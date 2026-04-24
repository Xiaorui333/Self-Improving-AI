"""Flow-GRPO Planner training for AgentFlow -- Phase 6 (full pipeline).

Trains Qwen3.5-0.8B with LoRA using GRPO so it acts as a better Planner
inside the full AgentFlow system (Planner → Executor → Verifier → Generator).

Training data covers all 10 benchmark categories:
  Search  (Bamboogle/2Wiki/HotpotQA/Musique) → HotpotQA train split
  Math    (AIME/AMC/Game-of-24)              → GSM8K train split
  Science (GPQA/MedQA)                       → MedMCQA train split
  Agentic (GAIA)                             → mixed QA from above

Reward functions (weighted 3:1):
  answer_reward  — F1 / numeric match / letter accuracy vs gold answer
  format_reward  — structured reasoning (<think> block + explicit answer line)

After training, the checkpoint is saved to a Modal Volume so
``train/serve_grpo_model.py`` can load and serve it immediately.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import string
import unicodedata

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------------------
# System prompt — mirrors AgentFlow Planner style
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert reasoning agent. "
    "Think through the problem carefully inside <think> </think> tags. "
    "Then state your final answer clearly on a new line starting with "
    "\"Answer:\" (for QA/science) or just the number/letter for math/MC."
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"answer\s*[:：]\s*(.+)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Data loading — one function per category
# ---------------------------------------------------------------------------

def _load_hotpotqa(n: int) -> list[dict]:
    """Multi-hop search QA (covers Bamboogle, 2Wiki, HotpotQA, Musique)."""
    ds = load_dataset("hotpot_qa", "distractor", split="train", trust_remote_code=True)
    records = []
    for ex in ds.shuffle(seed=42).select(range(min(n * 3, len(ds)))):
        q = ex.get("question", "")
        a = ex.get("answer", "")
        if q and a:
            records.append({"question": q, "gold": a, "task_type": "qa"})
        if len(records) >= n:
            break
    return records


def _load_gsm8k(n: int) -> list[dict]:
    """Math word problems (covers AIME, AMC, Game-of-24 category)."""
    ds = load_dataset("openai/gsm8k", "main", split="train", trust_remote_code=True)
    records = []
    for ex in ds.shuffle(seed=42).select(range(min(n * 2, len(ds)))):
        q = ex.get("question", "")
        a = ex.get("answer", "")
        # GSM8K answers are like "...#### 42"
        m = re.search(r"####\s*(-?[\d,]+)", a)
        if q and m:
            records.append({
                "question": q,
                "gold": m.group(1).replace(",", ""),
                "task_type": "math",
            })
        if len(records) >= n:
            break
    return records


def _load_medmcqa(n: int) -> list[dict]:
    """Medical MCQ (covers GPQA, MedQA category)."""
    ds = load_dataset("openlifescienceai/medmcqa", split="train", trust_remote_code=True)
    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    records = []
    for ex in ds.shuffle(seed=42).select(range(min(n * 2, len(ds)))):
        q = ex.get("question", "")
        cop = ex.get("cop", -1)  # correct option index 0-3
        opa = ex.get("opa", "")
        opb = ex.get("opb", "")
        opc = ex.get("opc", "")
        opd = ex.get("opd", "")
        if q and cop in (0, 1, 2, 3):
            full_q = (
                f"{q}\n"
                f"A. {opa}\nB. {opb}\nC. {opc}\nD. {opd}"
            )
            records.append({
                "question": full_q,
                "gold": option_map[cop],
                "task_type": "mc",
            })
        if len(records) >= n:
            break
    return records


def load_agentflow_train_data(smoke_test: bool = False) -> tuple[Dataset, Dataset]:
    """Load and mix training data covering all 10 benchmark categories."""
    if smoke_test:
        # Tiny inline set for smoke tests (no internet needed)
        records = [
            {"question": "What do David Gilmour and Roger Waters have in common?",
             "gold": "Pink Floyd", "task_type": "qa"},
            {"question": "Sarah has 5 apples. She gives 2 to Tom and buys 3 more. How many does she have?",
             "gold": "6", "task_type": "math"},
            {"question": "Which vitamin deficiency causes scurvy?\nA. Vitamin A\nB. Vitamin B12\nC. Vitamin C\nD. Vitamin D",
             "gold": "C", "task_type": "mc"},
            {"question": "The Eiffel Tower is located in which city?",
             "gold": "Paris", "task_type": "qa"},
            {"question": "If a train travels 60 mph for 2.5 hours, how far does it go?",
             "gold": "150", "task_type": "math"},
            {"question": "What is the powerhouse of the cell?\nA. Nucleus\nB. Ribosome\nC. Mitochondria\nD. Golgi apparatus",
             "gold": "C", "task_type": "mc"},
            {"question": "Who wrote the play Hamlet?",
             "gold": "Shakespeare", "task_type": "qa"},
            {"question": "A rectangle is 8 cm wide and 5 cm tall. What is its area?",
             "gold": "40", "task_type": "math"},
        ]
    else:
        per_category = 200  # 600 total training examples
        print("Loading HotpotQA (search/agentic)...")
        qa = _load_hotpotqa(per_category)
        print(f"  {len(qa)} QA problems loaded")

        print("Loading GSM8K (math)...")
        math = _load_gsm8k(per_category)
        print(f"  {len(math)} math problems loaded")

        print("Loading MedMCQA (science)...")
        mc = _load_medmcqa(per_category)
        print(f"  {len(mc)} MC problems loaded")

        records = qa + math + mc

    import random as _rng
    _rng.seed(42)
    _rng.shuffle(records)

    split = max(1, int(len(records) * 0.9))
    train_recs = records[:split]
    eval_recs = records[split:]

    def to_row(r: dict) -> dict:
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["question"]},
            ],
            "gold": r["gold"],
            "task_type": r["task_type"],
        }

    train_ds = Dataset.from_list([to_row(r) for r in train_recs])
    eval_ds = Dataset.from_list([to_row(r) for r in eval_recs])
    print(f"AgentFlow GRPO data: {len(train_ds)} train, {len(eval_ds)} eval")
    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _normalise(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    p_toks = set(_normalise(pred).split())
    g_toks = set(_normalise(gold).split())
    if not p_toks or not g_toks:
        return 0.0
    common = p_toks & g_toks
    if not common:
        return 0.0
    prec = len(common) / len(p_toks)
    rec = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def _numeric(pred: str, gold: str) -> float:
    try:
        pv = float(re.sub(r"[^0-9.\-]", "", pred.split()[0]))
        gv = float(re.sub(r"[^0-9.\-]", "", gold))
        return 1.0 if abs(pv - gv) < max(1e-3, 0.01 * abs(gv)) else 0.0
    except Exception:
        return 0.0


def _extract_answer(text: str) -> str:
    """Pull answer from model output: Answer: line > after </think> > full text."""
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    parts = re.split(r"</think>", text, flags=re.IGNORECASE)
    return parts[-1].strip() if len(parts) > 1 else text.strip()


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def _completion_text(c) -> str:
    if isinstance(c, list):
        return "".join(msg.get("content", "") for msg in c)
    if isinstance(c, dict):
        return c.get("content", "")
    return str(c)


def answer_reward(completions, gold, task_type, **kwargs):
    """Correctness reward: F1 for QA, numeric match for math, letter for MC."""
    rewards = []
    for c, g, tt in zip(completions, gold, task_type):
        text = _completion_text(c)
        pred = _extract_answer(text)
        if tt == "math":
            r = _numeric(pred, g)
        elif tt == "mc":
            # letter match: first uppercase letter A-D in prediction
            letters = re.findall(r"\b([A-D])\b", pred.upper())
            r = 1.0 if letters and letters[0] == g.upper() else 0.0
        else:
            r = _f1(pred, g)
        rewards.append(r)
    return rewards


def format_reward(completions, **kwargs):
    """Structured reasoning reward: 1.0 for <think> + Answer:, partial credit."""
    rewards = []
    for c in completions:
        text = _completion_text(c)
        has_think = bool(THINK_RE.search(text))
        has_answer = bool(ANSWER_RE.search(text))
        if has_think and has_answer:
            rewards.append(1.0)
        elif has_think or has_answer:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, dataset, max_new_tokens: int = 512, num_samples: int = 50):
    model.eval()
    subset = dataset.select(range(min(num_samples, len(dataset))))
    correct = fmt_ok = total = 0

    for ex in subset:
        prompt_text = tokenizer.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
            )
        text = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )
        pred = _extract_answer(text)
        tt = ex["task_type"]
        g = ex["gold"]

        if tt == "math":
            correct += _numeric(pred, g)
        elif tt == "mc":
            letters = re.findall(r"\b([A-D])\b", pred.upper())
            correct += 1.0 if letters and letters[0] == g.upper() else 0.0
        else:
            correct += _f1(pred, g)

        has_think = bool(THINK_RE.search(text))
        has_answer = bool(ANSWER_RE.search(text))
        if has_think or has_answer:
            fmt_ok += 1
        total += 1

    n = max(total, 1)
    res = {"accuracy": correct / n, "format_rate": fmt_ok / n, "n": total}
    print(
        f"  Eval ({total}): accuracy={res['accuracy']:.3f}  "
        f"format={res['format_rate']:.1%}"
    )
    return res


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        r = logs.get("reward")
        ans_r = logs.get("rewards/answer_reward/mean")
        fmt_r = logs.get("rewards/format_reward/mean")
        if r is not None:
            print(
                f"[step {state.global_step:>4}] "
                f"reward={r:.4f}  answer={ans_r or 0:.4f}  fmt={fmt_r or 0:.4f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VOLUME_ROOT = "/vol/flow_grpo_agentflow"    # Modal Volume mount point
CHECKPOINT_DIR = "/vol/flow_grpo_agentflow/run"  # writable subdir within volume


def main():
    model_name = "Qwen/Qwen3.5-0.8B"
    smoke_test = os.environ.get("SMOKE_TEST", "0") == "1"
    output_dir = CHECKPOINT_DIR

    print(f"Model:      {model_name}")
    print(f"Smoke test: {smoke_test}")
    print(f"Output dir: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_data, eval_data = load_agentflow_train_data(smoke_test=smoke_test)

    # Phase 1 — baseline
    print("\n=== Phase 1: Baseline Evaluation ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    baseline = evaluate(base_model, tokenizer, eval_data)
    del base_model
    torch.cuda.empty_cache()

    # Phase 2 — Flow-GRPO with LoRA
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    print("\n=== Phase 2: Flow-GRPO Training with LoRA ===")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=output_dir,
        max_steps=8 if smoke_test else 400,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        num_generations=8,
        max_completion_length=512,
        beta=0.001,
        logging_steps=1,
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=10 if smoke_test else 100,
        save_total_limit=5,
        report_to="none",
        reward_weights=[3.0, 1.0],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[answer_reward, format_reward],
        args=training_args,
        train_dataset=train_data,
        peft_config=peft_config,
    )
    trainer.add_callback(LogCallback())
    trainer.train()
    del trainer
    torch.cuda.empty_cache()

    # Phase 3 — evaluate checkpoints
    print("\n=== Phase 3: Evaluate Checkpoints ===")
    ckpt_dirs = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
        key=lambda d: int(d.split("-")[1]),
    )

    results: dict[int, dict] = {}
    for cn in ckpt_dirs:
        cp = os.path.join(output_dir, cn)
        step = int(cn.split("-")[1])
        print(f"\nEvaluating {cn} ...")
        bm = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(bm, cp).merge_and_unload()
        results[step] = evaluate(model, tokenizer, eval_data)
        del model, bm
        torch.cuda.empty_cache()

    best_step = max(results, key=lambda s: results[s]["accuracy"]) if results else None
    if best_step is not None:
        best = results[best_step]
        print(f"\nBest checkpoint: step {best_step}")
        print(f"  accuracy:    {best['accuracy']:.3f} (baseline {baseline['accuracy']:.3f})")
        print(f"  format_rate: {best['format_rate']:.1%}")

        # Write marker at volume root so serve_grpo_model.py can find it
        best_path = os.path.join(output_dir, f"checkpoint-{best_step}")
        marker = os.path.join(VOLUME_ROOT, "best_checkpoint.txt")
        os.makedirs(VOLUME_ROOT, exist_ok=True)
        with open(marker, "w") as f:
            f.write(best_path)
        print(f"  checkpoint:  {best_path}")

    final = {
        "model": model_name,
        "baseline": baseline,
        "checkpoints": {str(k): v for k, v in results.items()},
        "best_step": best_step,
    }
    results_path = os.path.join(VOLUME_ROOT, "agentflow_grpo_results.json")
    with open(results_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nResults and checkpoint saved to {VOLUME_ROOT}")
    print("Next step: modal run train/run_train.py --serve")


if __name__ == "__main__":
    main()
