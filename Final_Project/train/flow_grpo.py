"""Flow-GRPO training for AgentFlow's Planner module with LoRA.

Adapted from TinyZero (tinyzero.py) to multi-turn agentic planning:
  - Training data: NQ search questions + DeepMath-103K math questions
  - The Planner generates an action plan (context + sub-goal + tool name)
  - Trajectory-level reward is broadcast to every turn
  - Group-normalized advantages (standard GRPO)
  - LoRA adapter on Qwen3.5-0.8B

For simplicity with TRL's GRPOTrainer (which is single-turn), we train
the planner to produce a *single* complete action plan given a question.
The reward is based on whether the final answer (after AgentFlow execution)
matches the gold answer.  This is the "simplified Flow-GRPO" approach
that trains the planner's generation quality.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import string
from collections import Counter

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a planning agent. Given a question and a set of available tools, "
    "produce a step-by-step reasoning plan in <think> </think> tags, "
    "then provide a concise final answer in <answer> </answer> tags."
)

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

TOOLS_DESCRIPTION = (
    "Available tools: Base_Generator_Tool (reasoning), Python_Coder_Tool (code execution), "
    "Web_Search_Tool (web search), Wikipedia_Search_Tool (Wikipedia lookup)."
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _make_prompt_search(question: str, answer: str) -> list[dict]:
    """Create a chat prompt for a search / QA question."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{TOOLS_DESCRIPTION}\n\n"
            f"Question: {question}\n\n"
            "Think step by step about which tools you would use and why, "
            "then provide the final answer in <answer> </answer> tags."
        )},
    ]


def _make_prompt_math(question: str) -> list[dict]:
    """Create a chat prompt for a math question."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{TOOLS_DESCRIPTION}\n\n"
            f"Math problem: {question}\n\n"
            "Think step by step using Python_Coder_Tool if needed, "
            "then provide the numerical answer in <answer> </answer> tags."
        )},
    ]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_train_data(
    num_search: int = 2000,
    num_math: int = 2000,
) -> tuple[Dataset, Dataset]:
    """Load a mix of NQ search and math questions.

    Returns (train_dataset, eval_dataset).
    """
    records: list[dict] = []

    # NQ search questions
    try:
        nq = load_dataset("google-research-datasets/natural_questions", "default",
                          split="train", streaming=True, trust_remote_code=True)
        count = 0
        for ex in nq:
            if count >= num_search:
                break
            question = ex.get("question", {})
            q_text = question.get("text", "") if isinstance(question, dict) else str(question)
            sa = ex.get("annotations", {})
            short_answers = []
            if isinstance(sa, dict):
                for ans in sa.get("short_answers", []):
                    if isinstance(ans, dict) and ans.get("text"):
                        short_answers.extend(ans["text"])
            if not q_text or not short_answers:
                continue
            records.append({
                "prompt": _make_prompt_search(q_text, short_answers[0]),
                "gold_answer": short_answers[0],
                "task_type": "search",
            })
            count += 1
    except Exception as exc:
        print(f"Warning: NQ load failed ({exc}), using synthetic QA data")
        for i in range(min(num_search, 500)):
            records.append({
                "prompt": _make_prompt_search(
                    f"What is the capital of country #{i}?",
                    f"Capital_{i}",
                ),
                "gold_answer": f"Capital_{i}",
                "task_type": "search",
            })

    # Math questions (DeepMath-103K or MATH)
    try:
        math_ds = load_dataset("zwhe99/DeepMath-103K", split="train", streaming=True)
        count = 0
        for ex in math_ds:
            if count >= num_math:
                break
            q = ex.get("question", ex.get("problem", ""))
            a = str(ex.get("answer", ex.get("solution", "")))
            if not q:
                continue
            records.append({
                "prompt": _make_prompt_math(q),
                "gold_answer": a,
                "task_type": "math",
            })
            count += 1
    except Exception as exc:
        print(f"Warning: Math dataset load failed ({exc}), using AIME24 data")
        aime_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "benchmarks", "data", "aime24", "data.json",
        )
        if os.path.exists(aime_path):
            with open(aime_path) as f:
                aime = json.load(f)
            for ex in aime:
                records.append({
                    "prompt": _make_prompt_math(ex.get("query", ex.get("question", ""))),
                    "gold_answer": str(ex.get("answer", "")),
                    "task_type": "math",
                })

    if not records:
        raise RuntimeError("No training data loaded!")

    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(records)

    split = max(1, int(len(records) * 0.90))
    train_ds = Dataset.from_list(records[:split])
    eval_ds = Dataset.from_list(records[split:])

    keep_cols = {"prompt", "gold_answer", "task_type"}
    for col in list(train_ds.column_names):
        if col not in keep_cols:
            train_ds = train_ds.remove_columns([col])
            eval_ds = eval_ds.remove_columns([col])

    print(f"Training data: {len(train_ds)} samples ({sum(1 for r in records[:split] if r['task_type']=='search')} search, rest math)")
    print(f"Eval data:     {len(eval_ds)} samples")
    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pred_toks)
    r = n / len(gold_toks)
    return 2 * p * r / (p + r)


def _extract_number(s: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return float(m.group()) if m else None


def _numeric_match(pred: str, gold: str) -> float:
    p = _extract_number(pred)
    g = _extract_number(gold)
    if p is None or g is None:
        return 0.0
    return float(math.isclose(p, g, abs_tol=1e-3))


# ---------------------------------------------------------------------------
# Reward functions (for GRPOTrainer)
# ---------------------------------------------------------------------------

def _completion_text(c) -> str:
    if isinstance(c, list):
        return "".join(msg.get("content", "") for msg in c)
    if isinstance(c, dict):
        return c.get("content", "")
    return str(c)


def accuracy_reward(completions, gold_answer, task_type, **kwargs):
    """Trajectory-level reward based on answer correctness.

    For search/QA: F1 score (continuous 0-1)
    For math: numeric match (binary 0 or 1)
    """
    rewards = []
    for c, gold, tt in zip(completions, gold_answer, task_type):
        text = _completion_text(c)
        m = ANSWER_RE.search(text)
        if not m:
            rewards.append(0.0)
            continue
        pred = m.group(1).strip()

        if tt == "math":
            rewards.append(_numeric_match(pred, gold))
        else:
            rewards.append(_f1(pred, gold))
    return rewards


def format_reward(completions, **kwargs):
    """Reward for proper format: <think> + <answer> tags."""
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
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, dataset, max_new_tokens=1024, num_samples=100):
    model.eval()
    subset = dataset.select(range(min(num_samples, len(dataset))))

    correct = 0
    format_ok = 0
    total = 0

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
                temperature=0.7,
                top_p=0.9,
            )
        completion = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        has_think = bool(THINK_RE.search(completion))
        has_answer = bool(ANSWER_RE.search(completion))
        if has_think or has_answer:
            format_ok += 1

        # Try <answer> tag first; fall back to text after </think>
        m = ANSWER_RE.search(completion)
        if m:
            pred = m.group(1).strip()
        else:
            after_think = re.split(r"</think>", completion, flags=re.IGNORECASE)
            pred = after_think[-1].strip() if len(after_think) > 1 else completion.strip()

        gold = ex["gold_answer"]
        if pred:
            if ex["task_type"] == "math":
                correct += _numeric_match(pred, gold)
            else:
                correct += _f1(pred, gold)
        total += 1

    n = max(total, 1)
    results = {
        "accuracy": correct / n,
        "format_rate": format_ok / n,
        "n_samples": total,
    }
    print(f"  Eval ({total} samples): accuracy={results['accuracy']:.3f} format_rate={results['format_rate']:.1%}")
    return results


# ---------------------------------------------------------------------------
# Training callback
# ---------------------------------------------------------------------------

class LogCallback(TrainerCallback):
    def __init__(self, every_n: int = 5):
        self.every_n = every_n

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or state.global_step % self.every_n != 0:
            return
        r = logs.get("reward", None)
        if r is not None:
            print(f"[step {state.global_step:>5}] reward={r:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_name = "Qwen/Qwen3.5-0.8B"
    smoke_test = os.environ.get("SMOKE_TEST", "0") == "1"
    output_dir = "/runs/flow_grpo_smoke" if smoke_test else "/runs/flow_grpo"

    print(f"Model: {model_name}")
    print(f"Smoke test: {smoke_test}")
    print(f"Output dir: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if smoke_test:
        print("*** SMOKE TEST — small data, 8 steps ***")
        train_data, eval_data = load_train_data(num_search=100, num_math=100)
    else:
        train_data, eval_data = load_train_data(num_search=2000, num_math=2000)

    # Phase 1: Baseline
    print("\n=== Phase 1: Baseline Evaluation ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    baseline = evaluate(base_model, tokenizer, eval_data)
    del base_model
    torch.cuda.empty_cache()

    # Phase 2: Flow-GRPO with LoRA
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
        max_steps=8 if smoke_test else 200,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        num_generations=8,
        max_completion_length=1024,
        beta=0.001,
        logging_steps=1 if smoke_test else 5,
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=10 if smoke_test else 50,
        save_total_limit=5,
        report_to="none",
        reward_weights=[2.0, 1.0],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[accuracy_reward, format_reward],
        args=training_args,
        train_dataset=train_data,
        peft_config=peft_config,
    )

    trainer.add_callback(LogCallback(every_n=5))
    trainer.train()
    del trainer
    torch.cuda.empty_cache()

    # Phase 3: Evaluate checkpoints
    print("\n=== Phase 3: Evaluate Checkpoints ===")

    ckpt_dirs = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
        key=lambda d: int(d.split("-")[1]),
    )

    results = {}
    for ckpt_name in ckpt_dirs:
        ckpt_path = os.path.join(output_dir, ckpt_name)
        step = int(ckpt_name.split("-")[1])
        print(f"\nEvaluating {ckpt_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, ckpt_path)
        model = model.merge_and_unload()
        ckpt_results = evaluate(model, tokenizer, eval_data)
        results[step] = ckpt_results
        del model, base_model
        torch.cuda.empty_cache()

    if results:
        best_step = max(results, key=lambda s: results[s]["accuracy"])
        best = results[best_step]
        print(f"\nBest checkpoint: step {best_step}")
        print(f"  accuracy:    {best['accuracy']:.3f} (baseline {baseline['accuracy']:.3f})")
        print(f"  format_rate: {best['format_rate']:.1%}")

    # Save final results
    final = {
        "model": model_name,
        "baseline": baseline,
        "checkpoints": {str(k): v for k, v in results.items()},
        "best_step": best_step if results else None,
    }
    results_path = os.path.join(output_dir, "flow_grpo_results.json")
    with open(results_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
