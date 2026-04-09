"""
Reproduce TinyZero countdown task with LoRA + GRPO on a Modal GPU.

Three phases:
  1. Baseline evaluation (pre-training accuracy)
  2. GRPO fine-tuning with LoRA adapter
  3. Evaluate all checkpoints, pick the best
"""

import ast
import math
import operator
import os
import re
import shutil
from collections import Counter

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# ── Constants ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant. You first think about the reasoning "
    "process in the mind and then provide the user with the answer."
)

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+")

ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


# ── Safe arithmetic evaluator (no eval()) ────────────────────────────────────

def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in ALLOWED_OPS:
            raise ValueError(f"unsupported op {op}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if op is ast.Div and abs(right) < 1e-12:
            raise ValueError("division by zero")
        return ALLOWED_OPS[op](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _eval_node(node.operand)
        return val if isinstance(node.op, ast.UAdd) else -val
    raise ValueError(f"unsupported node {type(node)}")


def safe_eval(expr: str) -> float:
    return _eval_node(ast.parse(expr, mode="eval"))


# ── Prompt construction (TinyZero style) ─────────────────────────────────────

def _user_message(target, numbers):
    return (
        f"Using the numbers {numbers}, create an equation that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can "
        "only be used once. Show your work in <think> </think> tags. "
        "And return the final answer in <answer> </answer> tags, "
        "for example <answer> (1 + 2) / 3 </answer>."
    )


def _make_prompt(example):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_message(example["target"], example["nums"])},
    ]


# ── Dataset ──────────────────────────────────────────────────────────────────

def load_countdown_data(num_train=32768, num_eval=1024):
    raw = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

    def process(ex):
        return {
            "prompt": _make_prompt(ex),
            "target": ex["target"],
            "numbers": ex["nums"],
        }

    train = raw.select(range(num_train)).map(process)
    val = raw.select(range(num_train, num_train + num_eval)).map(process)

    keep = ["prompt", "target", "numbers"]
    train = train.remove_columns([c for c in train.column_names if c not in keep])
    val = val.remove_columns([c for c in val.column_names if c not in keep])
    return train, val


# ── Reward helpers ───────────────────────────────────────────────────────────

def _completion_text(completion):
    """Normalize a TRL completion (string, message-dict, or message-list) to plain text."""
    if isinstance(completion, list):
        return "".join(msg.get("content", "") for msg in completion)
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _strip_eq(answer: str) -> str:
    """Remove '= result' suffix that models typically append inside <answer> tags.

    E.g. '(77 - 4) * 4 = 15' → '(77 - 4) * 4'
    Without this, NUMBER_RE picks up the RHS result number and Counter never matches.
    """
    return answer.split("=")[0].strip() if "=" in answer else answer.strip()


def _check_answer(text: str, target, numbers) -> bool:
    """Return True if <answer> contains a valid expression equalling target with exactly the given numbers."""
    m = ANSWER_RE.search(text)
    if not m:
        return False
    expr = _strip_eq(m.group(1))
    used = [int(x) for x in NUMBER_RE.findall(expr)]
    if Counter(used) != Counter(numbers):
        return False
    try:
        return math.isclose(safe_eval(expr), float(target), abs_tol=1e-6)
    except Exception:
        return False


# ── Reward functions ─────────────────────────────────────────────────────────

def countdown_accuracy_reward(completions, target, numbers, **kwargs):
    """Graded reward with smooth partial credit for number overlap.

    The model typically writes 'expr = result' inside <answer>, so we strip
    the '= result' suffix before extracting numbers (see _strip_eq).

    Reward tiers (continuous on the overlap dimension):
      0.0                        no <answer> or no numbers in expression
      overlap/total * 0.5        partial number overlap (continuous 0…0.5)
      0.5                        exact number set but wrong/unparseable result
      1.0                        correct equation reaching the target
    """
    rewards = []
    for c, tgt, nums in zip(completions, target, numbers):
        text = _completion_text(c)
        m = ANSWER_RE.search(text)
        if not m:
            rewards.append(0.0)
            continue

        expr = _strip_eq(m.group(1))
        used = [int(x) for x in NUMBER_RE.findall(expr)]
        if not used:
            rewards.append(0.0)
            continue

        nums_counter = Counter(nums)
        used_counter = Counter(used)
        overlap = sum((nums_counter & used_counter).values())
        total = sum(nums_counter.values())

        if used_counter != nums_counter:
            rewards.append(overlap / total * 0.5)
            continue

        try:
            if math.isclose(safe_eval(expr), float(tgt), abs_tol=1e-6):
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        except Exception:
            rewards.append(0.5)
    return rewards


def format_reward(completions, **kwargs):
    """1.0 if both <think> and <answer> tags present, 0.5 for partial, 0.0 for neither."""
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


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, dataset, max_new_tokens=512, num_samples=100, show_samples=5):
    """Evaluate with metrics aligned to the training reward functions.

    Uses the same _strip_eq and grading tiers as countdown_accuracy_reward,
    so training metrics and eval metrics measure the exact same thing.

    Returns a dict with:
      exact_acc              — expression evaluates to target with correct numbers
      format_rate            — has both <think> and <answer> tags
      numbers_rate           — exact number set match (after stripping '= result')
      overlap_mean           — mean number-overlap ratio (continuous 0…1)
      accuracy_reward_mean   — same scale as countdown_accuracy_reward
    """
    model.eval()
    subset = dataset.select(range(min(num_samples, len(dataset))))

    stats = {
        "exact": 0, "format_ok": 0, "numbers_ok": 0,
        "acc_reward_sum": 0.0, "overlap_sum": 0.0,
    }
    total = 0
    samples = []

    for ex in subset:
        prompt_text = tokenizer.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        completion = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        has_think = bool(THINK_RE.search(completion))
        has_answer = bool(ANSWER_RE.search(completion))
        if has_think and has_answer:
            stats["format_ok"] += 1

        m = ANSWER_RE.search(completion)
        if m:
            expr = _strip_eq(m.group(1))
            used = [int(x) for x in NUMBER_RE.findall(expr)]
            nums_counter = Counter(ex["numbers"])

            if used:
                used_counter = Counter(used)
                overlap = sum((nums_counter & used_counter).values())
                total_nums = sum(nums_counter.values())
                overlap_ratio = overlap / total_nums if total_nums else 0
                stats["overlap_sum"] += overlap_ratio

                if used_counter == nums_counter:
                    stats["numbers_ok"] += 1
                    try:
                        if math.isclose(safe_eval(expr), float(ex["target"]), abs_tol=1e-6):
                            stats["exact"] += 1
                            stats["acc_reward_sum"] += 1.0
                        else:
                            stats["acc_reward_sum"] += 0.5
                    except Exception:
                        stats["acc_reward_sum"] += 0.5
                else:
                    stats["acc_reward_sum"] += overlap_ratio * 0.5

        total += 1
        if len(samples) < show_samples:
            samples.append((ex["target"], ex["numbers"], completion))
        if total % 20 == 0:
            print(f"  evaluated {total}/{len(subset)}  running_exact_acc={stats['exact'] / total:.1%}")

    n = max(total, 1)
    results = {
        "exact_acc": stats["exact"] / n,
        "format_rate": stats["format_ok"] / n,
        "numbers_rate": stats["numbers_ok"] / n,
        "overlap_mean": stats["overlap_sum"] / n,
        "accuracy_reward_mean": stats["acc_reward_sum"] / n,
    }

    print(f"\n  --- Eval Results ({total} samples) ---")
    print(f"  format_rate (think+answer):  {results['format_rate']:.1%}")
    print(f"  overlap_mean (num overlap):  {results['overlap_mean']:.3f}")
    print(f"  numbers_rate (exact nums):   {results['numbers_rate']:.1%}")
    print(f"  accuracy_reward_mean:        {results['accuracy_reward_mean']:.3f}")
    print(f"  exact_acc:                   {results['exact_acc']:.1%}")

    if samples:
        print(f"\n  --- Sample Completions (first {len(samples)}) ---")
        for i, (tgt, nums, comp) in enumerate(samples):
            print(f"  [{i}] target={tgt} nums={nums}")
            print(f"       {comp[:300]}{'...' if len(comp) > 300 else ''}")

    return results


# ── Callback ─────────────────────────────────────────────────────────────────

class ShowExamplesCallback(TrainerCallback):
    """Print the mean reward every N steps."""

    def __init__(self, tokenizer, every_n_steps=25):
        self.tokenizer = tokenizer
        self.every_n_steps = every_n_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or state.global_step % self.every_n_steps != 0:
            return
        reward = logs.get("reward", None)
        if reward is not None:
            print(f"[step {state.global_step:>5}]  mean_reward={reward:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    smoke_test = os.environ.get("SMOKE_TEST", "0") == "1"
    output_dir = "/runs/countdown_smoke" if smoke_test else "/runs/countdown_grpo_output"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if smoke_test:
        print("*** SMOKE TEST MODE — small dataset, 20 steps ***")
        train_data, eval_data = load_countdown_data(num_train=256, num_eval=32)
    else:
        train_data, eval_data = load_countdown_data()

    # ── Phase 1: Baseline ────────────────────────────────────────────────
    print("\n=== Phase 1: Baseline Evaluation ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    baseline = evaluate(base_model, tokenizer, eval_data)
    baseline_acc = baseline["exact_acc"]
    print(f"Baseline exact accuracy: {baseline_acc:.1%}")
    del base_model
    torch.cuda.empty_cache()

    # ── Phase 2: GRPO with LoRA ──────────────────────────────────────────
    if os.path.exists(output_dir):
        print(f"Cleaning stale output_dir: {output_dir}")
        shutil.rmtree(output_dir)

    print("\n=== Phase 2: GRPO Fine-Tuning with LoRA ===")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=output_dir,
        max_steps=20 if smoke_test else 200,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        num_generations=8,
        max_completion_length=256,
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
        reward_funcs=[countdown_accuracy_reward, format_reward],
        args=training_args,
        train_dataset=train_data,
        peft_config=peft_config,
    )

    trainer.add_callback(ShowExamplesCallback(tokenizer, every_n_steps=25))
    trainer.train()
    del trainer
    torch.cuda.empty_cache()

    # ── Phase 3: Evaluate all checkpoints ────────────────────────────────
    print("\n=== Phase 3: Evaluate All Checkpoints ===")

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
        print(f"  Step {step}: exact={ckpt_results['exact_acc']:.1%}  "
              f"numbers={ckpt_results['numbers_rate']:.1%}  "
              f"reward_mean={ckpt_results['accuracy_reward_mean']:.3f}")
        del model, base_model
        torch.cuda.empty_cache()

    if results:
        best_step = max(results, key=lambda s: results[s]["exact_acc"])
        best = results[best_step]
        print(f"\n{'=' * 60}")
        print(f"Best checkpoint: step {best_step}")
        print(f"  exact_acc:            {best['exact_acc']:.1%}  (baseline {baseline_acc:.1%}, delta {best['exact_acc'] - baseline_acc:+.1%})")
        print(f"  format_rate:          {best['format_rate']:.1%}")
        print(f"  numbers_rate:         {best['numbers_rate']:.1%}")
        print(f"  accuracy_reward_mean: {best['accuracy_reward_mean']:.3f}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
