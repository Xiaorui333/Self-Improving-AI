"""Flow-GRPO training on SecBench (security vulnerabilities) -- Phase 6.

Same architecture as flow_grpo.py but with SecBench SAQ training data
and security-focused reward function.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import string
from collections import Counter

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

SYSTEM_PROMPT = (
    "You are a cybersecurity expert. Think through the problem step by step "
    "in <think> </think> tags, then provide a precise answer in <answer> </answer> tags."
)

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _make_prompt(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Available tools: Web_Search_Tool (web search), "
            "Base_Generator_Tool (reasoning), Python_Coder_Tool (code execution).\n\n"
            f"Cybersecurity question: {question}\n\n"
            "Think step by step, then answer in <answer> </answer> tags."
        )},
    ]


def load_secbench_data() -> tuple[Dataset, Dataset]:
    """Load SecBench SAQ data for training."""
    secbench_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "benchmarks", "data", "secbench", "secbench_saq_en.json",
    )

    if os.path.exists(secbench_path):
        with open(secbench_path) as f:
            raw = json.load(f)
    else:
        # Use the synthetic fallback from run_secbench
        from benchmarks.run_secbench import _synthetic_security_data
        raw = _synthetic_security_data()

    records = []
    for item in raw:
        records.append({
            "prompt": _make_prompt(item["query"]),
            "gold_answer": item["answer"],
        })

    import random
    random.seed(42)
    random.shuffle(records)

    split = max(1, int(len(records) * 0.9))
    train_ds = Dataset.from_list(records[:split])
    eval_ds = Dataset.from_list(records[split:])

    print(f"SecBench training: {len(train_ds)} samples, eval: {len(eval_ds)} samples")
    return train_ds, eval_ds


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    pt = _normalize(pred).split()
    gt = _normalize(gold).split()
    if not pt or not gt:
        return 0.0
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pt)
    r = n / len(gt)
    return 2 * p * r / (p + r)


def _completion_text(c) -> str:
    if isinstance(c, list):
        return "".join(msg.get("content", "") for msg in c)
    if isinstance(c, dict):
        return c.get("content", "")
    return str(c)


def accuracy_reward(completions, gold_answer, **kwargs):
    rewards = []
    for c, gold in zip(completions, gold_answer):
        text = _completion_text(c)
        m = ANSWER_RE.search(text)
        if not m:
            rewards.append(0.0)
            continue
        pred = m.group(1).strip()
        rewards.append(_f1(pred, gold))
    return rewards


def format_reward(completions, **kwargs):
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


def evaluate(model, tokenizer, dataset, max_new_tokens=512, num_samples=50):
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
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        comp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        if THINK_RE.search(comp) and ANSWER_RE.search(comp):
            format_ok += 1
        m = ANSWER_RE.search(comp)
        if m:
            correct += _f1(m.group(1).strip(), ex["gold_answer"])
        total += 1

    n = max(total, 1)
    results = {"accuracy": correct / n, "format_rate": format_ok / n, "n": total}
    print(f"  Eval ({total}): accuracy={results['accuracy']:.3f} format={results['format_rate']:.1%}")
    return results


def main():
    model_name = "Qwen/Qwen3.5-0.8B"
    smoke_test = os.environ.get("SMOKE_TEST", "0") == "1"
    output_dir = "/runs/flow_grpo_secbench_smoke" if smoke_test else "/runs/flow_grpo_secbench"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_data, eval_data = load_secbench_data()

    print("\n=== Baseline ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    baseline = evaluate(base_model, tokenizer, eval_data)
    del base_model
    torch.cuda.empty_cache()

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    print("\n=== Flow-GRPO Training ===")
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
        max_completion_length=512,
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
    trainer.train()
    del trainer
    torch.cuda.empty_cache()

    print("\n=== Evaluate Checkpoints ===")
    ckpt_dirs = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
        key=lambda d: int(d.split("-")[1]),
    )

    results = {}
    for cn in ckpt_dirs:
        cp = os.path.join(output_dir, cn)
        step = int(cn.split("-")[1])
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, cp).merge_and_unload()
        results[step] = evaluate(model, tokenizer, eval_data)
        del model, base_model
        torch.cuda.empty_cache()

    final = {
        "model": model_name, "baseline": baseline,
        "checkpoints": {str(k): v for k, v in results.items()},
    }
    with open(os.path.join(output_dir, "secbench_results.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to {output_dir}/secbench_results.json")


if __name__ == "__main__":
    main()
