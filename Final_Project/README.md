# TinyZero Countdown with LoRA + GRPO

Reproducing the [TinyZero](https://github.com/Jiayi-Pan/TinyZero) countdown task using **LoRA** (parameter-efficient fine-tuning) instead of full fine-tuning, trained with **GRPO** (Group Relative Policy Optimization) on a cloud GPU via [Modal](https://modal.com).

## Task

Given a set of numbers (e.g. `[43, 55, 53]`) and a target (e.g. `65`), the model must produce an arithmetic expression using each number exactly once that evaluates to the target. The model reasons inside `<think>` tags and returns the final expression in `<answer>` tags.

**Example prompt:**

> Using the numbers [43, 55, 53], create an equation that equals 65. You can use basic arithmetic operations (+, -, \*, /) and each number can only be used once.

**Expected output:**

```
<think> 55 - 43 = 12, and 12 + 53 = 65. </think>
<answer> 55 - 43 + 53 </answer>
```

## Method

**Model:** Qwen/Qwen2.5-0.5B-Instruct with LoRA adapter (r=16, alpha=64, all-linear)

**Algorithm:** GRPO from TRL — an online RL method that generates multiple completions per prompt, scores them with reward functions, and updates the policy toward higher-reward completions.

**Reward functions** (weighted 2:1):

| Function | Signal |
|----------|--------|
| `countdown_accuracy_reward` | Continuous partial credit based on number-overlap ratio (0.0–0.5), plus 1.0 for a correct equation |
| `format_reward` | 1.0 for both `<think>` and `<answer>` tags, 0.5 for one, 0.0 for neither |

**Dataset:** [Jiayi-Pan/Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) (3–4 numbers per problem)

## Pipeline

The script runs three phases automatically:

1. **Baseline evaluation** — greedy-decode the pre-trained model on the eval set
2. **GRPO fine-tuning** — train the LoRA adapter with the two reward functions
3. **Checkpoint evaluation** — load each saved checkpoint, evaluate, and report the best

## Results (200 steps on L40S)

| Metric | Baseline | Step 50 | Step 100 | Step 150 | Step 200 |
|--------|----------|---------|----------|----------|----------|
| format_rate | 0.0% | 0.0% | **100%** | 100% | 100% |
| overlap_mean | 0.781 | 0.888 | 0.983 | 0.990 | **0.993** |
| numbers_rate | 39.0% | 54.0% | 50.0% | **59.0%** | 52.0% |
| accuracy_reward | 0.400 | 0.449 | 0.492 | 0.495 | **0.496** |
| exact_acc | 2.0% | 1.0% | 0.0% | 0.0% | 0.0% |

**What the model learned:** format compliance (`<thought>` → `<think>`) and number usage (overlap 0.78 → 0.99). **What it didn't learn:** actual arithmetic — the model discovered a reward-hacking strategy of placing all correct numbers into a template like `(a * b - c) / d` without computing the right result, plateauing at reward ≈ 0.5.

## Project Structure

```
Final_Project/
├── tinyzero.py       # Training & evaluation script (all three phases)
├── run.py            # Modal deployment (GPU provisioning + entrypoint)
├── requirements.txt  # Python dependencies (for local reference)
└── README.md
```

## How to Run

### Prerequisites

1. Install the [Modal CLI](https://modal.com/docs/guide):

```bash
pip install modal
modal setup
```

2. Clone this repo and `cd` into `Final_Project/`.

### Smoke Test (~2 min, verifies the full pipeline)

```bash
modal run run.py --smoke-test
```

Runs 20 training steps on a small data subset. Watch for `accuracy_reward/std > 0` in the logs to confirm the reward signal is alive.

### Full Run (~15 min on L40S)

```bash
modal run run.py
```

Runs 200 training steps, saves checkpoints every 50 steps, and evaluates all checkpoints at the end.

### Configuration

Key parameters in `tinyzero.py` that you may want to adjust:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `max_steps` | 200 | Increase for longer training (e.g. 1000–4000) |
| `learning_rate` | 1e-5 | |
| `num_generations` | 8 | Completions per prompt for GRPO |
| `max_completion_length` | 256 | Max tokens per completion |
| `save_steps` | 50 | Checkpoint frequency |
| `reward_weights` | [2.0, 1.0] | [accuracy, format] |

GPU and timeout are configured in `run.py` (`gpu="L40S"`, `timeout=10800`).

## Known Limitations

- **Reward hacking at the 0.5 plateau:** The model learns to use correct numbers without solving the arithmetic. The gap between 0.5 (right numbers, wrong math) and 1.0 (correct) is too large for GRPO to bridge with a 0.5B model. A potential fix is adding intermediate reward tiers for "result close to target."
- **`<thought>` vs `<think>`:** The base Qwen model defaults to `<thought>` tags. Training pushes toward `<think>` (as specified in the prompt), but greedy eval may lag behind sampling-based training.
- **No `eval()` used:** All expression evaluation goes through a safe AST walker (`safe_eval`) that only allows `+`, `-`, `*`, `/`.

## References

- [TinyZero](https://github.com/Jiayi-Pan/TinyZero) — Jiayi Pan et al.
- [DeepSeekMath: GRPO](https://arxiv.org/abs/2402.03300)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [PEFT LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
