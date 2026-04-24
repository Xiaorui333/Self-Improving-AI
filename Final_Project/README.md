# Final Project: TinyZero + AgentFlow

This repository has **two parts**: **Part 1** is a self-contained [TinyZero](https://github.com/Jiayi-Pan/TinyZero)-style **countdown task** with LoRA + GRPO on Modal. **Part 2** reimplements [AgentFlow](https://agentflow.stanford.edu/), runs **QA + HumanEval** benchmarks (Portkey → DeepInfra), and experiments with **simplified Flow-GRPO** on Qwen3.5-0.8B (Modal L40S).

---

# Part 1: TinyZero Countdown with LoRA + GRPO

Reproducing the TinyZero countdown task using **LoRA** instead of full fine-tuning, trained with **GRPO** (Group Relative Policy Optimization from TRL) on a cloud GPU via [Modal](https://modal.com).

## Task

Given numbers (e.g. `[43, 55, 53]`) and a target (e.g. `65`), the model must produce an arithmetic expression using **each number exactly once**. The prompt asks for reasoning in `<think>…</think>` and the expression in `<answer>…</answer>` (see `tinyzero.py` for regex-based scoring).

## Method

| Piece | Choice |
|-------|--------|
| **Model** | `Qwen/Qwen2.5-0.5B-Instruct` + LoRA (r=16, α=64, all-linear) |
| **Algorithm** | GRPO — multiple completions per prompt, reward-weighted update |
| **Rewards** (≈2:1 accuracy:format) | `countdown_accuracy_reward` (overlap + exact equation), `format_reward` (both tags) |
| **Data** | [Jiayi-Pan/Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) |

## Pipeline

`tinyzero.py` runs: **(1)** baseline eval → **(2)** GRPO training → **(3)** checkpoint eval.

## Results (200 steps, L40S)

| Metric | Baseline | Step 50 | Step 100 | Step 150 | Step 200 |
|--------|----------|---------|----------|----------|----------|
| format_rate | 0.0% | 0.0% | **100%** | 100% | 100% |
| overlap_mean | 0.781 | 0.888 | 0.983 | 0.990 | **0.993** |
| numbers_rate | 39.0% | 54.0% | 50.0% | **59.0%** | 52.0% |
| accuracy_reward | 0.400 | 0.449 | 0.492 | 0.495 | **0.496** |
| exact_acc | 2.0% | 1.0% | 0.0% | 0.0% | 0.0% |

**Analysis:** The policy learns **format compliance** and **high number overlap**, but **exact arithmetic** barely improves. The model tends to **reward-hack** by placing the right numbers in a template (e.g. `(a * b - c) / d`) without the expression evaluating to the target—hence accuracy_reward plateaus near **0.5** and exact_acc stays ~**0%**.

### Part 1 — Project layout

```
Final_Project/
├── tinyzero.py       # Training & eval (three phases)
├── run.py            # Modal: GPU + entrypoint
└── requirements.txt
```

### Part 1 — How to run

```bash
pip install modal && modal setup
cd Final_Project
modal run run.py --smoke-test    # ~2 min, 20 steps
modal run run.py                 # ~15 min, 200 steps on L40S
```

Key hyperparameters: `tinyzero.py` (`max_steps`, `learning_rate`, `num_generations`, `save_steps`, `reward_weights`); GPU and timeout in `run.py` (`gpu="L40S"`).

### Part 1 — Limitations

- **0.5 plateau:** Gap between “right numbers, wrong value” and exact solve is hard for GRPO on 0.5B; intermediate rewards (e.g. closeness to target) could help.
- **Tag mismatch:** Base Qwen may emit `<thought>`-style content; training pushes `<think>` per prompt—greedy eval can lag sampling-based training.
- **Safety:** Expression checking uses a safe AST walker (`safe_eval`), not raw `eval()`.

### Part 1 — References

- [TinyZero](https://github.com/Jiayi-Pan/TinyZero) · [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) · [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) · [PEFT LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)

---

# Part 2: AgentFlow Reimplementation

Reimplementation of [AgentFlow](https://agentflow.stanford.edu/) (ICLR 2026 Oral): a **Planner → Executor → Verifier** loop with shared memory and tools (Python coder, web/Wikipedia search). Inference goes through **Portkey** to DeepInfra unless you attach a **local GRPO endpoint** after training.

## Results at a glance (Part 2)

| Track | What we measured | Outcome |
|------|------------------|---------|
| **QA benchmarks (Phases 2–3)** | EM/F1/accuracy, n≈20 | **Qwen2.5-7B** strongest overall. **Qwen3.5** scales on easy search (e.g. Bamboogle: **27B** best in-family). Multi-hop (2Wiki, Musique, GAIA) ≈ **0** for all sizes. |
| **HumanEval + AgentFlow (Phase 4)** | pass@1, n=20 | **2B 0.85**, **9B 0.80**, **4B 0.50**, **Qwen2.5-7B 0.70**; **0.8B 0.00**; **27B 0.30†** (partial / budget). |
| **Flow-GRPO NQ+Math (Phase 5)** | LoRA on 0.8B | Training rewards ↑; **checkpoint eval &lt; baseline** (format vs native thinking output). |
| **Flow-GRPO HumanEval (Phase 6)** | LoRA on 0.8B | **100 steps** ~20 min. Held-out **n=4**: baseline **0.25**, best **ckpt-50 0.50**—noisy; fair test = **20-problem** `run_humaneval.py` with LoRA. |
| **SecBench** | LLM-judge, small n | Exploratory only—not used for scaling claims. |
| **Planner GRPO → 10 benchmarks** | `qwen3.5-0.8b-grpo` | Train often completes; **long eval / deploy + full benchmarks** may be incomplete (timeouts)—see Phase 6b. |

**Artifacts:** `results/humaneval_combined.json`, `results/secbench_combined.json`; Modal: `humaneval_grpo_results.json` under `/runs/flow_grpo_humaneval/` on the `runs` volume.

## Part 2 — Repository layout

```
Final_Project/
├── agentflow/           # engines, planner/executor/verifier, tools
├── benchmarks/          # run_benchmark.py, run_all_models.py, run_humaneval.py, run_secbench.py, score.py
├── train/               # flow_grpo*.py, run_train.py, serve_grpo_model.py, run_agentflow_pipeline.py
├── results/
├── tinyzero.py, run.py  # Part 1
└── requirements.txt
```

## Setup (Part 2)

- **Python 3.11+**, **Portkey** key, **Modal** for training.
- `pip install -r requirements.txt` (add benchmark deps: `portkey-ai duckduckgo-search wikipedia pydantic datasets`, etc.).
- `python3.11 benchmarks/download_data.py`

## Architecture (brief)

1. **Planner** — query analysis, tool choice, subgoals  
2. **Executor** — tool commands (code, search, …)  
3. **Verifier** — STOP / CONTINUE  
4. **Memory** — accumulates observations until STOP or step limit  

## Benchmarks

| Category | Tasks |
|----------|--------|
| Search | Bamboogle, 2Wiki, HotpotQA, Musique |
| Agentic | GAIA |
| Math | AIME 2024, AMC 23, Game of 24 |
| Science | GPQA, MedQA |

**HumanEval (extra):** code completion; **pass@1** by execution—not EM/F1. **`run_humaneval.py`** runs full AgentFlow with **PythonCoderTool** (20 problems, seed **42**).

---

## Experiments & analysis

### QA baselines and scaling (n=20)

**Phase 2 — Qwen2.5-7B-Instruct**

| Area | Benchmark | Score |
|------|-----------|--------|
| Search | Bamboogle EM/F1 | 0.60 / 0.70 |
| Search | HotpotQA EM/F1 | 0.25 / 0.36 |
| Search | 2Wiki, Musique | ~0 |
| Agentic | GAIA | ~0 |
| Math | AIME 2024 acc | 0.75* |
| Math | AMC 23, Game of 24 | 0.00 |
| Science | GPQA, MedQA | 0.20, 0.00 |

\*Small sample; high variance at full scale.

**Phase 3 — Qwen3.5 vs reference**

| Model | Bamboogle EM/F1 | HotpotQA EM/F1 | Notes |
|-------|-----------------|----------------|--------|
| 3.5-0.8B | 0 / 0 | 0 / 0.07 | — |
| 3.5-2B | 0.05 / 0.06 | **0.30 / 0.37** | Strong HotpotQA vs larger 3.5 |
| 3.5-4B | 0 / 0 † | 0 / 0 † | API errors—unreliable |
| 3.5-9B | 0.10 / 0.14 | 0.10 / 0.10 | — |
| 3.5-27B | **0.40 / 0.49** | 0.15 / 0.20 | Best Bamboogle in 3.5 family |
| 2.5-7B | 0.60 / 0.70 | 0.25 / 0.36 | Overall reference |

**Observations:** **Bamboogle** shows a rough size trend within 3.5 (27B &gt; 9B &gt; 2B). **HotpotQA** is not monotone in size (2B beats several larger 3.5s). **AIME** only **2.5-7B** scores in our table. **GPQA** is flat (~0.20–0.25). **Multi-hop / GAIA** stay near zero—retrieval + agentic depth remains the bottleneck vs single-hop QA.

### HumanEval (Phase 4) — n=20, pass@1

| Model | pass@1 | Passed |
|-------|--------|--------|
| Qwen3.5-0.8B | 0.00 | 0/20 |
| Qwen3.5-2B | **0.85** | 17/20 |
| Qwen3.5-4B | 0.50 | 10/20 |
| Qwen3.5-9B | 0.80 | 16/20 |
| Qwen3.5-27B | 0.30† | 6/20 |
| Qwen2.5-7B | 0.70 | 14/20 |

† **27B:** budget/API stopped part of the run; unrated problems count as fail → **lower bound**.

**Why HumanEval differs from paper QA:** execution is **binary**; failure modes are **syntax / logic**, not missing entities. It stresses the **PythonCoderTool** loop more than retrieval.

**Condensed per-tier analysis**

- **0.8B:** Often natural-language “summaries” instead of a function body—insufficient capacity for structured code under the same AgentFlow prompt as larger models.  
- **2B:** Strong formatting; failures cluster on **spec misreads** (e.g. edge cases, semantic nuance like dedupe vs elimination).  
- **4B:** Passes many string tasks; fails more on **numeric / state** bugs and occasional **indent/syntax** issues.  
- **9B:** Similar error profile to 4B but **more multi-turn refinement** in logs.  
- **27B:** Too few completed items for a clean scaling conclusion—rerun with budget.

**Per-problem matrix** (✓ / ✗ / —):

| # | Function | 0.8B | 2B | 4B | 9B | 27B |
|---|----------|------|----|----|----|-----|
| 1 | `strlen` | ✗ | ✓ | ✓ | ✓ | ✓ |
| 2 | `below_zero` | ✗ | ✓ | ✗ | ✗ | ✗ |
| 3 | `has_close_elements` | ✗ | ✓ | ✗ | ✓ | ✗ |
| 4 | `flip_case` | ✗ | ✓ | ✓ | ✓ | ✗ |
| 5 | `sum_product` | ✗ | ✓ | ✗ | ✗ | ✗ |
| 6 | `filter_by_substring` | ✗ | ✓ | ✓ | ✓ | ✗ |
| 7 | `largest_divisor` | ✗ | ✓ | ✗ | ✓ | ✗ |
| 8 | `mean_absolute_deviation` | ✗ | ✓ | ✗ | ✗ | ✗ |
| 9 | `get_positive` | ✗ | ✓ | ✗ | ✓ | ✓ |
| 10 | `how_many_times` | ✗ | ✓ | ✓ | ✓ | ✗ |
| 11 | `truncate_number` | ✗ | ✓ | ✓ | ✓ | ✓ |
| 12 | `all_prefixes` | ✗ | ✗ | ✓ | ✓ | ✓ |
| 13 | `concatenate` | ✗ | ✓ | ✓ | ✓ | — |
| 14 | `string_sequence` | ✗ | ✓ | ✓ | ✓ | — |
| 15 | `separate_paren_groups` | ✗ | ✗ | ✗ | ✓ | — |
| 16 | `find_closest_elements` | ✗ | ✓ | ✗ | ✓ | — |
| 17 | `longest` | ✗ | ✓ | ✗ | ✓ | — |
| 18 | `remove_duplicates` | ✗ | ✗ | ✗ | ✗ | — |
| 19 | `count_distinct_characters` | ✗ | ✓ | ✓ | ✓ | — |
| 20 | `string_xor` | ✗ | ✓ | ✓ | ✓ | — |

### Flow-GRPO NQ+Math (Phase 5)

- **Data:** ~3.8k NQ + math-style items; **200** steps full (8 smoke).  
- Training logs show **non-zero** format/accuracy rewards.  
- **Checkpoint eval:** best ~step **50** still **under baseline** on mixed held-out sample—partly because **Qwen3.5** prefers native `<think>` + free text vs strict `<answer>` tags expected by reward/eval.  
- Post-hoc eval can parse answers after `</think>`; **distribution shift** vs training target remains.

### Flow-GRPO HumanEval (Phase 6)

`train/flow_grpo_humaneval.py`: **single-turn** GRPO + **subprocess** execution (timeout)—not full AgentFlow inside each RL step.

```bash
modal run train/run_train.py --humaneval --smoke-test
modal run train/run_train.py --humaneval
```

| Stage | pass@1 (4 dev tasks) | format_rate |
|-------|----------------------|-------------|
| Baseline | 0.25 | 100% |
| **Checkpoint 50** | **0.50** | 100% |
| Others | 0.25 | 100% |

**Analysis:** Execution reward avoids the NQ+Math **tag war**; **n=4** is still too small for stable pass@1. **Next step:** load best LoRA and run **`run_humaneval.py`** on the **same 20** problems as Phase 4.

### Planner GRPO + serve + benchmarks (Phase 6b)

```bash
modal run train/run_train.py --agentflow --smoke-test
modal run train/run_train.py --agentflow
modal run train/run_train.py --serve
export GRPO_MODEL_URL=https://<endpoint>    # origin; OpenAI client may append /v1
python3.11 benchmarks/run_benchmark.py --model qwen3.5-0.8b-grpo --benchmarks all --sample_size 20
```

**Observed issues:** Training **400/400** steps can finish while **checkpoint eval** hits Modal **3 h** timeout—no final merged summary. **Dev eval (50-example mix):** baseline ~**0.365** acc; checkpoint **100** ~**0.432** (non-monotonic). **Mitigation:** increase `timeout` in `train/run_train.py`, reduce eval breadth in `flow_grpo_agentflow.py`, redeploy `serve_grpo_model.py`, rerun benchmarks.

`python3 train/run_agentflow_pipeline.py` automates train → deploy → sample benchmark (see flags).

---

## Training configuration (Flow-GRPO)

| Parameter | NQ+Math | HumanEval |
|-----------|---------|-----------|
| Base | Qwen3.5-0.8B | Qwen3.5-0.8B |
| LoRA r / α | 16 / 32 | 16 / 32 |
| LR | 1e-5 | 5e-6 |
| Max steps | 200 / 8 smoke | 100 / 8 smoke |
| Max completion len | 1024 | 512 |
| Rewards | F1 / numeric + format | exec + format |

---

## Comparison with the AgentFlow paper (qualitative)

| Aspect | Paper | This repo |
|--------|-------|-----------|
| Models | Large Qwen-class, strong baselines | API 7B + 3.5 family |
| Training | Multi-turn trajectories, vLLM, long RL | **Single-turn** GRPO (TRL), short runs, 0.8B LoRA |
| Gains on search/agentic | Reported +8–15pp style gains | **Not reproduced**—scope is reimplementation + ablation |

Our **Qwen2.5-7B** closes part of the gap on **simple search** vs paper-scale models; **multi-hop** stays ~0 without stronger retrieval + scale.

---

## Design choices (Part 2)

1. **Portkey** — all default LLM calls; no local GPU for benchmarks.  
2. **Modal** — training only; 10 min dev vs 3 h full timeouts.  
3. **Simplified Flow-GRPO** — TRL `GRPOTrainer`, rule rewards, not full paper trajectory server.  
4. **HumanEval training** — subprocess + timeout for safe execution.  
5. **DuckDuckGo** — search without extra API keys.

---

## Future improvements (Part 2)

1. **Align base model with reward** — e.g. **Qwen2.5-Instruct** variants if you need strict `<answer>` tags; or change prompts/rewards to score **post-`</think>`** text only (already partially done for eval).  
2. **More optimization steps / epochs** — NQ+Math saw ~**0.05 epochs** over 200 steps; scaling steps (and budget) helps GRPO shift the policy.  
3. **Full multi-turn trajectory RL** — paper-style rollouts need **vLLM** (or similar) in the training container + trajectory buffer + credit assignment per turn.  
4. **Larger backbone (e.g. 7B LoRA)** — Modal GPU memory permitting; matches paper’s stronger base.  
5. **Reward shaping** — sparse EM/F1; add partial credit (tool type, intermediate format).  
6. **Eval protocol** — greedy **and** pass@k / majority for GRPO checkpoints (training uses sampling).  
7. **Operational** — raise Modal **timeout**, shrink checkpoint eval matrix, finish **deploy + `run_benchmark.py`** for `qwen3.5-0.8b-grpo`.  
8. **HumanEval GRPO** — evaluate merged adapter on **20** AgentFlow problems to match Phase 4.

---

## Command reference (Part 2)

```bash
python3.11 benchmarks/run_benchmark.py --model qwen2.5-7b --benchmarks all --sample_size 20
python3.11 benchmarks/run_all_models.py --sample_size 20
python3.11 benchmarks/run_humaneval.py --models all --sample_size 20
python3.11 benchmarks/run_secbench.py --models all --sample_size 15

pip install modal && modal setup
modal run train/run_train.py --smoke-test
modal run train/run_train.py
modal run train/run_train.py --humaneval --smoke-test
modal run train/run_train.py --humaneval
modal run train/run_train.py --agentflow --smoke-test
modal run train/run_train.py --agentflow
modal run train/run_train.py --serve
```

---

## References (Part 2)

- [AgentFlow](https://arxiv.org/abs/2510.05592) (Li et al., ICLR 2026) · [code](https://github.com/lupantech/AgentFlow)  
- [TRL](https://huggingface.co/docs/trl) · [PEFT / LoRA](https://huggingface.co/docs/peft)  
- [HumanEval](https://github.com/openai/human-eval)
