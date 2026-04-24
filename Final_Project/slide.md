---
marp: true
theme: default
paginate: true
size: 16:9
header: Self-Improving AI — Final Project
footer: TinyZero + AgentFlow + Flow-GRPO
---

<!-- ~4 min @ ~30–40 s/slide: title → setup → TinyZero → AgentFlow → QA → HumanEval → GRPO → takeaway -->

# Self-Improving AI — Final Project

**TinyZero + AgentFlow reimplementation + Flow-GRPO**

_Portkey · DeepInfra · Modal (L40S)_

<!-- 20 s: state two parts + one sentence goal -->

---

# What this project is

| Part | Focus |
|------|--------|
| **1 — TinyZero** | Countdown arithmetic task: **LoRA + GRPO** on Modal (sanity check for TRL pipeline) |
| **2 — AgentFlow** | **Planner → Executor → Verifier** agents + 10 **QA** benchmarks + **HumanEval** (code) |
| **Training** | **Simplified Flow-GRPO**: single-turn policy, rule-based rewards (vs full multi-turn RL in paper) |

**Goal:** Reproduce benchmark trends, add a **non-QA** task (code), stress-test **GRPO** on the smallest Qwen3.5.

<!-- 40 s: don’t read table verbatim; emphasize “two tracks + GRPO ablation” -->

---

# Part 1 — TinyZero (warmup)

- **Task:** Use each number once to hit a target; reward = overlap + exact equation + **format** (reasoning + answer tags).
- **Setup:** Qwen2.5-**0.5B** + LoRA · GRPO · **200 steps** on L40S.

**Takeaway**

| Learned | Not learned |
|---------|-------------|
| Format → **100%** · number overlap → **~0.99** | **exact_acc ≈ 0%** — reward hack: right numbers, wrong value |

<!-- 35 s: “GRPO works mechanically; sparse exact reward bites small models” -->

---

# Part 2 — AgentFlow stack

```
Inference (benchmarks)          Training (Modal)
Portkey → DeepInfra      vs.     LoRA + GRPO (flow_grpo*.py)
```

- **Tools:** Python sandbox, DuckDuckGo, Wikipedia.
- **Extra benchmark:** **HumanEval** (n=20, seed 42) — **pass@1** by **execution**, full agent loop.

<!-- 30 s: one diagram mentally—API eval vs GPU train -->

---

# QA results (n ≈ 20 / task)

- **Best overall:** **Qwen2.5-7B** on single-hop search (e.g. Bamboogle **0.60 EM**).
- **Qwen3.5 scaling:** **27B** best **in-family** on Bamboogle (**0.40 EM**); **not** monotone (e.g. **2B** strong on HotpotQA).
- **Hard wall:** **2Wiki, Musique, GAIA** → ~**0** — multi-hop + agentic needs more than our stack/size.

*AIME etc.: high variance on small n; multi-hop gap dominates the story.*

<!-- 45 s: three bullets only; skip numbers if short on time -->

---

# HumanEval + AgentFlow (pass@1, n=20)

| Model | pass@1 |
|-------|--------|
| 3.5-**0.8B** | **0.00** |
| 3.5-**2B** | **0.85** |
| 3.5-4B | 0.50 |
| 3.5-9B | 0.80 |
| 2.5-7B | 0.70 |
| 3.5-27B | 0.30† |

† Partial / budget — lower bound.

**Insight:** Large jump **0.8B→2B**; code stresses **tool + syntax** vs string-match QA; scaling **non-monotonic** within 3.5.

<!-- 45 s: table is the slide; one spoken line on 0.8B failure mode -->

---

# Flow-GRPO on Qwen3.5-0.8B

| Setting | Finding |
|---------|---------|
| **NQ + Math** (200 steps) | Training rewards ↑ · **checkpoints &lt; baseline** on held-out — **format clash** (native reasoning vs answer-tag rewards) |
| **HumanEval** (100 steps) | Exec reward stable; **n=4** dev pass@1: **0.25 → 0.50** best ckpt — **noisy**; fair test = **same 20** AgentFlow eval **with LoRA** |
| **Planner GRPO → 10 benches** | Train can finish; **long checkpoint eval** may **timeout** before full benchmark sweep |

**vs paper:** We use **single-turn TRL GRPO**, not vLLM + full trajectory RL — **gains not reproduced**; scope = **reimplementation + ablation**.

<!-- 50 s: this is the “methods result” slide—speak slowly -->

---

# Takeaway

1. **Agents + APIs** reproduce **directional** QA gaps (easy search vs multi-hop).
2. **HumanEval** isolates **code**; **0.8B** is a strong candidate for **RL** (0% → headroom).
3. **GRPO** is easy to run; **alignment** of base model **format** with **reward** matters as much as algorithm choice.
4. **Next:** more **epochs / 7B** / **multi-turn** training — or **eval LoRA on 20-problem HumanEval** for a clean score.

**Repo:** `Final_Project/` · `README.md` for tables & commands

<!-- 30 s: end on “headroom + format alignment” -->

---

# References

- [AgentFlow](https://arxiv.org/abs/2510.05592) (ICLR 2026) · [TinyZero](https://github.com/Jiayi-Pan/TinyZero)
- [TRL GRPO](https://huggingface.co/docs/trl/main/en/grpo_trainer) · [HumanEval](https://github.com/openai/human-eval)
