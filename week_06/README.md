# Week 06 - Polychromic Objectives (BabyAI Goto)

This week reproduces the assignment around the paper:

- [Polychromic Objectives for Reinforcement Learning](https://arxiv.org/abs/2509.25424)

Environment chosen from Table 1:

- **Goto** (BabyAI), implemented as `BabyAI-GoToObj-v0`.

## What is implemented

`train_goto.py` includes:

- **Pretraining base policy (Behavior Cloning)**:
  - source is **official BabyAI demonstrations file** (`.pkl`),
  - by default, follows BabyAI naming and searches:
    - train: `demos/BabyAI-GoToObj-v0.pkl`
    - valid (optional): `demos/BabyAI-GoToObj-v0_valid.pkl`
  - you can override with `--official-demos-path` and `--official-valid-demos-path`,
  - train language-conditioned policy with action cross-entropy + entropy regularization,
  - if valid file is missing, split train demonstrations by **episode** with ratio `80/20`,
  - save checkpoint as `checkpoints/goto_pretrained.pt`.
- **REINFORCE with value baseline**
- **PPO with GAE + clipping**
- **Poly-PPO**:
  - vine sampling (`N=8` vines from rollout states),
  - set objective with `n=4` trajectories and `M=4` sets,
  - set score follows paper form:
    - `f_poly = mean_i R(tau_i) * d(s, tau_1:n)`,
    - with `R` and `d` normalized to `[0, 1]`,
  - diversity term in BabyAI uses fraction of semantically distinct trajectories
    (distinct visited-room sets),
  - advantage is assigned as **per-set shared signal**
    `A_hat = f_poly(set) - mean_j f_poly(g_j)` (not per-trajectory averaging),
  - window trick (`W=5`) to apply polychromic advantage beyond rollout state.
- **Policy architecture**:
  - image encoder uses a CNN,
  - mission text is encoded with token embedding + GRU (sequence-aware),
  - fused features are fed to actor/critic heads.

All RLFT methods (REINFORCE / PPO / Poly-PPO) start from the same pretrained checkpoint,
matching the paper's comparison setting.

The defaults follow Appendix A / Table 3 from the paper where possible:

- `gamma=1.0`, `lambda=0.95`, `clip=0.2`,
- PPO epochs `2`, minibatch size `64`,
- actor lr `1e-5`, critic lr `1e-4`,
- optimizer param groups:
  - shared encoder/trunk lr `1e-5` (`--shared-lr`),
  - policy head lr `1e-5`,
  - value head lr `1e-4`,
- episode horizon / max steps `100` (BabyAI/MiniGrid setting),
- RLFT on `50` fixed configurations (deterministic seed list),
- reward aligned to paper setting:
  - success reward `1 - 0.5 * t / H`,
  - failure reward `0`,
- value coef `0.5`, max grad norm `0.5`,
- KL penalty enabled in PPO objective (`kl_coef=0.01` by default),
- vine parameters `N=8`, `n=4`, `M=4`, `p=2`, `W=5`.
- pretraining demos split: `80/20` train/test.
- IL pretraining defaults for this GoToObj setup follow the official "big level" guidance:
  - `pretrain_batch_size=128`
  - `pretrain_lr=5e-5`

This implementation keeps these paper-style Poly-PPO hyperparameters fixed by default
and does not run automatic hyperparameter sweeps.

## Table 1 (Goto) reference targets

From Table 1 of the paper (no UCB variants):

- Pretrained policy: `(0.246, 34.2%)`
- REINFORCE: `(0.533, 73.0%)`
- PPO: `(0.406, 46.2%)`
- Poly-PPO: `(0.575, 80.2%)`

## Run

```bash
cd week_06
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run all three methods and save local replication metrics
python train_goto.py \
  --algo all \
  --official-demos-path /absolute/path/to/BabyAI-GoToObj-v0.pkl \
  --official-valid-demos-path /absolute/path/to/BabyAI-GoToObj-v0_valid.pkl \
  --output results/goto_replication_local.json
```

The script will:
1. pretrain/load a base policy checkpoint,
2. evaluate pretrained performance,
3. run RLFT for the selected algorithm(s) from that same checkpoint.

Checkpoint loading validates metadata consistency (`env_id`, `max_steps`,
`mission_vocab_size`, `hidden_dim`, and official-demos requirement). If mismatched,
rerun with `--force-pretrain`.

To run one method only:

```bash
python train_goto.py --algo reinforce
python train_goto.py --algo ppo
python train_goto.py --algo poly-ppo
```


