# Week 06 - Polychromic Objectives (BabyAI Goto)

This week reproduces the assignment around the paper:

- [Polychromic Objectives for Reinforcement Learning](https://arxiv.org/abs/2509.25424)

Environment chosen from Table 1:

- **Goto** (BabyAI), implemented as `BabyAI-GoToObj-v0`.

## What is implemented

`train_goto.py` includes:

- **REINFORCE with value baseline**
- **PPO with GAE + clipping**
- **Poly-PPO**:
  - vine sampling (`N=8` vines from rollout states),
  - set objective with `n=4` trajectories and `M=4` sets,
  - diversity term based on distinct visited-room signatures,
  - window trick (`W=5`) to apply polychromic advantage beyond rollout state.

The defaults follow Appendix A / Table 3 from the paper where possible:

- `gamma=1.0`, `lambda=0.95`, `clip=0.2`,
- PPO epochs `2`, minibatch size `64`,
- actor lr `1e-5`, critic lr `1e-4`,
- value coef `0.5`, max grad norm `0.5`,
- vine parameters `N=8`, `n=4`, `M=4`, `p=2`, `W=5`.

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
python train_goto.py --algo all --output results/goto_replication_local.json
```

To run one method only:

```bash
python train_goto.py --algo reinforce
python train_goto.py --algo ppo
python train_goto.py --algo poly-ppo
```

## Notes on replication

- The script records:
  - `paper_table1_reference` (the target numbers),
  - `local_replication` (your run on this machine / seed / budget).
- Exact match to paper numbers may require the paper's exact pretrained checkpoint and full evaluation protocol (50 fixed configs x 3 seeds x 100 rollouts).
