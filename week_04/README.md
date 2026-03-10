# Week 04 — Monte Carlo Policy Gradient (REINFORCE)

The deliverable for this assignment is for you to create a video lecture of this material where you walk through the mathematical derivation and Pytorch code in your own words and with your own slides.

video walkthrough: https://youtu.be/EOWqPy49jo8





A from-scratch implementation of the REINFORCE algorithm (vanilla policy gradient without a baseline) applied to the CartPole-v1 environment.

## Prerequisites

```
pip install gymnasium torch
```

## Run

```
python "Monte Carlo Policy Gradient.py"
```

## Code Walkthrough

### 1. Policy Network (`Policy`)

A small two-layer neural network that represents the stochastic policy π_θ(a | s):

| Layer | Shape | Purpose |
|-------|-------|---------|
| `fc1` | obs_dim → 128 | Maps the raw state to a learned feature representation |
| `fc2` | 128 → act_dim | Produces logits over the discrete action space |

A softmax on the output of `fc2` gives a valid probability distribution over actions. For CartPole this means two probabilities: push-left and push-right.

### 2. Batch Trainer (`BatchTrainer`)

Implements the REINFORCE update rule. Each call to `step()` does the following:

**a) Compute returns-to-go.** For every episode in the batch, walk backwards through the rewards to compute the discounted return at each timestep:

```
G_t = r_t + γ · r_{t+1} + γ² · r_{t+2} + …
```

This is done efficiently with the recursion `G_t = r_t + γ · G_{t+1}`.

**b) Form the surrogate loss.** The policy gradient theorem says to ascend in the direction:

```
ĝ = (1/N) Σ_i Σ_t ∇_θ log π_θ(a_t | s_t) · G_t
```

Because PyTorch optimizers *minimize*, the code constructs the equivalent *loss*:

```
L(θ) = (1/N) Σ_i Σ_t [ −log π_θ(a_t | s_t) · G_t ]
```

Calling `loss.backward()` followed by `optimizer.step()` then performs gradient **ascent** on the expected return.

**c) Update parameters** via Adam with learning rate α.

### 3. Training Loop (`main`)

Each of the 1 000 update steps proceeds as:

1. **Collect** a batch of N = 10 episodes by rolling out the current policy in CartPole-v1.
   - At every timestep the state is fed through the policy network, an action is sampled from the resulting categorical distribution, and the log-probability + reward are stored.
2. **Update** θ with `trainer.step()` using the collected batch.
3. **Log** the average episode return for monitoring.

### Hyperparameters

| Symbol | Value | Meaning |
|--------|-------|---------|
| α | 2 × 10⁻⁴ | Adam learning rate |
| N | 10 | Episodes per policy update |
| γ | 0.98 | Discount factor |
| steps | 1 000 | Total policy updates |
| hidden_dim | 128 | Hidden layer width |

### Expected Behaviour

CartPole gives a reward of +1 every timestep the pole stays upright (max 500). Over the course of training the average return should climb from ~10–20 towards 400–500, indicating the agent has learned to balance the pole.


