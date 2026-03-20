import copy
from typing import SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal, Independent
from torch.optim import Adam

NDArrayFloat = npt.NDArray[np.floating]
NDArrayBool = npt.NDArray[np.bool_]


class Policy:
    """Base policy class implementing REINFORCE-style gradient: ∇J(θ) ≈ E[∇log π(a|s) · A(s,a)]."""
    num_states: int
    opt: Adam

    def pi(self, s_t: torch.Tensor | NDArrayFloat) -> torch.distributions.Distribution:
        """Return the action distribution π(·|s_t). Subclasses define the parameterization."""
        raise NotImplementedError

    def act(self, s_t: NDArrayFloat) -> torch.Tensor:
        """Sample a_t ~ π(·|s_t)."""
        return self.pi(s_t).sample()

    def learn(self, states: NDArrayFloat, actions: NDArrayFloat,
              advantages: NDArrayFloat, entropy_coeff: float = 0.0) -> torch.Tensor:
        """Policy gradient step: minimize -E[log π(a|s) · Â(s,a)] with optional entropy bonus."""
        actions_t: torch.Tensor = torch.tensor(actions)
        advantages_t: torch.Tensor = torch.tensor(advantages)
        dist = self.pi(states)
        log_prob: torch.Tensor = dist.log_prob(actions_t)
        loss: torch.Tensor = torch.mean(-log_prob * advantages_t)
        if entropy_coeff > 0:
            loss = loss - entropy_coeff * dist.entropy().mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss


class ValueEstimator:
    """Critic network: learns V(s) by minimizing MSE against return targets."""

    def __init__(self, env: gym.Env, lr: float = 1e-2,
                 hidden_sizes: tuple[int, ...] = (64,)) -> None:
        self.num_states = env.observation_space.shape[0]
        layers: list[nn.Module] = []
        prev = self.num_states
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.V: nn.Sequential = nn.Sequential(*layers).double()
        self.opt: Adam = Adam(self.V.parameters(), lr=lr)

    def predict(self, s_t: NDArrayFloat | torch.Tensor) -> torch.Tensor:
        """V(s_t) — scalar value estimate for each state in the batch."""
        s_t_tensor: torch.Tensor = torch.as_tensor(s_t).double()
        return self.V(s_t_tensor).squeeze(-1)

    def learn(self, v_pred: torch.Tensor, returns: NDArrayFloat) -> torch.Tensor:
        """Minimize L = E[(V(s) - G_t)^2] where G_t is the (bootstrapped) return."""
        returns_t: torch.Tensor = torch.tensor(returns)
        loss: torch.Tensor = torch.mean((v_pred - returns_t) ** 2)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss


class CategoricalPolicy(Policy):
    """Actor for discrete action spaces: linear logits → softmax → Categorical(a|s)."""
    num_actions: int

    def __init__(self, env: gym.Env, lr: float = 1e-2) -> None:
        self.num_states = env.observation_space.shape[0]
        self.num_actions: int = env.action_space.n
        self.p: nn.Sequential = nn.Sequential(
            nn.Linear(self.num_states, self.num_actions),
        ).double()
        self.opt: Adam = Adam(self.p.parameters(), lr=lr)

    def pi(self, s_t: torch.Tensor | NDArrayFloat) -> Categorical:
        s_t_tensor: torch.Tensor = torch.as_tensor(s_t).double()
        logits: torch.Tensor = self.p(s_t_tensor)
        return Categorical(logits=logits)


class GaussianPolicy(Policy):
    """Actor for continuous action spaces: MLP → μ(s), learned log σ → Independent(Normal(μ, σ), 1).

    The mean network uses Tanh activations (standard for continuous control).
    log_std is a free parameter (state-independent) — one scalar per action dimension.
    """
    num_actions: int

    def __init__(self, env: gym.Env, lr: float = 3e-4,
                 hidden_sizes: tuple[int, ...] = (64, 64)) -> None:
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]

        layers: list[nn.Module] = []
        prev = self.num_states
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.Tanh()])
            prev = h
        layers.append(nn.Linear(prev, self.num_actions))
        self.mu_net: nn.Sequential = nn.Sequential(*layers).double()

        self.log_std = nn.Parameter(torch.zeros(self.num_actions, dtype=torch.float64))
        self.opt: Adam = Adam(list(self.mu_net.parameters()) + [self.log_std], lr=lr)

    def pi(self, s_t: torch.Tensor | NDArrayFloat) -> Independent:
        mu = self.mu_net(torch.as_tensor(s_t).double())
        std = self.log_std.exp().expand_as(mu)
        return Independent(Normal(mu, std), 1)


class VectorizedEnvWrapper(gym.Wrapper):
    """Runs N independent env copies in lockstep with episode return tracking."""

    def __init__(self, env: gym.Env, num_envs: int = 1) -> None:
        super().__init__(env)
        self.num_envs: int = num_envs
        self.envs: list[gym.Env] = [copy.deepcopy(env) for _ in range(num_envs)]
        self._ep_returns: NDArrayFloat = np.zeros(num_envs)
        self._completed_returns: list[float] = []

    def reset_all(self) -> NDArrayFloat:
        self._ep_returns = np.zeros(self.num_envs)
        self._completed_returns.clear()
        return np.asarray([env.reset()[0] for env in self.envs])

    def get_completed_returns(self) -> list[float]:
        """Drain and return all episode returns completed since the last call."""
        ret = list(self._completed_returns)
        self._completed_returns.clear()
        return ret

    def step(self, actions: torch.Tensor) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayBool]:
        next_states: list[npt.NDArray] = []
        rewards: list[SupportsFloat] = []
        dones: list[bool] = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            a = action.item() if action.dim() == 0 else action.detach().numpy()
            next_state, reward, terminated, truncated, _info = env.step(a)
            done: bool = terminated or truncated
            self._ep_returns[i] += reward
            if done:
                self._completed_returns.append(float(self._ep_returns[i]))
                self._ep_returns[i] = 0.0
                next_states.append(env.reset()[0])
            else:
                next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
        return (
            np.asarray(next_states),
            np.asarray(rewards),
            np.asarray(dones),
        )


def calculate_returns(rewards: NDArrayFloat, dones: NDArrayFloat, gamma: float) -> NDArrayFloat:
    """Compute discounted returns G_t = r_t + γ·(1-d_t)·G_{t+1} via backward pass.
    The (1-d_t) term zeros out the future when an episode terminates."""
    result: NDArrayFloat = np.empty_like(rewards)
    result[-1] = rewards[-1]
    for t in range(len(rewards) - 2, -1, -1):
        result[t] = rewards[t] + gamma * (1 - dones[t]) * result[t + 1]
    return result


def calculate_advantages(td_errors: NDArrayFloat, dones: NDArrayFloat, lam: float, gamma: float) -> NDArrayFloat:
    """GAE(γ,λ): Â_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}, computed as a backward recursion.
    λ=0 gives pure TD (low variance, high bias); λ=1 gives Monte Carlo (high variance, low bias)."""
    result: NDArrayFloat = np.empty_like(td_errors)
    result[-1] = td_errors[-1]
    for t in range(len(td_errors) - 2, -1, -1):
        result[t] = td_errors[t] + gamma * lam * (1 - dones[t]) * result[t + 1]
    return result


def a2c(env: VectorizedEnvWrapper, agent: Policy, value_estimator: ValueEstimator,
        gamma: float, lam: float, epochs: int, train_v_iters: int,
        rollout_traj_len: int, entropy_coeff: float = 0.0,
        verbose: bool = True) -> list[float]:
    """Advantage Actor-Critic (A2C): synchronous variant of A3C.
    Each epoch: (1) collect rollout, (2) fit critic, (3) compute GAE, (4) update actor.
    Returns per-epoch average episode returns for plotting learning curves."""

    states: NDArrayFloat = np.empty((rollout_traj_len + 1, env.num_envs, agent.num_states))
    if isinstance(env.action_space, gym.spaces.Box):
        actions: NDArrayFloat = np.empty((rollout_traj_len, env.num_envs, env.action_space.shape[0]))
    else:
        actions: NDArrayFloat = np.empty((rollout_traj_len, env.num_envs))
    rewards: NDArrayFloat = np.empty((rollout_traj_len, env.num_envs))
    dones: NDArrayFloat = np.empty((rollout_traj_len, env.num_envs))

    avg_returns: list[float] = []
    s_t: NDArrayFloat = env.reset_all()
    last_avg: float = 0.0

    for epoch in range(epochs):
        for t in range(rollout_traj_len):
            a_t: torch.Tensor = agent.act(s_t)
            s_t_next, r_t, d_t = env.step(a_t)
            states[t] = s_t
            actions[t] = a_t.detach().numpy()
            rewards[t] = r_t
            dones[t] = d_t
            s_t = s_t_next

        states[rollout_traj_len] = s_t

        V_pred_pre: NDArrayFloat = value_estimator.predict(states).detach().numpy()

        bootstrap_rewards: NDArrayFloat = rewards.copy()
        bootstrap_rewards[-1] += gamma * (1 - dones[-1]) * V_pred_pre[-1]
        returns: NDArrayFloat = calculate_returns(bootstrap_rewards, dones, gamma)

        for _i in range(train_v_iters):
            V_pred_train: torch.Tensor = value_estimator.predict(states[:-1])
            value_estimator.learn(V_pred_train, returns)

        td_errors: NDArrayFloat = rewards + gamma * (1 - dones) * V_pred_pre[1:] - V_pred_pre[:-1]
        advantages: NDArrayFloat = calculate_advantages(td_errors, dones, lam, gamma)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        agent.learn(states[:-1], actions, advantages, entropy_coeff=entropy_coeff)

        completed = env.get_completed_returns()
        if completed:
            last_avg = float(np.mean(completed))
        avg_returns.append(last_avg)

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}\tAvg Return: {last_avg:.1f}")

    env.close()
    return avg_returns


def main():
    import matplotlib.pyplot as plt
    import seaborn as sns

    env: VectorizedEnvWrapper = VectorizedEnvWrapper(
        gym.make("CartPole-v1"), num_envs=8
    )

    assert isinstance(env.observation_space, gym.spaces.Box), "This example assumes a Box observation space."
    assert isinstance(env.action_space, gym.spaces.Discrete), "This example assumes a Discrete action space."

    categorical: CategoricalPolicy = CategoricalPolicy(env, lr=1e-1)
    value_est: ValueEstimator = ValueEstimator(env, lr=1e-2)
    returns = a2c(env, categorical, value_est, gamma=0.99, lam=0.95,
                  epochs=100, train_v_iters=80, rollout_traj_len=4052)

    sns.lineplot(x=range(len(returns)), y=returns)
    plt.show()


if __name__ == "__main__":
    main()
