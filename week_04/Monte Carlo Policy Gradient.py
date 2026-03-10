import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    """
    Categorical policy network: π_θ(a | s)

    For discrete action spaces:
      - input:  s_t  ∈ R^{obs_dim}
      - output: π_θ(. | s_t) ∈ Δ^{act_dim}  (probabilities over act_dim actions)
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        # θ includes all weights/biases in these layers
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, act_dim)

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        """
        s_t -> π_θ(. | s_t)
        """
        x = F.relu(self.fc1(s_t))               # feature transform (part of θ)
        probs = F.softmax(self.fc2(x), dim=-1)  # π_θ(a | s_t) for all a
        return probs


class BatchTrainer:
    """
    Vanilla policy gradient (REINFORCE) without baseline, batched over N episodes.

    Algorithm (no baseline):
      Collect N episodes τ_i ~ π_θ
      For each episode i and timestep t:
        G_t^{(i)} = sum_{k=t..T-1} γ^{k-t} r_k^{(i)}            (return-to-go)
      Gradient estimate:
        ĝ = (1/N) * sum_i sum_t ∇_θ log π_θ(a_t^{(i)}|s_t^{(i)}) * G_t^{(i)}
      Update (gradient ascent):
        θ ← θ + α ĝ

    In PyTorch we typically do gradient *descent* on:
      L(θ) = (1/N) * sum_i sum_t [ - log π_θ(a_t|s_t) * G_t ]
    because:  ∇_θ L(θ) = -(ĝ)  =>  θ ← θ - α ∇_θ L = θ + α ĝ
    """
    def __init__(self, pi: nn.Module, learning_rate: float, gamma: float):
        self.pi = pi
        self.optimizer = optim.Adam(self.pi.parameters(), lr=learning_rate)  # α (via Adam)
        self.gamma = gamma                                                   # γ

    def step(self, episodes: list):
        """
        episodes: list of episodes, where each episode is a list of (log_prob_t, r_t)
          log_prob_t = log π_θ(a_t | s_t)
          r_t        = reward at timestep t

        Computes:
          L(θ) = (1/N) * Σ_i Σ_t [ - log π_θ(a_t^{(i)}|s_t^{(i)}) * G_t^{(i)} ]
        Then: θ ← θ - α ∇_θ L(θ)  (equivalently θ ← θ + α ĝ)
        """
        self.optimizer.zero_grad()  # clear old gradients

        total_loss = 0.0
        N = len(episodes)           # N = number of sampled trajectories (episodes) in the batch

        # Σ_i over episodes τ_i ~ π_θ
        for ep in episodes:
            # ep = [(log π_θ(a_0|s_0), r_0), ..., (log π_θ(a_{T-1}|s_{T-1}), r_{T-1})]

            # Compute return-to-go G_t by backward recursion:
            #   G_t = r_t + γ G_{t+1}, with G_T = 0
            R = 0.0                 # will hold G_{t+1} as we iterate backwards
            returns = []            # will hold [G_{T-1}, ..., G_0] (temporarily)

            for (log_prob_t, r_t) in reversed(ep):
                # R <- r_t + γ * R  ==  G_t <- r_t + γ * G_{t+1}
                R = r_t + self.gamma * R
                returns.append(R)

            returns.reverse()       # now returns[t] = G_t aligned with forward time

            # Σ_t over timesteps in this episode
            for (log_prob_t, _), G_t in zip(ep, returns):
                # Per-timestep contribution to L(θ):
                #   ℓ_t(θ) = - log π_θ(a_t|s_t) * G_t
                total_loss = total_loss + (-log_prob_t * G_t)

        # (1/N) * Σ_i Σ_t ...
        loss = total_loss / N

        # Compute ∇_θ L(θ)
        loss.backward()

        # Gradient step: θ ← θ - α ∇_θ L(θ)  (equivalently ascent on J)
        self.optimizer.step()


def main():
    learning_rate = 2e-4  # α (via Adam)
    batch_size = 10       # N: episodes per update
    gamma = 0.98          # γ
    steps = 1000

    env = gym.make("CartPole-v1")

    # Derive dimensions from the environment (no hard-coded 4,2)
    assert isinstance(env.observation_space, gym.spaces.Box), "This example assumes a Box observation space."
    assert isinstance(env.action_space, gym.spaces.Discrete), "This example assumes a Discrete action space."

    obs_dim = int(env.observation_space.shape[0])  # dim(s_t)
    act_dim = int(env.action_space.n)              # number of discrete actions

    # Initialize π_θ
    pi = Policy(obs_dim=obs_dim, act_dim=act_dim)

    trainer = BatchTrainer(pi, learning_rate=learning_rate, gamma=gamma)

    # Repeat policy updates
    for step in range(steps):
        batch_episodes = []
        score_sum = 0.0

        # Collect N trajectories: τ_i ~ π_θ
        for _ in range(batch_size):
            s_t, _ = env.reset()
            ep = []  # episode storage: [(log π_θ(a_t|s_t), r_t), ...]

            while True:
                # Convert s_t to tensor (state input to π_θ)
                s_tensor = torch.from_numpy(s_t).float()

                # π_θ(. | s_t)
                probs = pi(s_tensor)

                # Sample action a_t ~ π_θ(. | s_t)
                dist = Categorical(probs)
                a_t = dist.sample()

                # Environment transition: (s_{t+1}, r_t) ~ P(. | s_t, a_t)
                s_next, r_t, terminated, truncated, _ = env.step(a_t.item())

                # Store log π_θ(a_t | s_t) and reward r_t for REINFORCE:
                #   needed to form Σ_t ∇_θ log π_θ(a_t|s_t) * G_t
                log_prob_t = dist.log_prob(a_t)  # log π_θ(a_t | s_t)
                ep.append((log_prob_t, r_t))

                s_t = s_next
                score_sum += r_t

                if terminated or truncated:
                    break

            batch_episodes.append(ep)

        # Compute REINFORCE loss and update θ once per batch
        trainer.step(batch_episodes)

        # Reporting: average episodic return in the batch (CartPole has r_t=1 each step)
        avg_return = score_sum / batch_size
        print(f"update {step}, avg return per episode: {avg_return:.1f}")

    env.close()


if __name__ == "__main__":
    main()