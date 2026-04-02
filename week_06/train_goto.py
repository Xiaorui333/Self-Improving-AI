"""Train REINFORCE, PPO, and Poly-PPO on BabyAI Goto.

This script implements:
1) REINFORCE with value baseline
2) PPO with GAE
3) Poly-PPO (vine sampling + polychromic advantage windows)

The implementation follows the key details from:
Polychromic Objectives for Reinforcement Learning (ICLR 2026).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401  # side-effect: registers BabyAI envs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch.distributions import Categorical


DEFAULT_ENV_ID = "BabyAI-GoToObj-v0"

# Mission vocabulary used in BabyAI text instructions.
MISSION_VOCAB = [
    "go",
    "to",
    "a",
    "the",
    "red",
    "green",
    "blue",
    "purple",
    "yellow",
    "grey",
    "box",
    "ball",
    "key",
    "door",
]
TOKEN_TO_ID = {tok: i + 1 for i, tok in enumerate(MISSION_VOCAB)}  # 0 is PAD/UNK
PAD_ID = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BabyAIObsWrapper(gym.ObservationWrapper):
    """Convert BabyAI dict obs into numeric tensors-friendly dict."""

    def __init__(self, env: gym.Env, max_mission_len: int = 12):
        super().__init__(env)
        self.max_mission_len = max_mission_len
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(7, 7, 3), dtype=np.uint8),
                "direction": spaces.Box(low=0.0, high=3.0, shape=(1,), dtype=np.float32),
                "mission_tokens": spaces.Box(
                    low=0,
                    high=len(TOKEN_TO_ID),
                    shape=(max_mission_len,),
                    dtype=np.int32,
                ),
            }
        )

    def _encode_mission(self, mission: str) -> np.ndarray:
        toks = mission.lower().replace(",", " ").split()
        ids = [TOKEN_TO_ID.get(t, PAD_ID) for t in toks[: self.max_mission_len]]
        if len(ids) < self.max_mission_len:
            ids.extend([PAD_ID] * (self.max_mission_len - len(ids)))
        return np.asarray(ids, dtype=np.int32)

    def observation(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        image = observation["image"].astype(np.uint8)
        direction = np.asarray([float(observation["direction"])], dtype=np.float32)
        mission_tokens = self._encode_mission(observation["mission"])
        return {
            "image": image,
            "direction": direction,
            "mission_tokens": mission_tokens,
        }


def make_env(env_id: str, seed: int | None = None) -> gym.Env:
    env = gym.make(env_id)
    env = BabyAIObsWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


class ActorCritic(nn.Module):
    def __init__(
        self,
        action_dim: int,
        mission_vocab_size: int,
        mission_embed_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.mission_emb = nn.Embedding(mission_vocab_size, mission_embed_dim, padding_idx=0)

        image_dim = 7 * 7 * 3
        direction_dim = 1
        mission_dim = mission_embed_dim
        trunk_in = image_dim + direction_dim + mission_dim

        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def encode_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        image = obs["image"].float() / 255.0
        image = image.view(image.shape[0], -1)
        direction = obs["direction"].float()
        mission_tokens = obs["mission_tokens"].long()
        mission_emb = self.mission_emb(mission_tokens)
        mission_emb = mission_emb.mean(dim=1)
        x = torch.cat([image, direction, mission_emb], dim=1)
        return self.trunk(x)

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode_obs(obs)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def act(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value, entropy

    def evaluate_actions(
        self, obs: dict[str, torch.Tensor], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value


def obs_to_tensor(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "image": torch.from_numpy(obs["image"]).unsqueeze(0).to(device),
        "direction": torch.from_numpy(obs["direction"]).unsqueeze(0).to(device),
        "mission_tokens": torch.from_numpy(obs["mission_tokens"]).unsqueeze(0).to(device),
    }


def batch_obs(obs_list: list[dict[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "image": torch.from_numpy(np.stack([o["image"] for o in obs_list], axis=0)).to(device),
        "direction": torch.from_numpy(np.stack([o["direction"] for o in obs_list], axis=0)).to(device),
        "mission_tokens": torch.from_numpy(
            np.stack([o["mission_tokens"] for o in obs_list], axis=0)
        ).to(device),
    }


@dataclass
class StepRecord:
    obs: dict[str, np.ndarray]
    action: int
    reward: float
    done: bool
    log_prob: float
    value: float
    poly_adv_override: float | None = None


@dataclass
class Episode:
    steps: list[StepRecord]
    success: bool
    episodic_reward: float
    reset_seed: int
    visited_rooms: set[tuple[int, int]]


def room_signature(env_unwrapped: Any, pos: tuple[int, int]) -> tuple[int, int]:
    # BabyAI room size is usually 8. We infer robustly and fallback to 8.
    room_size = getattr(env_unwrapped, "room_size", 8)
    x, y = pos
    return (int(x) // int(room_size), int(y) // int(room_size))


def rollout_episode(
    env: gym.Env,
    model: ActorCritic,
    device: torch.device,
    gamma: float,
    max_steps: int = 200,
    reset_seed: int | None = None,
    action_prefix: list[int] | None = None,
) -> Episode:
    if reset_seed is None:
        reset_seed = random.randint(0, 1_000_000_000)
    obs, _ = env.reset(seed=reset_seed)
    unwrapped = env.unwrapped

    visited_rooms: set[tuple[int, int]] = set()
    if hasattr(unwrapped, "agent_pos"):
        visited_rooms.add(room_signature(unwrapped, tuple(unwrapped.agent_pos)))

    steps: list[StepRecord] = []
    done = False
    t = 0

    if action_prefix is None:
        action_prefix = []

    for a in action_prefix:
        next_obs, reward, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        if hasattr(unwrapped, "agent_pos"):
            visited_rooms.add(room_signature(unwrapped, tuple(unwrapped.agent_pos)))
        if done:
            obs = next_obs
            break
        obs = next_obs
        t += 1
        if t >= max_steps:
            done = True
            break

    while not done and t < max_steps:
        obs_t = obs_to_tensor(obs, device)
        with torch.no_grad():
            action_t, log_prob_t, value_t, _ = model.act(obs_t)
        action = int(action_t.item())
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps.append(
            StepRecord(
                obs=copy.deepcopy(obs),
                action=action,
                reward=float(reward),
                done=done,
                log_prob=float(log_prob_t.item()),
                value=float(value_t.item()),
                poly_adv_override=None,
            )
        )
        if hasattr(unwrapped, "agent_pos"):
            visited_rooms.add(room_signature(unwrapped, tuple(unwrapped.agent_pos)))
        obs = next_obs
        t += 1

    episodic_reward = float(sum(s.reward for s in steps))
    # BabyAI success gives reward > 0 near end.
    success = episodic_reward > 0.0
    return Episode(
        steps=steps,
        success=success,
        episodic_reward=episodic_reward,
        reset_seed=reset_seed,
        visited_rooms=visited_rooms,
    )


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    ret = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        ret[i] = running
    return ret


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> np.ndarray:
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv


def flatten_episodes_for_pg(episodes: list[Episode], gamma: float) -> dict[str, Any]:
    obs_list: list[dict[str, np.ndarray]] = []
    actions: list[int] = []
    advantages: list[float] = []
    returns: list[float] = []
    for ep in episodes:
        rewards = [s.reward for s in ep.steps]
        if not rewards:
            continue
        ret = discounted_returns(rewards, gamma)
        for i, step in enumerate(ep.steps):
            obs_list.append(step.obs)
            actions.append(step.action)
            returns.append(float(ret[i]))
    return {
        "obs": obs_list,
        "actions": np.asarray(actions, dtype=np.int64),
        "returns": np.asarray(returns, dtype=np.float32),
    }


def flatten_episodes_for_ppo(episodes: list[Episode], gamma: float, lam: float) -> dict[str, Any]:
    obs_list: list[dict[str, np.ndarray]] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    returns: list[float] = []
    advantages: list[float] = []

    for ep in episodes:
        if not ep.steps:
            continue
        rewards = np.asarray([s.reward for s in ep.steps], dtype=np.float32)
        values = np.asarray([s.value for s in ep.steps], dtype=np.float32)
        dones = np.asarray([1.0 if s.done else 0.0 for s in ep.steps], dtype=np.float32)
        adv = compute_gae(rewards, values, dones, gamma, lam, last_value=0.0)
        ret = adv + values

        for i, step in enumerate(ep.steps):
            obs_list.append(step.obs)
            actions.append(step.action)
            old_log_probs.append(step.log_prob)
            returns.append(float(ret[i]))
            advantages.append(float(adv[i]))

    advantages_np = np.asarray(advantages, dtype=np.float32)
    if len(advantages_np) > 1:
        advantages_np = (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)

    return {
        "obs": obs_list,
        "actions": np.asarray(actions, dtype=np.int64),
        "old_log_probs": np.asarray(old_log_probs, dtype=np.float32),
        "returns": np.asarray(returns, dtype=np.float32),
        "advantages": advantages_np,
    }


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    device: torch.device,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    kl_coef: float,
    ppo_epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
) -> None:
    n = len(batch["actions"])
    if n == 0:
        return
    obs = batch["obs"]
    actions = torch.from_numpy(batch["actions"]).to(device)
    old_log_probs = torch.from_numpy(batch["old_log_probs"]).to(device)
    returns = torch.from_numpy(batch["returns"]).to(device)
    advantages = torch.from_numpy(batch["advantages"]).to(device)

    indices = np.arange(n)
    for _ in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, n, minibatch_size):
            mb_idx = indices[start : start + minibatch_size]
            mb_obs = batch_obs([obs[i] for i in mb_idx], device)
            mb_actions = actions[mb_idx]
            mb_old_lp = old_log_probs[mb_idx]
            mb_ret = returns[mb_idx]
            mb_adv = advantages[mb_idx]

            new_log_prob, entropy, value = model.evaluate_actions(mb_obs, mb_actions)
            ratio = torch.exp(new_log_prob - mb_old_lp)
            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = F.mse_loss(value, mb_ret)
            entropy_loss = -entropy.mean()

            kl = (mb_old_lp - new_log_prob).mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss + kl_coef * kl
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()


def reinforce_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    episodes: list[Episode],
    device: torch.device,
    gamma: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> None:
    data = flatten_episodes_for_pg(episodes, gamma=gamma)
    if len(data["actions"]) == 0:
        return
    obs = batch_obs(data["obs"], device)
    actions = torch.from_numpy(data["actions"]).to(device)
    returns = torch.from_numpy(data["returns"]).to(device)

    log_prob, entropy, values = model.evaluate_actions(obs, actions)
    advantages = returns - values.detach()
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_loss = -(log_prob * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    entropy_loss = -entropy.mean()
    loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()


def choose_rollout_indices(ep_len: int, p: int) -> list[int]:
    if ep_len <= 1:
        return [0]
    idxs = []
    for i in range(1, p + 1):
        idx = int((i * ep_len) / (p + 1))
        idx = min(max(idx, 0), ep_len - 1)
        idxs.append(idx)
    return sorted(set(idxs))


def poly_diversity(visited_room_sets: list[set[tuple[int, int]]]) -> float:
    signatures = {frozenset(v) for v in visited_room_sets}
    return float(len(signatures)) / float(len(visited_room_sets))


def poly_ppo_collect(
    env_id: str,
    model: ActorCritic,
    device: torch.device,
    gamma: float,
    n_vines: int = 8,
    set_size: int = 4,
    num_sets: int = 4,
    rollout_states_per_seed: int = 2,
    window_w: int = 5,
    max_steps: int = 200,
) -> list[Episode]:
    """Collect episodes and assign Poly-PPO advantages using vine sampling."""
    env = make_env(env_id)
    seed_episodes: list[Episode] = []
    for _ in range(n_vines):
        seed_ep = rollout_episode(env, model, device, gamma=gamma, max_steps=max_steps)
        seed_episodes.append(seed_ep)

    all_eps: list[Episode] = list(seed_episodes)

    # Branch from p rollout states per seed trajectory.
    for seed_ep in seed_episodes:
        if len(seed_ep.steps) == 0:
            continue
        rollout_idxs = choose_rollout_indices(len(seed_ep.steps), rollout_states_per_seed)
        for ridx in rollout_idxs:
            prefix_actions = [s.action for s in seed_ep.steps[:ridx]]
            branch_eps: list[Episode] = []
            for _ in range(n_vines):
                branch_ep = rollout_episode(
                    env,
                    model,
                    device,
                    gamma=gamma,
                    max_steps=max_steps,
                    reset_seed=seed_ep.reset_seed,
                    action_prefix=prefix_actions,
                )
                branch_eps.append(branch_ep)
                all_eps.append(branch_ep)

            if len(branch_eps) < set_size:
                continue

            # Build M trajectory sets and compute set scores.
            set_indices: list[list[int]] = []
            set_scores: list[float] = []
            all_idx = list(range(len(branch_eps)))
            for _ in range(num_sets):
                chosen = random.sample(all_idx, set_size)
                selected = [branch_eps[i] for i in chosen]
                returns = [
                    discounted_returns([s.reward for s in ep.steps], gamma=gamma)[0] if ep.steps else 0.0
                    for ep in selected
                ]
                diversity = poly_diversity([ep.visited_rooms for ep in selected])
                score = float(np.mean(returns)) * diversity
                set_indices.append(chosen)
                set_scores.append(score)

            baseline = float(np.mean(set_scores))

            # Per-trajectory poly advantage from sets that include it.
            per_traj_poly_adv = np.zeros(len(branch_eps), dtype=np.float32)
            per_traj_count = np.zeros(len(branch_eps), dtype=np.int32)
            for chosen, score in zip(set_indices, set_scores):
                for idx in chosen:
                    per_traj_poly_adv[idx] += float(score - baseline)
                    per_traj_count[idx] += 1
            for i in range(len(branch_eps)):
                if per_traj_count[i] > 0:
                    per_traj_poly_adv[i] /= float(per_traj_count[i])

            for i, ep in enumerate(branch_eps):
                poly_adv = float(per_traj_poly_adv[i])
                for t, step in enumerate(ep.steps):
                    if ridx <= t <= ridx + window_w:
                        step.poly_adv_override = poly_adv

    env.close()
    return all_eps


def flatten_poly_episodes_for_ppo(episodes: list[Episode], gamma: float, lam: float) -> dict[str, Any]:
    """Build PPO batch and replace selected windows by poly advantages."""
    obs_list: list[dict[str, np.ndarray]] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    returns: list[float] = []
    advantages: list[float] = []

    for ep in episodes:
        if not ep.steps:
            continue
        rewards = np.asarray([s.reward for s in ep.steps], dtype=np.float32)
        raw_values = np.asarray([s.value for s in ep.steps], dtype=np.float32)
        dones = np.asarray([1.0 if s.done else 0.0 for s in ep.steps], dtype=np.float32)
        adv = compute_gae(rewards, raw_values, dones, gamma, lam, last_value=0.0)
        ret = adv + raw_values

        for i, step in enumerate(ep.steps):
            obs_list.append(step.obs)
            actions.append(step.action)
            adv_i = (
                float(step.poly_adv_override)
                if step.poly_adv_override is not None
                else float(adv[i])
            )
            advantages.append(adv_i)
            returns.append(float(ret[i]))
            old_log_probs.append(step.log_prob)

    advantages_np = np.asarray(advantages, dtype=np.float32)
    if len(advantages_np) > 1:
        advantages_np = (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)

    return {
        "obs": obs_list,
        "actions": np.asarray(actions, dtype=np.int64),
        "old_log_probs": np.asarray(old_log_probs, dtype=np.float32),
        "returns": np.asarray(returns, dtype=np.float32),
        "advantages": advantages_np,
    }


def evaluate_policy(
    env_id: str,
    model: ActorCritic,
    device: torch.device,
    n_episodes: int = 100,
    max_steps: int = 200,
    base_seed: int = 10_000,
) -> tuple[float, float]:
    env = make_env(env_id)
    rewards = []
    successes = []
    with torch.no_grad():
        for i in range(n_episodes):
            ep = rollout_episode(
                env,
                model,
                device,
                gamma=1.0,
                max_steps=max_steps,
                reset_seed=base_seed + i,
            )
            rewards.append(ep.episodic_reward)
            successes.append(1.0 if ep.success else 0.0)
    env.close()
    return float(np.mean(rewards)), 100.0 * float(np.mean(successes))


def train_reinforce(
    env_id: str,
    model: ActorCritic,
    device: torch.device,
    updates: int,
    episodes_per_update: int,
    gamma: float,
    lr_actor: float,
    lr_critic: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    max_steps: int,
) -> None:
    optimizer = torch.optim.Adam(
        [
            {"params": list(model.trunk.parameters()) + list(model.policy_head.parameters()), "lr": lr_actor},
            {"params": list(model.value_head.parameters()) + list(model.mission_emb.parameters()), "lr": lr_critic},
        ]
    )
    env = make_env(env_id)
    for upd in range(1, updates + 1):
        episodes = [
            rollout_episode(env, model, device, gamma=gamma, max_steps=max_steps)
            for _ in range(episodes_per_update)
        ]
        reinforce_update(
            model,
            optimizer,
            episodes,
            device=device,
            gamma=gamma,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
        )
        if upd % max(1, updates // 10) == 0:
            avg_r = np.mean([e.episodic_reward for e in episodes])
            succ = 100.0 * np.mean([1.0 if e.success else 0.0 for e in episodes])
            print(f"[REINFORCE] update {upd}/{updates}  avg_reward={avg_r:.3f}  success={succ:.1f}%")
    env.close()


def train_ppo(
    env_id: str,
    model: ActorCritic,
    device: torch.device,
    updates: int,
    episodes_per_update: int,
    gamma: float,
    lam: float,
    lr_actor: float,
    lr_critic: float,
    value_coef: float,
    entropy_coef: float,
    kl_coef: float,
    clip_eps: float,
    ppo_epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
    max_steps: int,
) -> None:
    optimizer = torch.optim.Adam(
        [
            {"params": list(model.trunk.parameters()) + list(model.policy_head.parameters()), "lr": lr_actor},
            {"params": list(model.value_head.parameters()) + list(model.mission_emb.parameters()), "lr": lr_critic},
        ]
    )
    env = make_env(env_id)
    for upd in range(1, updates + 1):
        episodes = [
            rollout_episode(env, model, device, gamma=gamma, max_steps=max_steps)
            for _ in range(episodes_per_update)
        ]
        batch = flatten_episodes_for_ppo(episodes, gamma=gamma, lam=lam)
        ppo_update(
            model=model,
            optimizer=optimizer,
            batch=batch,
            device=device,
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            kl_coef=kl_coef,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            max_grad_norm=max_grad_norm,
        )
        if upd % max(1, updates // 10) == 0:
            avg_r = np.mean([e.episodic_reward for e in episodes])
            succ = 100.0 * np.mean([1.0 if e.success else 0.0 for e in episodes])
            print(f"[PPO] update {upd}/{updates}  avg_reward={avg_r:.3f}  success={succ:.1f}%")
    env.close()


def train_poly_ppo(
    env_id: str,
    model: ActorCritic,
    device: torch.device,
    updates: int,
    gamma: float,
    lam: float,
    lr_actor: float,
    lr_critic: float,
    value_coef: float,
    entropy_coef: float,
    kl_coef: float,
    clip_eps: float,
    ppo_epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
    max_steps: int,
    n_vines: int = 8,
    set_size: int = 4,
    num_sets: int = 4,
    rollout_states_per_seed: int = 2,
    window_w: int = 5,
) -> None:
    optimizer = torch.optim.Adam(
        [
            {"params": list(model.trunk.parameters()) + list(model.policy_head.parameters()), "lr": lr_actor},
            {"params": list(model.value_head.parameters()) + list(model.mission_emb.parameters()), "lr": lr_critic},
        ]
    )
    for upd in range(1, updates + 1):
        episodes = poly_ppo_collect(
            env_id=env_id,
            model=model,
            device=device,
            gamma=gamma,
            n_vines=n_vines,
            set_size=set_size,
            num_sets=num_sets,
            rollout_states_per_seed=rollout_states_per_seed,
            window_w=window_w,
            max_steps=max_steps,
        )
        batch = flatten_poly_episodes_for_ppo(episodes, gamma=gamma, lam=lam)
        ppo_update(
            model=model,
            optimizer=optimizer,
            batch=batch,
            device=device,
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            kl_coef=kl_coef,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            max_grad_norm=max_grad_norm,
        )
        if upd % max(1, updates // 10) == 0:
            avg_r = np.mean([e.episodic_reward for e in episodes])
            succ = 100.0 * np.mean([1.0 if e.success else 0.0 for e in episodes])
            print(f"[Poly-PPO] update {upd}/{updates}  avg_reward={avg_r:.3f}  success={succ:.1f}%")


def build_model(env_id: str, device: torch.device) -> ActorCritic:
    env = make_env(env_id)
    action_dim = int(env.action_space.n)
    env.close()
    model = ActorCritic(
        action_dim=action_dim,
        mission_vocab_size=len(TOKEN_TO_ID) + 1,
        mission_embed_dim=32,
        hidden_dim=256,
    )
    return model.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Goto replication: REINFORCE, PPO, Poly-PPO")
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument("--algo", choices=["reinforce", "ppo", "poly-ppo", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--episodes-per-update", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default="results/goto_replication_local.json")

    # Paper-inspired defaults (Table 3 / Appendix A).
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    # Poly-PPO specific
    parser.add_argument("--n-vines", type=int, default=8)
    parser.add_argument("--set-size", type=int, default=4)
    parser.add_argument("--num-sets", type=int, default=4)
    parser.add_argument("--rollout-states-per-seed", type=int, default=2)
    parser.add_argument("--window-w", type=int, default=5)
    return parser.parse_args()


def run_single(algo: str, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    set_seed(args.seed)
    model = build_model(args.env_id, device)

    if algo == "reinforce":
        train_reinforce(
            env_id=args.env_id,
            model=model,
            device=device,
            updates=args.updates,
            episodes_per_update=args.episodes_per_update,
            gamma=args.gamma,
            lr_actor=args.actor_lr,
            lr_critic=args.critic_lr,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            max_steps=args.max_steps,
        )
    elif algo == "ppo":
        train_ppo(
            env_id=args.env_id,
            model=model,
            device=device,
            updates=args.updates,
            episodes_per_update=args.episodes_per_update,
            gamma=args.gamma,
            lam=args.lam,
            lr_actor=args.actor_lr,
            lr_critic=args.critic_lr,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            kl_coef=args.kl_coef,
            clip_eps=args.clip_eps,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            max_grad_norm=args.max_grad_norm,
            max_steps=args.max_steps,
        )
    elif algo == "poly-ppo":
        train_poly_ppo(
            env_id=args.env_id,
            model=model,
            device=device,
            updates=args.updates,
            gamma=args.gamma,
            lam=args.lam,
            lr_actor=args.actor_lr,
            lr_critic=args.critic_lr,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            kl_coef=args.kl_coef,
            clip_eps=args.clip_eps,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            max_grad_norm=args.max_grad_norm,
            max_steps=args.max_steps,
            n_vines=args.n_vines,
            set_size=args.set_size,
            num_sets=args.num_sets,
            rollout_states_per_seed=args.rollout_states_per_seed,
            window_w=args.window_w,
        )
    else:
        raise ValueError(f"Unsupported algo: {algo}")

    avg_reward, success_rate = evaluate_policy(
        env_id=args.env_id,
        model=model,
        device=device,
        n_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        base_seed=args.seed + 123456,
    )
    return {
        "avg_reward": round(avg_reward, 3),
        "success_rate_pct": round(success_rate, 1),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else "cpu")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paper_table1_goto = {
        "pretrained_policy": {"avg_reward": 0.246, "success_rate_pct": 34.2},
        "reinforce": {"avg_reward": 0.533, "success_rate_pct": 73.0},
        "ppo": {"avg_reward": 0.406, "success_rate_pct": 46.2},
        "poly-ppo": {"avg_reward": 0.575, "success_rate_pct": 80.2},
    }

    if args.algo == "all":
        algos = ["reinforce", "ppo", "poly-ppo"]
    else:
        algos = [args.algo]

    local_results: dict[str, dict[str, float]] = {}
    for algo in algos:
        print(f"\n=== Training {algo} on {args.env_id} ===")
        local_results[algo] = run_single(algo, args, device)
        print(
            f"[{algo}] eval avg_reward={local_results[algo]['avg_reward']:.3f}, "
            f"success={local_results[algo]['success_rate_pct']:.1f}%"
        )

    payload = {
        "environment": "Goto (BabyAI)",
        "env_id": args.env_id,
        "paper_table1_reference": paper_table1_goto,
        "local_replication": local_results,
        "config": vars(args),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
