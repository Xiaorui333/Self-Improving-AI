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
import hashlib
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blosc
import gymnasium as gym
import minigrid  # noqa: F401  # side-effect: registers BabyAI envs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch.distributions import Categorical


DEFAULT_ENV_ID = "BabyAI-GoToObj-v0"
MODEL_MISSION_EMBED_DIM = 32
MODEL_HIDDEN_DIM = 256

# Use a hashed token space so mission encoding is not constrained by a fixed word list.
MISSION_HASH_VOCAB_SIZE = 4096
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
                    high=MISSION_HASH_VOCAB_SIZE,
                    shape=(max_mission_len,),
                    dtype=np.int32,
                ),
            }
        )

    @staticmethod
    def _hash_token(token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        # Reserve 0 for padding.
        return int(digest, 16) % MISSION_HASH_VOCAB_SIZE + 1

    def _encode_mission(self, mission: str) -> np.ndarray:
        toks = mission.lower().replace(",", " ").split()
        ids = [self._hash_token(t) for t in toks[: self.max_mission_len]]
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
        self.mission_vocab_size = mission_vocab_size
        self.hidden_dim = hidden_dim
        self.mission_embed_dim = mission_embed_dim
        self.mission_emb = nn.Embedding(mission_vocab_size, mission_embed_dim, padding_idx=0)
        self.mission_gru = nn.GRU(
            input_size=mission_embed_dim,
            hidden_size=mission_embed_dim,
            num_layers=1,
            batch_first=True,
        )
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
        )
        direction_dim = 1
        mission_dim = mission_embed_dim
        trunk_in = hidden_dim + direction_dim + mission_dim

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
        image = image.permute(0, 3, 1, 2)
        image_feat = self.image_encoder(image)
        direction = obs["direction"].float()
        mission_tokens = obs["mission_tokens"].long()
        mission_emb = self.mission_emb(mission_tokens)  # [B, T, E]
        _, h_n = self.mission_gru(mission_emb)
        mission_feat = h_n[-1]  # [B, E]
        x = torch.cat([image_feat, direction, mission_feat], dim=1)
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


def paper_success_reward(step_idx: int, horizon: int, decay_coeff: float = 0.5) -> float:
    """Paper-aligned success reward: 1 - decay_coeff * t / H, clipped to [0, 1]."""
    shaped = 1.0 - decay_coeff * (float(step_idx) / float(horizon))
    return float(np.clip(shaped, 0.0, 1.0))


def build_fixed_config_seeds(num_configs: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(0, 1_000_000_000, size=num_configs, endpoint=False)]


def choose_reset_seed(
    config_seeds: list[int] | None,
    rng: np.random.Generator | None = None,
) -> int:
    if config_seeds and len(config_seeds) > 0:
        if rng is None:
            return int(random.choice(config_seeds))
        return int(rng.choice(np.asarray(config_seeds, dtype=np.int64)))
    return random.randint(0, 1_000_000_000)


def encode_mission_tokens(mission: str, max_mission_len: int = 12) -> np.ndarray:
    toks = mission.lower().replace(",", " ").split()
    ids = [BabyAIObsWrapper._hash_token(t) for t in toks[:max_mission_len]]
    if len(ids) < max_mission_len:
        ids.extend([PAD_ID] * (max_mission_len - len(ids)))
    return np.asarray(ids, dtype=np.int32)


def encode_raw_observation(raw_obs: dict[str, Any], max_mission_len: int = 12) -> dict[str, np.ndarray]:
    return {
        "image": np.asarray(raw_obs["image"], dtype=np.uint8),
        "direction": np.asarray([float(raw_obs["direction"])], dtype=np.float32),
        "mission_tokens": encode_mission_tokens(str(raw_obs["mission"]), max_mission_len=max_mission_len),
    }


def _describe_demo_item(demo: Any) -> str:
    if isinstance(demo, dict):
        return f"dict keys={list(demo.keys())}"
    if isinstance(demo, (list, tuple)):
        return f"{type(demo).__name__}(len={len(demo)})"
    return str(type(demo))


def _maybe_unpack_images(images_obj: Any) -> np.ndarray:
    if isinstance(images_obj, np.ndarray):
        return images_obj
    if isinstance(images_obj, (bytes, bytearray)):
        return blosc.unpack_array(images_obj)
    return np.asarray(images_obj)


def _parse_demo_episode_to_samples(
    demo: Any,
    max_steps: int,
) -> list[tuple[dict[str, np.ndarray], int]]:
    """Parse one demo episode from multiple common BabyAI formats."""
    # Format A (official legacy): (mission, packed_images, directions, actions)
    if isinstance(demo, (tuple, list)) and len(demo) >= 4 and not (
        len(demo) > 0 and isinstance(demo[0], dict)
    ):
        mission, images_obj, directions, actions = demo[0], demo[1], demo[2], demo[3]
        images = _maybe_unpack_images(images_obj)
        n = min(len(actions), len(directions), int(images.shape[0]), max_steps)
        if n <= 0:
            return []
        out: list[tuple[dict[str, np.ndarray], int]] = []
        for i in range(n):
            raw_obs = {"image": images[i], "direction": directions[i], "mission": mission}
            out.append((encode_raw_observation(raw_obs), int(actions[i])))
        return out

    # Format B (dict): supports raw or packed image key naming variants.
    if isinstance(demo, dict):
        mission = demo.get("mission")
        directions = demo.get("directions")
        actions = demo.get("actions")
        images_obj = demo.get("images", demo.get("packed_images"))
        if mission is None or directions is None or actions is None or images_obj is None:
            raise ValueError(f"Missing required keys in demo dict: {_describe_demo_item(demo)}")
        images = _maybe_unpack_images(images_obj)
        n = min(len(actions), len(directions), int(images.shape[0]), max_steps)
        if n <= 0:
            return []
        out: list[tuple[dict[str, np.ndarray], int]] = []
        for i in range(n):
            raw_obs = {"image": images[i], "direction": directions[i], "mission": mission}
            out.append((encode_raw_observation(raw_obs), int(actions[i])))
        return out

    # Format C (already transformed): [(obs, action, done), ...]
    if isinstance(demo, list) and len(demo) > 0 and isinstance(demo[0], (tuple, list)):
        out: list[tuple[dict[str, np.ndarray], int]] = []
        for tr in demo[:max_steps]:
            if len(tr) < 2:
                continue
            obs_raw, action = tr[0], tr[1]
            if not isinstance(obs_raw, dict):
                raise ValueError("Transformed demo transition must have dict obs")
            out.append((encode_raw_observation(obs_raw), int(action)))
        return out

    raise ValueError(f"Unsupported demo format: {_describe_demo_item(demo)}")


def rollout_episode(
    env: gym.Env,
    model: ActorCritic,
    device: torch.device,
    gamma: float,
    max_steps: int = 100,
    reset_seed: int | None = None,
    action_prefix: list[int] | None = None,
    reward_decay_coeff: float = 0.5,
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
    success = False

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
        step_idx = t + 1
        step_success = done and (float(reward) > 0.0)
        if step_success:
            success = True
            shaped_reward = paper_success_reward(
                step_idx=step_idx,
                horizon=max_steps,
                decay_coeff=reward_decay_coeff,
            )
        else:
            shaped_reward = 0.0
        steps.append(
            StepRecord(
                obs=copy.deepcopy(obs),
                action=action,
                reward=float(shaped_reward),
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


def normalized_episode_return(ep: Episode, gamma: float) -> float:
    """Return normalized trajectory return in [0, 1]."""
    if not ep.steps:
        return 0.0
    ret0 = float(discounted_returns([s.reward for s in ep.steps], gamma=gamma)[0])
    # Paper uses normalized reward and diversity terms for f_poly.
    return float(np.clip(ret0, 0.0, 1.0))


def clone_episode_with_poly_window(
    ep: Episode,
    rollout_local_idx: int,
    window_w: int,
    poly_adv: float,
) -> Episode:
    """Clone episode and apply one shared set-level poly advantage on a window."""
    ep_copy = copy.deepcopy(ep)
    for t, step in enumerate(ep_copy.steps):
        if rollout_local_idx <= t <= rollout_local_idx + window_w:
            step.poly_adv_override = float(poly_adv)
    return ep_copy


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
    max_steps: int = 100,
    config_seeds: list[int] | None = None,
    rng: np.random.Generator | None = None,
    reward_decay_coeff: float = 0.5,
) -> list[Episode]:
    """Collect episodes and assign Poly-PPO advantages using vine sampling.

    Uses per-set shared advantages:
      A_hat(set) = f_poly(set) - mean_j f_poly(g_j)
    and applies that same advantage to all actions in the set trajectories
    within the rollout window, matching the paper's construction.
    """
    env = make_env(env_id)
    seed_episodes: list[Episode] = []
    for _ in range(n_vines):
        reset_seed = choose_reset_seed(config_seeds=config_seeds, rng=rng)
        seed_ep = rollout_episode(
            env,
            model,
            device,
            gamma=gamma,
            max_steps=max_steps,
            reset_seed=reset_seed,
            reward_decay_coeff=reward_decay_coeff,
        )
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
                    reward_decay_coeff=reward_decay_coeff,
                )
                branch_eps.append(branch_ep)

            if len(branch_eps) < set_size:
                continue

            # The rollout state for each branch trajectory is at local index 0:
            # we reset to s_t and start recording from that state onward.
            rollout_local_idx = 0

            # Build M trajectory sets and compute f_poly(set).
            set_indices: list[list[int]] = []
            set_scores: list[float] = []
            all_idx = list(range(len(branch_eps)))
            for _ in range(num_sets):
                chosen = random.sample(all_idx, set_size)
                selected = [branch_eps[i] for i in chosen]
                # f_poly(s, tau_1:n) = mean_i R(tau_i) * d(s, tau_1:n), both in [0, 1]
                returns = [normalized_episode_return(ep, gamma=gamma) for ep in selected]
                diversity = float(np.clip(poly_diversity([ep.visited_rooms for ep in selected]), 0.0, 1.0))
                score = float(np.mean(returns)) * diversity
                set_indices.append(chosen)
                set_scores.append(score)

            baseline = float(np.mean(set_scores))

            # Per-set shared advantage assignment (no per-trajectory averaging).
            for chosen, score in zip(set_indices, set_scores):
                shared_adv = float(score - baseline)
                for idx in chosen:
                    ep_with_set_adv = clone_episode_with_poly_window(
                        branch_eps[idx],
                        rollout_local_idx=rollout_local_idx,
                        window_w=window_w,
                        poly_adv=shared_adv,
                    )
                    all_eps.append(ep_with_set_adv)

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
    max_steps: int = 100,
    base_seed: int = 10_000,
    config_seeds: list[int] | None = None,
    rollouts_per_config: int = 2,
    reward_decay_coeff: float = 0.5,
) -> tuple[float, float]:
    env = make_env(env_id)
    rewards = []
    successes = []
    eval_seeds: list[int]
    if config_seeds and len(config_seeds) > 0:
        eval_seeds = [int(s) for s in config_seeds for _ in range(rollouts_per_config)]
    else:
        eval_seeds = [base_seed + i for i in range(n_episodes)]

    with torch.no_grad():
        for reset_seed in eval_seeds:
            ep = rollout_episode(
                env,
                model,
                device,
                gamma=1.0,
                max_steps=max_steps,
                reset_seed=reset_seed,
                reward_decay_coeff=reward_decay_coeff,
            )
            rewards.append(ep.episodic_reward)
            successes.append(1.0 if ep.success else 0.0)
    env.close()
    return float(np.mean(rewards)), 100.0 * float(np.mean(successes))


def load_official_demos(
    demos_path: str,
    max_steps: int,
    reward_decay_coeff: float = 0.5,
) -> tuple[list[list[tuple[dict[str, np.ndarray], int]]], dict[str, float]]:
    """Load official BabyAI demos with format validation and compatibility parsing."""
    with open(demos_path, "rb") as f:
        demos = pickle.load(f)
    if not isinstance(demos, list) or len(demos) == 0:
        raise ValueError(f"Demo file is empty or not a list: {demos_path}")

    first_demo = demos[0]
    print(f"[demos] loaded {len(demos)} episodes from {demos_path}")
    print(f"[demos] first item format: {_describe_demo_item(first_demo)}")

    demo_episodes: list[list[tuple[dict[str, np.ndarray], int]]] = []
    rewards: list[float] = []
    successes: list[float] = []

    for epi, demo in enumerate(demos):
        try:
            ep_samples = _parse_demo_episode_to_samples(demo, max_steps=max_steps)
        except Exception as e:
            raise ValueError(
                f"Failed to parse demo episode #{epi} ({_describe_demo_item(demo)}): {e}"
            ) from e
        if len(ep_samples) == 0:
            continue
        demo_episodes.append(ep_samples)

        # Official demos are successful expert trajectories.
        rewards.append(
            paper_success_reward(
                step_idx=len(ep_samples),
                horizon=max_steps,
                decay_coeff=reward_decay_coeff,
            )
        )
        successes.append(1.0)

    stats = {
        "demo_avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "demo_success_rate_pct": 100.0 * (float(np.mean(successes)) if successes else 0.0),
        "num_episodes": float(len(demo_episodes)),
        "num_samples": float(sum(len(ep) for ep in demo_episodes)),
        "source": "official_file",
        "demos_path": demos_path,
    }
    return demo_episodes, stats


def resolve_official_demos_paths(
    env_id: str,
    user_train_path: str,
    user_valid_path: str,
) -> tuple[Path, Path | None]:
    """Resolve official train/valid demos following BabyAI naming conventions."""
    if user_train_path:
        train_path = Path(user_train_path).expanduser()
        if not train_path.exists():
            raise FileNotFoundError(f"Official train demos path does not exist: {train_path}")
    else:
        train_candidates = [
            Path(f"demos/{env_id}.pkl"),
            Path(f"../demos/{env_id}.pkl"),
            Path(f"../../demos/{env_id}.pkl"),
        ]
        train_path = next((c.resolve() for c in train_candidates if c.exists()), None)
        if train_path is None:
            raise FileNotFoundError(
                "Official demonstrations are required. Set --official-demos-path, "
                f"or place file at demos/{env_id}.pkl."
            )

    if user_valid_path:
        valid_path = Path(user_valid_path).expanduser()
        if not valid_path.exists():
            raise FileNotFoundError(f"Official valid demos path does not exist: {valid_path}")
        return train_path.resolve(), valid_path.resolve()

    valid_candidates = [
        Path(f"demos/{env_id}_valid.pkl"),
        Path(f"../demos/{env_id}_valid.pkl"),
        Path(f"../../demos/{env_id}_valid.pkl"),
    ]
    valid_resolved = next((c.resolve() for c in valid_candidates if c.exists()), None)
    return train_path.resolve(), valid_resolved


def flatten_demo_episodes(
    demo_episodes: list[list[tuple[dict[str, np.ndarray], int]]]
) -> list[tuple[dict[str, np.ndarray], int]]:
    return [sample for ep in demo_episodes for sample in ep]


def split_demo_episodes(
    demo_episodes: list[list[tuple[dict[str, np.ndarray], int]]],
    train_ratio: float,
    seed: int,
) -> tuple[list[tuple[dict[str, np.ndarray], int]], list[tuple[dict[str, np.ndarray], int]]]:
    """Split by episode first, then flatten into transition samples."""
    idx = np.arange(len(demo_episodes))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(len(demo_episodes) * train_ratio)
    train_idx = idx[:cut]
    val_idx = idx[cut:]
    train_eps = [demo_episodes[i] for i in train_idx]
    val_eps = [demo_episodes[i] for i in val_idx]
    train_samples = [sample for ep in train_eps for sample in ep]
    val_samples = [sample for ep in val_eps for sample in ep]
    return train_samples, val_samples


def train_behavior_cloning(
    model: ActorCritic,
    device: torch.device,
    train_samples: list[tuple[dict[str, np.ndarray], int]],
    val_samples: list[tuple[dict[str, np.ndarray], int]],
    epochs: int,
    batch_size: int,
    lr: float,
    entropy_coef: float,
) -> None:
    """Behavior cloning pretraining with action cross-entropy + entropy regularization."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        random.shuffle(train_samples)
        epoch_losses: list[float] = []
        for start in range(0, len(train_samples), batch_size):
            batch = train_samples[start : start + batch_size]
            if not batch:
                continue
            obs_batch = [b[0] for b in batch]
            act_batch = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
            obs_t = batch_obs(obs_batch, device)
            logits, _ = model.forward(obs_t)
            ce = F.cross_entropy(logits, act_batch)
            entropy = Categorical(logits=logits).entropy().mean()
            loss = ce - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        # Validation accuracy for sanity checking.
        val_acc = 0.0
        if val_samples:
            with torch.no_grad():
                correct = 0
                total = 0
                for start in range(0, len(val_samples), batch_size):
                    batch = val_samples[start : start + batch_size]
                    obs_batch = [b[0] for b in batch]
                    act_batch = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
                    obs_t = batch_obs(obs_batch, device)
                    logits, _ = model.forward(obs_t)
                    pred = torch.argmax(logits, dim=-1)
                    correct += int((pred == act_batch).sum().item())
                    total += int(act_batch.numel())
                if total > 0:
                    val_acc = correct / total

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print(
            f"[BC] epoch {epoch}/{epochs}  loss={mean_loss:.4f}  "
            f"val_acc={100.0 * val_acc:.1f}%"
        )


def save_pretrained_checkpoint(path: Path, model: ActorCritic, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, path)


def _check_pretrained_metadata(metadata: dict[str, Any], expected: dict[str, Any]) -> None:
    missing = [k for k in expected.keys() if k not in metadata]
    mismatches = [(k, metadata.get(k), v) for k, v in expected.items() if metadata.get(k) != v]
    if missing or mismatches:
        parts = []
        if missing:
            parts.append(f"missing keys={missing}")
        if mismatches:
            parts.append(
                "mismatches="
                + ", ".join([f"{k}: ckpt={ck} expected={ex}" for k, ck, ex in mismatches])
            )
        raise ValueError("Pretrained checkpoint metadata mismatch: " + " | ".join(parts))


def load_pretrained_checkpoint(
    path: Path,
    model: ActorCritic,
    device: torch.device,
    expected_metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata is missing or invalid.")
    _check_pretrained_metadata(metadata, expected_metadata)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return metadata


def make_actor_critic_optimizer(
    model: ActorCritic,
    lr_actor: float,
    lr_critic: float,
    lr_shared: float,
) -> torch.optim.Optimizer:
    shared_params = (
        list(model.image_encoder.parameters())
        + list(model.mission_emb.parameters())
        + list(model.mission_gru.parameters())
        + list(model.trunk.parameters())
    )
    actor_head_params = list(model.policy_head.parameters())
    critic_params = list(model.value_head.parameters())
    return torch.optim.Adam(
        [
            {"params": shared_params, "lr": lr_shared},
            {"params": actor_head_params, "lr": lr_actor},
            {"params": critic_params, "lr": lr_critic},
        ]
    )


def train_reinforce(
    env_id: str,
    model: ActorCritic,
    device: torch.device,
    updates: int,
    episodes_per_update: int,
    gamma: float,
    lr_actor: float,
    lr_critic: float,
    lr_shared: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    max_steps: int,
    config_seeds: list[int] | None = None,
    seed: int = 42,
    reward_decay_coeff: float = 0.5,
) -> None:
    optimizer = make_actor_critic_optimizer(
        model,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        lr_shared=lr_shared,
    )
    env = make_env(env_id)
    rng = np.random.default_rng(seed)
    for upd in range(1, updates + 1):
        episodes = [
            rollout_episode(
                env,
                model,
                device,
                gamma=gamma,
                max_steps=max_steps,
                reset_seed=choose_reset_seed(config_seeds=config_seeds, rng=rng),
                reward_decay_coeff=reward_decay_coeff,
            )
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
    lr_shared: float,
    value_coef: float,
    entropy_coef: float,
    kl_coef: float,
    clip_eps: float,
    ppo_epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
    max_steps: int,
    config_seeds: list[int] | None = None,
    seed: int = 42,
    reward_decay_coeff: float = 0.5,
) -> None:
    optimizer = make_actor_critic_optimizer(
        model,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        lr_shared=lr_shared,
    )
    env = make_env(env_id)
    rng = np.random.default_rng(seed)
    for upd in range(1, updates + 1):
        episodes = [
            rollout_episode(
                env,
                model,
                device,
                gamma=gamma,
                max_steps=max_steps,
                reset_seed=choose_reset_seed(config_seeds=config_seeds, rng=rng),
                reward_decay_coeff=reward_decay_coeff,
            )
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
    lr_shared: float,
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
    config_seeds: list[int] | None = None,
    seed: int = 42,
    reward_decay_coeff: float = 0.5,
) -> None:
    optimizer = make_actor_critic_optimizer(
        model,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        lr_shared=lr_shared,
    )
    rng = np.random.default_rng(seed)
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
            config_seeds=config_seeds,
            rng=rng,
            reward_decay_coeff=reward_decay_coeff,
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
        mission_vocab_size=MISSION_HASH_VOCAB_SIZE + 1,
        mission_embed_dim=MODEL_MISSION_EMBED_DIM,
        hidden_dim=MODEL_HIDDEN_DIM,
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
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--num-fixed-configs", type=int, default=50)
    parser.add_argument("--eval-rollouts-per-config", type=int, default=2)
    parser.add_argument("--output", type=str, default="results/goto_replication_local.json")
    parser.add_argument("--pretrained-ckpt", type=str, default="checkpoints/goto_pretrained.pt")
    parser.add_argument("--force-pretrain", action="store_true")
    parser.add_argument("--official-demos-path", type=str, default="")
    parser.add_argument("--official-valid-demos-path", type=str, default="")
    parser.add_argument("--pretrain-train-ratio", type=float, default=0.8)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--pretrain-batch-size", type=int, default=128)
    parser.add_argument("--pretrain-lr", type=float, default=5e-5)
    parser.add_argument("--pretrain-entropy-coef", type=float, default=0.01)
    parser.add_argument("--success-reward-decay", type=float, default=0.5)

    # Paper-inspired defaults (Table 3 / Appendix A).
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--shared-lr", type=float, default=1e-5)
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


def run_single(
    algo: str,
    args: argparse.Namespace,
    device: torch.device,
    pretrained_state_dict: dict[str, torch.Tensor] | None = None,
    config_seeds: list[int] | None = None,
) -> dict[str, float]:
    set_seed(args.seed)
    model = build_model(args.env_id, device)
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)

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
            lr_shared=args.shared_lr,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            max_steps=args.max_steps,
            config_seeds=config_seeds,
            seed=args.seed,
            reward_decay_coeff=args.success_reward_decay,
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
            lr_shared=args.shared_lr,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            kl_coef=args.kl_coef,
            clip_eps=args.clip_eps,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            max_grad_norm=args.max_grad_norm,
            max_steps=args.max_steps,
            config_seeds=config_seeds,
            seed=args.seed,
            reward_decay_coeff=args.success_reward_decay,
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
            lr_shared=args.shared_lr,
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
            config_seeds=config_seeds,
            seed=args.seed,
            reward_decay_coeff=args.success_reward_decay,
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
        config_seeds=config_seeds,
        rollouts_per_config=args.eval_rollouts_per_config,
        reward_decay_coeff=args.success_reward_decay,
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
    pretrained_ckpt_path = Path(args.pretrained_ckpt)

    paper_table1_goto = {
        "pretrained_policy": {"avg_reward": 0.246, "success_rate_pct": 34.2},
        "reinforce": {"avg_reward": 0.533, "success_rate_pct": 73.0},
        "ppo": {"avg_reward": 0.406, "success_rate_pct": 46.2},
        "poly-ppo": {"avg_reward": 0.575, "success_rate_pct": 80.2},
    }

    set_seed(args.seed)
    base_model = build_model(args.env_id, device)
    pretrain_metadata: dict[str, Any] = {}
    fixed_config_seeds = build_fixed_config_seeds(args.num_fixed_configs, seed=args.seed)
    expected_ckpt_metadata = {
        "env_id": args.env_id,
        "max_steps": args.max_steps,
        "mission_vocab_size": int(base_model.mission_emb.num_embeddings),
        "hidden_dim": int(base_model.hidden_dim),
        "official_demos_required": True,
    }

    if pretrained_ckpt_path.exists() and not args.force_pretrain:
        try:
            pretrain_metadata = load_pretrained_checkpoint(
                pretrained_ckpt_path,
                base_model,
                device,
                expected_metadata=expected_ckpt_metadata,
            )
        except ValueError as e:
            raise ValueError(
                f"{e}. Use --force-pretrain with a matching official demos file to regenerate checkpoint."
            ) from e
        print(f"Loaded pretrained base policy from {pretrained_ckpt_path}")
    else:
        print("\n=== Pretraining base policy with expert demonstrations ===")
        demos_path, valid_demos_path = resolve_official_demos_paths(
            env_id=args.env_id,
            user_train_path=args.official_demos_path,
            user_valid_path=args.official_valid_demos_path,
        )
        print(f"[demos] using official train demos file: {demos_path}")
        demo_episodes, demo_stats = load_official_demos(
            demos_path=str(demos_path),
            max_steps=args.max_steps,
            reward_decay_coeff=args.success_reward_decay,
        )
        if valid_demos_path is not None:
            print(f"[demos] using official valid demos file: {valid_demos_path}")
            valid_demo_episodes, valid_demo_stats = load_official_demos(
                demos_path=str(valid_demos_path),
                max_steps=args.max_steps,
                reward_decay_coeff=args.success_reward_decay,
            )
            train_demos = flatten_demo_episodes(demo_episodes)
            val_demos = flatten_demo_episodes(valid_demo_episodes)
        else:
            valid_demo_stats = None
            train_demos, val_demos = split_demo_episodes(
                demo_episodes, train_ratio=args.pretrain_train_ratio, seed=args.seed
            )
        print(
            f"[BC] episodes={len(demo_episodes)}  train_samples={len(train_demos)}  "
            f"val_samples={len(val_demos)}  "
            f"demo_avg_reward={demo_stats['demo_avg_reward']:.3f}  "
            f"demo_success={demo_stats['demo_success_rate_pct']:.1f}%"
        )
        train_behavior_cloning(
            model=base_model,
            device=device,
            train_samples=train_demos,
            val_samples=val_demos,
            epochs=args.pretrain_epochs,
            batch_size=args.pretrain_batch_size,
            lr=args.pretrain_lr,
            entropy_coef=args.pretrain_entropy_coef,
        )
        pretrain_metadata = {
            "env_id": args.env_id,
            "seed": args.seed,
            "official_demos_path": str(demos_path),
            "official_valid_demos_path": str(valid_demos_path) if valid_demos_path else None,
            "official_demos_required": True,
            "max_steps": args.max_steps,
            "mission_vocab_size": int(base_model.mission_emb.num_embeddings),
            "hidden_dim": int(base_model.hidden_dim),
            "pretrain_train_ratio": args.pretrain_train_ratio,
            "pretrain_epochs": args.pretrain_epochs,
            "pretrain_batch_size": args.pretrain_batch_size,
            "pretrain_lr": args.pretrain_lr,
            "pretrain_entropy_coef": args.pretrain_entropy_coef,
            "demo_stats": demo_stats,
            "valid_demo_stats": valid_demo_stats,
        }
        save_pretrained_checkpoint(pretrained_ckpt_path, base_model, pretrain_metadata)
        print(f"Saved pretrained base policy to {pretrained_ckpt_path}")

    pre_avg_reward, pre_success = evaluate_policy(
        env_id=args.env_id,
        model=base_model,
        device=device,
        n_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        base_seed=args.seed + 99999,
        config_seeds=fixed_config_seeds,
        rollouts_per_config=args.eval_rollouts_per_config,
        reward_decay_coeff=args.success_reward_decay,
    )
    local_pretrained = {
        "avg_reward": round(pre_avg_reward, 3),
        "success_rate_pct": round(pre_success, 1),
    }
    print(
        f"[pretrained] eval avg_reward={local_pretrained['avg_reward']:.3f}, "
        f"success={local_pretrained['success_rate_pct']:.1f}%"
    )
    pretrained_state_dict = copy.deepcopy(base_model.state_dict())

    if args.algo == "all":
        algos = ["reinforce", "ppo", "poly-ppo"]
    else:
        algos = [args.algo]

    local_results: dict[str, dict[str, float]] = {}
    for algo in algos:
        print(f"\n=== Training {algo} on {args.env_id} ===")
        local_results[algo] = run_single(
            algo,
            args,
            device,
            pretrained_state_dict=pretrained_state_dict,
            config_seeds=fixed_config_seeds,
        )
        print(
            f"[{algo}] eval avg_reward={local_results[algo]['avg_reward']:.3f}, "
            f"success={local_results[algo]['success_rate_pct']:.1f}%"
        )

    payload = {
        "environment": "Goto (BabyAI)",
        "env_id": args.env_id,
        "paper_table1_reference": paper_table1_goto,
        "local_pretrained_policy": local_pretrained,
        "local_replication": local_results,
        "pretraining": {
            "checkpoint": str(pretrained_ckpt_path),
            "metadata": pretrain_metadata,
        },
        "rlft_setting": {
            "num_fixed_configurations": args.num_fixed_configs,
            "fixed_config_seeds": fixed_config_seeds,
            "episode_horizon": args.max_steps,
            "success_reward_formula": "1 - c*t/H, fail=0",
            "success_reward_decay_c": args.success_reward_decay,
            "eval_rollouts_per_config": args.eval_rollouts_per_config,
        },
        "config": vars(args),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
