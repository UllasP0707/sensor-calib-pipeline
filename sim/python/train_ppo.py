"""Compact PPO training with explicit time-limit bootstrapping and batched simulation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from truck_env import TruckEnv


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(6, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
        )
        self.actor = nn.Linear(64, 2)
        self.critic = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.full((2,), -1.0))
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def distribution_value(self, obs):
        features = self.trunk(obs)
        return Normal(
            self.actor(features), self.log_std.clamp(-5, 1).exp()
        ), self.critic(features).squeeze(-1)

    def sample(self, obs):
        distribution, value = self.distribution_value(obs)
        latent = distribution.sample()
        action = latent.tanh()
        log_prob = (
            distribution.log_prob(latent) - torch.log(1 - action.square() + 1e-6)
        ).sum(-1)
        return action, latent, log_prob, value


def advantages(
    rewards, values, next_values, terminated, truncated, gamma=0.99, lam=0.95
):
    """Bootstrap time limits, but never propagate GAE across any episode boundary."""
    result = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        carry = delta + gamma * lam * (~(terminated[t] | truncated[t])) * carry
        result[t] = carry
    return result


def evaluate(policy, *, count=64, seed=2026, backend="cpu", workers=1, library=None):
    """Evaluate exactly one episode per environment, without resets or frozen-step rewards."""
    device = next(policy.parameters()).device
    with TruckEnv(count, backend=backend, workers=workers, library=library) as env:
        obs, _ = env.reset(seed=seed)
        active = np.ones(count, bool)
        returns = np.zeros(count, np.float64)
        successes = np.zeros(count, bool)
        for _ in range(env.horizon):
            with torch.no_grad():
                distribution, _ = policy.distribution_value(
                    torch.as_tensor(obs, device=device)
                )
                actions = distribution.mean.tanh().cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(actions)
            returns += reward * active
            successes |= active & terminated & (env.state()[:, 0] >= 300) & (reward > 0)
            active &= ~(terminated | truncated)
            if not active.any():
                break
        return {
            "mean_return": float(returns.mean()),
            "success_rate": float(successes.mean()),
            "episodes": count,
            "seed": seed,
        }


def train(args):
    if (
        min(
            args.envs,
            args.updates,
            args.rollout,
            args.epochs,
            args.minibatch,
            args.torch_threads,
        )
        < 1
    ):
        raise ValueError("Training counts must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA device unavailable")
    policy = ActorCritic().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    before = evaluate(
        policy, backend=args.backend, workers=args.workers, library=args.library
    )
    history = []
    start = time.perf_counter()
    with TruckEnv(
        args.envs, backend=args.backend, workers=args.workers, library=args.library
    ) as env:
        obs, _ = env.reset(seed=args.seed)
        for update in range(args.updates):
            storage = {
                key: []
                for key in (
                    "obs",
                    "latent",
                    "log_prob",
                    "value",
                    "reward",
                    "next_value",
                    "terminated",
                    "truncated",
                )
            }
            rollout_start = time.perf_counter()
            for _ in range(args.rollout):
                obs_tensor = torch.as_tensor(obs, device=device)
                with torch.no_grad():
                    action, latent, log_prob, value = policy.sample(obs_tensor)
                # This host interface transfers actions and observations every step, even with CUDA simulation.
                next_obs, reward, terminated, truncated, _ = env.step(
                    action.cpu().numpy()
                )
                with torch.no_grad():
                    _, next_value = policy.distribution_value(
                        torch.as_tensor(next_obs, device=device)
                    )
                data = {
                    "obs": obs_tensor,
                    "latent": latent,
                    "log_prob": log_prob,
                    "value": value,
                    "reward": torch.as_tensor(reward, device=device),
                    "next_value": next_value,
                    "terminated": torch.as_tensor(terminated, device=device),
                    "truncated": torch.as_tensor(truncated, device=device),
                }
                for key, value in data.items():
                    storage[key].append(value)
                obs = env.reset_done() if (terminated | truncated).any() else next_obs
            batch = {key: torch.stack(value) for key, value in storage.items()}
            advantage = advantages(
                batch["reward"],
                batch["value"],
                batch["next_value"],
                batch["terminated"],
                batch["truncated"],
            )
            returns = (advantage + batch["value"]).flatten()
            advantage = advantage.flatten()
            advantage = (advantage - advantage.mean()) / (
                advantage.std(unbiased=False) + 1e-8
            )
            observations = batch["obs"].flatten(0, 1)
            latents = batch["latent"].flatten(0, 1)
            old_log = batch["log_prob"].flatten()
            rollout_seconds = time.perf_counter() - rollout_start
            optimize_start = time.perf_counter()
            for _ in range(args.epochs):
                indices = torch.randperm(len(returns), device=device)
                for offset in range(0, len(indices), args.minibatch):
                    index = indices[offset : offset + args.minibatch]
                    distribution, value = policy.distribution_value(observations[index])
                    latent = latents[index]
                    log_prob = (
                        distribution.log_prob(latent)
                        - torch.log(1 - latent.tanh().square() + 1e-6)
                    ).sum(-1)
                    ratio = (log_prob - old_log[index]).exp()
                    objective = torch.minimum(
                        ratio * advantage[index],
                        ratio.clamp(0.8, 1.2) * advantage[index],
                    )
                    loss = (
                        -objective.mean()
                        + 0.5 * (value - returns[index]).square().mean()
                        - 0.001 * distribution.entropy().sum(-1).mean()
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite PPO loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row = {
                "update": update + 1,
                "mean_step_reward": batch["reward"].mean().item(),
                "loss": loss.item(),
                "rollout_seconds": rollout_seconds,
                "optimization_seconds": time.perf_counter() - optimize_start,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    elapsed = time.perf_counter() - start
    after = evaluate(
        policy, backend=args.backend, workers=args.workers, library=args.library
    )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": policy.state_dict(),
            "seed": args.seed,
            "observation_size": 6,
            "action_size": 2,
        },
        args.output / "policy.pt",
    )
    report = {
        "seed": args.seed,
        "backend": args.backend,
        "policy_device": str(device),
        "torch_version": torch.__version__,
        "transitions": args.updates * args.rollout * args.envs,
        "training_seconds": elapsed,
        "before": before,
        "after": after,
        "history": history,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--envs", type=int, default=128)
    result.add_argument("--updates", type=int, default=30)
    result.add_argument("--rollout", type=int, default=128)
    result.add_argument("--epochs", type=int, default=4)
    result.add_argument("--minibatch", type=int, default=512)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--workers", type=int, default=1)
    result.add_argument("--torch-threads", type=int, default=1)
    result.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    result.add_argument("--device", default="cpu")
    result.add_argument("--library", type=Path)
    result.add_argument("--output", type=Path, default=Path("runs/ppo"))
    return result


if __name__ == "__main__":
    train(parser().parse_args())
