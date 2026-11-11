"""Evaluate a saved PPO policy against its initial zero-action baseline on held-out seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from train_ppo import ActorCritic, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/ppo/policy.pt"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 200, 300])
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    baseline = ActorCritic()
    trained = ActorCritic()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["observation_size"] != 6 or checkpoint["action_size"] != 2:
        raise ValueError("Checkpoint does not match the environment interface")
    trained.load_state_dict(checkpoint["model"])
    rows = []
    for seed in args.seeds:
        options = {
            "count": args.episodes,
            "seed": seed,
            "backend": args.backend,
            "workers": args.workers,
            "library": args.library,
        }
        rows.append(
            {
                "seed": seed,
                "baseline": evaluate(baseline, **options),
                "trained": evaluate(trained, **options),
            }
        )
    report = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "backend": args.backend,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
