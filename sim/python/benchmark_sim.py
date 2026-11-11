"""Measure useful batched simulation transitions, including reset and host-interface costs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from truck_env import TruckEnv


def benchmark(count, steps, workers=1, backend="cpu", seed=42, library=None):
    if steps < 1:
        raise ValueError("steps must be positive")
    with TruckEnv(count, backend=backend, workers=workers, library=library) as env:
        rng = np.random.default_rng(seed)
        # Pregenerate a bounded action table; generation is outside the measured region.
        actions = rng.uniform(-0.08, 0.08, size=(32, count, 2)).astype(np.float32)
        env.reset(seed=seed)
        for i in range(10):
            _, _, term, trunc, _ = env.step(actions[i % 32])
            if (term | trunc).any():
                env.reset_done()
        env.reset(seed=seed)
        latencies, phase_times = [], []
        resets = 0
        start = time.perf_counter()
        for i in range(steps):
            tick = time.perf_counter()
            _, _, term, trunc, info = env.step(actions[i % 32])
            if (term | trunc).any():
                resets += int((term | trunc).sum())
                env.reset_done()
            latencies.append((time.perf_counter() - tick) * 1000)
            phase_times.append(list(info["timing_ms"].values()))
        elapsed = time.perf_counter() - start
        state = env.state()
        phases = np.mean(phase_times, axis=0)
        return {
            "backend": backend,
            "environments": count,
            "workers": min(workers, count) if backend == "cpu" else None,
            "steps": steps,
            "useful_transitions": steps * count,
            "seed": seed,
            "seconds": elapsed,
            "transitions_per_second": steps * count / elapsed,
            "batch_latency_ms_p50": float(np.percentile(latencies, 50)),
            "batch_latency_ms_p95": float(np.percentile(latencies, 95)),
            "mean_step_phase_ms": dict(
                zip(("upload", "compute", "download"), map(float, phases))
            ),
            "reset_episodes": resets,
            "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
            "state_checksum": float(state.astype(np.float64).sum()),
            "cuda_step_transfer_bytes": (
                count * (2 * 4 + 6 * 4 + 6 * 4 + 4 + 2) if backend == "cuda" else 0
            ),
            "timing_scope": "step + native output copies + Python arrays + completed-episode resets; excludes policy and action generation",
        }


def metadata():
    result = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        result["git_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        )
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[128, 4096, 65536])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark.json"))
    args = parser.parse_args()
    report = {
        "metadata": metadata(),
        "results": [
            benchmark(
                count, args.steps, args.workers, args.backend, args.seed, args.library
            )
            for count in args.counts
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
