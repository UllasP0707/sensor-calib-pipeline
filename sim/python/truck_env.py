"""Batched C++/CUDA highway environment, with explicit reset semantics.

This lightweight API follows reset/step conventions but is not a Gymnasium VectorEnv.
All returned arrays own their storage. Each instance is single-caller, not thread-safe.
"""

from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
F32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
U8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")


def load_library(path: str | Path | None = None):
    if path is None:
        path = os.environ.get("TRUCK_SIM_LIBRARY")
    if path is None:
        candidates = [
            ROOT / "build" / name for name in ("libtruck_sim.so", "libtruck_sim.dylib")
        ]
        path = next((p for p in candidates if p.exists()), candidates[0])
    library = ct.CDLL(str(Path(path).resolve()))
    library.sim_abi_version.restype = ct.c_int
    if library.sim_abi_version() != 1:
        raise RuntimeError("Unsupported truck_sim ABI")
    library.sim_last_error.restype = ct.c_char_p
    library.sim_create.argtypes = [
        ct.c_int,
        ct.c_float,
        ct.c_int,
        ct.c_uint,
        ct.c_int,
        ct.POINTER(ct.c_void_p),
    ]
    library.sim_create.restype = ct.c_int
    library.sim_destroy.argtypes = [ct.c_void_p]
    library.sim_destroy.restype = None
    library.sim_reset.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_int]
    library.sim_step.argtypes = [ct.c_void_p, F32]
    library.sim_read.argtypes = [ct.c_void_p, F32, F32, U8, U8]
    library.sim_read_state.argtypes = [ct.c_void_p, F32]
    library.sim_timing.argtypes = [ct.c_void_p, F32]
    return library


class TruckEnv:
    observation_size = 6
    action_size = 2

    def __init__(
        self, count=256, *, backend="cpu", workers=1, dt=0.1, horizon=300, library=None
    ):
        if backend not in {"cpu", "cuda"}:
            raise ValueError("backend must be cpu or cuda")
        if type(count) is not int or not 1 <= count <= 1_000_000:
            raise ValueError("count must be an integer in [1, 1000000]")
        if type(workers) is not int or not 1 <= workers <= 256:
            raise ValueError("workers must be an integer in [1, 256]")
        if type(horizon) is not int or not 1 <= horizon <= 100000:
            raise ValueError("horizon must be an integer in [1, 100000]")
        self.count, self.backend = count, backend
        self.dt, self.horizon, self.workers = dt, horizon, workers
        self._library = load_library(library)
        self._handle = ct.c_void_p()
        self._check(
            self._library.sim_create(
                count,
                dt,
                horizon,
                workers,
                int(backend == "cuda"),
                ct.byref(self._handle),
            )
        )
        self._seed = 0

    def _check(self, code):
        if code:
            raise RuntimeError(self._library.sim_last_error().decode("utf-8"))

    def _ensure_open(self):
        if not self._handle:
            raise RuntimeError("Environment is closed")

    def _read(self):
        observation = np.empty((self.count, 6), np.float32)
        reward = np.empty(self.count, np.float32)
        terminated = np.empty(self.count, np.uint8)
        truncated = np.empty(self.count, np.uint8)
        self._check(
            self._library.sim_read(
                self._handle, observation, reward, terminated, truncated
            )
        )
        return observation, reward, terminated.astype(bool), truncated.astype(bool)

    def reset(self, *, seed=0):
        self._ensure_open()
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("seed must be a uint32 integer")
        self._seed = seed
        self._check(self._library.sim_reset(self._handle, seed, 1))
        return self._read()[0], {"backend": self.backend, "seed": seed}

    def reset_done(self):
        """Reset only finished episodes; callers must save terminal observations first."""
        self._ensure_open()
        self._check(self._library.sim_reset(self._handle, self._seed, 0))
        return self._read()[0]

    def step(self, actions):
        self._ensure_open()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.count, 2) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite actions with shape ({self.count}, 2)")
        actions = np.ascontiguousarray(actions)
        self._check(self._library.sim_step(self._handle, actions))
        observation, reward, terminated, truncated = self._read()
        timing = np.empty(3, np.float32)
        self._check(self._library.sim_timing(self._handle, timing))
        return (
            observation,
            reward,
            terminated,
            truncated,
            {
                "timing_ms": dict(
                    zip(("upload", "compute", "download"), map(float, timing))
                )
            },
        )

    def state(self):
        self._ensure_open()
        state = np.empty((self.count, 6), np.float32)
        self._check(self._library.sim_read_state(self._handle, state))
        return state

    def close(self):
        if getattr(self, "_handle", None):
            self._library.sim_destroy(self._handle)
            self._handle = ct.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def __del__(self):
        self.close()
