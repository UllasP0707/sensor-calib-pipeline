"""Native integration, deterministic episodes, learning semantics, and calibrated projection."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from benchmark_sim import benchmark  # noqa: E402
from calibrated_camera import CalibratedCamera  # noqa: E402
from truck_env import TruckEnv, load_library  # noqa: E402


@pytest.fixture(params=["cpu", "cuda"])
def backend(request):
    if request.param == "cuda" and not load_library().sim_cuda_available():
        pytest.skip("CUDA device/backend unavailable")
    return request.param


def test_reset_reproducibility_and_independent_observations(backend):
    with TruckEnv(33, backend=backend) as env:
        first, _ = env.reset(seed=42)
        saved = first.copy()
        env.step(np.zeros((33, 2), np.float32))
        np.testing.assert_array_equal(first, saved)
        second, _ = env.reset(seed=42)
        np.testing.assert_array_equal(first, second)
        third, _ = env.reset(seed=43)
        assert not np.array_equal(first, third)
        assert np.unique(first[:, 0]).size > 1


def test_terminal_freeze_and_selective_reset(backend):
    with TruckEnv(8, horizon=1, backend=backend) as env:
        first, _ = env.reset(seed=7)
        obs, _, term, trunc, _ = env.step(np.zeros((8, 2)))
        assert trunc.all() and not term.any()
        state = env.state()
        obs2, reward, _, _, _ = env.step(np.ones((8, 2)))
        np.testing.assert_array_equal(obs, obs2)
        np.testing.assert_array_equal(state, env.state())
        assert not reward.any()
        reset = env.reset_done()
        assert not np.array_equal(reset, first)
        assert (reset[:, 5] == 1).all()
        np.testing.assert_array_equal(env.reset_done(), reset)


@pytest.mark.parametrize(
    "actions", [np.zeros((4, 3)), np.full((4, 2), np.nan), np.full((4, 2), np.inf)]
)
def test_bad_actions_leave_state_unchanged(actions):
    with TruckEnv(4) as env:
        state = env.state()
        with pytest.raises(ValueError):
            env.step(actions)
        np.testing.assert_array_equal(state, env.state())


def test_noncontiguous_actions_and_close():
    env = TruckEnv(4)
    env.step(np.zeros((4, 4), np.float32)[:, ::2])
    env.close()
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 0},
        {"dt": float("nan")},
        {"workers": 0},
        {"backend": "magic"},
        {"horizon": 0},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises((ValueError, RuntimeError)):
        TruckEnv(**kwargs)


def test_worker_parity():
    with TruckEnv(257, workers=1) as a, TruckEnv(257, workers=4) as b:
        a.reset(seed=8)
        b.reset(seed=8)
        rng = np.random.default_rng(123)
        for _ in range(20):
            actions = rng.uniform(-0.2, 0.2, (257, 2))
            np.testing.assert_array_equal(a.step(actions)[0], b.step(actions)[0])
            np.testing.assert_array_equal(a.reset_done(), b.reset_done())


def test_benchmark_counts_active_transitions_and_reproducibility():
    a = benchmark(8, 100)
    b = benchmark(8, 100)
    assert a["useful_transitions"] == 800
    assert a["reset_episodes"] > 0
    assert a["state_sha256"] == b["state_sha256"]
    assert a["transitions_per_second"] > 0
    assert a["batch_latency_ms_p95"] >= a["batch_latency_ms_p50"]


@pytest.fixture
def calibration(tmp_path):
    path = tmp_path / "fit.json"
    path.write_text(
        json.dumps(
            {
                "quality_gate": {"passed": True},
                "calibration": {
                    "status": "SUCCESS",
                    "rms_reprojection_error_px": 0.2,
                    "frames": {"valid_used": 45, "total_processed": 45},
                    "intrinsics": {"fx": 1000, "fy": 1000, "cx": 640, "cy": 360},
                    "distortion": {
                        "model": "brown_conrady_5",
                        "coefficients": [0, 0, 0, 0, 0],
                    },
                    "image_dimensions": {"width": 1280, "height": 720},
                },
            }
        )
    )
    return path


def test_calibrated_projection_and_visibility(calibration):
    camera = CalibratedCamera(calibration)
    pixels, visible = camera.project([[0, 0, 10], [1, 1, 10], [0, 0, -1]])
    np.testing.assert_allclose(pixels[:2], [[640, 360], [740, 460]])
    assert visible.tolist() == [True, True, False]
    with TruckEnv(4) as env:
        lanes, mask = camera.lane_pixels(env.state())
        assert lanes.shape == (4, 9, 2) and mask.shape == (4, 9)
        assert np.isfinite(lanes).all()


def test_rejected_calibration_is_not_used(calibration):
    record = json.loads(calibration.read_text())
    record["quality_gate"]["passed"] = False
    calibration.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="accepted"):
        CalibratedCamera(calibration)


def test_gae_bootstraps_truncation_but_not_termination():
    torch = pytest.importorskip("torch")
    from train_ppo import advantages

    rewards = torch.tensor([[1.0, 1.0], [100.0, 100.0]])
    zeros = torch.zeros_like(rewards)
    term = torch.tensor([[True, False], [False, False]])
    trunc = torch.tensor([[False, True], [False, False]])
    result = advantages(
        rewards, zeros, torch.full_like(rewards, 10), term, trunc, gamma=0.9, lam=1
    )
    torch.testing.assert_close(result[0], torch.tensor([1.0, 10.0]))


def test_ppo_updates_parameters_and_writes_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from train_ppo import ActorCritic, parser, train

    args = parser().parse_args(
        [
            "--envs",
            "8",
            "--updates",
            "1",
            "--rollout",
            "8",
            "--epochs",
            "1",
            "--minibatch",
            "32",
            "--output",
            str(tmp_path),
        ]
    )
    report = train(args)
    checkpoint = torch.load(tmp_path / "policy.pt", weights_only=True)
    assert report["transitions"] == 64
    assert np.isfinite(report["history"][0]["loss"])
    assert not torch.equal(
        checkpoint["model"]["actor.weight"], ActorCritic().state_dict()["actor.weight"]
    )
    assert (tmp_path / "training.json").exists()
