"""Project synthetic lane points using an accepted calibration audit artifact.

Assumed mount: optical axis forward, x right, y down, level camera 2.5m above road.
Intrinsics do not determine this extrinsic pose; it is an explicit simulation assumption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from truck_env import TruckEnv


class CalibratedCamera:
    def __init__(self, path: Path):
        record = json.loads(path.read_text())
        if record.get("quality_gate", {}).get("passed") is not True:
            raise ValueError("Only accepted calibration audit records can be used")
        fit = record["calibration"]
        rms = fit["rms_reprojection_error_px"]
        frames = fit["frames"]
        if (
            fit["status"] != "SUCCESS"
            or not np.isfinite(rms)
            or not 0 <= rms <= 0.40
            or not 3 <= frames["valid_used"] <= frames["total_processed"]
            or frames["valid_used"] / frames["total_processed"] < 0.8
        ):
            raise ValueError("Calibration fails quality thresholds")
        self.intrinsics = np.array(
            [fit["intrinsics"][k] for k in ("fx", "fy", "cx", "cy")], dtype=np.float64
        )
        self.distortion = np.asarray(
            fit["distortion"]["coefficients"], dtype=np.float64
        )
        if (
            fit["distortion"]["model"] != "brown_conrady_5"
            or self.distortion.shape != (5,)
            or not np.isfinite(self.intrinsics).all()
            or not np.isfinite(self.distortion).all()
            or (self.intrinsics[:2] <= 0).any()
        ):
            raise ValueError("Invalid camera model")
        self.width, self.height = (
            fit["image_dimensions"][key] for key in ("width", "height")
        )
        if any(
            type(size) is not int or size <= 0 for size in (self.width, self.height)
        ):
            raise ValueError("Invalid resolution")

    def project(self, optical_points):
        points = np.asarray(optical_points, dtype=np.float64)
        if points.ndim < 2 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("Expected finite optical-frame points [..., 3]")
        front = points[..., 2] > 0.1
        z = np.where(front, points[..., 2], 1.0)
        x, y = points[..., 0] / z, points[..., 1] / z
        k1, k2, p1, p2, k3 = self.distortion
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        fx, fy, cx, cy = self.intrinsics
        pixels = np.stack((fx * xd + cx, fy * yd + cy), axis=-1)
        visible = (
            front
            & (pixels[..., 0] >= 0)
            & (pixels[..., 0] < self.width)
            & (pixels[..., 1] >= 0)
            & (pixels[..., 1] < self.height)
        )
        return pixels, visible

    def lane_pixels(self, state):
        state = np.asarray(state, dtype=np.float64)
        if state.ndim != 2 or state.shape[1] != 6 or not np.isfinite(state).all():
            raise ValueError("Expected finite [N, 6] simulator state")
        distance = np.repeat([20.0, 40.0, 60.0], 3)[None, :]
        lateral = np.tile([-3.5, 0.0, 3.5], 3)[None, :] - state[:, 1, None]
        yaw = state[:, 2, None]
        forward = distance * np.cos(yaw) + lateral * np.sin(yaw)
        left = -distance * np.sin(yaw) + lateral * np.cos(yaw)
        optical = np.stack((-left, np.full_like(left, 2.5), forward), axis=-1)
        return self.project(optical)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-log", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/lane_projection.json")
    )
    args = parser.parse_args()
    camera = CalibratedCamera(args.calibration_log)
    with TruckEnv(4) as env:
        env.reset(seed=42)
        state = env.state()
        pixels, visible = camera.lane_pixels(state)
    report = {
        "calibration_log": str(args.calibration_log),
        "resolution": [camera.width, camera.height],
        "mount_assumption": "level, optical z forward, x right, y down; height 2.5m",
        "state": state.tolist(),
        "pixels": pixels.tolist(),
        "visible": visible.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved calibrated lane projections to {args.output}")


if __name__ == "__main__":
    main()
