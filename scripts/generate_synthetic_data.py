#!/usr/bin/env python3
"""Render repeatable planar checkerboards with shared pinhole intrinsics/distortion."""

from __future__ import annotations

import json
from pathlib import Path

import click
import cv2
import numpy as np


def generate(
    output_dir: Path,
    count: int = 45,
    scale: float = 1.0,
    seed: int = 42,
    board_w: int = 9,
    board_h: int = 6,
    square_size: float = 0.025,
) -> dict:
    if not np.isfinite(scale) or not 0.2 <= scale <= 2:
        raise ValueError("Scale must be finite and between 0.2 and 2")
    if not 3 <= count <= 10000 or not 3 <= board_w <= 100 or not 3 <= board_h <= 100:
        raise ValueError("Invalid frame count or board dimensions")
    if not np.isfinite(square_size) or not 0 < square_size <= 100:
        raise ValueError("Square size must be finite and in (0, 100]")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(p.suffix.lower() in {".png", ".jpg", ".jpeg"} for p in output_dir.iterdir()):
        raise ValueError(
            "Output directory already contains images; use a fresh directory"
        )
    width, height = round(3840 * scale), round(2160 * scale)
    camera = np.array(
        [[2840 * scale, 0, width / 2], [0, 2845 * scale, height / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    distortion = np.array([-0.06, 0.015, 0.0002, -0.0001, -0.002])
    rng = np.random.default_rng(seed)
    # Inverse distortion: each distorted output pixel samples an undistorted input pixel.
    yy, xx = np.indices((height, width), dtype=np.float32)
    pixels = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
    inverse = cv2.undistortPoints(pixels, camera, distortion, P=camera).reshape(
        height, width, 2
    )
    map_x, map_y = inverse[:, :, 0].copy(), inverse[:, :, 1].copy()
    del pixels, inverse, xx, yy
    tile = 96
    texture = np.full(((board_h + 1) * tile, (board_w + 1) * tile), 245, np.uint8)
    for y in range(board_h + 1):
        for x in range(board_w + 1):
            if (x + y) % 2 == 0:
                texture[y * tile : (y + 1) * tile, x * tile : (x + 1) * tile] = 15
    bw, bh = (board_w + 1) * square_size, (board_h + 1) * square_size
    outer = np.array(
        [
            [-bw / 2, -bh / 2, 0],
            [bw / 2, -bh / 2, 0],
            [bw / 2, bh / 2, 0],
            [-bw / 2, bh / 2, 0],
        ],
        dtype=np.float64,
    )
    source = np.array(
        [
            [0, 0],
            [texture.shape[1], 0],
            [texture.shape[1], texture.shape[0]],
            [0, texture.shape[0]],
        ],
        dtype=np.float32,
    )
    poses = []
    for index in range(count):
        for _ in range(1000):
            rotation = rng.uniform([-0.40, -0.45, -0.25], [0.40, 0.45, 0.25])
            distance = max(bw / 0.25, bh / 0.175) * rng.uniform(0.46, 0.70)
            translation = (
                np.array([rng.uniform(-0.09, 0.09), rng.uniform(-0.04, 0.04), 1])
                * distance
            )
            projected, _ = cv2.projectPoints(
                outer, rotation, translation, camera, np.zeros(5)
            )
            projected = projected.reshape(4, 2)
            if (
                projected[:, 0].min() > width * 0.06
                and projected[:, 0].max() < width * 0.94
                and projected[:, 1].min() > height * 0.06
                and projected[:, 1].max() < height * 0.94
            ):
                break
        else:
            raise ValueError("Could not place requested board fully in frame")
        # Supersampling limits rasterization bias in the corner detector.
        homography = cv2.getPerspectiveTransform(
            source, (projected * 2).astype(np.float32)
        )
        undistorted = cv2.warpPerspective(
            texture,
            homography,
            (width * 2, height * 2),
            flags=cv2.INTER_LINEAR,
            borderValue=195,
        )
        undistorted = cv2.resize(
            undistorted, (width, height), interpolation=cv2.INTER_AREA
        )
        frame = cv2.remap(undistorted, map_x, map_y, cv2.INTER_LINEAR, borderValue=195)
        frame = cv2.GaussianBlur(frame, (3, 3), 0.55)
        filename = f"frame_{index:03d}.png"
        if not cv2.imwrite(str(output_dir / filename), frame):
            raise OSError(f"Could not write {filename}")
        poses.append(
            {
                "file": filename,
                "rotation_vector": rotation.tolist(),
                "translation_m": translation.tolist(),
            }
        )
    truth = {
        "seed": seed,
        "image_dimensions": {"width": width, "height": height},
        "camera_matrix": camera.tolist(),
        "distortion": distortion.tolist(),
        "board_w": board_w,
        "board_h": board_h,
        "square_size_m": square_size,
        "poses": poses,
    }
    (output_dir / "ground_truth.json").write_text(
        json.dumps(truth, indent=2) + "\n", encoding="utf-8"
    )
    return truth


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("sample_data/checkerboard_frames"),
)
@click.option("--count", type=click.IntRange(3, 10000), default=45, show_default=True)
@click.option("--scale", type=click.FloatRange(0.2, 2), default=1.0, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--board-w", type=click.IntRange(3, 100), default=9)
@click.option("--board-h", type=click.IntRange(3, 100), default=6)
@click.option(
    "--square-size", type=click.FloatRange(min=0, max=100, min_open=True), default=0.025
)
def main(output_dir, count, scale, seed, board_w, board_h, square_size):
    """Generate images and ground_truth.json; existing image directories are protected."""
    cv2.setNumThreads(1)
    try:
        generate(output_dir, count, scale, seed, board_w, board_h, square_size)
    except (OSError, ValueError, cv2.error) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Generated {count} frames in {output_dir}")


if __name__ == "__main__":
    main()
