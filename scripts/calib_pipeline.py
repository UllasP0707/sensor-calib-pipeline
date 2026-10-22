#!/usr/bin/env python3
"""Operator CLI: calibrate, validate, persist every attempt, upload accepted results."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import boto3
import click
from boto3.exceptions import S3UploadFailedError
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
MAX_RMS = 0.40
MIN_RATIO = 0.80
console = Console(stderr=True)


def finite_number(value: object, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} is non-finite or out of range")
    return float(value)


def validate_result(result: dict) -> None:
    """Reject malformed success payloads before applying the quality gate."""
    if not isinstance(result, dict) or result.get("status") != "SUCCESS":
        raise ValueError(f"Engine did not succeed: {result}")
    finite_number(result["rms_reprojection_error_px"], "RMS", 0)
    finite_number(result["execution_time_ms"], "latency", 0)
    for key in ("width", "height"):
        value = result["image_dimensions"][key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"Invalid image {key}")
    total = result["frames"]["total_processed"]
    valid = result["frames"]["valid_used"]
    if type(total) is not int or type(valid) is not int or not 3 <= valid <= total:
        raise ValueError("Invalid frame counts (at least three valid frames required)")
    for key in ("fx", "fy", "cx", "cy"):
        value = finite_number(result["intrinsics"][key], key)
        if key in ("fx", "fy") and value <= 0:
            raise ValueError("Focal lengths must be positive")
    distortion = result["distortion"]
    coefficients = distortion["coefficients"]
    if (
        distortion["model"] != "brown_conrady_5"
        or not isinstance(coefficients, list)
        or len(coefficients) != 5
    ):
        raise ValueError("Expected five Brown-Conrady coefficients")
    for value in coefficients:
        finite_number(value, "distortion coefficient")
    errors = result["per_view_errors"]
    if not isinstance(errors, list) or len(errors) != valid:
        raise ValueError("Per-view errors must match valid frame count")
    for value in errors:
        finite_number(value, "per-view error", 0)
    # Each view has the same number of checkerboard corners.
    computed_rms = math.sqrt(sum(value * value for value in errors) / valid)
    if not math.isclose(
        computed_rms, result["rms_reprojection_error_px"], rel_tol=1e-4, abs_tol=1e-6
    ):
        raise ValueError("Global RMS is inconsistent with per-view errors")


def quality_gate(result: dict, min_ratio: float = MIN_RATIO) -> tuple[bool, list[str]]:
    validate_result(result)
    reasons = []
    if result["rms_reprojection_error_px"] > MAX_RMS:
        reasons.append(f"RMS exceeds {MAX_RMS:.2f} px")
    ratio = result["frames"]["valid_used"] / result["frames"]["total_processed"]
    if ratio < min_ratio:
        reasons.append(f"Usable frame ratio {ratio:.1%} is below {min_ratio:.1%}")
    return not reasons, reasons


def run_engine(
    binary: Path,
    image_dir: Path,
    board_w: int,
    board_h: int,
    square_size: float,
    timeout: float,
) -> dict:
    command = [
        str(binary.resolve()),
        "--image_dir",
        str(image_dir.resolve()),
        "--board_w",
        str(board_w),
        "--board_h",
        str(board_h),
        "--square_size",
        str(square_size),
    ]
    # stderr inherits the operator terminal and streams live; only stdout is captured.
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )
    try:
        result = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Engine stdout is not a single JSON document") from exc
    if completed.returncode != 0:
        raise ValueError(f"Engine exited {completed.returncode}: {result}")
    validate_result(result)
    return result


def write_record(
    root: Path, vehicle: str, sensor: str, timestamp: str, record: dict
) -> Path:
    directory = root / vehicle / sensor
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"calib_{timestamp}.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, prefix=".calib_", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        # Persist the rename as well as file contents on POSIX filesystems.
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def upload_s3(path: Path, bucket: str | None, key: str) -> bool:
    if not bucket:
        console.print(
            "[yellow]Cloud upload skipped: no S3 bucket configured; local log preserved.[/yellow]"
        )
        return False
    try:
        client = boto3.client(
            "s3",
            config=Config(
                connect_timeout=5,
                read_timeout=30,
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        )
        client.upload_file(
            str(path), bucket, key, ExtraArgs={"ContentType": "application/json"}
        )
    except (BotoCoreError, ClientError, S3UploadFailedError, OSError) as exc:
        console.print(
            f"Cloud upload failed; local log preserved: {exc}",
            style="yellow",
            markup=False,
        )
        return False
    console.print(f"Uploaded s3://{bucket}/{key}", markup=False)
    return True


def display_result(
    vehicle: str, sensor: str, result: dict | None, passed: bool, reasons: list[str]
) -> None:
    table = Table(title="Automated Sensor Calibration", show_header=False)
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Vehicle ID", vehicle)
    table.add_row("Sensor ID", sensor)
    if result is not None:
        dims, frames, intrinsics = (
            result["image_dimensions"],
            result["frames"],
            result["intrinsics"],
        )
        table.add_row("Resolution", f"{dims['width']} x {dims['height']}")
        table.add_row(
            "Frame Counts",
            f"{frames['valid_used']} / {frames['total_processed']} usable",
        )
        table.add_row("Engine Latency", f"{result['execution_time_ms']:.2f} ms")
        table.add_row(
            "Intrinsics (px)",
            ", ".join(
                f"{key}={intrinsics[key]:.4f}" for key in ("fx", "fy", "cx", "cy")
            ),
        )
        table.add_row(
            "RMS Reprojection Error", f"{result['rms_reprojection_error_px']:.6f} px"
        )
    table.add_row(
        "Quality Gate",
        "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]",
    )
    console.print(table)
    for reason in reasons:
        console.print(reason, style="red", markup=False)


def identifier(_ctx, _param, value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise click.BadParameter(
            "Use 1–64 letters, digits, underscores or hyphens, starting with a letter or digit"
        )
    return value


def load_config(ctx, _param, value: Path) -> Path:
    try:
        config = json.loads(value.read_text(encoding="utf-8"))
        required = {
            "board_w",
            "board_h",
            "square_size",
            "min_valid_ratio",
            "timeout_seconds",
        }
        if not isinstance(config, dict) or set(config) != required:
            raise ValueError(f"Expected exactly these config keys: {sorted(required)}")
        for key in ("board_w", "board_h"):
            if type(config[key]) is not int or not 3 <= config[key] <= 100:
                raise ValueError(f"Invalid {key}")
        size = finite_number(config["square_size"], "square_size")
        ratio = finite_number(config["min_valid_ratio"], "min_valid_ratio")
        timeout = finite_number(config["timeout_seconds"], "timeout_seconds")
        if (
            not 0 < size <= 100
            or not MIN_RATIO <= ratio <= 1
            or not 0 < timeout <= 86400
        ):
            raise ValueError("Config values are outside allowed ranges")
        ctx.default_map = {
            "board_w": config["board_w"],
            "board_h": config["board_h"],
            "square_size": size,
            "min_valid_ratio": ratio,
            "timeout": timeout,
        }
    except (OSError, ValueError, TypeError) as exc:
        raise click.BadParameter(str(exc)) from exc
    return value


@click.command()
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    default=ROOT / "configs/default_calibration.json",
    callback=load_config,
    is_eager=True,
    expose_value=False,
    show_default=True,
)
@click.option("--image-dir", required=True, type=click.Path(path_type=Path))
@click.option("--vehicle-id", required=True, callback=identifier)
@click.option("--sensor-id", required=True, callback=identifier)
@click.option(
    "--binary",
    default=ROOT / "build/calib_engine",
    type=click.Path(path_type=Path),
    show_default=True,
)
@click.option("--board-w", type=click.IntRange(3, 100), default=9)
@click.option("--board-h", type=click.IntRange(3, 100), default=6)
@click.option(
    "--square-size", type=click.FloatRange(min=0, max=100, min_open=True), default=0.025
)
@click.option(
    "--min-valid-ratio", type=click.FloatRange(MIN_RATIO, 1.0), default=MIN_RATIO
)
@click.option(
    "--timeout", type=click.FloatRange(min=0, max=86400, min_open=True), default=300.0
)
@click.option("--logs-dir", type=click.Path(path_type=Path), default=Path("logs"))
@click.option("--s3-bucket", envvar="CALIB_S3_BUCKET", default=None)
@click.option(
    "--no-upload", is_flag=True, help="Explicit local-only run; do not access AWS."
)
def main(
    image_dir,
    vehicle_id,
    sensor_id,
    binary,
    board_w,
    board_h,
    square_size,
    min_valid_ratio,
    timeout,
    logs_dir,
    s3_bucket,
    no_upload,
):
    """Calibrate a sensor. Exit 0: PASS, 2: quality rejection, 1: operational failure."""
    for name, value in (
        ("square size", square_size),
        ("valid ratio", min_valid_ratio),
        ("timeout", timeout),
    ):
        if not math.isfinite(value):
            raise click.BadParameter(f"{name} must be finite")
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:12]
    result = None
    try:
        result = run_engine(binary, image_dir, board_w, board_h, square_size, timeout)
        passed, reasons = quality_gate(result, min_valid_ratio)
        exit_code = 0 if passed else 2
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        passed, reasons, exit_code = False, [str(exc)], 1
        result = None
    record = {
        "schema_version": 1,
        "timestamp_utc": now.isoformat(),
        "vehicle_id": vehicle_id,
        "sensor_id": sensor_id,
        "request": {
            "image_dir": str(image_dir.resolve()),
            "binary": str(binary.resolve()),
            "board_w": board_w,
            "board_h": board_h,
            "square_size_m": square_size,
        },
        "quality_gate": {
            "passed": passed,
            "max_rms_px": MAX_RMS,
            "min_valid_ratio": min_valid_ratio,
            "reasons": reasons,
        },
        "exit_code": exit_code,
        "calibration": result,
    }
    try:
        path = write_record(logs_dir, vehicle_id, sensor_id, timestamp, record)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"Unable to persist audit log: {exc}") from exc
    display_result(vehicle_id, sensor_id, result, passed, reasons)
    console.print(f"Local record: {path}", markup=False)
    if passed:
        if no_upload:
            console.print("Cloud upload disabled; local log preserved.", style="yellow")
        else:
            upload_s3(
                path, s3_bucket, f"calibrations/{vehicle_id}/{sensor_id}/{path.name}"
            )
    raise click.exceptions.Exit(exit_code)


if __name__ == "__main__":
    main()
