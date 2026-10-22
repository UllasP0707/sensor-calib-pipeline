"""IPC, fail-closed gates, audit persistence and mocked cloud behavior."""

import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError, NoCredentialsError
from click.testing import CliRunner

SPEC = importlib.util.spec_from_file_location(
    "calib_pipeline", Path(__file__).resolve().parents[1] / "scripts/calib_pipeline.py"
)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


@pytest.fixture
def result():
    return {
        "status": "SUCCESS",
        "rms_reprojection_error_px": 0.28,
        "execution_time_ms": 650.0,
        "image_dimensions": {"width": 3840, "height": 2160},
        "frames": {"total_processed": 5, "valid_used": 4},
        "intrinsics": {"fx": 2840.0, "fy": 2841.0, "cx": 1920.0, "cy": 1080.0},
        "distortion": {
            "model": "brown_conrady_5",
            "coefficients": [-0.06, 0.015, 0, 0, -0.002],
        },
        "per_view_errors": [0.28] * 4,
    }


def invoke(tmp_path, extra=()):
    return CliRunner().invoke(
        pipeline.main,
        [
            "--image-dir",
            str(tmp_path / "images"),
            "--vehicle-id",
            "vehicle_01",
            "--sensor-id",
            "front",
            "--logs-dir",
            str(tmp_path / "logs"),
            *extra,
        ],
    )


def records(tmp_path):
    return list((tmp_path / "logs/vehicle_01/front").glob("calib_*.json"))


def mock_process(monkeypatch, result, code=0):
    run = Mock(return_value=subprocess.CompletedProcess([], code, json.dumps(result)))
    monkeypatch.setattr(pipeline.subprocess, "run", run)
    return run


def test_exec_and_parse(monkeypatch, tmp_path, result):
    run = mock_process(monkeypatch, result)
    assert pipeline.run_engine(Path("engine"), tmp_path, 9, 6, 0.025, 60) == result
    args, kwargs = run.call_args
    assert args[0][1:] == [
        "--image_dir",
        str(tmp_path),
        "--board_w",
        "9",
        "--board_h",
        "6",
        "--square_size",
        "0.025",
    ]
    assert kwargs["stderr"] is None
    assert kwargs["timeout"] == 60


@pytest.mark.parametrize(
    "rms,exit_code", [(0.0, 0), (0.28, 0), (0.40, 0), (0.400001, 2), (0.9, 2)]
)
def test_gate_boundary_and_persistence(monkeypatch, tmp_path, result, rms, exit_code):
    result["rms_reprojection_error_px"] = rms
    result["per_view_errors"] = [rms] * 4
    mock_process(monkeypatch, result)
    upload = Mock(return_value=True)
    monkeypatch.setattr(pipeline, "upload_s3", upload)
    response = invoke(tmp_path, ["--s3-bucket", "fleet-calibration"])
    assert response.exit_code == exit_code, response.output
    record = json.loads(records(tmp_path)[0].read_text())
    assert record["quality_gate"]["passed"] is (exit_code == 0)
    assert record["calibration"] == result
    assert upload.call_count == (1 if exit_code == 0 else 0)
    if exit_code == 0:
        path, bucket, key = upload.call_args.args
        assert bucket == "fleet-calibration"
        assert key == f"calibrations/vehicle_01/front/{path.name}"


def test_ratio_rejection(monkeypatch, tmp_path, result):
    result["frames"]["total_processed"] = 6
    mock_process(monkeypatch, result)
    upload = Mock()
    monkeypatch.setattr(pipeline, "upload_s3", upload)
    assert invoke(tmp_path).exit_code == 2
    upload.assert_not_called()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, "0.28", True])
def test_invalid_rms_fails_closed(monkeypatch, tmp_path, result, bad):
    result["rms_reprojection_error_px"] = bad
    mock_process(monkeypatch, result)
    assert invoke(tmp_path).exit_code == 1
    assert not json.loads(records(tmp_path)[0].read_text())["quality_gate"]["passed"]


@pytest.mark.parametrize(
    "payload,code",
    [
        ("garbage", 0),
        ("{} trailing", 0),
        ("[]", 0),
        ('{"status":"ERROR"}', 1),
        ("{}", 0),
    ],
)
def test_invalid_engine_output(monkeypatch, tmp_path, payload, code):
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], code, payload)),
    )
    assert invoke(tmp_path).exit_code == 1
    assert len(records(tmp_path)) == 1


@pytest.mark.parametrize(
    "exception", [FileNotFoundError("engine"), subprocess.TimeoutExpired("engine", 1)]
)
def test_process_failures_logged(monkeypatch, tmp_path, exception):
    monkeypatch.setattr(pipeline.subprocess, "run", Mock(side_effect=exception))
    assert invoke(tmp_path).exit_code == 1
    assert len(records(tmp_path)) == 1


def test_s3_upload(monkeypatch, tmp_path):
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(pipeline.boto3, "client", factory)
    path = tmp_path / "record.json"
    path.write_text("{}")
    assert pipeline.upload_s3(path, "bucket", "calibrations/v/s/record.json")
    client.upload_file.assert_called_once_with(
        str(path),
        "bucket",
        "calibrations/v/s/record.json",
        ExtraArgs={"ContentType": "application/json"},
    )


@pytest.mark.parametrize(
    "error",
    [
        NoCredentialsError(),
        S3UploadFailedError("upload denied"),
        ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "PutObject"
        ),
    ],
)
def test_cloud_fallback_preserves_log(monkeypatch, tmp_path, result, error):
    mock_process(monkeypatch, result)
    monkeypatch.setattr(pipeline.boto3, "client", Mock(side_effect=error))
    assert invoke(tmp_path, ["--s3-bucket", "bucket"]).exit_code == 0
    assert json.loads(records(tmp_path)[0].read_text())["calibration"] == result


def test_no_bucket_does_not_access_aws(monkeypatch, tmp_path):
    factory = Mock()
    monkeypatch.setattr(pipeline.boto3, "client", factory)
    assert not pipeline.upload_s3(tmp_path / "record.json", None, "key")
    factory.assert_not_called()


def test_no_upload_does_not_access_aws(monkeypatch, tmp_path, result):
    mock_process(monkeypatch, result)
    factory = Mock()
    monkeypatch.setattr(pipeline.boto3, "client", factory)
    assert invoke(tmp_path, ["--no-upload", "--s3-bucket", "bucket"]).exit_code == 0
    factory.assert_not_called()


def test_path_traversal_rejected(tmp_path):
    response = invoke(tmp_path, ["--vehicle-id", "../escape"])
    assert response.exit_code == 2
    assert not (tmp_path / "logs").exists()


def test_failed_persistence_prevents_upload(monkeypatch, tmp_path, result):
    mock_process(monkeypatch, result)
    monkeypatch.setattr(
        pipeline, "write_record", Mock(side_effect=OSError("disk full"))
    )
    upload = Mock()
    monkeypatch.setattr(pipeline, "upload_s3", upload)
    assert invoke(tmp_path).exit_code == 1
    upload.assert_not_called()


def test_custom_config_and_cli_override(monkeypatch, tmp_path, result):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "board_w": 8,
                "board_h": 5,
                "square_size": 0.03,
                "min_valid_ratio": 0.9,
                "timeout_seconds": 20,
            }
        )
    )
    run = mock_process(monkeypatch, result)
    assert invoke(tmp_path, ["--config", str(config), "--board-w", "10"]).exit_code == 2
    args = run.call_args.args[0]
    assert args[args.index("--board_w") + 1] == "10"
    assert args[args.index("--board_h") + 1] == "5"


def test_inconsistent_errors_rejected(result):
    result["per_view_errors"][0] = 2
    with pytest.raises(ValueError, match="inconsistent"):
        pipeline.validate_result(result)


def test_unique_records(monkeypatch, tmp_path, result):
    mock_process(monkeypatch, result)
    assert invoke(tmp_path, ["--no-upload"]).exit_code == 0
    assert invoke(tmp_path, ["--no-upload"]).exit_code == 0
    assert len(records(tmp_path)) == 2
