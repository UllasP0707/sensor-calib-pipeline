#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
BINARY="${CALIB_BINARY:-$ROOT/build/calib_engine}"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
"$PYTHON" "$ROOT/scripts/generate_synthetic_data.py" --output-dir "$TEMP_DIR/frames" --scale 0.25 --count 45
"$PYTHON" "$ROOT/scripts/calib_pipeline.py" --binary "$BINARY" --image-dir "$TEMP_DIR/frames" \
  --vehicle-id e2e_vehicle --sensor-id front_camera --logs-dir "$TEMP_DIR/logs" --no-upload
"$PYTHON" - "$TEMP_DIR" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
paths = list((root / "logs/e2e_vehicle/front_camera").glob("calib_*.json"))
assert len(paths) == 1
record = json.loads(paths[0].read_text())
truth = json.loads((root / "frames/ground_truth.json").read_text())
result = record["calibration"]
assert record["quality_gate"]["passed"]
assert result["rms_reprojection_error_px"] <= 0.40
assert result["frames"]["valid_used"] >= 36
assert result["frames"]["total_processed"] == 45
for name, row in (("fx", 0), ("fy", 1)):
    expected = truth["camera_matrix"][row][row]
    assert abs(result["intrinsics"][name] - expected) / expected < 0.03
print("End-to-end PASS: gate, audit record, frame counts, and focal recovery verified.")
PY
