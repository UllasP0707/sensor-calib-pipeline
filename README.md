# Automated Sensor Calibration & Cloud Pipeline

A C++17/OpenCV intrinsic camera calibration engine with a Python 3.10+ operator CLI, durable local audit records, and optional AWS S3 synchronization. It runs without a camera or cloud account using the supplied synthetic-data generator.

## Architecture

```text
PNG / JPG / JPEG frames
        │
        ▼
C++ engine: bounded parallel read + corner detection
        → subpixel refinement → Brown–Conrady fit → per-view + global RMS
        │ JSON stdout (diagnostics on stderr)
        ▼
Python operator: schema validation → quality gate → atomic local audit log
        │                                           │
        ├─ FAIL / operational error: local only      └─ PASS → optional S3
        └─ Rich operator table
```

`calib_core` exposes `calibrate()` for library users; `calib_engine` provides the subprocess boundary. Detection uses up to eight worker threads, retains only corner vectors, and preserves sorted filename order. The executable disables nested OpenCV threading. PNG/JPG/JPEG extensions are case insensitive; scanning is non-recursive. Corrupt images and missed boards count toward total supplied frames. Mixed decoded image resolutions reject the run. At least three detected views are required for fitting; substantially more diverse poses are recommended for meaningful calibration.

Detection uses adaptive thresholding, normalization and fast-check flags. Refinement uses an 11×11 half-window, 30 iterations and 0.001 pixel epsilon. Calibration estimates exactly five distortion terms, with no rational or fisheye model enabled. The engine independently recomputes residuals using `projectPoints`.

The operator accepts RMS **≤ 0.40 px** and a usable-frame ratio **≥ 80%**. Equality passes. The ratio may be tightened, never relaxed below 80%; the RMS limit is fixed. Missing fields, non-finite values, invalid dimensions/counts, and inconsistent per-view errors fail validation. Rejected fits retain all observations: the engine does not prune outliers to improve its score.

## Build and quick start

Ubuntu 22.04:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git libopencv-dev nlohmann-json3-dev libgtest-dev python3-venv python3-pip
cd sensor_calib_pipeline
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
make build
make generate-data SCALE=0.25
make run-pipeline
```

On macOS, install the development dependencies with `brew install cmake opencv nlohmann-json googletest`, then use the same virtual-environment and make commands. For a custom OpenCV installation, use `make build CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/path/to/opencv/install"`. If JSON or GoogleTest packages are unavailable, CMake fetches versioned upstream tags (v3.11.3 and v1.14.0 respectively), requiring Git and network access on first configure. Set `-DBUILD_TESTING=OFF` to build only the engine.

For the original local workspace only (these dependencies are not included in GitHub), the development environment was verified with a locally built OpenCV under `../.deps/install` and a Python environment at `../.venv`; from this checkout:

```bash
make test PYTHON=../.venv/bin/python CMAKE=../.venv/bin/cmake \
  CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$(cd ../.deps/install && pwd)"
make run-pipeline PYTHON=../.venv/bin/python
```

The generator defaults to **45 frames at 3840×2160**. `SCALE=0.25` renders 960×540 for fast testing. It uses fixed seed 42, shared K and distortion coefficients, varying physically consistent camera/board poses, supersampling, and slight blur. `ground_truth.json` stores the rendering parameters. It refuses directories containing existing images to prevent mixing datasets. Use a different output directory to generate 4K data:

```bash
make generate-data SCALE=1 IMAGE_DIR=sample_data/checkerboard_4k
make run-pipeline IMAGE_DIR=sample_data/checkerboard_4k
```

## Engine and operator interfaces

```bash
./build/calib_engine --image_dir sample_data/checkerboard_frames \
  --board_w 9 --board_h 6 --square_size 0.025 --output_json fit.json

python scripts/calib_pipeline.py --image-dir sample_data/checkerboard_frames \
  --vehicle-id fleet_042 --sensor-id front_camera \
  --binary build/calib_engine --no-upload
```

Board dimensions specify **internal corners**; a 9×6 target has 10×7 squares. Square size is in meters. Use the true target dimensions consistently in generation and calibration. Engine stdout always contains one JSON document; failures return `{"status":"ERROR","error":{"message":"..."}}` with exit 1. Successful documents include the specified intrinsic, distortion, dimensions, counts, timing and error fields plus `used_files` and `rejected_files` for traceability. The optional engine output file must have a writable parent and must not already exist; the operator's audit writer is the durable storage interface.

The operator reads `configs/default_calibration.json`; use `--config path.json` to replace defaults. Explicit CLI options override the config. `--timeout` limits engine runtime (default 300 seconds); stderr streams directly to the terminal while stdout is captured. Use `--help` for all options.

| Operator exit code | Meaning |
| --- | --- |
| 0 | Calibration accepted, local record persisted; cloud may be unavailable |
| 2 | Quality gate rejected, or Click CLI usage/configuration error |
| 1 | Engine failure, malformed IPC, timeout, or audit persistence failure |

Every engine attempt writes `./logs/{vehicle_id}/{sensor_id}/calib_{timestamp}.json`, including rejected and failed attempts. CLI usage errors occur before an attempt and are not logged. Records contain schema version, UTC timestamp, vehicle/sensor IDs, request geometry, full calibration result (or null on operational failure), gate thresholds/reasons and exit code. Filenames include microseconds and a random suffix. Writes use a same-directory temporary file, file fsync, atomic replace and directory fsync. IDs are constrained to safe path components. Store logs in a trusted directory; hostile filesystem writers/symlink attacks are outside the local service's trust boundary.

## Cloud synchronization

```bash
export CALIB_S3_BUCKET=my-calibration-bucket
# Use the standard AWS credential provider chain: profile, environment, or workload role.
python scripts/calib_pipeline.py --image-dir sample_data/checkerboard_frames \
  --vehicle-id fleet_042 --sensor-id front_camera
```

`--s3-bucket` overrides the environment variable. Only accepted, already-persisted records upload, using the exact local bytes and key `calibrations/{vehicle_id}/{sensor_id}/calib_{timestamp}.json`. The S3 client uses bounded connection/read timeouts and retries. Supply IAM `s3:PutObject` for the intended prefix and any KMS permissions required by your bucket. No credentials are stored in the repository or image.

Without a bucket, the operator warns immediately without accessing AWS. Missing credentials, denied access or cloud failures preserve the local record, warn explicitly and leave a quality PASS at exit 0. `--no-upload` guarantees local-only execution. Upload status is reported separately in the terminal and is not written back into the immutable calibration artifact; there is no persistent retry queue. Local retained artifacts can be uploaded later using your fleet's sync service. GCP is not implemented: this project selects the requested S3 alternative.

## Calibration mathematics

For a board point transformed into camera coordinates `(Xc,Yc,Zc)`, define `x=Xc/Zc`, `y=Yc/Zc`, and `r²=x²+y²`. The Brown–Conrady model is:

```text
radial = 1 + k1·r² + k2·r⁴ + k3·r⁶
xd = x·radial + 2·p1·x·y + p2·(r² + 2·x²)
yd = y·radial + p1·(r² + 2·y²) + 2·p2·x·y
u = fx·xd + cx
v = fy·yd + cy

K = [ fx   0  cx ]          D = [k1, k2, p1, p2, k3]
    [  0  fy  cy ]
    [  0   0   1 ]
```

For view i with Ni detected points and measured/projected locations `qij` and `q̂ij`:

```text
RMS_i = sqrt(Σj ||qij − q̂ij||² / Ni)
RMS_global = sqrt(Σi Σj ||qij − q̂ij||² / Σi Ni)
```

These are Euclidean pixel errors per **point**, not per scalar coordinate. `square_size` establishes translation scale; intrinsics are in pixels. Resizing input images changes the intrinsic parameters. The current model assumes a fixed-focus, fixed-resolution pinhole camera, negligible rolling-shutter/motion artifacts, and a flat, dimensionally accurate target. Low RMS alone does not establish physical correctness or operational readiness: production fleet qualification also needs pose/field-of-view coverage, parameter plausibility and uncertainty assessment, independent validation images, and controlled capture conditions. This software does not certify a vehicle for driving.

## Validation and benchmarks

```bash
make test          # C++ GoogleTest + pytest + Ruff + Black
make e2e           # 45 generated 960×540 frames; audit + focal recovery assertions
make asan          # separate AddressSanitizer build and the same tests
```

GoogleTest renders its own ideal camera views and checks RMS < 0.35 px, approximate ground-truth focal recovery, 3×3 K, 1×5 distortion, structured CLI errors, mixed-resolution rejection, and corrupt-frame accounting. Python tests mock IPC and S3 to check threshold equality, rejection, malformed payloads, timeouts, logs, upload failure fallback and path validation. The end-to-end test uses the real engine and checks recovered focal lengths within 3% of known values. Unit tests do not access AWS. ASan instruments this project's C++ targets; prebuilt OpenCV/system libraries are not instrumented.

**Requested reference figures: 0.28 px RMS, 650 ms runtime, 45 4K frames. These are illustrative targets, not measured performance claims.** Image decoding, corner detection, CPU parallelism, image content, OpenCV build and optimizer convergence affect latency. The engine's `execution_time_ms` covers scanning through residual computation, excluding Python startup, serialization, persistence and cloud upload. Measure on the intended deployment hardware; a low-resolution test cannot substantiate a 4K benchmark.

### Measured verification in this workspace

On Apple M2, macOS 26.5.1, AppleClang 17, locally built OpenCV 4.10.0 (Release), eight detection workers, seed 42:

| Dataset | Usable frames | Global RMS | Engine latency |
| --- | --- | --- | --- |
| 960×540 synthetic | 45 / 45 | 0.045064 px | 302.05 ms |
| 3840×2160 synthetic | 45 / 45 | 0.030666 px | 1800.96 ms |

These are individual observed runs, not a statistically controlled benchmark; other verification jobs were active. Synthetic images are deliberately clean and do not predict real-camera accuracy. Both focal lengths recovered within 0.1% of rendering truth. Generated datasets, audit logs, and build outputs are excluded from Git; use the commands above to reproduce them after cloning. In the original workspace, local artifacts are retained under `logs/demo_vehicle/`; both generated datasets remain under `sample_data/`.

`make test` passed all 6 GoogleTests and 30 pytest cases, plus Ruff and Black checks. The shell end-to-end test passed with real generated images and the compiled engine. Docker and live S3 were not exercised: no Docker executable/daemon was available, and S3 was verified through mocks without performing a cloud upload.

AddressSanitizer compilation succeeded, but test discovery timed out before `main()` on the installed AppleClang 17/macOS runtime. A sampled stack showed recursive allocation in `__asan::InitializeShadowMemory` / `get_dyld_hdr` and a sanitizer spin lock, matching the [upstream macOS sanitizer initialization report](https://github.com/llvm/llvm-project/issues/200447). No ASan test pass is claimed. Run `make asan` on Linux or with an updated LLVM runtime; a newer compiler must use a fresh build directory, for example `make test BUILD_DIR=build-asan-llvm CMAKE_ARGS="-DENABLE_ASAN=ON -DCMAKE_CXX_COMPILER=/path/to/clang++"`.

## Docker

```bash
docker build -t sensor-calib .
mkdir -p logs
# Ensure logs is writable by container UID 10001.
docker run --rm -v "$PWD/sample_data:/data:ro" -v "$PWD/logs:/app/logs" \
  sensor-calib --image-dir /data/checkerboard_frames \
  --vehicle-id demo_vehicle --sensor-id front_camera --no-upload

# Generate data with the same image (the mounted directory must be writable).
docker run --rm --entrypoint python3 -v "$PWD/sample_data:/app/sample_data" \
  sensor-calib /app/scripts/generate_synthetic_data.py --scale 0.25
```

The Ubuntu 22.04 builder compiles and runs GoogleTest; the runtime contains Python, OpenCV runtime packages, the binary and scripts, runs as UID 10001, and excludes compiler headers/build tools. Mount credentials or attach a workload role only when enabling uploads. Runtime Python requirements also include test/lint tools to allow identical local validation dependencies; deployment-specific images may separate those packages. Docker execution requires an available Docker daemon.

## References

- [OpenCV camera calibration and Brown–Conrady model](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [Boto3 S3 upload_file API](https://docs.aws.amazon.com/boto3/latest/reference/services/s3/client/upload_file.html)
