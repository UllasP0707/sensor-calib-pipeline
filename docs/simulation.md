# Batched simulation and reinforcement learning

This extension adds a C++17 highway-control simulator, an optional CUDA backend, a Python batch API, PPO training, and a calibration-to-projection bridge. It demonstrates simulation infrastructure and performance engineering alongside the original camera-calibration pipeline. It is an independently implemented portfolio project, with no affiliation to Bot Auto.

## What this demonstrates

| Engineering area | Implementation | Evidence / limit |
| --- | --- | --- |
| Modern C++ and concurrency | Persistent worker pool, condition variables, RAII, exception-safe C ABI | Serial vs multithread trajectory tests |
| CUDA execution and memory | One environment per GPU thread; SoA state, persistent device/pinned host buffers, stream, fused dynamics/reward/done kernel | CUDA compile CI; GPU runtime validation requires NVIDIA hardware |
| Batched RL | Explicit reset/step, actor-critic PPO, terminal observations, time-limit bootstrapping | GAE boundary tests and training smoke test |
| Profiling | p50/p95 latency, useful transitions/s, CUDA event phase timings, reproducibility hash | Includes Python/copy/reset costs; no invented GPU speedup |
| Calibration integration | Accepted camera audit artifact projects simulated lane points with Brown–Conrady distortion | Projection tests and actual calibration log demo |
| Linux engineering | CPU tests, sanitizer job, CUDA Docker compilation | GitHub Actions workflow |

## Build and run

From the repository root, using Python 3.10+ and a C++17 compiler:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt -r sim/requirements.txt
cmake -S . -B build -DBUILD_CALIBRATION=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
python -m pytest -q sim/tests/test_simulation.py
python sim/python/benchmark_sim.py --counts 128 4096 65536 --workers 4
python sim/python/train_ppo.py --envs 128 --updates 30
```

The simulation-only build has no OpenCV or JSON dependency. CMake finds GoogleTest or fetches its versioned source for tests. The default full build also requires the calibration dependencies described in the main README. PyTorch is optional for simulation/benchmarks and required for training; it is separate from the original requirements. A test run without PyTorch explicitly skips the two training tests.

The default library search is `build/libtruck_sim.so` (Linux) or `.dylib` (macOS). Set `TRUCK_SIM_LIBRARY=/absolute/path/to/libtruck_sim.so` or pass `--library` to training/benchmark commands for other build directories. In the original workspace, Python is available as `../.venv/bin/python`.

### CUDA

On Linux with an NVIDIA GPU, driver and CUDA Toolkit:

```bash
cmake -S . -B build-cuda -DBUILD_CALIBRATION=OFF -DENABLE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda --parallel 4
export TRUCK_SIM_LIBRARY="$PWD/build-cuda/libtruck_sim.so"
PYTHONPATH=sim/python python -c 'from truck_env import load_library; assert load_library().sim_cuda_available()'
ctest --test-dir build-cuda --output-on-failure
python -m pytest -q sim/tests/test_simulation.py
python sim/python/benchmark_sim.py --backend cuda --counts 128 4096 65536
compute-sanitizer --tool memcheck --error-exitcode=1 ./build-cuda/test_sim --gtest_filter=Cuda.*
```

Choose the architecture for your GPU; 86 is an example, not a universal setting. CMake 3.24+ supports `native` on a machine with the target GPU. `--backend cuda` fails explicitly when CUDA was not compiled or no GPU is available; it never silently substitutes CPU timings. CPU/GPU tests compare observations, rewards and termination flags with float tolerances, not bitwise equality.

```bash
docker build -f sim/Dockerfile.cuda -t truck-sim-cuda .
docker run --rm --gpus all truck-sim-cuda
# For parity tests inside the image:
docker run --rm --gpus all --entrypoint /app/test_sim truck-sim-cuda
```

The CUDA image includes benchmarking dependencies, not PyTorch. Install a suitable PyTorch build in a training environment when using `--device cuda`. The simulator and policy-device selections are independent. This initial Python interface exchanges NumPy host arrays: a CUDA policy still copies actions to CPU and observations back to GPU each step. It is **not** a zero-copy or all-GPU training pipeline, and this cost is not hidden. A device-tensor / DLPack interface is a logical future optimization, but is not implemented here.

## Model and episode contract

Each environment simulates one controlled rigid vehicle and a constant-speed lead vehicle on a straight road. There is no rendering, articulated trailer, tire model, map, traffic intelligence, multi-agent learning or distributed training. The camera bridge is a projection model; the RL policy currently consumes numerical state features, not camera images.

Units are meters, seconds and radians. Native state uses structure-of-arrays layout `[field * N + environment]` for coalesced adjacent-thread state access. Public observations/state are conventional `[N, 6]` arrays.

State: `[x, y, yaw, speed, lead_x, lead_speed]`.

Observation: `[y/2.25, yaw, speed/20, (lead_x-x)/100, lead_speed/20, remaining_horizon_fraction]`.

Actions: `[normalized_acceleration, normalized_steering]`, clamped to `[-1,1]` and scaled to ±3 m/s² and ±0.35 rad. Non-finite actions are rejected before any state update. For wheelbase `L=5m` and timestep `dt=0.1s`:

```text
v_next = clamp(v + acceleration * dt, 0, 35)
x_next = x + v_next * cos(yaw) * dt
y_next = y + v_next * sin(yaw) * dt
yaw_next = wrap(yaw + v_next / L * tan(steering) * dt)
lead_x_next = lead_x + lead_speed * dt
```

The lead vehicle is at road-center lateral position zero. Collision uses a conservative overlap test on the swept longitudinal interval (±7m clearance) and lateral proximity (<2.5m). Leaving the allowed center range ±2.25m terminates the episode. This range is a 3.5m road half-width minus a 1.25m vehicle half-width; it is a toy single-road corridor, not a standard lane specification. Reaching `x>=300m` is success. The default time limit is 300 steps. Collision/road-departure termination takes precedence over success, and all physical terminations take precedence over truncation.

```text
reward = dt * (v_next*cos(yaw_next)/20
               - 0.5*y_next² - ((v_next-20)/20)² - 0.05*steering²)
         - 10 for collision or offroad
         + 10 for success without collision/offroad
```

Counter-based hashing of `(seed, environment index, episode index, field)` initializes position, heading, ego speed, lead gap and lead speed. Full `reset(seed=...)` repeats the same initial conditions. `reset_done()` increments episode counters only for finished environments and leaves the others untouched. Finished environments remain frozen, return zero reward on later `step()` calls, and retain their terminal observation until reset. Returned Python arrays own their memory so saved rollouts are not overwritten by subsequent steps. A handle is single-caller; independent handles may be used concurrently. Close an environment/context manager to join CPU workers and release GPU allocations.

## PPO and reproducibility

The trainer uses a shared two-layer 64-unit tanh network, Gaussian latent actions transformed by tanh, a value head, clipped PPO objective, GAE, gradient clipping and minibatch Adam updates. It stores latent actions and the change-of-variable log probability for the policy ratio. The entropy bonus uses latent Gaussian entropy as a regularizer, rather than claiming an exact squashed-policy entropy.

Critically, values for **terminal observations are computed before resetting**. True terminations suppress value bootstrapping. Time limits bootstrap the final observation's value, while both termination and truncation stop GAE from propagating into a new episode. Unit tests exercise this distinction.

Outputs under `runs/ppo/` include `policy.pt` and `training.json` with configuration, seed, PyTorch version, losses, rollout vs optimization time, and deterministic evaluation before/after training on seed 2026. Evaluation runs exactly one episode per environment. Seeds and deterministic PyTorch algorithms make same-environment reruns inspectable; bitwise reproducibility across CPU/GPU/compiler/library versions is not promised. One successful training seed is not proof of robust policy quality. Use held-out scenarios and multiple training/evaluation seeds for an actual study.

## Bridge to existing calibration

Generate and calibrate frames using the existing pipeline, then:

```bash
python sim/python/calibrated_camera.py \
  --calibration-log logs/demo_vehicle/front_camera/calib_YOUR_TIMESTAMP.json
```

The bridge requires an accepted local audit record, rechecks the fixed quality thresholds, and uses its intrinsic parameters and distortion coefficients to project 9 lane points per vehicle. The assumed extrinsics are explicitly fixed: camera height 2.5m, level optical axis forward, optical x right, optical y down. **Intrinsic calibration does not estimate this mounting pose.** Pixel coordinates and visibility masks are saved to `runs/lane_projection.json`. This connects the original artifact format to simulated sensor geometry without pretending the original calibration measured vehicle extrinsics.

## Benchmark and profiling methodology

```bash
python sim/python/benchmark_sim.py --workers 1 --output runs/cpu-one.json
python sim/python/benchmark_sim.py --workers 4 --output runs/cpu-four.json
python sim/python/benchmark_sim.py --backend cuda --output runs/cuda.json
nsys profile --trace=cuda,osrt -o runs/sim-profile \
  python sim/python/benchmark_sim.py --backend cuda --counts 65536
ncu --target-processes all --kernel-name regex:step_kernel \
  python sim/python/benchmark_sim.py --backend cuda --counts 65536 --steps 20
```

Benchmarks warm up 10 batches, reset the seed, and time actual steps **including resets of finished environments**. They count useful transitions, rather than benchmarking frozen terminal slots. The fixed 32-step action table keeps action-generation noise outside the measured region. JSON captures batch size, workers, steps, seed, git revision/dirty status, platform, NumPy/Python versions, wall time, throughput, p50/p95 latency, final-state SHA-256 and checksum. SHA-256 is useful for same-backend reproducibility; CPU/GPU comparison should use numerical tolerances.

CPU compute timing includes the worker dispatch/wait. CUDA events split upload, kernel, and download times, excluding reset kernels and host work; the aggregate wall-clock throughput includes them. State and output buffers remain allocated across steps. CUDA allocates approximately 70 bytes/environment of device buffers and 70 bytes/environment of pinned host buffers, excluding CUDA runtime overhead; CPU native buffers use 62 bytes/environment plus thread stacks. A CUDA step transfers 62 bytes/environment, including raw-state diagnostics as well as RL observations. These are design-derived buffer/transfer counts, not a measured peak-memory claim. Profile with Nsight before optimizing. Avoid assuming that more CPU threads or GPU execution helps small batches.

CI compiles CUDA without a GPU on an ordinary GitHub Linux runner. The separately dispatched GPU validation workflow requires a **trusted self-hosted Linux NVIDIA runner labeled `gpu`** and explicitly checks that GPU execution is available before running parity/memory checks. It is never triggered automatically for public pull requests. No self-hosted runner is provisioned by this project. Local macOS verification exercises CPU code only.

## References

- [NVIDIA CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [Gymnasium termination and truncation semantics](https://gymnasium.farama.org/api/env/)
- [PyTorch PPO tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html)
