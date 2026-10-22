PYTHON ?= python3
CMAKE ?= cmake
BUILD_DIR ?= build
CMAKE_ARGS ?=
JOBS ?= 4
SCALE ?= 1.0
IMAGE_DIR ?= sample_data/checkerboard_frames
VEHICLE_ID ?= demo_vehicle
SENSOR_ID ?= front_camera

.PHONY: build test lint generate-data run-pipeline e2e asan
build:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release $(CMAKE_ARGS)
	$(CMAKE) --build $(BUILD_DIR) --parallel $(JOBS)

lint:
	$(PYTHON) -m ruff check --isolated --target-version py310 --select E4,E7,E9,F,I scripts tests
	$(PYTHON) -m black --check scripts tests

test: build lint
	$(CMAKE) --build $(BUILD_DIR) --target test -- ARGS=--output-on-failure
	$(PYTHON) -m pytest -q tests/test_pipeline.py

generate-data:
	$(PYTHON) scripts/generate_synthetic_data.py --scale $(SCALE) --output-dir "$(IMAGE_DIR)"

run-pipeline:
	$(PYTHON) scripts/calib_pipeline.py --binary "$(BUILD_DIR)/calib_engine" --image-dir "$(IMAGE_DIR)" --vehicle-id "$(VEHICLE_ID)" --sensor-id "$(SENSOR_ID)"

e2e: build
	PYTHON="$(PYTHON)" CALIB_BINARY="$(abspath $(BUILD_DIR))/calib_engine" bash scripts/run_end_to_end_test.sh

asan:
	$(MAKE) test BUILD_DIR=build-asan CMAKE_ARGS="$(CMAKE_ARGS) -DENABLE_ASAN=ON"
