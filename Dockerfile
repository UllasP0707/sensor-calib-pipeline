FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ cmake make git ca-certificates libopencv-dev nlohmann-json3-dev libgtest-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY include ./include
COPY src ./src
COPY tests/test_calib_engine.cpp ./tests/test_calib_engine.cpp
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    && cmake --build build --parallel 4 \
    && ctest --test-dir build --output-on-failure

FROM ubuntu:22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates libopencv-core4.5d libopencv-imgproc4.5d \
    libopencv-imgcodecs4.5d libopencv-calib3d4.5d libopencv-highgui4.5d \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 calib
WORKDIR /app
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY --from=builder /src/build/calib_engine ./build/calib_engine
COPY scripts ./scripts
COPY configs ./configs
RUN mkdir -p /app/logs /app/sample_data && chown -R calib:calib /app
USER calib
ENTRYPOINT ["python3", "/app/scripts/calib_pipeline.py"]
CMD ["--help"]
