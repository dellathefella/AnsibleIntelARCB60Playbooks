# =============================================================================
# engine/Dockerfile — arc-flash-engine: llama.cpp SYCL for Intel Arc Pro B60
#
# Qwen3.8-Flash-Next (qwen4exp) capable build of llama.cpp master for the B60
# (BMG-G21, Xe2). Derived from kernel-work/Dockerfile.tuned (same pinned IGC /
# compute-runtime stack as upstream full-intel) with:
#
#   - UPSTREAM_SHA on master (qwen4exp arch + gated-DeltaNet SYCL kernels +
#     MTP support, merged 2026-10-01..02; the ghcr full-intel release image
#     lags master and cannot load Flash-Next yet)
#   - no patch step (engine fork work lands via UPSTREAM_REPO once the fork
#     repo exists; until then this is pristine master)
#   - llama-cli + llama-bench included for viability/Phase-0 testing
#   - safe fast-math subset + GGML_SYCL_F16=ON (measured +0.5% decode,
#     neutral-to-positive; full -ffast-math is UB for -INFINITY masking)
#
# Build (on halo1, rootful podman):
#   podman build -f Dockerfile -t localhost/llama.cpp:arc-engine .
# =============================================================================

ARG ONEAPI_VERSION=2026.1.1-devel-ubuntu24.04
# Builds from the dellathefella fork; arc-flash-engine branch tracks upstream
# master (qwen4exp capable). Engine changes land on this branch.
ARG UPSTREAM_REPO=https://github.com/dellathefella/llama.cpp.git
ARG UPSTREAM_SHA=arc-flash-engine
# bump to invalidate the clone+build layer cache on rebuilds
ARG ENGINE_REV=1
ARG GGML_SYCL_DEVICE_ARCH=

## Build Image
FROM docker.io/intel/oneapi-toolkit:${ONEAPI_VERSION} AS build

ARG UPSTREAM_REPO
ARG UPSTREAM_SHA
ARG ENGINE_REV
ARG GGML_SYCL_DEVICE_ARCH
ARG LEVEL_ZERO_VERSION=1.28.2
ARG LEVEL_ZERO_UBUNTU_VERSION=u24.04

RUN echo "engine rev ${ENGINE_REV}" && \
    apt-get update && \
    apt-get install -y git libssl-dev wget ca-certificates && \
    cd /tmp && \
    wget -q "https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/level-zero_${LEVEL_ZERO_VERSION}%2B${LEVEL_ZERO_UBUNTU_VERSION}_amd64.deb" -O level-zero.deb && \
    wget -q "https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/level-zero-devel_${LEVEL_ZERO_VERSION}%2B${LEVEL_ZERO_UBUNTU_VERSION}_amd64.deb" -O level-zero-devel.deb && \
    apt-get -o Dpkg::Options::="--force-overwrite" install -y ./level-zero.deb ./level-zero-devel.deb && \
    rm -f /tmp/level-zero.deb /tmp/level-zero-devel.deb

RUN git config --global --add safe.directory /app && \
    git clone ${UPSTREAM_REPO} /app && \
    cd /app && \
    git checkout ${UPSTREAM_SHA}

WORKDIR /app

RUN AOT_FLAGS="" && \
    if [ -n "${GGML_SYCL_DEVICE_ARCH}" ]; then \
        AOT_FLAGS="-DGGML_SYCL_DEVICE_ARCH=${GGML_SYCL_DEVICE_ARCH}"; \
    fi && \
    cmake -B build \
      -DGGML_NATIVE=OFF \
      -DGGML_SYCL=ON \
      -DCMAKE_C_COMPILER=icx \
      -DCMAKE_CXX_COMPILER=icpx \
      -DGGML_BACKEND_DL=ON \
      -DGGML_CPU_ALL_VARIANTS=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DGGML_SYCL_F16=ON \
      -DCMAKE_CXX_FLAGS="-fno-math-errno -fno-signed-zeros -fno-trapping-math -ffp-contract=fast" \
      ${AOT_FLAGS} && \
    cmake --build build --config Release -j$(nproc)

RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

## Runtime — same pinned compute stack as upstream full-intel
FROM docker.io/intel/oneapi-toolkit:${ONEAPI_VERSION} AS base

ARG IGC_VERSION=v2.34.4
ARG IGC_VERSION_FULL=2_2.34.4+21428
ARG COMPUTE_RUNTIME_VERSION=26.18.38308.1
ARG COMPUTE_RUNTIME_VERSION_FULL=26.18.38308.1-0
ARG IGDGMM_VERSION=22.10.0
RUN mkdir /tmp/neo/ && cd /tmp/neo/ \
  && wget https://github.com/intel/intel-graphics-compiler/releases/download/$IGC_VERSION/intel-igc-core-${IGC_VERSION_FULL}_amd64.deb \
  && wget https://github.com/intel/intel-graphics-compiler/releases/download/$IGC_VERSION/intel-igc-opencl-${IGC_VERSION_FULL}_amd64.deb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/intel-ocloc_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/libigdgmm12_${IGDGMM_VERSION}_amd64.deb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
  && dpkg --install *.deb

RUN apt-get update \
    && apt-get install -y libgomp1 curl \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

### Server + CLI tools
FROM base AS server

COPY --from=build /app/lib/ /app
COPY --from=build /app/build/bin/llama-server /app/llama-server
COPY --from=build /app/build/bin/llama-cli /app/llama-cli
COPY --from=build /app/build/bin/llama-bench /app/llama-bench

WORKDIR /app

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
