#!/bin/bash
# qwen38-27b-vllm-xpu-pp4-start.sh — Qwen3.8-27B (AWQ W4A16) via vLLM XPU, PP=4
# Pipeline parallel across four Arc Pro B60 cards (port 8082).
set -euo pipefail

CONTAINER="qwen38-27b-vllm-xpu-pp4"
PORT="8082"
REPO_DIR="/home/jdella/.local/share/b60-llama/models/Qwen3.8-27B-W4A16-AWQ"
CONTAINER_REPO="/models/Qwen3.8-27B-W4A16-AWQ"
IMAGE="docker.io/vllm/vllm-openai-xpu:latest"
CTX="262144"
PP="4"
XPU_DEVICES="0,1,2,3"
GPU_MEM_UTIL="0.90"
KV_DTYPE="fp8"
BATCH="16384"
SEQS="32"
DTYPE="float16"
SERVED_MODEL="qwen3.8-27b"

podman rm -f "$CONTAINER" 2>/dev/null || true
echo "Starting vLLM XPU PP=$PP container on port $PORT"
setsid podman run -d --name "$CONTAINER" --restart=unless-stopped \
  --device /dev/dri \
  --group-add keep-groups \
  --security-opt label=disable \
  --ipc=host \
  -v "${REPO_DIR}:${CONTAINER_REPO}:ro" \
  -e ZE_AFFINITY_MASK=${XPU_DEVICES} \
  -e HF_HUB_OFFLINE=1 \
  -e CCL_ENABLE_SYCL_KERNELS=1 \
  -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
  -e CCL_ATL_TRANSPORT=ofi \
  -e CCL_ZE_IPC_EXCHANGE=pidfd \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:* \
  -p ${PORT}:${PORT} \
  "$IMAGE" \
  "${CONTAINER_REPO}" \
    --served-model-name "$SERVED_MODEL" \
    --pipeline-parallel-size "$PP" \
    --tensor-parallel-size 1 \
    --max-model-len "$CTX" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-cache-dtype "$KV_DTYPE" \
    --max-num-batched-tokens "$BATCH" \
    --max-num-seqs "$SEQS" \
    --dtype "$DTYPE" \
    --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image": 0}' \
    --host 0.0.0.0 --port "$PORT"

echo "Waiting for PP=4 health..."
for i in $(seq 1 240); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "=== PP=4 ready on port $PORT ==="
    exit 0
  fi
  sleep 5
done
echo "WARNING: PP=4 did not become healthy within 20 minutes" >&2
echo "         check: podman logs $CONTAINER" >&2
exit 1
