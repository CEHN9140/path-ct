#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/qijunchen/miniconda3/envs/agent/bin/python
MODEL_PATH=/data/qijun/path-ct/models/qwen2.5-72b-instruct
MODEL_NAME=qwen2.5
LLM_HOST=${LLM_HOST:-127.0.0.1}
LLM_PORT=${LLM_PORT:-8001}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.80}
# Set the GPU IDs used by the LLM server here.
# Examples:
#   LLM_GPUS=${LLM_GPUS:-0,1,2,3}
#   LLM_GPUS=${LLM_GPUS:-0,1}
LLM_GPUS=${LLM_GPUS:-0,1,2,3}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-}
LLM_BASE_URL=http://${LLM_HOST}:${LLM_PORT}/v1
LLM_LOG=${LLM_LOG:-/data/qijun/path-ct/output_kirc_v3/llm_server.log}

mkdir -p "$(dirname "${LLM_LOG}")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${LLM_GPUS}" ]; then
  export CUDA_VISIBLE_DEVICES="${LLM_GPUS}"
fi
if [ -z "${TENSOR_PARALLEL_SIZE}" ]; then
  if [ -n "${LLM_GPUS}" ]; then
    TENSOR_PARALLEL_SIZE=$("${PYTHON}" - <<PY
gpus = "${LLM_GPUS}".strip()
print(len([item for item in gpus.split(",") if item.strip()]))
PY
)
  else
    TENSOR_PARALLEL_SIZE=1
  fi
fi

if curl -fsS "${LLM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "[llm] Local LLM server is already running at ${LLM_BASE_URL}."
  exit 0
fi

echo "[llm] Starting local LLM server at ${LLM_BASE_URL}."
echo "[llm] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}, tensor_parallel_size=${TENSOR_PARALLEL_SIZE}."
echo "[llm] Log: ${LLM_LOG}"

exec "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME}" \
  --host "${LLM_HOST}" \
  --port "${LLM_PORT}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  2>&1 | tee "${LLM_LOG}"
