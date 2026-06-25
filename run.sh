#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/qijunchen/miniconda3/envs/agent/bin/python
LLM_BASE_URL=${LLM_BASE_URL:-http://127.0.0.1:8001/v1}

if [ "${SUBTYPE_REVIEW_ALLOW_LLM_FALLBACK:-0}" != "1" ]; then
  if ! curl -fsS "${LLM_BASE_URL}/models" >/dev/null 2>&1; then
    echo "[run] Local LLM server is not available at ${LLM_BASE_URL}."
    echo "[run] Start it first with ./llm.sh, or set SUBTYPE_REVIEW_ALLOW_LLM_FALLBACK=1 to run diagnostic fallback only."
    exit 2
  fi
fi

"${PYTHON}" main.py \
  --data-json-path /data/qijun/path-ct/data/tcga_kirc_data.json \
  --output-root /data/qijun/path-ct/output_kirc \
  --config-dir /data/qijun/path-ct/configs
