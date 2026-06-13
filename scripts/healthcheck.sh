#!/usr/bin/env bash
# /usr/local/bin/healthcheck.sh
# Healthcheck for DeepSeek-V4-Flash NVFP4 serving container

set -euo pipefail

PORT="${PORT:-8000}"

# Check if vLLM API server is responding
curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1 || {
    echo "Health check failed: vLLM API not responding on port ${PORT}"
    exit 1
}

# Check GPU memory utilization (should be < 95% to avoid OOM)
if command -v nvidia-smi > /dev/null 2>&1; then
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    if [ "$GPU_UTIL" -gt 95 ]; then
        echo "Health check warning: GPU utilization at ${GPU_UTIL}%"
        exit 1
    fi
fi

echo "Health check passed"
exit 0
