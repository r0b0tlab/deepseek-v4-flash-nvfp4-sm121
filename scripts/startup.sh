#!/usr/bin/env bash
# /usr/local/bin/startup.sh
# DeepSeek-V4-Flash NVFP4 serving startup script for SM121 dual GB10

set -euo pipefail

MODEL_DIR="/mnt/model"
HF_TOKEN="${HF_TOKEN:-}"
MODEL_ID="${MODEL_ID:-nvidia/DeepSeek-V4-Flash-NVFP4}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_SEQS="${MAX_SEQS:-2}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Auto-download if weights not mounted
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ] && [ -n "$HF_TOKEN" ]; then
    echo "Downloading model from HuggingFace: $MODEL_ID"
    mkdir -p "$MODEL_DIR"
    huggingface-cli download "$MODEL_ID" \
        --local-dir "$MODEL_DIR" \
        --token "$HF_TOKEN" \
        --local-dir-use-symlinks False
fi

# Verify checkpoint
if [ ! -f "$MODEL_DIR/hf_quant_config.json" ]; then
    echo "ERROR: hf_quant_config.json not found. Mount model weights or set HF_TOKEN." >&2
    exit 1
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: config.json not found. Invalid model directory." >&2
    exit 1
fi

# Log configuration
echo "Starting DeepSeek-V4-Flash NVFP4 on SM121"
echo "  Model: $MODEL_ID"
echo "  Model dir: $MODEL_DIR"
echo "  Port: $PORT"
echo "  GPU memory utilization: $GPU_MEM_UTIL"
echo "  Max model length: $MAX_MODEL_LEN"
echo "  Max sequences: $MAX_SEQS"
echo "  KV cache dtype: $KV_CACHE_DTYPE"

# SM121-specific environment
export VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass
export VLLM_FP8_MOE_BACKEND=flashinfer_cutlass
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUTE_DSL_ARCH=sm_121a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Network (QSFP) - adjust for your setup
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp1s0f0np0}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-enp1s0f0np0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f0np0}"

# Launch vLLM
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --port "$PORT" \
    --tensor-parallel-size 2 \
    --distributed-executor-backend ray \
    --trust-remote-code \
    --dtype auto \
    --quantization modelopt \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --attention-backend flashinfer \
    --moe-backend flashinfer_cutlass \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --max-num-batched-tokens 32768 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --no-enable-expert-parallel \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    $EXTRA_ARGS
