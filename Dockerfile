FROM vllm/vllm-openai:v0.22.0-aarch64-ubuntu2404

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/deepseek-v4-flash-nvfp4-sm121"
LABEL org.opencontainers.image.description="DeepSeek-V4-Flash NVFP4 serving container for SM121 (dual GB10)"
LABEL org.opencontainers.image.authors="r0b0tlab"

# Install huggingface-cli for auto-download
RUN pip install -U huggingface-hub

# DeepSeek model code dependencies (trust_remote_code)
RUN pip install -U einops transformers-stream-generator

# Copy scripts
COPY scripts/startup.sh /usr/local/bin/startup.sh
COPY scripts/healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/*.sh

# Environment defaults
ENV MODEL_ID=nvidia/DeepSeek-V4-Flash-NVFP4
ENV PORT=8000
ENV GPU_MEM_UTIL=0.70
ENV MAX_MODEL_LEN=32768
ENV MAX_SEQS=2
ENV KV_CACHE_DTYPE=fp8
ENV EXTRA_ARGS=""
ENV HF_TOKEN=""

# SM121-specific optimizations
ENV VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass
ENV VLLM_FP8_MOE_BACKEND=flashinfer_cutlass
ENV FLASHINFER_DISABLE_VERSION_CHECK=1
ENV CUTE_DSL_ARCH=sm_121a
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Network (QSFP) - adjust for your setup
ENV GLOO_SOCKET_IFNAME=enp1s0f0np0
ENV TP_SOCKET_IFNAME=enp1s0f0np0
ENV NCCL_SOCKET_IFNAME=enp1s0f0np0

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD bash /usr/local/bin/healthcheck.sh

EXPOSE 8000

ENTRYPOINT ["bash", "/usr/local/bin/startup.sh"]
