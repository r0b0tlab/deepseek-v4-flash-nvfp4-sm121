# DeepSeek-V4-Flash NVFP4 — SM121 Optimized Serving

Containerized serving stack for [NVIDIA's DeepSeek-V4-Flash NVFP4 checkpoint](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4) on dual NVIDIA DGX Spark (GB10, SM12.1).

## Hardware Requirements

- **Dual NVIDIA DGX Spark** (GB10, SM12.1)
- **Total VRAM:** 243 GiB (~260 GB)
- **Interconnect:** QSFP (100 Gbps)
- **Storage:** 200 GB free for model weights

## Model Details

| Property | Value |
|----------|-------|
| Base model | `deepseek-ai/DeepSeek-V4-Flash` |
| Checkpoint | `nvidia/DeepSeek-V4-Flash-NVFP4` |
| Size | 168.30 GB |
| Quantization | MIXED_PRECISION: Experts NVFP4 (W4A4), Attention/BF16 |
| Total params | 685B |
| Active params | ~350B (MoE top-6) |
| MTP modules | 1 |
| Max context | 1,048,576 tokens |

## Quick Start

### Option A: Mount Pre-downloaded Weights

```bash
docker run -d --name deepseek-v4 \
  --gpus all --ipc=host --network=host \
  -v /path/to/weights:/mnt/model:ro \
  -e GPU_MEM_UTIL=0.70 \
  ghcr.io/r0b0tlab/deepseek-v4-flash-nvfp4-sm121:latest
```

### Option B: Auto-download from HuggingFace

```bash
docker run -d --name deepseek-v4 \
  --gpus all --ipc=host --network=host \
  -e HF_TOKEN=*** \
  -e GPU_MEM_UTIL=0.70 \
  ghcr.io/r0b0tlab/deepseek-v4-flash-nvfp4-sm121:latest
```

### Option C: Docker Compose

```bash
# Edit docker-compose.yml to set your paths and token
docker-compose up -d
```

## Verified Launch Flags

| Flag | Value | Why |
|------|-------|-----|
| `--gpu-memory-utilization` | 0.70 | 168 GB model on 243 GB total; 0.85 causes worker death |
| `--kv-cache-dtype` | fp8 | NVFP4 KV blocked on SM12.1 (torch.nvfp4 missing in public PyTorch) |
| `--moe-backend` | flashinfer_cutlass | Marlin buggy on SM12.x for large MoE |
| `--attention-backend` | flashinfer | Native SM120/SM121 FP4 tensor cores |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":3}` | +63-77% throughput if acceptance >60% |
| `--max-num-seqs` | 2 | Memory headroom for 168 GB model |
| `--no-enable-expert-parallel` | | EP regresses on SM121 (SymmMem unavailable) |
| `--compilation-config` | `mode:3, cudagraph:none` | CUDA graph deadlock on cross-node SHM |

## vLLM Support Status

- **Native support:** Pending PR [#43477](https://github.com/vllm-project/vllm/pull/43477) ("Enable DeepSeek V4 on SM120")
- **Fallback:** `trust_remote_code=True` with manual model class registration
- **Current approach:** Container uses `trust_remote_code` until PR is merged

## Known Limitations

1. **NVFP4 KV cache blocked:** SM12.1 lacks `torch.nvfp4` dtype in public PyTorch. Use FP8 KV workaround.
2. **MTP speculative decoding:** Requires vLLM PR #43477 or dev build. Container includes fallback config.
3. **Worker stability:** GPU_UTIL > 0.70 causes worker death after 3-5 min on dual GB10.
4. **CUDA graphs:** Disabled (`cudagraph_mode: none`) due to cross-node SHM deadlock.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Worker dies after idle | Lower `GPU_MEM_UTIL` to 0.70 |
| NVFP4 KV error | Switch to `fp8` KV cache dtype |
| MTP not accepted | Verify `num_mtp_modules=1` in config.json |
| OOM during load | Use `device_map="cpu"` + manual `.cuda()` transfer |
| vLLM registry error | Track PR #43477; use `trust_remote_code` fallback |
| `cudaErrorLaunchFailure` | See vLLM issue #45475; disable KV offloading |

## Benchmarks

Benchmarks run on dual GB10 (SM12.1) with MTP K=3:

| Concurrency | Context | Throughput (tok/s) | TTFT (ms) | GPU Util |
|-------------|---------|-------------------|-----------|----------|
| 1 | 128 | TBD | TBD | TBD |
| 1 | 2048 | TBD | TBD | TBD |
| 2 | 128 | TBD | TBD | TBD |
| 5 | 128 | TBD | TBD | TBD |

*Results to be populated after benchmark phase.*

## Building from Source

```bash
# Clone
git clone https://github.com/r0b0tlab/deepseek-v4-flash-nvfp4-sm121.git
cd deepseek-v4-flash-nvfp4-sm121

# Build
docker build -t deepseek-v4-nvfp4-sm121 .

# Run
docker run -d --name deepseek-v4 \
  --gpus all --ipc=host --network=host \
  -v /path/to/weights:/mnt/model:ro \
  -e GPU_MEM_UTIL=0.70 \
  deepseek-v4-nvfp4-sm121
```

## Credits

- **DeepSeek-AI:** Base model architecture (`deepseek-ai/DeepSeek-V4-Flash`)
- **NVIDIA:** ModelOpt quantization + NVFP4 checkpoint (`nvidia/DeepSeek-V4-Flash-NVFP4`)
- **r0b0tlab:** Container, SM121 tuning, documentation

## License

Container scripts: MIT  
Model weights: Subject to DeepSeek-AI and NVIDIA license terms

## References

- [NVIDIA Checkpoint](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4)
- [vLLM PR #43477](https://github.com/vllm-project/vllm/pull/43477)
- [vLLM Issue #45475](https://github.com/vllm-project/vllm/issues/45475)
- [DeepSeek-V4 Paper](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4/resolve/main/DeepSeek_V4.pdf)
