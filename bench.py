#!/usr/bin/env python3
"""Benchmark script for DeepSeek-V4-Flash NVFP4 on dual GB10."""
import argparse
import json
import time
import sys
import os
from datetime import datetime

def run_benchmark(model_path, concurrency, context_length, mtp_k, output_file):
    """Run vLLM benchmark and capture metrics."""
    
    print(f"Benchmark: DeepSeek-V4-Flash NVFP4")
    print(f"  Model: {model_path}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Context: {context_length}")
    print(f"  MTP K: {mtp_k}")
    print(f"  Output: {output_file}")
    print()
    
    # Import vLLM
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vLLM not installed. Run: pip install vllm")
        return False
    
    # Build launch args
    launch_args = {
        "model": model_path,
        "tensor_parallel_size": 2,
        "distributed_executor_backend": "ray",
        "trust_remote_code": True,
        "dtype": "auto",
        "quantization": "modelopt",
        "kv_cache_dtype": "fp8",
        "attention_backend": "flashinfer",
        "moe_backend": "flashinfer_cutlass",
        "gpu_memory_utilization": 0.70,
        "max_model_len": 32768,
        "max_num_seqs": concurrency,
        "max_num_batched_tokens": 32768,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
    }
    
    if mtp_k > 0:
        launch_args["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": mtp_k,
        }
    
    print("Launching vLLM...")
    start_time = time.time()
    
    try:
        llm = LLM(**launch_args)
    except Exception as e:
        print(f"ERROR: Failed to launch vLLM: {e}")
        return False
    
    load_time = time.time() - start_time
    print(f"  Model loaded in {load_time:.1f}s")
    
    # Prepare prompts
    prompts = [
        "The future of artificial intelligence is" * (context_length // 10)
        for _ in range(concurrency)
    ]
    
    # Sampling params
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=128,
    )
    
    # Warmup
    print("Warming up...")
    try:
        _ = llm.generate(prompts[0], sampling_params)
    except Exception as e:
        print(f"WARNING: Warmup failed: {e}")
    
    # Benchmark run
    print(f"Running benchmark with {concurrency} concurrent requests...")
    start_time = time.time()
    
    try:
        outputs = llm.generate(prompts, sampling_params)
    except Exception as e:
        print(f"ERROR: Benchmark failed: {e}")
        return False
    
    total_time = time.time() - start_time
    
    # Collect metrics
    total_tokens = 0
    total_prompt_tokens = 0
    for output in outputs:
        total_tokens += len(output.outputs[0].token_ids)
        total_prompt_tokens += len(output.prompt_token_ids)
    
    throughput = total_tokens / total_time
    ttft = total_time / concurrency  # Approximate
    
    # GPU metrics (if available)
    gpu_util = 0
    gpu_temp = 0
    gpu_power = 0
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                util, temp, power = line.split(", ")
                gpu_util += float(util)
                gpu_temp += float(temp)
                gpu_power += float(power)
            gpu_util /= len(lines)
            gpu_temp /= len(lines)
            gpu_power /= len(lines)
    except Exception:
        pass
    
    # Build results
    results = {
        "model": model_path,
        "concurrency": concurrency,
        "context_length": context_length,
        "mtp_k": mtp_k,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "load_time_sec": load_time,
            "total_time_sec": total_time,
            "throughput_tok_per_sec": throughput,
            "ttft_ms": ttft * 1000,
            "total_tokens": total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "gpu_utilization": gpu_util,
            "gpu_temperature": gpu_temp,
            "gpu_power_watts": gpu_power,
        }
    }
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print()
    print("Results:")
    print(f"  Throughput: {throughput:.1f} tok/s")
    print(f"  TTFT: {ttft * 1000:.1f} ms")
    print(f"  GPU util: {gpu_util:.1f}%")
    print(f"  GPU temp: {gpu_temp:.1f}°C")
    print(f"  GPU power: {gpu_power:.1f}W")
    print(f"  Saved to: {output_file}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V4-Flash NVFP4")
    parser.add_argument("--model", default="/home/r0b0tdgx/deepseek-v4-nvfp4/weights", help="Model path")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--context", type=int, default=128, help="Context length")
    parser.add_argument("--mtp", type=int, default=3, help="MTP K value (0 to disable)")
    parser.add_argument("--output", default="benchmark_result.json", help="Output file")
    
    args = parser.parse_args()
    
    success = run_benchmark(
        args.model,
        args.concurrency,
        args.context,
        args.mtp,
        args.output
    )
    
    sys.exit(0 if success else 1)
