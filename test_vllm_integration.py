#!/usr/bin/env python3
"""Test vLLM DeepSeek-V4 registry support and trust_remote_code fallback."""
import sys
import json

def check_vllm_registry():
    """Check if vLLM has DeepSeek-V4 support."""
    try:
        from vllm.model_executor.models import ModelRegistry
        supported = ModelRegistry.get_supported_archs()
        supported_str = [str(a) for a in supported]
        
        deepseek_v4 = any('DeepseekV4' in s for s in supported_str)
        deepseek_v2 = any('DeepseekV2' in s for s in supported_str)
        deepseek_mtp = any('DeepseekMTP' in s for s in supported_str)
        
        print(f"vLLM registry check:")
        print(f"  DeepSeek-V4 support: {'YES' if deepseek_v4 else 'NO'}")
        print(f"  DeepSeek-V2 support: {'YES' if deepseek_v2 else 'NO'}")
        print(f"  DeepSeek-MTP support: {'YES' if deepseek_mtp else 'NO'}")
        
        if deepseek_v4:
            print("  ✅ Native DeepSeek-V4 support available")
            return True
        else:
            print("  ❌ No native DeepSeek-V4 support")
            print(f"  Available DeepSeek variants: {[s for s in supported_str if 'Deepseek' in s]}")
            return False
            
    except ImportError as e:
        print(f"  ❌ vLLM not installed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error checking registry: {e}")
        return False

def test_trust_remote_code(model_path: str = "deepseek-ai/DeepSeek-V4-Flash"):
    """Test loading with trust_remote_code."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        
        print(f"\nTesting trust_remote_code with {model_path}:")
        
        # Test config loading
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"  Config loaded: {config.model_type}")
        print(f"  Architecture: {config.architectures}")
        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Experts: {config.n_routed_experts}")
        print(f"  Experts per tok: {config.num_experts_per_tok}")
        print(f"  MTP modules: {getattr(config, 'num_nextn_predict_layers', 'N/A')}")
        
        # Check auto_map
        auto_map = getattr(config, 'auto_map', None)
        if auto_map:
            print(f"  auto_map: {auto_map}")
        else:
            print(f"  ❌ No auto_map — vLLM may need manual registration")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_nvfp4_checkpoint(checkpoint_path: str = "nvidia/DeepSeek-V4-Flash-NVFP4"):
    """Test loading NVIDIA NVFP4 checkpoint config."""
    try:
        from transformers import AutoConfig
        
        print(f"\nTesting NVIDIA NVFP4 checkpoint {checkpoint_path}:")
        
        config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
        print(f"  Config loaded: {config.model_type}")
        
        # Check quantization config
        quant_cfg = getattr(config, 'quantization_config', None)
        if quant_cfg:
            print(f"  Quantization: {quant_cfg.get('quant_algo', 'N/A')}")
            print(f"  MoE quant: {quant_cfg.get('moe_quant_algo', 'N/A')}")
            print(f"  Group size: {quant_cfg.get('group_size', 'N/A')}")
            print(f"  Ignore list: {quant_cfg.get('ignore', 'N/A')}")
        else:
            print(f"  ❌ No quantization_config found")
        
        # Check for hf_quant_config.json
        import os
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
        print(f"  HF cache: {hf_cache}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("DeepSeek-V4 vLLM Integration Test")
    print("=" * 60)
    
    # Check vLLM registry
    vllm_ok = check_vllm_registry()
    
    # Test trust_remote_code with base model
    base_ok = test_trust_remote_code("deepseek-ai/DeepSeek-V4-Flash")
    
    # Test NVFP4 checkpoint (if cached)
    nvfp4_ok = test_nvfp4_checkpoint("nvidia/DeepSeek-V4-Flash-NVFP4")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  vLLM registry: {'PASS' if vllm_ok else 'FAIL'}")
    print(f"  trust_remote_code: {'PASS' if base_ok else 'FAIL'}")
    print(f"  NVFP4 checkpoint: {'PASS' if nvfp4_ok else 'FAIL'}")
    print("=" * 60)
    
    sys.exit(0 if (vllm_ok or base_ok) else 1)
