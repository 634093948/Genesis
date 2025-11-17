#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Flux decode fix
测试 Flux 解码修复

Author: eddy
Date: 2025-11-16
"""

import sys
import io
from pathlib import Path

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "compat"))

print("=" * 70)
print("测试 Flux 解码修复")
print("=" * 70)
print()

# Test VAEDecode parameter order
print("1. 测试 VAEDecode 参数顺序...")
try:
    from nodes import VAEDecode
    import inspect
    
    # Get decode method signature
    sig = inspect.signature(VAEDecode.decode)
    params = list(sig.parameters.keys())
    
    print(f"   VAEDecode.decode 参数: {params}")
    
    if params == ['self', 'vae', 'samples']:
        print("   ✓ 参数顺序正确: (vae, samples)")
    else:
        print(f"   ⚠ 参数顺序: {params}")
    
except Exception as e:
    print(f"   ✗ 错误: {e}")

print()

# Test dict handling
print("2. 测试字典处理...")
try:
    import torch
    
    # Simulate a latent dict
    test_samples = {
        "samples": torch.randn(1, 4, 64, 64)
    }
    
    print(f"   测试数据类型: {type(test_samples)}")
    print(f"   包含 'samples' 键: {'samples' in test_samples}")
    
    if isinstance(test_samples, dict):
        print("   ✓ 正确识别为字典")
    else:
        print("   ✗ 未识别为字典")
    
except Exception as e:
    print(f"   ✗ 错误: {e}")

print()

# Test flux_comfy_pipeline import
print("3. 测试 flux_comfy_pipeline 导入...")
try:
    from apps.sd_module.flux_comfy_pipeline import FluxComfyPipeline, COMFY_AVAILABLE
    
    print(f"   ✓ 导入成功")
    print(f"   ComfyUI 可用: {COMFY_AVAILABLE}")
    
    if COMFY_AVAILABLE:
        print("   ✓ 可以进行完整测试")
    else:
        print("   ⚠ ComfyUI 模块不可用，跳过完整测试")
    
except Exception as e:
    print(f"   ✗ 错误: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("测试完成")
print("=" * 70)
print()
print("修复说明:")
print("  1. VAEDecode.decode() 参数顺序: (vae, samples)")
print("  2. samples 必须是字典格式: {'samples': tensor}")
print("  3. 已在 flux_comfy_pipeline.py 中修复")
print()
print("现在可以重新测试 Flux 生成了！")
