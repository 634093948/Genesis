#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 WanVideoSampler 参数"""

import sys
from pathlib import Path
import inspect

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "compat"))
sys.path.insert(0, str(project_root / "custom_nodes" / "Comfyui"))

print("Loading WanVideoSampler...")
sys.path.insert(0, str(project_root / "custom_nodes" / "Comfyui" / "ComfyUI-WanVideoWrapper"))

# Need to load dependencies first
from apps.wanvideo_module import server_stub
sys.modules['server'] = server_stub

from nodes_sampler import WanVideoSampler

# Get the process method signature
process_method = WanVideoSampler.process
sig = inspect.signature(process_method)

print("\n" + "="*60)
print("WanVideoSampler.process() Parameters")
print("="*60)

for param_name, param in sig.parameters.items():
    if param_name == 'self':
        continue
    
    default = param.default
    if default == inspect.Parameter.empty:
        default_str = "REQUIRED"
    else:
        default_str = repr(default)
    
    print(f"  {param_name}: {default_str}")

print("\n" + "="*60)
print("Checking for experimental parameters...")
print("="*60)

experimental_params = [
    "use_tf32",
    "use_cublas_gemm", 
    "force_contiguous_tensors",
    "fuse_qkv_projections"
]

for param in experimental_params:
    if param in sig.parameters:
        print(f"  ✓ {param}: Direct parameter")
    else:
        print(f"  ✗ {param}: NOT a direct parameter (should use experimental_args)")

print("\n" + "="*60)
print("Checking experimental_args parameter...")
print("="*60)

if 'experimental_args' in sig.parameters:
    print("  ✓ experimental_args: Available")
    default = sig.parameters['experimental_args'].default
    print(f"    Default: {default}")
else:
    print("  ✗ experimental_args: NOT available")
