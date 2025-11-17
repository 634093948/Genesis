"""
测试 Sage3 FP4 功能
验证 sage3 包是否正确安装并可以调用
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*60)
print("Sage3 FP4 测试")
print("="*60)

# Test 1: Import sage3
print("\n[测试 1] 导入 sage3 包...")
try:
    import sage3
    print(f"✓ sage3 版本: {sage3.__version__}")
    print(f"✓ SAGEATTENTION_AVAILABLE: {sage3.SAGEATTENTION_AVAILABLE}")
    print(f"✓ SAGEATTN3_AVAILABLE: {sage3.SAGEATTN3_AVAILABLE}")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# Test 2: Import sageattn3_blackwell
print("\n[测试 2] 导入 sageattn3_blackwell 函数...")
try:
    from sage3 import sageattn3_blackwell
    print(f"✓ sageattn3_blackwell: {sageattn3_blackwell}")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# Test 3: Test with dummy tensors
print("\n[测试 3] 使用虚拟张量测试...")
try:
    import torch
    
    # Create dummy tensors (small size for testing)
    batch_size = 1
    seq_len = 16
    num_heads = 8
    head_dim = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  使用设备: {device}")
    
    if device.type == "cuda":
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        
        print(f"  输入形状: q={q.shape}, k={k.shape}, v={v.shape}")
        
        # Test sageattn3_blackwell
        output = sageattn3_blackwell(q, k, v, per_block_mean=True)
        print(f"  输出形状: {output.shape}")
        print(f"✓ sageattn3_blackwell 测试成功!")
    else:
        print("  ⚠ 需要 CUDA 设备才能测试 FP4 功能")
        
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check WanVideoWrapper attention module
print("\n[测试 4] 检查 WanVideoWrapper attention 模块...")
try:
    wrapper_path = project_root / "custom_nodes" / "Comfyui" / "ComfyUI-WanVideoWrapper"
    sys.path.insert(0, str(wrapper_path))
    
    from wanvideo.modules.attention import SAGE3_AVAILABLE, sageattn_blackwell
    print(f"✓ WanVideoWrapper SAGE3_AVAILABLE: {SAGE3_AVAILABLE}")
    print(f"✓ WanVideoWrapper sageattn_blackwell: {sageattn_blackwell}")
except Exception as e:
    print(f"✗ 检查失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("测试完成!")
print("="*60)
print("\n如果所有测试都通过,说明 sage3 FP4 已正确配置。")
print("现在可以在 UI 中选择 'sageattn_3_fp4' attention 模式。")
