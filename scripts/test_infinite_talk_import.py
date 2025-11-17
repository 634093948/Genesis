#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Infinite Talk import after server stub fix
测试 Infinite Talk 导入修复

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
print("测试 Infinite Talk 导入")
print("=" * 70)
print()

# Setup triton stub
try:
    from utils import triton_ops_stub
    print("✓ Triton stub loaded")
except:
    print("⚠ Triton stub not loaded")

print()

# Test import
print("1. 测试导入 infinite_talk_pipeline...")
try:
    from apps.wanvideo_module.infinite_talk_pipeline import (
        WANVIDEO_AVAILABLE,
        InfiniteTalkPipeline
    )
    
    print(f"   ✓ 导入成功")
    print(f"   WANVIDEO_AVAILABLE: {WANVIDEO_AVAILABLE}")
    
    if WANVIDEO_AVAILABLE:
        print("   ✓ WanVideo 节点可用")
        
        # Test pipeline creation
        print()
        print("2. 测试创建 pipeline...")
        try:
            pipeline = InfiniteTalkPipeline()
            print("   ✓ Pipeline 创建成功")
            print(f"   设备: {pipeline.device}")
        except Exception as e:
            print(f"   ✗ Pipeline 创建失败: {e}")
    else:
        print("   ⚠ WanVideo 节点不可用")
        print("   这是正常的，因为依赖的模型和环境可能不完整")
        
except Exception as e:
    print(f"   ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("测试完成")
print("=" * 70)
print()

if WANVIDEO_AVAILABLE:
    print("✅ Infinite Talk 可以在 WebUI 中使用!")
else:
    print("⚠️ Infinite Talk 导入成功但节点不可用")
    print("   可能原因:")
    print("   - WanVideoWrapper 节点加载失败")
    print("   - 缺少必要的依赖")
    print("   但至少不会报 'No module named genesis' 错误了!")
