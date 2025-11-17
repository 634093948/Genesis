#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify model files integrity
验证模型文件完整性

Author: eddy
Date: 2025-11-16
"""

import os
import sys
import io
from pathlib import Path

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add compat to path
project_root = Path(__file__).parent.parent
compat_dir = project_root / "compat"
sys.path.insert(0, str(compat_dir))

print("=" * 70)
print("模型文件完整性验证")
print("=" * 70)
print()

try:
    import folder_paths
    print("✓ folder_paths 导入成功")
    print()
except Exception as e:
    print(f"✗ folder_paths 导入失败: {e}")
    sys.exit(1)

def check_file(file_path: str, file_type: str) -> dict:
    """Check a single file"""
    result = {
        'path': file_path,
        'exists': False,
        'size': 0,
        'readable': False,
        'valid': False,
        'error': None
    }
    
    try:
        # Check exists
        if not os.path.exists(file_path):
            result['error'] = "文件不存在"
            return result
        result['exists'] = True
        
        # Check size
        size = os.path.getsize(file_path)
        result['size'] = size
        
        if size < 1024:  # Less than 1KB
            result['error'] = f"文件太小 ({size} bytes)，可能损坏"
            return result
        
        # Check readable
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Try to read first 1KB
            result['readable'] = True
        except Exception as e:
            result['error'] = f"无法读取文件: {e}"
            return result
        
        # Try to load with safetensors
        if file_path.endswith('.safetensors') or file_path.endswith('.sft'):
            try:
                import safetensors
                with safetensors.safe_open(file_path, framework="pt") as f:
                    # Just check if we can open it
                    pass
                result['valid'] = True
            except Exception as e:
                result['error'] = f"Safetensors 验证失败: {e}"
                return result
        else:
            # For other formats, just check if readable
            result['valid'] = True
        
    except Exception as e:
        result['error'] = f"检查失败: {e}"
    
    return result

def format_size(size: int) -> str:
    """Format file size"""
    if size < 1024:
        return f"{size} B"
    elif size < 1024**2:
        return f"{size / 1024:.2f} KB"
    elif size < 1024**3:
        return f"{size / (1024**2):.2f} MB"
    else:
        return f"{size / (1024**3):.2f} GB"

# Check models
model_types = ['unet', 'diffusion_models', 'clip', 'text_encoders', 'vae']

print("检查模型文件...")
print()

total_files = 0
valid_files = 0
invalid_files = 0

for model_type in model_types:
    try:
        files = folder_paths.get_filename_list(model_type)
        if not files:
            continue
        
        print(f"📁 {model_type}:")
        print()
        
        for filename in files:
            total_files += 1
            full_path = folder_paths.get_full_path(model_type, filename)
            
            if full_path is None:
                print(f"  ✗ {filename}")
                print(f"     错误: 无法找到完整路径")
                invalid_files += 1
                continue
            
            result = check_file(full_path, model_type)
            
            if result['valid']:
                print(f"  ✓ {filename}")
                print(f"     大小: {format_size(result['size'])}")
                valid_files += 1
            else:
                print(f"  ✗ {filename}")
                print(f"     路径: {full_path}")
                if result['exists']:
                    print(f"     大小: {format_size(result['size'])}")
                print(f"     错误: {result['error']}")
                invalid_files += 1
            print()
        
    except Exception as e:
        print(f"✗ {model_type}: {e}")
        print()

print("=" * 70)
print("验证结果")
print("=" * 70)
print(f"总文件数: {total_files}")
print(f"有效文件: {valid_files}")
print(f"无效文件: {invalid_files}")
print()

if invalid_files > 0:
    print("⚠️ 发现无效文件，请检查并重新下载")
    print()
    print("建议:")
    print("  1. 删除损坏的文件")
    print("  2. 重新下载模型")
    print("  3. 验证下载的文件完整性（MD5/SHA256）")
else:
    print("✅ 所有文件验证通过!")

print("=" * 70)
