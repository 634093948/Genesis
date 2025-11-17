#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copy necessary ComfyUI modules to compat folder
复制必要的 ComfyUI 模块到 compat 文件夹

Author: eddy
Date: 2025-11-16
"""

import os
import sys
import shutil
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 源路径和目标路径
COMFY_SOURCE = Path(r"E:\liliyuanshangmie\comfyUI origin\ComfyUI_0811\ComfyUI")
PROJECT_ROOT = Path(__file__).parent.parent
COMPAT_DIR = PROJECT_ROOT / "compat"

# 需要复制的 comfy 模块文件
COMFY_FILES = [
    "sd.py",
    "model_management.py",
    "model_patcher.py",
    "model_base.py",
    "model_detection.py",
    "model_sampling.py",
    "utils.py",
    "samplers.py",
    "sampler_helpers.py",
    "sample.py",
    "latent_formats.py",
    "clip_model.py",
    "sd1_clip.py",
    "sdxl_clip.py",
    "supported_models.py",
    "supported_models_base.py",
    "lora.py",
    "controlnet.py",
    "conds.py",
    "ops.py",
    "float.py",
    "options.py",
    "cli_args.py",
    "checkpoint_pickle.py",  # 新增
    "hooks.py",              # 新增
    "diffusers_load.py",     # 新增
    "diffusers_convert.py",  # 新增
    "patcher_extension.py",  # 新增
]

# 需要复制的 comfy 子文件夹
COMFY_DIRS = [
    "ldm",
    "k_diffusion",
    "text_encoders",
    "comfy_types",
]

# 需要复制的顶层文件
TOP_LEVEL_FILES = [
    "nodes.py",
    "node_helpers.py",
    "folder_paths.py",
]

def copy_file(src, dst):
    """复制文件"""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"✓ 复制: {src.name}")
        return True
    except Exception as e:
        print(f"✗ 失败: {src.name} - {e}")
        return False

def copy_directory(src, dst):
    """复制目录"""
    try:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"✓ 复制目录: {src.name}")
        return True
    except Exception as e:
        print(f"✗ 失败: {src.name} - {e}")
        return False

def main():
    print("=" * 70)
    print("复制 ComfyUI 模块")
    print("=" * 70)
    print()
    
    if not COMFY_SOURCE.exists():
        print(f"✗ 错误: ComfyUI 源路径不存在: {COMFY_SOURCE}")
        return
    
    print(f"源路径: {COMFY_SOURCE}")
    print(f"目标路径: {COMPAT_DIR}")
    print()
    
    # 创建目标目录
    comfy_dst = COMPAT_DIR / "comfy"
    comfy_dst.mkdir(parents=True, exist_ok=True)
    
    # 复制 comfy 模块文件
    print("[1/3] 复制 comfy 模块文件...")
    success_count = 0
    for filename in COMFY_FILES:
        src = COMFY_SOURCE / "comfy" / filename
        dst = comfy_dst / filename
        if src.exists():
            if copy_file(src, dst):
                success_count += 1
        else:
            print(f"⚠ 跳过: {filename} (不存在)")
    print(f"完成: {success_count}/{len(COMFY_FILES)} 文件")
    print()
    
    # 复制 comfy 子文件夹
    print("[2/3] 复制 comfy 子文件夹...")
    success_count = 0
    for dirname in COMFY_DIRS:
        src = COMFY_SOURCE / "comfy" / dirname
        dst = comfy_dst / dirname
        if src.exists():
            if copy_directory(src, dst):
                success_count += 1
        else:
            print(f"⚠ 跳过: {dirname} (不存在)")
    print(f"完成: {success_count}/{len(COMFY_DIRS)} 文件夹")
    print()
    
    # 复制顶层文件
    print("[3/3] 复制顶层文件...")
    success_count = 0
    for filename in TOP_LEVEL_FILES:
        src = COMFY_SOURCE / filename
        dst = COMPAT_DIR / filename
        if src.exists():
            if copy_file(src, dst):
                success_count += 1
        else:
            print(f"⚠ 跳过: {filename} (不存在)")
    print(f"完成: {success_count}/{len(TOP_LEVEL_FILES)} 文件")
    print()
    
    # 创建 __init__.py
    print("[额外] 创建 __init__.py 文件...")
    init_files = [
        comfy_dst / "__init__.py",
        comfy_dst / "ldm" / "__init__.py",
        comfy_dst / "k_diffusion" / "__init__.py",
        comfy_dst / "text_encoders" / "__init__.py",
        comfy_dst / "comfy_types" / "__init__.py",
    ]
    
    for init_file in init_files:
        if not init_file.exists():
            init_file.parent.mkdir(parents=True, exist_ok=True)
            init_file.write_text("# ComfyUI module\n")
            print(f"✓ 创建: {init_file.relative_to(COMPAT_DIR)}")
    print()
    
    print("=" * 70)
    print("✅ 完成!")
    print("=" * 70)
    print()
    print("现在可以运行 Flux 了:")
    print("  start.bat")

if __name__ == "__main__":
    main()
