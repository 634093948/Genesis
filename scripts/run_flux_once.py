#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run Flux pipeline once with predefined parameters.
使用 Flux 管道生成一张图，方便后台调试。

Usage:
    python313\python.exe scripts\run_flux_once.py
"""

import sys
import io
from pathlib import Path
import json

# Fix stdout encoding on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.sd_module.flux_comfy_pipeline import FluxComfyPipeline, COMFY_AVAILABLE

if not COMFY_AVAILABLE:
    print("❌ ComfyUI 模块不可用，无法运行 Flux")
    sys.exit(1)

pipeline = FluxComfyPipeline()

# Model selections (per user request)
# UNET 模型 (flux1-krea-dev fp8 scaled)
UNET = "flux1-krea-dev_fp8_scaled.safetensors"
# T5 模型
CLIP1 = "t5xxl_fp16.safetensors"
# CLIP-L 模型
CLIP2 = "clip_l.safetensors"
# 默认 VAE
VAE = "ae.safetensors"

print("=" * 70)
print("加载模型...")
print(f"UNET: {UNET}")
print(f"CLIP1: {CLIP1}")
print(f"CLIP2: {CLIP2}")
print(f"VAE: {VAE}")
print("=" * 70)

if not pipeline.load_models(UNET, CLIP1, CLIP2, VAE):
    print("❌ 模型加载失败")
    sys.exit(1)

print("✅ 模型加载成功，开始生成\n")

# Prompts
prompt = "a high quality cinematic portrait, intricate details, ultra realistic"
negative_prompt = "worst quality, low quality, blurry, distorted"

result = pipeline.generate(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1024,
    height=1024,
    steps=20,
    cfg=3.5,
    sampler_name="dpmpp_2m",
    scheduler="sgm_uniform",
    seed=-1
)

if not result:
    print("❌ 生成失败")
    sys.exit(1)

output_dir = project_root / "output"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "flux_test_run.png"
result[0].save(output_path)

print("=" * 70)
print("✅ 生成完成")
print(f"输出: {output_path}")
print("提示词:", prompt)
print("=" * 70)
