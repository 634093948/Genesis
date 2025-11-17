#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run Infinite Talk generation once with specified parameters.
Usage: python scripts/run_infinite_talk_once.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from apps.wanvideo_module.infinite_talk_pipeline import InfiniteTalkPipeline, WANVIDEO_AVAILABLE


def main():
    if not WANVIDEO_AVAILABLE:
        raise SystemExit("WanVideo nodes unavailable")

    image_path = r"D:\rh推广项目\工作流及图形\1ecf8fdfbb57ef8ebb1cbec5973d5cc732c1a12cda6b2e71ac46a1aa4af40548.jpg"
    audio_path = r"D:\rh推广项目\工作流及图形\20250921\贾维斯-成功着陆欢迎回家_爱给网_aigei_com.mp3"
    model_name = r"video/eddy/infinite_talk/eedy_Wan2_IceCannon2.1_InfiniteTalk.safetensors"
    vae_name = "Wan2_1_VAE_bf16.safetensors"
    t5_model = "video/models_t5_umt5-xxl-enc-fp8_fully_uncensored.safetensors"  # 使用本地 T5 模型
    clip_vision = "clip_vision_g.safetensors"
    wav2vec_model = "TencentGameMate/chinese-wav2vec2-base"

    pipeline = InfiniteTalkPipeline()
    print("[Run] Loading models...")
    if not pipeline.load_models(model_name, vae_name, t5_model, clip_vision, wav2vec_model):
        raise SystemExit("Model loading failed")

    print("[Run] Generating video...")
    output = pipeline.generate(
        image_path=image_path,
        audio_path=audio_path,
        prompt="",
        negative_prompt="",
        width=768,
        height=768,
        video_length=49,
        steps=6,
        cfg=1.0,
        sampler_name="euler",
        scheduler="multitalk",
        shift=7.0,
        seed=-1,
        fps=8,
        optimization_args={
            'blocks_to_swap': 40,
            'enable_cuda_optimization': True,
            'enable_dram_optimization': True,
            'auto_hardware_tuning': True,
        }
    )

    print("[Run] Output:", output)


if __name__ == "__main__":
    main()
