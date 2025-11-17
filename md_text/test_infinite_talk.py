#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test Infinite Talk Pipeline with specific parameters"""

import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the pipeline
from apps.wanvideo_module.infinite_talk_pipeline import InfiniteTalkPipeline

def test_infinite_talk():
    """Test Infinite Talk with user-specified parameters"""
    
    print("="*70)
    print("Initializing Infinite Talk Pipeline...")
    print("="*70)
    
    # Initialize pipeline
    pipeline = InfiniteTalkPipeline()
    
    # Test parameters from user's screenshot
    test_params = {
        # Input files
        "image_path": r"D:\rh推广项目\工作流及图形\20250921\1ecf8fdfbb57ef8ebb1cbec5973d5cc732c1a12cda6b2e71ac46a1aa4af40548.jpg",
        "audio_path": r"D:\rh推广项目\工作流及图形\20250921\HI，美女，在找我吗+蜡笔小新-搞笑BGM_爱给网_aigei_com.mp3",
        
        # Model settings (from screenshot)
        "model_name": "video/eddy/infinite_talk/eedy_LWan2_IceCannon2.1_InfiniteTalk.safetensors",
        "vae_name": "wan_2.1_vae.safetensors",
        "t5_name": "video/models_t5_umt5-xxl-enc-fp8_fully_uncensored.safetensors",
        "clip_name": "clip_vision_vit_h.safetensors",
        "wav2vec_name": "TencentGameMate/chinese-wav2vec2-base",
        
        # Generation parameters (from screenshot)
        "width": 768,
        "height": 768,
        "video_length": 81,  # frames
        "fps": 8,
        "steps": 6,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "dpm++_sde",
        
        # Prompt
        "prompt": "worst quality, low quality, blurry, distorted",
        "negative_prompt": "worst quality, low quality, blurry, distorted",
    }
    
    print("\n" + "="*70)
    print("Test Parameters:")
    print("="*70)
    for key, value in test_params.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("Starting Generation...")
    print("="*70)
    
    try:
        # Run generation
        result = pipeline.generate(
            image_path=test_params["image_path"],
            audio_path=test_params["audio_path"],
            prompt=test_params["prompt"],
            negative_prompt=test_params["negative_prompt"],
            width=test_params["width"],
            height=test_params["height"],
            video_length=test_params["video_length"],
            fps=test_params["fps"],
            steps=test_params["steps"],
            cfg=test_params["cfg"],
            seed=-1,  # Random seed
        )
        
        print("\n" + "="*70)
        print("✓ Generation Successful!")
        print("="*70)
        print(f"Output: {result}")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("✗ Generation Failed!")
        print("="*70)
        print(f"Error: {e}")
        
        import traceback
        traceback.print_exc()
        
        return False

if __name__ == "__main__":
    success = test_infinite_talk()
    sys.exit(0 if success else 1)
