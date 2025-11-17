#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Complete Infinite Talk test with model loading"""

import sys
import os
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the pipeline
from apps.wanvideo_module.infinite_talk_pipeline import InfiniteTalkPipeline

def test_infinite_talk_complete():
    """Complete test with model loading"""
    
    print("="*70)
    print("Initializing Infinite Talk Pipeline...")
    print("="*70)
    
    # Initialize pipeline
    pipeline = InfiniteTalkPipeline()
    
    # Model names (relative to models folder)
    model_config = {
        "model_name": "video/eddy/infinite_talk/eedy_Wan2_IceCannon2.1_InfiniteTalk.safetensors",
        "vae_name": "wan_2.1_vae.safetensors",
        "t5_model_name": "video/models_t5_umt5-xxl-enc-fp8_fully_uncensored.safetensors",
        "clip_vision_name": "clip_vision_vit_h.safetensors",
        "wav2vec_model_name": "TencentGameMate/chinese-wav2vec2-base",
    }
    
    # Load models
    print("\n" + "="*70)
    print("Loading Models...")
    print("="*70)
    print("Model configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    try:
        pipeline.load_models(
            model_name=model_config["model_name"],
            vae_name=model_config["vae_name"],
            t5_model_name=model_config["t5_model_name"],
            clip_vision_name=model_config["clip_vision_name"],
            wav2vec_model_name=model_config["wav2vec_model_name"],
            model_precision="bf16",
            model_quantization="fp4_scaled",
            model_attention="sageattn_3_fp4",
        )
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test parameters
    test_params = {
        "image_path": r"D:\rh推广项目\工作流及图形\20250921\1ecf8fdfbb57ef8ebb1cbec5973d5cc732c1a12cda6b2e71ac46a1aa4af40548.jpg",
        "audio_path": r"D:\rh推广项目\工作流及图形\20250921\HI，美女，在找我吗+蜡笔小新-搞笑BGM_爱给网_aigei_com.mp3",
        "prompt": "worst quality, low quality, blurry, distorted",
        "negative_prompt": "worst quality, low quality, blurry, distorted",
        "width": 768,
        "height": 768,
        "video_length": 81,
        "fps": 8,
        "steps": 6,
        "cfg": 1.0,
        "seed": -1,
    }
    
    print("\n" + "="*70)
    print("Test Parameters:")
    print("="*70)
    for key, value in test_params.items():
        print(f"  {key}: {value}")
    
    # Check input files
    print("\n" + "="*70)
    print("Checking Input Files...")
    print("="*70)
    image_exists = Path(test_params["image_path"]).exists()
    audio_exists = Path(test_params["audio_path"]).exists()
    print(f"  Image: {'✓' if image_exists else '✗'} {test_params['image_path']}")
    print(f"  Audio: {'✓' if audio_exists else '✗'} {test_params['audio_path']}")
    
    if not image_exists or not audio_exists:
        print("\n✗ Input files not found!")
        return False
    
    print("\n" + "="*70)
    print("Starting Generation...")
    print("="*70)
    print("This may take several minutes...")
    print("Watch for 'Sampling audio indices' progress...")
    print("="*70)
    
    try:
        result = pipeline.generate(**test_params)
        
        print("\n" + "="*70)
        print("✓ Generation Successful!")
        print("="*70)
        if result:
            print(f"Output video: {result}")
        
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
    success = test_infinite_talk_complete()
    sys.exit(0 if success else 1)
