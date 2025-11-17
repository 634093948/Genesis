#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flux Standalone Implementation
Independent Flux text-to-image generation using diffusers library
No ComfyUI dependencies required

Author: eddy
Date: 2025-11-16
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import torch
import numpy as np
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import folder_paths
try:
    from core import folder_paths
except ImportError:
    import importlib.util
    folder_paths_file = project_root / "core" / "folder_paths.py"
    spec = importlib.util.spec_from_file_location("folder_paths", folder_paths_file)
    folder_paths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(folder_paths)

# Try to import diffusers
try:
    from diffusers import FluxPipeline, FluxTransformer2DModel
    from diffusers.models import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    DIFFUSERS_AVAILABLE = True
    logger.info("✓ diffusers library available")
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("⚠ diffusers library not available. Install with: pip install diffusers transformers accelerate")


class FluxStandalonePipeline:
    """
    Standalone Flux pipeline using diffusers
    Completely independent from ComfyUI
    """
    
    def __init__(self):
        """Initialize pipeline"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Pipeline components
        self.pipe = None
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.tokenizer = None
        self.tokenizer_2 = None
        
        # Current loaded models
        self.current_model = None
        self.current_vae = None
        
        logger.info(f"Flux Standalone Pipeline initialized on {self.device}")
        
        if not DIFFUSERS_AVAILABLE:
            logger.error("diffusers not available - pipeline will not work!")
    
    def load_from_single_file(
        self,
        model_path: str,
        vae_path: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ) -> bool:
        """
        Load Flux model from single safetensors file
        
        Args:
            model_path: Path to Flux model file
            vae_path: Optional path to VAE file
            dtype: Data type for model
            
        Returns:
            Success status
        """
        if not DIFFUSERS_AVAILABLE:
            logger.error("diffusers not available")
            return False
        
        try:
            logger.info(f"Loading Flux model from: {model_path}")
            
            if dtype is None:
                dtype = self.dtype
            
            # Load pipeline from single file
            self.pipe = FluxPipeline.from_single_file(
                model_path,
                torch_dtype=dtype,
                use_safetensors=True
            )
            
            # Load custom VAE if provided
            if vae_path and os.path.exists(vae_path):
                logger.info(f"Loading custom VAE from: {vae_path}")
                vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)
                self.pipe.vae = vae
                self.current_vae = vae_path
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Enable optimizations
            if self.device == "cuda":
                # Enable memory efficient attention
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info("✓ xformers enabled")
                except:
                    logger.info("xformers not available, using default attention")
                
                # Enable CPU offload for large models
                # self.pipe.enable_model_cpu_offload()
            
            self.current_model = model_path
            logger.info("✓ Flux model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_from_pretrained(
        self,
        model_id: str,
        dtype: Optional[torch.dtype] = None
    ) -> bool:
        """
        Load Flux model from HuggingFace
        
        Args:
            model_id: HuggingFace model ID (e.g., "black-forest-labs/FLUX.1-dev")
            dtype: Data type
            
        Returns:
            Success status
        """
        if not DIFFUSERS_AVAILABLE:
            logger.error("diffusers not available")
            return False
        
        try:
            logger.info(f"Loading Flux model from HuggingFace: {model_id}")
            
            if dtype is None:
                dtype = self.dtype
            
            # Load pipeline
            self.pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Enable optimizations
            if self.device == "cuda":
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info("✓ xformers enabled")
                except:
                    logger.info("xformers not available")
            
            self.current_model = model_id
            logger.info("✓ Flux model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        max_sequence_length: int = 256
    ) -> Optional[List[Image.Image]]:
        """
        Generate images from text prompt
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt (not used in Flux)
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            num_images_per_prompt: Number of images to generate
            seed: Random seed
            max_sequence_length: Max sequence length for text encoder
            
        Returns:
            List of generated images
        """
        if self.pipe is None:
            logger.error("Model not loaded. Call load_from_single_file() or load_from_pretrained() first")
            return None
        
        try:
            logger.info("=" * 70)
            logger.info("Flux Standalone Generation")
            logger.info("=" * 70)
            logger.info(f"Prompt: {prompt[:100]}...")
            logger.info(f"Size: {width}x{height}")
            logger.info(f"Steps: {num_inference_steps}")
            logger.info(f"Guidance: {guidance_scale}")
            
            # Set seed
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
                logger.info(f"Seed: {seed}")
            else:
                generator = None
            
            # Generate
            output = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                max_sequence_length=max_sequence_length
            )
            
            images = output.images
            
            logger.info("=" * 70)
            logger.info(f"✅ Generated {len(images)} image(s)")
            logger.info("=" * 70)
            
            return images
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def unload(self):
        """Unload model to free memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded")


def get_available_models() -> Dict[str, List[str]]:
    """
    Get available Flux models
    
    Returns:
        Dictionary of model lists
    """
    models = {
        'unet': [],
        'diffusion_models': [],
        'vae': [],
        'clip': [],
        'text_encoders': [],
        'pretrained': [
            'black-forest-labs/FLUX.1-dev',
            'black-forest-labs/FLUX.1-schnell'
        ]
    }
    
    # Get local models from project models folder
    for key in ['unet', 'diffusion_models', 'vae', 'clip', 'text_encoders']:
        try:
            files = folder_paths.get_filename_list(key)
            models[key] = files
        except:
            pass
    
    return models


def test_flux_standalone():
    """Test standalone Flux pipeline"""
    logger.info("Testing Flux Standalone Pipeline")
    logger.info("=" * 70)
    
    if not DIFFUSERS_AVAILABLE:
        logger.error("diffusers not available. Install with:")
        logger.error("pip install diffusers transformers accelerate")
        return
    
    # Create pipeline
    pipeline = FluxStandalonePipeline()
    
    # Get available models
    models = get_available_models()
    logger.info("\nAvailable models:")
    for key, value in models.items():
        logger.info(f"  {key}: {len(value)} models")
        if value and key != 'pretrained':
            logger.info(f"    Examples: {value[:3]}")
    
    # Try to load a model
    logger.info("\nAttempting to load model...")
    
    # Option 1: Load from local file
    if models['unet']:
        model_path = folder_paths.get_full_path('unet', models['unet'][0])
        logger.info(f"Trying local model: {model_path}")
        success = pipeline.load_from_single_file(model_path)
    elif models['diffusion_models']:
        model_path = folder_paths.get_full_path('diffusion_models', models['diffusion_models'][0])
        logger.info(f"Trying local model: {model_path}")
        success = pipeline.load_from_single_file(model_path)
    else:
        # Option 2: Load from HuggingFace (requires download)
        logger.info("No local models found. You can:")
        logger.info("1. Place Flux model in models/unet/ or models/diffusion_models/")
        logger.info("2. Or load from HuggingFace (will download):")
        logger.info("   pipeline.load_from_pretrained('black-forest-labs/FLUX.1-schnell')")
        return
    
    if success:
        logger.info("\n✓ Model loaded successfully!")
        logger.info("\nYou can now generate images:")
        logger.info("images = pipeline.generate('a beautiful landscape')")
    else:
        logger.error("\n❌ Failed to load model")


if __name__ == "__main__":
    test_flux_standalone()
