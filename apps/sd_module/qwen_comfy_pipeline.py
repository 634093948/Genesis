#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen Image ComfyUI Pipeline
使用 ComfyUI 兼容的 Qwen Image 节点

Based on: custom_nodes/Comfyui/ComfyUI-QwenImageWrapper
Workflow: qwen3 edy.json

Author: eddy
Date: 2025-11-16
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
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
sys.path.insert(0, str(project_root / "compat"))

# Import folder_paths
try:
    from core import folder_paths
except ImportError:
    import importlib.util
    folder_paths_file = project_root / "compat" / "folder_paths.py"
    spec = importlib.util.spec_from_file_location("folder_paths", folder_paths_file)
    folder_paths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(folder_paths)

# Try to import Qwen nodes
QWEN_AVAILABLE = False
try:
    # Add Qwen node path
    qwen_path = project_root / "custom_nodes" / "Comfyui" / "ComfyUI-QwenImageWrapper"
    if qwen_path.exists() and str(qwen_path) not in sys.path:
        sys.path.insert(0, str(qwen_path))
    
    # Import Qwen node
    from standalone_official_nodes import EddyQwenImageBlockSwap
    
    QWEN_AVAILABLE = True
    logger.info("✓ Qwen Image nodes available")
except ImportError as e:
    logger.warning(f"⚠ Qwen Image nodes not available: {e}")
    QWEN_AVAILABLE = False


class QwenComfyPipeline:
    """
    Qwen Image pipeline using ComfyUI format
    Compatible with ComfyUI workflow
    """
    
    def __init__(self):
        """Initialize pipeline"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Qwen node instance
        self.qwen_node = None
        
        # Current loaded models
        self.current_unet = None
        self.current_clip = None
        self.current_vae = None
        
        logger.info(f"Qwen ComfyUI Pipeline initialized on {self.device}")
        
        if not QWEN_AVAILABLE:
            logger.error("Qwen Image nodes not available - pipeline will not work!")
        else:
            self.qwen_node = EddyQwenImageBlockSwap()
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        unet_name: str = "qwen_image_fp8_e4m3fn.safetensors",
        clip_name: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        vae_name: str = "qwen_image_vae.safetensors",
        width: int = 1328,
        height: int = 1328,
        steps: int = 8,
        cfg: float = 2.5,
        sampler_name: str = "sa_solver",
        scheduler: str = "beta",
        seed: int = -1,
        quantization_dtype: str = "fp16_fast",
        # LoRA settings
        lora_1_name: str = "none",
        lora_1_strength: float = 1.0,
        lora_2_name: str = "none",
        lora_2_strength: float = 0.0,
        lora_3_name: str = "none",
        lora_3_strength: float = 0.0,
        lora_4_name: str = "none",
        lora_4_strength: float = 0.0,
        # BlockSwap settings
        use_blockswap: bool = True,
        blockswap_blocks: int = 20,
        blockswap_model_size: str = "auto",
        blockswap_use_recommended: bool = True,
        # Optimization settings
        enable_matmul_optimization: bool = True,
        use_torch_compile: bool = False,
        matmul_precision: str = "high",
        use_autocast: bool = False,
        autocast_dtype: str = "bfloat16",
        use_channels_last: bool = False,
        enable_flash_attention: bool = True,
        compile_mode: str = "default",
        enable_kv_cache: bool = True
    ) -> Optional[List[Image.Image]]:
        """
        Generate images from text prompt using Qwen Image
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            unet_name: UNET model name
            clip_name: CLIP model name
            vae_name: VAE model name
            width: Image width
            height: Image height
            steps: Number of steps
            cfg: CFG scale
            sampler_name: Sampler name
            scheduler: Scheduler name
            seed: Random seed
            quantization_dtype: Quantization type
            lora_*: LoRA settings
            use_blockswap: Enable BlockSwap
            blockswap_*: BlockSwap settings
            enable_matmul_optimization: Enable matmul optimization
            use_torch_compile: Enable torch.compile
            matmul_precision: Matmul precision
            use_autocast: Enable autocast
            autocast_dtype: Autocast dtype
            use_channels_last: Use channels last
            enable_flash_attention: Enable flash attention
            compile_mode: Compile mode
            enable_kv_cache: Enable KV cache
            
        Returns:
            List of generated images
        """
        if not QWEN_AVAILABLE:
            logger.error("Qwen Image nodes not available")
            return None
        
        if self.qwen_node is None:
            logger.error("Qwen node not initialized")
            return None
        
        try:
            logger.info("=" * 70)
            logger.info("Qwen Image ComfyUI Generation")
            logger.info("=" * 70)
            logger.info(f"Prompt: {prompt[:100]}...")
            logger.info(f"Size: {width}x{height}")
            logger.info(f"Steps: {steps}")
            logger.info(f"CFG: {cfg}")
            logger.info(f"Sampler: {sampler_name}")
            logger.info(f"Scheduler: {scheduler}")
            
            # Set seed
            if seed == -1:
                seed = torch.randint(0, 2**32, (1,)).item()
            logger.info(f"Seed: {seed}")
            
            # Store current models
            self.current_unet = unet_name
            self.current_clip = clip_name
            self.current_vae = vae_name
            
            # Generate using Qwen node
            logger.info("\nGenerating with Qwen Image node...")
            result = self.qwen_node.generate(
                positive=prompt,
                negative=negative_prompt,
                unet_name=unet_name,
                clip_name=clip_name,
                vae_name=vae_name,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                seed=seed,
                quantization_dtype=quantization_dtype,
                lora_1_name=lora_1_name,
                lora_1_strength=lora_1_strength,
                lora_2_name=lora_2_name,
                lora_2_strength=lora_2_strength,
                lora_3_name=lora_3_name,
                lora_3_strength=lora_3_strength,
                lora_4_name=lora_4_name,
                lora_4_strength=lora_4_strength,
                use_blockswap=use_blockswap,
                blockswap_blocks=blockswap_blocks,
                blockswap_model_size=blockswap_model_size,
                blockswap_use_recommended=blockswap_use_recommended,
                enable_matmul_optimization=enable_matmul_optimization,
                use_torch_compile=use_torch_compile,
                matmul_precision=matmul_precision,
                use_autocast=use_autocast,
                autocast_dtype=autocast_dtype,
                use_channels_last=use_channels_last,
                enable_flash_attention=enable_flash_attention,
                compile_mode=compile_mode,
                enable_kv_cache=enable_kv_cache
            )
            
            # Convert result to PIL images
            images = []
            if result and len(result) > 0:
                images_tensor = result[0]
                
                # Convert from tensor to PIL
                for img_tensor in images_tensor:
                    # img_tensor shape: [H, W, C]
                    # Detach from computation graph before converting to numpy
                    img_np = img_tensor.detach().cpu().numpy()
                    # Scale to 0-255
                    img_np = (img_np * 255).astype(np.uint8)
                    # Convert to PIL
                    img_pil = Image.fromarray(img_np)
                    images.append(img_pil)
            
            # Save images to output directory
            from pathlib import Path
            from datetime import datetime
            output_dir = Path(__file__).parent.parent.parent / "output"
            output_dir.mkdir(exist_ok=True)
            
            saved_paths = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for i, img in enumerate(images):
                filename = f"qwen_{timestamp}_{i:04d}.png"
                output_path = output_dir / filename
                img.save(output_path)
                saved_paths.append(str(output_path))
                logger.info(f"  Saved: {output_path}")
            
            logger.info("=" * 70)
            logger.info(f"✅ Generated {len(images)} image(s)")
            logger.info(f"📁 Saved to: {output_dir}")
            logger.info("=" * 70)
            
            return images
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def unload(self):
        """Unload models to free memory"""
        if self.qwen_node is not None:
            if hasattr(self.qwen_node, 'processed_model') and self.qwen_node.processed_model is not None:
                del self.qwen_node.processed_model
                self.qwen_node.processed_model = None
            
            if hasattr(self.qwen_node, 'processed_clip') and self.qwen_node.processed_clip is not None:
                del self.qwen_node.processed_clip
                self.qwen_node.processed_clip = None
            
            if hasattr(self.qwen_node, 'processed_vae') and self.qwen_node.processed_vae is not None:
                del self.qwen_node.processed_vae
                self.qwen_node.processed_vae = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Models unloaded")


def get_available_models() -> Dict[str, List[str]]:
    """
    Get available Qwen models from project folders
    
    Returns:
        Dictionary of model lists
    """
    models = {
        'unet': [],
        'diffusion_models': [],
        'clip': [],
        'text_encoders': [],
        'vae': [],
        'loras': []
    }
    
    # Get local models from project models folder
    for key in ['unet', 'diffusion_models', 'clip', 'text_encoders', 'vae', 'loras']:
        try:
            files = folder_paths.get_filename_list(key)
            models[key] = files
        except:
            pass
    
    return models


if __name__ == "__main__":
    # Test
    logger.info("Testing Qwen ComfyUI Pipeline")
    logger.info("=" * 70)
    
    if not QWEN_AVAILABLE:
        logger.error("Qwen Image nodes not available")
        sys.exit(1)
    
    # Create pipeline
    pipeline = QwenComfyPipeline()
    
    # Get available models
    models = get_available_models()
    logger.info("\nAvailable models:")
    for key, value in models.items():
        logger.info(f"  {key}: {len(value)} models")
        if value:
            logger.info(f"    Examples: {value[:3]}")
    
    logger.info("\n✓ Pipeline ready")
    logger.info("Generate with: pipeline.generate(prompt, ...)")
