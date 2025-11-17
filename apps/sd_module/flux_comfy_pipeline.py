#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flux ComfyUI Pipeline
使用 ComfyUI 兼容的 UNET 格式，不是 diffusers 格式

基于工作流: flux文生图.json
节点:
- UNETLoader: 加载 Flux UNET 模型
- DualCLIPLoader: 加载 T5XXL + CLIP-L
- VAELoader: 加载 VAE
- CLIPTextEncode: 编码提示词
- FluxGuidance: 应用引导
- KSamplerAdvanced: 高级采样
- VAEDecode: 解码图像

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

# Import folder_paths
try:
    from core import folder_paths
except ImportError:
    import importlib.util
    folder_paths_file = project_root / "core" / "folder_paths.py"
    spec = importlib.util.spec_from_file_location("folder_paths", folder_paths_file)
    folder_paths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(folder_paths)

# Try to import ComfyUI modules
COMFY_AVAILABLE = False
try:
    # Add ComfyUI paths
    comfy_paths = [
        project_root / "custom_nodes" / "Comfyui",
        project_root / "compat",
    ]
    for p in comfy_paths:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    
    # Import ComfyUI modules
    import comfy.sd
    import comfy.model_management
    import comfy.utils
    from nodes import UNETLoader, DualCLIPLoader, VAELoader, CLIPTextEncode, EmptyLatentImage, VAEDecode, KSamplerAdvanced
    from comfy_extras.nodes_flux import FluxGuidance
    
    COMFY_AVAILABLE = True
    logger.info("✓ ComfyUI modules available")
except ImportError as e:
    logger.warning(f"⚠ ComfyUI modules not available: {e}")
    COMFY_AVAILABLE = False


class FluxComfyPipeline:
    """
    Flux pipeline using ComfyUI UNET format
    Compatible with ComfyUI workflow
    """
    
    def __init__(self):
        """Initialize pipeline"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Pipeline components
        self.model = None
        self.clip = None
        self.vae = None
        
        # Current loaded models
        self.current_unet = None
        self.current_clip1 = None
        self.current_clip2 = None
        self.current_vae = None
        
        logger.info(f"Flux ComfyUI Pipeline initialized on {self.device}")
        
        if not COMFY_AVAILABLE:
            logger.error("ComfyUI modules not available - pipeline will not work!")
    
    def load_models(
        self,
        unet_name: str,
        clip_name1: str,
        clip_name2: str,
        vae_name: Optional[str] = None,
        weight_dtype: str = "default"
    ) -> bool:
        """
        Load Flux models (ComfyUI format)
        
        Args:
            unet_name: UNET model name (from models/unet or models/diffusion_models)
            clip_name1: CLIP model 1 name (T5XXL, from models/clip or models/text_encoders)
            clip_name2: CLIP model 2 name (CLIP-L, from models/clip or models/text_encoders)
            vae_name: VAE model name (from models/vae)
            weight_dtype: Weight data type
            
        Returns:
            Success status
        """
        if not COMFY_AVAILABLE:
            logger.error("ComfyUI modules not available")
            return False
        
        try:
            logger.info("=" * 70)
            logger.info("Loading Flux Models (ComfyUI Format)")
            logger.info("=" * 70)
            
            # Load UNET
            logger.info(f"Loading UNET: {unet_name}")
            try:
                # Verify file exists and is readable
                unet_path = folder_paths.get_full_path('unet', unet_name)
                if unet_path is None:
                    unet_path = folder_paths.get_full_path('diffusion_models', unet_name)
                
                if unet_path is None:
                    raise FileNotFoundError(f"UNET model not found: {unet_name}")
                
                # Check file size
                import os
                file_size = os.path.getsize(unet_path)
                logger.info(f"  File path: {unet_path}")
                logger.info(f"  File size: {file_size / (1024**3):.2f} GB")
                
                if file_size < 1024:  # Less than 1KB
                    raise ValueError(f"UNET file is too small ({file_size} bytes), possibly corrupted")
                
                unet_loader = UNETLoader()
                result = unet_loader.load_unet(unet_name, weight_dtype)
                self.model = result[0]
                self.current_unet = unet_name
                logger.info("✓ UNET loaded")
            except Exception as e:
                logger.error(f"✗ UNET loading failed: {e}")
                raise
            
            # Load Dual CLIP
            logger.info(f"Loading CLIP1 (T5XXL): {clip_name1}")
            logger.info(f"Loading CLIP2 (CLIP-L): {clip_name2}")
            try:
                # Verify CLIP files
                clip1_path = folder_paths.get_full_path('clip', clip_name1)
                if clip1_path is None:
                    clip1_path = folder_paths.get_full_path('text_encoders', clip_name1)
                
                clip2_path = folder_paths.get_full_path('clip', clip_name2)
                if clip2_path is None:
                    clip2_path = folder_paths.get_full_path('text_encoders', clip_name2)
                
                if clip1_path is None:
                    raise FileNotFoundError(f"CLIP1 model not found: {clip_name1}")
                if clip2_path is None:
                    raise FileNotFoundError(f"CLIP2 model not found: {clip_name2}")
                
                logger.info(f"  CLIP1 path: {clip1_path}")
                logger.info(f"  CLIP2 path: {clip2_path}")
                
                # Temporarily suppress text_projection.weight warning (it's optional for CLIP)
                import logging as temp_logging
                root_logger = temp_logging.getLogger()
                old_level = root_logger.level
                root_logger.setLevel(temp_logging.ERROR)
                
                try:
                    clip_loader = DualCLIPLoader()
                    result = clip_loader.load_clip(clip_name1, clip_name2, "flux")
                    self.clip = result[0]
                    self.current_clip1 = clip_name1
                    self.current_clip2 = clip_name2
                finally:
                    root_logger.setLevel(old_level)
                
                logger.info("✓ CLIP models loaded")
            except Exception as e:
                logger.error(f"✗ CLIP loading failed: {e}")
                raise
            
            # Load VAE
            if vae_name:
                logger.info(f"Loading VAE: {vae_name}")
                try:
                    vae_path = folder_paths.get_full_path('vae', vae_name)
                    if vae_path is None:
                        raise FileNotFoundError(f"VAE model not found: {vae_name}")
                    
                    logger.info(f"  VAE path: {vae_path}")
                    
                    vae_loader = VAELoader()
                    result = vae_loader.load_vae(vae_name)
                    self.vae = result[0]
                    self.current_vae = vae_name
                    logger.info("✓ VAE loaded")
                except Exception as e:
                    logger.error(f"✗ VAE loading failed: {e}")
                    raise
            
            logger.info("=" * 70)
            logger.info("✓ All models loaded successfully")
            logger.info("=" * 70)
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to load models: {error_msg}")
            
            # Provide helpful error messages
            if "incomplete metadata" in error_msg or "not fully covered" in error_msg:
                logger.error("")
                logger.error("=" * 70)
                logger.error("模型文件损坏或不完整")
                logger.error("=" * 70)
                logger.error("可能的原因:")
                logger.error("  1. 模型下载未完成")
                logger.error("  2. 文件传输中断")
                logger.error("  3. 磁盘空间不足")
                logger.error("  4. 文件系统错误")
                logger.error("")
                logger.error("解决方法:")
                logger.error("  1. 重新下载模型文件")
                logger.error("  2. 检查文件完整性（MD5/SHA256）")
                logger.error("  3. 确保有足够的磁盘空间")
                logger.error("  4. 尝试从不同的源下载")
                logger.error("=" * 70)
            
            import traceback
            traceback.print_exc()
            return False
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg: float = 1.0,
        guidance: float = 3.5,
        sampler_name: str = "dpmpp_2m",
        scheduler: str = "sgm_uniform",
        seed: int = -1,
        denoise: float = 1.0
    ) -> Optional[List[Image.Image]]:
        """
        Generate images from text prompt
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            steps: Number of steps
            cfg: CFG scale
            guidance: Flux guidance value
            sampler_name: Sampler name
            scheduler: Scheduler name
            seed: Random seed
            denoise: Denoise strength (0.0-1.0)
            
        Returns:
            List of generated images
        """
        if not COMFY_AVAILABLE:
            logger.error("ComfyUI modules not available")
            return None
        
        if self.model is None or self.clip is None or self.vae is None:
            logger.error("Models not loaded. Call load_models() first")
            return None
        
        try:
            logger.info("=" * 70)
            logger.info("Flux ComfyUI Generation")
            logger.info("=" * 70)
            logger.info(f"Prompt: {prompt[:100]}...")
            logger.info(f"Size: {width}x{height}")
            logger.info(f"Steps: {steps}")
            logger.info(f"Guidance: {guidance}")
            logger.info(f"Sampler: {sampler_name}")
            logger.info(f"Scheduler: {scheduler}")
            
            # Set seed
            if seed == -1:
                seed = torch.randint(0, 2**32, (1,)).item()
            logger.info(f"Seed: {seed}")
            
            # 1. Encode prompts
            logger.info("\n[1/5] Encoding prompts...")
            text_encoder = CLIPTextEncode()
            
            try:
                positive_result = text_encoder.encode(self.clip, prompt)
                # Handle both tuple and NodeOutput types
                if hasattr(positive_result, '__getitem__'):
                    positive_cond = positive_result[0]
                else:
                    positive_cond = positive_result
                
                negative_result = text_encoder.encode(self.clip, negative_prompt)
                if hasattr(negative_result, '__getitem__'):
                    negative_cond = negative_result[0]
                else:
                    negative_cond = negative_result
            except (IndexError, TypeError) as e:
                logger.error(f"Failed to encode prompts: {e}")
                logger.error(f"Positive result type: {type(positive_result) if 'positive_result' in locals() else 'not created'}")
                logger.error(f"Negative result type: {type(negative_result) if 'negative_result' in locals() else 'not created'}")
                raise
            
            # 2. Apply Flux Guidance
            logger.info("[2/5] Applying Flux guidance...")
            flux_guidance = FluxGuidance()
            try:
                guidance_result = flux_guidance.append(positive_cond, guidance)
                # Handle both tuple and NodeOutput types
                if hasattr(guidance_result, '__getitem__'):
                    positive_cond = guidance_result[0]
                else:
                    positive_cond = guidance_result
            except (IndexError, TypeError) as e:
                logger.error(f"Failed to apply Flux guidance: {e}")
                logger.error(f"Guidance result type: {type(guidance_result)}")
                logger.error(f"Guidance result: {guidance_result}")
                raise
            
            # 3. Create empty latent
            logger.info("[3/5] Creating latent...")
            latent_creator = EmptyLatentImage()
            try:
                latent_result = latent_creator.generate(width, height, 1)
                # Handle both tuple and NodeOutput types
                if hasattr(latent_result, '__getitem__'):
                    latent = latent_result[0]
                else:
                    latent = latent_result
            except (IndexError, TypeError) as e:
                logger.error(f"Failed to create latent: {e}")
                logger.error(f"Latent result type: {type(latent_result) if 'latent_result' in locals() else 'not created'}")
                raise
            
            # 4. Sample
            logger.info(f"[4/5] Sampling with {sampler_name}...")
            sampler = KSamplerAdvanced()
            
            # Calculate start and end steps
            start_step = 0
            end_step = steps
            
            try:
                sample_result = sampler.sample(
                    self.model,
                    "enable",  # add_noise
                    seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive_cond,
                    negative_cond,
                    latent,
                    start_step,
                    end_step,
                    "disable"  # return_with_leftover_noise
                )
                # Handle both tuple and NodeOutput types
                if hasattr(sample_result, '__getitem__'):
                    samples = sample_result[0]
                else:
                    samples = sample_result
            except (IndexError, TypeError) as e:
                logger.error(f"Failed to sample: {e}")
                logger.error(f"Sample result type: {type(sample_result) if 'sample_result' in locals() else 'not created'}")
                raise
            
            # 5. Decode
            logger.info("[5/5] Decoding to image...")
            decoder = VAEDecode()
            
            # samples is a dict with 'samples' key containing the latent tensor
            if isinstance(samples, dict):
                latent_samples = samples
            else:
                # If it's already a tensor, wrap it
                latent_samples = {"samples": samples}
            
            # VAEDecode.decode(vae, samples) - note the parameter order!
            try:
                decode_result = decoder.decode(self.vae, latent_samples)
                # Handle both tuple and NodeOutput types
                if hasattr(decode_result, '__getitem__'):
                    images_tensor = decode_result[0]
                else:
                    images_tensor = decode_result
            except (IndexError, TypeError) as e:
                logger.error(f"Failed to decode: {e}")
                logger.error(f"Decode result type: {type(decode_result) if 'decode_result' in locals() else 'not created'}")
                raise
            
            # Convert to PIL images
            images = []
            for img_tensor in images_tensor:
                # Detach from computation graph and convert to numpy
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
                filename = f"flux_{timestamp}_{i:04d}.png"
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
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.clip is not None:
            del self.clip
            self.clip = None
            
        if self.vae is not None:
            del self.vae
            self.vae = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Models unloaded")


def get_available_models() -> Dict[str, List[str]]:
    """
    Get available Flux models from project folders
    
    Returns:
        Dictionary of model lists
    """
    models = {
        'unet': [],
        'diffusion_models': [],
        'clip': [],
        'text_encoders': [],
        'vae': []
    }
    
    # Get local models from project models folder
    for key in ['unet', 'diffusion_models', 'clip', 'text_encoders', 'vae']:
        try:
            files = folder_paths.get_filename_list(key)
            models[key] = files
        except:
            pass
    
    return models


if __name__ == "__main__":
    # Test
    logger.info("Testing Flux ComfyUI Pipeline")
    logger.info("=" * 70)
    
    if not COMFY_AVAILABLE:
        logger.error("ComfyUI modules not available")
        sys.exit(1)
    
    # Create pipeline
    pipeline = FluxComfyPipeline()
    
    # Get available models
    models = get_available_models()
    logger.info("\nAvailable models:")
    for key, value in models.items():
        logger.info(f"  {key}: {len(value)} models")
        if value:
            logger.info(f"    Examples: {value[:3]}")
    
    logger.info("\n✓ Pipeline ready")
    logger.info("Load models with: pipeline.load_models(unet_name, clip_name1, clip_name2, vae_name)")
