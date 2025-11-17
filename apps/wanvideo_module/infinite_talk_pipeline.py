#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Infinite Talk Pipeline
基于 ComfyUI Infinite Talk 工作流的管道

Workflow: Infinite Talk test(1).json
核心功能: 图像 + 音频 -> 说话视频（MultiTalk）

Author: eddy
Date: 2025-11-16
"""

import sys
import os
import importlib.util
from pathlib import Path

# Add project root FIRST
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "compat"))

# Setup triton stub BEFORE any other imports
try:
    from utils import triton_ops_stub
    print("[Infinite Talk] Triton stub loaded")
except Exception as e:
    print(f"[Infinite Talk] Failed to load triton stub: {e}")

# Pre-load comfy module to prevent conflicts later
try:
    import comfy
    import comfy.k_diffusion
    import comfy.k_diffusion.utils
    print("[Infinite Talk] Comfy module pre-loaded")
except Exception as e:
    print(f"[Infinite Talk] Failed to pre-load comfy: {e}")

# Inject server stub BEFORE any custom nodes are loaded
# This prevents all 'import server' from trying to load genesis
try:
    from apps.wanvideo_module import server_stub
    sys.modules['server'] = server_stub
    print("[Infinite Talk] Server stub injected (no genesis dependency)")
except Exception as e:
    print(f"[Infinite Talk] Warning: failed to inject server stub: {e}")

# Try to import genesis (optional, for compatibility)
try:
    import genesis  # noqa: F401
    print("[Infinite Talk] Genesis module available")
except Exception as e:
    print(f"[Infinite Talk] Genesis not available (using stubs): {e}")

# Now import other modules
from typing import Optional, Dict, List, Tuple
import torch
import numpy as np
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add custom nodes path
custom_nodes_path = project_root / "custom_nodes" / "Comfyui"
if str(custom_nodes_path) not in sys.path:
    sys.path.insert(0, str(custom_nodes_path))

# Ensure VideoHelperSuite package is importable (for LoadAudio, etc.)
vhs_package_path = custom_nodes_path / "ComfyUI-VideoHelperSuite"
if str(vhs_package_path) not in sys.path:
    sys.path.insert(0, str(vhs_package_path))

# Try to import WanVideo nodes
WANVIDEO_AVAILABLE = False
WanVideoModelLoader = None
WanVideoVAELoader = None
WanVideoTextEncode = None
WanVideoClipVisionEncode = None
WanVideoImageToVideoMultiTalk = None
WanVideoSampler = None
LoadWanVideoT5TextEncoder = None
MultiTalkWav2VecEmbeds = None
DownloadAndLoadWav2VecModel = None
WanVideoDecode = None
LoadAudio = None
VHS_VideoCombine = None
ImageResizeKJ = None
AudioSeparation = None
AudioCrop = None
AudioDuration = None
SimpleMathNode = None
IntNode = None

try:
    # Check if WanVideoWrapper exists
    wanvideo_wrapper_path = custom_nodes_path / "ComfyUI-WanVideoWrapper"
    if not wanvideo_wrapper_path.exists():
        raise ImportError(f"ComfyUI-WanVideoWrapper not found at {wanvideo_wrapper_path}")
    
    # Add WanVideoWrapper to path
    sys.path.insert(0, str(wanvideo_wrapper_path))
    
    # Import as package to handle relative imports
    import importlib.util
    init_file = wanvideo_wrapper_path / "__init__.py"
    if init_file.exists():
        spec = importlib.util.spec_from_file_location(
            "ComfyUI_WanVideoWrapper",
            init_file,
            submodule_search_locations=[str(wanvideo_wrapper_path)]
        )
        wanvideo_package = importlib.util.module_from_spec(spec)
        sys.modules['ComfyUI_WanVideoWrapper'] = wanvideo_package
        spec.loader.exec_module(wanvideo_package)
        
        # Get NODE_CLASS_MAPPINGS
        if hasattr(wanvideo_package, 'NODE_CLASS_MAPPINGS'):
            mappings = wanvideo_package.NODE_CLASS_MAPPINGS
            
            # Get node classes from mappings
            WanVideoModelLoader = mappings.get('WanVideoModelLoader')
            WanVideoVAELoader = mappings.get('WanVideoVAELoader')
            WanVideoTextEncode = mappings.get('WanVideoTextEncode')
            WanVideoClipVisionEncode = mappings.get('WanVideoClipVisionEncode')
            WanVideoImageToVideoMultiTalk = mappings.get('WanVideoImageToVideoMultiTalk')
            WanVideoSampler = mappings.get('WanVideoSampler')
            LoadWanVideoT5TextEncoder = mappings.get('LoadWanVideoT5TextEncoder')
            MultiTalkWav2VecEmbeds = mappings.get('MultiTalkWav2VecEmbeds')
            DownloadAndLoadWav2VecModel = mappings.get('DownloadAndLoadWav2VecModel')
    
    # Import WanVideoDecode
    decode_path = custom_nodes_path / "ComfyUI-WanVideoDecode-Standalone"
    if decode_path.exists():
        sys.path.insert(0, str(decode_path))
        decode_init = decode_path / "__init__.py"
        if decode_init.exists():
            spec = importlib.util.spec_from_file_location(
                "ComfyUI_WanVideoDecode",
                decode_init
            )
            decode_module = importlib.util.module_from_spec(spec)
            sys.modules['ComfyUI_WanVideoDecode'] = decode_module
            spec.loader.exec_module(decode_module)
            
            if hasattr(decode_module, 'NODE_CLASS_MAPPINGS'):
                decode_mappings = decode_module.NODE_CLASS_MAPPINGS
                WanVideoDecode = decode_mappings.get('WanVideoDecode')
    
    # Load additional nodes from other packages
    # KJNodes for image resize (requires comfy.samplers and nodes module)
    kjnodes_path = custom_nodes_path / "ComfyUI-KJNodes"
    if kjnodes_path.exists():
        sys.path.insert(0, str(kjnodes_path))
        try:
            # Ensure comfy.samplers is accessible before loading KJNodes
            import comfy.samplers  # noqa: F401
            
            # Ensure compat/nodes.py is loaded as 'nodes' module before KJNodes tries to import it
            if 'nodes' not in sys.modules:
                compat_nodes_path = project_root / "compat" / "nodes.py"
                spec = importlib.util.spec_from_file_location("nodes", compat_nodes_path)
                nodes_module = importlib.util.module_from_spec(spec)
                sys.modules['nodes'] = nodes_module
                spec.loader.exec_module(nodes_module)
                logger.debug("Loaded compat/nodes.py as 'nodes' module")
            
            kjnodes_init = kjnodes_path / "__init__.py"
            if kjnodes_init.exists():
                spec = importlib.util.spec_from_file_location("ComfyUI_KJNodes", kjnodes_init)
                kjnodes_module = importlib.util.module_from_spec(spec)
                sys.modules['ComfyUI_KJNodes'] = kjnodes_module
                spec.loader.exec_module(kjnodes_module)
                if hasattr(kjnodes_module, 'NODE_CLASS_MAPPINGS'):
                    ImageResizeKJ = kjnodes_module.NODE_CLASS_MAPPINGS.get('ImageResizeKJ')
                    logger.info("✓ KJNodes loaded (ImageResizeKJ)")
        except Exception as e:
            logger.warning(f"Failed to load KJNodes: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    # Audio separation nodes (requires librosa)
    audio_sep_path = custom_nodes_path / "audio-separation-nodes-comfyui"
    if audio_sep_path.exists():
        sys.path.insert(0, str(audio_sep_path))
        try:
            # Verify librosa is available
            import librosa  # noqa: F401
            
            audio_sep_init = audio_sep_path / "__init__.py"
            if audio_sep_init.exists():
                spec = importlib.util.spec_from_file_location("audio_separation_nodes", audio_sep_init)
                audio_sep_module = importlib.util.module_from_spec(spec)
                sys.modules['audio_separation_nodes'] = audio_sep_module
                spec.loader.exec_module(audio_sep_module)
                if hasattr(audio_sep_module, 'NODE_CLASS_MAPPINGS'):
                    AudioSeparation = audio_sep_module.NODE_CLASS_MAPPINGS.get('AudioSeparation')
                    AudioCrop = audio_sep_module.NODE_CLASS_MAPPINGS.get('AudioCrop')
                    logger.info("✓ Audio separation nodes loaded")
        except ImportError as e:
            logger.warning(f"Failed to load audio separation nodes (librosa issue): {e}")
            import traceback
            logger.debug(traceback.format_exc())
        except Exception as e:
            logger.warning(f"Failed to load audio separation nodes: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    # MTB nodes for audio duration (optional - requires server.router)
    mtb_path = custom_nodes_path / "comfy-mtb"
    if mtb_path.exists():
        sys.path.insert(0, str(mtb_path))
        try:
            mtb_init = mtb_path / "__init__.py"
            if mtb_init.exists():
                spec = importlib.util.spec_from_file_location("comfy_mtb", mtb_init)
                mtb_module = importlib.util.module_from_spec(spec)
                sys.modules['comfy_mtb'] = mtb_module
                spec.loader.exec_module(mtb_module)
                if hasattr(mtb_module, 'NODE_CLASS_MAPPINGS'):
                    AudioDuration = mtb_module.NODE_CLASS_MAPPINGS.get('Audio Duration (mtb)')
                    logger.info("✓ MTB nodes loaded (Audio Duration)")
        except Exception as e:
            logger.warning(f"Failed to load MTB nodes: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    # Comfyroll nodes for SimpleMath+ (optional)
    comfyroll_path = custom_nodes_path / "ComfyUI_Comfyroll_CustomNodes"
    if comfyroll_path.exists():
        # Add parent path instead of the package path itself
        if str(custom_nodes_path) not in sys.path:
            sys.path.insert(0, str(custom_nodes_path))
        try:
            comfyroll_init = comfyroll_path / "__init__.py"
            if comfyroll_init.exists():
                # Import as a package to support relative imports
                spec = importlib.util.spec_from_file_location(
                    "ComfyUI_Comfyroll_CustomNodes", 
                    comfyroll_init,
                    submodule_search_locations=[str(comfyroll_path)]
                )
                comfyroll_module = importlib.util.module_from_spec(spec)
                comfyroll_module.__package__ = "ComfyUI_Comfyroll_CustomNodes"
                sys.modules['ComfyUI_Comfyroll_CustomNodes'] = comfyroll_module
                spec.loader.exec_module(comfyroll_module)
                if hasattr(comfyroll_module, 'NODE_CLASS_MAPPINGS'):
                    SimpleMathNode = comfyroll_module.NODE_CLASS_MAPPINGS.get('SimpleMath+')
                    logger.info("✓ Comfyroll nodes loaded (SimpleMath+)")
        except Exception as e:
            logger.warning(f"Failed to load Comfyroll nodes: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    # ComfyLiterals for Int node
    literals_path = custom_nodes_path / "ComfyLiterals"
    if literals_path.exists():
        sys.path.insert(0, str(literals_path))
        try:
            literals_init = literals_path / "__init__.py"
            if literals_init.exists():
                spec = importlib.util.spec_from_file_location("ComfyLiterals", literals_init)
                literals_module = importlib.util.module_from_spec(spec)
                sys.modules['ComfyLiterals'] = literals_module
                spec.loader.exec_module(literals_module)
                if hasattr(literals_module, 'NODE_CLASS_MAPPINGS'):
                    IntNode = literals_module.NODE_CLASS_MAPPINGS.get('Int')
                    logger.info("✓ ComfyLiterals loaded (Int)")
        except Exception as e:
            logger.warning(f"Failed to load ComfyLiterals: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    # VideoHelperSuite is optional and has import conflicts
    # Skipping for now - core Infinite Talk functionality doesn't require it
    logger.info("Skipping VideoHelperSuite (optional, has import conflicts with comfy module)")
    
    # Check if we have the minimum required nodes
    if (WanVideoModelLoader and WanVideoVAELoader and 
        WanVideoTextEncode and WanVideoSampler):
        WANVIDEO_AVAILABLE = True
        logger.info("✓ WanVideo Infinite Talk nodes available")
    else:
        logger.warning("⚠ Some WanVideo nodes missing")
        logger.warning(f"  WanVideoModelLoader: {WanVideoModelLoader is not None}")
        logger.warning(f"  WanVideoVAELoader: {WanVideoVAELoader is not None}")
        logger.warning(f"  WanVideoTextEncode: {WanVideoTextEncode is not None}")
        logger.warning(f"  WanVideoSampler: {WanVideoSampler is not None}")
        WANVIDEO_AVAILABLE = False
        
except Exception as e:
    logger.warning(f"⚠ WanVideo Infinite Talk nodes not available: {e}")
    import traceback
    traceback.print_exc()
    WANVIDEO_AVAILABLE = False


class InfiniteTalkPipeline:
    """
    Infinite Talk pipeline using ComfyUI format
    图像 + 音频 -> 说话视频
    """
    
    def __init__(self):
        """Initialize pipeline"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Model instances
        self.model = None
        self.vae = None
        self.t5_encoder = None
        self.clip_vision = None
        self.wav2vec_model = None
        
        # Current loaded models
        self.current_model = None
        self.current_vae = None
        
        logger.info(f"Infinite Talk Pipeline initialized on {self.device}")
        
        if not WANVIDEO_AVAILABLE:
            logger.error("WanVideo nodes not available - pipeline will not work!")
    
    def load_models(
        self,
        model_name: str,
        vae_name: str,
        t5_model_name: str = "google/umt5-xxl",
        clip_vision_name: str = "clip_vision_g.safetensors",
        wav2vec_model_name: str = "facebook/wav2vec2-base-960h",
        model_quantization: str = "fp4_scaled",
        model_attention: str = "sageattn_3_fp4",
        vae_precision: str = "bf16",
        model_precision: str = "bf16"
    ) -> bool:
        """
        Load all required models
        
        Args:
            model_name: WanVideo model name
            vae_name: VAE model name
            t5_model_name: T5 text encoder model
            clip_vision_name: CLIP Vision model
            wav2vec_model_name: Wav2Vec model for audio
            model_quantization: Model quantization mode
            model_attention: Attention mode
            vae_precision: VAE precision
            model_precision: Model base precision
            
        Returns:
            Success status
        """
        if not WANVIDEO_AVAILABLE:
            logger.error("WanVideo nodes not available")
            return False
        
        try:
            logger.info("=" * 70)
            logger.info("Loading Infinite Talk Models")
            logger.info("=" * 70)
            
            # Load WanVideo model
            logger.info(f"Loading WanVideo model: {model_name}")
            logger.info(f"  Quantization: {model_quantization}")
            logger.info(f"  Attention mode: {model_attention}")
            logger.info(f"  Base precision: {model_precision}")
            model_loader = WanVideoModelLoader()
            # WanVideoModelLoader uses 'loadmodel' method, not 'load_model'
            # Use parameters from UI
            result = model_loader.loadmodel(
                model=model_name,
                base_precision=model_precision,
                quantization=model_quantization,
                load_device="main_device",
                attention_mode=model_attention,
                rms_norm_function="default",
                compile_args=None,  # No torch.compile
                block_swap_args=None  # No block swap for now
            )
            # Handle different return types
            if isinstance(result, tuple):
                self.model = result[0]
            elif hasattr(result, '__getitem__') and not isinstance(result, (dict, torch.Tensor)):
                try:
                    self.model = result[0]
                except (IndexError, TypeError):
                    self.model = result
            else:
                self.model = result
            self.current_model = model_name
            logger.info("✓ WanVideo model loaded")
            
            # Load VAE
            logger.info(f"Loading VAE: {vae_name}")
            logger.info(f"  VAE precision: {vae_precision}")
            vae_loader = WanVideoVAELoader()
            # WanVideoVAELoader uses 'loadmodel' method
            # Use VAE precision from UI
            result = vae_loader.loadmodel(
                model_name=vae_name,
                precision=vae_precision,
                compile_args=None  # No torch.compile
            )
            # Handle different return types
            if isinstance(result, tuple):
                self.vae = result[0]
            elif hasattr(result, '__getitem__') and not isinstance(result, (dict, torch.Tensor)):
                try:
                    self.vae = result[0]
                except (IndexError, TypeError):
                    self.vae = result
            else:
                self.vae = result
            self.current_vae = vae_name
            logger.info("✓ VAE loaded")
            
            # Load T5 encoder
            logger.info(f"Loading T5 encoder: {t5_model_name}")
            t5_loader = LoadWanVideoT5TextEncoder()
            # LoadWanVideoT5TextEncoder uses 'loadmodel' method
            # Match workflow parameters
            result = t5_loader.loadmodel(
                model_name=t5_model_name,
                precision="bf16",
                load_device="offload_device",
                quantization="disabled"  # Match workflow
            )
            # Handle different return types
            if isinstance(result, tuple):
                self.t5_encoder = result[0]
            elif hasattr(result, '__getitem__') and not isinstance(result, (dict, torch.Tensor)):
                try:
                    self.t5_encoder = result[0]
                except (IndexError, TypeError):
                    self.t5_encoder = result
            else:
                self.t5_encoder = result
            logger.info("✓ T5 encoder loaded")
            
            # Load CLIP Vision
            logger.info(f"Loading CLIP Vision: {clip_vision_name}")
            # Use CLIPVisionLoader from nodes_stub.py (same as workflow uses)
            import importlib.util
            vhs_stub_path = custom_nodes_path / "ComfyUI-VideoHelperSuite" / "nodes_stub.py"
            spec = importlib.util.spec_from_file_location("nodes_stub_clip", vhs_stub_path)
            nodes_stub = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(nodes_stub)
            
            # Temporarily suppress text_projection.weight warning (it's optional for CLIP Vision)
            import logging as temp_logging
            root_logger = temp_logging.getLogger()
            old_level = root_logger.level
            root_logger.setLevel(temp_logging.ERROR)
            
            try:
                clip_loader = nodes_stub.CLIPVisionLoader()
                result = clip_loader.load_clip(clip_vision_name)
                # Handle different return types
                if isinstance(result, tuple):
                    self.clip_vision = result[0]
                elif hasattr(result, '__getitem__') and not isinstance(result, (dict, torch.Tensor)):
                    try:
                        self.clip_vision = result[0]
                    except (IndexError, TypeError):
                        self.clip_vision = result
                else:
                    self.clip_vision = result
            finally:
                root_logger.setLevel(old_level)
            
            logger.info("✓ CLIP Vision loaded")
            
            # Load Wav2Vec model
            logger.info(f"Loading Wav2Vec model: {wav2vec_model_name}")
            
            # Check if it's a local path (from models/audio_encoders)
            # or a HuggingFace model name
            if not wav2vec_model_name.startswith("facebook/") and not wav2vec_model_name.startswith("TencentGameMate/"):
                # Local path - convert to full path (project_root already defined at top of file)
                audio_encoders_dir = project_root / "models" / "audio_encoders"
                full_model_path = audio_encoders_dir / wav2vec_model_name
                
                # For local models, we need to use the path directly
                # The node expects HuggingFace format, so we'll use a workaround
                logger.info(f"Using local Wav2Vec model: {full_model_path}")
                
                # Import transformers directly for local models
                from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
                import torch
                
                # Determine which type based on path
                if "chinese" in wav2vec_model_name.lower():
                    # Chinese model uses MultiTalkWav2Vec2Model
                    import sys
                    wrapper_path = project_root / "custom_nodes" / "Comfyui" / "ComfyUI-WanVideoWrapper"
                    if str(wrapper_path) not in sys.path:
                        sys.path.insert(0, str(wrapper_path))
                    from multitalk.wav2vec2 import Wav2Vec2Model as MultiTalkWav2Vec2Model
                    
                    wav2vec = MultiTalkWav2Vec2Model.from_pretrained(str(full_model_path)).to(torch.float16).eval()
                    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(full_model_path), local_files_only=True)
                    self.wav2vec_model = {
                        "processor": None,
                        "feature_extractor": wav2vec_feature_extractor,
                        "model": wav2vec,
                        "model_type": "TencentGameMate/chinese-wav2vec2-base",
                        "dtype": torch.float16
                    }
                else:
                    # English model
                    wav2vec_processor = Wav2Vec2Processor.from_pretrained(str(full_model_path))
                    wav2vec = Wav2Vec2Model.from_pretrained(str(full_model_path)).to(torch.float16).eval()
                    self.wav2vec_model = {
                        "processor": wav2vec_processor,
                        "feature_extractor": None,
                        "model": wav2vec,
                        "model_type": "facebook/wav2vec2-base",
                        "dtype": torch.float16
                    }
            else:
                # HuggingFace model - use the node
                wav2vec_loader = DownloadAndLoadWav2VecModel()
                result = wav2vec_loader.loadmodel(
                    model=wav2vec_model_name,
                    base_precision="fp16",
                    load_device="main_device"
                )
                # Handle different return types
                if isinstance(result, tuple):
                    self.wav2vec_model = result[0]
                elif isinstance(result, dict):
                    self.wav2vec_model = result
                elif hasattr(result, '__getitem__') and not isinstance(result, torch.Tensor):
                    try:
                        self.wav2vec_model = result[0]
                    except (IndexError, TypeError):
                        self.wav2vec_model = result
                else:
                    self.wav2vec_model = result
            
            logger.info("✓ Wav2Vec model loaded")
            
            logger.info("=" * 70)
            logger.info("✓ All models loaded successfully")
            logger.info("=" * 70)
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate(
        self,
        image_path: str,
        audio_path: str,
        prompt: str = "",
        negative_prompt: str = "",
        width: int = 832,
        height: int = 480,
        video_length: int = 117,  # frame_window_size for video
        steps: int = 6,
        cfg: float = 1.0,
        sampler_name: str = "dpm++_sde",
        scheduler: str = "unipc",
        shift: float = 7.0,
        seed: int = -1,
        fps: int = 25,
        # Audio parameters (match workflow)
        audio_num_frames: int = 33,  # num_frames for audio processing
        audio_scale: float = 1.0,
        audio_cfg_scale: float = 1.0,
        normalize_loudness: bool = True,
        # Video generation parameters
        motion_frame: int = 25,
        colormatch: str = "mkl",
        # Image preprocessing parameters
        use_image_resize: bool = True,
        resize_interpolation: str = "lanczos",
        resize_method: str = "stretch",
        resize_condition: str = "always",
        # Audio preprocessing parameters
        enable_audio_crop: bool = False,
        audio_start_time: float = 0.0,
        audio_crop_duration: float = 0.0,
        enable_audio_separation: bool = False,
        separation_model: str = "UVR-MDX-NET-Inst_HQ_3",
        # Auto calculation parameters
        auto_calculate_frames: bool = True,
        max_frames: int = 200,
        optimization_args: dict = None
    ):
        """
        Generate Infinite Talk video
        
        Args:
            image_path: Path to input image
            audio_path: Path to input audio
            prompt: Text prompt
            negative_prompt: Negative prompt
            width: Video width
            height: Video height
            video_length: Number of video frames (frame_window_size)
            steps: Sampling steps
            cfg: CFG scale
            sampler_name: Sampler name
            scheduler: Scheduler name
            shift: Shift parameter for timestep
            seed: Random seed
            fps: Output FPS
            audio_num_frames: Audio processing frames (for duration calculation)
            audio_scale: Audio conditioning strength
            audio_cfg_scale: Audio CFG scale
            normalize_loudness: Normalize audio loudness
            motion_frame: Motion frame parameter
            colormatch: Color matching method
            optimization_args: Optimization arguments (BlockSwap, CUDA, etc.)
            
        Returns:
            Path to output video
        """
        if optimization_args is None:
            optimization_args = {
                'blocks_to_swap': 20,
                'enable_cuda_optimization': True,
                'auto_hardware_tuning': True,
            }
        if not WANVIDEO_AVAILABLE:
            logger.error("WanVideo nodes not available")
            return None
        
        if self.model is None or self.vae is None:
            logger.error("Models not loaded")
            return None
        
        try:
            logger.info("=" * 70)
            logger.info("Infinite Talk Generation")
            logger.info("=" * 70)
            logger.info(f"Image: {image_path}")
            logger.info(f"Audio: {audio_path}")
            logger.info(f"Size: {width}x{height}")
            logger.info(f"Frames: {video_length}")
            logger.info(f"Steps: {steps}")
            
            # Set seed
            if seed == -1:
                seed = torch.randint(0, 2**32, (1,)).item()
            logger.info(f"Seed: {seed}")
            
            # Load image
            logger.info("\n1. Loading and preprocessing image...")
            from nodes import LoadImage
            image_loader = LoadImage()
            image_result = image_loader.load_image(image_path)
            # Handle different return types
            if isinstance(image_result, tuple):
                image = image_result[0]
            elif isinstance(image_result, torch.Tensor):
                image = image_result
            elif hasattr(image_result, '__getitem__') and not isinstance(image_result, dict):
                try:
                    image = image_result[0]
                except (IndexError, TypeError):
                    image = image_result
            else:
                image = image_result
            
            logger.info(f"  Original image shape: {image.shape}")
            
            # Image preprocessing with ImageResizeKJ
            if use_image_resize and ImageResizeKJ:
                logger.info(f"  Resizing image to {width}x{height} using {resize_interpolation}")
                logger.info(f"  Resize method: {resize_method}, condition: {resize_condition}")
                
                try:
                    resizer = ImageResizeKJ()
                    resize_result = resizer.resize(
                        image=image,
                        width=width,
                        height=height,
                        interpolation=resize_interpolation,
                        method=resize_method,
                        condition=resize_condition,
                        multiple_of=8  # Ensure dimensions are multiples of 8
                    )
                    # Handle different return types
                    if isinstance(resize_result, tuple):
                        image = resize_result[0]
                    elif isinstance(resize_result, torch.Tensor):
                        image = resize_result
                    elif hasattr(resize_result, '__getitem__') and not isinstance(resize_result, dict):
                        try:
                            image = resize_result[0]
                        except (IndexError, TypeError):
                            image = resize_result
                    else:
                        image = resize_result
                    logger.info(f"  Resized image shape: {image.shape}")
                except Exception as e:
                    logger.warning(f"  ImageResizeKJ failed, using fallback: {e}")
                    # Fallback: simple resize
                    if image.shape[1] != height or image.shape[2] != width:
                        import torch.nn.functional as F
                        image = F.interpolate(
                            image.permute(0, 3, 1, 2),
                            size=(height, width),
                            mode='bilinear',
                            align_corners=False
                        ).permute(0, 2, 3, 1)
            elif image.shape[1] != height or image.shape[2] != width:
                # Fallback resize if ImageResizeKJ not available
                logger.info(f"  Fallback resizing to {width}x{height}")
                import torch.nn.functional as F
                image = F.interpolate(
                    image.permute(0, 3, 1, 2),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            
            # Load audio using VideoHelperSuite's LoadAudio node
            logger.info("\n2. Loading audio...")
            # Import LoadAudio from VideoHelperSuite as a proper package
            from importlib import import_module
            
            # Import videohelpersuite.nodes (VHS package already in sys.path)
            vhs_nodes = import_module("videohelpersuite.nodes")
            
            audio_loader = vhs_nodes.LoadAudio()
            audio_result = audio_loader.load_audio(audio_path, seek_seconds=0, duration=0)
            
            # Handle different return types
            if isinstance(audio_result, tuple):
                audio = audio_result[0]  # (audio, duration)
                duration = audio_result[1] if len(audio_result) > 1 else 0
            elif hasattr(audio_result, '__getitem__') and not isinstance(audio_result, dict):
                try:
                    audio = audio_result[0]
                    duration = audio_result[1] if len(audio_result) > 1 else 0
                except (IndexError, TypeError):
                    audio = audio_result
                    duration = 0
            else:
                audio = audio_result
                duration = 0
            
            logger.info(f"  Audio loaded: waveform shape={audio['waveform'].shape}, sample_rate={audio['sample_rate']}, duration={duration:.2f}s")
            
            # Audio preprocessing
            logger.info("\n2.1. Audio preprocessing...")
            
            # Audio cropping (optional)
            if enable_audio_crop and AudioCrop and audio_crop_duration > 0:
                logger.info(f"  Cropping audio: start={audio_start_time}s, duration={audio_crop_duration}s")
                try:
                    crop_node = AudioCrop()
                    audio_result = crop_node.crop_audio(
                        audio=audio,
                        start_time=audio_start_time,
                        end_time=audio_start_time + audio_crop_duration
                    )
                    if isinstance(audio_result, tuple):
                        audio = audio_result[0]
                    else:
                        audio = audio_result
                    logger.info(f"  Cropped audio shape: {audio['waveform'].shape}")
                except Exception as e:
                    logger.warning(f"  Audio crop failed: {e}")
            
            # Audio separation (optional)
            if enable_audio_separation and AudioSeparation:
                logger.info(f"  Separating audio using model: {separation_model}")
                try:
                    sep_node = AudioSeparation()
                    audio_result = sep_node.separate(
                        audio=audio,
                        model=separation_model,
                        segment=256,
                        overlap=0.25,
                        denoise=True,
                        output="Vocals"  # Extract vocals only
                    )
                    if isinstance(audio_result, tuple):
                        audio = audio_result[0]
                    else:
                        audio = audio_result
                    logger.info(f"  Separated audio shape: {audio['waveform'].shape}")
                except Exception as e:
                    logger.warning(f"  Audio separation failed: {e}")
            
            # Get audio duration for auto frame calculation
            audio_duration_seconds = duration if duration > 0 else (audio['waveform'].shape[-1] / audio['sample_rate'])
            logger.info(f"  Final audio duration: {audio_duration_seconds:.2f}s")
            
            # Auto calculate video frames based on audio duration
            if auto_calculate_frames and audio_duration_seconds > 0:
                calculated_frames = int(audio_duration_seconds * fps)
                original_video_length = video_length
                video_length = min(calculated_frames, max_frames)
                logger.info(f"  Auto-calculated frames: {calculated_frames} (capped at {max_frames})")
                if video_length != original_video_length:
                    logger.info(f"  Adjusted video_length from {original_video_length} to {video_length}")
                
                # Also adjust audio_num_frames proportionally
                if calculated_frames > 0:
                    audio_num_frames = min(int(audio_duration_seconds * fps / 4), video_length)
                    logger.info(f"  Adjusted audio_num_frames to {audio_num_frames}")
            
            # Encode image with CLIP Vision
            logger.info("\n3. Encoding image...")
            # Directly use clip_vision.encode_image() to bypass isinstance check
            # This avoids the ClipVisionModel type mismatch issue
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            clip_output = self.clip_vision.encode_image(image.to(device))
            # Handle different output types
            if hasattr(clip_output, 'penultimate_hidden_states'):
                clip_embeds_raw = clip_output.penultimate_hidden_states
            elif hasattr(clip_output, '__getitem__'):
                clip_embeds_raw = clip_output['penultimate_hidden_states'] if isinstance(clip_output, dict) else clip_output[0]
            else:
                clip_embeds_raw = clip_output
            
            # Apply strength and prepare embeds dict (matching WanVideoClipVisionEncode output format)
            strength_1 = 1.0
            weighted_embeds = clip_embeds_raw[0:1] * strength_1
            
            # Create the expected output format
            clip_embeds = {
                "cond": weighted_embeds,
                "uncond": torch.zeros_like(weighted_embeds)
            }
            logger.info(f"  CLIP embeds shape: {weighted_embeds.shape}")
            
            # Encode text
            logger.info("\n4. Encoding text...")
            text_encoder = WanVideoTextEncode()
            # WanVideoTextEncode.process() parameters
            text_result = text_encoder.process(
                t5=self.t5_encoder,
                positive_prompt=prompt,
                negative_prompt=negative_prompt,
                force_offload=True,
                use_disk_cache=False,
                device="gpu"
            )
            # Handle different return types: tuple, NodeOutput, or direct value
            if isinstance(text_result, tuple):
                positive_embeds = text_result[0]
            elif hasattr(text_result, '__getitem__') and not isinstance(text_result, (dict, torch.Tensor)):
                try:
                    positive_embeds = text_result[0]
                except (IndexError, TypeError):
                    positive_embeds = text_result
            else:
                positive_embeds = text_result
            # For Infinite Talk, we don't need separate negative embeds
            negative_embeds = None
            
            # Extract audio embeddings
            logger.info("\n5. Extracting audio embeddings...")
            logger.info(f"  Audio num_frames: {audio_num_frames}, fps: {fps}")
            logger.info(f"  Audio duration: {audio_num_frames / fps:.2f}s")
            wav2vec_embedder = MultiTalkWav2VecEmbeds()
            
            wav2vec_result = wav2vec_embedder.process(
                wav2vec_model=self.wav2vec_model,
                audio_1=audio,
                normalize_loudness=normalize_loudness,
                num_frames=audio_num_frames,
                fps=fps,
                audio_scale=audio_scale,
                audio_cfg_scale=audio_cfg_scale,
                multi_audio_type="para"
            )
            # Handle different return types: tuple (multitalk_embeds, audio, num_frames)
            if isinstance(wav2vec_result, tuple):
                audio_embeds = wav2vec_result[0]  # multitalk_embeds
            elif hasattr(wav2vec_result, '__getitem__') and not isinstance(wav2vec_result, (dict, torch.Tensor)):
                try:
                    audio_embeds = wav2vec_result[0]
                except (IndexError, TypeError):
                    audio_embeds = wav2vec_result
            else:
                audio_embeds = wav2vec_result
            
            # Generate latents
            logger.info("\n6. Generating video latents...")
            video_generator = WanVideoImageToVideoMultiTalk()
            # WanVideoImageToVideoMultiTalk.process() parameters from workflow
            latents_result = video_generator.process(
                vae=self.vae,
                start_image=image,
                clip_embeds=clip_embeds,
                width=width,
                height=height,
                frame_window_size=video_length,
                motion_frame=motion_frame,
                force_offload=False,
                colormatch=colormatch,
                tiled_vae=False,
                mode="infinitetalk",
                output_path=""
            )
            # Handle different return types: tuple (image_embeds, output_path)
            if isinstance(latents_result, tuple):
                image_embeds = latents_result[0]  # image_embeds
                output_path_from_node = latents_result[1] if len(latents_result) > 1 else ""
            elif hasattr(latents_result, '__getitem__') and not isinstance(latents_result, (dict, torch.Tensor)):
                try:
                    image_embeds = latents_result[0]
                    output_path_from_node = latents_result[1] if len(latents_result) > 1 else ""
                except (IndexError, TypeError):
                    image_embeds = latents_result
                    output_path_from_node = ""
            else:
                image_embeds = latents_result
                output_path_from_node = ""
            
            # Verify multitalk_sampling flag is set
            logger.info(f"  image_embeds keys: {image_embeds.keys() if isinstance(image_embeds, dict) else 'not a dict'}")
            if isinstance(image_embeds, dict):
                logger.info(f"  multitalk_sampling: {image_embeds.get('multitalk_sampling', 'NOT SET')}")
                logger.info(f"  frame_window_size: {image_embeds.get('frame_window_size', 'NOT SET')}")
                logger.info(f"  motion_frame: {image_embeds.get('motion_frame', 'NOT SET')}")
            
            # Sample
            logger.info("\n7. Sampling...")
            
            # Monkey-patch latent_preview to avoid genesis/server dependency
            # This must be done before creating WanVideoSampler
            try:
                from apps.wanvideo_module import latent_preview_standalone
                
                # Replace the module in sys.modules so nodes_sampler imports our version
                # Try multiple possible import paths
                sys.modules['latent_preview'] = latent_preview_standalone
                sys.modules['custom_nodes.Comfyui.ComfyUI-WanVideoWrapper.latent_preview'] = latent_preview_standalone
                sys.modules['ComfyUI-WanVideoWrapper.latent_preview'] = latent_preview_standalone
                
                logger.info("  Using standalone latent_preview (no genesis dependency)")
            except Exception as e:
                logger.warning(f"  Failed to inject latent_preview stub: {e}")
            
            sampler = WanVideoSampler()
            # WanVideoSampler.process() parameters from workflow
            # IMPORTANT: When multitalk_sampling=True in image_embeds, scheduler MUST be "multitalk" (string)
            # Check if image_embeds has multitalk_sampling flag
            is_multitalk = isinstance(image_embeds, dict) and image_embeds.get('multitalk_sampling', False)
            
            # For multitalk mode, use "multitalk" string directly
            # The WanVideoSampler will handle it internally with fixed timesteps
            if is_multitalk:
                actual_scheduler = "multitalk"
                logger.info(f"  multitalk_sampling detected: True")
                logger.info(f"  Using scheduler: multitalk (string)")
            else:
                actual_scheduler = scheduler
                logger.info(f"  multitalk_sampling detected: False")
                logger.info(f"  Using scheduler: {actual_scheduler}")
            
            logger.info(f"  audio_embeds present: {audio_embeds is not None}")
            
            sampled_result = sampler.process(
                model=self.model,
                image_embeds=image_embeds,
                text_embeds=positive_embeds,
                multitalk_embeds=audio_embeds,
                shift=shift,
                steps=steps,
                cfg=cfg,
                seed=seed,
                scheduler=actual_scheduler,
                riflex_freq_index=0,
                force_offload=True
            )
            # Handle different return types: tuple (samples, denoised_samples)
            if isinstance(sampled_result, tuple):
                sampled_latents = sampled_result[0]  # samples
            elif isinstance(sampled_result, dict):
                # If it's already a latent dict, use it directly
                sampled_latents = sampled_result
            elif hasattr(sampled_result, '__getitem__') and not isinstance(sampled_result, torch.Tensor):
                try:
                    sampled_latents = sampled_result[0]
                except (IndexError, TypeError):
                    sampled_latents = sampled_result
            else:
                sampled_latents = sampled_result
            
            # Check if sampler already returned video frames
            logger.info("\n8. Processing sampler output...")
            logger.info(f"  Samples type: {type(sampled_latents)}")
            if isinstance(sampled_latents, dict):
                logger.info(f"  Samples keys: {sampled_latents.keys()}")
                
                # Check if video is already decoded
                if 'video' in sampled_latents:
                    logger.info("  Sampler returned decoded video, using directly")
                    frames = sampled_latents['video']
                    
                    # Check if output_path is provided
                    if 'output_path' in sampled_latents and sampled_latents['output_path']:
                        output_path = sampled_latents['output_path']
                        logger.info(f"  Video already saved to: {output_path}")
                        logger.info("=" * 70)
                        return output_path
                    
                    # If no output_path, we need to save the frames ourselves
                    logger.info("  Video frames available, will save manually")
                elif 'samples' in sampled_latents:
                    # Standard latent dict, need to decode
                    logger.info("  Samples contain latents, decoding with VAE...")
                    
                    # Verify VAE is loaded
                    if self.vae is None:
                        raise RuntimeError("VAE is not loaded. Please load models first.")
                    
                    # Verify VAE has decode method
                    if not hasattr(self.vae, 'decode') or self.vae.decode is None:
                        raise RuntimeError("VAE decode method is not available. Model may not be loaded correctly.")
                    
                    logger.info(f"  VAE type: {type(self.vae).__name__}")
                    
                    decoder = WanVideoDecode()
                    # WanVideoDecode.decode() parameters from workflow
                    frames_result = decoder.decode(
                        vae=self.vae,
                        samples=sampled_latents,
                        enable_vae_tiling=False,
                        tile_x=272,
                        tile_y=272,
                        tile_stride_x=144,
                        tile_stride_y=128,
                        normalization="default"
                    )
                    # Handle different return types: tuple (images,)
                    if isinstance(frames_result, tuple):
                        frames = frames_result[0]  # images tensor
                    elif isinstance(frames_result, torch.Tensor):
                        frames = frames_result
                    elif hasattr(frames_result, '__getitem__') and not isinstance(frames_result, dict):
                        try:
                            frames = frames_result[0]
                        except (IndexError, TypeError):
                            frames = frames_result
                    else:
                        frames = frames_result
                else:
                    raise RuntimeError(f"Unexpected sampled_latents structure: {sampled_latents.keys()}")
            else:
                # Not a dict, assume it's latent tensor
                logger.info("  Samples are tensor, decoding with VAE...")
                
                if self.vae is None:
                    raise RuntimeError("VAE is not loaded. Please load models first.")
                
                decoder = WanVideoDecode()
                frames_result = decoder.decode(
                    vae=self.vae,
                    samples={'samples': sampled_latents},
                    enable_vae_tiling=False,
                    tile_x=272,
                    tile_y=272,
                    tile_stride_x=144,
                    tile_stride_y=128,
                    normalization="default"
                )
                if isinstance(frames_result, tuple):
                    frames = frames_result[0]
                else:
                    frames = frames_result
            
            # Save video frames
            logger.info("\n9. Saving video...")
            # Since VHS_VideoCombine is not available, save frames directly
            from pathlib import Path
            from datetime import datetime
            import imageio
            
            output_dir = Path(__file__).parent.parent.parent / "output"
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"infinite_talk_{timestamp}.mp4"
            
            # Convert frames to numpy arrays
            frames_np = []
            for frame in frames:
                # frame shape: [H, W, C], range [0, 1]
                frame_np = (frame.cpu().numpy() * 255).astype('uint8')
                frames_np.append(frame_np)
            
            # Write video with imageio
            imageio.mimsave(str(output_path), frames_np, fps=fps, codec='libx264')
            
            logger.info("=" * 70)
            logger.info(f"✅ Video generated: {output_path}")
            logger.info(f"📁 Saved to: {output_dir}")
            logger.info("=" * 70)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # Test
    logger.info("Testing Infinite Talk Pipeline")
    logger.info("=" * 70)
    
    if not WANVIDEO_AVAILABLE:
        logger.error("WanVideo nodes not available")
        sys.exit(1)
    
    # Create pipeline
    pipeline = InfiniteTalkPipeline()
    
    logger.info("\n✓ Pipeline ready")
    logger.info("Load models with: pipeline.load_models(model_name, vae_name)")
    logger.info("Generate with: pipeline.generate(image_path, audio_path, ...)")
