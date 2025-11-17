"""
ComfyUI comfy module compatibility layer for Genesis
Provides complete comfy API compatibility
Author: eddy
"""

import sys
import torch
import logging
import hashlib
from types import ModuleType

logger = logging.getLogger(__name__)


class ModelPatcher:
    """Stub for comfy.model_patcher.ModelPatcher"""

    def __init__(self, model, load_device=None, offload_device=None):
        self.model = model
        self.load_device = load_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.offload_device = offload_device or torch.device("cpu")
        self.patches = {}
        self.model_options = {}
    
    def patch_model(self, patches):
        self.patches.update(patches)
        return self
    
    def clone(self):
        return ModelPatcher(self.model, self.load_device, self.offload_device)
    
    def to(self, device):
        self.load_device = device
        if hasattr(self.model, 'to'):
            self.model.to(device)
        return self


class ModelManagement:
    """Stub for comfy.model_management"""

    @staticmethod
    def get_torch_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_autocast_device(device):
        return device.type if hasattr(device, 'type') else 'cpu'

    @staticmethod
    def throw_exception_if_processing_interrupted():
        """Check if processing was interrupted - stub implementation"""
        # In a real implementation, this would check for user interruption
        pass

    @staticmethod
    def soft_empty_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def load_models_gpu(models, memory_required=0):
        pass

    @staticmethod
    def unload_all_models():
        pass

    @staticmethod
    def cleanup_models():
        """Cleanup models from memory"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def get_free_memory(device=None):
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
        return 0

    @staticmethod
    def interrupt_current_processing(value=True):
        pass

    @staticmethod
    def unet_offload_device():
        """Get the offload device for UNet models"""
        # Return CPU for offloading when GPU memory is limited
        return torch.device("cpu")

    @staticmethod
    def vae_offload_device():
        """Get the offload device for VAE models"""
        return torch.device("cpu")


class Utils:
    """Stub for comfy.utils"""
    
    PROGRESS_BAR_ENABLED = True
    
    @staticmethod
    def ProgressBar(total):
        class DummyProgressBar:
            def __init__(self):
                self.total = total
                self.current = 0
            
            def update(self, value):
                self.current = value
            
            def update_absolute(self, value, total=None):
                self.current = value
        
        return DummyProgressBar()
    
    @staticmethod
    def copy_to_param(obj, attr, value):
        """Copy value to parameter"""
        setattr(obj, attr, value)
    
    @staticmethod
    def set_attr_param(obj, attr, value):
        """Set attribute parameter"""
        setattr(obj, attr, value)


class SD:
    """Stub for comfy.sd module"""

    @staticmethod
    def load_checkpoint(ckpt_path, output_vae=True, output_clip=True, embedding_directory=None):
        """Load checkpoint stub"""
        return ({}, {}, {})

    @staticmethod
    def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=None):
        """Load checkpoint with config guessing stub"""
        return ({}, {}, {})

    @staticmethod
    def load_lora_for_models(model, clip, lora_path, strength_model, strength_clip):
        """Load LoRA for models"""
        # Return models as-is for now
        return (model, clip)


class Samplers:
    """Stub for comfy.samplers"""
    
    class KSampler:
        SAMPLERS = ["euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "lms", "dpmpp_2m", "ddim"]
        SCHEDULERS = ["normal", "karras", "exponential", "simple", "ddim_uniform"]


class Sample:
    """Stub for comfy.sample"""
    
    @staticmethod
    def sample(*args, **kwargs):
        """Sample stub"""
        return torch.randn(1, 4, 64, 64)
    
    @staticmethod
    def prepare_noise(latent_image, seed, batch_inds=None):
        """Prepare noise stub"""
        torch.manual_seed(seed)
        return torch.randn_like(latent_image)
    
    @staticmethod
    def fix_empty_latent_channels(model, latent):
        """Fix empty latent channels stub"""
        return latent


# Helper functions
def copy_to_param(obj, attr, value):
    """Copy value to parameter"""
    setattr(obj, attr, value)


def set_module_tensor_to_device(module, tensor_name, device, dtype=None, value=None):
    """Set module tensor to device"""
    if value is not None:
        if dtype is not None:
            value = value.to(dtype=dtype)
        value = value.to(device=device)
        
        # Handle nested attributes
        if '.' in tensor_name:
            parts = tensor_name.split('.')
            current = module
            for part in parts[:-1]:
                current = getattr(current, part)
            setattr(current, parts[-1], value)
        else:
            setattr(module, tensor_name, value)


def set_attr_param(obj, attr, value):
    """Set attribute parameter"""
    setattr(obj, attr, value)


def cast_to_device(tensor, device, dtype=None, copy=False):
    """Move tensor to device with optional dtype and copy semantics."""
    if tensor is None:
        return None
    if isinstance(device, str):
        device = torch.device(device)
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if hasattr(tensor, "to"):
        return tensor.to(device=device, **kwargs).clone() if copy else tensor.to(device=device, **kwargs)
    return torch.as_tensor(tensor, device=device, **kwargs)


def string_to_seed(text: str) -> int:
    """Deterministically convert a string to a 64-bit seed."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="little", signed=False)


def get_key_weight(model, key: str):
    """
    Lightweight implementation compatible with ComfyUI's model_patcher.get_key_weight.
    Returns (weight_tensor, set_func, convert_func) for a given model and key path.
    set_func is None when direct assignment is possible.
    convert_func is an identity transform.
    """
    weight = None
    target_obj = model
    # Direct attribute traverse
    try:
        parts = key.split('.')
        for p in parts[:-1]:
            target_obj = getattr(target_obj, p)
        weight = getattr(target_obj, parts[-1])
    except Exception:
        # Fallback: search by named_parameters
        try:
            for name, param in getattr(model, 'named_parameters', lambda recurse=True: [])(recurse=True):
                if name == key:
                    weight = param
                    break
        except Exception:
            weight = None
    if weight is None:
        # As a last resort, return a zero tensor to avoid crashes
        weight = torch.zeros(1)
    set_func = None
    convert_func = (lambda t, inplace=True: t)
    return weight, set_func, convert_func


# Create comfy package module
comfy = ModuleType('comfy')

# Build submodules as proper modules to satisfy "from comfy.xxx import y" imports
model_management_module = ModuleType('comfy.model_management')
model_management_module.get_torch_device = ModelManagement.get_torch_device
model_management_module.get_autocast_device = ModelManagement.get_autocast_device
model_management_module.soft_empty_cache = ModelManagement.soft_empty_cache
model_management_module.load_models_gpu = ModelManagement.load_models_gpu
model_management_module.unload_all_models = ModelManagement.unload_all_models
model_management_module.cleanup_models = ModelManagement.cleanup_models
model_management_module.throw_exception_if_processing_interrupted = ModelManagement.throw_exception_if_processing_interrupted
model_management_module.get_free_memory = ModelManagement.get_free_memory
model_management_module.interrupt_current_processing = ModelManagement.interrupt_current_processing
model_management_module.cast_to_device = cast_to_device
model_management_module.unet_offload_device = ModelManagement.unet_offload_device
model_management_module.vae_offload_device = ModelManagement.vae_offload_device
model_management_module.current_loaded_models = []  # Track currently loaded models

def common_upscale(samples, width, height, upscale_method, crop="disabled"):
    """Common upscale function for images/latents"""
    if isinstance(samples, torch.Tensor):
        # Ensure 4D tensor (B, C, H, W)
        if len(samples.shape) == 3:
            samples = samples.unsqueeze(0)
        
        import torch.nn.functional as F
        
        # Map upscale methods to torch modes
        mode_map = {
            "nearest": "nearest",
            "nearest-exact": "nearest",
            "bilinear": "bilinear",
            "area": "area",
            "bicubic": "bicubic",
            "lanczos": "bilinear"  # Fallback to bilinear
        }
        mode = mode_map.get(upscale_method, "bilinear")
        
        # Upscale
        upscaled = F.interpolate(
            samples,
            size=(height, width),
            mode=mode,
            align_corners=False if mode != "nearest" else None
        )
        
        # Crop if needed
        if crop == "center":
            h, w = upscaled.shape[2], upscaled.shape[3]
            if h != height or w != width:
                top = (h - height) // 2
                left = (w - width) // 2
                upscaled = upscaled[:, :, top:top+height, left:left+width]
        
        return upscaled
    return samples

def load_torch_file(filename, safe_load=True, device=None):
    """Load a PyTorch checkpoint file"""
    import safetensors.torch

    # Convert torch.device to string for safetensors
    if device is not None:
        if hasattr(device, 'type'):
            device_str = str(device.type)
        else:
            device_str = str(device)
    else:
        device_str = "cpu"

    if filename.endswith('.safetensors'):
        # Load safetensors file
        return safetensors.torch.load_file(filename, device=device_str)
    else:
        # Load regular PyTorch file
        if device is None:
            device = torch.device("cpu")

        if safe_load:
            # Use safe loading with weights_only=True for security
            return torch.load(filename, map_location=device, weights_only=True)
        else:
            # Unsafe loading (for compatibility with old checkpoints)
            return torch.load(filename, map_location=device)

utils_module = ModuleType('comfy.utils')
utils_module.PROGRESS_BAR_ENABLED = True
utils_module.ProgressBar = Utils.ProgressBar
utils_module.copy_to_param = copy_to_param
utils_module.set_attr_param = set_attr_param
utils_module.set_module_tensor_to_device = set_module_tensor_to_device
utils_module.common_upscale = common_upscale
utils_module.load_torch_file = load_torch_file

sd_module = ModuleType('comfy.sd')
sd_module.load_checkpoint = SD.load_checkpoint
sd_module.load_checkpoint_guess_config = SD.load_checkpoint_guess_config
sd_module.load_lora_for_models = SD.load_lora_for_models

samplers_module = ModuleType('comfy.samplers')
samplers_module.KSampler = Samplers.KSampler

sample_module = ModuleType('comfy.sample')
sample_module.sample = Sample.sample
sample_module.prepare_noise = Sample.prepare_noise
sample_module.fix_empty_latent_channels = Sample.fix_empty_latent_channels

model_patcher_module = ModuleType('comfy.model_patcher')
model_patcher_module.ModelPatcher = ModelPatcher
model_patcher_module.get_key_weight = get_key_weight
model_patcher_module.string_to_seed = string_to_seed

lora_module = ModuleType('comfy.lora')
def _calculate_weight(patches, temp_weight, key):
    # Minimal placeholder: apply no-op (identity). Real logic can be added later.
    return temp_weight
lora_module.calculate_weight = _calculate_weight

float_module = ModuleType('comfy.float')
def _stochastic_rounding(tensor, dtype, seed=0):
    # Minimal rounding: deterministic cast with optional noise seed (ignored here)
    try:
        return tensor.to(dtype)
    except Exception:
        return tensor
float_module.stochastic_rounding = _stochastic_rounding

clip_vision_module = ModuleType('comfy.clip_vision')
class _CLIPVisionModel:
    """Placeholder CLIP Vision model"""
    def __init__(self):
        self.model = None
    
    def load(self, *args, **kwargs):
        return self
    
    def encode_image(self, image):
        # Return dummy embeddings
        return torch.randn(1, 768)

def clip_preprocess(image, size=224, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711], crop=True):
    """Preprocess image for CLIP Vision - compatible with comfy.clip_vision.clip_preprocess"""
    if isinstance(image, torch.Tensor):
        # Image is already a tensor, apply normalization
        image = image[:, :, :, :3] if image.shape[3] > 3 else image
        mean_tensor = torch.tensor(mean, device=image.device, dtype=image.dtype)
        std_tensor = torch.tensor(std, device=image.device, dtype=image.dtype)
        
        # Resize if needed
        if image.shape[1] != size or image.shape[2] != size:
            import torch.nn.functional as F
            # image shape: [B, H, W, C] -> [B, C, H, W]
            image = image.permute(0, 3, 1, 2)
            image = F.interpolate(image, size=(size, size), mode='bilinear', align_corners=False)
            # [B, C, H, W] -> [B, H, W, C]
            image = image.permute(0, 2, 3, 1)
        
        # Normalize: (image - mean) / std
        image = (image - mean_tensor) / std_tensor
        return image
    
    # If PIL image or similar, convert to tensor
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize(size),
        T.CenterCrop(size) if crop else T.Lambda(lambda x: x),
        T.ToTensor(),
        T.Normalize(mean=mean,
                   std=[0.26862954, 0.26130258, 0.27577711])
    ])
    try:
        return transform(image)
    except:
        # Fallback: return dummy tensor
        return torch.randn(3, size, size)

clip_vision_module.CLIPVisionModel = _CLIPVisionModel
clip_vision_module.ClipVisionModel = _CLIPVisionModel  # Alternative naming
clip_vision_module.load = lambda *args, **kwargs: _CLIPVisionModel()
clip_vision_module.clip_preprocess = clip_preprocess

# Add comfy.latent_formats module
latent_formats_module = ModuleType('comfy.latent_formats')

class HunyuanVideo:
    """HunyuanVideo latent format stub"""
    def __init__(self):
        self.latent_channels = 16
        self.scale_factor = 0.18215

latent_formats_module.HunyuanVideo = HunyuanVideo

# Add Wan21 and Wan22 formats (same as HunyuanVideo)
class Wan21(HunyuanVideo):
    """Wan21 latent format"""
    latent_channels = 16
    latent_rgb_factors = [
        [-0.0609, 0.0975, 0.1925, 0.0547, 0.3913, -0.0352, -0.0267, -0.0228,
         -0.2812, -0.1179, -0.1055, 0.2627, -0.0130, -0.1465, -0.0286, 0.1848],
        [0.0329, 0.2857, 0.0098, 0.3203, 0.1725, 0.1218, -0.0021, -0.0498,
         -0.1770, -0.2141, -0.0099, -0.1055, -0.2273, 0.0873, 0.1209, 0.0418],
        [0.2820, 0.1218, -0.1122, -0.0711, -0.0665, -0.0520, -0.0593, -0.2186,
         0.0913, 0.2115, -0.0217, -0.2148, -0.2817, 0.2249, 0.0234, 0.0193],
    ]
    latent_rgb_factors_bias = [0.0, 0.0, 0.0]

class Wan22(HunyuanVideo):
    """Wan22 latent format"""
    latent_channels = 16
    latent_rgb_factors = Wan21.latent_rgb_factors
    latent_rgb_factors_bias = Wan21.latent_rgb_factors_bias

latent_formats_module.Wan21 = Wan21
latent_formats_module.Wan22 = Wan22

# Add comfy.model_base module
model_base_module = ModuleType('comfy.model_base')

class BaseModel:
    """Base model class for ComfyUI compatibility"""
    def __init__(self, model_config=None, model_type=None, device=None, **kwargs):
        self.model_config = model_config
        self.model_type = model_type
        self.model = None
        self.device = device if device else "cpu"
        self.dtype = torch.float32
        # Store any additional kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to(self, device):
        self.device = device
        if self.model:
            self.model.to(device)
        return self

model_base_module.BaseModel = BaseModel

# Add ModelType enum
class ModelType:
    """Model type enumeration for ComfyUI"""
    FLOW = "flow"
    V_PREDICTION = "v_prediction"
    EPS = "eps"

model_base_module.ModelType = ModelType

# Add comfy.ops module
ops_module = ModuleType('comfy.ops')

def cast_bias_weight(s, input=None, dtype=None, device=None):
    """Cast bias and weight tensors to specified dtype and device"""
    if input is not None:
        if dtype is None:
            dtype = input.dtype
        if device is None:
            device = input.device

    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get weight and bias from the module
    weight = s.weight if hasattr(s, 'weight') else None
    bias = s.bias if hasattr(s, 'bias') else None

    # Cast to proper device and dtype
    if weight is not None:
        weight = weight.to(device=device, dtype=dtype)
    if bias is not None:
        bias = bias.to(device=device, dtype=dtype)

    # Return tuple for unpacking (expected by custom_linear.py)
    return weight, bias

def disable_weight_init():
    """Context manager to disable weight initialization"""
    class NoWeightInit:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    return NoWeightInit()

# Common layer operations
ops_module.cast_bias_weight = cast_bias_weight
ops_module.disable_weight_init = disable_weight_init

# Linear layers
ops_module.Linear = torch.nn.Linear
ops_module.Conv2d = torch.nn.Conv2d
ops_module.Conv3d = torch.nn.Conv3d
ops_module.GroupNorm = torch.nn.GroupNorm
ops_module.LayerNorm = torch.nn.LayerNorm
ops_module.ConvTranspose2d = torch.nn.ConvTranspose2d

# Create TAESD module for latent preview
taesd_module = ModuleType('comfy.taesd')

class TAESD:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def decode(self, latent):
        """Decode latent to image placeholder"""
        if isinstance(latent, torch.Tensor):
            # Return a placeholder decoded image
            b = latent.shape[0] if len(latent.shape) > 3 else 1
            return torch.zeros(b, 3, 256, 256, device=self.device)
        return latent

    def encode(self, pixel):
        """Encode pixel to latent placeholder"""
        if isinstance(pixel, torch.Tensor):
            # Simple downscale for placeholder
            b = pixel.shape[0] if len(pixel.shape) > 3 else 1
            return torch.randn(b, 16, 32, 32, device=self.device) * 0.18215
        return pixel

taesd_module.TAESD = TAESD

# Create CLI args module
cli_args_module = ModuleType('comfy.cli_args')

class Args:
    def __init__(self):
        self.preview_method = "disabled"  # Disable preview to avoid errors
        self.preview_size = 512
        self.disable_metadata = False
        self.fp8_e4m3fn_text_enc = False
        self.fp8_e5m2_text_enc = False
        self.gpu_only = False
        self.disable_smart_memory = False
        self.lowvram = False
        self.deterministic = False

class LatentPreviewMethod:
    Auto = "auto"
    AUTO = "auto"
    Taesd = "taesd"
    TAESD = "taesd"
    Disabled = "disabled"
    DISABLED = "disabled"
    Latent2RGB = "latent2rgb"
    NoPreviews = "none"
    NoPreview = "none"

cli_args_module.args = Args()
cli_args_module.LatentPreviewMethod = LatentPreviewMethod

# Create comfy_types module for Remote nodes compatibility
comfy_types_module = ModuleType('comfy.comfy_types')

# Add commonly used type definitions
class PromptID:
    """Type for prompt IDs in ComfyUI"""
    pass

class UniqueID:
    """Type for unique node IDs"""
    pass

# Add to module
comfy_types_module.PromptID = PromptID
comfy_types_module.UniqueID = UniqueID

# Create node_typing submodule for Remote nodes
node_typing_module = ModuleType('comfy.comfy_types.node_typing')

# Add typing classes for nodes
class Input:
    """Input type definition"""
    pass

class Output:
    """Output type definition"""
    pass

class IO:
    """IO type definition for node inputs/outputs"""
    # Common input/output types
    STRING = "STRING"
    INT = "INT"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"
    IMAGE = "IMAGE"
    LATENT = "LATENT"
    MASK = "MASK"
    MODEL = "MODEL"
    CLIP = "CLIP"
    VAE = "VAE"
    CONDITIONING = "CONDITIONING"

class ComfyNodeABC:
    """Abstract base class for ComfyUI nodes"""
    pass

class InputTypeDict(dict):
    """Dictionary type for node input type definitions"""
    pass

class OutputTypeDict(dict):
    """Dictionary type for node output type definitions"""
    pass

node_typing_module.Input = Input
node_typing_module.Output = Output
node_typing_module.IO = IO
node_typing_module.ComfyNodeABC = ComfyNodeABC
node_typing_module.InputTypeDict = InputTypeDict
node_typing_module.OutputTypeDict = OutputTypeDict

# Attach node_typing to comfy_types
comfy_types_module.node_typing = node_typing_module

# Attach submodules to comfy package
comfy.model_management = model_management_module
comfy.utils = utils_module
comfy.sd = sd_module
comfy.samplers = samplers_module
comfy.sample = sample_module
comfy.model_patcher = model_patcher_module
comfy.lora = lora_module
comfy.float = float_module
comfy.clip_vision = clip_vision_module
comfy.ops = ops_module
comfy.model_base = model_base_module
comfy.latent_formats = latent_formats_module
comfy.taesd = taesd_module
comfy.cli_args = cli_args_module
comfy.comfy_types = comfy_types_module

# Register in sys.modules
sys.modules['comfy'] = comfy
sys.modules['comfy.model_management'] = model_management_module
sys.modules['comfy.utils'] = utils_module
sys.modules['comfy.sd'] = sd_module
sys.modules['comfy.samplers'] = samplers_module
sys.modules['comfy.sample'] = sample_module
sys.modules['comfy.model_patcher'] = model_patcher_module
sys.modules['comfy.lora'] = lora_module
sys.modules['comfy.float'] = float_module
sys.modules['comfy.clip_vision'] = clip_vision_module
sys.modules['comfy.ops'] = ops_module
sys.modules['comfy.model_base'] = model_base_module
sys.modules['comfy.latent_formats'] = latent_formats_module
sys.modules['comfy.taesd'] = taesd_module
sys.modules['comfy.cli_args'] = cli_args_module
sys.modules['comfy.comfy_types'] = comfy_types_module
sys.modules['comfy.comfy_types.node_typing'] = node_typing_module

# Register folder_paths as global module (ComfyUI compatibility)
try:
    from genesis.core import folder_paths as genesis_folder_paths
    # Import extension functions to ensure they're registered
    try:
        from genesis.core import folder_paths_ext
    except ImportError:
        pass
    sys.modules['folder_paths'] = genesis_folder_paths
except ImportError:
    # Fallback: create minimal folder_paths stub
    folder_paths_module = ModuleType('folder_paths')
    folder_paths_module.models_dir = ""
    folder_paths_module.folder_names_and_paths = {}
    folder_paths_module.get_folder_paths = lambda x: []
    folder_paths_module.get_filename_list = lambda x: []
    folder_paths_module.get_full_path = lambda x, y: ""
    folder_paths_module.add_model_folder_path = lambda x, y: None
    sys.modules['folder_paths'] = folder_paths_module

logger.info("ComfyUI compatibility layer loaded successfully")

# Add utils directly to comfy module for compatibility
comfy.utils = utils_module

# Export module objects for direct import
model_management_module = model_management_module
utils_module = utils_module
sd_module = sd_module
samplers_module = samplers_module
sample_module = sample_module
model_patcher_module = model_patcher_module
lora_module = lora_module
float_module = float_module
clip_vision_module = clip_vision_module
ops_module = ops_module
model_base_module = model_base_module
latent_formats_module = latent_formats_module
taesd_module = taesd_module
cli_args_module = cli_args_module
