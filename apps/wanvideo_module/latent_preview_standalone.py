"""
Standalone Latent Preview for Infinite Talk Pipeline
Independent version that doesn't require genesis/server modules

Author: eddy
Date: 2025-11-17
"""

import torch
from PIL import Image
from comfy.cli_args import args, LatentPreviewMethod
import comfy.model_management
import comfy.utils


MAX_PREVIEW_RESOLUTION = args.preview_size


def preview_to_image(latent_image):
    """Convert latent tensor to PIL Image"""
    latents_ubyte = (((latent_image + 1.0) / 2.0).clamp(0, 1)  # change scale from -1..1 to 0..1
                        .mul(0xFF)  # to 0..255
                        )
    if comfy.model_management.directml_enabled:
        latents_ubyte = latents_ubyte.to(dtype=torch.uint8)
    latents_ubyte = latents_ubyte.to(
        device="cpu", 
        dtype=torch.uint8, 
        non_blocking=comfy.model_management.device_supports_non_blocking(latent_image.device)
    )
    return Image.fromarray(latents_ubyte.numpy())


class LatentPreviewer:
    """Base class for latent previewers"""
    def decode_latent_to_preview(self, x0):
        pass

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return ("JPEG", preview_image, MAX_PREVIEW_RESOLUTION)


class Latent2RGBPreviewer(LatentPreviewer):
    """RGB-based latent previewer (doesn't require TAESD model)"""
    def __init__(self, latent_rgb_factors, latent_rgb_factors_bias=None):
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu").transpose(0, 1)
        self.latent_rgb_factors_bias = None
        if latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device="cpu")

    def decode_latent_to_preview(self, x0):
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        if self.latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)

        if x0.ndim == 5:
            x0 = x0[0, :, 0]
        else:
            x0 = x0[0]

        latent_image = torch.nn.functional.linear(
            x0.movedim(0, -1), 
            self.latent_rgb_factors, 
            bias=self.latent_rgb_factors_bias
        )
        return preview_to_image(latent_image)


def get_previewer(device, latent_format):
    """
    Get a latent previewer (simplified version without TAESD)
    Always uses Latent2RGB to avoid model loading
    """
    previewer = None
    method = args.preview_method
    
    if method != LatentPreviewMethod.NoPreviews:
        # Always use Latent2RGB for simplicity
        if latent_format.latent_rgb_factors is not None:
            previewer = Latent2RGBPreviewer(
                latent_format.latent_rgb_factors, 
                latent_format.latent_rgb_factors_bias
            )
    
    return previewer


def prepare_callback(model, steps, x0_output_dict=None):
    """
    Prepare callback function for sampling progress
    Standalone version that doesn't require server module
    
    Args:
        model: Model patcher object
        steps: Total number of sampling steps
        x0_output_dict: Optional dict to store x0 output
        
    Returns:
        Callback function for sampling loop
    """
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    # Disable previewer to avoid dimension mismatch errors in MultiTalk mode
    # MultiTalk has custom sampling loop that may not be compatible with standard preview
    previewer = None
    
    # Create progress bar if steps provided
    pbar = None
    if steps is not None:
        pbar = comfy.utils.ProgressBar(steps)
    
    def callback(step, x0, x, total_steps):
        """
        Callback function called during sampling
        
        Args:
            step: Current step number
            x0: Denoised latent at current step
            x: Noisy latent at current step
            total_steps: Total number of steps
        """
        # Store x0 if output dict provided
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0

        # Update progress bar if available
        if pbar is not None and step is not None:
            # Simple progress update without preview
            pbar.update_absolute(step + 1, total_steps)
    
    return callback
