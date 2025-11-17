#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Utilities for WanVideo Module
WanVideo 模块的模型工具

Author: eddy
Date: 2025-11-16
"""

import os
from pathlib import Path
from typing import List, Dict


def get_models_from_directory(base_dir: str, extensions: List[str] = None) -> List[str]:
    """
    Get all model files from directory and subdirectories
    从目录及其子目录获取所有模型文件
    
    Args:
        base_dir: Base directory path
        extensions: List of file extensions to include (e.g., ['.safetensors', '.pt'])
    
    Returns:
        List of model file paths (relative to base_dir)
    """
    if extensions is None:
        extensions = ['.safetensors', '.pt', '.pth', '.ckpt', '.bin']
    
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    models = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            # Check if file has valid extension
            if any(file.lower().endswith(ext) for ext in extensions):
                # Get relative path from base_dir
                full_path = Path(root) / file
                rel_path = full_path.relative_to(base_path)
                # Use forward slashes for consistency
                models.append(str(rel_path).replace('\\', '/'))
    
    return sorted(models)


def get_audio_encoder_models() -> List[str]:
    """
    Get audio encoder models (Wav2Vec, etc.)
    获取音频编码器模型
    
    Returns:
        List of model paths (directories containing model files)
    """
    project_root = Path(__file__).parent.parent.parent
    audio_encoders_dir = project_root / "models" / "audio_encoders"
    
    models = []
    
    if audio_encoders_dir.exists():
        # Walk through subdirectories to find model directories
        for root, dirs, files in os.walk(audio_encoders_dir):
            # Check if this directory contains model files
            has_model_files = any(
                f.endswith(('.bin', '.safetensors', '.pt', '.pth')) or 
                f in ['config.json', 'pytorch_model.bin']
                for f in files
            )
            
            if has_model_files:
                # Get relative path from audio_encoders_dir
                rel_path = Path(root).relative_to(audio_encoders_dir)
                models.append(str(rel_path).replace('\\', '/'))
    
    return sorted(models)


def get_wanvideo_models() -> Dict[str, List[str]]:
    """
    Get all WanVideo-related models
    获取所有 WanVideo 相关模型
    
    Returns:
        Dictionary with model types as keys and file lists as values
    """
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "models"
    
    model_types = {
        'diffusion_models': get_models_from_directory(
            str(models_dir / "diffusion_models"),
            ['.safetensors', '.pt']
        ),
        'unet': get_models_from_directory(
            str(models_dir / "unet"),
            ['.safetensors', '.pt']
        ),
        'vae': get_models_from_directory(
            str(models_dir / "vae"),
            ['.safetensors', '.pt']
        ),
        'text_encoders': get_models_from_directory(
            str(models_dir / "text_encoders"),
            ['.safetensors', '.pt']
        ),
        'clip': get_models_from_directory(
            str(models_dir / "clip"),
            ['.safetensors', '.pt']
        ),
        'clip_vision': get_models_from_directory(
            str(models_dir / "clip_vision"),
            ['.safetensors', '.pt']
        ),
        'audio_encoders': get_audio_encoder_models(),
    }
    
    # Merge diffusion_models and unet
    all_diffusion = list(set(model_types['diffusion_models'] + model_types['unet']))
    model_types['diffusion_models'] = sorted(all_diffusion)
    
    # Merge text_encoders and clip
    all_text_encoders = list(set(model_types['text_encoders'] + model_types['clip']))
    model_types['text_encoders'] = sorted(all_text_encoders)
    
    return model_types


def format_model_choices(models: List[str], add_none: bool = False) -> List[str]:
    """
    Format model list for Gradio dropdown
    格式化模型列表用于 Gradio 下拉框
    
    Args:
        models: List of model paths
        add_none: Whether to add "None" option
    
    Returns:
        Formatted list for dropdown
    """
    if not models:
        return ["无可用模型 / No models available"]
    
    choices = models.copy()
    if add_none:
        choices.insert(0, "None")
    
    return choices


# WanVideo scheduler list (from WanVideoWrapper)
WANVIDEO_SCHEDULERS = [
    "unipc",
    "unipc/beta",
    "dpm++",
    "dpm++/beta",
    "dpm++_sde",
    "dpm++_sde/beta",
    "euler",
    "euler/beta",
    "deis",
    "lcm",
    "lcm/beta",
    "res_multistep",
    "flowmatch_causvid",
    "flowmatch_distill",
    "flowmatch_pusa",
    "flowmatch_lowstep_d",
    "flowmatch_sa_ode_stable",
    "sa_ode_stable/lowstep",
    "ode/+",
    "humo_lcm",
    "multitalk",
    "iching/wuxing",
    "iching/wuxing-strong",
    "iching/wuxing-stable",
    "iching/wuxing-smooth",
    "iching/wuxing-clean",
    "iching/wuxing-sharp",
    "iching/wuxing-lowstep",
    "rcm"
]


# Common samplers (fallback if WanVideo not available)
COMMON_SAMPLERS = [
    "euler",
    "euler_a",
    "heun",
    "dpm_2",
    "dpm_2_a",
    "lms",
    "dpmpp_2m",
    "dpmpp_sde",
    "dpmpp_2m_sde",
    "dpmpp_3m_sde",
]
