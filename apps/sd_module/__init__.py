"""
Stable Diffusion Module
Provides SD generation interface for Genesis WebUI

Includes:
- Standard SD text-to-image
- Flux text-to-image (integrated)

Author: eddy
Date: 2025-11-14
Updated: 2025-11-16 (Added Flux integration)
"""

import sys
import torch
from pathlib import Path
import gradio as gr
from typing import Optional, List
from PIL import Image
import numpy as np
import random

# Import Flux integrated UI
try:
    from apps.sd_module.flux_integrated import create_flux_subtab
    FLUX_INTEGRATED = True
except Exception as e:
    print(f"Flux integration not available: {e}")
    FLUX_INTEGRATED = False
    def create_flux_subtab():
        gr.Markdown("## ⚠️ Flux 不可用\n\n请安装依赖: `pip install diffusers transformers accelerate`")

# Import Qwen Image integrated UI
try:
    from apps.sd_module.qwen_integrated import create_qwen_subtab
    QWEN_INTEGRATED = True
except Exception as e:
    print(f"Qwen Image integration not available: {e}")
    QWEN_INTEGRATED = False
    def create_qwen_subtab():
        gr.Markdown("## ⚠️ Qwen Image 不可用\n\n请确保 custom_nodes/Comfyui/ComfyUI-QwenImageWrapper 已安装")

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import importlib.util
folder_paths_file = project_root / "core" / "folder_paths.py"
if folder_paths_file.exists():
    spec = importlib.util.spec_from_file_location("folder_paths", folder_paths_file)
    folder_paths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(folder_paths)
else:
    class folder_paths:
        @staticmethod
        def get_full_path(folder_type, filename):
            return str(project_root / "models" / folder_type / filename)
        @staticmethod
        def get_filename_list(folder_type):
            model_dir = project_root / "models" / folder_type
            if not model_dir.exists():
                return []
            return [f.name for f in model_dir.glob("*.safetensors")] + \
                   [f.name for f in model_dir.glob("*.ckpt")]

try:
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False


class SDGenerator:
    """Stable Diffusion Generator"""

    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.current_model = None

    def load_model(self, model_name, progress=None):
        """Load SD model"""
        if not SD_AVAILABLE:
            return "Error: Diffusers library not available"

        try:
            if progress:
                progress(0.1, desc="Loading model...")

            if model_name.startswith("HF:"):
                model_path = model_name[3:]
            else:
                model_path = folder_paths.get_full_path('checkpoints', model_name)

            if self.current_model == model_path:
                return f"Model already loaded: {model_name}"

            if progress:
                progress(0.3, desc="Loading pipeline...")

            self.pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=self.dtype,
                use_safetensors=model_path.endswith('.safetensors')
            )

            if progress:
                progress(0.6, desc="Moving to device...")

            self.pipe = self.pipe.to(self.device)

            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except:
                    pass

            self.current_model = model_path

            if progress:
                progress(1.0, desc="Model loaded!")

            return f"Model loaded successfully: {model_name}"

        except Exception as e:
            return f"Error loading model: {str(e)}"

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 7.0,
        seed: int = -1,
        batch_count: int = 1,
        batch_size: int = 1,
        progress=gr.Progress()
    ) -> List[Image.Image]:
        """Generate images"""
        if not SD_AVAILABLE:
            return []

        if self.pipe is None:
            progress(0, desc="No model loaded, please load a model first")
            return []

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        all_images = []
        total_batches = batch_count
        total_steps = steps * total_batches

        for batch_idx in range(batch_count):
            current_seed = seed + batch_idx
            generator = torch.Generator(device=self.device).manual_seed(int(current_seed))

            progress(
                batch_idx / total_batches,
                desc=f"Generating batch {batch_idx + 1}/{total_batches} (seed: {current_seed})"
            )

            def callback(step, timestep, latents):
                overall_step = batch_idx * steps + step
                progress(
                    overall_step / total_steps,
                    desc=f"Batch {batch_idx + 1}/{total_batches} - Step {step}/{steps}"
                )

            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
                num_images_per_prompt=batch_size,
                callback=callback,
                callback_steps=1
            )

            all_images.extend(result.images)

        progress(1.0, desc=f"Generated {len(all_images)} images successfully!")
        return all_images


def get_available_models():
    """Get list of available models"""
    try:
        models = folder_paths.get_filename_list('checkpoints')
        hf_models = [
            "HF:runwayml/stable-diffusion-v1-5",
            "HF:stabilityai/stable-diffusion-2-1"
        ]
        return hf_models + models
    except:
        return []


def create_sd_tab():
    """Create SD generation tab with Flux integration"""
    
    with gr.Tab("文生图 (Text-to-Image)"):
        gr.Markdown("""
        # 🎨 文生图生成
        支持多种模型: Stable Diffusion, Flux
        """)
        
        with gr.Tabs():
            # SD 标签
            with gr.Tab("Stable Diffusion"):
                generator = SDGenerator()
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 模型设置")
                        model_dropdown = gr.Dropdown(
                            label="模型",
                            choices=get_available_models(),
                            value=None
                        )
                        load_btn = gr.Button("加载模型", variant="secondary")
                        model_status = gr.Textbox(
                            label="状态",
                            value="未加载模型",
                            interactive=False
                        )
                        
                        gr.Markdown("### 生成设置")
                        prompt = gr.Textbox(
                            label="正向提示词",
                            placeholder="输入提示词...",
                            lines=3
                        )
                        negative_prompt = gr.Textbox(
                            label="负向提示词",
                            value="nsfw, lowres, bad anatomy, bad hands, text, error",
                            lines=2
                        )

                        with gr.Row():
                            width = gr.Slider(256, 1024, 512, step=64, label="宽度")
                            height = gr.Slider(256, 1024, 512, step=64, label="高度")

                        with gr.Row():
                            steps = gr.Slider(1, 100, 20, step=1, label="步数")
                            cfg = gr.Slider(1, 20, 7, step=0.5, label="CFG")

                        seed = gr.Number(label="种子 (-1随机)", value=-1, precision=0)

                        with gr.Row():
                            batch_count = gr.Slider(1, 10, 1, step=1, label="批次数")
                            batch_size = gr.Slider(1, 4, 1, step=1, label="批次大小")

                        generate_btn = gr.Button("生成", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        output_gallery = gr.Gallery(
                            label="生成的图像",
                            show_label=True,
                            elem_id="gallery",
                            columns=2,
                            rows=2,
                            height="auto"
                        )

                load_btn.click(
                    fn=generator.load_model,
                    inputs=[model_dropdown],
                    outputs=[model_status]
                )

                generate_btn.click(
                    fn=generator.generate,
                    inputs=[
                        prompt, negative_prompt, width, height,
                        steps, cfg, seed, batch_count, batch_size
                    ],
                    outputs=[output_gallery]
                )
            
            # Flux 标签
            with gr.Tab("Flux"):
                create_flux_subtab()
            
            # Qwen Image 标签
            with gr.Tab("Qwen Image"):
                create_qwen_subtab()
    
    return None


def create_flux_tab():
    """Create Flux generation tab"""
    try:
        from apps.sd_module.flux_gradio_ui import FluxGradioUI
        
        flux_ui = FluxGradioUI()
        
        # Create the Flux UI components
        with gr.Tab("Flux txt2img"):
            gr.Markdown("""
            # 🎨 Flux Text-to-Image Generator
            
            **High-quality image generation using Flux models**
            
            Based on ComfyUI workflow: `flux文生图.json`
            """)
            
            with gr.Row():
                # Left column: Settings
                with gr.Column(scale=1):
                    gr.Markdown("### 🔧 Model Settings")
                    
                    with gr.Accordion("Load Models", open=True):
                        unet_selector = gr.Dropdown(
                            label="UNET Model",
                            choices=flux_ui.models['unet'],
                            value=flux_ui.models['unet'][0] if flux_ui.models['unet'] else None
                        )
                        
                        clip1_selector = gr.Dropdown(
                            label="CLIP Model 1 (T5XXL)",
                            choices=flux_ui.models['clip'],
                            value="sd3/t5xxl_fp16.safetensors" if "sd3/t5xxl_fp16.safetensors" in flux_ui.models['clip'] else (flux_ui.models['clip'][0] if flux_ui.models['clip'] else None)
                        )
                        
                        clip2_selector = gr.Dropdown(
                            label="CLIP Model 2 (CLIP-L)",
                            choices=flux_ui.models['clip'],
                            value="clip_l.safetensors" if "clip_l.safetensors" in flux_ui.models['clip'] else (flux_ui.models['clip'][0] if flux_ui.models['clip'] else None)
                        )
                        
                        vae_selector = gr.Dropdown(
                            label="VAE Model",
                            choices=flux_ui.models['vae'],
                            value="ae.sft" if "ae.sft" in flux_ui.models['vae'] else (flux_ui.models['vae'][0] if flux_ui.models['vae'] else None)
                        )
                        
                        load_btn = gr.Button("📥 Load Models", variant="secondary")
                        model_status = gr.Textbox(
                            label="Model Status",
                            value="Models not loaded",
                            interactive=False,
                            lines=3
                        )
                    
                    gr.Markdown("### 📝 Prompts")
                    
                    prompt = gr.Textbox(
                        label="Positive Prompt",
                        placeholder="Describe the image you want to generate...",
                        lines=4,
                        value="a beautiful landscape with mountains and lake, sunset, 4k, highly detailed"
                    )
                    
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="What to avoid...",
                        lines=2,
                        value="worst quality, low quality, blurry"
                    )
                    
                    gr.Markdown("### ⚙️ Generation Parameters")
                    
                    with gr.Row():
                        width = gr.Slider(
                            label="Width",
                            minimum=512,
                            maximum=2048,
                            step=64,
                            value=1080
                        )
                        
                        height = gr.Slider(
                            label="Height",
                            minimum=512,
                            maximum=2048,
                            step=64,
                            value=1920
                        )
                    
                    with gr.Row():
                        steps = gr.Slider(
                            label="Steps",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=20
                        )
                        
                        cfg = gr.Slider(
                            label="CFG Scale",
                            minimum=0.0,
                            maximum=20.0,
                            step=0.1,
                            value=1.0
                        )
                    
                    with gr.Row():
                        guidance = gr.Slider(
                            label="Flux Guidance",
                            minimum=0.0,
                            maximum=10.0,
                            step=0.1,
                            value=3.5
                        )
                        
                        seed = gr.Number(
                            label="Seed (-1 for random)",
                            value=-1,
                            precision=0
                        )
                    
                    with gr.Row():
                        sampler_name = gr.Dropdown(
                            label="Sampler",
                            choices=["dpmpp_2m", "euler", "euler_a", "heun", "dpm_2", "lms"],
                            value="dpmpp_2m"
                        )
                        
                        scheduler = gr.Dropdown(
                            label="Scheduler",
                            choices=["sgm_uniform", "normal", "karras", "exponential", "simple"],
                            value="sgm_uniform"
                        )
                    
                    gr.Markdown("### 🎭 LoRA Settings")
                    
                    with gr.Row():
                        lora1_name = gr.Dropdown(
                            label="LoRA 1",
                            choices=["None"] + flux_ui.models['loras'],
                            value="None"
                        )
                        
                        lora1_strength = gr.Slider(
                            label="Strength",
                            minimum=0.0,
                            maximum=2.0,
                            step=0.05,
                            value=0.75
                        )
                    
                    with gr.Row():
                        lora2_name = gr.Dropdown(
                            label="LoRA 2",
                            choices=["None"] + flux_ui.models['loras'],
                            value="None"
                        )
                        
                        lora2_strength = gr.Slider(
                            label="Strength",
                            minimum=0.0,
                            maximum=2.0,
                            step=0.05,
                            value=0.6
                        )
                    
                    generate_btn = gr.Button(
                        "🎨 Generate Image",
                        variant="primary",
                        size="lg"
                    )
                
                # Right column: Output
                with gr.Column(scale=1):
                    gr.Markdown("### 🖼️ Generated Image")
                    
                    output_image = gr.Image(
                        label="Result",
                        type="pil"
                    )
                    
                    output_info = gr.Markdown(
                        value="Load models and click Generate to start..."
                    )
            
            # Connect buttons
            load_btn.click(
                fn=flux_ui.load_models,
                inputs=[unet_selector, clip1_selector, clip2_selector, vae_selector],
                outputs=[model_status]
            )
            
            generate_btn.click(
                fn=flux_ui.generate_image,
                inputs=[
                    prompt, negative_prompt,
                    width, height,
                    steps, cfg, guidance, seed,
                    sampler_name, scheduler,
                    lora1_name, lora1_strength,
                    lora2_name, lora2_strength
                ],
                outputs=[output_image, output_info]
            )
        
        return True
        
    except Exception as e:
        print(f"Failed to create Flux tab: {e}")
        import traceback
        traceback.print_exc()
        return False
