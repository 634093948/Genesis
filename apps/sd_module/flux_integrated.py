#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flux Integrated UI
Flux UI integrated into main WebUI

Author: eddy
Date: 2025-11-16
"""

import sys
from pathlib import Path
import gradio as gr

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import ComfyUI pipeline
try:
    # Direct import to avoid circular dependencies
    import importlib.util
    flux_file = project_root / "apps" / "sd_module" / "flux_comfy_pipeline.py"
    spec = importlib.util.spec_from_file_location("flux_comfy_pipeline", flux_file)
    flux_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flux_module)
    
    FluxComfyPipeline = flux_module.FluxComfyPipeline
    get_available_models = flux_module.get_available_models
    COMFY_AVAILABLE = flux_module.COMFY_AVAILABLE
except Exception as e:
    print(f"Failed to import Flux: {e}")
    COMFY_AVAILABLE = False


def _unique(seq):
    """Keep list order while removing duplicates"""
    seen = set()
    result = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def create_flux_subtab():
    """Create Flux UI as a subtab"""
    
    if not COMFY_AVAILABLE:
        gr.Markdown("""
        ## ⚠️ Flux 不可用
        
        ComfyUI 模块未加载。请确保:
        1. custom_nodes/Comfyui 文件夹存在
        2. compat 文件夹中有 ComfyUI 兼容层
        3. 相关依赖已安装
        
        查看文档: docs/FLUX_MODEL_GUIDE.md
        """)
        return
    
    # Create pipeline
    pipeline = FluxComfyPipeline()
    models = get_available_models()
    local_model_choices = ["无"] + _unique(models['unet'] + models['diffusion_models'])
    vae_choices = ["无"] + _unique(models['vae'])
    clip_choices = _unique(models.get('clip', []) + models.get('text_encoders', []))
    
    # 默认CLIP模型
    default_clip1 = "sd3/t5xxl_fp16.safetensors" if "sd3/t5xxl_fp16.safetensors" in clip_choices else (clip_choices[0] if clip_choices else None)
    default_clip2 = "clip_l.safetensors" if "clip_l.safetensors" in clip_choices else (clip_choices[1] if len(clip_choices) > 1 else clip_choices[0] if clip_choices else None)
    
    # Sampler and scheduler options
    # KSampler基础采样器
    ksampler_list = [
        "euler", "euler_a", "heun", "dpm_2", "dpm_2_a",
        "lms", "dpm_fast", "dpm_adaptive", 
        "dpmpp_2s_a", "dpmpp_2m", "dpmpp_sde",
        "ddim", "ddpm", "uni_pc", "uni_pc_bh2"
    ]
    
    # RES4LYF高级采样器
    res4lyf_samplers = [
        # Multistep
        "res_2m", "res_3m",
        "dpmpp_2m", "dpmpp_3m",
        "abnorsett_2m", "abnorsett_3m", "abnorsett_4m",
        "deis_2m", "deis_3m", "deis_4m",
        # Exponential
        "res_2s", "res_2s_stable", "res_3s", "res_3s_alt",
        "res_4s_krogstad", "res_5s", "res_6s", "res_8s",
        "etdrk2_2s", "etdrk3_a_3s", "etdrk4_4s",
        "dpmpp_2s", "dpmpp_sde_2s", "dpmpp_3s",
        # Linear
        "ralston_2s", "ralston_3s", "ralston_4s",
        "midpoint_2s", "heun_2s", "heun_3s",
        "rk4_4s", "rk5_7s", "rk6_7s",
        "dormand-prince_6s", "dormand-prince_13s"
    ]
    
    # 合并所有采样器
    sampler_list = ksampler_list + res4lyf_samplers
    
    scheduler_list = [
        "normal", "karras", "exponential", "sgm_uniform",
        "simple", "ddim_uniform", "beta_linear", "beta_cosine"
    ]
    
    gr.Markdown("""
    ## 🎨 Flux 文生图
    
    **高质量图像生成，使用 Flux 模型**
    """)
    
    with gr.Row():
        # Left column: Settings
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 模型设置")
            
            with gr.Accordion("加载模型", open=True):
                gr.Markdown("**使用 ComfyUI 格式 (UNET + Dual CLIP)**")
                
                with gr.Group() as local_group:
                    local_model = gr.Dropdown(
                        label="UNET 模型",
                        choices=local_model_choices,
                        value="无"
                    )
                    
                    with gr.Row():
                        clip_model1 = gr.Dropdown(
                            label="CLIP 1 (T5XXL)",
                            choices=clip_choices,
                            value=default_clip1
                        )
                        
                        clip_model2 = gr.Dropdown(
                            label="CLIP 2 (CLIP-L)",
                            choices=clip_choices,
                            value=default_clip2
                        )
                    
                    vae_model = gr.Dropdown(
                        label="VAE",
                        choices=vae_choices,
                        value="无"
                    )
                
                load_btn = gr.Button("📥 加载模型", variant="primary")
                model_status = gr.Textbox(
                    label="状态",
                    value="未加载模型",
                    interactive=False,
                    lines=3
                )
            
            gr.Markdown("### 📝 提示词")
            
            prompt = gr.Textbox(
                label="提示词",
                placeholder="描述你想生成的图像...",
                lines=4,
                value="a beautiful landscape with mountains and lake, sunset, highly detailed, 4k"
            )
            
            negative_prompt = gr.Textbox(
                label="负向提示词 (Flux 不使用)",
                placeholder="Flux 模型不使用负向提示词",
                lines=2,
                value=""
            )
            
            gr.Markdown("### ⚙️ 生成设置")
            
            with gr.Row():
                width = gr.Slider(
                    label="宽度",
                    minimum=256,
                    maximum=2048,
                    step=64,
                    value=1024
                )
                
                height = gr.Slider(
                    label="高度",
                    minimum=256,
                    maximum=2048,
                    step=64,
                    value=1024
                )
            
            with gr.Row():
                steps = gr.Slider(
                    label="步数",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=28
                )
                
                guidance = gr.Slider(
                    label="引导强度 (CFG)",
                    minimum=0.0,
                    maximum=20.0,
                    step=0.1,
                    value=3.5
                )
            
            with gr.Accordion("高级采样设置", open=False):
                with gr.Row():
                    sampler = gr.Dropdown(
                        label="采样器",
                        choices=sampler_list,
                        value="euler"
                    )
                    
                    scheduler = gr.Dropdown(
                        label="调度器",
                        choices=scheduler_list,
                        value="normal"
                    )
                
                denoise = gr.Slider(
                    label="重绘幅度 (Denoise)",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=1.0,
                    info="1.0 = 完全重绘, 0.0 = 不重绘"
                )
            
            with gr.Row():
                seed = gr.Number(
                    label="种子 (-1随机)",
                    value=-1,
                    precision=0
                )
                
                num_images = gr.Slider(
                    label="生成数量",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=1
                )
            
            generate_btn = gr.Button(
                "🎨 生成图像",
                variant="primary",
                size="lg"
            )
        
        # Right column: Output
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ 生成结果")
            
            output_image = gr.Image(
                label="结果",
                type="pil"
            )
            
            output_info = gr.Markdown(
                value="加载模型后点击生成开始..."
            )
    
    # Event handlers
    def load_model_wrapper(local_model, clip_model1, clip_model2, vae_model, progress=gr.Progress()):
        """Load model wrapper"""
        try:
            if not COMFY_AVAILABLE:
                return "❌ ComfyUI 模块未加载"
            
            progress(0.1, desc="加载模型...")
            
            if not local_model or local_model == "无":
                return "❌ 请选择 UNET 模型"
            
            if not clip_model1 or not clip_model2:
                return "❌ 请选择两个 CLIP 模型 (T5XXL + CLIP-L)"
            
            # Get VAE name
            vae_name = vae_model if vae_model and vae_model != "无" else None
            
            progress(0.5, desc="加载 ComfyUI 模型...")
            success = pipeline.load_models(
                unet_name=local_model,
                clip_name1=clip_model1,
                clip_name2=clip_model2,
                vae_name=vae_name,
                weight_dtype="default"
            )
            
            progress(1.0, desc="完成!")
            
            if success:
                return f"""✅ 模型加载成功!

UNET: {local_model}
CLIP 1: {clip_model1}
CLIP 2: {clip_model2}
VAE: {vae_name if vae_name else '默认'}

设备: {pipeline.device}
格式: ComfyUI UNET"""
            else:
                return "❌ 模型加载失败，查看控制台了解详情"
                
        except Exception as e:
            err_msg = str(e)
            if "attempted relative import beyond top-level package" in err_msg:
                return """❌ 模型格式不兼容

当前选择的模型是 ComfyUI 格式，本项目使用 diffusers 库，不支持该格式。

📋 解决方案：

1️⃣ 使用 HuggingFace 模型（推荐）
   - 切换到 "HuggingFace" 选项
   - 选择: black-forest-labs/FLUX.1-schnell
   - 首次使用会自动下载

2️⃣ 查看详细说明
   - 打开: docs/FLUX_MODEL_GUIDE.md
   - 了解模型格式和转换方法

💡 提示：FLUX.1-schnell 是快速版本，无需登录，推荐新手使用。
"""
            return f"❌ 错误: {err_msg}"
    
    def generate_wrapper(
        prompt, negative_prompt, width, height,
        steps, guidance, sampler, scheduler, denoise,
        seed, num_images,
        progress=gr.Progress()
    ):
        """Generate image wrapper"""
        try:
            if not COMFY_AVAILABLE:
                return None, "❌ ComfyUI 模块未加载"
            
            if pipeline.model is None:
                return None, "❌ 请先加载模型"
            
            if not prompt:
                return None, "❌ 请输入提示词"
            
            progress(0, desc="生成中...")
            
            # Generate (只生成一张，因为ComfyUI的批量生成需要修改latent)
            images = pipeline.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=1.0,  # Flux通常使用1.0
                guidance=guidance,
                sampler_name=sampler,
                scheduler=scheduler,
                seed=seed if seed >= 0 else -1,
                denoise=denoise
            )
            
            if images:
                info = f"""
## ✅ 生成完成!

**提示词:** {prompt[:100]}...

**参数:**
- 尺寸: {width} x {height}
- 步数: {steps}
- 引导: {guidance}
- 种子: {seed if seed >= 0 else '随机'}
- 数量: {len(images)}

**模型:** {pipeline.current_unet}
"""
                return images[0] if images else None, info
            else:
                return None, "❌ 生成失败，查看控制台了解详情"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"❌ 错误: {e}"
    
    # Connect events
    load_btn.click(
        fn=load_model_wrapper,
        inputs=[local_model, clip_model1, clip_model2, vae_model],
        outputs=[model_status]
    )
    
    generate_btn.click(
        fn=generate_wrapper,
        inputs=[
            prompt, negative_prompt,
            width, height,
            steps, guidance,
            sampler, scheduler, denoise,
            seed, num_images
        ],
        outputs=[output_image, output_info]
    )
