#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen Image UI Integration
Qwen Image UI 集成到文生图标签

Based on: custom_nodes/Comfyui/ComfyUI-QwenImageWrapper
Workflow: qwen3 edy.json (without image interrogation nodes)

Author: eddy
Date: 2025-11-16
"""

import sys
from pathlib import Path
import gradio as gr

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import Qwen pipeline
try:
    # Direct import to avoid circular dependencies
    import importlib.util
    qwen_file = project_root / "apps" / "sd_module" / "qwen_comfy_pipeline.py"
    spec = importlib.util.spec_from_file_location("qwen_comfy_pipeline", qwen_file)
    qwen_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qwen_module)
    
    QwenComfyPipeline = qwen_module.QwenComfyPipeline
    get_available_models = qwen_module.get_available_models
    QWEN_AVAILABLE = qwen_module.QWEN_AVAILABLE
except Exception as e:
    print(f"Failed to import Qwen: {e}")
    QWEN_AVAILABLE = False


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


def create_qwen_subtab():
    """Create Qwen Image UI as a subtab"""
    
    if not QWEN_AVAILABLE:
        gr.Markdown("""
        ## ⚠️ Qwen Image 不可用
        
        Qwen Image 节点未加载。请确保:
        1. custom_nodes/Comfyui/ComfyUI-QwenImageWrapper 文件夹存在
        2. 相关依赖已安装
        
        查看文档: docs/QWEN_IMAGE_GUIDE.md
        """)
        return
    
    # Create pipeline
    pipeline = QwenComfyPipeline()
    models = get_available_models()
    
    # Model choices
    unet_choices = _unique(models['unet'] + models['diffusion_models'])
    clip_choices = _unique(models.get('clip', []) + models.get('text_encoders', []))
    vae_choices = _unique(models['vae'])
    lora_choices = ["none"] + _unique(models.get('loras', []))
    
    # Sampler and scheduler options
    sampler_list = [
        "sa_solver", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a",
        "lms", "dpm_fast", "dpm_adaptive", 
        "dpmpp_2s_a", "dpmpp_2m", "dpmpp_sde",
        "ddim", "ddpm", "uni_pc", "uni_pc_bh2"
    ]
    
    scheduler_list = ["beta", "normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
    
    gr.Markdown("""
    # 🎨 Qwen Image 文生图
    
    **高质量图像生成，使用 Qwen Image 模型**
    """)
    
    with gr.Row():
        # Left column: Settings
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 模型设置")
            
            with gr.Accordion("模型选择", open=True):
                unet_model = gr.Dropdown(
                    label="UNET 模型",
                    choices=unet_choices,
                    value=unet_choices[0] if unet_choices else None
                )
                
                clip_model = gr.Dropdown(
                    label="CLIP 模型",
                    choices=clip_choices,
                    value=clip_choices[0] if clip_choices else None
                )
                
                vae_model = gr.Dropdown(
                    label="VAE 模型",
                    choices=vae_choices,
                    value=vae_choices[0] if vae_choices else None
                )
            
            gr.Markdown("### 📝 提示词")
            
            prompt = gr.Textbox(
                label="正向提示词",
                placeholder="描述你想生成的图像...",
                lines=4
            )
            
            negative_prompt = gr.Textbox(
                label="负向提示词",
                placeholder="描述你不想要的内容...",
                lines=2
            )
            
            gr.Markdown("### ⚙️ 生成参数")
            
            with gr.Row():
                width = gr.Slider(
                    label="宽度",
                    minimum=256,
                    maximum=2048,
                    value=1328,
                    step=16
                )
                
                height = gr.Slider(
                    label="高度",
                    minimum=256,
                    maximum=2048,
                    value=1328,
                    step=16
                )
            
            with gr.Row():
                steps = gr.Slider(
                    label="采样步数",
                    minimum=1,
                    maximum=100,
                    value=8,
                    step=1
                )
                
                cfg = gr.Slider(
                    label="CFG Scale",
                    minimum=0.0,
                    maximum=20.0,
                    value=2.5,
                    step=0.1
                )
            
            with gr.Row():
                sampler = gr.Dropdown(
                    label="采样器",
                    choices=sampler_list,
                    value="sa_solver"
                )
                
                scheduler = gr.Dropdown(
                    label="调度器",
                    choices=scheduler_list,
                    value="beta"
                )
            
            seed = gr.Number(
                label="种子 (-1 为随机)",
                value=-1,
                precision=0
            )
            
            quantization_dtype = gr.Dropdown(
                label="量化精度",
                choices=["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "fp16_fast", "bf16", "bf16_fast"],
                value="fp16_fast",
                info="fp8=最快+50% VRAM节省, bf16_fast=平衡2.5x速度, default=无量化"
            )
            
            with gr.Accordion("LoRA 设置", open=False):
                with gr.Row():
                    lora_1_name = gr.Dropdown(
                        label="LoRA 1",
                        choices=lora_choices,
                        value="none"
                    )
                    lora_1_strength = gr.Slider(
                        label="强度",
                        minimum=-10.0,
                        maximum=10.0,
                        value=1.0,
                        step=0.05
                    )
                
                with gr.Row():
                    lora_2_name = gr.Dropdown(
                        label="LoRA 2",
                        choices=lora_choices,
                        value="none"
                    )
                    lora_2_strength = gr.Slider(
                        label="强度",
                        minimum=-10.0,
                        maximum=10.0,
                        value=0.0,
                        step=0.05
                    )
                
                with gr.Row():
                    lora_3_name = gr.Dropdown(
                        label="LoRA 3",
                        choices=lora_choices,
                        value="none"
                    )
                    lora_3_strength = gr.Slider(
                        label="强度",
                        minimum=-10.0,
                        maximum=10.0,
                        value=0.0,
                        step=0.05
                    )
                
                with gr.Row():
                    lora_4_name = gr.Dropdown(
                        label="LoRA 4",
                        choices=lora_choices,
                        value="none"
                    )
                    lora_4_strength = gr.Slider(
                        label="强度",
                        minimum=-10.0,
                        maximum=10.0,
                        value=0.0,
                        step=0.05
                    )
            
            with gr.Accordion("优化设置", open=False):
                use_blockswap = gr.Checkbox(
                    label="启用 BlockSwap (30-60% VRAM 节省)",
                    value=True
                )
                
                with gr.Row():
                    blockswap_blocks = gr.Slider(
                        label="BlockSwap 块数",
                        minimum=1,
                        maximum=50,
                        value=20,
                        step=1
                    )
                    
                    blockswap_model_size = gr.Dropdown(
                        label="模型大小",
                        choices=["auto", "small", "medium", "large", "xl"],
                        value="auto"
                    )
                
                blockswap_use_recommended = gr.Checkbox(
                    label="使用推荐配置",
                    value=True
                )
                
                enable_matmul_optimization = gr.Checkbox(
                    label="启用矩阵乘法优化 (1.5-2x 加速)",
                    value=True
                )
                
                use_torch_compile = gr.Checkbox(
                    label="启用 Torch Compile (20-60% 加速，首次慢)",
                    value=False
                )
                
                matmul_precision = gr.Dropdown(
                    label="矩阵精度",
                    choices=["highest", "high", "medium"],
                    value="high"
                )
                
                use_autocast = gr.Checkbox(
                    label="启用混合精度 (30-50% 加速)",
                    value=False
                )
                
                autocast_dtype = gr.Dropdown(
                    label="Autocast 类型",
                    choices=["float16", "bfloat16"],
                    value="bfloat16"
                )
                
                use_channels_last = gr.Checkbox(
                    label="使用 Channels Last (10-20% 加速)",
                    value=False
                )
                
                enable_flash_attention = gr.Checkbox(
                    label="启用 Flash Attention (2-4x 加速)",
                    value=True
                )
                
                compile_mode = gr.Dropdown(
                    label="编译模式",
                    choices=["default", "reduce-overhead", "max-autotune"],
                    value="default"
                )
                
                enable_kv_cache = gr.Checkbox(
                    label="启用 KV Cache",
                    value=True
                )
            
            generate_btn = gr.Button("🎨 生成图像", variant="primary", size="lg")
        
        # Right column: Output
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ 生成结果")
            
            output_image = gr.Image(
                label="结果",
                type="pil"
            )
            
            output_info = gr.Markdown(
                value="点击生成开始..."
            )
    
    # Event handlers
    def generate_wrapper(
        prompt, negative_prompt, unet_model, clip_model, vae_model,
        width, height, steps, cfg, sampler, scheduler, seed,
        quantization_dtype,
        lora_1_name, lora_1_strength,
        lora_2_name, lora_2_strength,
        lora_3_name, lora_3_strength,
        lora_4_name, lora_4_strength,
        use_blockswap, blockswap_blocks, blockswap_model_size, blockswap_use_recommended,
        enable_matmul_optimization, use_torch_compile, matmul_precision,
        use_autocast, autocast_dtype, use_channels_last,
        enable_flash_attention, compile_mode, enable_kv_cache,
        progress=gr.Progress()
    ):
        """Generate image wrapper"""
        try:
            if not QWEN_AVAILABLE:
                return None, "❌ Qwen Image 模块未加载"
            
            if not prompt:
                return None, "❌ 请输入提示词"
            
            progress(0, desc="生成中...")
            
            # Generate
            images = pipeline.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                unet_name=unet_model,
                clip_name=clip_model,
                vae_name=vae_model,
                width=int(width),
                height=int(height),
                steps=int(steps),
                cfg=cfg,
                sampler_name=sampler,
                scheduler=scheduler,
                seed=int(seed),
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
                blockswap_blocks=int(blockswap_blocks),
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
            
            if images:
                info = f"""
## ✅ 生成完成!

**提示词:** {prompt[:100]}...

**参数:**
- 尺寸: {width} x {height}
- 步数: {steps}
- CFG: {cfg}
- 采样器: {sampler}
- 调度器: {scheduler}
- 种子: {seed if seed >= 0 else '随机'}
- 量化: {quantization_dtype}

**模型:**
- UNET: {unet_model}
- CLIP: {clip_model}
- VAE: {vae_model}
"""
                return images[0] if images else None, info
            else:
                return None, "❌ 生成失败，查看控制台了解详情"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"❌ 错误: {e}"
    
    # Connect events
    generate_btn.click(
        fn=generate_wrapper,
        inputs=[
            prompt, negative_prompt, unet_model, clip_model, vae_model,
            width, height, steps, cfg, sampler, scheduler, seed,
            quantization_dtype,
            lora_1_name, lora_1_strength,
            lora_2_name, lora_2_strength,
            lora_3_name, lora_3_strength,
            lora_4_name, lora_4_strength,
            use_blockswap, blockswap_blocks, blockswap_model_size, blockswap_use_recommended,
            enable_matmul_optimization, use_torch_compile, matmul_precision,
            use_autocast, autocast_dtype, use_channels_last,
            enable_flash_attention, compile_mode, enable_kv_cache
        ],
        outputs=[output_image, output_info]
    )


if __name__ == "__main__":
    # Test UI
    with gr.Blocks() as demo:
        create_qwen_subtab()
    
    demo.launch()
