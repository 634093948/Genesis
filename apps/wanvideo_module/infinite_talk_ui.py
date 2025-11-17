#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Infinite Talk UI Integration
Infinite Talk UI 集成

Based on: Infinite Talk test(1).json workflow
功能: 图像 + 音频 -> 说话视频

Author: eddy
Date: 2025-11-16
"""

import sys
from pathlib import Path
import gradio as gr

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import model utilities
from .model_utils import get_wanvideo_models, format_model_choices, WANVIDEO_SCHEDULERS
from .optimization_settings import create_optimization_settings

# Import Infinite Talk pipeline
try:
    import importlib.util
    pipeline_file = project_root / "apps" / "wanvideo_module" / "infinite_talk_pipeline.py"
    spec = importlib.util.spec_from_file_location("infinite_talk_pipeline", pipeline_file)
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)
    
    InfiniteTalkPipeline = pipeline_module.InfiniteTalkPipeline
    WANVIDEO_AVAILABLE = pipeline_module.WANVIDEO_AVAILABLE
except Exception as e:
    print(f"Failed to import Infinite Talk pipeline: {e}")
    WANVIDEO_AVAILABLE = False


def create_infinite_talk_tab():
    """Create Infinite Talk UI tab"""
    
    if not WANVIDEO_AVAILABLE:
        gr.Markdown("""
        ## ⚠️ Infinite Talk 暂时不可用
        
        ### 原因
        WanVideo 节点依赖 Triton，但 Windows 上 Triton 不可用。
        
        ### 解决方案
        
        **推荐：使用完整 ComfyUI 环境**
        
        Infinite Talk 工作流已在以下环境中可用：
        ```
        E:\\liliyuanshangmie\\Fuxkcomfy_lris_kernel_gen2-4_speed_safe\\FuxkComfy\\
        ```
        
        该环境已正确配置所有依赖。
        
        **工作流文件:**
        ```
        E:\\liliyuanshangmie\\Fuxkcomfy_lris_kernel_gen2-4_speed_safe\\FuxkComfy\\user\\default\\workflows\\Infinite Talk test(1).json
        ```
        
        ### 已完成的工作
        - ✅ 11 个 custom_nodes 包已复制
        - ✅ 管道代码已创建
        - ✅ UI 代码已创建
        - ✅ 工作流已分析
        
        ### 技术细节
        查看文档: `docs/INFINITE_TALK_STATUS.md`
        
        ---
        
        **提示**: 其他功能（Flux, Qwen Image, WanVideo Generation）仍然可用！
        """)
        return
    
    # Create pipeline
    pipeline = InfiniteTalkPipeline()
    
    # Get available models
    available_models = get_wanvideo_models()
    
    gr.Markdown("""
    # 🎤 Infinite Talk - 说话视频生成
    
    **从图像和音频生成说话视频（MultiTalk）**
    
    基于 ComfyUI Infinite Talk 工作流
    """)
    
    with gr.Row():
        # Left column: Settings
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 模型设置")
            
            with gr.Accordion("加载模型", open=True):
                model_name = gr.Dropdown(
                    label="WanVideo 模型",
                    choices=format_model_choices(available_models['diffusion_models']),
                    value=available_models['diffusion_models'][0] if available_models['diffusion_models'] else "wan2_1_dit.safetensors",
                    allow_custom_value=True,
                    info="从 models/diffusion_models 或 models/unet 扫描"
                )
                
                vae_name = gr.Dropdown(
                    label="VAE 模型",
                    choices=format_model_choices(available_models['vae']),
                    value=available_models['vae'][0] if available_models['vae'] else "Wan2_1_VAE_bf16.safetensors",
                    allow_custom_value=True,
                    info="从 models/vae 扫描"
                )
                
                t5_model = gr.Dropdown(
                    label="T5 文本编码器",
                    choices=format_model_choices(available_models['text_encoders']) + ["google/umt5-xxl"],
                    value="google/umt5-xxl",
                    allow_custom_value=True,
                    info="从 models/text_encoders 或 models/clip 扫描，或使用 HuggingFace 模型名"
                )
                
                clip_vision = gr.Dropdown(
                    label="CLIP Vision 模型",
                    choices=format_model_choices(available_models['clip_vision']),
                    value=available_models['clip_vision'][0] if available_models['clip_vision'] else "clip_vision_g.safetensors",
                    allow_custom_value=True,
                    info="从 models/clip_vision 扫描"
                )
                
                # Build Wav2Vec choices: local models + HuggingFace models
                local_audio_models = available_models.get('audio_encoders', [])
                huggingface_models = [
                    "facebook/wav2vec2-base-960h",
                    "facebook/wav2vec2-large-960h",
                    "facebook/wav2vec2-large-960h-lv60-self"
                ]
                wav2vec_choices = local_audio_models + huggingface_models
                
                # Set default value
                if local_audio_models:
                    wav2vec_default = local_audio_models[0]
                else:
                    wav2vec_default = "facebook/wav2vec2-base-960h"
                
                wav2vec_model = gr.Dropdown(
                    label="Wav2Vec 模型",
                    choices=wav2vec_choices if wav2vec_choices else ["facebook/wav2vec2-base-960h"],
                    value=wav2vec_default,
                    allow_custom_value=True,
                    info="音频编码器模型（本地: models/audio_encoders，或 HuggingFace 模型名）"
                )
                
                load_btn = gr.Button("📥 加载模型", variant="secondary")
                model_status = gr.Textbox(
                    label="模型状态",
                    value="未加载模型",
                    interactive=False,
                    lines=3
                )
            
            gr.Markdown("### 🔧 模型加载高级参数")
            
            with gr.Row():
                model_quantization = gr.Dropdown(
                    label="模型量化",
                    choices=[
                        "disabled",
                        "fp8_e4m3fn",
                        "fp8_e4m3fn_fast",
                        "fp8_e4m3fn_scaled",
                        "fp8_e5m2",
                        "fp8_e5m2_fast",
                        "fp8_e5m2_scaled",
                        "fp4_experimental",
                        "fp4_scaled",
                        "fp4_scaled_fast"
                    ],
                    value="fp4_scaled",
                    info="模型量化方式（与ComfyUI节点完全匹配）"
                )
                model_attention = gr.Dropdown(
                    label="注意力模式",
                    choices=["default", "sageattn", "sageattn_3", "sageattn_3_fp4"],
                    value="sageattn_3_fp4",
                    info="注意力计算模式（sageattn_3_fp4 配合 fp4_scaled）"
                )
            
            with gr.Row():
                vae_precision = gr.Dropdown(
                    label="VAE 精度",
                    choices=["fp32", "fp16", "bf16"],
                    value="bf16",
                    info="VAE 模型精度"
                )
                model_precision = gr.Dropdown(
                    label="模型精度",
                    choices=["fp32", "fp16", "bf16"],
                    value="bf16",
                    info="主模型基础精度"
                )
            
            gr.Markdown("### 📁 输入文件")
            
            image_input = gr.Image(
                label="输入图像",
                type="filepath",
                sources=["upload"]
            )
            
            audio_input = gr.Audio(
                label="输入音频",
                type="filepath",
                sources=["upload"]
            )
            
            gr.Markdown("### 📝 提示词")
            
            prompt = gr.Textbox(
                label="正向提示词",
                placeholder="描述视频内容...",
                lines=3
            )
            
            negative_prompt = gr.Textbox(
                label="负向提示词",
                placeholder="描述不想要的内容...",
                lines=2,
                value="worst quality, low quality, blurry, distorted"
            )
            
            gr.Markdown("### ⚙️ 生成参数")
            
            with gr.Row():
                width = gr.Slider(
                    label="宽度",
                    minimum=256,
                    maximum=1024,
                    value=768,
                    step=64
                )
                
                height = gr.Slider(
                    label="高度",
                    minimum=256,
                    maximum=1024,
                    value=768,
                    step=64
                )
            
            with gr.Row():
                video_length = gr.Slider(
                    label="视频长度（帧数）",
                    minimum=1,
                    maximum=200,
                    value=49,
                    step=1
                )
                
                fps = gr.Slider(
                    label="帧率 (FPS)",
                    minimum=1,
                    maximum=60,
                    value=8,
                    step=1
                )
            
            with gr.Row():
                steps = gr.Slider(
                    label="采样步数",
                    minimum=1,
                    maximum=100,
                    value=30,
                    step=1,
                    info="注意：MultiTalk 模式固定使用 4 步采样 [1000, 750, 500, 250]，此参数仅用于其他模式"
                )
                
                cfg = gr.Slider(
                    label="CFG Scale",
                    minimum=0.0,
                    maximum=20.0,
                    value=7.0,
                    step=0.5
                )
            
            with gr.Row():
                sampler = gr.Dropdown(
                    label="采样器 (Sampler)",
                    choices=["euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "lms", "dpmpp_2m", "dpmpp_sde"],
                    value="euler",
                    info="基础采样器（某些调度器会忽略此项）"
                )
                
                scheduler = gr.Dropdown(
                    label="调度器 (Scheduler)",
                    choices=WANVIDEO_SCHEDULERS,
                    value="multitalk",
                    info="WanVideo 专用调度器，推荐 multitalk"
                )
            
            with gr.Row():
                shift = gr.Slider(
                    label="Shift 参数",
                    info="时间步偏移量，影响生成质量",
                    minimum=0.0,
                    maximum=10.0,
                    value=1.0,
                    step=0.1
                )
            
            seed = gr.Number(
                label="种子 (-1 为随机)",
                value=-1,
                precision=0
            )
            
            gr.Markdown("### 🎵 音频参数")
            
            with gr.Row():
                audio_num_frames = gr.Slider(
                    label="音频帧数 (num_frames)",
                    info="用于计算音频时长，工作流默认33",
                    minimum=1,
                    maximum=200,
                    value=33,
                    step=1
                )
                
                normalize_loudness = gr.Checkbox(
                    label="归一化音量",
                    value=True
                )
            
            with gr.Row():
                audio_scale = gr.Slider(
                    label="音频强度 (audio_scale)",
                    info="音频条件强度",
                    minimum=0.0,
                    maximum=10.0,
                    value=1.0,
                    step=0.1
                )
                
                audio_cfg_scale = gr.Slider(
                    label="音频CFG (audio_cfg_scale)",
                    info="音频CFG缩放",
                    minimum=0.0,
                    maximum=10.0,
                    value=1.0,
                    step=0.1
                )
            
            gr.Markdown("### 🎬 视频生成参数")
            
            with gr.Row():
                motion_frame = gr.Slider(
                    label="运动帧 (motion_frame)",
                    info="重叠帧长度",
                    minimum=1,
                    maximum=100,
                    value=25,
                    step=1
                )
                
                colormatch = gr.Dropdown(
                    label="颜色匹配 (colormatch)",
                    choices=["disabled", "mkl", "hm", "reinhard", "mvgd", "hm-mvgd-hm", "hm-mkl-hm"],
                    value="mkl",
                    info="窗口间颜色匹配方法"
                )
            
            gr.Markdown("### 🖼️ 图像预处理")
            
            with gr.Row():
                use_image_resize = gr.Checkbox(
                    label="启用图像缩放",
                    value=True,
                    info="使用 ImageResizeKJ 进行高质量缩放"
                )
                resize_interpolation = gr.Dropdown(
                    label="插值方法",
                    choices=["lanczos", "bicubic", "bilinear", "nearest"],
                    value="lanczos",
                    info="缩放插值算法"
                )
            
            with gr.Row():
                resize_method = gr.Dropdown(
                    label="缩放方法",
                    choices=["stretch", "keep proportion", "fill / crop", "pad"],
                    value="stretch",
                    info="如何处理宽高比"
                )
                resize_condition = gr.Dropdown(
                    label="缩放条件",
                    choices=["always", "downscale if bigger", "upscale if smaller", "if bigger area", "if smaller area"],
                    value="always",
                    info="何时执行缩放"
                )
            
            gr.Markdown("### 🎵 音频预处理")
            
            with gr.Row():
                enable_audio_crop = gr.Checkbox(
                    label="启用音频裁剪",
                    value=False,
                    info="裁剪音频到指定时间段"
                )
                audio_start_time = gr.Slider(
                    label="开始时间 (秒)",
                    minimum=0,
                    maximum=60,
                    value=0,
                    step=0.1
                )
            
            with gr.Row():
                audio_crop_duration = gr.Slider(
                    label="裁剪时长 (秒)",
                    minimum=0,
                    maximum=60,
                    value=0,
                    step=0.1,
                    info="0 表示到音频结尾"
                )
                enable_audio_separation = gr.Checkbox(
                    label="启用音频分离",
                    value=False,
                    info="分离人声和背景音（仅保留人声）"
                )
            
            with gr.Row():
                separation_model = gr.Dropdown(
                    label="分离模型",
                    choices=["UVR-MDX-NET-Inst_HQ_3", "UVR_MDXNET_KARA_2", "Kim_Vocal_2"],
                    value="UVR-MDX-NET-Inst_HQ_3",
                    info="音频分离使用的模型"
                )
            
            gr.Markdown("### 🔢 自动参数计算")
            
            with gr.Row():
                auto_calculate_frames = gr.Checkbox(
                    label="根据音频时长自动计算帧数",
                    value=True,
                    info="使用音频时长 × FPS 自动计算视频帧数"
                )
                max_frames = gr.Slider(
                    label="最大帧数限制",
                    minimum=1,
                    maximum=500,
                    value=200,
                    step=1,
                    info="自动计算时的上限"
                )
            
            # Use shared optimization settings
            opt_components = create_optimization_settings(default_blocks=20, show_vae_blocks=True)
            
            generate_btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
        
        # Right column: Output
        with gr.Column(scale=1):
            gr.Markdown("### 🎬 生成结果")
            
            output_video = gr.Video(
                label="输出视频"
            )
            
            output_info = gr.Markdown(
                value="加载模型并上传文件后点击生成..."
            )
    
    # Event handlers
    def load_models_wrapper(model_name, vae_name, t5_model, clip_vision, wav2vec_model,
                           model_quantization, model_attention, vae_precision, model_precision):
        """Load models wrapper"""
        try:
            if not WANVIDEO_AVAILABLE:
                return "❌ WanVideo 模块未加载"
            
            success = pipeline.load_models(
                model_name=model_name,
                vae_name=vae_name,
                t5_model_name=t5_model,
                clip_vision_name=clip_vision,
                wav2vec_model_name=wav2vec_model,
                model_quantization=model_quantization,
                model_attention=model_attention,
                vae_precision=vae_precision,
                model_precision=model_precision
            )
            
            if success:
                return f"""✅ 模型加载成功!

**已加载:**
- WanVideo: {model_name}
- VAE: {vae_name}
- T5: {t5_model}
- CLIP Vision: {clip_vision}
- Wav2Vec: {wav2vec_model}

现在可以生成视频了！
"""
            else:
                return "❌ 模型加载失败，查看控制台了解详情"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ 错误: {e}"
    
    def generate_wrapper(
        image_path, audio_path, prompt, negative_prompt,
        width, height, video_length, steps, cfg,
        sampler, scheduler, shift, seed, fps,
        audio_num_frames, normalize_loudness, audio_scale, audio_cfg_scale,
        motion_frame, colormatch,
        use_image_resize, resize_interpolation, resize_method, resize_condition,
        enable_audio_crop, audio_start_time, audio_crop_duration,
        enable_audio_separation, separation_model,
        auto_calculate_frames, max_frames,
        blocks_to_swap, vae_blocks_to_swap,
        enable_cuda_optimization, enable_dram_optimization,
        auto_hardware_tuning, vram_threshold_percent,
        num_cuda_streams, bandwidth_target,
        offload_txt_emb, offload_img_emb, debug_mode,
        progress=gr.Progress()
    ):
        """Generate video wrapper"""
        try:
            if not WANVIDEO_AVAILABLE:
                return None, "❌ WanVideo 模块未加载"
            
            if not image_path:
                return None, "❌ 请上传图像"
            
            if not audio_path:
                return None, "❌ 请上传音频"
            
            progress(0, desc="生成中...")
            
            # Build optimization args
            optimization_args = {
                'blocks_to_swap': int(blocks_to_swap),
                'vae_blocks_to_swap': int(vae_blocks_to_swap),
                'enable_cuda_optimization': enable_cuda_optimization,
                'enable_dram_optimization': enable_dram_optimization,
                'auto_hardware_tuning': auto_hardware_tuning,
                'vram_threshold_percent': vram_threshold_percent,
                'num_cuda_streams': int(num_cuda_streams),
                'bandwidth_target': bandwidth_target,
                'offload_txt_emb': offload_txt_emb,
                'offload_img_emb': offload_img_emb,
                'debug_mode': debug_mode,
            }
            
            # Generate
            output_path = pipeline.generate(
                image_path=image_path,
                audio_path=audio_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(width),
                height=int(height),
                video_length=int(video_length),
                steps=int(steps),
                cfg=cfg,
                sampler_name=sampler,
                scheduler=scheduler,
                shift=shift,
                seed=int(seed),
                fps=int(fps),
                audio_num_frames=int(audio_num_frames),
                audio_scale=audio_scale,
                audio_cfg_scale=audio_cfg_scale,
                normalize_loudness=normalize_loudness,
                motion_frame=int(motion_frame),
                colormatch=colormatch,
                use_image_resize=use_image_resize,
                resize_interpolation=resize_interpolation,
                resize_method=resize_method,
                resize_condition=resize_condition,
                enable_audio_crop=enable_audio_crop,
                audio_start_time=audio_start_time,
                audio_crop_duration=audio_crop_duration,
                enable_audio_separation=enable_audio_separation,
                separation_model=separation_model,
                auto_calculate_frames=auto_calculate_frames,
                max_frames=int(max_frames),
                optimization_args=optimization_args
            )
            
            if output_path:
                info = f"""
## ✅ 视频生成完成!

**参数:**
- 尺寸: {width} x {height}
- 帧数: {video_length}
- FPS: {fps}
- 步数: {steps}
- CFG: {cfg}
- 采样器: {sampler}
- 调度器: {scheduler}
- Shift: {shift}
- 种子: {seed if seed >= 0 else '随机'}
- BlockSwap: {blocks_to_swap} 块

**输出:** {output_path}
"""
                return output_path, info
            else:
                return None, "❌ 生成失败，查看控制台了解详情"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"❌ 错误: {e}"
    
    # Connect events
    load_btn.click(
        fn=load_models_wrapper,
        inputs=[model_name, vae_name, t5_model, clip_vision, wav2vec_model,
                model_quantization, model_attention, vae_precision, model_precision],
        outputs=[model_status]
    )
    
    generate_btn.click(
        fn=generate_wrapper,
        inputs=[
            image_input, audio_input, prompt, negative_prompt,
            width, height, video_length, steps, cfg,
            sampler, scheduler, shift, seed, fps,
            audio_num_frames, normalize_loudness, audio_scale, audio_cfg_scale,
            motion_frame, colormatch,
            use_image_resize, resize_interpolation, resize_method, resize_condition,
            enable_audio_crop, audio_start_time, audio_crop_duration,
            enable_audio_separation, separation_model,
            auto_calculate_frames, max_frames,
            opt_components['blocks_to_swap'],
            opt_components.get('vae_blocks_to_swap', gr.Number(value=0)),
            opt_components['enable_cuda_optimization'],
            opt_components['enable_dram_optimization'],
            opt_components['auto_hardware_tuning'],
            opt_components['vram_threshold_percent'],
            opt_components['num_cuda_streams'],
            opt_components['bandwidth_target'],
            opt_components['offload_txt_emb'],
            opt_components['offload_img_emb'],
            opt_components['debug_mode']
        ],
        outputs=[output_video, output_info]
    )


if __name__ == "__main__":
    # Test UI
    with gr.Blocks() as demo:
        create_infinite_talk_tab()
    
    demo.launch()
