#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video Generation Optimization Settings
视频生成优化设置（共享模块）

Based on IntelligentVRAMNode for advanced VRAM management
基于 IntelligentVRAMNode 的高级 VRAM 管理

Author: eddy
Date: 2025-11-16
"""

import gradio as gr


def create_optimization_settings(default_blocks=0, show_vae_blocks=False):
    """
    Create optimization settings UI component
    创建优化设置 UI 组件
    
    Args:
        default_blocks: Default number of blocks to swap
        show_vae_blocks: Whether to show VAE blocks option
    
    Returns:
        Dictionary of Gradio components
    """
    
    components = {}
    
    with gr.Accordion("🚀 优化设置 (Optimization)", open=False):
        gr.Markdown("""
        **基于 IntelligentVRAMNode 的智能 VRAM 管理**
        
        自动检测硬件并优化性能，支持 BlockSwap、CUDA 优化等高级功能。
        """)
        
        with gr.Tab("基础设置"):
            components['blocks_to_swap'] = gr.Slider(
                label="BlockSwap 块数",
                info="交换到 CPU 的 Transformer 块数量（0=禁用，越高越省 VRAM 但越慢）",
                minimum=0,
                maximum=40,
                value=default_blocks,
                step=1
            )
            
            if show_vae_blocks:
                components['vae_blocks_to_swap'] = gr.Slider(
                    label="VAE BlockSwap 块数",
                    info="交换到 CPU 的 VAE 块数量",
                    minimum=0,
                    maximum=15,
                    value=0,
                    step=1
                )
            
            components['enable_cuda_optimization'] = gr.Checkbox(
                label="启用 CUDA 优化",
                info="使用 CUDA 流和固定内存加速传输",
                value=True
            )
            
            components['enable_dram_optimization'] = gr.Checkbox(
                label="启用 DRAM 优化",
                info="优化系统内存使用",
                value=True
            )
        
        with gr.Tab("自动调优"):
            components['auto_hardware_tuning'] = gr.Checkbox(
                label="自动硬件调优",
                info="根据 GPU 型号和 VRAM 自动配置最优参数",
                value=True
            )
            
            components['vram_threshold_percent'] = gr.Slider(
                label="VRAM 阈值 (%)",
                info="VRAM 使用率超过此值时触发警告",
                minimum=30.0,
                maximum=90.0,
                value=50.0,
                step=5.0
            )
            
            gr.Markdown("""
            **自动调优说明:**
            - RTX 5090/4090: 16 CUDA 流, 90% 带宽
            - RTX 3090/4080: 12 CUDA 流, 80% 带宽  
            - 其他 GPU: 8 CUDA 流, 70% 带宽
            """)
        
        with gr.Tab("高级设置"):
            components['num_cuda_streams'] = gr.Slider(
                label="CUDA 流数量",
                info="并行传输流数量（自动调优时忽略）",
                minimum=1,
                maximum=16,
                value=8,
                step=1
            )
            
            components['bandwidth_target'] = gr.Slider(
                label="带宽目标比例",
                info="PCIe 带宽使用目标（自动调优时忽略）",
                minimum=0.1,
                maximum=1.0,
                value=0.8,
                step=0.1
            )
            
            components['offload_txt_emb'] = gr.Checkbox(
                label="卸载文本嵌入",
                info="将文本嵌入卸载到 CPU（节省 VRAM）",
                value=False
            )
            
            components['offload_img_emb'] = gr.Checkbox(
                label="卸载图像嵌入",
                info="将图像嵌入卸载到 CPU（节省 VRAM）",
                value=False
            )
        
        with gr.Tab("调试"):
            components['debug_mode'] = gr.Checkbox(
                label="调试模式",
                info="输出详细的内存和性能日志",
                value=False
            )
            
            gr.Markdown("""
            **调试信息将输出到控制台**
            
            包含:
            - VRAM 使用统计
            - 传输速度监控
            - 硬件检测结果
            - 优化配置详情
            """)
    
    return components


def get_optimization_args(components):
    """
    Extract optimization arguments from components
    从组件中提取优化参数
    
    Args:
        components: Dictionary of Gradio components
    
    Returns:
        Dictionary of optimization arguments
    """
    return {
        'blocks_to_swap': components.get('blocks_to_swap'),
        'vae_blocks_to_swap': components.get('vae_blocks_to_swap', 0),
        'enable_cuda_optimization': components.get('enable_cuda_optimization'),
        'enable_dram_optimization': components.get('enable_dram_optimization'),
        'auto_hardware_tuning': components.get('auto_hardware_tuning'),
        'vram_threshold_percent': components.get('vram_threshold_percent'),
        'num_cuda_streams': components.get('num_cuda_streams'),
        'bandwidth_target': components.get('bandwidth_target'),
        'offload_txt_emb': components.get('offload_txt_emb'),
        'offload_img_emb': components.get('offload_img_emb'),
        'debug_mode': components.get('debug_mode'),
    }
