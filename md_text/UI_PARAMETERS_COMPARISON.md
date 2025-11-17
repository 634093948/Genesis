# UI 参数对比与改进建议

## 当前 UI 参数 vs 工作流参数

### ✅ 已有参数（WanVideoWrapper）

| 参数 | UI 当前值 | 工作流值 | 状态 |
|------|-----------|----------|------|
| `quantization` | fp4_scaled | fp4_scaled | ✅ 完全匹配 |
| `attention_mode` | sageattn | sageattn_3_fp4 | ⚠️ 默认值不同 |
| `base_precision` | fp16_fast | bf16 | ⚠️ 不同 |
| `load_device` | offload_device | main_device | ⚠️ 不同 |

### ✅ 已有参数（BlockSwap）

| 参数 | UI 当前值 | 工作流值 | 状态 |
|------|-----------|----------|------|
| `block_swap_enabled` | false | true | ⚠️ 默认值不同 |
| `blocks_to_swap` | 16 | 40 | ⚠️ 默认值不同 |

### ❌ 缺失参数（WanVideoWrapper）

| 参数 | 工作流值 | 重要性 | 说明 |
|------|----------|--------|------|
| `rms_norm_function` | default | 中 | RMS 归一化函数选择 |

### ❌ 缺失参数（BlockSwap/IntelligentVRAMNode）

| 参数 | 工作流值 | 重要性 | 说明 |
|------|----------|--------|------|
| `enable_cuda_optimization` | true | 高 | CUDA 优化开关 |
| `enable_dram_optimization` | true | 高 | DRAM 优化开关 |
| `auto_hardware_tuning` | false | 中 | 自动硬件调优 |
| `vram_threshold_percent` | 80 | 高 | VRAM 阈值百分比 |
| `num_cuda_streams` | 16 | 高 | CUDA 流数量 |
| `bandwidth_target` | 1.0 | 中 | 带宽目标 |
| `offload_txt_emb` | false | 低 | 卸载文本嵌入 |
| `offload_img_emb` | false | 低 | 卸载图像嵌入 |
| `vace_blocks_to_swap` | 0 | 低 | VACE 块交换数量 |
| `debug_mode` | false | 低 | 调试模式 |

### ❌ 缺失参数（WanVideoSampler）

| 参数 | 工作流值 | 重要性 | 说明 |
|------|----------|--------|------|
| `use_tf32` | false | 中 | 使用 TF32 |
| `use_cublas_gemm` | false | 中 | 使用 cuBLAS GEMM |
| `force_contiguous_tensors` | false | 低 | 强制连续张量（我们的修复已足够） |
| `fuse_qkv_projections` | false | 低 | 融合 QKV 投影 |

## 推荐的 UI 改进

### 方案 1: 最小改进（推荐）

**只添加最关键的参数**，保持 UI 简洁：

#### Optimization Tab 增强

```python
with gr.Tab("Optimization"):
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Block Swap (IntelligentVRAMNode)")
            block_swap_enabled = gr.Checkbox(label="Enable Block Swap", value=False)
            blocks_to_swap = gr.Slider(0, 48, value=16, step=1, label="Blocks to Swap")
            
            # 新增 ⭐
            vram_threshold = gr.Slider(
                30, 90, value=70, step=5, 
                label="VRAM Threshold (%)",
                info="触发 BlockSwap 的 VRAM 使用率阈值"
            )
            num_cuda_streams = gr.Slider(
                1, 16, value=8, step=1,
                label="CUDA Streams",
                info="并发 CUDA 流数量（越高越快但占用更多资源）"
            )
            
            # 新增 ⭐
            with gr.Accordion("Advanced BlockSwap", open=False):
                enable_cuda_optimization = gr.Checkbox(
                    label="Enable CUDA Optimization", 
                    value=True,
                    info="启用 CUDA 优化（推荐）"
                )
                auto_hardware_tuning = gr.Checkbox(
                    label="Auto Hardware Tuning", 
                    value=False,
                    info="自动根据硬件调整参数"
                )
                bandwidth_target = gr.Slider(
                    0.1, 1.0, value=0.8, step=0.1,
                    label="Bandwidth Target",
                    info="带宽目标比例"
                )
```

#### Advanced Settings 增强

```python
with gr.Column():
    gr.Markdown("### Advanced Settings")
    quantization = gr.Dropdown(
        choices=["disabled", "fp8_scaled", "fp4_scaled", "int8"],
        value="fp4_scaled",
        label="Quantization"
    )
    attention_mode = gr.Dropdown(
        choices=["sageattn", "sageattn_3", "sageattn_3_fp4", "sageattn_3_fp8", "flash_attn", "sdpa", "xformers"],
        value="sageattn_3_fp4",  # 改为 fp4 默认 ⭐
        label="Attention Mode"
    )
    
    # 新增 ⭐
    base_precision = gr.Dropdown(
        choices=["fp16", "bf16", "fp16_fast", "bf16_fast"],
        value="bf16",
        label="Base Precision",
        info="模型基础精度"
    )
    load_device = gr.Dropdown(
        choices=["main_device", "offload_device"],
        value="main_device",
        label="Load Device",
        info="模型加载设备"
    )
    
    # 新增 ⭐
    with gr.Accordion("Sampler Advanced", open=False):
        use_tf32 = gr.Checkbox(
            label="Use TF32", 
            value=False,
            info="使用 TensorFloat-32（Ampere+ GPU）"
        )
        use_cublas_gemm = gr.Checkbox(
            label="Use cuBLAS GEMM", 
            value=False,
            info="使用 cuBLAS 矩阵乘法"
        )
```

### 方案 2: 完整改进（高级用户）

**添加所有参数**，提供完全控制：

#### 新增 "Expert Settings" Tab

```python
with gr.Tab("Expert Settings"):
    gr.Markdown("""
    ### ⚠️ 专家设置
    这些参数对性能和稳定性有重大影响，请谨慎修改。
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Model Loading")
            base_precision = gr.Dropdown(
                choices=["fp16", "bf16", "fp16_fast", "bf16_fast"],
                value="bf16",
                label="Base Precision"
            )
            load_device = gr.Dropdown(
                choices=["main_device", "offload_device"],
                value="main_device",
                label="Load Device"
            )
            rms_norm_function = gr.Dropdown(
                choices=["default", "fast", "apex"],
                value="default",
                label="RMS Norm Function"
            )
        
        with gr.Column():
            gr.Markdown("### BlockSwap Advanced")
            enable_cuda_optimization = gr.Checkbox(
                label="Enable CUDA Optimization", 
                value=True
            )
            enable_dram_optimization = gr.Checkbox(
                label="Enable DRAM Optimization", 
                value=True
            )
            auto_hardware_tuning = gr.Checkbox(
                label="Auto Hardware Tuning", 
                value=False
            )
            vram_threshold = gr.Slider(
                30, 90, value=70, step=5,
                label="VRAM Threshold (%)"
            )
            num_cuda_streams = gr.Slider(
                1, 16, value=8, step=1,
                label="CUDA Streams"
            )
            bandwidth_target = gr.Slider(
                0.1, 1.0, value=0.8, step=0.1,
                label="Bandwidth Target"
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Embedding Offload")
            offload_txt_emb = gr.Checkbox(
                label="Offload Text Embeddings", 
                value=False
            )
            offload_img_emb = gr.Checkbox(
                label="Offload Image Embeddings", 
                value=False
            )
            vace_blocks_to_swap = gr.Slider(
                0, 15, value=0, step=1,
                label="VACE Blocks to Swap"
            )
        
        with gr.Column():
            gr.Markdown("### Sampler Advanced")
            use_tf32 = gr.Checkbox(
                label="Use TF32", 
                value=False
            )
            use_cublas_gemm = gr.Checkbox(
                label="Use cuBLAS GEMM", 
                value=False
            )
            force_contiguous_tensors = gr.Checkbox(
                label="Force Contiguous Tensors", 
                value=False,
                info="强制张量连续（我们的修复已足够，通常不需要）"
            )
            fuse_qkv_projections = gr.Checkbox(
                label="Fuse QKV Projections", 
                value=False
            )
    
    with gr.Row():
        debug_mode = gr.Checkbox(
            label="Debug Mode", 
            value=False,
            info="启用详细日志输出"
        )
```

### 方案 3: 预设配置（最简单）

**提供预设配置**，用户只需选择场景：

```python
with gr.Tab("Quick Presets"):
    gr.Markdown("""
    ### 🚀 快速预设
    根据你的硬件和需求选择预设配置
    """)
    
    preset = gr.Radio(
        choices=[
            "High Performance (24GB+ VRAM)",
            "Balanced (16GB VRAM)",
            "Memory Efficient (12GB VRAM)",
            "Ultra Low VRAM (8GB VRAM)",
            "Custom"
        ],
        value="Balanced (16GB VRAM)",
        label="Select Preset"
    )
    
    gr.Markdown("""
    **预设说明**：
    - **High Performance**: 无 BlockSwap，FP4 量化，Sage3 FP4 attention
    - **Balanced**: 20 blocks swap，FP4 量化，8 CUDA streams
    - **Memory Efficient**: 30 blocks swap，FP4 量化，16 CUDA streams
    - **Ultra Low VRAM**: 40 blocks swap，FP8 量化，16 CUDA streams
    - **Custom**: 手动配置所有参数
    """)
    
    # 预设配置映射
    preset_configs = {
        "High Performance (24GB+ VRAM)": {
            "quantization": "fp4_scaled",
            "attention_mode": "sageattn_3_fp4",
            "blocks_to_swap": 0,
            "num_cuda_streams": 8,
            "vram_threshold": 90
        },
        "Balanced (16GB VRAM)": {
            "quantization": "fp4_scaled",
            "attention_mode": "sageattn_3_fp4",
            "blocks_to_swap": 20,
            "num_cuda_streams": 8,
            "vram_threshold": 70
        },
        "Memory Efficient (12GB VRAM)": {
            "quantization": "fp4_scaled",
            "attention_mode": "sageattn_3_fp4",
            "blocks_to_swap": 30,
            "num_cuda_streams": 16,
            "vram_threshold": 60
        },
        "Ultra Low VRAM (8GB VRAM)": {
            "quantization": "fp8_scaled",
            "attention_mode": "sageattn_3_fp8",
            "blocks_to_swap": 40,
            "num_cuda_streams": 16,
            "vram_threshold": 50
        }
    }
```

## 推荐实施方案

### 阶段 1: 立即改进（方案 1）

**优先级高的参数**：
1. ✅ `vram_threshold` - VRAM 阈值
2. ✅ `num_cuda_streams` - CUDA 流数量
3. ✅ `base_precision` - 基础精度
4. ✅ `load_device` - 加载设备
5. ✅ 修改 `attention_mode` 默认值为 `sageattn_3_fp4`

**预计工作量**：30 分钟

### 阶段 2: 完整改进（方案 2）

**添加所有参数**，提供完全控制。

**预计工作量**：1-2 小时

### 阶段 3: 用户友好（方案 3）

**添加预设配置**，简化用户选择。

**预计工作量**：1 小时

## 当前 UI 的主要问题

### 1. 默认值不匹配工作流

| 参数 | UI 默认 | 工作流 | 建议 |
|------|---------|--------|------|
| `attention_mode` | sageattn | sageattn_3_fp4 | 改为 sageattn_3_fp4 |
| `blocks_to_swap` | 16 | 40 | 保持 16（更安全） |
| `block_swap_enabled` | false | true | 保持 false（默认不启用） |

### 2. 缺少关键参数

- ❌ `vram_threshold` - 无法控制何时触发 BlockSwap
- ❌ `num_cuda_streams` - 无法优化并发性能
- ❌ `base_precision` - 无法选择精度
- ❌ `load_device` - 无法选择加载设备

### 3. 没有预设配置

用户需要手动配置多个参数，容易出错。

## 建议的代码修改

### 修改 1: 更新默认值

```python
# 文件: wanvideo_gradio_app.py, 行 702-706

attention_mode = gr.Dropdown(
    choices=["sageattn", "sageattn_3", "sageattn_3_fp4", "sageattn_3_fp8", "flash_attn", "sdpa", "xformers"],
    value="sageattn_3_fp4",  # ⭐ 改为 fp4
    label="Attention Mode",
    info="推荐使用 sageattn_3_fp4 以获得最佳性能"
)
```

### 修改 2: 添加关键参数

```python
# 文件: wanvideo_gradio_app.py, 行 740-743

with gr.Column():
    gr.Markdown("### Block Swap")
    block_swap_enabled = gr.Checkbox(label="Enable Block Swap", value=False)
    blocks_to_swap = gr.Slider(0, 48, value=16, step=1, label="Blocks to Swap")
    
    # ⭐ 新增
    vram_threshold = gr.Slider(
        30, 90, value=70, step=5,
        label="VRAM Threshold (%)",
        info="当 VRAM 使用率超过此值时触发 BlockSwap"
    )
    num_cuda_streams = gr.Slider(
        1, 16, value=8, step=1,
        label="CUDA Streams",
        info="并发 CUDA 流数量（RTX 4090/5090 推荐 16）"
    )
```

### 修改 3: 更新函数签名

```python
# 文件: wanvideo_gradio_app.py, 行 159-174

def generate_video(
    self,
    # ... 其他参数 ...
    quantization: str,
    attention_mode: str,
    # ⭐ 新增
    base_precision: str,
    load_device: str,
    # LoRA parameters
    lora_enabled: bool,
    lora_name: str,
    lora_strength: float,
    # Optimization parameters
    compile_enabled: bool,
    compile_backend: str,
    block_swap_enabled: bool,
    blocks_to_swap: int,
    # ⭐ 新增
    vram_threshold: float,
    num_cuda_streams: int,
    enable_cuda_optimization: bool,
    # Output parameters
    output_format: str,
    fps: int,
    progress_callback=None
):
```

## 总结

### 当前状态

- ✅ 基础参数已暴露（quantization, attention_mode, blocks_to_swap）
- ⚠️ 默认值不匹配工作流
- ❌ 缺少关键 BlockSwap 参数
- ❌ 缺少模型加载参数
- ❌ 没有预设配置

### 推荐行动

1. **立即修改**：更新 `attention_mode` 默认值为 `sageattn_3_fp4`
2. **短期添加**：`vram_threshold` 和 `num_cuda_streams`（方案 1）
3. **中期改进**：添加所有专家参数（方案 2）
4. **长期优化**：添加预设配置（方案 3）

这样用户就能完全控制所有参数，同时保持 UI 的易用性！
