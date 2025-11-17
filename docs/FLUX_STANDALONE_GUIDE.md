# Flux Standalone 使用指南

## 概述

Flux Standalone 是一个**完全独立**的 Flux 文生图实现，不依赖 ComfyUI，使用 `diffusers` 库实现。

### 特点

- ✅ **完全独立**: 不需要 ComfyUI 任何组件
- ✅ **简单易用**: 基于 diffusers 库，标准化接口
- ✅ **灵活加载**: 支持本地文件和 HuggingFace 模型
- ✅ **高性能**: 支持 xformers、CPU offload 等优化
- ✅ **易于集成**: 清晰的 API，易于集成到其他项目

## 安装

### 1. 安装依赖

```bash
pip install -r requirements_flux.txt
```

或手动安装核心依赖:

```bash
pip install torch torchvision
pip install diffusers transformers accelerate
pip install gradio pillow numpy
```

### 2. 可选优化包

```bash
# 内存优化 (推荐)
pip install xformers

# 量化支持
pip install bitsandbytes
```

## 模型准备

### 方式 1: 本地模型

将 Flux 模型文件放到以下目录:

```
models/
├── unet/                          # 或
├── diffusion_models/              # 放这里
│   └── flux1-dev-fp8.safetensors
└── vae/                           # 可选
    └── ae.sft
```

支持的模型格式:
- `.safetensors` (推荐)
- `.ckpt`
- `.pt`

### 方式 2: HuggingFace 模型

直接从 HuggingFace 加载 (首次会下载):

- `black-forest-labs/FLUX.1-dev`
- `black-forest-labs/FLUX.1-schnell`

## 使用方法

### 方式 1: Gradio UI (推荐)

#### 使用启动脚本

```batch
scripts\start_flux_standalone.bat
```

#### 或直接运行

```bash
python apps\sd_module\flux_gradio_standalone.py
```

访问: http://localhost:7861

### 方式 2: Python API

```python
from apps.sd_module.flux_standalone import FluxStandalonePipeline

# 创建管道
pipeline = FluxStandalonePipeline()

# 方式 A: 从本地文件加载
pipeline.load_from_single_file(
    "models/unet/flux1-dev-fp8.safetensors",
    vae_path="models/vae/ae.sft"  # 可选
)

# 方式 B: 从 HuggingFace 加载
pipeline.load_from_pretrained("black-forest-labs/FLUX.1-schnell")

# 生成图像
images = pipeline.generate(
    prompt="a beautiful landscape with mountains and lake",
    width=1024,
    height=1024,
    num_inference_steps=28,
    guidance_scale=3.5,
    seed=42
)

# 保存
images[0].save("output.png")
```

## 参数说明

### 生成参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | str | - | 正向提示词 |
| `negative_prompt` | str | "" | 负向提示词 (Flux 不使用) |
| `width` | int | 1024 | 图像宽度 |
| `height` | int | 1024 | 图像高度 |
| `num_inference_steps` | int | 28 | 推理步数 |
| `guidance_scale` | float | 3.5 | 引导强度 |
| `num_images_per_prompt` | int | 1 | 生成图像数量 |
| `seed` | int | None | 随机种子 |
| `max_sequence_length` | int | 256 | 文本编码器最大长度 |

### 推荐参数

#### 标准质量
```python
width=1024
height=1024
num_inference_steps=28
guidance_scale=3.5
```

#### 快速生成 (FLUX.1-schnell)
```python
width=1024
height=1024
num_inference_steps=4
guidance_scale=0.0  # schnell 不需要 guidance
```

#### 高质量
```python
width=1024
height=1024
num_inference_steps=50
guidance_scale=3.5
```

## 性能优化

### 1. 启用 xformers

```python
# 自动启用 (如果已安装)
pipeline.load_from_single_file(model_path)
# 会自动调用 enable_xformers_memory_efficient_attention()
```

### 2. CPU Offload (低显存)

```python
# 在代码中取消注释
# self.pipe.enable_model_cpu_offload()
```

### 3. 使用量化模型

使用 fp8 量化的模型文件:
- `flux1-dev-fp8.safetensors`
- `flux1-schnell-fp8.safetensors`

### 4. 调整分辨率

降低分辨率可以显著减少显存使用:
```python
width=512
height=512
```

## 示例代码

### 基础生成

```python
from apps.sd_module.flux_standalone import FluxStandalonePipeline

pipeline = FluxStandalonePipeline()
pipeline.load_from_single_file("models/unet/flux1-dev-fp8.safetensors")

images = pipeline.generate(
    prompt="a cat wearing a hat",
    width=1024,
    height=1024,
    num_inference_steps=28,
    guidance_scale=3.5
)

images[0].save("cat.png")
```

### 批量生成

```python
# 生成多张图片
for i in range(5):
    images = pipeline.generate(
        prompt="a beautiful landscape",
        seed=i,
        num_images_per_prompt=1
    )
    images[0].save(f"landscape_{i}.png")
```

### 使用不同种子

```python
import random

for _ in range(10):
    seed = random.randint(0, 2**32 - 1)
    images = pipeline.generate(
        prompt="abstract art",
        seed=seed
    )
    images[0].save(f"art_{seed}.png")
```

## 与 ComfyUI 版本对比

| 特性 | Flux Standalone | ComfyUI 版本 |
|------|----------------|--------------|
| **依赖** | diffusers, transformers | ComfyUI 完整环境 |
| **安装** | pip install 简单 | 需要完整 ComfyUI |
| **模型加载** | 标准 diffusers 格式 | ComfyUI 格式 |
| **易用性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **灵活性** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **性能** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **独立性** | ✅ 完全独立 | ❌ 依赖 ComfyUI |

## 常见问题

### Q: 需要 ComfyUI 吗?

**A:** 不需要！Flux Standalone 完全独立，只需要 diffusers 库。

### Q: 支持哪些模型?

**A:** 
- 本地: 任何 Flux 格式的 safetensors/ckpt 文件
- 在线: HuggingFace 上的 Flux 模型

### Q: 显存不够怎么办?

**A:** 
1. 使用 fp8 量化模型
2. 降低分辨率
3. 启用 CPU offload
4. 减少推理步数

### Q: 如何加速生成?

**A:**
1. 安装 xformers
2. 使用 FLUX.1-schnell 模型
3. 减少推理步数
4. 使用较小的分辨率

### Q: 生成质量不好?

**A:**
1. 增加推理步数 (50+)
2. 调整 guidance_scale (3.0-4.0)
3. 优化提示词
4. 尝试不同的种子

## 技术细节

### 架构

```
FluxStandalonePipeline
├── diffusers.FluxPipeline
│   ├── FluxTransformer2DModel (UNET)
│   ├── CLIPTextModel (CLIP-L)
│   ├── T5EncoderModel (T5-XXL)
│   └── AutoencoderKL (VAE)
└── 优化
    ├── xformers (内存优化)
    ├── CPU offload (显存优化)
    └── torch.compile (速度优化)
```

### 模型加载流程

1. **从单文件加载**:
   ```
   safetensors → diffusers.FluxPipeline.from_single_file()
   ```

2. **从 HuggingFace 加载**:
   ```
   model_id → diffusers.FluxPipeline.from_pretrained()
   ```

3. **自动优化**:
   - 检测 CUDA 可用性
   - 启用 xformers (如果可用)
   - 设置最佳 dtype

### 生成流程

1. **文本编码**: CLIP-L + T5-XXL
2. **Latent 初始化**: 随机噪声
3. **去噪循环**: Flux Transformer
4. **VAE 解码**: Latent → Image

## 更新日志

### v1.0.0 (2025-11-16)
- ✅ 初始版本
- ✅ 支持本地文件和 HuggingFace 模型
- ✅ Gradio UI
- ✅ 完全独立于 ComfyUI

## 贡献

欢迎提交 Issue 和 Pull Request!

## 许可

MIT License

## 作者

**eddy** - 2025-11-16

## 参考

- [diffusers](https://github.com/huggingface/diffusers)
- [FLUX.1](https://github.com/black-forest-labs/flux)
- [Transformers](https://github.com/huggingface/transformers)
