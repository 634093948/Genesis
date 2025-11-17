# Flux ComfyUI 格式使用指南

## ✅ 已更新为 ComfyUI 格式

Flux 现在使用 **ComfyUI 兼容的 UNET 格式**，而不是 diffusers 格式。

## 📋 模型格式

### ComfyUI 格式 ✅ 支持
- **UNET 模型**: `.safetensors` 或 `.ckpt` 文件
- **CLIP 模型**: 两个独立的文件
  - CLIP 1: T5XXL (例如: `sd3/t5xxl_fp16.safetensors`)
  - CLIP 2: CLIP-L (例如: `clip_l.safetensors`)
- **VAE 模型**: `.safetensors` 或 `.sft` 文件

### Diffusers 格式 ❌ 不支持
- HuggingFace 预训练模型
- Diffusers 文件夹结构

## 📁 模型路径

### 项目文件夹结构
```
E:\liliyuanshangmie\Genesis-webui-modular-integration\
├── models/
│   ├── unet/                    # UNET 模型 ✅
│   │   └── flux1-dev-fp8.safetensors
│   ├── diffusion_models/        # 扩散模型 ✅
│   │   └── flux1-schnell.safetensors
│   ├── clip/                    # CLIP 模型 ✅
│   │   ├── clip_l.safetensors
│   │   └── sd3/
│   │       └── t5xxl_fp16.safetensors
│   ├── text_encoders/           # 文本编码器 ✅
│   │   └── (其他 CLIP 模型)
│   └── vae/                     # VAE 模型 ✅
│       └── ae.sft
└── custom_nodes/
    └── Comfyui/                 # ComfyUI 节点 ✅
        ├── flux/
        └── RES4LYF/
```

## 🎯 工作流节点

基于 `F:\工作流\flux文生图.json`：

### 1. UNETLoader
```python
节点: UNETLoader
输入:
  - unet_name: "flux1-dev-fp8.safetensors"
  - weight_dtype: "default"
输出:
  - MODEL
```

### 2. DualCLIPLoader
```python
节点: DualCLIPLoader
输入:
  - clip_name1: "sd3/t5xxl_fp16.safetensors"  # T5XXL
  - clip_name2: "clip_l.safetensors"          # CLIP-L
  - type: "flux"
输出:
  - CLIP
```

### 3. VAELoader
```python
节点: VAELoader
输入:
  - vae_name: "ae.sft"
输出:
  - VAE
```

### 4. CLIPTextEncode
```python
节点: CLIPTextEncode
输入:
  - clip: CLIP
  - text: "提示词"
输出:
  - CONDITIONING
```

### 5. FluxGuidance
```python
节点: FluxGuidance
输入:
  - conditioning: CONDITIONING
  - guidance: 3.5
输出:
  - CONDITIONING
```

### 6. KSamplerAdvanced
```python
节点: KSamplerAdvanced
输入:
  - model: MODEL
  - positive: CONDITIONING
  - negative: CONDITIONING
  - latent_image: LATENT
  - steps: 20
  - cfg: 1.0
  - sampler_name: "dpmpp_2m"
  - scheduler: "sgm_uniform"
输出:
  - LATENT
```

### 7. VAEDecode
```python
节点: VAEDecode
输入:
  - samples: LATENT
  - vae: VAE
输出:
  - IMAGE
```

## 🚀 使用方法

### 1. 准备模型

**UNET 模型:**
- 下载 Flux UNET 模型 (`.safetensors`)
- 放入 `models/unet/` 或 `models/diffusion_models/`

**CLIP 模型:**
- 下载 T5XXL: `sd3/t5xxl_fp16.safetensors`
- 下载 CLIP-L: `clip_l.safetensors`
- 放入 `models/clip/` 或 `models/text_encoders/`

**VAE 模型:**
- 下载 Flux VAE: `ae.sft` 或 `ae.safetensors`
- 放入 `models/vae/`

### 2. 启动 UI

```batch
start.bat
```

### 3. 加载模型

1. 进入 **文生图 > Flux** 标签
2. 选择 **UNET 模型**
3. 选择 **CLIP 1 (T5XXL)**
4. 选择 **CLIP 2 (CLIP-L)**
5. 选择 **VAE** (可选)
6. 点击 **📥 加载模型**

### 4. 生成图像

1. 输入提示词
2. 设置参数:
   - 尺寸: 1024x1024
   - 步数: 20-50
   - 引导: 3.5
   - 采样器: dpmpp_2m
   - 调度器: sgm_uniform
3. 点击 **🎨 生成图像**

## 📊 推荐配置

### 标准配置
```yaml
UNET: flux1-dev-fp8.safetensors
CLIP 1: sd3/t5xxl_fp16.safetensors
CLIP 2: clip_l.safetensors
VAE: ae.sft

尺寸: 1024 x 1024
步数: 28
引导: 3.5
采样器: dpmpp_2m
调度器: sgm_uniform
```

### 快速配置
```yaml
UNET: flux1-schnell.safetensors
CLIP 1: sd3/t5xxl_fp16.safetensors
CLIP 2: clip_l.safetensors
VAE: ae.sft

尺寸: 1024 x 1024
步数: 4-8
引导: 0
采样器: euler
调度器: simple
```

## 🔧 ComfyUI 节点

### 已集成的节点

**核心节点:**
- UNETLoader
- DualCLIPLoader
- VAELoader
- CLIPTextEncode
- EmptyLatentImage
- KSamplerAdvanced
- VAEDecode

**Flux 节点:**
- FluxGuidance (from comfy_extras.nodes_flux)

**RES4LYF 采样器:**
- 45+ 高级采样器
- 位置: `custom_nodes/Comfyui/RES4LYF/`

## ⚠️ 重要说明

### 不支持的功能

❌ **HuggingFace 加载**
- Flux ComfyUI 格式不支持从 HuggingFace 直接加载
- 必须使用本地文件

❌ **Diffusers 格式**
- 不支持 diffusers 文件夹结构
- 不支持 `FluxPipeline.from_pretrained()`

### 支持的功能

✅ **ComfyUI 工作流兼容**
- 完全兼容 ComfyUI 工作流
- 支持所有 ComfyUI 节点

✅ **本地模型加载**
- 从 models 文件夹加载
- 支持 .safetensors 和 .ckpt

✅ **高级采样器**
- KSampler 基础采样器
- RES4LYF 高级采样器

## 📚 参考资源

### 工作流
- **示例工作流**: `F:\工作流\flux文生图.json`
- **节点文档**: `custom_nodes/Comfyui/`

### 模型下载
- **Flux UNET**: https://huggingface.co/black-forest-labs/FLUX.1-dev
- **T5XXL**: https://huggingface.co/stabilityai/stable-diffusion-3-medium
- **CLIP-L**: https://huggingface.co/openai/clip-vit-large-patch14

### 文档
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI
- **Flux**: https://github.com/black-forest-labs/flux
- **RES4LYF**: https://github.com/blepping/comfyui_res4lyf

## ❓ 常见问题

### Q: 为什么不用 diffusers 格式？
A: ComfyUI 格式是 Flux 的原生格式，兼容性更好，支持更多高级功能和自定义节点。

### Q: 如何获取 CLIP 模型？
A: 
1. T5XXL: 从 SD3 模型包中提取
2. CLIP-L: 从 OpenAI CLIP 或 Flux 官方包中获取

### Q: VAE 是必须的吗？
A: 是的，Flux 需要 VAE 来解码 latent 为图像。

### Q: 支持 LoRA 吗？
A: 支持，使用 LoraLoaderModelOnly 节点（工作流中已包含）。

## ✅ 总结

**格式:** ComfyUI UNET ✅
**模型路径:** 项目 models 文件夹 ✅
**节点:** ComfyUI 兼容 ✅
**采样器:** KSampler + RES4LYF ✅

---

**现在可以使用 ComfyUI 格式的 Flux 了！** 🎉
