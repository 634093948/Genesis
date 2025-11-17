# Flux 模型使用指南

## ⚠️ 重要说明

### 错误：attempted relative import beyond top-level package

**原因：**
你选择的模型文件不是 Flux diffusers 格式，而是 ComfyUI 格式的模型。

**Flux 有两种格式：**

1. **ComfyUI 格式** ❌ 不支持
   - 文件名通常包含：`flux1-dev.safetensors`, `flux1-schnell.safetensors`
   - 这些是 ComfyUI 专用格式
   - **本项目的 Flux Standalone 不支持这种格式**

2. **Diffusers 格式** ✅ 支持
   - 通常是文件夹结构，包含多个子文件
   - 或者是转换后的 safetensors 文件
   - 可以从 HuggingFace 直接加载

## 🎯 解决方案

### 方案 1: 使用 HuggingFace 模型（推荐）

**步骤：**
1. 在 Flux 标签中，选择 **"HuggingFace"** 单选按钮
2. 从下拉框选择：
   - `black-forest-labs/FLUX.1-dev` （高质量，需要登录）
   - `black-forest-labs/FLUX.1-schnell` （快速版本）
3. 点击 "📥 加载模型"
4. 首次使用会自动下载（需要网络连接）

**优点：**
- ✅ 官方支持
- ✅ 自动下载
- ✅ 保证兼容

**缺点：**
- ⚠️ 需要网络连接
- ⚠️ 首次下载较大（约 20-30GB）
- ⚠️ FLUX.1-dev 需要 HuggingFace 账号登录

### 方案 2: 转换 ComfyUI 模型为 Diffusers 格式

如果你已经有 ComfyUI 格式的 Flux 模型，需要转换：

**使用转换脚本：**

```python
# convert_flux_to_diffusers.py
from diffusers import FluxPipeline
import torch

# 加载 ComfyUI 格式模型
comfy_model_path = "models/unet/flux1-dev.safetensors"

# 转换为 diffusers 格式
pipe = FluxPipeline.from_single_file(
    comfy_model_path,
    torch_dtype=torch.float16
)

# 保存为 diffusers 格式
output_path = "models/diffusers/flux1-dev-diffusers"
pipe.save_pretrained(output_path)

print(f"✅ 转换完成！保存到: {output_path}")
```

**运行转换：**
```bash
python313\python.exe convert_flux_to_diffusers.py
```

**注意：**
- 转换需要大量内存（建议 32GB+）
- 转换后的模型会占用更多磁盘空间

### 方案 3: 下载 Diffusers 格式模型

**从 HuggingFace 手动下载：**

1. 访问 https://huggingface.co/black-forest-labs/FLUX.1-schnell
2. 点击 "Files and versions"
3. 下载整个仓库或使用 git clone：
   ```bash
   git lfs install
   git clone https://huggingface.co/black-forest-labs/FLUX.1-schnell
   ```
4. 将下载的文件夹放到 `models/diffusers/` 目录
5. 在 UI 中使用 HuggingFace 选项，输入本地路径

## 📁 推荐的目录结构

```
models/
├── diffusers/                    # Diffusers 格式模型
│   ├── flux1-dev-diffusers/     # 转换后的模型
│   │   ├── model_index.json
│   │   ├── scheduler/
│   │   ├── text_encoder/
│   │   ├── text_encoder_2/
│   │   ├── tokenizer/
│   │   ├── tokenizer_2/
│   │   ├── transformer/
│   │   └── vae/
│   └── flux1-schnell/           # HuggingFace 下载的模型
│
├── unet/                         # ComfyUI 格式（不支持）
│   └── flux1-dev.safetensors    # ❌ 无法直接使用
│
└── vae/                          # VAE 模型
    └── ae.safetensors
```

## 🚀 快速开始（推荐方式）

### 使用 FLUX.1-schnell（快速版本）

1. 打开主 UI：`start.bat`
2. 进入 **文生图 > Flux** 标签
3. 选择 **"HuggingFace"**
4. 选择 `black-forest-labs/FLUX.1-schnell`
5. 点击 "📥 加载模型"
6. 等待下载完成（首次使用）
7. 输入提示词，点击生成

**推荐参数：**
- 尺寸: 1024 x 1024
- 步数: 4-8（schnell 版本很快）
- 引导: 0（schnell 不需要引导）

### 使用 FLUX.1-dev（高质量版本）

**前提条件：**
- 需要 HuggingFace 账号
- 需要接受模型许可协议

**步骤：**
1. 访问 https://huggingface.co/black-forest-labs/FLUX.1-dev
2. 点击 "Agree and access repository"
3. 在本地登录 HuggingFace：
   ```bash
   python313\python.exe -m pip install huggingface-hub
   python313\python.exe -c "from huggingface_hub import login; login()"
   ```
4. 输入你的 HuggingFace token
5. 在 UI 中选择 `black-forest-labs/FLUX.1-dev`
6. 加载并使用

**推荐参数：**
- 尺寸: 1024 x 1024
- 步数: 20-50
- 引导: 3.5

## ❓ 常见问题

### Q: 为什么我的 ComfyUI 模型不能用？

A: 本项目使用 diffusers 库，它与 ComfyUI 的模型格式不兼容。需要：
- 使用 HuggingFace 模型（推荐）
- 或转换 ComfyUI 模型为 diffusers 格式

### Q: 如何判断模型是什么格式？

A: 
- **ComfyUI 格式**：单个 `.safetensors` 或 `.ckpt` 文件
- **Diffusers 格式**：包含多个子文件夹的目录结构

### Q: 下载速度慢怎么办？

A: 
1. 使用国内镜像：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
2. 或使用代理
3. 或手动下载后放到本地

### Q: 内存不足怎么办？

A: 
1. 使用 FP8 量化版本
2. 启用 CPU offload
3. 减少分辨率
4. 关闭其他程序

### Q: 生成速度慢怎么办？

A: 
1. 使用 FLUX.1-schnell（快速版本）
2. 减少步数
3. 确保使用 GPU
4. 安装 xformers

## 📚 相关资源

- **Flux 官方**: https://github.com/black-forest-labs/flux
- **HuggingFace 模型**:
  - FLUX.1-dev: https://huggingface.co/black-forest-labs/FLUX.1-dev
  - FLUX.1-schnell: https://huggingface.co/black-forest-labs/FLUX.1-schnell
- **Diffusers 文档**: https://huggingface.co/docs/diffusers

## ✅ 总结

**推荐使用方式：**
1. ✅ 直接使用 HuggingFace 模型（最简单）
2. ✅ 使用 FLUX.1-schnell（快速，无需登录）
3. ⚠️ 转换 ComfyUI 模型（需要技术能力）

**不推荐：**
- ❌ 直接使用 ComfyUI 格式模型（不兼容）

---

**如有问题，请查看：**
- `FLUX_INTEGRATION_COMPLETE.md` - 完整集成文档
- `QUICK_START.md` - 快速开始指南
