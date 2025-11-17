# ✅ Qwen Image 集成完成

## 🎉 功能概述

Qwen Image 已成功集成到 Genesis WebUI 的文生图标签中，与 Flux 并列。

### 基于工作流
- **源工作流**: `custom_nodes/Comfyui/ComfyUI-QwenImageWrapper/qwen3 edy.json`
- **节点**: `eddy_qwen_image_blockswap`
- **去除**: 图片反推节点（MemoryCleaner, PreviewImage）

## 📋 集成架构

### 文件结构
```
apps/
├── sd_module/
│   ├── __init__.py                 # 主入口（已更新）
│   ├── flux_integrated.py          # Flux UI
│   ├── flux_comfy_pipeline.py      # Flux 管道
│   ├── qwen_integrated.py          # Qwen Image UI ✅ 新增
│   └── qwen_comfy_pipeline.py      # Qwen Image 管道 ✅ 新增
└── genesis_webui_integrated.py     # 主 UI

custom_nodes/
└── Comfyui/
    └── ComfyUI-QwenImageWrapper/   # Qwen 节点 ✅
        ├── __init__.py
        ├── standalone_official_nodes.py
        ├── qwen3 edy.json
        └── ...
```

### UI 层级
```
主界面
└── 文生图 (Text-to-Image)
    ├── Stable Diffusion
    ├── Flux
    └── Qwen Image ✅ 新增
```

## 🎯 核心功能

### 1. Qwen Image 管道 (`qwen_comfy_pipeline.py`)

**功能:**
- 使用 ComfyUI 兼容的 Qwen Image 节点
- 支持完整的 Qwen Image 生成流程
- 集成 BlockSwap 内存优化
- 支持 LoRA、量化、编译等高级功能

**关键类:**
```python
class QwenComfyPipeline:
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        unet_name: str = "qwen_image_fp8_e4m3fn.safetensors",
        clip_name: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        vae_name: str = "qwen_image_vae.safetensors",
        width: int = 1328,
        height: int = 1328,
        steps: int = 8,
        cfg: float = 2.5,
        sampler_name: str = "sa_solver",
        scheduler: str = "beta",
        seed: int = -1,
        quantization_dtype: str = "fp16_fast",
        # LoRA settings
        lora_1_name: str = "none",
        lora_1_strength: float = 1.0,
        # ... 更多参数
    ) -> Optional[List[Image.Image]]
```

### 2. Qwen Image UI (`qwen_integrated.py`)

**功能:**
- Gradio 界面集成
- 完整的参数控制
- 实时生成进度
- 结果展示

**UI 组件:**
- ✅ 模型选择（UNET, CLIP, VAE）
- ✅ 提示词输入（正向/负向）
- ✅ 生成参数（尺寸、步数、CFG、采样器）
- ✅ LoRA 设置（4 个 LoRA 插槽）
- ✅ 优化设置（BlockSwap、量化、编译）
- ✅ 结果展示

## 🚀 使用方法

### 1. 启动 UI

```batch
start.bat
```

### 2. 访问界面

```
http://localhost:7860
主界面 > 文生图 > Qwen Image
```

### 3. 选择模型

**UNET 模型:**
- `qwen_image_fp8_e4m3fn.safetensors`
- 或其他 Qwen UNET 模型

**CLIP 模型:**
- `qwen_2.5_vl_7b_fp8_scaled.safetensors`
- 或其他 Qwen CLIP 模型

**VAE 模型:**
- `qwen_image_vae.safetensors`

### 4. 设置参数

**基础参数:**
- 尺寸: 1328x1328（推荐）
- 步数: 8（快速）/ 20（高质量）
- CFG: 2.5
- 采样器: sa_solver
- 调度器: beta

**量化精度:**
- `fp16_fast`: 平衡速度和质量（推荐）
- `fp8_e4m3fn`: 最快，50% VRAM 节省
- `bf16_fast`: 稳定，2.5x 速度

### 5. 生成图像

1. 输入提示词
2. 设置参数
3. 点击"🎨 生成图像"
4. 等待生成完成

## 📊 参数说明

### 模型设置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| UNET 模型 | Qwen Image UNET | qwen_image_fp8_e4m3fn.safetensors |
| CLIP 模型 | Qwen CLIP | qwen_2.5_vl_7b_fp8_scaled.safetensors |
| VAE 模型 | Qwen VAE | qwen_image_vae.safetensors |

### 生成参数

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| 宽度 | 256-2048 | 1328 | 图像宽度（16的倍数）|
| 高度 | 256-2048 | 1328 | 图像高度（16的倍数）|
| 步数 | 1-100 | 8 | 采样步数 |
| CFG | 0-20 | 2.5 | 引导强度 |
| 种子 | -1或正整数 | -1 | 随机种子（-1为随机）|

### 采样器

| 采样器 | 特点 |
|--------|------|
| sa_solver | 推荐，快速收敛 |
| euler | 稳定 |
| dpmpp_2m | 高质量 |
| ddim | 经典 |

### 调度器

| 调度器 | 特点 |
|--------|------|
| beta | 推荐 |
| normal | 标准 |
| karras | 平滑 |
| exponential | 快速 |

### LoRA 设置

- **LoRA 1-4**: 最多 4 个 LoRA
- **强度**: -10.0 到 10.0
- **默认**: none（禁用）

### 优化设置

#### BlockSwap
- **启用**: 30-60% VRAM 节省
- **块数**: 1-50（推荐 20）
- **模型大小**: auto（自动检测）
- **使用推荐**: 自动优化

#### 量化精度
- **fp8_e4m3fn**: 最快，50% VRAM 节省
- **fp16_fast**: 平衡（推荐）
- **bf16_fast**: 稳定，2.5x 速度
- **default**: 无量化

#### 高级优化
- **矩阵乘法优化**: 1.5-2x 加速
- **Torch Compile**: 20-60% 加速（首次慢）
- **混合精度**: 30-50% 加速
- **Flash Attention**: 2-4x 加速

## 🔧 工作流对比

### 原始工作流 (qwen3 edy.json)
```
eddy_qwen_image_blockswap
    ↓
MemoryCleaner (已去除)
    ↓
PreviewImage (已去除)
```

### 集成后的流程
```
QwenComfyPipeline.generate()
    ↓
EddyQwenImageBlockSwap.generate()
    ↓
直接返回 PIL Image
```

**去除的节点:**
- ❌ `MemoryCleaner`: 内存清理（UI 中不需要）
- ❌ `PreviewImage`: 图像预览（UI 直接显示）

**保留的核心:**
- ✅ `eddy_qwen_image_blockswap`: 完整的生成逻辑
- ✅ 所有参数和优化选项

## 📁 模型路径

### 项目模型文件夹
```
models/
├── unet/                    # Qwen UNET 模型
│   └── qwen_image_fp8_e4m3fn.safetensors
├── diffusion_models/        # 备用 UNET 路径
├── clip/                    # Qwen CLIP 模型
│   └── qwen_2.5_vl_7b_fp8_scaled.safetensors
├── text_encoders/           # 备用 CLIP 路径
├── vae/                     # Qwen VAE 模型
│   └── qwen_image_vae.safetensors
└── loras/                   # LoRA 模型
    └── (可选 LoRA 文件)
```

## ✅ 功能清单

### 核心功能
- [x] Qwen Image 节点集成
- [x] ComfyUI 兼容管道
- [x] Gradio UI 集成
- [x] 模型自动扫描
- [x] 子文件夹支持

### 生成功能
- [x] 文本到图像
- [x] 正向/负向提示词
- [x] 自定义尺寸
- [x] 采样器选择
- [x] 调度器选择
- [x] 种子控制

### 高级功能
- [x] LoRA 支持（4个插槽）
- [x] BlockSwap 内存优化
- [x] 量化精度选择
- [x] Torch Compile 加速
- [x] 混合精度训练
- [x] Flash Attention
- [x] KV Cache

### UI 功能
- [x] 实时进度显示
- [x] 参数验证
- [x] 错误提示
- [x] 结果展示
- [x] 参数说明

## 🎯 推荐配置

### 快速配置（8步）
```yaml
UNET: qwen_image_fp8_e4m3fn.safetensors
CLIP: qwen_2.5_vl_7b_fp8_scaled.safetensors
VAE: qwen_image_vae.safetensors

尺寸: 1328 x 1328
步数: 8
CFG: 2.5
采样器: sa_solver
调度器: beta
量化: fp16_fast

BlockSwap: 启用
矩阵优化: 启用
Flash Attention: 启用
```

### 高质量配置（20步）
```yaml
UNET: qwen_image_fp8_e4m3fn.safetensors
CLIP: qwen_2.5_vl_7b_fp8_scaled.safetensors
VAE: qwen_image_vae.safetensors

尺寸: 1328 x 1328
步数: 20
CFG: 3.0
采样器: dpmpp_2m
调度器: karras
量化: bf16_fast

BlockSwap: 启用
矩阵优化: 启用
Flash Attention: 启用
Torch Compile: 启用（首次慢）
```

## 📚 相关文档

- **Qwen Image 节点**: `custom_nodes/Comfyui/ComfyUI-QwenImageWrapper/README.md`
- **ComfyUI 格式**: `docs/FLUX_COMFYUI_FORMAT.md`
- **ComfyUI 设置**: `COMFY_SETUP_COMPLETE.md`
- **模型验证**: `MODEL_VERIFICATION_REPORT.md`

## ⚠️ 注意事项

### 模型要求
- ✅ Qwen Image UNET 模型
- ✅ Qwen CLIP 模型
- ✅ Qwen VAE 模型
- ❌ 不兼容 Flux 或 SD 模型

### 内存要求
- **最小**: 8GB VRAM（使用 BlockSwap + fp8）
- **推荐**: 12GB VRAM（使用 BlockSwap + fp16）
- **最佳**: 16GB+ VRAM（无 BlockSwap）

### 性能优化
1. **启用 BlockSwap**: 30-60% VRAM 节省
2. **使用 fp8 量化**: 50% VRAM 节省
3. **启用矩阵优化**: 1.5-2x 加速
4. **启用 Flash Attention**: 2-4x 加速
5. **Torch Compile**: 20-60% 加速（首次慢）

## 🎉 总结

**状态**: ✅ 完成并集成
**位置**: 主界面 > 文生图 > Qwen Image
**功能**: 完整的 Qwen Image 生成
**优化**: BlockSwap + 量化 + 编译

---

**现在可以使用 Qwen Image 生成高质量图像了！** 🎉

```batch
start.bat
```

访问: http://localhost:7860 > 文生图 > Qwen Image
