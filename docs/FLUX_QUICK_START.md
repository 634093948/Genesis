# Flux 文生图快速开始指南

## 快速启动

### 方法 1: 使用启动脚本

```batch
scripts\start_flux_ui.bat
```

### 方法 2: 命令行启动

```bash
python apps\sd_module\flux_gradio_ui.py
```

## 基本使用流程

### 1. 准备模型

将模型文件放置到对应目录:

```
models/
├── unet/flux1-dev-fp8.safetensors
├── clip/sd3/t5xxl_fp16.safetensors
├── clip/clip_l.safetensors
└── vae/ae.sft
```

### 2. 加载模型

在 UI 中:
1. 选择 UNET 模型
2. 选择 CLIP 模型 (T5XXL + CLIP-L)
3. 选择 VAE 模型
4. 点击 "📥 Load Models"

### 3. 生成图像

1. 输入正向提示词
2. 输入负向提示词 (可选)
3. 调整参数:
   - 分辨率: 1080x1920 (推荐)
   - 步数: 20 (推荐)
   - CFG: 1.0 (Flux 推荐值)
   - Guidance: 3.5 (推荐)
4. 点击 "🎨 Generate Image"

## 推荐参数

### 标准设置
- **分辨率**: 1080x1920 或 1024x1024
- **步数**: 20-30
- **CFG Scale**: 1.0
- **Flux Guidance**: 3.5
- **采样器**: dpmpp_2m
- **调度器**: sgm_uniform

### 高质量设置
- **分辨率**: 1920x1080
- **步数**: 30-40
- **CFG Scale**: 1.0
- **Flux Guidance**: 4.0
- **采样器**: euler
- **调度器**: karras

### 快速测试
- **分辨率**: 512x512
- **步数**: 10-15
- **CFG Scale**: 1.0
- **Flux Guidance**: 3.0
- **采样器**: euler_a
- **调度器**: simple

## 提示词建议

### 正向提示词模板

```
[主体描述], [风格], [质量标签], [细节描述]
```

示例:
```
a beautiful landscape with mountains and lake, 
sunset, cinematic lighting, 
4k, highly detailed, masterpiece, 
professional photography
```

### 负向提示词模板

```
worst quality, low quality, normal quality, 
blurry, jpeg artifacts, 
ugly, bad anatomy, distorted
```

## LoRA 使用

### 加载 LoRA

1. 在 "LoRA Settings" 区域选择 LoRA 模型
2. 调整强度 (推荐 0.6-0.8)
3. 可同时加载 2 个 LoRA

### LoRA 强度建议

- **风格 LoRA**: 0.6-0.8
- **角色 LoRA**: 0.7-0.9
- **细节 LoRA**: 0.5-0.7

## 常见问题

### Q: 模型加载失败?
A: 检查模型文件路径是否正确,确保模型在 `models/` 对应目录下。

### Q: 生成速度慢?
A: 
- 使用 fp8 量化模型
- 降低分辨率
- 减少步数
- 确保使用 GPU

### Q: 显存不足?
A:
- 使用 fp8 模型
- 降低分辨率
- 关闭其他占用显存的程序

### Q: 生成质量不好?
A:
- 增加步数 (30-40)
- 调整 Guidance (3.5-4.5)
- 优化提示词
- 尝试不同采样器

## 代码示例

### Python 调用

```python
from apps.sd_module.flux_text2img import FluxText2ImgPipeline

# 创建管道
pipeline = FluxText2ImgPipeline()

# 加载模型
pipeline.load_unet("flux1-dev-fp8.safetensors")
pipeline.load_dual_clip(
    "sd3/t5xxl_fp16.safetensors",
    "clip_l.safetensors"
)
pipeline.load_vae("ae.sft")

# 生成图像
image = pipeline.generate(
    prompt="a beautiful landscape with mountains",
    negative_prompt="low quality, blurry",
    width=1080,
    height=1920,
    steps=20,
    guidance=3.5,
    seed=42
)

# 保存
image.save("output.png")
```

### 批量生成

```python
for i in range(5):
    image = pipeline.generate(
        prompt="a beautiful landscape",
        seed=i,  # 不同种子
        width=1024,
        height=1024
    )
    image.save(f"output_{i}.png")
```

## 性能优化

### GPU 优化
- 使用 CUDA
- 启用 FP16
- 使用 fp8 量化模型

### 内存优化
- 使用注意力切片
- 降低批次大小
- 及时释放不用的模型

## 下一步

- 查看完整文档: `docs/FLUX_INTEGRATION.md`
- 查看集成总结: `FLUX_INTEGRATION_SUMMARY.md`
- 探索更多参数组合
- 尝试不同的 LoRA 组合

## 支持

如有问题,请查看:
1. 完整文档
2. 错误日志
3. 测试文件

---

**快速开始指南** | eddy | 2025-11-16
