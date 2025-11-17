# Sage3 FP4 使用说明

## 概述

Sage3 是一个统一的 SageAttention 包,集成了:
- **SageAttention 2.x**: 标准的 INT8 量化注意力机制
- **SageAttention3 Blackwell**: 专为 NVIDIA Blackwell 架构优化的 FP4 量化注意力机制

## 已安装的包

在 `python313` 环境中已安装:
- `sage3==3.0.0` - 统一的 SageAttention 包
- `sageattention==2.2.0.post1` - 原始 SageAttention 包

## 修复内容

### 1. 修复 attention.py 导入逻辑

**文件**: `custom_nodes/Comfyui/ComfyUI-WanVideoWrapper/wanvideo/modules/attention.py`

**修改**:
- 优先从 `sage3` 包导入 `sageattn3_blackwell`
- 如果 `sage3` 不可用,回退到 `sageattention` 包
- 正确处理 `SAGE3_AVAILABLE` 标志

### 2. 添加 UI 选项

**文件**: `apps/wanvideo_module/wanvideo_gradio_app.py`

**新增 Attention Mode 选项**:
- `sageattn` - 标准 SageAttention (INT8)
- `sageattn_3` - SageAttention3 Blackwell (默认精度)
- `sageattn_3_fp4` - SageAttention3 Blackwell FP4 (最高性能)
- `sageattn_3_fp8` - SageAttention3 Blackwell FP8 (平衡性能和精度)
- `flash_attn` - Flash Attention
- `sdpa` - PyTorch SDPA
- `xformers` - xFormers

## 使用方法

### 1. 验证安装

运行测试脚本:
```bash
python313\python.exe test_sage3_fp4.py
```

### 2. 在 UI 中使用

1. 启动 WebUI:
   ```bash
   start.bat
   ```

2. 在 **Model Settings** 标签页中:
   - **Attention Mode**: 选择 `sageattn_3_fp4`
   - **Quantization**: 选择 `fp4_scaled`

3. 配置其他参数后点击 **Generate Video**

## Attention Mode 对比

| 模式 | 精度 | 速度 | VRAM | 适用场景 |
|------|------|------|------|----------|
| `sageattn` | INT8 | 快 | 中 | 通用场景 |
| `sageattn_3` | BF16 | 中 | 高 | 高质量生成 |
| `sageattn_3_fp4` | FP4 | 最快 | 最低 | 低显存/高速度 |
| `sageattn_3_fp8` | FP8 | 很快 | 低 | 平衡方案 |
| `flash_attn` | BF16 | 快 | 中 | 长序列 |
| `sdpa` | BF16 | 中 | 中 | 兼容性最好 |

## 技术细节

### FP4 量化原理

SageAttention3 Blackwell 使用块级 FP4 量化:
- **块大小**: 每个块包含多个元素
- **量化方式**: 每个块使用独立的缩放因子
- **精度**: 4-bit 浮点数 (1 符号位 + 3 数值位)
- **性能**: 相比 FP16 可节省 75% 内存

### 适用条件

- **GPU**: NVIDIA Blackwell 架构 (RTX 50 系列)
- **Head Dimension**: < 256 (代码会自动检查)
- **数据类型**: BF16 或 FP16

### 回退机制

如果 FP4 不可用,代码会自动回退:
1. 尝试 SageAttention3 Blackwell (默认精度)
2. 尝试标准 SageAttention (INT8)
3. 回退到 PyTorch SDPA

## 常见问题

### Q: 为什么选择 sageattn_3_fp4 后还是使用其他模式?

A: 可能原因:
1. GPU 不支持 Blackwell 架构
2. Head dimension >= 256
3. sage3 包未正确安装

查看日志中的警告信息了解具体原因。

### Q: FP4 会影响生成质量吗?

A: 
- 对于大多数场景,FP4 的质量损失很小
- 如果追求极致质量,建议使用 `sageattn_3` 或 `flash_attn`
- 可以先用 FP4 快速预览,再用高精度模式生成最终结果

### Q: 如何确认正在使用 FP4?

A: 查看控制台日志:
- 启动时会显示: `SageAttention3 Blackwell (sage3) loaded successfully`
- 生成时如果回退会显示警告信息

## 性能优化建议

### 最佳配置 (低显存)

```
Attention Mode: sageattn_3_fp4
Quantization: fp4_scaled
Block Swap: Enabled (16-20 blocks)
Torch Compile: Disabled (首次生成)
```

### 最佳配置 (高质量)

```
Attention Mode: sageattn_3 或 flash_attn
Quantization: fp8_scaled
Block Swap: Disabled
Torch Compile: Enabled (inductor backend)
```

### 最佳配置 (平衡)

```
Attention Mode: sageattn_3_fp8
Quantization: fp8_scaled
Block Swap: Enabled (8-12 blocks)
Torch Compile: Enabled (inductor backend)
```

## 更新日志

### 2025-01-17
- 修复 sage3 导入逻辑
- 添加 sageattn_3_fp4/fp8 UI 选项
- 创建测试脚本和使用文档

## 参考资料

- [SageAttention GitHub](https://github.com/thu-ml/SageAttention)
- [Sage3 统一包文档](python313/Lib/site-packages/sage3/)
- [WanVideoWrapper 文档](custom_nodes/Comfyui/ComfyUI-WanVideoWrapper/)
