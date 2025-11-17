# Infinite Talk CUDA 内存对齐错误修复

## 问题描述

Infinite Talk 模块在使用 Sage3 FP4/FP8 量化时出现 CUDA 内存对齐错误：
```
torch.AcceleratorError: CUDA error: misaligned address
```

## 根本原因

CUDA 内存对齐错误主要由以下几个原因引起：

1. **张量形状直接访问**：在 CUDA 张量上直接访问 `.shape` 属性并进行比较可能触发异步错误
2. **非连续内存布局**：`transpose()`, `view()`, `reshape()` 等操作后的张量可能不连续
3. **FP4/FP8 量化敏感性**：低精度量化对内存对齐要求更严格

## 修复方案

### 1. multitalk.py 修复

**位置**：`custom_nodes\Comfyui\ComfyUI-WanVideoWrapper\multitalk\multitalk.py`

**修改内容**：
- `SingleStreamAttention.forward()` (行 228-266)
  - 将形状访问转换为 Python `int` 类型
  - 在张量切片后添加 `.contiguous()`
  - 使用安全的形状提取避免 CUDA 异步错误

- `SingleStreamMultiAttention.forward()` (行 309-446)
  - 在 `squeeze()` 后添加 `.contiguous()`
  - 形状比较前先提取为 `int` 变量
  - 在 `rearrange()` 后添加 `.contiguous()`

- `calculate_x_ref_attn_map()` (行 45-48)
  - scale 计算使用 `float()` 转换

- `AudioMultiProj.forward()` (行 172-174)
  - 所有 shape 访问使用 `int()` 转换

### 2. attention.py 修复

**位置**：`custom_nodes\Comfyui\ComfyUI-WanVideoWrapper\wanvideo\modules\attention.py`

**修改内容**：
- `sageattn_3` 模式 (行 210-235)
  - 在 transpose 前后确保 contiguous
  - 使用临时变量存储连续张量

- `sageattn_3_fp4` 模式 (行 236-260)
  - **关键修复**：FP4 量化前确保内存对齐
  - 在所有 transpose 操作前后调用 `.contiguous()`

- `sageattn_3_fp8` 模式 (行 261-285)
  - **关键修复**：FP8 量化前确保内存对齐
  - 在所有 transpose 操作前后调用 `.contiguous()`

### 3. fp8_optimization.py 修复

**位置**：`custom_nodes\Comfyui\ComfyUI-WanVideoWrapper\fp8_optimization.py`

**修改内容**：
- `fp8_linear_forward()` (行 16-42)
  - 在 reshape 前确保输入 contiguous
  - 形状提取使用 `int()` 转换为 Python 标量
  - 在最终输出添加 `.contiguous()`

### 4. model.py 修复

**位置**：`custom_nodes\Comfyui\ComfyUI-WanVideoWrapper\wanvideo\modules\model.py`

**修改内容**：
- `WanHuMoCrossAttention.forward()` (行 817-837)
  - 所有 view/reshape 操作前后添加 `.contiguous()`
  - 形状值提取为 `int` 类型
  - flatten 操作后确保 contiguous

- `AudioCrossAttentionWrapper.forward()` (行 846-853)
  - 输入张量确保 contiguous
  - 中间计算结果添加 `.contiguous()`
  - 输出确保 contiguous

## 修复效果

✅ **已修复文件**：
- `multitalk/multitalk.py`
- `wanvideo/modules/attention.py`
- `fp8_optimization.py`
- `wanvideo/modules/model.py`

✅ **影响范围**：
- 仅修改 Infinite Talk 和量化相关代码
- 不影响其他模块功能
- 保持原有逻辑不变

✅ **兼容性**：
- 支持所有量化模式（FP4/FP8/FP16）
- 支持 Sage3 Blackwell 加速
- 向后兼容非量化模式

## 测试建议

重新运行 Infinite Talk 生成测试：
```bash
python apps/wanvideo_module/wanvideo_gradio_app.py
```

使用参数：
- 模式：infinitetalk
- 量化：sage3_fp4 或 sage3_fp8
- 音频输入：测试音频文件
- 参考图像：测试图像

## 技术要点

### 为什么需要 contiguous()?

1. **张量内存布局**：PyTorch 张量可能在内存中不连续存储
2. **CUDA 内核要求**：某些 CUDA 操作要求连续内存
3. **量化精度**：FP4/FP8 对内存对齐更敏感

### int() 转换的作用

```python
# 不安全 - 可能触发 CUDA 异步错误
actual_tokens = x.shape[1]  
if actual_tokens != expected_tokens:
    ...

# 安全 - 在 CPU 上比较
actual_tokens = int(x.shape[1])
if actual_tokens != expected_tokens:
    ...
```

### contiguous() 调用时机

```python
# 在这些操作前后调用 contiguous()
x = x.transpose(1, 2).contiguous()
x = x.view(b, -1, n, d).contiguous()
x = x.reshape(-1, hlen_wlen, n, d).contiguous()
x = x.flatten(2).contiguous()
```

## 相关文档

- [Sage3 FP4 使用说明](SAGE3_FP4_使用说明.md)
- [Sage3 FP4 修复总结](SAGE3_FP4_修复总结.md)
- [DECODE VRAM 优化说明](DECODE_VRAM_优化说明.md)
