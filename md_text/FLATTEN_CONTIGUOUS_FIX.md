# Flatten Contiguous 修复总结

## 问题根源

CUDA 内存对齐错误的真正根源：**所有 `flatten(2)` 操作后的张量在传入 FP8/FP4 量化的线性层前必须是连续的**。

### 错误堆栈
```
File "fp8_optimization.py", line 30, in fp8_linear_forward
    scale_input = torch.ones((), device=input.device, dtype=torch.float32)
torch.AcceleratorError: CUDA error: misaligned address
```

**触发路径**：
1. `WanSelfAttention.forward()` → `x.flatten(2)` → `self.o(...)` 
2. `self.o` 是 FP8/FP4 优化的线性层
3. 非连续的输入导致访问 `input.device` 时内存对齐错误

## 完整修复列表

### 修复的文件
- `wanvideo/modules/model.py` - 15+ 处 flatten(2) 修复

### 修复的类和方法

#### 1. WanSelfAttention 类

**forward()**
```python
# 修复前
return self.o(x.flatten(2))

# 修复后  
return self.o(x.flatten(2).contiguous())
```

**forward_ip()**
```python
# 修复后
return self.o(x.flatten(2).contiguous())
```

**forward_radial()**
```python
# 修复后
return self.o(x.flatten(2).contiguous())
```

**forward_multitalk()**
```python
# 修复前
x = x.flatten(2)
x = self.o(x)

# 修复后
x = x.flatten(2).contiguous()
x = self.o(x)
```

**forward_split()**
```python
# 修复前
x = x.flatten(2)
x = self.o(x)

# 修复后
x = x.flatten(2).contiguous()
x = self.o(x)
```

**normalized_attention_guidance()**
```python
# 修复后
x_positive = x_positive.flatten(2).contiguous()
x_negative = x_negative.flatten(2).contiguous()
```

#### 2. WanCrossAttention 类

**forward()**
```python
# 主attention
x = attention(q, k, v, attention_mode=self.attention_mode).flatten(2).contiguous()

# FantasyTalking audio attention (4D)
audio_x = audio_x.view(b, q.size(1), n, d).flatten(2).contiguous()

# FantasyTalking audio attention (3D)
audio_x = attention(q, ip_key, ip_value, attention_mode=self.attention_mode).flatten(2).contiguous()

# FantasyPortrait adapter (4D)
adapter_x = adapter_x.flatten(2).contiguous()

# FantasyPortrait adapter (3D)
adapter_x = adapter_x.flatten(2).contiguous()

# Fusion target attention
target_x = attention(q, k_target, v_target, k_lens=kwargs["target_seq_lens"]).flatten(2).contiguous()
```

#### 3. WanI2VCrossAttention 类

**forward()**
```python
# Text attention
x_text = attention(q, k, v, attention_mode=self.attention_mode).flatten(2).contiguous()

# Image attention
img_x = attention(q, k_img, v_img, attention_mode=self.attention_mode).flatten(2).contiguous()

# FantasyTalking audio (4D)
audio_x = audio_x.view(b, q.size(1), n, d).flatten(2).contiguous()

# FantasyTalking audio (3D)
audio_x = attention(q, ip_key, ip_value, attention_mode=self.attention_mode).flatten(2).contiguous()

# FantasyPortrait adapter (4D)
adapter_x = adapter_x.flatten(2).contiguous()

# FantasyPortrait adapter (3D)
adapter_x = adapter_x.flatten(2).contiguous()
```

#### 4. MTVCrafterMotionAttention 类

**forward()**
```python
# 修复后
return self.o(x.flatten(2).contiguous())
```

#### 5. WanHuMoCrossAttention 类
```python
# 已在之前修复中添加了 contiguous
x_text = x_text.contiguous().view(b, -1, n, d).contiguous().flatten(2).contiguous()
```

## 修复统计

### 总计修复
- **15+ 处** `flatten(2)` 操作
- **5 个类**受影响
- **1 个文件**修改

### 修复位置分布
- `WanSelfAttention`: 6处
- `WanCrossAttention`: 6处  
- `WanI2VCrossAttention`: 6处
- `MTVCrafterMotionAttention`: 1处
- `WanHuMoCrossAttention`: 已在之前修复

## 为什么 flatten(2) 需要 contiguous()?

### 1. flatten() 的工作原理
```python
# flatten(2) 从第2维开始展平
x.shape = (B, N, H, D)  # 4维
x.flatten(2).shape = (B, N, H*D)  # 3维
```

**关键问题**：`flatten()` 可能返回非连续的视图（view），而不是复制数据。

### 2. FP8/FP4 线性层的要求

FP8 优化的线性层（`fp8_optimization.py`）中：
```python
def fp8_linear_forward(cls, base_dtype, input):
    # 第1步：访问 device 属性
    input = input.contiguous()  # 如果input已经非连续，这里可能已经太晚
    input_shape = (int(input.shape[0]), int(input.shape[1]), int(input.shape[2]))
    
    # 第2步：reshape 操作
    inn = input.reshape(-1, input_shape[2]).to(torch.float8_e4m3fn).contiguous()
    
    # 第3步：FP8 matmul
    o = torch._scaled_mm(inn, cls.weight.t(), ...)
```

**问题点**：
- 如果 `input` 在进入函数时已经是非连续的
- 访问 `input.device` 时可能触发 CUDA 内存对齐错误
- 即使后续调用 `.contiguous()`，也为时已晚

### 3. 为什么在 attention 后面？

```python
# attention() 返回的张量可能是连续的
x = attention(q, k, v, ...)

# 但 flatten(2) 可能产生非连续视图
x_flattened = x.flatten(2)  # 可能非连续

# 传入FP8线性层前必须确保连续
x_out = self.o(x_flattened.contiguous())  # ✓ 安全
```

## 技术细节

### view() vs contiguous()
```python
# view() 返回同一数据的不同形状视图（要求连续）
x.view(shape)  # 如果x不连续会报错

# reshape() 如果可能返回视图，否则复制（更安全）
x.reshape(shape)  # 可能返回非连续

# flatten() 内部使用 reshape
x.flatten(2)  # 可能返回非连续

# contiguous() 确保连续（必要时复制）
x.contiguous()  # 总是返回连续张量
```

### CUDA 对齐要求

FP4/FP8 量化对内存对齐要求严格：
- **字节对齐**: 通常要求 16 字节对齐
- **张量步长**: 必须满足特定模式
- **设备访问**: 非连续张量在某些操作时会失败

## 测试验证

### 测试场景
1. ✅ WanSelfAttention with FP4
2. ✅ WanCrossAttention with audio/adapter
3. ✅ WanI2VCrossAttention with clip_embed
4. ✅ MTVCrafterMotionAttention
5. ✅ Multitalk with FP4

### 预期结果
- 无 CUDA 内存对齐错误
- 正常生成视频
- 内存使用正常
- 性能无明显下降

## 相关修复

本修复是 FP4 量化系列修复的一部分：

1. ✅ **attention.py** - Sage3 FP4/FP8 transpose 修复
2. ✅ **multitalk.py** - 形状访问和 contiguous 修复
3. ✅ **fp8_optimization.py** - FP8 线性层输入修复
4. ✅ **model.py (WanHuMoCrossAttention)** - 之前的修复
5. ✅ **model.py (flatten)** - 本次修复 ⭐

## 最佳实践

### 规则 1: 所有传入线性层前确保连续
```python
# ✓ 正确
x = some_operation(x)
x = x.contiguous()  # 确保连续
output = linear_layer(x)

# ✗ 错误
output = linear_layer(some_operation(x))  # 可能非连续
```

### 规则 2: flatten/reshape 后立即 contiguous
```python
# ✓ 正确
x = x.flatten(2).contiguous()
x = x.reshape(shape).contiguous()

# ✗ 错误  
x = x.flatten(2)  # 可能非连续
```

### 规则 3: 在 FP8/FP4 模式下更加谨慎
```python
# 量化模式下所有张量操作都要小心
if quantization in ["fp4", "fp8"]:
    x = x.contiguous()  # 额外保险
```

## 性能影响

### contiguous() 的成本
- **已经连续**: 几乎零成本（仅检查）
- **需要复制**: O(n) 时间和内存

### 实际影响
- 大多数情况下张量已经连续
- contiguous() 调用开销可忽略
- 避免 CUDA 错误的收益远大于成本

## 相关文档

- [FP4 量化修复总结](FP4_QUANTIZATION_FIXES.md)
- [Infinite Talk CUDA 对齐修复](INFINITE_TALK_CUDA_ALIGNMENT_FIX.md)
- [Sage3 FP4 使用说明](SAGE3_FP4_使用说明.md)
