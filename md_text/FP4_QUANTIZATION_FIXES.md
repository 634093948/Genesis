# FP4 量化修复总结

## 问题概述

Infinite Talk 模块在使用 Sage3 FP4/FP8 量化时出现 CUDA 内存对齐错误，导致生成失败。

## 根本原因分析

### 1. CUDA 内存对齐问题

**症状**：
```
torch.AcceleratorError: CUDA error: misaligned address
```

**原因**：
- FP4/FP8 量化对内存对齐要求极为严格
- `transpose()`, `view()`, `reshape()` 等操作可能产生非连续内存布局
- 直接访问 CUDA 张量的 `.shape` 属性可能触发异步错误
- 在量化计算前未确保张量连续性

### 2. 模块量化配置问题

**症状**：
- multitalk 模块在量化后可能行为异常
- 某些应该保留的层被错误量化

**原因**：
- `params_to_keep` 配置不完整
- multitalk 相关模块未被正确保护

## 修复方案

### 修复 1: attention.py - Sage3 注意力层内存对齐

**文件**：`wanvideo/modules/attention.py`

**修改内容**：
```python
# sageattn_3 模式
q_contig = q.contiguous().transpose(1,2).contiguous()
k_contig = k.contiguous().transpose(1,2).contiguous()
v_contig = v.contiguous().transpose(1,2).contiguous()
return sageattn_blackwell(q_contig, k_contig, v_contig, per_block_mean=True).transpose(1,2).contiguous()

# sageattn_3_fp4 模式 (关键修复)
q_contig = q.contiguous().transpose(1,2).contiguous()
k_contig = k.contiguous().transpose(1,2).contiguous()
v_contig = v.contiguous().transpose(1,2).contiguous()
return sageattn_blackwell(q_contig, k_contig, v_contig, per_block_mean=True).transpose(1,2).contiguous()

# sageattn_3_fp8 模式 (关键修复)
q_contig = q.contiguous().transpose(1,2).contiguous()
k_contig = k.contiguous().transpose(1,2).contiguous()
v_contig = v.contiguous().transpose(1,2).contiguous()
return sageattn_blackwell(q_contig, k_contig, v_contig, per_block_mean=True).transpose(1,2).contiguous()
```

**关键点**：
- 在 transpose 前后都调用 `.contiguous()`
- 确保传入 sageattn_blackwell 的张量内存连续
- 输出结果也确保连续

### 修复 2: multitalk.py - MultiTalk 模块内存对齐

**文件**：`multitalk/multitalk.py`

**修改内容**：

1. **SingleStreamAttention.forward()**
```python
# 安全的形状提取
expected_tokens = int(N_t * N_h * N_w)
actual_tokens = int(x.shape[1])

# 切片后确保连续
x_extra = x[:, -N_h * N_w:, :].contiguous()
x = x[:, :-N_h * N_w, :].contiguous()

# 形状值转换为 Python int
B = int(x.shape[0])
S = int(N_h * N_w)
enc_seq_len = int(encoder_hidden_states.shape[1])
```

2. **SingleStreamMultiAttention.forward()**
```python
# 安全的形状比较
x_dim0 = int(x.shape[0])
enc_dim0 = int(encoder_hidden_states.shape[0])
if x_dim0 * N_t != enc_dim0:
    ...

# 维度值转换
B, N, C = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
_, N_a, _ = int(encoder_hidden_states.shape[0]), int(encoder_hidden_states.shape[1]), int(encoder_hidden_states.shape[2])
```

3. **其他函数**
```python
# calculate_x_ref_attn_map
scale = 1.0 / float(visual_q.shape[-1]) ** 0.5

# AudioMultiProj.forward
video_length = int(audio_embeds.shape[1]) + int(audio_embeds_vf.shape[1])
B, _, _, S, C = int(audio_embeds.shape[0]), ...
```

### 修复 3: fp8_optimization.py - FP8 线性层

**文件**：`fp8_optimization.py`

**修改内容**：
```python
def fp8_linear_forward(cls, base_dtype, input):
    # 确保输入连续
    input = input.contiguous()
    input_shape = (int(input.shape[0]), int(input.shape[1]), int(input.shape[2]))
    
    # 操作后确保连续
    input = torch.clamp(input, min=-448, max=448, out=input).contiguous()
    inn = input.reshape(-1, input_shape[2]).to(torch.float8_e4m3fn).contiguous()
    
    # 输出确保连续
    return o.reshape((-1, input_shape[1], cls.weight.shape[0])).contiguous()
```

### 修复 4: model.py - WanHuMoCrossAttention

**文件**：`wanvideo/modules/model.py`

**修改内容**：

1. **WanHuMoCrossAttention.forward()**
```python
b, n, d = int(x.size(0)), self.num_heads, self.head_dim
q = self.norm_q(self.q(x)).contiguous().view(b, -1, n, d).contiguous()
k = self.norm_k(self.k(context)).contiguous().view(b, -1, n, d).contiguous()
v = self.v(context).contiguous().view(b, -1, n, d).contiguous()

hlen_wlen = int(grid_sizes[0][1]) * int(grid_sizes[0][2])
q = q.reshape(-1, hlen_wlen, n, d).contiguous()
k = k.reshape(-1, 16, n, d).contiguous()
v = v.reshape(-1, 16, n, d).contiguous()

x_text = attention(q, k, v, attention_mode=self.attention_mode)
x_text = x_text.contiguous().view(b, -1, n, d).contiguous().flatten(2).contiguous()
```

2. **AudioCrossAttentionWrapper.forward()**
```python
x = x.contiguous()
audio = audio.contiguous()
normed_x = self.norm1_audio(x).contiguous()
attn_result = self.audio_cross_attn(normed_x, audio, grid_sizes).contiguous()
x = x + attn_result * humo_audio_scale
return x.contiguous()
```

### 修复 5: nodes_model_loading.py - 量化参数配置

**文件**：`nodes_model_loading.py`

**修改内容**：
```python
params_to_keep = {
    "norm", "bias", "time_in", "patch_embedding", "time_", 
    "img_emb", "modulation", "text_embedding", "adapter", 
    "add", "ref_conv", "audio_proj", "multitalk_audio_proj"
}
```

**说明**：
- `"multitalk_audio_proj"`: 保护 multitalk 音频投影层不被量化
- `"norm"`: 保护所有归一化层（包括 norm_x）
- `"audio_proj"`: 保护通用音频投影层
- **允许 audio_cross_attn 的线性层被量化**，因为它们使用相同的 attention_mode

## 技术要点

### 为什么需要 contiguous()?

1. **张量内存布局**
   - PyTorch 张量可能在内存中非连续存储
   - `transpose()`, `view()`, `reshape()` 可能改变内存布局
   - 非连续张量在 CUDA 操作时可能导致未对齐访问

2. **CUDA 内核要求**
   - 某些 CUDA 操作要求连续内存
   - FP4/FP8 量化对内存对齐更敏感
   - Sage3 Blackwell 加速器要求严格的内存对齐

3. **调用时机**
   ```python
   # 在 transpose 前后
   x.contiguous().transpose(1,2).contiguous()
   
   # 在 view/reshape 前后
   x.contiguous().view(b, -1, n, d).contiguous()
   
   # 在 flatten 后
   x.flatten(2).contiguous()
   ```

### int() 转换的作用

```python
# 不安全 - CUDA 张量上的形状访问可能触发异步错误
if x.shape[0] != expected:
    ...

# 安全 - 在 CPU 上进行比较
if int(x.shape[0]) != expected:
    ...
```

**原因**：
- 直接访问 CUDA 张量的 shape 可能在量化模式下触发异步错误
- `int()` 转换将值从 CUDA 传输到 CPU
- 在 CPU 上进行比较避免了 CUDA 同步问题

### params_to_keep 的工作原理

```python
for name, submodule in module.named_modules():
    if not any(keyword in name for keyword in params_to_keep):
        # 应用量化
```

- 如果模块名称包含 params_to_keep 中的关键词，跳过量化
- 这保护了关键层（如 norm, bias）不被量化
- multitalk_audio_proj 被保护，但 audio_cross_attn 可以被量化

## 测试验证

### 测试命令
```bash
python apps/wanvideo_module/wanvideo_gradio_app.py
```

### 测试配置
- **模式**: infinitetalk
- **量化**: fp4_scaled 或 fp4_scaled_fast
- **注意力模式**: sageattn_3_fp4
- **输入**: 测试音频 + 参考图像

### 预期结果
- ✅ 无 CUDA 内存对齐错误
- ✅ 正常生成视频
- ✅ 内存占用在预期范围内
- ✅ 生成质量正常

## 相关文件修改列表

1. ✅ `multitalk/multitalk.py` - MultiTalk 模块内存对齐
2. ✅ `wanvideo/modules/attention.py` - Sage3 注意力层
3. ✅ `fp8_optimization.py` - FP8 线性层优化
4. ✅ `wanvideo/modules/model.py` - 模型层内存对齐
5. ✅ `nodes_model_loading.py` - 量化参数配置

## 兼容性说明

### 支持的量化模式
- ✅ `fp4` - 基础 FP4 模式
- ✅ `fp4_scaled` - FP4 缩放模式
- ✅ `fp4_scaled_fast` - FP4 快速模式
- ✅ `fp8_e4m3fn` - FP8 e4m3fn 模式
- ✅ `fp8_e5m2` - FP8 e5m2 模式

### 支持的注意力模式
- ✅ `sageattn_3_fp4` - 推荐用于 FP4
- ✅ `sageattn_3_fp8` - 推荐用于 FP8
- ✅ `sageattn_3` - Sage3 Blackwell
- ✅ `sageattn` - 标准 Sage 注意力
- ✅ `sdpa` - PyTorch SDPA（回退）

### 硬件要求
- NVIDIA GPU with Compute Capability >= 9.0 (Blackwell) for FP4
- NVIDIA GPU with Compute Capability >= 8.0 (Ampere) for FP8
- CUDA 12.0+ recommended

## 注意事项

1. **必须配对使用**
   - FP4 量化 → sageattn_3_fp4 注意力模式
   - FP8 量化 → sageattn_3_fp8 注意力模式

2. **内存对齐关键**
   - 所有 transpose/view/reshape 操作确保 contiguous
   - 形状访问使用 int() 转换

3. **不影响其他功能**
   - 修复仅针对量化模式
   - 非量化模式不受影响
   - 其他模块功能正常

## 相关文档

- [Infinite Talk CUDA 对齐修复](INFINITE_TALK_CUDA_ALIGNMENT_FIX.md)
- [Sage3 FP4 使用说明](SAGE3_FP4_使用说明.md)
- [Sage3 FP4 修复总结](SAGE3_FP4_修复总结.md)
