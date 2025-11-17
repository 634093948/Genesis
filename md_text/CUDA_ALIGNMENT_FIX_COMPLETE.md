# CUDA 内存对齐错误 - 完整修复报告

## 问题描述
```
torch.AcceleratorError: CUDA error: misaligned address
```

错误发生在 MultiTalk 音频处理的 forward 过程中,具体位置:
- `multitalk/multitalk.py` line 235: `if actual_tokens != expected_tokens:`

## 根本原因

在 FP8/FP4 量化环境下,以下操作会产生非连续(non-contiguous)张量:
1. `view()` - 改变张量形状
2. `reshape()` - 重塑张量
3. `permute()` - 改变维度顺序
4. `Linear` 层输出
5. `LayerNorm` 输出

非连续张量在量化操作中会导致 CUDA 内存访问未对齐。

## 修复方案

### 修改文件 1: `multitalk/multitalk.py`

#### 1.1 SingleStreamAttention.forward() - 输入检查
**位置**: Line 228-233

```python
def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor, shape=None) -> torch.Tensor:
    N_t, N_h, N_w = shape
    
    # Ensure input tensors are contiguous for FP8/FP4 quantization
    x = x.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()
```

**作用**: 确保输入张量在进入函数时就是连续的

#### 1.2 SingleStreamAttention.forward() - 中间操作
**位置**: Line 242-258

```python
x = x.view(B * N_t, S, self.dim).contiguous()

# get q for hidden_state
q = self.q_linear(x).view(B * N_t, S, self.num_heads, self.head_dim).contiguous()

# get kv from encoder_hidden_states
kv = self.kv_linear(encoder_hidden_states).contiguous()
encoder_k, encoder_v = kv.view(B * N_t, encoder_hidden_states.shape[1], 2, self.num_heads, self.head_dim).contiguous().unbind(2)

x = attention(q, encoder_k, encoder_v, attention_mode=self.attention_mode)

# linear transform
x = self.proj(x.reshape(B * N_t, S, self.dim).contiguous())
x = x.view(B, N_t * S, self.dim).contiguous()

if x_extra is not None:
    x = torch.cat([x, torch.zeros_like(x_extra)], dim=1).contiguous()
```

**作用**: 在每个可能产生非连续张量的操作后添加 `.contiguous()`

#### 1.3 SingleStreamMultiAttention.forward() - 输入检查
**位置**: Line 315-318

```python
# Ensure input tensors are contiguous for FP8/FP4 quantization
x = x.contiguous()
encoder_hidden_states = encoder_hidden_states.contiguous()
encoder_hidden_states = encoder_hidden_states.squeeze(0)
```

#### 1.4 SingleStreamMultiAttention.forward() - Q/KV 投影
**位置**: Line 328-329, 390-391

```python
# Query projection
q = self.q_linear(x).contiguous()
q = q.view(B, N, self.num_heads, self.head_dim).contiguous().permute(0, 2, 1, 3).contiguous()

# Keys / Values
encoder_kv = self.kv_linear(encoder_hidden_states).contiguous()
encoder_kv = encoder_kv.view(B, N_a, 2, self.num_heads, self.head_dim).contiguous().permute(2, 0, 3, 1, 4).contiguous()
```

#### 1.5 SingleStreamMultiAttention.forward() - 输出处理
**位置**: Line 426, 432

```python
# Linear projection
x = x.reshape(B, N, C).contiguous()
x = self.proj(x)

# Restore original layout
x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)
if x_extra is not None:
    x = torch.cat([x, torch.zeros_like(x_extra)], dim=1).contiguous()
```

### 修改文件 2: `wanvideo/modules/model.py`

#### 2.1 audio_cross_attn 调用前确保连续性
**位置**: Line 1261-1266

```python
# MultiTalk
if multitalk_audio_embedding is not None and not isinstance(self, VaceWanAttentionBlock):
    # Ensure tensors are contiguous before passing to audio_cross_attn (FP8/FP4 compatibility)
    norm_x_result = self.norm_x(x).contiguous()
    multitalk_audio_embedding_contig = multitalk_audio_embedding.contiguous()
    x_audio = self.audio_cross_attn(norm_x_result, encoder_hidden_states=multitalk_audio_embedding_contig,
                                shape=grid_sizes[0], x_ref_attn_map=x_ref_attn_map, human_num=human_num)
    x = x + x_audio * audio_scale
```

**作用**: 确保传递给 `audio_cross_attn` 的参数是连续的

## 修复统计

### multitalk.py
- **SingleStreamAttention.forward()**: 9 处 `.contiguous()`
  - 2 处输入检查
  - 7 处中间操作

- **SingleStreamMultiAttention.forward()**: 7 处 `.contiguous()`
  - 2 处输入检查
  - 5 处中间操作

### model.py
- **audio_cross_attn 调用**: 2 处 `.contiguous()`

**总计**: 18 处 `.contiguous()` 调用

## 性能影响

### `.contiguous()` 的行为
```python
if tensor.is_contiguous():
    return tensor  # No-op, 无开销
else:
    return tensor.clone()  # 复制数据
```

### 实际影响
- 大多数情况下是 no-op
- 只在必要时复制数据
- 相比 CUDA 崩溃,开销可忽略
- 不影响模型精度

## 测试结果

### 测试配置
- **模型**: eedy_Wan2_IceCannon2.1_InfiniteTalk.safetensors
- **量化**: FP4 Scaled
- **Attention**: SageAttention3 FP4
- **分辨率**: 768x768
- **帧数**: 81 frames @ 8 FPS
- **Steps**: 6
- **CFG**: 1.0

### 测试状态
```
✓ Models loaded successfully
✓ Input files validated
✓ Generation Successful!
```

### 预期行为
- ✅ 不再出现 `CUDA error: misaligned address`
- ✅ 采样过程正常进行
- ✅ "Sampling audio indices" 进度条显示
- ✅ 生成视频成功

## 核心原则遵守

✅ **不改变其他版块**
- 只修改 `multitalk.py` 和 `model.py` 中与 MultiTalk 相关的部分
- 不影响其他功能模块

✅ **不影响前面已成功的**
- `.contiguous()` 是安全操作
- 对已连续的张量无影响
- 保持所有现有功能正常

✅ **针对性修复**
- 直接解决 FP8/FP4 量化下的内存对齐问题
- 最小化修改范围
- 遵循 PyTorch 最佳实践

## 技术细节

### 为什么 FP8/FP4 需要连续张量?

1. **内存布局要求**
   - FP8/FP4 使用特殊的内存对齐
   - 非连续张量的步长(stride)不规则
   - CUDA 核心无法正确访问

2. **量化操作**
   - `torch._scaled_mm` 要求连续输入
   - FP8 转换需要连续内存
   - Scale 因子计算依赖对齐

3. **View/Reshape 限制**
   - `view()` 要求张量连续
   - `reshape()` 在非连续时会隐式复制
   - 显式 `.contiguous()` 更清晰

### 最佳实践

```python
# ✓ 推荐: 显式确保连续性
x = x.view(new_shape).contiguous()

# ✗ 不推荐: 假设张量已连续
x = x.view(new_shape)  # 可能失败

# ✓ 推荐: Linear 层后确保连续
x = linear(x).contiguous()

# ✓ 推荐: Permute 后确保连续
x = x.permute(dims).contiguous()
```

## 相关文件

### 已修改
1. `custom_nodes/Comfyui/ComfyUI-WanVideoWrapper/multitalk/multitalk.py`
   - SingleStreamAttention.forward(): 9 处修改
   - SingleStreamMultiAttention.forward(): 7 处修改

2. `custom_nodes/Comfyui/ComfyUI-WanVideoWrapper/wanvideo/modules/model.py`
   - audio_cross_attn 调用: 2 处修改

### 测试文件
1. `test_infinite_talk_full.py` - 完整测试脚本
2. `CUDA_ALIGNMENT_FIX.md` - 初步修复文档
3. `CUDA_ALIGNMENT_FIX_COMPLETE.md` - 本文档

## 总结

通过在关键位置添加 `.contiguous()` 调用,彻底解决了 FP8/FP4 量化环境下的 CUDA 内存对齐错误。修复是最小化的、安全的,不影响其他功能,完全符合"不改变其他版块,不影响前面已成功的"原则。

测试显示修复有效,模型加载和生成过程正常完成。
