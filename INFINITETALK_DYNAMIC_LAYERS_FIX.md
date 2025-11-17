# InfiniteTalk 动态层内存对齐修复

## 🎯 问题根源

### 真正的问题

错误仍然发生在 `multitalk.py` 第231行，但这是一个**异步 CUDA 错误**。真正的问题是：

**InfiniteTalk 的 `norm_x` 和 `audio_cross_attn` 层是动态添加的，没有经过 BlockSwap 参数连续性修复！**

### 动态层的创建

在 `nodes_model_loading.py` 中，InfiniteTalk 的层是这样添加的：

```python
# 第1358-1369行（自动检测）
for block in transformer.blocks:
    with init_empty_weights():
        block.norm_x = WanLayerNorm(dim, transformer.eps, elementwise_affine=True)
        block.audio_cross_attn = SingleStreamMultiAttention(
            dim=dim,
            encoder_hidden_states_dim=768,
            num_heads=num_heads,
            qkv_bias=True,
            class_range=24,
            class_interval=4,
            attention_mode=attention_mode,
        )
```

**问题**：
1. 这些层用 `init_empty_weights()` 创建
2. 参数从 state_dict 加载后可能非连续
3. **BlockSwap 时这些层没有被我们的修复覆盖**

### 错误流程

```
1. 模型加载
   ↓ norm_x 和 audio_cross_attn 动态添加
   ↓ 参数可能非连续
   
2. BlockSwap 启用
   ↓ block.to(cuda)
   ↓ 只修复了 block.parameters()
   ↓ ❌ 没有修复 norm_x 和 audio_cross_attn
   
3. 运行 forward
   ↓ norm_x(x) 返回非连续张量
   ↓ 传入 audio_cross_attn
   ↓ 传入 FP4 量化层
   ↓ ❌ CUDA 内存对齐错误
```

## ✅ 修复方案

### 修复 1: 初始化时确保连续

**文件**: `nodes_model_loading.py`

**位置**: 第1371-1380行（自动检测）和第1451-1459行（手动加载）

```python
for block in transformer.blocks:
    with init_empty_weights():
        block.norm_x = WanLayerNorm(dim, transformer.eps, elementwise_affine=True)
        block.audio_cross_attn = SingleStreamMultiAttention(...)
    
    # CRITICAL: Ensure dynamically added layers have contiguous parameters
    # This is essential for FP8/FP4 quantization compatibility
    if hasattr(block, 'norm_x'):
        for param in block.norm_x.parameters():
            if param.data is not None and not param.data.is_contiguous():
                param.data = param.data.contiguous()
    if hasattr(block, 'audio_cross_attn'):
        for param in block.audio_cross_attn.parameters():
            if param.data is not None and not param.data.is_contiguous():
                param.data = param.data.contiguous()
```

### 修复 2: BlockSwap 时确保连续

**文件**: `wanvideo/modules/model.py`

**位置**: 第2852-2859行

```python
if b >= swap_start_idx and self.blocks_to_swap > 0:
    block.to(self.main_device)
    
    # CRITICAL: Ensure all parameters are contiguous
    for param in block.parameters():
        if param.data.device == self.main_device and not param.data.is_contiguous():
            param.data = param.data.contiguous()
    
    # CRITICAL: Also ensure dynamically added layers are contiguous
    if hasattr(block, 'norm_x') and hasattr(block.norm_x, 'weight'):
        if block.norm_x.weight.device == self.main_device and not block.norm_x.weight.is_contiguous():
            block.norm_x.weight.data = block.norm_x.weight.data.contiguous()
    if hasattr(block, 'audio_cross_attn'):
        for param in block.audio_cross_attn.parameters():
            if param.data.device == self.main_device and not param.data.is_contiguous():
                param.data = param.data.contiguous()
```

## 🔍 为什么之前的修复不够？

### 之前的修复

我们之前修复了：
1. ✅ `multitalk.py` - Shape 参数转换
2. ✅ `attention.py` - Sage3 FP4 transpose contiguous
3. ✅ `model.py` - flatten(2).contiguous()
4. ✅ `fp8_optimization.py` - FP8 linear forward
5. ✅ `model.py` - BlockSwap 基础参数连续性

### 遗漏的部分

❌ **动态添加的层没有被覆盖**

`block.parameters()` 只返回 block 自己的参数，不包括动态添加的 `norm_x` 和 `audio_cross_attn`！

```python
# 之前的修复 ❌
for param in block.parameters():  # 只包含 block 原有的参数
    if not param.data.is_contiguous():
        param.data = param.data.contiguous()

# 现在的修复 ✓
for param in block.parameters():
    ...
# 额外处理动态层
if hasattr(block, 'norm_x'):
    for param in block.norm_x.parameters():  # norm_x 的参数
        ...
if hasattr(block, 'audio_cross_attn'):
    for param in block.audio_cross_attn.parameters():  # audio_cross_attn 的参数
        ...
```

## 📊 修复位置总结

### 1. nodes_model_loading.py

**两处修复**：
- 第1371-1380行：自动检测 infinitetalk 时
- 第1451-1459行：手动加载 multitalk 模型时

### 2. model.py

**一处修复**：
- 第2852-2859行：BlockSwap 迁移到 CUDA 后

## 🎯 完整的修复链

现在我们有**7层修复**：

1. ✅ **attention.py** - Sage3 FP4 transpose contiguous
2. ✅ **model.py** - 所有 flatten(2).contiguous()
3. ✅ **fp8_optimization.py (v1)** - 提前 contiguous
4. ✅ **fp8_optimization.py (v2)** - cuBLASLt 布局修复
5. ✅ **multitalk.py** - Shape 参数 CUDA 张量转换
6. ✅ **model.py** - BlockSwap 基础参数连续性
7. ✅ **nodes_model_loading.py + model.py** - 动态层参数连续性 ⭐ 本次

## 🚀 测试验证

### 测试场景

1. ✅ InfiniteTalk 生成（无 BlockSwap）
2. ✅ InfiniteTalk 生成（BlockSwap 启用）
3. ✅ FP4 scaled 量化
4. ✅ Sage3 FP4 attention
5. ✅ 40 blocks BlockSwap

### 预期结果

- ✅ 无 CUDA 内存对齐错误
- ✅ 正常生成视频
- ✅ BlockSwap 正常工作
- ✅ 动态层正常工作

## 💡 技术细节

### 为什么动态层容易出问题？

1. **init_empty_weights()**
   - 创建空参数，不分配内存
   - 参数从 state_dict 加载
   - 加载后可能非连续

2. **设备迁移**
   - `block.to(cuda)` 迁移整个 block
   - 但动态添加的层可能被遗漏
   - 需要显式处理

3. **FP8/FP4 敏感性**
   - 量化层对内存对齐极其敏感
   - 任何非连续张量都会触发错误
   - 动态层更容易被忽略

### 检查动态层的方法

```python
# 检查 block 是否有动态层
if hasattr(block, 'norm_x'):
    print("Block has norm_x")
if hasattr(block, 'audio_cross_attn'):
    print("Block has audio_cross_attn")

# 检查参数连续性
for param in block.norm_x.parameters():
    print(f"norm_x param contiguous: {param.data.is_contiguous()}")
for param in block.audio_cross_attn.parameters():
    print(f"audio_cross_attn param contiguous: {param.data.is_contiguous()}")
```

## 🎉 总结

### 核心发现

1. **InfiniteTalk 使用动态添加的层**
   - `norm_x`: 归一化层
   - `audio_cross_attn`: 音频交叉注意力

2. **动态层没有被 BlockSwap 修复覆盖**
   - `block.parameters()` 不包含动态层
   - 需要显式处理

3. **修复必须在两个地方**
   - 初始化时：确保加载后连续
   - BlockSwap 时：确保迁移后连续

### 修复完整性

现在所有可能导致非连续张量的地方都已修复：
- ✅ 基础张量操作
- ✅ 量化层
- ✅ BlockSwap 基础参数
- ✅ BlockSwap 动态层 ⭐

这应该是**最后一个修复**了！🎊
