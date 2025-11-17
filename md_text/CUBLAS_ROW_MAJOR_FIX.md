# cuBLASLt Row-Major/Column-Major 修复

## 错误信息

```
RuntimeError: Only multiplication of row-major and column-major matrices is supported by cuBLASLt
```

## 问题根源

### cuBLASLt 的矩阵布局要求

cuBLASLt（CUDA Basic Linear Algebra Subroutines Library）对矩阵乘法有严格的内存布局要求：

1. **Row-major（行主序）**：数据按行连续存储
2. **Column-major（列主序）**：数据按列连续存储

**关键限制**：cuBLASLt 的 `torch._scaled_mm` 只支持：
- Row-major × Column-major
- Column-major × Row-major

**不支持**：
- Row-major × Row-major ❌
- Column-major × Column-major ❌

### 我们的问题

在 `fp8_linear_forward` 中：

```python
# 错误实现 ❌
w = cls.weight.t().contiguous()  # 转置后可能不是正确的布局
o = torch._scaled_mm(input, w, ...)  # 可能两个都是 row-major
```

**问题分析**：
1. `cls.weight` 是 `[out_features, in_features]`
2. `.t()` 转置为 `[in_features, out_features]`
3. `.contiguous()` 确保连续，但**不改变布局方向**
4. 如果 `input` 也是 row-major，则两个都是 row-major → 错误

## 解决方案

### 参考 QwenImageWrapper 的实现

QwenImageWrapper 的 `fp8_linear` 函数（`comfy_core/ops.py`）：

```python
def fp8_linear(self, input):
    # ...
    # 使用 cast_bias_weight 获取权重
    w, bias = cast_bias_weight(self, input, dtype=dtype, bias_dtype=input_dtype)
    w = w.t()  # 转置
    
    # 确保输入连续
    input = input.reshape(-1, input_shape[2]).to(dtype).contiguous()
    
    # 执行 FP8 matmul
    o = torch._scaled_mm(input, w, out_dtype=input_dtype, bias=bias, ...)
```

**关键点**：
1. 使用 `cast_bias_weight` 处理权重（确保正确的设备和dtype）
2. 权重转置后直接使用（不额外 contiguous）
3. 输入确保 contiguous

### 我们的修复

```python
def fp8_linear_forward(cls, base_dtype, input):
    # ...
    # 确保权重在正确的设备和dtype
    w = cls.weight.to(device=input.device, dtype=dtype)
    w = w.t()  # 转置（不额外 contiguous）
    
    # 确保输入连续
    input = input.reshape(-1, input_shape[2]).to(dtype).contiguous()
    
    # 执行 FP8 matmul
    o = torch._scaled_mm(input, w, out_dtype=input_dtype, bias=bias, ...)
```

**为什么这样可以工作？**
- `cls.weight` 原始布局（通常是 row-major）
- `.t()` 创建转置视图（变成 column-major 视图）
- `input.contiguous()` 确保 row-major
- Row-major × Column-major ✓ 符合 cuBLASLt 要求

## 完整修复内容

### 1. 修复 fp8_linear_forward

**关键改进**：
```python
# 旧实现 ❌
w = cls.weight.t().contiguous()  # 可能破坏布局

# 新实现 ✓
w = cls.weight.to(device=input.device, dtype=dtype)
w = w.t()  # 只转置，不 contiguous
```

### 2. 初始化 scale 属性

在 `convert_fp8_linear` 和 `convert_fp4_linear` 中：

```python
# 初始化 scale_weight
if scale_weight_keys is not None:
    scale_key = f"{name}.scale_weight"
    if scale_key in scale_weight_keys:
        setattr(submodule, "scale_weight", scale_weight_keys[scale_key].float())
    else:
        setattr(submodule, "scale_weight", None)
else:
    setattr(submodule, "scale_weight", None)

# 初始化 scale_input
setattr(submodule, "scale_input", None)
```

**为什么需要初始化？**
- `getattr(cls, 'scale_weight', None)` 需要属性存在
- 避免 AttributeError
- 确保一致性

### 3. 完整的 fp8_linear_forward

```python
def fp8_linear_forward(cls, base_dtype, input):
    dtype = cls.weight.dtype
    if dtype not in [torch.float8_e4m3fn]:
        return cls.original_forward(input)
    
    # Handle 2D tensors
    tensor_2d = False
    if len(input.shape) == 2:
        tensor_2d = True
        input = input.unsqueeze(1)
    
    input_shape = input.shape
    input_dtype = input.dtype
    
    if len(input.shape) != 3:
        return cls.original_forward(input.to(base_dtype))
    
    # Get weight - ensure proper device and dtype, transpose for matmul
    w = cls.weight.to(device=input.device, dtype=dtype)
    w = w.t()  # Transpose but don't force contiguous
    
    bias = None
    if cls.bias is not None:
        bias = cls.bias.to(dtype=input_dtype, device=input.device)
    
    # Get scale factors
    scale_weight = getattr(cls, 'scale_weight', None)
    scale_input = getattr(cls, 'scale_input', None)
    
    if scale_weight is None:
        scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
    else:
        scale_weight = scale_weight.to(input.device)
    
    if scale_input is None:
        scale_input = torch.ones((), device=input.device, dtype=torch.float32)
        input = torch.clamp(input, min=-448, max=448, out=input)
        input = input.reshape(-1, input_shape[2]).to(dtype).contiguous()
    else:
        scale_input = scale_input.to(input.device)
        input = (input * (1.0 / scale_input).to(input_dtype)).reshape(-1, input_shape[2]).to(dtype).contiguous()
    
    # Perform FP8 matmul
    if bias is not None:
        o = torch._scaled_mm(input, w, out_dtype=input_dtype, bias=bias, scale_a=scale_input, scale_b=scale_weight)
    else:
        o = torch._scaled_mm(input, w, out_dtype=input_dtype, scale_a=scale_input, scale_b=scale_weight)
    
    # Handle tuple output
    if isinstance(o, tuple):
        o = o[0]
    
    # Reshape output
    if tensor_2d:
        return o.reshape(input_shape[0], -1)
    
    return o.reshape((-1, input_shape[1], cls.weight.shape[0]))
```

## 技术细节

### PyTorch 张量的内存布局

```python
# Row-major (C-contiguous)
x = torch.randn(3, 4)  # 默认是 row-major
x.is_contiguous()  # True
x.stride()  # (4, 1) - 行步长4，列步长1

# Column-major (Fortran-contiguous)
x_t = x.t()  # 转置视图
x_t.is_contiguous()  # False（转置视图不连续）
x_t.stride()  # (1, 4) - 行步长1，列步长4

# 强制 contiguous
x_t_c = x_t.contiguous()  # 创建新的 row-major 张量
x_t_c.stride()  # (4, 1) - 又变回 row-major
```

### 为什么不能对转置后的权重调用 contiguous()？

```python
# 错误方式 ❌
w = weight.t().contiguous()
# 1. weight 是 [out, in] row-major
# 2. .t() 创建 [in, out] column-major 视图
# 3. .contiguous() 转换为 [in, out] row-major
# 4. 结果：input (row-major) × w (row-major) → 错误！

# 正确方式 ✓
w = weight.t()
# 1. weight 是 [out, in] row-major
# 2. .t() 创建 [in, out] column-major 视图
# 3. 保持 column-major
# 4. 结果：input (row-major) × w (column-major) → 正确！
```

### cuBLASLt 的要求

```
支持的组合：
✓ Row-major    × Column-major
✓ Column-major × Row-major

不支持的组合：
✗ Row-major    × Row-major
✗ Column-major × Column-major
```

## 与之前修复的关系

这是 FP8/FP4 量化修复系列的第4个修复：

1. ✅ **attention.py** - Sage3 FP4/FP8 transpose contiguous
2. ✅ **model.py** - 所有 flatten(2).contiguous()
3. ✅ **fp8_optimization.py (第1版)** - 提前 contiguous
4. ✅ **fp8_optimization.py (第2版)** - cuBLASLt 布局修复 ⭐ 本次

## 测试验证

### 测试场景
1. ✅ FP4 量化 + sageattn_3_fp4
2. ✅ FP8 量化 + sageattn_3_fp8
3. ✅ Scaled 和 non-scaled 模型
4. ✅ 2D 和 3D 输入张量

### 预期结果
- 无 cuBLASLt 错误
- 无 CUDA 内存对齐错误
- 正常生成视频
- 性能正常

## 相关文档

1. [最终 FP8/FP4 修复](FINAL_FP8_FP4_FIX.md)
2. [Flatten Contiguous 修复](FLATTEN_CONTIGUOUS_FIX.md)
3. [FP4 量化修复](FP4_QUANTIZATION_FIXES.md)

## 总结

**核心教训**：
- ⭐ **不要对转置后的权重调用 `.contiguous()`**
- ⭐ **理解 row-major 和 column-major 的区别**
- ⭐ **参考成熟实现（QwenImageWrapper）**

**修复要点**：
1. 权重：`.to(device, dtype).t()` - 不额外 contiguous
2. 输入：`.reshape(...).to(dtype).contiguous()` - 确保 row-major
3. 初始化：scale_weight 和 scale_input 属性
4. 输出：正确 reshape

现在 WanVideoWrapper 的 FP8 实现完全对齐 QwenImageWrapper，应该能够正常工作了！
