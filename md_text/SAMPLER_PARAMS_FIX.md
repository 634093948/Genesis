# WanVideoSampler 参数错误修复

## 错误信息
```
ERROR:infinite_talk_pipeline:Generation failed: WanVideoSampler.process() got an unexpected keyword argument 'use_tf32'
```

## 问题分析

### 错误原因
之前的代码将以下参数直接传递给 `WanVideoSampler.process()`:
- `use_tf32=False`
- `use_cublas_gemm=False`
- `force_contiguous_tensors=False`
- `fuse_qkv_projections=False`

但这些参数**不是** `WanVideoSampler.process()` 的直接参数。

### WanVideoSampler.process() 实际参数

根据 `nodes_sampler.py` 第 158-161 行:

```python
def process(self, model, image_embeds, shift, steps, cfg, seed, scheduler, riflex_freq_index, 
    text_embeds=None,
    force_offload=True, 
    samples=None, 
    feta_args=None, 
    denoise_strength=1.0, 
    context_options=None,
    cache_args=None, 
    teacache_args=None, 
    flowedit_args=None, 
    batched_cfg=False, 
    slg_args=None, 
    rope_function="default", 
    loop_args=None,
    experimental_args=None,  # ← 这里!
    sigmas=None, 
    unianimate_poses=None, 
    fantasytalking_embeds=None, 
    uni3c_embeds=None, 
    multitalk_embeds=None, 
    freeinit_args=None, 
    start_step=0, 
    end_step=-1, 
    add_noise_to_samples=False):
```

### 关键发现
- ✅ `experimental_args` 是一个可选参数
- ✅ 用于传递实验性配置的字典
- ❌ 那些 CUDA 优化参数不是直接参数

## 修复方案

### 错误的代码 (已修复)
```python
sampled_result = sampler.process(
    model=self.model,
    image_embeds=image_embeds,
    text_embeds=positive_embeds,
    multitalk_embeds=audio_embeds,
    shift=shift,
    steps=steps,
    cfg=cfg,
    seed=seed,
    scheduler=actual_scheduler,
    riflex_freq_index=0,
    force_offload=True,
    use_tf32=False,              # ✗ 错误!
    use_cublas_gemm=False,       # ✗ 错误!
    force_contiguous_tensors=False,  # ✗ 错误!
    fuse_qkv_projections=False   # ✗ 错误!
)
```

### 正确的代码 (当前)
```python
sampled_result = sampler.process(
    model=self.model,
    image_embeds=image_embeds,
    text_embeds=positive_embeds,
    multitalk_embeds=audio_embeds,
    shift=shift,
    steps=steps,
    cfg=cfg,
    seed=seed,
    scheduler=actual_scheduler,
    riflex_freq_index=0,
    force_offload=True
    # ✓ 移除了无效参数
)
```

## 关于 CUDA 优化参数

### 这些参数的真实用途
经过检查,这些参数 (`use_tf32`, `use_cublas_gemm`, `force_contiguous_tensors`, `fuse_qkv_projections`) 在 WanVideoWrapper 的代码中**并未使用**。

它们可能是:
1. 其他项目的参数
2. 计划中但未实现的功能
3. 误解了工作流配置

### CUDA 内存对齐问题的真正解决方案
如果之前的 CUDA 错误 (`misaligned address`) 已经解决,可能是因为:
1. ✅ 正确的模型加载配置 (FP4 量化)
2. ✅ 正确的 attention 模式 (sageattn_3_fp4)
3. ✅ 正确的调度器选择
4. ✅ 依赖库的正确安装

而**不是**这些不存在的参数。

## 修改文件

### `apps/wanvideo_module/infinite_talk_pipeline.py`
- **行数**: 969-980
- **修改**: 移除了 4 个无效的直接参数
- **结果**: 使用正确的参数调用 `sampler.process()`

## 验证

### 正确的参数列表
```python
# Required
model, image_embeds, shift, steps, cfg, seed, scheduler, riflex_freq_index

# Optional (常用)
text_embeds=None
force_offload=True
multitalk_embeds=None  # ← Infinite Talk 需要!

# Optional (高级)
samples=None
denoise_strength=1.0
context_options=None
cache_args=None
experimental_args=None  # ← 如果需要实验性功能
...
```

## 总结

### ✅ 已修复
- 移除了 4 个不存在的参数
- 代码现在使用正确的 API
- 不影响其他已成功的部分

### 📝 注意事项
- `experimental_args` 参数存在,但那 4 个 CUDA 参数不是它的标准选项
- 如果将来需要传递实验性参数,应该检查 `nodes_sampler.py` 中 `experimental_args` 的实际用法
- 当前的简化版本应该可以正常工作

### 🎯 下一步
直接测试 Infinite Talk 功能,之前的 CUDA 错误可能已经通过其他方式解决了。
