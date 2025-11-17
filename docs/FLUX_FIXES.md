# Flux 生成修复记录

## 🐛 已修复的问题

### 1. VAEDecode 参数顺序错误

**错误信息:**
```
ERROR: 'dict' object has no attribute 'decode'
```

**原因:**
VAEDecode.decode() 的参数顺序是 `(vae, samples)`，但代码中使用了 `(samples, vae)`

**修复:**
```python
# 修复前
images_tensor = decoder.decode(samples, self.vae)[0]

# 修复后
images_tensor = decoder.decode(self.vae, latent_samples)[0]
```

**文件:** `apps/sd_module/flux_comfy_pipeline.py` 第 352 行

---

### 2. Tensor 需要 detach

**错误信息:**
```
ERROR: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
```

**原因:**
从 VAE 解码出来的 tensor 还在计算图中（requires_grad=True），不能直接转换为 numpy

**修复:**
```python
# 修复前
img_np = img_tensor.cpu().numpy()

# 修复后
img_np = img_tensor.detach().cpu().numpy()
```

**文件:** `apps/sd_module/flux_comfy_pipeline.py` 第 358 行

---

## ✅ 完整的解码流程

```python
# 1. 采样得到 latent
samples = sampler.sample(
    self.model,
    "enable",
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive_cond,
    negative_cond,
    latent,
    start_step,
    end_step,
    "disable"
)[0]

# 2. 确保 latent 是字典格式
if isinstance(samples, dict):
    latent_samples = samples
else:
    latent_samples = {"samples": samples}

# 3. 解码（注意参数顺序）
decoder = VAEDecode()
images_tensor = decoder.decode(self.vae, latent_samples)[0]

# 4. 转换为 PIL 图像（注意 detach）
images = []
for img_tensor in images_tensor:
    img_np = img_tensor.detach().cpu().numpy()  # detach() 很重要！
    img_np = (img_np * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    images.append(img_pil)
```

---

## 🔍 技术细节

### VAEDecode 方法签名

```python
class VAEDecode:
    def decode(self, vae, samples):
        """
        Args:
            vae: VAE 模型对象
            samples: 字典格式 {"samples": tensor}
        
        Returns:
            (images_tensor,) - 解码后的图像 tensor
        """
        images = vae.decode(samples["samples"])
        return (images,)
```

### Tensor 梯度管理

PyTorch 中的 tensor 如果参与了计算图（requires_grad=True），需要：

1. **detach()**: 从计算图中分离
2. **cpu()**: 移到 CPU（如果在 GPU 上）
3. **numpy()**: 转换为 numpy 数组

正确顺序：
```python
tensor.detach().cpu().numpy()
```

---

## 🧪 测试验证

运行测试脚本：
```bash
python313\python.exe scripts\test_flux_decode.py
```

预期输出：
```
✓ VAEDecode.decode 参数顺序正确: (vae, samples)
✓ 正确识别字典格式
✓ flux_comfy_pipeline 导入成功
✓ ComfyUI 可用
```

---

## 🚀 使用方法

1. 启动 UI:
   ```bash
   start.bat
   ```

2. 访问: http://localhost:7860

3. 导航: 主界面 > 文生图 > Flux

4. 选择模型:
   - UNET: `flux1-krea-dev_fp8_scaled.safetensors`
   - CLIP 1: `video\models_t5_umt5-xxl-enc-fp8_fully_uncensored.safetensors`
   - CLIP 2: `clip_l.safetensors`
   - VAE: `ae.safetensors`

5. 点击"加载模型"

6. 输入提示词并生成

---

## 📝 相关文件

- **管道**: `apps/sd_module/flux_comfy_pipeline.py`
- **UI**: `apps/sd_module/flux_integrated.py`
- **测试**: `scripts/test_flux_decode.py`
- **文档**: `docs/FLUX_COMFYUI_FORMAT.md`

---

## ✅ 状态

- [x] VAEDecode 参数顺序修复
- [x] Tensor detach 修复
- [x] 测试验证通过
- [x] 文档更新

**现在 Flux 生成应该可以正常工作了！** 🎉
