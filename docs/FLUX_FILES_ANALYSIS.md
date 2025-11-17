# Flux 文件分析与合并方案

## 当前文件列表

### apps/sd_module/ 下的 Flux 文件

1. **flux_text2img.py** (16.8 KB)
   - 用途: 占位符实现，基于 ComfyUI 工作流逻辑
   - 依赖: ComfyUI (但实际是占位符)
   - 状态: ⚠️ 旧版，不推荐使用

2. **flux_pipeline_real.py** (16.9 KB)
   - 用途: 尝试使用复制的 ComfyUI 节点
   - 依赖: custom_nodes/Comfyui/flux 中的 ComfyUI 节点
   - 状态: ⚠️ 依赖复杂，不完整

3. **flux_standalone.py** (12.0 KB)
   - 用途: ✅ **独立实现**，使用 diffusers 库
   - 依赖: diffusers, transformers (标准库)
   - 状态: ✅ **推荐使用**

4. **flux_gradio_ui.py** (20.7 KB)
   - 用途: Gradio UI，尝试使用 flux_pipeline_real
   - 依赖: flux_pipeline_real 或 flux_text2img
   - 状态: ⚠️ 依赖旧版实现

5. **flux_gradio_standalone.py** (12.6 KB)
   - 用途: ✅ **独立 Gradio UI**，使用 flux_standalone
   - 依赖: flux_standalone (独立实现)
   - 状态: ✅ **推荐使用**

## 文件关系图

```
旧版实现 (ComfyUI 依赖):
┌─────────────────────┐
│ flux_text2img.py    │ ← 占位符
└─────────────────────┘
          ↓
┌─────────────────────┐
│flux_pipeline_real.py│ ← 依赖 ComfyUI 节点
└─────────────────────┘
          ↓
┌─────────────────────┐
│ flux_gradio_ui.py   │ ← UI (旧版)
└─────────────────────┘

新版实现 (独立):
┌─────────────────────┐
│ flux_standalone.py  │ ← 独立实现 ✅
└─────────────────────┘
          ↓
┌──────────────────────────┐
│flux_gradio_standalone.py │ ← UI (新版) ✅
└──────────────────────────┘
```

## 合并方案

### 方案 1: 保留独立版本 (推荐)

**保留文件:**
- ✅ `flux_standalone.py` - 核心实现
- ✅ `flux_gradio_standalone.py` - UI

**删除文件:**
- ❌ `flux_text2img.py` - 占位符，无实际功能
- ❌ `flux_pipeline_real.py` - 依赖复杂，不完整
- ❌ `flux_gradio_ui.py` - 依赖旧版

**优点:**
- 完全独立，无 ComfyUI 依赖
- 代码清晰，易于维护
- 真实推理功能

**缺点:**
- 需要安装 diffusers 等依赖

### 方案 2: 合并为单文件

将 `flux_standalone.py` 和 `flux_gradio_standalone.py` 合并为一个文件

**优点:**
- 文件更少
- 部署简单

**缺点:**
- 文件较大
- 不利于模块化

## Python 版本分析

### 当前环境
```
Python 3.12.9
```

### 依赖检查

1. **系统 Python**: Python 3.12.9
2. **项目是否使用独立 Python313**: 需要检查

让我检查项目中是否有 Python313 相关配置...

## 推荐方案

### 立即执行: 清理旧文件

```bash
# 删除旧版文件
rm apps/sd_module/flux_text2img.py
rm apps/sd_module/flux_pipeline_real.py
rm apps/sd_module/flux_gradio_ui.py

# 保留独立版本
# apps/sd_module/flux_standalone.py ✅
# apps/sd_module/flux_gradio_standalone.py ✅
```

### 重命名为标准名称

```bash
# 重命名为更简洁的名称
mv apps/sd_module/flux_standalone.py apps/sd_module/flux_pipeline.py
mv apps/sd_module/flux_gradio_standalone.py apps/sd_module/flux_ui.py
```

### 更新 __init__.py

在 `apps/sd_module/__init__.py` 中只保留对新版的引用:

```python
# Flux (独立实现)
def create_flux_tab():
    """Create Flux generation tab"""
    try:
        from apps.sd_module.flux_ui import FluxStandaloneUI
        
        flux_ui = FluxStandaloneUI()
        return flux_ui.create_ui()
    except Exception as e:
        print(f"Failed to create Flux tab: {e}")
        return None
```

## Python 版本使用

### 检查项目 Python 配置

需要检查以下文件:
- `.python-version`
- `pyproject.toml`
- `setup.py`
- 启动脚本中的 Python 路径

### 当前使用

项目当前使用系统 Python 3.12.9，不是独立的 Python313。

如果需要使用特定 Python 版本:

```bash
# 方式 1: 使用虚拟环境
python -m venv venv
venv\Scripts\activate

# 方式 2: 指定 Python 路径
C:\path\to\python313\python.exe script.py
```

## 总结

### 当前状态
- ✅ 有 5 个 Flux 相关文件
- ✅ 其中 2 个是独立实现 (推荐)
- ⚠️ 其中 3 个是旧版 (可删除)
- ✅ 使用系统 Python 3.12.9

### 建议操作
1. **删除旧文件** (flux_text2img.py, flux_pipeline_real.py, flux_gradio_ui.py)
2. **保留独立版本** (flux_standalone.py, flux_gradio_standalone.py)
3. **可选: 重命名** 为更简洁的名称
4. **更新文档** 和启动脚本

### Python 版本
- 当前使用 Python 3.12.9 (系统 Python)
- 无需特定 Python313
- diffusers 支持 Python 3.8+
