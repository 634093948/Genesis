# Infinite Talk 完整工作流集成状态

## ✅ 已完成的工作

### 1. 节点包复制
- ✅ ComfyUI_Comfyroll_CustomNodes (SimpleMath+)
- ✅ ComfyLiterals (Int)

### 2. 节点导入集成
已在 `infinite_talk_pipeline.py` 中添加以下节点的导入逻辑：
- ✅ ImageResizeKJ (ComfyUI-KJNodes)
- ✅ AudioSeparation, AudioCrop (audio-separation-nodes-comfyui)
- ✅ AudioDuration (comfy-mtb)
- ✅ SimpleMathNode (ComfyUI_Comfyroll_CustomNodes)
- ✅ IntNode (ComfyLiterals)

### 3. Server Stub 系统
- ✅ 完整的 server_stub.py
- ✅ 完整的 latent_preview_standalone.py
- ✅ 在 pipeline 启动时自动注入，避免 genesis 依赖

### 4. 调度器修复
- ✅ multitalk 模式使用字符串 "multitalk" 而不是 dict
- ✅ 符合 WanVideoSampler 的预期输入格式

## 📋 下一步需要完成的工作

### 1. Pipeline 增强
需要在 `generate()` 方法中添加：

#### 图像预处理流程
```python
# 1. 使用 ImageResizeKJ 调整图像尺寸
if ImageResizeKJ and use_image_resize:
    resize_node = ImageResizeKJ()
    image = resize_node.resize(
        image=image,
        width=width,
        height=height,
        interpolation=resize_interpolation,  # 新UI参数
        method=resize_method,  # 新UI参数
        condition=resize_condition,  # 新UI参数
        multiple_of=8  # 确保是8的倍数
    )
```

#### 音频预处理流程
```python
# 2. 音频裁剪 (可选)
if AudioCrop and enable_audio_crop:
    crop_node = AudioCrop()
    audio = crop_node.crop(
        audio=audio,
        start_time=audio_start_time,  # 新UI参数
        duration=audio_duration  # 新UI参数
    )

# 3. 音频分离 (可选)
if AudioSeparation and enable_audio_separation:
    sep_node = AudioSeparation()
    audio = sep_node.separate(
        audio=audio,
        model=separation_model,  # 新UI参数
        device=self.device
    )

# 4. 获取音频时长
if AudioDuration:
    duration_node = AudioDuration()
    audio_duration = duration_node.get_duration(audio=audio)
```

#### 动态参数计算
```python
# 5. 使用 SimpleMath+ 计算帧数
if SimpleMathNode and auto_calculate_frames:
    math_node = SimpleMathNode()
    # 根据音频时长计算视频帧数
    calculated_frames = math_node.calculate(
        a=audio_duration,
        b=fps,
        operation="multiply"  # duration * fps
    )
    video_length = min(calculated_frames, max_frames)
```

### 2. UI 参数扩展
需要在 `infinite_talk_ui.py` 中添加：

#### 图像处理参数组
```python
gr.Markdown("### 🖼️ 图像预处理")

with gr.Row():
    use_image_resize = gr.Checkbox(
        label="启用图像缩放",
        value=True
    )
    resize_interpolation = gr.Dropdown(
        label="插值方法",
        choices=["lanczos", "bicubic", "bilinear", "nearest"],
        value="lanczos"
    )

with gr.Row():
    resize_method = gr.Dropdown(
        label="缩放方法",
        choices=["stretch", "keep proportion", "fill / crop", "pad"],
        value="stretch"
    )
    resize_condition = gr.Dropdown(
        label="缩放条件",
        choices=["always", "downscale if bigger", "upscale if smaller", "if bigger area", "if smaller area"],
        value="always"
    )
```

#### 音频处理参数组
```python
gr.Markdown("### 🎵 音频预处理")

with gr.Row():
    enable_audio_crop = gr.Checkbox(
        label="启用音频裁剪",
        value=False
    )
    audio_start_time = gr.Slider(
        label="开始时间 (秒)",
        minimum=0,
        maximum=60,
        value=0,
        step=0.1
    )
    audio_crop_duration = gr.Slider(
        label="裁剪时长 (秒)",
        minimum=0,
        maximum=60,
        value=0,
        step=0.1,
        info="0 表示到结尾"
    )

with gr.Row():
    enable_audio_separation = gr.Checkbox(
        label="启用音频分离",
        value=False,
        info="分离人声和背景音"
    )
    separation_model = gr.Dropdown(
        label="分离模型",
        choices=["UVR-MDX-NET-Inst_HQ_3", "UVR_MDXNET_KARA_2", "Kim_Vocal_2"],
        value="UVR-MDX-NET-Inst_HQ_3"
    )
```

#### 自动计算参数
```python
gr.Markdown("### 🔢 自动参数计算")

with gr.Row():
    auto_calculate_frames = gr.Checkbox(
        label="根据音频时长自动计算帧数",
        value=True,
        info="使用 SimpleMath+ 动态计算"
    )
    max_frames = gr.Slider(
        label="最大帧数限制",
        minimum=1,
        maximum=500,
        value=200,
        step=1
    )
```

### 3. 参数传递
需要更新 `generate_wrapper()` 和 `generate()` 方法签名，添加所有新参数。

### 4. 测试和验证
- [ ] 测试图像缩放功能
- [ ] 测试音频裁剪功能
- [ ] 测试音频分离功能
- [ ] 测试自动帧数计算
- [ ] 验证所有参数可以在 UI 中正确设置
- [ ] 确保不影响其他版块（Flux、Qwen 等）

## 🎯 优先级

1. **高优先级**：图像预处理（ImageResizeKJ）- 确保输入尺寸正确
2. **中优先级**：音频时长计算 - 自动匹配视频长度
3. **低优先级**：音频分离、裁剪 - 高级功能，可选

## 📝 注意事项

1. 所有新节点都通过 server_stub 避免 genesis 依赖
2. 参数默认值应匹配原工作流
3. UI 中所有参数都应该有清晰的说明
4. 保持代码只在 Infinite Talk 模块内修改
5. 添加充分的错误处理和日志输出
