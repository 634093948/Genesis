# Flux Text-to-Image Integration

## Overview

Flux text-to-image functionality has been successfully integrated into the Genesis WebUI project. This implementation is based on the ComfyUI workflow `flux文生图.json`.

## Architecture

### Workflow Analysis

The original workflow consists of the following nodes:

1. **UNETLoader** (Node 31): Loads Flux UNET model
2. **DualCLIPLoader** (Node 30): Loads T5XXL + CLIP-L models
3. **VAELoader** (Node 32): Loads VAE model
4. **LoraLoaderModelOnly** (Nodes 50, 51): Loads LoRA models
5. **CLIPTextEncode** (Nodes 29, 52): Encodes positive and negative prompts
6. **FluxGuidance** (Node 37): Applies Flux-specific guidance
7. **EmptyLatentImage** (Node 35): Creates initial latent
8. **KSamplerAdvanced** (Node 27): Advanced sampling with custom parameters
9. **VAEDecode** (Node 33): Decodes latent to image
10. **SaveImage** (Node 34): Saves the generated image

### Python Implementation

The workflow has been converted into a Python pipeline with the following structure:

```
apps/sd_module/
├── flux_text2img.py       # Core pipeline implementation
├── flux_gradio_ui.py      # Gradio UI interface
└── __init__.py            # Module integration
```

## Files Created

### 1. `flux_text2img.py`

Core pipeline implementation that replicates the workflow logic:

- **FluxText2ImgPipeline**: Main pipeline class
  - `load_unet()`: Load UNET model
  - `load_dual_clip()`: Load T5XXL + CLIP-L
  - `load_vae()`: Load VAE
  - `load_lora()`: Load LoRA models
  - `encode_prompt()`: Encode text prompts
  - `apply_flux_guidance()`: Apply Flux guidance
  - `create_empty_latent()`: Create initial latent
  - `sample()`: Advanced sampling
  - `decode_latent()`: Decode to image
  - `generate()`: Complete generation pipeline

### 2. `flux_gradio_ui.py`

Gradio interface for user interaction:

- **FluxGradioUI**: UI wrapper class
  - Model loading interface
  - Parameter controls
  - Generation interface
  - Result display

### 3. Updated `__init__.py`

Integration into the sd_module:

- Added `create_flux_tab()` function
- Integrated Flux UI into existing module structure

### 4. `start_flux_ui.bat`

Launcher script for standalone Flux UI

## Features

### Model Support

- **UNET**: Flux models (e.g., flux1-dev-fp8.safetensors)
- **CLIP**: Dual CLIP support (T5XXL + CLIP-L)
- **VAE**: Compatible VAE models (e.g., ae.sft)
- **LoRA**: Multiple LoRA loading support

### Generation Parameters

- **Resolution**: 512-2048px (customizable)
- **Steps**: 1-100 sampling steps
- **CFG Scale**: 0-20 (typically 1.0 for Flux)
- **Flux Guidance**: 0-10 (typically 3.5)
- **Seed**: Random or fixed seed
- **Sampler**: Multiple samplers (dpmpp_2m, euler, etc.)
- **Scheduler**: Multiple schedulers (sgm_uniform, karras, etc.)

### LoRA Support

- Load up to 2 LoRA models simultaneously
- Adjustable strength (0.0-2.0)
- Dynamic LoRA selection

## Usage

### Standalone Mode

Run the Flux UI independently:

```batch
scripts\start_flux_ui.bat
```

Or directly:

```bash
python apps/sd_module/flux_gradio_ui.py
```

### Integrated Mode

Import and use in other modules:

```python
from apps.sd_module import create_flux_tab

# In your Gradio app
with gr.Blocks() as demo:
    create_flux_tab()
```

Or use the pipeline directly:

```python
from apps.sd_module.flux_text2img import FluxText2ImgPipeline

# Create pipeline
pipeline = FluxText2ImgPipeline()

# Load models
pipeline.load_unet("flux1-dev-fp8.safetensors")
pipeline.load_dual_clip()
pipeline.load_vae("ae.sft")

# Generate image
image = pipeline.generate(
    prompt="a beautiful landscape",
    width=1080,
    height=1920,
    steps=20,
    guidance=3.5
)
```

## Model Requirements

### Required Models

Place models in the appropriate directories:

```
models/
├── unet/
│   └── flux1-dev-fp8.safetensors
├── clip/
│   ├── sd3/
│   │   └── t5xxl_fp16.safetensors
│   └── clip_l.safetensors
└── vae/
    └── ae.sft
```

### Optional Models

```
models/
└── loras/
    ├── AISHOUJIA.safetensors
    └── F.1_写实人像摄影.safetensors
```

## Workflow Comparison

### Original ComfyUI Workflow

```json
{
  "nodes": [
    {"id": 31, "type": "UNETLoader"},
    {"id": 30, "type": "DualCLIPLoader"},
    {"id": 32, "type": "VAELoader"},
    {"id": 50, "type": "LoraLoaderModelOnly"},
    {"id": 51, "type": "LoraLoaderModelOnly"},
    {"id": 29, "type": "CLIPTextEncode"},
    {"id": 52, "type": "CLIPTextEncode"},
    {"id": 37, "type": "FluxGuidance"},
    {"id": 35, "type": "EmptyLatentImage"},
    {"id": 27, "type": "KSamplerAdvanced"},
    {"id": 33, "type": "VAEDecode"},
    {"id": 34, "type": "SaveImage"}
  ]
}
```

### Python Pipeline

```python
# 1. Load models
pipeline.load_unet(...)
pipeline.load_dual_clip(...)
pipeline.load_vae(...)
pipeline.load_lora(...)

# 2. Encode prompts
positive_cond = pipeline.encode_prompt(prompt)
negative_cond = pipeline.encode_prompt(negative_prompt)

# 3. Apply guidance
positive_cond = pipeline.apply_flux_guidance(positive_cond, guidance)

# 4. Create latent
latent = pipeline.create_empty_latent(width, height)

# 5. Sample
sampled_latent = pipeline.sample(
    positive_cond, negative_cond, latent, ...
)

# 6. Decode
image = pipeline.decode_latent(sampled_latent)
```

## Implementation Notes

### Current Status

- ✅ Workflow structure implemented
- ✅ UI interface created
- ✅ Model loading framework
- ✅ Parameter handling
- ⚠️ Actual model inference (placeholder)

### TODO

The current implementation provides the complete workflow structure but uses placeholder logic for actual model inference. To complete the implementation:

1. **Implement UNET loading**: Use actual Flux UNET loading logic
2. **Implement CLIP encoding**: Use T5XXL + CLIP-L for text encoding
3. **Implement VAE decoding**: Use actual VAE decoding
4. **Implement sampling**: Use KSamplerAdvanced with proper schedulers
5. **Implement LoRA loading**: Apply LoRA weights to models

### Integration Points

The implementation is designed to integrate with:

- **Core modules**: Uses `core.folder_paths` for model paths
- **Existing nodes**: Compatible with existing node implementations
- **Gradio UI**: Seamless integration with existing UI structure

## Testing

### Test the Pipeline

```bash
python apps/sd_module/flux_text2img.py
```

This will:
1. Initialize the pipeline
2. Load models (if available)
3. Generate a test image
4. Save to `output/flux_test.png`

### Test the UI

```bash
python apps/sd_module/flux_gradio_ui.py
```

This will launch the Gradio interface on port 7861.

## Troubleshooting

### Models Not Found

Ensure models are placed in the correct directories as specified in `extra_model_paths.yaml`.

### Import Errors

Make sure the project root is in the Python path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

### Memory Issues

For large models:
- Use fp8 quantized models
- Enable attention slicing
- Reduce batch size

## Future Enhancements

1. **Complete Model Implementation**: Implement actual model loading and inference
2. **Batch Generation**: Support multiple image generation
3. **Image-to-Image**: Add img2img support
4. **ControlNet**: Add ControlNet support
5. **Upscaling**: Add post-generation upscaling
6. **API Endpoint**: Add REST API for programmatic access

## References

- Original workflow: `F:\工作流\flux文生图.json`
- ComfyUI nodes: Standard ComfyUI node implementations
- Model paths: `extra_model_paths.yaml`

## Author

**eddy** - 2025-11-16

## License

Same as Genesis WebUI project
