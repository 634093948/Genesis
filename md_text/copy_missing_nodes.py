import shutil
from pathlib import Path

# 源目录和目标目录
source_base = Path(r"E:\liliyuanshangmie\Fuxkcomfy_lris_kernel_gen2-4_speed_safe\FuxkComfy\custom_nodes")
target_base = Path(r"E:\liliyuanshangmie\Genesis-webui-modular-integration\custom_nodes\Comfyui")

# 工作流中使用的节点对应的包
# 格式: {节点类型: 包名}
node_to_package = {
    # 已存在的节点包
    "WanVideoModelLoader": "ComfyUI-WanVideoWrapper",  # 已存在
    "WanVideoVAELoader": "ComfyUI-WanVideoWrapper",  # 已存在
    "WanVideoSampler": "ComfyUI-WanVideoWrapper",  # 已存在
    "WanVideoTextEncode": "ComfyUI-WanVideoWrapper",  # 已存在
    "WanVideoClipVisionEncode": "ComfyUI-WanVideoWrapper",  # 已存在
    "WanVideoImageToVideoMultiTalk": "ComfyUI-WanVideoWrapper",  # 已存在
    "WanVideoDecode": "ComfyUI-WanVideoDecode-Standalone",  # 已存在
    "WanVideoTorchCompileSettings": "ComfyUI-TorchCompileSpeed",  # 已存在
    "WanVideoEnhancedBlockSwap": "ComfyUI-UniversalBlockSwap",  # 已存在
    "LoadWanVideoT5TextEncoder": "ComfyUI-WanVideoWrapper",  # 已存在
    "MultiTalkWav2VecEmbeds": "ComfyUI-WanVideoWrapper",  # 已存在
    "DownloadAndLoadWav2VecModel": "ComfyUI-WanVideoWrapper",  # 已存在
    "CLIPVisionLoader": "ComfyUI-WanVideoWrapper",  # 已存在
    "LoadImage": "ComfyUI_essentials",  # 已存在
    "LoadAudio": "ComfyUI-VideoHelperSuite",  # 已存在
    "VHS_VideoCombine": "ComfyUI-VideoHelperSuite",  # 已存在
    "AudioSeparation": "audio-separation-nodes-comfyui",  # 已存在
    "AudioCrop": "audio-separation-nodes-comfyui",  # 已存在
    
    # 需要复制的节点包
    "ImageResizeKJv2": "ComfyUI-KJNodes",  # 已存在但可能不完整
    "SimpleMath+": "ComfyUI_Comfyroll_CustomNodes",  # 需要复制
    "easy showAnything": "comfyui-easy-use",  # 已存在
    "ttN int": "comfyui_tinyterranodes",  # 已存在
    "ttN text": "comfyui_tinyterranodes",  # 已存在
    "Audio Duration (mtb)": "comfy-mtb",  # 已存在
    "Int": "ComfyLiterals",  # 需要复制
}

# 需要复制的包
packages_to_copy = [
    "ComfyUI_Comfyroll_CustomNodes",
    "ComfyLiterals",
]

print("=== Checking and Copying Missing Packages ===\n")

for package in packages_to_copy:
    source_path = source_base / package
    target_path = target_base / package
    
    if not source_path.exists():
        print(f"❌ Source not found: {package}")
        continue
    
    if target_path.exists():
        print(f"✓ Already exists: {package}")
        continue
    
    print(f"📦 Copying: {package}")
    try:
        shutil.copytree(source_path, target_path)
        print(f"✅ Copied: {package}")
    except Exception as e:
        print(f"❌ Failed to copy {package}: {e}")

print("\n=== Copy Complete ===")
