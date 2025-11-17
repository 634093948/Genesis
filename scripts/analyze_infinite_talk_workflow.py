#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze Infinite Talk workflow and identify required custom nodes
分析 Infinite Talk 工作流并识别所需的自定义节点

Author: eddy
Date: 2025-11-16
"""

import json
import sys
import io
from pathlib import Path

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("=" * 80)
print("Infinite Talk Workflow Analysis")
print("=" * 80)

# Load workflow
workflow_path = Path(r"E:\liliyuanshangmie\Fuxkcomfy_lris_kernel_gen2-4_speed_safe\FuxkComfy\user\default\workflows\Infinite Talk test(1).json")

if not workflow_path.exists():
    print(f"✗ 工作流文件不存在: {workflow_path}")
    sys.exit(1)

print(f"✓ 加载工作流: {workflow_path.name}")

with open(workflow_path, 'r', encoding='utf-8') as f:
    workflow = json.load(f)

# Find all nodes
nodes = workflow.get('nodes', [])
print(f"\nTotal nodes: {len(nodes)}")

# Find key nodes
key_types = [
    'WanVideoModelLoader',
    'WanVideoVAELoader', 
    'LoadWanVideoT5TextEncoder',
    'CLIPVisionLoader',
    'DownloadAndLoadWav2VecModel',
    'LoadImage',
    'LoadAudio',
    'WanVideoImageToVideoMultiTalk',
    'WanVideoSampler',
    'WanVideoDecode',
    'MultiTalkWav2VecEmbeds',
    'WanVideoClipVisionEncode',
    'WanVideoTextEncode'
]

print("\n" + "=" * 80)
print("Key Nodes Found:")
print("=" * 80)

for node in nodes:
    node_type = node.get('type', '')
    if any(key in node_type for key in key_types):
        node_id = node.get('id')
        widgets = node.get('widgets_values', [])
        inputs = node.get('inputs', [])
        print(f"\n[{node_type}] (ID: {node_id})")
        print(f"  Widgets: {widgets}")
        if inputs:
            print(f"  Inputs: {[inp.get('name') for inp in inputs]}")

# Find workflow execution order
print("\n" + "=" * 80)
print("Workflow Execution Flow:")
print("=" * 80)

links = workflow.get('links', [])
print(f"\nTotal links: {len(links)}")

print("\n" + "=" * 70)
print("节点分类")
print("=" * 70)
print()

# WanVideo nodes (核心)
wanvideo_nodes = [nt for nt in node_types.keys() if 'WanVideo' in nt or 'MultiTalk' in nt]
print("📹 WanVideo 核心节点:")
for nt in sorted(wanvideo_nodes):
    print(f"  - {nt} (x{len(node_types[nt])})")
print()

# Audio nodes
audio_nodes = [nt for nt in node_types.keys() if 'Audio' in nt or 'Wav2Vec' in nt]
print("🎵 音频处理节点:")
for nt in sorted(audio_nodes):
    print(f"  - {nt} (x{len(node_types[nt])})")
print()

# Image/Video nodes
image_nodes = [nt for nt in node_types.keys() if 'Image' in nt or 'Video' in nt or 'CLIP' in nt]
image_nodes = [nt for nt in image_nodes if nt not in wanvideo_nodes]
print("🖼️ 图像/视频节点:")
for nt in sorted(image_nodes):
    print(f"  - {nt} (x{len(node_types[nt])})")
print()

# Utility nodes
utility_nodes = [nt for nt in node_types.keys() if nt not in wanvideo_nodes and nt not in audio_nodes and nt not in image_nodes]
print("🔧 工具节点:")
for nt in sorted(utility_nodes):
    print(f"  - {nt} (x{len(node_types[nt])})")
print()

# Map nodes to custom_nodes folders
print("=" * 70)
print("节点来源映射")
print("=" * 70)
print()

node_source_map = {
    # WanVideo nodes
    'WanVideoModelLoader': 'ComfyUI-WanVideoWrapper',
    'WanVideoVAELoader': 'ComfyUI-WanVideoWrapper',
    'WanVideoTextEncode': 'ComfyUI-WanVideoWrapper',
    'WanVideoClipVisionEncode': 'ComfyUI-WanVideoWrapper',
    'WanVideoImageToVideoMultiTalk': 'ComfyUI-WanVideoWrapper',
    'WanVideoSampler': 'ComfyUI-WanVideoWrapper',
    'WanVideoDecode': 'ComfyUI-WanVideoDecode-Standalone',
    'WanVideoEnhancedBlockSwap': 'ComfyUI-UniversalBlockSwap',
    'WanVideoTorchCompileSettings': 'ComfyUI-TorchCompileSpeed',
    'LoadWanVideoT5TextEncoder': 'ComfyUI-WanVideoWrapper',
    'MultiTalkWav2VecEmbeds': 'ComfyUI-WanVideoWrapper',
    
    # Audio nodes
    'LoadAudio': 'ComfyUI-VideoHelperSuite',
    'AudioSeparation': 'audio-separation-nodes-comfyui',
    'AudioCrop': 'audio-separation-nodes-comfyui',
    'DownloadAndLoadWav2VecModel': 'ComfyUI-WanVideoWrapper',
    'Audio Duration (mtb)': 'comfy-mtb',
    
    # Image/Video nodes
    'LoadImage': 'ComfyUI (built-in)',
    'CLIPVisionLoader': 'ComfyUI (built-in)',
    'ImageResizeKJv2': 'ComfyUI-KJNodes',
    'VHS_VideoCombine': 'ComfyUI-VideoHelperSuite',
    
    # Utility nodes
    'ttN text': 'comfyui_tinyterranodes',
    'ttN int': 'comfyui_tinyterranodes',
    'easy showAnything': 'comfyui-easy-use',
    'SimpleMath+': 'ComfyUI_essentials',
    'Int': 'ComfyUI (built-in)',
}

# Group by source
source_groups = {}
for node_type in sorted(node_types.keys()):
    source = node_source_map.get(node_type, '❓ Unknown')
    if source not in source_groups:
        source_groups[source] = []
    source_groups[source].append(node_type)

for source in sorted(source_groups.keys()):
    print(f"📦 {source}:")
    for nt in source_groups[source]:
        print(f"    - {nt}")
    print()

# Summary
print("=" * 70)
print("需要复制的 custom_nodes")
print("=" * 70)
print()

required_folders = set()
for node_type in node_types.keys():
    source = node_source_map.get(node_type)
    if source and source != 'ComfyUI (built-in)':
        required_folders.add(source)

print("需要从 FuxkComfy 复制到 Genesis:")
print()
for folder in sorted(required_folders):
    source_path = Path(r"E:\liliyuanshangmie\Fuxkcomfy_lris_kernel_gen2-4_speed_safe\FuxkComfy\custom_nodes") / folder
    target_path = Path(r"E:\liliyuanshangmie\Genesis-webui-modular-integration\custom_nodes\Comfyui") / folder
    
    if source_path.exists():
        print(f"✓ {folder}")
        print(f"  源: {source_path}")
        print(f"  目标: {target_path}")
    else:
        print(f"✗ {folder} (源文件夹不存在)")
    print()

print("=" * 70)
print(f"总计: {len(required_folders)} 个 custom_nodes 文件夹")
print("=" * 70)
