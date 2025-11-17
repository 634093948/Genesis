#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试所有自定义节点的加载情况
"""

import sys
import os
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "compat"))

# Setup triton stub
try:
    from utils import triton_ops_stub
    print("[✓] Triton stub loaded")
except Exception as e:
    print(f"[✗] Failed to load triton stub: {e}")

# Pre-load comfy module
try:
    import comfy
    import comfy.samplers
    print("[✓] Comfy module pre-loaded")
    print(f"    - comfy.samplers.sampling_function: {hasattr(comfy.samplers, 'sampling_function')}")
except Exception as e:
    print(f"[✗] Failed to pre-load comfy: {e}")

# Inject server stub
try:
    from apps.wanvideo_module import server_stub
    sys.modules['server'] = server_stub
    print("[✓] Server stub injected")
except Exception as e:
    print(f"[✗] Failed to inject server stub: {e}")

# Test librosa
try:
    import librosa
    print(f"[✓] librosa {librosa.__version__} available")
except ImportError as e:
    print(f"[✗] librosa not available: {e}")

custom_nodes_path = project_root / "custom_nodes" / "Comfyui"

print("\n" + "="*60)
print("Testing Custom Nodes Loading")
print("="*60)

# Test 1: KJNodes
print("\n[1/5] Testing KJNodes...")
try:
    import importlib.util
    
    # Pre-load compat/nodes.py as 'nodes' module to prevent import conflicts
    if 'nodes' not in sys.modules:
        compat_nodes_path = project_root / "compat" / "nodes.py"
        spec = importlib.util.spec_from_file_location("nodes", compat_nodes_path)
        nodes_module = importlib.util.module_from_spec(spec)
        sys.modules['nodes'] = nodes_module
        spec.loader.exec_module(nodes_module)
        print(f"    [✓] Pre-loaded compat/nodes.py as 'nodes' module")
    
    kjnodes_path = custom_nodes_path / "ComfyUI-KJNodes"
    if kjnodes_path.exists():
        sys.path.insert(0, str(kjnodes_path))
        kjnodes_init = kjnodes_path / "__init__.py"
        spec = importlib.util.spec_from_file_location("ComfyUI_KJNodes", kjnodes_init)
        kjnodes_module = importlib.util.module_from_spec(spec)
        sys.modules['ComfyUI_KJNodes'] = kjnodes_module
        spec.loader.exec_module(kjnodes_module)
        if hasattr(kjnodes_module, 'NODE_CLASS_MAPPINGS'):
            ImageResizeKJ = kjnodes_module.NODE_CLASS_MAPPINGS.get('ImageResizeKJ')
            print(f"    [✓] KJNodes loaded successfully")
            print(f"    [✓] ImageResizeKJ: {ImageResizeKJ is not None}")
    else:
        print(f"    [✗] KJNodes path not found")
except Exception as e:
    print(f"    [✗] Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Audio Separation
print("\n[2/5] Testing Audio Separation Nodes...")
try:
    audio_sep_path = custom_nodes_path / "audio-separation-nodes-comfyui"
    if audio_sep_path.exists():
        sys.path.insert(0, str(audio_sep_path))
        audio_sep_init = audio_sep_path / "__init__.py"
        spec = importlib.util.spec_from_file_location("audio_separation_nodes", audio_sep_init)
        audio_sep_module = importlib.util.module_from_spec(spec)
        sys.modules['audio_separation_nodes'] = audio_sep_module
        spec.loader.exec_module(audio_sep_module)
        if hasattr(audio_sep_module, 'NODE_CLASS_MAPPINGS'):
            AudioSeparation = audio_sep_module.NODE_CLASS_MAPPINGS.get('AudioSeparation')
            AudioCrop = audio_sep_module.NODE_CLASS_MAPPINGS.get('AudioCrop')
            print(f"    [✓] Audio separation nodes loaded successfully")
            print(f"    [✓] AudioSeparation: {AudioSeparation is not None}")
            print(f"    [✓] AudioCrop: {AudioCrop is not None}")
    else:
        print(f"    [✗] Audio separation path not found")
except Exception as e:
    print(f"    [✗] Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: MTB Nodes
print("\n[3/5] Testing MTB Nodes...")
try:
    mtb_path = custom_nodes_path / "comfy-mtb"
    if mtb_path.exists():
        sys.path.insert(0, str(mtb_path))
        mtb_init = mtb_path / "__init__.py"
        spec = importlib.util.spec_from_file_location("comfy_mtb", mtb_init)
        mtb_module = importlib.util.module_from_spec(spec)
        sys.modules['comfy_mtb'] = mtb_module
        spec.loader.exec_module(mtb_module)
        if hasattr(mtb_module, 'NODE_CLASS_MAPPINGS'):
            AudioDuration = mtb_module.NODE_CLASS_MAPPINGS.get('Audio Duration (mtb)')
            print(f"    [✓] MTB nodes loaded successfully")
            print(f"    [✓] Audio Duration: {AudioDuration is not None}")
    else:
        print(f"    [✗] MTB path not found")
except Exception as e:
    print(f"    [✗] Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Comfyroll
print("\n[4/5] Testing Comfyroll Nodes...")
try:
    comfyroll_path = custom_nodes_path / "ComfyUI_Comfyroll_CustomNodes"
    if comfyroll_path.exists():
        sys.path.insert(0, str(comfyroll_path))
        comfyroll_init = comfyroll_path / "__init__.py"
        spec = importlib.util.spec_from_file_location("ComfyUI_Comfyroll", comfyroll_init)
        comfyroll_module = importlib.util.module_from_spec(spec)
        sys.modules['ComfyUI_Comfyroll'] = comfyroll_module
        spec.loader.exec_module(comfyroll_module)
        if hasattr(comfyroll_module, 'NODE_CLASS_MAPPINGS'):
            SimpleMathNode = comfyroll_module.NODE_CLASS_MAPPINGS.get('SimpleMath+')
            print(f"    [✓] Comfyroll nodes loaded successfully")
            print(f"    [✓] SimpleMath+: {SimpleMathNode is not None}")
    else:
        print(f"    [✗] Comfyroll path not found")
except Exception as e:
    print(f"    [✗] Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: ComfyLiterals
print("\n[5/5] Testing ComfyLiterals...")
try:
    literals_path = custom_nodes_path / "ComfyLiterals"
    if literals_path.exists():
        sys.path.insert(0, str(literals_path))
        literals_init = literals_path / "__init__.py"
        spec = importlib.util.spec_from_file_location("ComfyLiterals", literals_init)
        literals_module = importlib.util.module_from_spec(spec)
        sys.modules['ComfyLiterals'] = literals_module
        spec.loader.exec_module(literals_module)
        if hasattr(literals_module, 'NODE_CLASS_MAPPINGS'):
            IntNode = literals_module.NODE_CLASS_MAPPINGS.get('Int')
            print(f"    [✓] ComfyLiterals loaded successfully")
            print(f"    [✓] Int: {IntNode is not None}")
    else:
        print(f"    [✗] ComfyLiterals path not found")
except Exception as e:
    print(f"    [✗] Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test Complete")
print("="*60)
