#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""最终节点加载测试"""

import sys
import os
import importlib.util
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
except Exception as e:
    print(f"[✗] Failed to pre-load comfy: {e}")

# Inject server stub
try:
    from apps.wanvideo_module import server_stub
    sys.modules['server'] = server_stub
    print("[✓] Server stub injected")
    print(f"    - PromptServer.instance.app.router: {server_stub.PromptServer.instance.app.router}")
except Exception as e:
    print(f"[✗] Failed to inject server stub: {e}")

# Test librosa
try:
    import librosa
    print(f"[✓] librosa {librosa.__version__} available")
except ImportError as e:
    print(f"[✗] librosa not available: {e}")

# Test moviepy
try:
    import moviepy
    print(f"[✓] moviepy available")
except ImportError as e:
    print(f"[✗] moviepy not available: {e}")

custom_nodes_path = project_root / "custom_nodes" / "Comfyui"

print("\n" + "="*60)
print("Testing Custom Nodes Loading")
print("="*60)

# Test 1: MTB Nodes
print("\n[1/4] Testing MTB Nodes...")
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
            print(f"    [✓] Total nodes: {len(mtb_module.NODE_CLASS_MAPPINGS)}")
    else:
        print(f"    [✗] MTB path not found")
except Exception as e:
    print(f"    [✗] Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Audio Separation
print("\n[2/4] Testing Audio Separation Nodes...")
try:
    import librosa  # Verify librosa first
    
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
except ImportError as e:
    print(f"    [✗] Failed (librosa not installed): {e}")
except Exception as e:
    print(f"    [✗] Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Comfyroll
print("\n[3/4] Testing Comfyroll Nodes...")
try:
    comfyroll_path = custom_nodes_path / "ComfyUI_Comfyroll_CustomNodes"
    if comfyroll_path.exists():
        if str(custom_nodes_path) not in sys.path:
            sys.path.insert(0, str(custom_nodes_path))
        
        comfyroll_init = comfyroll_path / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "ComfyUI_Comfyroll_CustomNodes", 
            comfyroll_init,
            submodule_search_locations=[str(comfyroll_path)]
        )
        comfyroll_module = importlib.util.module_from_spec(spec)
        comfyroll_module.__package__ = "ComfyUI_Comfyroll_CustomNodes"
        sys.modules['ComfyUI_Comfyroll_CustomNodes'] = comfyroll_module
        spec.loader.exec_module(comfyroll_module)
        if hasattr(comfyroll_module, 'NODE_CLASS_MAPPINGS'):
            SimpleMathNode = comfyroll_module.NODE_CLASS_MAPPINGS.get('SimpleMath+')
            CR_ImageSize = comfyroll_module.NODE_CLASS_MAPPINGS.get('CR Image Size')
            print(f"    [✓] Comfyroll nodes loaded successfully")
            print(f"    [✓] SimpleMath+: {SimpleMathNode is not None}")
            print(f"    [✓] CR Image Size: {CR_ImageSize is not None}")
            print(f"    [✓] Total nodes: {len(comfyroll_module.NODE_CLASS_MAPPINGS)}")
    else:
        print(f"    [✗] Comfyroll path not found")
except Exception as e:
    print(f"    [✗] Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: ComfyLiterals
print("\n[4/4] Testing ComfyLiterals...")
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
