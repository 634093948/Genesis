#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 Comfyroll 节点加载"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "compat"))
sys.path.insert(0, str(project_root / "custom_nodes" / "Comfyui"))
sys.path.insert(0, str(project_root / "custom_nodes" / "Comfyui" / "ComfyUI_Comfyroll_CustomNodes"))

# Inject server stub
from apps.wanvideo_module import server_stub
sys.modules['server'] = server_stub

print("Testing Comfyroll Essential nodes...")
try:
    from nodes.nodes_essential import *
    print("✓ Essential nodes loaded")
except Exception as e:
    print(f"✗ Essential nodes failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting Comfyroll Graphics nodes...")
try:
    from nodes.nodes_graphics import *
    print("✓ Graphics nodes loaded")
except Exception as e:
    print(f"✗ Graphics nodes failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting Comfyroll Animation nodes...")
try:
    from nodes.nodes_animation import *
    print("✓ Animation nodes loaded")
except Exception as e:
    print(f"✗ Animation nodes failed: {e}")
    import traceback
    traceback.print_exc()
