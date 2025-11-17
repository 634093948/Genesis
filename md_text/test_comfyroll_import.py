#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "compat"))
sys.path.insert(0, str(project_root / "custom_nodes" / "Comfyui"))

from apps.wanvideo_module import server_stub
sys.modules['server'] = server_stub

print("Testing individual Comfyroll imports...")

try:
    from ComfyUI_Comfyroll_CustomNodes.nodes.nodes_core import *
    print("✓ nodes_core loaded")
except Exception as e:
    print(f"✗ nodes_core failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from ComfyUI_Comfyroll_CustomNodes.nodes.nodes_legacy import *
    print("✓ nodes_legacy loaded")
    print(f"  CR_ImageSize defined: {'CR_ImageSize' in dir()}")
except Exception as e:
    print(f"✗ nodes_legacy failed: {e}")
    import traceback
    traceback.print_exc()
