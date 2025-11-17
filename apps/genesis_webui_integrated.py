#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genesis WebUI - Fully Integrated Interface with Dynamic Module Loading
Automatically loads all app modules and registers them as tabs

Author: eddy
Date: 2025-11-14
"""

import sys
import os
from pathlib import Path
import importlib

os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'
os.environ['GRADIO_SERVER_PORT'] = '7860'

project_root = Path(__file__).parent.parent
apps_dir = Path(__file__).parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(apps_dir) not in sys.path:
    sys.path.insert(0, str(apps_dir))

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("="*70)
print("Genesis WebUI - Dynamic Module Loading")
print("="*70)

try:
    import gradio as gr
    print(f"✓ Gradio {gr.__version__}")
except ImportError:
    print("✗ Gradio not installed")
    sys.exit(1)

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("✗ PyTorch not installed")
    sys.exit(1)

print()
print("="*70)
print("Loading App Modules...")
print("="*70)

loaded_modules = {}

def load_module(module_name):
    """Dynamically load a module and return its components"""
    try:
        module = importlib.import_module(module_name)
        print(f"✓ {module_name} loaded")
        return module
    except Exception as e:
        print(f"✗ {module_name} failed: {e}")
        return None

sd_module = load_module('sd_module')
wanvideo_module = load_module('wanvideo_module')

if sd_module:
    loaded_modules['sd'] = {
        'create_tab': sd_module.create_sd_tab
    }

if wanvideo_module and wanvideo_module.WAN_VIDEO_AVAILABLE:
    try:
        loaded_modules['wanvideo'] = {
            'workflow': wanvideo_module.WanVideoWorkflow(),
            'create_tab': wanvideo_module.create_wanvideo_tab
        }
    except Exception as e:
        print(f"✗ WanVideo workflow init failed: {e}")
        loaded_modules['wanvideo'] = {
            'workflow': None,
            'create_tab': wanvideo_module.create_wanvideo_tab
        }
elif wanvideo_module:
    loaded_modules['wanvideo'] = {
        'workflow': None,
        'create_tab': wanvideo_module.create_wanvideo_tab
    }

print("="*70)
print()


def create_models_tab():
    """Model management tab"""
    with gr.Tab("Models"):
        gr.Markdown("""
        ## Model Management

        ### Model Directories
        - **Checkpoints**: `models/checkpoints/`
        - **VAE**: `models/vae/`
        - **LoRA**: `models/loras/`
        - **WanVideo**: `models/unet/`
        """)


def create_settings_tab():
    """Settings tab"""
    with gr.Tab("Settings"):
        gr.Markdown(f"""
        ## System Information

        **Device**: {'CUDA' if torch.cuda.is_available() else 'CPU'}
        **GPU**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}
        **PyTorch**: {torch.__version__}
        **Gradio**: {gr.__version__}

        ### Loaded Modules
        - **SD Generator**: {'✓ Loaded' if 'sd' in loaded_modules else '✗ Not Available'}
        - **WanVideo**: {'✓ Loaded' if 'wanvideo' in loaded_modules and loaded_modules['wanvideo']['workflow'] else '✗ Not Available'}

        ### Configuration
        Model paths: `extra_model_paths.yaml`
        Documentation: [docs/](../docs/)
        """)


def main():
    """Main WebUI application"""

    print("="*70)
    print("Building WebUI Interface...")
    print("="*70)

    with gr.Blocks(
        title="Genesis WebUI",
        theme=gr.themes.Soft(),
        css="""
        #gallery { min-height: 400px; }
        .gradio-container { max-width: 100% !important; }
        """
    ) as demo:
        gr.Markdown("""
        # Genesis WebUI

        Unified interface for AI generation - Stable Diffusion and WanVideo
        """)

        with gr.Tabs():
            if 'sd' in loaded_modules:
                loaded_modules['sd']['create_tab']()
            else:
                with gr.Tab("txt2img"):
                    gr.Markdown("""
                    ## ⚠️ SD Module Not Available

                    Install dependencies:
                    ```bash
                    pip install diffusers transformers accelerate
                    ```
                    """)

            with gr.Tab("img2img"):
                gr.Markdown("## img2img (Coming Soon)")

            if 'wanvideo' in loaded_modules:
                loaded_modules['wanvideo']['create_tab'](loaded_modules['wanvideo']['workflow'])
            else:
                with gr.Tab("WanVideo"):
                    gr.Markdown("## ⚠️ WanVideo Module Not Loaded")

            create_models_tab()
            create_settings_tab()

        gr.Markdown("""
        ---
        **Genesis AI Engine** | [Documentation](../docs/) | [GitHub](https://github.com/eddyhhlure1Eddy/Genesis)
        """)

    print()
    print("="*70)
    print("Launching Genesis WebUI...")
    print("="*70)
    
    # Try to find available port
    import socket
    def is_port_available(port):
        """Check if port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return True
        except:
            return False
    
    # Try ports in order
    ports_to_try = [7860, 7861, 7862, 7863, 7864]
    selected_port = None
    
    for port in ports_to_try:
        if is_port_available(port):
            selected_port = port
            print(f"✓ Port {port} is available")
            break
        else:
            print(f"✗ Port {port} is occupied")
    
    if selected_port is None:
        print("⚠ All preferred ports are occupied, using random port...")
        selected_port = 0
    
    print(f"\n🚀 Starting server on port {selected_port}...")
    print("="*70)
    
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=selected_port,
            share=False,
            inbrowser=True,
            show_error=True,
            quiet=False
        )
    except Exception as e:
        print(f"\n❌ Launch failed: {e}")
        print("\nTrying alternative port...")
        try:
            demo.launch(
                server_name="0.0.0.0",
                server_port=0,
                share=False,
                inbrowser=True,
                quiet=False
            )
        except Exception as e2:
            print(f"\n❌ All launch attempts failed: {e2}")
            print("\nPlease check:")
            print("  1. No other instance is running")
            print("  2. Firewall is not blocking the ports")
            print("  3. Run: netstat -ano | findstr '7860' to check port usage")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutdown by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
