#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Install dependencies for Infinite Talk nodes
安装 Infinite Talk 节点所需的依赖
"""

import subprocess
import sys
from pathlib import Path

def install_package(package):
    """Install a package using pip"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    print("=" * 70)
    print("Installing Infinite Talk Dependencies")
    print("=" * 70)
    
    # Dependencies for audio-separation-nodes-comfyui
    audio_sep_deps = [
        "librosa==0.10.2",
        "torchaudio>=2.3.0",
        "numpy",
        "moviepy",
        "soundfile",  # Required by librosa
    ]
    
    # Dependencies for comfy-mtb (Audio Duration node)
    mtb_deps = [
        "qrcode[pil]",
        "onnxruntime-gpu",
        "requirements-parser",
        "rembg",
        "imageio_ffmpeg",
        "rich",
        "rich_argparse",
        "matplotlib",
        "pillow",
        "cachetools",
        "transformers",
    ]
    
    # All dependencies
    all_deps = {
        "Audio Separation Nodes": audio_sep_deps,
        "MTB Nodes (Audio Duration)": mtb_deps,
    }
    
    failed = []
    
    for category, deps in all_deps.items():
        print(f"\n### {category}")
        for dep in deps:
            if not install_package(dep):
                failed.append(dep)
    
    print("\n" + "=" * 70)
    if not failed:
        print("✅ All dependencies installed successfully!")
    else:
        print(f"⚠ {len(failed)} packages failed to install:")
        for pkg in failed:
            print(f"  - {pkg}")
        print("\nYou may need to install these manually.")
    print("=" * 70)

if __name__ == "__main__":
    main()
