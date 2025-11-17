import json
from pathlib import Path

wf_path = Path(r"E:\liliyuanshangmie\Fuxkcomfy_lris_kernel_gen2-4_speed_safe\FuxkComfy\user\default\workflows\Infinite Talk test(1).json")

with wf_path.open('r', encoding='utf-8') as f:
    data = json.load(f)

targets = [
    "WanVideoModelLoader",
    "WanVideoVAELoader",
    "WanVideoSampler",
    "WanVideoImageToVideoMultiTalk",
    "WanVideoDecode",
    "WanCompileArgs",
    "WanVideoTorchCompileSettings",
]

nodes = [n for n in data['nodes'] if n['type'] in targets]

for node in nodes:
    print("==", node['type'], "(id", node['id'], ")")
    print("widgets:", node.get('widgets_values'))
    print("inputs:")
    for inp in node.get('inputs', []):
        print("  -", inp.get('name'), inp.get('link'), inp.get('widget', {}).get('name'))
    print()
