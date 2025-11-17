import json
from pathlib import Path

workflow_path = Path(r"E:\liliyuanshangmie\Fuxkcomfy_lris_kernel_gen2-4_speed_safe\FuxkComfy\user\default\workflows\Infinite Talk test(1).json")

with open(workflow_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 80)
print("Workflow vs Our Implementation - Parameter Comparison")
print("=" * 80)

# Key nodes to check
nodes_to_check = {
    'WanVideoModelLoader': 122,
    'WanVideoVAELoader': 129,
    'LoadWanVideoT5TextEncoder': 136,
    'WanVideoTextEncode': 135,
    'WanVideoClipVisionEncode': 193,
    'MultiTalkWav2VecEmbeds': 194,
    'WanVideoImageToVideoMultiTalk': 192,
    'WanVideoSampler': 128,
    'WanVideoDecode': 130
}

nodes_dict = {n['id']: n for n in data['nodes']}

for node_type, node_id in nodes_to_check.items():
    if node_id in nodes_dict:
        node = nodes_dict[node_id]
        print(f"\n{'='*80}")
        print(f"{node_type} (ID: {node_id})")
        print(f"{'='*80}")
        
        # Get inputs
        inputs = node.get('inputs', [])
        widgets = node.get('widgets_values', [])
        
        print(f"\nInputs ({len(inputs)}):")
        for i, inp in enumerate(inputs):
            inp_name = inp.get('name', f'input_{i}')
            inp_type = inp.get('type', 'unknown')
            link = inp.get('link')
            print(f"  {i}: {inp_name} ({inp_type}) - {'connected' if link else 'NOT connected'}")
        
        print(f"\nWidget Values ({len(widgets)}):")
        for i, val in enumerate(widgets):
            print(f"  {i}: {val}")
        
        # Get outputs
        outputs = node.get('outputs', [])
        print(f"\nOutputs ({len(outputs)}):")
        for i, out in enumerate(outputs):
            out_name = out.get('name', f'output_{i}')
            out_type = out.get('type', 'unknown')
            links = out.get('links', [])
            print(f"  {i}: {out_name} ({out_type}) - {len(links) if links else 0} connections")

print("\n" + "=" * 80)
print("Critical Findings:")
print("=" * 80)

# Check specific nodes
print("\n1. WanVideoModelLoader:")
if 122 in nodes_dict:
    node = nodes_dict[122]
    inputs = node.get('inputs', [])
    print(f"   Inputs: {[inp['name'] for inp in inputs]}")
    print(f"   Widgets: {node.get('widgets_values', [])}")

print("\n2. WanVideoImageToVideoMultiTalk:")
if 192 in nodes_dict:
    node = nodes_dict[192]
    inputs = node.get('inputs', [])
    print(f"   Inputs: {[inp['name'] for inp in inputs]}")
    print(f"   Widgets: {node.get('widgets_values', [])}")

print("\n3. WanVideoSampler:")
if 128 in nodes_dict:
    node = nodes_dict[128]
    inputs = node.get('inputs', [])
    print(f"   Inputs: {[inp['name'] for inp in inputs]}")
    print(f"   Widgets: {node.get('widgets_values', [])}")
