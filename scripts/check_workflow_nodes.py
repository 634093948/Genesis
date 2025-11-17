import json
from pathlib import Path

workflow_path = Path(r"E:\liliyuanshangmie\Fuxkcomfy_lris_kernel_gen2-4_speed_safe\FuxkComfy\user\default\workflows\Infinite Talk test(1).json")

with open(workflow_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 80)
print("Infinite Talk Workflow Node Analysis")
print("=" * 80)

# Find key nodes
key_nodes = []
for node in data['nodes']:
    node_type = node.get('type', '')
    if any(key in node_type for key in ['WanVideo', 'MultiTalk', 'CLIP', 'LoadAudio', 'LoadImage', 'Wav2Vec']):
        key_nodes.append(node)

# Sort by ID
key_nodes.sort(key=lambda n: n['id'])

print("\nKey Nodes in Workflow:")
print("-" * 80)
for node in key_nodes:
    node_id = node['id']
    node_type = node['type']
    outputs = node.get('outputs', [])
    
    print(f"\n[{node_id}] {node_type}")
    
    # Show outputs and their connections
    for i, output in enumerate(outputs):
        output_name = output.get('name', f'output_{i}')
        output_links = output.get('links', [])
        print(f"  Output '{output_name}': {len(output_links)} connections")
        
        # Find where this output goes
        if output_links:
            for link_id in output_links:
                # Find the link
                for link in data['links']:
                    if link[0] == link_id:
                        target_node_id = link[3]
                        target_slot = link[4]
                        # Find target node
                        for target_node in data['nodes']:
                            if target_node['id'] == target_node_id:
                                target_inputs = target_node.get('inputs', [])
                                target_input_name = target_inputs[target_slot]['name'] if target_slot < len(target_inputs) else f'input_{target_slot}'
                                print(f"    -> [{target_node_id}] {target_node['type']}.{target_input_name}")
                                break
                        break

print("\n" + "=" * 80)
print("Execution Flow:")
print("=" * 80)

# Build execution order
execution_order = [
    "LoadImage",
    "ImageResizeKJ",
    "CLIPVisionLoader",
    "WanVideoClipVisionEncode",
    "LoadWanVideoT5TextEncoder",
    "WanVideoTextEncode",
    "LoadAudio",
    "DownloadAndLoadWav2VecModel",
    "MultiTalkWav2VecEmbeds",
    "WanVideoModelLoader",
    "WanVideoVAELoader",
    "WanVideoImageToVideoMultiTalk",
    "WanVideoSampler",
    "WanVideoDecode"
]

for i, node_type in enumerate(execution_order, 1):
    # Find if this node exists in workflow
    found = False
    for node in key_nodes:
        if node['type'] == node_type:
            found = True
            print(f"{i:2d}. ✓ {node_type} (ID: {node['id']})")
            break
    if not found:
        print(f"{i:2d}. ✗ {node_type} (NOT FOUND)")

print("\n" + "=" * 80)
