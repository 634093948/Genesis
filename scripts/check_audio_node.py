import json
from pathlib import Path

workflow_path = Path(r"E:\liliyuanshangmie\Fuxkcomfy_lris_kernel_gen2-4_speed_safe\FuxkComfy\user\default\workflows\Infinite Talk test(1).json")

with open(workflow_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

nodes = {n['id']: n for n in data['nodes']}

print("=" * 80)
print("Audio Nodes Analysis")
print("=" * 80)

# LoadAudio node
if 267 in nodes:
    print("\n[267] LoadAudio:")
    print(f"  Widgets: {nodes[267].get('widgets_values', [])}")
    print(f"  Outputs: {[o['name'] for o in nodes[267].get('outputs', [])]}")

# MultiTalkWav2VecEmbeds node
if 194 in nodes:
    n194 = nodes[194]
    print("\n[194] MultiTalkWav2VecEmbeds:")
    print(f"  Widgets: {n194.get('widgets_values', [])}")
    print("\n  Inputs:")
    for i, inp in enumerate(n194.get('inputs', [])):
        link = inp.get('link')
        link_info = f"connected (link {link})" if link else "NOT connected"
        print(f"    {i}: {inp['name']} - {link_info}")
    
    # Check what's connected to num_frames
    for inp in n194.get('inputs', []):
        if inp['name'] == 'num_frames' and inp.get('link'):
            link_id = inp['link']
            # Find the source
            for link in data['links']:
                if link[0] == link_id:
                    source_node_id = link[1]
                    source_slot = link[2]
                    if source_node_id in nodes:
                        source_node = nodes[source_node_id]
                        print(f"\n  num_frames comes from:")
                        print(f"    Node: [{source_node_id}] {source_node['type']}")
                        print(f"    Output slot: {source_slot}")
                        if source_node.get('outputs'):
                            print(f"    Output name: {source_node['outputs'][source_slot]['name']}")
                    break

# Check WanVideoImageToVideoMultiTalk frame_window_size
if 192 in nodes:
    n192 = nodes[192]
    print("\n[192] WanVideoImageToVideoMultiTalk:")
    print(f"  Widgets: {n192.get('widgets_values', [])}")
    print("\n  Inputs:")
    for i, inp in enumerate(n192.get('inputs', [])):
        if inp['name'] in ['frame_window_size', 'width', 'height']:
            link = inp.get('link')
            link_info = f"connected (link {link})" if link else "NOT connected"
            print(f"    {inp['name']}: {link_info}")

print("\n" + "=" * 80)
print("Key Findings:")
print("=" * 80)
print("\nWorkflow uses:")
print("  - MultiTalkWav2VecEmbeds.num_frames = 33 (from connected node)")
print("  - MultiTalkWav2VecEmbeds.fps = 25")
print("  - WanVideoImageToVideoMultiTalk.frame_window_size = 117")
print("\nOur implementation:")
print("  - We pass video_length to num_frames")
print("  - Need to check if this matches!")
