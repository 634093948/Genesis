import json
from pathlib import Path
wf_path = Path(r"E:\\liliyuanshangmie\\Fuxkcomfy_lris_kernel_gen2-4_speed_safe\\FuxkComfy\\user\\default\\workflows\\Infinite Talk test(1).json")
text = wf_path.read_text(encoding='utf-8')
data = json.loads(text)
nodes = {node['id']: node for node in data['nodes']}
links = {link['id']: link for link in data['links']}

sampler_node = None
for node in nodes.values():
    if node['type'] == 'WanVideoSampler':
        sampler_node = node
        break

if not sampler_node:
    print('WanVideoSampler not found')
    exit(0)

print('WanVideoSampler widgets:', sampler_node.get('widgets_values'))

def resolve_input(node, name):
    for inp in node.get('inputs', []):
        if inp['name'] == name and inp['link'] is not None:
            link = links.get(inp['link'])
            if not link:
                return None
            origin = nodes.get(link['origin_id'])
            return origin
    return None

scheduler_node = resolve_input(sampler_node, 'scheduler')
sampler_input_node = resolve_input(sampler_node, 'sampler')

print('Scheduler source:', scheduler_node['type'] if scheduler_node else None)
if scheduler_node:
    print('Scheduler widgets:', scheduler_node.get('widgets_values'))

print('Sampler source:', sampler_input_node['type'] if sampler_input_node else None)
if sampler_input_node:
    print('Sampler widgets:', sampler_input_node.get('widgets_values'))
