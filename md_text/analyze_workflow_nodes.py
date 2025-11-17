import json
import sys

wf_path = r"E:\liliyuanshangmie\Fuxkcomfy_lris_kernel_gen2-4_speed_safe\FuxkComfy\user\default\workflows\Infinite Talk test(1).json"

with open(wf_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

node_types = set(n['type'] for n in data['nodes'])

print("=== All Node Types in Workflow ===")
for nt in sorted(node_types):
    print(nt)

print("\n=== Node Type Counts ===")
from collections import Counter
type_counts = Counter(n['type'] for n in data['nodes'])
for nt, count in type_counts.most_common():
    print(f"{nt}: {count}")
