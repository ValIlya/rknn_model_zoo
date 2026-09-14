import onnx
from collections import defaultdict

m = onnx.load('../model/mms_tts_eng_encoder_200.onnx')
g = m.graph

producer = {}
consumers = defaultdict(list)
for i, n in enumerate(g.node):
    for o in n.output:
        producer[o] = i
    for inp in n.input:
        consumers[inp].append(i)

graph_inputs = {x.name for x in g.input}
graph_outputs = {o.name for o in g.output}

dp_nodes = {i for i, n in enumerate(g.node) if 'duration_predictor' in (n.name or '')}
trunk_tensors = set()
for i in dp_nodes:
    for inp in g.node[i].input:
        if inp:
            pi = producer.get(inp)
            if pi not in dp_nodes:
                trunk_tensors.add(inp)

def walk_to_trunk(target_name):
    """Walk backward from target_name within dp_nodes, yield (op_type, name) in order."""
    pi = producer.get(target_name)
    if pi is None or pi not in dp_nodes:
        return
    stack = [pi]
    seen = set()
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        n = g.node[i]
        yield (i, n.op_type, n.name)
        for inp in n.input:
            if not inp:
                continue
            if inp in trunk_tensors or inp in graph_inputs:
                continue
            p = producer.get(inp)
            if p is not None and p in dp_nodes:
                stack.append(p)

path_nodes = list(walk_to_trunk('log_duration'))
print('=== LOG_DURATION PATH: unique op types (excluding Constant) ===')
op_counts = defaultdict(int)
for i, op, nm in path_nodes:
    if op != 'Constant':
        op_counts[op] += 1
for op in sorted(op_counts, key=op_counts.get, reverse=True):
    print('%-25s %4d' % (op, op_counts[op]))

print()
print('=== Where/Expand/Equal/ConstantOfShape nodes on log_duration path ===')
for i, op, nm in sorted(path_nodes):
    if op in ('Where', 'Expand', 'Equal', 'ConstantOfShape', 'Range', 'ScatterND', 'GatherElements'):
        print('node[%05d] %-20s %s' % (i, op, nm))
