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

graph_outputs = {o.name for o in g.output}
graph_inputs = {x.name for x in g.input}

# === duration_predictor region ===
dp_idx = [i for i, n in enumerate(g.node) if 'duration_predictor' in (n.name or '')]
dp_nodes = {i for i in dp_idx}
print('duration_predictor node count:', len(dp_nodes))

# Trunk tensors: inputs of dp nodes produced OUTSIDE the region
trunk_tensors = set()
for i in dp_idx:
    n = g.node[i]
    for inp in n.input:
        if not inp:
            continue
        pi = producer.get(inp)
        if pi not in dp_nodes:
            trunk_tensors.add(inp)
print()
print('=== TRUNK INPUTS INTO duration_predictor region ===')
for t in sorted(trunk_tensors):
    pi = producer.get(t)
    src = 'graph-output' if t in graph_outputs else ('graph-input' if t in graph_inputs else ('init/' + str(pi) if pi is None else ('node[%d]/%s' % (pi, g.node[pi].name or g.node[pi].op_type))))
    print('%-40s <- %s' % (t[:40], src))

# all nodes in region sorted topologically
print()
print('=== ALL duration_predictor NODES (topological) ===')
for i in sorted(dp_nodes):
    n = g.node[i]
    print('node[%05d] %-15s %s' % (i, n.op_type, n.name))

# Walk backward from log_duration restricted to the dp region, to get ordering
# from trunk to output
print()
print('=== log_duration path: nodes in reverse order (output -> trunk) ===')
out_t = 'log_duration'
pi = producer[out_t]
stack = [pi]
seen = set()
while stack:
    i = stack.pop()
    if i is None or i in seen:
        continue
    seen.add(i)
    n = g.node[i]
    print('node[%05d] %-15s %-60s outputs=%s' % (i, n.op_type, n.name, [o[:30] for o in n.output]))
    # push producers of inputs in topological priority
    preds = []
    for inp in n.input:
        if not inp:
            continue
        p = producer.get(inp)
        if p is not None:
            preds.append(p)
    # continue walking into the region only
    preds = [p for p in preds if p in dp_nodes]
    # reverse so lower indexes get popped first -> we walk forward eventually
    stack.extend(reversed(preds))