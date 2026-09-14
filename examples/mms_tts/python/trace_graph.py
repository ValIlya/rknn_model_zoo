import sys
import onnx
from collections import deque

model_path = '../model/mms_tts_eng_encoder_200.onnx'
m = onnx.load(model_path)

graph = m.graph

print('=== GRAPH META ===')
print('producer_name:', m.producer_name)
print('producer_version:', m.producer_version)
print('opset imports:', [(o.domain, o.version) for o in m.opset_import])

print()
print('=== GRAPH INPUTS ===')
for i, inp in enumerate(graph.input):
    shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
    print('input[%d] name=%r shape=%s dtype=%d' % (i, inp.name, shape, inp.type.tensor_type.elem_type))

print()
print('=== GRAPH OUTPUTS ===')
for i, out in enumerate(graph.output):
    shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
    print('output[%d] name=%r shape=%s dtype=%d' % (i, out.name, shape, out.type.tensor_type.elem_type))

print()
print('=== ALL NODES (topological index / name / op_type) ===')
for i, n in enumerate(graph.node):
    print('node[%03d] %-50s %s' % (i, n.name or '(no-name)', n.op_type))

output_names = {o.name for o in graph.output}

# Build forward adjacency: tensor -> producer node index
producer = {}
for i, n in enumerate(graph.node):
    for out in n.output:
        producer[out] = i

# Build consumers: tensor -> list of node indices
consumers = {}
for i, n in enumerate(graph.node):
    for inp in n.input:
        if inp and inp not in output_names:
            consumers.setdefault(inp, []).append(i)

# Backward walk from each graph output until reaching a graph input.
print()
print('=== BACKWARD WALK FROM EACH GRAPH OUTPUT ===')
for out in graph.output:
    seen = set()
    stack = [out.name]
    path = []
    while stack:
        t = stack.pop()
        if t in seen or not t:
            continue
        seen.add(t)
        if t in [i.name for i in graph.input]:
            path.append(('INPUT', t))
            break
        pi = producer.get(t)
        if pi is None:
            path.append(('UNKNOWN_SRC', t))
            continue
        node = graph.node[pi]
        path.append((node.op_type, node.name or ('node[%d]' % pi)))
        for tin in node.input:
            if tin and tin not in [i.name for i in graph.input]:
                if tin not in seen:
                    stack.append(tin)
    print()
    print('--- path to output %r ---' % out.name)
    for step in reversed(path):
        print('   ', step)