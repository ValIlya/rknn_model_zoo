import os
import sys

import numpy as np
import onnx
from onnx import helper, TensorProto

SRC = '/home/ubuntu/rknn_model_zoo/examples/mms_tts/model/mms_tts_eng_encoder_200.onnx'


def patch_mask_ones(dst):
    m = onnx.load(SRC)
    g = m.graph
    N = 200
    for n in list(g.node):
        # Rewrite the producer of the mask feeding Mul_11 in flows.2
        if n.op_type == 'Mul' and n.name == '/duration_predictor/flows.2/Mul_11':
            n.input[0] = '/duration_predictor/flows.2/ones_mask'
    ones = helper.make_node(
        'Constant', [], ['/duration_predictor/flows.2/ones_mask'],
        name='/duration_predictor/flows.2/ones_mask_const',
        value=helper.make_tensor('/duration_predictor/flows.2/ones_mask',
                                 TensorProto.FLOAT, [1, 1, 1, N],
                                 np.ones(N, dtype=np.float32)))
    g.node.insert(0, ones)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print('saved', dst)


def patch_split_copy(dst):
    m = onnx.load(SRC)
    g = m.graph
    # Insert an Identity copying flows.2/Split_output_1 -> copy, point consumers at it
    consumers = []
    for n in g.node:
        for i, inp in enumerate(n.input):
            if inp == '/duration_predictor/flows.2/Split_output_1':
                consumers.append((n.name, i))
    print('consumers of flows.2 Split_output_1:', consumers)
    for n in g.node:
        for i, inp in enumerate(n.input):
            if inp == '/duration_predictor/flows.2/Split_output_1':
                n.input[i] = '/duration_predictor/flows.2/Split_output_1_copy'
    copy = helper.make_node('Identity',
                            ['/duration_predictor/flows.2/Split_output_1'],
                            ['/duration_predictor/flows.2/Split_output_1_copy'],
                            name='/duration_predictor/flows.2/Split_output_1_copy')
    g.node.insert(0, copy)
    # fix topological order: Identity must come after its producer Split
    for idx, n in enumerate(g.node):
        if n.name == '/duration_predictor/flows.2/Split':
            g.node.remove(copy)
            g.node.insert(idx + 1, copy)
            break
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print('saved', dst, 'with', len(consumers), 'consumers rerouted')


if __name__ == '__main__':
    which = sys.argv[1]
    dst = sys.argv[2]
    if which == 'mask_ones':
        patch_mask_ones(dst)
    elif which == 'split_copy':
        patch_split_copy(dst)
    else:
        raise SystemExit('unknown patch')