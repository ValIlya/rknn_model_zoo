import os
import sys

import numpy as np
import onnx
from onnx import helper, TensorProto

SRC = '/home/ubuntu/rknn_model_zoo/examples/mms_tts/model/mms_tts_eng_encoder_200.onnx'
N = 200


def patch(dst):
    m = onnx.load(SRC)
    g = m.graph
    # semantic values of the flows.2 mask chain (all degenerate for |x|<5)
    #  1: Less(Neg(x),5)  Cast Cast_1  Greater(Neg,-5) Cast_2 Cast_3 Mul_1 Cast_5..Cast_12
    #  0: Sub = Mul_1-1   Cast_4..9 -> Mul_10 = 0 * Split_output_1 (this one is Mul(op, Split))
    repl = {
        '/duration_predictor/flows.2/Cast_12_output_0': ('ones_200', [1, 1, N]),
        '/duration_predictor/flows.2/Cast_15_output_0': ('ones_200x1', [1, 1, N, 1]),
        '/duration_predictor/flows.2/Cast_5_output_0': ('ones_200', [1, 1, N]),
        '/duration_predictor/flows.2/Cast_10_output_0': ('ones_200', [1, 1, N]),
        '/duration_predictor/flows.2/Cast_11_output_0': ('ones_200', [1, 1, N]),
        '/duration_predictor/flows.2/Cast_1_output_0': ('ones_200', [1, 1, N]),
        '/duration_predictor/flows.2/Cast_3_output_0': ('ones_200', [1, 1, N]),
        '/duration_predictor/flows.2/Mul_1_output_0': ('ones_200', [1, 1, N]),
        '/duration_predictor/flows.2/Sub_output_0': ('zero_200', [1, 1, N]),
        '/duration_predictor/flows.2/Cast_4_output_0': ('zero_200', [1, 1, N]),
        '/duration_predictor/flows.2/Cast_7_output_0': ('zero_200', [1, 1, N]),
        '/duration_predictor/flows.2/Cast_8_output_0': ('zero_200', [1, 1, N]),
        '/duration_predictor/flows.2/Cast_9_output_0': ('zero_200', [1, 1, N]),
    }
    consts = {}
    for t, (cname, shape) in repl.items():
        if cname not in consts:
            consts[cname] = (np.ones(shape, dtype=np.float32) if cname.startswith('ones') else np.zeros(shape, dtype=np.float32))
    # reroute consumers into the mask-chain tensors
    rewired = 0
    for n in g.node:
        for i, inp in enumerate(n.input):
            if inp in repl:
                n.input[i] = repl[inp][0]
                rewired += 1
    # insert Constant nodes
    for cname, arr in consts.items():
        node = helper.make_node(
            'Constant', [], [cname], name=cname + '_const',
            value=helper.make_tensor(cname, TensorProto.FLOAT, list(arr.shape), arr.flatten().astype(np.float32)))
        g.node.insert(0, node)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print('saved', dst, 'with', rewired, 'input rewires')


if __name__ == '__main__':
    patch(sys.argv[1])