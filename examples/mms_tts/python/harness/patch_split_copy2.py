import sys

import onnx
from onnx import helper

SRC = '/tmp/mtts_exp/patched_mask_all.onnx'


def patch(dst):
    m = onnx.load(SRC)
    g = m.graph
    split_out = '/duration_predictor/flows.2/Split_output_1'
    copy_out = '/duration_predictor/flows.2/Split_output_1_via_copy'

    consumers = [n for n in g.node if split_out in n.input]
    print('consumers of Split_output_1:', [n.name for n in consumers])

    for n in consumers:
        for i, inp in enumerate(n.input):
            if inp == split_out:
                n.input[i] = copy_out
    split_node = next(n for n in g.node if split_out in n.output)
    g.node.insert(list(g.node).index(split_node) + 1,
                  helper.make_node('Identity', [split_out], [copy_out], name='flows.2_split_copy'))
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print('saved', dst)


if __name__ == '__main__':
    patch(sys.argv[1])