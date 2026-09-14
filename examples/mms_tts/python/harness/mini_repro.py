import argparse
import os
import sys

import numpy as np
import onnx
from onnx import helper, TensorProto

sys.path.insert(0, '/home/ubuntu/rknn_model_zoo/examples/mms_tts/python')
from rknn.api import RKNN

N = 200


def build_mask_graph(out_path):
    backend = []
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 1, N])
    neg = helper.make_node('Neg', ['x'], ['neg_out'], name='/m/Neg')
    c15 = helper.make_node('Constant', [], ['/m/Constant_15_output_0'],
                           name='/m/Constant_15', value=helper.make_tensor('/m/Constant_15_output_0', TensorProto.FLOAT, [1], [5.0]))
    less = helper.make_node('Less', ['neg_out', '/m/Constant_15_output_0'], ['less_out'], name='/m/Less')
    cast1 = helper.make_node('Cast', ['less_out'], ['/m/Cast_output_0'], to=TensorProto.FLOAT, name='/m/Cast')
    cast2 = helper.make_node('Cast', ['/m/Cast_output_0'], ['/m/Cast_1_output_0'], to=TensorProto.FLOAT, name='/m/Cast_1')
    c16 = helper.make_node('Constant', [], ['/m/Constant_16_output_0'],
                           name='/m/Constant_16', value=helper.make_tensor('/m/Constant_16_output_0', TensorProto.FLOAT, [1], [-5.0]))
    great = helper.make_node('Greater', ['neg_out', '/m/Constant_16_output_0'], ['great_out'], name='/m/Greater')
    cast3 = helper.make_node('Cast', ['great_out'], ['/m/Cast_2_output_0'], to=TensorProto.FLOAT, name='/m/Cast_2')
    cast4 = helper.make_node('Cast', ['/m/Cast_2_output_0'], ['/m/Cast_3_output_0'], to=TensorProto.FLOAT, name='/m/Cast_3')
    mul1 = helper.make_node('Mul', ['/m/Cast_1_output_0', '/m/Cast_3_output_0'], ['/m/Mul_1_output_0'], name='/m/Mul_1')
    c17 = helper.make_node('Constant', [], ['/m/Constant_17_output_0'],
                           name='/m/Constant_17', value=helper.make_tensor('/m/Constant_17_output_0', TensorProto.INT32, [1], [1]))
    sub = helper.make_node('Sub', ['/m/Mul_1_output_0', '/m/Constant_17_output_0'], ['/m/Sub_output_0'], name='/m/Sub')
    cast5 = helper.make_node('Cast', ['/m/Mul_1_output_0'], ['/m/Cast_5_output_0'], to=TensorProto.FLOAT, name='/m/Cast_5')
    cast10 = helper.make_node('Cast', ['/m/Cast_5_output_0'], ['/m/Cast_10_output_0'], to=TensorProto.FLOAT, name='/m/Cast_10')
    cast11 = helper.make_node('Cast', ['/m/Cast_10_output_0'], ['/m/Cast_11_output_0'], to=TensorProto.FLOAT, name='/m/Cast_11')
    cast12 = helper.make_node('Cast', ['/m/Cast_11_output_0'], ['/m/Cast_12_output_0'], to=TensorProto.FLOAT, name='/m/Cast_12')
    mul11 = helper.make_node('Mul', ['/m/Cast_12_output_0', 'x'], ['/m/Mul_11_output_0'], name='/m/Mul_11')
    y_out = helper.make_tensor_value_info('/m/Mul_11_output_0', TensorProto.FLOAT, [1, 1, 1, N])
    out_m1 = helper.make_tensor_value_info('/m/Mul_1_output_0', TensorProto.FLOAT, [1, 1, 1, N])
    out_c12 = helper.make_tensor_value_info('/m/Cast_12_output_0', TensorProto.FLOAT, [1, 1, 1, N])
    graph = helper.make_graph([c15, neg, less, cast1, cast2, c16, great,
                               cast3, cast4, mul1, c17, sub, cast5, cast10,
                               cast11, cast12, mul11],
                              'mini_mask', [x], [y_out, out_m1, out_c12])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    print('saved mini ONNX:', out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mode', default='real', choices=['real', 'sim'])
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    onnx_path = os.path.join(args.outdir, 'mini_mask.onnx')
    build_mask_graph(onnx_path)

    rng = np.random.default_rng(args.seed)
    x = rng.uniform(-2.03, 1.77, size=(1, 1, 1, N)).astype(np.float32)

    rknn = RKNN(verbose=False)
    print('config...', flush=True)
    rknn.config(target_platform='rk3588')
    rknn.load_onnx(onnx_path)
    print('build...', flush=True)
    rknn.build(do_quantization=False)
    rknn_path = os.path.join(args.outdir, 'mini_mask.rknn')
    rknn.export_rknn(rknn_path)
    rknn.release()

    rknn2 = RKNN(verbose=False)
    rknn2.load_rknn(rknn_path)
    if args.mode == 'real':
        ret = rknn2.init_runtime(target='rk3588')
    else:
        ret = rknn2.init_runtime(target=None)
    print('init_runtime ret:', ret)
    outs = rknn2.inference(inputs=[x])
    names = ['Mul_11(x*mask)', 'Mul_1(mask)', 'Cast_12(mask2)']
    for name, o in zip(names, outs):
        a = np.array(o).flatten()
        print('%-18s n=%-6d [%11.6g, %11.6g] uniq=%-5d' % (name, a.size, a.min(), a.max(), len(np.unique(np.round(a, 4)))))
    rknn2.release()


if __name__ == '__main__':
    main()