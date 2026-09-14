import sys
import numpy as np
from rknn.api import RKNN

DEFAULT_QUANT = False

MASK_FLOWS = ('/duration_predictor/flows.2', '/duration_predictor/flows.3')
MASK_ONEA = ('Cast', 'Cast_1', 'Cast_2', 'Cast_3', 'Mul_1',
             'Cast_5', 'Cast_10', 'Cast_11', 'Cast_12', 'Cast_15')
MASK_ZEROA = ('Sub', 'Cast_4', 'Cast_7', 'Cast_8', 'Cast_9')


def patch_flows_masks(src):
    import onnx
    from onnx import TensorProto, helper, shape_inference

    m = onnx.load(src)
    have_mask = any('/duration_predictor/flows.2/Mul_1' in o for n in m.graph.node
                    for o in n.output)
    if not have_mask:
        return src

    mi = shape_inference.infer_shapes(m)
    g = m.graph
    vi = {v.name: v for v in mi.graph.value_info}

    consts = {}
    rewired = 0
    for fl in MASK_FLOWS:
        for vtype, names in (('ones', MASK_ONEA), ('zeros', MASK_ZEROA)):
            for t in names:
                full = fl + '/' + t + '_output_0'
                v = vi.get(full)
                if v is None:
                    continue
                shape = [d.dim_value for d in v.type.tensor_type.shape.dim]
                cname = fl.split('.')[1] + '_' + vtype + '_' + str(len(shape))
                consts.setdefault((cname, tuple(shape), vtype), []).append(full)

    for n in g.node:
        for i, inp in enumerate(n.input):
            for (cname, shape, vtype), srcs in consts.items():
                if inp in srcs:
                    n.input[i] = cname
                    rewired += 1
                    break

    added = []
    for (cname, shape, vtype), srcs in consts.items():
        arr = (np.ones(shape, dtype=np.float32) if vtype == 'ones'
               else np.zeros(shape, dtype=np.float32))
        added.append(helper.make_node(
            'Constant', [], [cname], name=cname + '_c',
            value=helper.make_tensor(cname, TensorProto.FLOAT, list(shape),
                                     arr.flatten().astype(np.float32))))
    for nd in reversed(added):
        g.node.insert(0, nd)

    dst = src.replace('.onnx', '_maskpatched.onnx')
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print('patched spline masks (%d rewires) -> %s' % (rewired, dst))
    return dst

def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]))
        print("       platform choose from [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b]")
        print("       dtype choose from [fp] for [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['i8', 'u8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type in ['i8', 'u8']:
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = model_path.replace('.onnx', '.rknn')

    return model_path, platform, do_quant, output_path

if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()

    model_path = patch_flows_masks(model_path)

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
