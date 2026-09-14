import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference

SRC = '/home/ubuntu/rknn_model_zoo/examples/mms_tts/model/mms_tts_eng_encoder_200.onnx'

# The BP mask chain in each spline flow (identical topology in flows.2 and .3):
#   mask(x) = Cast(Less(Neg,5)) & Cast(Greater(Neg,-5))  -> value 1 for |x|<5
#   mask-1   = Sub(Mul_1, 1) -> value 0
# tensors with value 1:  Cast, Cast_1, Cast_2, Cast_3, Mul_1, Cast_5, Cast_10, Cast_11, Cast_12, Cast_15
# tensors with value 0:  Sub, Cast_4, Cast_7, Cast_8, Cast_9
ONEA = ('Cast', 'Cast_1', 'Cast_2', 'Cast_3', 'Mul_1', 'Cast_5', 'Cast_10', 'Cast_11', 'Cast_12', 'Cast_15')
ZEROA = ('Sub', 'Cast_4', 'Cast_7', 'Cast_8', 'Cast_9')
FLOWS = ('/duration_predictor/flows.2', '/duration_predictor/flows.3')


def patch(dst):
    m = onnx.load(SRC)
    mi = shape_inference.infer_shapes(m)
    g = m.graph
    vi = {v.name: v for v in mi.graph.value_info}

    def shape_of(t):
        v = vi.get(t)
        if v is None or not v.type.tensor_type.HasField('shape'):
            return None
        return [d.dim_value for d in v.type.tensor_type.shape.dim]

    # build constant arrays per (flow, value-type) from real inferred shapes
    const_names = {}
    rewired = 0
    for fl in FLOWS:
        for vtype, names in (('ones', ONEA), ('zeros', ZEROA)):
            for t in names:
                full = fl + '/' + t + '_output_0'
                if full not in vi:
                    continue
                shape = shape_of(full)
                if shape is None:
                    print('WARN no shape for', full)
                    continue
                cname = fl.split('.')[1] + '_' + vtype + '_' + str(len(shape))
                const_names.setdefault((cname, tuple(shape), vtype), []).append(full)
    print('user indicates chains:', {k[0]: len(v) for k, v in const_names.items()})

    # reroute consumers
    for n in g.node:
        for i, inp in enumerate(n.input):
            for (cname, shape, vtype), srcs in const_names.items():
                if inp in srcs:
                    n.input[i] = cname
                    rewired += 1
                    break
    # add Constant nodes
    added = []
    for (cname, shape, vtype), srcs in const_names.items():
        arr = np.ones(shape, dtype=np.float32) if vtype == 'ones' else np.zeros(shape, dtype=np.float32)
        added.append(helper.make_node(
            'Constant', [], [cname], name=cname + '_c',
            value=helper.make_tensor(cname, TensorProto.FLOAT, list(shape), arr.flatten().astype(np.float32))))
    for nd in reversed(added):
        g.node.insert(0, nd)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print('saved', dst, 'rewired', rewired)


if __name__ == '__main__':
    patch(sys.argv[1])