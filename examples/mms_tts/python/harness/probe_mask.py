import argparse
import numpy as np
import onnx
import onnxruntime as ort
import sys
sys.path.insert(0, '/home/ubuntu/rknn_model_zoo/examples/mms_tts/python')
from mms_tts import vocab, MAX_LENGTH, preprocess_input

FLOW2_X = '/duration_predictor/flows.2/Neg_output_0'
FLOW2_M = '/duration_predictor/flows.2/Mul_1_output_0'
FLOW3_X = '/duration_predictor/flows.3/Neg_output_0'
FLOW3_M = '/duration_predictor/flows.3/Mul_1_output_0'
PROBES = [FLOW2_X, FLOW2_M, FLOW3_X, FLOW3_M]


def main():
    p = argparse.ArgumentParser(
        description='Probe original (unpatched) encoder ONNX: emit spline x and mask (Mul_1=1 iff |x|<5) for each segment.')
    p.add_argument('--onnx', required=True)
    p.add_argument('--text', required=True)
    args = p.parse_args()

    m = onnx.load(args.onnx)
    existing = {o for n in m.graph.node for o in n.output}
    for name in PROBES:
        assert name in existing, 'missing tensor %s' % name
    for name in PROBES:
        vi = m.graph.output.add()
        vi.name = name
    sess = ort.InferenceSession(m.SerializeToString(), providers=['CPUExecutionProvider'])

    for seg_i, seg in enumerate(chunk_text(args.text)):
        inp, mask = preprocess_input(seg, vocab, MAX_LENGTH)
        res = sess.run(PROBES, {'input_ids': inp, 'attention_mask': mask})
        out = dict(zip(PROBES, res))
        real = int(mask.sum())
        print('== segment %d: chars=%d real_tokens=%d' % (seg_i, len(seg), real))
        for flow in ('flows.2', 'flows.3'):
            x = out['/duration_predictor/%s/Neg_output_0' % flow][0, 0, :real]
            mm = out['/duration_predictor/%s/Mul_1_output_0' % flow][0, 0, :real]
            missed = np.nonzero(np.abs(mm - 1) > 1e-6)[0]
            print('  %s: x real range=[%.4f, %.4f] max|x|=%.4f  mask==1 at all %d real pos: %s'
                  % (flow, x.min(), x.max(), np.abs(x).max(), len(x),
                     'YES' if len(missed) == 0 else ('NO -> idx ' + str(missed.tolist()))))


def chunk_text(text, max_chars=90):
    segs, cur = [], ''
    for ch in text:
        cur += ch
        if len(cur) >= max_chars and ch in '.!?;:,':
            segs.append(cur.strip())
            cur = ''
        elif len(cur) >= max_chars:
            sp = cur.rfind(' ')
            if sp >= max_chars // 2:
                segs.append(cur[:sp].strip())
                cur = cur[sp:]
    if cur.strip():
        segs.append(cur.strip())
    return segs


if __name__ == '__main__':
    main()