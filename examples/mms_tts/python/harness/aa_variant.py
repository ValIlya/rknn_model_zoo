import sys
import os
import json
import argparse
sys.path.insert(0, '/home/ubuntu/rknn_model_zoo/examples/mms_tts/python')
from rknn.api import RKNN

ONNX_MODEL = '/home/ubuntu/rknn_model_zoo/examples/mms_tts/model/mms_tts_eng_encoder_200.onnx'
INPUT_IDS = '/tmp/input_ids.npy'
ATTN_MASK = '/tmp/attention_mask.npy'


def main():
    p = argparse.ArgumentParser(description='Build + accuracy_analysis with a given op_target.')
    p.add_argument('--outdir', required=True)
    p.add_argument('--onnx', default=None, help='override ONNX path')
    p.add_argument('--op-target', default='{}', help='JSON dict: node name -> "cpu"')
    p.add_argument('--opt-level', type=int, default=3)
    p.add_argument('--quant', dest='quant', action='store_true', help='do_quantization=True (default False)')
    p.add_argument('--build-only', dest='build_only', action='store_true',
                   help='stop after build (no accuracy_analysis)')
    p.add_argument('--verbose-log', default=None, help='path for RKNN verbose build log')
    args = p.parse_args()

    onnx_path = args.onnx or ONNX_MODEL
    op_target = json.loads(args.op_target)
    print('onnx   =', onnx_path, flush=True)
    print('op_target =', json.dumps(op_target, indent=2), flush=True)
    os.makedirs(args.outdir, exist_ok=True)

    rknn = RKNN(verbose=args.verbose_log is not None, verbose_file=args.verbose_log)

    print('[1] Config (target_platform=rk3588, optimization_level=%d, op_target=%s)...'
          % (args.opt_level, 'SET' if op_target else 'NONE'), flush=True)
    ret = rknn.config(mean_values=[], std_values=[], target_platform='rk3588',
                      optimization_level=args.opt_level, op_target=op_target)
    print('config ret:', ret, flush=True)

    print('[2] Loading ONNX...', flush=True)
    ret = rknn.load_onnx(onnx_path)
    if ret != 0:
        print('load_onnx failed:', ret)
        sys.exit(1)

    print('[3] Building (do_quantization=%s)...' % args.quant, flush=True)
    ret = rknn.build(do_quantization=args.quant)
    if ret != 0:
        print('build failed:', ret)
        sys.exit(1)

    rknn_out = os.path.join(args.outdir, 'model.rknn')
    print('[4] Exporting %s ...' % rknn_out, flush=True)
    ret = rknn.export_rknn(rknn_out)
    print('export ret:', ret, flush=True)

    if args.build_only:
        print('[build-only] done.', flush=True)
        rknn.release()
        return

    import numpy as np
    print('[4] Loading inputs...', flush=True)
    input_ids = np.load(INPUT_IDS)
    attn_mask = np.load(ATTN_MASK)
    print('  input_ids:', input_ids.shape, input_ids.dtype)
    print('  attn_mask:', attn_mask.shape, attn_mask.dtype)

    print('[5] Running accuracy_analysis into %s ...' % args.outdir, flush=True)
    ret = rknn.accuracy_analysis(inputs=[input_ids, attn_mask],
                                 output_dir=args.outdir,
                                 target='rk3588')
    print('accuracy_analysis returned:', ret, flush=True)
    rknn.release()
    print('Done.', flush=True)


if __name__ == '__main__':
    main()