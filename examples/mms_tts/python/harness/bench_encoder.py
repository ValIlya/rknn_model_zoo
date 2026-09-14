import sys
import time
import argparse
import statistics
import numpy as np
sys.path.insert(0, '/home/ubuntu/rknn_model_zoo/examples/mms_tts/python')
from rknn.api import RKNN

INPUT_IDS = '/tmp/input_ids.npy'
ATTN_MASK = '/tmp/attention_mask.npy'


def main():
    p = argparse.ArgumentParser(description='NPU latency + log_duration sanity for the encoder rknn.')
    p.add_argument('--rknn', required=True)
    p.add_argument('--runs', type=int, default=8)
    p.add_argument('--warmup', type=int, default=2)
    args = p.parse_args()

    input_ids = np.load(INPUT_IDS)
    attn_mask = np.load(ATTN_MASK)

    rknn = RKNN()
    print('loading', args.rknn, flush=True)
    ret = rknn.load_rknn(args.rknn)
    if ret != 0:
        print('load_rknn failed:', ret)
        sys.exit(1)
    ret = rknn.init_runtime(target='rk3588')
    if ret != 0:
        print('init_runtime failed:', ret)
        sys.exit(1)

    for _ in range(args.warmup):
        rknn.inference(inputs=[input_ids, attn_mask])

    times = []
    last = None
    for _ in range(args.runs):
        t0 = time.perf_counter()
        last = rknn.inference(inputs=[input_ids, attn_mask])
        times.append(time.perf_counter() - t0)

    ld = np.array(last[0])
    runiq = len(np.unique(np.round(ld.ravel(), 4)))
    print('log_duration: shape=%s range=[%.6f, %.6f] uniq(4dp)=%d'
          % (ld.shape, ld.min(), ld.max(), runiq), flush=True)
    print('latency: mean=%.5fs min=%.5fs max=%.5fs' % (statistics.mean(times), min(times), max(times)), flush=True)
    print('all: %s' % [round(t, 5) for t in times], flush=True)
    rknn.release()


if __name__ == '__main__':
    main()