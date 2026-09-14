import os
import sys

sys.path.insert(0, '/home/ubuntu/rknn_model_zoo/examples/mms_tts/python')
from rknn.api import RKNN


def main():
    onnx_path = sys.argv[1]
    outdir = sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[], std_values=[], target_platform='rk3588')
    rknn.load_onnx(onnx_path)
    rknn.build(do_quantization=False)
    rknn_path = os.path.join(outdir, 'model.rknn')
    rknn.export_rknn(rknn_path)
    rknn.release()
    print('saved', rknn_path)


if __name__ == '__main__':
    main()