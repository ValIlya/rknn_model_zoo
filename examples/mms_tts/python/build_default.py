import sys
from rknn.api import RKNN

rknn = RKNN(verbose=True, verbose_file='/tmp/rknn_build_default.log')
print('--> Config')
rknn.config(target_platform='rk3588')
print('done')
print('--> Loading model')
ret = rknn.load_onnx(model='../model/mms_tts_eng_encoder_200.onnx')
if ret != 0:
    print('Load model failed!', ret)
    sys.exit(ret)
print('done')
print('--> Building model')
ret = rknn.build(do_quantization=False)
if ret != 0:
    print('Build model failed!', ret)
    sys.exit(ret)
print('done')
ret = rknn.export_rknn('/tmp/mms_tts_eng_encoder_200_default.rknn')
if ret != 0:
    print('Export rknn failed!', ret)
    sys.exit(ret)
print('done')
rknn.release()
print('ALL OK')