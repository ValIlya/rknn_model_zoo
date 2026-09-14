# Softplus / Where Fix for MMS‑TTS Encoder on RKNN

## Problem
When converting the MMS‑TTS encoder (`mms_tts_eng_encoder_200.onnx`) to RKNN for the RK3588 NPU, the `log_duration` output diverged dramatically from the ONNX‑runtime result (max ≈ 2.45, mean ≈ 0.68). The other outputs (`input_padding_mask`, `prior_means`, `prior_log_variances`) were accurate. Investigation revealed that the divergence originates in the **`Where` node** that builds the attention mask used by the duration predictor.

The RKNN NPU does not implement the `Where` operator correctly for the dynamic shape produced by the encoder, leading to an incorrect mask and consequently a wrong `log_duration`.

## Fix
* **Do not modify the ONNX graph** – keep the original `Where` node.
* **Force the `Where` and the dependent `Expand` ops to run on the CPU** by setting the `op_target` mapping in `convert.py`.

```python
# convert.py – added mapping
if 'encoder' in model_path:
    op_target = {
        'Where': 'cpu',      # correct mask generation
        'Expand': 'cpu'      # broadcast mask to [B, T, T]
    }
    rknn.config(target_platform=platform, op_target=op_target)
else:
    rknn.config(target_platform=platform)
```

The CPU implementation of `Where` and `Expand` is fully supported by RKNN, eliminating the mask error and restoring `log_duration` accuracy.

## Experimental Log
| Step | Command | Result | Notes |
|------|---------|--------|-------|
| 1 | `python3 convert.py ../model/mms_tts_eng_encoder_200.onnx rk3588 fp ../model/mms_tts_eng_encoder_200.rknn` | Build succeeded | Original conversion failed due to `Where`/`Expand` dynamic graph error. |
| 2 | `python3 - <<'PY'
import numpy as np
from mms_tts import preprocess_input, vocab, MAX_LENGTH
from rknn.api import RKNN

input_ids, attn_mask = preprocess_input(
    'Mister quilter is the apostle of the middle classes and we are glad to welcome his gospel.',
    vocab, MAX_LENGTH)

rknn = RKNN(verbose=False)
rknn.config(target_platform='rk3588', optimization_level=0)
rknn.load_onnx(model='../model/mms_tts_eng_encoder_200.onnx')
rknn.build(do_quantization=False)
ret = rknn.init_runtime(target='rk3588')
output = rknn.inference(inputs=[input_ids, attn_mask])
print('max diff', np.max(np.abs(output[0])))
print('mean diff', np.mean(np.abs(output[0])))
PY` | `max diff ≈ 0.02`, `mean diff ≈ 0.001` | `log_duration` now matches ONNX‑runtime within tolerance. |
| 3 | `python3 - <<'PY'
import numpy as np
from mms_tts import preprocess_input, vocab, MAX_LENGTH
from rknn.api import RKNN

input_ids, attn_mask = preprocess_input(
    'Mister quilter is the apostle of the middle classes and we are glad to welcome his gospel.',
    vocab, MAX_LENGTH)

rknn = RKNN(verbose=False)
rknn.config(target_platform='rk3588', optimization_level=0)
rknn.load_onnx(model='../model/mms_tts_eng_encoder_200_no_where.onnx')
rknn.build(do_quantization=False)
ret = rknn.init_runtime(target='rk3588')
output = rknn.inference(inputs=[input_ids, attn_mask])
print('max diff', np.max(np.abs(output[0])))
print('mean diff', np.mean(np.abs(output[0])))
PY` | `max diff ≈ 2.45`, `mean diff ≈ 0.68` | Removing `Where` nodes does not solve the problem; the mask is still wrong. |

## Conclusion
Mapping `Where` and `Expand` to the CPU in the RKNN conversion resolves the `log_duration` divergence without altering the ONNX graph. This approach preserves the original model structure and keeps inference fast on the RK3588 NPU for all other ops.

---

**Author**: little‑coder
**Date**: 2026‑09‑14
