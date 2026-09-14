Context: I'm converting a VITS-based MMS-TTS model (facebook/mms-tts-eng) to RKNN 
format for Rockchip RK3588 NPU inference, using rknn-toolkit2 v2.3.2 and the 
conversion script at rknn_model_zoo/examples/mms_tts/python/convert.py + 
mms_tts.py. The encoder ONNX model was exported via a patched HuggingFace 
transformers VitsModel (modeling_vits_for_export_onnx.py from rknn_model_zoo).

Problem: When comparing encoder outputs between the ONNX model (onnxruntime, 
ground truth) and the RKNN-converted model (RKNNLite/RKNN on-device inference) 
for the *same* text input, on the *same* RK3588 device:
  - input_padding_mask: max_abs_diff = 0.0000 (perfect match)
  - log_duration: max_abs_diff = 2.4481, mean_abs_diff = 0.6801 (BAD)
  - prior_means: max_abs_diff = 2.09-2.21, mean_abs_diff = 0.005-0.05 (mostly OK, 
    but has occasional large outliers)
  - prior_log_variances: max_abs_diff = 0.22-0.38, mean_abs_diff = 0.0003-0.003 
    (small, likely acceptable)

This numerical divergence in log_duration causes the final synthesized audio to 
have correct pitch and overall duration/length, but with word chunks/phonemes 
scrambled/out of order — the alignment computed from log_duration (via the 
attn matrix in middle_process()) is wrong, even though the total predicted 
length stays similar.

Setting rknn.config(optimization_level=0) made ZERO difference (identical 
diffs to optimization_level=3 default), which rules out standard graph fusion/
constant-folding as the cause.

I need you to investigate WHY log_duration diverges so much more than the 
other three outputs. Please:

1. Locate the ONNX graph nodes that produce log_duration specifically — trace 
   backward from the "log_duration" output tensor through the duration_predictor 
   subgraph (stochastic duration predictor, piecewise rational quadratic spline 
   flows named /duration_predictor/flows.0 through flows.5 or similar in the 
   graph). Identify which ops sit on this path that DON'T sit on the path to 
   prior_means/prior_log_variances (which are comparatively much more accurate).

2. In the RKNN build verbose log (rebuild with rknn.config(verbose=True, 
   verbose_file='...')), search for any node fusion/rewrite specifically 
   touching Softplus, Exp, Where, Equal, ConstantOfShape, ScatterND, Range, 
   Gather, or Cumsum ops within the duration_predictor/flows.* subgraphs. 
   Pay special attention to entries like "unsqueeze_to_4d_softplus" or 
   "bypass_two_reshape" that rewrite Softplus/Squeeze sequences — these look 
   suspicious because Softplus normalizes spline "bin widths/heights" in VITS's 
   stochastic duration predictor, and any precision loss there would directly 
   distort log_duration.

3. Check whether RKNN is silently running any of these ops in reduced 
   precision (fp16 instead of fp32) even with do_quantization=False — RKNN 
   NPU compute is natively fp16, and ops NOT explicitly forced to CPU run in 
   NPU fp16, which can be enough to disturb a stochastic sampling / cumulative 
   sum pipeline like this one. Look for a way to force the duration_predictor 
   subgraph nodes onto CPU (fp32) via rknn.config(op_target={...}), and 
   identify the correct *current* internal node names for this graph (not the 
   old '7398-rs'/'5773-rs' from the original rknn_model_zoo script, which don't 
   exist in this exported graph — confirmed by testing, they raised 
   "Invalid key" errors).

4. As a diagnostic, write a small script that runs BOTH onnxruntime and RKNN 
   inference on the same input, but instead of just comparing the 4 final 
   outputs, use rknn.accuracy_analysis() (if available in this rknn-toolkit2 
   version) or manually extract intermediate tensor values layer-by-layer 
   within the duration_predictor subgraph, to find the earliest point in the 
   graph where RKNN's numerical output starts diverging meaningfully from 
   ONNX's. This will pinpoint the exact problematic op instead of guessing.

5. Cross-check torch/transformers version sensitivity: the ONNX export was 
   done with torch==2.2.0 (or 2.4.1, inconsistent installs happened during 
   setup) and transformers==4.39.3, whereas the original rknn_model_zoo 
   MMS-TTS example may have been validated against older torch (~1.10-1.13). 
   Check the export patch file (modeling_vits_for_export_onnx.py) for any 
   version-sensitive tensor ops (e.g. torch.cumsum, searchsorted-like logic in 
   the spline flow) that could produce a subtly different ONNX graph structure 
   depending on torch version, which RKNN then handles differently than 
   onnxruntime does.

Repository/paths for reference:
- Model: rknn_model_zoo/examples/mms_tts/python/
- README_EXPORT.md - instructions
- convert_to_rknn.sh high-level installation+conversion
- convert.py (conversion), mms_tts.py (inference/test script)
- ONNX models: ../model/mms_tts_eng_encoder_200.onnx, ../model/mms_tts_eng_decoder_200.onnx
- RKNN models: ../model/mms_tts_eng_encoder_200.rknn
- Venv: .rknn_venv (rknn-toolkit2==2.3.2, torch==2.2.0, onnx==1.16.1, onnxruntime==1.23.2)
- Target platform: rk3588 (Orange Pi 5, RK3588 SoC)

Try to fix: (a) the exact ops on the log_duration-only path, (b) any 
RKNN-side rewrite/precision issue found on that path, (c) a concrete fix — 
either op_target CPU-forcing with correct node names, a different opset/export 
setting, or a torch/transformers version pin — with instructions to verify the 
fix numerically (re-run the ONNX-vs-RKNN diff comparison and confirm 
log_duration mean_abs_diff drops to the same order of magnitude as 
prior_log_variances, i.e. < 0.01).
Feel free to ask if stuck.
