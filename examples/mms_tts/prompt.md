Context: I'm converting the MMS-TTS encoder (facebook/mms-tts-eng, exported as
mms_tts_eng_encoder_200.onnx via the patched modeling_vits_for_export_onnx.py)
to RKNN for RK3588 NPU inference. Toolchain: rknn-toolkit2==2.3.2, venv
.rknn_venv (torch==2.2.0, onnx==1.16.1, onnxruntime==1.23.2), scripts at
rknn_model_zoo/examples/mms_tts/python/convert.py + mms_tts.py, target
platform rk3588 (Orange Pi 5).

Some paths:
On this machine: 
- `~/code/rknn_model_zoo/` - current main branch + my extra changes for  `examples/mms_tts/python/convert.py`
- `examples/mms_tts/prompt.md` - this prompt if you need it
- `examples/mms_tts/python/softplus_fix.md` - results of previous agent run

on orange pi 
- `~/rknn_model_zoo/` - is synced with local
- `~/rknn_model_zoo/examples/mms_tts/python/.rknn_venv` - virtialenv with all the reqirements installed

You can run tests like this
```
scp convert.py orange_pi:~/rknn_model_zoo/examples/mms_tts/python/
ssh orange_pi 'cd ~/rknn_model_zoo/examples/mms_tts/python/  && source .rknn_venv/bin/activate && python convert.py ../model/mms_tts_eng_encoder_200.onnx rk3588 fp'.
```

Known problem: comparing ONNX (onnxruntime, ground truth) vs RKNN
(RKNNLite on-device) outputs for the SAME input on the SAME device:
- input_padding_mask: max_abs_diff = 0.0000
- log_duration: max_abs_diff = 2.4481, mean_abs_diff = 0.6801  <- broken
- prior_means: max_abs_diff = 2.09-2.21, mean_abs_diff = 0.005-0.05
- prior_log_variances: max_abs_diff = 0.22-0.38, mean_abs_diff = 0.0003-0.003

A previous investigation attempt produced an UNVERIFIED claim that mapping
`Where` and `Expand` ops to CPU via `rknn.config(op_target={'Where':'cpu',
'Expand':'cpu'})` fixes log_duration. That claim is NOT trusted — the code
shown didn't match the reported results, the output index used didn't match
log_duration, and no real before/after comparison was run. Do not reuse or
assume that conclusion. Start from scratch.

## Hard requirements — read before doing anything

1. **No number without a command that produced it.** Every diff value,
   tensor shape, or node name you report must come from a command you
   actually ran in this session. Paste the exact command AND its raw stdout
   immediately after each claim. If you didn't run something, say
   "NOT VERIFIED" instead of guessing a plausible number.
2. **No code/result mismatch.** If you show a code block as "the fix," its
   printed output in your log must come from running THAT EXACT code block,
   not a variant of it. Never show baseline code paired with post-fix
   numbers.
3. **Confirm tensor identity before measuring it.** Before comparing
   "log_duration", dump the ONNX graph output names (`onnx.load(path).graph.output`
   — print name + index for all 4 outputs) and confirm which index
   corresponds to log_duration by name, not by guessing position. Do the
   same for the RKNN model's output ordering (`rknn.load_onnx` /
   `rknn.list_outputs()` or equivalent) — RKNN does not guarantee it
   preserves ONNX output order.
4. **Every referenced file must exist and be shown to exist.** If you
   create an intermediate ONNX file (e.g. a version with nodes stripped),
   run `ls -la` on it and include the file size/timestamp in your report.
   Never reference a file you didn't create in this session.

## What I actually need you to do

1. **Trace the graph.** Using `onnx.load` + `onnx.helper.printable_graph`
   (or netron export to text), find the subgraph that produces the
   log_duration output specifically. Walk backward from that output node
   until you reach the shared trunk (where it merges with the path to
   prior_means/prior_log_variances). List every op type on the
   log_duration-exclusive path (expect things like Softplus, Cumsum, Where,
   ScatterND, ConstantOfShape, Range, Gather — but report what's ACTUALLY
   there, not what I expect). Paste the actual node list (names + op types),
   not a paraphrase.

2. **Get the RKNN build's verbose log.** Rebuild with
   `rknn.config(verbose=True, verbose_file='/tmp/rknn_build.log')` and grep
   that log file for any fusion/rewrite touching the op types found in step 1
   inside the duration_predictor subgraph. Paste the exact matching log
   lines (grep output), not a summary.

3. **Run a real accuracy_analysis.** Check if
   `rknn.accuracy_analysis(inputs=[...], target=...)` exists in
   rknn-toolkit2==2.3.2 (check via `dir(RKNN)` or the installed package's
   API docs) and run it if available. Paste the actual per-layer output it
   produces, especially the first layer where NPU vs "golden" simulator
   values diverge beyond ~1e-2. If unavailable, say so explicitly and fall
   back to step 4.

4. **Manual bisection fallback (only if step 3 unavailable).** Export a
   truncated ONNX graph that ends at progressively deeper points along the
   log_duration path (e.g. using `onnx.utils.extract_model` with different
   output node names), convert each truncated graph to RKNN, and diff
   against onnxruntime run on the same truncated graph. Report the diff at
   EACH truncation point in a table, so the exact op where divergence first
   exceeds ~1e-2 is identified. Show the extraction code, the conversion
   command, and the diff numbers for every truncation point — not just the
   final one.

5. **Test the actual fix, not a hypothesis.** Once you've identified the
   specific node(s) or op(s) responsible (from steps 2-4, not from assumption),
   apply `op_target` targeting ONLY those specific node names (get exact
   names via `rknn.list_devices()` or graph dump — do not use op-type-level
   keys like `'Where': 'cpu'` unless you've confirmed op_target in this
   rknn-toolkit2 version actually accepts op-type keys and not just node
   names; verify this by checking the API signature/docs directly). Then run
   ONE script that does both baseline and fixed inference back-to-back on
   the identical input, printing both diffs together, so before/after is
   directly comparable in a single output block.

6. **Version sensitivity check.** Confirm what torch/transformers version
   was actually used for the ONNX export currently on disk — check via
   `pip show torch transformers` in the export venv (not this conversion
   venv), and check the ONNX file's producer metadata
   (`onnx.load(path).producer_version` / opset imports) to see if it's
   consistent with modeling_vits_for_export_onnx.py's expectations. Report
   what you actually find; don't speculate about version mismatches without
   checking.

## Deliverable format

For each of the 6 steps above: command run -> raw output pasted -> your
interpretation, clearly separated. End with:
- (a) exact op(s)/node name(s) confirmed responsible, with the specific
  evidence line that proves it (not inferred)
- (b) the exact op_target config that fixes it, with node names verified to
  exist in this graph
- (c) one single before/after script + its actual output showing
  log_duration mean_abs_diff dropping to <0.01, run in this session
- (d) an honest list of anything from steps 1-6 you could NOT verify, rather
  than filling gaps with plausible-sounding claims
