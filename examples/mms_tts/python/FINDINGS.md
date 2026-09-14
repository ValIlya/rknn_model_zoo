# MMS-TTS RKNN: `log_duration` NPU Collapse — Findings

## 1. Problem

The RKNN-converted MMS-TTS encoder (`mms_tts_eng_encoder_200.onnx`, RK3588 NPU)
produces a **completely wrong and constant** `log_duration` on the real NPU,
while the RKNN simulator (host-side emulation) matches PyTorch/ONNX ground truth
nearly exactly.

Measured on the exact per-layer snapshots dumped by `rknn-toolkit2`'s
`accuracy_analysis` (golden = ONNX fp32, simulator = RKNN sim, runtime = real NPU):

| tensor | golden range | simulator range | runtime range |
|---|---|---|---|
| `log_duration` | -2.23 … 2.44 | -2.23 … 2.44 | **-0.0114 … -0.0000 (constant)** |
| `flows.2/Split_output_0` | -1.20 … 1.33 | -1.19 … 1.33 | **0.0000 … 0.0010 (constant)** |
| `flows.2/Concat_57` | -2.18 … 2.41 | -2.18 … 2.41 | **0.0000 … 0.0010 (constant)** |
| `flows.2/Softplus`, `Pow`, `Add_39/40` chain | variable | variable | **constant** (softplus→ln2=0.6929) |
| `flows.3/Concat_57`, `Mul_81` | -2.03 … 1.77 | -2.03 … 1.77 | -2.03 … 1.77 (OK) |
| `prior_means` | variable | variable | close (cos 0.9973) |
| `prior_log_variances` | variable | variable | close (cos 0.99997) |

`error_analysis.txt` (full, 992 rows) is in `acc_data/error_analysis.txt`; name→file
map is `acc_data/map_name_to_file.txt`. Per-tensor snapshots for the tensors above
are in `acc_data/{golden,simulator,runtime}/`.

## 2. Where exactly the collapse happens

The duration predictor is a stack of 4 affine coupling "flows" applied to the
text-encoder features. Numeric path:

```
flows.conv_pre → conv_dds (3 dilated blocks) → conv_proj
   → flows.4 → flows.3 → flows.2 → flows.0 → Split → log_duration
```

Each flow:
- `Split` splits its input into `Split_output_0` (data to transform) and
  `Split_output_1` (pass-through),
- `Concat_57` = `Concat(Split_output_0, Add_40)` (transformed),
- `Mul_81` = `Concat_57 * input_padding_mask`,
- passed to the next flow.

Collapse trace on the real NPU:

1. **`flows.4` and `flows.3` are fine.** Their intermediate Conv/Softplus/Pow and
   final `flows.3/Mul_81` outputs match golden closely (max|rt−sim| ≈ 1.1 on a
   single element, cos > 0.99).
2. **`flows.2` breaks.** Inside `flows.2` the whole spline softplus/variance
   chain (`GatherElements → Softplus → Pow → Div → Add_37…Add_40`)
   comes out **constant on the NPU** (`Softplus` = `ln 2 = 0.6929` for every
   element, i.e. `softplus(0)`, then the chain collapses to `0.001`), so
   `flows.2/Add_40 = 0.001` and `flows.2/Split_output_0 = 0.001`.
   Because `Concat_57 = Concat(Split_output_0, Add_40)`, the whole
   `flows.2/Concat_57` becomes `0.001`.
3. `flows.2/Mul_81` (= `Concat_57 * mask`) is therefore `≈ 0.001`,
   `flows.0` (a small net on top of it) computes a constant, and the final
   `log_duration` = constant `-0.0114`.

Key evidence (direct file comparisons, `check_concat.py` output, stored locally):

```
== flows.2/Concat_57_output_0
   gold: [−0.6966, −0.4081, −0.189, …]  rt: [0.001, 0.001, …]   max|rt−sim| = 2.411
== flows.2/Split_output_0
   gold: [−0.6966, −0.4081, −0.189, …]  rt: [0.001, 0.001, …]   max|rt−sim| = 1.331
== flows.2/Split_output_1          (pass-through)
   gold: [1.3372, 0.6998, …]        rt: [0.9492, −0.4121, …]    max|rt−sim| = 1.109
== flows.3/Split_output_1          (pass-through)
   rt == sim exactly                max|rt−sim| = 0.0
== flows.3/Concat_57 / Mul_81      (flows.3 OK)
   rt ≈ sim                         max|rt−sim| = 1.331 at one elt
== log_duration
   gold: [2.3453, 1.5657, 1.536, …] rt: [−0.0114 × 200]        max|rt−sim| = 2.45
```

The text-encoder part of the graph is NOT the problem: `input_padding_mask` is
exact, `prior_means`/`prior_log_variances` are close, and the big runtime `euc`
errors reported for `Pad`/attention `MatMul` in `error_analysis.txt` live in the
padded/garbage region that is sliced away before the flows.

## 3. Root-cause hypothesis

The **simulator** for the exact same `flows.2` chain is near-perfect
(`simulator_error` cos ≈ 1.0, euc ≈ 0.07), so the RKNN graph **as emulated** is
correct. Only the real NPU collapses, and it collapses specifically inside the
softplus/spline-mask computation of `flows.2` (identical op structure to
`flows.3`/`flows.4`, which are fine — so this is data/value-dependent).

Suspects, in order of likelihood:
1. **Softplus (or its decomposed `Log/Exp/Max/Where` pattern)** implemented by the
   NPU in a way that mishandles this data range → everything after it becomes `ln2`/`0.001`.
2. The **GatherElements / spline basis index** path feeding the variance term.
3. The **Where/Equal/ConstantOfShape "exp_clamp" mask** (previous lead) — `Where` is
   currently forced off-NPU in `convert.py`'s `op_target` **without evidence**.
4. Constant folding/elementwise fusion of that subexpression at build time.

The shortest distinguishing experiment: force softplus (and, separately, each
suspect node) to CPU via `op_target` (node names, per official docs), rebuild with
`do_quantization=False`, re-run `accuracy_analysis`, and see whether
`log_duration` becomes non-constant.

## 4. What was already checked / ruled out

- `VertexInsertPadding` (transformer) path: ruled out by the trace — encoder
  outputs are fine.
- Quantization (`do_quantization=True`, i16): `log_duration` was already wrong with
  `do_quantization=False`. Not the cause.
- `optimization_level=1`: same wrong constant output (build log compared). Not the cause.
- Model-side: the ONNX export (`export_onnx.py`) runs correctly in PyTorch and
  matches the reference `mms_tts.py` output. Not a float/export bug.
- Simulator vs NPU: simulator is accurate for the same model — therefore a
  compile/NPU-backend issue, not a model issue.

## 5. Environment & artifacts (all numbers verified against real runs)

- Orange Pi (ARM64, an RK3588 itself) runs the full `rknn-toolkit2==2.3.2` in
  venv `.rknn_venv`:
  `/home/ubuntu/rknn_model_zoo/examples/mms_tts/python/.rknn_venv`
  (includes `accuracy_analysis`; the other venv `rknn_lite_venv` only has
  RKNNLite, no build/analysis — wrong one).
- ONNX model: `/home/ubuntu/rknn_model_zoo/examples/mms_tts/model/mms_tts_eng_encoder_200.onnx`
- Inputs (both int64 `(1,200)`): `/tmp/input_ids.npy`, `/tmp/attention_mask.npy`,
  generated with `mms_tts.preprocess_input("Mister quilter is the apostle …")`.
- `RKNN.config` requires `target_platform="rk3588"` (a plain `target=` key fails).
- Analysis run: `/tmp/accuracy_analysis.py` on the Orange Pi wrote
  `/tmp/acc_analysis_results/` `{golden, simulator, runtime}` + `error_analysis.txt` + `map_name_to_file.txt`.
- Build flags that mattered: `do_quantization=False`, `optimization_level=3`,
  `target_platform="rk3588"`.
- Useful debugging scripts (on the Orange Pi `/tmp/`): `accuracy_analysis.py`,
  `parse_errors.py`, `trace_const.py`, `scan_flow.py`, `check_concat.py`,
  `dump_mul81.py`, `trace_flows.py`; reproducible subset stored locally as
  `illustrate_findings.py` (below).

## 6. Next step

Pick the single most likely suspect and prove it:

1. Run `illustrate_findings.py` (local, uses `acc_data/`) → prints `history.txt`
   showing golden vs simulator vs runtime for the collapsed chain — quick reference.
2. On the Orange Pi, build twice with `op_target` pinning only the softplus chain
   of `flows.2` to CPU, then FE `Softplus` to CPU separately
   (node-name keys), each time `do_quantization=False` + `accuracy_analysis`.
3. The version that makes `runtime/log_duration-rs.txt` non-constant and
   `max|rt−gold| < ~0.1` identifies the culprit node → commit that `op_target`.

## Data reference for the illustrative script

- `acc_data/golden/`, `acc_data/simulator/`, `acc_data/runtime/` — per-tensor
  text dumps (one float per line) for the 18 tensors listed in
  `illustrate_findings.py` (all `flows.2`/`flows.3`/`flows.0` collapse-relevant
  tensors + `log_duration` + `input_padding_mask` + priors).
- `acc_data/error_analysis.txt` — 992-row full per-layer error report.
- `acc_data/map_name_to_file.txt` — RKNN tensor name → dump filename.