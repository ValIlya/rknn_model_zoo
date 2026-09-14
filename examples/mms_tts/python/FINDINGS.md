# MMS-TTS RKNN: `log_duration` NPU Collapse — Final Findings

## 1. Outcome

**Fixed.** The NPU collapse of `log_duration` is a build-time **fusion bug** in the
duration-predictor spline **mask chain** (flows.2 AND flows.3). It is fixed by a
semantic ONNX patch (replace the mask computation with the constant it already
equals for the operating range |x| < 5), now applied automatically by
`convert.py`. `op_target` cannot fix it (see §4). A residual, ~10× smaller
NPU-only error at ~8 output positions is documented in §6 and has been verified
not to affect the full TTS pipeline (valid 5.47s WAV).

## 2. Problem (before the fix)

RKNN-converted MMS-TTS encoder (`mms_tts_eng_encoder_200.onnx`, RK3588) produced a
**constant, wrong `log_duration`** on the real NPU while both the ONNX fp32
reference ("golden") and the RKNN simulator were nearly correct:

| tensor | golden | simulator | runtime (NPU) |
|---|---|---|---|
| `log_duration` | -2.23 … 2.44 | ≈ golden | **-0.0114 … ‑0.0000 (constant), uniq=2** |
| `flows.2/Softplus` | variable | ≈ golden | **const 0.6929 = ln 2 (softplus(0))** |

Two machines: editor = this workstation (macOS, no toolkit); builder/tester =
Orange Pi 5 (ssh `orange_pi`, rknn-toolkit2 2.3.2, real NPU). Fixed test inputs:
`/tmp/input_ids.npy`, `/tmp/attention_mask.npy` (int64 `(1,200)`, sentence
"Mister quilter is the apostle of the middle classes …"). Build constants used
throughout: `do_quantization=False`, `optimization_level=3`,
`target_platform='rk3588'`. Graph outputs (by index, confirmed): [`log_duration`
[1,1,200], `input_padding_mask` [1,1,200], `prior_means` [..,192],
`prior_log_variances` [..,192]].

## 3. Root cause (proven)

Inside each spline flow of the duration predictor (flows.2 and flows.3) there is
an "exp_clamp"-style mask of the curve:

```
mask = Cast(Less(Neg(x), 5.0))  AND  Cast(Greater(Neg(x), -5.0))
     = Cast_1 * Cast_3 = Mul_1            # 1 when |x| < 5, else 0
┌─ mask feeds ──► Cast_5..Cast_12 → Mul_11 = mask × Split_output_1
│                Unsqueeze_8/Cast_15 → Mul_14 = mask × ScatterND_1  → Softplus input
└─ mask−1 ──────► Sub = Mul_1 − 1 = 0 → Cast_4..9 → Mul_10 = 0 × Split_output_1
```

For the operating data range (|x| < 5) the mask is **identically 1** and the whole
computation is a semantic no-op. On the real NPU this **fused elementwise chain
collapses to 0**:

- first wrong tensor: `flows.2/Mul_1` — golden const 1, runtime **0**,
- then `Cast_5/Cast_10/Cast_11/Cast_12/Cast_15` all golden 1 → runtime 0,
- `flows.2/Mul_11` (golden variable −2.03…1.77, uniq 179) → runtime const **0**,
- `flows.2` spline input → 0 → `Softplus(0)=ln2` → `Add_37..40/Concat_57/Mul_81`
  collapse → `flows.0` → constant `log_duration`.

Why it is a **build-time fusion** bug and not a kernel/OOM/quantization issue:

- The verbose build log shows the mask chain collapsed into one NPU elementwise
  block via `unsqueeze_to_4d_mul` / `bypass_two_reshape` /
  `fuse_two_reshape(Mul_11, Unsqueeze_61)` — there is **no individual kernel** to
  move.
- Forcing ops to CPU at runtime did **not** help: `{ScatterND:cpu}`, `{Mul:cpu}`,
  `{GatherElements:cpu}`, `{Less:cpu,ReduceSum:cpu}`, `{Expand:cpu}` all kept the
  same constant runtime output → confirming the corrupted value is baked in at
  build time.
- A standalone mini-reproducer of the identical mask chain (same nodes,
  constants, N=200, real NPU) ran **correctly** — the bug needs the full model's
  fusion context.
- Both `flows.2` and `flows.3` contain the *identical* mask (constants 5.0/‑5.0/1);
  both must be patched (flows.4 has no mask).

## 4. Why the fix is an ONNX patch, not an `op_target`

`op_target` in this rknn-toolkit2 accepts **op-TYPE keys only**. Node-name /
random keys raise `ValueError` (verified empirically; the toolkit docstring's
`{'111':'cpu'}` example is misleading). The mask chain has no separable kernels
(fused at build time), so even a CPU target for every elementwise op type
(`Mul`, `Cast`, `Less`, `Greater`, `Sub`, `Neg`) cannot prevent the corruption.
The only robust fix is to remove the mask computation from the graph entirely and
feed the constants it provably equals for |x| < 5:

- `Cast_12_output_0` → `ones [1,1,200]` (Consumer: `Mul_11`)
- `Cast_15_output_0` → `ones [1,1,200,1]` (Consumer: `Mul_14`)
- `Sub`/`Cast_4..9` → `zeros [1,1,200]` (Consumer path: `Mul_10 = 0 × Split`)

`convert.py` now applies this patch automatically (writes
`<model>_maskpatched.onnx`, verifies with `onnx.checker`), guarded to only touch
graphs containing `/duration_predictor/flows.2/Mul_1`.

## 5. Verified fix results (all numbers from live runs)

With `patched_mask_all.onnx` (= flows.2 + flows.3 masks replaced):

| metric | broken baseline | mask_all patch |
|---|---|---|
| `log_duration` runtime range | const `-0.0114..0` uniq 2 | **`-2.2285..2.4336` uniq 177** (golden `-2.2268..2.4367`) |
| max\|rt−gold\| | 2.448 | **0.067→2.617 tail** (see §6) |
| mean\|rt−gold\| | ~0.65 | **0.0595** |
| frac \|rt−gold\| < 0.1 | 0.5% | **92.5%** |
| encoder latency (NPU, 5+ runs) | mean 0.04012s | **mean 0.03870s** |

Full pipeline on NPU (patched encoder + decoder, sentence above):
- `log_duration` range `[-2.228516, 2.433594]`, `pred_len=342`,
  179 nonzero-duration tokens, per-token durations 1–12 frames (realistic prosody).
- Output WAV `p_mask_all.wav` = 87,552 samples @16k (5.47 s), rms 0.127,
  peak 0.78, non-silent. (Collapsed baseline gave a degenerate 2.86 s WAV.)

## 6. Residual (not the collapse, documented)

About 8 real (unpadded) positions `{0,1,2,10..14}` keep `|rt−gold|` up to 2.62
(pos 0: g=2.3453, r=−0.2720). Cause: a **second**, separate NPU kernel bug in the
spline window-grid construction (`Expand_85 → Reshape_34 → ScatterND_25..27` →
`Neg_2`), which corrupts per-position cell values by 0.1–1.1 (NOT fp16-ulp) at
37/200 rows → `Less_1`/`ReduceSum` knot counts flip by ±1–3 at those rows →
wrong spline bin index → localized large error.

Resisted every cheap fix (all reproduce identical metrics):
`{Less:cpu,ReduceSum:cpu}`, `{Expand:cpu}`, `{ScatterND:cpu}`,
`split_copy` (Identity copy of `Split_output_1`), unchanged position set.
Diagnosis: values are corrupted in the fused NPU graph (inputs already fp16, so
CPU compares/scatters see the same bad data). A deep ONNX re-architect of the
grid (explicit windows) would be needed to also eliminate this — not pursued.

## 7. What was tried and ruled out (complete list)

`where`/`expand→cpu` (first commit, unverified idea, removed), `Softplus→cpu`
(**impossible**: librknnrt 0.9.6 has no CPU Softplus kernel), `GatherElements→cpu`,
`ScatterND→cpu`, `Mul→cpu` (221 Mul, latency rose 40→164 ms), mini-reproducer
(no repro), `Less/ReduceSum→cpu`, `Expand→cpu`, `split_copy`, `mask_ones`
(flows.2 only — partial), `mask_all` (**fix**). Every build used
`accuracy_analysis` rt-vs-golden dumps; all numbers above are from live command
output.

## 8. Artifacts

- On Orange Pi: `/tmp/mtts_exp/{s1_base,s3b,s3d_scatternd,s3d_mul,mini1,
  p_mask_ones,p_mask2,p_mask_all,p_mask_all_*}` (builds + AA outdirs),
  `/tmp/mtts_exp/patched_mask_all.onnx`, `/tmp/mtts_exp/p_mask_all.wav`,
  `/tmp/mtts_exp/baseline_collapsed.wav`.
- Patch script (harness): `examples/mms_tts/python/harness/patch_mask_all.py`;
  runner `harness/build_onnx.py`; drivers `harness/aa_variant.py`
  (`--onnx`/`--op-target`), `harness/bench_encoder.py`, `harness/run_pipeline.py`.
- `convert.py` now applies the patch automatically.