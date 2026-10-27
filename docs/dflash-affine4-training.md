# Experimental AF4G64 projection training (#543)

Depends on training #499 and MPS #537. Add `affine4_scope: selector_context`
to an existing training recipe to fake-quantize `fc` and the selector gate;
`all` also covers every backbone Linear projection. `off` is the unchanged
control. Run with the existing `python -m draft_training train` command.
The recipe and checkpoint identities already bind resume and final selection.

The reconstruction follows the actual native loader: BF16 source weights,
64-column groups, signed affine scale, halves-away code rounding, and **BF16**
scale/bias metadata. The original issue's FP16 wording was inaccurate.
The codebook, normalization and base convolution tensors retain their existing
precision. FP32 masters, optimizer identity and BF16 export names are preserved.
A straight-through gradient is an experimental surrogate. This reference uses
FP32 matmul and BF16 outputs; native Metal accumulation is a separate check.

Tests compare packed bytes against the repository's independent native weight
preparation oracle, cover constant/negative/tiny/tie groups, optimizer updates,
state names, and actual loss reduction on CPU and available MPS hardware.

This implements the draft-projection factor of #543. The teacher remains the
BF16 teacher required by #499. It does not supply deployed-target features or
a low-bit teacher/head adapter. Do not call it quantized-teacher adaptation.
Compare off/selector_context/all at equal training tokens, then prepare the
selected BF16 export with the real loader and run held-out native rollouts.
No real checkpoint acceptance or throughput improvement has been measured.

Local verification (Apple M5 Pro, macOS 26.6, Python 3.12.14, PyTorch 2.13.0,
Transformers 5.15.1): `56` training tests passed, including the pinned upstream
reference comparison; none skipped. Actual MPS was required and CPU fallback disabled:

```sh
DFLASH_REFERENCE=/path/to/pinned/dflash/model.py \
DFLASH_REQUIRE_MPS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  python -m unittest discover -s dev/tests/training -t . -v
```
