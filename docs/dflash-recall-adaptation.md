# Train recall before adding capacity (#547)

Depends on #499/#537. Three opt-in recipe scopes retain the native checkpoint
architecture: `trainable: context` adapts fc/hidden_norm; `late_backbone`
adapts the final `recall_last_layers` draft layers plus final norm; `recall`
combines those scopes. All freeze both selector codebooks and selector gate.
Existing `selector` and `all` controls remain unchanged. Start with
`selector_loss_weight: 0` to isolate unary recall training; optionally compare
the combined existing CE with equal data/steps. `recall_last_layers: 1` is the
default; values exceeding the actual model depth fail before optimization.

Use the existing train CLI, validation-only checkpoint selection, and BF16
export. Trainable masters remain FP32. No convolution taps, candidate support,
rank, attention geometry or model depth are changed. These options adapt
released native-compatible weights, rather than claiming an untrained larger
architecture would help. The deployed target/head matching factor in #543
remains separate.

Evaluation now logs reached, covered and correctly selected counts per proposal
position, with recall and conditional selection rates using those denominators.
A verifier suffix after the first mismatch never enters a reached denominator.
Ordinary all-position top-k coverage remains available for comparison.

Tests verify exact scope membership and frozen weights, reject excessive depth,
check conditional counts by hand, and reduce unary loss with real CPU/MPS
optimization for all three scopes. The inherited full training tests exercise
resume, isolated block masking and BF16 exports. Tiny loss reduction is a
mechanics check. Multi-conversation/domain native rollouts, real exported
checkpoint acceptance and latency are required before selecting any scope or
considering wider convolution, per-layer context fusion or larger capacity.

Local verification (Apple M5 Pro, macOS 26.6, Python 3.12.14, PyTorch 2.13.0,
Transformers 5.15.1): `54` training tests passed, including the pinned upstream
reference comparison; none skipped. Actual MPS was required and CPU fallback disabled:

```sh
DFLASH_REFERENCE=/path/to/pinned/dflash/model.py \
DFLASH_REQUIRE_MPS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  python -m unittest discover -s dev/tests/training -t . -v
```
