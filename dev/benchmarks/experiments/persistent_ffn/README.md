# Persistent weighted feed-forward experiment

This extends the task-scheduling investigation in issue #760 / draft #762 toward
weighted model operations, following the layer-fusion and persistent-execution
ideas in [Muse Spark Open Ultrafast](https://www.lmsys.org/blog/2026-11-05-muse-spark-open-ultrafast).
It is a complete feed-forward block, not an attention layer or full model VM.
Production operators and selection are unchanged.

The block normalizes BF16 rows, computes AF4G64 gate/up projections and SwiGLU,
then a split-K down projection and residual output. Its matrix sizes default to
the dense 27B's 5120 hidden and 17408 intermediate values. Four modes run:

0. The production norm and staged projection entry points, three dispatches.
1. Four static dispatches of the prototype's exact helpers.
2. One persistent dispatch with fixed strided jobs and phase barriers.
3. One persistent dispatch with an atomic task queue and phase barriers.

A SIMD group owns a matrix tile. The single-SIMD norm virtualizes the native
256-thread reduction in its original summation order. The native gate/up tile
height and down split/height are observed from the production selection;
`native_selection_matched` records whether the explicit native staged baseline
matches that selection. Cases marked false are correctness fixtures, not
production-selected performance comparisons.

Cross-group intermediates use device coherence and device-scope fences. MPP
rejects coherent BF16 tensor operands directly. The default copies each input
slice into a threadgroup tile; `--register-inputs` instead fills MPP cooperative
input fragments from coherent scalar loads. Both preserve every tested BF16
rounding boundary and split summation order.

A phase barrier has a bounded poll and an error flag. A failed or unfinished
barrier fails the run. This detects observed progress failures; it does not
prove all worker groups can reside simultaneously on other GPUs. The worker
count is experimental and is not a production tuning rule.

Timing streams a ring of physically distinct weight copies covering at least
384 MiB. Four warm-up ABBA blocks precede recorded blocks. Full tensor checks
occur before warmup and after timing; guard and completion checks remain.
Shader validation repeats all modes with full comparisons and no recorded
timing. Normalized values, SwiGLU output, split partials (when split), final
output, input preservation, counters and buffer guards are checked bitwise
against the native kernels. The existing ABBA rule is unchanged; pass means
non-regression, not proof of improvement. All raw samples and verdicts remain
under ignored build paths. There is no native-model TTFT, tokens/sec or energy
claim.

```sh
python3 dev/benchmarks/experiments/persistent_ffn/run.py --shader-validation --output build/experiments/persistent-ffn/shader
python3 dev/benchmarks/experiments/persistent_ffn/run.py --output build/experiments/persistent-ffn/timing
python3 dev/benchmarks/experiments/persistent_ffn/run.py --workers 20 --rows 8 --output build/experiments/persistent-ffn/pool-ablation
python3 dev/benchmarks/experiments/persistent_ffn/run.py --register-inputs --shader-validation --rows 8 --output build/experiments/persistent-ffn/register-shader
python3 dev/benchmarks/experiments/persistent_ffn/run.py --register-inputs --rows 8 --output build/experiments/persistent-ffn/register-timing
```
