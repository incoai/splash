# Larger draft-attention fusion experiment

This benchmark explores the drafting-fusion direction in [Muse Spark Open
Ultrafast](https://www.lmsys.org/blog/2026-11-05-muse-spark-open-ultrafast).
It measures a component, not a full-layer megakernel or model inference.

From the repository root, on a supported Mac with Xcode's Metal tools:

```sh
python3 dev/benchmarks/experiments/draft_fusion/run.py --output build/experiments/fusion/normal
python3 dev/benchmarks/experiments/draft_fusion/run.py --shader-validation --output build/experiments/fusion/shader
```

The baseline is the current production prepare, split, reduce and reorder
sequence (four dispatches). The two candidates use separate output storage:
prepare plus persistent split/reduce (two dispatches), or per-head prepare
and attention/reduce (one). They use the production split arithmetic and
compiled four-split order; the experimental reduction writes row-major output.
The latter layout is also explored independently in PR #756.

Every case checks exact BF16 output, prepared Q/K/V, input preservation, and
output and KV guards. Cases cover B1–B4 and cache lengths 0, 127, 2047, 2048,
and 6000, including wrapped rings. Seed 817 is fixed. Each command repeats the
whole sequence 64 times; timings divide GPU command duration by 64. There are
8 warmup and 16 recorded ABBA blocks per variant. All blocks are saved and
summarized using `dev/benchmarks/abba.py`; a pass is a non-regression verdict,
not proof of a gain. Shader-validation timings are correctness diagnostics.

`raw.json`, `stderr.log` and `summary.json` stay in the requested build directory.
The summary includes source/binary hashes and individual block verdicts.
TTFT and model tokens/sec are unmeasured and recorded as null. This benchmark
is isolated from production dispatch selection and runtime defaults.
