# Draft fusion and parallel preparation experiments

These component and native Engine probes follow the drafting-fusion direction in
[Muse Spark Open Ultrafast](https://www.lmsys.org/blog/2026-11-05-muse-spark-open-ultrafast).
They are experiments in draft PR #761, tracked by issue #759.

The production baseline runs prepare, split, grouped reduce and reorder in four
dispatches. A stronger three-dispatch control writes reduction directly into
row-major output, the separate improvement in PR #756. New methods compare with
that control; it is not counted as a new gain here.

| Method | Dispatches | Preparation tasks/group | Attention split groups at B1 |
| --- | ---: | ---: | ---: |
| Production | 4 | 1, 256 threads | 32 |
| Parallel row-major control | 3 | 1, 256 threads | 32 |
| Original persistent split/reduce | 2 | 1, 256 threads | 8, four splits serially |
| Original full fusion | 1 | 40 serial tasks per KV head | 8, four splits serially |
| Serial preparation ablation | 3 | 40 serial tasks per KV head | 32 |
| Serial split ablation | 3 | 1, 256 threads | 8, separate reduction |
| Last-arriving split reduction | 2 | 1, 256 threads | 32 |
| Compact preparation | 3 | 1, 128 threads | 32 |
| Packed preparation | 3 | 2, 4 or 8 parallel tasks; 128 threads/task | 32 |

The split ablation keeps the control's dispatch count and reduction while
changing split parallelism. The preparation ablation keeps parallel attention
while changing preparation ordering. This separates the two serialization costs.
No hardware occupancy counter is collected; group counts are source geometry.

The compiled head dimension is 128, leaving four of the production preparation
kernel's eight SIMD groups contributing zero to its RMS sum. Compact preparation
uses the same four nonzero SIMD partials in the same order. Packing gives each
128-thread subteam one head, reducing scheduling overhead without serializing
heads. The selected confirmation method uses four tasks per group for every case.

The last-arriving split publishes device-coherent partials, increments a completion
counter and lets the last group reduce. No group spins on another group. The
counter resets before the next dispatch. See memory coherence and fences in
[Apple's Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
sections 4.8 and 6.16.3. MPP rejects coherent tensor stores, so a copy of the
production split helper publishes already computed fragments by their documented
multidimensional indices. Arithmetic and split order are preserved. This method
and the old slower variants remain available for reproduction, not selected
as production defaults.

## Component reproduction

On a supported Mac with Xcode's Metal tools, from the repository root:

```sh
python3 dev/benchmarks/experiments/draft_fusion/run.py --output build/experiments/fusion/full
python3 dev/benchmarks/experiments/draft_fusion/run.py --shader-validation --output build/experiments/fusion/shader
python3 dev/benchmarks/experiments/draft_fusion/run.py --shader-validation --holdout --seed 20261113 --nonuniform-weights --output build/experiments/fusion/holdout-shader
python3 dev/benchmarks/experiments/draft_fusion/run.py --confirm --holdout --seed 20261113 --nonuniform-weights --output build/experiments/fusion/confirm-holdout
```

The initial matrix is B1–B4 × histories 0, 127, 2047, 2048, 6000, seed 817,
unit normalization weights. `--screen` limits it to six initial diagnostic cases.
`--holdout` uses histories 1, 128, 129, 511, 2055, 8192; `--nonuniform-weights`
uses deterministic non-unit weights. `--confirm` tests only the fixed four-task
packing chosen on the initial matrix. No per-case choice occurs in confirmation.

Every case checks exact BF16 output and prepared Q/K/V, input preservation,
output/KV guards and reset completion counters. Commands repeat 64 times, and
GPU duration is divided by 64. Timing uses 8 warmup and 16 recorded ABBA blocks.
The unchanged `dev/benchmarks/abba.py` rule saves aggregate and individual block
verdicts; pass means non-regression, not a demonstrated gain. Shader validation
runs correctness only and produces no timing comparisons. Raw JSON, stderr,
parameters, hashes and summaries stay under ignored `build/`.

## Native model reproduction

With the normal installed Qwen3.8-27B 4-bit target and compatible draft:

```sh
python3 dev/benchmarks/experiments/draft_fusion/model_probe.py --model /path/to/Qwen3.8-27B-4bit --shader-validation --output build/experiments/fusion/model-shader
python3 dev/benchmarks/experiments/draft_fusion/model_probe.py --model /path/to/Qwen3.8-27B-4bit --output build/experiments/fusion/model-normal
```

The six frozen prompts in `model-cohort.txt` use target revision
`10c35caafbb80f7dc6a7a432cdd11af10a6d4818`. Each request is cold, greedy and
capped at 96 output tokens. The script generates an instrumented operator and
benchmark in `build/`, linking the operator before the production library.
One Engine/backend/model remains loaded, with all draft rows verified, default
int8 activations and ANE sharing disabled. Both arms use production four-dispatch
attention; only the preparation kernel changes. Counters require every candidate
preparation call to use packing and every baseline call to use production.
This is not an additive measurement with PR #756 or an HTTP serving benchmark.

Timing uses 40 samples per prompt: 8 excluded from timing only, then 8 balanced
alternating ABBA/BAAB blocks. Shader validation uses four samples per prompt
without a performance summary. Every sample counts toward full output-array
and cache metadata parity. The summary includes TTFT, whole-request tokens/sec
(including prefill), decode GPU duration, all metric blocks and source/model/
binary identities. Component GPU-time savings alone do not establish a model
TTFT, tokens/sec or energy improvement. No power measurements are collected.
