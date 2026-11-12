# Shared-prefix tree verification experiment

This controlled native Engine experiment follows the tree-verification direction
in [Muse Spark Open Ultrafast](https://www.lmsys.org/blog/2026-11-05-muse-spark-open-ultrafast).
It exercises the opt-in sixteen-node shared-prefix tree on the dense 27B target.

From this PR's repository root, with the installed Qwen3.8-27B 4-bit target and
its compatible eight-row draft available through the normal model installation:

```sh
python3 dev/benchmarks/experiments/tree_verify/run.py --model /path/to/Qwen3.8-27B-4bit --shader-validation --output build/experiments/tree/shader
python3 dev/benchmarks/experiments/tree_verify/run.py --model /path/to/Qwen3.8-27B-4bit --output build/experiments/tree/normal
```

The frozen `cohort.txt` contains twelve named prompts encoded as token IDs for
that target (tested public target revision
`10c35caafbb80f7dc6a7a432cdd11af10a6d4818`). Each cold greedy request is capped
at 96 output tokens. The helper generates instrumented copies of the production
runtime and backend benchmark under ignored `build/`, then links the runtime
ahead of the production library. Seam replacements must occur exactly once.
It uses one Engine/backend/model and the same tree-capable setup and cache
writer policy in both arms. Only a per-request tree-execution gate alternates.
This is a causal diagnostic, not a comparison with unmodified upstream, an
HTTP serving benchmark, or a cache-lifecycle qualification.

The benchmark forces verification of all draft rows, keeps default int8
activations and disables ANE sharing. Samples alternate ABBA then BAAB. A timing
run records 40 samples per prompt; the first eight are excluded from timing
only, leaving eight balanced blocks. Every sample counts toward full output
array and cache metadata parity. A shader-validation run uses four samples per
prompt and produces no performance summary. It requires positive observed
wide-cycle counts and zero wide cycles in the chain arm.

`summary.json` records TTFT, whole-request tokens/sec (including prefill),
decode GPU duration, every block's unchanged repository ABBA verdict, and
source/binary/model identity. A pass is a non-regression verdict, not evidence
of a gain. Raw output, stderr and per-request progress remain beside it.
The cohort was originally split into six screen and six structured holdout
prompts; rerunning it here is confirmation, not a new independent holdout.

The retained-token average starts at the maximum eight rows. With the existing
6.5 threshold and 0.125 update weight, the runtime first observes chain
acceptance before widening. High-acceptance streams can remain on the chain.
Tree verification remains default off and this submission remains experimental.
Sampled-distribution, long-context, active mixed-tree batches and broader
conversation/cache-lifecycle qualification are still required.
