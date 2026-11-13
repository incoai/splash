# GPU task scheduling and static DAG fusion experiment

This benchmark follows the ready-task scheduling direction in
[Muse Spark Open Ultrafast](https://www.lmsys.org/blog/2026-11-05-muse-spark-open-ultrafast).
It is research for issue #760 / draft PR #762. It does not implement a model
executor or a complete GPU virtual machine.

## Workload and alternatives

The fixed DAG has 32 rows and eight stages: affine, two-parent SiLU product,
RMS, affine, SiLU, RMS, affine, SiLU. The initial columns are 256, 2048 and
5120; worker counts are 1, 4, 8 and 20. Static dispatches always use 32 groups
and are recorded once per column in the 20-worker fixture.

- The original static baseline launches each of eight stages separately.
- The original persistent scan and fine ready queue execute 256 tasks.
- A coordination-only queue removes all arithmetic while retaining task
  discovery, dependencies, completion and fences. It checks completion only,
  not tensor output. It is a diagnostic, not an equivalent-work optimization
  or an additive decomposition of runtime; removing arithmetic changes contention.
- Coarsening groups affine/SiLU/RMS into three macro stages, yielding 96 tasks.
  Recomputing a neighboring affine value removes an internal dependency.
  The same operation can run as a ready queue or three static dispatches.
- A halo-recomputation kernel gives each row its own group and recomputes the
  neighboring rows needed by later stages, using three private scratch rows.
  It removes cross-group coordination at the cost of redundant arithmetic.
- Ordinary-buffer static controls avoid device coherence inside a dispatch;
  command boundaries synchronize the static graph. Both eight- and three-stage
  versions are included so this is not counted as a fusion gain by itself.

See [Metal resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization).
The queue still needs device-coherent data and synchronized publication inside
its persistent dispatch. Its 4096-attempt bound and completion/error counters
reject incomplete scheduling; these tests do not prove forward progress on
all GPU architectures.

Naive macro fusion initially changed affine contraction by one FP32 ULP.
Explicit `fma(column, bias, scaled_input)` preserves the original GPU stage's
contraction. No tolerance was relaxed. Every equivalent-work variant must
match every intermediate bit-for-bit against an immutable GPU reference that
also passes an independent CPU oracle (`2e-5 + 2e-5 * abs(reference)`). Buffer,
scratch, state and completion guards are checked.

## Validation and timing

Full mapped CPU tensor comparisons between timed runs changed shared-buffer
cache ownership, creating a repeatable first/second-run timing gap for large
vectors. Default timing now performs full exact checks before warmup and after
the timed sequence. Dynamic completion and small guards are checked on every
run. Separate shader validation repeats each arm eight times with full exact
checks after every execution. `--readback-each-run` reproduces the observer
effect; those runs are diagnostic and kept separate from GPU-only measurements.

There are four warmup ABBA blocks and eight recorded blocks by default.
`--blocks 16` is used for confirmation. The unchanged repository ABBA rule
saves aggregate and per-block verdicts; spread above 5% is inconclusive and
pass means non-regression. CPU tensor validation is outside the measured region.
GPU clocks are not fixed: interpret paired comparisons within a run, rather
than absolute microseconds across processes or warmup contexts.

Metrics are separate:

- GPU time excludes CPU queue initialization.
- Host wall time includes queue initialization, command preparation/submission
  and waiting for completion; it excludes tensor validation and one-time graph
  construction. It is standalone DAG latency, not model serving latency.
- Host submission time includes queue initialization and the submission call,
  before waiting. It can overlap GPU work and must not be added to GPU time.

No model TTFT, tokens/sec or energy improvement is inferred from these metrics.
No power measurements or production model integration are included.

## Reproduce

From the repository root on a supported Mac with Xcode's Metal tools:

```sh
python3 dev/benchmarks/experiments/gpu_scheduler/run.py --output build/experiments/scheduler/full
python3 dev/benchmarks/experiments/gpu_scheduler/run.py --shader-validation --output build/experiments/scheduler/shader
python3 dev/benchmarks/experiments/gpu_scheduler/run.py --shader-validation --holdout --seed 20261113 --output build/experiments/scheduler/heldout-shader
python3 dev/benchmarks/experiments/gpu_scheduler/run.py --holdout --seed 20261113 --blocks 16 --output build/experiments/scheduler/heldout-full
python3 dev/benchmarks/experiments/gpu_scheduler/run.py --confirm --blocks 16 --output build/experiments/scheduler/confirm
python3 dev/benchmarks/experiments/gpu_scheduler/run.py --confirm --blocks 16 --readback-each-run --output build/experiments/scheduler/readback-control
python3 dev/benchmarks/experiments/gpu_scheduler/run.py --confirm --holdout --seed 20261113 --blocks 16 --output build/experiments/scheduler/heldout-confirm
```

`--holdout` uses previously unmeasured columns 512, 1024 and 4096. The seed
changes the input values deterministically. `--confirm` tests only the fixed
three-dispatch ordinary-buffer implementation against the eight-dispatch
ordinary-buffer control; the same implementation is used for every column.
There is no per-shape selector. Initial and held-out full sweeps each contain
81 comparisons: 69 equivalent-work cases and 12 coordination-only diagnostics.

Raw timings, stderr, all block verdicts, input parameters and source/binary
hashes remain under ignored `build/`. Slower and inconclusive methods remain
available for reproduction. Production scheduling defaults are unchanged.
