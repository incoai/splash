# Dynamic GPU scheduler experiment

This bounded task-DAG probe explores the ready-task scheduling direction in
[Muse Spark Open Ultrafast](https://www.lmsys.org/blog/2026-11-05-muse-spark-open-ultrafast).
It does not implement a complete GPU virtual machine or model executor.

From the repository root, on a supported Mac with Xcode's Metal tools:

```sh
python3 dev/benchmarks/experiments/gpu_scheduler/run.py --output build/experiments/scheduler/normal
python3 dev/benchmarks/experiments/gpu_scheduler/run.py --shader-validation --output build/experiments/scheduler/shader
```

The baseline launches eight static stages, each with 32 independent tasks.
Persistent scan and ready-queue candidates execute the same 256-task DAG in
one dispatch with 1, 4, 8 or 20 worker threadgroups. Task columns are 256,
2048 or 5120. Stages alternate affine transforms, a two-parent SiLU product,
and RMS normalization. Every GPU intermediate must match the static execution
exactly and an independent CPU reference within `2e-5 + 2e-5 * abs(reference)`.
The 4096-attempt bound and completion/error counters turn a stalled scheduler
into a failed check. These checks cover observed behavior on the tested GPU;
they are not a proof of forward progress across all GPU architectures.

Each case uses 4 warmup and 8 recorded ABBA blocks. Queue initialization on
the CPU is excluded from GPU timing, favoring the candidates. `raw.json`,
`stderr.log` and `summary.json` stay in the requested build directory. The
summary saves source/binary hashes and each block's unchanged repository
ABBA verdict. A pass means non-regression under that rule. Shader-validation
timings are correctness diagnostics. Model TTFT and tokens/sec are unmeasured
and recorded as null. Production scheduling and runtime defaults are unchanged.
