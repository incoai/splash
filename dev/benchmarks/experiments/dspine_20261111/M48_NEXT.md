# M48 implementation and qualification plan

The research tile computes 48 live rows for three sixteen-row lanes. It keeps
the existing 64-row allocation capacity, split scratch and traversal. The
primary operator screen improves from 0.8120 to 0.6025 ms, or +34.78% throughput.
All four registered split/spread configurations and the failed setup attempts
remain in the evidence. Static threadgroup storage stays 8,208 bytes.
Allocated registers, spills and occupancy remain open measurements.

The source is committed separately as `5f873426be3d8bedaba0d3dac3db1e20b18805b9`.
The main report gives the operator and whole-model results for this separate
research snapshot.

## Current guard

The prototype requires all of the following conditions.

- The existing measured Apple10 / 20-GPU-core M5 Pro calibration enables the
  wide separate-pass gate/up policy.
- Decode geometry is sixteen rows, with 48 logical rows and 64 storage rows.
- The epilogue is gate/up with separate passes and BF16 output.
- The matrix has `N=17408, K=5120`.
- Each weight is one unrotated AF4G64 segment starting at column zero.

The diagnostic production graph confirms 69 M48 gate and 69 M48 up calls at
B3, and zero M48 calls at B1, B2 and B4. Nine complete 64-token cloned-lane
arrays match the frozen serving build.

## Make compute height explicit

The prototype chooses a local dispatch height in `Linear::addGgufStaged`.
Before a production proposal, represent compute height separately from storage
capacity in the linear plan. Validate that live rows fit compute rows, compute
rows fit allocated rows, and each supported tile has matching dispatch,
partial-reduction and counter geometry. Keep Table16 packing's physical
eight-row unit independent of both values.

Preserve K partitions, split counters, buffer strides, output masks and the
existing M64 fallback. Keep the first proposal scoped to the tested gate/up
case with 64-row allocation capacity.

## Add permanent selection and numerical checks

Extend the permanent GGUF projection tests with M48 rather than relying only on
the research harness. Cover separate gate/up epilogues, BF16 output, one/two K
splits, spread on/off, NaN padding, output guards, counter release and independent
FP64 bounds. Retain same-split bitwise comparison against the existing path.

Add a positive dispatch test for the measured B3 shape and negative cases for
eight-row execution, B1/B2/B4, other matrices, other formats, multiple/offset
segments, rotated weights, FP32 output and unmeasured devices. Assert actual
selected entry names as well as plan capacity.

## Require a new deployment comparison

The registered research comparison uses three complete ABBA groups against the
already corrected sixteen-row serving path. Its verdict retains the recorded
output, operating, noise and effect checks for every group.

A later production proposal needs a new frozen workload with heterogeneous
prompts, lengths, lane arrivals, output caps and prefix histories. Charge
prefill, queueing, checkpoint creation and later reuse in client wall time.
Include priority suspension, pending-state replay, disk restore and memory
pressure. Compare sixteen-row DSpine with DFlash2 and eight-row DSpine after
any M48 adoption.

If the corrected M48 graph qualifies, ordinary M64 projections are the next
shape family to inspect. Their residual/output rules, formats, segment layouts
and FP32 destinations need separate kernels and numerical and selection
coverage. Hardware counters would help distinguish register pressure, spills
and arithmetic limits.

At B4 all 64 rows are useful. Test two M32 row tiles with the existing 64-row
allocation. This could
reduce each group's accumulator working set while adding weight-access
requests. Measure whether cache reuse offsets that cost.
It needs independent row-offset, prepared-input, counter, partial-reduction
and output/residual-view tests before a full B4 comparison.

## Rebuild the research snapshot

The full research patch applies to baseline `425ad7400f3a00d2a1e6745bdacfa2881922d544`.
The smaller `m48-research-only-v1.patch` is the two-file delta on top of the
frozen geometry source. The build command uses the full patch.

```sh
python3 reproduce.py build --repo /path/to/splash --output /tmp/dspine-m48 \
  --variant research-m48 --python /path/to/python --metal
python3 reproduce.py run --build /tmp/dspine-m48 --model-root MODEL_ROOT \
  --rows 16 --width 3 --cohort cohorts/calibration-smoke3-v1.txt --shader \
  --output /tmp/dspine-m48-functional.json
```

These commands rebuild and run the functional smoke fixture.
