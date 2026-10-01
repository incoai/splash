# Device policy and performance qualification

The convergence pass starts at `2a43027`. It preserves the measured default
execution paths while making offline Q4 calibration independent of fixed
threadgroup counts from individual GPUs. It adds no kernels, weight formats,
production allocations, startup benchmarks or per-model/per-SKU tables.

## Policy ownership

`runtime/ops/Linear.cpp` owns Q4 selection. Apple9 decode uses bfloat
simdgroup matrices with 1/2/4/8 K partitions
([apple9-simdgroup.md](apple9-simdgroup.md)). Apple10 and later use MPP
tiles, with shape and core count selecting grids. A projection with at most
two N128 tiles per core splits K instead: `LinearTile::Split128` runs the N128
tile over the largest power of two, up to eight, of K partitions whose grid
still fits four 256-thread threadgroups per core, each partition at least one
256-input block. The rule depends on the grid per core only, so every batch
width takes the same plan and a request's sums do not depend on the requests
it is batched with. Every other projection keeps the sequential tiles, paired
N256 and M24 included, bit for bit. The obsolete Apple9 one-lane MPP branches
have been removed; those kernels remain useful as qualification references
and offline candidates.

The split rule follows the policy bench's DRAM-cold kernel times (eight
alternating runs) on the two Apple10/11 GPUs we have, a 20-core M5 Pro and a
12-core M6. Core counts emulated by width misjudge split tiles by -26% to
+20% against a 40-core M5 Max, so a rule takes a gain only where a native
machine measures one.

- The one-lane Split32/Split64 tiles, withdrawn as defaults after
  speculative-acceptance reductions on some M5 prompts, remain offline
  candidates: no rule of the grid per core selects them on both machines. On
  the M5 Pro they beat today's plan on the 35B's mixer and draft outputs by
  4%, about their run-to-run spread, and on its draft gate/up by 9%; the same
  shapes run 15-17% slower in them on the M6. The M6's one-lane wins with them
  (4-8% on the 27B's 5120-wide projections) are smaller than those of the
  unpaired N128 or N256 tile on the same shapes (17-18%), and on the M5 Pro
  the unpaired N128 tile likewise beats both the paired plan and Split64 on
  the 35B's attention input: that headroom belongs to the sequential tiles'
  pairing rule, not to a split.

On the 27B and 35B projections neither machine runs a split plan slower than
its fastest sequential tile. The exceptions are off those shapes: on the
bench's fitting grid the M6 runs 3072 x 2048 and 3072 x 4096 in two splits
4-7% slower at one and two lanes, and with 20 cores emulated on the M6 the
27B draft output (5120 x 4096) and mixer output at one lane take 1.044x and
1.010x the paired N128 time (spread 1.9%), about 0.06 ms of a 58 ms one-lane
Q4 step there.

Core count comes from the Metal device's IORegistry property. Missing metadata
uses one 32-core estimate across families, an intermediate value in the
16–40-core range of our reference machines. This is not a calibrated optimum
or a performance guarantee for unidentified GPUs. A nonzero reported count
always overrides it. Family 11 (the M6) runs the family 10 policy; its policy
tests check extrapolation only, and actual validation here covers families 9
and 10, with the Apple10 split rule also timed on a 12-core M6. Core count
alone cannot describe memory bandwidth, cache capacity, power state or
compiler behavior.

MoE routing scales its row threshold with core count. The expert tile's
four-SIMD-group Apple9 decode default remains a family rule, measured on the
40-core M3 Max. Smaller Apple9 devices need an expert-kernel comparison before
claiming that rule is optimal. Prefill and Apple10 retain eight SIMD groups.

## Bounded offline calibration

Q4 candidates always start with the shipped baseline. Persistent grids now
include two, three and four threadgroups per reported core plus the full grid,
instead of fixed counts 36/60/80. One lane lists the Split32/Split64 tiles
when K % 1024 == 0. Apple9 additionally exposes every valid simdgroup split in
1/2/4/8, and Apple10 every Split128 split of two to eight its blocks allow at
every lane count. The maximum candidate count is 20, derived beside
`Linear::kMaximumCandidates`; deduplication handles small grids. Prefill
candidates are unchanged. New candidates do not automatically change serving.

The existing offline tuner qualifies numerical results, admits the maximum
candidate workspace, alternates baseline/candidate timing and requires both
GPU and wall-time evidence. Interrupted or inconclusive runs keep the default.
Its scratch admission already covered candidate maxima; the independent batch
reuse test was corrected to do the same after the extra K splits exposed its
baseline-only allocation assumption.

On a new device, run the tuning tool on an installed model:

```sh
make tune-kernels MODEL=mlx-community/Qwen3.8-27B-4bit \
  TUNE_ARGS='--seconds 30 --pairs 31 --candidates --confirm 12'
```

Record GPU family/core count, power mode, OS/toolchain and source identity.
Keep other GPU work idle. The tool prints measurements and confirms complete
prefill/decode graphs; it does not persist a serving profile. Operator timings
include standalone preparation and do not substitute for fused-producer or
cold full-model measurements. Promote a default change only after repeatable
whole-model A/B results, unchanged correctness/state-restoration behavior,
and acceptable speculative acceptance and memory use. Check both models,
short/long prompts and batch widths 1–4. Preserve the baseline when evidence
is mixed. For the family-only MoE rule, separately compare four/eight groups
on the missing hardware; the existing MoE tuner does not vary that rule.

## Convergence validation (2026-09-21)

- An independent before/after snapshot compares 1,591,200 default plans across
  families 9/10/11, every core count 1–128, unknown count, the IORegistry reader's
  upper bound of 4096, production-like shapes and dispatch boundaries. Configs,
  pipeline names and workspace sizes are byte-identical. These simulated core
  counts establish policy consistency, not measured performance on those GPUs.
- Permanent CPU tests cover 84,240 decode workload/device combinations: valid
  grids, bounded unique candidates, retained defaults and workspace admission
  after installing each candidate. `test-engine-cpu` now runs `linear-plan
  --cpu`, so these checks do not depend on a Metal test run.
- M3 Max 40, M5 Pro 16 and M5 Pro 20 pass full builds and CPU suites, Linear
  candidate numerical tests, tuning controls/batch reuse, and the 168-case
  independent fp64 simdgroup test with GPU shader validation. Both remote
  machines are on AC. All build the same production source identity:
  `src-a996ac63153606d2ab3534be64d2ca804dc396d44e06ab222cfc42922b2a3740`.
- On each machine, the production metallib is byte-identical to its pre-cleanup
  library. This pass establishes unchanged default GPU work and valid expanded
  calibration; it does not claim a new serving speedup or repeat the previous
  whole-model ABBA measurements.

The policy snapshot harness is not in the repository. Earlier serving results
remain in
[remaining-decode-optimizations.md](remaining-decode-optimizations.md) and
[apple9-simdgroup.md](apple9-simdgroup.md).

## Unknown-core fallback follow-up

The unknown-core path now uses a single 32-core estimate. The convergence
snapshot reported above precedes this fallback change; unknown-core plans
intentionally differ. Existing CPU policy tests cover unknown counts in both
prefill and decode, including equivalence to an explicitly reported 32-core
GPU. Known-core plans remain byte-identical in a separate before/after
comparison. This fallback does not require startup or user-run calibration.
