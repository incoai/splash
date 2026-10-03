# Decode follow-up on Apple9 and Apple10

This follow-up starts at `1beb863`, after the Q4/MoE integration and Apple9
bfloat simdgroup decode implementation. It does not establish that all possible
optimizations are exhausted, or a ranking against other serving engines.

## Production changes

GDN and the attention gate emit the Q4 operand layout and row sums alongside
their ordinary rounded bfloat output when the following mixer projection reads
a prepared table (Apple9; one lane when measured here, one to four lanes since
`3a20983`). Both paths reproduce separate preparation bit for bit, and reuse
the existing arena allocation. The FFN
producer is intentionally unchanged; its experimental fusion did not justify
another live operand buffer and a second row-sum format.

Verify attention shares the maximum and exponential weights across all 256
output dimensions of a row. Two threadgroup barriers publish eight SIMD-group
maxima and at most 128 split weights. The numerator **and denominator** retain
the original ascending split order. An earlier parallel denominator reduction
passed numerical tests but changed speculative acceptance enough to reduce
throughput on some prompts; it was rejected. Prefill is unchanged.

Apple10 M24 plain and residual projections use the existing eight-SIMD-group
N128 kernel when there is at most one output tile per GPU core and K >= 4096.
Other M24 workloads keep four SIMD groups. The conservative boundary excludes
short dot products and grids that can benefit more from occupancy. This adds
no kernel, scratch buffer or weight format. Apple9 policy is unchanged. (Those
narrow grids now split K at every batch width, [device-policy.md](device-policy.md).)

`attention-sweep` can compare two libraries with alternating order, GPU warmup
and medians, and can select verify or prefill. Its figures describe one layer's
store/attention graph, not whole-model throughput.

## Validation

Validation on M3 Max (40 GPU cores), M5 Pro (16 GPU cores), and M5 Pro
(20 GPU cores) combines full builds, CPU tests and a full Metal integration
run, followed by focused tests of the final ordered merge and M24 policy. Kernel correctness tests enable GPU
shader validation; performance and native-model runs use ordinary production
execution. Final model oracles passed for both models on M3 and for 35B on
M5 16, including state restore, preemption continuation and batch transitions.

The merge test independently evaluates fp64 results for split counts
1/3/7/32/65/128, both head geometries, inactive rows, cancellation, fractional
weights, maxima differing by 2000, and untouched output padding. Producer
fusion tests compare ordinary output, prepared table and row sums bitwise.
For split counts through 32, the merge also matches the unchanged sequential
prefill reduction bitwise. Separate comparisons against the baseline library
cover all tested counts through 128 on all three devices.
The M24 tests cover both sides of the N and K dispatch boundaries and execute
all supported candidate kernels against numerical references.

Experimental kernels are not part of the production build.

## Measurements (2026-09-21)

Baseline: `1beb863`, the previously completed sgmatrix implementation.
Candidate production build ID:
`src-7f61e003bf7f314127ee7be781d7553da5e9a21297689bc648b3756bb227a9b4`.
All three machines built the same source identity. The M3 was connected to AC
with high-power mode; these absolute timings should not be combined with the
previous battery measurements.

The attention sweep alternates libraries, warms the GPU, and takes medians of
31 samples at 2K/8K/32K/nearly 128K history, both model head geometries and
one/three/four lanes (24 cases per machine). Speedups are baseline/candidate.

```sh
build/engine-tests/attention-sweep build/splash.metallib --phases verify \
  --compare-metallib BASELINE/splash.metallib --lanes 1,3,4 --repeat 31 \
  --histories LENGTHS
```

| Device | Merge kernel speedup range | Whole attention graph range | Graph median |
| --- | ---: | ---: | ---: |
| M3 Max 40 | 1.21–1.73x | 1.005–1.057x | 1.018x |
| M5 Pro 16 | 1.25–1.62x | 1.014–1.133x | 1.038x |
| M5 Pro 20 | 1.18–1.69x | 1.010–1.106x | 1.029x |

A separate M3 paired producer benchmark (101 alternating samples, 100 ms
warmup per variant, repeated twice) measured GDN plus preparation at roughly
1.11x/1.13x for 48/32 value heads. Attention gate plus preparation measured
1.10x–1.20x for the two geometries. These fixtures reuse hot state/activations;
they are not a cold full-model benchmark. Both fusions remove a dispatch and
reuse the existing operand buffers without another live allocation.

These are isolated attention measurements. Whole-model gains at the measured
short contexts are much smaller and near the noise floor. ABBA serving runs
use fresh servers, the same package/prompts, temperature zero, one warmup per
server, and engine token/time counters. M3: 256 tokens, four scenarios, two
samples per phase (four measured requests per build/scenario). M5 Pro 16:
192 tokens, two scenarios, one sample per phase. Every tested output text and
acceptance statistic matches the baseline after restoring split-order sums.

| Device/model | Scenario | Baseline tok/s | Candidate tok/s | Change |
| --- | --- | ---: | ---: | ---: |
| M3 / 27B | python_code | 93.5 | 93.5 | +0.01% |
| M3 / 27B | repeat | 144.9 | 145.2 | +0.21% |
| M3 / 27B | technical | 49.6 | 49.6 | +0.01% |
| M3 / 27B | typescript_code | 109.2 | 109.5 | +0.30% |
| M3 / 35B | python_code | 190.2 | 190.8 | +0.27% |
| M3 / 35B | repeat | 323.9 | 324.5 | +0.17% |
| M3 / 35B | technical | 113.8 | 114.0 | +0.25% |
| M3 / 35B | typescript_code | 212.2 | 212.8 | +0.29% |
| M5 16 / 27B | repeat | 99.4 | 100.2 | +0.73% |
| M5 16 / 27B | technical | 37.4 | 37.6 | +0.69% |
| M5 16 / 35B | repeat | 335.3 | 337.4 | +0.61% |
| M5 16 / 35B | technical | 106.2 | 106.0 | -0.16% |

The existing-kernel M24 comparison rotates weight copies, warms each choice
for 100 ms of GPU time, alternates 31 paired samples and batches 16 dispatches
per sample. The measured 35B draft context projection improves 1.064x on M5
20 and 1.077x on M5 16; draft down improves 1.047x and 1.064x. Small K and wider
grids can regress, hence the conservative dispatch boundary. A desktop timing
outlier for draft out was rechecked twice (1.045x each); it is reported here
rather than discarded silently.

A real 35B decode-profile ABBA on M5 16, with 2048 prompt tokens and nine
cycles per width/phase (`make benchmark-decode-profile MODEL=...
DECODE_PROFILE_ARGS='--prompt-tokens 2048 --cycles 9'` per build and phase),
measured the following medians of phase medians:

| Batch width | Baseline cycle ms | Candidate cycle ms | Change |
| --- | ---: | ---: | ---: |
| 1 | 22.540 | 22.405 | +0.60% |
| 2 | 35.300 | 35.130 | +0.48% |
| 3 | 45.005 | 44.890 | +0.26% |
| 4 | 50.105 | 49.840 | +0.53% |

Host work between decode commands (graph build, submission, completion,
scheduling) does not show in GPU timings or in `metrics.decode_wall_ms`, which
starts at submission. Judge changes to it by `metrics.decode_cycle_ms` of
`/status`, which the HTTP ABBA decode verdict uses, or by decode-profile's
wall minus fused GPU time.

The M5 20-core machine has kernel/policy validation and microbenchmarks here;
this table does not imply an end-to-end serving run on that machine. No new
comparison against MLX Serve or Ollama was performed.

## Experiments not selected

- Register-resident simdgroup attention: correct in the exercised cases, but
  substantially slower on M3 and M5. Fully unrolling its fragment loop did not
  recover the loss. This does not disprove the original Q-traffic diagnosis:
  the prototype also changes matrix execution, occupancy and softmax work.
- Reusing a cooperative Q tensor in the existing eight-group MPP kernel:
  rejected by the compiler; cooperative input tensors require a one-group
  execution scope. A different algorithm would be required.
- Two KV pages per attention iteration: approximately neutral on M3 and slower
  on relevant M5 long-context cases. It increases threadgroup storage.
- Four SIMD groups per original attention tile, and halving its query rows:
  no consistent cross-shape/device improvement. The latter also doubles tiles.
- Four independent numerator chains in the merge: no useful gain over the
  simpler sequential chain.
- B3 WIP two-fragment kernel: passed its qualification but slower than existing
  kernels. A separate padded-32 variant produced wrong results and was rejected.
  Only the measured existing-kernel policy change is retained.
- FFN gate/up preparation fusion: on M3, a paired cold-weight benchmark measured
  about 1.035x on the small draft FFN and 1.003x on the dense target FFN. The
  dense result is noise-scale and does not justify the additional buffer and
  consumer format.

Long-context attention remains a substantial optimization opportunity. The
experiments above rule out these implementations, not every possible redesign.

The subsequent [device-policy convergence pass](device-policy.md) preserves
these defaults, removes superseded policy code and expands bounded offline
calibration across core counts.
