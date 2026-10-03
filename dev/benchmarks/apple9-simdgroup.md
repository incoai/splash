# Apple9 Q4 decode with simdgroup matrices

Apple9 Q4 decode uses eight-row bfloat matrix operands, one independent tile
per request lane at one to four lanes. Plain projections at three or four lanes
whose grid holds at least two N256 tiles per core keep the broad-column path
(`widePlain` in `runtime/ops/Linear.cpp`). Apple10 and prefill retain their
policies. The packed Q4 weights and the `Q4Params` ABI are unchanged. The
measurements below are of the one-lane version; `3a20983` extended it to two
through four lanes with identical output hashes and acceptance counts.

Each SIMD group computes two 8-column fragments, or one fragment each for
gate and up. Nibbles become exact bfloat values `128 + q`; fp32 accumulation
subtracts `128 * sum(x)` before applying the existing scale and bias. The
activation table stays bfloat, including FFN down inputs that can exceed half's
range. The residual and gate/up paths retain the original bfloat rounding
boundary before their epilogues. Arithmetic is reassociated and is not bitwise
identical to the sequential Q4 kernels.

`LinearPlan` owns the grid, split count and scratch sizes. The split rule picks
powers of two up to eight to approach 16 column/K threadgroups per GPU core,
while retaining at least 12 quantization groups per partition. The final
arriving group reduces partials in a fixed order and resets its counter; no
group spins. Every matrix accumulator chain is initialized before the loop.
Initializing fragments only within the loop produced wrong results under
Apple10 GPU validation; the dedicated regression test covers this case.

The decode arena owns one reusable activation table, row-sum buffer, partial
buffer and counter buffer, charged to the memory budget. It is private to the
serial decode command stream. Counters are zeroed at allocation and reset by
each completed dispatch. Startup choices reserve the maximum of the default
and selected plans. No scratch allocation occurs while encoding a projection.

RMSNorm writes its ordinary output and the matrix operand layout from the same
rounded bfloat values. Its next Q4 consumer marks the input prepared. GDN and the attention gate also emit
the prepared layout from their rounded output. The FFN gate/up producer still
uses a separate preparation dispatch. Reused draft head/selector and
context inputs share their prepared table until another producer overwrites it.
Producer fusion reuses the same arena buffers and does not change Q4 weights.
See [remaining-decode-optimizations.md](remaining-decode-optimizations.md) for the
follow-up measurements and rejected experiments.

## Validation

```sh
make -j8 all test-engine-cpu test-engine-metal
```

The Metal target enables GPU shader validation. `q4-sgmatrix` compares against
an independent fp64 packed-weight reference for all three epilogues, every
valid split count in 1/2/4/8, small and irregular K, and real FFN widths. Inputs
include values above half's range, cancellation and zero weights. It also
checks guard bytes, repeated workspace use, deterministic reductions, zeroed
counters, and bitwise equality of separate versus fused RMSNorm preparation.

Error bounds use input and weight magnitudes, so cancellation does not hide
behind a relative-output threshold. They cover fp32 dot/affine reassociation,
bfloat output rounding and epilogue propagation. The tuning qualification uses
the same operand-based reasoning; its timing includes preparation when the
input has no fused producer. End-to-end measurements are required to assess
the benefit of producer fusion and changes in speculative acceptance.

## End-to-end measurements (2026-09-20)

M3 Max, 40 GPU cores, Apple9, macOS 27.0.0. Baseline: Q4/MoE integration
`7a63b4c`; candidate: `2c788c0`, production build ID
`src-c380c7f4024d8118a26d95b5ddaa32ce4eff149287e94c4cdd8db7f1ddd65566`.
The candidate includes the later candidate-count fix and integration cleanup.
The same local Splash packages, prompts, temperature 0, disabled reasoning,
256 generated tokens and 32768 context limit were used for both builds.
Each fresh server had one 64-token warmup. ABBA order, two repetitions per
scenario per phase: 32 measured requests per model, four per build/scenario.

All values below are medians. Decode throughput comes from engine token/time
counters; a streamed message chunk is not assumed to equal a token.

| Model | Scenario | Baseline tok/s | Candidate tok/s | Gain | Cycle ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 27B | technical | 34.3 | 46.5 | 35.6% | 89.2 → 60.0 |
| 27B | python_code | 59.2 | 87.9 | 48.6% | 88.5 → 59.6 |
| 27B | typescript_code | 64.8 | 101.7 | 56.9% | 88.0 → 60.0 |
| 27B | repeat | 93.5 | 135.8 | 45.3% | 85.8 → 59.0 |
| 35B | technical | 87.2 | 110.2 | 26.5% | 27.2 → 24.7 |
| 35B | python_code | 178.0 | 185.1 | 4.0% | 27.1 → 24.7 |
| 35B | typescript_code | 183.0 | 205.6 | 12.4% | 27.4 → 24.9 |
| 35B | repeat | 285.1 | 315.6 | 10.7% | 28.1 → 25.3 |

The M3 was on battery with automatic power mode. Absolute timings drifted
through the longer 27B run: baseline cycles ranged roughly 77–94 ms and
candidate cycles 55–67 ms. Preserve this limitation when quoting the results.
An earlier shorter ABBA run measured approximately 77.1 → 55.8 ms. The real-model
oracle independently measured a 57.1 ms one-lane cycle. These measurements do
not establish sustained AC/high-power performance or a same-condition ranking
against another serving engine. The serving ABBA harness is not in the
repository; `make benchmark-decode-profile MODEL=... DECODE_PROFILE_ARGS='--cycles 7'`
reports one build's cycle time at each batch width.

The M5 Pro 16-core control used the same integration/candidate comparison
with 192 tokens, two scenarios and one repetition per phase. Decode throughput
was 37.7 and 61.0 tok/s, changing by +0.1% and 0.0%; output hashes matched.

Builds and CPU/Metal suites passed on M3 Max 40-core, M5 Pro 16-core, and
M5 Pro 20-core. The dedicated simdgroup test passes with GPU validation on
all three, including explicit matrix-kernel execution on Apple10. Full native
model-runtime oracles pass for both 27B and 35B on the M3, including state
restore, preemption continuation and one- through four-lane batch transitions.
