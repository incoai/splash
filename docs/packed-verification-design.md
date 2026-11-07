# Packed verification: proposed allocation boundary

Status: design for discussion. No implementation or performance result is
defined by this document. The serving policy and defaults remain unchanged.

## Existing contracts

`runtime/metal/kernels/decode/sampling.metal` already implements
`draft_verify_rows`: it ranks calibrated prefix-survival gains across lanes
and selects the allocation with the greatest expected committed tokens per
supplied total-row cost. `runtime/model/VerifyBudget.hpp` maintains the
acceptance calibration, censored labels, exploration and a linear cost fit.
Reuse those contracts rather than add another allocation algorithm.

Dense Auto currently verifies all rows. A packed dense policy therefore
needs an explicit, validated eligibility rule; the existing MoE prior does
not qualify it automatically. Extending the lane count also needs its own
calibration, rather than extrapolation of the four-lane prior.

## The execution boundary

A compact projection can gather selected prefixes into 32 physical rows.
Larger totals require the original 64-row projection. Attention, GDN
recurrence, state addressing and the draft retain their original layouts.
Consequently, a linear function of selected rows does not represent the
projection mode boundary. Include the complete cycle cost: selection,
map/gather/scatter, unchanged operators and any original-wide fallback.

For a lane with `r(i)` verification rows and calibrated prefix survival
`s(i,j)`, expected committed tokens are approximately:

```
progress(i, r) = 1 + sum(s(i,j), j < r(i)-1)
utility(r) = sum(progress(i, r)) / cycle_cost(mode, sum(r(i)))
```

Prefixes remain contiguous. Requests near their length limit bound their
own available rows. An unsupported or uncalibrated configuration retains
the established original path.

Start with fixed, independently measured costs for the compact and wide
modes. Where cost is flat within a mode, the initial decision can compare
the best allocation at each feasible mode boundary. A new online cost
estimator is a separate step, justified only after a paired comparison
beats that fixed baseline. Do not introduce another experimental serving
flag as a substitute for a general support rule.

## Operator work before policy work

1. Establish support for every affected projection at model construction,
   including formats, mixed floating segments, epilogues, input plans,
   output padding and scratch extents. Apply no retention restriction on
   the strength of a model-geometry match alone.
2. Use a global selected-row map and its inverse. The fixed four-slots per
   lane mapping cannot represent an allocation that gives one lane a
   longer prefix and another a shorter one.
3. Select the original-wide grids on the GPU when the total exceeds the
   compact capacity. Every affected projection must consume a consistent
   map/mode, with no host readback or partially encoded failure path.
4. Preserve original wide attention/GDN storage, pending-token semantics,
   committed-state addressing and target selection. This proposal does
   not shorten the GDN scan or change the model/draft weights or precision.
5. Account compact buffers and the control map in the existing arena and
   memory plan; retain the wide scratch required by fallback.

The exact target still checks every committed token. A prediction of
survival chooses work to attempt; it does not authorize approximate
committed target selection.

## Feedback and calibration

Keep full-row exploration so late positions remain observable. Proposals
past the verified prefix are censored, not rejection labels. Reuse the
existing acceptance model and measure its calibration on an independent
pilot before a dense policy uses it. Separate lane counts and constrained
policies where their work differs. Sampling/thinking paths need their own
contracts and are outside an initial greedy implementation.

Calibration must charge map/selector overhead, share the actual projection
plans and distinguish compact from fallback mode. Freeze the pilot and
evaluation sets before inspecting final gains. Preserve a fixed baseline
and record all attempted modes, including failures.

## Acceptance evidence for a later implementation

- Map/extent/alias/padding tests with empty, uneven, boundary and overfull
  allocations; actual compact and wide GPU dispatches under shader
  validation; unchanged inputs and reusable scratch.
- Selected full-vocabulary F32 and canonical committed-state comparisons on
  matched histories, with varied lane counts, pending rows, long contexts
  and draft-ring wrap boundaries. Exclude only undefined storage by a
  canonicalization rule fixed before data.
- Unforced raw target-maximum witnesses and natural-response checks;
  real Engine warm reuse, eviction, cancellation, followers and priority
  suspend/resume, including every emitted token's actual prefix/position.
- Fresh balanced cohorts against both uncapped eight-request decoding and
  fixed-cap combined packing. Include mixed and high-acceptance/code-heavy
  prompts, small-width controls, per-request completion, TTFT, memory and
  startup as well as group throughput. Keep models and quantization fixed.
- Freeze source/model/toolchain identities, operational conditions,
  estimator and thresholds before timing. Retain every failed interval;
  do not pool interrupted cohorts or replace an unfavorable sample.

Keep generated reports and local experiment data outside source commits,
as `validate.md` requires. An implementation should remove the prototype
mechanism it supersedes after a general rule and its measurements justify
replacement. This design supplies neither that evidence nor an enablement
recommendation.
