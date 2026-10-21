# Host preparation experiment and decision

**Historical data:** these measurements apply to the source/build identities in this bundle. Current main has since merged a different runtime stack; see [the source reconciliation](../../current-main-audit.md). They are not current-main qualification.


Step 5 of [the Qwen roadmap](../../deepinfer-qwen-plan.md) was evaluated on the
M5 Pro / 20 GPU core / 48 GiB / macOS 26.6.0 machine used for the other local
experiments. **Keep the runtime baseline.** A safe allocation reduction works
and passes ownership checks, but does not produce a qualified serving gain.
Larger graph caching or speculative preparation is not justified by these
measurements. This completes the local conditional experiment, not a rollout.

## Measured opportunity

The ordinary serving traces contain 306 MoE and 510 dense B1 decode submissions.
For each `runtime.decode_submit` parent, `host_bound.py` subtracts the union of
intersecting `backend.prepare` and `metal.encode` intervals. It does not
subtract nested medians or add overlapping scopes. `host-bound.json` records
the result:

| Model | Median submit span | Median remainder | Maximum remainder |
| --- | ---: | ---: | ---: |
| MoE GGUF | 0.3229 ms | 0.1615 ms | 0.1924 ms |
| Dense MLX | 0.3475 ms | 0.1684 ms | 0.2565 ms |

The remainder includes graph construction and other host work outside those
two child scopes. Reusing a graph description cannot remove all of it.
For scale, median complete native decode cycles in these traced workloads
are 30.36 ms and 92.36 ms. Ratios of these medians are not per-cycle bounds or
predicted speedups. The full span analysis identifies a small opportunity;
only serving measurements can establish whether exploiting it matters.

The dense trace exercises real ANE handoffs, but the two-resident-server
configuration serialized arriving long prefill. It is valid for this host
submission analysis and handoff execution, **not** contended-prefill latency.
The subsequent mixed harness requires native evidence of actual overlap.

## Safe experiment

Implementation commit `7a843cb852d6d7f63fa3ec395dc7316c0baba9d8` over
measurement base `bc09da81` is retained as `experiment.patch`. It copies a
fixed initializer-list of Metal buffer handles directly into the dispatch's
owned storage. It removes the temporary vector allocation and retains the
existing vector move path. Each dispatch still owns its parameter bytes;
current positions, parity, page references and indirect grids are bound anew.
It does not cache request state or move state transitions off the engine thread.

The added GPU test constructs 512 distinct parameter payloads, grows/moves
the graph, executes it and checks every indexed result. The complete
shader-validated backend suite passes, including event-split traces, command
failure signalling, residency lifetime/races and table-addressed buffers.
`backend-validation.log` retains the result. Production and profile builds
also pass.

`allocation_probe.mm` constructs 204,800 synthetic dispatches per round.
`allocation-counts.json` records baseline/candidate/candidate/baseline counts:
**1,027,000 / 822,200 / 822,200 / 1,027,000 allocations**. Exactly one
allocation per dispatch is removed (19.94% of counted allocations). The probe
counts ordinary `operator new` calls in this construction workload; it is not
a process memory or GPU allocation measurement. Its recorded elapsed times ran
alongside correctness/install work and are not performance evidence.

To reproduce the count, compile this probe with the same C++20/O3 flags once
against the base headers and once against the patch, then run ABBA. For example,
from each worktree:

```sh
clang++ -std=c++20 -O3 -Iruntime \
  /path/to/allocation_probe.mm -o /tmp/graph-allocation-probe
/tmp/graph-allocation-probe
```

## Production serving result

Both report files compare production binaries with threshold 0 and no tracing
or shader validation, B1 code, 2,048 prompt / 512 output tokens, three trials
and paired resident MoE servers. Binaries/metallibs are frozen; downloads are
stopped. The metric is `decode_cycle_ms / decode_output_tokens`, including
inter-command gaps. AC and thermal snapshots, exact tool/build hashes,
memory, usage and hashes are retained. Prompt/answer text is removed from the
portable copies. All output hashes and sampled/constrained probes match.

| Series | Trial gains | Noise | Decision |
| --- | --- | --- | --- |
| Initial | +0.404%, -0.022% in the two stable trials | 1.25%, 1.27%; first trial 26.89% | No qualified gain |
| Matched sustained warmup | -0.844%, +0.482%, +0.757% | 1.00%, 0.36%, 21.44% | First two within regression floor; third inconclusive |

The warmed series gives each server the same number of complete 2k/512-token
warmup requests until both accumulate at least ten seconds of warmup work.
Warmups are outside measured deltas. Every stable gain is below the promotion
rule `max(2%,2*noise)`; noise above 5% is inconclusive, never a pass. Neither
series supports shipping the change for inference throughput.

**Power-log audit correction:** the warmed series crossed a sleep/wake
transition and is excluded from performance qualification in its entirety,
including the two low-spread trials. Its raw numbers above are retained as
historical diagnostics, not usable evidence. An awake repeat is queued with
process-scoped sleep prevention and an elapsed UTC/awake-clock validity check.
The initial series is unaffected. See the [audit](../qwen-serving-qualification-m5-pro/power-audit.json).

The patch stays an experiment. Device-side propagation, double-buffered arenas
and preparing a future request before acceptance would add ownership and
invalidation work, with no established serving benefit here. Other hardware
may expose a different host budget; the maintainer package in PR #510 can
collect that evidence before revisiting the decision.
