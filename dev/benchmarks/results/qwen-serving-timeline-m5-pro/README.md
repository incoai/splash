# CPU, GPU, mask and ANE diagnostic timelines

These are **display windows from validated historical captures**, not current
main and not production latency datasets. The MoE/dense scheduler traces use
implementation `02a1162e` on measurement base `bc09da81`; the earlier constrained
capture uses `bc09da81`. The native trace hooks are archived experimental code,
not a current-main build target. See [the source audit](../../current-main-audit.md).

Load a `.trace.json` file into a compatible Chrome Trace viewer. Events use
microsecond timestamps and retain separate CPU, GPU, ANE-notification and
request tracks. The windows rebase timestamps to zero and clip crossing
intervals; those events have `args.excerpt_clipped=true`. `windows.json` records
the source bounds, source SHA-256, completion/overflow status and event counts.
Do not run whole-workload timing aggregates on these clipped display files.

| Window | What to inspect |
| --- | --- |
| [moe-2k.trace.json](moe-2k.trace.json) | Bucket-policy mixed prefill/decode, including the larger ragged command in the 2k latency-regression scenario. |
| [moe.trace.json](moe.trace.json) | EMA control with an arriving 8k prompt and interactive request beside active decoding. |
| [dense.trace.json](dense.trace.json) | Dense contention uses small GPU chunks; configured ANE availability does not make these chunks ANE work. |
| [dense-ane.trace.json](dense-ane.trace.json) | A separate actual ANE-enabled 2,048-row prefill command, including host handoff notifications and event-split GPU segments. This is outside active-decode contention. |
| [masks.trace.json](masks.trace.json) | Constrained target forward, host mask roundtrip and remaining mask-tail wait. |

The [prefill evidence bundle](../qwen-prefill-cost-m5-pro/README.md) includes
the full original MoE/dense trace bytes in deterministic `*.trace.json.gz`
archives, with archive/source hashes and per-case geometry in
`trace-analysis.json`. The complete earlier constrained trace is
`masks.full.trace.json.gz` here. Decompress an archive to recover its original
Chrome Trace JSON. The full source validators reject dropped events,
unfinished commands, unsuccessful serving processes and invalid intervals.
The interrupted earlier dense diagnostic is explicitly excluded in
`excluded-dense-stale-status.json`, rather than substituted for a complete run.

The dense repeat has 2,048 ANE queued jobs across its four full round traces.
During the eight measured active-decoding intervals it instead submits 304
GPU prefill commands of 128–256 rows, below its 1,801-row split minimum; none
queues an ANE job. Handoffs elsewhere belong to warmups, initial prefill or
work after the active decoder finishes. Native HTTP progress independently
confirms real contention. This distinction prevents an ANE-configuration field
from being presented as evidence of contended ANE execution.

Host scopes overlap. GPU command spans can include event waits; segment
boundaries do not by themselves identify hardware wait duration. ANE
queue/ready/report intervals include callback latency and are not hardware ANE
execution timestamps. Instrumentation changes timing. Use the separate frozen
production reports for throughput/latency gates; use these views to inspect
execution order, geometry, CPU preparation and synchronization.
