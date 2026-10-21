# Serving measurements for the Qwen optimization roadmap

This follow-up to [the initial MoE experiment](deepinfer-qwen-results.md)
provides the full-engine measurements required by steps 1, 4, 5 and 6 of the
[plan](deepinfer-qwen-plan.md). Tracking: [#508](https://github.com/incoai/splash/issues/508).
It adds measurement tools; no scheduling or GDN arithmetic is changed here.

## Instrumented serving

Build `make -j8 build/serving-profile`. This is the native protocol executable
with benchmark-only Metal instrumentation, beside `build/splash.metallib`.
The regular server can select it with `--binary build/serving-profile`.
Only this executable reads `SPLASH_SERVING_TRACE=/path/to/trace.json`; production
`build/splash` does not expose this environment option.

An optional trace sink records the following on the awake/system mach clock:

| Events | Meaning |
| --- | --- |
| `frontend.receive`, `native.control`, `native.tick` | Protocol processing, control work and engine advancement |
| `scheduler.plan`, `engine.admit`, `engine.prepare` | Selection and resource/cache preparation, including unsuccessful attempts |
| `runtime.prefill_submit`, `runtime.decode_submit` | Graph construction and command submission; width, rows and maximum prior history are retained |
| `engine.result`, `engine.apply` | Ticket result consumption and host application/cache/state bookkeeping |
| `constraint.initial_mask_roundtrip`, `constraint.mask_roundtrip`, `constraint.mask_tail_wait` | Initial-selection/verification mask roundtrips versus the remaining wait after target forward finishes |
| `ane.queue_to_ready`, `ane.ready_to_report` | Host event notification and report intervals; not hardware ANE execution timestamps |
| `backend.prepare`, `metal.encode`, `metal.completion` | Pipeline/binding preparation, encoding and completion notification |
| `metal.command_span`, `gpu.segment` | Whole command span and individual event-split command-buffer GPU intervals |
| `request.active` | Request lifetime, used to separate idle gaps from gaps while work is pending |
| `batch.observation`, `batch.cycle`, `batch.counters` | Ticket wall time, existing engine cycle metric, actual output/proposal/acceptance counts |
| `draft.confidence` | Existing live-prefix confidence and observed prefix survival for greedy MoE; absent when the live selector does not run |

Host scopes overlap and cannot be summed. ANE intervals include callback delay
and can overlap GPU work. Command-buffer spans can include event waits; they
do not reveal the exact hardware duration of an individual wait instruction.
Request-active gaps can also include resource waits or constrained stages.

The host buffer reserves 200,000 events. The command trace retains at most
65,536 commands. Overflow drops records without changing serving and is
reported. Trace IO happens after the workload. Shared ownership keeps late
ANE callbacks' sinks alive. Names have static storage; runtime events contain
geometry and opaque ids, without prompt text or token ids. Null sinks read no
instrumentation clock and allocate no trace buffer.

`batch.observation` flags are prefill=1, representative timing=2 and
constrained=4; its value is ticket wall milliseconds. `batch.cycle` stores the
existing engine cycle milliseconds. `batch.counters` uses rows for committed
output tokens, context for proposed tokens and flags for accepted tokens.
For `draft.confidence`, rows is the one-based position, context is eligible
rows and value is cumulative selector confidence. Flags observed=1 and
accepted-prefix=2 distinguish rejection from a censored suffix. An accepted
truncated prefix does not label its unverified suffix as rejected.

Validate and summarize a completed trace with:

```sh
.venv/bin/python -m dev.benchmarks.trace_report build/run/server-1-trace.json \
  --output build/run/trace-summary.json
```

The report rejects dropped, unfinished, unsuccessful or invalid-clock traces.
It preserves prefill rows/history/width and representative flags for estimator
analysis. Full traces remain ignored build artifacts.

## Paired cold HTTP throughput

Each benchmark holds a process-scoped `caffeinate -is` assertion on macOS;
it releases that assertion on success or failure. It does not change global
power settings. Reports compare elapsed UTC and the Mac's awake monotonic
clock, and reject a difference of one second or more as suspension or a wall
clock adjustment, even if ABBA costs happen to look stable. A power-log audit
invalidated several overnight captures; those are retained as excluded data
and repeated while awake. Power/thermal snapshots alone do not detect sleep.

The new harness keeps two servers resident, then issues serial requests in
baseline/candidate/candidate/baseline order. It never contacts an existing
server. The shorter separation between arms reduces one source of thermal
drift. Enough memory must remain for both models; shared residency and GPU
cache effects limit generalization. This complements the standard sequential
release gate and does not replace it.

Every request gets a unique cache salt, and actual cached prompt tokens must
be zero. The first smoke test found that repeating a prompt otherwise mixes
cold and cached numerical graphs: both baseline and candidate changed output
between those two conditions, while corresponding conditions matched.
Exact-context prompts, actual output budgets, all four arm measurements,
message hashes, fingerprints, thermal state and memory snapshots are retained.
Incomplete runs are marked explicitly and both servers are cleaned up.

For final speed comparisons use production binaries **without `--trace`**:

```sh
.venv/bin/python -m dev.benchmarks.serving_regression \
  --model unsloth/Qwen3.6-35B-A3B-GGUF --model-root /path/to/installed-model \
  --baseline-binary /path/to/baseline/build/splash --binary build/splash \
  --baseline-env SPLASH_LIVE_VERIFY_THRESHOLD=0 \
  --candidate-env SPLASH_LIVE_VERIFY_THRESHOLD=0.1 \
  --contexts 2048,8192,31744 --tokens 512 --samples 3 \
  --output build/paired-serving
```

For a sustained warmup, add `--warmup-context 2048 --warmup-tokens 512
--warmup-seconds 10` (on one command line). Complete matched rounds alternate
between both servers until each has accumulated at least that much request
wall time. The faster arm receives the same number of warmup requests.
Warmups use unique cache salts and remain outside measured metric deltas;
counts, durations, settings and the harness source hash are recorded. This
helps investigate startup transients; it does not excuse an inconclusive
spread or establish the cause of observed noise. The default retains two
128-token warmups per arm for short functional diagnostics.

Use `--residency sequential --warmup-seconds 0 --warmup-rounds 4` when two
resident models cause memory pressure. This starts one server per complete
ABBA round, runs the same workload sequence and fixed number of warmup
requests, and closes it before the next arm starts. It trades longer time
between corresponding cases for natural single-server memory conditions.
Actual warmup times and per-round statuses are retained. Do not interpret
paired-resident memory-pressure results as isolated production latency.

The default per-server memory limit is 20 GiB. Set `--max-memory` as appropriate;
the existing engine admission/reserve rules remain in force. Widths 2–4 use
`--batch`, preferably with the smaller contexts on this machine. A requested
prompt plus output must fit `--max-context`.

Add `--trace --probes` and select `build/serving-profile` to collect a diagnostic
timeline plus seeded sampling and multi-cycle strict-JSON checks. Trace overhead
is then part of execution, so these are diagnostic measurements. In particular,
an instrumented candidate versus an uninstrumented baseline is not an isolated
verification-policy comparison.

## Mixed prefill and decode

Add `--scenario mixed --batch 1 --tokens 512`. Each arm starts an active
2,048-token streaming request, waits for content, introduces the selected long
prompt after 50 ms, then a 512-token interactive request after another 50 ms.
Both arriving peers request 32 output tokens. `--arrival-delay` changes those
offsets. The harness polls native request progress every 100 ms and requires actual
long-prefill progress while the original native request still decodes. HTTP
request overlap by itself is insufficient. Fully cached continuations with no
new prefill are labelled separately. The samples and polling interval are
retained, and the same control traffic is present in both arms.

`--cache-mode cached` primes the complete long prompt before the measurement.
`--cache-mode prefix-resume` primes its leading half by character count, then
extends that shared prefix in the measured request. Each arm uses a fresh salt;
the primer must be cold, the measured long request must reuse cached tokens,
and active/interactive peers remain cold. Actual cached counts are retained.
These are RAM-prefix reuse/continuation scenarios; they do not simulate a
request suspended under memory pressure. Priming is outside the measured
interval, while restore and suffix prefill in the arriving request are inside.

The primary mixed metric is total workload wall seconds per output token,
including prefill and frontend work. It also records short/long HTTP TTFT and
p95/maximum content-chunk gaps during the contended interval. A content chunk
is not necessarily one token; timestamps are client arrival times. Native
decode metric deltas are retained separately. Sampling enough complete trials
is necessary before interpreting p95 as evidence about contended latency.

## Completed local checks

- Production/instrumented builds, engine/native-loop tests and architecture checks pass.
- Complete CPU and ASAN/UBSAN/TSAN suites pass after the instrumentation changes.
- Shader-validated backend tests cover ordinary and event-split GPU traces.
- Handoff success/failure/timeout/late-report/cancellation tests pass with a sink attached.
- Six report/environment tests and 38 existing benchmark/smoke tests pass.
- Real mixed HTTP smoke completes with matching outputs and valid seeded/strict-JSON probes.
  Both traces contain 26 verification-mask roundtrips/tail waits, no dropped events and exit status zero.
  The final audit adds a distinct initial-mask span, verifies cut commands' actual
  row counts, and excludes provisional cancelled output from committed counters.

[Portable diagnostics](results/qwen-serving-trace-m5-pro/diagnostics.json)
retain the numerical evidence. The three cold code diagnostics had 0.10–0.31%
spread and matching outputs; their candidate had tracing enabled, so they do
not qualify a policy speedup. The mixed smoke ran alongside installation and
sanitizer activity and is execution validation, not latency evidence. Quiet
production comparisons, dense-model ANE traces and the expanded model matrix
are the next qualification runs.

## Overlap guard audit

The first dense ANE diagnostic used two resident servers. Both exercised real
ANE handoffs, but memory pressure serialized admission: no long prefill command
was submitted before the original decode request ended. The former client
lifetime check accepted that case. It is now rejected by native-progress
checks, with unit coverage for serialized admission and cache-restore-only
progress. That earlier capture validates handoffs/execution, not contended
prefill latency. Single-resident runs are required for this dense configuration.
Per-round memory snapshots now use the actual status schema, and scheduled
decode-width histograms are retained separately from requested concurrency.
