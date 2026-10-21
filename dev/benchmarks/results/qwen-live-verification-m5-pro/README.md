# M5 Pro live-verification experiment, October 20, 2026

**Historical data:** these measurements apply to the source/build identities in this bundle. Current main has since merged a different runtime stack; see [the source reconciliation](../../current-main-audit.md). They are not current-main qualification.


This bundle makes the measurements in [the results report](../../deepinfer-qwen-results.md)
reviewable without the author's ignored build directory. Baseline:
`be3b083c5f413673d44e44881042e875e674e333`. See issue
[#506](https://github.com/incoai/splash/issues/506) for qualification tracking.

| File | Contents |
| --- | --- |
| [measurements.json](measurements.json) | Environment, settings, binary/model fingerprints, all final AC native summaries, HTTP summaries and round measurements, output hashes and native host/GPU interval summaries |
| [native-cycles.csv](native-cycles.csv) | All 1,024 measured native cycles: two threshold experiments × four ABBA rounds × four widths × 32 cycles |
| [http_workload.py](http_workload.py) | Exact HTTP workload snapshot, formatted for readability with a required model-root argument replacing the local installation default |
| [validation.md](validation.md) | Test commands, scope, completed results and limits |

## Reading the evidence

In JSON, `native` includes thresholds 0.1 and 0.2. Its `widths[].rounds` are
seconds per committed token in off/on/on/off order. `noise`, `regression` and
`bound` are fractional values under [ABBA](../../abba.py), not percentages.
`tokens_per_cycle` and `eligible_rows_per_cycle` have one value per round.
Spread above 0.05 is inconclusive even if the candidate's average is faster.

In CSV, `threshold` identifies the experiment. The effective threshold is zero
when `enabled` is zero. `eligible_rows` is the sum across lanes; proposals and
acceptances are actual counters. `prefix_survival_1` through `_7` count lanes
whose accepted draft prefix reached that position. The unchanged full-width
target operators still process eight physical rows per lane.

HTTP data is keyed by series:

- `ac-http-0`: candidate disabled versus clean main; 128 requested tokens.
- `ac-http-0.1` and `ac-http-0.2`: enabled policies versus clean main;
  512 requested tokens.
- `ac-disabled-prose-warmed`: disabled prose control with a ten-second warmup;
  128 requested tokens.
- `ac-enabled-warmed`: threshold 0.1 short workloads and batch-four with a
  ten-second warmup; 512 requested tokens.

Each HTTP series contains `summary` and `rounds`. Rows retain the measured
`decode_cycle_ms`, `decode_wall_ms`, decode output, drafted and accepted counts,
actual prompt/completion usage, finish reasons and SHA256 of generated message
JSON. Generated answer text is omitted. Sampled and strict-schema probe hashes
are also retained. Hash equality is a finite observation, not a universal
reproducibility guarantee.

The native comparison uses the same instrumented candidate binary, with the
policy off/on, and traces all arms symmetrically. HTTP uses separate clean-main
and candidate server binaries and includes between-command time. Consequently
native and HTTP rates are different metrics and must not be compared directly.

The HTTP snapshot targets the named Qwen3.6 MoE model and the original 34 GiB
memory/32K context setup. It is experiment documentation, not a general release
gate. Use the standard regression and release harnesses for broader rollout.
Early battery runs and intermediate conservative-gather builds are excluded
from this final comparison bundle. Complete Chrome traces and full logs remain
local; no model weights or machine-specific absolute paths are included.
