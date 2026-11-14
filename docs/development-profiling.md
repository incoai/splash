# Flash-Next profiling tooling and disabled capture fusion fixture

This change adds development tooling and model-free correctness tests. No production
runtime, Metal kernel, serving option, model, queue policy or cache setting changes.
The experimental fusion is compiled only into its dedicated test library; no
production graph can select it. This does not qualify a serving-speed or accuracy
gain.

## Profiler configuration and input replay

`decode-profile` accepts these additional development options:

- `--activations auto|bf16`, using the serving option's meaning: `auto` permits
  INT8 target activations and `bf16` disables them. Legacy `auto` remains the
  default. Recurrent state still defaults to FP32 unless explicitly selected.
- `--max-context N` explicitly chooses the descriptor's served window; omitting it
  preserves the installation's existing context. The fixture conservatively
  bounds prompt plus normal/replayed cycle growth to 262144 tokens by default.
- `--max-memory SIZE` prechecks estimated resident weights and sets the KV/state
  memory governor's limit. It does not reproduce the engine's entire streaming,
  host-memory, warm-cache or request-admission budget. Reports state that limit's
  scope; it must not be called a full production memory-plan qualification.
- `--input-tokens FILE` reads a bounded regular file containing whitespace-separated
  unsigned decimal token IDs. It is exclusive with `--prompt-tokens`. The fixture
  must already contain the exact formatted request, including its assistant
  prefix, using the installation's tokenizer. No retokenization or template
  substitution occurs. Token IDs are vocabulary-checked before model allocation.
- `--require-flash-next` requires BF16 activations/state, INT8 KV, explicit 96G
  memory and 262144 context, and a token fixture. The actual text execution path
  also checks Flash-Next target geometry. This validates family/settings, not
  an exact Q2_0 installation or production equivalence; pin model/source identity
  and all loaded weight formats separately.
- `--report FILE` writes a private, exclusively created JSON report after successful
  text execution. It records requested settings (including expert/PLE streaming budgets and generation
  reserve), installation source identity,
  effective native limits, prefill GPU time, whole-cycle wall/emitted-token
  denominators and per-width command observations. HTTP TTFT, Next Queue wait,
  CPU-only graph time and SSD cache reuse remain unknown, not measured zero.
- `--config-only` validates inputs and prints configuration JSON, then exits before
  opening Metal, inspecting/loading a model, or writing report/dispatch files.
  A successful dry run does not validate an installation or output permissions.

The private replay identity is SHA-256 over the consumed token IDs serialized
little-endian uint32. It identifies tokens rather than the input file's spacing.
Raw token IDs, input filenames and prompt/output text are not included in the
configuration JSON. Keep the fixture and any external provenance manifest local.

Output files must be new. Existing report/dispatch files and duplicate output
paths are rejected; exclusive creation also prevents races or symlink clobbering.
A write failure is an error and can leave a partial owned output file; it is not
reported as success. No partial report qualifies a run.

Normal native command timing and separate-dispatch attribution retain separate
scopes. Replay changes cache/scheduling behavior; subtracting its duration from
whole-command time does not isolate dispatch overhead. This tool does not send
HTTP requests, exercise the 512 GiB persistent KV cache, or measure eight-user
admission latency. Those require a separate, explicitly authorized experiment.
No performance claim should be based on the new dry-run or unit-test fixtures.

## Safe validation commands

These targets load no model and collect no timings:

```sh
make test-decode-profile-options test-hyper-capture-contract
make test-hyper-capture-parity
```

The last target runs the dedicated shader fixture normally and with Metal shader
validation. On a machine with multiple developer installations, set
`DEVELOPER_DIR` for the command instead of changing global Xcode selection.

Compile the native profiler without running it:

```sh
make -j2 build/engine-tests/decode-profile
```

Then test the real executable's early-exit path with nonexistent model/Metal
library paths:

```sh
python3 dev/tests/tools/profile_cli_preflight.py build/engine-tests/decode-profile
```

Do not run `benchmark-decode-profile` or invoke the native profiler without
`--config-only` during the current model-benchmark pause.

## Exact GPU parity fixture

The test Metal library includes the actual unchanged production
`hyper_connection.metal`, then adds a test-only candidate that combines the
residual update and stream-mean capture. Every thread retains the original FP32
stream update and ordered FP32 sum, writes all residual streams back and converts
only the final mean to BF16. No persistent queue, global barrier, normalization,
prefill path or production operator is changed.

The candidate's test-only contract rejects invalid extents, alignment, geometry,
integer overflow and overlapping writable byte ranges before command encoding.
Read-only inputs may alias each other; disjoint views of a common buffer remain
valid. This fixture uses fresh Metal buffers with explicit identity/offset ranges;
it is not an arbitrary-resource alias proof for future production buffer wrappers.

The 360 cases cover one through four streams, rows 1/5/8/16/24/32, hidden widths 1/255/256/257/2560 and capture
slots zero/middle/last. B3 has padded B4 storage. Captures use a nonzero row offset
and wider destination stride. The complete residual, capture, input, guard and
padding buffers are compared byte-for-byte against the original two GPU kernels.
Finite random values, deterministic cancellation, signed zero and BF16 rounding
boundaries are included. NaN payload and infinity behavior are not qualified.

A deliberately reordered FP32 reduction must disagree on the cancellation fixture
in all 90 four-stream cases. This proves the observer can detect at least that important fault;
it does not prove completeness. The fixture collects no elapsed-time data.

## Validation recorded 2026-11-13

| Check | Result |
| --- | --- |
| Profiler parsing, token input and private report output | 59 checks passed; also passed address/undefined-behavior sanitizers |
| Buffer, alias and geometry contract | 33 checks passed; also passed address/undefined-behavior sanitizers |
| Real profiler executable with nonexistent model/Metal paths | 11 early-exit CLI cases passed |
| M5 GPU operator parity | 360 cases passed normally and 360 with Metal shader validation |
| Invalid bindings / deliberate faulty reductions | 1800 rejected / 90 detected in each GPU mode |
| Native profiler | Compiled and linked on M5 with command-scoped Xcode 27.2.0 |
| Runtime architecture / whitespace | Passed; production runtime remains 30105 lines |

GPU validation used Apple M5 Max. The candidate and reference were built with the
same production Metal compiler options. No model was loaded for these tests, no
serving benchmark or generation probe was sent, and no deployment occurred.
The post-test health snapshot retained the same running Splash instance, Flash-Next
Q2_0, 96 GiB RAM, 512 GiB cache, 262144 context, eight queue workers and M4 overflow
disabled. Splash and queue health/readiness checks passed; queued/running/failed
counts were zero at that snapshot.

Raw build output, source/binary SHA-256 hashes and the latest production health
snapshot are retained locally in ignored `build/pr-review-validation.json`.
The earlier implementation receipt remains in `build/profiling-validation.json`. These results
qualify the development tooling's tested paths and finite operator fixture only.

## Correctness review

The review added bounds for the kernels' uint32 row/stream indices; checking only
uint64 buffer sizes could otherwise accept dimensions whose device indices wrap.
Tests isolate those cases with generous synthetic buffer capacities and accept the
last-row boundary exactly at UINT32_MAX. GPU parity now covers each stream count
accepted by the contract, rather than only four streams.

Reports now include expert/PLE streaming budgets, generation reserve and the
Flash-Next matching requirement, so distinct native configurations are visible.
The two model-free CPU checks also run through the existing `test-engine-cpu` CI
gate. The GPU fixture remains an explicit development target; these local results
do not claim that GitHub-hosted Metal validation has passed.

## Hosted CI Metal bootstrap

The production/app and Python jobs request the standard GitHub macOS 26 image.
The earlier Blacksmith stable-default image booted macOS 26.3, below the 26.4
production floor, while its macOS 27 image selected an Xcode release candidate
whose Metal catalog repeatedly failed. Native sanitizers retain macOS 27
coverage; the production deployment floor is unchanged.

Installer lifecycle fixtures supply their own supported OS/architecture. Separate
regressions verify rejection of macOS 26.3 and non-Apple-Silicon hosts before any
installation; the production installer guard is unchanged. CI's app step uses
`sh dev/tools/check_app_ci.sh` to disable Reduce Motion only for the animation
suite and restore the runner's original value (or absent key) on exit. Seven
subprocess tests cover restoration, absent/disabled preferences, application
failure, mutation failure and unexpected input without touching real preferences.
The combined targeted installer/preference-guard suite passed 19 tests on M5.

Hosted CI runs `python3 dev/tools/ensure_metal_toolchain.py` before the native
and Python compatibility gates. An unconditional component download previously
failed fetching Apple's MetalToolchain catalog before any project compilation.
The bootstrap first compiles BF16 Metal 4.0 source with the production deployment
target and links a nonempty metallib. A usable installation needs no download.
If the probe fails, it makes at most five timed component-install attempts with bounded backoff,
reprobes after each attempt (even when the installer exits with an error), and
fails closed if compilation/linking remain unusable. It logs the selected Xcode
and SDK without changing global Xcode selection or executing GPU work.

Six model-free subprocess regression tests cover the existing-toolchain/catalog
failure, installation, transient failure, installer-error-but-usable-tools,
installer-success-without-tools, and broken/missing/empty compiler/linker outputs.
Run them with `python3 -m unittest dev.tests.tools.test_metal_toolchain`.
The real compile/link probe also passed on the isolated M5 development checkout
using command-scoped Xcode 27.2.0, without a component download. These local
checks do not replace successful hosted CI at the reviewed commit.

## Remaining gates

The candidate is not integrated into `Qwen4ExpTarget`, even behind a runtime flag.
Production fusion remains disabled by construction. Before integration, review
actual resource aliases/lifetimes, declare all resource dependencies, preserve
capture-slot ordinals and every active verify row, and retain the unfused fallback.
Prefill's per-sequence capture slices require a separate design.

Exact operator parity on finite test tensors does not establish full logits,
acceptance decisions, persistent draft/recurrent state, cache restore, cancellation,
or generated-code correctness. Real-model qualification and TTFT/throughput
measurements remain deferred. Keep Flash-Next, 96 GiB RAM, 512 GiB external cache,
M5-only routing and the existing activation/state/KV settings unchanged.
