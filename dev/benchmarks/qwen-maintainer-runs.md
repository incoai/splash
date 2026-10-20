# Repeatable Qwen qualification on maintainer Macs

Tracking: [#508](https://github.com/incoai/splash/issues/508). The local machine
is an Apple10 M5 Pro with 20 GPU cores and 48 GiB. Only that machine is
available to the contributor. Apple9, another Apple10 configuration and smaller
memory configurations are **unexecuted**; preparing this package does not
qualify those devices or a shared default.

## Freeze builds and model identities

Use isolated checkouts and build directories. Build `splash`,
`engine-tests/model-runtime-oracle` and `engine-tests/decode-profile` with the
same production flags. Keep `splash.metallib` beside each executable. Never
rebuild a binary or metallib used by a live server. Record both commit ids,
dirty diffs, Xcode/Metal toolchain, OS, GPU family/core count, memory, power and
thermal state. The runner hashes executable, library and installed-manifest
bytes before executing, and refuses artifact changes during the run.

Copy [the model list](qwen-qualification-models.example.json), replace its four
root paths and check each installed `model.json` against the pinned target and
draft revisions listed there. The recorded revision fields are documentation;
the runner retains the actual manifest hash, not a claim that arbitrary files
match the example. Keep the same model bytes and settings in both arms. The
MoE GGUF cell uses the IQ3_XXS qualification target; the dense GGUF cell uses
UD-Q4_K_M. Install through the normal pinned installer rather than modifying
prepared model files.

## Prepare, review and execute the matrix

Run from the candidate checkout with its Python environment:

```sh
.venv/bin/python -m dev.benchmarks.qualification_matrix \
  --models /path/to/models.json \
  --baseline /path/to/baseline/build --candidate /path/to/candidate/build \
  --output build/qualification-apple9 --live-gdn
```

This writes commands and marks every job `unexecuted`. Review memory settings
and commands, then repeat with `--execute`. Omit `--live-gdn` for a candidate
without the optional GDN experiment. `--suite correctness` and `--suite serving`
allow separate runs. Save separate output directories so one scope does not
overwrite another's evidence.

For each of the four model/storage cells, the full package prepares:

- The real-model oracle with INT8/BF16 KV crossed with FP32/BF16 recurrent state,
  including shader validation, snapshot/restore, lane movement, preemption,
  masks, thinking limits and sampled-path checks already in that oracle.
- Three single-resident ABBA trials for code, math and prose at B1 and exact prompt
  contexts 2,048, 8,192 and 31,744, each with 512 committed output tokens.
- B2/B3/B4 at 2,048 prompt rows, and a B1 mixed streaming/long-prefill/interactive
  workload at the three context sizes, plus seeded sampling and strict JSON.

Correctness and timing use separate processes. Serving speed runs turn tracing
and shader validation off. A zero command exit means the harness executed;
each child `report.json` retains its performance verdict, noise and output-hash
diagnostics. The package's `complete` field means all jobs executed successfully,
not that every policy passed a promotion gate. Incomplete jobs and failures
retain their logs and do not become passing results.

The default `--residency sequential` starts one server per complete ABBA round
and closes it before starting the next, with four matched 2k/512-token warmup
requests per round. This avoids the observed second-server Metal out-of-memory
failure for the MoE safetensors model and admission serialization for dense ANE
workloads on the 48 GiB Mac. `--residency paired` remains available when there is
enough memory for both; record that method separately. Native progress checks
must still prove actual long-prefill contention rather than HTTP overlap.
Per-model memory limits are upper bounds and must fit the runner's capacity.
If a cell cannot run at the chosen settings, report it as unexecuted.
Do not silently lower model quality or compare unlike memory/ANE settings.
B2–B4 at 8K/32K and long-context limits are additional memory qualification,
not claimed by the initial 2K multi-lane matrix.

## Isolate individual decisions

The main matrix compares a clean baseline against the candidate configuration.
For GDN's incremental gain, use the same frozen candidate binary for both arms
of `serving_regression`: threshold 0.1 in both, `SPLASH_LIVE_GDN=0` versus `1`.
Use `decode-profile --live-gdn off|on` with the same threshold, context, KV and
state format for native ABBA runs. Native kernels retain eight physical rows;
eligible rows and committed output are different quantities.

For host allocation changes, keep verification settings identical and compare
the unchanged measurement base with the host candidate. Collect symmetric
`serving-profile` traces separately for attribution. Do not qualify speed from
an instrumented candidate versus an uninstrumented baseline.

Adaptive-policy [#487](https://github.com/incoai/splash/pull/487) sits on a larger
dependency stack and speaks protocol 11; this roadmap's original base speaks
protocol 10. Run its own checkout/server with both arms on immutable revision
`67bb94069179470125d393da5e5725e3f1501f9d`. The
[qualification-only patch](patches/pr-487-fixed-budget.patch) adds
`SPLASH_BENCH_FIXED_VERIFY_ROWS=1..8` for unconstrained greedy MoE requests.
Without that environment value the upstream adaptive choice is unchanged.
The fixed control caps each lane before the existing verify-row selector and
uses its full capped prefix; it does not replace the upstream confidence
feature with this roadmap's cumulative-confidence heuristic. The upstream
online fitting overhead remains present in the controls.

Apply the patch in an isolated checkout and copy `serving_regression.py` and
`trace_report.py` into its `dev/benchmarks` directory. Build and run its existing
`verify-budget` CPU test. Select the best fixed setting on training prompts,
then compare adaptive against that setting on held-out code/math/prose and
contexts; retain the full-width comparison too. Keep raw arm timings and
server-history order. Changes in greedy/seeded outputs are an explicitly
documented behavior of that upstream policy, and finite seeded probes do not
establish distribution preservation.

## Return reviewable evidence

Provide the qualification manifest, child reports, logs, actual model/binary
hashes and trace summaries. Exclude model weights and generated answer text
from the public bundle; retain answer hashes and committed counts. For useful
output throughput, a gain must exceed `max(2%, 2 * measured noise)`; spread
above 5% is inconclusive. Report contended TTFT/chunk gaps and throughput
separately, each device and workload separately, and any output differences.
Never infer cross-device qualification from one device's aggregate result.
