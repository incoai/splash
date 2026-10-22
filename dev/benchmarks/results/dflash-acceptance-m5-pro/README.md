# DFlash acceptance measurements on M5 Pro

These October 22, 2026 diagnostic probes support the
[model analysis](../../dflash-model-analysis.md). They characterize the released
checkpoints using ordinary greedy inference. No new model was trained, no
alternative selector was deployed, and throughput was not measured.

`manifest.json` records the base commit, checkpoint revisions, source hashes,
configuration and SHA256 checksums of the raw and compressed captures. Each
`*.report.json` is produced by `dev.benchmarks.draft_replay`; the authored
prompts are in `prompts.json` and their exact chat-formatted token IDs in
`*.u32`. The same dense tokenizer's IDs were fed to both targets.

The six main B1 rows use 64 cycles for each of four dense-model prompts and
two MoE prompts. The presence-penalty probe is a separate changed trajectory.
B2/B4 and shader-validated B3 probes exercise the reader's packed batch
layout; their cloned lanes are not independent prompt samples. The ten
captures contain 552 lane-cycle records in total.

The ten `*.jsonl.gz` captures are included here, about 4.4 MiB compressed.
Uncompressed copies and native logs are also retained in the ignored directory
`build/dflash-acceptance-m5-pro/`. Replay reads the gzip files directly.
Do not concatenate captures: each file has its own metadata, request IDs and
completion record. Verify each raw SHA256 against the manifest before replay.

For example, after selecting an installation of the pinned target and draft:

```sh
make build/engine-tests/decode-profile
build/engine-tests/decode-profile build/splash.metallib "$MODEL_ROOT" \
  --cycles 64 --width 1 --draft-dump build/code.jsonl \
  --prompt-file dev/benchmarks/results/dflash-acceptance-m5-pro/code.u32
.venv/bin/python -m dev.benchmarks.draft_replay build/code.jsonl \
  --output build/code.report.json
```

For the other B1 prompts, substitute `math.u32`, `prose.u32` or `json.u32`.
The penalty probe adds `--presence-penalty 1.5`. Batch probes use code tokens:
`--width 2 --cycles 8`, `--width 4 --cycles 16`, and
`MTL_SHADER_VALIDATION=1 ... --width 3 --cycles 8`. All use INT8 KV, FP32
recurrent state, thinking disabled in the prompt, and all verify rows.

The optional `--edge-scales 0,0.25,0.5,0.75,1,1.25,1.5,2` replay generates the
exploratory results in `edge-scale-replay.json`; no scale is selected for
deployment. Captured-source hashes describe the original capture code;
analysis-source hashes describe the replay/tests used for the published PR.

Validation of the original captures:

- Production binary and instrumented decode-profile build successfully.
- 14 replay unit tests, including exhaustive checks of both dynamic programs,
  FP32 ties, censoring, complete batches and malformed captures, pass.
- 62 tests across replay and the existing serving/backend/HTTP reporting tools
  pass. The existing HTTP test harness emits ResourceWarnings while passing.
- Native draft-selector and target-sampling tests pass with Metal shader
  validation enabled.
- All ten real-model captures pass greedy token and confidence replay,
  including every tested batch width and the shader-validated B3 capture.
- Separate smoke checks pass for the original dispatch-profiling mode and
  an early end-of-sequence capture, whose unavailable suffix is censored.
- Architecture validation, `git diff --check`, and Ruff checks of the new
  Python files pass.

The PR follow-up adds 20 replay tests, including checksum and acceptance
reproduction from every archived capture, and the scalar calibration sweep.
Its final-source qualification is recorded in `pr-validation.json`.

`capture-source.patch.gz` reconstructs the fingerprinted last capture-source
variant on a clean `1482f6bb` checkout. The manifest distinguishes the early
B1 captures taken before its terminal-step guard. The published replay also
fixes terminal correction censoring and adds calibration/validation options;
the archived summaries reproduce with that current replay.

These are the checks relevant to the diagnostic change. They do not qualify
new weights, a sampling proposal change, or a different production kernel.
