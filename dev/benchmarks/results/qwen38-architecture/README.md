# Qwen3.8-27B local architecture experiment evidence

See the [analysis and decisions](../../qwen38-architecture.md). These are private experimental variants, not supported runtime flags or production model exports.

- `summary.json`: all structural screening, paired draft confirmation, same-prefix target sensitivity, low-rank calibration, and fixed-input timing summaries.
- `measurements.jsonl.gz`: original measurement and provenance documents, one `{source,data}` record per document. Absolute workspace/home paths are replaced by `${SPLASH_ROOT}`, `${ARCH_WORKTREE}`, and `${USER_HOME}`. Serving responses include the authored prompts' generated outputs and native metrics.
- `manifest.json`: source, binary, corpus, full-logit, BF16 capture, calibrated tensor, and portable-file hashes; baseline parity and CPU/native surrogate-math validation.
- `experimental-sources.tar.gz`: production-base patch, exact v1/v2 experimental source overrides and build identities, original one-off drivers/analyzers, and exact token corpora. No original or calibrated model weights, large captured activations, or native binaries are included.

The raw local artifacts are under `build/qwen38-architecture/` (about 7.5 GB); the private source worktree is `/private/tmp/splash-qwen38-architecture`, branch `experiment/qwen38-architecture`. Existing production source changes belong to the earlier M5 investigation and are not altered by this experiment. Installed model files are read only.

## Scope and qualification

The experiments cover 11 draft variants, 6 approximate target FFN-deletion variants, and 2 calibrated target surrogates. There are 31 fresh serving processes and 124 measured HTTP requests, 19200 emitted tokens, excluding warmups. Additional teacher-forced experiments check full-vocabulary distributions and fixed-input cycle cost. Token/sec screens of approximate targets generate different answers. They are not equivalent-quality speedup claims. No architecture variant is qualified for production.

Private v1 source identity is `9c24d771d2f11ce4fa778e74d54cba4dccba572a76b9213cfd40bd3e36e93426`. V2, which adds calibration captures and rank-64 residual predictors, is `e5b2efa4e6dfda0d2959a0faceb15bb16aa281dcd33d46c59aaf89267e655895`. The separate fixed-prefix driver source/binary hashes are in the measurement archive. Runtime serving identity additionally incorporates architecture settings and surrogate tensor content.

## Reconstruction

Use a separate clone/workspace so that original measurements remain intact. This is a source archive of exploratory work, not a portable product command. Drivers have the measured workspace's relative output layout and a private worktree path at their top; adapt those paths consistently when using another location.

1. Start with upstream `3135ab45` and apply `production-base.patch`. This reconstructs the runtime/library ABI used for the experiments, including the previous M16/short-prefill changes. The original local integration commit was `950ffef9`; its runtime is byte-identical to the measured production checkout. Build with `make -j6 teacher-logits` to obtain the matching engine library, instrumented runtime, and Metal libraries. Python requires the repository environment plus NumPy/Accelerate and the existing model/tokenizer dependencies.
2. Supply the pinned model roots recorded in the manifest. No model download is part of these scripts. Recreate the private architecture worktree from that same production base and overlay `v1/` or `v2/` source files. Place the matching `BuildIdentity.hpp` and `architecture.metal` in `build/qwen38-architecture/`. Do not combine new headers with an old engine library.
3. Restore the archived scripts to `build/qwen38-architecture/` and the token corpora to their original relative paths. `numerical-probe.u32` belongs under `build/qwen38-investigation/`; the other `.u32` files belong under `build/qwen38-architecture/`. Measurement JSON records can be unpacked to those paths, replacing path placeholders with the new roots. The original generator is preserved for audit; exact archived source overrides are the authoritative source for each version.
4. `build_experiments.py` compiles v1 or v2 against that matching production library. Preserve v1 executables/libraries under `build/qwen38-architecture/v1/` before building v2. The no-op full-logit parity controls should pass before any timed run. `collect_surrogate.py` gathers 512 rows from each calibration prefix; `fit_surrogate.py` recreates the rank-64 tensor files using only those calibration prefixes. `verify_surrogate_math.py` checks the actual native FFN output against its CPU mathematical reference.
5. Use `run_screen.py` for the original draft screen, `run_quality.py` and `run_surrogate_quality.py` for held-out same-prefix distributions, and `run_compare.py` with each archived protocol for serving. `build_fixed_prefix.py` / `run_fixed_prefix.py` reproduce the fixed-input ABBA cost experiment. The comparison driver skips completed outputs; use fresh output names/workspaces for a new experiment.
6. Re-run `analyze_all_quality.py`, `analyze_serving.py`, and `analyze_fixed_prefix.py`. Keep one GPU experiment active at a time, use AC power and the original protocol flags, and retain the control drift instead of silently normalizing it away.

The shared serving arguments used in this round were:

```sh
--model mlx-community/Qwen3.8-27B-4bit \
--model-root install/models/mlx-community/Qwen3.8-27B-4bit \
--max-context 32768 --max-memory 34G --max-cache-disk 0 \
--disable-ane --state-format fp32
```

Draft screening and repeated draft confirmation use v1; the surrogate screen and fixed-input cost experiment use v2. The target-deletion serving screen used v1. INT8 KV is the default in these protocols and is asserted. Full teacher-forced sensitivity runs explicitly request INT8 KV/FP32 state. Captures must be disabled for timing.

The surrogate target keeps original weights resident and adds about 10.5 MB of predictor matrices; its status memory plan does not separately account for that allocation. A real compressed-model export and loader would be required to demonstrate capacity savings. No result here establishes broad task quality, long-context behavior beyond the tested windows, or multi-lane throughput.
