# Qwen3.8 DFlash2 deployment evidence

See the [results and decisions](../../qwen38-rejection-audit.md). These files document local experiments on the released five-layer model and one rejected selector adaptation. No weights or production runtime changes are shipped.

## Portable verification

From this directory, run:

```sh
python3 verify_bundle.py
```

It requires only the Python standard library. It checks portable/source/prompt hashes, reconstructs first-miss and conditional-acceptance counts from all 192 baseline cases, recomputes both native selector comparisons, and validates native greedy/confidence replay on eight full representative captures. `replay_reference.py` is an exact copy of the reviewed development replay implementation from PR #550, extended by PR #556; it is isolated here for artifact verification.

- `protocol.json`, `audit-final-summary.json`: all 32 task families, 192 cases, per-case/stratum first-rejection and acceptance summaries.
- `compact-traces.jsonl.gz`: 288 case records / 9216 cycles across baseline, selector validation/test and restored-context pairs. It retains candidates, proposed/target/output tokens, retained counts, context hashes and confidence. Unary/edge matrices and hidden sidecars are omitted. Only reached-prefix/first-rejection target labels have a correctness guarantee.
- `representative-captures.tar.gz`: eight exact original captures, preselected as family index zero, 2K context, both thinking modes, all four domains. These include full candidate scores for independent native greedy/confidence replay.
- `selector-*.json`, `training-plan.json`, `frozen-dataset.json`, `final-test-lock.json`: frozen experiment, all four fits, selected tensor hash, native validation and held-out failure, plus the descriptive output-parity sensitivity check.
- `validation-reference-audit.json`: post-decision proxy-tail correction. Only the released and already-locked checkpoint are rescored; no refit/reselection occurs.
- `policy-summary.json`, `restore-summary.json`, `restore-output-parity.json`: matched-prefix distribution references and materialized/restore results, with scope limitations.
- `forensic-analysis.json`: exact-rollout reproduction, close target-logit argmaxes and fixed-schedule teacher-logit equality for the sole divergent selector continuation.
- `m16-{protocol,measurements,summary}.json`: complete representative original-checkpoint ABBA requests/responses/counters, serving identities, status memory plans and OS boundary observations. Results do not qualify the M16 change.
- `*-build.json`, `*-provenance.json`, controls, quantization and feature-math checks: source identities, model/export controls and numerical checks.
- `experimental-sources.tar.gz`: exact one-off drivers, native source overrides, read-only accessor patch, production-base patch, original prompt protocol, all exact prompt text/token files, references/hashes for historical diagnostic-test fixtures and the existing quantizer reference. It preserves the original measured proxy and the later corrective audit separately. Historical fixtures already published with PRs #550/#556 are not duplicated here; the portable verifier uses the fresh captures in this bundle.
- `manifest.json`: portable/source hashes and hashes/sizes for local raw captures, hidden features, logits, candidate tensors and binaries that are not shipped. Installed source model files are checked against their original SHA256/Git-blob digests and recorded unchanged.

Portable JSON paths use `${SPLASH_ROOT}`, `${AUDIT_WORKTREE}` and `${USER_HOME}`. Representative captures retain exact original metadata bytes. Generated text belongs to authored fictional tasks; it is not a quality benchmark. Full activations, full-vocabulary logits, native binaries, virtual environments and private model assemblies remain ignored under `build/qwen38-rejection-audit/`.

## Native reconstruction

The archive is a record of exploratory work, not a supported cross-platform CLI. Original drivers contain measured workspace paths and expect repository build outputs. Use a separate workspace, adapt paths consistently and use fresh output names. The final-test families in this bundle have been exposed and must not become a tuning set.

1. Start at upstream `3135ab45` and apply `production-base.patch` to reconstruct the measured runtime/library ABI (`950ffef9`). Build the native engine, instrumented backend and teacher-logit tool with the repository's `make` targets. Retain the pinned target/draft revisions in the report; no download is included in these drivers.
2. Create the isolated audit worktree at that combined base. Overlay `audit-source/`; source accessors are guarded by `SPLASH_RUNTIME_INSTRUMENTATION`. Place drivers under `build/qwen38-rejection-audit/`, exact prompts under its `prompts/`, and reconstruct protocol output paths in a fresh directory. Source hashes are recorded per native tool. Compile using `prepare_native.py`, `build_batch.py`, `build_policy_probe.py`, and `build_forensic.py`, preserving matching engine objects and Metal libraries. `prepare_native.py` originally imports accessor definitions from the existing PR #556 worktree; either reconstruct it or use the archived accessor patch/overrides directly. Assert production preprocessing and single/batch controls before new captures.
3. `run_audit.py` captures native features with 32 cycles/case. `audit_analysis.py` excludes final test by default. `frozen_selector.py dataset` builds reached-prefix examples; the `train` phase uses isolated PyTorch 2.13.0/MPS, NumPy and the archived `draft_training.quantization` module. Main repository dependencies are left unchanged. The original training call uses `PYTORCH_ENABLE_MPS_FALLBACK=0` and `PYTORCH_MPS_FAST_MATH=0`. Before any new selection, correct `val_cycles()` to construct complete references solely from emitted tokens; `check_validation_reference.py` documents the required mask and only rescored the locked model in this run.
4. `export_selector.py` creates a private copy, changes only the specified BF16 tensor and verifies all other bytes. Never overwrite an installed model. Run the unchanged-export/native control, then validation. Freeze selection before testing; the existing `run_followup.py`/`run_final_phases.py` record this run's orchestration, not an invitation to tune on its exposed final set.
5. `analyze_policy.py`, `analyze_restore.py` and `analyze_forensics.py` explain the separate policy, restore and divergence probes. They require large local raw sidecars listed in the manifest; those cannot be reconstructed from compact traces alone. Policy overlap uses a FP64 reference, not exact native sampling math.
6. `qualify_m16.py` runs four fresh HTTP servers A/B/B/A with the original checkpoint. It expects the original baseline and M16-only build directories, the repository Python environment and `PYTHONPATH` pointing at the repository. Its predeclared no-promotion result and raw drift are retained. Disable capture during serving timings; serialize GPU work.

The diagnostic module tests (26) passed in the isolated worktree and with the archived source files in a temporary directory. Repeating their two historical-fixture tests requires the original fixture directories from PRs #550/#556, whose exact file hashes and source commits are listed in the archive. The portable verifier additionally checks the published reduction of the fresh raw evidence without those historical fixtures. Original production correctness/oracle checks are documented with the earlier runtime PRs; no modified selector or architecture has passed a production promotion gate in this round.
