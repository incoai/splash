# Reproducing the final review

These instructions refer to the frozen-source report and its separately published release assets in [assets.json](assets.json).

These instructions reproduce **frozen source A, before engine 1.6.1**: `37bdd4f8710319bfad43709de2725df89ef776d3` plus the selected patch, preserved as `fa62be81f2e9c48959f10a96ae3866a30c824075` and originally built as `src-1c82f6775b84007778a42bac408d53e5b095ac731ed27f479e126f7203fe0283`. They do not rebuild or validate the later integration with upstream `767c646ed91eb45a1e13a59525da7373123ac4cf` (engine 1.6.1). Its activation policy, memory accounting and warmups differ, so new source identities, receipts and measurements are required for that integration.

The [portable helper](scripts/reproduce_review_v1.py) provides exactly four actions: `check`, `build`, `run` and native-ABBA `analyze`. The examples below follow its current CLI. It does not provide turnkey relocated sampling, HTTP or prefix-statistics subcommands.

Set paths to a Splash checkout containing the base commit and a report directory containing `scripts/`, `patches/`, `cohorts/`, `notes/` and `results/`:

```sh
REVIEW_REPO=/path/to/splash
REVIEW_REPORT=/path/to/extracted-final-review
REVIEW_BUILD=/tmp/splash-review-combined-build
REVIEW_PY=python3
```

Check source assets and clean patch application without touching the checkout:

```sh
"$REVIEW_PY" "$REVIEW_REPORT/scripts/reproduce_review_v1.py" check \
  --repo "$REVIEW_REPO" --evidence "$REVIEW_REPORT"
```

The helper checks the pinned `37bdd4f8710319bfad43709de2725df89ef776d3` base, both patch hashes and study-driver source hash, using temporary Git indexes. It selects that historical base even if the checkout's current branch has advanced. This is a source check, not validation of every archive file or an execution test.

On a compatible Apple Silicon Mac with the required Xcode/Metal toolchain, rebuild an exclusive detached worktree and bundle:

```sh
MTL_DEBUG_LAYER=0 "$REVIEW_PY" \
  "$REVIEW_REPORT/scripts/reproduce_review_v1.py" build \
  --repo "$REVIEW_REPO" --evidence "$REVIEW_REPORT" \
  --variant combined-v2 --output "$REVIEW_BUILD" \
  --python /path/to/build-python --jobs 2 --cpu --metal
```

Variants are `baseline`, `qkv` and `combined-v2`; use a different new output directory for each. `--cpu` runs the CPU recipe. `--metal` builds and runs the targeted linear-plan and sixteen-row register checks, plus M48 for `combined-v2`. It does not run the complete all-format projection recipe. The original full projection check can be run separately under the repository's standard shader-validation setting:

```sh
MTL_DEBUG_LAYER=0 MTL_SHADER_VALIDATION=1 \
  "$REVIEW_BUILD/source/build/engine-tests/gguf-projection" \
  "$REVIEW_BUILD/source/build/engine-tests/production-and-test.metallib"
```

The extra API-plus-shader IQ4_XS prefill abort is retained and classified in [the paired diagnostic](results/prefill-iq4xs-validation-v1.analysis.json). Do not label the full suite as passing that stronger combination. Rebuild manifests record new artifact hashes; another SDK/compiler need not produce the original bytes.

Model weights and executables are not supplied by these evidence archives. Provide a resolved model root with the required target/draft checkpoints and metadata. For example, save one greedy functional DSpine16 B3 fixture:

```sh
REVIEW_DSPINE_ROOT=/path/to/resolved/dspine-model
"$REVIEW_PY" "$REVIEW_REPORT/scripts/reproduce_review_v1.py" run \
  --build "$REVIEW_BUILD" --model-root "$REVIEW_DSPINE_ROOT" \
  --cohort "$REVIEW_REPORT/cohorts/kernel-smoke3-cap256-v1.txt" \
  --rows 16 --width 3 --context 32768 --shader \
  --output /tmp/splash-review-functional.stdout
```

This creates exclusive stdout, stderr and receipt files, checks bundle/cohort hashes before and after, and uses one sample. It is not an ABBA timing comparison. DFlash2 uses its own resolved root and eight rows. New controlled measurements require a new nomination and all registered groups; the original receipts must not be reused for a rebuilt binary.

After extracting the new evidence archive, recompute saved native ABBA group gates without a GPU or model weights:

```sh
"$REVIEW_PY" "$REVIEW_REPORT/scripts/reproduce_review_v1.py" analyze \
  --evidence "$REVIEW_REPORT" \
  --protocol "$REVIEW_REPORT/notes/agent12-dflash8-vs-review16-v1.protocol.json" \
  --output /tmp/splash-review-agent12-reanalysis.json
```

The same command accepts the M48, default-eight and historical31 native protocols. It runs the unchanged SHA-pinned `analyze_abba_v2.py` and `abba.py` against saved round records, retaining failed gates. It does **not** recompute the supplemental cross-group/provenance reviewer or the manual HTTP audit. Original absolute paths remain provenance keys in the recorded commands and artifact maps; they are not paths to rewrite in frozen JSON.

The [new release](https://github.com/linson007/splash/releases/tag/dspine-final-review-20261111-v1) contains:

- `dspine-final-review-20261111-v1-evidence.tar.gz`: saved protocols, scripts, cohorts, receipts and report records.
- `dspine-final-review-20261111-v1-logits.tar.gz`: raw pre-policy logit captures.
- `dspine-final-review-20261111-v1-hardware.tar.gz`: retained hardware observations.
- `assets.json`: the three archive URLs, sizes and SHA-256 hashes, also uploaded as a manifest asset.

Download the assets and compare their sizes/hashes with `assets.json` before extracting. Keep new and old studies separate:

```sh
REVIEW_DOWNLOADS=/path/to/downloaded-final-review-assets
REVIEW_NEW_EVIDENCE=/tmp/splash-final-review-evidence
REVIEW_OLD_EVIDENCE=/tmp/splash-13h-evidence
REVIEW_OLD_REPORT="$REVIEW_REPO/dev/benchmarks/experiments/dspine_20261111"
mkdir -p "$REVIEW_NEW_EVIDENCE" "$REVIEW_OLD_EVIDENCE"
shasum -a 256 "$REVIEW_DOWNLOADS"/*.tar.gz
tar -xzf "$REVIEW_DOWNLOADS/dspine-final-review-20261111-v1-evidence.tar.gz" \
  -C "$REVIEW_NEW_EVIDENCE"
tar -xzf "$REVIEW_DOWNLOADS/dspine-final-review-20261111-v1-logits.tar.gz" \
  -C "$REVIEW_NEW_EVIDENCE"
tar -xzf "$REVIEW_DOWNLOADS/dspine-final-review-20261111-v1-hardware.tar.gz" \
  -C "$REVIEW_NEW_EVIDENCE"
tar -xzf "$REVIEW_OLD_REPORT/evidence/evidence.tar.gz" \
  -C "$REVIEW_OLD_EVIDENCE"
"$REVIEW_PY" "$REVIEW_OLD_REPORT/reproduce.py" verify \
  --evidence "$REVIEW_OLD_EVIDENCE" \
  --manifest "$REVIEW_OLD_REPORT/evidence/evidence-manifest.json"
```

The old report and its [thirteen-hour release](https://github.com/linson007/splash/releases/tag/dspine-13h-20261111-v1) document their separate optional logit/hardware assets. If downloaded, extract old logit data into the old evidence root. Extraction preserves the two source histories; it does not substitute the recovered historical screen for the selected candidate's new sampling family.

The raw sampling scripts are retained for inspection, but their registration, nomination, baseline and artifact provenance includes original study paths. A fully relocated statistical workflow across the two archives has not been verified by this helper. Extraction plus `--source-root` alone is not a verified reproduction procedure for the new screen. The canonical [source A comparison](results/candidate-sampling-v1.comparison.json) covers 4,000 new candidate requests and 4,000 reused recovered-control requests, with all 140 reserved tests evaluated, 98 informative and zero rejections. It is a finite regression screen, not an equivalence test or validation of the engine 1.6.1 integration. Use that comparison and its pinned source records until a separately validated relocation/reanalysis procedure is supplied.
