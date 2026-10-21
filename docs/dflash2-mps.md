# Experimental DFlash2 training on Apple Silicon

This builds on [incoai/splash#499](https://github.com/incoai/splash/pull/499).
The Mac branch adds explicit `--device mps` support to `train`, `capture`,
`evaluate`, and the source `splash train-draft` launcher. CUDA remains the
default. Training uses PyTorch MPS, BF16 forward computation, and FP32
optimizer masters. The BF16 export format and training objective are inherited
from the parent PR.

Validation transfers FP32 masters to CPU and rounds them there using the same
conversion as export, then scores temporary BF16 parameters on MPS. It restores
the original optimizer parameter objects even if evaluation fails. This adds
validation transfer overhead while keeping checkpoint selection tied to the
published weights.

The small tests exercise the actual Apple GPU, including teacher generation,
feature capture, selector and full-draft optimization, optimizer/RNG resume,
BF16 export, and final evaluation. Production Qwen3.8-27B memory, throughput,
training quality, and trained-export native qualification still need testing.
Keep this change draft until that evidence is available.

## Run the smoke tests

Use an Apple Silicon Mac and an MPS-enabled PyTorch installation. Dependencies
are the same pinned optional packages as the parent. Run from the checkout root:

```bash
python3.12 -m venv /path/to/training-env
. /path/to/training-env/bin/activate
python -m pip install -r training/requirements.txt

DFLASH_REQUIRE_MPS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  python -m unittest dev.tests.training.test_mps -v
```

These tests create tiny random checkpoints locally and download no model weights.
`DFLASH_REQUIRE_MPS=1` makes an unavailable Apple GPU fail the run. Ordinary test
discovery skips the hardware cases on machines without MPS. The default smoke
command disables explicit CPU fallback so unsupported operators surface.

Run the parent training/reference suite as well:

```bash
DFLASH_REQUIRE_MPS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  python -m unittest discover -s dev/tests/training -t . -v
```

The upstream numerical parity test separately requires `DFLASH_REFERENCE` pointing
to the pinned reference described in the parent guide. CPU-versus-MPS bitwise
equality is not claimed. The MPS tests compare uninterrupted and resumed runs on
the same backend and check that BF16 export replay matches validation.

An optional `Experimental DFlash2 MPS smoke` workflow can run these tests on a
trusted self-hosted Apple Silicon runner. Dispatch it on this branch and supply
the runner's label (default `splash-mps-training`) after the workflow is present
on the repository's default branch, as required by
[GitHub manual dispatch](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).
While the workflow is only in this PR, use the local commands above. The job
requires MPS, disables explicit CPU fallback, uses an existing runner, and
downloads no real-model weights. It is manual and does not execute pull-request
code automatically on a self-hosted machine.

## Check memory before a real run

The current command captures features and trains on the selected device. It
loads the full unquantized BF16 teacher before releasing its decoder and vision
weights. Existing feature caches and checkpoints remain bound to their original
device context; cross-device cached-feature training needs a separate workflow.

For a 27B teacher, weights alone are approximately 54 GB (50.3 GiB), before
working memory, sequence activations, and macOS. A 48 GB Mac cannot keep those
weights resident alongside the required working memory. Larger-memory Macs
still need a measured preflight. Quantized serving memory is not a measurement
of this BF16 training path.

Start with the parent guide's disjoint training and validation JSONL files and
local BF16 checkpoints. Inspect metadata first:

```bash
python -m draft_training train \
  --target /models/target-transformers-bf16 \
  --init-draft /models/released-dflash2 \
  --data train.jsonl --validation-data validation.jsonl \
  --config training/qwen3_8_27b.yaml \
  --output runs/mac-draft --device mps --dry-run
```

Replace `--dry-run` with `--preflight` to generate one continuation and execute
an optimizer probe without writing a run. Use an example with enough assistant
tokens for a full query block. For MPS, the result includes synchronized
`capture_memory` and `training_memory` snapshots containing tensor allocation,
driver allocation, and recommended working-set bytes. These are end-of-phase
snapshots, **not peak memory measurements**. Monitor system memory and swap
separately during real-checkpoint testing.

After preflight succeeds, remove the probe flag to train. The default recipe
trains only the selector. Full-draft optimization needs additional optimizer
and activation memory. Resume with the same arguments plus `--resume`.
Checkpoints store CPU and MPS RNG states; changing the backend, macOS version,
Apple GPU name, or explicit CPU-fallback setting rejects resume and final
evaluation. Use a new run directory when changing the execution context.

If an operation is unsupported, report its error and the exact versions first.
An independently measured experiment can set `PYTORCH_ENABLE_MPS_FALLBACK=1`
before starting Python. This setting is recorded in run provenance; it can
change performance and must stay fixed throughout that run.

Evaluate the selected export on a third untouched split using the parent
`evaluate` command with `--device mps`. Both the released baseline and trained
draft are scored against the same frozen teacher traces. Native Splash loading,
acceptance, throughput, and latency remain separate qualification steps.

## Testing help wanted

Contributors with sufficient unified memory can help qualify the real BF16
target and released draft. Report results on the Mac PR with:

- Commit, Apple chip, unified memory, macOS, Python, PyTorch, and Transformers
  versions, plus the explicit CPU-fallback setting.
- Exact target and draft revisions, recipe, and training/validation/test split
  fingerprints from `run.json` and the selection manifest.
- Smoke-test results, preflight output, observed system memory/swap, step timing,
  and any unsupported operation or nonfinite loss/gradient.
- Interrupted-versus-resumed training results and successful BF16 export.
- Frozen test replay and native Splash loading and benchmarks against the
  released draft with identical deployment settings and alternating runs.

Use shareable prompts when reporting examples. Tiny-model success establishes
backend integration; the real-model results above establish practical capacity
and deployment behavior.
