# DFlash experiment integration

The changes are optional training/research infrastructure. Merge readiness
covers their implementation and reproducible tests; trained-checkpoint
acceptance, native loading and paired serving performance remain separate
qualification gates. No experimental architecture or freshly trained export
is selected as a serving default.

The training PRs share configuration and trainer entry points. Their branches
now include their prerequisites and resolve those shared edits in this order:

1. #499: training command and BF16 adapter/export contracts.
2. #537: optional MPS backend.
3. #551: AF4G64 projection training.
4. #552: prefix-survival and position-weight objectives.
5. #553: training-only contextual selector research artifacts.
6. #554: selective recall scopes and reached-prefix metrics.

#550 precedes #556 on the independent native-diagnostics chain. #555 is
independent. PRs target main; their descriptions link the isolated change
against the immediate prerequisite. Merge commits preserve the tested
ancestry. If prerequisites are squash-merged or rebased upstream, refresh the
remaining dependent branches against main before merging them; do not count
an old head's green checks as verification of a changed merge result.

A local integration branch merged all nine PRs in this order without conflicts.
The combined optional training suite passes with the pinned DFlash reference,
actual MPS required and CPU fallback disabled. The tests include full-projection
QAT plus prefix/position objectives and recall adaptation, finite gradients,
frozen-selector weights, recipe loading and exact resumed BF16 exports.

```sh
DFLASH_REFERENCE=/path/to/dflash/dflash/model.py \
DFLASH_REQUIRE_MPS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  python -m unittest discover -s dev/tests/training -t . -v
```

The richer selector modules remain research-only: their artifacts have a
capability marker and are not native serving checkpoints. They need their
own matched training and versioned native implementation before deployment.
The native context probe's combined Metal API+shader validation assertion was
also reproduced on its unchanged baseline; shader-validation-only capture and
the ordinary native matrix passed. That pre-existing validation-mode limitation
is recorded in #556's artifacts, not described as a passing check.

Real-checkpoint CUDA qualification on one A100 80GB is recorded in the
[Qwen3.8 smoke results](experiments/qwen38-a100-smoke/README.md). Four recipes
completed 20 steps each, and a separate full-draft optimizer preflight passed.
Validation acceptance stayed unchanged on the tiny smoke split; implementation
success does not establish a trained-model or serving improvement.
