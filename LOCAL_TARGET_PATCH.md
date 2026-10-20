# Local target directories

Use `splash serve --model-dir /path/to/model` for a local target, or
`--model OWNER/REPO[:VARIANT]` for a Hub target. These options are mutually
exclusive. Local paths are resolved before use and cannot take `--revision`.

Local targets are inspected on every start and linked into a hashed selection
under the models root. The existing source directory is never modified.
The matching DFlash2 draft remains required: use `--draft-model /path/to/draft`
for a local draft or allow the normal draft download.

Regression coverage: CLI exclusivity, unchanged Hub-ID validation, missing
paths, spaces, selection containment, symlink normalization, revision
rejection, reinspection on restart, no local target downloads, and unchanged
source bytes. Existing installer, launcher and serve-option tests also run.

Validation: full native rebuild with `make -B -j4 all`, installer and launcher
regression suites using the rebuilt engine, and Ruff lint/format checks.
No native source changes. A prior live HTTP inference test used the patched
Python code with the released 1.3.2 engine and local target and draft weights.
