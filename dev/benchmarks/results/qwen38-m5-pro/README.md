# Qwen3.8-27B / M5 Pro investigation evidence

Read the [report](../../qwen38-m5-pro.md) for the source audit, cost model,
priorities, decisions, gains and limitations. This directory contains the
bounded local investigation's evidence, not a portable speed guarantee.

- `summary.json`: comparisons, numerical sensitivity, research decisions,
  issue statuses and final validation.
- `manifest.json`: pinned model revisions, baseline commit, hardware,
  compiler, source/measurement hashes, and retained executable/logit hashes.
- `measurements.jsonl.gz`: one JSON record per raw text file, with `path`
  and `text`. Text paths are normalized to `$REPO` or named historical
  checkout placeholders; the manifest preserves both original and normalized
  SHA256s. Timing, output hashes and counter values are unmodified.
- `exploration-sources.tar.gz`: diagnostic drivers, generated prototypes,
  reconstruction patches and the new CLI validation test. Members preserve
  paths relative to the repository. Regenerate prototype sources with their
  makers when rebuilding in another directory.

Checkpoint weights, executables, Metal libraries and full vocabulary logits
are excluded. They remain in the original machine's ignored
`build/qwen38-investigation/`. The manifest records their hashes where useful.
The original source tensors and model install were read without modification.

## Reproduction

Use the pinned target/draft revisions from the manifest on the same GPU/core
count, compiler and OS to reproduce the local scope. Build and retain a
pristine `3135ab45` baseline in an isolated checkout. Retain its executable,
`splash.metallib`, `host-step.metallib`, backend/projection benchmarks and
`weight-digests`. Build tools against that checkout's private headers and
library: `ProjectionShape` changed layout in the candidate.

The archived `m16-wide.patch` reconstructs the expanded M16-only control;
`current-runtime.patch` reconstructs the final combined candidate from the
base. The CLI test is a separate archive member. Keep each retained binary
beside its own libraries, and use the source hashes to distinguish the
frozen experiments from the final build.

The report provides the main commands. Drivers are archived at
`build/qwen38-investigation/`, including:

- `run_native_abba.py`, `run_native_wide_abba.py`,
  `run_transcript_abba.py`, and `run_short_http_abba.py`;
- `run_capacity_attention.py`, `run_long_logits.py`,
  `run_precision_domains.py`, and the logit analysis scripts;
- `make_k64_variant.py`, `make_prefill_variant.py`,
  `make_attention_partition_variant.py`, `run_attention_partitions.py`;
- `make_draft_precision_variants.py` and `analyze_acceptance.py`.

Run local Python drivers from the repository with its virtual environment
and `PYTHONPATH=.`. The attention override and BF16-draft loaders are isolated
development builds. The draft variants' explicit experimental build identity
algorithm is documented in the manifest; it is not the normal repository
source-identity algorithm. Neither prototype changes production policy.

The acceptance analysis reads a separate, older capture checkout. Those
October 22 records are historical diagnostic probes of the pinned weights,
not current-head timing measurements. Raw captures are not duplicated here;
verification results, reached-prefix denominators and source provenance are
retained. Existing upstream research links are in the report and upstream issues542–548.

Connect AC power and run one GPU performance job at a time. Each paired
comparison uses the repository's ABBA rule: round medians, a 2% floor,
twice measured same-build spread, and an inconclusive result above 5% spread.
A neutral regression pass does not establish a gain. Keep actual output and
speculative counts rather than normalizing by a requested token budget.

The repeated decode counting cells show 2.32–2.45% throughput gains. Both
aggregate HTTP runs remain inconclusive because cold 2K TTFT spread exceeds
5%. Representative transcript timing did not establish a general coding
speedup. Precision and draft screens use few authored prompts and do not
establish held-out quality parity or trained-model improvement.

## Upstream publication

`publication.json` maps the original ten workstreams to upstream tracking.
The `issues` array in `summary.json` preserves historical fork states.
Measurements, original source patches, and raw-file checksums are unchanged;
publication branch validation is recorded separately from the measured
combined patch.
