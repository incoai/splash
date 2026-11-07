# Qwen3.8 inference research

The [archived architecture and DFlash2 rejection studies](https://github.com/linson007/splash/releases/tag/qwen38-research-audit-20261107)
retain the original reports, measurements, experimental sources, token corpora
and offline verification tools outside the production source tree. The archive
preserves all 51 files from research commit
`7a34e311481c314e881e236ac2dffb9a9c78685b`, with a SHA-256 manifest.

These studies identify opportunities and record unsuccessful approaches. They
do not qualify a change to the current runtime:

- Architecture ablations lost draft acceptance or changed target outputs. No
  architecture variant qualified for production.
- In 192 cases and 6,144 greedy cycles, 84.89% of first rejections had the
  target token in the draft's top-16 support. A small selector pilot still
  lost 1.17% tokens per cycle on held-out task families.
- The historical matrix-tile serving comparison was inconclusive because
  timing drift exceeded the observed differences.

The reports distinguish same-prefix diagnostics from new native rollouts and
preserve negative results, output differences, timing drift and a corrected
proxy-tail audit. Model and draft revisions, source identities and the limits
of reconstructing the historical experiments are recorded in the archive.

To check the evidence, download the archive and its checksum file from the
release, verify the SHA-256 checksum, extract it, and run:

```sh
python3 -B qwen38-research-20261107/verify.py
```

This needs only Python's standard library. It verifies all original file
hashes and recomputes the rejection counts, selector comparisons and eight
representative native greedy replays without loading a model or using a GPU.
For validation of changes to today's runtime, use [Validate](validate.md).
