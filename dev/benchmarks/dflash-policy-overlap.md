# Matched-context proposal policy reference (#545)

Run `python -m dev.benchmarks.dflash_policy_overlap capture.json --output report.json`.
The input is a `dflash-policy-overlap-v1` object with `distribution_source`
(`native_capture` or `test_fixture`), target/draft/tokenizer identity strings,
and `records`. See `dev/tests/tools/test_dflash_policy_overlap.py:fixture` for
a complete small schema example.

Each record supplies the **full**, normalized deployed target distribution
AFTER filtering, prompt IDs, output IDs before this block, the correct
within-block prefix, candidate IDs and already combined unary/edge scores at
that correct predecessor, proposal temperature, and repetition/presence/frequency
values. Both target/proposal context SHA256 fields must match the canonical
hash of those three token lists. A hash is an alignment check, not proof that
a capture producer actually evaluated the right model/context.

The report compares existing, history-only, and history-plus-prefix penalties
on the SAME target p and frozen support. Prompt occurrences affect repetition;
presence/frequency count only output. Signed penalties, greedy ties and p tail
mass are covered. It reports overlap, support mass and q entropy per position.
It never conditions/renormalizes target p onto the candidate set. The exact
one-step rejection reference uses the actual q, including zero support, and
verifies that accept-plus-residual correction recovers the complete target p.

This is a CPU numerical reference, not a native FP32/Metal parity claim.
Top-k/top-p/min-p, stop/constraint/thinking policies must already be reflected
in input p. No candidate retrieval policy is changed. Changing within-block
predecessors requires new edge evaluation; this tool scores fixed correct
prefixes, not independent modified rollouts. PR #550's greedy token capture
does NOT contain these sampled distributions. A native distribution capture
adapter, state/mixed-lane qualification and paired serving runs remain gates.
Fixture tests are not evidence of improved real-model overlap or throughput.
