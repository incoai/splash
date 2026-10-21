# Greedy selector confidence calibration

**Versioned historical experiment:** runtime instructions and results below apply to their pinned legacy revisions. The runtime flags/build targets are not current-main settings. See [the October 21 current-main audit](current-main-audit.md) and [current qualification recipe](qwen-maintainer-runs.md). Replacement runs and final validation are still being completed.


`confidence_calibration.py` compares the existing cumulative selected-path
selector confidence with greedy target-prefix survival. It does not fit a
new predictor or change runtime policy. The feature differs from PR #487's
maximum-candidate-confidence feature and from DeepInfer's trained head.

Capture a complete B1 decode workload with `serving_regression.py --trace`
and the development `serving-profile` executable in both arms. Give the
reference a near-zero live threshold (`1e-9`) so the selector runs while all
eight rows remain eligible, and the candidate threshold 0.1. Use one resident
server at a time for the larger MLX model on this 48 GiB Mac. Then run:

```sh
python -m dev.benchmarks.confidence_calibration build/confidence-run \
  --output build/confidence-calibration.json
```

The analysis validates trace completion, maps request lifetimes to the exact
recorded B1 case sequence, checks prompt geometry and excludes both startup
and HTTP warmups. It rejects a reference containing truncated or censored
labels. Workload, context and prefix position stay separate. Each ten-point
confidence bin retains count, mean confidence, observed prefix survival and
Brier score. Censored suffixes remain in the totals but are excluded from
calibration; they are not failures. All traces have hashes and the report pins
tool/build/model identities and explicit environments.

These are descriptive, correlated labels: a deterministic repeated request
is not a new independent trial, and positions within one prefix are dependent.
The candidate's observed subset is selected by its own policy, so its bins
cannot establish unbiased full-suffix calibration. Sampled lanes do not enter
this greedy dataset. Seeded equality is not distributional validation.

The first GGUF capture on this M5 Pro has 4,676 reference labels with no
censoring and 4,676 candidate records, of which 22 are censored. The reference
mean confidence and actual seventh-proposal survival are:

| Context / workload | Labels | Mean cumulative confidence | Prefix survival |
| --- | ---: | ---: | ---: |
| 2k code | 144 | 0.312 | 0.111 |
| 2k prose | 176 | 0.195 | 0.034 |
| 8k code | 164 | 0.201 | 0.085 |
| 8k prose | 184 | 0.193 | 0.022 |

This is evidence against interpreting selector confidence directly as calibrated
acceptance probability. It does not predict the result of a cost-aware policy
using another feature or estimate. That decision needs held-out total-cycle
serving comparisons against the best fixed policy and full-width reference.
The MLX capture is being repeated with one resident server after dual residency
failed during model startup; the failed startup is not a calibration result.
