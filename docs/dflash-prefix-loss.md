# Prefix-loss experiment (#544)

Depends on #499 and #537. Existing recipe defaults remain `prefix_loss_weight: 0`
and `position_decay: 1`. A treatment can set `prefix_loss_weight: 0.25` and
`position_decay: 0.9` through the existing training CLI. Sweep each factor
separately against equal-token continuation before combining them.

The auxiliary loss is minus the mean cumulative product of correct-token
selector probabilities under teacher-forced correct predecessors. A natural
candidate miss has probability zero; the unary CE still trains recall. The
position decay weights unary and selector CE, normalized over known labels.
This is a smooth greedy-prefix **surrogate**, not exact sampled acceptance or
an unbiased rollout gradient. Teacher-forced survival is logged separately.
No gold token is inserted into candidate retrieval. Unknown/padded labels and
transitions with unknown predecessors are masked; censoring stops survival.

The unchanged trainer's conversation split, feature isolation, optimizer
checkpoint/RNG and validation-only model selection remain in force. Tests
include exhaustive prefix enumeration, numerical gradients, candidate misses,
censoring, sentinel padding, chunking, and actual auxiliary gradients. The
optional training suite also checks deterministic CPU/MPS resume and exports.

Verification-error mining/mixing is still a separate experiment: do not label
a verifier suffix behind an incorrect proposal as a correct continuation.
This PR supplies the objective factor, not a mined replay corpus. Native
exported-checkpoint held-out acceptance and throughput remain unmeasured.

Local verification (Apple M5 Pro, macOS 26.6, Python 3.12.14, PyTorch 2.13.0,
Transformers 5.15.1): `56` training tests passed, including the pinned upstream
reference comparison; none skipped. Actual MPS was required and CPU fallback disabled:

```sh
DFLASH_REFERENCE=/path/to/pinned/dflash/model.py \
DFLASH_REQUIRE_MPS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  python -m unittest discover -s dev/tests/training -t . -v
```
