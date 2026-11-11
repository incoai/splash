# Selector architecture prototypes (#546)

Depends on #499/#537. `ExperimentalSelector` supplies equal-support bilinear,
explicit previous-hidden pair context, and candidate-conditioned nonlinear
correction. Pair and nonlinear additions initialize to zero, reproducing the
released selector before adaptation. Rank/support/codebooks and the greedy
walk stay fixed. Added parameters are H*R for pair context or (2R+1)*bottleneck
for the nonlinear variant; measure actual compute, not only parameter counts.

Use independently split, cached target features through the existing model:

```python
import torch
from draft_training.selector_experiments import ExperimentalSelector
candidate = ExperimentalSelector(model.candidate_selector, variant="pair_context").float()
optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-5)
# tokens/features/embedding/head are on the same selected device.
device_type = tokens.device.type
with torch.no_grad(), torch.autocast(device_type, dtype=torch.bfloat16):
    block = model.block_hidden(tokens, features, anchor, embedding, include_anchor=True)
    logits = head(block[:, 1:])
optimizer.zero_grad()
with torch.autocast(device_type, dtype=torch.bfloat16):
    loss, metrics = candidate.loss(block, logits, labels, anchor_ids)
loss.backward()
optimizer.step()
```

`labels` is [batch,7] correct-continuation tokens, `anchor_ids` [batch], and
`block` [batch,8,H], including each block's own contextual anchor hidden row.
Each block remains independent. Neither a lane boundary nor a dropped anchor
may be inferred with a flattened shift. No future target features enter
`block_hidden`; candidates are the unmodified natural unary top-k.

`save_research`/`load_research` use an explicit capability/variant/geometry and
baseline identity. They emit no serving `config.json`, so these prototypes
cannot silently load as the old native selector. They are not wired into the
serving-export trainer. Do not rename the research safetensors into a serving
checkpoint. Optimizer/resume orchestration for architecture trials and a native
versioned checkpoint/kernel/loader implementation remain follow-ups after a
trained candidate qualifies. QAT from #543 must be combined and tested
separately; BF16 smoke tests alone do not establish native quantized parity.

Tests cover baseline equivalence, anchor/lane gradients, natural support
misses, strict capability round-trips, BF16 forward/backward, and real CPU/MPS
optimization for all three variants. Loss reduction on tiny fixtures does not
establish acceptance, added latency, or an advantage over adapted bilinear.

Local verification (Apple M5 Pro, macOS 26.6, Python 3.12.14, PyTorch 2.13.0,
Transformers 5.15.1): `55` training tests passed, including the pinned upstream
reference comparison; none skipped. Actual MPS was required and CPU fallback disabled:

```sh
DFLASH_REFERENCE=/path/to/pinned/dflash/model.py \
DFLASH_REQUIRE_MPS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  python -m unittest discover -s dev/tests/training -t . -v
```
