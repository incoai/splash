"""Training-only selector candidates. Artifacts are NOT native checkpoints."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .config import TrainingError
from .model import top_candidates


class ExperimentalSelector(nn.Module):
    def __init__(self, baseline, *, variant="bilinear", bottleneck=8):
        super().__init__()
        if variant not in ("bilinear", "pair_context", "nonlinear"):
            raise TrainingError("unsupported experimental selector variant")
        if type(bottleneck) is not int or bottleneck <= 0:
            raise TrainingError("selector bottleneck must be positive")
        self.base, self.variant = baseline, variant
        projection = baseline.hidden_projection
        hidden, rank = projection.in_features, projection.out_features
        self.geometry = dict(
            hidden=hidden,
            rank=rank,
            top_k=baseline.top_k,
            vocabulary=baseline.predecessor_codebook.shape[0],
            bottleneck=bottleneck,
        )
        options = dict(device=projection.weight.device, dtype=projection.weight.dtype)
        if variant == "pair_context":
            self.previous_projection = nn.Linear(hidden, rank, bias=False, **options)
            nn.init.zeros_(self.previous_projection.weight)
        if variant == "nonlinear":
            self.correction_input = nn.Linear(
                2 * rank, bottleneck, bias=False, **options
            )
            self.correction_output = nn.Linear(bottleneck, 1, bias=False, **options)
            nn.init.zeros_(self.correction_output.weight)

    def row_scores(self, current, previous, unary, candidates, predecessors):
        gate = self.base.hidden_projection(current)
        if self.variant == "pair_context":
            gate = (gate.float() + self.previous_projection(previous).float()).to(
                gate.dtype
            )
        before = (
            F.embedding(predecessors, self.base.predecessor_codebook)
            .to(gate.dtype)
            .float()
        )
        after = (
            F.embedding(candidates, self.base.successor_codebook).to(gate.dtype).float()
        )
        context = before * gate.float()
        scores = unary.float() + (context.unsqueeze(-2) * after).sum(-1)
        if self.variant == "nonlinear":
            pair = torch.cat(
                (context.unsqueeze(-2).expand_as(after), after), dim=-1
            ).to(gate.dtype)
            delta = self.correction_output(
                torch.tanh(self.correction_input(pair))
            ).squeeze(-1)
            scores = scores + delta.float()
        return scores

    def scores(self, block_hidden, unary, candidates, predecessors):
        if block_hidden.ndim != 3 or block_hidden.shape[1] != unary.shape[1] + 1:
            raise TrainingError(
                "supply each complete block including its own anchor hidden row"
            )
        # Explicit slicing retains batch/lane boundaries; never torch.roll.
        return self.row_scores(
            block_hidden[:, 1:], block_hidden[:, :-1], unary, candidates, predecessors
        )

    def loss(self, block_hidden, logits, labels, anchor):
        unary, candidates = top_candidates(logits.detach(), self.base.top_k)
        predecessors = torch.cat((anchor[:, None], labels[:, :-1]), dim=1)
        scores = self.scores(block_hidden.detach(), unary, candidates, predecessors)
        matches = candidates == labels.unsqueeze(-1)
        valid = matches.any(-1)
        gold = matches.long().argmax(-1)
        # Natural support only. Zero covered rows must give zero gradients,
        # not NaN and not an injected gold candidate.
        loss = scores.sum() * 0
        if valid.any():
            loss = (
                F.cross_entropy(scores[valid], gold[valid], reduction="sum")
                / labels.numel()
            )
        return loss, {
            "tokens": labels.numel(),
            "covered": int(valid.sum()),
            "correct_when_covered": int(((scores.argmax(-1) == gold) & valid).sum()),
        }

    def path(self, block_hidden, logits, anchor):
        unary, candidates = top_candidates(logits, self.base.top_k)
        predecessor, path = anchor, []
        for row in range(logits.shape[1]):
            scores = self.row_scores(
                block_hidden[:, row + 1],
                block_hidden[:, row],
                unary[:, row],
                candidates[:, row],
                predecessor,
            )
            predecessor = (
                candidates[:, row]
                .gather(-1, scores.argmax(-1, keepdim=True))
                .squeeze(-1)
            )
            path.append(predecessor)
        return torch.stack(path, dim=1)

    def save_research(self, directory, baseline_identity):
        from safetensors.torch import save_file

        if not isinstance(baseline_identity, str) or not baseline_identity:
            raise TrainingError("record the baseline checkpoint identity")
        directory = Path(directory)
        # mkdir refuses replacement. No config.json/model.safetensors pair is
        # emitted, so the serving loader cannot silently accept this artifact.
        directory.mkdir(parents=True, exist_ok=False)
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise TrainingError("nonfinite research selector")
        save_file(state, str(directory / "research-selector.safetensors"))
        (directory / "experiment.json").write_text(
            json.dumps(
                {
                    "capability": "splash-research-selector-v1",
                    "variant": self.variant,
                    "geometry": self.geometry,
                    "baseline_identity": baseline_identity,
                    "native_compatible": False,
                },
                indent=2,
            )
            + "\n"
        )

    def load_research(self, directory, baseline_identity):
        from safetensors.torch import load_file

        directory = Path(directory)
        metadata = json.loads((directory / "experiment.json").read_text())
        if metadata != {
            "capability": "splash-research-selector-v1",
            "variant": self.variant,
            "geometry": self.geometry,
            "baseline_identity": baseline_identity,
            "native_compatible": False,
        }:
            raise TrainingError(
                "research selector capability, geometry, variant or baseline differs"
            )
        state = load_file(str(directory / "research-selector.safetensors"))
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise TrainingError("nonfinite research selector")
        self.load_state_dict(state, strict=True)
