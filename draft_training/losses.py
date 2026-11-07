"""Experimental unary CE and strict-candidate selector CE, in bounded chunks."""

import torch
from torch.nn import functional as F

from .model import top_candidates


def block_loss(
    model, hidden, tokens, anchor, output_head, selector_weight, chunk_rows=1
):
    labels = tokens[anchor + 1 : anchor + 1 + hidden.shape[1]]
    predecessors = tokens[anchor : anchor + hidden.shape[1]]
    unary_sum = hidden.new_zeros((), dtype=torch.float32)
    selector_sum = hidden.new_zeros((), dtype=torch.float32)
    covered, correct = 0, 0
    for start in range(0, hidden.shape[1], chunk_rows):
        stop = min(hidden.shape[1], start + chunk_rows)
        rows = hidden[0, start:stop]
        # Head parameters stay frozen, but this multiplication must preserve
        # gradients to the draft when training its backbone.
        logits = output_head(rows).float()
        target = labels[start:stop]
        unary_sum = unary_sum + F.cross_entropy(logits, target, reduction="sum")
        unary, candidates = top_candidates(logits, model.flash["selector_top_k"])
        scores = model.candidate_selector.scores(
            rows, unary, candidates, predecessors[start:stop]
        )
        matches = candidates == target[:, None]
        valid = matches.any(-1)
        covered += int(valid.sum())
        if valid.any():
            gold = matches.long().argmax(-1)
            selector_sum = selector_sum + F.cross_entropy(
                scores[valid], gold[valid], reduction="sum"
            )
            correct += int((scores[valid].argmax(-1) == gold[valid]).sum())
    count = len(labels)
    loss = unary_sum / count
    if selector_weight:
        loss = loss + selector_weight * selector_sum / count
    return loss, {
        "unary_ce": float(unary_sum.detach()) / count,
        "selector_ce": float(selector_sum.detach()) / count,
        "tokens": count,
        "covered": covered,
        "selector_correct": correct,
    }
