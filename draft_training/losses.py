"""Experimental unary CE and strict-candidate selector CE, in bounded chunks."""

import math

import torch
from torch.nn import functional as F

from .model import top_candidates


def prefix_survival(probabilities, known):
    """Teacher-forced soft correct-token prefix length, not rollout acceptance.

    Missing support supplies probability zero. Censoring stops the prefix at
    the first unknown label; later labels cannot restart it.
    """
    if probabilities.ndim != 1 or known.shape != probabilities.shape:
        raise ValueError("prefix probabilities and known mask must be vectors")
    return (probabilities * known.to(probabilities.dtype)).cumprod(0).sum()


def block_loss(
    model,
    hidden,
    tokens,
    anchor,
    output_head,
    selector_weight,
    chunk_rows=1,
    *,
    prefix_weight=0.0,
    position_decay=1.0,
    label_mask=None,
):
    labels = tokens[anchor + 1 : anchor + 1 + hidden.shape[1]]
    predecessors = tokens[anchor : anchor + hidden.shape[1]]
    known = (
        torch.ones_like(labels, dtype=torch.bool) if label_mask is None else label_mask
    )
    if known.shape != labels.shape or known.dtype != torch.bool or not known.any():
        raise ValueError("label_mask must select at least one known block label")
    log_weights = math.log(position_decay) * torch.arange(
        len(labels), device=hidden.device, dtype=torch.float32
    )
    log_weights = log_weights.masked_fill(~known, -torch.inf)
    weights = (log_weights - log_weights.max()).exp()
    weights = weights * known.sum() / weights.sum()
    unary_sum = hidden.new_zeros((), dtype=torch.float32)
    selector_sum = hidden.new_zeros((), dtype=torch.float32)
    prefix_probabilities = []
    covered, correct = 0, 0
    for start in range(0, hidden.shape[1], chunk_rows):
        stop = min(hidden.shape[1], start + chunk_rows)
        rows = hidden[0, start:stop]
        # Keep gradients through the frozen output head for backbone training.
        logits = output_head(rows).float()
        target = labels[start:stop]
        chunk_known = known[start:stop]
        # Unknown labels may be padding sentinels. Never gather or score them.
        safe_target = torch.where(chunk_known, target, 0)
        unary_sum = (
            unary_sum
            + (
                F.cross_entropy(logits, safe_target, reduction="none")
                * weights[start:stop]
            ).sum()
        )
        unary, candidates = top_candidates(logits, model.flash["selector_top_k"])
        # For a prefix with an unknown predecessor, this row cannot supervise
        # a correct-prefix transition, even if its own token is known.
        predecessor_known = torch.cat((known.new_ones(1), known[:-1]))[start:stop]
        safe_predecessors = torch.where(predecessor_known, predecessors[start:stop], 0)
        scores = model.candidate_selector.scores(
            rows, unary, candidates, safe_predecessors
        )
        matches = candidates == safe_target[:, None]
        valid = matches.any(-1) & chunk_known & predecessor_known
        gold = matches.long().argmax(-1)
        covered += int(valid.sum())
        log_gold = F.log_softmax(scores, dim=-1).gather(-1, gold[:, None]).squeeze(-1)
        prefix_probabilities.append(log_gold.exp() * valid)
        if valid.any():
            selector_sum = (
                selector_sum
                + (
                    F.cross_entropy(scores[valid], gold[valid], reduction="none")
                    * weights[start:stop][valid]
                ).sum()
            )
            correct += int((scores[valid].argmax(-1) == gold[valid]).sum())
    count = int(known.sum())
    soft_prefix = prefix_survival(torch.cat(prefix_probabilities), known)
    # Normalize by the number of observed consecutive labels. A censored
    # suffix is neither a failure nor free extra survival reward.
    observed_prefix = int(known.long().cumprod(0).sum())
    prefix_loss = -soft_prefix / max(1, observed_prefix)
    loss = unary_sum / count
    if selector_weight:
        loss = loss + selector_weight * selector_sum / count
    if prefix_weight:
        loss = loss + prefix_weight * prefix_loss
    return loss, {
        "unary_ce": float(unary_sum.detach()) / count,
        "selector_ce": float(selector_sum.detach()) / count,
        "prefix_survival_surrogate": float(soft_prefix.detach()),
        "tokens": count,
        "covered": covered,
        "selector_correct": correct,
    }
