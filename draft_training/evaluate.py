"""Greedy replay metrics; native deployment speed is measured separately."""

import random
from pathlib import Path

import torch
from safetensors import safe_open

from .capture import load_sample
from .config import (
    TrainingError,
    checkpoint_identity,
    file_digest,
    read_json,
    write_json,
)
from .data import anchors, disjoint
from .losses import block_loss


@torch.no_grad()
def evaluate_cached(model, embedding, head, directory, recipe, device):
    # Select the BF16 artifact that serving will load, rather than the FP32
    # master parameters. Restore their original storage even if replay fails;
    # the optimizer must retain every small update for the next training step.
    masters = [(p, p.data) for p in model.parameters() if p.dtype != torch.bfloat16]
    try:
        for parameter, value in masters:
            parameter.data = value.to(torch.bfloat16)
        return _evaluate_cached(model, embedding, head, directory, recipe, device)
    finally:
        for parameter, value in masters:
            parameter.data = value


def evaluation_plan(directory, manifest, block_size, recipe):
    # Inspect only the small masks while planning. Keep feature tensors bounded
    # to one continuation when scoring, and never spend budget on empty answers.
    selected = []
    shards = random.Random(recipe.seed).sample(
        manifest["shards"], len(manifest["shards"])
    )
    for shard in shards:
        path = Path(directory) / shard["file"]
        with safe_open(path, framework="pt", device="cpu") as source:
            mask = source.get_tensor("assistant_mask")
        if mask.ndim != 1:
            raise TrainingError("feature shard assistant mask must be one-dimensional")
        choices = list(anchors({"assistant_mask": mask}, block_size))
        if choices:
            selected.append((shard, choices))
            if len(selected) == recipe.max_validation_blocks:
                break
        elif file_digest(path) != shard["sha256"]:
            # Scored shards are checked by load_sample. Check skipped shards
            # here so a changed mask cannot silently remove an example.
            raise TrainingError("feature shard digest mismatch")

    quotas = [0] * len(selected)
    remaining = recipe.max_validation_blocks
    while remaining:
        available = False
        for i, (_, choices) in enumerate(selected):
            if quotas[i] < len(choices):
                available = True
                quotas[i] += 1
                remaining -= 1
                if not remaining:
                    break
        if not available:
            break
    return [
        (shard, random.Random(recipe.seed + i).sample(choices, quotas[i]))
        for i, (shard, choices) in enumerate(selected)
    ]


def _evaluate_cached(model, embedding, head, directory, recipe, device):
    model.eval()
    manifest = read_json(Path(directory) / "manifest.json")
    totals = {
        "blocks": 0,
        "tokens": 0,
        "covered": 0,
        "selector_correct": 0,
        "unary_ce": 0.0,
        "selector_ce": 0.0,
    }
    histogram = [0] * model.flash["block_size"]
    # Spend the frozen block budget across conversations, rather than letting
    # the first long continuation consume the entire validation/test set.
    selected = evaluation_plan(directory, manifest, model.flash["block_size"], recipe)
    totals["continuations_evaluated"] = 0
    for shard, choices in selected:
        sample = load_sample(directory, shard, device)
        totals["continuations_evaluated"] += bool(choices)
        for anchor in choices:
            with torch.autocast(
                device_type=torch.device(device).type, dtype=torch.bfloat16
            ):
                hidden = model.block_hidden(
                    sample["tokens"], sample["features"], anchor, embedding
                )
                _, metrics = block_loss(
                    model,
                    hidden,
                    sample["tokens"],
                    anchor,
                    head,
                    1,
                    recipe.vocabulary_chunk_rows,
                )
                logits = head(hidden).float()
                path = model.candidate_selector.path(
                    hidden, logits, sample["tokens"][anchor : anchor + 1]
                )
            labels = sample["tokens"][anchor + 1 : anchor + 1 + hidden.shape[1]]
            accepted = int((path[0] == labels).long().cumprod(0).sum())
            histogram[accepted] += 1
            totals["blocks"] += 1
            for key in ("tokens", "covered", "selector_correct"):
                totals[key] += metrics[key]
            for key in ("unary_ce", "selector_ce"):
                totals[key] += metrics[key] * metrics["tokens"]
            if totals["blocks"] >= recipe.max_validation_blocks:
                break
        if totals["blocks"] >= recipe.max_validation_blocks:
            break
    if not totals["blocks"]:
        raise TrainingError("evaluation set contains no full assistant query blocks")
    count = totals["tokens"]
    return {
        "scope": "frozen-teacher greedy trace replay; not native throughput or sampled acceptance",
        "weight_precision": "BF16 serving export",
        "supervision": manifest["binding"]["supervision"],
        "blocks": totals["blocks"],
        "continuations_evaluated": totals["continuations_evaluated"],
        "total_continuations": len(manifest["shards"]),
        "block_budget": recipe.max_validation_blocks,
        "unary_ce": totals["unary_ce"] / count,
        "selector_ce": totals["selector_ce"] / count,
        "top_k_coverage": totals["covered"] / count,
        "selector_accuracy_when_covered": totals["selector_correct"] / totals["covered"]
        if totals["covered"]
        else None,
        "mean_matching_proposals": sum(i * n for i, n in enumerate(histogram))
        / totals["blocks"],
        "matching_proposal_histogram": histogram,
    }


def validate_selection(selection_path, draft, test_manifest, teacher_identity, recipe):
    selection = read_json(selection_path)
    if selection["export_identity"] != checkpoint_identity(draft):
        raise TrainingError("export differs from the frozen selection manifest")
    if (
        selection["teacher_identity"] != teacher_identity
        or selection["recipe"] != recipe.to_dict()
    ):
        raise TrainingError(
            "teacher or evaluation recipe differs from frozen selection"
        )
    if test_manifest is not None:
        for split in ("training_split", "validation_split"):
            disjoint(selection[split], test_manifest)
    return selection


def write_test_report(path, selection, test_manifest, metrics):
    path = Path(path)
    if path.exists():
        raise TrainingError(
            "test report already exists; final test results must not be overwritten"
        )
    write_json(
        path,
        {
            "selection": selection,
            "test_split": test_manifest,
            "metrics": metrics,
            "status": "experimental; native deployment validation still required",
        },
    )
