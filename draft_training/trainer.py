"""Deterministic optimizer-step training with bounded feature loading."""

import random
import time
from pathlib import Path

import torch

from .capture import load_sample
from .config import TrainingError, read_json, write_json
from .data import anchors
from .evaluate import evaluate_cached
from .export import export_model, restore_checkpoint, save_checkpoint, verify_export
from .losses import block_loss


def block_stream(directory, model, device, cursor=0, seed=42):
    manifest = read_json(Path(directory) / "manifest.json")
    # Count on CPU without retaining all samples or a per-token index in memory.
    count = sum(
        sum(
            1 for _ in anchors(load_sample(directory, shard), model.flash["block_size"])
        )
        for shard in manifest["shards"]
    )
    if not count:
        raise TrainingError("training set contains no full assistant query blocks")
    epoch = cursor // count
    while True:
        absolute = epoch * count
        rng = random.Random(seed + epoch)
        shards = list(manifest["shards"])
        rng.shuffle(shards)
        for shard in shards:
            sample = load_sample(directory, shard, device)
            choices = list(anchors(sample, model.flash["block_size"]))
            rng.shuffle(choices)
            for anchor in choices:
                if absolute >= cursor:
                    yield absolute, sample, anchor
                absolute += 1
        epoch += 1


def configure_trainable(model, mode):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            mode == "all" or name.startswith("candidate_selector.")
        )
        if parameter.requires_grad:
            # Preserve small optimizer updates with FP32 master parameters.
            # Forward matmuls use BF16 autocast; Adam moments remain FP32.
            parameter.data = parameter.data.float()


def train(
    model,
    embedding,
    head,
    training_dir,
    validation_dir,
    output,
    recipe,
    run_identity,
    provenance,
    device,
    *,
    resume=False,
    family=True,
    stop_after=None,
):
    output = Path(output)
    configure_trainable(model, recipe.trainable)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
    )
    progress = {
        "step": 0,
        "cursor": 0,
        "run_identity": run_identity,
        "best_score": None,
        "best_checkpoint": None,
        "updates": 0,
    }
    if resume:
        # A directory rename commits a checkpoint. Recover even if interruption
        # happened before updating latest.json, or before the first checkpoint.
        completed = sorted((output / "checkpoints").glob("step-*"))
        if completed:
            progress = restore_checkpoint(model, optimizer, completed[-1], run_identity)
        log = output / "metrics.jsonl"
        if log.exists():
            import json

            kept = []
            for line in log.read_text().splitlines():
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if item["step"] <= progress["step"]:
                    kept.append(json.dumps(item, allow_nan=False))
            log.write_text("\n".join(kept) + ("\n" if kept else ""))
    stream = block_stream(training_dir, model, device, progress["cursor"], recipe.seed)
    start = time.monotonic()
    for step in range(progress["step"] + 1, recipe.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        alpha = recipe.selector_loss_weight * min(
            1.0, step / max(1, recipe.selector_warmup_steps)
        )
        latest_metrics = None
        micro_count = recipe.batch_size * recipe.gradient_accumulation
        for _ in range(micro_count):
            cursor, sample, anchor = next(stream)
            with torch.autocast(
                device_type=torch.device(device).type, dtype=torch.bfloat16
            ):
                hidden = model.block_hidden(
                    sample["tokens"], sample["features"], anchor, embedding
                )
                loss, latest_metrics = block_loss(
                    model,
                    hidden,
                    sample["tokens"],
                    anchor,
                    head,
                    alpha,
                    recipe.vocabulary_chunk_rows,
                )
            if not torch.isfinite(loss):
                raise TrainingError(
                    "nonfinite training loss; previous checkpoints are preserved"
                )
            if loss.requires_grad:
                (loss / micro_count).backward()
            progress["cursor"] = cursor + 1
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), recipe.max_grad_norm, error_if_nonfinite=True
        )
        if norm > 0:
            optimizer.step()
            progress["updates"] += 1
        progress["step"] = step
        record = {
            "step": step,
            "seconds": time.monotonic() - start,
            "gradient_norm": float(norm),
            **latest_metrics,
        }
        due = step % recipe.validation_interval == 0 or step == recipe.max_steps
        if due:
            metrics = evaluate_cached(
                model, embedding, head, validation_dir, recipe, device
            )
            record["validation"] = metrics
            # Selection uses only validation replay. Native speed remains an
            # independent gate before any performance claim or ready PR.
            score = metrics["mean_matching_proposals"]
            if progress["best_score"] is None or score > progress["best_score"]:
                progress["best_score"] = score
                progress["best_checkpoint"] = f"step-{step:08d}"
        with (output / "metrics.jsonl").open("a") as stream_log:
            import json

            stream_log.write(json.dumps(record, allow_nan=False) + "\n")
        print(
            f"step {step}/{recipe.max_steps}: unary CE {latest_metrics['unary_ce']:.4f}",
            flush=True,
        )
        if due or step % recipe.checkpoint_interval == 0 or step == stop_after:
            save_checkpoint(model, optimizer, output / "checkpoints", progress)
        if step == stop_after:
            return progress
    if progress["updates"] == 0:
        raise TrainingError(
            "training made no gradient updates; inspect candidate coverage and supervision"
        )
    if progress["best_checkpoint"] is None:
        raise TrainingError("no validation checkpoint was selected")
    from safetensors.torch import load_file

    selected = output / "checkpoints" / progress["best_checkpoint"]
    model.load_state_dict(load_file(str(selected / "model.safetensors")))
    export_path = output / "export"
    export_provenance = {
        **provenance,
        "selected_checkpoint": progress["best_checkpoint"],
    }
    if export_path.exists():
        if not resume:
            raise TrainingError(
                "final export already exists; use --resume to verify it"
            )
        # Export publication and selection publication are separate atomic
        # operations. Recover the gap without replacing a published checkpoint.
        export_identity = verify_export(
            model, export_path, export_provenance, family=family
        )
    else:
        export_identity = export_model(
            model, export_path, export_provenance, family=family
        )
    selection = {
        **provenance,
        "export_identity": export_identity,
        "recipe": recipe.to_dict(),
        "selected_checkpoint": progress["best_checkpoint"],
        "criterion": "maximum validation mean_matching_proposals; earliest step wins ties",
        "native_validation": "not run",
        "status": "experimental",
    }
    selection_path = output / "selection.json"
    if selection_path.exists():
        if read_json(selection_path) != selection:
            raise TrainingError("selection manifest differs from the completed run")
    else:
        write_json(selection_path, selection)
    return progress
