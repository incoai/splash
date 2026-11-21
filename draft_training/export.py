"""Atomic, finite BF16 serving exports and safe local training checkpoints."""

import os
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .config import (
    TrainingError,
    checkpoint_identity,
    checkpoint_index,
    read_json,
    validate_draft,
    write_json,
)


def export_model(model, destination, provenance, *, family=True, shard_bytes=512 << 20):
    destination = Path(destination)
    if destination.exists():
        raise TrainingError(f"export already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".draft-export-", dir=destination.parent))
    try:
        shards, current, size = [], {}, 0
        for name, tensor in model.state_dict().items():
            value = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            if not torch.isfinite(value).all():
                raise TrainingError(f"nonfinite exported tensor: {name}")
            if current and size + value.numel() * value.element_size() > shard_bytes:
                filename = f"model-{len(shards) + 1:05d}.safetensors"
                save_file(current, str(stage / filename))
                shards.append((filename, list(current)))
                current, size = {}, 0
            current[name] = value.clone()
            size += value.numel() * value.element_size()
        filename = (
            f"model-{len(shards) + 1:05d}.safetensors"
            if shards
            else "model.safetensors"
        )
        save_file(current, str(stage / filename))
        shards.append((filename, list(current)))
        if len(shards) > 1:
            write_json(
                stage / "model.safetensors.index.json",
                {
                    "weight_map": {
                        name: file for file, names in shards for name in names
                    }
                },
            )
        write_json(stage / "config.json", model.raw_config)
        write_json(stage / "training-provenance.json", provenance)
        validate_draft(stage, family=family)
        # Destination must be new; callers use one export path per selected step.
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return checkpoint_identity(destination)


def verify_export(model, directory, provenance, *, family=True):
    """Recover a published export only when it matches the selected checkpoint."""
    directory = Path(directory)
    if (
        validate_draft(directory, family=family) != model.raw_config
        or read_json(directory / "training-provenance.json") != provenance
    ):
        raise TrainingError(
            "existing export differs from the selected checkpoint provenance"
        )
    index = checkpoint_index(directory)
    state = model.state_dict()
    for filename in sorted({item["file"] for item in index.values()}):
        with safe_open(directory / filename, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                expected = state[name].detach().to(device="cpu", dtype=torch.bfloat16)
                if not torch.equal(shard.get_tensor(name), expected):
                    raise TrainingError(
                        f"existing export differs from the selected checkpoint: {name}"
                    )
    return checkpoint_identity(directory)


def save_checkpoint(model, optimizer, directory, state):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    step_path = directory / f"step-{state['step']:08d}"
    if step_path.exists():
        raise TrainingError(f"checkpoint already exists: {step_path}")
    stage = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=directory))
    try:
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
            str(stage / "model.safetensors"),
        )
        # Only local checkpoints created by this command are resumed. Loading
        # uses weights_only=True; arbitrary Python objects are never unpickled.
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
            },
            stage / "state.pt",
        )
        write_json(stage / "progress.json", state)
        os.rename(stage, step_path)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    write_json(directory / "latest.json", {"checkpoint": step_path.name})
    return step_path


def restore_checkpoint(model, optimizer, path, run_identity):
    path = Path(path)
    progress = read_json(path / "progress.json")
    if progress["run_identity"] != run_identity:
        raise TrainingError(
            "resume checkpoint belongs to different data, recipe, target, or draft"
        )
    model.load_state_dict(load_file(str(path / "model.safetensors")))
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["cpu_rng"])
    if state["cuda_rng"]:
        if (
            not torch.cuda.is_available()
            or len(state["cuda_rng"]) != torch.cuda.device_count()
        ):
            raise TrainingError("resume requires the original CUDA device topology")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return progress
