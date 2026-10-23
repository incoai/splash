"""Optional DFlash2 training commands; help and metadata validation need no torch."""

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

from .config import (
    Recipe,
    TrainingError,
    checkpoint_identity,
    digest,
    file_digest,
    read_json,
    validate_draft,
    validate_target,
    write_json,
)
from .data import disjoint, split_manifest

REFERENCE_REVISION = "07ebd93db9f472af339b644bb70221ad8428328a"


def implementation_identity():
    return {p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def execution_context(torch, device):
    context = {
        "device": device,
        "device_name": torch.cuda.get_device_name()
        if device == "cuda"
        else platform.machine(),
        "cuda_version": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "forward_precision": "BF16 autocast; FP32 rotary frequencies",
        "optimizer_precision": "FP32 trainable masters and Adam moments",
        "attention": "sdpa",
    }
    if device == "mps":
        context.update(
            device_name=torch.backends.mps.get_name(),
            macos_version=platform.mac_ver()[0],
            mps_cpu_fallback=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        )
    return context


def memory_snapshot(torch, device):
    """Synchronize MPS before recording allocator snapshots, not peak estimates."""
    if device != "mps":
        return None
    torch.mps.synchronize()
    return {
        "device": "mps",
        "measurement": "end-of-phase snapshot; not peak memory",
        "allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "recommended_max_bytes": torch.mps.recommended_max_memory(),
    }


def add_train_arguments(parser):
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="local BF16 Transformers Qwen3.8-27B checkpoint",
    )
    parser.add_argument(
        "--init-draft",
        type=Path,
        required=True,
        help="local released BF16 DFlash2 draft",
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="training conversations JSONL"
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        required=True,
        help="disjoint validation conversations JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new run directory, or original directory with --resume",
    )
    parser.add_argument("--config", type=Path, help="YAML training recipe")
    parser.add_argument(
        "--supervision",
        choices=("target-generated", "supplied-answers"),
        default="target-generated",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "mps"),
        default="cuda",
        help="CUDA by default; experimental Apple GPU MPS; CPU reference diagnostics",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="validate metadata and splits without loading models or writing output",
    )
    group.add_argument(
        "--preflight",
        action="store_true",
        help="capture one continuation and probe gradients without starting a run",
    )
    group.add_argument(
        "--resume",
        action="store_true",
        help="continue the matching run from its latest complete checkpoint",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "capture"):
        add_train_arguments(commands.add_parser(name))
    validate = commands.add_parser(
        "validate", help="check draft/target metadata without allocating model tensors"
    )
    validate.add_argument("--target", type=Path, required=True)
    validate.add_argument("--draft", type=Path, required=True)
    evaluate = commands.add_parser(
        "evaluate", help="final frozen-checkpoint greedy replay on untouched test data"
    )
    for name in (
        "target",
        "draft",
        "baseline-draft",
        "test-data",
        "selection-manifest",
        "output",
    ):
        evaluate.add_argument("--" + name, type=Path, required=True)
    evaluate.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    export = commands.add_parser(
        "export", help="export a checkpoint produced by this trainer"
    )
    for name in ("checkpoint", "init-draft", "output"):
        export.add_argument("--" + name, type=Path, required=True)
    return parser.parse_args(argv)


def require_torch(device):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise TrainingError(
            "CUDA is unavailable. Use --device mps on a supported Apple Silicon Mac, or --device cpu for reference diagnostics."
        )
    if device == "mps" and not torch.backends.mps.is_available():
        raise TrainingError(
            "MPS is unavailable. Apple GPU training needs a supported Mac, macOS, and an MPS-enabled PyTorch build. Use --device cpu for reference diagnostics."
        )
    return torch


def inspect_training(args):
    validate_target(args.target)
    config = validate_draft(args.init_draft)
    recipe = Recipe.load(args.config)
    if recipe.sequence_length > config["max_position_embeddings"]:
        raise TrainingError("sequence_length exceeds model position limit")
    training, validation = (
        split_manifest(args.data),
        split_manifest(args.validation_data),
    )
    disjoint(training, validation)
    if any(
        source.resolve() == args.output.resolve()
        or source.resolve() in args.output.resolve().parents
        for source in (args.target, args.init_draft)
    ):
        raise TrainingError("run output must not be a source checkpoint")
    if (
        not args.dry_run
        and not args.preflight
        and args.output.exists()
        and any(args.output.iterdir())
        and not args.resume
    ):
        raise TrainingError(
            "output is nonempty; use --resume for a matching interrupted run"
        )
    return config, recipe, training, validation


def run_training(args, *, family=True):
    config, recipe, training, validation = inspect_training(args)
    if args.dry_run:
        return {
            "mode": "metadata-only",
            "family": "Qwen3.8-27B",
            "recipe": recipe.to_dict(),
            "training": training,
            "validation": validation,
            "feature_bytes_per_token": len(config["dflash_config"]["target_layer_ids"])
            * config["hidden_size"]
            * 2,
            "native_validation": "not run",
        }
    torch = require_torch(args.device)
    from .capture import TargetAdapter, capture_dataset
    from .evaluate import evaluate_cached
    from .model import load_draft
    from .trainer import train

    torch.manual_seed(recipe.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(recipe.seed)
    if args.preflight:
        from .data import conversations, examples
        from .losses import block_loss
        from .trainer import configure_trainable

        adapter = TargetAdapter.load(
            args.target, config["dflash_config"]["target_layer_ids"], args.device
        )
        messages, answer = next(examples(conversations(args.data), args.supervision))
        sample = {
            name: value.to(args.device)
            for name, value in adapter.make_sample(messages, answer, recipe).items()
        }
        from .data import anchors

        capture_peak = (
            torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        )
        capture_memory = memory_snapshot(torch, args.device)
        embedding, head = adapter.release_teacher()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = load_draft(args.init_draft, config, args.device)
        anchor = next(anchors(sample, model.flash["block_size"]), None)
        if anchor is None:
            raise TrainingError(
                "preflight continuation has no full assistant query block"
            )
        configure_trainable(model, recipe.trainable)
        with torch.autocast(
            device_type=torch.device(args.device).type, dtype=torch.bfloat16
        ):
            hidden = model.block_hidden(
                sample["tokens"], sample["features"], anchor, embedding
            )
        with torch.autocast(
            device_type=torch.device(args.device).type, dtype=torch.bfloat16
        ):
            loss, metrics = block_loss(
                model,
                hidden,
                sample["tokens"],
                anchor,
                head,
                1,
                recipe.vocabulary_chunk_rows,
            )
        if not loss.requires_grad:
            raise TrainingError(
                "preflight has no selector targets; inspect candidate coverage"
            )
        loss.backward()
        if not all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise TrainingError("preflight produced nonfinite gradients")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=recipe.learning_rate,
            weight_decay=recipe.weight_decay,
        )
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), recipe.max_grad_norm, error_if_nonfinite=True
        )
        optimizer.step()
        return {
            "mode": "device-preflight",
            "capture_peak_cuda_bytes": capture_peak,
            "metrics": metrics,
            "training_peak_cuda_bytes": torch.cuda.max_memory_allocated()
            if args.device == "cuda"
            else None,
            "capture_memory": capture_memory,
            "training_memory": memory_snapshot(torch, args.device),
        }
    teacher_identity = checkpoint_identity(args.target)
    initial_identity = checkpoint_identity(args.init_draft)
    provenance = {
        "teacher_identity": teacher_identity,
        "initial_draft_identity": initial_identity,
        "training_split": training,
        "validation_split": validation,
        "recipe": recipe.to_dict(),
        "supervision": args.supervision,
        "reference_revision": REFERENCE_REVISION,
        "splash_contract": {
            name: file_digest(Path(__file__).parent.parent / name)
            for name in (
                "runtime/model/Qwen3_8.hpp",
                "runtime/model/DraftCheckpoint.cpp",
                "runtime/model/ModelDescriptor.mm",
                "runtime/metal/abi/ExecutionGeometry.h",
                "runtime/metal/abi/DraftAttention.h",
            )
        },
        "implementation": implementation_identity(),
        "execution": execution_context(torch, args.device),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "safetensors")
        },
        "deployment_validation": "pending: BF16 teacher and native quantized target are distinct artifacts",
    }
    run_identity = digest(provenance)
    if args.resume:
        if read_json(args.output / "run.json")["run_identity"] != run_identity:
            raise TrainingError(
                "resume configuration or input files differ from the original run"
            )
    # Reject an incompatible resume before allocating the full BF16 teacher.
    adapter = TargetAdapter.load(
        args.target, config["dflash_config"]["target_layer_ids"], args.device
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "run.json",
        {**provenance, "run_identity": run_identity, "draft_config": config},
    )
    capture_identity = {
        "teacher": teacher_identity,
        "tokenizer": {
            k: v for k, v in teacher_identity.items() if "token" in k or "template" in k
        },
        "layers": config["dflash_config"]["target_layer_ids"],
        "versions": provenance["versions"],
        "execution": provenance["execution"],
    }
    training_dir, validation_dir = (
        args.output / "features/train",
        args.output / "features/validation",
    )
    capture_dataset(
        adapter, args.data, training_dir, recipe, capture_identity, args.supervision
    )
    capture_dataset(
        adapter,
        args.validation_data,
        validation_dir,
        recipe,
        capture_identity,
        args.supervision,
    )
    if args.command == "capture":
        return {"mode": "capture", "output": str(args.output)}
    # Model allocation follows teacher release, so target decoder weights and
    # training optimizer state do not have to coexist on the accelerator.
    embedding, head = adapter.release_teacher()
    model = load_draft(args.init_draft, config, args.device)
    if not (args.output / "baseline-validation.json").exists():
        write_json(
            args.output / "baseline-validation.json",
            evaluate_cached(
                model, embedding, head, validation_dir, recipe, args.device
            ),
        )
    return train(
        model,
        embedding,
        head,
        training_dir,
        validation_dir,
        args.output,
        recipe,
        run_identity,
        provenance,
        args.device,
        resume=args.resume,
        family=family,
    )


def run_evaluation(args):
    validate_target(args.target)
    config = validate_draft(args.draft)
    validate_draft(args.baseline_draft)
    selection = read_json(args.selection_manifest)
    recipe = Recipe.load()
    recipe = type(recipe)(**selection["recipe"])
    from .evaluate import validate_selection

    teacher = checkpoint_identity(args.target)
    validate_selection(args.selection_manifest, args.draft, None, teacher, recipe)
    if checkpoint_identity(args.baseline_draft) != selection["initial_draft_identity"]:
        raise TrainingError("baseline differs from the original draft")
    if args.output.exists():
        raise TrainingError(
            "evaluation output already exists; final reports are not overwritten"
        )
    torch = require_torch(args.device)
    if selection["implementation"] != implementation_identity():
        raise TrainingError(
            "evaluator implementation differs from the frozen selection"
        )
    if selection["execution"] != execution_context(torch, args.device):
        raise TrainingError(
            "evaluation device context differs from the frozen selection"
        )
    if any(
        importlib.metadata.version(name) != version
        for name, version in selection["versions"].items()
    ):
        raise TrainingError(
            "evaluation library versions differ from the frozen selection"
        )
    test = split_manifest(args.test_data)
    for split in ("training_split", "validation_split"):
        disjoint(selection[split], test)
    from .capture import TargetAdapter, capture_dataset
    from .evaluate import evaluate_cached, write_test_report
    from .model import load_draft

    adapter = TargetAdapter.load(
        args.target, config["dflash_config"]["target_layer_ids"], args.device
    )
    args.output.mkdir(parents=True)
    features = args.output / "features"
    capture_dataset(
        adapter,
        args.test_data,
        features,
        recipe,
        {"teacher": teacher, "layers": config["dflash_config"]["target_layer_ids"]},
        selection["supervision"],
    )
    embedding, head = adapter.release_teacher()
    metrics = {}
    for name, directory in [("baseline", args.baseline_draft), ("trained", args.draft)]:
        model = load_draft(directory, config, args.device)
        metrics[name] = evaluate_cached(
            model, embedding, head, features, recipe, args.device
        )
        del model
    write_test_report(args.output / "test-report.json", selection, test, metrics)
    return metrics


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.command in ("train", "capture"):
            result = run_training(args)
        elif args.command == "validate":
            validate_target(args.target)
            validate_draft(args.draft)
            result = {
                "metadata": "valid",
                "native_loading": "not run",
                "tensor_finiteness": "not scanned",
            }
        elif args.command == "evaluate":
            result = run_evaluation(args)
        else:
            from .export import export_model
            from .model import load_draft

            config = validate_draft(args.init_draft)
            progress = read_json(args.checkpoint / "progress.json")
            model = load_draft(args.checkpoint, config)
            result = export_model(model, args.output, progress)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except ModuleNotFoundError as error:
        print(
            f"error: optional training dependency {error.name} is missing; install training/requirements.txt in a separate environment",
            file=sys.stderr,
        )
        return 1
    except (
        TrainingError,
        OSError,
        KeyError,
        TypeError,
        RuntimeError,
        StopIteration,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "training interrupted; resume from the latest complete checkpoint with --resume",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
