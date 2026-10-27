#!/usr/bin/env python3
"""Remove one Splash model selection without deleting shared Hub cache files."""

import argparse
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from install import assembly, hub, models


def remove(models_root: Path, model_id: str) -> None:
    selection = models.Selection.of(models_root, model_id)
    with models.installation_lock(selection.models_root):
        if not selection.link.is_symlink():
            raise models.ModelError("model selection is not installed")
        record = assembly.recorded(selection.link)
        old_refs = [
            hub.pinned(snapshot, selection.link)
            for snapshot, _ in assembly.pins(record)
        ] if record else []
        selection.link.unlink()
        hub.retire_other_pins([], old_refs)
        assembly.collect_garbage(selection.models_root)


def estimate_sizes(format_name: str, repository: str, variants: list[str]) -> dict[str, int]:
    from install import upstream

    if format_name == "GGUF":
        repo = hub.Repository.resolve(repository)
        result = {}
        for variant in variants:
            try:
                names, _ = upstream.select_gguf(repo.files, variant)
            except models.ModelError:
                continue
            if all(name in repo.sizes for name in names):
                result[variant] = sum(repo.sizes[name][0] for name in names)
        return result
    if format_name == "MLX":
        result = {}
        for variant in variants:
            model_id = f"{repository}-{variant}"
            try:
                repo = hub.Repository.resolve(model_id)
            except Exception as error:
                print(f"Size unavailable for {model_id}: {error}", file=sys.stderr)
                continue
            if (any(name.endswith(".safetensors") for name in repo.files)
                    and all(name in repo.sizes for name in repo.files)):
                result[variant] = sum(repo.sizes[name][0] for name in repo.files)
        return result
    raise ValueError(f"unsupported model format: {format_name}")


def draft_files(repo) -> set[str]:
    from install import upstream

    return {"config.json", *upstream._weight_files(repo),
            *({upstream.DRAFT_MASK_EMBEDDING} & set(repo.files))}


def download_plan(models_root: Path, model_id: str, language_only: bool) -> dict:
    from install import upstream

    selection = models.Selection.of(models_root, model_id, language_only=language_only)
    if models.installation_kind(selection.link) == models.PACKAGE:
        models.refuse_package(model_id, models.read_json(selection.link / "manifest.json"), selection.link)
    target_repo = hub.Repository.resolve(selection.repo_id, installation=selection.link)
    if "manifest.json" in target_repo.files:
        models.refuse_package(model_id, models.read_json(target_repo.file("manifest.json")))
    with tempfile.TemporaryDirectory(prefix="splash-download-plan-") as scratch:
        # inspect_target reports its selection on stdout; reserve stdout for JSON.
        with redirect_stdout(sys.stderr):
            target = upstream.inspect_target(
                target_repo, selection.variant, language_only, Path(scratch)
            )
        draft = upstream._resolve_draft(target.family, selection, None, target_repo)
        names = draft_files(draft)
        config = draft.file("config.json")
        upstream.check_model(
            target.format, target.vision_format, target.config,
            gguf_metadata=target.gguf_metadata, draft=config,
        )
    from huggingface_hub import try_to_load_from_cache
    missing = [] if draft.paths is not None else [
        name for name in names
        if not isinstance(try_to_load_from_cache(draft.name, name, revision=draft.revision), str)
    ]
    blobs = hub.folder(draft.name) / "blobs" if draft.paths is None else None
    transfer = [name for name in missing if name not in draft.sizes or not (
        draft.sizes[name][1] and (blobs / draft.sizes[name][1]).exists()
    )]
    size = sum(draft.sizes[name][0] for name in transfer) if all(
        name in draft.sizes for name in transfer
    ) else None
    return {
        "model_revision": target_repo.revision,
        "draft_repository": draft.name,
        "draft_revision": draft.revision,
        "draft_needed": bool(missing),
        "draft_size_bytes": size,
    }


def fetch_model(models_root: Path, model_id: str, language_only: bool, revision: str) -> None:
    from install import upstream

    selection = models.Selection.of(models_root, model_id, language_only=language_only)
    repo = hub.Repository.resolve(selection.repo_id, revision, installation=selection.link)
    with tempfile.TemporaryDirectory(prefix="splash-model-fetch-") as scratch:
        target = upstream.inspect_target(repo, selection.variant, language_only, Path(scratch))
        repo.download(set(target.files.values()))


def fetch_draft(repository: str, revision: str) -> None:
    repo = hub.Repository.resolve(repository, revision)
    repo.download(draft_files(repo))


def assemble_model(models_root: Path, model_id: str, language_only: bool,
                   model_revision: str, draft_repository: str, draft_revision: str) -> None:
    from install import upstream

    selection = models.Selection.of(models_root, model_id, language_only=language_only)
    if models.installation_kind(selection.link) == models.PACKAGE:
        models.refuse_package(model_id, models.read_json(selection.link / "manifest.json"), selection.link)
    target = hub.Repository.cached(selection.repo_id, model_revision)
    if "manifest.json" in target.files:
        models.refuse_package(model_id, models.read_json(target.file("manifest.json")))
    draft = hub.Repository.cached(draft_repository, draft_revision)
    upstream._install(selection, target, None, draft)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--repository")
    parser.add_argument("--format", choices=["GGUF", "MLX"])
    parser.add_argument("--variants")
    parser.add_argument("--revision")
    parser.add_argument("--draft-revision")
    parser.add_argument("--language-only", action="store_true")
    parser.add_argument("command", choices=["remove", "sizes", "plan", "fetch-model", "fetch-draft", "assemble"])
    args = parser.parse_args()
    try:
        if args.command == "remove":
            if args.models is None or args.model is None:
                parser.error("remove requires --models and --model")
            remove(args.models, args.model)
        elif args.command == "sizes":
            if args.repository is None or args.format is None or args.variants is None:
                parser.error("sizes requires --repository, --format, and --variants")
            print(json.dumps(estimate_sizes(args.format, args.repository, args.variants.split(","))))
        elif args.command in ("plan", "fetch-model", "assemble"):
            if args.models is None or args.model is None:
                parser.error(f"{args.command} requires --models and --model")
            if args.command == "plan":
                print(json.dumps(download_plan(args.models, args.model, args.language_only)))
            elif args.command == "fetch-model":
                if args.revision is None:
                    parser.error("fetch-model requires --revision")
                fetch_model(args.models, args.model, args.language_only, args.revision)
            else:
                if args.revision is None or args.repository is None or args.draft_revision is None:
                    parser.error("assemble requires --revision, --repository, and --draft-revision")
                assemble_model(args.models, args.model, args.language_only,
                               args.revision, args.repository, args.draft_revision)
        else:
            if args.repository is None or args.revision is None:
                parser.error("fetch-draft requires --repository and --revision")
            fetch_draft(args.repository, args.revision)
    except (models.ModelError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
