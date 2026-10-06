"""Reuse external model libraries through the ordinary native model assembly."""

from __future__ import annotations

import shlex
from dataclasses import replace
from pathlib import Path

if __package__:
    from . import assembly, families, hub, library, models, upstream
else:
    import assembly
    import families
    import hub
    import library
    import models
    import upstream


def _missing_draft(selection, name, detail=None):
    command = ["splash", "download-draft", "--model", selection.model]
    if selection.model_dir is not None:
        command += ["--model-dir", str(selection.model_dir)]
    if selection.draft_model:
        command += ["--draft-model", selection.draft_model]
    if selection.revision:
        command += ["--revision", selection.revision]
    return models.ModelError(
        f"Missing draft {name}. Run: {shlex.join(command)}"
        + (f". Cached checkpoint is unusable: {detail}" if detail else "")
    )


def _installed_drafts(selection, name):
    for link in models.selection_links(selection.models_root):
        try:
            record = assembly.verify(link)
            source = record["sources"]["draft"]
            if source["repo"] != name:
                continue
            directory = Path(record["files"]["draft/config.json"]["path"]).parent
            local = hub.Repository.local_directory(directory)
            yield hub.Repository(name, source["revision"], local.files, directory)
        except (models.ModelError, OSError, KeyError):
            # A damaged unrelated installation is not a usable cache candidate.
            continue


def _cached_drafts(selection, name):
    yield from _installed_drafts(selection, name)
    snapshots = hub.folder(name) / "snapshots"
    if snapshots.is_dir():
        for path in sorted(
            snapshots.iterdir(), key=lambda p: p.stat().st_mtime_ns, reverse=True
        ):
            if path.is_dir() and models.is_hex_digest(path.name, 40):
                yield hub.Repository.cached(name, path.name)


def resolve_draft(selection, family):
    name = selection.draft_model or family.draft.repo
    if Path(name).is_absolute():
        repo = hub.Repository.local_directory(name)
        return repo, upstream._draft_files(repo, family)
    detail = None
    if selection.model_dir is not None:
        directory = selection.model_dir / name
        if directory.exists():
            repo = hub.Repository.local_directory(directory)
            try:
                return repo, upstream._draft_files(repo, family)
            except upstream.MissingCheckpoint as error:
                detail = str(error)
    for repo in _cached_drafts(selection, name):
        try:
            names = upstream.inspect_draft(repo, family)
        except (models.ModelError, OSError) as error:
            detail = str(error)
            continue
        if selection.download_draft and selection.model_dir is not None:
            repo = library.materialize(
                repo, _source_files(repo, names), selection.model_dir
            )
        return repo, upstream._draft_files(repo, family)
    if selection.download_draft:
        repo = hub.Repository.resolve(name, installation=selection.link)
        names = upstream.inspect_draft(repo, family)
        if selection.model_dir is not None:
            repo = library.materialize(
                repo, _source_files(repo, names), selection.model_dir
            )
        return repo, upstream._draft_files(repo, family)
    raise _missing_draft(selection, name, detail)


def _source_files(repo, names):
    """Keep checkpoint indices and processor metadata for later local starts."""
    companions = {"model.safetensors.index.json", "preprocessor_config.json"}
    return set(names) | (companions & repo.files)


def target_repository(selection):
    if selection.model_dir is not None:
        if selection.model_dir.exists() and not selection.model_dir.is_dir():
            raise models.ModelError(
                f"model library is not a directory: {selection.model_dir}"
            )
        directory = selection.model_dir / selection.repo_id
        if directory.exists():
            repo = hub.Repository.local_directory(directory)
            missing_variant = selection.variant and not any(
                "/" not in name
                and name.lower().endswith(f"-{selection.variant.lower()}.gguf")
                and "mmproj" not in name.lower()
                for name in repo.files
            )
            has_weights = any(
                name.endswith((".safetensors", ".gguf")) for name in repo.files
            )
            if has_weights and not missing_variant:
                if selection.revision:
                    commit = library.revision(directory)
                    if commit is None:
                        raise models.ModelError(
                            "cannot verify --revision for unmanaged or changed local weights; "
                            "omit --revision or use another model directory"
                        )
                    requested = selection.revision
                    if not models.is_hex_digest(requested, 40):
                        requested = hub.Repository.resolve(
                            selection.repo_id, requested
                        ).revision
                    if commit != requested.lower():
                        raise models.ModelError(
                            "local weights have a different revision; use another model directory"
                        )
                return repo
    return hub.Repository.resolve(
        selection.repo_id, selection.revision, installation=selection.link
    )


def download_draft(selection):
    with hub.as_model_errors(f"cannot download draft for {selection.model}"):
        repo = target_repository(selection)
        # Draft pairing needs target geometry, not vision or target weights.
        target = upstream.inspect_target(repo, selection.variant, True)
        family = families.family_for(target.config)
        draft, _ = resolve_draft(replace(selection, download_draft=True), family)
        destination = draft.directory or hub.snapshot(draft.name, draft.revision)
        print(f"Draft for {selection.model} is ready in {destination}.", flush=True)


def prepare(selection):
    with hub.as_model_errors(f"cannot install {selection.model}"):
        _prepare(selection)


def _reuse(selection, repo, target, draft, files):
    if not selection.link.exists():
        return False
    with models.installation_lock(selection.models_root):
        try:
            installed = assembly.verify(selection.link)
        except (models.ModelError, OSError):
            return False
        sources = {"target": repo.identity(), "draft": draft.identity()}
        if installed["sources"] != sources or any(
            installed["files"].get(name, {}).get("path") != str(path.absolute())
            for name, path in files.items()
        ):
            return False
        if target.format == "gguf" and installed["metadata"] != assembly.metadata_key(
            installed["files"]
        ):
            return False
        hub.repair_pins(selection.link, assembly.pins(installed))
        print(
            f"Splash model {selection.model} is already installed in {selection.link}",
            flush=True,
        )
        return True


def _prepare(selection):
    repo = target_repository(selection)
    try:
        target = upstream.inspect_target(
            repo, selection.variant, selection.language_only
        )
    except upstream.MissingCheckpoint:
        if not Path(repo.name).is_absolute():
            raise
        # An interrupted publication may have left only some checkpoint shards.
        # Materialization checks every existing byte against the resolved commit.
        repo = hub.Repository.resolve(
            selection.repo_id, selection.revision, installation=selection.link
        )
        target = upstream.inspect_target(
            repo, selection.variant, selection.language_only
        )
    family = families.family_for(target.config)
    draft, files = resolve_draft(selection, family)
    if not Path(repo.name).is_absolute():
        repo = library.materialize(
            repo, _source_files(repo, target.files.values()), selection.model_dir
        )
    downloaded = repo.download(set(target.files.values()))
    files = files | {path: downloaded[name] for path, name in target.files.items()}
    if not _reuse(selection, repo, target, draft, files):
        upstream.publish(selection, repo, target, draft, files)
