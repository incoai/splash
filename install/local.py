"""Prepare a model for the server from files already on this Mac.

prepare_local_model is the installer's public interface, for applications
that download models themselves and start the server on what it prepares,
with Splash's own Python and this checkout or release on PYTHONPATH:

    from install.local import ModelError, prepare_local_model

    model = prepare_local_model(
        "/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf",
        draft="/models/Qwen3.8-27B-DFlash2",
        mmproj="/models/Qwen3.8-27B-GGUF/mmproj-BF16.gguf",
    )

and then `python -m server.server MODEL --model OWNER/REPO --binary ENGINE`,
ENGINE being the engine of the same checkout or release (build/splash, or
engine/splash in a release). It checks the files with the rules the
installer and the engine hold a download to, and it makes no network
request.

Stability: prepare_local_model and ModelError keep their names, parameters,
defaults, results and errors across the releases of one major version; a
release may add keyword parameters whose defaults keep that behavior.
Nothing else under install/ is public, nor is the layout of the directory
prepared, which is for the server and the engine of the release that
prepared it: after an upgrade, prepare it again, as `splash serve` does at
every start, at little cost.
"""

from __future__ import annotations

import functools
import os
import signal
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from . import gguf, hub, models, paths, upstream
from .models import ModelError

__all__ = ["ModelError", "prepare_local_model"]


def prepare_local_model(
    target, draft=None, *, mmproj=None, language_only=False, models_dir=paths.MODELS
):
    """Prepare the target and its DFlash2 draft for the server, and return
    the directory the server loads, an absolute pathlib.Path.

    target: a GGUF file, any part of a split GGUF naming all of them, which
    lie beside it; or a directory whose root holds an MLX model (config.json,
    tokenizer files and a safetensors checkpoint) or else a single GGUF
    target, whole or split, beside any vision projector named mmproj.
    draft: the directory of a DFlash2 draft (config.json and safetensors
    weights) trained for the target's family, or "none" (models.NO_DRAFT)
    to decode one token a cycle without one, as a family no draft is
    released for does. Without one, the draft released for the family is
    taken from the Hugging Face cache (HF_HUB_CACHE), at the commit last
    recorded there for its default branch, as `hf download REPO` and
    `splash serve` record it.
    mmproj: a GGUF target's vision projector, a GGUF file, with which it
    serves images; a GGUF target serves text only without one, as one of a
    family without a vision tower (Flint) always does. An MLX target's
    vision tower is in its checkpoint, and it serves images unless
    language_only.
    language_only: serve text only.
    models_dir: where prepared models are kept, Splash's own models
    directory by default. Preparations there run one at a time, beside the
    installations `splash serve` makes.

    The directory links the files and never changes. Preparing the same
    files again returns it again: that reads their metadata and their sizes
    and times, never their content. The first preparation reads each file
    outside the Hugging Face cache once, for its digest. Once a file
    changes, or a download moves the draft's default branch, preparing again
    returns another directory: serve that one. The server holds the
    directory it serves until it exits; one a later preparation no longer
    returns is removed only once no server holds it. The server takes
    models_dir's installation lock to hold it when it starts, so it needs
    write access there.

    A ModelError, "cannot prepare TARGET: REASON", refuses the files Splash
    cannot serve without a download (a missing file, an unsupported format,
    a draft of another model), naming any file of them but the target in
    its reason; an OSError is a failure of the file system.
    """
    target = _path(target)
    if draft is not None and draft != models.NO_DRAFT:
        draft = _path(draft)
    mmproj = None if mmproj is None else _path(mmproj)
    models_root = Path(models_dir).expanduser().resolve()
    try:
        return _prepare(target, draft, mmproj, language_only, models_root)
    except ModelError as error:
        raise type(error)(f"cannot prepare {target}: {error}") from None


def _prepare(target, draft, mmproj, language_only, models_root):
    """prepare_local_model, of its paths made absolute."""
    if mmproj and language_only:
        raise ModelError(
            f"{mmproj} is a vision projector, which a model that serves text "
            "only (language_only) does not load"
        )
    if draft and draft != models.NO_DRAFT and not draft.is_dir():
        raise ModelError(
            f"draft {draft} is not a directory: a DFlash2 draft is the "
            "directory of its config.json and safetensors weights"
        )
    if mmproj and not mmproj.is_file():
        raise ModelError(
            f"mmproj {mmproj} is not a file: a vision projector is a GGUF file"
        )
    link = models.local_selection_link(
        models_root, target, draft, mmproj, language_only
    )
    with tempfile.TemporaryDirectory(prefix="splash-target-") as scratch:
        source, inspected = _inspect(target, mmproj, language_only, Path(scratch))
        repo = _draft(inspected.family, draft)
        files = (
            {} if repo is upstream.NO_DRAFT else upstream.draft_files(repo, inspected)
        )
    files |= {path: source.paths[name] for path, name in inspected.files.items()}
    record = upstream.assembly_record(str(target), inspected, source, repo)
    return upstream.publish(models_root, link, record, files)


def _path(value) -> Path:
    """value as an absolute path whose parent is resolved: a file's own name
    stays, as a Hub cache file's link to its blob does."""
    path = Path(os.path.abspath(Path(value).expanduser()))
    return path.parent.resolve() / path.name


def _inspect(target, mmproj, language_only, scratch):
    """target's files (hub.Repository) and the target upstream finds in
    them, its configuration and metadata written into scratch."""
    if target.is_dir():
        repo = hub.Repository.local_directory(target)
        # The files at its root are the target's: a safetensors file in a
        # subdirectory, a draft's say, makes no MLX model.
        if any("/" not in n and n.endswith(".safetensors") for n in repo.files):
            if mmproj:
                raise ModelError(
                    f"{mmproj} is a GGUF target's vision projector; an MLX "
                    "target keeps its vision tower in its checkpoint"
                )
            try:
                return repo, upstream.mlx_target(repo, language_only)
            except models.VisionError as error:
                raise models.VisionError(
                    f"{error}; pass language_only=True to serve text only"
                ) from None
        directory = target
    elif target.is_file():
        if not target.name.endswith(".gguf"):
            raise ModelError(
                "it is not a GGUF file; a target is a GGUF file or a model directory"
            )
        directory = target.parent
    else:
        raise ModelError("it does not exist")
    # The last part a split GGUF's variants share under a name of its own, as
    # DeepSeek-V4.1's Engram tables, is no target; where one lies at the
    # root, the split GGUF beside it may lack its last part, which
    # upstream.gguf_target finds by its header among them.
    shared = {
        path.name: path
        for path in sorted(directory.iterdir())
        if _shared_part(directory, path.name)
    }
    if target.is_dir():
        names = _gguf_in(
            {n for n in repo.files if n not in shared}, shared_last=bool(shared)
        )
    else:
        names = _split_parts(target, shared_last=bool(shared))
    files = {name: directory / name for name in names} | shared
    projector = None
    if mmproj:
        if mmproj.name in files:
            raise ModelError(f"{mmproj} is named as one of the target's files")
        files[mmproj.name] = mmproj
        projector = functools.partial(_projector, mmproj)
    repo = hub.Repository.local_files(str(target), files)
    try:
        inspected = upstream.gguf_target(repo, names, scratch, projector=projector)
    except models.VisionError as error:
        raise models.VisionError(f"{error}; omit mmproj to serve text only") from None
    if mmproj and not inspected.family.vision:
        raise ModelError(
            f"{mmproj} is a vision projector, which a {inspected.family.name} "
            "model, without a vision tower, does not load"
        )
    return repo, inspected


def _shared_part(directory, name):
    """Whether the file name of directory is a GGUF at its root, not named as
    a split GGUF's part, whose header makes it a later part of one: the last
    part the variants of a split GGUF share under a name of its own
    (gguf.last_part), as DeepSeek-V4.1's Engram tables."""
    path = directory / name
    if (
        "/" in name
        or not name.endswith(".gguf")
        or upstream.SPLIT_NAME.search(name)
        or not path.is_file()
    ):
        return False
    try:
        part = gguf.Metadata(path).values.get("split.no")
    except (ModelError, OSError):
        return False
    return type(part) is int and part > 0


def _model_of(name):
    """The model a GGUF file holds: NAME for a split GGUF's part,
    NAME-0000K-of-0000N.gguf, else the file's own name."""
    match = upstream.SPLIT_NAME.search(name)
    return name[: match.start()] if match else name


def _gguf_in(files, *, shared_last=False):
    """The files of the one GGUF target at the root of a directory of files,
    in split order, whatever vision projectors (named mmproj) lie beside it;
    with shared_last, a split GGUF's but its last (upstream.split_order)."""
    candidates = [
        name
        for name in upstream.root_ggufs(files)
        if not upstream.projector_named(name)
    ]
    if not candidates:
        raise ModelError(
            "it holds no model at its root: no safetensors checkpoint, and no "
            "GGUF file but vision projectors (named mmproj)"
        )
    if len({_model_of(name) for name in candidates}) > 1:
        raise ModelError(
            f"it holds several GGUF targets ({', '.join(candidates)}); pass the "
            "target's file"
        )
    if _model_of(candidates[0]) == candidates[0]:
        return candidates
    return upstream.split_order(candidates, "its split GGUF", shared_last=shared_last)


def _split_parts(target, *, shared_last=False):
    """The files of the GGUF target, in split order: target alone, or every
    part of the split GGUF it is one part of, beside it (with shared_last,
    every part but the last, upstream.split_order)."""
    model = _model_of(target.name)
    if model == target.name:
        return [target.name]
    parts = [
        path.name
        for path in target.parent.iterdir()
        if path.is_file()
        and path.name.endswith(".gguf")
        and _model_of(path.name) == model
    ]
    return upstream.split_order(
        parts, f"the split GGUF {model} in {target.parent}", shared_last=shared_last
    )


def _projector(mmproj, repo):
    """mmproj's name and header in repo, if it is a vision projector Splash
    loads (upstream.read_projector); a refusal names mmproj."""
    try:
        header, precision, kind = upstream.read_projector(repo, mmproj.name)
    except ModelError as error:
        raise models.VisionError(f"{mmproj}: {error}") from None
    if precision is None:
        raise models.VisionError(
            f"{mmproj} ({kind}) is not a vision projector Splash loads, a clip "
            "model of BF16, F32 or F16 weights"
        )
    return mmproj.name, header


def _draft(family, draft):
    """The draft's files: the directory draft, or else, without one, the
    draft released for family at the commit the Hugging Face cache records
    for its default branch, with no request to the Hub
    (hub.Repository.resolve). "none", and a family no draft is released for
    without one, install none (upstream.NO_DRAFT)."""
    if draft == models.NO_DRAFT or (not draft and family.draft_repo is None):
        return upstream.NO_DRAFT
    if draft:
        return hub.Repository.local_directory(draft)
    try:
        return hub.Repository.resolve(
            family.draft_repo,
            unreachable="no draft was given, and preparing local files asks no Hub",
        )
    except ModelError as error:
        raise ModelError(
            f"{error}; download {family.draft_repo}, or pass the directory of a "
            f"{family.name} DFlash2 draft as draft"
        ) from None


def _main(argv=None):
    """The launcher's helper, run by Splash's Python after setup. Its stdout
    is only prepare_local_model's result; preparation messages use stderr."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare local Splash model files")
    parser.add_argument("target")
    parser.add_argument("--models", type=Path, default=paths.MODELS)
    parser.add_argument("--draft-model")
    parser.add_argument("--mmproj")
    parser.add_argument("--language-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))
        with redirect_stdout(sys.stderr):
            prepared = prepare_local_model(
                args.target,
                args.draft_model,
                mmproj=args.mmproj,
                language_only=args.language_only,
                models_dir=args.models,
            )
        print(prepared)
    except (ModelError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
