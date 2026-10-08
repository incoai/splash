#!/usr/bin/env python3

"""Install and verify the model a Splash selection names.

Terms the installer modules share:
- selection: what `--model OWNER/REPO[:VARIANT]` and its source options name
  (Selection); its selection link, under the models root, points to what
  serves it.
- assembly: the directory an upstream model is served from, of links to
  source snapshot files and its record, model.json (assembly.py), built and
  published by upstream.py. A legacy Splash package is served from its own
  Hub snapshot instead (legacy.py).
- source snapshot: a repository at one commit in the Hub cache, or a local
  draft directory (hub.py).

This module holds what the installers share, and the command line.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

if __name__ == "__main__" and not __package__:
    # Run as a script by make, the launcher and `python install/models.py`:
    # import siblings as the install package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "install"

# The model ID syntax, which --model shares with the server, in modules of
# the standard library alone: the launcher imports this before .venv exists.
from server import serve_options

from . import paths

MODELS = paths.MODELS
# The bound on one JSON metadata file the installer reads; such files are
# kilobytes, so a larger one is refused unread. The engine bounds the files it
# reads by its own rule.
MAX_JSON_BYTES = 4 * 1024 * 1024
# What installation_kind finds at a selection link.
ASSEMBLY, PACKAGE = "assembly", "package"
# Staging an interrupted installation leaves, which garbage collection
# removes: a selection link being replaced (beside the link), and an assembly
# or derived metadata entry being written (beside the published entries).
LINK_STAGING = ".prepare-"
ENTRY_STAGING = ".loading-"


class ModelError(RuntimeError):
    pass


def warn(message):
    """Report something the user may need to act on, on stderr."""
    print(f"Warning: {message}", file=sys.stderr, flush=True)


def run_engine(arguments, check):
    """The output of one of the engine's checks, `ENGINE ARGUMENTS`, which
    check names. A ModelError names an engine that cannot run, or that
    refuses the command line as one built before the check does, and what
    replaces it; else it gives the engine's refusal, its error: line without
    that prefix, or the whole report of an engine that ended before main()
    (dyld on an older macOS)."""
    engine = paths.BINARY
    replace = "reinstall Splash" if paths.PACKAGED else "rebuild it with make"
    try:
        result = subprocess.run(
            [str(engine), *arguments], capture_output=True, text=True
        )
    except OSError as error:
        raise ModelError(
            f"cannot run the engine {engine} ({error.strerror}); {replace}"
        ) from None
    if result.returncode == os.EX_USAGE:
        raise ModelError(f"the engine {engine} does not know this {check}; {replace}")
    if result.returncode:
        report = result.stderr.strip()
        refusals = [
            line.removeprefix("error: ")
            for line in report.splitlines()
            if line.startswith("error: ")
        ]
        raise ModelError(
            refusals[-1]
            if result.returncode > 0 and refusals
            else f"the engine's {check} failed: {report or f'status {result.returncode}'}"
        )
    return result.stdout


def is_hex_digest(value, length: int) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is not None
    )


def is_safe_path(name) -> bool:
    """Whether name is a plain relative POSIX path: no absolute, empty, "."
    or ".." part, no backslash or control character, and none of the
    characters Hub download patterns read as globs (* ? [ ])."""
    if not isinstance(name, str):
        return False
    path = PurePosixPath(name)
    return (
        bool(path.parts)  # "" and "." have none
        and not path.is_absolute()
        and path.as_posix() == name
        and ".." not in path.parts
        and not any(character in name for character in "\\*?[]")
        and all(ord(character) >= 32 for character in name)
    )


def validate_repo_id(value: str) -> str:
    try:
        return serve_options.check_repo_id(value)
    except ValueError as error:
        raise ModelError(str(error)) from None


def split_model_id(value: str) -> tuple[str, str | None]:
    """owner/repo[:variant] -> (repository ID, variant or None)."""
    try:
        return serve_options.split_model_id(value)
    except ValueError as error:
        raise ModelError(str(error)) from None


def parse_draft_model(value: str) -> str:
    # A local draft directory is recorded as an absolute path, so the
    # installation it selects does not depend on the working directory.
    if value and (local := Path(value).expanduser()).is_dir():
        return str(local.resolve())
    try:
        return validate_repo_id(value)
    except ModelError:
        raise argparse.ArgumentTypeError(
            "must be a local DFlash2 draft directory or a Hugging Face "
            "repository ID (owner/repo)"
        ) from None


def mmproj_path(value) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ModelError(f"--mmproj must name an existing GGUF file: {path}")
    return path


def parse_mmproj(value: str) -> Path:
    try:
        return mmproj_path(value)
    except (ModelError, OSError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def hash_file(path: Path, digest) -> str:
    """The hex digest of path's content fed to digest, a hashlib object."""
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    return hash_file(path, hashlib.sha256())


def read_json(path: Path):
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ModelError(f"JSON metadata is too large: {path}")
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ModelError(f"expected a JSON object in {path}")
    return value


def json_bytes(value) -> bytes:
    """Canonical JSON: the same value always encodes to the same bytes."""
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode()


def selection_link(
    models: Path,
    model_id: str,
    *,
    revision=None,
    language_only=False,
    draft_model=None,
    model_dir=None,
    mmproj=None,
) -> Path:
    """The selection link of model_id with these source options:
    OWNER/REPO[:VARIANT] under models for the model alone, else
    .selections/<hash of the selection>."""
    repo_id, variant = split_model_id(model_id)
    if (
        revision
        or language_only
        or draft_model
        or model_dir is not None
        or mmproj is not None
    ):
        values = [model_id, revision, language_only, draft_model]
        if model_dir is not None:
            values = [*values, str(Path(model_dir).expanduser().resolve())]
        if mmproj is not None:
            values.append(["mmproj", str(Path(mmproj).expanduser().resolve())])
        selection = json.dumps(values, separators=(",", ":"))
        return models / ".selections" / hashlib.sha256(selection.encode()).hexdigest()
    if variant is None:
        return models / repo_id
    return models / f"{repo_id}{serve_options.VARIANT_SEPARATOR}{variant}"


def selection_links(models: Path):
    """Every selection link under models, in the places selection_link names."""
    return [path for path in models.glob("*/*") if path.is_symlink()]


@dataclass(frozen=True)
class Selection:
    """What --model and its source options select, and its selection link."""

    model: str
    repo_id: str
    variant: str | None
    revision: str | None
    language_only: bool
    draft_model: str | None
    models_root: Path
    link: Path
    model_dir: Path | None = None
    download_draft: bool = False
    mmproj: Path | None = None

    @classmethod
    def of(
        cls,
        models_root,
        model,
        *,
        revision=None,
        language_only=False,
        draft_model=None,
        model_dir=None,
        download_draft=False,
        mmproj=None,
    ):
        repo_id, variant = split_model_id(model)
        models_root = Path(models_root).resolve()
        model_dir = (
            Path(model_dir).expanduser().resolve() if model_dir is not None else None
        )
        if mmproj is not None:
            if language_only:
                raise ModelError("--mmproj cannot be combined with --language-only")
            mmproj = mmproj_path(mmproj)
        link = selection_link(
            models_root,
            model,
            revision=revision,
            language_only=language_only,
            draft_model=draft_model,
            model_dir=model_dir,
            mmproj=mmproj,
        )
        return cls(
            model,
            repo_id,
            variant,
            revision,
            language_only,
            draft_model,
            models_root,
            link,
            model_dir,
            download_draft,
            mmproj,
        )


def installation_kind(link: Path):
    """What serves a selection link: an assembly (its model.json), a legacy
    Splash package (its manifest.json), or nothing (None)."""
    if (link / "model.json").exists():
        return ASSEMBLY
    if (link / "manifest.json").exists():
        return PACKAGE
    return None


@contextmanager
def installation_lock(models: Path):
    """Exclusive access to everything written under models."""
    lock_path = models / ".install.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                "Another Splash model installation is running; waiting...",
                flush=True,
            )
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def link_selection(link: Path, target: Path):
    """Point the selection link at target, atomically. Only a link is ever
    replaced: a directory at its path is someone's own files."""
    if link.exists() and not link.is_symlink():
        raise ModelError(f"refusing to replace non-symlink model path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f"{LINK_STAGING}{link.name}-", dir=link.parent)
    )
    temporary = stage / "model"
    try:
        os.symlink(target, temporary, target_is_directory=True)
        os.replace(temporary, link)
    finally:
        temporary.unlink(missing_ok=True)
        stage.rmdir()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Install Splash runtime weights")
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument(
        "--model",
        required=True,
        type=serve_options.parse_model_id,
        help="Hugging Face repository ID (owner/repo[:variant])",
    )
    parser.add_argument("--revision", help="optional upstream branch, tag or commit")
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="reuse or download OWNER/REPO files in this library",
    )
    parser.add_argument(
        "--download-draft",
        action="store_true",
        help="allow missing draft weights to be downloaded",
    )
    parser.add_argument(
        "--draft-model",
        type=parse_draft_model,
        help="override the automatically selected DFlash2 repository or local directory",
    )
    vision = parser.add_mutually_exclusive_group()
    vision.add_argument(
        "--mmproj",
        type=parse_mmproj,
        metavar="PATH",
        help="override a GGUF target's vision projector with a local BF16/F32 GGUF",
    )
    vision.add_argument(
        "--language-only",
        action="store_true",
        help="skip vision preparation and loading",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser(
        "download-draft",
        help="download the selected target's compatible draft, then exit",
    )
    commands.add_parser("verify").add_argument("--full", action="store_true")
    commands.add_parser("link", help="print the selection link")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    selection = Selection.of(
        args.models,
        args.model,
        revision=args.revision,
        language_only=args.language_only,
        draft_model=args.draft_model,
        model_dir=args.model_dir,
        download_draft=args.download_draft or args.command == "download-draft",
        mmproj=args.mmproj,
    )
    if args.command == "link":
        print(selection.link)
        return 0
    # The installers import this module, so it imports them once it exists.
    from . import assembly, legacy, local_models, upstream

    try:
        # The launcher starts this with the stop signals blocked, so that one
        # sent while it starts is not lost; it arrives here.
        signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))
        if args.command == "prepare":
            upstream.prepare(selection)
        elif args.command == "download-draft":
            local_models.download_draft(selection)
        else:
            kind = installation_kind(selection.link)
            if kind == ASSEMBLY:
                assembly.verify(selection.link, full=args.full)
            elif kind == PACKAGE:
                legacy.verify(selection.link, selection.repo_id, full=args.full)
            else:
                raise ModelError(f"{args.model} is not installed in {args.models}")
            print(
                f"Splash model {args.model} preflight passed "
                f"({'full' if args.full else 'quick'})."
            )
    except (ModelError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    # The other installer modules import this file by its module name. Run
    # that module, not this __main__ copy, so the ModelError they raise is the
    # one main() reports.
    import importlib

    installer = importlib.import_module("install.models")
    raise SystemExit(installer.main())
