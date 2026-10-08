"""Verified physical downloads into an external OWNER/REPO model library.

Only requested files are fetched, at a resolved commit, outside the Hub cache.
Private provenance and a per-repository process lock live under root/.splash.
Existing files are never replaced. A failed publication may leave verified files;
retrying reuses them. The provenance record is published last, atomically.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from . import hub, models


def _private(root, repo_id):
    return root / ".splash" / hashlib.sha256(repo_id.encode()).hexdigest()


def _plain(root, relative, *, directory=False):
    """Reject symlink components before reading or writing a library path."""
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise models.ModelError(
                f"local model library contains a symlink: {current}"
            )
    if directory:
        current.mkdir(parents=True, exist_ok=True)
    return current


def _stamp(path):
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise models.ModelError(f"expected a regular model file: {path}")
    return [info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_dev, info.st_ino]


def _record(path):
    if path.is_symlink():
        raise models.ModelError(f"local model library contains a symlink: {path}")
    try:
        return models.read_json(path)
    except models.ModelError:
        if path.exists():
            raise
        return None


def revision(directory: Path) -> str | None:
    """Verified commit, or None for unmanaged/changed/mixed library contents.

    Content hashes are verified on download/adoption. Later checks compare the
    complete file listing plus size, mtime, ctime, device and inode to that record,
    avoiding reading many gigabytes on each startup. This is a local integrity
    check, not authentication against someone able to rewrite private records.
    """
    directory = Path(directory).absolute()
    root = directory.parent.parent
    repo_id = f"{directory.parent.name}/{directory.name}"
    try:
        _plain(root, repo_id)
        _plain(root, ".splash")
        record = _record(_private(root, repo_id).with_suffix(".json"))
        if not _valid_record(record, repo_id):
            return None
        files = record["files"]
        listing = {
            p.relative_to(directory).as_posix()
            for p in directory.rglob("*")
            if not p.is_dir()
        }
        if listing != set(files):
            return None
        for name, identity in files.items():
            if (
                not models.is_safe_path(name)
                or _stamp(_plain(directory, name)) != identity["stat"]
            ):
                return None
        return record["revision"]
    except (OSError, models.ModelError, KeyError, TypeError, ValueError):
        return None


def _valid_record(record, repo_id):
    return (
        isinstance(record, dict)
        and record.get("repo") == repo_id
        and models.is_hex_digest(record.get("revision"), 40)
        and isinstance(record.get("files"), dict)
    )


@contextmanager
def _locked(root, repo_id):
    _plain(root, ".splash", directory=True)
    path = _private(root, repo_id).with_suffix(".lock")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _expected(repo, name):
    """Prefer source metadata or the content-addressed Hub blob filename.

    A regular copied cache file lacking either has no authoritative digest:
    hashing it can verify faithful copying, but cannot prove Hub authenticity.
    Unidentifiable symlinks are rejected instead of adopting arbitrary data.
    """
    size, digest = repo.sizes.get(name, (None, None))
    if (
        isinstance(size, int)
        and size >= 0
        and (models.is_hex_digest(digest, 40) or models.is_hex_digest(digest, 64))
    ):
        return size, digest.lower()
    if repo.directory is not None:
        path = repo.directory / name
        resolved = path.resolve(strict=True)
        if (
            resolved.parent.name == "blobs"
            and resolved.parent.parent.name == hub.folder_name(repo.name)
            and (
                models.is_hex_digest(resolved.name, 40)
                or models.is_hex_digest(resolved.name, 64)
            )
        ):
            return resolved.stat().st_size, resolved.name.lower()
        if not path.is_symlink():
            return _stamp(path)[0], models.sha256(path)
    raise models.ModelError(f"missing size/hash metadata for {repo.name}/{name}")


def _verify(path, expected):
    size, digest = expected
    before = _stamp(path)
    if before[0] != size:
        raise models.ModelError(
            f"model file size mismatch; preserve or move the existing file: {path}"
        )
    hasher = (
        hashlib.sha256()
        if len(digest) == 64
        else hashlib.sha1(f"blob {size}\0".encode())
    )
    actual = models.hash_file(path, hasher)
    if actual != digest or _stamp(path) != before:
        raise models.ModelError(
            f"model file hash mismatch; preserve or move the existing file: {path}"
        )
    return {"size": size, "hash": digest, "stat": before}


def _fetch(repo, name, stage, expected):
    path = stage / name
    if repo.directory is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo.directory / name, path)
    else:
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(
                repo.name,
                name,
                revision=repo.revision,
                local_dir=stage,
            )
        )
        if downloaded != path:
            raise models.ModelError(
                f"download did not use requested local directory: {name}"
            )
    _plain(stage, name)
    _verify(path, expected)
    return path


def _publish(repo, names, root, previous):
    destination = _plain(root, repo.name, directory=True)
    expected = {name: _expected(repo, name) for name in names}
    existing = {
        name: _verify(_plain(destination, name), expected[name])
        for name in names
        if (destination / name).exists() or (destination / name).is_symlink()
    }
    with tempfile.TemporaryDirectory(
        prefix="download-", dir=root / ".splash"
    ) as temporary:
        stage = Path(temporary)
        staged = {
            name: _fetch(repo, name, stage, expected[name])
            for name in sorted(names - existing.keys())
        }
        for name, path in staged.items():
            target = _plain(destination, name)
            _plain(destination, str(Path(name).parent), directory=True)
            try:
                os.link(path, target)
            except FileExistsError:
                _verify(_plain(destination, name), expected[name])
            path.unlink()
        files = (previous or {}).get("files", {}) | {
            name: _verify(_plain(destination, name), expected[name]) for name in names
        }
        record = {"repo": repo.name, "revision": repo.revision, "files": files}
        pending = stage / "provenance.json"
        with pending.open("w") as output:
            json.dump(record, output, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(pending, _private(root, repo.name).with_suffix(".json"))
    return hub.Repository.local_directory(destination)


def materialize(repo: hub.Repository, names: set[str], root: Path) -> hub.Repository:
    """Copy/download selected, pinned source files into root/OWNER/REPO."""
    models.validate_repo_id(repo.name)
    if not models.is_hex_digest(repo.revision, 40):
        raise models.ModelError("local library downloads require a resolved Hub commit")
    names = frozenset(names)
    if not names:
        raise models.ModelError("no model files selected for download")
    for name in names:
        repo._require(name)
        if any(part.startswith(".") for part in Path(name).parts):
            raise models.ModelError(f"unsupported hidden model file path: {name}")
    root = Path(root).expanduser().resolve()
    with hub.as_model_errors(f"cannot install {repo.name} into {root}"):
        root.mkdir(parents=True, exist_ok=True)
        with _locked(root, repo.name):
            previous = _record(_private(root, repo.name).with_suffix(".json"))
            if previous is not None and not _valid_record(previous, repo.name):
                raise models.ModelError(
                    f"invalid local library provenance for {repo.name}"
                )
            if previous and previous["revision"] != repo.revision:
                raise models.ModelError(
                    f"{repo.name} already contains a different commit; use another model directory"
                )
            return _publish(repo, names, root, previous)
