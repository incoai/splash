"""Assemblies: the directories upstream models are served from.

An assembly is models/.resolved/<SHA-256 of its record>: links to its source
snapshot's files (hub.py), and model.json, the record naming its source and
the identity of every linked file. It never changes once published.
Selection links point to assemblies; one that no selection links and no
server holds is collected.

The paths an assembly links, and what reads each:
  config.json            the target configuration, the MLX config.json
                         (runtime/omni OmniConfig.mm)
  target/config.json     the MLX config.json again, as the checkpoint
                         directory's own config
  target/<any file>      the model's whole snapshot, speech_tokenizer/
                         included, which the audio engine loads in place
                         (runtime/omni)
  tokenizer/config.json  the same configuration again, from which the
                         server's AutoTokenizer.from_pretrained chooses the
                         tokenizer class
  tokenizer/<file>       the MLX tokenizer files (audio_server.py --tokenizer)
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from . import hub, models

# mlx-affine: projections quantized as MLX affine Q4; mlx-bf16: unquantized.
TARGET_FORMATS = ("mlx-affine", "mlx-bf16")
RECORD_KEYS = {"version", "model", "family", "target_format", "sources", "files"}
FILE_RECORD_KEYS = {"path", "bytes", "mtime_ns", "ctime_ns", "digest"}


def build(models_root: Path, record, files) -> Path:
    """The assembly of record, whose files map each assembly path to its
    source file: an existing one if it verifies, else built again."""
    encoded = models.json_bytes(record)

    def write(stage):
        for name, path in files.items():
            link = stage / name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(path.absolute())
        _write_durably(stage / "model.json", encoded)

    def check(assembly):
        verify(assembly)
        # Its name is its record's digest: a record changed since is damage.
        if (assembly / "model.json").read_bytes() != encoded:
            raise models.ModelError("resolved model record changed")

    destination = models_root / ".resolved" / hashlib.sha256(encoded).hexdigest()
    _reuse_or_write(destination, write, check)
    return destination


def _reuse_or_write(destination, write, check):
    """Reuse the entry at destination if check passes; else write it with
    write(stage) into a staging folder beside it, published by one rename and
    checked again. Call it under the installation lock, which serializes the
    writers of every entry."""
    if destination.exists():
        try:
            check(destination)
            return
        except (models.ModelError, OSError) as error:
            print(f"Rebuilding the damaged {destination}: {error}", flush=True)
            shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=models.ENTRY_STAGING, dir=destination.parent))
    try:
        write(stage)
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    check(destination)


def _write_durably(path, data):
    """Write a file of an entry, on disk before the entry is published."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def file_record(path: Path):
    """What model.json records of a linked file: its path, size and times,
    which verify compares on every start, and its content digest, which
    verify(full=True) recomputes (_digest_of)."""
    stat = path.stat()
    return {
        "path": str(path.absolute()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "digest": _digest(path),
    }


def _digest(path):
    """A Hub cache file's name in blobs/: the digest the Hub publishes for it,
    the Git blob SHA-1 of a small file (40 hex digits) or the SHA-256 of an
    LFS file (64). Any other file's SHA-256."""
    blob = path.resolve()
    if blob.parent.name == "blobs" and (
        models.is_hex_digest(blob.name, 40) or models.is_hex_digest(blob.name, 64)
    ):
        return blob.name
    return models.sha256(path)


def _digest_of(path, digest):
    """path's content digest in the scheme of digest (_digest)."""
    if len(digest) == 64:
        return models.sha256(path)
    size = path.stat().st_size
    return models.hash_file(path, hashlib.sha1(f"blob {size}\0".encode()))


def verify(assembly: Path, *, full=False):
    """The record of the assembly at assembly (an entry of .resolved, or a
    selection link to one) if it verifies: model.json has the shape build
    records, and each assembly path still links the file it records, with
    the recorded size and times (with full, the recorded content). Needs no
    Hub access."""
    record = models.read_json(assembly / "model.json")
    if not _well_formed(record):
        raise models.ModelError("invalid resolved model record")
    for name, entry in record["files"].items():
        path = assembly / name
        stat = path.stat()
        if (
            not path.is_symlink()
            or str(path.readlink()) != entry["path"]
            or stat.st_size != entry["bytes"]
            or stat.st_mtime_ns != entry["mtime_ns"]
            or stat.st_ctime_ns != entry["ctime_ns"]
        ):
            raise models.ModelError("resolved model file changed: " + name)
        if full and _digest_of(path, entry["digest"]) != entry["digest"]:
            raise models.ModelError("source content hash mismatch: " + name)
    return record


def _well_formed(record):
    sources = record.get("sources")
    files = record.get("files")
    return (
        set(record) == RECORD_KEYS
        and type(record["version"]) is int
        and record["version"] == 1
        and isinstance(record["model"], str)
        and isinstance(record["family"], str)
        and record["target_format"] in TARGET_FORMATS
        and isinstance(sources, dict)
        and set(sources) == {"target"}
        and all(
            isinstance(source, dict)
            and set(source) == {"repo", "revision"}
            and isinstance(source["repo"], str)
            and (
                source["revision"] is None
                or models.is_hex_digest(source["revision"], 40)
            )
            for source in sources.values()
        )
        and isinstance(files, dict)
        and all(
            models.is_safe_path(name)
            and isinstance(entry, dict)
            and set(entry) == FILE_RECORD_KEYS
            and isinstance(entry["path"], str)
            and all(
                type(entry[key]) is int for key in ("bytes", "mtime_ns", "ctime_ns")
            )
            and (
                models.is_hex_digest(entry["digest"], 40)
                or models.is_hex_digest(entry["digest"], 64)
            )
            for name, entry in files.items()
        )
    )


def pins(record):
    """The Hub snapshot of each source an assembly links files from, with its
    repository ID: the snapshots its installation pins. Each is found from a
    linked file's path, not the cache location: HF_HUB_CACHE may name another
    folder now than when the assembly was built."""
    sources = {
        (hub.folder_name(source["repo"]), source["revision"]): source["repo"]
        for source in record["sources"].values()
        if source["revision"] is not None
    }
    snapshots = {}
    for entry in record["files"].values():
        path = hub.snapshot_of(entry["path"])
        key = path and (path.parent.parent.name, path.name)
        if key in sources:
            snapshots[path] = sources[key]
    return sorted(snapshots.items())


def hold(link: Path, models_root: Path):
    """The directory a selection link serves from, and what holds it: for an
    assembly, its model.json opened under a shared lock. While that file is
    open, in this process or one that inherited it, collect_garbage keeps the
    assembly. The link is read under the installation lock, where collection
    runs, so there is no moment the assembly is neither linked nor held. A
    link that names no assembly is served from itself: (link, None)."""
    if models.installation_kind(link) != models.ASSEMBLY:
        return link, None
    with models.installation_lock(models_root):
        assembly = link.resolve(strict=True)
        record = (assembly / "model.json").open("rb")
        fcntl.flock(record, fcntl.LOCK_SH)
    return assembly, record


def is_held(assembly: Path) -> bool:
    """Whether a server holds the assembly (hold)."""
    try:
        with (assembly / "model.json").open("rb") as record:
            fcntl.flock(record, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except OSError:
        return False
    return False


def collect_garbage(models_root: Path):
    """Remove what no installation uses: assemblies that no selection links
    and no server holds, and staging an interrupted installation left. Call
    it under the installation lock, where all of these are written."""
    resolved = models_root / ".resolved"
    linked = {link.resolve() for link in models.selection_links(models_root)}
    for assembly in sorted(resolved.iterdir()) if resolved.is_dir() else ():
        if assembly not in linked and not is_held(assembly):
            shutil.rmtree(assembly)
    for stage in models_root.glob(f"*/{models.LINK_STAGING}*"):
        shutil.rmtree(stage)
