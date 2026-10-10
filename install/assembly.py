"""Assemblies: the directories upstream models are served from.

An assembly is models/.resolved/<SHA-256 of its record>: links to its source
snapshots' files (hub.py) and to metadata derived from a GGUF
(models/.metadata/<key>), and model.json, the record naming its sources and
the identity of every linked file. It never changes once published.
Selection links point to assemblies; one that no selection links and no
server holds is collected.

The paths an assembly links, and what reads each:
  config.json            the target configuration: the MLX config.json, or
                         the one derived from the GGUF (ModelDescriptor.mm
                         inspectSourceModel)
  target/config.json, vision/config.json
                         the MLX config.json again, as each checkpoint
                         directory's own config, read by engines up to 1.2,
                         which a release check runs on this installation
  target/<shard>         the MLX safetensors shards except those holding only
                         vision_tower.* (SafetensorsCheckpoint.mm)
  target/<name>.gguf     the GGUF target, or the parts of a split one
                         (GgufTarget.cpp findTargetGgufs)
  tokenizer/config.json  the same configuration again, from which the
                         server's AutoTokenizer.from_pretrained chooses the
                         tokenizer class
  tokenizer/<file>       the MLX tokenizer files, or tokenizer.json,
                         tokenizer_config.json and chat_template.jinja derived
                         from the GGUF (server.py --tokenizer)
  draft/config.json, draft/<name>.safetensors
                         the DFlash2 checkpoint (ModelDescriptor.mm,
                         DraftCheckpoint.cpp); none for a family without one
  vision/<shard>         the MLX shards holding vision_tower.*
                         (VisionLoader.cpp, through SafetensorsCheckpoint.mm)
  vision/mmproj.gguf     the GGUF vision projector (VisionLoader.cpp)
  mtp/<name>.gguf        Qwen3.8-Flash-Next's MTP head (GgufTarget.cpp
                         GgufMtpLoader), which drafts in place of a DFlash2
                         checkpoint
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from . import gguf, hub, models

GGUF_VISION = "vision/mmproj.gguf"
TARGET_FORMATS = ("mlx-affine", "gguf")
VISION_FORMATS = ("none", "safetensors", "gguf")
RECORD_KEYS = {
    "version",
    "model",
    "family",
    "target_format",
    "vision_format",
    "sources",
    "files",
}
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

    destination = models_root / ".resolved" / hashlib.sha256(encoded).hexdigest()
    _reuse_or_write(destination, write, verify)
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
    gguf_target = record.get("target_format") == "gguf"
    sources = record.get("sources")
    files = record.get("files")
    return (
        set(record) == RECORD_KEYS | ({"metadata"} if gguf_target else set())
        and type(record["version"]) is int
        and record["version"] == 1
        and isinstance(record["model"], str)
        and isinstance(record["family"], str)
        and record["target_format"] in TARGET_FORMATS
        and record["vision_format"] in VISION_FORMATS
        and (not gguf_target or models.is_hex_digest(record["metadata"], 64))
        and isinstance(sources, dict)
        and set(sources) in ({"target"}, {"target", "draft"})
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


def recorded_pins(link: Path):
    """pins of the assembly a selection link names, as its record states
    them whether or not the assembly still verifies; none without a record
    of the shape build writes."""
    try:
        record = models.read_json(link / "model.json")
    except models.ModelError:
        return []
    return pins(record) if _well_formed(record) else []


def _metadata_inputs(files):
    """The assembly paths of the GGUF files the derived metadata comes from,
    in order: the target, then the vision projector."""
    targets = [
        n for n in sorted(files) if n.startswith("target/") and n.endswith(".gguf")
    ]
    return targets + ([GGUF_VISION] if GGUF_VISION in files else [])


def _metadata_key(sources):
    """The .metadata entry derived from these source file records. The same
    content at another path or time is the same source; the adapter
    (gguf.py) and the tokenizers library that writes tokenizer.json are part
    of the derivation."""
    from importlib.metadata import version

    identity = {
        "sources": [{"bytes": s["bytes"], "digest": s["digest"]} for s in sources],
        "adapter": models.sha256(Path(gguf.__file__)),
        "tokenizers": version("tokenizers"),
    }
    return hashlib.sha256(models.json_bytes(identity)).hexdigest()


def metadata_key(file_records):
    """The key of the metadata this release derives from an assembly's GGUF
    files, given each assembly path's file record."""
    return _metadata_key(
        [file_records[name] for name in _metadata_inputs(file_records)]
    )


def derived_metadata(models_root: Path, files):
    """The key and the files, by assembly path, of the metadata derived from
    the GGUF files among files (assembly path -> source file): derived once,
    and again when its entry is damaged. Call it under the installation lock."""
    names = _metadata_inputs(files)
    inputs = [files[name] for name in names]
    sources = [file_record(path) for path in inputs]
    key = _metadata_key(sources)
    # A split target's metadata is its first file's (split order is name
    # order); the vision projector comes last.
    derive = [inputs[0]] + ([files[GGUF_VISION]] if GGUF_VISION in files else [])

    def write(stage):
        contents = gguf.derived_files(*derive)
        if [file_record(path) for path in inputs] != sources:
            raise models.ModelError("GGUF source changed while reading metadata")
        for name, data in contents.items():
            _write_durably(stage / name, data)
        hashes = {
            name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
        }
        _write_durably(stage / "files.json", models.json_bytes(hashes))

    entry = models_root / ".metadata" / key
    _reuse_or_write(entry, write, _check_metadata)
    derived = {name: entry / name for name in gguf.DERIVED_FILES}
    derived["tokenizer/config.json"] = entry / "config.json"
    return key, derived


def _check_metadata(entry):
    """Raise unless a metadata entry holds its files as its files.json lists them."""
    hashes = models.read_json(entry / "files.json")
    if set(hashes) != set(gguf.DERIVED_FILES):
        raise models.ModelError("invalid prepared GGUF metadata record")
    for name in gguf.DERIVED_FILES:
        if models.sha256(entry / name) != hashes[name]:
            raise models.ModelError("prepared GGUF metadata changed: " + name)


def hold(link: Path, models_root: Path):
    """The assembly a selection link serves from, and what holds it: its
    model.json opened under a shared lock. While that file is open, in this
    process or one that inherited it, collect_garbage keeps the assembly. The
    link is read under the installation lock, where collection runs, so there
    is no moment the assembly is neither linked nor held."""
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
    and no server holds, metadata entries the remaining assemblies do not
    link, and staging an interrupted installation left. Call it under the
    installation lock, where all of these are written."""
    resolved, derived = models_root / ".resolved", models_root / ".metadata"
    linked = {link.resolve() for link in models.selection_links(models_root)}
    used = set()
    for assembly in sorted(resolved.iterdir()) if resolved.is_dir() else ():
        if assembly not in linked and not is_held(assembly):
            shutil.rmtree(assembly)
            continue
        # A damaged record protects no metadata: its assembly is rebuilt.
        try:
            files = models.read_json(assembly / "model.json").get("files")
        except models.ModelError:
            continue
        for entry in files.values() if isinstance(files, dict) else ():
            path = Path(entry.get("path", "")) if isinstance(entry, dict) else None
            if path and path.is_relative_to(derived):
                used.add(path.relative_to(derived).parts[0])
    for entry in sorted(derived.iterdir()) if derived.is_dir() else ():
        if entry.name not in used:
            shutil.rmtree(entry)
    for stage in models_root.glob(f"*/{models.LINK_STAGING}*"):
        shutil.rmtree(stage)
