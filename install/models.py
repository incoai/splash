#!/usr/bin/env python3

"""Verify and atomically install Splash packages from Hugging Face snapshots.

The native model descriptor validates architecture, tensors, headers, and
execution geometry before mapping weights.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

if __package__:
    from . import paths
else:
    import paths

MODELS = paths.MODELS
ALIGNMENT = 16384
HUB_ENDPOINT = "https://huggingface.co"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
TOKENIZER_FILES = {
    "chat_template.jinja",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
REPO_ID = re.compile(
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?/"
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?"
)
PACKAGE_FORMATS = {
    "splash-packed-q4": (3, "MDFL0006"),
    "splash-packed-q4-moe": (4, "MDFM0001"),
    "splash-packed-q8": (5, "MDFL0008"),
}


class ModelError(RuntimeError):
    pass


def is_hex_digest(value, length: int) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is not None
    )


def validate_repo_id(value: str) -> str:
    # Keep argument validation available before the Hub dependency is installed.
    if (
        not isinstance(value, str)
        or not REPO_ID.fullmatch(value)
        or "--" in value
        or ".." in value
        or value.endswith(".git")
    ):
        raise ModelError("model must be a full Hugging Face repository ID (owner/repo)")
    return value


def parse_repo_id(value: str) -> str:
    try:
        return validate_repo_id(value)
    except ModelError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    try:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ModelError(f"JSON metadata is too large: {path}")
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ModelError(f"expected a JSON object in {path}")
    return value


def validate_package_manifest(path: Path):
    manifest = read_json(path)
    format_ = manifest.get("format")
    format_name = format_.get("name") if isinstance(format_, dict) else None
    layout = PACKAGE_FORMATS.get(format_name) if isinstance(format_name, str) else None
    if (
        layout is None
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != layout[0]
        or not isinstance(manifest.get("model"), str)
        or not manifest["model"].strip()
        or not isinstance(manifest.get("execution_geometry"), dict)
    ):
        raise ModelError("repository is not a supported Splash runtime package")
    expected_format = {
        "section_alignment_bytes": ALIGNMENT,
        "target_layer_magic": layout[1],
        "draft_layer_magic": "MDFD0004",
        "vision_magic": "MDFV0001",
    }
    if any(
        type(format_.get(key)) is not type(value) or format_[key] != value
        for key, value in expected_format.items()
    ):
        raise ModelError("runtime package has an unsupported packed weight format")
    if layout[0] == 4:
        for key, architecture in (
            ("target", "qwen3_5_moe"),
            ("draft", "DFlash2DraftModel"),
        ):
            declaration = manifest.get(key)
            if (
                not isinstance(declaration, dict)
                or declaration.get("architecture") != architecture
            ):
                raise ModelError(
                    f"runtime package has an unsupported {key} architecture"
                )

    records = manifest.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ModelError("runtime package manifest has no artifact list")
    artifact_paths = set()
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "size", "sha256"}
            or not isinstance(record["path"], str)
        ):
            raise ModelError("runtime package manifest has an invalid artifact")
        pure = PurePosixPath(record["path"])
        if (
            pure.is_absolute()
            or not pure.parts
            or ".." in pure.parts
            or pure.as_posix() != record["path"]
            or any(character in record["path"] for character in "\\*?[]")
            or any(ord(character) < 32 for character in record["path"])
            or record["path"] == "manifest.json"
            or type(record["size"]) is not int
            or record["size"] <= 0
            or not is_hex_digest(record["sha256"], 64)
        ):
            raise ModelError("runtime package manifest has an invalid artifact")
        if record["path"] in artifact_paths:
            raise ModelError("runtime package artifact paths are not unique")
        if pure.suffix == ".bin" and record["size"] % ALIGNMENT:
            raise ModelError(
                f"runtime package packed file is unaligned: {record['path']}"
            )
        artifact_paths.add(record["path"])
    if any(
        parent.as_posix() in artifact_paths
        for name in artifact_paths
        for parent in PurePosixPath(name).parents
    ):
        raise ModelError("runtime package artifact paths overlap")
    target_layers, draft_layers = (64, 5) if layout[0] in (3, 5) else (40, 6)
    required_files = {
        "target/embedding.bin",
        "target/head.bin",
        "draft/model.bin",
        "vision/model.bin",
        *(f"target/layer-{index}.bin" for index in range(target_layers)),
        *(f"draft/layer-{index}.bin" for index in range(draft_layers)),
        *(f"tokenizer/{name}" for name in TOKENIZER_FILES),
    }
    missing = required_files - artifact_paths
    if missing:
        raise ModelError(
            "runtime package artifact list is missing: " + ", ".join(sorted(missing))
        )
    return manifest


def verify_artifacts(root: Path, manifest, *, full: bool):
    for record in manifest["artifacts"]:
        path = root / record["path"]
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise ModelError(f"installed artifact has the wrong size: {record['path']}")
        if path.suffix == ".bin" and record["size"] % ALIGNMENT:
            raise ModelError(f"installed packed file is unaligned: {record['path']}")
        if full and sha256(path) != record["sha256"].lower():
            raise ModelError(f"installed artifact checksum changed: {record['path']}")


def installed_root(models: Path, model_id: str) -> Path:
    return models / validate_repo_id(model_id)


def verify_installed(
    models: Path,
    *,
    model_id: str,
    full: bool,
):
    root = installed_root(models, model_id)
    manifest = validate_package_manifest(root / "manifest.json")
    verify_artifacts(root, manifest, full=full)
    return model_id


def _snapshot_revision(snapshot: Path, model_id: str) -> str:
    if (
        snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != "models--" + model_id.replace("/", "--")
        or not is_hex_digest(snapshot.name, 40)
    ):
        raise ModelError(
            "installed package is not a snapshot of the requested Hub repository"
        )
    return snapshot.name


def _retain_snapshot_ref(snapshot: Path, model_id: str, installation: Path):
    revision = _snapshot_revision(snapshot, model_id)
    # Each installation owns its references; Hub branch updates and other
    # installations must not unpin this installation's current weights.
    owner_path = installation.parent.resolve() / installation.name
    owner = hashlib.sha256(os.fsencode(owner_path)).hexdigest()
    ref = snapshot.parent.parent / "refs" / "splash" / owner / revision
    try:
        existing = ref.read_text()
    except FileNotFoundError:
        pass
    else:
        if existing != revision:
            raise ModelError(f"invalid installed snapshot reference: {ref}")
        return ref
    ref.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=ref.parent) as temporary:
        temporary.write(revision.encode())
        temporary.flush()
        try:
            os.link(temporary.name, ref)
        except FileExistsError:
            if ref.read_text() != revision:
                raise ModelError(f"invalid installed snapshot reference: {ref}")
    return ref


def _download_snapshot(model_id: str, token):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    options = {
        "repo_id": model_id,
        "repo_type": "model",
        "token": token or False,
        "endpoint": HUB_ENDPOINT,
    }
    for _ in range(3):
        info = HfApi(endpoint=HUB_ENDPOINT, token=token or False).model_info(
            model_id, revision="main", files_metadata=True
        )
        if not is_hex_digest(info.sha, 40):
            raise ModelError("Hub did not resolve the model to a snapshot commit")
        manifest_file = next(
            (item for item in info.siblings if item.rfilename == "manifest.json"), None
        )
        if manifest_file is None:
            raise ModelError("repository has no Splash runtime package manifest.json")
        if (
            type(manifest_file.size) is not int
            or not 0 < manifest_file.size <= MAX_MANIFEST_BYTES
        ):
            raise ModelError("runtime package manifest.json has an invalid size")
        # Resolve the named revision through the Hub cache before pinning every
        # artifact. A branch update between metadata and download retries before
        # any weights are downloaded.
        manifest_path = Path(
            hf_hub_download(filename="manifest.json", revision="main", **options)
        )
        revision = _snapshot_revision(manifest_path.parent, model_id)
        if revision == info.sha:
            break
    else:
        raise ModelError("Hub main changed repeatedly during installation; retry")

    options["revision"] = revision
    try:
        manifest = validate_package_manifest(manifest_path)
    except ModelError:
        manifest_path = Path(
            hf_hub_download(
                filename="manifest.json",
                force_download=True,
                **options,
            )
        )
        manifest = validate_package_manifest(manifest_path)
    manifest_sha = sha256(manifest_path)
    published = {item.rfilename: item for item in info.siblings}
    for record in manifest["artifacts"]:
        item = published.get(record["path"])
        if item is None or item.size != record["size"]:
            raise ModelError(f"Hub artifact does not match manifest: {record['path']}")
        lfs = getattr(item, "lfs", None)
        if lfs is not None and lfs.sha256.lower() != record["sha256"].lower():
            raise ModelError(
                f"Hub artifact hash does not match manifest: {record['path']}"
            )
    snapshot = Path(
        snapshot_download(
            allow_patterns=[
                "manifest.json",
                *(r["path"] for r in manifest["artifacts"]),
            ],
            **options,
        )
    )
    if sha256(snapshot / "manifest.json") != manifest_sha:
        raise ModelError("runtime package manifest changed during download")
    if _snapshot_revision(snapshot, model_id) != revision:
        raise ModelError("Hub returned a different runtime package revision")
    # Repair only corrupt cached artifacts. A manifest error is deterministic
    # and must not trigger a second download of all model weights.
    for record in manifest["artifacts"]:
        path = snapshot / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != record["size"]
            or sha256(path) != record["sha256"].lower()
        ):
            hf_hub_download(filename=record["path"], force_download=True, **options)
            verify_artifacts(snapshot, {"artifacts": [record]}, full=True)
    return snapshot.resolve()


def _cached_snapshot(model_id):
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(model_id, "manifest.json", revision="main")
    if not isinstance(path, str):
        return None
    snapshot = Path(path).parent
    try:
        _snapshot_revision(snapshot, model_id)
        manifest = validate_package_manifest(snapshot / "manifest.json")
        verify_artifacts(snapshot, manifest, full=True)
    except (ModelError, OSError):
        return None
    return snapshot.resolve()


def resolve_snapshot(model_id: str):
    validate_repo_id(model_id)
    try:
        from huggingface_hub import get_token
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as error:
        raise ModelError(
            "missing dependency huggingface_hub; reinstall Splash"
        ) from error
    token = os.environ.get("HF_TOKEN") or get_token()
    import httpx
    from huggingface_hub.errors import OfflineModeIsEnabled

    try:
        return _download_snapshot(model_id, token)
    except Exception as error:
        if isinstance(error, (OfflineModeIsEnabled, httpx.TransportError)):
            if cached := _cached_snapshot(model_id):
                print("Using a verified cached model while offline.", flush=True)
                return cached
        if isinstance(error, ModelError):
            raise
        message = str(error)
        if token:
            message = message.replace(token, "[redacted]")
        if (
            isinstance(error, HfHubHTTPError)
            and error.response is not None
            and error.response.status_code in (401, 403)
        ):
            message += (
                "; set HF_TOKEN or run 'hf auth login' with access to this repository"
            )
        raise ModelError(
            f"could not download Splash runtime package {model_id}@main: {message}"
        ) from error


@contextmanager
def installation_lock(models: Path):
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


def install_snapshot(snapshot: Path, destination: Path):
    if destination.exists() and not destination.is_symlink():
        raise ModelError(f"refusing to replace non-symlink model path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".prepare-{destination.name}-", dir=destination.parent)
    )
    temporary = stage / "model"
    try:
        os.symlink(snapshot, temporary, target_is_directory=True)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
        stage.rmdir()


def prepare(args):
    validate_repo_id(args.model)
    models = args.models.resolve()
    models.mkdir(parents=True, exist_ok=True)
    root = installed_root(models, args.model)
    with installation_lock(models):
        try:
            _snapshot_revision(root.resolve(), args.model)
            manifest = validate_package_manifest(root / "manifest.json")
            verify_artifacts(root, manifest, full=False)
        except (ModelError, OSError):
            if root.exists() and not root.is_symlink():
                raise ModelError(
                    f"cannot identify the local package at {root}; move it aside before installing"
                ) from None
            print(
                f"Installing {args.model}; missing artifacts will be downloaded.",
                flush=True,
            )
            snapshot = resolve_snapshot(args.model)
            ref = _retain_snapshot_ref(snapshot, args.model, root)
            install_snapshot(snapshot, root)
            manifest = validate_package_manifest(root / "manifest.json")
            verify_artifacts(root, manifest, full=False)
            print(f"Installed verified Splash model {args.model} in {root}")
        else:
            print(f"Splash model {args.model} is already installed in {root}")
            try:
                ref = _retain_snapshot_ref(root.resolve(), args.model, root)
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
                    raise
                print(
                    "Warning: the verified model can be used, but its Hub cache "
                    "reference could not be retained; protect this snapshot from "
                    f"external cache pruning: {error}",
                    file=sys.stderr,
                )
                return
        # Retire this installation's previous pins only after publishing and
        # verifying its new destination. Other installations own other folders.
        try:
            for previous in ref.parent.iterdir():
                if previous != ref and is_hex_digest(previous.name, 40):
                    previous.unlink()
        except OSError as error:
            # Keeping an old pin uses cache space but cannot invalidate the
            # verified installation or its successfully retained current pin.
            print(
                f"Warning: could not retire old Hub cache references: {error}",
                file=sys.stderr,
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Install Splash runtime weights")
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument(
        "--model",
        required=True,
        type=parse_repo_id,
        help="Hugging Face repository ID (owner/repo)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("verify").add_argument("--full", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args)
        else:
            selected = verify_installed(
                args.models.resolve(),
                model_id=args.model,
                full=args.full,
            )
            print(
                f"Splash model {selected} preflight passed "
                f"({'full' if args.full else 'quick'})."
            )
    except (ModelError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
