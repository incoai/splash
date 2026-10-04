"""Legacy Splash packages: prebuilt packed weights published on the Hub with a
manifest.json, which predate upstream loading and stay installable.

A package is installed as a selection link to its verified Hub snapshot,
pinned for that installation. The native model descriptor validates
architecture, tensors, headers and execution geometry before loading weights.
huggingface_hub is imported where it is used, for the reasons hub.py gives.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import NamedTuple

from . import families, hub, models

ALIGNMENT = 16384
# The tokenizer/ files a package ships.
PACKAGE_TOKENIZER_FILES = {
    "chat_template.jinja",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


class PackageFormat(NamedTuple):
    schema_version: int
    target_layer_magic: str
    # The family whose target and draft layout the format packs.
    family: str
    # Manifest section -> the architecture it must declare.
    declarations: dict


PACKAGE_FORMATS = {
    "splash-packed-q4": PackageFormat(3, "MDFL0006", "Qwen3.8-27B", {}),
    "splash-packed-q4-moe": PackageFormat(
        4,
        "MDFM0001",
        "Qwen3.6-35B-A3B",
        {"target": "qwen3_5_moe", "draft": "DFlash2DraftModel"},
    ),
}


def is_package_manifest(manifest) -> bool:
    """Whether a manifest.json names a Splash package format (another tool's
    manifest.json does not)."""
    format_ = manifest.get("format")
    return isinstance(format_, dict) and format_.get("name") in PACKAGE_FORMATS


def _validate_records(records, artifact_paths):
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "size", "sha256"}
            or not models.is_safe_path(record["path"])
            or record["path"] == "manifest.json"
            or type(record["size"]) is not int
            or record["size"] <= 0
            or not models.is_hex_digest(record["sha256"], 64)
        ):
            raise models.ModelError("runtime package manifest has an invalid artifact")
        if record["path"] in artifact_paths:
            raise models.ModelError("runtime package artifact paths are not unique")
        if (
            PurePosixPath(record["path"]).suffix == ".bin"
            and record["size"] % ALIGNMENT
        ):
            raise models.ModelError(
                f"runtime package packed file is unaligned: {record['path']}"
            )
        artifact_paths.add(record["path"])


def validate_manifest(path: Path):
    manifest = models.read_json(path)
    format_ = manifest.get("format")
    format_name = format_.get("name") if isinstance(format_, dict) else None
    layout = PACKAGE_FORMATS.get(format_name) if isinstance(format_name, str) else None
    if (
        layout is None
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != layout.schema_version
        or not isinstance(manifest.get("model"), str)
        or not manifest["model"].strip()
        or not isinstance(manifest.get("execution_geometry"), dict)
    ):
        raise models.ModelError("repository is not a supported Splash runtime package")
    expected_format = {
        "section_alignment_bytes": ALIGNMENT,
        "target_layer_magic": layout.target_layer_magic,
        "draft_layer_magic": "MDFD0004",
        "vision_magic": "MDFV0001",
    }
    if any(
        type(format_.get(key)) is not type(value) or format_[key] != value
        for key, value in expected_format.items()
    ):
        raise models.ModelError(
            "runtime package has an unsupported packed weight format"
        )
    for key, architecture in layout.declarations.items():
        declaration = manifest.get(key)
        if (
            not isinstance(declaration, dict)
            or declaration.get("architecture") != architecture
        ):
            raise models.ModelError(
                f"runtime package has an unsupported {key} architecture"
            )

    records = manifest.get("artifacts")
    if not isinstance(records, list) or not records:
        raise models.ModelError("runtime package manifest has no artifact list")
    artifact_paths = set()
    _validate_records(records, artifact_paths)
    if any(
        parent.as_posix() in artifact_paths
        for name in artifact_paths
        for parent in PurePosixPath(name).parents
    ):
        raise models.ModelError("runtime package artifact paths overlap")
    family = families.named(layout.family)
    target_layers = dict(family.signature)["num_hidden_layers"]
    required_files = {
        "target/embedding.bin",
        "target/head.bin",
        *(f"target/layer-{index}.bin" for index in range(target_layers)),
        "draft/model.bin",
        *(f"draft/layer-{index}.bin" for index in range(family.draft.layers)),
        "vision/model.bin",
        *(f"tokenizer/{name}" for name in PACKAGE_TOKENIZER_FILES),
    }
    missing = required_files - artifact_paths
    if missing:
        raise models.ModelError(
            "runtime package artifact list is missing: " + ", ".join(sorted(missing))
        )
    return manifest


def verify_artifacts(root: Path, manifest, *, full: bool):
    for record in manifest["artifacts"]:
        path = root / record["path"]
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise models.ModelError(
                f"installed artifact has the wrong size: {record['path']}"
            )
        if path.suffix == ".bin" and record["size"] % ALIGNMENT:
            raise models.ModelError(
                f"installed packed file is unaligned: {record['path']}"
            )
        if full and models.sha256(path) != record["sha256"].lower():
            raise models.ModelError(
                f"installed artifact checksum changed: {record['path']}"
            )


def installed_snapshot(link: Path) -> Path:
    """The Hub snapshot an installed package's selection link points to."""
    if not link.is_symlink():
        raise models.ModelError(
            f"installed model root is not a Splash installation: {link}"
        )
    return link.resolve()


def verify(link: Path, repo_id: str, *, full: bool):
    """Check the package at a selection link: a snapshot of repo_id whose
    manifest validates and whose artifacts match it (with full, by hash)."""
    hub.snapshot_commit(installed_snapshot(link), repo_id)
    verify_artifacts(link, validate_manifest(link / "manifest.json"), full=full)


def _download_snapshot(repo_id: str, token):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    options = {
        "repo_id": repo_id,
        "repo_type": "model",
        "token": token or False,
    }
    for _ in range(3):
        info = HfApi(token=token or False).model_info(
            repo_id, revision="main", files_metadata=True
        )
        if not models.is_hex_digest(info.sha, 40):
            raise models.ModelError(
                "Hub did not resolve the model to a snapshot commit"
            )
        manifest_file = next(
            (item for item in info.siblings if item.rfilename == "manifest.json"), None
        )
        if manifest_file is None:
            raise models.ModelError(
                "repository has no Splash runtime package manifest.json"
            )
        if (
            type(manifest_file.size) is not int
            or not 0 < manifest_file.size <= models.MAX_JSON_BYTES
        ):
            raise models.ModelError("runtime package manifest.json has an invalid size")
        # Resolve the named revision through the Hub cache before pinning every
        # artifact. A branch update between metadata and download retries before
        # any weights are downloaded.
        manifest_path = Path(
            hf_hub_download(filename="manifest.json", revision="main", **options)
        )
        revision = hub.snapshot_commit(manifest_path.parent, repo_id)
        if revision == info.sha:
            break
    else:
        raise models.ModelError(
            "Hub main changed repeatedly during installation; retry"
        )

    options["revision"] = revision
    try:
        manifest = validate_manifest(manifest_path)
    except models.ModelError:
        manifest_path = Path(
            hf_hub_download(
                filename="manifest.json",
                force_download=True,
                **options,
            )
        )
        manifest = validate_manifest(manifest_path)
    manifest_sha = models.sha256(manifest_path)
    records = manifest["artifacts"]
    published = {item.rfilename: item for item in info.siblings}
    for record in records:
        item = published.get(record["path"])
        if item is None or item.size != record["size"]:
            raise models.ModelError(
                f"Hub artifact does not match manifest: {record['path']}"
            )
        lfs = getattr(item, "lfs", None)
        if lfs is not None and lfs.sha256.lower() != record["sha256"].lower():
            raise models.ModelError(
                f"Hub artifact hash does not match manifest: {record['path']}"
            )
    snapshot = Path(
        snapshot_download(
            allow_patterns=["manifest.json", *(r["path"] for r in records)],
            **options,
        )
    )
    if models.sha256(snapshot / "manifest.json") != manifest_sha:
        raise models.ModelError("runtime package manifest changed during download")
    if hub.snapshot_commit(snapshot, repo_id) != revision:
        raise models.ModelError("Hub returned a different runtime package revision")
    # Repair only corrupt cached artifacts. A manifest error is deterministic
    # and must not trigger a second download of all model weights.
    for record in records:
        path = snapshot / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != record["size"]
            or models.sha256(path) != record["sha256"].lower()
        ):
            hf_hub_download(filename=record["path"], force_download=True, **options)
            verify_artifacts(snapshot, {"artifacts": [record]}, full=True)
    return snapshot.resolve()


def _cached_snapshot(repo_id):
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo_id, "manifest.json", revision="main")
    if not isinstance(path, str):
        return None
    snapshot = Path(path).parent
    try:
        hub.snapshot_commit(snapshot, repo_id)
        verify_artifacts(
            snapshot, validate_manifest(snapshot / "manifest.json"), full=True
        )
    except (models.ModelError, OSError):
        return None
    return snapshot.resolve()


def resolve_snapshot(repo_id: str):
    """The verified Hub snapshot of the package at repo_id's main branch,
    downloading what the cache lacks; offline, a verified cached one."""
    models.validate_repo_id(repo_id)
    try:
        token = hub.token()
    except ImportError as error:
        raise models.ModelError(
            "missing dependency huggingface_hub; reinstall Splash"
        ) from error
    import httpx
    from huggingface_hub.errors import OfflineModeIsEnabled

    try:
        return _download_snapshot(repo_id, token)
    except Exception as error:
        if isinstance(error, (OfflineModeIsEnabled, httpx.TransportError)):
            if cached := _cached_snapshot(repo_id):
                print("Using a verified cached model while offline.", flush=True)
                return cached
        if isinstance(error, models.ModelError):
            raise
        raise models.ModelError(
            "could not download Splash runtime package "
            f"{repo_id}@main: {hub.reason(error, token)}"
        ) from error


def prepare(selection):
    """Start the package the selection link names, or install it: a link to
    its verified Hub snapshot, pinned for this installation before it is
    published. The download runs outside the installation lock, which every
    start takes."""
    if selection.variant is not None:
        raise models.ModelError(
            "this runtime package has no variants; drop the :VARIANT suffix"
        )
    if selection.revision or selection.language_only or selection.draft_model:
        raise models.ModelError("source selection options require an upstream model ID")
    link = selection.link
    selection.models_root.mkdir(parents=True, exist_ok=True)
    with models.installation_lock(selection.models_root):
        try:
            verify(link, selection.repo_id, full=False)
        except (models.ModelError, OSError):
            if link.exists() and not link.is_symlink():
                raise models.ModelError(
                    f"cannot identify the local package at {link}; move it aside before installing"
                ) from None
        else:
            print(f"Splash model {selection.model} is already installed in {link}")
            hub.repair_pins(link, [(installed_snapshot(link), selection.repo_id)])
            return
    print(
        f"Installing {selection.model}; missing artifacts will be downloaded.",
        flush=True,
    )
    snapshot = resolve_snapshot(selection.repo_id)
    with models.installation_lock(selection.models_root):
        pin = hub.pin(snapshot, selection.repo_id, link)
        models.link_selection(link, snapshot)
        verify(link, selection.repo_id, full=False)
        print(f"Installed verified Splash model {selection.model} in {link}")
        hub.retire_other_pins([pin])
