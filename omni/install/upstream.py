"""Install the supported upstream models, as assemblies (assembly.py).

The Hub owns downloads and snapshots; this module inspects the target and
decides when the Hub is asked. The engine owns the model rules, which it
applies to the target's configuration before any weight download
(check_audio_model). A target's family is the one the engine finds its
configuration to describe (families.py); repository names play no part.

A start asks the Hub only for what the machine lacks: a verified
installation starts as it is, a model whole in the Hub cache is assembled
from it, and only a model in neither is downloaded, with the default draft
of a Qwen3-ASR-1.7B model when that is not cached either. A model once here
is never checked for a newer commit. The setup page decides what a start
downloads with the same functions (present, draft_present).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from server import serve_options

from . import assembly, families, hub, models

# The tokenizer files an MLX target may supply, linked when present.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
)


@dataclass(frozen=True)
class Target:
    """What the target repository supplies, known before any weight download."""

    # The record's target_format: "mlx-affine" or "mlx-bf16".
    format: str
    # The configuration the engine reads, the assembly's config.json.
    config: Path
    # The family the engine finds the configuration to describe.
    family: families.ModelFamily
    # Assembly path -> repository file, for every path a repository file
    # serves (the layout in assembly.py).
    files: dict[str, str]


def unsupported(name, reason):
    """The error refusing a repository Splash Omni does not serve."""
    return models.ModelError(
        f"{name} is not a model Splash Omni serves ({reason}); the supported "
        f"checkpoints are {families.supported_models_text()}"
    )


def check_audio_model(config):
    """The family the engine finds a speech or transcription model's
    configuration to describe, by the rules it holds the model to at every
    start (its omni-check command, given the directory holding config.json):
    config, the target's config.json in its snapshot. A ModelError gives the
    engine's reason when it refuses it. An engine built before the check
    is tolerated, with a warning: the family is then the one config.json's
    model_type names, and the engine checks the model when it serves it."""
    try:
        output = models.run_engine(["omni-check", str(config.parent)], "omni check")
    except models.UnknownCheck as error:
        models.warn(f"{error}; taking the family from config.json")
        name = families.AUDIO_MODEL_TYPES.get(
            models.read_json(config).get("model_type")
        )
    else:
        name = json.loads(output)["family"]
    if (family := families.named(name)) is None:
        raise models.ModelError(f"no audio model family is named {name}")
    return family


def inspect_target(repo):
    """What the target repository supplies, from metadata alone: its config
    and tokenizer files, and the family the engine finds its configuration
    to describe. A repository without an MLX checkpoint of a supported
    family is refused before any weight download."""
    if not any(n.endswith(".safetensors") for n in repo.files):
        raise unsupported(repo.name, "it has no safetensors checkpoint")
    required = {"config.json", "tokenizer_config.json"}
    # A tokenizer is tokenizer.json, or the vocabulary and merges it is
    # built from, as the audio repositories ship it.
    if not {"vocab.json", "merges.txt"} <= set(repo.files):
        required.add("tokenizer.json")
    if missing := required - repo.files:
        raise models.ModelError(
            f"target repository {repo.name} is missing: {', '.join(sorted(missing))}. "
            "Configuration and tokenizer must come from the target repository."
        )
    config = repo.file("config.json")
    model_type = models.read_json(config).get("model_type")
    if model_type not in families.AUDIO_MODEL_TYPES:
        raise unsupported(repo.name, f"its model_type is {model_type!r}")
    return _audio_target(repo, config)


def _config_files(repo):
    """The assembly paths of the configuration and tokenizer files an MLX
    repository supplies: config.json where the engine, the checkpoint
    loader and the tokenizer each read it, and the tokenizer files."""
    files = {
        path: "config.json"
        for path in ("config.json", "target/config.json", "tokenizer/config.json")
    }
    files |= {"tokenizer/" + n: n for n in TOKENIZER_FILES if n in repo.files}
    return files


def _audio_target(repo, config):
    """A speech or transcription model: every file of the snapshot under
    target/, the speech tokenizer's subdirectory included, as the audio
    engine loads the checkpoint in place."""
    family = check_audio_model(config)
    quantization = models.read_json(config).get("quantization_config")
    affine = (
        isinstance(quantization, dict)
        and quantization.get("mode", "affine") == "affine"
        and type(quantization.get("bits")) is int
    )
    files = _config_files(repo)
    for name in sorted(repo.files):
        if not models.is_safe_path(name):
            raise models.ModelError(f"unsupported file name in {repo.name}: {name}")
        files["target/" + name] = name
    return Target("mlx-affine" if affine else "mlx-bf16", config, family, files)


def installed(selection):
    """The record of selection's installation, if it verifies (assembly.verify,
    without the Hub) and this release serves its family; a ModelError or
    OSError says why it does not."""
    record = assembly.verify(selection.link)
    if families.named(record["family"]) is None:
        raise models.ModelError(f"no supported family is named {record['family']}")
    return record


def cached_commit(selection):
    """The commit of a whole snapshot in the Hub cache that selection's model
    is assembled from without the Hub (hub.cached_commit), else None."""
    return hub.cached_commit(selection.repo_id, selection.revision, selection.link)


def present(selection):
    """Whether selection's model starts without the Hub: installed, or whole
    in the Hub cache."""
    if models.installation_kind(selection.link) == models.ASSEMBLY:
        try:
            installed(selection)
            return True
        except (models.ModelError, OSError):
            pass
    return cached_commit(selection) is not None


def draft_present(draft):
    """Whether the Hub cache holds the draft whole, where the server finds it
    (hub.whole_snapshot)."""
    return hub.whole_snapshot(draft) is not None


def prepare(selection):
    """Start selection's model, installing what it needs.

    A verified installation starts as it is; else (none, or damaged) the
    model is assembled from a whole snapshot in the Hub cache; else the Hub
    is asked, with one request, for the commit selection's revision names,
    which is downloaded and assembled. Only that last step makes a Hub
    request: a model once here is not checked for a newer commit."""
    if models.installation_kind(selection.link) == models.ASSEMBLY:
        # Verified under the lock, where a concurrent installation may have
        # replaced it; its pins are repaired there too.
        with models.installation_lock(selection.models_root):
            try:
                record = installed(selection)
            except (models.ModelError, OSError) as error:
                print(f"Reinstalling {selection.model}: {error}", flush=True)
            else:
                hub.repair_pins(selection.link, assembly.pins(record))
                print(
                    f"Splash model {selection.model} is already installed in "
                    f"{selection.link}",
                    flush=True,
                )
                return
    if (commit := cached_commit(selection)) is not None:
        print(
            f"Using {selection.repo_id}@{commit[:12]} from the Hugging Face "
            "cache; the Hub is not asked.",
            flush=True,
        )
        repo = hub.Repository.cached(selection.repo_id, commit)
    else:
        repo = hub.Repository.resolve(selection.repo_id, selection.revision)
    _install(selection, repo)


def fetch_draft(selection):
    """Fetch the draft serve decodes selection's model with by default
    (serve_options.default_draft_model: Qwen3-ASR-1.7B's DFlash draft)
    into the Hub cache, beside the installed model.
    Nothing is fetched for other models or a draft already cached whole
    (draft_present). A failure warns: the server tries again at its start
    and otherwise serves without speculative decoding."""
    draft = serve_options.default_draft_model(selection.link)
    if draft is None or draft_present(draft):
        return
    print(
        f"Downloading {draft}, the speculative draft of {selection.model} "
        "(about 0.7 GB; --no-draft-model skips it).",
        flush=True,
    )
    try:
        serve_options.download_model(draft)
    except Exception as error:
        # Offline, unreachable or refused: whatever the Hub client raises
        # leaves the model installed.
        models.warn(
            f"cannot download the draft {draft}: {hub.reason(error)}; the "
            "server will serve without speculative decoding"
        )


def _install(selection, repo):
    """Assemble repo's target and publish the assembly at the selection
    link."""
    # Every Hub request but the listing (config.json, downloads) happens
    # here; none for a cached snapshot.
    with hub.as_model_errors(f"cannot install {selection.model}"):
        target = inspect_target(repo)
        print(
            f"Installing {selection.model} as {target.family.name} ({target.format}).",
            flush=True,
        )
        downloaded = repo.download(set(target.files.values()))
    files = {path: downloaded[name] for path, name in target.files.items()}
    record = {
        "version": 1,
        "model": selection.model,
        "family": target.family.name,
        "target_format": target.format,
        "sources": {"target": repo.identity()},
    }
    models_root = selection.models_root
    models_root.mkdir(parents=True, exist_ok=True)
    # Everything written under the models root is written under the lock.
    with models.installation_lock(models_root):
        # One record per linked file, however many assembly paths link it.
        records = {path: assembly.file_record(path) for path in set(files.values())}
        record["files"] = {name: records[path] for name, path in sorted(files.items())}
        # A new installation requires its pins before it is published; the
        # pins of a commit it replaces are retired once it is.
        pins = [
            hub.pin(snapshot, repo_id, selection.link)
            for snapshot, repo_id in assembly.pins(record)
        ]
        models.link_selection(
            selection.link, assembly.build(models_root, record, files)
        )
        hub.retire_other_pins(pins)
        assembly.collect_garbage(models_root)
