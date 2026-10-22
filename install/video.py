"""MiniMax-H3, a video model: assemblies of target format h3-video.

Its sources: MiniMaxAI/MiniMax-H3, the target, whose FL2VA holds the
transformer, the video and audio VAEs and the tokenizer; the VDN-H3 stage
checkpoints its DiT is converted from (OpenVDN/vdn-minimax-h3); the TAEH3
preview decoder (madebyollin/taehv); and its text encoder, Qwen3-VL-32B's
MLX checkpoint (mlx-community/Qwen3-VL-32B-Instruct-4bit). The vendored
engine's converters (install/video, runtime/video/VENDORED) turn the DiT,
the VAEs and TAEH3 into the engine's images once, into .converted/<key>
under the models root, keyed by the converters and the commits converted
(written into .converting/<key> first, outside the installation lock, and
published by one rename); assemblies link them as they link a GGUF's
derived metadata. Images the converters converted before the video VAE's
encoder joined them are completed with the encoder's alone. A local
directory of images converted before (dit-turbo/, dit-default/, vae/,
audio/, preview/ and fl2va/, as h3 converts them) prepares without
converting.

The paths an assembly links (assembly.py lists a language model's):
  weights/dit-turbo/, weights/dit-default/, weights/vae/, weights/audio/,
  weights/preview/       the engine's images (runtime/video/h3/h3.h)
  weights/fl2va/transformer/
                         FL2VA's transformer shards and index, whose adaLN
                         bases the engine's request preparation reads
  encoder/               the text encoder's config.json and shards
                         (runtime/video/TextEncoder.hpp)
  tokenizer/             FL2VA's tokenizer, which the server encodes prompts
                         with (server/videos.py)
  LICENSE                the MiniMax H3 Community License Agreement

Its record's sources are the target and the encoder; `video` adds the
license the user acknowledged and the images' origin: the converters'
identity and the commits they converted, or the local directory.
"""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

from . import hub, models, paths, upstream

FORMAT = "h3-video"
FAMILY = "MiniMax-H3"
TARGET = "MiniMaxAI/MiniMax-H3"
VDN = "OpenVDN/vdn-minimax-h3"
PREVIEW = "madebyollin/taehv"
ENCODER = "mlx-community/Qwen3-VL-32B-Instruct-4bit"
# The DiT's variants: the OpenVDN stage each converts from, and whether its
# turbo adapter applies (8 distilled steps; the default variant takes 50).
VARIANTS = {
    "turbo": ("stage-dmd-step-250", True),
    "default": ("stage-b-step-2000", False),
}
# The image directories the engine reads, and the files of each it reads
# (None: all of them); the video VAE's are the W8 decoder the engine runs and
# the encoder of a clip's first and last frames.
IMAGES = {
    "dit-turbo": None,
    "dit-default": None,
    "vae": (
        "manifest_w8.json",
        "vae_dec_w8.bin",
        "manifest_enc.json",
        "vae_enc_f16.bin",
    ),
    "audio": None,
    "preview": None,
}
CONVERTERS = paths.ROOT / "install" / "video"
# converters_identity() before the video VAE's encoder joined the converters
# (h3 5b55ab7): an entry they converted holds every image but the encoder's,
# and the same commits' entry starts from it (_complete).
ENCODERLESS_CONVERTERS = (
    "cab145e7ae06aa29dc0f4a6619cfbb643cca48529a487cf0ff37b55d2c29f6c5"
)
LICENSE_NAME = "MiniMax H3 Community License Agreement"
# The value of SPLASH_ACCEPT_LICENSE that acknowledges the license without a
# prompt, for an installation with no terminal to ask on.
ACCEPT = "minimax-h3"
NOTICE = """\
{model} is licensed under the {license}.
Its rights hold only within its Applicable Territory, and it incorporates an
Acceptable Use Policy. In its own words:

{territory}

Read the agreement, {path}, before you go on: Splash records that you
acknowledged it, and every use of the model is subject to it."""


def is_source(files):
    """Whether a repository's files are MiniMax-H3's: its FL2VA pipeline."""
    return "FL2VA/model_index.json" in files


def is_local(path: Path):
    """Whether a local directory holds MiniMax-H3: as its repository holds
    it, or as images converted before."""
    return (path / "FL2VA" / "model_index.json").is_file() or _is_converted(path)


def _is_converted(path: Path):
    return all((path / name).is_dir() for name in IMAGES) and (path / "fl2va").is_dir()


def _refuse_options(model, *options):
    """A video model takes none of a language model's source options."""
    if any(options):
        raise models.ModelError(
            f"{model} is a video model: it takes no variant, DFlash2 draft, vision "
            "projector or language-only option"
        )


def _territory(text):
    """The license's sentences on its territory, quoted."""
    markers = (
        "expressly limited",
        "Territory” means",
        "Territories” means",
        "outside the Applicable",
    )
    lines = [
        line.strip()
        for line in text.splitlines()
        if any(marker in line for marker in markers) and len(line) < 600
    ]
    return "\n".join(f"  {line}" for line in lines)


def acknowledge(models_root: Path, model, license_path: Path):
    """The license at license_path, acknowledged: once per license text, on
    a terminal's prompt or by SPLASH_ACCEPT_LICENSE, and recorded in the
    models root's .licenses.json; the record names it by its digest. Asked
    before any weight download."""
    text = license_path.read_text(errors="replace")
    digest = hashlib.sha256(text.encode()).hexdigest()
    record = models_root / ".licenses.json"
    try:
        acknowledged = models.read_json(record)
    except models.ModelError:
        acknowledged = {}
    if digest not in acknowledged:
        print(
            NOTICE.format(
                model=model,
                license=LICENSE_NAME,
                territory=_territory(text),
                path=license_path,
            ),
            flush=True,
        )
        if os.environ.get("SPLASH_ACCEPT_LICENSE") == ACCEPT:
            how = "SPLASH_ACCEPT_LICENSE"
        elif sys.stdin.isatty():
            answer = input("Type 'acknowledge' to acknowledge the agreement: ")
            if answer.strip().lower() != "acknowledge":
                raise models.ModelError(
                    f"the {LICENSE_NAME} was not acknowledged; nothing was installed"
                )
            how = "prompt"
        else:
            raise models.ModelError(
                f"{model} needs its {LICENSE_NAME} acknowledged: run splash serve on "
                f"a terminal, or set SPLASH_ACCEPT_LICENSE={ACCEPT}"
            )
        now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        acknowledged[digest] = {"license": LICENSE_NAME, "at": now, "by": how}
        models_root.mkdir(parents=True, exist_ok=True)
        with models.installation_lock(models_root):
            _write(record, models.json_bytes(acknowledged))
    return {"name": LICENSE_NAME, "sha256": digest}


def _write(path: Path, data: bytes):
    staging = path.with_name(path.name + ".partial")
    staging.write_bytes(data)
    os.replace(staging, path)


def _target_files(repo):
    """The target's files the model needs, by repository name: its license,
    FL2VA's transformer, VAEs and tokenizer, read from FL2VA's index."""
    index = models.read_json(
        repo.file("FL2VA/transformer/model.safetensors.index.json")
    )
    shards = {
        f"FL2VA/transformer/{name}" for name in index.get("weight_map", {}).values()
    }
    names = {
        "LICENSE",
        "FL2VA/transformer/model.safetensors.index.json",
        "FL2VA/video_vae/source/model.safetensors",
        "FL2VA/video_vae/source/config.json",
        "FL2VA/audio_vae/config.json",
        "FL2VA/audio_vae/model.safetensors",
        *shards,
        *(n for n in repo.files if n.startswith("FL2VA/tokenizer/")),
    }
    _require(repo, names)
    return names


def _vdn_files(repo):
    stages = tuple(f"{stage}/" for stage, _ in VARIANTS.values())
    names = {
        n
        for n in repo.files
        if n.startswith(stages)
        and n.split("/")[1]
        in ("adapters", "linear_branch", "metadata.json", "model_spec.json")
    }
    _require(repo, names, ("linear_branch/model.safetensors",))
    return names


def _encoder_files(repo):
    """The text encoder's config.json and safetensors shards at its root;
    its shard index is not read (TextEncoder.hpp)."""
    names = {n for n in repo.files if "/" not in n and n.endswith(".safetensors")}
    names.add("config.json")
    _require(repo, names)
    return names


def _require(repo, names, endings=()):
    missing = sorted(n for n in names if n not in repo.files)
    missing += [e for e in endings if not any(n.endswith(e) for n in names)]
    if missing or not names:
        raise models.ModelError(
            f"{repo.name} is missing files MiniMax-H3 needs: {', '.join(missing) or 'all'}"
        )


def _companion(name, link, installed, unreachable):
    """A companion source at its default branch's commit, as a draft is
    resolved: one Hub request, or without one the commit the installation
    records, or the Hub cache's."""
    recorded = None
    if installed and name == ENCODER:
        recorded = installed["sources"]["encoder"]["revision"]
    elif installed:
        recorded = installed["video"]["images"].get("commits", {}).get(name)
    return hub.Repository.resolve(
        name, installation=link, installed=recorded, unreachable=unreachable
    )


def _root(files):
    """The directory a source's downloaded files (name -> path) lie in."""
    name, path = next(iter(files.items()))
    return path.parents[len(Path(name).parts) - 1]


def converters_identity():
    """The vendored converters, by content: an edit converts again."""
    digest = hashlib.sha256()
    for path in sorted(CONVERTERS.rglob("*.py")):
        digest.update(path.relative_to(CONVERTERS).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _require_converter_environment():
    """The video VAE's converter computes its rotary tables with MLX and the
    MiniMax-H3 MLX port (H3_MLX_PORT): refused before any download."""
    port = os.environ.get("H3_MLX_PORT")
    if importlib.util.find_spec("mlx") is None or not (
        port and (Path(port) / "minimax_h3_mlx").is_dir()
    ):
        raise models.ModelError(
            "converting MiniMax-H3's video VAE needs MLX in Splash's environment and "
            "H3_MLX_PORT naming a checkout of https://github.com/PipeNetwork/"
            "minimax-h3-mlx; or prepare it from images converted before "
            "(install.local.prepare_local_model)"
        )


def _run(arguments, environment):
    print("Converting · " + " ".join(arguments[:2]), flush=True)
    command = [sys.executable, str(CONVERTERS / arguments[0]), *arguments[1:]]
    if subprocess.run(command, cwd=CONVERTERS, env=environment).returncode:
        raise models.ModelError("MiniMax-H3 conversion failed: " + " ".join(arguments))


@contextmanager
def converted(models_root: Path, fl2va: Path, vdn: Path, taeh3: Path, commits):
    """The engine's images of these sources, FL2VA (fl2va), the OpenVDN
    stages (vdn) and TAEH3's safetensors (taeh3), and their identity: the
    entry .converted/<key>, converted once and reused for the same
    converters and commits, held by its model.json (assembly.is_held) while
    the block runs, so that no collection removes it before an assembly
    links it. A conversion writes about 100 GB, holding no installation
    lock."""
    identity = {"converters": converters_identity(), "commits": commits}
    entry = _entry(models_root, identity)
    with _held(entry) or _convert(entry, identity, fl2va, vdn, taeh3):
        yield entry, identity


def _entry(models_root: Path, identity):
    return (
        models_root
        / ".converted"
        / hashlib.sha256(models.json_bytes(identity)).hexdigest()
    )


def _held(entry: Path):
    """The record of entry, held, if it is there."""
    try:
        record = (entry / "model.json").open("rb")
    except FileNotFoundError:
        return None
    fcntl.flock(record, fcntl.LOCK_SH)
    # A collection may have removed it before it was held.
    if os.fstat(record.fileno()).st_nlink:
        return record
    record.close()
    return None


def _convert(entry: Path, identity, fl2va: Path, vdn: Path, taeh3: Path):
    """Converts the images into .converting/<key>, one conversion of an
    entry at a time, and publishes them as entry by one rename with their
    record held, which it returns; a conversion that was interrupted left
    its staging, which is removed. The same commits' images without the
    encoder's, when there, are completed instead (_complete)."""
    encoderless = _entry(
        entry.parents[1], identity | {"converters": ENCODERLESS_CONVERTERS}
    )
    converting = entry.parents[1] / ".converting"
    converting.mkdir(parents=True, exist_ok=True)
    entry.parent.mkdir(exist_ok=True)
    with (converting / f"{entry.name}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if record := _held(entry):
            return record
        staging = converting / entry.name
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        try:
            if held := _held(encoderless):
                with held:
                    _complete(staging, encoderless, fl2va)
            else:
                _require_converter_environment()
                _convert_into(staging, fl2va, vdn, taeh3)
            _write(staging / "model.json", models.json_bytes(identity))
            record = _held(staging)
            os.rename(staging, entry)
            return record
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def _convert_into(staging: Path, fl2va: Path, vdn: Path, taeh3: Path):
    """The converters' steps (h3's README, Weights) into staging, which they
    see as their weights root (H3_WEIGHTS_DIR) with the stages in vdn/; they
    find FL2VA as the only snapshot of the Hub cache they are given."""
    cache = staging / ".cache"
    snapshot = cache / f"models--{TARGET.replace('/', '--')}" / "snapshots" / "source"
    snapshot.mkdir(parents=True)
    (snapshot / "FL2VA").symlink_to(fl2va)
    (staging / "vdn").symlink_to(vdn)
    environment = dict(
        os.environ,
        H3_WEIGHTS_DIR=str(staging),
        HF_HUB_CACHE=str(cache),
        PYTHONPATH=str(CONVERTERS),
    )
    for variant, (stage, turbo) in VARIANTS.items():
        out, source = staging / f"dit-{variant}", staging / "vdn" / stage
        adapter = ["--turbo", str(source / "adapters" / "turbo")] if turbo else []
        for step in ("blocks", "qkfix"):
            _run(["convert_dit.py", step, "--stage", str(source), *adapter, "--out", str(out)],
                 environment)  # fmt: skip
        _run(["convert_dit.py", "io", "--out", str(out)], environment)
        _run(
            ["convert_prep.py", "--variant", variant, "--weights", str(staging)],
            environment,
        )
    _run(["convert_vae.py", "--src", str(fl2va / "video_vae"), "--out",
          str(staging / "vae"), "--tag", "w8", "--i8", "qkv,o,1,2"], environment)  # fmt: skip
    _convert_encoder(staging, fl2va)
    _run(
        ["convert_audio.py", str(fl2va / "audio_vae"), str(staging / "audio")],
        environment,
    )
    _run(["convert_preview.py", str(taeh3), str(staging / "preview")], environment)
    for link in ("vdn", "fl2va"):
        (staging / link).unlink()
    shutil.rmtree(cache)


def _convert_encoder(staging: Path, fl2va: Path):
    """The video VAE encoder's images, of a clip's first and last frames."""
    _run(["convert_vae.py", "--src", str(fl2va / "video_vae"), "--out", str(staging / "vae"),
          "--encoder"], dict(os.environ, PYTHONPATH=str(CONVERTERS)))  # fmt: skip


def _complete(staging: Path, encoderless: Path, fl2va: Path):
    """The images of an entry the converters converted before the video
    VAE's encoder joined them, cloned into staging (APFS copies on write: no
    bytes are copied, and the clones outlive that entry's collection), and
    the encoder's."""

    def clone(source, destination):
        subprocess.run(["/bin/cp", "-c", source, destination], check=True)

    def record(directory, _names):
        return ["model.json"] if Path(directory) == encoderless else []

    shutil.copytree(encoderless, staging, copy_function=clone, dirs_exist_ok=True,
                    ignore=record)  # fmt: skip
    _convert_encoder(staging, fl2va)


def _files_of(directory: Path):
    """Each file in directory, links followed, by its name."""
    return {path.name: path for path in sorted(directory.iterdir()) if path.is_file()}


def _links(images: Path, fl2va: Path, encoder):
    """The assembly's files (assembly path -> source file) of the images in
    images, FL2VA's directory and the encoder's files."""
    files = {}
    for name, wanted in IMAGES.items():
        found = _files_of(images / name)
        for file in wanted or found:
            if file not in found:
                raise models.ModelError(f"{images / name} holds no {file}")
            files[f"weights/{name}/{file}"] = found[file]
    transformer = fl2va / "transformer"
    index = models.read_json(transformer / "model.safetensors.index.json")
    for shard in {
        "model.safetensors.index.json",
        *index.get("weight_map", {}).values(),
    }:
        files[f"weights/fl2va/transformer/{shard}"] = transformer / shard
    files |= {f"tokenizer/{n}": p for n, p in _files_of(fl2va / "tokenizer").items()}
    files["LICENSE"] = fl2va.parent / "LICENSE"
    files |= {f"encoder/{n}": p for n, p in encoder.items()}
    missing = sorted(name for name, path in files.items() if not path.is_file())
    if missing:
        raise models.ModelError("missing files: " + ", ".join(missing))
    return files


def _record(model, target, encoder, license, images):
    return {
        "version": 1,
        "model": model,
        "family": FAMILY,
        "target_format": FORMAT,
        "vision_format": "none",
        "sources": {"target": target, "encoder": encoder.identity()},
        "video": {"license": license, "images": images},
    }


def prepare(selection, target, installed):
    """Install selection's MiniMax-H3 from target, its repository as this
    start resolved it, and its companion sources at their default
    branches, as upstream.prepare installs a language model and its draft:
    the license acknowledged before any weight download, the images
    converted once, and the assembly published at the selection link. A
    ModelError refuses what cannot be installed."""
    _refuse_options(
        selection.model,
        selection.variant,
        selection.draft_model,
        selection.language_only,
    )
    with hub.as_model_errors(f"cannot install {selection.model}"):
        repos = {
            name: _companion(name, selection.link, installed, target.unreachable_reason)
            for name in (VDN, PREVIEW, ENCODER)
        }
        repos[TARGET] = target
        wanted = {
            TARGET: _target_files(target),
            VDN: _vdn_files(repos[VDN]),
            PREVIEW: {"safetensors/taeh3.safetensors"},
            ENCODER: _encoder_files(repos[ENCODER]),
        }
        _require(repos[PREVIEW], wanted[PREVIEW])
        license = acknowledge(
            selection.models_root, selection.model, target.file("LICENSE")
        )
        print(f"Installing {selection.model} as {FAMILY} ({FORMAT}).", flush=True)
        downloaded = {
            name: repos[name].download(names) for name, names in wanted.items()
        }
    fl2va = _root(downloaded[TARGET]) / "FL2VA"
    with converted(
        selection.models_root,
        fl2va,
        _root(downloaded[VDN]),
        downloaded[PREVIEW]["safetensors/taeh3.safetensors"],
        {name: repo.revision for name, repo in repos.items() if name != ENCODER},
    ) as (images, identity):
        record = _record(
            selection.model, target.identity(), repos[ENCODER], license, identity
        )
        files = _links(images, fl2va, downloaded[ENCODER])
        upstream.publish(selection.models_root, selection.link, record, files)
    print(
        f"Splash model {selection.model} is installed in {selection.link}", flush=True
    )


def prepare_local(
    target: Path, models_root: Path, link: Path, draft, mmproj, language_only
):
    """prepare_local_model of MiniMax-H3 in the directory target: images
    converted before, or its repository's layout, converted with the
    companion sources the Hugging Face cache holds at their default
    branches' commits; the text encoder comes from there too. No network
    request."""
    _refuse_options(str(target), draft, mmproj, language_only)
    unreachable = "preparing local files asks no Hub"
    encoder = hub.Repository.resolve(ENCODER, unreachable=unreachable)
    encoder_files = encoder.download(_encoder_files(encoder))
    converted_images = _is_converted(target)
    fl2va = (target / "fl2va").resolve() if converted_images else target / "FL2VA"
    license = acknowledge(models_root, str(target), fl2va.parent / "LICENSE")
    with ExitStack() as held:
        if converted_images:
            images, identity = target, {"directory": str(target)}
        else:
            vdn = hub.Repository.resolve(VDN, unreachable=unreachable)
            preview = hub.Repository.resolve(PREVIEW, unreachable=unreachable)
            images, identity = held.enter_context(
                converted(
                    models_root,
                    fl2va,
                    _root(vdn.download(_vdn_files(vdn))),
                    preview.download({"safetensors/taeh3.safetensors"})[
                        "safetensors/taeh3.safetensors"
                    ],
                    {TARGET: str(target), VDN: vdn.revision, PREVIEW: preview.revision},
                )
            )
        record = _record(
            str(target),
            {"repo": str(target), "revision": None},
            encoder,
            license,
            identity,
        )
        files = _links(images, fl2va, encoder_files)
        return upstream.publish(models_root, link, record, files)
