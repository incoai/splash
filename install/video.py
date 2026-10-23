"""MiniMax-H3, a video model: assemblies of target format h3-video.

Splash serves it from a repository of its engine's own format, as it serves
a GGUF or MLX repository: an installation downloads the repository's files
and links them, converting nothing. Such a repository (REPOSITORY, which
Splash's documentation names; any other its marker describes serves too)
holds, at its root:

  splash-video.json      its marker (MARKER): the format h3-video and its
                         version, the format of its int8 weights (WEIGHTS),
                         its variants and the commits of the sources they
                         were converted from
  dit-<variant>/         the DiT of each variant, turbo and or default: its
                         int8 blocks (QUANT_VIDEO_W8 of
                         runtime/metal/abi/QuantFormat.h), QKFIX blocks,
                         globals, patch projections (io), request-independent
                         prep inputs, adaLN tables and manifest
  vae/                   the video VAE's W8 decoder, and its encoder of a
                         clip's first and last frames
  audio/, preview/       the audio VAE's decoder and TAEH3's
  tokenizer/             FL2VA's tokenizer, which the server encodes prompts
                         with (server/videos.py)
  LICENSE, NOTICE        the MiniMax H3 Community License Agreement, and the
                         notice it requires; README.md, its model card

dev/tools/publish_video_repo.py makes one from MiniMax's bf16 release
(RELEASE), which Splash does not serve: a model naming it is refused, with
the way to the engine's format. The text encoder, Qwen3-VL-32B's MLX
checkpoint (ENCODER), comes from its own repository at its default branch's
commit, as a language model's draft does.

The paths an assembly links (assembly.py lists a language model's):
  weights/dit-<variant>/, weights/vae/, weights/audio/, weights/preview/
                         the repository's images (runtime/video/model) of the
                         variants installed
  tokenizer/, LICENSE    the repository's
  encoder/               the text encoder's config.json and shards
                         (runtime/video/model/TextEncoder.hpp)

An installation holds the turbo variant, and the default one once asked for
(`--variants`, `splash generate --variant default`). Its record's sources
are the repository and the encoder; `video` adds the license the user
acknowledged and the variants. prepare_local_model takes a directory of the
repository's layout, or of h3's converted images (dit-<variant>/, vae/,
audio/, preview/ and fl2va/, a link to FL2VA, whose tokenizer and license it
takes), and links its files as they are.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import sys
from pathlib import Path

from . import hub, models, upstream

FORMAT = "h3-video"
FAMILY = "MiniMax-H3"
# The repository of MiniMax-H3 in the engine's format that Splash's
# documentation, help and tests name.
REPOSITORY = "bzzzy1234/MiniMax-H3-Splash"
# MiniMax's release, of bf16 weights, which dev/tools/publish_video_repo.py
# converts; its FL2VA pipeline names it.
RELEASE = "MiniMaxAI/MiniMax-H3"
RELEASE_INDEX = "FL2VA/model_index.json"
ENCODER = "mlx-community/Qwen3-VL-32B-Instruct-4bit"
# A repository of the engine's format: its marker, and the marker's version
# and weights' format this release serves (QUANT_VIDEO_W8 of
# runtime/metal/abi/QuantFormat.h, which the engine checks the DiT's
# manifest against).
MARKER = "splash-video.json"
VERSION = 1
WEIGHTS = "h3-w8a8-vdn-v1"
# The DiT's variants: turbo (8 distilled steps) and default (50).
VARIANTS = ("turbo", "default")
# The variants an installation holds unless asked for others.
DEFAULT_VARIANTS = ("turbo",)
# The adaLN tables of a variant's schedule in its DiT's directory, and its
# manifest, which names the weights' format.
ADALN = "adaln.safetensors"
DIT_MANIFEST = "manifest.json"
# What h3 converts that the engine does not read: the DiT's stream image.
UNREAD = ("stream.bin",)
# The image directories the engine reads beside the variants' dit-<variant>,
# and the files of each it reads (None: all of them); the video VAE's are
# the W8 decoder the engine runs and the encoder of a clip's first and last
# frames.
IMAGES = {
    "vae": (
        "manifest_w8.json",
        "vae_dec_w8.bin",
        "manifest_enc.json",
        "vae_enc_f16.bin",
    ),
    "audio": None,
    "preview": None,
}
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
    """Whether a repository's files are MiniMax-H3's: in the engine's format,
    which its marker names, or MiniMax's release."""
    return MARKER in files or RELEASE_INDEX in files


def is_local(path: Path):
    """Whether a local directory holds MiniMax-H3: as a repository of the
    engine's format or MiniMax's release holds it, or as h3's converted
    images."""
    return (
        (path / MARKER).is_file()
        or (path / RELEASE_INDEX).is_file()
        or _is_converted(path)
    )


def _is_converted(path: Path):
    return (
        all((path / name).is_dir() for name in IMAGES)
        and (path / "fl2va").is_dir()
        and any((path / f"dit-{v}").is_dir() for v in VARIANTS)
    )


def _refuse_release(model):
    raise models.ModelError(
        f"{model} is MiniMax's release of MiniMax-H3, which Splash serves only in "
        f"its engine's format: serve {REPOSITORY}, or convert the release with "
        "dev/tools/publish_video_repo.py and prepare the directory it writes "
        "(install.local.prepare_local_model)"
    )


def _marker(repo):
    """The marker of repo, a repository of the engine's format, which must
    describe one this release serves."""
    marker = models.read_json(repo.file(MARKER))
    described = isinstance(marker, dict) and (
        marker.get("format"),
        marker.get("version"),
        marker.get("weights"),
    ) == (FORMAT, VERSION, WEIGHTS)
    variants = marker.get("variants") if described else None
    if not (isinstance(variants, list) and variants and set(variants) <= set(VARIANTS)):
        raise models.ModelError(
            f"{repo.name}'s {MARKER} does not describe MiniMax-H3 as this release "
            f"serves it: {FORMAT} version {VERSION}, weights {WEIGHTS}, variants of "
            + ", ".join(VARIANTS)
        )
    return marker


def _variants(selection, installed, available):
    """The variants an installation holds: those selection asks for (the
    defaults without), and those it installed before that the repository
    still has."""
    asked = selection.variants or DEFAULT_VARIANTS
    if missing := [v for v in asked if v not in available]:
        raise models.ModelError(
            f"{selection.model} has no variant {missing[0]}: its variants are "
            + ", ".join(available)
        )
    before = (installed or {}).get("video", {}).get("variants", [])
    return [v for v in VARIANTS if v in asked or (v in before and v in available)]


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


def _encoder_files(repo):
    """The text encoder's config.json and safetensors shards at its root;
    its shard index is not read (TextEncoder.hpp)."""
    names = {n for n in repo.files if "/" not in n and n.endswith(".safetensors")}
    names.add("config.json")
    if missing := sorted(names - set(repo.files)):
        raise models.ModelError(
            f"{repo.name} is missing files MiniMax-H3 needs: {', '.join(missing)}"
        )
    return names


def repository_files(repo, variants):
    """The files of repo, of the repository's layout, an installation of
    variants links, by name: each variant's DiT, the images the engine reads,
    the tokenizer and the license."""
    images = {f"dit-{v}": None for v in variants} | IMAGES
    names = {"LICENSE"}
    required = {"LICENSE"}
    for directory, wanted in (images | {"tokenizer": None}).items():
        found = {
            n
            for n in repo.files
            if n.count("/") == 1
            and n.startswith(f"{directory}/")
            and n.rpartition("/")[2] not in UNREAD
        }
        names |= {f"{directory}/{n}" for n in wanted} if wanted else found
        required |= {f"{directory}/{n}" for n in wanted or ()}
        if directory.startswith("dit-"):
            required |= {f"{directory}/{ADALN}", f"{directory}/{DIT_MANIFEST}"}
        elif not wanted and not found:
            required.add(f"{directory}/")
    if missing := sorted(required - set(repo.files)):
        raise models.ModelError(
            f"{repo.name} is missing files MiniMax-H3 needs: {', '.join(missing)}"
        )
    return names


def local_files(path: Path):
    """The files of a local directory, by their names in the repository's
    layout: its own, or, of h3's converted images, FL2VA's tokenizer and
    license through their fl2va/ link."""
    fl2va = None if (path / MARKER).is_file() else (path / "fl2va").resolve()
    directories = {d: path / d for d in (*IMAGES, *(f"dit-{v}" for v in VARIANTS))}
    directories["tokenizer"] = (fl2va or path) / "tokenizer"
    files = {
        f"{directory}/{name}": file
        for directory, source in directories.items()
        if source.is_dir()
        for name, file in _files_of(source).items()
    }
    for name, file in (
        (MARKER, path / MARKER),
        ("LICENSE", (fl2va.parent if fl2va else path) / "LICENSE"),
    ):
        if file.is_file():
            files[name] = file
    return files


def _companion(name, link, installed, unreachable):
    """A companion source at its default branch's commit, as a draft is
    resolved: one Hub request, or without one the commit the installation
    records, or the Hub cache's."""
    recorded = installed and installed["sources"]["encoder"]["revision"]
    return hub.Repository.resolve(
        name, installation=link, installed=recorded, unreachable=unreachable
    )


def _listed(repo):
    """repo with its files listed: a commit the Hub did not list (a verified
    installation's, which stands in when the Hub cannot answer) is read from
    its snapshot in the Hub cache, which the installation pins."""
    if repo.files or repo.revision is None:
        return repo
    return hub.Repository.cached(
        repo.name, repo.revision, unreachable_reason=repo.unreachable_reason
    )


def _files_of(directory: Path):
    """Each file in directory, links followed, by its name."""
    return {path.name: path for path in sorted(directory.iterdir()) if path.is_file()}


def _links(files, encoder):
    """The assembly's files (assembly path -> source file) of a repository's
    files and the encoder's, by name: the images under weights/."""

    def linked(name):
        return (
            name
            if name == "LICENSE" or name.startswith("tokenizer/")
            else f"weights/{name}"
        )

    links = {linked(name): path for name, path in files.items()}
    links |= {f"encoder/{name}": path for name, path in encoder.items()}
    missing = sorted(name for name, path in links.items() if not path.is_file())
    if missing:
        raise models.ModelError("missing files: " + ", ".join(missing))
    return links


def _record(model, target, encoder, license, variants):
    return {
        "version": 1,
        "model": model,
        "family": FAMILY,
        "target_format": FORMAT,
        "vision_format": "none",
        "sources": {"target": target, "encoder": encoder.identity()},
        "video": {"license": license, "variants": list(variants)},
    }


def prepare(selection, target, installed):
    """Install selection's MiniMax-H3 from target, a repository of the
    engine's format as this start resolved it, and its text encoder at its
    default branch's commit, as upstream.prepare installs a language model
    and its draft: the license acknowledged before any weight download, and
    the files linked as they are downloaded, into the assembly published at
    the selection link. MiniMax's release is refused. A ModelError refuses
    what cannot be installed."""
    _refuse_options(
        selection.model,
        selection.variant,
        selection.draft_model,
        selection.language_only,
    )
    with hub.as_model_errors(f"cannot install {selection.model}"):
        target = _listed(target)
        if MARKER not in target.files:
            _refuse_release(selection.model)
        variants = _variants(selection, installed, _marker(target)["variants"])
        encoder = _listed(
            _companion(ENCODER, selection.link, installed, target.unreachable_reason)
        )
        names = repository_files(target, variants)
        encoder_names = _encoder_files(encoder)
        license = acknowledge(
            selection.models_root, selection.model, target.file("LICENSE")
        )
        print(
            f"Installing {selection.model} as {FAMILY} ({FORMAT}); "
            f"variants {', '.join(variants)}.",
            flush=True,
        )
        files = _links(target.download(names), encoder.download(encoder_names))
    record = _record(selection.model, target.identity(), encoder, license, variants)
    upstream.publish(selection.models_root, selection.link, record, files)
    print(
        f"Splash model {selection.model} is installed in {selection.link}", flush=True
    )


def prepare_local(
    target: Path, models_root: Path, link: Path, draft, mmproj, language_only
):
    """prepare_local_model of MiniMax-H3 in the directory target, of the
    repository's layout or h3's converted images, every variant it holds;
    the text encoder comes from the Hugging Face cache at its default
    branch's commit. MiniMax's release is refused. No network request."""
    _refuse_options(str(target), draft, mmproj, language_only)
    if not (target / MARKER).is_file() and not _is_converted(target):
        _refuse_release(str(target))
    repo = hub.Repository.local_files(str(target), local_files(target))
    if MARKER in repo.files:
        variants = _marker(repo)["variants"]
    else:
        variants = [v for v in VARIANTS if (target / f"dit-{v}").is_dir()]
    encoder = hub.Repository.resolve(
        ENCODER, unreachable="preparing local files asks no Hub"
    )
    names = repository_files(repo, variants)
    encoder_files = encoder.download(_encoder_files(encoder))
    license = acknowledge(models_root, str(target), repo.file("LICENSE"))
    record = _record(
        str(target), {"repo": str(target), "revision": None}, encoder, license, variants
    )
    files = _links(repo.download(names), encoder_files)
    return upstream.publish(models_root, link, record, files)
