#!/usr/bin/env python3
"""Make a repository of MiniMax-H3 in the video engine's format
(install/video.py) to upload to the Hugging Face Hub.

Its files are the engine's images in IMAGES, h3's layout (dit-<variant>/
with its adaLN tables, vae/, audio/, preview/ and fl2va/, a link to FL2VA):
converted before, or converted here (--convert) from MiniMax's bf16 release
and its companions at their default branches' commits, into IMAGES, of
which a conversion converts only what it lacks. OUT, a new directory on
IMAGES' APFS volume, receives clones of them (no bytes are copied), but for
the images' manifests, written again without the converting machine's
paths; FL2VA's tokenizer and license; and the marker, NOTICE and model card
written here. The converters are h3's (dev/tools/video); the video VAE's
needs MLX in Splash's environment and H3_MLX_PORT naming a checkout of
https://github.com/PipeNetwork/minimax-h3-mlx, and the engine (build/splash
video-adaln) tabulates each DiT's adaLN tables.

    .venv/bin/python dev/tools/publish_video_repo.py --images IMAGES \\
        [--convert [--variants turbo,default]] [--commit REPO=COMMIT ...] OUT
    hf upload OWNER/REPO OUT .
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from install import hub, models, video  # noqa: E402

VDN = "OpenVDN/vdn-minimax-h3"
PREVIEW = "madebyollin/taehv"
SOURCES = (video.RELEASE, VDN, PREVIEW)
# Each variant's OpenVDN stage, and whether its turbo adapter applies.
STAGES = {
    "turbo": ("stage-dmd-step-250", True),
    "default": ("stage-b-step-2000", False),
}
CONVERTERS = ROOT / "dev" / "tools" / "video"
TAEH3 = "safetensors/taeh3.safetensors"
MINIMAX_NOTICE = (
    "MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, "
    "Copyright © 2026 MiniMax. All Rights Reserved."
)
NOTICE = """\
{minimax}

This repository holds MiniMax H3 in the int8 format of Splash's video engine
({weights}, https://github.com/incoai/splash), converted from:

{sources}

Every file but those of tokenizer/ is modified: the weights are converted
from those repositories' (the DiT's and video VAE's linears rotated and
quantized to int8, the other tensors laid out for the engine's kernels), and
each DiT's adaLN tables computed from them.

The VDN-H3 weights are under the same agreement. TAEH3 (preview/), Copyright
(c) 2025 Ollin Boer Bohan, is under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
CARD = """\
---
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
base_model:
- {release}
- {vdn}
pipeline_tag: text-to-video
tags:
- splash
---

# MiniMax-H3 for Splash

MiniMax H3, text to video with audio, in the int8 format of
[Splash](https://github.com/incoai/splash)'s video engine, which serves it on
an M5 Mac with 48 GB through OpenAI's video API:

```bash
splash serve --model OWNER/REPO
```

Splash downloads the turbo variant (8 steps), and the default one (50 steps)
once asked for (`splash generate --variant default`), and loads the files as
they are. Its text encoder comes from
[{encoder}](https://huggingface.co/{encoder}).

## License

{minimax}

Its rights hold only within its Applicable Territory, which excludes the
European Union, the United Kingdom, the Republic of Korea and the United
States of America, and it incorporates an Acceptable Use Policy: read
[LICENSE](LICENSE) before you download or use these files. Splash asks you to
acknowledge it before it downloads them. [NOTICE](NOTICE) names the sources
and what was modified.

## Files

| Path | Holds |
| --- | --- |
| `{marker}` | the format ({format} version {version}), its weights' format, the variants and the sources' commits |
| {dits} | the DiT: 50 int8 blocks, 3 QKFIX blocks, globals, patch projections (`io`), request-independent prep inputs, adaLN tables and manifest |
| `vae/` | the video VAE's int8 decoder and fp16 encoder |
| `audio/`, `preview/` | the audio VAE's decoder and TAEH3's |
| `tokenizer/` | FL2VA's tokenizer, which encodes prompts |
| `LICENSE`, `NOTICE` | the license agreement and its notice |

The int8 weights ({weights}) are dense row-major [N][K] with one fp32 scale
per output row (absmax / 127, rounded half to even), their inputs rotated
by a Hadamard transform in groups of 128 with splitmix64 signs (seed
0x48335138); the engine quantizes the activations the same way, per token.

## Sources

{sources}
"""


def convert(images: Path, variants):
    """Convert what images lacks of the engine's images, with the DiT of each
    of variants, from the sources at their default branches' commits, and
    return those commits. FL2VA's transformer shards (62 GB) download to the
    Hugging Face cache; the turbo variant's images take 26 GB, the default
    one's 25 GB more."""
    _require_converter_environment()
    with hub.as_model_errors("cannot download MiniMax-H3's sources"):
        repos = {name: hub.Repository.resolve(name) for name in SOURCES}
        downloaded = {
            name: repos[name].download(names)
            for name, names in (
                (video.RELEASE, _release_files(repos[video.RELEASE])),
                (VDN, _vdn_files(repos[VDN], variants)),
                (PREVIEW, _require(repos[PREVIEW], {TAEH3})),
            )
        }
    fl2va = _root(downloaded[video.RELEASE]) / "FL2VA"
    images.mkdir(parents=True, exist_ok=True)
    _convert_into(
        images, fl2va, _root(downloaded[VDN]), downloaded[PREVIEW][TAEH3], variants
    )
    (images / "fl2va").unlink(missing_ok=True)
    (images / "fl2va").symlink_to(fl2va)
    return {name: repo.revision for name, repo in repos.items()}


def _release_files(repo):
    """The release's files a conversion reads, by name: FL2VA's transformer,
    read from its index, its VAEs, tokenizer and license."""
    index = models.read_json(
        repo.file("FL2VA/transformer/model.safetensors.index.json")
    )
    names = {
        "LICENSE",
        "FL2VA/transformer/model.safetensors.index.json",
        "FL2VA/video_vae/source/model.safetensors",
        "FL2VA/video_vae/source/config.json",
        "FL2VA/audio_vae/config.json",
        "FL2VA/audio_vae/model.safetensors",
        *(n for n in repo.files if n.startswith("FL2VA/tokenizer/")),
        *(f"FL2VA/transformer/{n}" for n in index.get("weight_map", {}).values()),
    }
    return _require(repo, names)


def _vdn_files(repo, variants):
    """The OpenVDN stages of the variants converted."""
    stages = tuple(f"{STAGES[variant][0]}/" for variant in variants)
    names = {
        n
        for n in repo.files
        if n.startswith(stages)
        and n.split("/")[1]
        in ("adapters", "linear_branch", "metadata.json", "model_spec.json")
    }
    _require(repo, names | {f"{s}linear_branch/model.safetensors" for s in stages})
    return names


def _require(repo, names):
    if missing := sorted(names - set(repo.files)):
        raise models.ModelError(
            f"{repo.name} is missing files MiniMax-H3 needs: {', '.join(missing)}"
        )
    return names


def _root(files):
    """The directory a source's downloaded files (name -> path) lie in."""
    name, path = next(iter(files.items()))
    return path.parents[len(Path(name).parts) - 1]


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
            "minimax-h3-mlx"
        )


def _run(arguments, environment):
    print("Converting · " + " ".join(arguments[:2]), flush=True)
    command = [sys.executable, str(CONVERTERS / arguments[0]), *arguments[1:]]
    if subprocess.run(command, cwd=CONVERTERS, env=environment).returncode:
        raise models.ModelError("MiniMax-H3 conversion failed: " + " ".join(arguments))


def _convert_into(images: Path, fl2va: Path, vdn: Path, taeh3: Path, variants):
    """What images lacks (h3's README, Weights): the decoders', the video VAE
    encoder's, and each variant's DiT and its adaLN tables. The converters
    see images as their weights root (H3_WEIGHTS_DIR) with the stages in
    vdn/, and find FL2VA as the only snapshot of the Hub cache they are
    given; an interrupted conversion's links and cache are replaced."""
    cache = images / ".cache"
    shutil.rmtree(cache, ignore_errors=True)
    for link in ("vdn", "fl2va"):
        (images / link).unlink(missing_ok=True)
    snapshot = cache / hub.folder_name(video.RELEASE) / "snapshots" / "source"
    snapshot.mkdir(parents=True)
    (snapshot / "FL2VA").symlink_to(fl2va)
    (images / "vdn").symlink_to(vdn)
    environment = dict(
        os.environ,
        H3_WEIGHTS_DIR=str(images),
        HF_HUB_CACHE=str(cache),
        PYTHONPATH=str(CONVERTERS),
    )
    try:
        if (
            not all((images / name).is_dir() for name in video.IMAGES)
            or not (images / "vae" / "vae_dec_w8.bin").is_file()
        ):
            _convert_decoders(images, fl2va, taeh3, environment)
        if not (images / "vae" / "vae_enc_f16.bin").is_file():
            _convert_encoder(images, fl2va)
        for variant in variants:
            if not (images / f"dit-{variant}").is_dir():
                _convert_dit(images, variant, environment)
            if not (images / f"dit-{variant}" / video.ADALN).is_file():
                _tabulate_adaln(images, variant, fl2va)
    finally:
        for link in ("vdn", "fl2va"):
            (images / link).unlink(missing_ok=True)
        shutil.rmtree(cache)


def _convert_decoders(images: Path, fl2va: Path, taeh3: Path, environment):
    """The video VAE's W8 decoder, the audio VAE's and TAEH3's images."""
    _run(["convert_vae.py", "--src", str(fl2va / "video_vae"), "--out",
          str(images / "vae"), "--tag", "w8", "--i8", "qkv,o,1,2"], environment)  # fmt: skip
    _run(
        ["convert_audio.py", str(fl2va / "audio_vae"), str(images / "audio")],
        environment,
    )
    _run(["convert_preview.py", str(taeh3), str(images / "preview")], environment)


def _convert_encoder(images: Path, fl2va: Path):
    """The video VAE encoder's images, of a clip's first and last frames."""
    _run(["convert_vae.py", "--src", str(fl2va / "video_vae"), "--out", str(images / "vae"),
          "--encoder"], dict(os.environ, PYTHONPATH=str(CONVERTERS)))  # fmt: skip


def _convert_dit(images: Path, variant, environment):
    """The variant's DiT, dit-<variant>: its blocks, QKFIX blocks, patch
    projections and request-independent prep inputs."""
    stage, turbo = STAGES[variant]
    out, source = images / f"dit-{variant}", images / "vdn" / stage
    adapter = ["--turbo", str(source / "adapters" / "turbo")] if turbo else []
    for step in ("blocks", "qkfix"):
        _run(["convert_dit.py", step, "--stage", str(source), *adapter, "--out", str(out)],
             environment)  # fmt: skip
    _run(["convert_dit.py", "io", "--out", str(out)], environment)
    _run(
        ["convert_prep.py", "--variant", variant, "--weights", str(images)],
        environment,
    )


def _tabulate_adaln(images: Path, variant, fl2va: Path):
    """The adaLN tables of the variant's schedule (dit-<variant>/ADALN), from
    FL2VA's transformer shards, on the engine."""
    print(f"Converting · adaLN tables of dit-{variant}", flush=True)
    models.run_engine(
        ["video-adaln", str(images), variant, str(fl2va / "transformer")],
        "conversion of the adaLN tables",
    )


def release_commit(images: Path):
    """The commit of the release whose FL2VA images' fl2va/ links to in a Hub
    cache, or None."""
    snapshot = (images / "fl2va").resolve().parent
    try:
        return hub.snapshot_commit(snapshot, video.RELEASE)
    except models.ModelError:
        return None


def portable(value):
    """A manifest's JSON value without the converting machine's paths: each
    absolute path relative to its source, the part after a Hub snapshot's
    commit or after h3's weights/ or vdn/, else its name."""
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if not (isinstance(value, str) and value.startswith("/")):
        return value
    parts = PurePosixPath(value).parts
    for anchor, skipped in (("snapshots", 2), ("weights", 1), ("vdn", 1)):
        if anchor in parts[1:-1]:
            index = len(parts) - 1 - parts[::-1].index(anchor)
            return "/".join(parts[index + skipped :]) or parts[-1]
    return parts[-1]


def assemble(images: Path, out: Path, commits):
    """Write the repository of images, h3's converted images, into out, a
    new directory on their APFS volume; commits names each source's
    (SOURCES) commit they were converted from."""
    if missing := [name for name in SOURCES if not commits.get(name)]:
        raise models.ModelError(
            f"name the commit of {missing[0]} the images were converted from: "
            f"--commit {missing[0]}=COMMIT"
        )
    variants = [v for v in video.VARIANTS if (images / f"dit-{v}").is_dir()]
    if not variants:
        raise models.ModelError(f"{images} holds no DiT of MiniMax-H3")
    repo = hub.Repository.local_files(str(images), video.local_files(images))
    names = video.repository_files(repo, variants)
    out.mkdir(parents=True)
    for name in sorted(names):
        source, destination = repo.paths[name], out / name
        destination.parent.mkdir(exist_ok=True)
        if name.endswith(".json") and not name.startswith("tokenizer/"):
            manifest = portable(json.loads(source.read_text()))
            destination.write_text(json.dumps(manifest, indent=1) + "\n")
        else:
            subprocess.run(["/bin/cp", "-c", source, destination], check=True)
    sources = "\n".join(f"- {name}@{commits[name]}" for name in SOURCES)
    marker = {
        "format": video.FORMAT,
        "version": video.VERSION,
        "weights": video.WEIGHTS,
        "variants": variants,
        "sources": {name: commits[name] for name in SOURCES},
    }
    (out / video.MARKER).write_text(json.dumps(marker, indent=2) + "\n")
    (out / "NOTICE").write_text(
        NOTICE.format(minimax=MINIMAX_NOTICE, weights=video.WEIGHTS, sources=sources)
    )
    (out / "README.md").write_text(
        CARD.format(
            release=video.RELEASE,
            vdn=VDN,
            encoder=video.ENCODER,
            minimax=MINIMAX_NOTICE,
            marker=video.MARKER,
            format=video.FORMAT,
            version=video.VERSION,
            weights=video.WEIGHTS,
            dits=", ".join(f"`dit-{v}/`" for v in variants),
            sources=sources,
        )
    )
    return marker


def _commit(value):
    name, _, commit = value.partition("=")
    if name not in SOURCES or not models.is_hex_digest(commit, 40):
        raise argparse.ArgumentTypeError(
            f"{value}: expected REPO=COMMIT, REPO one of {', '.join(SOURCES)}"
        )
    return name, commit


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("out", type=Path, help="the repository's new directory")
    parser.add_argument(
        "--images", type=Path, required=True, help="the engine's images, h3's layout"
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="convert what IMAGES lacks from the sources' default branches",
    )
    parser.add_argument(
        "--variants",
        type=lambda value: tuple(v for v in value.split(",") if v),
        default=video.DEFAULT_VARIANTS,
        help="the DiT variants to convert (turbo, default; turbo alone without)",
    )
    parser.add_argument(
        "--commit",
        type=_commit,
        action="append",
        default=[],
        metavar="REPO=COMMIT",
        help="a source's commit the images were converted from",
    )
    arguments = parser.parse_args(argv)
    try:
        if unknown := sorted(set(arguments.variants) - set(STAGES)):
            raise models.ModelError(f"MiniMax-H3 has no variant {unknown[0]}")
        commits = {video.RELEASE: release_commit(arguments.images)}
        if arguments.convert:
            commits = convert(arguments.images, arguments.variants)
        commits |= dict(arguments.commit)
        marker = assemble(arguments.images, arguments.out, commits)
    except models.ModelError as error:
        print(f"publish_video_repo: {error}", file=sys.stderr)
        return 1
    print(json.dumps(marker, indent=2))
    print(f"Upload it with `hf upload OWNER/REPO {arguments.out} .`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
