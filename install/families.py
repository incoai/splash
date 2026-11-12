"""The model families Splash serves, the DFlash2 draft released for each and
the chat format the server reads and writes each one's output in; and the
video model's family, H3, its installer's and server's constants.

A target's family is the one the engine finds its configuration to describe,
by the rules it holds every start to (upstream.check_model), never its
repository's name.

Run as a script, it reports what each family makes of the options of
`splash serve`, which the desktop app reads as it launches to offer only the
models of these families, each only the options it can use (option_report):

    python install/families.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

if __name__ == "__main__" and not __package__:
    # Run as a script, from the installation's install/ folder: import its
    # siblings as the install package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "install"

from server import serve_options
from server.chat_format import FLINT, QWEN, ChatFormat

from . import models


@dataclass(frozen=True)
class SamplingDefaults:
    """The sampling a family's publisher recommends for its models, which a
    request gets for each of these fields it leaves out."""

    temperature: float
    top_p: float
    top_k: int


# What Qwen publishes with its models: temperature 1.0, top_p 0.95, top_k 20.
QWEN_SAMPLING = SamplingDefaults(temperature=1.0, top_p=0.95, top_k=20)
# What Flint's GGUF release states (general.sampling.temp, top_p, top_k),
# which are Qwen's numbers.
FLINT_SAMPLING = SamplingDefaults(temperature=1.0, top_p=0.95, top_k=20)


@dataclass(frozen=True)
class ModelFamily:
    # The family's name, as the engine names it (each target layout's family,
    # runtime/model/Qwen3_8.hpp, Qwen3_6Moe.hpp, Qwen4Exp.hpp and Flint.hpp).
    name: str
    # The repository of the DFlash2 draft released for the family, as its
    # release publishes it (config.json and BF16 safetensors). Installations
    # follow its default branch as they follow the target's.
    draft_repo: str
    # How the family's model reasons and calls tools, and the tokens it
    # shares with the engine (server/chat_format.py).
    chat_format: ChatFormat
    # The sampling a request gets unless it names its own.
    sampling: SamplingDefaults
    # Whether the family's models have a vision tower, which an installation
    # loads from the target repository unless --language-only. The models of
    # a family without one serve text only, whatever their repository holds.
    vision: bool = True


FAMILIES = (
    ModelFamily("Qwen3.8-27B", "incoai/Qwen3.8-27B-DFlash2", QWEN, QWEN_SAMPLING),
    ModelFamily(
        "Qwen3.6-35B-A3B", "incoai/Qwen3.6-35B-A3B-DFlash2", QWEN, QWEN_SAMPLING
    ),
    # Its fine-tune Ornith-1.5-122B is the family too, and the draft
    # released for it the family's.
    ModelFamily(
        "Qwen3.5-122B-A10B", "incoai/Ornith-1.5-122B-DFlash2", QWEN, QWEN_SAMPLING
    ),
    ModelFamily(
        "Qwen3.8-Flash-Next",
        "incoai/Qwen3.8-Flash-Next-DFlash2",
        QWEN,
        QWEN_SAMPLING,
    ),
    # It has no vision tower.
    ModelFamily("Flint", "incoai/Flint-DFlash2", FLINT, FLINT_SAMPLING, vision=False),
)


def named(name):
    """The family called name, or None."""
    return next((family for family in FAMILIES if family.name == name), None)


@dataclass(frozen=True)
class Variant:
    """A DiT a model ID's variant installs: its denoising steps, its images'
    directory, dit-<dit> (h3.STAGES; FastH3's by the variant's name), the
    repository the recipe reads with the release, the recipe, h3.py or
    fasth3.py, and for fasth3.py's spec-driven mode the spec (fasth3.SPEC:
    name, checkpoint, adapter, schedule, references, qkfix), which the
    entry's key names."""

    steps: int
    dit: str
    companion: str
    recipe: str = "h3"
    spec: dict | None = None


@dataclass(frozen=True)
class VideoFamily:
    # The family's name, and the target format of its assemblies
    # (install/video.py; assembly.TARGET_FORMATS and the engine name it too).
    name: str
    target_format: str
    # The release Splash installs it from, the file that marks a repository
    # as the release, and the folder its tokenizer/ lies in.
    release: str
    release_index: str
    tokenizer_prefix: str
    # The variants a model ID names; a model ID without one keeps the variant
    # its installation holds, or else installs default_variant, which a
    # server serves where it holds more than one.
    variants: dict
    default_variant: str
    # The text encoder's checkpoint, the decoder layers of it the engine
    # reads, and its package's name, in the package's entry key
    # (video.encoder_package).
    encoder: str
    encoder_layers: int
    encoder_package: str
    # The image directories the engine reads beside the variants' DiTs, and
    # in a DiT's directory the adaLN tables of its schedule and its manifest,
    # which names the weights' format.
    images: tuple
    adaln: str
    dit_manifest: str
    # Printed at installation and by `splash generate`, and the release's
    # license files an assembly links, where it has them.
    license: str
    license_files: tuple
    # A clip's frame rate, its frames frame_period n + frame_offset
    # (runtime/video/ops/Geometry.cpp holds the engine's copy), and OpenAI's
    # `seconds` (of the video API): the whole numbers a clip takes and the
    # default.
    fps: int
    frame_period: int
    frame_offset: int
    seconds: range
    default_seconds: str
    # OpenAI's `size`, WIDTHxHEIGHT: a canvas of a tier, by its short edge,
    # both sides multiples of canvas_step and the long edge at most twice the
    # short (the engine's ops::canvasTier); the named canvases, any other of
    # a tier rendering too, and the default.
    canvas_tiers: tuple
    canvas_step: int
    sizes: tuple
    size: str
    # The presentation's text rows (one prefill pass of the text encoder),
    # and the text encoder's vision tokens, for the server's prompt layout.
    max_text_tokens: int
    vision_start: int
    vision_end: int
    image_pad: int
    video_pad: int
    # Ref2VA references as the server normalizes them: their audio's
    # sampling rate, the canvas rule per tier and Pika's limits.
    sample_rate: int
    reference_tiers: dict
    aspect_range: tuple
    max_source_seconds: float
    video_seconds: tuple
    max_total_video_seconds: float
    max_images: int
    max_videos: int
    max_audio: int
    max_references: int
    min_video_frames: int


# Pika H3 Flex, whose adapters and Ref2VA base the pika variants read, and
# its released sampler (pika_h3.core.sampler's DEFAULT_KNOTS): six forwards at its
# knots, shifts 12 / 3, sigmas in float64 then fp32, fresh noise after each
# forward but the last from the request's seed + 1.
PIKA = "Pika-Labs/Pika-H3-Flex"
PIKA_SCHEDULE = {
    "rungs": [1000, 750, 500, 250, 100, 40],
    "video_shift": 12.0,
    "audio_shift": 3.0,
    "sampler": "renoise",
    "sigma_precision": "float64",
    "renoise_seed_offset": 1,
    "source": "Pika-Labs/Pika-H3-Flex src/pika_h3/core/sampler.py DEFAULT_KNOTS, "
    "schedules.H3SigmaSchedule, sampling.few_step_trajectory (renoise)",
}

H3 = VideoFamily(
    name="MiniMax-H3",
    target_format="h3-video",
    # MiniMax's release, of bf16 weights.
    release="MiniMaxAI/MiniMax-H3",
    release_index="FL2VA/model_index.json",
    tokenizer_prefix="FL2VA/",
    variants={
        # The OpenVDN stages the recipe reads with the release.
        "vdn-8": Variant(8, "turbo", "OpenVDN/vdn-minimax-h3"),
        "vdn-50": Variant(50, "default", "OpenVDN/vdn-minimax-h3"),
        # FastH3's: the 8-step student, and the LoRA bundle of the 4-step
        # one, which applies to the release's transformer (fasth3.py).
        "fasth3-8": Variant(
            8, "fasth3-8", "FastVideo/FastVideo-FastH3-8-Step-V2", "fasth3"
        ),
        "fasth3-4": Variant(
            4, "fasth3-4", "FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA", "fasth3"
        ),
        # Pika H3 Flex: the release's base DiT (FL2VA's transformer at the
        # commit Pika's adapters were trained on) with its FL2VA adapter at
        # run time (runtime/video/model/Adapter.hpp); and MiniMax-H3's Ref2VA
        # base (Pika's re-sharded transformer_ref/, diffusers names) with
        # Pika's ref2va adapter, the variant of Ref2VA requests
        # (references). Both are specs of fasth3.py's recipe.
        "pika-6": Variant(
            6,
            "pika-6",
            PIKA,
            "fasth3",
            {
                "name": "pika-6",
                "checkpoint": {
                    "repo": "MiniMaxAI/MiniMax-H3",
                    "revision": "42ed227ee7df40d41602854ae760620d6eb651fe",
                    "prefix": "FL2VA/transformer/",
                },
                "adapter": {
                    "file": "fl2va/adapter.safetensors",
                    "name": "default",
                    "alpha": 128.0,
                },
                "schedule": PIKA_SCHEDULE,
                "references": False,
                "qkfix": [45, 48, 49],
            },
        ),
        "pika-ref2va-6": Variant(
            6,
            "pika-ref2va-6",
            PIKA,
            "fasth3",
            {
                "name": "pika-ref2va-6",
                "checkpoint": {
                    "repo": PIKA,
                    "revision": None,
                    "prefix": "transformer_ref/",
                },
                "adapter": {
                    "file": "ref2va/adapter.safetensors",
                    "name": "default",
                    "alpha": 128.0,
                },
                "schedule": PIKA_SCHEDULE,
                "references": True,
                "qkfix": [45, 48, 49],
            },
        ),
    },
    default_variant="pika-6",
    # Qwen3-VL-32B's MLX checkpoint: decoder layers 0 to 49
    # (TextEncoder::kLayers: MiniMax-H3 conditions on hidden_states[50]).
    encoder="mlx-community/Qwen3-VL-32B-Instruct-8bit",
    encoder_layers=50,
    encoder_package="qwen3-vl-encoder-package-v1",
    images=("vae", "audio", "preview"),
    adaln="adaln.safetensors",
    dit_manifest="manifest.json",
    license=(
        "MiniMax-H3 weights are under the MiniMax H3 Community License (not "
        "licensed for use in the EU, UK, Republic of Korea or the USA): "
        "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
    ),
    license_files=("LICENSE", "NOTICE"),
    # A whole number of seconds from 4 to 15 (Pika's `--duration`, 5 to 15,
    # and 4). A clip of s seconds has the smallest 17 n + 5 frames at least
    # s x 24 at 24 fps (Pika's align_num_frames): 4, 5, 7, 10 and 15 s are
    # 107, 124, 175, 243 and 362.
    fps=24,
    frame_period=17,
    frame_offset=5,
    seconds=range(4, 16),
    default_seconds="5",
    # 768p and 480p.
    canvas_tiers=(768, 480),
    canvas_step=32,
    sizes=("1344x768", "768x1344", "832x480", "864x480", "480x832"),
    size="1344x768",
    # Pika H3 Flex's MAX_TEXT_TOKENS: a prompt and its labels, and a clip
    # from keyframes as a whole; a Ref2VA presentation's vision rows run past
    # it up to the engine's bound (Ready max_context_tokens).
    max_text_tokens=2048,
    # Qwen3-VL's <|vision_start|>, <|vision_end|>, <|image_pad|> and
    # <|video_pad|>.
    vision_start=151652,
    vision_end=151653,
    image_pad=151655,
    video_pad=151656,
    sample_rate=32000,
    # Pika's DEFAULT_TIER_RULES of the tiers Splash renders, by the canvas's
    # short edge: the canvas rule a reference video follows (short edge, area
    # budget) and an image reference's short edge.
    reference_tiers={
        480: (480, 832 * 480, 1024),
        768: (768, 1344 * 768, 2048),
    },
    aspect_range=(1 / 4, 4.0),
    max_source_seconds=16.0,
    video_seconds=(2.0, 15.0),
    max_total_video_seconds=15.0,
    max_images=9,
    max_videos=3,
    max_audio=3,
    max_references=12,
    min_video_frames=22,
)


# The fact a model's family must have for an option of the launcher's own,
# beside the shared ones (serve_options.SERVE_OPTIONS), to act on it, as
# ServeOption.needs names one: --language-only leaves out a vision tower.
LAUNCHER_NEEDS = {"--language-only": "vision"}


def option_report(engine_facts):
    """What each family, by name, makes of the options of `splash serve`,
    from its facts: engine_facts, each family's facts of its layout by name,
    as the engine's families command gives them, and its own vision tower
    and reasoning switch. Each family's report holds the flags of the
    options that need a fact it lacks (ServeOption.needs, LAUNCHER_NEEDS),
    which act on none of its models, and the context its models serve:
    their native window, and the most past it, with YaRN."""
    needs = {
        option.flag: option.needs
        for option in serve_options.SERVE_OPTIONS
        if option.needs is not None
    } | LAUNCHER_NEEDS
    report = {}
    for family in FAMILIES:
        engine = engine_facts[family.name]
        facts = engine | {
            "vision": family.vision,
            "reasoning_switch": family.chat_format.reasoning_switch,
        }
        report[family.name] = {
            "unusable": [flag for flag, fact in needs.items() if not facts[fact]],
            "context_tokens": engine["context_tokens"],
            "max_context_tokens": engine["max_context_tokens"],
        }
    return report


def main():
    try:
        engine_facts = json.loads(models.run_engine(["families"], "family report"))
    except models.ModelError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(option_report(engine_facts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
