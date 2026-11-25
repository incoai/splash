"""The model families Splash serves, the DFlash2 draft released for each and
the chat format the server reads and writes each one's output in.

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
from server.chat_format import FLINT, INKLING2, QWEN, ChatFormat
from server.images import INKLING_IMAGES, QWEN_IMAGES, ImageFormat

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
# What Inkling 2's GGUF release states (general.sampling.temp, top_p, top_k):
# temperature 1.0 over every token.
INKLING2_SAMPLING = SamplingDefaults(temperature=1.0, top_p=1.0, top_k=0)


@dataclass(frozen=True)
class ModelFamily:
    # The family's name, as the engine names it (each target layout's family,
    # runtime/model/Qwen3_8.hpp, Qwen3_6Moe.hpp, Qwen4Exp.hpp, Flint.hpp and
    # Inkling2Loop.hpp).
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
    # How its tower takes images (server/images.py), and the weight types of
    # the GGUF vision projectors (mmproj) an installation loads for it, in
    # the order it prefers them when a repository holds several
    # (upstream.select_vision). Qwen's tower loads only exact ones, which
    # preparation copies or converts without rounding; Inkling's also Q8_0,
    # last, whose values preparation rounds to BF16.
    image_format: ImageFormat = QWEN_IMAGES
    projectors: tuple[str, ...] = ("BF16", "F32", "F16")
    # Whether a source whose tower the installer refuses serves text, with
    # the refusal noted, rather than being refused: Inkling 2's, many of whose
    # conversions hold no tower, and which 1.7.2 served as text only. Another
    # family's such source asks for --language-only.
    text_fallback: bool = False


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
    # Its vision tower is its release's and its MLX conversions' own, and an
    # inkling2v projector beside its GGUFs: Unsloth publishes it in BF16, F16
    # and F32, ggml-org in BF16 and Q8_0.
    ModelFamily(
        "Inkling-2-Mini",
        "incoai/Inkling-2-Mini-DFlash2",
        INKLING2,
        INKLING2_SAMPLING,
        image_format=INKLING_IMAGES,
        projectors=("BF16", "F32", "F16", "Q8_0"),
        text_fallback=True,
    ),
)


def named(name):
    """The family called name, or None."""
    return next((family for family in FAMILIES if family.name == name), None)


# The fact a model's family must have for an option of the launcher's own,
# beside the shared ones (serve_options.SERVE_OPTIONS), to act on it, as
# ServeOption.needs names one: --language-only leaves out a vision tower.
LAUNCHER_NEEDS = {"--language-only": "vision"}


def option_report(engine_facts):
    """What each family, by name, makes of the options of `splash serve`,
    from its facts: engine_facts, each family's facts of its layout by name,
    as the engine's families command gives them, and its own vision tower
    and what a request's effort does to its reasoning (whether it reaches
    the model, and whether it turns the reasoning on and off). Each family's
    report holds the flags of the options that need a fact it lacks
    (ServeOption.needs, LAUNCHER_NEEDS), which act on none of its models,
    and the context its models serve: their native window, and the most past
    it, with YaRN."""
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
            "prompt_switches_reasoning": family.chat_format.prompt_switches_reasoning,
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
