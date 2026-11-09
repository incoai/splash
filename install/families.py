"""The model families Splash serves, the DFlash2 draft released for each and
the chat format the server reads and writes each one's output in.

A target's family is the one the engine finds its configuration to describe,
by the rules it holds every start to (upstream.check_model), never its
repository's name.

Run as a script, it reports what each family makes of the options of
`splash serve`, which the desktop app reads as it launches to offer a model
only the options it can use (option_report):

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
