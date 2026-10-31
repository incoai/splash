"""The model families Splash serves, the DFlash2 draft released for each and
the chat format the server reads and writes each one's output in.

A target's family is the one the engine finds its configuration to describe,
by the rules it holds every start to (upstream.check_model), never its
repository's name.
"""

from __future__ import annotations

from dataclasses import dataclass

from server.chat_format import DEEPSEEK, FLINT, QWEN, ChatFormat


@dataclass(frozen=True)
class ModelFamily:
    # The family's name, as the engine names it (each target layout's family,
    # runtime/model/Qwen3_8.hpp, Qwen3_6Moe.hpp, Qwen4Exp.hpp and Flint.hpp).
    name: str
    # The repository of the DFlash2 draft released for the family, as its
    # release publishes it (config.json and BF16 safetensors). Installations
    # follow its default branch as they follow the target's. None while no
    # draft is released for the family: an installation decodes without one
    # unless --draft-model names one.
    draft_repo: str | None
    # How the family's model reasons and calls tools, and the tokens it
    # shares with the engine (server/chat_format.py).
    chat_format: ChatFormat
    # Whether the family's models have a vision tower, which an installation
    # loads from the target repository unless --language-only. The models of
    # a family without one serve text only, whatever their repository holds.
    vision: bool = True
    # Whether the family's drafts are DSpark GGUFs, the one dspark*.gguf file
    # of a repository or a local file (DeepSeek-V4.1-Flash's own drafter),
    # rather than DFlash2 checkpoints.
    dspark: bool = False


FAMILIES = (
    ModelFamily("Qwen3.8-27B", "incoai/Qwen3.8-27B-DFlash2", QWEN),
    ModelFamily("Qwen3.6-35B-A3B", "incoai/Qwen3.6-35B-A3B-DFlash2", QWEN),
    # Its fine-tune Ornith-1.5-122B is the family too, and the draft
    # released for it the family's.
    ModelFamily("Qwen3.5-122B-A10B", "incoai/Ornith-1.5-122B-DFlash2", QWEN),
    ModelFamily("Qwen3.8-Flash-Next", "incoai/Qwen3.8-Flash-Next-DFlash2", QWEN),
    # It has no vision tower.
    ModelFamily("Flint", "incoai/Flint-DFlash2", FLINT, vision=False),
    # Splash serves its text alone, and an installation decodes without its
    # DSpark drafter unless --draft-model names one (its 8 GB of weights
    # beside the target's).
    ModelFamily("DeepSeek-V4.1-Flash", None, DEEPSEEK, vision=False, dspark=True),
)


def named(name):
    """The family called name, or None."""
    return next((family for family in FAMILIES if family.name == name), None)
