"""The model families Splash serves and the DFlash2 draft trained for each,
if one exists.

A target's family is the one the engine finds its configuration to describe,
by the rules it holds every start to (upstream.check_model), never its
repository's name.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFamily:
    # The family's name, as the engine names it (each target layout's family,
    # runtime/model/Qwen3_8.hpp, Qwen3_6Moe.hpp and Qwen4Exp.hpp).
    name: str
    # The repository of the DFlash2 checkpoint trained for the family, as its
    # release publishes it: config.json and BF16 safetensors. Installations
    # follow its default branch as they follow the target's. None for a family
    # no DFlash2 draft was trained for: it decodes without one, and
    # --draft-model is refused for it.
    draft_repo: str | None


FAMILIES = (
    ModelFamily("Qwen3.8-27B", "incoai/Qwen3.8-27B-DFlash2"),
    ModelFamily("Qwen3.6-35B-A3B", "incoai/Qwen3.6-35B-A3B-DFlash2"),
    ModelFamily("Qwen3.8-Flash-Next", None),
)


def named(name):
    """The family called name, or None."""
    return next((family for family in FAMILIES if family.name == name), None)
