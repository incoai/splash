"""The model families Splash serves and the DFlash2 draft released for each.

A target's family is the one the engine finds its configuration to describe,
by the rules it holds every start to (upstream.check_model), never its
repository's name.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFamily:
    # The family's name, as the engine names it (each target layout's family,
    # runtime/model/Qwen3_8.hpp and Qwen3_6Moe.hpp).
    name: str
    # The repository of the DFlash2 draft released for the family, as its
    # release publishes it (config.json and BF16 safetensors). Installations
    # follow its default branch as they follow the target's. None while no
    # draft is released for the family: an installation names one with
    # --draft-model.
    draft_repo: str | None


FAMILIES = (
    ModelFamily("Qwen3.8-27B", "incoai/Qwen3.8-27B-DFlash2"),
    ModelFamily("Qwen3.6-35B-A3B", "incoai/Qwen3.6-35B-A3B-DFlash2"),
)


def named(name):
    """The family called name, or None."""
    return next((family for family in FAMILIES if family.name == name), None)
