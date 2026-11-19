"""The model families Splash Omni serves: the speech family and the
transcription family, each served by the audio server.

A target's family is the one the engine finds its configuration to describe,
by the rules it holds every start to (upstream.check_audio_model), never its
repository's name.
"""

from __future__ import annotations

from dataclasses import dataclass

# The server package module that serves every family (install/launcher.py).
SERVER = "server.audio_server"
# The checkpoints Splash Omni serves, which the README and the installer's
# refusals name.
SUPPORTED_MODELS = (
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    "mlx-community/Qwen3-ASR-1.7B-4bit",
    "mlx-community/Qwen3-ASR-1.7B-bf16",
    # The same talker, prompted with an instruction describing the voice
    # (VoiceDesign), or with a preset speaker and an optional instruction
    # (CustomVoice).
    "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
)


@dataclass(frozen=True)
class ModelFamily:
    # The family's name, as the engine names it (runtime/omni).
    name: str
    # The config.json model_type of the family's checkpoints: what omni-check
    # reads, and what stands in for it before the engine knows the check.
    model_type: str


FAMILIES = (
    ModelFamily("qwen3-tts", "qwen3_tts"),
    ModelFamily("qwen3-asr", "qwen3_asr"),
)
# The config.json model_type of each family, by family name.
AUDIO_MODEL_TYPES = {family.model_type: family.name for family in FAMILIES}


def named(name):
    """The family called name, or None."""
    return next((family for family in FAMILIES if family.name == name), None)


def supported_models_text():
    """The supported checkpoints, as a refusal names them."""
    return ", ".join(SUPPORTED_MODELS)
