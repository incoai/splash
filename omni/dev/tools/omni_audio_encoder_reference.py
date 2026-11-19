#!/usr/bin/env python3
"""mlx-audio's Qwen3-ASR audio tower output for the Omni encoder fixtures.

Encodes the synthetic signals dev/tools/omni_mel_reference.py wrote through
mlx-audio's Qwen3-ASR model (features from transformers'
WhisperFeatureExtractor, the tower in fp32 over its bf16 weights) and writes
the 2048-dim audio embeddings, [tokens][2048] float32, next to them:

    python dev/tools/omni_audio_encoder_reference.py MODEL_DIR dev/tests/fixtures/omni

MODEL_DIR is a local snapshot of mlx-community/Qwen3-ASR-1.7B-4bit (or
-bf16). Run it in the baseline venv (mlx-audio, transformers).
"""

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from omni_mel_reference import (  # noqa: E402
    DURATIONS_MS,
    SAMPLE_RATE,
    TONE_FIRST_ROW,
    TONE_MS,
    tone_signal,
    whisper_log_mel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("fixture_dir", type=Path)
    args = parser.parse_args()
    from mlx_audio.stt.utils import load

    model = load(str(args.model_dir))
    tower = model._model.audio_tower

    def encode(signal: np.ndarray) -> np.ndarray:
        mel = whisper_log_mel(signal)  # [frames][128]
        features = mx.array(mel.T)[None]  # [1][128][frames] fp32
        mask = mx.ones((1, mel.shape[0]), dtype=mx.int32)
        embeddings = tower(features, mask)
        mx.eval(embeddings)
        return np.array(embeddings.astype(mx.float32))

    for milliseconds in DURATIONS_MS:
        path = args.fixture_dir / f"audio_{milliseconds}ms.f32"
        signal = np.fromfile(path, dtype=np.float32)
        assert len(signal) == SAMPLE_RATE * milliseconds // 1000, path
        out = encode(signal)
        out.astype(np.float32).tofile(args.fixture_dir / f"tower_{milliseconds}ms.f32")
        print(
            f"tower_{milliseconds}ms: {len(signal) // 160} frames -> {out.shape}, |max| {np.abs(out).max():.4f}"
        )
    out = encode(tone_signal(TONE_MS))
    tail = np.ascontiguousarray(out[TONE_FIRST_ROW:])
    tail.astype(np.float32).tofile(
        args.fixture_dir / f"tower_{TONE_MS}ms_rows{TONE_FIRST_ROW}.f32"
    )
    print(
        f"tower_{TONE_MS}ms rows {TONE_FIRST_ROW}..{out.shape[0] - 1}: {tail.shape}, |max| {np.abs(tail).max():.4f}"
    )


if __name__ == "__main__":
    main()
