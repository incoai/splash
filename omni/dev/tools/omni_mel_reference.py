#!/usr/bin/env python3
"""Golden inputs and log-mel features for the Omni audio front end.

Writes the synthetic 16 kHz signals the audio encoder tests encode (a
chirp plus noise from a fixed seed, at a few durations) and the Whisper
log-mel features of the 2-second one as transformers' WhisperFeatureExtractor
(128 Slaney mels, n_fft 400, hop 160) computes them, which
runtime/omni/MelSpectrogram.cpp must reproduce:

    python dev/tools/omni_mel_reference.py dev/tests/fixtures/omni

Files are raw little-endian float32: audio_<ms>ms.f32 holds the samples;
mel_2000ms.f32 holds the features frame-major, [frames][128].
"""

import argparse
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
MEL_BINS = 128
# Durations in milliseconds: two full 100-frame chunks; one short chunk; a
# full chunk followed by a short, zero-padded one.
DURATIONS_MS = (2000, 700, 1300)
SEED = 1234
# A longer signal spanning two 104-token attention windows (117 tokens),
# computed from a closed form so the C++ test regenerates it instead of
# storing it; only the tail rows of its tower output are stored.
TONE_MS = 9000
TONE_FIRST_ROW = 96


def synthetic_signal(milliseconds: int) -> np.ndarray:
    """A chirp from 200 Hz to 4 kHz under white noise, deterministic."""
    samples = SAMPLE_RATE * milliseconds // 1000
    rng = np.random.default_rng(SEED + milliseconds)
    t = np.arange(samples, dtype=np.float64) / SAMPLE_RATE
    duration = samples / SAMPLE_RATE
    f0, f1 = 200.0, 4000.0
    phase = 2.0 * np.pi * (f0 * t + (f1 - f0) * t * t / (2.0 * duration))
    chirp = 0.5 * np.sin(phase)
    noise = 0.05 * rng.standard_normal(samples)
    return (chirp + noise).astype(np.float32)


def tone_signal(milliseconds: int) -> np.ndarray:
    """A closed-form mixture of tones, bit-for-bit reproducible in C++."""
    samples = SAMPLE_RATE * milliseconds // 1000
    t = np.arange(samples, dtype=np.float64) / SAMPLE_RATE
    tau = 2.0 * np.pi
    signal = (
        0.3 * np.sin(tau * 440.0 * t)
        + 0.2 * np.sin(tau * 1234.5 * t + 1.0)
        + 0.1 * np.sin(tau * 3000.0 * t)
        + 0.2 * np.sin(tau * 220.0 * t) * np.sin(tau * 0.5 * t)
    )
    return signal.astype(np.float32)


def whisper_log_mel(samples: np.ndarray) -> np.ndarray:
    """transformers' features for one unpadded signal, [frames][128]."""
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor(
        feature_size=MEL_BINS,
        sampling_rate=SAMPLE_RATE,
        hop_length=160,
        n_fft=400,
        dither=0.0,
    )
    features = extractor(
        samples,
        sampling_rate=SAMPLE_RATE,
        return_attention_mask=True,
        truncation=False,
        padding=True,
        return_tensors="np",
    )
    mask = features["attention_mask"][0]
    assert int(mask.sum()) == mask.shape[0] == len(samples) // 160
    return np.ascontiguousarray(features["input_features"][0].T.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for milliseconds in DURATIONS_MS:
        signal = synthetic_signal(milliseconds)
        signal.tofile(args.out_dir / f"audio_{milliseconds}ms.f32")
        if milliseconds == 2000:
            mel = whisper_log_mel(signal)
            mel.tofile(args.out_dir / f"mel_{milliseconds}ms.f32")
            print(
                f"mel_{milliseconds}ms: {mel.shape[0]} frames, range {mel.min():.4f}..{mel.max():.4f}"
            )
        print(f"audio_{milliseconds}ms: {len(signal)} samples")


if __name__ == "__main__":
    main()
