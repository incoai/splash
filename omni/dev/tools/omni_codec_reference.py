#!/usr/bin/env python3
"""Reference waveform of the Qwen3-TTS 12 Hz speech tokenizer decoder.

Decodes a fixed pseudo-random code sequence with mlx-audio's speech tokenizer
decoder (fp32 weights, whole sequence, no chunking) and writes the samples as
float32 to a .bin that dev/tests/engine/omni_codec_test.mm compares against.
The codes are drawn with the same xorshift32 generator the test uses, so no
code file is needed: codes[frame][q] = next() % codebook_size, frame-major.

Usage:
  omni_codec_reference.py --model-dir <snapshot dir of the TTS model> \
      [--frames 40] [--seed 1] [--output dev/tests/fixtures/omni/codec_40_1.bin]
"""

import argparse
import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np


def xorshift_codes(
    frames: int, quantizers: int, seed: int, codebook_size: int
) -> np.ndarray:
    state = seed & 0xFFFFFFFF
    codes = np.zeros((frames, quantizers), dtype=np.int64)
    for f in range(frames):
        for q in range(quantizers):
            state ^= (state << 13) & 0xFFFFFFFF
            state ^= state >> 17
            state ^= (state << 5) & 0xFFFFFFFF
            codes[f, q] = state % codebook_size
    return codes


def load_tokenizer(model_dir: Path):
    from mlx_audio.tts.models.qwen3_tts.config import (
        Qwen3TTSTokenizerConfig,
        Qwen3TTSTokenizerDecoderConfig,
        filter_dict_for_dataclass,
    )
    from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import Qwen3TTSSpeechTokenizer

    tokenizer_dir = model_dir / "speech_tokenizer"
    with open(tokenizer_dir / "config.json") as f:
        config_dict = json.load(f)
    decoder_config = Qwen3TTSTokenizerDecoderConfig(
        **filter_dict_for_dataclass(
            Qwen3TTSTokenizerDecoderConfig, config_dict["decoder_config"]
        )
    )
    config = Qwen3TTSTokenizerConfig(encoder_config=None, decoder_config=decoder_config)
    for k, v in config_dict.items():
        if k not in ("decoder_config", "encoder_config") and hasattr(config, k):
            setattr(config, k, v)
    tokenizer = Qwen3TTSSpeechTokenizer(config)
    weights = {}
    for wf in tokenizer_dir.glob("*.safetensors"):
        weights.update(mx.load(str(wf)))
    weights = {k: v for k, v in weights.items() if not k.startswith("encoder")}
    weights = Qwen3TTSSpeechTokenizer.sanitize(weights)
    tokenizer.load_weights(list(weights.items()), strict=False)
    mx.eval(tokenizer.parameters())
    tokenizer.eval()
    return tokenizer, decoder_config


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=Path("dev/tests/fixtures/omni/codec_40_1.bin")
    )
    args = parser.parse_args()

    tokenizer, decoder_config = load_tokenizer(args.model_dir)
    codes = xorshift_codes(
        args.frames,
        decoder_config.num_quantizers,
        args.seed,
        decoder_config.codebook_size,
    )
    # Whole-sequence decode (no chunking): [batch, quantizers, time].
    wav = tokenizer.decoder(mx.array(codes.T[None]))
    mx.eval(wav)
    samples = np.asarray(wav, dtype=np.float32).reshape(-1)
    expected = args.frames * tokenizer.decoder.total_upsample
    if samples.shape[0] != expected:
        raise SystemExit(
            f"decoder produced {samples.shape[0]} samples, expected {expected}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(
            struct.pack("<4sIII", b"OCRF", args.frames, args.seed, samples.shape[0])
        )
        f.write(samples.astype("<f4").tobytes())
    print(
        f"wrote {samples.shape[0]} samples to {args.output} (peak {np.abs(samples).max():.4f})"
    )


if __name__ == "__main__":
    main()
