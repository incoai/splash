#!/usr/bin/env python3
"""Writes the MLX quantization fixture that pins the MLX formats of the GGUF
CPU reference and of weight preparation to MLX
(dev/tests/engine/gguf_reference_test.mm, gguf_preparation_test.mm): for each
affine bit width and group size Splash loads, and for mxfp4, MLX's
quantization of two rows of 256 random bf16 weights (mlx.core.quantize) and
MLX's own reading of it: the codes (an affine quantization dequantized with
unit scales and zero biases, which is exact) and the fp32 values.

    python dev/tools/mlx_quantization_fixture.py dev/tests/fixtures/mlx-quantization/fixture.json

needs MLX (pip install mlx); the fixture is committed, so the tests do not.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

BITS = (2, 3, 4, 5, 6, 8)
GROUPS = (32, 64, 128)
ROWS, COLUMNS = 2, 256


def hexbytes(array) -> str:
    return np.asarray(array).tobytes().hex()


def entry(mode: str, bits: int, group: int, seed: int) -> dict:
    weights = (
        mx.random.normal((ROWS, COLUMNS), key=mx.random.key(seed)) * 0.02
    ).astype(mx.bfloat16)
    read = {"group_size": group, "bits": bits, "mode": mode}
    quantized = mx.quantize(weights, **read)
    packed, scales = quantized[0], quantized[1]
    result = {
        "mode": mode,
        "bits": bits,
        "group": group,
        "rows": ROWS,
        "columns": COLUMNS,
        "weight": hexbytes(packed),
    }
    if mode == "affine":
        biases = quantized[2]
        assert scales.dtype == mx.bfloat16 and biases.dtype == mx.bfloat16
        codes = mx.dequantize(
            packed, mx.ones(scales.shape), mx.zeros(biases.shape), **read
        )
        values = mx.dequantize(
            packed, scales.astype(mx.float32), biases.astype(mx.float32), **read
        )
        result |= {
            "scales": hexbytes(scales.view(mx.uint16)),
            "biases": hexbytes(biases.view(mx.uint16)),
            "codes": hexbytes(codes.astype(mx.uint8)),
        }
    else:
        assert scales.dtype == mx.uint8
        values = mx.dequantize(packed, scales, **read)
        result["scales"] = hexbytes(scales)
    result["values"] = hexbytes(values.astype(mx.float32))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    formats = [
        entry("affine", bits, group, 100 * bits + group)
        for bits in BITS
        for group in GROUPS
    ]
    formats.append(entry("mxfp4", 4, 32, 7))
    args.output.write_text(
        json.dumps({"mlx": mx.__version__, "formats": formats}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
