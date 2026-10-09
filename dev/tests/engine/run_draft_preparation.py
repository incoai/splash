"""A small DFlash2 draft checkpoint and the af4g64 rows MLX's affine
quantization rounds each of its projections to, an independent oracle of the
draft's prepared planes, then the golden hashes of its images."""

import argparse
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dev.tests.fixture_files import write_safetensors  # noqa: E402

F32 = struct.Struct("<f")


def f32(value):
    """value rounded once to float32: a float32 sum, difference or quotient
    computed in double and rounded once is the float32 operation's result."""
    return F32.unpack(F32.pack(value))[0]


def bf16(value):
    """The bits of the BF16 nearest a finite value, ties to even."""
    bits = struct.unpack("<I", F32.pack(value))[0]
    return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16


def half_away(value):
    """value rounded to an integer, halves away from zero."""
    return math.copysign(math.floor(abs(value) + 0.5), value)


def quantized_group(weights):
    """64 weights quantized to 4 bits as MLX's affine quantization rounds them
    (mlx.core.quantize, its Metal kernel, whose maximum starts at 0), as a
    native af4g64 block: the scale and bias as BF16, then the packed codes."""
    low, high = min(weights), max(0.0, *weights)
    low_edge = abs(low) > abs(high)
    scale = max(f32(f32(high - low) / 15), f32(1e-7))
    if not low_edge:
        scale = -scale
    edge = low if low_edge else high
    q0 = half_away(f32(edge / scale))
    bias = 0.0
    if q0 != 0:
        scale, bias = f32(edge / q0), edge
    codes = [
        int(min(max(half_away(f32(f32(w - bias) / scale)), 0), 15)) for w in weights
    ]
    packed = bytes(codes[i] | codes[i + 1] << 4 for i in range(0, len(codes), 2))
    return struct.pack("<HH", bf16(scale), bf16(bias)) + packed


def draft_fixture(root):
    """A two-layer DFlash2 checkpoint of width 256, its BF16 safetensors as
    its repository releases them, and expected/native.safetensors: each
    projection's native af4g64 rows, as MLX's affine quantization rounds
    them."""
    tensors, values, expected = {}, {}, {}

    def add(name, shape, data=None):
        count = math.prod(shape)
        if data is None:
            # Multiples of 1/64 within 4 of zero, each exactly a BF16.
            seed = len(tensors) + 1
            data = [((i * 37 + seed * 11) % 509 - 254) / 64 for i in range(count)]
        values[name] = data
        tensors[name] = (shape, "BF16", struct.pack(f"<{count}H", *map(bf16, data)))

    def projection(name, rows, columns, data=None):
        add(name + ".weight", [rows, columns], data)
        weights = values[name + ".weight"]
        native = b"".join(
            quantized_group(weights[start : start + 64])
            for start in range(0, len(weights), 64)
        )
        expected[name] = ([rows, columns // 64 * 36], "U8", native)

    for layer in range(2):
        p = f"layers.{layer}."
        a = p + "self_attn."
        dynamic = None
        if layer == 0:
            # Two first rows of edge cases: all zero, where no scale puts 0 on
            # a code; constant; 0 to 15 with halves, which round away from
            # zero; a minimum farther from zero than the maximum; then groups
            # below zero, whose range still ends at 0: constant, spread, a
            # maximum near zero, and -15 to -1 with halves.
            dynamic = [0.0] * 64 + [0.75] * 64
            dynamic += [2.5, 8.5, *(float(i % 16) for i in range(62))]
            dynamic += [(i % 20) / 4 - 4 for i in range(64)]
            dynamic += [-0.75] * 64
            dynamic += [-(i % 16 + 1) / 4 for i in range(64)]
            dynamic += [-1 / 64, *(-(i % 32 + 1) / 8 for i in range(63))]
            dynamic += [-2.5, -8.5, *(-float(i % 15 + 1) for i in range(62))]
            dynamic += [((i * 37 + 11) % 509 - 254) / 64 for i in range(254 * 256)]
        add(p + "input_layernorm.weight", [256])
        add(p + "attention_conv.base_kernel", [2, 2, 256])
        projection(p + "attention_conv.kernel_projection", 256, 256, dynamic)
        # The fused q|k|v projection's parts.
        for name, rows in (("q_proj", 256), ("k_proj", 128), ("v_proj", 128)):
            projection(a + name, rows, 256)
        add(a + "q_norm.weight", [64])
        add(a + "k_norm.weight", [64])
        projection(a + "o_proj", 256, 256)
        add(p + "post_attention_layernorm.weight", [256])
        add(p + "mlp_conv.base_kernel", [2, 2, 256])
        projection(p + "mlp_conv.kernel_projection", 256, 256)
        projection(p + "mlp.gate_proj", 256, 256)
        projection(p + "mlp.up_proj", 256, 256)
        projection(p + "mlp.down_proj", 256, 256)
    projection("fc", 256, 256)
    add("hidden_norm.weight", [256])
    add("norm.weight", [256])
    projection("candidate_selector.hidden_projection", 256, 256)
    add("candidate_selector.predecessor_codebook", [256, 256])
    add("candidate_selector.successor_codebook", [256, 256])
    write_safetensors(root / "model.safetensors", tensors)
    (root / "expected").mkdir()
    write_safetensors(root / "expected" / "native.safetensors", expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("metallib", type=Path)
    parser.add_argument("goldens", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="splash-draft-preparation-") as directory:
        root = Path(directory)
        draft_fixture(root)
        command = [
            str(args.binary.resolve()),
            str(args.metallib.resolve()),
            str(root),
            str(args.goldens.resolve()),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        print(result.stdout.splitlines()[-1])


if __name__ == "__main__":
    main()
