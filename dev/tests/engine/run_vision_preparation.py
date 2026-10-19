"""Prepare tiny vision towers of MLX's and transformers' safetensors names and of
a GGUF, and towers of MLX affine matrices, and compare them with an
independently serialized vision/model.bin; check the exact-BF16 rule, that
invalid sources fail naming what is wrong, and that a padded section written
after another keeps its padding zero."""

import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dev.tests.fixture_files import (  # noqa: E402
    weight_file,
    write_gguf,
    write_safetensors,
)

DTYPES = ("BF16", "F16", "F32")
# The name prefix of each safetensors source's tower: MLX's two, and
# transformers' (as Model Optimizer keeps it).
TOWER = {"mlx": "vision_tower.", "visual": "visual.", "transformers": "model.visual."}
# The module beside whose weight each case of a quantized tensor puts scales.
QUANTIZED = {
    "quantized": "blocks.0.attn.qkv",
    "quantized-patch": "patch_embed.proj",
    "quantized-norm": "blocks.0.norm1",
}
GGML_TYPES = {"F32": 0, "F16": 1, "Q4_0": 2, "BF16": 30}
VISION_SHARD = "model-00001-of-00002.safetensors"
TEXT_SHARD = "model-00002-of-00002.safetensors"


def bfloat16(values):
    # The upper half of each value's F32 bits: the value itself when it is
    # exactly a BF16.
    return b"".join(struct.pack("<f", x)[2:] for x in values)


def encode(values, dtype):
    if dtype == "BF16":
        return bfloat16(values)
    code = {"F16": "e", "F32": "f", "F64": "d"}[dtype]
    return struct.pack("<" + code * len(values), *values)


def affine_matrix(name, rows, columns, bits, group):
    """The tensors of an MLX affine matrix of bits in groups of group, its U32
    codes and BF16 scales and biases, and the values its image holds: each
    s * q + z rounded to the nearest BF16, ties to even, which some are."""
    packed, scales, biases, values = bytearray(), [], [], []
    groups = columns // group
    for row in range(rows):
        scale = [(1, -3, 3, -1)[(row + g) % 4] / 256 for g in range(groups)]
        bias = [(1, -1, 0.5, -0.5)[(row * 3 + g) % 4] for g in range(groups)]
        codes = 0
        for column in range(columns):
            code = (row * 7 + column * 11) % (1 << bits)
            codes |= code << (column * bits)
            value = scale[column // group] * code + bias[column // group]
            word = struct.unpack("<I", struct.pack("<f", value))[0]
            upper, lower = divmod(word, 65536)
            if lower > 32768 or (lower == 32768 and upper % 2):
                upper += 1
            values.append(struct.unpack("<f", struct.pack("<I", upper << 16))[0])
        packed.extend(codes.to_bytes(columns * bits // 8, "little"))
        scales.extend(scale)
        biases.extend(bias)
    stem = name.removesuffix(".weight")
    return {
        name: ([rows, columns * bits // 32], "U32", bytes(packed)),
        stem + ".scales": ([rows, groups], "BF16", bfloat16(scales)),
        stem + ".biases": ([rows, groups], "BF16", bfloat16(biases)),
    }, values


def fixture(root, source, shift=0, case=None, quantization=None):
    """Writes a tiny tower (depth 2, width 8, 2x2 patches) of source "mlx",
    "visual", "transformers" or "gguf" whose tensors cycle through BF16, F16
    and F32, and the vision/model.bin it must prepare; with quantization, a
    (bits, group size) pair, the tower is 128 wide and its matrices MLX affine
    in that format."""
    tensors = {}
    sections = []
    width = 128 if quantization else 8
    intermediate, padded = (160, 256) if quantization else (10, 16)

    def add(mlx, gguf_name, rows, columns=1, padded_rows=None, padded_columns=None):
        index = len(sections)
        dtype = DTYPES[(index + shift) % 3]
        # Exact in every dtype: signed zero, infinity, 2^-24 (an F16
        # subnormal) and multiples of 1/64.
        values = [-0.0, math.inf, 2.0**-24] + [
            ((i * 7 + index * 11) % 255 - 127) / 64 for i in range(3, rows * columns)
        ]
        if case == "inexact-f32" and index == 0:
            values[5] = 1 + 2.0**-20
        if case == "inexact-f16" and index == 0:
            values[5] = 1 + 2.0**-10
        if index == 0:
            # One patch row is [frame, patch-row, patch-col, channel] in MLX and
            # [channel, patch-row, patch-col] per frame in GGUF. The image's rows
            # are [channel, frame, patch-row, patch-col], as transformers' are.
            def at(row, frame, pixel, channel):
                return values[row * 24 + (frame * 4 + pixel) * 3 + channel]

            image = [
                at(row, frame, pixel, channel)
                for row in range(rows)
                for channel in range(3)
                for frame in range(2)
                for pixel in range(4)
            ]
            # MLX's order and transformers', which "other-order" swaps.
            mlx_order = (source != "transformers") != (case == "other-order")
            if source in TOWER and mlx_order:
                tensors[TOWER[source] + mlx] = (
                    [rows, 2, 2, 2, 3],
                    dtype,
                    encode(values, dtype),
                )
            elif source in TOWER:
                tensors[TOWER[source] + mlx] = (
                    [rows, 3, 2, 2, 2],
                    dtype,
                    encode(image, dtype),
                )
            else:
                for frame, suffix in enumerate(("", ".1")):
                    frame_values = [
                        at(row, frame, pixel, channel)
                        for row in range(rows)
                        for channel in range(3)
                        for pixel in range(4)
                    ]
                    tensors[gguf_name + suffix] = (
                        [2, 2, 3, rows],
                        dtype,
                        encode(frame_values, dtype),
                    )
            values = image
        else:
            name = TOWER[source] + mlx if source in TOWER else gguf_name
            if columns == 1:
                shape = [rows]
            else:
                shape = [rows, columns] if source in TOWER else [columns, rows]
            if case == "dtype" and mlx == "merger.linear_fc1.weight":
                dtype = "F64" if source in TOWER else "Q4_0"
            if (
                quantization
                and columns > 1
                and columns % quantization[1] == 0
                and mlx.endswith(".weight")
            ):
                quantized, values = affine_matrix(name, rows, columns, *quantization)
                tensors.update(quantized)
            else:
                raw = encode(values, "F32" if dtype == "Q4_0" else dtype)
                tensors[name] = (shape, dtype, raw)
        output = []
        for row in range(padded_rows or rows):
            output.extend(
                values[row * columns : (row + 1) * columns]
                if row < rows
                else [0.0] * columns
            )
            output.extend([0.0] * ((padded_columns or columns) - columns))
        sections.append(bfloat16(output))

    def affine(mlx, gguf_name, rows, columns, padded_rows=None, padded_columns=None):
        add(
            mlx + ".weight",
            gguf_name + ".weight",
            rows,
            columns,
            padded_rows,
            padded_columns,
        )
        add(mlx + ".bias", gguf_name + ".bias", rows, padded_rows=padded_rows)

    def norm(mlx, gguf_name):
        add(mlx + ".weight", gguf_name + ".weight", width)
        add(mlx + ".bias", gguf_name + ".bias", width)

    affine("patch_embed.proj", "v.patch_embd", width, 24)
    add("pos_embed.weight", "v.position_embd.weight", 4, width)
    for layer in range(2):
        m, g = f"blocks.{layer}.", f"v.blk.{layer}."
        norm(m + "norm1", g + "ln1")
        affine(m + "attn.qkv", g + "attn_qkv", 3 * width, width)
        affine(m + "attn.proj", g + "attn_out", width, width)
        norm(m + "norm2", g + "ln2")
        affine(
            m + "mlp.linear_fc1", g + "ffn_up", intermediate, width, padded_rows=padded
        )
        affine(
            m + "mlp.linear_fc2",
            g + "ffn_down",
            width,
            intermediate,
            padded_columns=padded,
        )
    norm("merger.norm", "v.post_ln")
    affine("merger.linear_fc1", "mm.0", 4 * width, 4 * width)
    affine("merger.linear_fc2", "mm.2", width, 4 * width)
    first = next(iter(tensors))
    if case == "shape":
        shape, dtype, raw = tensors[first]
        tensors[first] = ([math.prod(shape)], dtype, raw)
    (root / "expected.bin").write_bytes(weight_file("MDFV0001", 2, 0, sections))
    if source in TOWER:
        # A quantized module's scales: MLX's beside its packed weight, Model
        # Optimizer's weight_scale beside its codes.
        if case in QUANTIZED:
            tensors[quantized_scales(source, case)] = ([24, 1], "BF16", bytes(48))
        # The position table's scales and biases made malformed, or uint8
        # scales without biases, as MLX keeps mxfp4 and nvfp4.
        stem = TOWER[source] + "pos_embed"
        if case == "missing-bias":
            del tensors[stem + ".biases"]
        elif case in ("affine-shape", "affine-dtype"):
            shape, dtype, raw = tensors[stem + ".scales"]
            tensors[stem + ".scales"] = (
                list(reversed(shape)) if case == "affine-shape" else shape,
                "F16" if case == "affine-dtype" else dtype,
                raw,
            )
        elif case in ("mxfp4", "nvfp4"):
            shape, _, raw = tensors[stem + ".scales"]
            tensors[stem + ".scales"] = (shape, "U8", raw[: len(raw) // 2])
            del tensors[stem + ".biases"]
        write_safetensors(root / VISION_SHARD, tensors)
        text_norm = (
            "model.language_model.norm.weight"
            if source == "transformers"
            else "language_model.model.norm.weight"
        )
        write_safetensors(
            root / TEXT_SHARD, {text_norm: ([width], "BF16", bytes(2 * width))}
        )
        return
    metadata = {
        "general.architecture": "clip",
        "clip.projector_type": "qwen3vl_merger",
        "clip.vision.projection_dim": 8,
        "clip.vision.patch_size": 2,
        "clip.vision.embedding_length": 8,
        "clip.vision.feed_forward_length": 10,
        "clip.vision.block_count": 2,
        "clip.vision.attention.head_count": 2,
        "clip.vision.spatial_merge_size": 2,
        "clip.use_gelu": True,
        "clip.vision.attention.layer_norm_epsilon": 1e-6,
        "clip.vision.image_mean": [0.5] * 3,
        "clip.vision.image_std": [0.5] * 3,
        "clip.vision.is_deepstack_layers": [False] * 2,
        # Unknown metadata must not make an otherwise supported projector fail.
        "clip.unused": ["ignored"],
    }
    if case == "epsilon":
        metadata["clip.vision.attention.layer_norm_epsilon"] = 1e-5
    if case == "deepstack":
        metadata["clip.vision.is_deepstack_layers"] = [True, False]
    if case == "no-deepstack":
        del metadata["clip.vision.is_deepstack_layers"]
    if case == "unused":
        tensors["v.deepstack.0.fc1.weight"] = ([8, 8], "BF16", bytes(128))
    gguf_tensors = [
        (name, shape, GGML_TYPES[dtype], raw)
        for name, (shape, dtype, raw) in tensors.items()
    ]
    write_gguf(root / "mmproj.gguf", metadata, gguf_tensors)


def quantized_scales(source, case="quantized"):
    scales = "weight_scale" if source == "transformers" else "scales"
    return TOWER[source] + QUANTIZED[case] + "." + scales


def prepare(binary, directory, source, affine=False):
    # Every safetensors name convention is the loader's safetensors source;
    # an affine tower's layout is wider.
    loader = "gguf" if source == "gguf" else "affine" if affine else "safetensors"
    command = [binary, loader, str(directory), str(directory / "expected.bin")]
    return subprocess.run(command, text=True, capture_output=True)


def check_refused(binary, directory, source, parts, affine=False):
    """Checks that preparing directory fails naming each of parts: the tensor
    or key, the kind of problem and, as "{}", the source file."""
    result = prepare(binary, directory, source, affine)
    errors = result.stderr.strip().splitlines()
    assert result.returncode == 1 and errors, (directory.name, result.stderr)
    file = VISION_SHARD if source in TOWER else "mmproj.gguf"
    named = [part.format(directory / file) for part in parts]
    assert all(part in errors[-1] for part in named), (
        directory.name,
        named,
        result.stderr,
    )


def main():
    binary = str(Path(sys.argv[1]).resolve())
    # Every fixture prepares the same file: the values do not depend on the
    # source format or the dtypes.
    golden = json.loads(Path(sys.argv[2]).read_text())["vision_image"]
    with tempfile.TemporaryDirectory(prefix="splash-vision-preparation-") as temp:
        root = Path(temp)
        for source in (*TOWER, "gguf"):
            safetensors = source in TOWER
            # Every tensor as BF16, F16 and F32; BF16 is copied and exact F32
            # or F16 values become their BF16 bits. A safetensors tower's patch
            # embedding loads in either order.
            cases = [(shift, None) for shift in range(3)]
            if safetensors:
                cases.append((0, "other-order"))
            for shift, case in cases:
                directory = root / f"{source}-{case or shift}"
                directory.mkdir()
                fixture(directory, source, shift, case)
                result = prepare(binary, directory, source)
                assert result.returncode == 0, (source, shift, case, result.stderr)
                digest = result.stdout.split()[0]
                assert digest == golden, (source, shift, case, digest)
            patch = (
                TOWER[source] + "patch_embed.proj.weight"
                if safetensors
                else "v.patch_embd.weight"
            )
            merger = (
                TOWER[source] + "merger.linear_fc1.weight"
                if safetensors
                else "mm.0.weight"
            )
            dtype = "F64" if safetensors else "Q4_0"
            # What each refusal names (check_refused).
            inexact = (patch, "{}", "BF16")
            # The patch embedding is F32 with shift 2 and F16 with shift 1.
            rejected = {
                ("inexact-f32", 2): inexact,
                ("inexact-f16", 1): inexact,
                ("shape", 0): (patch, "shape"),
                ("dtype", 0): (merger, "{}", dtype),
            }
            if safetensors:
                rejected |= {
                    (case, 0): ("quantized", quantized_scales(source, case), "{}")
                    for case in QUANTIZED
                }
            else:
                rejected |= {
                    ("epsilon", 0): ("epsilon",),
                    ("deepstack", 0): ("deepstack", "unsupported"),
                    ("no-deepstack", 0): ("clip.vision.is_deepstack_layers",),
                    ("unused", 0): ("v.deepstack.0.fc1.weight", "{}", "does not use"),
                }
            for (case, shift), parts in rejected.items():
                directory = root / f"{source}-{case}"
                directory.mkdir()
                fixture(directory, source, shift, case)
                check_refused(binary, directory, source, parts)

        # MLX affine matrices of every bits and group size, compared with
        # their nearest BF16 values, under either MLX prefix.
        for source in ("mlx", "visual"):
            for bits in (2, 3, 4, 5, 6, 8):
                for group in (32, 64, 128):
                    directory = root / f"{source}-affine-{bits}-{group}"
                    directory.mkdir()
                    fixture(directory, source, quantization=(bits, group))
                    result = prepare(binary, directory, source, affine=True)
                    assert result.returncode == 0, (source, bits, group, result.stderr)
            # A malformed position table, or one in a format but affine's.
            table = TOWER[source] + "pos_embed"
            for case, quantization, parts in (
                ("missing-bias", (4, 64), (table + ".biases", "missing")),
                ("affine-shape", (4, 64), (table, "does not match")),
                ("affine-dtype", (4, 64), (table + ".weight", "{}", "F16 scales")),
                ("unsupported-bits", (7, 64), (table + ".weight", "{}", "7-bit")),
                ("unsupported-group", (4, 16), (table + ".weight", "{}", "of 16")),
                ("mxfp4", (4, 32), (table + ".weight", "{}", "of 32 with U8")),
                ("nvfp4", (4, 16), (table + ".weight", "{}", "of 16 with U8")),
            ):
                directory = root / f"{source}-{case}"
                directory.mkdir()
                fixture(directory, source, case=case, quantization=quantization)
                check_refused(binary, directory, source, parts, affine=True)

        directory = root / "padding"
        directory.mkdir()
        result = subprocess.run(
            [binary, "padding", str(directory)], text=True, capture_output=True
        )
        assert result.returncode == 0, result.stderr
        print(
            "Vision layouts from MLX, transformers and GGUF, affine BF16 decoding, "
            "exact floating conversion, rejected sources and zero padding PASS"
        )


if __name__ == "__main__":
    main()
