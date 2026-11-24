"""Prepare tiny vision towers of MLX's and transformers' safetensors names and of
a GGUF and compare them with an independently serialized vision/model.bin;
check the exact-BF16 rule, that invalid sources fail naming what is wrong, and
that a padded section written after another keeps its padding zero. Then the
same for a tiny Inkling tower: an inkling2v mmproj whose matrices include Q8_0
ones, each value rounded to the nearest BF16, and the release's and
transformers' checkpoints and the two that rename the release's model.visual.
vision_tower., whose patch embedding orders a patch's values [row, column,
channel], each with its q, k and v stacked, its class token's two rows and
its zero biases."""

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
# The name prefix of each safetensors source's tower: MLX's, and transformers'
# (as Model Optimizer keeps it).
TOWER = {"mlx": "vision_tower.", "transformers": "model.visual."}
GGML_TYPES = {"F32": 0, "F16": 1, "Q4_0": 2, "Q8_0": 8, "BF16": 30}
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


def fixture(root, source, shift=0, case=None):
    """Writes a tiny tower (depth 2, width 8, 2x2 patches) of source "mlx",
    "transformers" or "gguf" whose tensors cycle through BF16, F16 and F32,
    and the vision/model.bin it must prepare."""
    tensors = {}
    sections = []

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
            if source == "mlx":
                tensors[TOWER[source] + mlx] = (
                    [rows, 2, 2, 2, 3],
                    dtype,
                    encode(values, dtype),
                )
            elif source == "transformers":
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
        add(mlx + ".weight", gguf_name + ".weight", 8)
        add(mlx + ".bias", gguf_name + ".bias", 8)

    affine("patch_embed.proj", "v.patch_embd", 8, 24)
    add("pos_embed.weight", "v.position_embd.weight", 4, 8)
    for layer in range(2):
        m, g = f"blocks.{layer}.", f"v.blk.{layer}."
        norm(m + "norm1", g + "ln1")
        affine(m + "attn.qkv", g + "attn_qkv", 24, 8)
        affine(m + "attn.proj", g + "attn_out", 8, 8)
        norm(m + "norm2", g + "ln2")
        affine(m + "mlp.linear_fc1", g + "ffn_up", 10, 8, padded_rows=16)
        affine(m + "mlp.linear_fc2", g + "ffn_down", 8, 10, padded_columns=16)
    norm("merger.norm", "v.post_ln")
    affine("merger.linear_fc1", "mm.0", 32, 32)
    affine("merger.linear_fc2", "mm.2", 8, 32)
    first = next(iter(tensors))
    if case == "shape":
        shape, dtype, raw = tensors[first]
        tensors[first] = ([math.prod(shape)], dtype, raw)
    (root / "expected.bin").write_bytes(weight_file("MDFV0001", 2, 0, sections))
    if source in TOWER:
        # A quantized module's scales: MLX's beside its packed weight, Model
        # Optimizer's weight_scale beside its codes.
        if case == "quantized":
            tensors[quantized_scales(source)] = ([24, 1], "BF16", bytes(48))
        write_safetensors(root / VISION_SHARD, tensors)
        norm = {"mlx": "language_model.model.norm.weight"}.get(
            source, "model.language_model.norm.weight"
        )
        write_safetensors(root / TEXT_SHARD, {norm: ([8], "BF16", bytes(16))})
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


def bf16_nearest(values):
    """The BF16 nearest each float32 value, ties to even (roundBfloat16)."""
    out = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        out += struct.pack("<H", (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16)
    return bytes(out)


def q8_0(rows, columns, seed):
    """A Q8_0 tensor of rows x columns: its bytes (an F16 scale and 32 int8
    quants a block) and its values, scale times quant, each exact in float32
    (an F16 scale and a 7-bit quant need 18 bits)."""
    raw, values = bytearray(), []
    for row in range(rows):
        for block in range(columns // 32):
            scale = struct.unpack(
                "<e", struct.pack("<e", 0.0123 * (1 + (row + block + seed) % 7))
            )[0]
            quants = [
                ((row * 31 + block * 17 + i * 7 + seed) % 255) - 127 for i in range(32)
            ]
            raw += struct.pack("<e", scale) + struct.pack("<32b", *quants)
            values.extend(scale * q for q in quants)
    return bytes(raw), values


# A tiny Inkling tower: depth 2, width 32, 2x2 patches (12 values, stored
# 16 wide), an MLP of 64, a 2x2 position grid, a merged width of 128 and a
# text width of 32 (inkling_layout in vision_preparation_test.cpp).
INKLING_HIDDEN, INKLING_FFN, INKLING_MERGED = 32, 64, 128
# The checkpoints' names for an mmproj's: the release's, under
# model.visual.backbone. with its projector's model.visual.proj_in and
# proj_out, and transformers' Inkling2VisionModel's, under model.vision_tower.
# (an MLX conversion's). A name ending in "." takes the mmproj's suffix.
INKLING_NAMES = {
    "release": {
        "tower": "model.visual.backbone.",
        "v.patch_embd.weight": "embeddings.patch_embedding.weight",
        "v.position_embd.weight": "embeddings.position_embedding",
        "v.class_embd": "embeddings.class_embedding",
        "v.class_position_embd.weight": "embeddings.cls_position_embedding",
        "v.pre_ln": "ln_pre.",
        "mm.1.weight": "/model.visual.proj_in.weight",
        "mm.2.weight": "/model.visual.proj_out.weight",
        "ln1": "ln_1.",
        "ln2": "ln_2.",
        "ffn_up": "mlp.c_fc.",
        "ffn_down": "mlp.c_proj.",
        "ls1": "ls_1.gamma",
        "ls2": "ls_2.gamma",
    },
    "transformers": {
        "tower": "model.vision_tower.",
        "v.patch_embd.weight": "embeddings.patch_embedding.weight",
        "v.position_embd.weight": "embeddings.position_embedding.weight",
        "v.class_embd": "embeddings.class_embedding",
        "v.class_position_embd.weight": "embeddings.cls_position_embedding",
        "v.pre_ln": "pre_layernorm.",
        "mm.1.weight": "merger.linear_1.weight",
        "mm.2.weight": "merger.linear_2.weight",
        "ln1": "layernorm_before.",
        "ln2": "layernorm_after.",
        "ffn_up": "mlp.fc1.",
        "ffn_down": "mlp.fc2.",
        "ls1": "lambda_1",
        "ls2": "lambda_2",
    },
}
# Either with the release's model.visual. renamed vision_tower.: oMLX's keep
# the backbone's names; thoddnn's nvfp4 conversion gives it transformers'
# instead, beside the release's position embedding and projector.
INKLING_NAMES["omlx"] = INKLING_NAMES["release"] | {
    "tower": "vision_tower.backbone.",
    "mm.1.weight": "/vision_tower.proj_in.weight",
    "mm.2.weight": "/vision_tower.proj_out.weight",
}
INKLING_NAMES["nvfp4"] = INKLING_NAMES["transformers"] | {
    "tower": "vision_tower.",
    "v.position_embd.weight": "embeddings.position_embedding",
    "mm.1.weight": "/vision_tower.proj_in.weight",
    "mm.2.weight": "/vision_tower.proj_out.weight",
}
for names in INKLING_NAMES.values():
    names |= {f"attn_{p}": f"self_attn.{p}_proj." for p in "qkv"}
    names["attn_out"] = "self_attn.out_proj."


def inkling_name(source, gguf):
    """The checkpoint `source`'s name for the mmproj's tensor `gguf`."""
    names = INKLING_NAMES[source]
    stem, _, suffix = gguf.rpartition(".")
    if gguf in names:
        name = names[gguf]
    elif stem == "v.pre_ln":
        name = names[stem]
    else:  # v.blk.N.PART
        _, _, layer, part = stem.split(".")
        name = f"layers.{layer}." + names[part]
    name = name + suffix if name.endswith(".") else name
    return name[1:] if name.startswith("/") else names["tower"] + name


def inkling_fixture(root, source="gguf", shift=0, case=None):
    """Writes the tiny tower from `source`, its matrices of 32-wide rows
    cycling through BF16, F16, F32 and, in an mmproj.gguf, Q8_0, its other
    tensors through the first three, and the vision/model.bin it must
    prepare, which does not depend on the source."""
    tensors, sections = {}, []
    dtypes = ("BF16", "F16", "F32", "Q8_0")
    quantized = source == "gguf"

    def values_of(index, count):
        # Exact in every float dtype, as the Qwen fixture's.
        return [((i * 7 + index * 11) % 255 - 127) / 64 for i in range(count)]

    def tensor(name, rows, columns, quantizable=True):
        """The GGUF tensor `name` of rows x columns and its values."""
        index = len(tensors)
        vector = rows == 1 or columns == 1
        quantizable = quantized and quantizable and not vector and columns % 32 == 0
        dtype = dtypes[(index + shift) % (4 if quantizable else 3)]
        shape = [rows * columns] if vector else [columns, rows]
        if name == "v.patch_embd.weight":
            shape = [2, 2, 3, rows]
        values = values_of(index, rows * columns)
        if dtype == "Q8_0":
            raw, values = q8_0(rows, columns, index)
        elif case == "q8-bias" and name == "v.blk.0.attn_out.bias":
            # A Q8_0 vector: one block of the 32 values.
            dtype, (raw, values) = "Q8_0", q8_0(1, rows * columns, index)
        else:
            raw = encode(values, dtype)
        tensors[name] = (shape, dtype, raw, values)
        return values

    def section(values, rows, columns, stored_columns=None):
        stored = stored_columns or columns
        out = []
        for row in range(rows):
            out.extend(values[row * columns : (row + 1) * columns])
            out.extend([0.0] * (stored - columns))
        sections.append(bf16_nearest(out))

    def single(name, rows, columns=1, stored_columns=None):
        section(tensor(name, rows, columns), rows, columns, stored_columns)

    def stacked(names, rows, columns=1):
        values = []
        for name in names:
            values += tensor(name, rows, columns)
        section(values, rows * len(names), columns)

    def zero(rows):
        section([0.0] * rows, rows, 1)

    def norm(name):
        single(name + ".weight", INKLING_HIDDEN)
        single(name + ".bias", INKLING_HIDDEN)

    single("v.patch_embd.weight", INKLING_HIDDEN, 12, 16)
    zero(INKLING_HIDDEN)
    single("v.position_embd.weight", 4, INKLING_HIDDEN)
    stacked(["v.class_embd", "v.class_position_embd.weight"], 1, INKLING_HIDDEN)
    norm("v.pre_ln")
    for layer in range(2):
        g = f"v.blk.{layer}."
        norm(g + "ln1")
        stacked(
            [g + p + ".weight" for p in ("attn_q", "attn_k", "attn_v")],
            INKLING_HIDDEN,
            INKLING_HIDDEN,
        )
        stacked(
            [g + p + ".bias" for p in ("attn_q", "attn_k", "attn_v")], INKLING_HIDDEN
        )
        single(g + "attn_out.weight", INKLING_HIDDEN, INKLING_HIDDEN)
        single(g + "attn_out.bias", INKLING_HIDDEN)
        single(g + "ls1.weight", INKLING_HIDDEN)
        norm(g + "ln2")
        single(g + "ffn_up.weight", INKLING_FFN, INKLING_HIDDEN)
        single(g + "ffn_up.bias", INKLING_FFN)
        single(g + "ffn_down.weight", INKLING_HIDDEN, INKLING_FFN)
        single(g + "ffn_down.bias", INKLING_HIDDEN)
        single(g + "ls2.weight", INKLING_HIDDEN)
    single("mm.1.weight", INKLING_MERGED, INKLING_MERGED)
    zero(INKLING_MERGED)
    single("mm.2.weight", INKLING_HIDDEN, INKLING_MERGED)
    zero(INKLING_HIDDEN)
    (root / "expected.bin").write_bytes(weight_file("MDFV0001", 2, 0, sections))
    if source != "gguf":
        inkling_checkpoint(root, source, tensors, case)
        return
    metadata = {
        "general.architecture": "clip",
        "clip.vision.projector_type": "inkling2v",
        "clip.vision.projection_dim": INKLING_HIDDEN,
        "clip.vision.image_size": 4,
        "clip.vision.patch_size": 2,
        "clip.vision.embedding_length": INKLING_HIDDEN,
        "clip.vision.feed_forward_length": INKLING_FFN,
        "clip.vision.block_count": 2,
        "clip.vision.attention.head_count": 2,
        "clip.vision.spatial_merge_size": 2,
        "clip.vision.attention.layer_norm_epsilon": 1e-5,
        "clip.vision.image_mean": [0.5] * 3,
        "clip.vision.image_std": [0.5] * 3,
    }
    if case == "epsilon":
        metadata["clip.vision.attention.layer_norm_epsilon"] = 1e-6
    if case == "projector":
        metadata["clip.vision.projector_type"] = "qwen3vl_merger"
    if case == "unused":
        tensors["v.post_ln.weight"] = (
            [INKLING_HIDDEN],
            "F32",
            bytes(4 * INKLING_HIDDEN),
            [0.0] * INKLING_HIDDEN,
        )
    if case == "shape":
        shape, dtype, raw, values = tensors["v.blk.1.attn_k.weight"]
        tensors["v.blk.1.attn_k.weight"] = (shape[::-1] + [1], dtype, raw, values)
    gguf_tensors = [
        (name, shape, GGML_TYPES[dtype], raw)
        for name, (shape, dtype, raw, _) in tensors.items()
    ]
    write_gguf(root / "mmproj.gguf", metadata, gguf_tensors)


def inkling_checkpoint(root, source, tensors, case):
    """Writes the mmproj's `tensors` as the checkpoint `source`'s: row-major
    shapes, and a patch embedding of [output, row, column, channel] where the
    mmproj's is [output, channel, row, column]."""
    out = {}
    for gguf, (shape, dtype, raw, values) in tensors.items():
        shape = shape[::-1]
        if gguf == "v.patch_embd.weight":
            rows, channels, side = shape[0], shape[1], shape[2]
            values = [
                values[row * channels * side * side + (c * side + y) * side + x]
                for row in range(rows)
                for y in range(side)
                for x in range(side)
                for c in range(channels)
            ]
            shape, raw = [rows, channels * side * side], encode(values, dtype)
        out[inkling_name(source, gguf)] = (shape, dtype, raw)
    key = inkling_name(source, "v.blk.1.attn_k.weight")
    if case == "shape":
        shape, dtype, raw = out[key]
        out[key] = (shape + [1], dtype, raw)
    if case == "dtype":
        shape, dtype, raw = out[key]
        out[key] = (shape, "F64", encode([1.0] * math.prod(shape), "F64"))
    if case == "unused":
        # A projector bias, which no Inkling tower has.
        bias = inkling_name(source, "mm.1.weight").removesuffix(".weight") + ".bias"
        out[bias] = ([INKLING_MERGED], "BF16", bytes(2 * INKLING_MERGED))
    if case == "quantized":
        out[key.removesuffix(".weight") + ".scales"] = (
            [INKLING_HIDDEN, 1],
            "BF16",
            bytes(64),
        )
    write_safetensors(root / VISION_SHARD, out)
    norm = "model.language_model.norm.weight"
    write_safetensors(root / TEXT_SHARD, {norm: ([INKLING_HIDDEN], "BF16", bytes(64))})


def prepare_inkling(binary, directory, source):
    loader = "gguf" if source == "gguf" else "safetensors"
    command = [
        binary,
        "inkling-" + loader,
        str(directory),
        str(directory / "expected.bin"),
    ]
    return subprocess.run(command, text=True, capture_output=True)


def check_inkling(binary, root):
    """The tiny Inkling tower prepares as its oracle from an mmproj of each
    dtype shift and from each checkpoint of each, and each invalid source is
    refused naming what is wrong."""
    runs = [("gguf", shift) for shift in range(4)]
    runs += [(source, shift) for source in INKLING_NAMES for shift in range(3)]
    for source, shift in runs:
        directory = root / f"inkling-{source}-{shift}"
        directory.mkdir()
        inkling_fixture(directory, source, shift)
        result = prepare_inkling(binary, directory, source)
        assert result.returncode == 0, ("inkling", source, shift, result.stderr)
    rejected = {
        ("gguf", "epsilon"): ("epsilon",),
        ("gguf", "projector"): ("unsupported vision GGUF architecture",),
        ("gguf", "unused"): ("v.post_ln.weight", "{}", "does not use"),
        ("gguf", "shape"): ("v.blk.1.attn_k.weight", "shape"),
        ("gguf", "q8-bias"): ("v.blk.0.attn_out.bias", "{}", "Q8_0"),
    }
    for source in INKLING_NAMES:
        key = inkling_name(source, "v.blk.1.attn_k.weight")
        rejected |= {
            (source, "shape"): (key, "shape"),
            (source, "dtype"): (key, "{}", "F64"),
            (source, "quantized"): (
                "quantized",
                key.removesuffix(".weight") + ".scales",
                "{}",
            ),
            (source, "unused"): (
                inkling_name(source, "mm.1.weight").removesuffix(".weight") + ".bias",
                "does not use",
            ),
        }
    for (source, case), parts in rejected.items():
        directory = root / f"inkling-{source}-{case}"
        directory.mkdir()
        inkling_fixture(directory, source, 0, case)
        result = prepare_inkling(binary, directory, source)
        errors = result.stderr.strip().splitlines()
        assert result.returncode == 1 and errors, (
            "inkling",
            source,
            case,
            result.stderr,
        )
        file = "mmproj.gguf" if source == "gguf" else VISION_SHARD
        named = [part.format(directory / file) for part in parts]
        assert all(part in errors[-1] for part in named), (
            "inkling",
            source,
            case,
            named,
            result.stderr,
        )


def quantized_scales(source):
    scales = "scales" if source == "mlx" else "weight_scale"
    return TOWER[source] + "blocks.0.attn.qkv." + scales


def prepare(binary, directory, source):
    # MLX's and transformers' towers are both the loader's safetensors source.
    loader = "gguf" if source == "gguf" else "safetensors"
    command = [binary, loader, str(directory), str(directory / "expected.bin")]
    return subprocess.run(command, text=True, capture_output=True)


def main():
    binary = str(Path(sys.argv[1]).resolve())
    # Every fixture prepares the same file: the values do not depend on the
    # source format or the dtypes.
    golden = json.loads(Path(sys.argv[2]).read_text())["vision_image"]
    with tempfile.TemporaryDirectory(prefix="splash-vision-preparation-") as temp:
        root = Path(temp)
        for source in ("mlx", "transformers", "gguf"):
            safetensors = source in TOWER
            file = VISION_SHARD if safetensors else "mmproj.gguf"
            # Every tensor as BF16, F16 and F32; BF16 is copied and exact F32
            # or F16 values become their BF16 bits.
            for shift in range(3):
                directory = root / f"{source}-{shift}"
                directory.mkdir()
                fixture(directory, source, shift)
                result = prepare(binary, directory, source)
                assert result.returncode == 0, (source, shift, result.stderr)
                digest = result.stdout.split()[0]
                assert digest == golden, (source, shift, digest)
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
            # What each refusal names: the tensor or key, the kind of problem
            # and, as "{}", the source file.
            inexact = (patch, "{}", "BF16")
            # The patch embedding is F32 with shift 2 and F16 with shift 1.
            rejected = {
                ("inexact-f32", 2): inexact,
                ("inexact-f16", 1): inexact,
                ("shape", 0): (patch, "shape"),
                ("dtype", 0): (merger, "{}", dtype),
            }
            if safetensors:
                rejected[("quantized", 0)] = (
                    "quantized",
                    quantized_scales(source),
                    "{}",
                )
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
                result = prepare(binary, directory, source)
                errors = result.stderr.strip().splitlines()
                assert result.returncode == 1 and errors, (source, case, result.stderr)
                named = [part.format(directory / file) for part in parts]
                assert all(part in errors[-1] for part in named), (
                    source,
                    case,
                    named,
                    result.stderr,
                )

        directory = root / "padding"
        directory.mkdir()
        result = subprocess.run(
            [binary, "padding", str(directory)], text=True, capture_output=True
        )
        assert result.returncode == 0, result.stderr
        check_inkling(binary, root)
        print(
            "Vision layouts from MLX, transformers and GGUF, exact BF16 conversion, "
            "rejected sources and zero padding, and Inkling's tower from an mmproj "
            "with Q8_0 matrices and from its release's and transformers' "
            "checkpoints and oMLX's and an nvfp4 conversion's under vision_tower. PASS"
        )


if __name__ == "__main__":
    main()
