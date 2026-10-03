"""Small standalone dense and MoE affine checkpoints and a DFlash2 draft
checkpoint, an independent byte-layout oracle of every prepared image and
their golden hashes."""

import argparse
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
    read_safetensors,
    safetensors_bytes,
    weight_file,
    write_safetensors,
)


def fixture(root, moe=False, rope_type_key="rope_type", rope_type="default"):
    tensors = {}
    quantization = {"bits": 4, "group_size": 64}

    def add(name, shape, dtype="BF16", data=None):
        size = math.prod(shape) * {"BF16": 2, "U32": 4, "F32": 4}[dtype]
        if data is None:
            # Byte i is (i * 31 + seed * 17) % 256, which repeats every 256.
            seed = len(tensors) + 1
            period = bytes((i * 31 + seed * 17) % 256 for i in range(256))
            data = (period * (size // 256 + 1))[:size]
        tensors[name] = (shape, dtype, data)
        return data

    def projection(name, rows, columns, bits=4, experts=1):
        lead = [experts] if experts > 1 else []
        add(name + ".weight", lead + [rows, columns * bits // 32], "U32")
        add(name + ".scales", lead + [rows, columns // 64])
        add(name + ".biases", lead + [rows, columns // 64])
        if bits != 4:
            quantization[name] = {"bits": bits, "group_size": 64}

    def packed(parts, rows, columns, bits=4, experts=1):
        # Per expert, each field's rows of the parts, then zero rows, in
        # [rows / 256][groups][256] tiles of the field's group bytes.
        result = bytearray()
        groups = columns // 64
        for expert in range(experts):
            for field, unit in (("weight", 8 * bits), ("scales", 2), ("biases", 2)):
                source = bytearray()
                for name in parts:
                    data = tensors[name + "." + field][2]
                    size = len(data) // experts
                    source += data[expert * size : (expert + 1) * size]
                source = source.ljust(rows * groups * unit, b"\0")
                for tile in range(0, rows, 256):
                    for group in range(groups):
                        for row in range(tile, tile + 256):
                            offset = (row * groups + group) * unit
                            result.extend(source[offset : offset + unit])
        return result

    expected = root / "expected"
    expected.mkdir()

    def image(name, magic, index, kind, sections):
        (expected / name).write_bytes(weight_file(magic, index, kind, sections))

    for layer in range(2):
        p = f"language_model.model.layers.{layer}."
        sections = [add(p + "input_layernorm.weight", [256])]
        if layer == 0:
            g = p + "linear_attn."
            names = [
                g + s for s in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
            ]
            for name, rows in zip(names, (512, 256, 4, 4)):
                projection(name, rows, 256)
            sections.append(packed(names, 1024, 256))
            sections.append(add(g + "conv1d.weight", [512, 4, 1]))
            if moe:
                # BF16 decay logarithms: 0.5, -1, 2 and 0.
                logarithms = (0.5, -1, 2, 0)
                add(g + "A_log", [4], data=bytes.fromhex("003f80bf00400000"))
            else:
                logarithms = (0, 0, 0, 0)
                add(g + "A_log", [4], "F32", bytes(16))
            # The decay -exp(A_log), rounded once from double to F32.
            sections.append(struct.pack("<4f", *(-math.exp(x) for x in logarithms)))
            sections.append(add(g + "dt_bias", [4]))
            sections.append(add(g + "norm.weight", [64]))
            projection(g + "out_proj", 256, 256)
            sections.append(packed([g + "out_proj"], 256, 256))
        else:
            a = p + "self_attn."
            names = [a + s for s in ("q_proj", "k_proj", "v_proj")]
            for name, rows in zip(names, (512, 128, 128)):
                projection(name, rows, 256)
            sections.append(packed(names, 768, 256))
            sections.append(add(a + "q_norm.weight", [64]))
            sections.append(add(a + "k_norm.weight", [64]))
            projection(a + "o_proj", 256, 256)
            sections.append(packed([a + "o_proj"], 256, 256))
        sections.append(add(p + "post_attention_layernorm.weight", [256]))
        if moe:
            # The 8-bit router and shared-expert scalar gate; the gate's one row is
            # padded to a 256-row tile.
            projection(p + "mlp.gate", 256, 256, bits=8)
            sections.append(packed([p + "mlp.gate"], 256, 256, bits=8))
            for name in ("gate_proj", "up_proj", "down_proj"):
                name = p + "mlp.switch_mlp." + name
                projection(name, 256, 256, experts=256)
                sections.append(packed([name], 256, 256, experts=256))
            for name in ("gate_proj", "up_proj", "down_proj"):
                name = p + "mlp.shared_expert." + name
                projection(name, 256, 256)
                sections.append(packed([name], 256, 256))
            projection(p + "mlp.shared_expert_gate", 1, 256, bits=8)
            sections.append(packed([p + "mlp.shared_expert_gate"], 256, 256, bits=8))
        else:
            for name, rows, columns in (
                ("gate_proj", 512, 256),
                ("up_proj", 512, 256),
                ("down_proj", 256, 512),
            ):
                name = p + "mlp." + name
                projection(name, rows, columns)
                sections.append(packed([name], rows, columns))
        magic = "MDFM0001" if moe else "MDFL0006"
        image(f"layer-{layer}.bin", magic, layer, layer, sections)
    norm = add("language_model.model.norm.weight", [256])
    projection("language_model.lm_head", 256, 256)
    image(
        "head.bin",
        "MDFM0002" if moe else "MDFL0002",
        2,
        2,
        [norm, packed(["language_model.lm_head"], 256, 256)],
    )
    projection("language_model.model.embed_tokens", 256, 256)
    image(
        "embedding.bin",
        "MDFE0001",
        256,
        256,
        [
            tensors["language_model.model.embed_tokens." + field][2]
            for field in ("weight", "scales", "biases")
        ],
    )
    config = {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 2,
        "hidden_size": 256,
        "vocab_size": 256,
        "head_dim": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 64,
        "linear_value_head_dim": 64,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 2,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "attn_output_gate": True,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "intermediate_size": 512,
        "layer_types": ["linear_attention", "full_attention"],
        "rope_parameters": {
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            rope_type_key: rope_type,
        },
    }
    if moe:
        del config["intermediate_size"]
        config |= {
            "model_type": "qwen3_5_moe_text",
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 256,
        }
    (root / "config.json").write_text(
        json.dumps({"text_config": config, "quantization": quantization})
    )
    write_safetensors(root / "model.safetensors", tensors)


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
    (mlx.core.quantize, its Metal kernel, whose maximum starts at 0): the
    packed codes, and the scale and bias as BF16."""
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
    return packed, struct.pack("<H", bf16(scale)), struct.pack("<H", bf16(bias))


def draft_fixture(root):
    """A two-layer DFlash2 checkpoint of width 256 as its repository releases
    it, config.json and BF16 safetensors, and the draft files its preparation
    must write: each projection quantized, every other tensor as stored."""
    tensors, values = {}, {}

    def add(name, shape, data=None):
        count = math.prod(shape)
        if data is None:
            # Multiples of 1/64 within 4 of zero, each exactly a BF16.
            seed = len(tensors) + 1
            data = [((i * 37 + seed * 11) % 509 - 254) / 64 for i in range(count)]
        values[name] = data
        tensors[name] = (shape, "BF16", struct.pack(f"<{count}H", *map(bf16, data)))
        return tensors[name][2]

    def quantized(names, columns):
        # The parts' rows quantized, each field in [rows / 256][groups][256]
        # tiles of its group bytes: codes, then scales, then biases.
        rows = [
            values[name][start : start + columns]
            for name in names
            for start in range(0, len(values[name]), columns)
        ]
        groups = columns // 64
        fields = [
            [quantized_group(row[g * 64 : (g + 1) * 64]) for g in range(groups)]
            for row in rows
        ]
        result = bytearray()
        for field in range(3):
            for tile in range(0, len(rows), 256):
                for group in range(groups):
                    for row in range(tile, tile + 256):
                        result += fields[row][group][field]
        return result

    def projection(name, rows, columns, data=None):
        add(name + ".weight", [rows, columns], data)
        return quantized([name + ".weight"], columns)

    expected = root / "expected"
    expected.mkdir()
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
        sections = [
            add(p + "input_layernorm.weight", [256]),
            add(p + "attention_conv.base_kernel", [2, 2, 256]),
            projection(p + "attention_conv.kernel_projection", 256, 256, dynamic),
        ]
        for name, rows in (("q_proj", 128), ("k_proj", 64), ("v_proj", 64)):
            add(a + name + ".weight", [rows, 256])
        sections.append(
            quantized([a + n + ".weight" for n in ("q_proj", "k_proj", "v_proj")], 256)
        )
        sections += [
            add(a + "q_norm.weight", [64]),
            add(a + "k_norm.weight", [64]),
            projection(a + "o_proj", 256, 128),
            add(p + "post_attention_layernorm.weight", [256]),
            add(p + "mlp_conv.base_kernel", [2, 2, 256]),
            projection(p + "mlp_conv.kernel_projection", 256, 256),
            projection(p + "mlp.gate_proj", 256, 256),
            projection(p + "mlp.up_proj", 256, 256),
            projection(p + "mlp.down_proj", 256, 256),
        ]
        (expected / f"layer-{layer}.bin").write_bytes(
            weight_file("MDFD0004", layer, 0, sections)
        )
    sections = [
        projection("fc", 256, 256),
        add("hidden_norm.weight", [256]),
        add("norm.weight", [256]),
        projection("candidate_selector.hidden_projection", 256, 256),
        add("candidate_selector.predecessor_codebook", [256, 256]),
        add("candidate_selector.successor_codebook", [256, 256]),
    ]
    (expected / "model.bin").write_bytes(weight_file("MDFD0004", 2, 1, sections))
    config = {
        "architectures": ["DFlash2DraftModel"],
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "vocab_size": 256,
        "dflash_config": {"selector_rank": 256},
    }
    (root / "config.json").write_text(json.dumps(config))
    write_safetensors(root / "model.safetensors", tensors)


def prepare(binary, metallib, root, kind, golden):
    command = [str(binary.resolve()), str(metallib.resolve()), str(root), kind]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    hashes = {
        name: digest
        for marker, name, digest in (
            line.split()
            for line in result.stdout.splitlines()
            if line.startswith("prepared ")
        )
    }
    assert hashes == golden, (kind, hashes)
    print(result.stdout.splitlines()[-1])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("metallib", type=Path)
    parser.add_argument("goldens", type=Path)
    args = parser.parse_args()
    goldens = json.loads(args.goldens.read_text())["affine_images"]
    with tempfile.TemporaryDirectory(prefix="splash-affine-preparation-") as directory:
        root = Path(directory) / "draft"
        root.mkdir()
        draft_fixture(root)
        prepare(args.binary, args.metallib, root, "draft", goldens["draft"])
        root = Path(directory) / "moe"
        root.mkdir()
        fixture(root, moe=True)
        prepare(args.binary, args.metallib, root, "moe", goldens["moe"])
        root = Path(directory) / "dense"
        root.mkdir()
        fixture(root)
        command = prepare(args.binary, args.metallib, root, "dense", goldens["dense"])
        # Fine-tunes may name the rope type by its older `type` key.
        legacy = Path(directory) / "legacy"
        legacy.mkdir()
        fixture(legacy, rope_type_key="type")
        prepare(args.binary, args.metallib, legacy, "dense", goldens["dense"])
        # Read from either key, a rope type other than the default is refused.
        refused = Path(directory) / "legacy-yarn"
        refused.mkdir()
        fixture(refused, rope_type_key="type", rope_type="yarn")
        result = subprocess.run(
            [*command[:2], str(refused), "dense"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0 and "rope_parameters.type" in result.stderr, (
            result.stderr
        )
        # Raw checkpoints need a different normalization convention. Refuse
        # their unsanitized convolution layout before publishing any weights.
        source = root / "model.safetensors"
        header, payload = read_safetensors(source)
        header["language_model.model.layers.0.linear_attn.conv1d.weight"]["shape"] = [
            512,
            1,
            4,
        ]
        before = set((root / "cache").glob("*/weights"))
        source.write_bytes(safetensors_bytes(header, payload))
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode != 0 and "conv1d.weight" in result.stderr, (
            result.stderr
        )
        assert set((root / "cache").glob("*/weights")) == before
        print("affine preparation: raw checkpoint rejected before conversion PASS")


if __name__ == "__main__":
    main()
