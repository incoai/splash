#!/usr/bin/env python3
"""Offline tool to pack true native 8-bit or mixed-precision Q8 model packages for Splash."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

try:
    import mlx.core as mx
    import numpy as np
except ImportError:
    print(
        "Error: mlx and numpy are required. Run with .mlx-env/bin/python",
        file=sys.stderr,
    )
    sys.exit(1)

ALIGNMENT = 16384

TARGET_Q8_LAYER_MAGIC = b"MDFL0008"
TARGET_Q8_HEAD_MAGIC = b"MDFL0008"
TARGET_Q8_EMBEDDING_MAGIC = b"MDFE0008"

DEFAULT_EXECUTION_GEOMETRY = {
    "allocation_extent_target_bytes": 134217728,
    "attention_history_group": 128,
    "draft_proposal_tokens": 7,
    "draft_query_rows": 8,
    "draft_sliding_window": 2048,
    "maximum_batch_width": 4,
    "prefill_token_budget": 2048,
    "target_kv_block_tokens": 32,
    "target_verify_rows": 8,
}

QWEN38_LAYOUT = {
    "layers": 64,
    "hidden_size": 5120,
    "vocabulary_size": 248320,
    "packed_gdn_width": 16640,
    "packed_full_width": 14336,
    "convolution_dim": 10240,
    "gdn_value_heads": 48,
    "gdn_head_dim": 128,
    "attention_width": 6144,
    "attention_head_dim": 256,
    "intermediate_size": 17408,
    "full_attention_period": 4,
}


def is_full_attention(layer_index: int, period: int = 4) -> bool:
    return period > 0 and (layer_index + 1) % period == 0


def align_offset(offset: int) -> int:
    return (offset + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def q8_packed_bytes(out_size: int, in_size: int) -> int:
    return (out_size * in_size * 17) // 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_packed_binary(
    dest_path: Path, magic: bytes, arg1: int, arg2: int, sections: List[bytes]
) -> int:
    """Write aligned binary weight file with 16-byte header and 16 KiB aligned sections."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("wb") as f:
        header = struct.pack("<8sII", magic, arg1, arg2)
        f.write(header)
        offset = len(header)
        for section in sections:
            aligned = align_offset(offset)
            pad = aligned - offset
            if pad:
                f.write(b"\x00" * pad)
                offset += pad
            f.write(section)
            offset += len(section)
        final_aligned = align_offset(offset)
        final_pad = final_aligned - offset
        if final_pad:
            f.write(b"\x00" * final_pad)
            offset += final_pad
    return offset


def read_q8_sections(path: Path, section_sizes: List[int]) -> List[bytes]:
    """Read sections from an aligned Splash Q8 weight file."""
    sections = []
    with path.open("rb") as f:
        offset = 16
        for size in section_sizes:
            offset = align_offset(offset)
            f.seek(offset)
            data = f.read(size)
            if len(data) != size:
                raise ValueError(
                    f"Truncated section in {path}: expected {size}, got {len(data)}"
                )
            sections.append(data)
            offset += size
    return sections


def pack_q8_projection_from_tensors(
    w_u8: np.ndarray, s_u16: np.ndarray, b_u16: np.ndarray, storage_n: int = 256
) -> bytes:
    """Pack weight (N, K), scales (N, K/64), biases (N, K/64) into Splash Q8 tiled layout.

    Layout:
    - weights: (num_tiles, num_groups, storage_n, 64) uint8
    - scales:  (num_tiles, num_groups, storage_n) uint16 (bfloat16)
    - biases:  (num_tiles, num_groups, storage_n) uint16 (bfloat16)
    """
    out_size, in_size = w_u8.shape
    if out_size % storage_n != 0 or in_size % 64 != 0:
        raise ValueError(
            f"Projection shape {w_u8.shape} incompatible with StorageN={storage_n}, Group=64"
        )
    num_tiles = out_size // storage_n
    num_groups = in_size // 64

    w_tiled = w_u8.reshape(num_tiles, storage_n, num_groups, 64).transpose(0, 2, 1, 3)
    s_tiled = s_u16.reshape(num_tiles, storage_n, num_groups).transpose(0, 2, 1)
    b_tiled = b_u16.reshape(num_tiles, storage_n, num_groups).transpose(0, 2, 1)

    return w_tiled.tobytes() + s_tiled.tobytes() + b_tiled.tobytes()


def tensor_to_u8(t: mx.array, shape: tuple[int, int]) -> np.ndarray:
    """Convert an MLX uint32 tensor to uint8 ndarray of given shape."""
    return np.frombuffer(bytes(memoryview(t)), dtype=np.uint8).reshape(shape)


def tensor_to_u16(t: mx.array, shape: tuple[int, int]) -> np.ndarray:
    """Convert an MLX bfloat16 tensor to uint16 ndarray of given shape."""
    return np.frombuffer(bytes(memoryview(t)), dtype=np.uint16).reshape(shape)


class SafetensorsStore:
    """Zero-copy tensor cache across sharded safetensors files."""

    def __init__(
        self, base_dir: Path, weight_map: Dict[str, str], max_cached_files: int = 2
    ):
        self.base_dir = base_dir
        self.weight_map = weight_map
        self.max_cached = max_cached_files
        self.cache: Dict[str, Dict[str, mx.array]] = {}

    def get(self, key: str) -> mx.array:
        fn = self.weight_map[key]
        if fn not in self.cache:
            if len(self.cache) >= self.max_cached:
                oldest = next(iter(self.cache))
                del self.cache[oldest]
                gc.collect()
            print(f"Loading shard {fn}...")
            self.cache[fn] = mx.load(str(self.base_dir / fn))
        return self.cache[fn][key]


def parse_upgrade_layers(spec: str, total_layers: int = 64) -> Set[int]:
    """Parse layer specification like 'all', '56-63', 'none', '56,57,58'."""
    spec = spec.strip().lower()
    if spec in ("all", "*"):
        return set(range(total_layers))
    if spec in ("none", ""):
        return set()

    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s.strip()), int(end_s.strip())
            for layer_index in range(start, end + 1):
                if 0 <= layer_index < total_layers:
                    layers.add(layer_index)
        else:
            layer_index = int(part)
            if 0 <= layer_index < total_layers:
                layers.add(layer_index)
    return layers


def pack_native_q8_model(
    source_splash_q8: Path,
    source_mtplx: Path,
    output_dir: Path,
    upgrade_layers_spec: str = "all",
    model_name: str = "incoai/Qwen3.8-27B-Splash-HQ",
    verbose: bool = True,
):
    start_time = time.time()
    geom = QWEN38_LAYOUT
    total_layers = geom["layers"]
    hidden_size = geom["hidden_size"]
    vocab_size = geom["vocabulary_size"]
    inter_size = geom["intermediate_size"]
    attn_width = geom["attention_width"]
    packed_gdn_w = geom["packed_gdn_width"]
    packed_full_w = geom["packed_full_width"]

    upgrade_layers = parse_upgrade_layers(upgrade_layers_spec, total_layers)
    if verbose:
        print(f"Target model: {model_name}")
        print(f"Output directory: {output_dir}")
        print(
            f"Upgraded layers ({len(upgrade_layers)}/{total_layers}): {sorted(list(upgrade_layers))}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    target_out = output_dir / "target"
    target_out.mkdir(parents=True, exist_ok=True)

    # 1. Symlink shared files (draft, vision, tokenizer)
    for folder in ("draft", "vision", "tokenizer"):
        src = source_splash_q8 / folder
        dst = output_dir / folder
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            (dst / item.name).symlink_to(item.resolve())
        if verbose:
            print(f"Populated {folder}/ with file symlinks -> {src}")

    # 2. Load MTPLX index and initialize store
    index_path = source_mtplx / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing index at {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    store = SafetensorsStore(source_mtplx, weight_map, max_cached_files=2)

    # Pre-calculate section sizes for reading non-projection parts from Splash Q8
    full_sizes = [
        hidden_size * 2,
        q8_packed_bytes(packed_full_w, hidden_size),
        256 * 2,
        256 * 2,
        q8_packed_bytes(hidden_size, attn_width),
        hidden_size * 2,
        q8_packed_bytes(inter_size, hidden_size),
        q8_packed_bytes(inter_size, hidden_size),
        q8_packed_bytes(hidden_size, inter_size),
    ]

    gdn_sizes = [
        hidden_size * 2,
        q8_packed_bytes(packed_gdn_w, hidden_size),
        10240 * 4 * 2,
        48 * 4,
        48 * 2,
        128 * 2,
        q8_packed_bytes(hidden_size, attn_width),
        hidden_size * 2,
        q8_packed_bytes(inter_size, hidden_size),
        q8_packed_bytes(inter_size, hidden_size),
        q8_packed_bytes(hidden_size, inter_size),
    ]

    # 3. Process embedding.bin
    if verbose:
        print("Packing target/embedding.bin with native 8-bit weights...")
    ew = tensor_to_u8(
        store.get("language_model.model.embed_tokens.weight"), (vocab_size, hidden_size)
    )
    es = tensor_to_u16(
        store.get("language_model.model.embed_tokens.scales"),
        (vocab_size, hidden_size // 64),
    )
    eb = tensor_to_u16(
        store.get("language_model.model.embed_tokens.biases"),
        (vocab_size, hidden_size // 64),
    )
    write_packed_binary(
        target_out / "embedding.bin",
        TARGET_Q8_EMBEDDING_MAGIC,
        vocab_size,
        hidden_size,
        [ew.tobytes(), es.tobytes(), eb.tobytes()],
    )
    if verbose:
        print("target/embedding.bin written successfully.")

    # 4. Process all 64 layers sequentially
    for layer_index in range(total_layers):
        dst_layer_path = target_out / f"layer-{layer_index}.bin"
        if layer_index not in upgrade_layers:
            src_layer_path = source_splash_q8 / "target" / f"layer-{layer_index}.bin"
            if dst_layer_path.exists() or dst_layer_path.is_symlink():
                dst_layer_path.unlink()
            dst_layer_path.symlink_to(src_layer_path.resolve())
            if verbose:
                print(f"Layer {layer_index:2d}: Kept baseline Q8 (symlinked)")
            continue

        full = is_full_attention(layer_index)
        src_layer_path = source_splash_q8 / "target" / f"layer-{layer_index}.bin"
        prefix = f"language_model.model.layers.{layer_index}."

        if full:
            # Read baseline sections from Splash Q8
            q8_base = read_q8_sections(src_layer_path, full_sizes)

            # 1. Packed QKV
            qw = tensor_to_u8(
                store.get(f"{prefix}self_attn.q_proj.weight"), (12288, hidden_size)
            )
            kw = tensor_to_u8(
                store.get(f"{prefix}self_attn.k_proj.weight"), (1024, hidden_size)
            )
            vw = tensor_to_u8(
                store.get(f"{prefix}self_attn.v_proj.weight"), (1024, hidden_size)
            )
            qkv_w = np.concatenate([qw, kw, vw], axis=0)

            qs = tensor_to_u16(
                store.get(f"{prefix}self_attn.q_proj.scales"),
                (12288, hidden_size // 64),
            )
            ks = tensor_to_u16(
                store.get(f"{prefix}self_attn.k_proj.scales"), (1024, hidden_size // 64)
            )
            vs = tensor_to_u16(
                store.get(f"{prefix}self_attn.v_proj.scales"), (1024, hidden_size // 64)
            )
            qkv_s = np.concatenate([qs, ks, vs], axis=0)

            qb = tensor_to_u16(
                store.get(f"{prefix}self_attn.q_proj.biases"),
                (12288, hidden_size // 64),
            )
            kb = tensor_to_u16(
                store.get(f"{prefix}self_attn.k_proj.biases"), (1024, hidden_size // 64)
            )
            vb = tensor_to_u16(
                store.get(f"{prefix}self_attn.v_proj.biases"), (1024, hidden_size // 64)
            )
            qkv_b = np.concatenate([qb, kb, vb], axis=0)

            packed_qkv = pack_q8_projection_from_tensors(
                qkv_w, qkv_s, qkv_b, storage_n=256
            )

            # 2. Output projection (o_proj)
            ow = tensor_to_u8(
                store.get(f"{prefix}self_attn.o_proj.weight"), (hidden_size, attn_width)
            )
            os_ = tensor_to_u16(
                store.get(f"{prefix}self_attn.o_proj.scales"),
                (hidden_size, attn_width // 64),
            )
            ob = tensor_to_u16(
                store.get(f"{prefix}self_attn.o_proj.biases"),
                (hidden_size, attn_width // 64),
            )
            packed_o = pack_q8_projection_from_tensors(ow, os_, ob, storage_n=256)

            # 3. MLP: gate, up, down
            gw = tensor_to_u8(
                store.get(f"{prefix}mlp.gate_proj.weight"), (inter_size, hidden_size)
            )
            gs = tensor_to_u16(
                store.get(f"{prefix}mlp.gate_proj.scales"),
                (inter_size, hidden_size // 64),
            )
            gb = tensor_to_u16(
                store.get(f"{prefix}mlp.gate_proj.biases"),
                (inter_size, hidden_size // 64),
            )
            packed_gate = pack_q8_projection_from_tensors(gw, gs, gb, storage_n=256)

            uw = tensor_to_u8(
                store.get(f"{prefix}mlp.up_proj.weight"), (inter_size, hidden_size)
            )
            us = tensor_to_u16(
                store.get(f"{prefix}mlp.up_proj.scales"),
                (inter_size, hidden_size // 64),
            )
            ub = tensor_to_u16(
                store.get(f"{prefix}mlp.up_proj.biases"),
                (inter_size, hidden_size // 64),
            )
            packed_up = pack_q8_projection_from_tensors(uw, us, ub, storage_n=256)

            dw = tensor_to_u8(
                store.get(f"{prefix}mlp.down_proj.weight"), (hidden_size, inter_size)
            )
            ds = tensor_to_u16(
                store.get(f"{prefix}mlp.down_proj.scales"),
                (hidden_size, inter_size // 64),
            )
            db = tensor_to_u16(
                store.get(f"{prefix}mlp.down_proj.biases"),
                (hidden_size, inter_size // 64),
            )
            packed_down = pack_q8_projection_from_tensors(dw, ds, db, storage_n=256)

            out_sections = [
                q8_base[0],  # inputNorm
                packed_qkv,  # native 8-bit QKV
                q8_base[2],  # q_norm
                q8_base[3],  # k_norm
                packed_o,  # native 8-bit o_proj
                q8_base[5],  # postAttentionNorm
                packed_gate,  # native 8-bit gate_proj
                packed_up,  # native 8-bit up_proj
                packed_down,  # native 8-bit down_proj
            ]

            write_packed_binary(
                dst_layer_path,
                TARGET_Q8_LAYER_MAGIC,
                layer_index,
                1,
                out_sections,
            )
            if verbose:
                print(f"Layer {layer_index:2d} (Full Attn): UPGRADED to native 8-bit")

        else:
            # GDN layer
            q8_base = read_q8_sections(src_layer_path, gdn_sizes)

            # 1. Packed GDN input projection
            # [in_proj_qkv (10240), in_proj_z (6144), in_proj_b (48), in_proj_a (48), pad (160)]
            qkv_w = tensor_to_u8(
                store.get(f"{prefix}linear_attn.in_proj_qkv.weight"),
                (10240, hidden_size),
            )
            z_w = tensor_to_u8(
                store.get(f"{prefix}linear_attn.in_proj_z.weight"), (6144, hidden_size)
            )
            b_w = tensor_to_u8(
                store.get(f"{prefix}linear_attn.in_proj_b.weight"), (48, hidden_size)
            )
            a_w = tensor_to_u8(
                store.get(f"{prefix}linear_attn.in_proj_a.weight"), (48, hidden_size)
            )
            pad_w = np.zeros((160, hidden_size), dtype=np.uint8)
            gdn_w = np.concatenate([qkv_w, z_w, b_w, a_w, pad_w], axis=0)

            qkv_s = tensor_to_u16(
                store.get(f"{prefix}linear_attn.in_proj_qkv.scales"),
                (10240, hidden_size // 64),
            )
            z_s = tensor_to_u16(
                store.get(f"{prefix}linear_attn.in_proj_z.scales"),
                (6144, hidden_size // 64),
            )
            b_s = tensor_to_u16(
                store.get(f"{prefix}linear_attn.in_proj_b.scales"),
                (48, hidden_size // 64),
            )
            a_s = tensor_to_u16(
                store.get(f"{prefix}linear_attn.in_proj_a.scales"),
                (48, hidden_size // 64),
            )
            pad_s = np.full((160, hidden_size // 64), 0x3F80, dtype=np.uint16)
            gdn_s = np.concatenate([qkv_s, z_s, b_s, a_s, pad_s], axis=0)

            qkv_b = tensor_to_u16(
                store.get(f"{prefix}linear_attn.in_proj_qkv.biases"),
                (10240, hidden_size // 64),
            )
            z_b = tensor_to_u16(
                store.get(f"{prefix}linear_attn.in_proj_z.biases"),
                (6144, hidden_size // 64),
            )
            b_b = tensor_to_u16(
                store.get(f"{prefix}linear_attn.in_proj_b.biases"),
                (48, hidden_size // 64),
            )
            a_b = tensor_to_u16(
                store.get(f"{prefix}linear_attn.in_proj_a.biases"),
                (48, hidden_size // 64),
            )
            pad_b = np.zeros((160, hidden_size // 64), dtype=np.uint16)
            gdn_b = np.concatenate([qkv_b, z_b, b_b, a_b, pad_b], axis=0)

            packed_gdn = pack_q8_projection_from_tensors(
                gdn_w, gdn_s, gdn_b, storage_n=256
            )

            # 2. Output projection (out_proj)
            ow = tensor_to_u8(
                store.get(f"{prefix}linear_attn.out_proj.weight"),
                (hidden_size, attn_width),
            )
            os_ = tensor_to_u16(
                store.get(f"{prefix}linear_attn.out_proj.scales"),
                (hidden_size, attn_width // 64),
            )
            ob = tensor_to_u16(
                store.get(f"{prefix}linear_attn.out_proj.biases"),
                (hidden_size, attn_width // 64),
            )
            packed_out = pack_q8_projection_from_tensors(ow, os_, ob, storage_n=256)

            # 3. MLP: gate, up, down
            gw = tensor_to_u8(
                store.get(f"{prefix}mlp.gate_proj.weight"), (inter_size, hidden_size)
            )
            gs = tensor_to_u16(
                store.get(f"{prefix}mlp.gate_proj.scales"),
                (inter_size, hidden_size // 64),
            )
            gb = tensor_to_u16(
                store.get(f"{prefix}mlp.gate_proj.biases"),
                (inter_size, hidden_size // 64),
            )
            packed_gate = pack_q8_projection_from_tensors(gw, gs, gb, storage_n=256)

            uw = tensor_to_u8(
                store.get(f"{prefix}mlp.up_proj.weight"), (inter_size, hidden_size)
            )
            us = tensor_to_u16(
                store.get(f"{prefix}mlp.up_proj.scales"),
                (inter_size, hidden_size // 64),
            )
            ub = tensor_to_u16(
                store.get(f"{prefix}mlp.up_proj.biases"),
                (inter_size, hidden_size // 64),
            )
            packed_up = pack_q8_projection_from_tensors(uw, us, ub, storage_n=256)

            dw = tensor_to_u8(
                store.get(f"{prefix}mlp.down_proj.weight"), (hidden_size, inter_size)
            )
            ds = tensor_to_u16(
                store.get(f"{prefix}mlp.down_proj.scales"),
                (hidden_size, inter_size // 64),
            )
            db = tensor_to_u16(
                store.get(f"{prefix}mlp.down_proj.biases"),
                (hidden_size, inter_size // 64),
            )
            packed_down = pack_q8_projection_from_tensors(dw, ds, db, storage_n=256)

            out_sections = [
                q8_base[0],  # inputNorm
                packed_gdn,  # native 8-bit GDN input
                q8_base[2],  # conv1d
                q8_base[3],  # A_log
                q8_base[4],  # dt_bias
                q8_base[5],  # mixerNorm
                packed_out,  # native 8-bit out_proj
                q8_base[7],  # postAttentionNorm
                packed_gate,  # native 8-bit gate_proj
                packed_up,  # native 8-bit up_proj
                packed_down,  # native 8-bit down_proj
            ]

            write_packed_binary(
                dst_layer_path,
                TARGET_Q8_LAYER_MAGIC,
                layer_index,
                0,
                out_sections,
            )
            if verbose:
                print(f"Layer {layer_index:2d} (GDN RNN  ): UPGRADED to native 8-bit")

    # 5. Process head.bin
    if verbose:
        print("\nPacking target/head.bin with native 8-bit lm_head...")
    head_q8_src = source_splash_q8 / "target" / "head.bin"
    head_sec = read_q8_sections(
        head_q8_src, [hidden_size * 2, q8_packed_bytes(vocab_size, hidden_size)]
    )
    norm_bytes = head_sec[0]

    hw = tensor_to_u8(
        store.get("language_model.lm_head.weight"), (vocab_size, hidden_size)
    )
    hs = tensor_to_u16(
        store.get("language_model.lm_head.scales"), (vocab_size, hidden_size // 64)
    )
    hb = tensor_to_u16(
        store.get("language_model.lm_head.biases"), (vocab_size, hidden_size // 64)
    )
    packed_logits = pack_q8_projection_from_tensors(hw, hs, hb, storage_n=256)

    write_packed_binary(
        target_out / "head.bin",
        TARGET_Q8_HEAD_MAGIC,
        total_layers,
        2,
        [norm_bytes, packed_logits],
    )
    if verbose:
        print("target/head.bin written successfully.")

    # 6. Generate manifest.json
    if verbose:
        print("\nComputing SHA-256 checksums and creating manifest.json...")
    artifacts = []
    for rel_dir in ("target", "draft", "vision", "tokenizer"):
        base_dir = output_dir / rel_dir
        if not base_dir.exists():
            continue
        for root_dir, _, files in os.walk(base_dir):
            for file_name in sorted(files):
                file_path = Path(root_dir) / file_name
                rel_path = file_path.relative_to(output_dir).as_posix()
                size = file_path.stat().st_size
                digest = sha256_file(file_path)
                artifacts.append(
                    {
                        "path": rel_path,
                        "sha256": digest,
                        "size": size,
                    }
                )

    artifacts.sort(key=lambda a: a["path"])

    manifest = {
        "model": "Qwen3.8-27B",
        "schema_version": 5,
        "format": {
            "name": "splash-packed-q8",
            "q8_bits": 8,
            "quant_group_size": 64,
            "storage_n": 256,
            "section_alignment_bytes": ALIGNMENT,
            "target_layer_magic": "MDFL0008",
            "draft_layer_magic": "MDFD0004",
            "vision_magic": "MDFV0001",
        },
        "execution_geometry": DEFAULT_EXECUTION_GEOMETRY,
        "artifacts": artifacts,
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    elapsed = time.time() - start_time
    if verbose:
        print("\n=======================================================")
        print(f"Successfully created {model_name} in {elapsed:.1f}s!")
        print(f"Location: {output_dir}")
        print(f"Total artifacts: {len(artifacts)}")
        print("=======================================================")


def main():
    parser = argparse.ArgumentParser(
        description="Pack true native Q8 or mixed-precision Q8 model."
    )
    parser.add_argument(
        "--source-splash-q8",
        type=Path,
        default=Path("install/models/incoai/Qwen3.8-27B-Splash-Q8"),
        help="Path to baseline Splash Q8 model directory.",
    )
    parser.add_argument(
        "--source-mtplx",
        type=Path,
        default=Path(
            os.path.expanduser("~/.mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Quality")
        ),
        help="Path to native 8-bit MTPLX checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination directory for the packed model package.",
    )
    parser.add_argument(
        "--upgrade-layers",
        type=str,
        default="all",
        help="Layers to upgrade: 'all', '56-63', 'none', or comma-separated indices.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="incoai/Qwen3.8-27B-Splash-HQ",
        help="Model identifier name.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet mode.",
    )

    args = parser.parse_args()
    pack_native_q8_model(
        source_splash_q8=args.source_splash_q8,
        source_mtplx=args.source_mtplx,
        output_dir=args.output,
        upgrade_layers_spec=args.upgrade_layers,
        model_name=args.model_name,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
