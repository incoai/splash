#!/usr/bin/env python3
"""Offline tool to convert and pack Q8 model packages for Splash."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required for pack_q8.py", file=sys.stderr)
    sys.exit(1)

ALIGNMENT = 16384

TARGET_Q8_LAYER_MAGIC = b"MDFL0008"
TARGET_Q8_HEAD_MAGIC = b"MDFL0008"
TARGET_Q8_EMBEDDING_MAGIC = b"MDFE0008"
DRAFT_LAYER_MAGIC = b"MDFD0004"
VISION_MAGIC = b"MDFV0001"

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def f32_to_bf16(arr: np.ndarray) -> np.ndarray:
    """Convert float32 numpy array to bfloat16 stored as uint16."""
    u32 = arr.astype(np.float32).view(np.uint32)
    lsb = (u32 >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    rounded = u32 + rounding_bias
    return (rounded >> 16).astype(np.uint16)


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    """Convert uint16 array of bfloat16 elements to float32."""
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def quantize_q8_projection(weight: np.ndarray, storage_n: int = 256) -> bytes:
    """Quantize a 2D float32 weight matrix (N, K) into packed Splash Q8 format.

    Packed layout:
    - weights: N * K bytes (uint8) in tiled (T, G, StorageN, 64) order
    - scales: (N * K / 32) bytes (bfloat16)
    - biases: (N * K / 32) bytes (bfloat16)
    Total bytes: N * K * 17 / 16.
    """
    out_size, in_size = weight.shape
    if out_size % storage_n != 0 or in_size % 64 != 0:
        raise ValueError(
            f"Weight shape {weight.shape} incompatible with StorageN={storage_n}, Group=64"
        )
    num_tiles = out_size // storage_n
    num_groups = in_size // 64

    # Reshape: (T, StorageN, G, 64) -> (T, G, StorageN, 64)
    w_tiled = weight.reshape(num_tiles, storage_n, num_groups, 64).transpose(0, 2, 1, 3)

    w_min = w_tiled.min(axis=-1)
    w_max = w_tiled.max(axis=-1)
    diff = w_max - w_min
    scales_f32 = np.where(diff > 0, diff / 255.0, 1.0)
    biases_f32 = w_min

    scales_bf16 = f32_to_bf16(scales_f32)
    biases_bf16 = f32_to_bf16(biases_f32)

    s_recon = bf16_to_f32(scales_bf16)
    z_recon = bf16_to_f32(biases_bf16)

    q = np.clip(
        np.round((w_tiled - z_recon[..., None]) / s_recon[..., None]), 0, 255
    ).astype(np.uint8)

    return q.tobytes() + scales_bf16.tobytes() + biases_bf16.tobytes()


def transcode_q4_to_q8_projection(
    q4_bytes: bytes, out_size: int, in_size: int
) -> bytes:
    """Transcode a packed Splash Q4 projection to Splash Q8 projection without loss."""
    elements = out_size * in_size
    weight_bytes = elements // 2
    param_bytes = elements // 32

    q4_weights = np.frombuffer(q4_bytes[:weight_bytes], dtype=np.uint8)
    q4_scales = np.frombuffer(
        q4_bytes[weight_bytes : weight_bytes + param_bytes], dtype=np.uint16
    )
    q4_biases = q4_bytes[weight_bytes + param_bytes : weight_bytes + 2 * param_bytes]

    # 1. Unpack 4-bit nibbles and map to 8-bit affine range (multiply by 17)
    low = (q4_weights & 0x0F) * 17
    high = ((q4_weights >> 4) & 0x0F) * 17
    q8_weights = np.empty(elements, dtype=np.uint8)
    q8_weights[0::2] = low
    q8_weights[1::2] = high

    # 2. Divide scales by 17 in bfloat16
    s4_f32 = bf16_to_f32(q4_scales)
    s8_f32 = s4_f32 * (1.0 / 17.0)
    s8_bf16 = f32_to_bf16(s8_f32)

    # 3. Packed Q8: weights + scales + biases
    return q8_weights.tobytes() + s8_bf16.tobytes() + q4_biases


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


def read_q4_sections(path: Path, section_sizes: List[int]) -> List[bytes]:
    """Read sections from an aligned Splash Q4 weight file."""
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


def q4_packed_bytes(out_size: int, in_size: int) -> int:
    return (out_size * in_size // 16) * 9


def q8_packed_bytes(out_size: int, in_size: int) -> int:
    return (out_size * in_size * 17) // 16


def convert_from_splash_q4(
    source_root: Path,
    output_root: Path,
    symlink_shared: bool = False,
    verbose: bool = False,
) -> Dict:
    """Convert an existing Splash Q4 package into a Splash Q8 package."""
    manifest_path = source_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest in {source_root}")

    with manifest_path.open("r", encoding="utf-8") as f:
        source_manifest = json.load(f)

    model_name = source_manifest.get("model", "Qwen3.8-27B")
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Share or copy tokenizer, vision, draft
    for folder in ("tokenizer", "vision", "draft"):
        src_folder = source_root / folder
        dst_folder = output_root / folder
        if not src_folder.exists():
            continue
        if dst_folder.exists():
            shutil.rmtree(dst_folder)
        if symlink_shared:
            dst_folder.symlink_to(src_folder.resolve(), target_is_directory=True)
            if verbose:
                print(f"Symlinked {folder}/ -> {src_folder}")
        else:
            shutil.copytree(src_folder, dst_folder)
            if verbose:
                print(f"Copied {folder}/")

    target_out = output_root / "target"
    target_out.mkdir(parents=True, exist_ok=True)

    geom = QWEN38_LAYOUT
    hidden_size = geom["hidden_size"]
    vocab_size = geom["vocabulary_size"]
    inter_size = geom["intermediate_size"]
    attn_width = geom["attention_width"]
    packed_gdn_w = geom["packed_gdn_width"]
    packed_full_w = geom["packed_full_width"]

    # 2. Transcode target/embedding.bin
    if verbose:
        print("Transcoding target/embedding.bin...")
    emb_path = source_root / "target" / "embedding.bin"
    emb_elements = vocab_size * hidden_size
    emb_q4_sizes = [
        emb_elements // 2,
        emb_elements // 32,
        emb_elements // 32,
    ]
    emb_q4_sections = read_q4_sections(emb_path, emb_q4_sizes)
    # Unpack embedding weights
    packed_u8 = np.frombuffer(emb_q4_sections[0], dtype=np.uint8)
    low = (packed_u8 & 0x0F) * 17
    high = ((packed_u8 >> 4) & 0x0F) * 17
    emb_q8_weights = np.empty(emb_elements, dtype=np.uint8)
    emb_q8_weights[0::2] = low
    emb_q8_weights[1::2] = high

    s4_f32 = bf16_to_f32(np.frombuffer(emb_q4_sections[1], dtype=np.uint16))
    s8_bf16 = f32_to_bf16(s4_f32 * (1.0 / 17.0))
    emb_q8_biases = emb_q4_sections[2]

    write_packed_binary(
        target_out / "embedding.bin",
        TARGET_Q8_EMBEDDING_MAGIC,
        vocab_size,
        hidden_size,
        [emb_q8_weights.tobytes(), s8_bf16.tobytes(), emb_q8_biases],
    )

    # 3. Transcode target/head.bin
    if verbose:
        print("Transcoding target/head.bin...")
    head_path = source_root / "target" / "head.bin"
    head_q4_sizes = [
        hidden_size * 2,
        q4_packed_bytes(vocab_size, hidden_size),
    ]
    head_q4_sections = read_q4_sections(head_path, head_q4_sizes)
    head_q8_logits = transcode_q4_to_q8_projection(
        head_q4_sections[1], vocab_size, hidden_size
    )
    write_packed_binary(
        target_out / "head.bin",
        TARGET_Q8_HEAD_MAGIC,
        geom["layers"],
        2,
        [head_q4_sections[0], head_q8_logits],
    )

    # 4. Transcode target/layer-*.bin
    total_layers = geom["layers"]
    start_time = time.time()
    for layer_idx in range(total_layers):
        full_attn = is_full_attention(layer_idx, geom["full_attention_period"])
        layer_file = f"layer-{layer_idx}.bin"
        src_layer_path = source_root / "target" / layer_file

        if full_attn:
            q4_sizes = [
                hidden_size * 2,
                q4_packed_bytes(packed_full_w, hidden_size),
                geom["attention_head_dim"] * 2,
                geom["attention_head_dim"] * 2,
                q4_packed_bytes(hidden_size, attn_width),
                hidden_size * 2,
                q4_packed_bytes(inter_size, hidden_size),
                q4_packed_bytes(inter_size, hidden_size),
                q4_packed_bytes(hidden_size, inter_size),
            ]
            q4_sec = read_q4_sections(src_layer_path, q4_sizes)
            q8_sec = [
                q4_sec[0],
                transcode_q4_to_q8_projection(q4_sec[1], packed_full_w, hidden_size),
                q4_sec[2],
                q4_sec[3],
                transcode_q4_to_q8_projection(q4_sec[4], hidden_size, attn_width),
                q4_sec[5],
                transcode_q4_to_q8_projection(q4_sec[6], inter_size, hidden_size),
                transcode_q4_to_q8_projection(q4_sec[7], inter_size, hidden_size),
                transcode_q4_to_q8_projection(q4_sec[8], hidden_size, inter_size),
            ]
        else:
            q4_sizes = [
                hidden_size * 2,
                q4_packed_bytes(packed_gdn_w, hidden_size),
                geom["convolution_dim"] * 4 * 2,
                geom["gdn_value_heads"] * 4,
                geom["gdn_value_heads"] * 2,
                geom["gdn_head_dim"] * 2,
                q4_packed_bytes(hidden_size, attn_width),
                hidden_size * 2,
                q4_packed_bytes(inter_size, hidden_size),
                q4_packed_bytes(inter_size, hidden_size),
                q4_packed_bytes(hidden_size, inter_size),
            ]
            q4_sec = read_q4_sections(src_layer_path, q4_sizes)
            q8_sec = [
                q4_sec[0],
                transcode_q4_to_q8_projection(q4_sec[1], packed_gdn_w, hidden_size),
                q4_sec[2],
                q4_sec[3],
                q4_sec[4],
                q4_sec[5],
                transcode_q4_to_q8_projection(q4_sec[6], hidden_size, attn_width),
                q4_sec[7],
                transcode_q4_to_q8_projection(q4_sec[8], inter_size, hidden_size),
                transcode_q4_to_q8_projection(q4_sec[9], inter_size, hidden_size),
                transcode_q4_to_q8_projection(q4_sec[10], hidden_size, inter_size),
            ]

        write_packed_binary(
            target_out / layer_file,
            TARGET_Q8_LAYER_MAGIC,
            layer_idx,
            1 if full_attn else 0,
            q8_sec,
        )

        if verbose and (layer_idx + 1) % 8 == 0:
            elapsed = time.time() - start_time
            print(
                f"Transcoded {layer_idx + 1}/{total_layers} layers ({elapsed:.1f}s)..."
            )

    # 5. Build artifacts list and manifest
    if verbose:
        print("Generating manifest.json and calculating artifact hashes...")
    artifacts = []
    # Collect all files under target, draft, vision, tokenizer
    for rel_dir in ("target", "draft", "vision", "tokenizer"):
        base_dir = output_root / rel_dir
        if not base_dir.exists():
            continue
        for root_dir, _, files in os.walk(base_dir):
            for file_name in sorted(files):
                file_path = Path(root_dir) / file_name
                rel_path = file_path.relative_to(output_root).as_posix()
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
        "model": model_name,
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
        "execution_geometry": source_manifest.get(
            "execution_geometry", DEFAULT_EXECUTION_GEOMETRY
        ),
        "artifacts": artifacts,
    }

    if "upstream" in source_manifest:
        manifest["upstream"] = source_manifest["upstream"]
    if "converter" in source_manifest:
        manifest["converter"] = source_manifest["converter"]

    out_manifest_path = output_root / "manifest.json"
    with out_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    if verbose:
        print(f"Successfully packaged Splash Q8 model to {output_root}")
    return manifest


def create_synthetic_q8_package(output_root: Path, verbose: bool = False) -> Dict:
    """Create a minimal synthetic Q8 package for testing and validation."""
    output_root.mkdir(parents=True, exist_ok=True)
    target_dir = output_root / "target"
    draft_dir = output_root / "draft"
    vision_dir = output_root / "vision"
    tok_dir = output_root / "tokenizer"

    for d in (target_dir, draft_dir, vision_dir, tok_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Tokenizer config
    tok_config = {
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "vocab_size": 248320,
            "max_position_embeddings": 32768,
        }
    }
    with (tok_dir / "config.json").open("w") as f:
        json.dump(tok_config, f, indent=2)

    for name in (
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ):
        (tok_dir / name).write_text("{}\n")

    # 2. Minimal synthetic draft & vision models
    # draft/model.bin (header: MDFD0004, layers=5, type=1)
    write_packed_binary(
        draft_dir / "model.bin", DRAFT_LAYER_MAGIC, 5, 1, [b"\x00" * 1024]
    )
    for layer_index in range(5):
        write_packed_binary(
            draft_dir / f"layer-{layer_index}.bin",
            DRAFT_LAYER_MAGIC,
            layer_index,
            0,
            [b"\x00" * 1024],
        )

    # vision/model.bin (header: MDFV0001, depth=1, type=0)
    write_packed_binary(vision_dir / "model.bin", VISION_MAGIC, 1, 0, [b"\x00" * 1024])

    # 3. Minimal synthetic target files
    geom = QWEN38_LAYOUT
    hidden_size = geom["hidden_size"]
    vocab_size = geom["vocabulary_size"]

    # embedding
    write_packed_binary(
        target_dir / "embedding.bin",
        TARGET_Q8_EMBEDDING_MAGIC,
        vocab_size,
        hidden_size,
        [b"\x00" * 1024, b"\x00" * 512, b"\x00" * 512],
    )

    # head
    write_packed_binary(
        target_dir / "head.bin",
        TARGET_Q8_HEAD_MAGIC,
        geom["layers"],
        2,
        [b"\x00" * 1024, b"\x00" * 2048],
    )

    # 64 layers
    for layer_index in range(geom["layers"]):
        full = is_full_attention(layer_index)
        num_sections = 9 if full else 11
        sections = [b"\x00" * 1024 for _ in range(num_sections)]
        write_packed_binary(
            target_dir / f"layer-{layer_index}.bin",
            TARGET_Q8_LAYER_MAGIC,
            layer_index,
            1 if full else 0,
            sections,
        )

    artifacts = []
    for rel_dir in ("target", "draft", "vision", "tokenizer"):
        base_dir = output_root / rel_dir
        for root_dir, _, files in os.walk(base_dir):
            for file_name in sorted(files):
                file_path = Path(root_dir) / file_name
                rel_path = file_path.relative_to(output_root).as_posix()
                artifacts.append(
                    {
                        "path": rel_path,
                        "sha256": sha256_file(file_path),
                        "size": file_path.stat().st_size,
                    }
                )
    artifacts.sort(key=lambda a: a["path"])

    manifest = {
        "model": "Qwen3.8-27B-Synthetic-Q8",
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
    with (output_root / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    if verbose:
        print(f"Created synthetic Q8 package at {output_root}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Pack weights into Splash Q8 package format."
    )
    parser.add_argument(
        "--source-splash",
        type=Path,
        help="Path to an existing Splash Q4 package to convert to Q8.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination directory for the packed Q8 package.",
    )
    parser.add_argument(
        "--symlink-shared",
        action="store_true",
        help="Symlink draft, vision, and tokenizer directories instead of copying.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a minimal synthetic Q8 package for testing.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose conversion logs."
    )

    args = parser.parse_args()

    if args.synthetic:
        create_synthetic_q8_package(args.output, verbose=args.verbose)
        return

    if args.source_splash:
        convert_from_splash_q4(
            args.source_splash,
            args.output,
            symlink_shared=args.symlink_shared,
            verbose=args.verbose,
        )
        return

    parser.error("Must specify either --source-splash or --synthetic")


if __name__ == "__main__":
    main()
