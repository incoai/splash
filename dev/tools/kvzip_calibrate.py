"""Calibrates ZBF16 KV bases (runtime/metal/abi/KvZip.h) from captured BF16 pages.

Capture: serve the model with --kv-format bf16 --max-cache-disk 40G
--persistent-cache --cache-dir DIR, send prompts as long as the contexts it
will serve (keys' exponent windows drift with RoPE position), stop the server
cleanly, then pass DIR/<namespace>/kv.slots here. Each slot is one BF16 page:
per attention layer the keys, then the values, of every KV head, 32 tokens x
256 dimensions (keys token-major, values dimension-major).

For every (layer, tensor, KV head, dimension) the base of the 8- and the
16-binade exponent window that covers the most elements is written to
runtime/model/kvzip/<family>.inc, which runtime/model/KvZipBases.cpp
compiles in (ZBF16 codes against the 16-binade window). It also reports how
many escapes ZBF16 slabs of these bases need.

    .venv/bin/python -m dev.tools.kvzip_calibrate --family Qwen3.8-27B \\
        --layers 16 --kv-heads 4 DIR/<namespace>/kv.slots
"""

from __future__ import annotations

import argparse
import pathlib
import re

import numpy as np

TOKENS, DIMENSIONS = 32, 256
ESCAPES = 256  # a slab's escape table (runtime/metal/abi/KvZip.h)
REPOSITORY = pathlib.Path(__file__).resolve().parents[2]


def page_view(path: pathlib.Path, layers: int, heads: int) -> np.ndarray:
    """uint16 [slots, layers, 2 (K, V), heads, 32 * 256] over the slot file."""
    page_bytes = layers * 2 * heads * TOKENS * DIMENSIONS * 2
    slot_bytes = (page_bytes + 16383) // 16384 * 16384
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    slots = raw.size // slot_bytes
    view = np.lib.stride_tricks.as_strided(
        raw.view(np.uint16), shape=(slots, page_bytes // 2), strides=(slot_bytes, 2)
    )
    return view.reshape(slots, layers, 2, heads, TOKENS * DIMENSIONS)


def token_major(slab: np.ndarray, value: bool) -> np.ndarray:
    """[..., 32*256] slab -> [..., 32 tokens, 256 dims] as ZBF16 stores both tensors."""
    shaped = slab.reshape(
        slab.shape[:-1] + ((DIMENSIONS, TOKENS) if value else (TOKENS, DIMENSIONS))
    )
    return np.swapaxes(shaped, -1, -2) if value else shaped


def windows(exponents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """exponents [N, 256] of one (layer, tensor, head) -> best 8 and 16 windows per dimension."""
    hist = np.zeros((DIMENSIONS, 256), np.int64)
    for dimension in range(DIMENSIONS):
        hist[dimension] = np.bincount(exponents[:, dimension], minlength=256)
    cumulative = np.concatenate(
        [np.zeros((DIMENSIONS, 1), np.int64), np.cumsum(hist, 1)], 1
    )
    base3 = np.argmax(cumulative[:, 8:] - cumulative[:, :-8], 1)
    base4 = np.argmax(cumulative[:, 16:] - cumulative[:, :-16], 1)
    return base3.astype(np.uint8), base4.astype(np.uint8)


def slab_escapes(exponents: np.ndarray, base4: np.ndarray) -> np.ndarray:
    """exponents [slabs, 32, 256] -> elements per slab outside the 16-binade window."""
    delta = exponents.astype(np.int16) - base4.astype(np.int16)
    return ((delta < 0) | (delta >= 16)).reshape(exponents.shape[0], -1).sum(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("slots", type=pathlib.Path, nargs="+")
    parser.add_argument(
        "--family", required=True, help="the model family, e.g. Qwen3.8-27B"
    )
    parser.add_argument("--layers", type=int, required=True, help="attention layers")
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument(
        "--output", type=pathlib.Path, help="default: runtime/model/kvzip/<family>.inc"
    )
    args = parser.parse_args()

    views = [page_view(path, args.layers, args.kv_heads) for path in args.slots]
    pages = sum(view.shape[0] for view in views)
    print(f"{pages} pages ({pages * TOKENS} tokens) from {len(views)} file(s)")
    bases = np.zeros((args.layers, 2, args.kv_heads, 2, DIMENSIONS), np.uint8)
    # Per slab, the elements outside their dimension's 16-binade window,
    # which the slab's escape table must hold.
    escapes = []
    for layer in range(args.layers):
        for tensor in range(2):
            for head in range(args.kv_heads):
                exponents = np.concatenate(
                    [
                        (
                            token_major(
                                np.asarray(view[:, layer, tensor, head]), tensor == 1
                            )
                            >> 7
                        )
                        & 0xFF
                        for view in views
                    ]
                ).astype(np.uint8)
                base3, base4 = windows(exponents.reshape(-1, DIMENSIONS))
                bases[layer, tensor, head] = base3, base4
                escapes.append(slab_escapes(exponents, base4))
    escapes = np.concatenate(escapes)
    print(
        f"escapes per slab: mean {escapes.mean():.2f}, 99.9th percentile {np.percentile(escapes, 99.9):.0f}, "
        f"largest {int(escapes.max())} of {ESCAPES}; slabs over the table: {int((escapes > ESCAPES).sum())}"
    )

    slug = re.sub(r"[^0-9A-Za-z]+", "_", args.family).strip("_").lower()
    output = args.output or REPOSITORY / "runtime" / "model" / "kvzip" / f"{slug}.inc"
    output.parent.mkdir(parents=True, exist_ok=True)
    flat = bases.reshape(-1)
    rows = [
        ", ".join(str(int(value)) for value in flat[index : index + 32])
        for index in range(0, flat.size, 32)
    ]
    output.write_text(
        f"// Generated by dev/tools/kvzip_calibrate.py from {pages * TOKENS} tokens of BF16 KV.\n"
        f"// {args.family}: {args.layers} attention layers x (keys, values) x {args.kv_heads} KV heads x\n"
        f"// (base3[256], base4[256]).\n" + ",\n".join(rows) + "\n"
    )
    print(
        f"wrote {output.relative_to(REPOSITORY) if output.is_relative_to(REPOSITORY) else output}"
    )


if __name__ == "__main__":
    main()
