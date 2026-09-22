#!/usr/bin/env python3
"""CPU-only projected-input oracle for the inspected Flash-Next QSA geometry.

Only NumPy is imported; this module never imports MLX, creates a GPU device, or
loads model weights. BF16 is represented by rounded float32 arrays in memory and
uint16 IEEE BF16 payloads in fixtures. The output ends before o_proj.

Audited source boundaries (omlx's vendored qwen4_exp):
  language.py:1084-1089  RMSNorm uses float32 (1 + weight), then input-dtype cast.
  language.py:1504-1517  q projection is [24, 256 Q + 256 gate]; K/V are [2,256].
  language.py:1534-1543  index projection is [4 query heads + 1 raw key,128].
  qsa_fast.py:180-189    mean4 float32 -> BF16 -> norm BF16 -> block-start RoPE.
  qsa_fast.py:136-142    score = sum(ReLU(dot) over four heads) / sqrt(128), F32.
  qsa_fast.py:626-724    completed causal blocks plus zero-to-three tail tokens.
  qsa_fast.py:734-744    softmax F32 -> BF16 probabilities -> BF16 output.
  language.py:1576-1580 output * sigmoid(gate) keeps BF16 at both operations.

Sigmoid follows the audited MLX GPU unary-precise BF16 contract in
build/release/flash/flash_hc_reference.py:84-120: BF16(exp(abs(BF16(gate)))) ->
BF16(1+exp) -> BF16(FTZ_F32(1/denominator)); positive is BF16(1-tail), selected
by BF16(gate)<0. Reciprocal FTZ precedes BF16 rounding and applies only to that
F32 intermediate. The BF16 output product preserves BF16 subnormals. NumPy's
F32 exp is a CPU approximation to the audited precise GPU intrinsic.

The canonical native path uses F32 RoPE trigonometry/products/addition, then a
BF16 cast, as installed mlx-vlm rope_utils.py's fused Metal arm:128-162 does.
``rope_mode=staged_bf16`` exposes the fallback:589-594,670-671 with BF16 cos/sin,
each product, then addition. ``final_bf16`` is an alias for ``fused_f32``.
Native omlx main attention also retains F32 probabilities until its BF16 output;
``probability_mode=float32`` exposes that alternative. Neither alternative claims
bit identity with Metal trigonometry, fast::exp2, or GPU reduction trees.

Equal FP32 scores choose the highest block IDs at the cutoff, then selected IDs
are returned in chronological U32[rows,512] order with UINT_MAX padding. Token
slots and probabilities use the native fixed width2051, with invalid slots -1
and zero probability; tails follow the visible selected-block prefix. This is
an explicit deterministic policy
matching the native omlx radix top-k tie branch. NumPy/BLAS and scalar F32 sums
may differ in their last bits, so selection margins are included in the trace.

Examples (every file write needs an explicit new destination):
  python flash_qsa_reference.py self-test
  python flash_qsa_reference.py fixture --output /tmp/qsa.npz --tokens 2055
  python flash_qsa_reference.py evaluate --input /tmp/qsa.npz --output /tmp/out.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

QUERY_HEADS = 24
KV_HEADS = 2
HEAD_DIM = 256
INDEX_HEADS = 4
INDEX_DIM = 128
ROTARY_DIM = 64
COMPRESSION = 4
BLOCK_BUDGET = 512
THETA = np.float32(10_000_000.0)
EPSILON = np.float32(1e-6)
TIE_POLICY = "highest_block_id_at_equal_score_then_chronological"


def bf16_bits(value: Any) -> np.ndarray:
    """Round float32 to BF16, round-to-nearest ties-to-even; preserve NaNs."""
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16
    nan = (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    # A low-payload NaN must not round to infinity. Quiet it in the BF16 payload.
    return np.where(nan, (bits >> 16) | np.uint32(0x0040), rounded).astype(np.uint16)


def bf16_from_bits(value: Any) -> np.ndarray:
    return (np.asarray(value, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16(value: Any) -> np.ndarray:
    return bf16_from_bits(bf16_bits(value))


def _sum_f32(value: np.ndarray, axis: int, reduction: str) -> np.ndarray:
    if reduction == "numpy":
        return np.sum(value, axis=axis, dtype=np.float32)
    moved = np.moveaxis(value, axis, -1)
    out = np.zeros(moved.shape[:-1], dtype=np.float32)
    for i in range(moved.shape[-1]):
        out = np.add(out, moved[..., i], dtype=np.float32)
    return out


def _dot_f32(left: np.ndarray, right: np.ndarray, reduction: str) -> np.ndarray:
    """[...,D] @ [D,N], with optional sequential scalar F32 accumulation."""
    if reduction == "numpy":
        return np.asarray(left @ right, dtype=np.float32)
    out = np.zeros((*left.shape[:-1], right.shape[-1]), dtype=np.float32)
    for d in range(left.shape[-1]):
        out = np.add(out, left[..., d, None] * right[d], dtype=np.float32)
    return out


def rms_norm(
    value: Any,
    weight: Any,
    *,
    norm_convention: str = "OnePlusWeight",
    epsilon: float = float(EPSILON),
    reduction: str = "numpy",
) -> np.ndarray:
    """Qwen4 norm with an explicitly supplied checkpoint audit convention."""
    x = bf16(value)
    w = np.asarray(weight, dtype=np.float32)
    if w.shape != (x.shape[-1],):
        raise ValueError("norm weight must match the final dimension")
    if norm_convention == "OnePlusWeight":
        scale = np.add(w, np.float32(1.0), dtype=np.float32)
    elif norm_convention == "DirectGamma":
        scale = w
    else:
        raise ValueError("norm_convention must be audited OnePlusWeight or DirectGamma")
    variance = _sum_f32(x * x, -1, reduction) / np.float32(x.shape[-1])
    inverse = np.float32(1.0) / np.sqrt(variance + np.float32(epsilon))
    # The scale stays F32. It must not be rounded to BF16 before this product.
    return bf16((x * inverse[..., None]) * scale)


def partial_rope(
    value: Any,
    positions: Any,
    *,
    rope_mode: str = "fused_f32",
) -> np.ndarray:
    """Token-major [T,...,D], rotating pairs d and d+32 only in first64."""
    x = bf16(value)
    positions = np.asarray(positions, dtype=np.int64)
    if x.ndim < 2 or x.shape[-1] < ROTARY_DIM or positions.shape != (x.shape[0],):
        raise ValueError(
            "RoPE needs [tokens,...,dim>=64] and one text position per token"
        )
    exponent = np.arange(0, ROTARY_DIM, 2, dtype=np.float32) / np.float32(ROTARY_DIM)
    inv_freq = np.float32(1.0) / np.power(THETA, exponent)
    angles = positions.astype(np.float32)[:, None] * inv_freq[None]
    coefficient_shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (ROTARY_DIM // 2,)
    cosine = np.cos(angles).reshape(coefficient_shape)
    sine = np.sin(angles).reshape(coefficient_shape)
    lo, hi = x[..., :32], x[..., 32:64]
    if rope_mode == "staged_bf16":
        cosine, sine = bf16(cosine), bf16(sine)
        rotated_lo = bf16(bf16(lo * cosine) + bf16((-hi) * sine))
        rotated_hi = bf16(bf16(hi * cosine) + bf16(lo * sine))
    elif rope_mode in {"fused_f32", "final_bf16"}:
        rotated_lo = bf16(lo * cosine - hi * sine)
        rotated_hi = bf16(hi * cosine + lo * sine)
    else:
        raise ValueError("rope_mode must be staged_bf16 or fused_f32")
    return np.concatenate((rotated_lo, rotated_hi, x[..., 64:]), axis=-1)


def split_query_gate(q_projection: Any) -> tuple[np.ndarray, np.ndarray]:
    q = bf16(q_projection)
    if q.ndim != 2 or q.shape[1] != QUERY_HEADS * HEAD_DIM * 2:
        raise ValueError("Q projection must be [query_tokens,12288]")
    heads = q.reshape(q.shape[0], QUERY_HEADS, HEAD_DIM * 2)
    return heads[..., :HEAD_DIM], heads[..., HEAD_DIM:]


def pool_index_keys(
    raw_keys: Any,
    positions: Any,
    weight: Any,
    *,
    norm_convention: str = "OnePlusWeight",
    rope_mode: str = "fused_f32",
    reduction: str = "numpy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = bf16(raw_keys)
    pos = np.asarray(positions, dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != INDEX_DIM or pos.shape != (raw.shape[0],):
        raise ValueError(
            "raw index keys must be [cached_tokens,128] with aligned positions"
        )
    blocks = raw.shape[0] // COMPRESSION
    grouped = raw[: blocks * COMPRESSION].reshape(blocks, COMPRESSION, INDEX_DIM)
    mean = bf16(_sum_f32(grouped, 1, reduction) / np.float32(COMPRESSION))
    normalized = rms_norm(
        mean, weight, norm_convention=norm_convention, reduction=reduction
    )
    rotated = partial_rope(
        mean if blocks == 0 else normalized, pos[: blocks * 4 : 4], rope_mode=rope_mode
    )
    return mean, normalized, rotated


def indexer_scores(
    index_queries: np.ndarray,
    pooled_keys: np.ndarray,
    *,
    reduction: str = "numpy",
) -> np.ndarray:
    q, k = (
        np.asarray(index_queries, dtype=np.float32),
        np.asarray(pooled_keys, dtype=np.float32),
    )
    if (
        q.ndim != 3
        or q.shape[1:] != (INDEX_HEADS, INDEX_DIM)
        or k.shape[1:] != (INDEX_DIM,)
    ):
        raise ValueError("index queries/pooled keys must be [M,4,128]/[blocks,128]")
    dots = _dot_f32(q, k.T, reduction)
    positive = np.maximum(dots, np.float32(0.0))
    return _sum_f32(positive, 1, reduction) / np.float32(np.sqrt(np.float32(INDEX_DIM)))


def select_blocks(
    scores: Any, query_indices: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return causal scores, chronological block IDs/counts, token IDs, margin."""
    s = np.asarray(scores, dtype=np.float32)
    q = np.asarray(query_indices, dtype=np.int64)
    if (
        s.ndim != 2
        or q.shape != (s.shape[0],)
        or np.any(q < 0)
        or not np.all(np.isfinite(s))
    ):
        raise ValueError(
            "selection needs finite [query,blocks] scores and nonnegative query IDs"
        )
    # Match the native fixed ABI even when the completed block bank is short.
    width = BLOCK_BUDGET
    selected = np.full((len(q), width), np.iinfo(np.uint32).max, dtype=np.uint32)
    token_ids = np.full((len(q), width * COMPRESSION + 3), -1, dtype=np.int32)
    counts = np.empty(len(q), dtype=np.int32)
    masked = s.copy()
    margins = np.full(len(q), np.float32(np.inf), dtype=np.float32)
    for row, absolute in enumerate(q):
        complete = int((absolute + 1) // COMPRESSION)
        if complete > s.shape[1]:
            raise ValueError("visible complete blocks exceed the supplied block bank")
        masked[row, complete:] = np.finfo(np.float32).min
        count = min(complete, BLOCK_BUDGET)
        counts[row] = count
        if complete <= BLOCK_BUDGET:
            picked = np.arange(complete, dtype=np.int32)
        else:
            ids = np.arange(complete, dtype=np.int32)
            ranked = np.lexsort((-ids, -s[row, :complete]))
            picked = np.sort(ids[ranked[:BLOCK_BUDGET]])
            margins[row] = (
                s[row, ranked[BLOCK_BUDGET - 1]] - s[row, ranked[BLOCK_BUDGET]]
            )
        selected[row, :count] = picked
        token_ids[row, : count * 4] = (picked[:, None] * 4 + np.arange(4)).reshape(-1)
        tail = np.arange(complete * 4, absolute + 1, dtype=np.int32)
        token_ids[row, count * 4 : count * 4 + len(tail)] = tail
    return masked, selected, counts, token_ids, margins


def gathered_gqa(
    queries: Any,
    keys: Any,
    values: Any,
    token_ids: Any,
    *,
    probability_mode: str = "bf16",
    reduction: str = "numpy",
) -> tuple[np.ndarray, np.ndarray]:
    """F32 scores/softmax, source-portable BF16 probabilities, BF16 output."""
    q, k, v = bf16(queries), bf16(keys), bf16(values)
    ids = np.asarray(token_ids, dtype=np.int32)
    if (
        q.shape[1:] != (QUERY_HEADS, HEAD_DIM)
        or k.shape[1:] != (KV_HEADS, HEAD_DIM)
        or v.shape != k.shape
    ):
        raise ValueError("GQA expects [M,24,256] Q and matching [N,2,256] K/V")
    if ids.ndim != 2 or ids.shape[0] != q.shape[0] or np.any(ids >= len(k)):
        raise ValueError(
            "selected token IDs must align with queries and fit the K/V bank"
        )
    if probability_mode not in {"bf16", "float32"}:
        raise ValueError("probability_mode must be bf16 or float32")
    output = np.zeros_like(q)
    probabilities = np.zeros((len(q), QUERY_HEADS, ids.shape[1]), dtype=np.float32)
    groups = QUERY_HEADS // KV_HEADS
    for row in range(len(q)):
        valid_slots = np.flatnonzero(ids[row] >= 0)
        if len(valid_slots) == 0:
            raise ValueError("every causal query must have at least one visible token")
        rows = ids[row, valid_slots]
        for kv_head in range(KV_HEADS):
            begin, end = kv_head * groups, (kv_head + 1) * groups
            logits = _dot_f32(
                q[row, begin:end], k[rows, kv_head].T, reduction
            ) / np.float32(16.0)
            exponent = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probability = exponent / _sum_f32(exponent, -1, reduction)[:, None]
            if probability_mode == "bf16":
                probability = bf16(probability)
            probabilities[row, begin:end, valid_slots] = probability.T
            output[row, begin:end] = bf16(
                _dot_f32(probability, v[rows, kv_head], reduction)
            )
    return output, probabilities


def sigmoid_gate(output: Any, gate: Any) -> tuple[np.ndarray, np.ndarray]:
    """Audited unary-precise BF16 sigmoid with reciprocal-F32 FTZ, then product."""
    x, g = bf16(output), bf16(gate)
    if x.shape != g.shape:
        raise ValueError("output and per-head gate shapes must match")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        exponent = bf16(np.exp(np.abs(g), dtype=np.float32))
        denominator = bf16(np.add(np.float32(1.0), exponent, dtype=np.float32))
        tail = np.divide(np.float32(1.0), denominator, dtype=np.float32)
        words = tail.view(np.uint32)
        subnormal = ((words & np.uint32(0x7F800000)) == 0) & (
            (words & np.uint32(0x007FFFFF)) != 0
        )
        # Flush the reciprocal's F32 subnormals before BF16 rounding, preserving
        # their sign. Do not flush BF16 inputs or the final BF16 product.
        tail = bf16(
            np.where(subnormal, words & np.uint32(0x80000000), words)
            .astype(np.uint32)
            .view(np.float32)
        )
        positive = bf16(np.subtract(np.float32(1.0), tail, dtype=np.float32))
        sigmoid = bf16(np.where(g < 0, tail, positive))
    return sigmoid, bf16(x * sigmoid)


def projected_qsa(
    q_projection: Any,
    k_projection: Any,
    v_projection: Any,
    index_projection: Any,
    norm_weights: dict[str, Any],
    *,
    query_offset: int | None = None,
    positions: Any | None = None,
    norm_convention: str = "OnePlusWeight",
    rope_mode: str = "fused_f32",
    probability_mode: str = "bf16",
    reduction: str = "numpy",
    norm_conventions: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate contiguous batch-one projected inputs, retaining stage evidence."""
    if reduction not in {"numpy", "scalar"}:
        raise ValueError("reduction must be numpy or scalar")
    query, gate = split_query_gate(q_projection)
    kp, vp, ip = bf16(k_projection), bf16(v_projection), bf16(index_projection)
    tokens = kp.shape[0]
    if kp.shape != (tokens, 512) or vp.shape != kp.shape or ip.shape != (tokens, 640):
        raise ValueError("K/V/index projections must be [N,512]/[N,512]/[N,640]")
    if query_offset is None:
        query_offset = tokens - len(query)
    if query_offset < 0 or query_offset + len(query) > tokens or not len(query):
        raise ValueError(
            "query rows must cover a nonempty contiguous portion of the cache"
        )
    pos = (
        np.arange(tokens, dtype=np.int64)
        if positions is None
        else np.asarray(positions, dtype=np.int64)
    )
    if pos.shape != (tokens,) or np.any(pos < 0):
        raise ValueError(
            "positions must contain one nonnegative text position per cached token"
        )
    if set(norm_weights) != {"q", "k", "index_q", "index_k"}:
        raise ValueError("norm_weights must supply q, k, index_q, and index_k")
    conventions = (
        {name: norm_convention for name in norm_weights}
        if norm_conventions is None
        else norm_conventions
    )
    if set(conventions) != set(norm_weights):
        raise ValueError(
            "norm_conventions must independently specify all four norm weights"
        )
    absolute = np.arange(query_offset, query_offset + len(query), dtype=np.int64)
    qn = rms_norm(
        query, norm_weights["q"], norm_convention=conventions["q"], reduction=reduction
    )
    kn = rms_norm(
        kp.reshape(tokens, KV_HEADS, HEAD_DIM),
        norm_weights["k"],
        norm_convention=conventions["k"],
        reduction=reduction,
    )
    qr = partial_rope(qn, pos[absolute], rope_mode=rope_mode)
    kr = partial_rope(kn, pos, rope_mode=rope_mode)
    index = ip.reshape(tokens, INDEX_HEADS + 1, INDEX_DIM)
    iqn = rms_norm(
        index[absolute, :INDEX_HEADS],
        norm_weights["index_q"],
        norm_convention=conventions["index_q"],
        reduction=reduction,
    )
    iqr = partial_rope(iqn, pos[absolute], rope_mode=rope_mode)
    mean, pooled_norm, pooled = pool_index_keys(
        index[:, INDEX_HEADS],
        pos,
        norm_weights["index_k"],
        rope_mode=rope_mode,
        norm_convention=conventions["index_k"],
        reduction=reduction,
    )
    scores = indexer_scores(iqr, pooled, reduction=reduction)
    masked, blocks, counts, selected_tokens, margins = select_blocks(scores, absolute)
    attention, probability = gathered_gqa(
        qr,
        kr,
        vp.reshape(tokens, KV_HEADS, HEAD_DIM),
        selected_tokens,
        probability_mode=probability_mode,
        reduction=reduction,
    )
    sigmoid, gated = sigmoid_gate(attention, gate)
    return {
        "query_indices": absolute,
        "positions": pos,
        "query_normalized": qn,
        "query_rotated": qr,
        "key_normalized": kn,
        "key_rotated": kr,
        "index_query_normalized": iqn,
        "index_query_rotated": iqr,
        "index_raw_keys": index[:, INDEX_HEADS],
        "index_pool_mean": mean,
        "index_pool_normalized": pooled_norm,
        "index_pool_rotated": pooled,
        "index_scores_unmasked": scores,
        "index_scores": masked,
        "selected_block_ids": blocks,
        "selected_block_counts": counts,
        "selected_token_ids": selected_tokens,
        "selection_margin": margins,
        "attention_probabilities": probability,
        "attention_output": attention,
        "gate": gate,
        "gate_sigmoid": sigmoid,
        "gated_output": gated.reshape(len(query), QUERY_HEADS * HEAD_DIM),
    }


def fixture_inputs(
    tokens: int, queries: int, seed: int, convention: str, tied: bool
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def random(shape):
        return bf16(rng.standard_normal(shape, dtype=np.float32))

    weight_center = np.float32(0.0 if convention == "OnePlusWeight" else 1.0)
    inputs = {
        "q_projection_bf16": bf16_bits(random((queries, 12288))),
        "k_projection_bf16": bf16_bits(random((tokens, 512))),
        "v_projection_bf16": bf16_bits(random((tokens, 512))),
        "index_projection_bf16": bf16_bits(random((tokens, 640))),
        "positions": np.arange(tokens, dtype=np.int64),
        "query_offset": np.asarray(tokens - queries, dtype=np.int64),
    }
    for name, dim in (
        ("q", HEAD_DIM),
        ("k", HEAD_DIM),
        ("index_q", INDEX_DIM),
        ("index_k", INDEX_DIM),
    ):
        inputs[f"{name}_norm_weight"] = bf16(
            weight_center + random((dim,)) * np.float32(0.03)
        )
    if tied:
        inputs["index_projection_bf16"][-queries:, : INDEX_HEADS * INDEX_DIM] = 0
    return inputs


def evaluate_inputs(
    inputs: dict[str, np.ndarray], **options: Any
) -> dict[str, np.ndarray]:
    projections = [
        bf16_from_bits(inputs[f"{name}_projection_bf16"])
        for name in ("q", "k", "v", "index")
    ]
    weights = {
        name: inputs[f"{name}_norm_weight"] for name in ("q", "k", "index_q", "index_k")
    }
    return projected_qsa(
        *projections,
        weights,
        query_offset=int(inputs["query_offset"]),
        positions=inputs["positions"],
        **options,
    )


def write_npz(
    path: Path,
    inputs: dict[str, np.ndarray],
    trace: dict[str, np.ndarray],
    options: dict[str, Any],
) -> None:
    metadata = {
        "schema": "splash.flash_qsa.cpu.v1",
        "backend": "numpy_cpu",
        "output_endpoint": "before_o_proj",
        "query_heads": QUERY_HEADS,
        "kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "index_heads": INDEX_HEADS,
        "index_dim": INDEX_DIM,
        "rotary_dim": ROTARY_DIM,
        "theta": float(THETA),
        "norm_epsilon": float(EPSILON),
        "compression": COMPRESSION,
        "block_budget": BLOCK_BUDGET,
        "tie_policy": TIE_POLICY,
        **options,
    }
    arrays = {
        **inputs,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for name, value in trace.items():
        arrays[f"expected_{name}"] = value
        if value.dtype == np.float32 and name not in {
            "index_scores",
            "index_scores_unmasked",
            "selection_margin",
        }:
            if (
                name != "attention_probabilities"
                or options["probability_mode"] == "bf16"
            ):
                arrays[f"expected_{name}_bf16"] = bf16_bits(value)
    # Refuse silent overwrites, and avoid a default fixture/artifact directory.
    with path.open("xb") as destination:
        np.savez_compressed(destination, **arrays)


def self_test() -> None:
    halfway = np.asarray(
        [0x3F808000, 0x3F818000, 0x80000000, 0x7F800001], dtype=np.uint32
    ).view(np.float32)
    np.testing.assert_array_equal(bf16_bits(halfway), [0x3F80, 0x3F82, 0x8000, 0x7FC0])
    sentinel = np.arange(12288, dtype=np.float32)[None]
    query, gate = split_query_gate(sentinel)
    np.testing.assert_array_equal(query[0, 1], bf16(sentinel[0, 512:768]))
    np.testing.assert_array_equal(gate[0, 1], bf16(sentinel[0, 768:1024]))
    x = bf16(np.arange(1, 257, dtype=np.float32))[None]
    np.testing.assert_array_equal(
        rms_norm(x, np.zeros(256)),
        rms_norm(x, np.ones(256), norm_convention="DirectGamma"),
    )
    scale_weight = np.full(256, np.float32(1.0 / 256), dtype=np.float32)
    correct = rms_norm(x, scale_weight)
    incorrectly_rounded_scale = rms_norm(
        x, bf16(np.float32(1.0) + scale_weight), norm_convention="DirectGamma"
    )
    assert np.any(correct != incorrectly_rounded_scale), (
        "norm must preserve float32 one-plus scale"
    )
    np.testing.assert_array_equal(partial_rope(x, [0]), x)
    rotated = partial_rope(x, [1703], rope_mode="staged_bf16")
    np.testing.assert_array_equal(rotated[..., 64:], x[..., 64:])
    lo, hi = x[0, 0], x[0, 32]
    cosine, sine = bf16(np.cos(np.float32(1703))), bf16(np.sin(np.float32(1703)))
    assert rotated[0, 0] == bf16(bf16(lo * cosine) + bf16(-hi * sine))
    raw = np.repeat(np.asarray([1, 2, 4, 8], dtype=np.float32)[:, None], 128, axis=1)
    mean, normalized, pooled = pool_index_keys(
        raw, [17, 18, 19, 20], np.zeros(128), rope_mode="staged_bf16"
    )
    np.testing.assert_array_equal(mean, np.full((1, 128), np.float32(3.75)))
    np.testing.assert_array_equal(
        pooled, partial_rope(normalized, [17], rope_mode="staged_bf16")
    )
    raw_halfway = np.repeat(
        np.asarray([1, 1.0078125, 1, 1.0078125], dtype=np.float32)[:, None], 128, axis=1
    )
    np.testing.assert_array_equal(
        pool_index_keys(raw_halfway, [0, 1, 2, 3], np.zeros(128))[0], np.ones((1, 128))
    )
    index_q = np.zeros((1, 4, 128), dtype=np.float32)
    index_k = np.zeros((1, 128), dtype=np.float32)
    index_q[:, :, :2] = 1
    index_k[0, :2] = [1, np.float32(1.0 / 256)]
    expected_score = np.float32(4.015625) / np.sqrt(np.float32(128))
    np.testing.assert_array_equal(indexer_scores(index_q, index_k), [[expected_score]])
    assert indexer_scores(index_q, index_k)[0, 0] != np.float32(4) / np.sqrt(
        np.float32(128)
    )
    scores = np.zeros((5, 514), dtype=np.float32)
    ids = np.asarray([2047, 2048, 2050, 2051, 2054], dtype=np.int64)
    masked, blocks, counts, tokens, margin = select_blocks(scores, ids)
    np.testing.assert_array_equal(counts, [512] * 5)
    np.testing.assert_array_equal(blocks[:3], np.broadcast_to(np.arange(512), (3, 512)))
    np.testing.assert_array_equal(
        blocks[3:], np.broadcast_to(np.arange(1, 513), (2, 512))
    )
    np.testing.assert_array_equal(tokens[1, -3:], [2048, -1, -1])
    np.testing.assert_array_equal(tokens[2, -3:], [2048, 2049, 2050])
    np.testing.assert_array_equal(tokens[4, -3:], [2052, 2053, 2054])
    assert masked[0, 512] == np.finfo(np.float32).min and margin[4] == 0
    short = select_blocks(np.zeros((2, 2), dtype=np.float32), [0, 4])
    np.testing.assert_array_equal(short[3][0, :2], [0, -1])
    np.testing.assert_array_equal(short[3][1, :6], [0, 1, 2, 3, 4, -1])
    np.testing.assert_array_equal(
        short[1][0], np.full(512, np.iinfo(np.uint32).max, dtype=np.uint32)
    )
    zero_query = np.zeros((1, 24, 256), dtype=np.float32)
    zero_key = np.zeros((1, 2, 256), dtype=np.float32)
    constant_value = np.repeat(
        np.asarray([3, 7], dtype=np.float32)[None, :, None], 256, axis=2
    )
    grouped_output, grouped_probability = gathered_gqa(
        zero_query, zero_key, constant_value, [[0, -1]]
    )
    np.testing.assert_array_equal(grouped_output[:, :12], np.full((1, 12, 256), 3))
    np.testing.assert_array_equal(grouped_output[:, 12:], np.full((1, 12, 256), 7))
    np.testing.assert_array_equal(grouped_probability[..., 1], np.zeros((1, 24)))
    sigmoid, gated = sigmoid_gate(grouped_output, zero_query)
    np.testing.assert_array_equal(sigmoid, np.full_like(sigmoid, 0.5))
    np.testing.assert_array_equal(gated, grouped_output * np.float32(0.5))
    sigmoid_traps = bf16([0.5, -2.0])
    staged_sigmoid, staged_product = sigmoid_gate(
        np.ones(2, dtype=np.float32), sigmoid_traps
    )
    np.testing.assert_array_equal(bf16_bits(staged_sigmoid), [0x3F20, 0x3DF5])
    np.testing.assert_array_equal(staged_product, staged_sigmoid)
    final_only_sigmoid = bf16(
        np.float32(1.0) / (np.float32(1.0) + np.exp(-sigmoid_traps))
    )
    np.testing.assert_array_equal(bf16_bits(final_only_sigmoid), [0x3F1F, 0x3DF4])
    assert np.all(bf16_bits(staged_sigmoid) != bf16_bits(final_only_sigmoid))
    saturation_gates = bf16([-90, -64, -8, -0.0, 0.0, 8, 64, 90])
    saturation, _ = sigmoid_gate(np.ones(8, dtype=np.float32), saturation_gates)
    np.testing.assert_array_equal(
        bf16_bits(saturation[[0, 3, 4, 7]]), [0, 0x3F00, 0x3F00, 0x3F80]
    )
    assert np.all(saturation[:3] <= np.float32(0.5)) and np.all(
        saturation[5:] >= np.float32(0.5)
    )
    finite_extremes = bf16_from_bits([0xFF7F, 0x7F7F])
    np.testing.assert_array_equal(sigmoid_gate(np.ones(2), finite_extremes)[0], [0, 1])
    reciprocal_gates = bf16_from_bits([0xC2AE, 0xC2AF, 0xC2B0, 0xC2B1])
    reciprocal_sigmoid, _ = sigmoid_gate(np.ones(4), reciprocal_gates)
    np.testing.assert_array_equal(bf16_bits(reciprocal_sigmoid), [0x00B3, 0, 0, 0])
    unflushed_tail = bf16(
        np.float32(1.0) / bf16(np.float32(1.0) + bf16(np.exp(np.abs(reciprocal_gates))))
    )
    np.testing.assert_array_equal(
        bf16_bits(unflushed_tail), [0x00B3, 0x006D, 0x0042, 0x0028]
    )
    negative_min_normal = bf16_from_bits([0x8080])
    tiny_sigmoid, tiny_product = sigmoid_gate(negative_min_normal, negative_min_normal)
    np.testing.assert_array_equal(bf16_bits(tiny_sigmoid), [0x3F00])
    np.testing.assert_array_equal(bf16_bits(tiny_product), [0x8040])
    _, tiny_products = sigmoid_gate(
        np.repeat(negative_min_normal, 3), bf16([0, 0.5, -2])
    )
    np.testing.assert_array_equal(bf16_bits(tiny_products), [0x8040, 0x8050, 0x800F])
    small = fixture_inputs(9, 3, 1926, "OnePlusWeight", False)
    vector = evaluate_inputs(small)
    scalar = evaluate_inputs(small, reduction="scalar")
    np.testing.assert_array_equal(
        vector["selected_token_ids"], scalar["selected_token_ids"]
    )
    np.testing.assert_allclose(
        vector["index_scores_unmasked"],
        scalar["index_scores_unmasked"],
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        vector["gated_output"], scalar["gated_output"], rtol=8e-3, atol=4e-3
    )
    assert np.any(bf16_bits(vector["attention_probabilities"]) != 0)
    print(
        "CPU QSA checks passed: BF16 RNE, layout, norm scale, partial RoPE, pool stages, F32 index scores, causality/tail, ties, GQA, staged sigmoid/reciprocal FTZ, scalar agreement"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "self-test", help="run bounded CPU checks without writing artifacts"
    )
    fixture = commands.add_parser(
        "fixture",
        help="generate projected BF16 inputs and expected stages in one new NPZ",
    )
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--tokens", type=int, default=2055)
    fixture.add_argument("--queries", type=int, default=4)
    fixture.add_argument("--seed", type=int, default=1926)
    fixture.add_argument("--tie-scores", action="store_true")
    evaluate = commands.add_parser(
        "evaluate", help="evaluate a projected-input NPZ into a new result NPZ"
    )
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    for command in (fixture, evaluate):
        command.add_argument(
            "--norm-convention", choices=("OnePlusWeight", "DirectGamma")
        )
        command.add_argument(
            "--rope-mode", choices=("fused_f32", "staged_bf16", "final_bf16")
        )
        command.add_argument("--probability-mode", choices=("bf16", "float32"))
        command.add_argument("--reduction", choices=("numpy", "scalar"))
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
        return
    defaults = {
        "norm_convention": "OnePlusWeight",
        "rope_mode": "fused_f32",
        "probability_mode": "bf16",
        "reduction": "numpy",
    }
    stored_options: dict[str, Any] = {}
    if args.command == "fixture":
        if not 1 <= args.queries <= 16 or not args.queries <= args.tokens <= 8192:
            parser.error(
                "bounded fixture requires 1..16 query rows and queries..8192 cached tokens"
            )
        convention = args.norm_convention or defaults["norm_convention"]
        inputs = fixture_inputs(
            args.tokens, args.queries, args.seed, convention, args.tie_scores
        )
    else:
        with np.load(args.input, allow_pickle=False) as source:
            if "metadata_json" in source:
                stored_options = json.loads(str(source["metadata_json"]))
            inputs = {
                name: source[name]
                for name in source.files
                if not name.startswith("expected_") and name != "metadata_json"
            }
        if (
            not 1 <= len(inputs["q_projection_bf16"]) <= 16
            or len(inputs["k_projection_bf16"]) > 8192
        ):
            parser.error(
                "bounded evaluation requires 1..16 query rows and at most 8192 cached tokens"
            )
    options = {
        name: getattr(args, name) or stored_options.get(name, fallback)
        for name, fallback in defaults.items()
    }
    trace = evaluate_inputs(inputs, **options)
    write_npz(args.output, inputs, trace, options)
    summary = {
        "output": str(args.output.resolve()),
        "backend": "numpy_cpu",
        "query_rows": len(trace["query_indices"]),
        "cached_tokens": len(inputs["k_projection_bf16"]),
        "tie_policy": TIE_POLICY,
        "selected_block_counts": trace["selected_block_counts"].tolist(),
        "selection_margin": trace["selection_margin"].tolist(),
        **options,
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
