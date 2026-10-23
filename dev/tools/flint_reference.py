#!/usr/bin/env python3
"""fp32 reference forward of Flint, read from a GGUF.

This is the executable specification Splash's Flint target is graded
against. It reads a GGUF of the ``flint`` architecture with gguf-py, which
also dequantizes its tensors, and runs the forward pass of the vendor's SGLang
implementation (flint_sglang 0.1.4) entirely in numpy fp32, one layer and one
routed expert at a time, so that it never holds more than one layer's
weights:

- the token table's rows, unscaled, into pre-norm layers whose RMSNorm is
  ``x * rsqrt(mean(x^2) + eps) * w``, with no ``1 + w`` offset;
- attention of 32 query heads over 4 KV heads of 128 (query head h reads KV
  head h // 8), each head's q and k RMS-normed by one weight the heads share,
  then rotated on the layers ``rope_pattern`` names (NeoX half-split pairs
  over the whole head, ``rope.freq_base``, absolute positions) and on no
  others (NoPE), scaled by 1/sqrt(128). A learned sink per query head
  (``attn_sinks``) is a logit in the softmax's denominator that adds no value.
  A sliding layer (``sliding_window_pattern``) attends the 513
  (``sliding_window``) keys ending at its row, the row's own included; a full
  layer, every key up to its row;
- dense SwiGLU on the leading_dense_block_count layers; on the others, the
  top 6 experts by router logit, weighted by their sigmoid scores normalized
  to sum to 1, plus the ungated shared expert;
- the final norm and the untied output head, in fp32.

The geometry comes from the GGUF's ``flint.*`` keys, so a small synthetic
Flint GGUF runs as the real one does.

    python dev/tools/flint_reference.py MODEL.gguf TOKENS OUT_DIR

TOKENS is a .npy array or a text file of whitespace-separated token ids, as
the model's tokenizer and chat template give them. OUT_DIR receives:

- tokens.npy (uint32), and logits.npy, fp32 [tokens, vocabulary]: the logits
  after each prefix of the tokens, the last after all of them;
- residual_norms.npy, fp32 [layers, 2, tokens]: the L2 norm of each row of
  the residual stream after each layer's attention (0) and after its FFN (1);
- residuals.npy, fp32 [layers named, 2, tokens, hidden]: those rows of the
  layers --residual-layers names. A layer's rows after its FFN are what a
  DFlash2 draft's target_layer_ids capture;
- experts.npy (int32) and expert_weights.npy (fp32), [layers named, tokens,
  6]: each row's routed experts, by descending score, and their weights, at
  the layers --router-layers names (the first routed, the middle and the last
  layer by default);
- sequences.txt, records.txt and logits.f32: the logits after --records
  prefixes spread over the tokens, the longest included, as a logits dump
  that ``dev/benchmarks/logit_compare.py`` compares with another engine's
  (``backend-benchmark --logits-from`` scores the same prefixes);
- reference.json: the GGUF, its layout and what was dumped.

needs gguf-py (pip install gguf, or llama.cpp's gguf-py/); its outputs are
reference data, not fixtures.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import gguf
import numpy as np

ARCHITECTURE = "flint"
# expert_gating_func's sigmoid, llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID.
SIGMOID_GATING = 2
# The vendor's guard in the routing weights' normalization.
ROUTING_EPSILON = 1e-20
# Query rows per attention block, which bounds its scores to heads x
# ATTENTION_ROWS x keys values.
ATTENTION_ROWS = 256
# Output head rows dequantized at once.
HEAD_ROWS = 16384
# A SwiGLU's projections, as their GGUF names spell them.
GLU = ("gate", "up", "down")


@dataclass(frozen=True)
class FlintLayout:
    """The model's geometry, as the GGUF's flint.* keys give it."""

    layers: int
    hidden: int
    vocabulary: int
    heads: int
    kv_heads: int
    head_dim: int
    # The leading layers whose FFN is dense, and its width.
    dense_layers: int
    dense_intermediate: int
    experts: int
    experts_per_token: int
    expert_intermediate: int
    shared_intermediate: int
    # The keys a sliding layer's row attends, its own included.
    window: int
    sliding: tuple[bool, ...]
    rotary: tuple[bool, ...]
    rope_theta: float
    norm_eps: float


class FlintModel:
    """A Flint GGUF's layout and its tensors, each dequantized to fp32 as it
    is read."""

    def __init__(self, path: Path):
        self.path = path
        self.reader = gguf.GGUFReader(path)
        self.tensors = {tensor.name: tensor for tensor in self.reader.tensors}
        self.layout = self.read_layout()

    def value(self, key, default=None):
        field = self.reader.get_field(key)
        if field is None:
            if default is None:
                raise ValueError(f"{self.path} lacks {key}")
            return default
        return field.contents()

    def read_layout(self) -> FlintLayout:
        if (architecture := self.value("general.architecture")) != ARCHITECTURE:
            raise ValueError(
                f"{self.path} is a {architecture} GGUF, not {ARCHITECTURE}"
            )

        def key(name, default=None):
            return self.value(f"{ARCHITECTURE}.{name}", default)

        layers, head_dim = key("block_count"), key("attention.key_length")
        # What the forward pass implements, which every Flint release has.
        if key("attention.value_length") != head_dim:
            raise ValueError("Flint's keys and values have one head dimension")
        if key("rope.dimension_count", head_dim) != head_dim:
            raise ValueError("Flint rotates the whole head")
        if key("expert_gating_func") != SIGMOID_GATING:
            raise ValueError("Flint routes by sigmoid scores")
        if not key("expert_weights_norm") or key("expert_weights_scale", 1.0) != 1.0:
            raise ValueError("Flint's routing weights are normalized, unscaled")
        if any(name.endswith(".ffn_exp_probs_b.bias") for name in self.tensors):
            raise ValueError("Flint's router has no expert bias")
        sliding = tuple(key("attention.sliding_window_pattern"))
        rotary = tuple(key("attention.rope_pattern"))
        if len(sliding) != layers or len(rotary) != layers:
            raise ValueError("Flint's layer patterns name every layer")
        return FlintLayout(
            layers=layers,
            hidden=key("embedding_length"),
            vocabulary=int(self.tensors["output.weight"].shape[1]),
            heads=key("attention.head_count"),
            kv_heads=key("attention.head_count_kv"),
            head_dim=head_dim,
            dense_layers=key("leading_dense_block_count"),
            dense_intermediate=key("feed_forward_length"),
            experts=key("expert_count"),
            experts_per_token=key("expert_used_count"),
            expert_intermediate=key("expert_feed_forward_length"),
            shared_intermediate=key("expert_shared_feed_forward_length"),
            window=key("attention.sliding_window"),
            sliding=sliding,
            rotary=rotary,
            rope_theta=key("rope.freq_base"),
            norm_eps=key("attention.layer_norm_rms_epsilon"),
        )

    def tensor(self, name: str, rows=None) -> np.ndarray:
        """A tensor's fp32 values, row-major in the reverse of the GGUF's
        dimensions, or those of the given indices of its outermost one: the
        token table's rows, or a routed expert's matrix."""
        tensor = self.tensors.get(name)
        if tensor is None:
            raise ValueError(f"{self.path} lacks {name}")
        data = tensor.data if rows is None else tensor.data[rows]
        return np.asarray(gguf.dequantize(data, tensor.tensor_type), dtype=np.float32)

    def weight(self, layer: int, name: str, rows=None) -> np.ndarray:
        return self.tensor(f"blk.{layer}.{name}.weight", rows)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * weight


def sigmoid(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def swiglu(x: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray):
    gated = x @ gate.T
    return (gated * sigmoid(gated) * (x @ up.T)) @ down.T


def rotary_table(count: int, head_dim: int, theta: float):
    """cos and sin, [count, 1, head_dim / 2], of the angle NeoX rotation turns
    position p's dimension pair (i, i + head_dim / 2) by,
    p * theta^(-2i / head_dim), computed in float64."""
    frequencies = theta ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    angles = np.arange(count, dtype=np.float64)[:, None, None] * frequencies
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def rotate(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return np.concatenate([first * cos - second * sin, second * cos + first * sin], -1)


def attention(model: FlintModel, layer: int, x: np.ndarray, table) -> np.ndarray:
    """Layer's attention output for the normed rows x."""
    layout = model.layout
    count, group, dim = x.shape[0], layout.heads // layout.kv_heads, layout.head_dim
    queries = (x @ model.weight(layer, "attn_q").T).reshape(count, layout.heads, dim)
    keys = (x @ model.weight(layer, "attn_k").T).reshape(count, layout.kv_heads, dim)
    values = (x @ model.weight(layer, "attn_v").T).reshape(count, layout.kv_heads, dim)
    queries = rms_norm(queries, model.weight(layer, "attn_q_norm"), layout.norm_eps)
    keys = rms_norm(keys, model.weight(layer, "attn_k_norm"), layout.norm_eps)
    if layout.rotary[layer]:
        queries, keys = rotate(queries, *table), rotate(keys, *table)
    span = layout.window if layout.sliding[layer] else count
    # [KV head, query head of its group, 1, 1], as the scores below.
    sinks = model.weight(layer, "attn_sinks").reshape(layout.kv_heads, group, 1, 1)
    context = np.empty((count, layout.heads, dim), np.float32)
    for start in range(0, count, ATTENTION_ROWS):
        stop = min(start + ATTENTION_ROWS, count)
        first = max(0, start - span + 1)
        rows = np.arange(start, stop)[:, None]
        columns = np.arange(first, stop)[None, :]
        visible = (columns <= rows) & (columns > rows - span)
        block = queries[start:stop].reshape(stop - start, layout.kv_heads, group, dim)
        block_keys = keys[first:stop].transpose(1, 2, 0)[:, None]
        scores = block.transpose(1, 2, 0, 3) @ block_keys
        scores = np.where(visible, scores * dim**-0.5, -np.inf)
        # The sink takes part in the maximum: it may exceed every score.
        peak = np.maximum(scores.max(axis=-1, keepdims=True), sinks)
        weights = np.exp(scores - peak)
        total = weights.sum(axis=-1, keepdims=True) + np.exp(sinks - peak)
        mixed = (weights @ values[first:stop].transpose(1, 0, 2)[:, None]) / total
        context[start:stop] = mixed.transpose(2, 0, 1, 3).reshape(
            stop - start, layout.heads, dim
        )
    return context.reshape(count, -1) @ model.weight(layer, "attn_output").T


def moe(model: FlintModel, layer: int, x: np.ndarray):
    """Layer's MoE output for the normed rows x, each row's routed experts by
    descending score and their weights."""
    logits = x @ model.weight(layer, "ffn_gate_inp").T
    # The sigmoid is monotonic, so the top experts by score are the top ones
    # by logit, which keeps scores that round to 1 apart.
    chosen = np.argsort(-logits, axis=-1, kind="stable")
    chosen = chosen[:, : model.layout.experts_per_token]
    scores = sigmoid(np.take_along_axis(logits, chosen, axis=-1))
    weights = scores / (scores.sum(axis=-1, keepdims=True) + ROUTING_EPSILON)
    routed = np.zeros_like(x)
    for expert in np.unique(chosen):
        rows, slots = np.nonzero(chosen == expert)
        matrices = (model.weight(layer, f"ffn_{name}_exps", expert) for name in GLU)
        routed[rows] += weights[rows, slots, None] * swiglu(x[rows], *matrices)
    shared = swiglu(x, *(model.weight(layer, f"ffn_{name}_shexp") for name in GLU))
    return routed + shared, chosen.astype(np.int32), weights


def head(model: FlintModel, x: np.ndarray) -> np.ndarray:
    layout = model.layout
    normed = rms_norm(x, model.tensor("output_norm.weight"), layout.norm_eps)
    logits = np.empty((x.shape[0], layout.vocabulary), np.float32)
    for start in range(0, layout.vocabulary, HEAD_ROWS):
        rows = slice(start, min(start + HEAD_ROWS, layout.vocabulary))
        logits[:, rows] = normed @ model.tensor("output.weight", rows).T
    return logits


@dataclass
class Forward:
    """What one forward pass over a token sequence dumps (see the module's
    description of OUT_DIR)."""

    logits: np.ndarray
    residual_norms: np.ndarray
    residuals: np.ndarray
    experts: np.ndarray
    expert_weights: np.ndarray


def forward(
    model: FlintModel, tokens: np.ndarray, residual_layers=(), router_layers=()
) -> Forward:
    """The forward pass over tokens, keeping the residual rows of
    residual_layers and the routes of router_layers."""
    layout, count = model.layout, len(tokens)
    residual_layers, router_layers = list(residual_layers), list(router_layers)
    if not count or tokens.min() < 0 or tokens.max() >= layout.vocabulary:
        raise ValueError(f"tokens must be 1 or more ids below {layout.vocabulary}")
    if any(not 0 <= layer < layout.layers for layer in residual_layers):
        raise ValueError(f"residual layers must be below {layout.layers}")
    if any(not layout.dense_layers <= layer < layout.layers for layer in router_layers):
        raise ValueError(
            f"router layers must be routed, {layout.dense_layers} to {layout.layers - 1}"
        )
    routes = (len(router_layers), count, layout.experts_per_token)
    norms = np.empty((layout.layers, 2, count), np.float32)
    residuals = np.empty((len(residual_layers), 2, count, layout.hidden), np.float32)
    experts, expert_weights = np.empty(routes, np.int32), np.empty(routes, np.float32)
    table = rotary_table(count, layout.head_dim, layout.rope_theta)
    hidden = model.tensor("token_embd.weight", tokens)
    for layer in range(layout.layers):
        started = time.monotonic()
        normed = rms_norm(hidden, model.weight(layer, "attn_norm"), layout.norm_eps)
        hidden = hidden + attention(model, layer, normed, table)
        stages = [hidden]
        normed = rms_norm(hidden, model.weight(layer, "ffn_norm"), layout.norm_eps)
        if layer < layout.dense_layers:
            ffn = swiglu(normed, *(model.weight(layer, f"ffn_{name}") for name in GLU))
        else:
            ffn, chosen, weights = moe(model, layer, normed)
            if layer in router_layers:
                index = router_layers.index(layer)
                experts[index], expert_weights[index] = chosen, weights
        hidden = hidden + ffn
        stages.append(hidden)
        for stage, rows in enumerate(stages):
            norms[layer, stage] = np.linalg.norm(rows, axis=-1)
            if layer in residual_layers:
                residuals[residual_layers.index(layer), stage] = rows
        print(
            f"layer {layer}: {time.monotonic() - started:.1f} s",
            file=sys.stderr,
            flush=True,
        )
    return Forward(head(model, hidden), norms, residuals, experts, expert_weights)


def record_prefixes(count: int, records: int) -> list[int]:
    """Up to records prefix lengths spread evenly over 1 to count - 1, the
    longest included: the proper prefixes a logits dump scores."""
    if count < 2 or records < 1:
        return []
    spread = np.linspace(1, count - 1, min(records, count - 1))
    return sorted({int(length) for length in spread.round()})


def read_tokens(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.int64).reshape(-1)
    return np.array(path.read_text().split(), dtype=np.int64)


def write_reference(
    path: Path,
    tokens: np.ndarray,
    out: Path,
    residual_layers=(),
    router_layers=None,
    records: int = 16,
) -> None:
    started = time.monotonic()
    model = FlintModel(path)
    layout = model.layout
    if router_layers is None:
        router_layers = sorted(
            {layout.dense_layers, layout.layers // 2, layout.layers - 1}
        )
    result = forward(model, tokens, residual_layers, router_layers)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "tokens.npy", tokens.astype(np.uint32))
    np.save(out / "logits.npy", result.logits)
    np.save(out / "residual_norms.npy", result.residual_norms)
    np.save(out / "residuals.npy", result.residuals)
    np.save(out / "experts.npy", result.experts)
    np.save(out / "expert_weights.npy", result.expert_weights)
    prefixes = record_prefixes(len(tokens), records)
    if prefixes:
        (out / "sequences.txt").write_text(" ".join(map(str, tokens)) + "\n")
        (out / "records.txt").write_text("".join(f"0 {n}\n" for n in prefixes))
        rows = result.logits[[length - 1 for length in prefixes]]
        rows.astype("<f4").tofile(out / "logits.f32")
    description = {
        "gguf": str(path.absolute()),
        "gguf_bytes": path.stat().st_size,
        "gguf_py": metadata.version("gguf"),
        "layout": asdict(layout),
        "tokens": len(tokens),
        "residual_layers": list(residual_layers),
        "router_layers": list(router_layers),
        "records": prefixes,
        "seconds": round(time.monotonic() - started, 1),
    }
    (out / "reference.json").write_text(json.dumps(description, indent=2) + "\n")
    print(
        f"wrote the reference of {len(tokens)} tokens to {out} in "
        f"{description['seconds']} s; the next token is "
        f"{int(result.logits[-1].argmax())}"
    )


def layer_list(text: str) -> list[int]:
    return [int(layer) for layer in text.split(",") if layer]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("tokens", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--residual-layers",
        type=layer_list,
        default=[],
        help="comma-separated layers whose residual rows to dump",
    )
    parser.add_argument(
        "--router-layers",
        type=layer_list,
        help="comma-separated routed layers whose routes to dump",
    )
    parser.add_argument(
        "--records", type=int, default=16, help="prefixes in the logits dump"
    )
    args = parser.parse_args()
    write_reference(
        args.model,
        read_tokens(args.tokens),
        args.output,
        args.residual_layers,
        args.router_layers,
        args.records,
    )
