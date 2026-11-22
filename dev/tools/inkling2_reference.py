#!/usr/bin/env python3
"""fp32 reference forward of Inkling-2-Mini's text model, read from a GGUF.

This is the executable specification Splash's Inkling-2-Mini target is graded
against. It reads a GGUF of the ``inkling2_loop`` architecture with gguf-py,
which also dequantizes its tensors (NVFP4 experts included), and runs the
forward pass of transformers' Inkling2LoopTextModel entirely in numpy fp32, one
layer application and one routed expert at a time, so that it never holds more
than one layer's weights. llama.cpp's converter and graph settle how the GGUF
lays out what the checkpoint holds.

- The token's row of the padded table, RMS-normed by ``token_embd_norm``.
  Every weighted RMSNorm is ``x * rsqrt(mean(x^2) + eps) * w``.
- 48 applications a token: layers before ``loop.start`` once, the loop's layers
  ``loop.count`` times with shared weights (passes), the layers from
  ``loop.end`` once. Each application has its own keys, values and short
  convolution states. A pass after the first restarts from the hidden rows and
  block states before the loop and takes messages from the previous pass.
- Block states: the input rows of a layer whose index is a multiple of
  ``attn_res.block_size`` join a per-token stack. A block mix normalizes each
  of [stack..., current] without a weight (eps 1e-6), scores it against a
  learned query scaled by 1/sqrt(hidden), and returns the softmax-weighted sum
  of the normed states, in fp32. A layer mixes once before attention (three
  queries on a full layer, for its q, k and v inputs; none on layer 0), once
  before its FFN, and the model once before ``output_norm``.
- Attention, 24 query heads over 8 KV heads of 128, no RoPE: q is projected,
  scaled per head and channel (``attn_q_clip``) and L2-clipped per head to
  ``qk_clip``; k and v are projected and pass a short convolution, then k is
  L2-clipped and v RMS-normed without a weight. Scores are q.k / sqrt(128)
  plus a relative bias: the row's relative state s (``attn_rel``, scaled by
  ``attn_rel_clip``, clipped to ``relative_clip``) dotted with the
  ``attn_rel_bias`` row of the key's distance d, zero past the table. Full
  layers multiply q and s by 1 + alpha * ln(max((position + 1) / floor, 1)).
  A sink logit per row and head (``attn_sink_proj``) joins the softmax and adds
  no value. A sliding layer's row at position p attends keys p - 1023 .. p;
  a full layer's, every key up to p. The context, gated by
  sigmoid(``attn_gate``), goes through ``attn_output`` and a short
  convolution.
- A layer starting a block passes its attention output on alone (its input is
  on the stack); any other adds it to its input.
- MoE: sigmoid router scores in fp32, the top 8 by score plus
  ``exp_probs_b``, weighted by score / (sum + 1e-20) * 8. An expert applies
  4 tanh(gate / 4) * sigmoid(gate) * 25 tanh(up / 25) to the halves of
  ``ffn_gate_up_exps``, then ``ffn_down_exps``; an expert tensor's ``.scale``
  multiplies each expert's matrix. A short convolution follows the sum.
- Short convolution, 4 taps per channel over a row and the 3 before it: the
  taps' sum plus the row, over sqrt(5). Full layers' k and v add to the static
  taps ones projected from the convolution's input (``attn_{k,v}_conv_proj``,
  one set per KV head), over sqrt(9).
- Messages: a loop layer of a later pass mixes, with its four
  ``loop_msg_query`` queries, the previous pass's block states after the
  loop's first and its output, into messages for its q, k, v and FFN inputs.
  A message gate g = ``down``(concat(norm(x), norm(message))) gives
  (1 - tanh(``forget``(g))) x + tanh(``gain``(g)) message.
- The final block mix, ``output_norm``, the output head over the real
  vocabulary (``vocab_size_unpadded``; the padded rows are never logits), and
  30 tanh(logits / 30).

The model is bf16. The source rounds to bf16 every bf16 projection's output,
the hidden rows between applications, every weighted norm's output, q, k, v
and the relative states after their clips and norms, each convolution's and
the MoE's output, the expert activations, and in its eager attention the
scores, the biased scores and the probabilities; it computes in fp32 the block
mixes, the message gates (but the input and output of their down projection),
the convolutions' sums, the residual sums, the router, the softmax with its
sink, the experts' weighted sum, and the softcap of the logits. This reference
is fp32 throughout, or with --rows bf16 rounds to bf16 (to nearest, ties to
even) exactly at those points (application() lists each with its source line),
so that the engine's rounding can be told from the source's own. Weights are
used as the GGUF holds them, in either mode.

The geometry comes from the GGUF's ``inkling2_loop.*`` keys, so a small
synthetic GGUF runs as the real one does.

    python dev/tools/inkling2_reference.py MODEL.gguf TOKENS OUT_DIR

TOKENS is a .npy array or a text file of whitespace-separated token ids, as
the model's tokenizer and chat template give them. Applications are numbered
in execution order (0-47 for Inkling-2-Mini: layers 0-5, the loop's three
passes over layers 6-17, layers 18-23). OUT_DIR receives:

- tokens.npy (uint32), and logits.npy, fp32 [tokens, real vocabulary]: the
  logits after each prefix of the tokens, the last after all of them; with
  --record-logits-only, after the records' prefixes, all the tokens and the
  rows --logit-rows adds alone, which logit_rows.npy (int64) names by their
  last row (a 9,000-token prompt's logits are 7 GB);
- residual_norms.npy, fp32 [applications, 2, tokens]: the L2 norm of each row
  of the residual stream after each application's attention (0) and after its
  FFN (1);
- residuals.npy, fp32 [applications named, 2, tokens, hidden]: those rows of
  the applications --residual-applications names;
- experts.npy (int32) and expert_weights.npy (fp32), [applications named,
  tokens, 8]: each row's routed experts, by descending biased score, and their
  weights, at the applications --router-applications names (the first, the
  middle and the last by default);
- trace.npz, with --trace-applications and --trace-rows: per chosen
  application a and row, each intermediate the engine also produces, as
  ``a{a:02d}.{name}`` (rows in --trace-rows order, ``rows`` holds them):
  mix_attention [1 or 3, hidden] (layer 0 has none), messages [4, hidden],
  gated_q/k/v (later passes), input_q/k/v (the normed projection inputs),
  q [heads, dim], k_projected and v_projected [kv heads * dim] before their
  convolution, k_convolved and v_convolved after it, k and v [kv heads, dim],
  relative [heads, relative length] (s), relative_bias [heads, extent] (the
  bias by distance), sink [heads], attention (the context before the gate),
  attention_gated, attention_projected, attention_output (after the
  convolution), residual, mix_ffn, gated_ffn (later passes), ffn_input,
  router_scores [experts], experts and expert_weights [8], moe (before the
  convolution), ffn_output and output;
- sequences.txt, records.txt and logits.f32: the logits after --records
  prefixes spread over the tokens, the longest included, as a logits dump
  that ``dev/benchmarks/logit_compare.py`` compares with another engine's
  (``backend-benchmark --logits-from`` scores the same prefixes);
- reference.json: the GGUF, its layout and what was dumped.

needs gguf-py with NVFP4 (the development requirements pin a release that
reads it); its outputs are reference data, not fixtures.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path

import gguf
import numpy as np

ARCHITECTURE = "inkling2_loop"
# Inkling2LoopL2Clip's guard in the norm it divides by.
CLIP_EPSILON = 1e-6
# The block mixes' unweighted norm keeps Inkling2LoopUnweightedRMSNorm's
# default eps, not the config's (llama.cpp's mix uses 1e-6 as well).
MIX_EPSILON = 1e-6
# The router's guard in the routing weights' normalization.
ROUTING_EPSILON = 1e-20
# Query rows per attention block, which bounds its scores to heads x
# ATTENTION_ROWS x keys values.
ATTENTION_ROWS = 128
# Output head rows dequantized at once.
HEAD_ROWS = 16384
# The order of a loop layer's messages and message gates, as the converter
# stacks loop_msg_query and as Inkling2LoopMessageAttentionResidual's
# messages unbind.
MESSAGES = ("q", "k", "v", "ffn")


@dataclass(frozen=True)
class InklingLayout:
    """The model's geometry, as the GGUF's inkling2_loop.* keys give it."""

    layers: int
    hidden: int
    # The real vocabulary; the token and output tables are padded past it.
    vocabulary: int
    heads: int
    kv_heads: int
    head_dim: int
    experts: int
    experts_per_token: int
    expert_intermediate: int
    # The keys a sliding layer's row attends, its own included.
    window: int
    sliding: tuple[bool, ...]
    relative_length: int
    qk_clip: float
    relative_clip: float
    log_scale_floor: int
    log_scale_alpha: float
    # Taps of every short convolution: the row and the taps - 1 before it.
    conv_taps: int
    block_size: int
    loop_start: int
    loop_end: int
    passes: int
    gate_rank: int
    gate_softcap: float
    up_softcap: float
    route_scale: float
    logit_softcap: float
    norm_eps: float

    def schedule(self) -> list[tuple[int, int]]:
        """(weight layer, pass) of each application, in execution order; the
        layers outside the loop run in pass 0."""
        loop = [
            (layer, index)
            for index in range(self.passes)
            for layer in range(self.loop_start, self.loop_end)
        ]
        before = [(layer, 0) for layer in range(self.loop_start)]
        after = [(layer, 0) for layer in range(self.loop_end, self.layers)]
        return before + loop + after


class InklingModel:
    """An inkling2_loop GGUF's layout and its tensors, each dequantized to
    fp32 as it is read."""

    def __init__(self, path: Path):
        self.path = path
        self.reader = gguf.GGUFReader(path)
        self.tensors = {tensor.name: tensor for tensor in self.reader.tensors}
        self.layout = self.read_layout()

    def value(self, key, default=None):
        entry = self.reader.get_field(key)
        if entry is None:
            if default is None:
                raise ValueError(f"{self.path} lacks {key}")
            return default
        return entry.contents()

    def read_layout(self) -> InklingLayout:
        if (architecture := self.value("general.architecture")) != ARCHITECTURE:
            raise ValueError(
                f"{self.path} is a {architecture} GGUF, not {ARCHITECTURE}"
            )

        def key(name, default=None):
            return self.value(f"{ARCHITECTURE}.{name}", default)

        layers, head_dim = key("block_count"), key("attention.key_length")
        heads, kv_heads = key("attention.head_count"), key("attention.head_count_kv")
        # What the forward pass implements, which every Inkling-2 loop has.
        lengths = ("value_length", "key_length_swa", "value_length_swa")
        if any(key(f"attention.{name}", head_dim) != head_dim for name in lengths):
            raise ValueError("Inkling-2's layers have one head dimension")
        if len(set(heads)) != 1 or len(set(kv_heads)) != 1:
            raise ValueError("Inkling-2's layers have one head count")
        if key("rope.dimension_count", 0):
            raise ValueError("Inkling-2 has no RoPE, its relative bias holds positions")
        if key("loop.varying", False):
            raise ValueError("Inkling-2-Mini's loop runs a fixed number of passes")
        sliding = tuple(key("attention.sliding_window_pattern"))
        if len(sliding) != layers:
            raise ValueError("the sliding window pattern names every layer")
        block, start, end = (
            key("attn_res.block_size"),
            key("loop.start"),
            key("loop.end"),
        )
        if not 0 < start < end <= layers or start % block or end % block:
            raise ValueError("the loop spans whole blocks after the first")
        return InklingLayout(
            layers=layers,
            hidden=key("embedding_length"),
            vocabulary=key("vocab_size_unpadded"),
            heads=heads[0],
            kv_heads=kv_heads[0],
            head_dim=head_dim,
            experts=key("expert_count"),
            experts_per_token=key("expert_used_count"),
            expert_intermediate=key("expert_feed_forward_length"),
            window=key("attention.sliding_window"),
            sliding=sliding,
            relative_length=key("attention.relative_length"),
            qk_clip=key("attention.qk_clip"),
            relative_clip=key("attention.relative_clip"),
            log_scale_floor=key("attention.log_scale_floor"),
            log_scale_alpha=key("attention.log_scale_alpha"),
            conv_taps=key("shortconv.l_cache"),
            block_size=block,
            loop_start=start,
            loop_end=end,
            passes=key("loop.count"),
            gate_rank=key("loop.gate_rank"),
            gate_softcap=key("activation.situ_beta"),
            up_softcap=key("activation.situ_linear_beta"),
            route_scale=key("expert_weights_scale"),
            logit_softcap=key("final_logit_softcapping"),
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

    def weight(self, layer: int, name: str) -> np.ndarray:
        return self.tensor(f"blk.{layer}.{name}.weight")

    def has(self, layer: int, name: str) -> bool:
        return f"blk.{layer}.{name}.weight" in self.tensors

    def expert(self, layer: int, name: str, expert: int) -> np.ndarray:
        """A routed expert's matrix [out, in], times the expert's entry of the
        tensor's .scale where it has one (the UD files' NVFP4 experts carry
        a per-expert F32 scale; llama.cpp multiplies the product by it)."""
        matrix = self.tensor(f"blk.{layer}.{name}.weight", expert)
        scale = f"blk.{layer}.{name}.scale"
        if scale in self.tensors:
            matrix = matrix * self.tensor(scale)[expert]
        return matrix


def exact(x: np.ndarray) -> np.ndarray:
    """The fp32 rows' rounding: none."""
    return x


def bf16(x: np.ndarray) -> np.ndarray:
    """x rounded to the nearest bfloat16, ties to even, as torch's
    .to(torch.bfloat16) rounds, kept in fp32."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bits = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (bits & np.uint32(0xFFFF0000)).view(np.float32)


# --rows: fp32 throughout, or the bf16 model's rows, rounded where the source
# rounds (see application()).
ROUNDINGS = {"fp32": exact, "bf16": bf16}


def unweighted_norm(x: np.ndarray, eps: float) -> np.ndarray:
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps))


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return unweighted_norm(x, eps) * weight


def sigmoid(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def softcap(x: np.ndarray, cap: float) -> np.ndarray:
    return np.tanh(x / cap) * cap


def l2_clip(x: np.ndarray, threshold: float) -> np.ndarray:
    """Each vector of the last axis scaled down to an L2 norm of threshold
    when longer; shorter ones unchanged."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x * np.minimum(1.0, threshold / (norm + CLIP_EPSILON))


def log_scale(positions: np.ndarray, floor: int, alpha: float) -> np.ndarray:
    """A full layer's factor on q and s at each position,
    1 + alpha * ln(max((position + 1) / floor, 1)), [count, 1, 1]."""
    ratio = (positions.astype(np.float32) + 1) / np.float32(floor)
    return (1 + np.float32(alpha) * np.log(np.maximum(ratio, 1)))[:, None, None]


def block_mix(states: list, current: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Each row's mix of its block states and current row, [count, queries,
    hidden], for queries [queries, hidden]: the softmax over the states of
    query . norm(state) / sqrt(hidden), weighting the normed states."""
    normed = unweighted_norm(np.stack([*states, current], axis=1), MIX_EPSILON)
    scores = normed @ (queries.T / np.sqrt(np.float32(current.shape[-1])))
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights.transpose(0, 2, 1) @ normed


def message_gate(x, message, down, forget, gain, eps, rounding=exact):
    """(1 - tanh(forget(g))) x + tanh(gain(g)) message, g = down(concat(
    norm(x), norm(message))). down is a bf16 module in the source: its input
    and output are rounded (modeling 92); forget and gain are fp32."""
    gate = np.concatenate([unweighted_norm(x, eps), unweighted_norm(message, eps)], -1)
    gate = rounding(rounding(gate) @ down.T)
    return (1 - np.tanh(gate @ forget.T)) * x + np.tanh(gate @ gain.T) * message


def short_convolution(x, taps, history, dynamic=None):
    """A causal convolution of each channel of x [count, channels] over the
    row and the taps - 1 before it (history [taps - 1, channels], oldest
    first, then x's own rows), plus the row, and the history the next rows
    read.

    taps [channels, taps] is in the GGUF's order, its last column weighting
    the row itself: ggml_ssm_conv's order, which a static convolution's
    conv1d weight already has and into which the converter flips a dynamic
    one's static_weight (indexed by backward distance). dynamic [count,
    groups, taps], by backward distance, adds per-row taps shared by each
    group of channels (a KV head). The sum over sqrt(taps + 1), or
    sqrt(2 taps + 1) with dynamic taps, as Inkling2LoopShortConvolution and
    Inkling2LoopDynamicShortConvolution scale it."""
    count, channels = x.shape
    width = taps.shape[1]
    window = np.concatenate([history, x])
    out = np.zeros_like(x)
    for lag in range(width):
        tap = taps[:, width - 1 - lag]
        if dynamic is not None:
            group = channels // dynamic.shape[1]
            tap = tap + np.repeat(dynamic[:, :, lag], group, axis=1)
        out += tap * window[width - 1 - lag : width - 1 - lag + count]
    terms = 2 * width + 1 if dynamic is not None else width + 1
    return (out + x) / np.sqrt(np.float32(terms)), window[count:]


def attend(
    queries, keys, values, relative, table, sinks, rows, columns, window, rounding=exact
):
    """The context [count, heads, dim] of queries [count, heads, dim] at
    positions rows over keys and values [keys, kv heads, dim] at positions
    columns (ascending): query head h reads KV head h // (heads / kv heads).
    Scores are q.k / sqrt(dim) plus the relative bias relative [count,
    heads, relative length] . table[distance] ([extent, relative length]),
    zero from the extent on. A sink logit per row and head, sinks [count,
    heads], joins the softmax and adds no value. A sliding layer (window)
    attends the window keys ending at the row's own; a full one (None),
    every key up to it. rounding is eager_attention_forward's in a bf16
    model: q.k, its scaling, the bias table and the biased scores, the
    probabilities and the context are bf16 (modeling 126-139, 267); the
    softmax with the sink is fp32."""
    count, heads, dim = queries.shape
    kv_heads = keys.shape[1]
    group, extent = heads // kv_heads, table.shape[0]
    context = np.empty_like(queries)
    keys_t, values_t = keys.transpose(1, 2, 0), values.transpose(1, 0, 2)
    for start in range(0, count, ATTENTION_ROWS):
        stop = min(start + ATTENTION_ROWS, count)
        block = stop - start
        first = (
            0 if window is None else np.searchsorted(columns, rows[start] - window + 1)
        )
        last = np.searchsorted(columns, rows[stop - 1], side="right")
        distance = rows[start:stop, None] - columns[None, first:last]
        visible = distance >= 0
        if window is not None:
            visible &= distance < window
        # Bias by distance, with a zero past the table for every distance from
        # the extent on.
        biases = rounding(relative[start:stop] @ table.T)
        biases = np.concatenate([biases, np.zeros((block, heads, 1), np.float32)], -1)
        index = np.broadcast_to(
            np.clip(distance, 0, extent)[:, None], (block, heads, last - first)
        )
        bias = np.take_along_axis(biases, index, axis=-1)
        grouped = queries[start:stop].reshape(block, kv_heads, group, dim)
        grouped = grouped.transpose(1, 0, 2, 3).reshape(kv_heads, block * group, dim)
        scores = rounding(grouped @ keys_t[:, :, first:last])
        scores = scores.reshape(kv_heads, block, group, -1).transpose(1, 0, 2, 3)
        scores = rounding(scores.reshape(block, heads, -1) * np.float32(dim**-0.5))
        scores = rounding(scores + bias)
        scores = np.where(visible[:, None], scores, -np.inf)
        sink = sinks[start:stop, :, None]
        # The sink takes part in the maximum: it may exceed every score.
        peak = np.maximum(scores.max(axis=-1, keepdims=True), sink)
        weights = np.exp(scores - peak)
        total = weights.sum(axis=-1, keepdims=True) + np.exp(sink - peak)
        weights = rounding(weights / total)
        weights = weights.reshape(block, kv_heads, group, -1).transpose(1, 0, 2, 3)
        mixed = weights.reshape(kv_heads, block * group, -1) @ values_t[:, first:last]
        mixed = mixed.reshape(kv_heads, block, group, dim).transpose(1, 0, 2, 3)
        context[start:stop] = rounding(mixed.reshape(block, heads, dim))
    return context


def route(x, router, bias, used, scale):
    """Each row's sigmoid scores [count, experts], its experts by descending
    score + bias, and their weights score / (sum + 1e-20) * scale: the bias
    only chooses."""
    scores = sigmoid(x @ router.T)
    chosen = np.argsort(-(scores + bias), axis=-1, kind="stable")[:, :used]
    weights = np.take_along_axis(scores, chosen, axis=-1)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + ROUTING_EPSILON) * scale
    return scores, chosen, weights.astype(np.float32)


def expert(x, gate_up, down, gate_softcap, up_softcap, rounding=exact):
    """One expert's output for rows x: gate and up are the halves of
    gate_up's output, activated as softcap(gate) * sigmoid(gate) *
    softcap(up). In a bf16 model both projections' outputs and the
    activation are bf16 (modeling 771, 802-803)."""
    gate, up = np.split(rounding(x @ gate_up.T), 2, axis=-1)
    act = softcap(gate, gate_softcap) * sigmoid(gate) * softcap(up, up_softcap)
    return rounding(rounding(act) @ down.T)


def moe(model: InklingModel, layer: int, x: np.ndarray, rounding=exact):
    """Layer's MoE output for the normed rows x, the router's scores, each
    row's experts and their weights. The router and the weighted sum are fp32;
    the sum is rounded as the experts' output (modeling 790-807, 1163-1172)."""
    layout = model.layout
    scores, chosen, weights = route(
        x,
        model.weight(layer, "ffn_gate_inp"),
        model.tensor(f"blk.{layer}.exp_probs_b.bias"),
        layout.experts_per_token,
        layout.route_scale,
    )
    out = np.zeros_like(x)
    for index in np.unique(chosen):
        rows, slots = np.nonzero(chosen == index)
        matrices = (
            model.expert(layer, name, index)
            for name in ("ffn_gate_up_exps", "ffn_down_exps")
        )
        caps = (layout.gate_softcap, layout.up_softcap)
        out[rows] += weights[rows, slots, None] * expert(
            x[rows], *matrices, *caps, rounding
        )
    return rounding(out), scores, chosen, weights


class Cache:
    """What a later chunk of a sequence reads from its earlier ones, per
    application: the positions, keys and values its rows may attend (a
    sliding application keeps the last window - 1) and the last taps - 1
    inputs of each of its four short convolutions."""

    def __init__(self, layout: InklingLayout):
        applications = len(layout.schedule())
        self.layout = layout
        self.length = 0
        self.positions = [np.zeros(0, np.int64) for _ in range(applications)]
        self.keys = [None] * applications
        self.values = [None] * applications
        self.history = [{} for _ in range(applications)]

    def convolve(self, application, name, x, taps, dynamic=None):
        width = self.layout.conv_taps
        history = self.history[application].get(name)
        if history is None:
            history = np.zeros((width - 1, x.shape[1]), np.float32)
        y, self.history[application][name] = short_convolution(
            x, taps, history, dynamic
        )
        return y

    def extend(self, application, positions, keys, values, sliding):
        """The application's keys and values with these rows', keeping what a
        later row can still attend."""
        if self.keys[application] is not None:
            positions = np.concatenate([self.positions[application], positions])
            keys = np.concatenate([self.keys[application], keys])
            values = np.concatenate([self.values[application], values])
        kept = slice(-(self.layout.window - 1), None) if sliding else slice(None)
        self.positions[application] = positions[kept]
        self.keys[application], self.values[application] = keys[kept], values[kept]
        return positions, keys, values


class Trace:
    """The intermediates of chosen applications at chosen rows of a run."""

    def __init__(self, applications=(), rows=()):
        self.applications = set(applications)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.values = {}

    def recorder(self, application: int):
        """A function noting an intermediate of the application's rows, or of
        what derive makes of the traced rows, which it computes only then."""
        if application not in self.applications:
            return lambda name, value, derive=None: None

        def note(name, value, derive=None):
            rows = value[self.rows]
            self.values[f"a{application:02d}.{name}"] = np.array(
                rows if derive is None else derive(rows)
            )

        return note


def application(
    model, cache, index, layer, hidden, stack, messages, positions, note, rounding
):
    """One application of layer to the rows hidden with the block states
    stack (and, in a later pass, messages [count, 4, hidden]): its output
    rows, the stack after it, its residual rows after attention, and its
    router's scores, experts and weights.

    rounding is applied where a bf16 model rounds, each at a .to(dtype) of
    modeling_inkling2_loop.py or the output of one of its bf16 Linear
    modules: the normed attention and FFN inputs (892, 908, 970-972,
    1042-1044), the q, k, v, relative, sink, gate, output and dynamic taps
    projections (473, 263, 523, 619, 541, 543, 430), q and relative after
    their clip (202, 235) and log scale (220), k after its clip (187), v after
    its norm (76), every convolution's output (393, 455), the gated context
    (542), the MoE's output (807) and the layer's output (910); the block
    mixes, the message gates (but their down projection), the residual sum
    (905) and the router stay fp32."""
    layout = model.layout
    eps, count = layout.norm_eps, hidden.shape[0]
    sliding = layout.sliding[layer]
    heads, kv_heads, dim = layout.heads, layout.kv_heads, layout.head_dim

    def weight(name):
        return model.weight(layer, name)

    # The attention's inputs: q, k and v share one mix on a sliding layer and
    # have one each on a full one; layer 0 has no stack to mix.
    inputs = [hidden] * 3
    if model.has(layer, "attn_res_query"):
        queries = weight("attn_res_query").reshape(-1, layout.hidden)
        mixed = block_mix(stack, hidden, queries)
        note("mix_attention", mixed)
        inputs = [mixed[:, 0 if sliding else j] for j in range(3)]
    if messages is not None:
        note("messages", messages)
        for j, kind in enumerate(MESSAGES[:3]):
            gates = (
                weight(f"loop_{kind}_ctrl_{part}")
                for part in ("down", "forget", "gain")
            )
            inputs[j] = message_gate(inputs[j], messages[:, j], *gates, eps, rounding)
            note(f"gated_{kind}", inputs[j])
    if sliding:
        norm = weight("attn_norm")
        inputs = [rounding(rms_norm(x, norm, eps)) for x in inputs]
    else:
        inputs = [
            rounding(rms_norm(x, weight(f"attn_{k}_inp_norm"), eps))
            for x, k in zip(inputs, "qkv")
        ]
    for x, kind in zip(inputs, "qkv"):
        note(f"input_{kind}", x)
    q_in, k_in, v_in = inputs

    scale = (
        1
        if sliding
        else log_scale(positions, layout.log_scale_floor, layout.log_scale_alpha)
    )
    q = rounding(q_in @ weight("attn_q").T).reshape(count, heads, dim)
    q = rounding(l2_clip(q * weight("attn_q_clip"), layout.qk_clip))
    q = rounding(q * scale)
    relative = rounding(q_in @ weight("attn_rel").T).reshape(count, heads, -1)
    relative = rounding(
        l2_clip(relative * weight("attn_rel_clip"), layout.relative_clip)
    )
    relative = rounding(relative * scale)
    sinks = rounding(q_in @ weight("attn_sink_proj").T)
    convolved = []
    for x, kind in ((k_in, "k"), (v_in, "v")):
        projected = rounding(x @ weight(f"attn_{kind}").T)
        dynamic = None
        if not sliding:
            dynamic = rounding(x @ weight(f"attn_{kind}_conv_proj").T).reshape(
                count, kv_heads, -1
            )
        out = rounding(
            cache.convolve(index, kind, projected, weight(f"attn_{kind}_conv"), dynamic)
        )
        note(f"{kind}_projected", projected)
        note(f"{kind}_convolved", out)
        convolved.append(out.reshape(count, kv_heads, dim))
    k = rounding(l2_clip(convolved[0], layout.qk_clip))
    v = rounding(unweighted_norm(convolved[1], eps))
    table = weight("attn_rel_bias")
    for name, value in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("relative", relative),
        ("sink", sinks),
    ):
        note(name, value)
    note("relative_bias", relative, lambda rows: rounding(rows @ table.T))
    columns, keys, values = cache.extend(index, positions, k, v, sliding)
    window = layout.window if sliding else None
    context = attend(
        q, keys, values, relative, table, sinks, positions, columns, window, rounding
    )
    context = context.reshape(count, heads * dim)
    gated = rounding(context * sigmoid(rounding(q_in @ weight("attn_gate").T)))
    projected = rounding(gated @ weight("attn_output").T)
    out = rounding(
        cache.convolve(index, "attention", projected, weight("attn_post_conv"))
    )
    for name, value in (
        ("attention", context),
        ("attention_gated", gated),
        ("attention_projected", projected),
        ("attention_output", out),
    ):
        note(name, value)

    # A layer starting a block puts its input on the stack and passes its
    # attention output on alone.
    if layer % layout.block_size == 0:
        stack, residual = [*stack, hidden], out
    else:
        residual = hidden + out
    note("residual", residual)

    mixed = block_mix(stack, residual, weight("ffn_res_query")[None])[:, 0]
    note("mix_ffn", mixed)
    if messages is not None:
        gates = (weight(f"loop_ffn_ctrl_{part}") for part in ("down", "forget", "gain"))
        mixed = message_gate(mixed, messages[:, 3], *gates, eps, rounding)
        note("gated_ffn", mixed)
    normed = rounding(rms_norm(mixed, weight("ffn_norm"), eps))
    routed, scores, chosen, weights = moe(model, layer, normed, rounding)
    ffn = rounding(cache.convolve(index, "ffn", routed, weight("ffn_post_conv")))
    output = rounding(residual + ffn)
    for name, value in (
        ("ffn_input", normed),
        ("router_scores", scores),
        ("experts", chosen),
        ("expert_weights", weights),
        ("moe", routed),
        ("ffn_output", ffn),
        ("output", output),
    ):
        note(name, value)
    return output, stack, residual, (chosen, weights)


def head(model: InklingModel, x: np.ndarray, rounding=exact) -> np.ndarray:
    """Softcapped logits over the real vocabulary for the final rows x; a
    bf16 model rounds the normed rows and the head's output, then caps in
    fp32 (modeling 1443, 1534-1535)."""
    layout = model.layout
    normed = rms_norm(x, model.tensor("output_norm.weight"), layout.norm_eps)
    normed = rounding(normed)
    logits = np.empty((x.shape[0], layout.vocabulary), np.float32)
    for start in range(0, layout.vocabulary, HEAD_ROWS):
        rows = slice(start, min(start + HEAD_ROWS, layout.vocabulary))
        logits[:, rows] = rounding(normed @ model.tensor("output.weight", rows).T)
    return softcap(logits, layout.logit_softcap)


@dataclass
class Forward:
    """What one forward pass over a token sequence dumps (see the module's
    description of OUT_DIR)."""

    logits: np.ndarray
    residual_norms: np.ndarray
    residuals: np.ndarray
    experts: np.ndarray
    expert_weights: np.ndarray
    trace: dict = field(default_factory=dict)


def forward(
    model: InklingModel,
    tokens: np.ndarray,
    cache: Cache | None = None,
    residual_applications=(),
    router_applications=(),
    trace: Trace | None = None,
    logit_rows=None,
    rounding=exact,
) -> Forward:
    """The forward pass over tokens, after the earlier chunks cache holds,
    keeping the residual rows of residual_applications, the routes of
    router_applications and trace's intermediates, with the logits of
    logit_rows (every row by default), its rows rounded by rounding (exact or
    bf16)."""
    layout, count = model.layout, len(tokens)
    schedule = layout.schedule()
    cache = cache or Cache(layout)
    trace = trace or Trace()
    residual_applications = list(residual_applications)
    router_applications = list(router_applications)
    if not count or tokens.min() < 0 or tokens.max() >= layout.vocabulary:
        raise ValueError(f"tokens must be 1 or more ids below {layout.vocabulary}")
    chosen = residual_applications + router_applications + list(trace.applications)
    if any(not 0 <= index < len(schedule) for index in chosen):
        raise ValueError(f"applications must be below {len(schedule)}")
    routes = (len(router_applications), count, layout.experts_per_token)
    norms = np.empty((len(schedule), 2, count), np.float32)
    residuals = np.empty(
        (len(residual_applications), 2, count, layout.hidden), np.float32
    )
    experts, expert_weights = np.empty(routes, np.int32), np.empty(routes, np.float32)
    positions = cache.length + np.arange(count, dtype=np.int64)

    def step(index, hidden, stack, messages=None):
        started = time.monotonic()
        layer, _ = schedule[index]
        note = trace.recorder(index)
        hidden, stack, residual, route = application(
            model,
            cache,
            index,
            layer,
            hidden,
            stack,
            messages,
            positions,
            note,
            rounding,
        )
        for stage, rows in enumerate((residual, hidden)):
            norms[index, stage] = np.linalg.norm(rows, axis=-1)
            if index in residual_applications:
                residuals[residual_applications.index(index), stage] = rows
        if index in router_applications:
            experts[router_applications.index(index)] = route[0]
            expert_weights[router_applications.index(index)] = route[1]
        print(
            f"application {index} (layer {layer}): {time.monotonic() - started:.1f} s",
            file=sys.stderr,
            flush=True,
        )
        return hidden, stack

    rows = model.tensor("token_embd.weight", tokens)
    # The table's row is the weight as read; its norm's output is rounded
    # (modeling 745, 1246).
    hidden = rms_norm(rows, model.tensor("token_embd_norm.weight"), layout.norm_eps)
    hidden = rounding(hidden)
    stack, index = [], 0
    for _ in range(layout.loop_start):
        hidden, stack = step(index, hidden, stack)
        index += 1
    prelude = hidden, stack
    for loop_pass in range(layout.passes):
        if loop_pass:
            # The previous pass's bank: its block states after the loop's
            # first (that one is the prelude's output, the same every pass)
            # and its output; then the pass restarts from the prelude.
            bank = stack[len(prelude[1]) + 1 :], hidden
            hidden, stack = prelude
        for layer in range(layout.loop_start, layout.loop_end):
            messages = None
            if loop_pass:
                messages = block_mix(*bank, model.weight(layer, "loop_msg_query"))
            hidden, stack = step(index, hidden, stack, messages)
            index += 1
    for _ in range(layout.loop_end, layout.layers):
        hidden, stack = step(index, hidden, stack)
        index += 1
    cache.length += count
    final = block_mix(stack, hidden, model.tensor("output_res_query.weight")[None])
    return Forward(
        head(
            model,
            final[slice(None) if logit_rows is None else logit_rows, 0],
            rounding,
        ),
        norms,
        residuals,
        experts,
        expert_weights,
        trace.values,
    )


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
    residual_applications=(),
    router_applications=None,
    trace_applications=(),
    trace_rows=(),
    records: int = 16,
    record_logits_only: bool = False,
    precision: str = "fp32",
    logit_rows=(),
) -> None:
    started = time.monotonic()
    model = InklingModel(path)
    schedule = model.layout.schedule()
    if router_applications is None:
        router_applications = sorted({0, len(schedule) // 2, len(schedule) - 1})
    trace = Trace(trace_applications, trace_rows)
    prefixes = record_prefixes(len(tokens), records)
    # The logits kept: every row, or the records' prefixes, the last row and
    # the rows asked for (a 9,000-token prompt's logits are 7 GB).
    kept = {n - 1 for n in prefixes} | {len(tokens) - 1} | set(logit_rows)
    record_logits_only = record_logits_only or bool(logit_rows)
    logit_rows = np.arange(len(tokens))
    if record_logits_only:
        logit_rows = np.array(sorted(kept))
    result = forward(
        model,
        tokens,
        None,
        residual_applications,
        router_applications,
        trace,
        logit_rows,
        ROUNDINGS[precision],
    )
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "tokens.npy", tokens.astype(np.uint32))
    np.save(out / "logits.npy", result.logits)
    if record_logits_only:
        np.save(out / "logit_rows.npy", logit_rows)
    np.save(out / "residual_norms.npy", result.residual_norms)
    np.save(out / "residuals.npy", result.residuals)
    np.save(out / "experts.npy", result.experts)
    np.save(out / "expert_weights.npy", result.expert_weights)
    if trace.applications:
        np.savez(out / "trace.npz", rows=trace.rows, **result.trace)
    if prefixes:
        (out / "sequences.txt").write_text(" ".join(map(str, tokens)) + "\n")
        (out / "records.txt").write_text("".join(f"0 {n}\n" for n in prefixes))
        rows = result.logits[np.searchsorted(logit_rows, [n - 1 for n in prefixes])]
        rows.astype("<f4").tofile(out / "logits.f32")
    description = {
        "gguf": str(path.absolute()),
        "gguf_bytes": path.stat().st_size,
        "gguf_py": metadata.version("gguf"),
        "layout": asdict(model.layout),
        "rows": precision,
        "schedule": schedule,
        "tokens": len(tokens),
        "residual_applications": list(residual_applications),
        "router_applications": list(router_applications),
        "trace_applications": sorted(trace.applications),
        "trace_rows": trace.rows.tolist(),
        "records": prefixes,
        "logit_rows": "every" if not record_logits_only else logit_rows.tolist(),
        "seconds": round(time.monotonic() - started, 1),
    }
    (out / "reference.json").write_text(json.dumps(description, indent=2) + "\n")
    print(
        f"wrote the reference of {len(tokens)} tokens to {out} in "
        f"{description['seconds']} s; the next token is "
        f"{int(result.logits[-1].argmax())}"
    )


def number_list(text: str) -> list[int]:
    """Comma-separated numbers or inclusive ranges FIRST:LAST."""
    numbers = []
    for value in filter(None, text.split(",")):
        first, _, last = value.partition(":")
        numbers += range(int(first), int(last or first) + 1)
    return numbers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("tokens", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--residual-applications",
        type=number_list,
        default=[],
        help="comma-separated applications whose residual rows to dump",
    )
    parser.add_argument(
        "--router-applications",
        type=number_list,
        help="comma-separated applications whose routes to dump",
    )
    parser.add_argument(
        "--trace-applications",
        type=number_list,
        default=[],
        help="comma-separated applications whose intermediates to dump",
    )
    parser.add_argument(
        "--trace-rows",
        type=number_list,
        default=[-1],
        help="comma-separated rows (negative from the end) to trace",
    )
    parser.add_argument(
        "--records", type=int, default=16, help="prefixes in the logits dump"
    )
    parser.add_argument(
        "--rows",
        choices=sorted(ROUNDINGS),
        default="fp32",
        help="fp32 throughout, or rounded to bf16 where the bf16 source rounds",
    )
    parser.add_argument(
        "--logit-rows",
        type=number_list,
        default=[],
        help="rows (FIRST:LAST ranges allowed) whose logits to keep beside the"
        " records' and the last; implies --record-logits-only",
    )
    parser.add_argument(
        "--record-logits-only",
        action="store_true",
        help="keep logits only after the records' prefixes and the whole tokens",
    )
    args = parser.parse_args()
    write_reference(
        args.model,
        read_tokens(args.tokens),
        args.output,
        args.residual_applications,
        args.router_applications,
        args.trace_applications,
        args.trace_rows,
        args.records,
        args.record_logits_only,
        args.rows,
        args.logit_rows,
    )
