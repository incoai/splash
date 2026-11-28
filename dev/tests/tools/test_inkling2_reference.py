"""The Inkling-2-Mini reference forward against slow float64 loops.

Each mechanism is checked against a per-row float64 loop over random weights,
and a small inkling2_loop GGUF written here (Q8_0, BF16, F32 and NVFP4 tensors
with per-expert scales) against a float64 forward of transformers'
Inkling2LoopTextModel, one token at a time, over the same values in the
checkpoint's orientation: the GGUF gets them as llama.cpp's converter writes
them, so that the reference's reading of each convention is tested too.
"""

import contextlib
import io
import math
import unittest
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import gguf
import numpy as np

from dev.benchmarks import logit_compare
from dev.tools import inkling2_reference as reference

EPS = 1e-6
# ggml's doubled E2M1 values and llama.cpp's UE4M3 block scale, halved to
# match them (ggml-common.h kvalues_fp4, ggml-impl.h ggml_ue4m3_to_fp32).
FP4_VALUES = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def norm64(x, eps=EPS):
    return x / math.sqrt(float(np.mean(x * x)) + eps)


def sigmoid64(x):
    return 1 / (1 + np.exp(-x))


def clip64(x, threshold):
    return x * min(1.0, threshold / (float(np.linalg.norm(x)) + 1e-6))


def mix64(states, current, queries):
    """Inkling2LoopBlockAttentionResidual for one row: per query, the softmax
    over the normed states of query . state / sqrt(hidden)."""
    normed = [norm64(state) for state in [*states, current]]
    out = []
    for query in queries:
        scores = np.array([query @ state for state in normed]) / math.sqrt(len(current))
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        out.append(sum(w * state for w, state in zip(weights, normed)))
    return out


def gate64(x, message, down, forget, gain, eps=EPS):
    g = down @ np.concatenate([norm64(x, eps), norm64(message, eps)])
    return (1 - np.tanh(forget @ g)) * x + np.tanh(gain @ g) * message


def situ64(gate, up, gate_cap=4.0, up_cap=25.0):
    return (
        gate_cap
        * np.tanh(gate / gate_cap)
        * sigmoid64(gate)
        * up_cap
        * np.tanh(up / up_cap)
    )


def ue4m3(byte):
    if byte in (0, 0x7F):
        return 0.0
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    raw = (
        mantissa * 2.0**-9
        if exponent == 0
        else (1 + mantissa / 8) * 2.0 ** (exponent - 7)
    )
    return raw * 0.5


def nvfp4_values(data):
    """llama.cpp's dequantize_row_nvfp4 over rows of block_nvfp4 bytes: per
    block of 64, four UE4M3 scales, then 32 bytes whose low nibbles are the
    first 8 of each 16 and high nibbles the last 8."""
    blocks = data.reshape(-1, 36)
    out = np.zeros((len(blocks), 64))
    for b, block in enumerate(blocks):
        for s in range(4):
            scale = ue4m3(int(block[s]))
            for j in range(8):
                byte = int(block[4 + s * 8 + j])
                out[b, s * 16 + j] = FP4_VALUES[byte & 15] * scale
                out[b, s * 16 + 8 + j] = FP4_VALUES[byte >> 4] * scale
    return out.reshape(*data.shape[:-1], data.shape[-1] // 36 * 64)


class MechanismTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def normal(self, *shape, scale=1.0):
        return (self.rng.standard_normal(shape) * scale).astype(np.float32)

    def test_block_mix_with_one_three_and_four_queries(self):
        for states, count in ((0, 1), (1, 3), (4, 4)):
            stack = [self.normal(5, 32, scale=3) for _ in range(states)]
            current = self.normal(5, 32)
            queries = self.normal(count, 32)
            mixed = reference.block_mix(stack, current, queries)
            self.assertEqual(mixed.shape, (5, count, 32))
            for row in range(5):
                expected = mix64([s[row] for s in stack], current[row], queries)
                np.testing.assert_allclose(mixed[row], expected, rtol=1e-5, atol=1e-5)

    def test_l2_clip_scales_only_longer_vectors(self):
        x = self.normal(6, 3, 16)
        x[0] *= 0.01
        clipped = reference.l2_clip(x, 2.0)
        np.testing.assert_array_equal(clipped[0], x[0])
        for vector, out in zip(x.reshape(-1, 16), clipped.reshape(-1, 16)):
            np.testing.assert_allclose(
                out, clip64(vector.astype(np.float64), 2.0), rtol=1e-6
            )

    def test_log_scale_is_one_up_to_the_floor_and_logarithmic_after(self):
        positions = np.array([0, 8190, 8191, 8192, 20000, 1 << 20])
        factors = reference.log_scale(positions, 8192, 0.1)[:, 0, 0]
        expected = [1 + 0.1 * math.log(max((p + 1) / 8192, 1)) for p in positions]
        np.testing.assert_array_equal(factors[:3], 1)
        np.testing.assert_allclose(factors, expected, rtol=1e-6)

    def static64(self, conv1d, x):
        """Inkling2LoopShortConvolution: conv1d with its last tap on the row,
        zero before the sequence, plus the row, over sqrt(taps + 1)."""
        channels, width = conv1d.shape
        out = np.zeros_like(x, dtype=np.float64)
        for t in range(len(x)):
            for j in range(width):
                source = t - (width - 1) + j
                if source >= 0:
                    out[t] += conv1d[:, j] * x[source]
        return (out + x) / math.sqrt(width + 1)

    def dynamic64(self, static, dynamic, x):
        """Inkling2LoopDynamicShortConvolution: per backward distance, the
        static tap plus the row's tap of the channel's head, over
        sqrt(2 taps + 1)."""
        width, channels = static.shape
        group = channels // dynamic.shape[1]
        out = np.zeros_like(x, dtype=np.float64)
        for t in range(len(x)):
            for lag in range(width):
                if t - lag >= 0:
                    taps = static[lag] + np.repeat(dynamic[t, :, lag], group)
                    out[t] += taps * x[t - lag]
        return (out + x) / math.sqrt(2 * width + 1)

    def convolve_in_pieces(self, x, taps, dynamic, pieces):
        history = np.zeros((taps.shape[1] - 1, x.shape[1]), np.float32)
        outputs, start = [], 0
        for size in pieces:
            rows = slice(start, start + size)
            part = None if dynamic is None else dynamic[rows]
            out, history = reference.short_convolution(x[rows], taps, history, part)
            outputs.append(out)
            start += size
        return np.concatenate(outputs)

    def test_static_convolution_matches_conv1d_and_carries_its_state(self):
        x, conv1d = self.normal(9, 12), self.normal(12, 4, scale=0.5)
        expected = self.static64(conv1d.astype(np.float64), x.astype(np.float64))
        whole = self.convolve_in_pieces(x, conv1d, None, [9])
        np.testing.assert_allclose(whole, expected, rtol=1e-5, atol=1e-6)
        pieces = self.convolve_in_pieces(x, conv1d, None, [2, 1, 4, 2])
        np.testing.assert_allclose(pieces, whole, rtol=0, atol=1e-6)

    def test_dynamic_convolution_flips_its_static_taps_and_carries_its_state(self):
        x = self.normal(9, 12)
        static, dynamic = self.normal(4, 12, scale=0.5), self.normal(9, 3, 4, scale=0.5)
        expected = self.dynamic64(
            static.astype(np.float64), dynamic.astype(np.float64), x
        )
        # The converter writes static_weight (row 0 on the row itself) as
        # flip(0).T, in the static convolution's order.
        taps = np.ascontiguousarray(static[::-1].T)
        whole = self.convolve_in_pieces(x, taps, dynamic, [9])
        np.testing.assert_allclose(whole, expected, rtol=1e-5, atol=1e-6)
        pieces = self.convolve_in_pieces(x, taps, dynamic, [1, 3, 3, 2])
        np.testing.assert_allclose(pieces, whole, rtol=0, atol=1e-6)

    def attention64(self, q, k, v, relative, table, sinks, rows, columns, window):
        heads, kv_heads = q.shape[1], k.shape[1]
        context = np.zeros(q.shape)
        for i, p in enumerate(rows):
            for h in range(heads):
                g = h // (heads // kv_heads)
                scores, values = [], []
                for j, c in enumerate(columns):
                    d = p - c
                    if d < 0 or (window is not None and d >= window):
                        continue
                    bias = relative[i, h] @ table[d] if d < len(table) else 0.0
                    scores.append(q[i, h] @ k[j, g] / math.sqrt(q.shape[2]) + bias)
                    values.append(v[j, g])
                scores = np.array(scores)
                peak = max(scores.max(), sinks[i, h])
                weights = np.exp(scores - peak)
                total = weights.sum() + math.exp(sinks[i, h] - peak)
                context[i, h] = weights @ np.array(values) / total
        return context

    def test_attention_window_relative_bias_and_sinks(self):
        rows, columns = np.arange(14, 21), np.arange(3, 21)
        q, k, v = self.normal(7, 4, 8), self.normal(18, 2, 8), self.normal(18, 2, 8)
        relative, table = self.normal(7, 4, 3), self.normal(5, 3)
        sinks = self.normal(7, 4)
        # A sink above every score and one below them all.
        sinks[0, 0], sinks[1, 1] = 40.0, -40.0
        for window in (None, 3, 6):
            with mock.patch.object(reference, "ATTENTION_ROWS", 3):
                out = reference.attend(
                    q, k, v, relative, table, sinks, rows, columns, window
                )
            expected = self.attention64(
                q, k, v, relative, table, sinks, rows, columns, window
            )
            np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
        # A window wider than every distance is full attention, bit for bit.
        full = reference.attend(q, k, v, relative, table, sinks, rows, columns, None)
        wide = reference.attend(q, k, v, relative, table, sinks, rows, columns, 64)
        np.testing.assert_array_equal(wide, full)

    def test_router_chooses_by_biased_score_and_weights_by_score(self):
        x, router = np.abs(self.normal(6, 16)) + 0.1, self.normal(8, 16, scale=0.25)
        bias = self.normal(8, scale=0.05)
        # Expert 0 scores lowest on every row but its bias puts it first.
        router[0], bias[0] = -1.0, 2.0
        scores, chosen, weights = reference.route(x, router, bias, 3, 8.0)
        self.assertTrue((scores[:, 0] < scores[:, 1:].min(axis=1)).all())
        for row in range(6):
            s = sigmoid64(router.astype(np.float64) @ x[row])
            order = sorted(range(8), key=lambda e: -(s[e] + bias[e]))[:3]
            self.assertEqual(list(chosen[row]), order)
            self.assertEqual(order[0], 0)
            np.testing.assert_allclose(scores[row], s, rtol=1e-6)
            expected = s[order] / (s[order].sum() + 1e-20) * 8
            # Expert 0's weight is about 1e-6: its fp32 sigmoid is near
            # underflow's precision.
            np.testing.assert_allclose(weights[row], expected, rtol=1e-5)

    def test_expert_softcaps_gate_and_up_and_not_the_sigmoid(self):
        x = self.normal(5, 16, scale=4)
        gate_up, down = self.normal(24, 16), self.normal(16, 12, scale=0.3)
        out = reference.expert(x, gate_up, down, 4.0, 25.0)
        projected = x.astype(np.float64) @ gate_up.T
        # Past both caps, where a softcap applied to the sigmoid's input
        # would differ.
        self.assertGreater(np.abs(projected[:, 12:]).max(), 25)
        self.assertGreater(np.abs(projected[:, :12]).max(), 8)
        for row in range(5):
            expected = down @ situ64(projected[row, :12], projected[row, 12:])
            np.testing.assert_allclose(out[row], expected, rtol=1e-5, atol=1e-5)

    def test_message_gate(self):
        x, message = self.normal(4, 16), self.normal(4, 16, scale=2)
        down = self.normal(6, 32, scale=0.3)
        forget, gain = self.normal(16, 6), self.normal(16, 6)
        out = reference.message_gate(x, message, down, forget, gain, EPS)
        for row in range(4):
            expected = gate64(x[row], message[row], down, forget, gain)
            np.testing.assert_allclose(out[row], expected, rtol=1e-5, atol=1e-6)

    def test_bf16_rounds_to_nearest_with_ties_to_even(self):
        one = np.float32(1.0)
        step = np.float32(2.0**-7)
        x = np.array(
            [
                1 + 2.0**-8,
                1 + 3 * 2.0**-8,
                1 + 2.0**-8 + 2.0**-20,
                -(1 + 2.0**-9),
                3.4e38,
            ],
            dtype=np.float32,
        )
        expected = [one, one + 2 * step, one + step, -one, np.inf]
        np.testing.assert_array_equal(reference.bf16(x), np.array(expected, np.float32))
        x = self.normal(1000, scale=100)
        rounded = reference.bf16(x)
        np.testing.assert_array_equal(rounded.view(np.uint32) & 0xFFFF, 0)
        # Within half a bf16 unit of x: 2^-8 of its magnitude's power of two.
        unit = 2.0 ** (np.floor(np.log2(np.abs(x))) - 7)
        self.assertTrue((np.abs(rounded.astype(np.float64) - x) <= unit / 2).all())


# fp32 rounding through the 16 applications leaves the logits within 1.4e-4
# of the float64 forward (median 8e-6) and a prompt in pieces within 1.8e-4 of
# it at once, where a convention gotten wrong (dynamic taps unflipped, the
# window one key wider) moves them by 1 to 17: the bound sits between.
LOGIT_TOLERANCE = 1e-3
# The synthetic model: 8 layers in blocks of 2, the loop over layers 2-5 run
# three times (16 applications), a window and a relative extent shorter than
# the prompt and a log-scale floor inside it.
LAYERS, HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 8, 64, 4, 2, 16
SLIDING = (True, True, False, True, True, False, True, False)
BLOCK, LOOP_START, LOOP_END, PASSES, RANK = 2, 2, 6, 3, 8
WINDOW, EXTENT, RELATIVE, TAPS = 4, 6, 4, 4
EXPERTS, USED, INTERMEDIATE = 8, 3, 64
VOCABULARY, PADDED = 37, 40
NVFP4_LAYERS = range(5)
# The float keys as the GGUF stores them, in fp32.
F32 = {
    "attention.layer_norm_rms_epsilon": 1e-6,
    "attention.qk_clip": 4.0,
    "attention.relative_clip": 2.0,
    "attention.log_scale_alpha": 0.1,
    "expert_weights_scale": 8.0,
    "final_logit_softcapping": 30.0,
    "activation.situ_beta": 4.0,
    "activation.situ_linear_beta": 25.0,
}
CONSTANT = {name: float(np.float32(value)) for name, value in F32.items()}
FLOOR = 4
TOKENS = np.array([3, 17, 0, 36, 5, 5, 22, 9, 30, 1, 14, 28, 7])


class SyntheticModel:
    """Random weights in the checkpoint's orientation (as transformers holds
    them), the values a GGUF of them holds, and that GGUF."""

    def __init__(self, seed=11):
        self.rng = np.random.default_rng(seed)
        self.tensors = []
        self.values = {
            name: self.tensor(name, values, kind)
            for name, values, kind in (
                ("token_embd.weight", self.normal(PADDED, HIDDEN, scale=1.0), "Q8_0"),
                ("token_embd_norm.weight", self.norm_weight(), "F32"),
                ("output.weight", self.normal(PADDED, HIDDEN, scale=0.5), "Q8_0"),
                ("output_norm.weight", self.norm_weight(), "F32"),
                ("output_res_query.weight", self.normal(HIDDEN, scale=1.0), "F32"),
            )
        }
        self.layers = [self.layer(layer) for layer in range(LAYERS)]

    def normal(self, *shape, scale=None):
        scale = 1 / math.sqrt(shape[-1]) if scale is None else scale
        return (self.rng.standard_normal(shape) * scale).astype(np.float32)

    def norm_weight(self):
        return 1 + self.normal(HIDDEN, scale=0.1)

    def tensor(self, name, values, kind="F32"):
        """Adds a GGUF tensor of values (numpy order) and returns the values
        it holds, in float64."""
        qtype = getattr(gguf.GGMLQuantizationType, kind)
        data = gguf.quantize(values, qtype)
        self.tensors.append((name, data, qtype))
        return gguf.dequantize(data, qtype).astype(np.float64)

    def nvfp4_experts(self, name, rows, columns):
        """Random NVFP4 expert matrices and their per-expert scales, as
        llama.cpp decodes and multiplies them."""
        blocks = self.rng.integers(
            0, 256, (EXPERTS, rows, columns // 64, 36), dtype=np.uint8
        )
        # Block scales of 2^-6 to 2^-4, so that weights are near 1/sqrt(64).
        blocks[..., :4] = (self.rng.integers(1, 4, blocks[..., :4].shape) << 3) | (
            self.rng.integers(0, 8, blocks[..., :4].shape)
        )
        data = blocks.reshape(EXPERTS, rows, -1)
        scale = (0.5 + self.rng.random(EXPERTS)).astype(np.float32)
        self.tensors.append((name + ".weight", data, gguf.GGMLQuantizationType.NVFP4))
        self.tensors.append((name + ".scale", scale, gguf.GGMLQuantizationType.F32))
        return nvfp4_values(data) * scale[:, None, None].astype(np.float64)

    def layer(self, layer):
        def tensor(name, values, kind="F32"):
            return self.tensor(f"blk.{layer}.{name}", values, kind)

        sliding = SLIDING[layer]
        kv = KV_HEADS * HEAD_DIM
        w = {"sliding": sliding}
        if sliding:
            w["attn_norm"] = tensor("attn_norm.weight", self.norm_weight())
        else:
            for kind in "qkv":
                w[f"{kind}_norm"] = tensor(
                    f"attn_{kind}_inp_norm.weight", self.norm_weight()
                )
        if layer:
            queries = self.normal(1 if sliding else 3, HIDDEN, scale=1.0)
            w["attn_queries"] = tensor(
                "attn_res_query.weight", queries[0] if sliding else queries
            ).reshape(-1, HIDDEN)
        w["ffn_query"] = tensor("ffn_res_query.weight", self.normal(HIDDEN, scale=1.0))
        w["q"] = tensor("attn_q.weight", self.normal(HEADS * HEAD_DIM, HIDDEN), "Q8_0")
        w["k"] = tensor("attn_k.weight", self.normal(kv, HIDDEN), "Q8_0")
        w["v"] = tensor("attn_v.weight", self.normal(kv, HIDDEN), "Q8_0")
        w["o"] = tensor(
            "attn_output.weight", self.normal(HIDDEN, HEADS * HEAD_DIM), "Q8_0"
        )
        w["gate"] = tensor(
            "attn_gate.weight", self.normal(HEADS * HEAD_DIM, HIDDEN), "Q8_0"
        )
        w["rel"] = tensor(
            "attn_rel.weight", self.normal(HEADS * RELATIVE, HIDDEN), "Q8_0"
        )
        w["sink"] = tensor("attn_sink_proj.weight", self.normal(HEADS, HIDDEN), "BF16")
        w["q_scale"] = tensor(
            "attn_q_clip.weight", 1 + self.normal(HEADS, HEAD_DIM, scale=0.1)
        )
        w["rel_scale"] = tensor(
            "attn_rel_clip.weight", 1 + self.normal(HEADS, RELATIVE, scale=0.1)
        )
        # proj is [relative length, extent]; the converter writes it transposed.
        proj = self.normal(RELATIVE, WINDOW if sliding else EXTENT, scale=1.0)
        w["proj"] = tensor("attn_rel_bias.weight", np.ascontiguousarray(proj.T)).T
        for kind in "kv":
            if sliding:
                # conv1d's weight, its last tap on the row itself, as written.
                w[f"{kind}_conv"] = tensor(
                    f"attn_{kind}_conv.weight", self.normal(kv, TAPS, scale=0.3)
                )
            else:
                # static_weight, row 0 on the row itself, written flip(0).T.
                static = self.normal(TAPS, kv, scale=0.3)
                taps = tensor(
                    f"attn_{kind}_conv.weight", np.ascontiguousarray(static[::-1].T)
                )
                w[f"{kind}_static"] = taps.T[::-1]
                w[f"{kind}_conv_proj"] = tensor(
                    f"attn_{kind}_conv_proj.weight",
                    self.normal(KV_HEADS * TAPS, HIDDEN),
                    "BF16",
                )
        w["attn_conv"] = tensor(
            "attn_post_conv.weight", self.normal(HIDDEN, TAPS, scale=0.3)
        )
        w["ffn_conv"] = tensor(
            "ffn_post_conv.weight", self.normal(HIDDEN, TAPS, scale=0.3)
        )
        w["ffn_norm"] = tensor("ffn_norm.weight", self.norm_weight())
        w["router"] = tensor("ffn_gate_inp.weight", self.normal(EXPERTS, HIDDEN))
        w["router_bias"] = tensor("exp_probs_b.bias", self.normal(EXPERTS, scale=0.1))
        if layer in NVFP4_LAYERS:
            w["gate_up"] = self.nvfp4_experts(
                f"blk.{layer}.ffn_gate_up_exps", 2 * INTERMEDIATE, HIDDEN
            )
            w["down"] = self.nvfp4_experts(
                f"blk.{layer}.ffn_down_exps", HIDDEN, INTERMEDIATE
            )
        else:
            gate_up = self.normal(EXPERTS, 2 * INTERMEDIATE, HIDDEN)
            w["gate_up"] = tensor("ffn_gate_up_exps.weight", gate_up, "Q8_0")
            down = self.normal(EXPERTS, HIDDEN, INTERMEDIATE)
            w["down"] = tensor("ffn_down_exps.weight", down, "Q8_0")
        if LOOP_START <= layer < LOOP_END:
            # The q, k, v and FFN message queries, in that order.
            w["message_queries"] = tensor(
                "loop_msg_query.weight", self.normal(4, HIDDEN, scale=1.0)
            )
            for kind in reference.MESSAGES:
                w[f"{kind}_gate"] = (
                    tensor(
                        f"loop_{kind}_ctrl_down.weight", self.normal(RANK, 2 * HIDDEN)
                    ),
                    tensor(
                        f"loop_{kind}_ctrl_forget.weight", self.normal(HIDDEN, RANK)
                    ),
                    tensor(f"loop_{kind}_ctrl_gain.weight", self.normal(HIDDEN, RANK)),
                )
        return w

    def write(self, path: Path) -> Path:
        writer = gguf.GGUFWriter(path, reference.ARCHITECTURE)
        arch = reference.ARCHITECTURE
        for key, value in {
            "block_count": LAYERS,
            "embedding_length": HIDDEN,
            "attention.key_length": HEAD_DIM,
            "attention.value_length": HEAD_DIM,
            "attention.key_length_swa": HEAD_DIM,
            "attention.value_length_swa": HEAD_DIM,
            "attention.relative_extent": EXTENT,
            "attention.relative_length": RELATIVE,
            "attention.log_scale_floor": FLOOR,
            "attention.sliding_window": WINDOW,
            "attn_res.block_size": BLOCK,
            "expert_count": EXPERTS,
            "expert_used_count": USED,
            "expert_feed_forward_length": INTERMEDIATE,
            "loop.start": LOOP_START,
            "loop.end": LOOP_END,
            "loop.count": PASSES,
            "loop.gate_rank": RANK,
            "rope.dimension_count": 0,
            "shortconv.l_cache": TAPS,
            "vocab_size": PADDED,
            "vocab_size_unpadded": VOCABULARY,
        }.items():
            writer.add_uint32(f"{arch}.{key}", value)
        for key, value in F32.items():
            writer.add_float32(f"{arch}.{key}", value)
        writer.add_array(f"{arch}.attention.head_count", [HEADS] * LAYERS)
        writer.add_array(f"{arch}.attention.head_count_kv", [KV_HEADS] * LAYERS)
        writer.add_array(f"{arch}.attention.sliding_window_pattern", list(SLIDING))
        writer.add_bool(f"{arch}.loop.varying", False)
        for name, data, qtype in self.tensors:
            writer.add_tensor(name, data, raw_dtype=qtype)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return path


class Forward64:
    """Inkling2LoopTextModel's forward in float64, one token at a time, over
    a SyntheticModel's values: every application keeps its own keys and
    values and its convolutions' past inputs."""

    def __init__(self, model: SyntheticModel):
        self.model = model
        self.keys = defaultdict(list)
        self.past = defaultdict(list)

    def convolve(self, app, name, x, conv1d=None, static=None, dynamic=None):
        past = self.past[app, name]
        width = TAPS

        def back(lag):
            return x if lag == 0 else (past[-lag] if len(past) >= lag else 0 * x)

        if conv1d is not None:
            out = sum(conv1d[:, width - 1 - lag] * back(lag) for lag in range(width))
            out = (out + x) / math.sqrt(width + 1)
        else:
            group = len(x) // len(dynamic)
            out = sum(
                (static[lag] + np.repeat(dynamic[:, lag], group)) * back(lag)
                for lag in range(width)
            )
            out = (out + x) / math.sqrt(2 * width + 1)
        past.append(x)
        return out

    def layer(self, app, layer, h, stack, messages, position):
        w = self.model.layers[layer]
        eps = CONSTANT["attention.layer_norm_rms_epsilon"]
        sliding = w["sliding"]
        inputs = [h] * 3
        if "attn_queries" in w:
            mixed = mix64(stack, h, w["attn_queries"])
            inputs = [mixed[0]] * 3 if sliding else mixed
        if messages is not None:
            inputs = [
                gate64(x, messages[j], *w[f"{kind}_gate"], eps)
                for j, (x, kind) in enumerate(zip(inputs, "qkv"))
            ]
        if sliding:
            inputs = [norm64(x, eps) * w["attn_norm"] for x in inputs]
        else:
            inputs = [norm64(x, eps) * w[f"{k}_norm"] for x, k in zip(inputs, "qkv")]
        xq, xk, xv = inputs
        tau = 1.0
        if not sliding:
            alpha = CONSTANT["attention.log_scale_alpha"]
            tau = 1 + alpha * math.log(max((position + 1) / FLOOR, 1))
        clip, rel_clip = (
            CONSTANT["attention.qk_clip"],
            CONSTANT["attention.relative_clip"],
        )
        q = (w["q"] @ xq).reshape(HEADS, HEAD_DIM) * w["q_scale"]
        q = np.array([clip64(head, clip) for head in q]) * tau
        s = (w["rel"] @ xq).reshape(HEADS, RELATIVE) * w["rel_scale"]
        s = np.array([clip64(head, rel_clip) for head in s]) * tau
        sinks = w["sink"] @ xq
        if sliding:
            k = self.convolve(app, "k", w["k"] @ xk, conv1d=w["k_conv"])
            v = self.convolve(app, "v", w["v"] @ xv, conv1d=w["v_conv"])
        else:
            dk = (w["k_conv_proj"] @ xk).reshape(KV_HEADS, TAPS)
            dv = (w["v_conv_proj"] @ xv).reshape(KV_HEADS, TAPS)
            k = self.convolve(app, "k", w["k"] @ xk, static=w["k_static"], dynamic=dk)
            v = self.convolve(app, "v", w["v"] @ xv, static=w["v_static"], dynamic=dv)
        k = np.array([clip64(head, clip) for head in k.reshape(KV_HEADS, HEAD_DIM)])
        v = np.array([norm64(head, eps) for head in v.reshape(KV_HEADS, HEAD_DIM)])
        self.keys[app].append((position, k, v))
        extent = w["proj"].shape[1]
        context = np.zeros((HEADS, HEAD_DIM))
        for head in range(HEADS):
            g = head // (HEADS // KV_HEADS)
            scores, values = [], []
            for key_position, key, value in self.keys[app]:
                d = position - key_position
                if sliding and d >= WINDOW:
                    continue
                bias = s[head] @ w["proj"][:, d] if d < extent else 0.0
                scores.append(q[head] @ key[g] / math.sqrt(HEAD_DIM) + bias)
                values.append(value[g])
            logits = np.array([*scores, sinks[head]])
            weights = np.exp(logits - logits.max())
            weights /= weights.sum()
            context[head] = weights[:-1] @ np.array(values)
        gated = context.reshape(-1) * sigmoid64(w["gate"] @ xq)
        o = self.convolve(app, "attention", w["o"] @ gated, conv1d=w["attn_conv"])
        if layer % BLOCK == 0:
            stack, r = [*stack, h], o
        else:
            r = h + o
        m = mix64(stack, r, [w["ffn_query"]])[0]
        if messages is not None:
            m = gate64(m, messages[3], *w["ffn_gate"], eps)
        x = norm64(m, eps) * w["ffn_norm"]
        scores = sigmoid64(w["router"] @ x)
        chosen = sorted(
            range(EXPERTS), key=lambda e: -(scores[e] + w["router_bias"][e])
        )[:USED]
        weights = (
            scores[chosen]
            / (scores[chosen].sum() + 1e-20)
            * CONSTANT["expert_weights_scale"]
        )
        f = 0
        for weight, e in zip(weights, chosen):
            projected = w["gate_up"][e] @ x
            act = situ64(projected[:INTERMEDIATE], projected[INTERMEDIATE:])
            f = f + weight * (w["down"][e] @ act)
        f = self.convolve(app, "ffn", f, conv1d=w["ffn_conv"])
        return r + f, stack

    def token(self, token, position):
        model = self.model
        eps = CONSTANT["attention.layer_norm_rms_epsilon"]
        h = norm64(model.values["token_embd.weight"][token], eps)
        h = h * model.values["token_embd_norm.weight"]
        stack, app = [], 0
        for layer in range(LOOP_START):
            h, stack = self.layer(app, layer, h, stack, None, position)
            app += 1
        prelude = h, stack
        for loop_pass in range(PASSES):
            if loop_pass:
                bank = stack[len(prelude[1]) + 1 :], h
                h, stack = prelude
            for layer in range(LOOP_START, LOOP_END):
                messages = None
                if loop_pass:
                    messages = mix64(*bank, model.layers[layer]["message_queries"])
                h, stack = self.layer(app, layer, h, stack, messages, position)
                app += 1
        for layer in range(LOOP_END, LAYERS):
            h, stack = self.layer(app, layer, h, stack, None, position)
            app += 1
        final = mix64(stack, h, [model.values["output_res_query.weight"]])[0]
        x = norm64(final, eps) * model.values["output_norm.weight"]
        logits = model.values["output.weight"][:VOCABULARY] @ x
        cap = CONSTANT["final_logit_softcapping"]
        return cap * np.tanh(logits / cap)

    def logits(self, tokens):
        return np.array([self.token(token, p) for p, token in enumerate(tokens)])


class SyntheticModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory()
        cls.synthetic = SyntheticModel()
        cls.path = cls.synthetic.write(Path(cls.directory.name) / "inkling.gguf")
        cls.model = reference.InklingModel(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def forward(self, tokens, cache=None, **options):
        with contextlib.redirect_stderr(io.StringIO()):
            return reference.forward(self.model, tokens, cache, **options)

    def test_the_header_gives_the_layout_and_the_schedule(self):
        layout = self.model.layout
        self.assertEqual(
            (layout.vocabulary, layout.window, layout.passes, layout.sliding),
            (VOCABULARY, WINDOW, PASSES, SLIDING),
        )
        schedule = layout.schedule()
        self.assertEqual(len(schedule), 16)
        self.assertEqual(schedule[:3], [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(schedule[6:8], [(2, 1), (3, 1)])
        self.assertEqual(schedule[-3:], [(5, 2), (6, 0), (7, 0)])

    def test_nvfp4_experts_are_decoded_and_scaled_as_llama_cpp_does(self):
        layer = self.synthetic.layers[0]
        for name, values in (
            ("ffn_gate_up_exps", layer["gate_up"]),
            ("ffn_down_exps", layer["down"]),
        ):
            for expert in (0, EXPERTS - 1):
                read = self.model.expert(0, name, expert)
                np.testing.assert_allclose(read, values[expert], rtol=1e-6, atol=0)

    def test_logits_match_a_float64_forward_over_the_same_values(self):
        logits = self.forward(TOKENS).logits
        expected = Forward64(self.synthetic).logits(TOKENS)
        self.assertEqual(logits.shape, (len(TOKENS), VOCABULARY))
        np.testing.assert_allclose(logits, expected, rtol=0, atol=LOGIT_TOLERANCE)

    def test_a_prompt_in_pieces_with_its_cache_is_the_prompt_at_once(self):
        whole = self.forward(TOKENS).logits
        cache = reference.Cache(self.model.layout)
        pieces, start = [], 0
        for size in (5, 1, 2, 5):
            pieces.append(self.forward(TOKENS[start : start + size], cache).logits)
            start += size
        self.assertEqual(cache.length, len(TOKENS))
        np.testing.assert_allclose(
            np.concatenate(pieces), whole, rtol=0, atol=LOGIT_TOLERANCE
        )

    def test_the_dump_is_a_logits_dump_with_the_traced_intermediates(self):
        out = Path(self.directory.name) / "reference"
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            reference.write_reference(
                self.path,
                TOKENS,
                out,
                residual_applications=[7],
                trace_applications=[0, 2, 7],
                trace_rows=[0, -1],
                records=4,
            )
        summary = logit_compare.compare(out, out)
        self.assertEqual((summary["records"], summary["top1_agreement"]), (4, 1.0))
        logits = np.load(out / "logits.npy")
        prefixes = [
            int(line.split()[1])
            for line in (out / "records.txt").read_text().splitlines()
        ]
        dumped = np.fromfile(out / "logits.f32", dtype="<f4").reshape(len(prefixes), -1)
        np.testing.assert_array_equal(dumped, logits[[n - 1 for n in prefixes]])
        norms = np.load(out / "residual_norms.npy")
        residuals = np.load(out / "residuals.npy")
        self.assertEqual(norms.shape, (16, 2, len(TOKENS)))
        trace = np.load(out / "trace.npz")
        # Layer 0 mixes nothing, full layer 2 mixes three inputs, and
        # application 7 (layer 3 in the second pass) takes messages.
        self.assertNotIn("a00.mix_attention", trace)
        self.assertNotIn("a02.messages", trace)
        self.assertEqual(trace["a02.mix_attention"].shape, (2, 3, HIDDEN))
        self.assertEqual(trace["a07.messages"].shape, (2, 4, HIDDEN))
        self.assertEqual(trace["a07.relative_bias"].shape, (2, HEADS, WINDOW))
        self.assertEqual(trace["a02.relative_bias"].shape, (2, HEADS, EXTENT))
        np.testing.assert_array_equal(trace["a07.residual"], residuals[0, 0, [0, -1]])
        np.testing.assert_array_equal(trace["a07.output"], residuals[0, 1, [0, -1]])
        np.testing.assert_allclose(
            np.linalg.norm(trace["a07.output"], axis=-1),
            norms[7, 1, [0, -1]],
            rtol=1e-6,
        )

    def test_bf16_rows_round_where_the_source_rounds_and_nowhere_else(self):
        trace = reference.Trace(range(16), range(len(TOKENS)))
        self.forward(TOKENS, trace=trace, rounding=reference.bf16)
        names = {key.split(".", 1)[1] for key in trace.values}

        def representable(name):
            values = [v for k, v in trace.values.items() if k.endswith("." + name)]
            return all(np.array_equal(reference.bf16(v), v) for v in values)

        rounded = {
            *("input_q", "input_k", "input_v", "q", "k", "v", "relative", "sink"),
            *("k_projected", "v_projected", "k_convolved", "v_convolved"),
            *("relative_bias", "attention", "attention_gated", "attention_projected"),
            *("attention_output", "ffn_input", "moe", "ffn_output", "output"),
        }
        fp32 = {
            *("mix_attention", "messages", "gated_q", "gated_k", "gated_v"),
            *("residual", "mix_ffn", "gated_ffn", "router_scores", "expert_weights"),
        }
        self.assertEqual(names, rounded | fp32 | {"experts"})
        self.assertEqual({n for n in names - {"experts"} if representable(n)}, rounded)

    def test_bf16_rows_move_the_logits_as_far_as_bf16_rounding_predicts(self):
        exact = self.forward(TOKENS).logits
        rounded = self.forward(TOKENS, rounding=reference.bf16).logits
        expected = Forward64(self.synthetic).logits(TOKENS)
        # Measured: fp32 is within 8.3e-5 of float64, 1400 fp32 units (2^-24);
        # bf16 rows move the logits by up to 7.5, 1900 bf16 units (2^-8). The
        # model amplifies a unit of rounding alike in both: the ratio is 1.4.
        fp32_units = np.abs(exact - expected).max() / 2.0**-24
        bf16_units = np.abs(rounded - exact).max() / 2.0**-8
        self.assertTrue(0.3 < bf16_units / fp32_units < 3, (bf16_units, fp32_units))

    def test_record_logits_only_keeps_the_dump_and_drops_the_other_rows(self):
        root = Path(self.directory.name)
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            for name, only, extra in (
                ("every", False, ()),
                ("records", True, ()),
                ("asked", False, (2, 3)),
            ):
                reference.write_reference(
                    self.path,
                    TOKENS,
                    root / name,
                    records=4,
                    record_logits_only=only,
                    logit_rows=extra,
                )
        every = np.load(root / "every" / "logits.npy")
        rows = np.load(root / "records" / "logit_rows.npy")
        prefixes = (root / "records" / "records.txt").read_text().split()[1::2]
        self.assertEqual(
            list(rows), sorted({int(n) - 1 for n in prefixes} | {len(TOKENS) - 1})
        )
        np.testing.assert_allclose(
            np.load(root / "records" / "logits.npy"), every[rows], rtol=0, atol=1e-5
        )
        self.assertEqual(
            logit_compare.compare(root / "every", root / "records")["top1_agreement"],
            1.0,
        )
        # Rows asked for join the records' and the last.
        asked = np.load(root / "asked" / "logit_rows.npy")
        self.assertEqual(list(asked), sorted({*rows, 2, 3}))
        np.testing.assert_allclose(
            np.load(root / "asked" / "logits.npy"), every[asked], rtol=0, atol=1e-5
        )
        self.assertEqual(reference.number_list("1,4:6,9"), [1, 4, 5, 6, 9])


if __name__ == "__main__":
    unittest.main()
