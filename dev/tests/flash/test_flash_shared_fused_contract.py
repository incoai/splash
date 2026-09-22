#!/usr/bin/env python3
"""CPU/source audit for a shared-expert gate/up + SwiGLU fusion.

These fixtures preserve the BF16 boundaries between matmul, sigmoid, SiLU and
up multiplication. CPU libm exp is suitable for the ordinary fixtures here;
it cannot predict Metal fast exp at a BF16 halfway boundary. The separately
injected exponent fixture documents that distinction. MPP dynamic-K reduction
order and full-model correctness must still be qualified on the actual GPU.
No Metal/MLX imports, model loading or GPU commands occur in this test.
"""

from __future__ import annotations

import math
import re
import struct
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ONE = 0x3F80
QNAN = 0x7FC0


def f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def number(word):
    return struct.unpack("<f", struct.pack("<I", word << 16))[0]


def bf16(value):
    word = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
    high, remainder = divmod(word, 65536)
    if word & 0x7F800000 == 0x7F800000:
        return high | (0x40 if word & 0x7FFFFF else 0)
    return (high + int(remainder > 32768 or (remainder == 32768 and high & 1))) & 65535


def cpu_exp(value):
    try:
        return f32(math.exp(value))
    except OverflowError:
        return math.inf


def canonical_trace(gate_word, up_word, *, exponent_word=None):
    """Literal BF16 boundaries; an optional already rounded Metal exp oracle."""
    gate, up = number(gate_word), number(up_word)
    if not math.isfinite(gate) or not math.isfinite(up):
        return {"diagnostics": 4, "output": QNAN}
    exponential = bf16(cpu_exp(abs(gate))) if exponent_word is None else exponent_word
    denominator = bf16(number(ONE) + number(exponential))
    tail = bf16(number(ONE) / number(denominator))
    sigmoid = tail if gate < 0 else bf16(number(ONE) - number(tail))
    silu = bf16(gate * number(sigmoid))
    output = bf16(number(silu) * up)
    invalid = not math.isfinite(number(output))
    return {
        "exponential": exponential,
        "denominator": denominator,
        "tail": tail,
        "sigmoid": sigmoid,
        "silu": silu,
        "output": QNAN if invalid else output,
        "diagnostics": 4 if invalid else 0,
    }


def from_matmul_dots(gate_f32, up_f32):
    gate_f32, up_f32 = f32(gate_f32), f32(up_f32)
    result = canonical_trace(bf16(gate_f32), bf16(up_f32))
    if not math.isfinite(gate_f32) or not math.isfinite(up_f32):
        result["diagnostics"] |= 4
    return result


class SharedActivationBoundaryTests(unittest.TestCase):
    def test_ordinary_existing_mlx_gpu_golden(self):
        # Existing actual MLX ordinary golden in test_flash_moe_reference.cpp.
        gates = [-7.96875, -1, 0, 1, 0.75, 0.5, -0.5]
        ups = [-4.625, 1, 7, 1, 2, 1, 1]
        outputs = [
            0.0126953125,
            -0.26953125,
            0,
            0.73046875,
            1.015625,
            0.3125,
            -0.1884765625,
        ]
        for g, u, expected in zip(gates, ups, outputs):
            trace = canonical_trace(bf16(g), bf16(u))
            self.assertEqual(trace["diagnostics"], 0)
            self.assertEqual(trace["output"], bf16(expected))

    def test_every_sigmoid_operation_keeps_bf16_result(self):
        positive = canonical_trace(bf16(0.5), ONE)
        negative = canonical_trace(bf16(-0.5), ONE)
        self.assertEqual(number(positive["exponential"]), 1.6484375)
        self.assertEqual(number(positive["denominator"]), 2.65625)
        self.assertEqual(number(positive["tail"]), 0.376953125)
        self.assertEqual(number(positive["sigmoid"]), 0.625)
        self.assertEqual(number(negative["sigmoid"]), 0.376953125)
        # The older CPU F32 tail would round the final sigmoid to .62109375.
        old_tail = f32(1 / f32(1 + number(positive["exponential"])))
        self.assertEqual(bf16(f32(1 - old_tail)), bf16(0.62109375))
        self.assertNotEqual(positive["sigmoid"], bf16(f32(1 - old_tail)))

    def test_missing_sigmoid_storage_changes_output(self):
        gate, up = -7.96875, -4.625
        correct = canonical_trace(bf16(gate), bf16(up))["output"]
        float_silu = bf16(f32(gate / f32(1 + cpu_exp(-gate))))
        missing = bf16(number(float_silu) * up)
        self.assertNotEqual(correct, missing)

    def test_fast_exp_threshold_cannot_be_qualified_by_cpu_libm(self):
        # Existing GPU golden: compiled gate=-6.84375, up=.125 -> BF16 0xba70.
        # A BF16 exp operand of 936 instead of CPU-libm's 940 changes the output.
        gate, up = bf16(-6.84375), bf16(0.125)
        precise = canonical_trace(gate, up)
        candidate_fast_operand = canonical_trace(gate, up, exponent_word=0x446A)
        self.assertEqual(precise["exponential"], 0x446B)
        self.assertEqual(precise["output"], 0xBA6E)
        self.assertEqual(candidate_fast_operand["output"], 0xBA70)
        self.assertNotEqual(precise["output"], candidate_fast_operand["output"])

    def test_matmul_dot_rounding_occurs_before_activation(self):
        # Different F32 dots that round to identical BF16 arrays are equivalent
        # inputs to canonical SwiGLU, regardless of their matmul reduction path.
        g, u = 0.75, 1.0
        reference = from_matmul_dots(g, u)
        for delta in [-0.0001, 0, 0.0001]:
            self.assertEqual(from_matmul_dots(g + delta, u + delta), reference)

    def test_gate_and_up_are_independent_operands(self):
        g, u = bf16(0.5), bf16(2)
        correct = canonical_trace(g, u)["output"]
        reversed_operands = canonical_trace(u, g)["output"]
        self.assertNotEqual(correct, reversed_operands)

    def test_exp_and_denominator_overflow_are_valid_saturation(self):
        negative = canonical_trace(0xFF7F, ONE)
        positive = canonical_trace(0x7F7F, ONE)
        self.assertEqual(negative["exponential"], 0x7F80)
        self.assertEqual(negative["denominator"], 0x7F80)
        self.assertEqual(negative["tail"], 0)
        self.assertEqual(negative["sigmoid"], 0)
        self.assertEqual(negative["output"], 0x8000)
        self.assertEqual(negative["diagnostics"], 0)
        self.assertEqual(positive["sigmoid"], ONE)
        self.assertEqual(positive["output"], 0x7F7F)
        self.assertEqual(positive["diagnostics"], 0)

    def test_signed_zero_survives_silu_and_up_multiply(self):
        self.assertEqual(canonical_trace(0x8000, ONE)["output"], 0x8000)
        self.assertEqual(canonical_trace(0x8000, 0xBF80)["output"], 0)
        self.assertEqual(canonical_trace(0, 0xBF80)["output"], 0x8000)

    def test_nonfinite_input_and_overflow_use_nan_with_sticky_diagnostic(self):
        for gate, up in [
            (0x7F80, ONE),
            (ONE, 0xFF80),
            (QNAN, ONE),
            (ONE, QNAN),
            (0x7F7F, bf16(2)),
        ]:
            trace = canonical_trace(gate, up)
            self.assertEqual(trace["output"], QNAN)
            self.assertEqual(trace["diagnostics"], 4)


class CanonicalSourceContractTests(unittest.TestCase):
    def test_cached_whole_k_descriptor_and_layout(self):
        source = (
            ROOT / "runtime/metal/kernels/shared/flash_dense_cache.metal"
        ).read_text()
        begin = source.index("inline void flash_dense_cache_tile(")
        end = source.index("#define FLASH_DENSE_CACHE_ENTRY", begin)
        tile = source[begin:end]
        self.assertIn("array<int, 2>{1, k}", tile)
        self.assertIn(
            "M, N, static_cast<int>(dynamic_extent), false, true, false", tile
        )
        self.assertIn("matmul2d_descriptor::mode::multiply", tile)
        self.assertIn("execution_simdgroups<4>", tile)
        self.assertIn("decltype(a), decltype(b), float", tile)
        self.assertIn("const bfloat value = bfloat(dot[i]);", tile)
        self.assertNotIn("multiply_accumulate", tile)

    def test_canonical_shared_swiglu_uses_compiled_sigmoid(self):
        source = (ROOT / "runtime/metal/kernels/shared/flash_moe.metal").read_text()
        begin = source.index("kernel void flash_moe_silu_multiply(")
        end = source.index("kernel void flash_moe_combine(", begin)
        activation = source[begin:end]
        self.assertIn("flash_moe_sigmoid_compiled(gate[index])", activation)
        self.assertNotIn("flash_moe_sigmoid_unary", activation)
        self.assertIn("const bfloat silu = gate[index] * sigmoid;", activation)
        self.assertIn("const bfloat result = silu * up[index];", activation)
        self.assertIn("bfloat(flash_moe_nan())", activation)

    def test_canonical_sigmoid_bf16_literals_and_operator_precision_distinction(self):
        source = (ROOT / "runtime/metal/kernels/shared/flash_moe.metal").read_text()
        start = source.index("inline bfloat flash_moe_sigmoid_compiled(")
        unary = source.index("inline bfloat flash_moe_sigmoid_unary(", start)
        compiled = source[start:unary]
        self.assertIn("bfloat(metal::exp(metal::abs(float(source))))", compiled)
        self.assertIn("bfloat(1.0f) + exponent", compiled)
        self.assertIn("bfloat(1.0f) / denominator", compiled)
        self.assertIn("bfloat(1.0f) - tail", compiled)
        self.assertIn("#pragma clang fp contract(off)", source[:start])
        self.assertIn("#pragma clang fp reassociate(off)", source[:start])
        self.assertIn("metal::precise::exp", source[unary:])

    def test_actual_prefill_shared_gate_up_selects_m16n64(self):
        # The route's N640 does not satisfy the N>=1024 branch for M32N128.
        projection = (ROOT / "runtime/flash/FlashForward.cpp").read_text()
        self.assertIn("rows >= 128 && projection.outputSize >= 1024", projection)
        self.assertIn(
            "? FlashAffineMPPTile::M32N128 : FlashAffineMPPTile::M16N64", projection
        )
        for rows in [256, 512, 1024, 2048, 4096, 8192]:
            self.assertEqual(rows % 16, 0)
            self.assertEqual((640 // 64) * (rows // 16), 10 * (rows // 16))

    def test_shared_weight_roles_and_byte_extents(self):
        rows, width, hidden = 256, 2560, 640
        self.assertEqual(rows * width * 2, 1310720)  # Direct device A.
        self.assertEqual(hidden * width * 2, 3276800)  # Each immutable cached B.
        self.assertEqual(rows * hidden * 2, 327680)  # Activated Y, distinct from A/B.
        self.assertEqual(2560 * 640 * 2, 3276800)  # Separate shared down B.
        host = (ROOT / "runtime/flash/FlashDenseCache.cpp").read_text()
        self.assertIn("if (overlaps(input, output)", host)
        self.assertIn("overlaps(b, weight.buffer)", host)
        self.assertIn("overlaps(diagnostics, input)", host)
        self.assertIn("overlaps(diagnostics, output)", host)


class CandidateSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shader = (
            ROOT / "dev/benchmarks/flash_shared_expert_fused.metal"
        ).read_text()
        cls.host = (ROOT / "dev/benchmarks/FlashSharedExpertFused.cpp").read_text()

    def test_whole_k_descriptor_matches_m16n64_control(self):
        shader = self.shader
        self.assertIn(
            "M, N,\n      static_cast<int>(dynamic_extent), false, true, false", shader
        )
        self.assertIn("matmul2d_descriptor::mode::multiply", shader)
        self.assertIn("execution_simdgroups<4>", shader)
        self.assertNotIn("multiply_accumulate", shader)
        self.assertIn("flash_shared_expert_fused_m16_n64, 16, 64", shader)
        self.assertIn("flash_shared_expert_fused_taps_m16_n64, 16, 64", shader)
        self.assertNotIn("threadgroup bfloat", shader)

    def test_gate_up_have_distinct_f32_fragments_and_direct_shared_a(self):
        shader = self.shader
        self.assertIn("decltype(a), decltype(gate), float", shader)
        self.assertIn("decltype(a), decltype(up), float", shader)
        self.assertIn("operation.run(a, gate, gateDot);", shader)
        self.assertIn("operation.run(a, up, upDot);", shader)
        self.assertIn("tensor(input + ulong(row) * 2560", shader)
        self.assertIn("tensor(gateWeights + ulong(column) * 2560", shader)
        self.assertIn("tensor(upWeights + ulong(column) * 2560", shader)
        self.assertEqual(shader.count("array<int, 2>{1, k}"), 3)

    def test_fast_exp_helper_has_explicit_bf16_boundaries_and_safe_mode_restored(self):
        shader = self.shader
        start = shader.index("inline bfloat flash_shared_fused_compiled_sigmoid(")
        stop = shader.index("template <ushort M, ushort N, bool Taps>", start)
        helper = shader[start:stop]
        self.assertGreater(
            shader.rfind("#pragma METAL fp math_mode(fast)", 0, start),
            shader.rfind("#pragma METAL fp math_mode(safe)", 0, start),
        )
        self.assertIn("#pragma clang fp contract(off)", helper)
        self.assertIn("#pragma clang fp reassociate(off)", helper)
        self.assertIn("bfloat(metal::fast::exp(metal::abs(float(source))))", helper)
        self.assertIn("bfloat(1.0f) + exponent", helper)
        self.assertIn("bfloat(1.0f) / denominator", helper)
        self.assertIn("bfloat(1.0f) - tail", helper)
        self.assertIn("#pragma METAL fp math_mode(safe)", helper)
        self.assertNotIn("metal::precise::exp", shader)

    def test_both_dot_fragments_round_before_sigmoid_and_products(self):
        shader = self.shader
        start = shader.index("inline void flash_shared_expert_fused_tile(")
        body = shader[start : shader.index("#define FLASH_SHARED_FUSED_ENTRY", start)]
        rounded = body.index(
            "const bfloat roundedGate = bfloat(gateDot[i]), roundedUp = bfloat(upDot[i]);"
        )
        sigmoid = body.index(
            "const bfloat sigmoid = flash_shared_fused_compiled_sigmoid(roundedGate);"
        )
        silu = body.index("const bfloat silu = roundedGate * sigmoid;")
        activated = body.index("const bfloat activated = silu * roundedUp;")
        self.assertLess(rounded, sigmoid)
        self.assertLess(sigmoid, silu)
        self.assertLess(silu, activated)

    def test_invalid_dot_operand_or_output_writes_canonical_nan(self):
        shader = self.shader
        for value in [
            "gateDot[i]",
            "upDot[i]",
            "roundedGate",
            "roundedUp",
            "activated",
        ]:
            self.assertIn(f"!flash_mpp_finite({value})", shader)
        self.assertIn("flash_mpp_error(diagnostics, 4u);", shader)
        self.assertIn(
            "output[destination] = bfloat(as_type<float>(0x7fc00000u));", shader
        )
        # Exp and denominator +inf are valid; diagnosing them would reject saturation.
        self.assertNotIn("!flash_mpp_finite(exponent)", shader)
        self.assertNotIn("!flash_mpp_finite(denominator)", shader)

    def test_host_rejects_weight_input_overlap_before_any_graph_mutation(self):
        host = self.host
        start = host.index("void validate(")
        validate = host[start : host.index("void complete(", start)]
        loop = re.search(
            r"for \(const auto &weight\s*:\s*\{gate\.buffer, up\.buffer\}\)\s*"
            r"for \(const auto &(\w+)\s*:\s*\{([^}]*)\}\)\s*"
            r"require\(!overlaps\(weight, \1\)",
            validate,
        )
        self.assertIsNotNone(loop, "immutable weight overlap validation is missing")
        checked = {word.strip() for word in loop.group(2).split(",")}
        self.assertEqual(checked, {"input", "activated", "diagnostics"})
        self.assertIn("!overlaps(activated, input)", validate)
        self.assertIn("!overlaps(diagnostics, input)", validate)
        self.assertIn("!overlaps(diagnostics, activated)", validate)
        self.assertNotIn("graph.add(", validate)

    def test_tail_scratch_is_validated_then_literal_baseline_fallback_runs(self):
        host = self.host
        start = host.index("void addSharedExpertFused(")
        body = host[start : host.index("void addSharedExpertFusedTaps(", start)]
        self.assertLess(body.index("validate(gate, up,"), body.index("complete(graph,"))
        self.assertLess(body.index("bytes(tail.gate,"), body.index("complete(graph,"))
        self.assertLess(body.index("bytes(tail.up,"), body.index("complete(graph,"))
        self.assertLess(
            body.index("!overlaps(tail.gate, tail.up)"), body.index("complete(graph,")
        )
        self.assertIn("{input, activated, diagnostics, gate.buffer, up.buffer}", body)
        self.assertIn(
            "addDenseBF16WholeK(backend, graph, tailInput, gate, tail.gate", body
        )
        self.assertIn("addDenseBF16WholeK(backend, graph, tailInput, up, tail.up", body)
        self.assertIn(
            "addSiLUMultiply(graph, tail.gate, tail.up, tailOutput, diagnostics, remaining, 640)",
            body,
        )

    def test_shader_and_host_use_same_prefill_scope_and_fixed_dimensions(self):
        self.assertIn("p.rows < 256", self.shader)
        self.assertIn("p.rows > 8192", self.shader)
        self.assertIn("p.input_size != 2560", self.shader)
        self.assertIn("p.output_size != 640", self.shader)
        self.assertIn("rows >= 256 && rows <= 8192", self.host)
        self.assertIn("std::vector<uint64_t>{640, 2560}", self.host)
        self.assertIn("value.logicalBytes == 640ULL * 2560 * 2", self.host)


if __name__ == "__main__":
    unittest.main(verbosity=2)
