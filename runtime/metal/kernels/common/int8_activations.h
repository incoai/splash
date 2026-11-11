#pragma once
#include <metal_stdlib>
using namespace metal;

// The int8-activation quantizers' rule, shared by gguf_rotate_int8 (a row's rotation block),
// gguf_prefill_int8_quantize (a row's 64-input group) and the attention tiles' INT8 queries (a fused row), each of
// which reduces its block's largest magnitude m itself:
// - an input x codes as q = clamp(rint(x * (127 / m)), -127, 127), ties to even, and stands for q * s, s = m / 127;
//   both quotients are correctly rounded (precise::divide, where fast math's divide is within 2.5 ulp), so a host
//   reference computes the same codes and scales bit for bit;
// - a block whose m is 2^-100 or less, zero among them, codes as zeros with scale 0: its products, 2^-100 times the
//   weights at most, vanish beside those of any input of normal size, and 127 / m stays finite (it overflows below
//   2^-121);
// - a block holding an infinity or a NaN, whose codes alone would round to finite values, takes a NaN scale, made
//   from its bits so that fast math cannot fold it, and its row goes on in NaN as it would in bf16: a projection's
//   outputs are not finite; an attention row's scores and softmax statistics are NaN, which the reduce writes as a
//   row of zeros.
struct Int8Block {
  float scale;    // s
  float inverse;  // 127 / m
};
inline Int8Block int8_activation_block(float largest) {
  const bool coded = largest > 0x1p-100f;
  return {coded ? precise::divide(largest, 127.0f) : 0.0f, coded ? precise::divide(127.0f, largest) : 0.0f};
}
// The code of input x, or of four, in a block of inverse `inverse`.
inline int int8_activation_code(float x, float inverse) { return int(clamp(rint(x * inverse), -127.0f, 127.0f)); }
inline int4 int8_activation_code(float4 x, float inverse) { return int4(clamp(rint(x * inverse), -127.0f, 127.0f)); }
// A block's value, its scale or a product of it, as stored: its bits, or a NaN's where the block is not finite.
inline uint int8_activation_bits(float value, bool finite) { return finite ? as_type<uint>(value) : 0x7fc00000u; }
// A bf16 input is found non-finite by its exponent bits: fast math may fold isfinite() to true.
inline bool int8_activation_not_finite(bfloat value) { return (as_type<ushort>(value) & 0x7f80) == 0x7f80; }
inline bool int8_activation_not_finite(bfloat4 value) {
  return any((as_type<ushort4>(value) & ushort(0x7f80)) == ushort(0x7f80));
}
