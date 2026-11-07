#pragma once
#include <metal_stdlib>
using namespace metal;

// The int8-activation projection quantizers' rule (gguf_rotate_int8,
// gguf_prefill_int8_quantize) for non-finite inputs: a block or group of
// inputs holding an infinity or a NaN, whose codes alone would round to finite
// values, takes NaN scales, stored as bits, which fast math cannot fold, so
// that its row's outputs are not finite, as the bf16 tiles' are. A bf16 input
// is found non-finite by its exponent bits: fast math may fold isfinite() to
// true.
inline bool int8_activation_not_finite(bfloat value) { return (as_type<ushort>(value) & 0x7f80) == 0x7f80; }
