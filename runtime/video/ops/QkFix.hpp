#pragma once

// QKFIX blocks (kernels/dit/qkfix.metal): fp16 normalised x and the fp16 q / k / v GEMM with an fp32 QK-norm / RoPE
// epilogue. Their attention is ops::windowQkFix.

#include "video/metal/Encoder.hpp"
#include "video/ops/Linear.hpp"

#include <string>
#include <vector>

namespace splash::video::ops {

// RMSNorm + adaLN modulate -> fp16 x [M][5376]
void qkfixNormalize(Encoder &e, BufferView x, int32_t rows, BufferView normWeight, BufferView ada, BufferView tags,
                    BufferView out);

struct QkFixTargets {
  QkvTargets qkv;               // fp16 Q / K (split dims zeroed), V, raw q / k
  BufferView qExact, kExact;    // fp32 [56][Spad][4]
};
void qkfixQkv(Encoder &e, BufferView in, int32_t rows, BufferView weight, BufferView splitDims, BufferView qNorm,
              BufferView kNorm, BufferView rope, const QkFixTargets &out);

} // namespace splash::video::ops
