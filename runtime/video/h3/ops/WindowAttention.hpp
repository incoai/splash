#pragma once

// Window-softmax attention of the DiT (kernels/dit/window_attention.metal, qkfix.metal): int8 preparation, the
// table-driven window kernel (int8 QK or the QKFIX fp16 QK + exact split term) and the key-chunk merge.
#import <Metal/Metal.h>

#include "video/h3/engine/WindowTable.hpp"
#include "video/h3/metal/MetalBackend.hpp"

#include <string>
#include <vector>

namespace h3::ops {

using metal::BufferView;

[[nodiscard]] std::vector<std::string> attentionKernels();

struct AttentionBuffers {
  BufferView qHalf, kHalf, v;     // fp16 [56][Spad][128]
  BufferView qInt8, kInt8;        // int8 [56][Spad][128]
  BufferView qScale, kScale;      // fp32 [56][Spad], [56][Spad / 32]
  BufferView keyMean, keyMeanPartial;
  BufferView partials, partialStats;  // key-chunk partials (only with chunked groups)
  int32_t rows, paddedRows;
};

// Key mean, Hadamard(128) int8 Q (per row) and K - mean (per 32-key block).
void quantize(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, const AttentionBuffers &a);
// int8 window attention -> bf16 O [M][7168] (+ combine of the key-chunk partials).
void window(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, const AttentionBuffers &a,
            const WindowTable &table, BufferView out);
// QKFIX: fp16 Q / K window attention with the fp32 split-dimension term.
void windowQkFix(id<MTLComputeCommandEncoder> e, const metal::MetalBackend &b, const AttentionBuffers &a,
                 BufferView qExact, BufferView kExact, const WindowTable &table, BufferView out);

} // namespace h3::ops
