#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/gguf_sgmatrix.h"

// The attention's output of one fused row of a reduce, as the layer's
// out-projection reads it: element `dimension` of query head `query_head` of
// the row, `value` (the reduce's, in bfloat) times the sigmoid of the head's
// gate in its packed row `packed` where the layer is Gated (a projection row
// holds each query head's queries, then its gate), and as it is where it is
// not, which reads no packed row (`packed_qkv` may be null). Written, where
// `writes`, at row `output` of `hidden` (QHeads x HeadDim elements a row)
// and, with Table, of the out-projection's Table16 input and its sums too: a
// thread of the first HeadDim / 2 a pair of the row's elements, whole
// simdgroups a 64-element span, from `pairs`, which holds every thread's
// element. `writes` is the threadgroup's, which every thread reaches.
template <uint QHeads, uint KHeads, uint HeadDim, bool Gated, bool Table>
inline void splash_attention_output_row(bfloat value, device const bfloat *packed_qkv, device bfloat *hidden,
                                        device bfloat *table, device float *sums, ulong packed, ulong output,
                                        uint query_head, uint dimension, bool writes, threadgroup bfloat *pairs) {
  constexpr uint Width = QHeads * HeadDim;
  static_assert(HeadDim % 64 == 0, "a head's elements fill whole simdgroups of pairs");
  bfloat element = value;
  if constexpr (Gated) {
    constexpr uint PackedStride = QHeads * 2 * HeadDim + 2 * KHeads * HeadDim;
    if (writes) {
      const float gate = float(packed_qkv[packed * PackedStride + query_head * 2 * HeadDim + HeadDim + dimension]);
      element = bfloat(float(value) * splash_sigmoid(gate));
    }
  }
  if (writes)
    hidden[output * Width + query_head * HeadDim + dimension] = element;
  if constexpr (Table) {
    pairs[dimension] = element;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (writes && dimension < HeadDim / 2) {
      const uint first = query_head * HeadDim + 2 * dimension;
      gguf_sg::Table16::write(table + output / 8 * Width * 8,
                              sums + output / 8 * gguf_sg::Table16::sums_per_tile(Width), Width, first / 64,
                              output % 8, dimension % 32, pairs[2 * dimension], pairs[2 * dimension + 1]);
    }
  }
}
