#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/gguf_sgmatrix.h"

// Where a reduce finds its rows' output gates, by its geometry's projection
// rows: none (SplashNoGates); each query head's gates after its queries in
// the row's packed projection row (Qwen3.5's, bfloat, SplashPackedGates); or
// every query head's after all their queries in the row's query side row (a
// relative geometry's, F32, metal/abi/PagedAttention.h), which holds the
// row's sinks too (SplashQuerySideGates, RowSinks). `index` is a row's gate
// of a query head's dimension, `sink` its sink.
struct SplashNoGates {
  using Element = bfloat;
  static constexpr constant bool Gated = false, RowSinks = false;
};

template <uint QHeads, uint KHeads, uint HeadDim> struct SplashPackedGates {
  using Element = bfloat;
  static constexpr constant bool Gated = true, RowSinks = false;
  static ulong index(ulong packed, uint query_head, uint dimension) {
    constexpr uint PackedStride = QHeads * 2 * HeadDim + 2 * KHeads * HeadDim;
    return packed * PackedStride + query_head * 2 * HeadDim + HeadDim + dimension;
  }
};

template <uint QHeads, uint HeadDim> struct SplashQuerySideGates {
  using Element = float;
  static constexpr constant bool Gated = true, RowSinks = true;
  static ulong index(ulong packed, uint query_head, uint dimension) {
    return packed * SPLASH_RELATIVE_QUERY_SIDE(QHeads, HeadDim) + SPLASH_RELATIVE_GATES(QHeads, HeadDim) +
           query_head * HeadDim + dimension;
  }
  static ulong sink(ulong packed, uint query_head) {
    return packed * SPLASH_RELATIVE_QUERY_SIDE(QHeads, HeadDim) + SPLASH_RELATIVE_SINKS(QHeads, HeadDim) + query_head;
  }
};

// The gates of a geometry gated (Qwen3.5's packed rows) or not.
template <bool Gated, uint QHeads, uint KHeads, uint HeadDim>
using SplashGatesOf = conditional_t<Gated, SplashPackedGates<QHeads, KHeads, HeadDim>, SplashNoGates>;

// The attention's output of one fused row of a reduce, as the layer's
// out-projection reads it: element `dimension` of query head `query_head` of
// the row, `value` (the reduce's, in bfloat) times the sigmoid of the head's
// gate in its projection row `packed` (Gates) where the layer is gated, and
// as it is where it is not, which reads no projection row (`gates` may be
// null). Written, where `writes`, at row `output` of `hidden` (QHeads x
// HeadDim elements a row) and, with Table, of the out-projection's Table16
// input and its sums too: a thread of the first HeadDim / 2 a pair of the
// row's elements, whole simdgroups a 64-element span, from `pairs`, which
// holds every thread's element. `writes` is the threadgroup's, which every
// thread reaches.
template <uint QHeads, uint KHeads, uint HeadDim, class Gates, bool Table>
inline void splash_attention_output_row(bfloat value, device const typename Gates::Element *gates,
                                        device bfloat *hidden, device bfloat *table, device float *sums,
                                        ulong packed, ulong output, uint query_head, uint dimension, bool writes,
                                        threadgroup bfloat *pairs) {
  constexpr uint Width = QHeads * HeadDim;
  static_assert(HeadDim % 64 == 0, "a head's elements fill whole simdgroups of pairs");
  bfloat element = value;
  if constexpr (Gates::Gated) {
    if (writes) {
      const float gate = float(gates[Gates::index(packed, query_head, dimension)]);
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
