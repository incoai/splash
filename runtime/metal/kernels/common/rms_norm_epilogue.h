#pragma once

#include "metal/abi/Normalization.h"
#include "metal/kernels/common/rms_inverse.h"
#include "metal/kernels/common/split_reduce.h"

#include <metal_stdlib>

// A producer's RMS norm epilogue (metal/abi/Normalization.h,
// ops::RmsNormEpilogue): a dispatch whose output tiles (threadgroups, or a K
// split's storing ones) each store some columns of every one of its rows,
// bf16 as the next layer reads them, writes the rows' weighted RMS norm in
// the tile that finishes last, so no norm dispatch follows it:
// 1. each tile, after its stores, writes its columns' sum of squares of each
//    row, of the bf16 values it stored, to partials [tiles][rows]
//    (rms_norm_epilogue_partial, a thread a row);
// 2. every thread of the tile then runs rms_norm_epilogue_finish: the tile
//    that arrives last at the dispatch's counter (split_arrive_last) adds each
//    row's partials, a simdgroup a row, takes its inverse RMS, writes every
//    value times it and the norm's weight, rounded to bf16 as
//    Normalization::addRms writes them, and returns the counter to zero.
// The producer stores its rows and the partials through device
// coherent(device) pointers, which the last tile reads. A tile of which one
// simdgroup alone stores and arrives runs rms_norm_epilogue_finish_simdgroup.
struct RmsNormEpilogue {
  device coherent(device) float *partials;
  device atomic_uint *counter;
  // The norm's weights, F32 or bf16 (RmsNormEpilogueParams::weights_f32).
  device const uchar *weights;
  device bfloat *output;
};

// Tile `tile`'s sum of squares of its columns of row `row`.
inline void rms_norm_epilogue_partial(RmsNormEpilogue e, constant RmsNormEpilogueParams &p, uint tile, uint row,
                                      float squares) {
  e.partials[ulong(tile) * p.rows + row] = squares;
}

// The last tile's part, by `Threads` threads (`stride` values from one of
// the producer's rows to the next): each row's inverse RMS into
// `inverses` (RMS_NORM_EPILOGUE_MAXIMUM_ROWS floats), then the rows.
template <uint Threads>
inline void rms_norm_epilogue_rows(RmsNormEpilogue e, constant RmsNormEpilogueParams &p,
                                   device coherent(device) const bfloat *values, uint stride, uint tid, uint lane,
                                   uint sg, threadgroup float *inverses) {
#pragma clang fp reassociate(off)
  constexpr uint kSimdgroups = Threads / 32;
  for (uint row = sg; row < p.rows; row += kSimdgroups) {
    float sum = 0.0f;
    for (uint tile = lane; tile < p.tiles; tile += 32) sum += e.partials[ulong(tile) * p.rows + row];
    sum = simd_sum(sum);
    if (lane == 0) inverses[row] = rsqrt(sum / p.width + kRmsEpsilon);
  }
  if (Threads > 32)
    threadgroup_barrier(mem_flags::mem_threadgroup);
  else
    simdgroup_barrier(mem_flags::mem_threadgroup);
  const uint quads = p.width / 4;
  for (uint i = tid; i < p.rows * quads; i += Threads) {
    const uint row = i / quads, quad = i % quads;
    const float4 x = float4(*(device coherent(device) const bfloat4 *)(values + ulong(row) * stride + 4 * quad));
    const float4 w = p.weights_f32 ? float4(((device const packed_float4 *)e.weights)[quad])
                                   : float4(((device const bfloat4 *)e.weights)[quad]);
    *(device bfloat4 *)(e.output + ulong(row) * p.width + 4 * quad) = bfloat4((x * inverses[row]) * w);
  }
}

// Step 2 by a tile's every thread (`Threads`, whole simdgroups).
template <uint Threads>
inline void rms_norm_epilogue_finish(RmsNormEpilogue e, constant RmsNormEpilogueParams &p,
                                     device coherent(device) const bfloat *values, uint stride, uint tid, uint lane,
                                     uint sg, threadgroup uint *arrival, threadgroup float *inverses) {
  if (!split_arrive_last(e.counter, p.tiles, tid, arrival)) return;
  rms_norm_epilogue_rows<Threads>(e, p, values, stride, tid, lane, sg, inverses);
  split_release(e.counter, tid);
}

// Step 2 by the one simdgroup of a tile that stores and arrives, the others
// having returned (split_arrive_last_simdgroup).
inline void rms_norm_epilogue_finish_simdgroup(RmsNormEpilogue e, constant RmsNormEpilogueParams &p,
                                               device coherent(device) const bfloat *values, uint stride, uint lane,
                                               threadgroup float *inverses) {
  if (!split_arrive_last_simdgroup(e.counter, p.tiles, lane)) return;
  rms_norm_epilogue_rows<32>(e, p, values, stride, lane, lane, 0, inverses);
  split_release(e.counter, lane);
}
