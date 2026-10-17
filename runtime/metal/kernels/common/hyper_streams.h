#pragma once

#include "metal/abi/HyperConnection.h"

#include <metal_stdlib>

using namespace metal;

// The per-row kernels over a row's hyper-connected streams
// (shared/hyper_connection.metal, shared/ngram_embedding.metal): a
// threadgroup of kRowThreads threads per row, of rows of up to kStreams
// streams.
constant constexpr uint kRowThreads = HYPER_CONNECTION_ROW_THREADS;
constant constexpr uint kSimdgroups = kRowThreads / 32;
constant constexpr uint kStreams = HYPER_CONNECTION_MAXIMUM_STREAMS;

// The sums of `values` (one per stream) over the threadgroup, in every thread.
inline void reduce_streams(thread float (&values)[kStreams], uint streams, threadgroup float *scratch,
                           uint lane, uint sg) {
  for (uint s = 0; s < streams; ++s) {
    const float sum = simd_sum(values[s]);
    if (lane == 0) scratch[sg * kStreams + s] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 0; s < streams; ++s) {
    float total = 0.0f;
    for (uint g = 0; g < kSimdgroups; ++g) total += scratch[g * kStreams + s];
    values[s] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
