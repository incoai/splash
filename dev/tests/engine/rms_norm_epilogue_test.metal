#include "metal/abi/Normalization.h"
#include "metal/kernels/common/rms_norm_epilogue.h"

#include <metal_stdlib>

using namespace metal;

// A producer of the RMS norm epilogue (kernels/common/rms_norm_epilogue.h),
// for its test: tile t of 256 threads stores columns 32 t to 32 t + 31 of
// every row, bf16(a + b) of two bf16 inputs, as a projection's tile stores
// its rows with a residual: a simdgroup's rows s, s + 8, ... (the matrix
// tile's), lane the column, each row's sum of squares by simd_sum. With
// `simdgroup`, simdgroup 0 alone stores every row and finishes, the others
// returning (the column tile's).
struct RmsNormEpilogueTestParams {
  struct RmsNormEpilogueParams norm;
  uint32_t simdgroup;
};

static_assert(sizeof(RmsNormEpilogueTestParams) == 20, "the test's parameters are 20 bytes on both sides");

kernel void rms_norm_epilogue_test(device const bfloat *a [[buffer(0)]], device const bfloat *b [[buffer(1)]],
                                   device coherent(device) bfloat *values [[buffer(2)]],
                                   device coherent(device) float *partials [[buffer(3)]],
                                   device atomic_uint *counter [[buffer(4)]],
                                   device const uchar *weights [[buffer(5)]], device bfloat *output [[buffer(6)]],
                                   constant RmsNormEpilogueTestParams &params [[buffer(7)]],
                                   uint tile [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]]) {
  constant RmsNormEpilogueParams &p = params.norm;
  threadgroup uint arrival;
  threadgroup float inverses[RMS_NORM_EPILOGUE_MAXIMUM_ROWS];
  const RmsNormEpilogue e{partials, counter, weights, output};
  const uint simdgroups = params.simdgroup ? 1u : 8u;
  if (sg >= simdgroups) return;
  for (uint row = sg; row < p.rows; row += simdgroups) {
    const ulong i = ulong(row) * p.width + tile * 32 + lane;
    const bfloat value = bfloat(float(a[i]) + float(b[i]));
    values[i] = value;
    const float squares = simd_sum(float(value) * float(value));
    if (lane == 0) rms_norm_epilogue_partial(e, p, tile, row, squares);
  }
  if (params.simdgroup)
    rms_norm_epilogue_finish_simdgroup(e, p, values, p.width, lane, inverses);
  else
    rms_norm_epilogue_finish<256>(e, p, values, p.width, tid, lane, sg, &arrival, inverses);
}
