#include "metal/abi/DSpark.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/rms_inverse.h"

// One 256-thread group per row (DSparkMixParams): the softmax-weighted sum of
// the row's SPLASH_DSPARK_TAPS bf16 features in fp32, RMS-normalized and
// multiplied by the consumer's F32 gamma, into a bf16 row. A thread keeps
// columns thread_index + 256 i of the mix in registers; rows up to
// 256 * kMixColumns values wide.
constant constexpr uint kMixColumns = 16;

kernel void dspark_mix_rms(device const bfloat *features [[buffer(0)]],
                           device const float *gamma [[buffer(1)]],
                           device bfloat *output [[buffer(2)]],
                           constant DSparkMixParams &params [[buffer(3)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint thread_index [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (row >= params.rows)
    return;
  const uint width = params.width;
  device const bfloat *taps = features + ulong(row) * SPLASH_DSPARK_TAPS * width;
  float mixed[kMixColumns];
  float squares = 0.0f;
  for (uint i = 0; i < kMixColumns; ++i) {
    const uint column = thread_index + 256 * i;
    float value = 0.0f;
    if (column < width)
      for (uint tap = 0; tap < SPLASH_DSPARK_TAPS; ++tap)
        value += params.alpha[tap] * float(taps[tap * width + column]);
    mixed[i] = value;
    squares += value * value;
  }
  threadgroup float reductions[8];
  const float inverse = rms_inverse_of_sums(squares, width, reductions, thread_index, lane, simd_group);
  device bfloat *destination = output + ulong(row) * width;
  for (uint i = 0; i < kMixColumns; ++i) {
    const uint column = thread_index + 256 * i;
    if (column < width)
      destination[column] = bfloat(mixed[i] * inverse * gamma[column]);
  }
}
