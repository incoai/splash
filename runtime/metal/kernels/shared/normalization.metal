#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/rms_inverse.h"
#include "metal/kernels/common/q4_sgmatrix.h"

kernel void norm_rms(device const bfloat *input [[buffer(0)]],
                        device const bfloat *weight [[buffer(1)]],
                        device bfloat *output [[buffer(2)]],
                        constant uint &width [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint thread_index [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float reductions[8];
  float inverse_rms = rms_inverse(input + row * width, width, reductions,
                                  thread_index, lane, simd_group);
  for (uint column = thread_index; column < width; column += 256) {
    output[row * width + column] =
        bfloat(float(input[row * width + column]) * inverse_rms *
               float(weight[column]));
  }
}

// Keep the ordinary output for non-Q4 consumers, and emit the matrix operand
// layout from the same rounded bfloat values. No additional dispatch is needed.
kernel void norm_rms_q4_decode(device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]], device bfloat *output [[buffer(2)]],
    device bfloat *table [[buffer(3)]], device float *sums [[buffer(4)]],
    constant uint &width [[buffer(5)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float reductions[8];
  const float inverse = rms_inverse(input + row * width, width, reductions, tid, lane, sg);
  for (uint g = sg; g < width / 64; g += 8) {
    const uint k = g * 64 + lane * 2;
    const bfloat a = bfloat(float(input[row * width + k]) * inverse * float(weight[k]));
    const bfloat b = bfloat(float(input[row * width + k + 1]) * inverse * float(weight[k + 1]));
    output[row * width + k] = a;
    output[row * width + k + 1] = b;
    q4sg::write_input(table + ulong(row / 8) * width * 8,
                       sums + ulong(row / 8) * width / 8, g, row % 8, lane, a, b);
  }
}

// One threadgroup per row stages the row in threadgroup memory so the apply
// pass reads it back instead of loading it from device memory a second time.
// Threads 0..255 repeat the column partition and the eight-partial summation
// order of rms_inverse: the reduction order is the one thing this kernel may
// not change, because every output byte has to match norm_rms. The group is
// 1024 threads, the widest every Apple GPU family 9 or later admits, so the
// load and the store stream the row with as many threads as possible. The
// staging array bounds the row width the kernel can take; at 5120 bfloat it
// holds 10 KiB of the 32 KiB a threadgroup has.
static_assert(SPLASH_STAGED_NORM_WIDTH % 4u == 0u,
              "the staged row is loaded and stored as bfloat4");
static_assert(SPLASH_STAGED_NORM_WIDTH * 2u <= 16384u,
              "the staged row leaves half of threadgroup memory free");
static_assert(SPLASH_STAGED_NORM_THREADS % 256u == 0u,
              "threads 0..255 must form complete simdgroups for the reduction");
kernel void norm_rms_staged(device const bfloat4 *input [[buffer(0)]],
                            device const bfloat4 *weight [[buffer(1)]],
                            device bfloat4 *output [[buffer(2)]],
                            constant uint &width [[buffer(3)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint thread_index [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup bfloat4 stage[SPLASH_STAGED_NORM_WIDTH / 4u];
  threadgroup float reductions[8];
  const uint vectors = width / 4;
  device const bfloat4 *row_input = input + row * vectors;
  for (uint column = thread_index; column < vectors;
       column += SPLASH_STAGED_NORM_THREADS)
    stage[column] = row_input[column];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup const bfloat *values = (threadgroup const bfloat *)stage;
  if (thread_index < 256) {
    float sum = 0.0f;
    for (uint column = thread_index; column < width; column += 256) {
      const float value = float(values[column]);
      sum += value * value;
    }
    sum = simd_sum(sum);
    if (lane == 0)
      reductions[simd_group] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i)
      total += reductions[i];
    reductions[0] = rsqrt(total / width + 1e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inverse = reductions[0];
  for (uint column = thread_index; column < vectors;
       column += SPLASH_STAGED_NORM_THREADS)
    output[row * vectors + column] =
        bfloat4(float4(stage[column]) * inverse * float4(weight[column]));
}
