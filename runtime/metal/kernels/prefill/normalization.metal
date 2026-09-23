#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/rms_inverse.h"

// W is the norm weights' stored type: bfloat in the packed formats, float for
// a GGUF's F32 norms (the _f32 entry point).
template <class W>
inline void norm_rms_sums32(device const bfloat *input, device const W *weight,
                            device bfloat *output, device float *sums,
                            uint width, uint row, uint thread_index, uint lane,
                            uint simd_group, threadgroup float *reductions) {
  constexpr uint TileM = 32;
  const float inverse_rms = rms_inverse(input + row * width, width, reductions,
                                        thread_index, lane, simd_group);
  const uint quant_groups = width / 64;
  const uint row_tile = row / TileM;
  const uint row_in_tile = row % TileM;
  for (uint quant_group = simd_group; quant_group < quant_groups;
       quant_group += 8) {
    uint origin = row * width + quant_group * 64 + lane;
    bfloat first = bfloat(float(input[origin]) * inverse_rms *
                          float(weight[quant_group * 64 + lane]));
    bfloat second = bfloat(float(input[origin + 32]) * inverse_rms *
                           float(weight[quant_group * 64 + lane + 32]));
    output[origin] = first;
    output[origin + 32] = second;
    float group_sum = simd_sum(float(first) + float(second));
    if (lane == 0) {
      sums[(ulong(row_tile) * quant_groups + quant_group) * TileM +
           row_in_tile] = group_sum;
    }
  }
}

#define PREFILL_NORM_RMS_SUMS32(Name, W)                                      \
  kernel void Name(device const bfloat *input [[buffer(0)]],                  \
                   device const W *weight [[buffer(1)]],                      \
                   device bfloat *output [[buffer(2)]],                       \
                   device float *sums [[buffer(3)]],                          \
                   constant uint &width [[buffer(4)]],                        \
                   uint row [[threadgroup_position_in_grid]],                 \
                   uint thread_index [[thread_index_in_threadgroup]],         \
                   uint lane [[thread_index_in_simdgroup]],                   \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {      \
    threadgroup float reductions[8];                                          \
    norm_rms_sums32(input, weight, output, sums, width, row, thread_index,    \
                    lane, simd_group, reductions);                            \
  }

PREFILL_NORM_RMS_SUMS32(prefill_norm_rms_sums32, bfloat)
PREFILL_NORM_RMS_SUMS32(prefill_norm_rms_sums32_f32, float)
#undef PREFILL_NORM_RMS_SUMS32
