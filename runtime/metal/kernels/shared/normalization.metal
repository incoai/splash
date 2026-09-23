#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/rms_inverse.h"
#include "metal/kernels/common/gguf_sgmatrix.h"

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

// Keep the ordinary output for non-matrix consumers, and emit the consumer's
// matrix operand table (Table: q4sg::Table64 affine, q16sg::Table16 GGUF) from
// the same rounded bfloat values. No additional dispatch is needed.
template <class Table>
inline void norm_rms_table(device const bfloat *input, device const bfloat *weight,
                           device bfloat *output, device bfloat *table, device float *sums,
                           uint width, uint row, uint tid, uint lane, uint sg,
                           threadgroup float *reductions) {
  const float inverse = rms_inverse(input + row * width, width, reductions, tid, lane, sg);
  for (uint g = sg; g < width / 64; g += 8) {
    const uint k = g * 64 + lane * 2;
    const bfloat a = bfloat(float(input[row * width + k]) * inverse * float(weight[k]));
    const bfloat b = bfloat(float(input[row * width + k + 1]) * inverse * float(weight[k + 1]));
    output[row * width + k] = a;
    output[row * width + k + 1] = b;
    Table::write(table + ulong(row / 8) * width * 8,
                 sums + ulong(row / 8) * Table::sums_per_tile(width), width, g, row % 8, lane, a, b);
  }
}
#define NORM_RMS_TABLE(Name, Table) \
  kernel void Name(device const bfloat *input [[buffer(0)]], \
      device const bfloat *weight [[buffer(1)]], device bfloat *output [[buffer(2)]], \
      device bfloat *table [[buffer(3)]], device float *sums [[buffer(4)]], \
      constant uint &width [[buffer(5)]], uint row [[threadgroup_position_in_grid]], \
      uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
      uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float reductions[8]; \
    norm_rms_table<Table>(input, weight, output, table, sums, width, row, tid, lane, sg, reductions); \
  }
NORM_RMS_TABLE(norm_rms_q4_decode, q4sg::Table64)
NORM_RMS_TABLE(norm_rms_q16_decode, q16sg::Table16)
#undef NORM_RMS_TABLE
