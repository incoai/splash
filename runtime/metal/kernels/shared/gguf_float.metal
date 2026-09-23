// fp32 projections of the GGUF float tensors llama.cpp keeps unquantized: the
// MoE router (ffn_gate_inp) and, when a GGUF stores them as F32, the GDN
// alpha/beta gates (ops/Linear.hpp float segments). out[r][n] = sum_k x[r][k]
// W[n][k] on simdgroup float 8x8 MMA with exact operands (bf16 activations
// widen to fp32, the weights as stored), so a result differs from the fp64
// product only by fp32 accumulation. llama.cpp's mul_mm stages F32 weights as
// half above eight rows; no phase here rounds them.
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/q4_sgmatrix.h"

namespace gguf_float {

constant constexpr uint kFragments = 4;          // 8-row fragments per threadgroup: 32 rows
constant constexpr uint kRows = 8 * kFragments;
// K parts, one per simdgroup. A decode dispatch has few threadgroups (8 or 32 at the 35B's alpha/beta and router
// widths), so each simdgroup's chain of K / (8 kSplits) dependent loads and MMAs sets its time: on the 40-core M3
// Max four parts took 26-28 us per decode router or alpha/beta dispatch.
constant constexpr uint kSplits = 16;

// One threadgroup: output columns [8 tg.x, 8 tg.x + 8) of rows [32 tg.y, 32 tg.y + 32). Simdgroup s accumulates
// K part s of every fragment; simdgroup f then adds fragment f's parts in order, so a result depends neither on
// the grid nor on the row count. Lane (fm, fn) holds x[row fm][k + fn, k + fn + 1] as A,
// W[column fn, fn + 1][k + fm] as B and the destination [row fm][column fn, fn + 1] (q4sg::lane_map).
template <typename T>
inline void project(device const bfloat *input, device const float *weights, device T *output,
                    constant GgufFloatParams &p, uint2 tg, uint sg, uint lane, threadgroup float2 *parts) {
  const q4sg::Lane l = q4sg::lane_map(lane);
  const uint K = p.input_size, column = tg.x * 8 + l.fn, row0 = tg.y * kRows;
  const uint live = min(kRows, p.rows - row0);
  const uint steps = K / 8, first = sg * steps / kSplits, last = (sg + 1) * steps / kSplits;
  device const float *w = weights + ulong(column) * K + l.fm;
  float2 acc[kFragments];
#pragma unroll
  for (uint f = 0; f < kFragments; ++f) acc[f] = float2(0);
#pragma unroll(4)
  for (uint step = first; step < last; ++step) {
    const uint k = step * 8;
    const float2 b(w[k], w[K + k]);
#pragma unroll
    for (uint f = 0; f < kFragments; ++f) {
      if (f * 8 >= live) break;
      const uint row = f * 8 + l.fm;
      const float2 a = row < live
          ? float2(*reinterpret_cast<device const bfloat2 *>(input + ulong(row0 + row) * K + k + l.fn))
          : float2(0);
      q4sg::mma_acc<float>(acc[f], a, b);
    }
  }
#pragma unroll
  for (uint f = 0; f < kFragments; ++f) parts[(sg * kFragments + f) * 32 + lane] = acc[f];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint f = sg, row = f * 8 + l.fm;
  if (f >= kFragments || row >= live) return;
  float2 total = float2(0);
  for (uint s = 0; s < kSplits; ++s) total += parts[(s * kFragments + f) * 32 + lane];
  device T *out = output + ulong(row0 + row) * p.out_stride + p.out_offset + column;
  out[0] = T(total.x);
  out[1] = T(total.y);
}

} // namespace gguf_float

static_assert(gguf_float::kSplits >= gguf_float::kFragments, "simdgroup f reduces fragment f");

// Grid (output_size / 8, ceil(rows / 32)), 32 kSplits threads.
#define GGUF_FLOAT_KERNEL(Name, T)                                                                             \
  kernel void Name(device const bfloat *input [[buffer(0)]], device const float *weights [[buffer(1)]],       \
                   device T *output [[buffer(2)]], constant GgufFloatParams &p [[buffer(3)]],                 \
                   uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],     \
                   uint lane [[thread_index_in_simdgroup]]) {                                                 \
    threadgroup float2 parts[gguf_float::kSplits * gguf_float::kFragments * 32];                              \
    gguf_float::project<T>(input, weights, output, p, tg, sg, lane, parts);                                   \
  }
GGUF_FLOAT_KERNEL(gguf_float_bf16, bfloat)
GGUF_FLOAT_KERNEL(gguf_float_f32, float)
#undef GGUF_FLOAT_KERNEL
