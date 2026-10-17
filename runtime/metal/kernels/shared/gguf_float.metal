// fp32 projections of the GGUF float tensors llama.cpp keeps unquantized: the
// MoE router (ffn_gate_inp, F32, or BF16 as some GGUFs store it) and, when a
// GGUF stores them as F32, the GDN alpha/beta gates (ops/Linear.hpp float
// segments). out[r][n] = sum_k x[r][k] W[n][k] on simdgroup float 8x8 MMA with
// exact operands (bf16 activations widen to fp32, the weights as stored, BF16
// ones widened), so a result differs from the fp64 product only by fp32
// accumulation, and a BF16 tensor's equals that of the F32 tensor of its
// values bit for bit. llama.cpp's mul_mm stages F32 weights as half above
// eight rows; no phase here rounds them.
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/sgmatrix.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

namespace gguf_float {

constant constexpr uint kFragments = 4;          // 8-row fragments per threadgroup: 32 rows
constant constexpr uint kRows = 8 * kFragments;
// K parts, one per simdgroup. A decode dispatch has few threadgroups (8 or 32 at the 35B's alpha/beta and router
// widths), so each simdgroup's chain of K / (8 kSplits) dependent loads and MMAs sets its time.
constant constexpr uint kSplits = 16;

// One threadgroup: output columns [8 tg.x, 8 tg.x + 8) of rows [32 tg.y, 32 tg.y + 32). Simdgroup s accumulates
// K part s of every fragment; simdgroup f then adds fragment f's parts in order, so a result depends neither on
// the grid nor on the row count. Lane (fm, fn) holds x[row fm][k + fn, k + fn + 1] as A,
// W[column fn, fn + 1][k + fm] as B and the destination [row fm][column fn, fn + 1] (sgmatrix::lane_map).
template <typename W, typename T>
inline void project(device const bfloat *input, device const W *weights, device T *output,
                    constant GgufFloatParams &p, uint2 tg, uint sg, uint lane, threadgroup float2 *parts) {
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint K = p.input_size, column = tg.x * 8 + l.fn, row0 = tg.y * kRows;
  const uint live = min(kRows, p.rows - row0);
  const uint steps = K / 8, first = sg * steps / kSplits, last = (sg + 1) * steps / kSplits;
  device const W *w = weights + ulong(column) * K + l.fm;
  float2 acc[kFragments];
#pragma unroll
  for (uint f = 0; f < kFragments; ++f) acc[f] = float2(0);
#pragma unroll(4)
  for (uint step = first; step < last; ++step) {
    const uint k = step * 8;
    const float2 b = float2(float(w[k]), float(w[K + k]));
#pragma unroll
    for (uint f = 0; f < kFragments; ++f) {
      if (f * 8 >= live) break;
      const uint row = f * 8 + l.fm;
      const float2 a = row < live
          ? float2(*reinterpret_cast<device const bfloat2 *>(input + ulong(row0 + row) * K + k + l.fn))
          : float2(0);
      sgmatrix::mma_acc<float>(acc[f], a, b);
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

// Grid (output_size / 8, ceil(rows / 32)), 32 kSplits threads; named by the weights' and the output's types.
#define GGUF_FLOAT_KERNEL(Name, W, T)                                                                          \
  kernel void Name(device const bfloat *input [[buffer(0)]], device const W *weights [[buffer(1)]],           \
                   device T *output [[buffer(2)]], constant GgufFloatParams &p [[buffer(3)]],                 \
                   uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],     \
                   uint lane [[thread_index_in_simdgroup]]) {                                                 \
    threadgroup float2 parts[gguf_float::kSplits * gguf_float::kFragments * 32];                              \
    gguf_float::project<W, T>(input, weights, output, p, tg, sg, lane, parts);                                \
  }
GGUF_FLOAT_KERNEL(gguf_float_f32_bf16, float, bfloat)
GGUF_FLOAT_KERNEL(gguf_float_f32_f32, float, float)
GGUF_FLOAT_KERNEL(gguf_float_bf16_bf16, bfloat, bfloat)
GGUF_FLOAT_KERNEL(gguf_float_bf16_f32, bfloat, float)
#undef GGUF_FLOAT_KERNEL

// The same projections on the neural accelerator (ops::FloatTile::NeuralAccelerator), whose matmul takes bf16 operands:
// each F32 weight is split into three bf16 parts, its leading 8 significant bits, the next 8 and the last 8, so
// hi + mid + lo == w exactly (fp32 has 24) and each part's product with a bf16 activation is exact in fp32; a BF16
// weight is its own one part (the F32 tensor of its values splits into it and two zero parts, whose products add
// zeros, so the two give the same bits). The results again differ from the fp64 products only by fp32 accumulation.
// The accelerator adds a matmul's products to a destination with a few ulps of error (a probe: products
// 1 + 31 x 0.75 ulp(1) added to 1 + 23 ulp(1) came out 3 ulp(2) above the rounded sum), so the hi part accumulates
// apart from the mid and lo parts: that cuts the error against fp64 2.2x (mean 1.5e-8 -> 6.6e-9 of sum |x w| at the
// router's K = 2048, 512 rows; this file's fp32 kernel: 5.5e-9) for 3-6% of the kernel's time on a 20-core M5 Pro.
namespace gguf_float_na {

constant constexpr ushort kRowsPerSimdgroup = 16;   // the accelerator's row unit
constant constexpr ushort kSimdgroups = 4;
constant constexpr ushort kRows = kRowsPerSimdgroup * kSimdgroups;
constant constexpr ushort kColumns = 32;
constant constexpr ushort kStep = 32;   // inputs per stage
constant constexpr ushort kPerThread = kColumns * kStep / (kSimdgroups * 32);
constant constexpr uint kPlane = kColumns * kStep;
// The bf16 parts of a weight of type W, and the stage of a K step's (two such stages alternate).
template <typename W> constant constexpr ushort kParts = is_same_v<W, float> ? 3 : 1;
template <typename W> constant constexpr uint kBuffer = kParts<W> * kPlane;

// The parts of w as bf16 bits: each truncates the remainder to its leading 8 significant bits, so every subtraction
// is exact (Sterbenz) and the last remainder has at most 8 significant bits left.
inline ushort3 split(float w) {
  const uint hi = as_type<uint>(w) & 0xFFFF0000u;
  const float r = w - as_type<float>(hi);
  const uint mid = as_type<uint>(r) & 0xFFFF0000u;
  const float lo = r - as_type<float>(mid);
  return ushort3(ushort(hi >> 16), ushort(mid >> 16), ushort(as_type<uint>(lo) >> 16));
}

// One threadgroup: kColumns output columns of kRows rows, kRowsPerSimdgroup per simdgroup. Every thread splits
// kPerThread weights of a K step into the stage's bf16 planes [part][column][input] (columns past the output stay
// zero); after the barrier each simdgroup with live rows runs a matmul per part, and one past the rows only stages. A
// simdgroup whose rows cross p.rows computes the 16 rows ending at p.rows instead and stores its own, so no row past
// the input is read (p.rows >= 16).
template <typename W, typename T>
inline void project(device bfloat *input, device const W *weights, device T *output,
                    constant GgufFloatParams &p, uint2 tg, ushort sg, ushort lane, threadgroup bfloat *stage) {
  constexpr ushort Parts = kParts<W>;
  constexpr uint Buffer = kBuffer<W>;
  typedef vec<W, 4> W4;
  const uint K = p.input_size, steps = K / kStep;
  const uint row0 = tg.y * kRows + sg * kRowsPerSimdgroup, first_row = min(row0, p.rows - kRowsPerSimdgroup);
  const uint col0 = tg.x * kColumns;
  const ushort t = sg * 32 + lane, column = t / (kStep / kPerThread), first = t % (kStep / kPerThread) * kPerThread;
  const bool live_column = col0 + column < p.output_size;
  device const W4 *w =
      reinterpret_cast<device const W4 *>(weights + ulong(live_column ? col0 + column : 0) * K + first);
  auto a = tensor(input + ulong(first_row) * K, dextents<int, 2>{int(K), kRowsPerSimdgroup}, array<int, 2>{1, int(K)});
  constexpr auto descriptor = matmul2d_descriptor(kRowsPerSimdgroup, kColumns, kStep, false, true, false,
                                                  matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  const auto plane = [&](uint buffer, uint part) {
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> b(
        stage + buffer * Buffer + part * kPlane, dextents<int, 2>{kStep, kColumns}, array<int, 2>{1, kStep});
    return b.template slice<kStep, kColumns>(0, 0);
  };
  // A weight of one part has no mid and lo planes: their views alias its one, and no matmul runs them. Its products
  // never reach `low`, which keeps the +0 that the F32 tensor of its values sums from its zero parts.
  constexpr uint mid = Parts == 3 ? 1 : 0, lo = Parts == 3 ? 2 : 0;
  auto h0 = plane(0, 0), m0 = plane(0, mid), l0 = plane(0, lo), h1 = plane(1, 0), m1 = plane(1, mid), l1 = plane(1, lo);
  auto a0 = a.template slice<kStep, kRowsPerSimdgroup>(0, 0);
  auto high = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(h0), float>();
  auto low = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(h0), float>();
#pragma unroll
  for (ushort i = 0; i < high.get_capacity(); ++i) high[i] = low[i] = 0.0f;
  W4 next[kPerThread / 4];
#pragma unroll
  for (ushort j = 0; j < kPerThread / 4; ++j) next[j] = live_column ? w[j] : W4(0);
  const bool owns_rows = row0 < p.rows;   // uniform per simdgroup
  for (uint step = 0; step < steps; ++step) {
    threadgroup ushort *parts =
        reinterpret_cast<threadgroup ushort *>(stage + (step & 1) * Buffer) + column * kStep + first;
#pragma unroll
    for (ushort j = 0; j < kPerThread / 4; ++j) {
      if constexpr (Parts == 1) {
        *reinterpret_cast<threadgroup W4 *>(parts + 4 * j) = next[j];
      } else {
        const ushort3 x = split(next[j].x), y = split(next[j].y), z = split(next[j].z), v = split(next[j].w);
#pragma unroll
        for (ushort part = 0; part < Parts; ++part)
          *reinterpret_cast<threadgroup ushort4 *>(parts + part * kPlane + 4 * j) =
              ushort4(x[part], y[part], z[part], v[part]);
      }
    }
    if (step + 1 < steps && live_column) {
#pragma unroll
      for (ushort j = 0; j < kPerThread / 4; ++j) next[j] = w[(step + 1) * (kStep / 4) + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (owns_rows) {
      auto slice = a.template slice<kStep, kRowsPerSimdgroup>(step * kStep, 0);
      if (step & 1) {
        operation.run(slice, h1, high);
        if constexpr (Parts == 3) {
          operation.run(slice, m1, low);
          operation.run(slice, l1, low);
        }
      } else {
        operation.run(slice, h0, high);
        if constexpr (Parts == 3) {
          operation.run(slice, m0, low);
          operation.run(slice, l0, low);
        }
      }
    }
  }
  if (!owns_rows) return;
#pragma unroll
  for (ushort i = 0; i < high.get_capacity(); ++i) {
    if (!high.is_valid_element(i)) continue;
    const auto index = high.get_multidimensional_index(i);
    const uint row = first_row + index[1], column_out = col0 + index[0];
    if (row >= row0 && column_out < p.output_size)
      output[ulong(row) * p.out_stride + p.out_offset + column_out] = T(high[i] + low[i]);
  }
}

} // namespace gguf_float_na

static_assert(gguf_float_na::kStep % gguf_float_na::kPerThread == 0 && gguf_float_na::kPerThread % 4 == 0,
              "a thread stages whole float4s of one column");

// Grid (ceil(output_size / 32), ceil(rows / 64)), 128 threads; rows >= 16, input_size a multiple of 32; named by the
// weights' and the output's types.
#define GGUF_FLOAT_NA_KERNEL(Name, W, T)                                                                       \
  kernel void Name(device bfloat *input [[buffer(0)]], device const W *weights [[buffer(1)]],                 \
                   device T *output [[buffer(2)]], constant GgufFloatParams &p [[buffer(3)]],                 \
                   uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],   \
                   ushort lane [[thread_index_in_simdgroup]]) {                                               \
    threadgroup bfloat stage[2 * gguf_float_na::kBuffer<W>];                                                   \
    gguf_float_na::project<W, T>(input, weights, output, p, tg, sg, lane, stage);                             \
  }
GGUF_FLOAT_NA_KERNEL(gguf_float_na_f32_bf16, float, bfloat)
GGUF_FLOAT_NA_KERNEL(gguf_float_na_f32_f32, float, float)
GGUF_FLOAT_NA_KERNEL(gguf_float_na_bf16_bf16, bfloat, bfloat)
GGUF_FLOAT_NA_KERNEL(gguf_float_na_bf16_f32, bfloat, float)
#undef GGUF_FLOAT_NA_KERNEL
