// fp32 projections of the tensors kept as rows (ops/Linear.hpp float segments): the MoE router, in the format its
// checkpoint holds it (a GGUF's F32, or BF16 as some store it; an MLX checkpoint's quantized rows), the QSA indexer's
// query and key, which share their input and, on one tile, a dispatch, and, when a GGUF stores them as F32, the GDN
// alpha/beta gates. out[r][n] = sum_k x[r][k] W[n][k] on simdgroup float 8x8 MMA with exact operands (bf16
// activations widen to fp32, the weights as stored, BF16 ones widened, a quantized format's each the fp32 value its
// decode gives), so a result differs from the fp64 product only by fp32 accumulation, and the rows of a BF16 or
// quantized tensor give the outputs of the F32 tensor of their values bit for bit. The reference GGUF
// implementation's Metal matrix multiply stages F32 weights as half above eight rows; no phase here rounds them.
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_embedding_formats.h"
#include "metal/kernels/common/sgmatrix.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// The weights' rows, one of K values per output column, of type W: float or bfloat, the F32 or BF16 values as stored,
// or Native<F>, the native blocks of quantized format F (kernels/common/gguf_embedding_formats.h), each value the fp32
// its decode gives, as the token gathers and the F32 values load-time widening writes have it.
template <typename F> struct Native {};
template <typename W> struct FloatRows {
  static ulong bytes(uint K) { return ulong(K) * sizeof(W); }
};
template <typename F> struct FloatRows<Native<F>> {
  static ulong bytes(uint K) { return ulong(K / F::Weights) * F::Bytes; }
};
// The row of output column `column`.
template <typename W> inline device const uchar *float_row(device const uchar *weights, uint column, uint K) {
  return weights + ulong(column) * FloatRows<W>::bytes(K);
}

// One projection of a dispatch (GgufFloatParams) and the column tile of a threadgroup within it: a dispatch's column
// tiles are its projections' in projection order.
template <typename T> struct FloatProjection {
  device const uchar *weights;
  device T *output;
  uint output_size, out_stride, out_offset, tile;
};
static_assert(GGUF_FLOAT_PROJECTIONS == 2, "a dispatch binds two projections");
// The projection of dispatch column tile `tile`, of `columns` columns.
template <typename T>
inline FloatProjection<T> float_projection(constant GgufFloatParams &p, uint tile, uint columns,
                                           device const uchar *weights0, device T *output0,
                                           device const uchar *weights1, device T *output1) {
  const uint tiles = (p.output_size[0] + columns - 1) / columns;
  return tile < tiles
             ? FloatProjection<T>{weights0, output0, p.output_size[0], p.out_stride[0], p.out_offset[0], tile}
             : FloatProjection<T>{weights1, output1, p.output_size[1], p.out_stride[1], p.out_offset[1], tile - tiles};
}

namespace gguf_float {

constant constexpr uint kFragments = 4;          // 8-row fragments per threadgroup: 32 rows
constant constexpr uint kRows = 8 * kFragments;
constant constexpr uint kStep = 64;              // inputs per step: eight runs of 8
// K parts, one per simdgroup: a router's decode dispatch has few threadgroups (32 to 64), and 8 parts are the
// fastest there, where 16 take up to 29% longer.
constant constexpr uint kSplits = 8;

// The eight consecutive values of a row from input k (a multiple of 8): 16 bytes of bf16 or 32 of fp32 as stored,
// each widened exactly to fp32 where read, or a quantized format's from the one block that holds them; zeros past
// the row's last run. values() gives them as a step's products read them.
template <typename W> struct Eight;
template <> struct Eight<bfloat> {
  uint4 pairs;
  static Eight load(device const uchar *row, uint k, bool inside) {
    return {inside ? *reinterpret_cast<device const uint4 *>(row + ulong(k) * 2) : uint4(0)};
  }
  float operator[](uint i) const { return float(as_type<bfloat2>(pairs[i / 2])[i % 2]); }
  Eight values() const { return *this; }
};
template <> struct Eight<float> {
  float4 low, high;
  static Eight load(device const uchar *row, uint k, bool inside) {
    device const float4 *quads = reinterpret_cast<device const float4 *>(row + ulong(k) * 4);
    return inside ? Eight{quads[0], quads[1]} : Eight{float4(0), float4(0)};
  }
  float operator[](uint i) const { return i < 4 ? low[i] : high[i - 4]; }
  Eight values() const { return *this; }
};
// A quantized format's 8 values: each its decode, read from the block as decoded, or, for an MLX affine format, the
// run's codes, scale and bias, read together and decoded only where the next step uses them (values()), so that
// the read of a step's weights stays in flight while the step before runs its products.
template <typename F> struct DecodedRun {
  float v[8];
  static DecodedRun fetch(device const uchar *block, uint k) {
    DecodedRun r;
#pragma unroll
    for (uint i = 0; i < 8; ++i) r.v[i] = F::decode(block, k + i);
    return r;
  }
  float operator[](uint i) const { return v[i]; }
};
template <typename F> struct RunOf {
  typedef DecodedRun<F> type;
};
template <uint Bits, uint Group> struct RunOf<GgufEmbedAffine<Bits, Group>> {
  typedef typename GgufEmbedAffine<Bits, Group>::Run type;
};
template <typename F> struct Eight<Native<F>> {
  typedef typename RunOf<F>::type Run;
  Run run;
  static Eight load(device const uchar *row, uint k, bool inside) {
    return {inside ? Run::fetch(row + ulong(k / F::Weights) * F::Bytes, k) : Run{}};
  }
  float operator[](uint i) const { return run[i]; }
  // The fp32 values, decoded once for every product of a step.
  Eight<float> values() const {
    return {float4(run[0], run[1], run[2], run[3]), float4(run[4], run[5], run[6], run[7])};
  }
};

// One threadgroup: output columns [8 tile, 8 tile + 8) of the projection's rows [32 y, 32 y + 32). K runs in steps of
// kStep inputs, simdgroup s taking part s of the steps for every fragment; simdgroup f then adds fragment f's parts in
// order, so a result depends neither on the grid nor on the row count. MMA j of the step from input k multiplies
// inputs k + 8 i + j over its index i, so each lane's operands are whole runs of 8 inputs: lane (fm, fn) loads
// W[column fn, fn + 1][k + 8 fm, + 8) as B and x[row fm][k + 8 fn, + 16) as A, a column's runs 128 contiguous bytes
// of bf16 over its eight lanes, and holds the destination [row fm][column fn, fn + 1] (sgmatrix::lane_map). The
// weights load a step ahead. Even and odd MMAs accumulate apart, which halves each chain of fp32 sums at no cost in
// time: at the router's K = 2048 the mean error stays 5.5e-9 of sum |x w| (7.4e-9 with one chain per part). Rows of
// K (a multiple of 8) start 16-byte aligned in page-aligned buffers.
template <typename W, typename T>
inline void project(device const bfloat *input, FloatProjection<T> projection, constant GgufFloatParams &p, uint y,
                    uint sg, uint lane, threadgroup float2 *parts) {
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint K = p.input_size, runs = K / 8, column = projection.tile * 8 + l.fn, row0 = y * kRows;
  const uint live = min(kRows, p.rows - row0);
  const uint steps = (K + kStep - 1) / kStep, first = sg * steps / kSplits, last = (sg + 1) * steps / kSplits;
  device const uchar *w0 = float_row<W>(projection.weights, column, K),
                     *w1 = float_row<W>(projection.weights, column + 1, K);
  const auto load = [&](uint step, thread Eight<W> (&b)[2]) {
    const uint run = step * 8 + l.fm;
    b[0] = Eight<W>::load(w0, run * 8, run < runs);
    b[1] = Eight<W>::load(w1, run * 8, run < runs);
  };
  float2 acc[kFragments][2];
#pragma unroll
  for (uint f = 0; f < kFragments; ++f) acc[f][0] = acc[f][1] = float2(0);
  Eight<W> next[2];
  load(first, next);
  for (uint step = first; step < last; ++step) {
    const auto b0 = next[0].values(), b1 = next[1].values();
    if (step + 1 < last) load(step + 1, next);
    const uint run = step * 8 + l.fn;
#pragma unroll
    for (uint f = 0; f < kFragments; ++f) {
      if (f * 8 >= live) break;
      const uint row = f * 8 + l.fm;
      device const uchar *x = reinterpret_cast<device const uchar *>(input + ulong(row0 + row) * K);
      const Eight<bfloat> a0 = Eight<bfloat>::load(x, run * 8, row < live && run < runs);
      const Eight<bfloat> a1 = Eight<bfloat>::load(x, run * 8 + 8, row < live && run + 1 < runs);
#pragma unroll
      for (uint j = 0; j < 8; ++j)
        sgmatrix::mma_acc<float>(acc[f][j % 2], float2(a0[j], a1[j]), float2(b0[j], b1[j]));
    }
  }
#pragma unroll
  for (uint f = 0; f < kFragments; ++f) parts[(sg * kFragments + f) * 32 + lane] = acc[f][0] + acc[f][1];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint f = sg, row = f * 8 + l.fm;
  if (f >= kFragments || row >= live) return;
  float2 total = float2(0);
  for (uint s = 0; s < kSplits; ++s) total += parts[(s * kFragments + f) * 32 + lane];
  device T *out = projection.output + ulong(row0 + row) * projection.out_stride + projection.out_offset + column;
  out[0] = T(total.x);
  out[1] = T(total.y);
}

} // namespace gguf_float

static_assert(gguf_float::kSplits >= gguf_float::kFragments, "simdgroup f reduces fragment f");

// Grid (the projections' output_size / 8 summed, ceil(rows / 32)), 32 kSplits threads; named by the weights' and the
// output's types.
#define GGUF_FLOAT_KERNEL(Name, W, T)                                                                          \
  kernel void Name(device const bfloat *input [[buffer(0)]], device const uchar *weights0 [[buffer(1)]],      \
                   device T *output0 [[buffer(2)]], device const uchar *weights1 [[buffer(3)]],               \
                   device T *output1 [[buffer(4)]], constant GgufFloatParams &p [[buffer(5)]],                \
                   uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],     \
                   uint lane [[thread_index_in_simdgroup]]) {                                                 \
    threadgroup float2 parts[gguf_float::kSplits * gguf_float::kFragments * 32];                              \
    gguf_float::project<W, T>(input, float_projection(p, tg.x, 8, weights0, output0, weights1, output1), p,   \
                              tg.y, sg, lane, parts);                                                         \
  }
GGUF_FLOAT_KERNEL(gguf_float_f32_bf16, float, bfloat)
GGUF_FLOAT_KERNEL(gguf_float_f32_f32, float, float)
GGUF_FLOAT_KERNEL(gguf_float_bf16_bf16, bfloat, bfloat)
GGUF_FLOAT_KERNEL(gguf_float_bf16_f32, bfloat, float)
// A quantized format's rows into fp32 (the MoE router's scores), by its kQuantFormats token: gguf_float_<token>_f32.
#define GGUF_FLOAT_NATIVE(f, F) GGUF_FLOAT_KERNEL(gguf_float_##f##_f32, Native<F>, float)
GGUF_QUANTIZED_EMBEDDING_FORMATS(GGUF_FLOAT_NATIVE)
#undef GGUF_FLOAT_NATIVE
#undef GGUF_FLOAT_KERNEL

// The same projections on the neural accelerator (ops::FloatTile::NeuralAccelerator), whose matmul takes bf16 operands:
// each F32 weight is split into three bf16 parts, its leading 8 significant bits, the next 8 and the last 8, so
// hi + mid + lo == w exactly (fp32 has 24) and each part's product with a bf16 activation is exact in fp32; a BF16
// weight is its own one part (the F32 tensor of its values splits into it and two zero parts, whose products add
// zeros, so the two give the same bits). The results again differ from the fp64 products only by fp32 accumulation.
// The accelerator adds a matmul's products to a destination with a few ulps of error, so the hi part accumulates apart
// from the mid and lo parts: that cuts the error against fp64 2.2x (to 6.6e-9 of sum |x w| at the router's K = 2048,
// near this file's fp32 kernel's 5.5e-9) for 3-6% of the kernel's time.
namespace gguf_float_na {

constant constexpr ushort kRowsPerSimdgroup = 16;   // the accelerator's row unit
constant constexpr ushort kSimdgroups = 4;
constant constexpr ushort kRows = kRowsPerSimdgroup * kSimdgroups;
constant constexpr ushort kColumns = 32;
constant constexpr ushort kStep = 32;   // inputs per stage
constant constexpr ushort kPerThread = kColumns * kStep / (kSimdgroups * 32);
constant constexpr uint kPlane = kColumns * kStep;
// The bf16 parts of a weight of type W, one for BF16 values and three for fp32 ones (F32 values, a quantized format's
// decoded), and the stage of a K step's (two such stages alternate).
template <typename W> constant constexpr ushort kParts = is_same_v<W, bfloat> ? 1 : 3;
template <typename W> constant constexpr uint kBuffer = kParts<W> * kPlane;
using gguf_float::Eight;
static_assert(kPerThread == 8, "a thread stages one run of 8 weights of a step");

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
inline void project(device bfloat *input, FloatProjection<T> projection, constant GgufFloatParams &p, uint y,
                    ushort sg, ushort lane, threadgroup bfloat *stage) {
  constexpr ushort Parts = kParts<W>;
  constexpr uint Buffer = kBuffer<W>;
  const uint K = p.input_size, steps = K / kStep;
  const uint row0 = y * kRows + sg * kRowsPerSimdgroup, first_row = min(row0, p.rows - kRowsPerSimdgroup);
  const uint col0 = projection.tile * kColumns;
  const ushort t = sg * 32 + lane, column = t / (kStep / kPerThread), first = t % (kStep / kPerThread) * kPerThread;
  const bool live_column = col0 + column < projection.output_size;
  device const uchar *w = float_row<W>(projection.weights, live_column ? col0 + column : 0, K);
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
  Eight<W> next = Eight<W>::load(w, first, live_column);
  const bool owns_rows = row0 < p.rows;   // uniform per simdgroup
  for (uint step = 0; step < steps; ++step) {
    threadgroup ushort *parts =
        reinterpret_cast<threadgroup ushort *>(stage + (step & 1) * Buffer) + column * kStep + first;
    if constexpr (Parts == 1) {
      *reinterpret_cast<threadgroup uint2 *>(parts) = next.pairs.xy;
      *reinterpret_cast<threadgroup uint2 *>(parts + 4) = next.pairs.zw;
    } else {
#pragma unroll
      for (ushort j = 0; j < kPerThread / 4; ++j) {
        const ushort3 x = split(next[4 * j]), y = split(next[4 * j + 1]), z = split(next[4 * j + 2]),
                      v = split(next[4 * j + 3]);
#pragma unroll
        for (ushort part = 0; part < Parts; ++part)
          *reinterpret_cast<threadgroup ushort4 *>(parts + part * kPlane + 4 * j) =
              ushort4(x[part], y[part], z[part], v[part]);
      }
    }
    if (step + 1 < steps && live_column) next = Eight<W>::load(w, (step + 1) * kStep + first, true);
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
    if (row >= row0 && column_out < projection.output_size)
      projection.output[ulong(row) * projection.out_stride + projection.out_offset + column_out] = T(high[i] + low[i]);
  }
}

} // namespace gguf_float_na

static_assert(gguf_float_na::kStep % gguf_float_na::kPerThread == 0, "a thread stages whole runs of one column");

// Grid (the projections' ceil(output_size / 32) summed, ceil(rows / 64)), 128 threads; rows >= 16, input_size a
// multiple of 32; named by the weights' and the output's types.
#define GGUF_FLOAT_NA_KERNEL(Name, W, T)                                                                       \
  kernel void Name(device bfloat *input [[buffer(0)]], device const uchar *weights0 [[buffer(1)]],            \
                   device T *output0 [[buffer(2)]], device const uchar *weights1 [[buffer(3)]],               \
                   device T *output1 [[buffer(4)]], constant GgufFloatParams &p [[buffer(5)]],                \
                   uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],   \
                   ushort lane [[thread_index_in_simdgroup]]) {                                               \
    threadgroup bfloat stage[2 * gguf_float_na::kBuffer<W>];                                                   \
    gguf_float_na::project<W, T>(input,                                                                       \
                                 float_projection(p, tg.x, gguf_float_na::kColumns, weights0, output0,        \
                                                  weights1, output1),                                         \
                                 p, tg.y, sg, lane, stage);                                                   \
  }
GGUF_FLOAT_NA_KERNEL(gguf_float_na_f32_bf16, float, bfloat)
GGUF_FLOAT_NA_KERNEL(gguf_float_na_f32_f32, float, float)
GGUF_FLOAT_NA_KERNEL(gguf_float_na_bf16_bf16, bfloat, bfloat)
GGUF_FLOAT_NA_KERNEL(gguf_float_na_bf16_f32, bfloat, float)
// A quantized format's rows into fp32: gguf_float_na_<token>_f32.
#define GGUF_FLOAT_NA_NATIVE(f, F) GGUF_FLOAT_NA_KERNEL(gguf_float_na_##f##_f32, Native<F>, float)
GGUF_QUANTIZED_EMBEDDING_FORMATS(GGUF_FLOAT_NA_NATIVE)
#undef GGUF_FLOAT_NA_NATIVE
#undef GGUF_FLOAT_NA_KERNEL
