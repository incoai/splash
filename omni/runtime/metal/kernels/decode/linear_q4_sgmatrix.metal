// Keep the source order of float operations, which Metal's default fast math
// lets the compiler reassociate, so the epilogue order below is fixed. Set
// before the includes, so it also holds for their code compiled here.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/q4_mpp_tiles.h"
#include "metal/kernels/common/q4_sgmatrix.h"
#include "metal/kernels/common/sgmatrix.h"
#include "metal/kernels/common/split_reduce.h"

// Packed Q4 keeps its storage tiles of kQ4StorageColumns columns
// (q4_mpp_tiles.h). Each simdgroup computes W X^T for 16 columns (8 for the
// two gate/up streams). The bfloat operand 128+q is exact for every nibble;
// subtracting 128*sum(x) in fp32 recovers q*x. This preserves the bf16
// activation range without relying on half denormals.
namespace q4sg {
enum class Epilogue { Affine, Residual, GateUp };

// The destination's type Out is bf16, or fp32 for a plain projection's
// logits (ops::Projection::destination), which keeps the sum unrounded.
template <Epilogue E, class Out>
__attribute__((always_inline)) inline void decode(device const bfloat *table, device const uchar *w0,
                   device const bfloat *sc0, device const bfloat *bi0,
                   device Out *out, device const float *sums,
                   device coherent(device) float *partials, device atomic_uint *counters,
                   device const bfloat *residual, device const uchar *w1,
                   device const bfloat *sc1, device const bfloat *bi1,
                   constant Q4Params &p, uint3 tg, uint3 grid, uint tid, uint sg,
                   uint lane, threadgroup uint *arrival) {
  constexpr bool gateUp = E == Epilogue::GateUp;
  constexpr uint tileN = gateUp ? 32 : 64;
  const uint N = p.output_size, groups = p.input_size / 64;
  // grid.y partitions K; the host takes only splits that divide the groups.
  const uint splits = grid.y;
  // Independent eight-row tiles share the weight layout and dispatch.
  // Each tile owns its activation workspace and split completion counters.
  table += ulong(tg.z) * p.input_size * 8;
  sums += ulong(tg.z) * p.input_size / 8;
  out += ulong(tg.z) * 8 * N;
  residual += ulong(tg.z) * 8 * N;
  if (splits > 1) {
    partials += ulong(tg.z) * splits * 16 * N;
    counters += ulong(tg.z) * N / tileN;
  }
  const uint first = tg.y * (groups / splits);
  const uint end = first + groups / splits;
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint fm = l.fm, fn = l.fn, c = fn / 2;
  const uint base = tg.x * tileN + sg * (gateUp ? 8 : 16);
  constexpr uint kGroupBytes = kQ4StorageColumns * 64 / 2;
  const uint tile = base / kQ4StorageColumns;
  const uint col0 = base % kQ4StorageColumns + fm;
  const uint col1 = gateUp ? col0 : col0 + 8;
  device const uchar *tile0 = w0 + ulong(tile) * groups * kGroupBytes;
  device const uchar *tile1 = w1 + ulong(tile) * groups * kGroupBytes;
  auto load = [&](uint g, thread uint2 (&w)[2]) __attribute__((always_inline)) {
    w[0] = *reinterpret_cast<device const uint2 *>(tile0 + ulong(g) * kGroupBytes + col0 * 32 + c * 8);
    w[1] = *reinterpret_cast<device const uint2 *>(tile1 + ulong(g) * kGroupBytes + col1 * 32 + c * 8);
  };
  float2 acc[2] = {float2(0), float2(0)};
  // Initialize every chain before the loop, including under GPU validation.
  // Per-fragment resets alone give wrong results with Apple10 GPU validation.
  float2 dot[2][2] = {{float2(0), float2(0)}, {float2(0), float2(0)}};
  uint2 words[2];
  auto run_group = [&](uint g, thread uint2 (&words)[2]) __attribute__((always_inline)) {
    const float2 sum = float2(sums[g * 8 + fn], sums[g * 8 + fn + 1]);
    device const vec<bfloat, 8> *xt =
        reinterpret_cast<device const vec<bfloat, 8> *>(table + ulong(g) * kXtPerGroup);
    vec<bfloat, 8> bq[2];
    bq[0] = xt[fm * 4 + c];
    bq[1] = xt[(8 + fm) * 4 + c];
#pragma unroll
    for (uint j = 0; j < 8; ++j) {
      const bfloat2 b = reinterpret_cast<thread bfloat2 *>(&bq[j >> 2])[j & 3];
#pragma unroll
      for (uint nf = 0; nf < 2; ++nf) {
        const uint word = j < 4 ? words[nf].x : words[nf].y;
        const uint pair = ((word >> (4 * (j & 3))) & 0x000F000Fu) | 0x43004300u;
        if (j < 2) dot[nf][j & 1] = float2(0);
        sgmatrix::mma_acc<bfloat>(dot[nf][j & 1], as_type<bfloat2>(pair), b);
      }
    }
    const ulong prm0 = (ulong(tile) * groups + g) * kQ4StorageColumns + col0;
    const ulong prm1 = (ulong(tile) * groups + g) * kQ4StorageColumns + col1;
    const float2 d0 = fma(-128.0f, sum, dot[0][0] + dot[0][1]);
    const float2 d1 = fma(-128.0f, sum, dot[1][0] + dot[1][1]);
    acc[0] = fma(d0, float(sc0[prm0]), acc[0]);
    acc[0] = fma(sum, float(bi0[prm0]), acc[0]);
    acc[1] = fma(d1, float(sc1[prm1]), acc[1]);
    acc[1] = fma(sum, float(bi1[prm1]), acc[1]);
  };
  load(first, words);
  for (uint g = first; g < end; ++g) {
    run_group(g, words);
    if (g + 1 < end) load(g + 1, words);
  }
  if (splits > 1) {
#pragma unroll
    for (uint nf = 0; nf < 2; ++nf) {
      const uint n = base + fm + (gateUp ? 0 : nf * 8);
      const auto slot = partials + ulong(tg.y * 2 + nf) * 8 * N + n;
      slot[fn * N] = acc[nf].x;
      slot[(fn + 1) * N] = acc[nf].y;
    }
    if (!split_arrive_last(counters + tg.x, splits, tid, arrival)) return;
#pragma unroll
    for (uint nf = 0; nf < 2; ++nf) {
      const uint n = base + fm + (gateUp ? 0 : nf * 8);
      acc[nf] = split_sum(acc[nf], tg.y, splits, [&](uint s) {
        const auto slot = partials + ulong(s * 2 + nf) * 8 * N + n;
        return float2(slot[fn * N], slot[(fn + 1) * N]);
      });
    }
    split_release(counters + tg.x, tid);
  }
  if (gateUp) {
    const uint n = base + fm;
    const float2 gate = float2(bfloat2(acc[0])), up = float2(bfloat2(acc[1]));
    const float2 value = splash_silu(gate) * up;
    out[fn * N + n] = bfloat(value.x);
    out[(fn + 1) * N + n] = bfloat(value.y);
  } else {
#pragma unroll
    for (uint nf = 0; nf < 2; ++nf) {
      const uint n = base + nf * 8 + fm;
      float2 value = acc[nf];
      if constexpr (!is_same_v<Out, float>) value = float2(bfloat2(value));
      if (E == Epilogue::Residual)
        value += float2(float(residual[fn * N + n]), float(residual[(fn + 1) * N + n]));
      out[fn * N + n] = Out(value.x);
      out[(fn + 1) * N + n] = Out(value.y);
    }
  }
}
} // namespace q4sg

kernel void decode_linear_q4_prepare(
    device const bfloat *input [[buffer(0)]], device bfloat *table [[buffer(1)]],
    device float *sums [[buffer(2)]], constant uint &width [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint group = (tg.x * 4 + sg) / 8, row = (tg.x * 4 + sg) % 8;
  input += ulong(tg.y) * width * 8;
  table += ulong(tg.y) * width * 8;
  sums += ulong(tg.y) * width / 8;
  const uint offset = row * width + group * 64 + 2 * lane;
  q4sg::write_input(table, sums, group, row, lane, input[offset], input[offset + 1]);
}

#define Q4_SG_INPUTS(Out) \
    device const bfloat *table [[buffer(0)]], device const uchar *weights [[buffer(1)]], \
    device const bfloat *scales [[buffer(2)]], device const bfloat *biases [[buffer(3)]], \
    device Out *output [[buffer(4)]], device const float *sums [[buffer(5)]], \
    device coherent(device) float *partials [[buffer(6)]], device atomic_uint *counters [[buffer(7)]]
#define Q4_SG_THREADS \
    uint3 tg [[threadgroup_position_in_grid]], uint3 grid [[threadgroups_per_grid]], \
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]

// The plain projection into bf16 and into fp32 (_f32: the logits,
// ops::Projection::destination); the input table stands in for the residual
// it does not read.
#define Q4_SG_AFFINE(Name, Out) \
  kernel void Name(Q4_SG_INPUTS(Out), constant Q4Params &p [[buffer(8)]], Q4_SG_THREADS) { \
    threadgroup uint arrival; \
    q4sg::decode<q4sg::Epilogue::Affine>(table, weights, scales, biases, output, sums, partials, \
        counters, table, weights, scales, biases, p, tg, grid, tid, sg, lane, &arrival); \
  }
Q4_SG_AFFINE(decode_linear_q4_sg, bfloat)
Q4_SG_AFFINE(decode_linear_q4_sg_f32, float)
#undef Q4_SG_AFFINE
kernel void decode_linear_q4_sg_residual(Q4_SG_INPUTS(bfloat),
    device const bfloat *residual [[buffer(8)]], constant Q4Params &p [[buffer(9)]], Q4_SG_THREADS) {
  threadgroup uint arrival;
  q4sg::decode<q4sg::Epilogue::Residual>(table, weights, scales, biases, output, sums,
      partials, counters, residual, weights, scales, biases, p, tg, grid, tid, sg, lane, &arrival);
}
kernel void decode_linear_q4_sg_gate_up(Q4_SG_INPUTS(bfloat), device const uchar *up [[buffer(8)]],
    device const bfloat *upScales [[buffer(9)]], device const bfloat *upBiases [[buffer(10)]],
    constant Q4Params &p [[buffer(11)]], Q4_SG_THREADS) {
  threadgroup uint arrival;
  q4sg::decode<q4sg::Epilogue::GateUp>(table, weights, scales, biases, output, sums,
      partials, counters, output, up, upScales, upBiases, p, tg, grid, tid, sg, lane, &arrival);
}
#undef Q4_SG_INPUTS
#undef Q4_SG_THREADS
