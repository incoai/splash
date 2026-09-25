#include "metal/kernels/common/q4_sgmatrix.h"

// Packed Q4 stays in its shipped StorageN=256 layout. Each simdgroup computes
// W X^T for 16 columns (8 for the two gate/up streams). The bfloat operand
// 128+q is exact for every nibble; subtracting 128*sum(x) in fp32 recovers q*x.
// This preserves the bf16 activation range without relying on half denormals.
namespace q4sg {
enum class Epilogue { Affine, Residual, GateUp };

template <Epilogue E>
__attribute__((always_inline)) inline void decode(device const bfloat *table, device const uchar *w0,
                   device const bfloat *sc0, device const bfloat *bi0,
                   device bfloat *out, device const float *sums,
                   device coherent(device) float *partials, device atomic_uint *counters,
                   device const bfloat *residual, device const uchar *w1,
                   device const bfloat *sc1, device const bfloat *bi1,
                   constant Q4Params &p, uint3 tg, uint tid, uint sg, uint lane,
                   threadgroup uint *arrival) {
  constexpr bool gateUp = E == Epilogue::GateUp;
  constexpr uint tileN = gateUp ? 32 : 64;
  const uint N = p.output_size, groups = p.input_size / 64;
  const uint splits = p.persistent_groups;
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
  const uint end = tg.y + 1 == splits ? groups : first + groups / splits;
  const Lane l = lane_map(lane);
  const uint fm = l.fm, fn = l.fn, c = fn / 2;
  const uint base = tg.x * tileN + sg * (gateUp ? 8 : 16);
  const uint tile = base / 256;
  const uint col0 = base % 256 + fm;
  const uint col1 = gateUp ? col0 : col0 + 8;
  device const uchar *tile0 = w0 + ulong(tile) * groups * 8192;
  device const uchar *tile1 = w1 + ulong(tile) * groups * 8192;
  auto load = [&](uint g, thread uint2 (&w)[2]) __attribute__((always_inline)) {
    w[0] = *reinterpret_cast<device const uint2 *>(tile0 + ulong(g) * 8192 + col0 * 32 + c * 8);
    w[1] = *reinterpret_cast<device const uint2 *>(tile1 + ulong(g) * 8192 + col1 * 32 + c * 8);
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
        mma_acc<bfloat>(dot[nf][j & 1], as_type<bfloat2>(pair), b);
      }
    }
    const ulong prm0 = (ulong(tile) * groups + g) * 256 + col0;
    const ulong prm1 = (ulong(tile) * groups + g) * 256 + col1;
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
      device coherent(device) float *slot = partials + ulong(tg.y * 2 + nf) * 8 * N + n;
      slot[fn * N] = acc[nf].x;
      slot[(fn + 1) * N] = acc[nf].y;
    }
    // Every writer publishes its partials before lane zero signals arrival.
    // Device-scope fences pair through the atomic counter. The last group
    // reduces in split order, independent of scheduling. No group spins.
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                          thread_scope::thread_scope_device);
      *arrival = atomic_fetch_add_explicit(counters + tg.x, 1u, memory_order_relaxed);
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                          thread_scope::thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (*arrival != splits - 1) return;
    float2 total[2] = {float2(0), float2(0)};
    for (uint s = 0; s < splits; ++s) {
#pragma unroll
      for (uint nf = 0; nf < 2; ++nf) {
        const uint n = base + fm + (gateUp ? 0 : nf * 8);
        const device coherent(device) float *slot = partials + ulong(s * 2 + nf) * 8 * N + n;
        total[nf] += s == tg.y ? acc[nf] : float2(slot[fn * N], slot[(fn + 1) * N]);
      }
    }
    acc[0] = total[0]; acc[1] = total[1];
    if (tid == 0) atomic_store_explicit(counters + tg.x, 0u, memory_order_relaxed);
  }
  if (gateUp) {
    const uint n = base + fm;
    const float2 gate = float2(bfloat2(acc[0])), up = float2(bfloat2(acc[1]));
    const float2 value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * up;
    out[fn * N + n] = bfloat(value.x);
    out[(fn + 1) * N + n] = bfloat(value.y);
  } else {
#pragma unroll
    for (uint nf = 0; nf < 2; ++nf) {
      const uint n = base + nf * 8 + fm;
      float2 value = float2(bfloat2(acc[nf]));
      if (E == Epilogue::Residual)
        value += float2(float(residual[fn * N + n]), float(residual[(fn + 1) * N + n]));
      out[fn * N + n] = bfloat(value.x);
      out[(fn + 1) * N + n] = bfloat(value.y);
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

#define Q4_SG_INPUTS \
    device const bfloat *table [[buffer(0)]], device const uchar *weights [[buffer(1)]], \
    device const bfloat *scales [[buffer(2)]], device const bfloat *biases [[buffer(3)]], \
    device bfloat *output [[buffer(4)]], device const float *sums [[buffer(5)]], \
    device float *partials [[buffer(6)]], device atomic_uint *counters [[buffer(7)]]
#define Q4_SG_THREADS \
    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]

kernel void decode_linear_q4_sg(Q4_SG_INPUTS, constant Q4Params &p [[buffer(8)]], Q4_SG_THREADS) {
  threadgroup uint arrival;
  q4sg::decode<q4sg::Epilogue::Affine>(table, weights, scales, biases, output, sums,
      partials, counters, output, weights, scales, biases, p, tg, tid, sg, lane, &arrival);
}
kernel void decode_linear_q4_sg_residual(Q4_SG_INPUTS,
    device const bfloat *residual [[buffer(8)]], constant Q4Params &p [[buffer(9)]], Q4_SG_THREADS) {
  threadgroup uint arrival;
  q4sg::decode<q4sg::Epilogue::Residual>(table, weights, scales, biases, output, sums,
      partials, counters, residual, weights, scales, biases, p, tg, tid, sg, lane, &arrival);
}
kernel void decode_linear_q4_sg_gate_up(Q4_SG_INPUTS, device const uchar *up [[buffer(8)]],
    device const bfloat *upScales [[buffer(9)]], device const bfloat *upBiases [[buffer(10)]],
    constant Q4Params &p [[buffer(11)]], Q4_SG_THREADS) {
  threadgroup uint arrival;
  q4sg::decode<q4sg::Epilogue::GateUp>(table, weights, scales, biases, output, sums,
      partials, counters, output, up, upScales, upBiases, p, tg, tid, sg, lane, &arrival);
}
#undef Q4_SG_INPUTS
#undef Q4_SG_THREADS
