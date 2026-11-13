#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_staged_tile.h"
#include "metal/kernels/common/rms_inverse.h"
struct FfnParams {
  uint hidden, intermediate, splits, workers, phase, dynamic, spread, pad;
};
// One SIMD virtualizes the production RMS's eight logical SIMD groups in their
// original order.
inline void muse_ffn_norm(device coherent(device) bfloat *input,
                          device bfloat *weight,
                          device coherent(device) bfloat *normalized,
                          uint width, uint row, uint lane,
                          threadgroup float *scratch) {
  float total = 0;
  for (uint sg = 0; sg < 8; ++sg) {
    float sum = 0;
    for (uint c = sg * 32 + lane; c < width; c += 256) {
      float v = float(input[ulong(row) * width + c]);
      sum += v * v;
    }
    sum = simd_sum(sum);
    if (lane == 0)
      total += sum;
  }
  if (lane == 0)
    scratch[0] = rsqrt(total / width + kRmsEpsilon);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = lane; c < width; c += 32)
    normalized[ulong(row) * width + c] = bfloat(
        (float(input[ulong(row) * width + c]) * scratch[0]) * float(weight[c]));
}
template <ushort Rows>
inline auto muse_ffn_accumulator(threadgroup bfloat *xs, threadgroup half *ws) {
  auto a = tensor(xs, dextents<int, 2>{32, Rows}, array<int, 2>{1, 32});
  constexpr auto descriptor =
      matmul2d_descriptor(Rows, 32, 32, false, true, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  auto a0 = a.slice<32, Rows>(0, 0);
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> b(
      ws, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
  auto b0 = b.slice<32, 32>(0, 0);
  return operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), float>();
}
template <ushort Rows, bool Pair, bool RegisterInput>
inline void
muse_ffn_projection(device coherent(device) bfloat *x, device uchar *w0,
                    device uchar *meta, device uchar *uw0, device uchar *umeta,
                    device coherent(device) bfloat *y,
                    device coherent(device) float *partials, uint K, uint N,
                    uint splits, uint job, uint spread, uint lane,
                    threadgroup half *ws, threadgroup bfloat *xs) {
  const uint tile = job % (N / 32), split = job / (N / 32),
             sb = split * (K / 32 / splits), se = sb + K / 32 / splits,
             first = staged_first_step(tile / 2, sb, se, spread),
             origin = tile * 32;
  auto g = muse_ffn_accumulator<Rows>(xs, ws);
  auto u = muse_ffn_accumulator<Rows>(xs, ws);
  gguf_zero(g);
  gguf_zero(u);
  auto a = tensor(xs, dextents<int, 2>{32, Rows}, array<int, 2>{1, 32});
  auto a0 = a.slice<32, Rows>(0, 0);
  constexpr auto descriptor =
      matmul2d_descriptor(Rows, 32, 32, false, true, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  auto a_regs =
      operation
          .template get_left_input_cooperative_tensor<bfloat, half, float>();
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(
      ws, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(
      ws + 1024, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
  auto b0 = bt0.slice<32, 32>(0, 0), b1 = bt1.slice<32, 32>(0, 0);
  StagedColumn<FmtAF4G64> gcol, ucol;
  gcol.begin(w0, w0, meta, K, origin, lane, first, true);
  if constexpr (Pair)
    ucol.begin(uw0, uw0, umeta, K, origin, lane, first, true);
  const auto step = [&](uint at, uint next, bool more,
                        uint) __attribute__((always_inline)) {
    if constexpr (RegisterInput) {
#pragma unroll
      for (ushort i = 0; i < a_regs.get_capacity(); ++i) {
        if (!a_regs.is_valid_element(i))
          continue;
        auto index = a_regs.get_multidimensional_index(i);
        a_regs[i] = x[ulong(index[1]) * K + at * 32 + index[0]];
      }
    } else
      for (uint row = 0; row < Rows; ++row)
        xs[row * 32 + lane] = x[ulong(row) * K + at * 32 + lane];
    gcol.stage(at, nullptr, ws, lane);
    if constexpr (Pair)
      ucol.stage(at, nullptr, ws + 1024, lane);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (more) {
      gcol.load(at, next);
      if constexpr (Pair)
        ucol.load(at, next);
    }
    if constexpr (RegisterInput) {
      operation.run(a_regs, b0, g);
      if constexpr (Pair)
        operation.run(a_regs, b1, u);
    } else {
      operation.run(a0, b0, g);
      if constexpr (Pair)
        operation.run(a0, b1, u);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  };
  gguf_staged_walk(sb, se, first, step);
#pragma unroll
  for (ushort i = 0; i < g.get_capacity(); ++i) {
    if (!g.is_valid_element(i))
      continue;
    auto at = g.get_multidimensional_index(i);
    const ulong out = ulong(at[1]) * N + origin + at[0];
    if constexpr (Pair)
      y[out] = bfloat(float(bfloat(u[i])) * splash_silu(float(bfloat(g[i]))));
    else
      partials[ulong(split) * Rows * N + out] = g[i];
  }
}
inline uint muse_ffn_jobs(uint phase, constant FfnParams &p, uint rows) {
  return phase == 0   ? rows
         : phase == 1 ? p.intermediate / 32
         : phase == 2 ? p.hidden / 32 * p.splits
                      : p.hidden / 32;
}
template <ushort Rows, bool RegisterInput>
inline void
muse_ffn_job(uint phase, uint job, device coherent(device) bfloat *input,
             device bfloat *weight, device uchar *gw, device uchar *gm,
             device uchar *uw, device uchar *um, device uchar *dw,
             device uchar *dm, device coherent(device) bfloat *normalized,
             device coherent(device) bfloat *intermediate,
             device coherent(device) float *partials,
             device coherent(device) bfloat *output, constant FfnParams &p,
             uint lane, threadgroup half *ws, threadgroup bfloat *xs,
             threadgroup float *scratch) {
  if (phase == 0)
    muse_ffn_norm(input, weight, normalized, p.hidden, job, lane, scratch);
  else if (phase == 1)
    muse_ffn_projection<Rows, true, RegisterInput>(
        normalized, gw, gm, uw, um, intermediate, partials, p.hidden,
        p.intermediate, 1, job, p.spread, lane, ws, xs);
  else if (phase == 2)
    muse_ffn_projection<Rows, false, RegisterInput>(
        intermediate, dw, dm, dw, dm, output, partials, p.intermediate,
        p.hidden, p.splits, job, p.spread, lane, ws, xs);
  else
    for (uint row = 0; row < Rows; ++row) {
      const ulong at = ulong(row) * p.hidden + job * 32 + lane;
      float total = 0;
      for (uint s = 0; s < p.splits; ++s)
        total += partials[ulong(s) * Rows * p.hidden + at];
      output[at] = bfloat(total + float(input[at]));
    }
}
inline bool muse_ffn_meet(device atomic_uint *state, uint workers,
                          thread uint &generation, uint lane,
                          threadgroup float *scratch) {
  simdgroup_barrier(mem_flags::mem_device);
  if (lane == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);
    uint arrived = atomic_fetch_add_explicit(state, 1u, memory_order_relaxed);
    if (arrived == workers - 1) {
      atomic_store_explicit(state, 0u, memory_order_relaxed);
      atomic_store_explicit(state + 4, 0u, memory_order_relaxed);
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                          thread_scope_device);
      atomic_store_explicit(state + 1, generation + 1, memory_order_relaxed);
    } else {
      uint spin = 0;
      while (atomic_load_explicit(state + 1, memory_order_relaxed) !=
                 generation + 1 &&
             atomic_load_explicit(state + 2, memory_order_relaxed) == 0u &&
             ++spin < (1u << 20)) {
      }
      if (spin == (1u << 20))
        atomic_store_explicit(state + 2, 1u, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);
    scratch[1] = float(atomic_load_explicit(state + 2, memory_order_relaxed));
  }
  simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  ++generation;
  return scratch[1] == 0;
}
template <ushort Rows, bool Persistent, bool RegisterInput>
kernel void
muse_ffn_ffn(device coherent(device) bfloat *input [[buffer(0)]],
             device bfloat *weight [[buffer(1)]],
             device uchar *gw [[buffer(2)]], device uchar *gm [[buffer(3)]],
             device uchar *uw [[buffer(4)]], device uchar *um [[buffer(5)]],
             device uchar *dw [[buffer(6)]], device uchar *dm [[buffer(7)]],
             device coherent(device) bfloat *normalized [[buffer(8)]],
             device coherent(device) bfloat *intermediate [[buffer(9)]],
             device coherent(device) float *partials [[buffer(10)]],
             device coherent(device) bfloat *output [[buffer(11)]],
             device atomic_uint *state [[buffer(12)]],
             constant FfnParams &p [[buffer(13)]],
             uint group [[threadgroup_position_in_grid]],
             uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half ws[2048];
  threadgroup bfloat xs[Rows * 32];
  threadgroup float scratch[8];
  if constexpr (!Persistent) {
    muse_ffn_job<Rows, RegisterInput>(
        p.phase, group, input, weight, gw, gm, uw, um, dw, dm, normalized,
        intermediate, partials, output, p, lane, ws, xs, scratch);
    return;
  }
  uint generation = atomic_load_explicit(state + 1, memory_order_relaxed);
  for (uint phase = 0; phase < 4; ++phase) {
    const uint count = muse_ffn_jobs(phase, p, Rows);
    if (p.dynamic) {
      while (true) {
        if (lane == 0)
          scratch[2] = as_type<float>(
              atomic_fetch_add_explicit(state + 4, 1u, memory_order_relaxed));
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint job = as_type<uint>(scratch[2]);
        if (job >= count)
          break;
        muse_ffn_job<Rows, RegisterInput>(
            phase, job, input, weight, gw, gm, uw, um, dw, dm, normalized,
            intermediate, partials, output, p, lane, ws, xs, scratch);
      }
    } else
      for (uint job = group; job < count; job += p.workers)
        muse_ffn_job<Rows, RegisterInput>(
            phase, job, input, weight, gw, gm, uw, um, dw, dm, normalized,
            intermediate, partials, output, p, lane, ws, xs, scratch);
    if (!muse_ffn_meet(state, p.workers, generation, lane, scratch))
      return;
  }
  if (group == 0 && lane == 0)
    atomic_fetch_add_explicit(state + 3, 1u, memory_order_relaxed);
}
// The native comparison uses the production norm and projection entry points.
// These entry points implement the same weighted block with coherent
// cross-group intermediates.
#define FFN_VARIANT(R, Reg, Prefix)                                            \
  template [[host_name(Prefix "static_m" #R)]] kernel void                     \
  muse_ffn_ffn<R, false, Reg>(                                                 \
      device coherent(device) bfloat *, device bfloat *, device uchar *,       \
      device uchar *, device uchar *, device uchar *, device uchar *,          \
      device uchar *, device coherent(device) bfloat *,                        \
      device coherent(device) bfloat *, device coherent(device) float *,       \
      device coherent(device) bfloat *, device atomic_uint *,                  \
      constant FfnParams &, uint, uint);                                       \
  template [[host_name(Prefix "persistent_m" #R)]] kernel void                 \
  muse_ffn_ffn<R, true, Reg>(                                                  \
      device coherent(device) bfloat *, device bfloat *, device uchar *,       \
      device uchar *, device uchar *, device uchar *, device uchar *,          \
      device uchar *, device coherent(device) bfloat *,                        \
      device coherent(device) bfloat *, device coherent(device) float *,       \
      device coherent(device) bfloat *, device atomic_uint *,                  \
      constant FfnParams &, uint, uint);
#define FFN(R)                                                                 \
  FFN_VARIANT(R, false, "muse_ffn_") FFN_VARIANT(R, true, "muse_ffn_register_")
FFN(8) FFN(16) FFN(32)
#undef FFN
#undef FFN_VARIANT
