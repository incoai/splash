#pragma once
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_staged.h"
#include "metal/kernels/common/gguf_tile.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;
using namespace mpp::tensor_ops;

// The staged tiles of kernels/shared/gguf_linear.metal and kernels/shared/moe_gguf.metal: dequantize the weights into
// a half stage and run matmul2d on it, in the decode tile each simdgroup its own columns alone, in the prefill tile
// every thread for the columns all simdgroups share. Includers set `#pragma clang fp reassociate(off)` first.

// The two stages of a simdgroup, and of the threadgroup's GGUF_TILE_COLUMNS / GGUF_STAGED_COLUMNS simdgroups.
constant constexpr uint kStagedSimdgroupStage = 2 * GGUF_STAGED_COLUMNS * GGUF_STAGED_STEP;
constant constexpr uint kStagedStages = GGUF_TILE_COLUMNS / GGUF_STAGED_COLUMNS * kStagedSimdgroupStage;

// The destination of a Rows-row tile, zeroed by the caller: returning an initialized cooperative tensor loses its
// initial values on Apple9 in runtime-format kernels (also with shader validation).
template <ushort Rows, ushort Cols, ushort KS>
inline auto staged_accumulator(device bfloat *input, uint input_size, threadgroup half *stage) {
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(Rows, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  auto a0 = a.template slice<KS, Rows>(0, 0);
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, Cols>(0, 0);
  return operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
}
template <class Acc> inline void gguf_zero(thread Acc &acc) {
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
}
// fn(row, column, value) for every element of a tile's destination.
template <class Acc, class Fn> inline void gguf_elements(thread Acc &acc, Fn fn) {
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto index = acc.get_multidimensional_index(i);
    fn(uint(index[1]), uint(index[0]), float(acc[i]));
  }
}

// The step at which a decode tile starts its walk over steps [step_begin, step_end) of K. Walking in lockstep
// (spread 0), every column tile of a projection fetches the same slice of the input at each step, which the operator
// before has just written and no cache holds yet, so the fetch sits on every tile's critical path. Spread (1), tile t
// starts (t * 40503 mod 2^16) / 2^16 of the way in, the golden ratio's fraction in 16 bits, which spreads the starts
// of any run of tiles evenly over the partition, and most tiles find their slice brought in by others. ops::Linear
// picks the walk (LinearConfig::spread); the MoE expert tiles and the prefill tile walk in lockstep.
inline uint staged_first_step(uint tile, uint step_begin, uint step_end, uint spread) {
  if (!spread) return step_begin;
  return step_begin + uint(ulong((tile * 40503u) & 0xFFFFu) * (step_end - step_begin) >> 16);
}

// The staged tile loop without the store, over steps [step_begin, step_end) of K from step_first, wrapping: the
// Threads threads (thread_index among them) dequantize one KS-input step of the Cols columns into the stage, meet at a
// barrier (a simdgroup's when one simdgroup stages alone, the threadgroup's otherwise), and, when `matmuls`, run the
// Rows-row matmul2d of `input` on the stage. Threads that run no matmuls still stage and meet every barrier: in MSL a
// barrier inside a conditional must be reached by every thread of the threadgroup. Each thread loads the next step's
// payload while the step's matmul runs, and so its meta unit when the next step enters a new one (a unit spans
// F::MetaGroups groups): both DRAM round trips overlap a matmul. Both go to the registers the step has just
// dequantized from, so the prefetch holds no second meta unit.
template <class F, ushort Rows, ushort Cols, ushort KS, ushort Threads, class Acc>
inline void gguf_staged_steps(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, uint input_size,
                              uint output_origin, threadgroup half *stage, threadgroup half2 *tl, uint thread_index,
                              uint step_begin, uint step_end, uint step_first, bool matmuls, thread Acc &acc) {
  constexpr ushort GPS = KS / 32, Items = Cols * GPS, IPT = (Items + Threads - 1) / Threads;
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(Rows, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  const uint groups = input_size / 32, units = groups / F::MetaGroups;
  const uint plane_tile = output_origin / QUANT_TILE_ROWS, plane_row = output_origin % QUANT_TILE_ROWS;
  device uchar *tw0 = w0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P0;
  device uchar *tw1 = w1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
  device uchar *tmeta = meta + (ulong(plane_tile) * units * QUANT_TILE_ROWS + plane_row) * F::MetaBytes;
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(stage + KS * Cols, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, Cols>(0, 0), b1 = bt1.slice<KS, Cols>(0, 0);
  // Per item: the payload of the step it stages next and that step's meta unit.
  typename F::Payload packed[IPT]; typename F::Meta hdr[IPT];
#pragma unroll
  for (ushort it = 0; it < IPT; ++it) {
    const uint item = thread_index + it * Threads; const bool live = item < Items;
    const uint col = live ? item % Cols : 0, gi = live ? item / Cols : 0;
    const ulong g = ulong(step_first) * GPS + gi;
    if (live && step_begin < step_end) packed[it] = F::load(tw0 + (g * QUANT_TILE_ROWS + col) * F::P0, tw1 + (g * QUANT_TILE_ROWS + col) * F::P1);
    hdr[it] = F::loadMeta(tmeta + (ulong(g / F::MetaGroups) * QUANT_TILE_ROWS + col) * F::MetaBytes);
  }
  // One step: dequantize `step` into stage `buffer`, load `next`'s payload and meta unit when there is a next step
  // (`more`), and run the step's matmul.
  const auto run_step = [&](uint step, uint next, bool more, uint buffer) __attribute__((always_inline)) {
    threadgroup half *buf = stage + buffer * (KS * Cols);
#pragma unroll
    for (ushort it = 0; it < IPT; ++it) {
      const uint item = thread_index + it * Threads; if (item >= Items) break;
      const uint col = item % Cols, gi = item / Cols, g = step * GPS + gi;
      dequant32<F>(packed[it], hdr[it], g % F::MetaGroups, tl, buf + col * KS + gi * 32);
    }
    if constexpr (Threads == 32) simdgroup_barrier(mem_flags::mem_threadgroup);
    else threadgroup_barrier(mem_flags::mem_threadgroup);
    if (more) {
#pragma unroll
      for (ushort it = 0; it < IPT; ++it) {
        const uint item = thread_index + it * Threads; if (item >= Items) break;
        const uint col = item % Cols, gi = item / Cols; const ulong g = ulong(next) * GPS + gi;
        packed[it] = F::load(tw0 + (g * QUANT_TILE_ROWS + col) * F::P0, tw1 + (g * QUANT_TILE_ROWS + col) * F::P1);
        const uint unit = uint(g / F::MetaGroups);
        if (unit != (step * GPS + gi) / F::MetaGroups)
          hdr[it] = F::loadMeta(tmeta + (ulong(unit) * QUANT_TILE_ROWS + col) * F::MetaBytes);
      }
    }
    if (matmuls) {
      auto a_slice = a.template slice<KS, Rows>(step * KS, 0);
      if (buffer) operation.run(a_slice, b1, acc); else operation.run(a_slice, b0, acc);
    }
  };
  if (step_first == step_begin) {
    for (uint step = step_begin; step < step_end; ++step) run_step(step, step + 1, step + 1 < step_end, step & 1);
    return;
  }
  // A walk from step_first wraps: two runs, [step_first, step_end) then [step_begin, step_first), the stages
  // alternating per step walked; the last step of the first run loads step_begin's payload.
  uint walked = 0;
  for (uint run = 0; run < 2; ++run) {
    const uint from = run ? step_begin : step_first, to = run ? step_first : step_end;
    for (uint step = from; step < to; ++step, ++walked)
      run_step(step, step + 1 < to ? step + 1 : step_begin, walked + 1 < step_end - step_begin, walked & 1);
  }
}

// The decode tile loop of one simdgroup without the store: its Cols columns over steps [step_begin, step_end) of K
// from step_first.
template <class F, ushort Rows, ushort Cols, ushort KS, class Acc>
inline void staged_accumulate(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, uint input_size, uint output_origin,
                     threadgroup half *stage, threadgroup half2 *tl, uint simd_lane, uint step_begin, uint step_end,
                     uint step_first, thread Acc &acc) {
  gguf_staged_steps<F, Rows, Cols, KS, 32>(input, w0, w1, meta, input_size, output_origin, stage, tl, simd_lane, step_begin,
                                           step_end, step_first, true, acc);
  simdgroup_barrier(mem_flags::mem_threadgroup);   // the stage may be reused by a following accumulate
}

// One tensor's weights in a one-simdgroup decode tile of GGUF_STAGED_COLUMNS columns: lane l stages column l, one
// 32-input group per step, as gguf_staged_steps does at 32 threads. It holds the payload of the step it stages next
// and that step's meta unit.
static_assert(GGUF_GATE_UP_THREADS == 32 && GGUF_STAGED_COLUMNS == 32 && GGUF_STAGED_STEP == 32,
              "lane l of one simdgroup stages column l, one 32-input group per step");
template <class F> struct StagedColumn {
  device uchar *w0, *w1, *meta;
  typename F::Payload packed;
  typename F::Meta hdr;
  // The planes of column `origin + lane`, and step `first`'s payload (when the walk has a step) and meta unit.
  void begin(device uchar *p0, device uchar *p1, device uchar *m, uint input_size, uint origin, uint lane, uint first,
             bool any) thread {
    const uint groups = input_size / 32, units = groups / F::MetaGroups;
    const uint plane_tile = origin / QUANT_TILE_ROWS, plane_row = origin % QUANT_TILE_ROWS + lane;
    w0 = p0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P0;
    w1 = p1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
    meta = m + (ulong(plane_tile) * units * QUANT_TILE_ROWS + plane_row) * F::MetaBytes;
    if (any) packed = F::load(w0 + ulong(first) * QUANT_TILE_ROWS * F::P0, w1 + ulong(first) * QUANT_TILE_ROWS * F::P1);
    hdr = F::loadMeta(meta + ulong(first / F::MetaGroups) * QUANT_TILE_ROWS * F::MetaBytes);
  }
  // Dequantizes `step` into the lane's column of `stage`.
  void stage(uint step, threadgroup half2 *tl, threadgroup half *stage, uint lane) thread {
    dequant32<F>(packed, hdr, step % F::MetaGroups, tl, stage + lane * GGUF_STAGED_STEP);
  }
  // After `step` is staged: loads `next`'s payload, and its meta unit when it enters a new one.
  void load(uint step, uint next) thread {
    packed = F::load(w0 + ulong(next) * QUANT_TILE_ROWS * F::P0, w1 + ulong(next) * QUANT_TILE_ROWS * F::P1);
    const uint unit = next / F::MetaGroups;
    if (unit != step / F::MetaGroups) hdr = F::loadMeta(meta + ulong(unit) * QUANT_TILE_ROWS * F::MetaBytes);
  }
};

// A gate/up pair's decode tile in one simdgroup: the gate's and the up's same GGUF_STAGED_COLUMNS columns over steps
// [step_begin, step_end) of K from step_first, wrapping as gguf_staged_steps walks. Each step dequantizes both tensors
// into their stages (one each), meets at a barrier, loads both tensors' next payloads, runs both matmuls on the step's
// input and meets again before the stages are rewritten. Each tensor sums its steps in the order of its own decode
// tile, so gate and up hold the bits of the gate and up passes over the same steps.
template <class F, ushort Rows, class Acc>
inline void gguf_staged_pair_steps(device bfloat *input, device uchar *gw0, device uchar *gw1, device uchar *gmeta,
                                   device uchar *uw0, device uchar *uw1, device uchar *umeta, uint input_size,
                                   uint origin, threadgroup half *stage, threadgroup half2 *tl, uint lane,
                                   uint step_begin, uint step_end, uint step_first, thread Acc &gate, thread Acc &up) {
  constexpr ushort Cols = GGUF_STAGED_COLUMNS, KS = GGUF_STAGED_STEP;
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(Rows, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(stage + KS * Cols, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  auto bg = bt0.slice<KS, Cols>(0, 0), bu = bt1.slice<KS, Cols>(0, 0);
  StagedColumn<F> g, u;
  g.begin(gw0, gw1, gmeta, input_size, origin, lane, step_first, step_begin < step_end);
  u.begin(uw0, uw1, umeta, input_size, origin, lane, step_first, step_begin < step_end);
  const auto run_step = [&](uint step, uint next, bool more) __attribute__((always_inline)) {
    g.stage(step, tl, stage, lane);
    u.stage(step, tl, stage + KS * Cols, lane);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (more) { g.load(step, next); u.load(step, next); }
    auto a_slice = a.template slice<KS, Rows>(step * KS, 0);
    operation.run(a_slice, bg, gate);
    operation.run(a_slice, bu, up);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  };
  uint walked = 0;
  for (uint run = 0; run < 2; ++run) {
    const uint from = run ? step_begin : step_first, to = run ? step_first : step_end;
    for (uint step = from; step < to; ++step, ++walked)
      run_step(step, step + 1 < to ? step + 1 : step_begin, walked + 1 < step_end - step_begin);
  }
}

// runtime dequantizer selection (uniform per threadgroup): the format's pair table, which every thread of the
// threadgroup fills, then its tile loop
template <ushort Rows, ushort Cols, ushort KS, class Acc>
inline void staged_accumulate_any(uint fmt, device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, uint input_size, uint origin,
                           threadgroup half *stage, threadgroup half2 *tl, uint thread_index, uint simd_lane, uint sb, uint se,
                           uint sf, thread Acc &acc) {
  quant_format_switch(fmt, [&](auto format) {
    typedef decltype(format) F;
    quant_pair_table<F>(tl, thread_index, GGUF_STAGED_THREADS);
    staged_accumulate<F, Rows, Cols, KS>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, sf, acc);
  });
}
