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

// The staged tile loop without the store, over steps [step_begin, step_end) of K: the Threads threads (thread_index
// among them) dequantize one KS-input step of the Cols columns into the stage, meet at a barrier (a simdgroup's when
// one simdgroup stages alone, the threadgroup's otherwise), and, when `matmuls`, run the Rows-row matmul2d of `input`
// on the stage. Threads that run no matmuls still stage and meet every barrier: in MSL a barrier inside a conditional
// must be reached by every thread of the threadgroup.
template <class F, ushort Rows, ushort Cols, ushort KS, ushort Threads, class Acc>
inline void gguf_staged_steps(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, uint input_size,
                              uint output_origin, threadgroup half *stage, threadgroup half2 *tl, uint thread_index,
                              uint step_begin, uint step_end, bool matmuls, thread Acc &acc,
                              uint plane_input_size = 0) {
  // Prefetch: the steps whose weights are loaded ahead of the one being staged.
  constexpr ushort Prefetch = 1, GPS = KS / 32, Items = Cols * GPS, IPT = (Items + Threads - 1) / Threads;
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(Rows, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  // A tile's groups lie in order: the first input_size / 32 of a view of the leading inputs of wider rows
  // (plane_input_size, 0 = input_size) are those it reads.
  const uint groups = (plane_input_size ? plane_input_size : input_size) / 32, units = groups / F::MetaGroups;
  const uint plane_tile = output_origin / QUANT_TILE_ROWS, plane_row = output_origin % QUANT_TILE_ROWS;
  device uchar *tw0 = w0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P0;
  device uchar *tw1 = w1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
  device uchar *tmeta = meta + (ulong(plane_tile) * units * QUANT_TILE_ROWS + plane_row) * F::MetaBytes;
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(stage + KS * Cols, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, Cols>(0, 0), b1 = bt1.slice<KS, Cols>(0, 0);
  typename F::Payload packed[Prefetch][IPT]; typename F::Meta hdr[IPT]; uint hdr_unit[IPT];
  const uint unit0 = (step_begin * GPS) / F::MetaGroups;
#pragma unroll
  for (ushort it = 0; it < IPT; ++it) {
    const uint item = thread_index + it * Threads; const bool live = item < Items;
    const uint col = live ? item % Cols : 0, gi = live ? item / Cols : 0;
#pragma unroll
    for (ushort pf = 0; pf < Prefetch; ++pf) {
      const ulong g = ulong(step_begin + pf) * GPS + gi;
      if (live && step_begin + pf < step_end) packed[pf][it] = F::load(tw0 + (g * QUANT_TILE_ROWS + col) * F::P0, tw1 + (g * QUANT_TILE_ROWS + col) * F::P1);
    }
    hdr[it] = F::loadMeta(tmeta + (ulong(unit0) * QUANT_TILE_ROWS + col) * F::MetaBytes); hdr_unit[it] = unit0;
  }
  for (uint step = step_begin; step < step_end; ++step) {
    threadgroup half *buf = stage + (step & 1) * (KS * Cols);
#pragma unroll
    for (ushort it = 0; it < IPT; ++it) {
      const uint item = thread_index + it * Threads; if (item >= Items) break;
      const uint col = item % Cols, gi = item / Cols, g = step * GPS + gi, unit = g / F::MetaGroups; const ushort j = g % F::MetaGroups;
      if (unit != hdr_unit[it]) { hdr[it] = F::loadMeta(tmeta + (ulong(unit) * QUANT_TILE_ROWS + col) * F::MetaBytes); hdr_unit[it] = unit; }
      dequant32<F>(packed[0][it], hdr[it], j, tl, buf + col * KS + gi * 32);
    }
    if constexpr (Threads == 32) simdgroup_barrier(mem_flags::mem_threadgroup);
    else threadgroup_barrier(mem_flags::mem_threadgroup);
    if (step + Prefetch < step_end) {
#pragma unroll
      for (ushort it = 0; it < IPT; ++it) {
        const uint item = thread_index + it * Threads; if (item >= Items) break;
        const uint col = item % Cols, gi = item / Cols; const ulong g = ulong(step + Prefetch) * GPS + gi;
        packed[Prefetch - 1][it] = F::load(tw0 + (g * QUANT_TILE_ROWS + col) * F::P0, tw1 + (g * QUANT_TILE_ROWS + col) * F::P1);
      }
    }
    if (matmuls) {
      auto a_slice = a.template slice<KS, Rows>(step * KS, 0);
      if (step & 1) operation.run(a_slice, b1, acc); else operation.run(a_slice, b0, acc);
    }
  }
}

// The decode tile loop of one simdgroup without the store: its Cols columns over steps [step_begin, step_end) of K.
template <class F, ushort Rows, ushort Cols, ushort KS, class Acc>
inline void staged_accumulate(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, uint input_size, uint output_origin,
                     threadgroup half *stage, threadgroup half2 *tl, uint simd_lane, uint step_begin, uint step_end, thread Acc &acc) {
  gguf_staged_steps<F, Rows, Cols, KS, 32>(input, w0, w1, meta, input_size, output_origin, stage, tl, simd_lane, step_begin,
                                           step_end, true, acc);
  simdgroup_barrier(mem_flags::mem_threadgroup);   // the stage may be reused by a following accumulate
}

// runtime dequantizer selection (uniform per threadgroup): the format's pair table, which every thread of the
// threadgroup fills, then its tile loop
template <ushort Rows, ushort Cols, ushort KS, class Acc>
inline void staged_accumulate_any(uint fmt, device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, uint input_size, uint origin,
                           threadgroup half *stage, threadgroup half2 *tl, uint thread_index, uint simd_lane, uint sb, uint se, thread Acc &acc) {
  quant_format_switch(fmt, [&](auto format) {
    typedef decltype(format) F;
    quant_pair_table<F>(tl, thread_index, GGUF_STAGED_THREADS);
    staged_accumulate<F, Rows, Cols, KS>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc);
  });
}
