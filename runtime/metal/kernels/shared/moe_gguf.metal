// GGUF MoE experts on the staged decode tile (kernels/common/gguf_staged_tile.h).
#pragma clang fp reassociate(off)
#include "metal/kernels/common/moe_gguf_expert_tile.h"

// An expert pass decodes its tensors' formats alone, function constants 0 and 1 for the down pass (routed, shared) and
// 0-3 for the gate/up pass (ops/MoE.cpp: each once in ascending order, the last repeated past them): the GPU compiler
// builds the loops of every format a pipeline decodes, for each of its live-row matmuls, and a cold 32-row down
// pipeline took 68 s to build with all 44 on the M5 Max, 0.25 s with one. moe_formats<Count> runs body(F()) with the
// format type of `format`, one body per distinct format; an id the pipeline does not decode decodes as its first, so
// every id runs one body.
constant uint kMoeFormat0 [[function_constant(0)]];
constant uint kMoeFormat1 [[function_constant(1)]];
constant uint kMoeFormat2 [[function_constant(2)]];
constant uint kMoeFormat3 [[function_constant(3)]];
template <uint Count> struct moe_formats {
  template <class Body> void operator()(uint format, Body body) const {
    if constexpr (Count > 3)
      if (kMoeFormat3 != kMoeFormat2 && format == kMoeFormat3) return quant_format_switch(kMoeFormat3, body);
    if constexpr (Count > 2)
      if (kMoeFormat2 != kMoeFormat1 && format == kMoeFormat2) return quant_format_switch(kMoeFormat2, body);
    if (kMoeFormat1 != kMoeFormat0 && format == kMoeFormat1) return quant_format_switch(kMoeFormat1, body);
    quant_format_switch(kMoeFormat0, body);
  }
};

// The down pass of the MoE experts (ops/MoE.cpp; kernels/shared/moe.metal groups the rows): projection 2 of the
// tiles' experts over the intermediate, the routed ones in the chunks of the table at the index after the parameters
// (metal/abi/MoE.h), into bf16 outputs (moe_gguf_expert_tile).
template <ushort Rows, uint Shared>
kernel void moe_expert_gguf_down(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                                 device const uint *tile_count [[buffer(2)]], device uchar *sw0 [[buffer(3)]],
                                 device uchar *sw1 [[buffer(4)]], device uchar *smeta [[buffer(5)]],
                                 device bfloat *output [[buffer(6)]], device const uint *slots [[buffer(7)]],
                                 constant MoeGgufExpertParams &p [[buffer(8)]], constant ulong *chunks [[buffer(9)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint simd_lane [[thread_index_in_simdgroup]],
                                 uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries];
  moe_gguf_expert_tile<Rows, Shared, 2>(input, tiles, tile_count, sw0, sw1, smeta, slots, p, chunks, group, simd_lane,
                                        simd_group, stage, tl, [&](ulong o, float v) { output[o] = bfloat(v); },
                                        moe_formats<2>());
}
using MoeExpertGgufKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *, device uchar *,
                                 device uchar *, device uchar *, device bfloat *, device const uint *,
                                 constant MoeGgufExpertParams &, constant ulong *, uint2, uint, uint);
template [[host_name("moe_expert_gguf_m8_down")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<8, 1>;
template [[host_name("moe_expert_gguf_m32_down")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<32, 1>;
template [[host_name("moe_expert_gguf_m8_down_s2")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<8, 2>;
template [[host_name("moe_expert_gguf_m32_down_s2")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<32, 2>;

// The gate and up passes in one (ops/MoE.cpp): threadgroup (x, y) runs a gate simdgroup and an up simdgroup on the same
// 32 columns of tile y, grid (N / 32, tiles), each through its own stage in its own pass's formats, as in
// moe_gguf_expert_tile. Every pass ends in a tail that leaves cores idle; one pass of gate and up has one tail, which
// runs 1-5% faster than the two passes from two lanes or 16-row chunks up. Where gate and up share their routed and
// their shared formats the two read one pair table, otherwise (_gate_up_mixed: a GGUF may store a shared expert's gate
// and up in different formats) each its own, a kilobyte more threadgroup memory that costs up to 5%. Every thread fills
// the tables before one barrier (none without a table), so the simdgroups' walks, of different formats, meet no barrier
// but their own. Each simdgroup parks bf16 of its sums in its stage, which its loop no longer reads, and the up
// simdgroup writes bf16(up) * silu(gate) as gguf_epilogue<EpUpWithGate> does, so the intermediate is bitwise that of
// the gate and up passes apart (moe_gguf_expert_tile of projections 0 and 1), which gguf-moe's reference runs.
// The gate/up pass's column tile (moe_column_tile), out of line: inlined in every format, it grew the 32-row kernel
// until a 10-core M4 (Apple9) under shader validation summed one element of the staged accumulator wrongly, the same
// kernel right without validation.
__attribute__((noinline)) void moe_gate_up_column(device bfloat *x, uint rows, uint K, uint N, MoeGgufSegment s,
                                                  uint origin, uint simd_lane, threadgroup half2 *table, bool up,
                                                  threadgroup bfloat *gate, device bfloat *output, ulong out) {
  float sums[MOE_GGUF_COLUMN_ROWS];
  moe_formats<4>()(s.format, [&](auto f) {
    column_dot<decltype(f), MOE_GGUF_COLUMN_ROWS>(x, rows, K, s.w0, s.w1, s.meta, origin + simd_lane, 0, K / 32, table,
                                                  sums);
  });
  if (!up)
    for (uint row = 0; row < rows; ++row) gate[row * GGUF_STAGED_COLUMNS + simd_lane] = bfloat(sums[row]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (up)
    for (uint row = 0; row < rows; ++row)
      output[out + ulong(row) * N + origin + simd_lane] =
          bfloat(float(bfloat(sums[row])) * splash_silu(float(gate[row * GGUF_STAGED_COLUMNS + simd_lane])));
}
// A gate/up simdgroup's staged tile of R rows in format F (moe_expert_gguf_gate_up), out of line as the column tile
// is: its sums parked as bf16 in `park`, its stage, which its loop no longer reads. Inlined in the pass, a 10-core M4
// (Apple9) under shader validation summed one element of the shared expert's staged accumulator wrongly once the pass
// decoded its formats alone (moe_formats).
template <class F, ushort R>
__attribute__((noinline)) void moe_gate_up_staged(device bfloat *x, MoeGgufSegment s, uint K, uint origin,
                                                  threadgroup half *my, threadgroup half2 *table, uint simd_lane,
                                                  threadgroup bfloat *park) {
  auto acc = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, K, my);
  gguf_zero(acc);
  staged_accumulate<F, R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, s.w0, s.w1, s.meta, K, origin, my, table,
                                                                 simd_lane, 0, K / GGUF_STAGED_STEP, 0, acc);
  gguf_elements(acc, [&](uint row, uint column, float v) { park[row * GGUF_STAGED_COLUMNS + column] = bfloat(v); });
}
template <ushort Rows, ushort Tables, uint Shared>
kernel void moe_expert_gguf_gate_up(device bfloat *input [[buffer(0)]],
                                    device const MoeTileDescriptor *tiles [[buffer(1)]],
                                    device const uint *tile_count [[buffer(2)]], device uchar *gsw0 [[buffer(3)]],
                                    device uchar *gsw1 [[buffer(4)]], device uchar *gsmeta [[buffer(5)]],
                                    device uchar *usw0 [[buffer(6)]], device uchar *usw1 [[buffer(7)]],
                                    device uchar *usmeta [[buffer(8)]], device bfloat *output [[buffer(9)]],
                                    device const uint *slots [[buffer(10)]],
                                    constant MoeGgufGateUpParams &p [[buffer(11)]],
                                    constant ulong *chunks [[buffer(12)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint simd_lane [[thread_index_in_simdgroup]],
                                    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[kStagedStages]; threadgroup half2 tl[Tables * kQuantPairTableEntries];
  const uint t = moe_first_tile(tile_count, p.tiles) + group.y;
  if (t >= moe_end_tile(tile_count, p.tiles)) return;
  const MoeTileDescriptor tile = tiles[t];
  const bool up = simd_group == 1;
  const MoeGgufSegment gs = moe_gguf_segment<Shared>(tile.expert, p.experts, p.input_size, p.output_size,
                                                     p.gate_routed_format, p.gate_shared_format, slots, chunks,
                                                     p.floor_slots, p.chunk_slots, 0, gsw0, gsw1, gsmeta);
  const MoeGgufSegment us = moe_gguf_segment<Shared>(tile.expert, p.experts, p.input_size, p.output_size,
                                                     p.up_routed_format, p.up_shared_format, slots, chunks,
                                                     p.floor_slots, p.chunk_slots, 1, usw0, usw1, usmeta);
  const uint thread_index = simd_group * 32 + simd_lane;
  bool tables = false;
  moe_formats<4>()(gs.format, [&](auto f) {
    tables |= quant_pair_entries<decltype(f)>(tl, thread_index, GGUF_STAGED_THREADS);
  });
  if constexpr (Tables == 2)
    moe_formats<4>()(us.format, [&](auto f) {
      tables |= quant_pair_entries<decltype(f)>(tl + kQuantPairTableEntries, thread_index, GGUF_STAGED_THREADS);
    });
  if (tables) threadgroup_barrier(mem_flags::mem_threadgroup);
  const MoeGgufSegment s = up ? us : gs;
  threadgroup half2 *table = tl + (Tables == 2 && up ? kQuantPairTableEntries : 0);
  const uint K = p.input_size, N = p.output_size;
  device bfloat *x = input + ulong(t) * Rows * K;
  const ulong out = ulong(t) * Rows * N;
  const uint origin = group.x * GGUF_STAGED_COLUMNS;
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  threadgroup bfloat *gate = reinterpret_cast<threadgroup bfloat *>(stage);
  if (moe_column_tile(tiles, t)) {
    moe_gate_up_column(x, tile.rows, K, N, s, origin, simd_lane, table, up, gate, output, out);
    return;
  }
  const auto run = [&](auto rows) {
    constexpr ushort R = decltype(rows)::value;
    static_assert(R * GGUF_STAGED_COLUMNS * sizeof(bfloat) <= kStagedSimdgroupStage * sizeof(half),
                  "the gate fits a stage");
    // Each simdgroup parks bf16 of its sums in its stage: the gate's is `gate`.
    threadgroup bfloat *park = reinterpret_cast<threadgroup bfloat *>(my);
    moe_formats<4>()(s.format, [&](auto f) {
      moe_gate_up_staged<decltype(f), R>(x, s, K, origin, my, table, simd_lane, park);
    });
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (up)
      for (uint row = 0; row < R; ++row)
        output[out + ulong(row) * N + origin + simd_lane] =
            bfloat(float(park[row * GGUF_STAGED_COLUMNS + simd_lane]) *
                   splash_silu(float(gate[row * GGUF_STAGED_COLUMNS + simd_lane])));
  };
  moe_live_rows<Rows>(tile.rows, run);
}
using MoeExpertGgufGateUpKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *,
                                       device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                       device uchar *, device bfloat *, device const uint *,
                                       constant MoeGgufGateUpParams &, constant ulong *, uint2, uint, uint);
template [[host_name("moe_expert_gguf_m8_gate_up")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<8, 1, 1>;
template [[host_name("moe_expert_gguf_m32_gate_up")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<32, 1, 1>;
template [[host_name("moe_expert_gguf_m8_gate_up_mixed")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<8, 2, 1>;
template [[host_name("moe_expert_gguf_m32_gate_up_mixed")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<32, 2, 1>;
template [[host_name("moe_expert_gguf_m8_gate_up_s2")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<8, 1, 2>;
template [[host_name("moe_expert_gguf_m32_gate_up_s2")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<32, 1, 2>;
template [[host_name("moe_expert_gguf_m8_gate_up_mixed_s2")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<8, 2, 2>;
template [[host_name("moe_expert_gguf_m32_gate_up_mixed_s2")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<32, 2, 2>;
