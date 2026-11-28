// GGUF MoE experts on the staged decode tile (kernels/common/gguf_staged_tile.h).
#pragma clang fp reassociate(off)
#include "metal/kernels/common/moe_gguf_expert_tile.h"
#include "metal/kernels/common/softcapped_swiglu.h"

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

// Function constant 4, which production never defines, so that kMoeColumnTiles folds to true and an expert's only
// tile of at most GGUF_COLUMN_ROWS live rows runs the column tile. The instrumented backend defines it true
// (metal/BackendInstrumentation.hpp: stageExpertTiles): every tile then runs the staged tile, whose 8-, 16-, 32- and
// 64-row matmuls give a row the same bits (gguf-moe), so that a route's output depends on its own row alone, not on how
// many rows its step routes to its expert (moe_expert_slab.h).
constant bool kMoeStagedOnly [[function_constant(4)]];
constant bool kMoeColumnTiles = !is_function_constant_defined(kMoeStagedOnly) || !kMoeStagedOnly;

// The down pass of the MoE experts (ops/MoE.cpp; kernels/shared/moe.metal groups the rows): projection 2 of the
// tiles' experts over the intermediate, the routed ones in the chunks of the table at the index after the parameters
// (metal/abi/MoE.h), times their down scales, into bf16 outputs (moe_gguf_expert_tile).
template <ushort Rows, uint Shared, ushort Partitions>
kernel void moe_expert_gguf_down(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                                 device const uint *tile_count [[buffer(2)]], device uchar *sw0 [[buffer(3)]],
                                 device uchar *sw1 [[buffer(4)]], device uchar *smeta [[buffer(5)]],
                                 device bfloat *output [[buffer(6)]], device const uint *slots [[buffer(7)]],
                                 device const float *expert_scales [[buffer(8)]],
                                 constant MoeGgufExpertParams &p [[buffer(9)]], constant ulong *chunks [[buffer(10)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint simd_lane [[thread_index_in_simdgroup]],
                                 uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[Partitions * kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries];
  moe_gguf_expert_tile<Rows, Shared, 2, Partitions>(input, tiles, tile_count, sw0, sw1, smeta, slots, expert_scales, p, chunks,
                                        group, simd_lane, simd_group, stage, tl,
                                        [&](ulong o, float v) { output[o] = bfloat(v); }, moe_formats<2>(),
                                        kMoeColumnTiles);
}
using MoeExpertGgufKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *, device uchar *,
                                 device uchar *, device uchar *, device bfloat *, device const uint *,
                                 device const float *, constant MoeGgufExpertParams &, constant ulong *, uint2, uint,
                                 uint);
template [[host_name("moe_expert_gguf_m8_down")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<8, 1, 1>;
template [[host_name("moe_expert_gguf_m16_down")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<16, 1, 1>;
template [[host_name("moe_expert_gguf_m32_down")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<32, 1, 1>;
template [[host_name("moe_expert_gguf_m8_down_s2")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<8, 2, 1>;
template [[host_name("moe_expert_gguf_m16_down_s2")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<16, 2, 1>;
template [[host_name("moe_expert_gguf_m32_down_s2")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<32, 2, 1>;
template [[host_name("moe_expert_gguf_m8_down_s0")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<8, 0, 1>;
template [[host_name("moe_expert_gguf_m16_down_s0")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<16, 0, 1>;
template [[host_name("moe_expert_gguf_m32_down_s0")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<32, 0, 1>;
// The 8-row passes of two K partitions (moe_gguf_expert_tile), decode steps' (ops::MoeConfig::gateUpPartitions, downPartitions).
template [[host_name("moe_expert_gguf_m8k2_down")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<8, 1, 2>;
template [[host_name("moe_expert_gguf_m8k2_down_s2")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<8, 2, 2>;
template [[host_name("moe_expert_gguf_m8k2_down_s0")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<8, 0, 2>;
// The 64-row passes, here and below, of a routing of one shared slice or none: two slices' (Flint's), whose prefill
// chunks average at most 48 routes an expert (ops::selectMoe), run none (ops::MoePlan refuses them).
template [[host_name("moe_expert_gguf_m64_down")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<64, 1, 1>;
template [[host_name("moe_expert_gguf_m64_down_s0")]] kernel MoeExpertGgufKernel moe_expert_gguf_down<64, 0, 1>;

// The gate and up passes in one (ops/MoE.cpp): threadgroup (x, y) runs a gate simdgroup and an up simdgroup on the same
// 32 columns of tile y, grid (N / 32, tiles), each through its own stage in its own pass's formats, as in
// moe_gguf_expert_tile. Every pass ends in a tail that leaves cores idle; one pass of gate and up has one tail, which
// runs 1-5% faster than the two passes from two lanes or 16-row chunks up. Where gate and up share their routed and
// their shared formats the two read one pair table, otherwise (_gate_up_mixed: a GGUF may store a shared expert's gate
// and up in different formats) each its own, a kilobyte more threadgroup memory that costs up to 5%. Every thread fills
// the tables before one barrier (none without a table), so the simdgroups' walks, of different formats, meet no barrier
// but their own. Each simdgroup parks bf16 of its sums in its stage, which its loop no longer reads, and the up
// simdgroup writes bf16(up) * silu(gate) as gguf_epilogue<EpUpWithGate> does, so the intermediate is bitwise that of
// the gate and up passes apart (moe_gguf_expert_tile of projections 0 and 1), which gguf-moe's reference runs. Each
// simdgroup's sums are its expert's scale of its projection (moe_expert_scale, `scale`) times the products'.
// Activation: the up simdgroup's output, Activation::value(up, gate) of the bf16 up and gate values: MoeSiluGate's
// bf16(up) * silu(gate), as gguf_epilogue<EpUpWithGate>, or a compiled routing's (kernels/common/softcapped_swiglu.h).
struct MoeSiluGate {
  static float value(float up, float gate) { return up * splash_silu(gate); }
};
// The input rows of a column tile's live rows: its grouped rows, or a narrow step's rows in place
// (moe_expert_gguf_gate_up).
struct MoeColumnRows {
  device bfloat *row[GGUF_COLUMN_ROWS];
};
// The gate/up pass's column tile (moe_column_tile): a simdgroup's sums of its 32 columns over groups [g_begin, g_end)
// of K, out of line: inlined in every format, it grew the 32-row kernel until a 10-core M4 (Apple9) under shader
// validation summed one element of the staged accumulator wrongly, the same kernel right without validation.
__attribute__((noinline)) void moe_gate_up_column(MoeColumnRows x, uint rows, uint K, MoeGgufSegment s, uint column,
                                                  uint g_begin, uint g_end, threadgroup half2 *table,
                                                  thread float (&sums)[GGUF_COLUMN_ROWS]) {
  moe_formats<4>()(s.format, [&](auto f) {
    column_dot_rows<decltype(f), GGUF_COLUMN_ROWS>([&](uint r) { return x.row[r]; }, rows, K, s.w0, s.w1, s.meta,
                                                   column, g_begin, g_end, table, sums);
  });
}
// A gate/up simdgroup's staged tile of R rows in format F over steps [step_begin, step_end) of K
// (moe_expert_gguf_gate_up), out of line as the column tile is, parking its sums in `park`, its stage, which its
// loop no longer reads: times `scale` as bf16 where it sums all of K, as they are in fp32 where it sums a partition
// (Partial). Inlined in the pass, a 10-core M4 (Apple9) under shader validation summed one element of the shared
// expert's staged accumulator wrongly once the pass decoded its formats alone (moe_formats).
template <class F, ushort R, bool Partial>
__attribute__((noinline)) void moe_gate_up_staged(device bfloat *x, MoeGgufSegment s, float scale, uint K,
                                                  uint origin, threadgroup half *my, threadgroup half2 *table,
                                                  uint simd_lane, uint step_begin, uint step_end) {
  auto acc = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, K, my);
  gguf_zero(acc);
  staged_accumulate<F, R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, s.w0, s.w1, s.meta, K, origin, my, table,
                                                                 simd_lane, step_begin, step_end, step_begin, acc);
  if constexpr (Partial)
    gguf_elements(acc, [&](uint row, uint column, float v) {
      reinterpret_cast<threadgroup float *>(my)[row * GGUF_STAGED_COLUMNS + column] = v;
    });
  else
    gguf_elements(acc, [&](uint row, uint column, float v) {
      reinterpret_cast<threadgroup bfloat *>(my)[row * GGUF_STAGED_COLUMNS + column] = bfloat(v * scale);
    });
}
// The gate and up passes in one over a tile's K partitions (Partitions, MoeConfig::gateUpPartitions, downPartitions): threadgroup (x, y)
// runs 2 * Partitions simdgroups on the same 32 columns of tile y, simdgroup s the gate (s % 2 == 0) or the up of
// partition s / 2 of K (moe_partition_first), each in its own stage. One partition: the gate simdgroup parks bf16 of
// its sums times its scale in its stage (`gate`), and the up simdgroup writes Activation::value of its own and the
// gate's. More: every simdgroup parks its fp32 sums in its stage, and after a barrier the partition-0 gate
// simdgroup parks bf16 of the partitions' sum in order times its scale past the sums it read (`gate`, a stage's
// second half), and after another the partition-0 up simdgroup writes the activation of its partitions' sum.
template <ushort Rows, ushort Tables, uint Shared, class Activation, ushort Partitions>
kernel void moe_expert_gguf_gate_up(device bfloat *input [[buffer(0)]],
                                    device const MoeTileDescriptor *tiles [[buffer(1)]],
                                    device const uint *tile_count [[buffer(2)]], device uchar *gsw0 [[buffer(3)]],
                                    device uchar *gsw1 [[buffer(4)]], device uchar *gsmeta [[buffer(5)]],
                                    device uchar *usw0 [[buffer(6)]], device uchar *usw1 [[buffer(7)]],
                                    device uchar *usmeta [[buffer(8)]], device bfloat *output [[buffer(9)]],
                                    device const uint *slots [[buffer(10)]],
                                    device const float *expert_scales [[buffer(11)]],
                                    device const uint *grouped_routes [[buffer(12)]],
                                    constant MoeGgufGateUpParams &p [[buffer(13)]],
                                    constant ulong *chunks [[buffer(14)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint simd_lane [[thread_index_in_simdgroup]],
                                    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[Partitions * kStagedStages]; threadgroup half2 tl[Tables * kQuantPairTableEntries];
  const uint t = moe_first_tile(tile_count, p.tiles) + group.y;
  if (t >= moe_end_tile(tile_count, p.tiles)) return;
  const MoeTileDescriptor tile = tiles[t];
  const bool up = simd_group % 2 == 1;
  const uint partition = simd_group / 2;
  const MoeGgufSegment gs = moe_gguf_segment<Shared>(tile.expert, p.experts, p.input_size, p.output_size,
                                                     p.gate_routed_format, p.gate_shared_format, slots, chunks,
                                                     p.floor_slots, p.chunk_slots, 0, gsw0, gsw1, gsmeta);
  const MoeGgufSegment us = moe_gguf_segment<Shared>(tile.expert, p.experts, p.input_size, p.output_size,
                                                     p.up_routed_format, p.up_shared_format, slots, chunks,
                                                     p.floor_slots, p.chunk_slots, 1, usw0, usw1, usmeta);
  const uint thread_index = simd_group * 32 + simd_lane;
  bool tables = false;
  moe_formats<4>()(gs.format, [&](auto f) {
    tables |= quant_pair_entries<decltype(f)>(tl, thread_index, Partitions * GGUF_STAGED_THREADS);
  });
  if constexpr (Tables == 2)
    moe_formats<4>()(us.format, [&](auto f) {
      tables |= quant_pair_entries<decltype(f)>(tl + kQuantPairTableEntries, thread_index,
                                                Partitions * GGUF_STAGED_THREADS);
    });
  if (tables) threadgroup_barrier(mem_flags::mem_threadgroup);
  const MoeGgufSegment s = up ? us : gs;
  const float scale = moe_expert_scale(expert_scales, tile.expert, p.experts, up ? 1 : 0);
  threadgroup half2 *table = tl + (Tables == 2 && up ? kQuantPairTableEntries : 0);
  const uint K = p.input_size, N = p.output_size;
  const uint steps = K / GGUF_STAGED_STEP, first = moe_partition_first(steps, partition, Partitions),
             last = moe_partition_first(steps, partition + 1, Partitions);
  // The input row of the tile's live row `row`: of its grouped row, or a narrow step's row of its route in place.
  const auto input_row = [&](uint row) -> uint {
    return p.input_rows ? grouped_routes[tile.first_row + row] / p.routes_per_row : tile.first_row + row;
  };
  const ulong out = ulong(tile.first_row) * N;
  const uint origin = group.x * GGUF_STAGED_COLUMNS;
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  threadgroup bfloat *gate =
      reinterpret_cast<threadgroup bfloat *>(stage + (Partitions > 1 ? kStagedSimdgroupStage / 2 : 0));
  // The tile's live rows, live row `row` at sum row at(row): the gate's sums times its scale parked as bf16, then
  // the up's activation written.
  const auto finish = [&](uint rows, auto sum, auto at) {
    if (partition == 0 && !up)
      for (uint row = 0; row < rows; ++row)
        gate[at(row) * GGUF_STAGED_COLUMNS + simd_lane] = bfloat(sum(at(row)) * scale);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (partition == 0 && up)
      for (uint row = 0; row < rows; ++row)
        output[out + ulong(row) * N + origin + simd_lane] = bfloat(Activation::value(
            float(bfloat(sum(at(row)) * scale)), float(gate[at(row) * GGUF_STAGED_COLUMNS + simd_lane])));
  };
  const auto parked = [&](uint row) { return moe_partition_total<Partitions>(stage, up ? 1 : 0, row, simd_lane); };
  if (kMoeColumnTiles && moe_column_tile(tiles, t)) {
    MoeColumnRows x;
    for (uint row = 0; row < GGUF_COLUMN_ROWS; ++row)
      x.row[row] = input + ulong(input_row(min(row, tile.rows - 1))) * K;
    float sums[GGUF_COLUMN_ROWS];
    moe_gate_up_column(x, tile.rows, K, s, origin + simd_lane, first, last, table, sums);
    if constexpr (Partitions > 1) {
      for (uint row = 0; row < tile.rows; ++row)
        reinterpret_cast<threadgroup float *>(my)[row * GGUF_STAGED_COLUMNS + simd_lane] = sums[row];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      finish(tile.rows, parked, [](uint row) { return row; });
    } else {
      finish(tile.rows, [&](uint row) { return sums[row]; }, [](uint row) { return row; });
    }
    return;
  }
  // The staged tile's matmul runs the tile's grouped rows, or a narrow step's input rows, all of them in one 8- or
  // 16-row matmul, of which the up simdgroup stores its routes' rows: a row's sums are the same bits on every tile's
  // matmul (moe_live_rows).
  device bfloat *x = input + (p.input_rows ? 0 : ulong(tile.first_row) * K);
  const auto at = [&](uint row) { return p.input_rows ? input_row(row) : row; };
  const auto run = [&](auto rows) {
    constexpr ushort R = decltype(rows)::value;
    static_assert(R * GGUF_STAGED_COLUMNS * sizeof(bfloat) <= kStagedSimdgroupStage * sizeof(half),
                  "the gate fits a stage");
    moe_formats<4>()(s.format, [&](auto f) {
      moe_gate_up_staged<decltype(f), R, (Partitions > 1)>(x, s, scale, K, origin, my, table, simd_lane, first,
                                                          last);
    });
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if constexpr (Partitions > 1) {
      static_assert(R * GGUF_STAGED_COLUMNS * (sizeof(float) + sizeof(bfloat)) <= kStagedSimdgroupStage * sizeof(half),
                    "a partition's sums and the gate fit a stage");
      finish(tile.rows, parked, at);
    } else if (up) {
      // Each simdgroup parked bf16 of its sums times its scale in its stage: the gate's is `gate`.
      threadgroup bfloat *park = reinterpret_cast<threadgroup bfloat *>(my);
      for (uint row = 0; row < tile.rows; ++row)
        output[out + ulong(row) * N + origin + simd_lane] = bfloat(Activation::value(
            float(park[at(row) * GGUF_STAGED_COLUMNS + simd_lane]),
            float(gate[at(row) * GGUF_STAGED_COLUMNS + simd_lane])));
    }
  };
  if (p.input_rows)
    moe_live_rows<(Rows > MOE_NARROW_ROWS ? Rows : MOE_NARROW_ROWS)>(p.input_rows, run);
  else
    moe_live_rows<Rows>(tile.rows, run);
}
using MoeExpertGgufGateUpKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *,
                                       device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                       device uchar *, device bfloat *, device const uint *, device const float *,
                                       device const uint *, constant MoeGgufGateUpParams &, constant ulong *, uint2,
                                       uint, uint);
template [[host_name("moe_expert_gguf_m8_gate_up")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<8, 1, 1, MoeSiluGate, 1>;
// The 16-row passes, a decode step's of 9 to 16 rows (ops::selectMoe).
template [[host_name("moe_expert_gguf_m16_gate_up")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<16, 1, 1, MoeSiluGate, 1>;
template [[host_name("moe_expert_gguf_m16_gate_up_mixed")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<16, 2, 1, MoeSiluGate, 1>;
template [[host_name("moe_expert_gguf_m16_gate_up_s2")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<16, 1, 2, MoeSiluGate, 1>;
template [[host_name("moe_expert_gguf_m16_gate_up_mixed_s2")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<16, 2, 2, MoeSiluGate, 1>;
template [[host_name("moe_expert_gguf_m16_gate_up_s0_softcap4_25")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<16, 1, 0, SoftcappedSwiglu<4, 25>, 1>;
template [[host_name("moe_expert_gguf_m16_gate_up_mixed_s0_softcap4_25")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<16, 2, 0, SoftcappedSwiglu<4, 25>, 1>;
template [[host_name("moe_expert_gguf_m32_gate_up")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<32, 1, 1, MoeSiluGate, 1>;
template [[host_name("moe_expert_gguf_m8_gate_up_mixed")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<8, 2, 1, MoeSiluGate, 1>;
template [[host_name("moe_expert_gguf_m32_gate_up_mixed")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<32, 2, 1, MoeSiluGate, 1>;
template [[host_name("moe_expert_gguf_m8_gate_up_s2")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<8, 1, 2, MoeSiluGate, 1>;
template [[host_name("moe_expert_gguf_m32_gate_up_s2")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<32, 1, 2, MoeSiluGate, 1>;
template [[host_name("moe_expert_gguf_m8_gate_up_mixed_s2")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<8, 2, 2, MoeSiluGate, 1>;
template [[host_name("moe_expert_gguf_m32_gate_up_mixed_s2")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<32, 2, 2, MoeSiluGate, 1>;
template [[host_name("moe_expert_gguf_m64_gate_up")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<64, 1, 1, MoeSiluGate, 1>;
template [[host_name("moe_expert_gguf_m64_gate_up_mixed")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<64, 2, 1, MoeSiluGate, 1>;
// A compiled routing's softcapped activation (ops::kCompiledMoeRoutings), without a shared expert.
template [[host_name("moe_expert_gguf_m8_gate_up_s0_softcap4_25")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<8, 1, 0, SoftcappedSwiglu<4, 25>, 1>;
template [[host_name("moe_expert_gguf_m32_gate_up_s0_softcap4_25")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<32, 1, 0, SoftcappedSwiglu<4, 25>, 1>;
template [[host_name("moe_expert_gguf_m8_gate_up_mixed_s0_softcap4_25")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<8, 2, 0, SoftcappedSwiglu<4, 25>, 1>;
template [[host_name("moe_expert_gguf_m32_gate_up_mixed_s0_softcap4_25")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<32, 2, 0, SoftcappedSwiglu<4, 25>, 1>;
template [[host_name("moe_expert_gguf_m64_gate_up_s0_softcap4_25")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<64, 1, 0, SoftcappedSwiglu<4, 25>, 1>;
template [[host_name("moe_expert_gguf_m64_gate_up_mixed_s0_softcap4_25")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<64, 2, 0, SoftcappedSwiglu<4, 25>, 1>;
// The 8-row passes of two K partitions, decode steps' (ops::MoeConfig::gateUpPartitions, downPartitions).
template [[host_name("moe_expert_gguf_m8k2_gate_up")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<8, 1, 1, MoeSiluGate, 2>;
template [[host_name("moe_expert_gguf_m8k2_gate_up_mixed")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<8, 2, 1, MoeSiluGate, 2>;
template [[host_name("moe_expert_gguf_m8k2_gate_up_s2")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<8, 1, 2, MoeSiluGate, 2>;
template [[host_name("moe_expert_gguf_m8k2_gate_up_mixed_s2")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<8, 2, 2, MoeSiluGate, 2>;
template [[host_name("moe_expert_gguf_m8k2_gate_up_s0_softcap4_25")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<8, 1, 0, SoftcappedSwiglu<4, 25>, 2>;
template [[host_name("moe_expert_gguf_m8k2_gate_up_mixed_s0_softcap4_25")]] kernel MoeExpertGgufGateUpKernel
    moe_expert_gguf_gate_up<8, 2, 0, SoftcappedSwiglu<4, 25>, 2>;
