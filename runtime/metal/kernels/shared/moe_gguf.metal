// GGUF MoE experts on the staged decode tile (kernels/common/gguf_staged_tile.h); the Apple9 register form is in
// kernels/decode/linear_gguf_sgmatrix.metal.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_column_tile.h"
#include "metal/kernels/common/gguf_staged_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"

// MoE experts (ops/MoE.cpp; kernels/shared/moe.metal groups the rows): threadgroup (x, y) computes
// 64 columns of tile y with the weights of the tile's expert (moe_gguf_segment) of a block whose shared expert is
// Shared slices (each instance's slices suffix, ops::kCompiledMoeRoutings), in the format the tile picks
// at run time: on a 16-core M5 Pro one run-time-format dispatch over two segments is within -11..+8% of a dispatch
// per format (23040x2048 Q4_K and 92160x512 Q5_K, one to four lanes). aux is the gate of the up pass.
// Two simdgroups each stream their own 32 columns through the decode tile, grid (N / 64, tiles), on 8-row tiles
// (decode steps, short prefill chunks) or 32-row tiles (longer chunks, ops::moeGgufPrefillTile), where a tile runs the
// 8-, 16- or 32-row matmul that holds its live rows (moe_live_rows): an expert's last tile is mostly partial. On the
// 35B's real prefill routes the 32-row tiles take 17-30% less time than 64-row tiles sharing one 64-column stage over
// four 16-row simdgroups (512 and 2048 rows, 16-core M5 Pro). An expert's only tile of at most MOE_GGUF_COLUMN_ROWS
// live rows runs the column tile instead (kernels/common/gguf_column_tile.h, moe_column_tile), each lane one of the
// simdgroup's columns over the live rows: routes spread over many experts leave most decode tiles at one or two rows,
// where the staged tile's padded matmul steps bound it.
template <ushort Rows, GgufEpilogue Ep, uint Shared>
inline void moe_gguf_expert_tile(device bfloat *input, device const MoeTileDescriptor *tiles,
                                 device const uint *tile_count, device uchar *w0, device uchar *w1, device uchar *meta,
                                 device uchar *sw0, device uchar *sw1, device uchar *smeta, device bfloat *output,
                                 device bfloat *aux, device const uint *slots, constant MoeGgufExpertParams &p,
                                 uint2 group, uint simd_lane, uint simd_group, threadgroup half *stage,
                                 threadgroup half2 *tl) {
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const MoeGgufSegment s = moe_gguf_segment<Shared>(tile.expert, p, slots, w0, w1, meta, sw0, sw1, smeta);
  device bfloat *x = input + ulong(group.y) * Rows * p.input_size;
  const ulong out = ulong(group.y) * Rows * p.output_size;
  const uint origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS;
  if (moe_column_tile(tiles, group.y)) {
    float sums[MOE_GGUF_COLUMN_ROWS];
    column_dot_any<MOE_GGUF_COLUMN_ROWS>(s.format, x, tile.rows, p.input_size, s.w0, s.w1, s.meta, origin + simd_lane,
                                         0, p.input_size / 32, tl, simd_group * 32 + simd_lane, GGUF_STAGED_THREADS,
                                         sums);
    for (uint row = 0; row < tile.rows; ++row) {
      const ulong o = out + ulong(row) * p.output_size + origin + simd_lane;
      output[o] = gguf_epilogue<Ep>(sums[row], aux, o);
    }
    return;
  }
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  const auto run = [&](auto rows) {
    constexpr ushort R = decltype(rows)::value;
    auto acc = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, p.input_size, my);
    gguf_zero(acc);
    staged_accumulate_any<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(s.format, x, s.w0, s.w1, s.meta, p.input_size, origin, my, tl,
                                            simd_group * 32 + simd_lane, simd_lane, 0, p.input_size / GGUF_STAGED_STEP, 0, acc);
    gguf_elements(acc, [&](uint row, uint column, float v) {
      const ulong o = out + ulong(row) * p.output_size + origin + column;
      output[o] = gguf_epilogue<Ep>(v, aux, o);
    });
  };
  moe_live_rows<Rows>(tile.rows, run);
}
template <ushort Rows, GgufEpilogue Ep, uint Shared>
kernel void moe_expert_gguf(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                            device const uint *tile_count [[buffer(2)]], device uchar *w0 [[buffer(3)]],
                            device uchar *w1 [[buffer(4)]], device uchar *meta [[buffer(5)]], device uchar *sw0 [[buffer(6)]],
                            device uchar *sw1 [[buffer(7)]], device uchar *smeta [[buffer(8)]],
                            device bfloat *output [[buffer(9)]], device bfloat *aux [[buffer(10)]],
                            device const uint *slots [[buffer(11)]], constant MoeGgufExpertParams &p [[buffer(12)]],
                            uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                            uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries];
  moe_gguf_expert_tile<Rows, Ep, Shared>(input, tiles, tile_count, w0, w1, meta, sw0, sw1, smeta, output, aux, slots,
                                         p, group, simd_lane, simd_group, stage, tl);
}
using MoeExpertGgufKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *, device uchar *,
                                 device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                 device bfloat *, device bfloat *, device const uint *, constant MoeGgufExpertParams &,
                                 uint2, uint, uint);
template [[host_name("moe_expert_gguf_m8_a")]] kernel MoeExpertGgufKernel moe_expert_gguf<8, EpNone, 1>;
template [[host_name("moe_expert_gguf_m8_g")]] kernel MoeExpertGgufKernel moe_expert_gguf<8, EpUpWithGate, 1>;
template [[host_name("moe_expert_gguf_m32_a")]] kernel MoeExpertGgufKernel moe_expert_gguf<32, EpNone, 1>;
template [[host_name("moe_expert_gguf_m32_g")]] kernel MoeExpertGgufKernel moe_expert_gguf<32, EpUpWithGate, 1>;
template [[host_name("moe_expert_gguf_m8_a_s2")]] kernel MoeExpertGgufKernel moe_expert_gguf<8, EpNone, 2>;
template [[host_name("moe_expert_gguf_m8_g_s2")]] kernel MoeExpertGgufKernel moe_expert_gguf<8, EpUpWithGate, 2>;
template [[host_name("moe_expert_gguf_m32_a_s2")]] kernel MoeExpertGgufKernel moe_expert_gguf<32, EpNone, 2>;
template [[host_name("moe_expert_gguf_m32_g_s2")]] kernel MoeExpertGgufKernel moe_expert_gguf<32, EpUpWithGate, 2>;

// The gate and up passes in one (ops/MoE.cpp): threadgroup (x, y) runs a gate simdgroup and an up simdgroup on the same
// 32 columns of tile y, grid (N / 32, tiles), each through its own stage in its own pass's formats, as in
// moe_gguf_expert_tile. Where gate and up share their routed and their shared formats the two read one pair table,
// otherwise (_gate_up_mixed) each its own, a kilobyte more threadgroup memory: two tables on a layer of hidden 2048 and
// 256 experts of width 512, 8 per row (gguf-moe-benchmark), took 1-5% longer to prefill 64- to 2048-row chunks and 0-3%
// to decode on a 20-core M5 Pro (Q4_K/Q5_K/Q8_0), 1-3% on a 12-core M6 (IQ2_XS/IQ3_XXS/Q5_K). Every thread fills the
// tables before one barrier (none without a table), so the simdgroups' walks, of different formats, meet no barrier but
// their own. The gate simdgroup parks bf16(gate) in its stage, which its loop no longer reads, and the up simdgroup
// writes bf16(up) * silu(gate) as the up pass does (gguf_epilogue<EpUpWithGate>), so the intermediate is bitwise the
// two passes'. A pass of few tiles leaves cores idle and every pass ends in a tail; one pass runs gate and up together
// and has one tail. Against the two passes, both on live grids, on a model of that layer's shape in MLX 4-bit and mxfp4
// and in GGUF experts of Q4_K/Q5_K and of IQ2_XS/IQ3_XXS: a 40-core M5 Max decodes 1.1-4.3% faster at two to four lanes
// and prefills chunks of 16 to 512 rows 1.1-5.1% faster, within noise (-1.4..+0.6%) at one lane and on 8-row chunks; a
// 12-core M6 runs 0.5-2.7% and a 40-core M3 Max 0.9-5.5% faster. A GGUF may store a shared expert's gate and up in
// different formats (34 of 48 layers of one 512-expert Q2_0 checkpoint).
// The gate/up pass's column tile (moe_column_tile), out of line. Inlined, its column dots in every format grew the
// 32-row kernel until a 10-core M4 (Apple9) under shader validation summed one element of the staged accumulator
// wrongly (row 0, columns 1, 3, 5 and 7 of each 32-column tile of an F16 shared expert's up; gguf-moe, both shapes),
// the same kernel right without validation, as was the kernel before the column tile and the kernel without its column
// branch.
__attribute__((noinline)) void moe_gate_up_column(device bfloat *x, uint rows, uint K, uint N, MoeGgufSegment s,
                                                  uint origin, uint simd_lane, threadgroup half2 *table, bool up,
                                                  threadgroup bfloat *gate, device bfloat *output, ulong out) {
  float sums[MOE_GGUF_COLUMN_ROWS];
  quant_format_switch(s.format, [&](auto f) {
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
template <ushort Rows, ushort Tables, uint Shared>
kernel void moe_expert_gguf_gate_up(device bfloat *input [[buffer(0)]],
                                    device const MoeTileDescriptor *tiles [[buffer(1)]],
                                    device const uint *tile_count [[buffer(2)]], device uchar *gw0 [[buffer(3)]],
                                    device uchar *gw1 [[buffer(4)]], device uchar *gmeta [[buffer(5)]],
                                    device uchar *gsw0 [[buffer(6)]], device uchar *gsw1 [[buffer(7)]],
                                    device uchar *gsmeta [[buffer(8)]], device uchar *uw0 [[buffer(9)]],
                                    device uchar *uw1 [[buffer(10)]], device uchar *umeta [[buffer(11)]],
                                    device uchar *usw0 [[buffer(12)]], device uchar *usw1 [[buffer(13)]],
                                    device uchar *usmeta [[buffer(14)]], device bfloat *output [[buffer(15)]],
                                    device const uint *slots [[buffer(16)]],
                                    constant MoeGgufGateUpParams &p [[buffer(17)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint simd_lane [[thread_index_in_simdgroup]],
                                    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[kStagedStages]; threadgroup half2 tl[Tables * kQuantPairTableEntries];
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const bool up = simd_group == 1;
  const MoeGgufSegment gs = moe_gguf_segment<Shared>(tile.expert, p.experts, p.input_size, p.output_size, p.gate_routed_format,
                                             p.gate_shared_format, slots, gw0, gw1, gmeta, gsw0, gsw1, gsmeta);
  const MoeGgufSegment us = moe_gguf_segment<Shared>(tile.expert, p.experts, p.input_size, p.output_size, p.up_routed_format,
                                             p.up_shared_format, slots, uw0, uw1, umeta, usw0, usw1, usmeta);
  const uint thread_index = simd_group * 32 + simd_lane;
  bool tables = false;
  quant_format_switch(gs.format, [&](auto f) {
    tables |= quant_pair_entries<decltype(f)>(tl, thread_index, GGUF_STAGED_THREADS);
  });
  if constexpr (Tables == 2)
    quant_format_switch(us.format, [&](auto f) {
      tables |= quant_pair_entries<decltype(f)>(tl + kQuantPairTableEntries, thread_index, GGUF_STAGED_THREADS);
    });
  if (tables) threadgroup_barrier(mem_flags::mem_threadgroup);
  const MoeGgufSegment s = up ? us : gs;
  threadgroup half2 *table = tl + (Tables == 2 && up ? kQuantPairTableEntries : 0);
  const uint K = p.input_size, N = p.output_size;
  device bfloat *x = input + ulong(group.y) * Rows * K;
  const ulong out = ulong(group.y) * Rows * N;
  const uint origin = group.x * GGUF_STAGED_COLUMNS;
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  threadgroup bfloat *gate = reinterpret_cast<threadgroup bfloat *>(stage);
  if (moe_column_tile(tiles, group.y)) {
    moe_gate_up_column(x, tile.rows, K, N, s, origin, simd_lane, table, up, gate, output, out);
    return;
  }
  const auto run = [&](auto rows) {
    constexpr ushort R = decltype(rows)::value;
    static_assert(R * GGUF_STAGED_COLUMNS * sizeof(bfloat) <= kStagedSimdgroupStage * sizeof(half),
                  "the gate fits a stage");
    auto acc = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, K, my);
    gguf_zero(acc);
    quant_format_switch(s.format, [&](auto f) {
      staged_accumulate<decltype(f), R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(
          x, s.w0, s.w1, s.meta, K, origin, my, table, simd_lane, 0, K / GGUF_STAGED_STEP, 0, acc);
    });
    if (!up) gguf_elements(acc, [&](uint row, uint column, float v) { gate[row * GGUF_STAGED_COLUMNS + column] = bfloat(v); });
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (up)
      gguf_elements(acc, [&](uint row, uint column, float v) {
        output[out + ulong(row) * N + origin + column] =
            bfloat(float(bfloat(v)) * splash_silu(float(gate[row * GGUF_STAGED_COLUMNS + column])));
      });
  };
  moe_live_rows<Rows>(tile.rows, run);
}
using MoeExpertGgufGateUpKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *,
                                       device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                       device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                       device uchar *, device uchar *, device bfloat *, device const uint *,
                                       constant MoeGgufGateUpParams &, uint2, uint, uint);
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
