#pragma once
#include "metal/kernels/common/gguf_column_tile.h"
#include "metal/kernels/common/gguf_staged_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"

// The column tile of an expert pass (moe_column_tile below): each lane's sums of one column over the live rows, in
// the run-time format `format` among those `formats` switches to. Out of line, as the gate/up pass's moe_gate_up_column
// is: inlined beside the staged tile, a 10-core M4 (Apple9) under shader validation summed one element of the down
// pass's staged accumulator wrongly once the pass decoded its formats alone (moe_gguf.metal's moe_formats).
template <class Formats>
__attribute__((noinline)) void moe_expert_column(uint format, device bfloat *x, uint rows, uint input_size,
                                                 device uchar *w0, device uchar *w1, device uchar *meta, uint column,
                                                 threadgroup half2 *tl, uint thread_index,
                                                 thread float (&sums)[MOE_GGUF_COLUMN_ROWS], Formats formats) {
  column_dot_any<MOE_GGUF_COLUMN_ROWS>(format, x, rows, input_size, w0, w1, meta, column, 0, input_size / 32, tl,
                                       thread_index, GGUF_STAGED_THREADS, sums, formats);
}

// A GGUF expert pass of projection Projection (metal/abi/MoE.h: 0 gate, 1 up, 2 down) over the grouped tiles: the
// down pass (kernels/shared/moe_gguf.metal), and the gate and up passes apart of gguf-moe's reference. Threadgroup
// (x, y) computes 64 columns of tile y with the weights of the tile's expert (moe_gguf_segment), the routed ones in the
// chunks of the table `chunks`, of a block whose shared expert is Shared slices (each instance's slices suffix,
// ops::kCompiledMoeRoutings), in the format the tile picks at run time (within -11..+8% of a dispatch per format). Two
// simdgroups each stream their own 32 columns through the decode tile, grid (N / 64, tiles), on 8-row tiles (decode
// steps, short prefill chunks) or 32-row tiles (longer chunks, ops::selectMoe), where a tile runs the 8-, 16- or
// 32-row matmul that holds its live rows (moe_live_rows): an expert's last tile is mostly partial, so on the 35B's
// prefill routes 32-row tiles take 17-30% less time than 64-row ones. An expert's only tile of at most
// MOE_GGUF_COLUMN_ROWS live rows runs the column tile instead (kernels/common/gguf_column_tile.h, moe_column_tile),
// each lane one of the simdgroup's columns over the live rows: routes spread over many experts leave most decode tiles
// at one or two rows, where the staged tile's padded matmul steps bound it. store(o, v) writes the output element o of
// the fp32 sum v; `formats` switches to the formats the pass decodes (moe_gguf.metal's moe_formats, every format for
// gguf-moe's reference).
template <ushort Rows, uint Shared, uint Projection, class Store, class Formats = any_quant_format>
inline void moe_gguf_expert_tile(device bfloat *input, device const MoeTileDescriptor *tiles,
                                 device const uint *tile_count, device uchar *sw0, device uchar *sw1,
                                 device uchar *smeta, device const uint *slots, constant MoeGgufExpertParams &p,
                                 constant ulong *chunks, uint2 group, uint simd_lane, uint simd_group,
                                 threadgroup half *stage, threadgroup half2 *tl, Store store,
                                 Formats formats = Formats()) {
  const uint t = moe_first_tile(tile_count, p.tiles) + group.y;
  if (t >= moe_end_tile(tile_count, p.tiles)) return;
  const MoeTileDescriptor tile = tiles[t];
  const MoeGgufSegment s =
      moe_gguf_segment<Shared>(tile.expert, p.experts, p.input_size, p.output_size, p.routed_format, p.shared_format,
                               slots, chunks, p.floor_slots, p.chunk_slots, Projection, sw0, sw1, smeta);
  device bfloat *x = input + ulong(t) * Rows * p.input_size;
  const ulong out = ulong(t) * Rows * p.output_size;
  const uint origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS;
  if (moe_column_tile(tiles, t)) {
    float sums[MOE_GGUF_COLUMN_ROWS];
    moe_expert_column(s.format, x, tile.rows, p.input_size, s.w0, s.w1, s.meta, origin + simd_lane, tl,
                      simd_group * 32 + simd_lane, sums, formats);
    for (uint row = 0; row < tile.rows; ++row) store(out + ulong(row) * p.output_size + origin + simd_lane, sums[row]);
    return;
  }
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  const auto run = [&](auto rows) {
    constexpr ushort R = decltype(rows)::value;
    auto acc = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, p.input_size, my);
    gguf_zero(acc);
    staged_accumulate_any<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(s.format, x, s.w0, s.w1, s.meta, p.input_size, origin, my, tl,
                                            simd_group * 32 + simd_lane, simd_lane, 0, p.input_size / GGUF_STAGED_STEP, 0, acc,
                                            formats);
    gguf_elements(acc, [&](uint row, uint column, float v) {
      store(out + ulong(row) * p.output_size + origin + column, v);
    });
  };
  moe_live_rows<Rows>(tile.rows, run);
}
