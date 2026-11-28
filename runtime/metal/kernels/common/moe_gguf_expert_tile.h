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
                                                 uint g_begin, uint g_end, threadgroup half2 *tl, uint thread_index,
                                                 uint threads, thread float (&sums)[GGUF_COLUMN_ROWS],
                                                 Formats formats) {
  column_dot_any<GGUF_COLUMN_ROWS>(format, x, rows, input_size, w0, w1, meta, column, g_begin, g_end, tl,
                                   thread_index, threads, sums, formats);
}

// A tile's K partitions (MoeConfig::gateUpPartitions, downPartitions): a threadgroup of 2 * Partitions simdgroups, simdgroup s summing
// its half's 32 columns (s % 2) over partition s / 2 of K, [partition * steps / Partitions, (partition + 1) *
// steps / Partitions) of its 32-input steps. Each parks its fp32 sums of the tile's rows, [row][column], in its own
// stage, which its walk no longer reads, and a half's partition-0 simdgroup adds the partitions' sums in their
// order (moe_partition_total) after the threadgroup's barrier. Partitions of one simdgroup a half run the walk of
// all of K and park nothing.
inline uint moe_partition_first(uint steps, uint partition, uint partitions) {
  return steps * partition / partitions;
}
template <ushort Partitions>
inline float moe_partition_total(threadgroup half *stage, uint half_index, uint row, uint lane) {
  float total = 0.0f;
  for (uint partition = 0; partition < Partitions; ++partition)
    total += reinterpret_cast<threadgroup const float *>(
        stage + (partition * 2 + half_index) * kStagedSimdgroupStage)[row * GGUF_STAGED_COLUMNS + lane];
  return total;
}

// A GGUF expert pass of projection Projection (metal/abi/MoE.h: 0 gate, 1 up, 2 down) over the grouped tiles: the
// down pass (kernels/shared/moe_gguf.metal), and the gate and up passes apart of gguf-moe's reference. Threadgroup
// (x, y) computes 64 columns of tile y with the weights of the tile's expert (moe_gguf_segment), the routed ones in the
// chunks of the table `chunks`, of a block whose shared expert is Shared slices (each instance's slices suffix,
// ops::kCompiledMoeRoutings), in the format the tile picks at run time (within -11..+8% of a dispatch per format). Two
// simdgroups each stream their own 32 columns through the decode tile, grid (N / 64, tiles), on 8-row tiles (decode
// steps, short prefill chunks), 16-row tiles (a decode step of 9 to 16 rows instead, each expert's routes one tile),
// 32-row tiles (longer chunks) or 64-row tiles (an M5 Pro's chunks whose routes average 64 an expert, ops::selectMoe),
// where a tile runs the 8-, 16-, 32- or 64-row matmul that holds its live rows (moe_live_rows): an expert's last tile
// is mostly partial. Each matmul sums a row's products over each 32-input step as the others do and the steps in one
// order, so a row's output is the same bits on every tile. An expert's only tile of at most GGUF_COLUMN_ROWS live rows
// runs the column tile instead (kernels/common/gguf_column_tile.h, moe_column_tile), each lane one of the simdgroup's
// columns over the live rows: routes spread over many experts leave most decode tiles at one or two rows, where the
// staged tile's padded matmul steps bound it. store(o, v) writes the output element o of the fp32 sum v, which it
// rounds to bf16 first: a partial staged tile hands it the sum it parked as bf16. `formats` switches to the formats the
// pass decodes (moe_gguf.metal's moe_formats, every format for gguf-moe's reference). Without `column_tiles` every tile
// runs the staged tile (moe_gguf.metal's kMoeColumnTiles). Each sum is its expert's scale of the projection
// (moe_expert_scale) times the products'.
template <ushort Rows, uint Shared, uint Projection, ushort Partitions, class Store, class Formats = any_quant_format>
inline void moe_gguf_expert_tile(device bfloat *input, device const MoeTileDescriptor *tiles,
                                 device const uint *tile_count, device uchar *sw0, device uchar *sw1,
                                 device uchar *smeta, device const uint *slots, device const float *expert_scales,
                                 constant MoeGgufExpertParams &p, constant ulong *chunks, uint2 group,
                                 uint simd_lane, uint simd_group, threadgroup half *stage, threadgroup half2 *tl,
                                 Store store, Formats formats = Formats(), bool column_tiles = true) {
  const uint t = moe_first_tile(tile_count, p.tiles) + group.y;
  if (t >= moe_end_tile(tile_count, p.tiles)) return;
  const MoeTileDescriptor tile = tiles[t];
  const MoeGgufSegment s =
      moe_gguf_segment<Shared>(tile.expert, p.experts, p.input_size, p.output_size, p.routed_format, p.shared_format,
                               slots, chunks, p.floor_slots, p.chunk_slots, Projection, sw0, sw1, smeta);
  const float scale = moe_expert_scale(expert_scales, tile.expert, p.experts, Projection);
  device bfloat *x = input + ulong(tile.first_row) * p.input_size;
  const ulong out = ulong(tile.first_row) * p.output_size;
  const uint half_index = simd_group % 2, partition = simd_group / 2;
  const uint origin = group.x * GGUF_TILE_COLUMNS + half_index * GGUF_STAGED_COLUMNS;
  const uint steps = p.input_size / GGUF_STAGED_STEP, first = moe_partition_first(steps, partition, Partitions),
             last = moe_partition_first(steps, partition + 1, Partitions);
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  threadgroup float *parked = reinterpret_cast<threadgroup float *>(my);
  if (column_tiles && moe_column_tile(tiles, t)) {
    float sums[GGUF_COLUMN_ROWS];
    moe_expert_column(s.format, x, tile.rows, p.input_size, s.w0, s.w1, s.meta, origin + simd_lane, first, last, tl,
                      simd_group * 32 + simd_lane, Partitions * GGUF_STAGED_THREADS, sums, formats);
    if constexpr (Partitions > 1) {
      for (uint row = 0; row < tile.rows; ++row) parked[row * GGUF_STAGED_COLUMNS + simd_lane] = sums[row];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (partition == 0)
        for (uint row = 0; row < tile.rows; ++row)
          store(out + ulong(row) * p.output_size + origin + simd_lane,
                moe_partition_total<Partitions>(stage, half_index, row, simd_lane) * scale);
      return;
    }
    for (uint row = 0; row < tile.rows; ++row)
      store(out + ulong(row) * p.output_size + origin + simd_lane, sums[row] * scale);
    return;
  }
  const auto run = [&](auto rows) {
    constexpr ushort R = decltype(rows)::value;
    auto acc = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, p.input_size, my);
    gguf_zero(acc);
    staged_accumulate_any<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(s.format, x, s.w0, s.w1, s.meta, p.input_size, origin, my, tl,
                                            simd_group * 32 + simd_lane, simd_lane, first, last, first, acc,
                                            formats);
    if constexpr (Partitions > 1) {
      static_assert(R * GGUF_STAGED_COLUMNS * sizeof(float) <= kStagedSimdgroupStage * sizeof(half),
                    "a partition's sums fit its stage");
      gguf_elements(acc, [&](uint row, uint column, float v) { parked[row * GGUF_STAGED_COLUMNS + column] = v; });
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (partition == 0)
        for (uint row = 0; row < tile.rows; ++row)
          store(out + ulong(row) * p.output_size + origin + simd_lane,
                moe_partition_total<Partitions>(stage, half_index, row, simd_lane) * scale);
      return;
    }
    // A tile of R live rows stores its sums as they are. A partial tile parks
    // them as bf16 in the simdgroup's stage, which its walk no longer reads,
    // and each lane stores its column of the live rows. On the down pass of a
    // 35B 2048-row chunk (16-core M5 Pro), a guard on each element's row took
    // 17% more time, and parking every tile 2%.
    if (tile.rows == R) {
      gguf_elements(acc, [&](uint row, uint column, float v) {
        store(out + ulong(row) * p.output_size + origin + column, v * scale);
      });
      return;
    }
    static_assert(R * GGUF_STAGED_COLUMNS * sizeof(bfloat) <= kStagedSimdgroupStage * sizeof(half),
                  "the sums fit a stage");
    threadgroup bfloat *park = reinterpret_cast<threadgroup bfloat *>(my);
    gguf_elements(acc, [&](uint row, uint column, float v) {
      park[row * GGUF_STAGED_COLUMNS + column] = bfloat(v * scale);
    });
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < tile.rows; ++row)
      store(out + ulong(row) * p.output_size + origin + simd_lane, float(park[row * GGUF_STAGED_COLUMNS + simd_lane]));
  };
  moe_live_rows<Rows>(tile.rows, run);
}
