#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/abi/MoE.h"
#include "metal/abi/QuantFormat.h"

// A grouped expert tile of tile_rows rows (kernels/shared/moe.metal) holds
// `rows` live rows first, then padding. An expert pass runs the matmul of the
// smallest of 8, 16 and tile_rows rows that holds the live rows, whose padding
// rows the grouping marks empty, but for the staged passes' column tiles
// (moe_column_tile), which run over their live rows alone. The gather fills
// exactly the rows its passes read: no kernel reads past them.
inline uint moe_matmul_rows(uint rows, uint tile_rows) {
  return min(tile_rows, rows <= 8 ? 8u : rows <= 16 ? 16u : tile_rows);
}
// The staged expert passes run an expert's only tile of at most
// MOE_GGUF_COLUMN_ROWS live rows on the column tile (moe_column_tile in
// abi/MoE.h). The two tiles sum a row's products in different orders, so a
// route's output depends on how many rows the step routes to its expert, up to
// MOE_GGUF_COLUMN_ROWS on the column tile and more on the staged tile: decode
// results of GGUF MoE models on the staged tile are not batch-invariant, since
// a row's experts see the other rows of its step (its lane's draft rows and the
// other lanes'). The grouping's atomics order an expert's routes over its
// tiles, so the tiles of an expert run one tile kind, or a route's output would
// also depend on that order (an expert of nine routes in an 8-row tile and a
// 1-row tile).
// The rows of tile t that the staged expert passes read.
inline uint moe_staged_rows(device const MoeTileDescriptor *tiles, uint t, uint tile_rows) {
  return moe_column_tile(tiles, t) ? tiles[t].rows : moe_matmul_rows(tiles[t].rows, tile_rows);
}

// Runs the pass `run` on the matmul rows of a Max-row tile:
// run(integral_constant<ushort, R>{}) for R = moe_matmul_rows(rows, Max).
template <ushort Max, typename Run>
inline void moe_live_rows(uint rows, Run run) {
  const uint matmul = moe_matmul_rows(rows, Max);
  if constexpr (8 < Max) {
    if (matmul == 8)
      return run(integral_constant<ushort, 8>{});
  }
  if constexpr (16 < Max) {
    if (matmul == 16)
      return run(integral_constant<ushort, 16>{});
  }
  run(integral_constant<ushort, Max>{});
}

// The shared slice that `id` names when the shared expert's slices take the
// ids from `first` on: a route slot of a row, whose top_k routed routes the
// slices follow (first = top_k), or an expert, the slices following the
// routed experts (first = experts). An id below `first` names a routed one
// and wraps past every slice.
inline uint moe_shared_slice(uint id, uint first) { return id - first; }

// A GGUF expert pass's weights for one tile (ops/MoE.cpp, metal/abi/MoE.h):
// the routed experts of the projection are one segment of slots *
// output_size rows, so routed expert e's planes start at tile slots[e] *
// output_size / QUANT_TILE_ROWS (metal/abi/QuantFormat.h), slots being the
// block's slot table; the Shared slices of the shared expert (ids `experts`
// on) have a segment of their own, possibly in another format, slice s's
// planes from tile s * output_size / QUANT_TILE_ROWS on. w1 is the meta
// plane for formats without a second plane, as the host binds it.
struct MoeGgufSegment {
  device uchar *w0;
  device uchar *w1;
  device uchar *meta;
  uint format;
};

// The planes of a segment in format f (id `format`) from row `row` on, of
// `groups` groups of 32 inputs a row.
inline MoeGgufSegment moe_gguf_planes(constant QuantFormat &f, uint format, device uchar *w0, device uchar *w1,
                                      device uchar *meta, ulong row, ulong groups) {
  return {w0 + row * groups * f.plane0_bytes, w1 + row * groups * f.plane1_bytes,
          meta + row * (groups / f.meta_groups) * f.meta_bytes, format};
}
template <uint Shared>
inline MoeGgufSegment moe_gguf_segment(uint expert, uint experts, uint input_size, uint output_size,
                                       uint routed_format, uint shared_format, device const uint *slots,
                                       device uchar *w0, device uchar *w1, device uchar *meta,
                                       device uchar *shared_w0, device uchar *shared_w1,
                                       device uchar *shared_meta) {
  const uint slice = moe_shared_slice(expert, experts);
  if (slice < Shared) {
    constant QuantFormat &f = kQuantFormats[shared_format];
    return moe_gguf_planes(f, shared_format, shared_w0, shared_w1, shared_meta, ulong(slice) * output_size,
                           input_size / 32);
  }
  constant QuantFormat &f = kQuantFormats[routed_format];
  return moe_gguf_planes(f, routed_format, w0, w1, meta, ulong(slots[expert]) * output_size, input_size / 32);
}
template <uint Shared>
inline MoeGgufSegment moe_gguf_segment(uint expert, constant MoeGgufExpertParams &p, device const uint *slots,
                                       device uchar *w0, device uchar *w1, device uchar *meta,
                                       device uchar *shared_w0, device uchar *shared_w1,
                                       device uchar *shared_meta) {
  return moe_gguf_segment<Shared>(expert, p.experts, p.input_size, p.output_size, p.routed_format, p.shared_format,
                                  slots, w0, w1, meta, shared_w0, shared_w1, shared_meta);
}
