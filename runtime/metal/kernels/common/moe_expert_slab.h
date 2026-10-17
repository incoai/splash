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
// Whether the staged expert passes run tile t on the column tile: at most
// MOE_GGUF_COLUMN_ROWS live rows, its expert's only tile. The two tiles sum a
// row's products in different orders, so a route's output depends on how
// many rows the step routes to its expert, up to MOE_GGUF_COLUMN_ROWS on the
// column tile and more on the staged tile: decode results of GGUF MoE models
// on the staged tile, the 35B's included, are not batch-invariant, since a
// row's experts see the other rows of its step (its lane's draft rows and the
// other lanes'). The grouping's atomics order an expert's routes over its
// tiles, so the tiles of an expert run one tile kind, or a route's output
// would also depend on that order (an expert of nine routes in an 8-row tile
// and a 1-row tile).
inline bool moe_column_tile(device const MoeTileDescriptor *tiles, uint t) {
  return tiles[t].rows <= MOE_GGUF_COLUMN_ROWS && (t == 0 || tiles[t - 1].expert != tiles[t].expert);
}
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

// A GGUF expert pass's weights for one tile (ops/MoE.cpp, metal/abi/MoE.h):
// the routed experts of the projection are one segment of slots *
// output_size rows, so routed expert e's planes start at tile slots[e] *
// output_size / QUANT_TILE_ROWS (metal/abi/QuantFormat.h), slots being the
// block's slot table; the shared expert (id `experts`) has a segment of its
// own, possibly in another format. w1 is the meta plane for formats without
// a second plane, as the host binds it.
struct MoeGgufSegment {
  device uchar *w0;
  device uchar *w1;
  device uchar *meta;
  uint format;
};

inline MoeGgufSegment moe_gguf_segment(uint expert, uint experts, uint input_size, uint output_size,
                                       uint routed_format, uint shared_format, device const uint *slots,
                                       device uchar *w0, device uchar *w1, device uchar *meta,
                                       device uchar *shared_w0, device uchar *shared_w1,
                                       device uchar *shared_meta) {
  if (expert == experts)
    return {shared_w0, shared_w1, shared_meta, shared_format};
  constant QuantFormat &f = kQuantFormats[routed_format];
  const ulong rows = ulong(slots[expert]) * output_size, groups = input_size / 32;
  return {w0 + rows * groups * f.plane0_bytes, w1 + rows * groups * f.plane1_bytes,
          meta + rows * (groups / f.meta_groups) * f.meta_bytes, routed_format};
}
inline MoeGgufSegment moe_gguf_segment(uint expert, constant MoeGgufExpertParams &p, device const uint *slots,
                                       device uchar *w0, device uchar *w1, device uchar *meta,
                                       device uchar *shared_w0, device uchar *shared_w1,
                                       device uchar *shared_meta) {
  return moe_gguf_segment(expert, p.experts, p.input_size, p.output_size, p.routed_format, p.shared_format, slots,
                          w0, w1, meta, shared_w0, shared_w1, shared_meta);
}
