#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/abi/QuantFormat.h"

// A grouped expert tile of tile_rows rows (kernels/shared/moe.metal) holds
// `rows` live rows first, then padding. Every expert pass runs the matmul of
// the smallest of 8, 16 and tile_rows rows that holds the live rows, and the
// grouping and the gather fill exactly those rows: no kernel reads past them.
inline uint moe_matmul_rows(uint rows, uint tile_rows) {
  return min(tile_rows, rows <= 8 ? 8u : rows <= 16 ? 16u : tile_rows);
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

// One expert's Q4 slab, [weights][BF16 scales][BF16 biases] in the same
// affine storage tiles as the dense kernels. Routed experts sit at
// their stride in the packed buffer; expert `experts` is the shared expert,
// whose single slab lives in its own buffer.
struct MoeQ4Slab {
  device uchar *weights;
  device bfloat *scales;
  device bfloat *biases;
};

inline MoeQ4Slab moe_q4_slab(device uchar *packed, device uchar *shared,
                             uint expert, uint experts,
                             ulong expert_stride_bytes, uint output_size,
                             uint input_size) {
  device uchar *base = expert == experts
                           ? shared
                           : packed + ulong(expert) * expert_stride_bytes;
  ulong elements = ulong(output_size) * input_size;
  ulong weight_bytes = elements / 2;
  ulong parameter_bytes = elements / 32;
  return {base, reinterpret_cast<device bfloat *>(base + weight_bytes),
          reinterpret_cast<device bfloat *>(base + weight_bytes +
                                            parameter_bytes)};
}

// A GGUF expert pass's weights for one tile (ops/MoE.cpp, metal/abi/MoE.h):
// every routed expert of the projection is one image segment of experts *
// output_size rows, so expert e's planes start at tile e * output_size /
// QUANT_TILE_ROWS (metal/abi/QuantFormat.h); the shared expert (id `experts`) has a segment
// of its own, possibly in another format. w1 is the meta plane for formats
// without a second plane, as the host binds it.
struct MoeGgufSegment {
  device uchar *w0;
  device uchar *w1;
  device uchar *meta;
  uint format;
};

inline MoeGgufSegment moe_gguf_segment(uint expert, constant MoeGgufExpertParams &p,
                                       device uchar *w0, device uchar *w1, device uchar *meta,
                                       device uchar *shared_w0, device uchar *shared_w1,
                                       device uchar *shared_meta) {
  if (expert == p.experts)
    return {shared_w0, shared_w1, shared_meta, p.shared_format};
  constant QuantFormat &f = kQuantFormats[p.routed_format];
  const ulong rows = ulong(expert) * p.output_size, groups = p.input_size / 32;
  return {w0 + rows * groups * f.plane0_bytes, w1 + rows * groups * f.plane1_bytes,
          meta + rows * (groups / f.meta_groups) * f.meta_bytes, p.routed_format};
}
