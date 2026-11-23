// The MoE experts' prefill passes on int8 activations (ops/MoE.cpp): the dense prefill's int8 tile
// (kernels/common/gguf_prefill_int8_tile.h) over the grouping's tiles (kernels/shared/moe.metal), each tile's expert
// reached as the staged passes reach it (moe_gguf_segment), in af4g64 or af5g64.
#include "metal/abi/MoE.h"
#include "metal/kernels/common/gguf_prefill_int8_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"

#include <metal_stdlib>
using namespace metal;

// Copies each grouped row's codes and scales from its chunk row's, so every expert tile is a dense matrix of its
// live rows. Threadgroup (tile, part) copies the part's 256 codes of the tile's rows, sixteen rows at once, each
// thread sixteen codes of one row, and its four 64-input groups' scales.
kernel void moe_gather_int8(device const char *codes [[buffer(0)]], device const float2 *scales [[buffer(1)]],
                            device const uint *grouped_routes [[buffer(2)]],
                            device const MoeTileDescriptor *tiles [[buffer(3)]],
                            device const uint *tile_count [[buffer(4)]], device char *grouped_codes [[buffer(5)]],
                            device float2 *grouped_scales [[buffer(6)]],
                            constant MoeGatherInt8Params &p [[buffer(7)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint thread_index [[thread_index_in_threadgroup]]) {
  if (group.x >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.x];
  const uint lane = thread_index % 16, column = group.y * 256 + lane * 16;
  for (uint row = tile.first_row + thread_index / 16; row < tile.first_row + tile.rows; row += 16) {
    const uint source = grouped_routes[row] / p.routes_per_row;
    *reinterpret_cast<device uint4 *>(grouped_codes + ulong(row) * p.input_size + column) =
        *reinterpret_cast<device const uint4 *>(codes + ulong(source) * p.input_size + column);
    if (lane < 4) {
      const uint g = group.y * 4 + lane;
      grouped_scales[ulong(g) * p.routes + row] = scales[ulong(g) * p.rows + source];
    }
  }
}

// An expert pass of the grouping's tiles: threadgroup (x, y) computes column tile x of tile moe_first_tile + y with
// its expert's weights, the tile's rows (at most GGUF_PREFILL_INT8_ROWS, the grouping's tile_rows) as the dense tile's
// one row tile, its sums times its expert's scale of the projection (moe_expert_scale). The up pass (EpUpWithGate)
// multiplies by the gate the gate pass wrote to `output`, each element in place: the thread that writes an element
// has read its gate there.
template <int BITS, GgufEpilogue Ep, uint Shared, uint Projection>
kernel void moe_expert_int8(device const char *codes [[buffer(0)]], device const float2 *scales [[buffer(1)]],
                            device const MoeTileDescriptor *tiles [[buffer(2)]],
                            device const uint *tile_count [[buffer(3)]], device uchar *sw0 [[buffer(4)]],
                            device uchar *sw1 [[buffer(5)]], device uchar *smeta [[buffer(6)]],
                            device bfloat *output [[buffer(7)]], device const uint *slots [[buffer(8)]],
                            device const float *expert_scales [[buffer(9)]],
                            constant MoeInt8ExpertParams &p [[buffer(10)]], constant ulong *chunks [[buffer(11)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            ushort lane [[thread_index_in_simdgroup]],
                            ushort simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint t = moe_first_tile(tile_count, p.tiles) + group.y;
  if (t >= moe_end_tile(tile_count, p.tiles)) return;
  const MoeTileDescriptor tile = tiles[t];
  const MoeGgufSegment s =
      moe_gguf_segment<Shared>(tile.expert, p.experts, p.input_size, p.output_size, p.routed_format, p.shared_format,
                               slots, chunks, p.floor_slots, p.chunk_slots, Projection, sw0, sw1, smeta);
  const ulong out = ulong(tile.first_row) * p.output_size;
  prefill_int8_tile<BITS, Ep>(codes + ulong(tile.first_row) * p.input_size, scales + tile.first_row, s.w0, s.w1,
                              s.meta, output + out, output + out,
                              {p.input_size, tile.rows, p.routes, p.output_size, 0, p.input_size},
                              uint2(group.x, 0), lane, simd_group,
                              moe_expert_scale(expert_scales, tile.expert, p.experts, Projection));
}

using MoeExpertInt8Kernel = void(device const char *, device const float2 *, device const MoeTileDescriptor *,
                                 device const uint *, device uchar *, device uchar *, device uchar *, device bfloat *,
                                 device const uint *, device const float *, constant MoeInt8ExpertParams &,
                                 constant ulong *, uint2, ushort, ushort);
// moe_expert_int8_<format>_<pass><slices suffix>: the gate (plain), up (times the gate's SiLU) and down passes.
#define MOE_EXPERT_INT8(B, f, Shared, suffix)                                                                         \
  template [[host_name("moe_expert_int8_" #f "_gate" suffix)]] kernel MoeExpertInt8Kernel                            \
      moe_expert_int8<B, EpNone, Shared, 0>;                                                                         \
  template [[host_name("moe_expert_int8_" #f "_up" suffix)]] kernel MoeExpertInt8Kernel                              \
      moe_expert_int8<B, EpUpWithGate, Shared, 1>;                                                                   \
  template [[host_name("moe_expert_int8_" #f "_down" suffix)]] kernel MoeExpertInt8Kernel                            \
      moe_expert_int8<B, EpNone, Shared, 2>;
MOE_EXPERT_INT8(4, af4g64, 1, "")
MOE_EXPERT_INT8(5, af5g64, 1, "")
MOE_EXPERT_INT8(4, af4g64, 2, "_s2")
MOE_EXPERT_INT8(5, af5g64, 2, "_s2")
#undef MOE_EXPERT_INT8
