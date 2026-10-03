#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/moe_expert_slab.h"
#include "metal/kernels/common/q4_mpp_tiles.h"

// Grouped prefill uses separate N256 gate, up and down projections. This
// avoids the register pressure of a fused gate/up tile. Gate and up round to
// bf16 before silu(gate) * up.
//
// Tiles with at most 8 or 16 live rows use smaller MPP tiles (moe_live_rows):
// every pass reads and writes the rows of that matmul, which the gather fills,
// and the combine reads only live rows. All row variants share one kernel:
// register allocation follows the 32-row path, while smaller paths save
// instructions. This path has been measured on Apple10; Apple9 performance
// remains unmeasured.
constant constexpr uint PrefillMoeTileRows = 32;

template <ushort Rows, bool MultiplySiluGate>
inline void prefill_moe_expert_tile(device bfloat *grouped_input,
                                    device const MoeTileDescriptor *tiles,
                                    device uchar *packed, device uchar *shared,
                                    device bfloat *gate, device bfloat *output,
                                    constant MoeExpertParams &params,
                                    uint2 group, threadgroup float *input_sums,
                                    uint simd_lane, uint simd_group) {
  const MoeQ4Slab slab = moe_q4_slab(
      packed, shared, tiles[group.y].expert, params.experts,
      params.expert_stride_bytes_0, params.output_size, params.input_size);
  const ulong row = ulong(group.y) * PrefillMoeTileRows;
  q4_mpp_tile<Rows, 256, false, false, MultiplySiluGate>(
      grouped_input + row * params.input_size, slab.weights, slab.scales,
      slab.biases, output + row * params.output_size, slab.weights,
      slab.scales, slab.biases, gate + row * params.output_size,
      params.output_size, params.input_size, input_sums, group.x * 256,
      simd_lane, simd_group);
}

template <bool MultiplySiluGate>
inline void prefill_moe_expert(device bfloat *grouped_input,
                               device const MoeTileDescriptor *tiles,
                               device const uint *tile_count,
                               device uchar *packed, device uchar *shared,
                               device bfloat *gate, device bfloat *output,
                               constant MoeExpertParams &params, uint2 group,
                               threadgroup float *input_sums, uint simd_lane,
                               uint simd_group) {
  if (group.y >= *tile_count)
    return;
  moe_live_rows<PrefillMoeTileRows>(tiles[group.y].rows, [&](auto rows) {
    prefill_moe_expert_tile<decltype(rows)::value, MultiplySiluGate>(
        grouped_input, tiles, packed, shared, gate, output, params, group,
        input_sums, simd_lane, simd_group);
  });
}

// Gate and down passes: one affine projection of each tile.
kernel void prefill_moe_expert_q4_n256_m32(
    device bfloat *grouped_input [[buffer(0)]],
    device const MoeTileDescriptor *tiles [[buffer(1)]],
    device const uint *tile_count [[buffer(2)]],
    device uchar *packed [[buffer(3)]],
    device uchar *shared [[buffer(4)]],
    device bfloat *output [[buffer(5)]],
    constant MoeExpertParams &params [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float input_sums[8 * PrefillMoeTileRows];
  prefill_moe_expert<false>(grouped_input, tiles, tile_count, packed, shared,
                            output, output, params, group, input_sums,
                            simd_lane, simd_group);
}

// Up pass: multiplies each output by silu of the gate pass's bf16 result.
kernel void prefill_moe_expert_q4_n256_up_silu_m32(
    device bfloat *grouped_input [[buffer(0)]],
    device const MoeTileDescriptor *tiles [[buffer(1)]],
    device const uint *tile_count [[buffer(2)]],
    device uchar *up_packed [[buffer(3)]],
    device uchar *shared_up [[buffer(4)]],
    device bfloat *gate [[buffer(5)]],
    device bfloat *output [[buffer(6)]],
    constant MoeExpertParams &params [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float input_sums[8 * PrefillMoeTileRows];
  prefill_moe_expert<true>(grouped_input, tiles, tile_count, up_packed,
                           shared_up, gate, output, params, group, input_sums,
                           simd_lane, simd_group);
}
