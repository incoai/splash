#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/q4_mpp_tiles.h"

// Split-K decode projections across the grid (q4_mpp_tile_grid_split): the
// N128 tile of linear_q4.metal, eight simdgroups, over grid (128-column
// tiles, K splits), every row of the step (8, 16, 24 or 32) in each
// threadgroup's tile, so a tile streams its weights once for every lane. A
// projection whose tiles cannot fill the GPU's cores splits K across
// threadgroups instead of narrowing its tile (ops::Linear's split rule). The
// buffers are the sequential kernels' of linear_q4.metal, then the split
// partials and counters (ops::LinearScratch), then the parameters: plain
// (input, weights, scales, biases, output), and residual or up-with-gate
// (..., residual or gate, output).
#define Q4_SPLIT_THREADS                                                       \
  uint2 group [[threadgroup_position_in_grid]],                                \
      uint2 grid [[threadgroups_per_grid]],                                    \
      uint simd_lane [[thread_index_in_simdgroup]],                            \
      uint simd_group [[simdgroup_index_in_threadgroup]]

// A plain projection reads no auxiliary input (its input stands in) and
// writes a destination of type Out.
#define Q4_SPLIT_OUTPUT(Name, Rows, Out)                                       \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device Out *output [[buffer(4)]],                           \
                   device coherent(device) float *partials [[buffer(5)]],      \
                   device atomic_uint *counters [[buffer(6)]],                 \
                   constant Q4Params &params [[buffer(7)]], Q4_SPLIT_THREADS) { \
    threadgroup float input_sums[8 * Rows];                                    \
    threadgroup uint arrival;                                                  \
    const Q4StoredInput<Rows, 8> in{input, params.input_size};                 \
    q4_mpp_tile_grid_split<Rows, 128, false, false, false, 8, false>(          \
        in, weights, scales, biases, weights, scales, biases, input,           \
        output, partials, counters, params.output_size, params.input_size,     \
        input_sums, &arrival, group, grid.y, simd_lane, simd_group);           \
  }
// Each plain projection into bf16 and into fp32 (Name_f32: the logits,
// ops::Projection::destination).
#define Q4_SPLIT_AFFINE(Name, Rows)                                            \
  Q4_SPLIT_OUTPUT(Name, Rows, bfloat)                                          \
  Q4_SPLIT_OUTPUT(Name##_f32, Rows, float)

#define Q4_SPLIT_AUXILIARY(Name, Rows, AddResidual, MultiplySiluGate)          \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *auxiliary [[buffer(4)]],                     \
                   device bfloat *output [[buffer(5)]],                        \
                   device coherent(device) float *partials [[buffer(6)]],      \
                   device atomic_uint *counters [[buffer(7)]],                 \
                   constant Q4Params &params [[buffer(8)]], Q4_SPLIT_THREADS) { \
    threadgroup float input_sums[8 * Rows];                                    \
    threadgroup uint arrival;                                                  \
    const Q4StoredInput<Rows, 8> in{input, params.input_size};                 \
    q4_mpp_tile_grid_split<Rows, 128, false, AddResidual, MultiplySiluGate, 8, \
                           false>(                                             \
        in, weights, scales, biases, weights, scales, biases, auxiliary,       \
        output, partials, counters, params.output_size, params.input_size,     \
        input_sums, &arrival, group, grid.y, simd_lane, simd_group);           \
  }

// Gate/up runs as a plain gate pass into the gate scratch and an up pass
// whose epilogue multiplies silu of that bf16 gate into the bf16 up value,
// as the sequential three- and four-lane kernels do.
#define Q4_SPLIT_ROWS(Suffix, Rows)                                            \
  Q4_SPLIT_AFFINE(decode_linear_q4_n128_split##Suffix, Rows)                   \
  Q4_SPLIT_AUXILIARY(decode_linear_q4_n128_split_residual##Suffix, Rows,       \
                     true, false)                                              \
  Q4_SPLIT_AUXILIARY(decode_linear_q4_n128_split_up_silu##Suffix, Rows,        \
                     false, true)
Q4_SPLIT_ROWS(, 8)
Q4_SPLIT_ROWS(_m16, 16)
Q4_SPLIT_ROWS(_m24, 24)
Q4_SPLIT_ROWS(_m32, 32)

// One lane's (eight rows') split tile, LinearTile::LaneSplit128: the N128
// tile of eight simdgroups issuing two quant groups' matmuls before their
// epilogues (the paired tile), over one to eight K partitions; one partition
// stores directly. Gate/up accumulates both weight streams in each
// threadgroup, so the pair is one dispatch at every split. The buffers are
// the sequential kernels' (gate/up: the gate's planes, the output, the up
// planes), then the partials and counters, then the parameters.
#define Q4_LANE_SPLIT_CALL(GateUp, AddResidual, weights_1, scales_1, biases_1, \
                           auxiliary)                                          \
  threadgroup float input_sums[64];                                            \
  threadgroup uint arrival;                                                    \
  const Q4StoredInput<8, 8> in{input, params.input_size};                      \
  q4_mpp_tile_grid_split<8, 128, GateUp, AddResidual, false, 8, true>(         \
      in, weights, scales, biases, weights_1, scales_1, biases_1,              \
      auxiliary, output, partials, counters, params.output_size,               \
      params.input_size, input_sums, &arrival, group, grid.y, simd_lane,       \
      simd_group);
#define Q4_LANE_SPLIT_PLAIN(Name, Out)                                         \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device Out *output [[buffer(4)]],                           \
                   device coherent(device) float *partials [[buffer(5)]],      \
                   device atomic_uint *counters [[buffer(6)]],                 \
                   constant Q4Params &params [[buffer(7)]], Q4_SPLIT_THREADS) { \
    Q4_LANE_SPLIT_CALL(false, false, weights, scales, biases, input)           \
  }
Q4_LANE_SPLIT_PLAIN(decode_linear_q4_n128_lane_split, bfloat)
Q4_LANE_SPLIT_PLAIN(decode_linear_q4_n128_lane_split_f32, float)
kernel void decode_linear_q4_n128_lane_split_residual(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device coherent(device) float *partials [[buffer(6)]],
    device atomic_uint *counters [[buffer(7)]],
    constant Q4Params &params [[buffer(8)]], Q4_SPLIT_THREADS) {
  Q4_LANE_SPLIT_CALL(false, true, weights, scales, biases, residual)
}
kernel void decode_linear_q4_n128_lane_split_gate_up(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]], device uchar *up_weights [[buffer(5)]],
    device bfloat *up_scales [[buffer(6)]],
    device bfloat *up_biases [[buffer(7)]],
    device coherent(device) float *partials [[buffer(8)]],
    device atomic_uint *counters [[buffer(9)]],
    constant Q4Params &params [[buffer(10)]], Q4_SPLIT_THREADS) {
  Q4_LANE_SPLIT_CALL(true, false, up_weights, up_scales, up_biases, input)
}

#undef Q4_LANE_SPLIT_PLAIN
#undef Q4_LANE_SPLIT_CALL
#undef Q4_SPLIT_ROWS
#undef Q4_SPLIT_AUXILIARY
#undef Q4_SPLIT_AFFINE
#undef Q4_SPLIT_OUTPUT
#undef Q4_SPLIT_THREADS
