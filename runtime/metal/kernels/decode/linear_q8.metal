#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/q8_mpp_tiles.h"

// Decode projections: persistent threadgroups stride over TileN-wide output
// tiles, TileCall names the q8_mpp_tiles.h instantiation and Sums holds eight
// input sums per row. The auxiliary buffer is the residual the epilogue adds
// or the gate it applies SiLU to.
#define Q8_DECODE_AFFINE(Name, TileCall, Sums, TileN)                          \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *output [[buffer(4)]],                        \
                   constant Q4Params &params [[buffer(5)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, output, params.output_size, params.input_size,          \
               input_sums, tile * TileN, simd_lane, simd_group);               \
    }                                                                          \
  }

#define Q8_DECODE_AUXILIARY(Name, Auxiliary, TileCall, Sums, TileN)            \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *Auxiliary [[buffer(4)]],                     \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q4Params &params [[buffer(6)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, Auxiliary, params.output_size, params.input_size,       \
               input_sums, tile * TileN, simd_lane, simd_group);               \
    }                                                                          \
  }

#define Q8_DECODE_GATE_UP(Name, TileCall, Sums, TileN)                         \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights_0 [[buffer(1)]],                      \
                   device bfloat *scales_0 [[buffer(2)]],                      \
                   device bfloat *biases_0 [[buffer(3)]],                      \
                   device bfloat *output [[buffer(4)]],                        \
                   device uchar *weights_1 [[buffer(5)]],                      \
                   device bfloat *scales_1 [[buffer(6)]],                      \
                   device bfloat *biases_1 [[buffer(7)]],                      \
                   constant Q4Params &params [[buffer(8)]],                    \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.persistent_groups) {  \
      TileCall(input, weights_0, scales_0, biases_0, output, weights_1,        \
               scales_1, biases_1, output, params.output_size,                 \
               params.input_size, input_sums, tile * TileN, simd_lane,         \
               simd_group);                                                    \
    }                                                                          \
  }

Q8_DECODE_AFFINE(decode_linear_q8_n128, (q8_mpp_tile<128, false, false, 256>), 64,
                 128)
Q8_DECODE_AFFINE(decode_linear_q8_n128_m16,
                 (q8_mpp_tile_batched<16, 128, false, false, 256>), 128, 128)
Q8_DECODE_AFFINE(decode_linear_q8_n128_m24,
                 (q8_mpp_tile_batched<24, 128, false, false, 256>), 192, 128)
Q8_DECODE_AFFINE(decode_linear_q8_n128_m24_sg4,
                 (q8_mpp_tile_batched<24, 128, false, false, 256, false, 4>), 192, 128)
Q8_DECODE_AFFINE(decode_linear_q8_n256_m16,
                 (q8_mpp_tile_batched<16, 256, false, false>), 128, 256)
Q8_DECODE_AFFINE(decode_linear_q8_n256_m24,
                 (q8_mpp_tile_batched<24, 256, false, false>), 192, 256)
Q8_DECODE_AFFINE(decode_linear_q8_n256, (q8_mpp_tile<256, false, false>), 64, 256)
Q8_DECODE_AFFINE(decode_linear_q8_n128_paired,
                 (q8_mpp_tile<128, false, false, 256, true>), 64, 128)
Q8_DECODE_AUXILIARY(decode_linear_q8_n128_residual_paired, residual,
                    (q8_mpp_tile<128, false, true, 256, true>), 64, 128)
Q8_DECODE_AUXILIARY(decode_linear_q8_n128_residual, residual,
                    (q8_mpp_tile<128, false, true, 256>), 64, 128)
Q8_DECODE_GATE_UP(decode_linear_q8_n256_gate_up, (q8_mpp_tile<256, true, false>), 64, 256)
Q8_DECODE_AUXILIARY(decode_linear_q8_n128_residual_m16, residual,
                    (q8_mpp_tile_batched<16, 128, false, true, 256>), 128, 128)
Q8_DECODE_AUXILIARY(decode_linear_q8_n128_residual_m24, residual,
                    (q8_mpp_tile_batched<24, 128, false, true, 256>), 192, 128)
Q8_DECODE_AUXILIARY(decode_linear_q8_n128_residual_m24_sg4, residual,
                    (q8_mpp_tile_batched<24, 128, false, true, 256, false, 4>), 192, 128)
Q8_DECODE_GATE_UP(decode_linear_q8_n256_gate_up_m16,
                  (q8_mpp_tile_batched<16, 256, true, false>), 128, 256)
Q8_DECODE_AFFINE(decode_linear_q8_n128_m32,
                 (q8_mpp_tile_batched<32, 128, false, false, 256>), 256, 128)
Q8_DECODE_AFFINE(decode_linear_q8_n256_m32,
                 (q8_mpp_tile_batched<32, 256, false, false>), 256, 256)
Q8_DECODE_AUXILIARY(decode_linear_q8_n128_residual_m32, residual,
                    (q8_mpp_tile_batched<32, 128, false, true, 256>), 256, 128)
Q8_DECODE_AUXILIARY(decode_linear_q8_n256_up_silu_m32, gate,
                    (q8_mpp_tile_batched<32, 256, false, false, 256, true>),
                    256, 256)
Q8_DECODE_AUXILIARY(decode_linear_q8_n256_up_silu_m24, gate,
                    (q8_mpp_tile_batched<24, 256, false, false, 256, true>),
                    192, 256)
#undef Q8_DECODE_AFFINE
#undef Q8_DECODE_AUXILIARY
#undef Q8_DECODE_GATE_UP
