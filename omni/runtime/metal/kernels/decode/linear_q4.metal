#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/q4_mpp_tiles.h"

// Decode projections: the grid's params.groups persistent threadgroups
// stride over TileN-wide output tiles, TileCall names the q4_mpp_tiles.h
// instantiation and Sums holds eight input sums per row. The auxiliary buffer
// is the residual the epilogue adds or the gate it applies SiLU to; a plain
// projection reads none (its input stands in) and writes a destination of
// type Out.
#define Q4_DECODE_OUTPUT(Name, TileCall, Sums, TileN, Out)                     \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device Out *output [[buffer(4)]],                           \
                   constant Q4PersistentParams &params [[buffer(5)]],          \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.groups) {             \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, input, params.output_size, params.input_size,           \
               input_sums, tile * TileN, simd_lane, simd_group);               \
    }                                                                          \
  }
// Each plain projection the policy runs, into bf16 and into fp32 (Name_f32:
// the logits, ops::Projection::destination). The plain tiles only
// tune-kernels times write bf16 alone.
#define Q4_DECODE_AFFINE(Name, TileCall, Sums, TileN)                          \
  Q4_DECODE_OUTPUT(Name, TileCall, Sums, TileN, bfloat)                        \
  Q4_DECODE_OUTPUT(Name##_f32, TileCall, Sums, TileN, float)

#define Q4_DECODE_AUXILIARY(Name, Auxiliary, TileCall, Sums, TileN)            \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *Auxiliary [[buffer(4)]],                     \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q4PersistentParams &params [[buffer(6)]],          \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.groups) {             \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, Auxiliary, params.output_size, params.input_size,       \
               input_sums, tile * TileN, simd_lane, simd_group);               \
    }                                                                          \
  }

#define Q4_DECODE_GATE_UP(Name, TileCall, Sums, TileN)                         \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights_0 [[buffer(1)]],                      \
                   device bfloat *scales_0 [[buffer(2)]],                      \
                   device bfloat *biases_0 [[buffer(3)]],                      \
                   device bfloat *output [[buffer(4)]],                        \
                   device uchar *weights_1 [[buffer(5)]],                      \
                   device bfloat *scales_1 [[buffer(6)]],                      \
                   device bfloat *biases_1 [[buffer(7)]],                      \
                   constant Q4PersistentParams &params [[buffer(8)]],          \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.groups) {             \
      TileCall(input, weights_0, scales_0, biases_0, output, weights_1,        \
               scales_1, biases_1, output, params.output_size,                 \
               params.input_size, input_sums, tile * TileN, simd_lane,         \
               simd_group);                                                    \
    }                                                                          \
  }

Q4_DECODE_OUTPUT(decode_linear_q4_n128, (q4_mpp_tile<8, 128, false, false>), 64,
                 128, bfloat)
Q4_DECODE_AFFINE(decode_linear_q4_n128_m16,
                 (q4_mpp_tile<16, 128, false, false>), 128, 128)
Q4_DECODE_OUTPUT(decode_linear_q4_n128_m24,
                 (q4_mpp_tile<24, 128, false, false>), 192, 128, bfloat)
Q4_DECODE_AFFINE(decode_linear_q4_n128_m24_sg4,
                 (q4_mpp_tile<24, 128, false, false, false, 4>), 192, 128)
Q4_DECODE_OUTPUT(decode_linear_q4_n256_m16,
                 (q4_mpp_tile<16, 256, false, false>), 128, 256, bfloat)
// Also the gate pass of three-lane gate/up projections.
Q4_DECODE_OUTPUT(decode_linear_q4_n256_m24,
                 (q4_mpp_tile<24, 256, false, false>), 192, 256, bfloat)
Q4_DECODE_OUTPUT(decode_linear_q4_n256, (q4_mpp_tile<8, 256, false, false>), 64,
                 256, bfloat)
Q4_DECODE_AFFINE(decode_linear_q4_n128_paired,
                 (q4_mpp_tile<8, 128, false, false, false, 8, true>), 64, 128)
// 128 threads: four 8 x 256 tiles per core reach the occupancy knee for very
// wide one-lane projections, with half the input re-reads of N128 tiles.
Q4_DECODE_AFFINE(decode_linear_q4_n256_paired_sg4,
                 (q4_mpp_tile<8, 256, false, false, false, 4, true>), 64, 256)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual_paired, residual,
                    (q4_mpp_tile<8, 128, false, true, false, 8, true>), 64, 128)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual, residual,
                    (q4_mpp_tile<8, 128, false, true>), 64, 128)
Q4_DECODE_GATE_UP(decode_linear_q4_n256_gate_up, (q4_mpp_tile<8, 256, true, false>), 64, 256)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual_m16, residual,
                    (q4_mpp_tile<16, 128, false, true>), 128, 128)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual_m24, residual,
                    (q4_mpp_tile<24, 128, false, true>), 192, 128)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual_m24_sg4, residual,
                    (q4_mpp_tile<24, 128, false, true, false, 4>), 192, 128)
Q4_DECODE_GATE_UP(decode_linear_q4_n256_gate_up_m16,
                  (q4_mpp_tile<16, 256, true, false>), 128, 256)
Q4_DECODE_AFFINE(decode_linear_q4_n128_m32,
                 (q4_mpp_tile<32, 128, false, false>), 256, 128)
Q4_DECODE_AFFINE(decode_linear_q4_n256_m32,
                 (q4_mpp_tile<32, 256, false, false>), 256, 256)
Q4_DECODE_AUXILIARY(decode_linear_q4_n128_residual_m32, residual,
                    (q4_mpp_tile<32, 128, false, true>), 256, 128)
Q4_DECODE_AUXILIARY(decode_linear_q4_n256_up_silu_m32, gate,
                    (q4_mpp_tile<32, 256, false, false, true>), 256, 256)
Q4_DECODE_AUXILIARY(decode_linear_q4_n256_up_silu_m24, gate,
                    (q4_mpp_tile<24, 256, false, false, true>), 192, 256)
#undef Q4_DECODE_AFFINE
#undef Q4_DECODE_OUTPUT
#undef Q4_DECODE_AUXILIARY
#undef Q4_DECODE_GATE_UP
