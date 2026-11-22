// Dense bf16 projections (ops/DenseLinear.hpp): output[r][n] =
// epilogue(sum_k input[r][k] weight[n][k] + bias[n]), weights [N][K] as a
// checkpoint stores them. A threadgroup of S simdgroups owns a TileM x TileN
// tile and runs MPP matmuls over static 128-wide K chunks with fp32
// accumulation: static extents avoid per-load bounds checks, accumulation
// stays in the destination cooperative tensor, and a mem_none barrier per
// chunk keeps the simdgroups cache-adjacent along K. An output's sum does
// not depend on the rows beside it, so a row's bits are the same in any
// tile and at any offset (dense_linear_metal_test). A tile reads whole row
// tiles of its input and stores its rows alone. Grid {row tile, output
// tile}.

#include "metal/abi/DenseLinear.h"
#include "metal/kernels/common/activation.h"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// What a tile does with a sum plus its bias: stores it (bf16), stores its
// GELU (bf16), or adds it to the fp32 rows it stores into.
enum DenseEpilogue { DenseNone, DenseGeluTanh, DenseGeluErf, DenseResidual };

template <ushort TileM, ushort TileN, ushort Simdgroups,
          DenseEpilogue Epilogue, typename Output>
inline void dense_linear_tile(
    device bfloat *input,
    device bfloat *weights,
    device bfloat *bias,
    device Output *output,
    constant DenseLinearParams &params,
    uint2 group)
{
    const uint row_origin = group.x * TileM;
    const uint output_origin = group.y * TileN;
    const uint input_size = params.input_size;
    device bfloat *rows = input + ulong(row_origin) * input_size;
    device bfloat *tile_weights = weights + ulong(output_origin) * input_size;
    auto a = tensor(rows,
                    dextents<int, 2>{int(input_size), TileM},
                    array<int, 2>{1, int(input_size)});
    auto b = tensor(tile_weights,
                    dextents<int, 2>{int(input_size), TileN},
                    array<int, 2>{1, int(input_size)});
    constexpr ushort BK = 128;
    constexpr auto descriptor = matmul2d_descriptor(
        TileM, TileN, BK, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
    auto a0 = a.template slice<BK, TileM>(0, 0);
    auto b0 = b.template slice<BK, TileN>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<
        decltype(a0), decltype(b0), float>();
    #pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        accumulated[i] = 0.0f;
    }
    for (uint chunk = 0; chunk < input_size / BK; ++chunk) {
        auto a_slice = a.template slice<BK, TileM>(chunk * BK, 0);
        auto b_slice = b.template slice<BK, TileN>(chunk * BK, 0);
        operation.run(a_slice, b_slice, accumulated);
        threadgroup_barrier(mem_flags::mem_none);
    }

    #pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        if (!accumulated.is_valid_element(i)) continue;
        auto index = accumulated.get_multidimensional_index(i);
        const uint row = row_origin + index[1];
        if (row >= params.rows) continue;
        const uint column = output_origin + index[0];
        float value = accumulated[i] + float(bias[column]);
        if constexpr (Epilogue == DenseGeluTanh) {
            value = splash_gelu_tanh(value);
        } else if constexpr (Epilogue == DenseGeluErf) {
            value = splash_gelu_erf(value);
        }
        device Output *out = output + ulong(row) * params.output_size + column;
        if constexpr (Epilogue == DenseResidual) {
            value += float(*out);
        }
        *out = Output(value);
    }
}

#define DENSE_LINEAR_KERNEL(name, tile_m, tile_n, simdgroups, epilogue,       \
                            output_type)                                      \
kernel void name(                                                             \
    device bfloat *input [[buffer(0)]],                                       \
    device bfloat *weights [[buffer(1)]],                                     \
    device bfloat *bias [[buffer(2)]],                                        \
    device output_type *output [[buffer(3)]],                                 \
    constant DenseLinearParams &params [[buffer(4)]],                         \
    uint2 group [[threadgroup_position_in_grid]])                             \
{                                                                             \
    dense_linear_tile<tile_m, tile_n, simdgroups, epilogue, output_type>(     \
        input, weights, bias, output, params, group);                         \
}

// dense_linear_m<TileM>n<TileN>s<Simdgroups> and its epilogues, on the tiles
// ops/DenseLinear.cpp picks. The residual rows, a tower's residual stream,
// stay fp32: rounded to bf16 at each of the vision tower's 55 updates, they
// carried most of its drift from the fp32 reference.
#define DENSE_LINEAR_TILE(tile, tile_m, tile_n, simdgroups)                   \
DENSE_LINEAR_KERNEL(dense_linear_##tile, tile_m, tile_n, simdgroups,          \
                    DenseNone, bfloat)                                        \
DENSE_LINEAR_KERNEL(dense_linear_##tile##_gelu_tanh, tile_m, tile_n,          \
                    simdgroups, DenseGeluTanh, bfloat)                        \
DENSE_LINEAR_KERNEL(dense_linear_##tile##_gelu_erf, tile_m, tile_n,           \
                    simdgroups, DenseGeluErf, bfloat)                         \
DENSE_LINEAR_KERNEL(dense_linear_##tile##_residual, tile_m, tile_n,           \
                    simdgroups, DenseResidual, float)

DENSE_LINEAR_TILE(m64n64s8, 64, 64, 8)
DENSE_LINEAR_TILE(m64n64s4, 64, 64, 4)
DENSE_LINEAR_TILE(m64n128s4, 64, 128, 4)
