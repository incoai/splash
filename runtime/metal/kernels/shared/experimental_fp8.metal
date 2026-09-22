#if __METAL_VERSION__ >= 410
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/ExperimentalFP8.h"

using namespace metal;
using namespace mpp::tensor_ops;

// Optional experimental runtime library; compile separately as Metal 4.1.
// The production Q4 shaders and standalone projection experiment are separate.
//
// Buffer ABI (parameters are appended after the eight resource buffers by
// CommandGraph.add):
//   0: half input[params.rows * K], input[row * K + k].
//   1: uchar E4M3 up weights[N * K], up_weights[n * K + k].
//   2: uchar UE8M0 up scales[N * scale_row_stride].
//   3: uchar E4M3 gate weights[N * K], used only by epilogue 2.
//   4: uchar UE8M0 gate scales[N * scale_row_stride], used only by epilogue 2.
//   5: bfloat auxiliary[params.rows * N]: residual in epilogue 1 or raw gate
//      projection in epilogue 3; an unused dummy in epilogues 0 and 2.
//   6: bfloat output[params.rows * N], output[row * N + n].
//   7: uint diagnostics[4] = {actual_data_row_stride,
//                            actual_scale_row_stride,
//                            expected_scale_row_stride,
//                            status}; status is sticky: 0 = OK,
//                            bit 0 = stride mismatch, bit 1 = invalid parameters.
//   8: ExperimentalFP8Params, five uint32 values (20 bytes), in shared order:
//      {output_size=N, input_size=K, rows, scale_row_stride, epilogue}.
//
// Each logical UE8M0 scale covers 32 adjacent K elements. tensor_blockwise
// exposes no independent scale-plane stride constructor, so this shader checks
// the actual per-tile operands' reported strides against tightly packed K/32
// manifest rows, without 128-byte row padding. This controlled experiment runs
// the matmuls before querying metadata, but checks both operands before any
// output store. A mismatch leaves the current tile untouched. The host uses a
// larger bounded scale test arena; exact mathematical controls validated the
// tight layout for every runtime K under test. Post-run queries test whether
// MPP lowering materializes auxiliary metadata only when the call executes;
// that hypothesis remains unproven until the real projection pilot passes.
//
// Host requirements: K > 0 and K % 128 == 0, N > 0 and N % 64 == 0,
// dimensions fit int, epilogue in [0,3], sufficient initialized storage for
// every participating row, and 128-byte-aligned weight bases. Input/scales use
// aligned Metal buffer bases too. Exactly eight SIMD groups (256 threads) are
// dispatched per projection threadgroup. Decode dispatch grid Y is one;
// params.rows covers the selected 8/16/24/32 rows. Prefill params.rows is padded
// to a multiple of 32, <= 2048, and grid Y covers params.rows / 32 row tiles.
// X may be a full output tile grid or a smaller positive persistent grid.
// Loading clears diagnostics once. Every failure atomically sets a sticky bit;
// valid projections never clear status. The host checks status after each
// command and rejects output when nonzero. Unused gate/aux buffers may be valid
// dummy buffers.

using ExperimentalFP8ScalePlane =
    tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format,
                     32, 1>;
using ExperimentalFP8WeightTensor =
    tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline,
           ExperimentalFP8ScalePlane>;

template <ushort Rows>
__attribute__((always_inline)) inline void experimental_fp8_project(
    device half *input, device uchar *up_weights, device uchar *up_scales,
    device uchar *gate_weights, device uchar *gate_scales,
    device bfloat *auxiliary, device bfloat *output, device uint *diagnostics,
    constant ExperimentalFP8Params &params, uint2 group, uint output_groups,
    uint row_origin, uint thread_index) {
  constexpr ushort TileN = 64;
  const int k = int(params.input_size);
  const bool fused_gate_up = params.epilogue == 2;
  if (!params.input_size || params.input_size % 128 ||
      !params.output_size || params.output_size % TileN ||
      !params.rows || params.epilogue > 3 ||
      !output_groups || output_groups > params.output_size / TileN ||
      row_origin + Rows > params.rows ||
      params.scale_row_stride != params.input_size / 32) {
    if (thread_index == 0)
      atomic_fetch_or_explicit(
          reinterpret_cast<device atomic_uint *>(diagnostics + 3), 2u,
          memory_order_relaxed);
    return;
  }

  constexpr auto descriptor = matmul2d_descriptor(
      Rows, TileN, static_cast<int>(dynamic_extent), false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<8>> operation;
  // Dynamic K is inferred from the complete input and transposed right views.
  // The operation owns internal K tiling; no caller-side quant-group loop or
  // scale/bias affine epilogue remains in the experimental projection.
  auto a = tensor(input + ulong(row_origin) * params.input_size,
                  dextents<int, 2>{k, Rows}, array<int, 2>{1, k});
  const uint output_tiles = params.output_size / TileN;
  for (uint tile = group.x; tile < output_tiles; tile += output_groups) {
    const uint output_origin = tile * TileN;
    ExperimentalFP8ScalePlane up_plane(
        up_scales + ulong(output_origin) * params.scale_row_stride);
    ExperimentalFP8WeightTensor up(
        up_weights + ulong(output_origin) * params.input_size,
        dextents<int, 2>{k, TileN}, array<int, 2>{1, k}, up_plane);
    auto up_accumulated = operation.template get_destination_cooperative_tensor<
        decltype(a), decltype(up), float>();
    operation.run(a, up, up_accumulated);

    if (fused_gate_up) {
      ExperimentalFP8ScalePlane gate_plane(
          gate_scales + ulong(output_origin) * params.scale_row_stride);
      ExperimentalFP8WeightTensor gate(
          gate_weights + ulong(output_origin) * params.input_size,
          dextents<int, 2>{k, TileN}, array<int, 2>{1, k}, gate_plane);
      auto gate_accumulated =
          operation.template get_destination_cooperative_tensor<
              decltype(a), decltype(gate), float>();
      operation.run(a, gate, gate_accumulated);
      // Query both actual operands only after both cooperative matmuls.
      const uint actual_data_stride = uint(up.get_stride(1));
      const uint actual_scale_stride =
          uint(up.get_stride<tensor_plane_scales>(1));
      const uint actual_gate_data_stride = uint(gate.get_stride(1));
      const uint actual_gate_scale_stride =
          uint(gate.get_stride<tensor_plane_scales>(1));
      const bool compatible =
          actual_data_stride == params.input_size &&
          actual_scale_stride == params.scale_row_stride &&
          actual_gate_data_stride == params.input_size &&
          actual_gate_scale_stride == params.scale_row_stride &&
          params.scale_row_stride == params.input_size / 32;
      if (group.x == 0 && group.y == 0 && thread_index == 0 && tile == 0) {
        diagnostics[0] = actual_data_stride;
        diagnostics[1] = actual_scale_stride;
        diagnostics[2] = params.scale_row_stride;
      }
      // All threads validate both actual operands before the SiLU epilogue or
      // any output write, so no partly valid fused projection is stored.
      if (!compatible) {
        if (thread_index == 0)
          atomic_fetch_or_explicit(
              reinterpret_cast<device atomic_uint *>(diagnostics + 3), 1u,
              memory_order_relaxed);
        return;
      }
#pragma unroll
      for (ushort i = 0; i < up_accumulated.get_capacity(); ++i) {
        if (up_accumulated.is_valid_element(i)) {
          const auto index = up_accumulated.get_multidimensional_index(i);
          const ulong output_index =
              ulong(row_origin + index[1]) * params.output_size +
              output_origin + index[0];
          // Production GateUp independently rounds both projections to BF16
          // before SiLU and multiplication. Buffer 1 is up; buffer 3 is gate.
          const float up_value = float(bfloat(up_accumulated[i]));
          const float gate_value = float(bfloat(gate_accumulated[i]));
          const float value =
              gate_value / (1.0f + fast::exp2(-1.44269504089f * gate_value)) *
              up_value;
          output[output_index] = bfloat(value);
        }
      }
    } else {
      const uint actual_data_stride = uint(up.get_stride(1));
      const uint actual_scale_stride =
          uint(up.get_stride<tensor_plane_scales>(1));
      const bool up_compatible =
          actual_data_stride == params.input_size &&
          actual_scale_stride == params.scale_row_stride &&
          params.scale_row_stride == params.input_size / 32;
      if (group.x == 0 && group.y == 0 && thread_index == 0 && tile == 0) {
        diagnostics[0] = actual_data_stride;
        diagnostics[1] = actual_scale_stride;
        diagnostics[2] = params.scale_row_stride;
      }
      if (!up_compatible) {
        if (thread_index == 0)
          atomic_fetch_or_explicit(
              reinterpret_cast<device atomic_uint *>(diagnostics + 3), 1u,
              memory_order_relaxed);
        return;
      }
#pragma unroll
      for (ushort i = 0; i < up_accumulated.get_capacity(); ++i) {
        if (up_accumulated.is_valid_element(i)) {
          const auto index = up_accumulated.get_multidimensional_index(i);
          const ulong output_index =
              ulong(row_origin + index[1]) * params.output_size +
              output_origin + index[0];
          float value = float(bfloat(up_accumulated[i]));
          if (params.epilogue == 1) {
            value += float(auxiliary[output_index]);
          } else if (params.epilogue == 3) {
            const float gate = float(auxiliary[output_index]);
            value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * value;
          }
          output[output_index] = bfloat(value);
        }
      }
    }
  }
}

#define EXPERIMENTAL_FP8_PROJECTION(Name, Rows, RowOrigin)                  \
  kernel void Name(device half *input [[buffer(0)]],                       \
                   device uchar *up_weights [[buffer(1)]],                \
                   device uchar *up_scales [[buffer(2)]],                 \
                   device uchar *gate_weights [[buffer(3)]],              \
                   device uchar *gate_scales [[buffer(4)]],               \
                   device bfloat *auxiliary [[buffer(5)]],                \
                   device bfloat *output [[buffer(6)]],                   \
                   device uint *diagnostics [[buffer(7)]],                \
                   constant ExperimentalFP8Params &params [[buffer(8)]],  \
                   uint2 group [[threadgroup_position_in_grid]],          \
                   uint2 groups [[threadgroups_per_grid]],                \
                   uint thread_index [[thread_index_in_threadgroup]]) {   \
    experimental_fp8_project<Rows>(                                       \
        input, up_weights, up_scales, gate_weights, gate_scales, auxiliary, \
        output, diagnostics, params, group, groups.x, RowOrigin,            \
        thread_index);                                                    \
  }

EXPERIMENTAL_FP8_PROJECTION(experimental_fp8_decode_m8_n64, 8, 0u)
EXPERIMENTAL_FP8_PROJECTION(experimental_fp8_decode_m16_n64, 16, 0u)
EXPERIMENTAL_FP8_PROJECTION(experimental_fp8_decode_m24_n64, 24, 0u)
EXPERIMENTAL_FP8_PROJECTION(experimental_fp8_decode_m32_n64, 32, 0u)
EXPERIMENTAL_FP8_PROJECTION(experimental_fp8_prefill_m32_n64, 32, group.y * 32u)

#undef EXPERIMENTAL_FP8_PROJECTION

// This conversion is a separate graph node. The caller provides one contiguous
// region's element count; input/output backing may be larger shared arenas.
kernel void experimental_fp8_bf16_to_half(
    device const bfloat *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &elements [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < elements)
    output[index] = half(float(input[index]));
}

#endif
