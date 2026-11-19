#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/q4_mpp_tiles.h"

kernel void prefill_linear_q4_sums32(device const bfloat *input [[buffer(0)]],
                              device float *sums [[buffer(1)]],
                              constant uint &input_size [[buffer(2)]],
                              uint tile [[threadgroup_position_in_grid]],
                              uint simd_lane [[thread_index_in_simdgroup]],
                              uint simd_group
                              [[simdgroup_index_in_threadgroup]]) {
  constexpr uint TileM = 32;
  uint quant_groups = input_size / 64;
  input += ulong(tile) * TileM * input_size;
  sums += ulong(tile) * TileM * quant_groups;
  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    for (uint row = simd_group; row < TileM; row += 8) {
      uint origin = row * input_size + quant_group * 64 + simd_lane;
      float sum = simd_sum(float(input[origin]) + float(input[origin + 32]));
      if (simd_lane == 0) {
        sums[quant_group * TileM + row] = sum;
      }
    }
  }
}

// Eligible Apple9/Apple10 devices allow 32 KiB of static threadgroup memory.
// Keeping 256 row sums resident lets the 5120/6144-wide projections run with
// one load and the 17408-wide down projection with only one refill.
constant constexpr ushort PrefillSumBatch = 256;

// Eight simdgroups stage row sums in threadgroup memory. Four simdgroups read
// sums from device memory to avoid staging barriers and increase residency.
// Both paths use the same accumulation order.
template <ushort TileM, ushort TileN, ushort Simdgroups, bool AddResidual,
          bool MultiplySiluGate>
inline void q4_mpp_prefill_tile(device bfloat *input, device uchar *weights,
                                device bfloat *scales, device bfloat *biases,
                                device bfloat *output, device bfloat *auxiliary,
                                uint output_size, uint input_size,
                                device const float *precomputed_sums,
                                uint output_origin, uint simd_lane,
                                uint simd_group,
                                threadgroup float *input_sums) {
  constexpr bool StagedSums = Simdgroups == 8;
  auto a = tensor(input, dextents<int, 2>{int(input_size), TileM},
                  array<int, 2>{1, int(input_size)});
  auto c = tensor(output, dextents<int, 2>{int(output_size), TileM},
                  array<int, 2>{1, int(output_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(TileM, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  auto a0 = a.slice<64, TileM>(0, 0);
  uint quant_groups = input_size / 64;
  uint tile = output_origin / kQ4StorageColumns;
  uint tile_column = output_origin % kQ4StorageColumns;
  device uchar *tile_weights =
      weights +
      (ulong(tile) * quant_groups * kQ4StorageColumns + tile_column) * 64 / 2;
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b(
      tile_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
  auto b0 = first_b.slice<64, TileN>(0, 0);
  auto accumulated = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    accumulated[i] = 0.0f;
  }

  auto load_sums = [&](uint start) {
    uint count = min(uint(PrefillSumBatch), quant_groups - start);
    uint thread_index = simd_group * 32 + simd_lane;
    for (uint index = thread_index; index < count * TileM;
         index += Simdgroups * 32) {
      uint quant_group = start + index / TileM;
      uint row = index % TileM;
      input_sums[index] = precomputed_sums[quant_group * TileM + row];
    }
  };
  if constexpr (StagedSums) {
    load_sums(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    uint input_origin = quant_group * 64;
    auto a_slice = a.slice<64, TileM>(input_origin, 0);
    device uchar *group_weights =
        tile_weights + ulong(quant_group) * kQ4StorageColumns * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b(
        group_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b_slice = b.slice<64, TileN>(0, 0);
    auto partial = operation.template get_destination_cooperative_tensor<
        decltype(a_slice), decltype(b_slice), float>();
    operation.run(a_slice, b_slice, partial);

#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
      auto index = accumulated.get_multidimensional_index(i);
      uint row = index[1];
      ulong parameter =
          (ulong(tile) * quant_groups + quant_group) * kQ4StorageColumns +
          tile_column + index[0];
      float sum = StagedSums
          ? input_sums[(quant_group % PrefillSumBatch) * TileM + row]
          : precomputed_sums[quant_group * TileM + row];
      accumulated[i] += partial[i] * float(scales[parameter]) +
                        sum * float(biases[parameter]);
    }
    if (StagedSums && quant_group % PrefillSumBatch == PrefillSumBatch - 1 &&
        quant_group + 1 < quant_groups) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      load_sums(quant_group + 1);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  auto converted = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), bfloat>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    float value = float(bfloat(accumulated[i]));
    if constexpr (MultiplySiluGate) {
      auto index = accumulated.get_multidimensional_index(i);
      float gate =
          float(auxiliary[index[1] * output_size + output_origin + index[0]]);
      value = splash_silu(gate) * value;
    }
    if constexpr (AddResidual) {
      auto index = accumulated.get_multidimensional_index(i);
      value +=
          float(auxiliary[index[1] * output_size + output_origin + index[0]]);
    }
    converted[i] = bfloat(value);
  }
  converted.store(c.slice<TileN, TileM>(output_origin, 0));
}

// The output's Q4 input sums of a TileM x TileN tile, for the projection that
// reads the output next.
template <ushort TileM, ushort TileN, ushort Simdgroups>
inline void q4_prefill_write_output_sums(device const bfloat *output,
                                         device float *output_sums,
                                         uint output_size, uint output_origin,
                                         uint simd_lane, uint simd_group) {
  constexpr uint QuantGroups = TileN / 64;
  threadgroup_barrier(mem_flags::mem_device);
  for (uint task = simd_group; task < TileM * QuantGroups;
       task += Simdgroups) {
    uint row = task / QuantGroups;
    uint local_group = task % QuantGroups;
    uint origin =
        row * output_size + output_origin + local_group * 64 + simd_lane;
    float sum = simd_sum(float(output[origin]) + float(output[origin + 32]));
    if (simd_lane == 0) {
      uint quant_group = output_origin / 64 + local_group;
      output_sums[quant_group * TileM + row] = sum;
    }
  }
}

// What the prefill kernels do with a tile's sums: store them (Plain), add the
// residual at buffer 4 (Residual), or multiply silu of the gate at buffer 4
// in and also write the output's Q4 input sums at buffer 7 (UpSiluSums).
enum class PrefillQ4Epilogue { Plain, Residual, UpSiluSums };

// The 32 x TileN tile of grid position (row tile, column tile). The kernels
// inline it as their bodies: when prefill_linear_q4_n256_up_silu_sums only
// calls it, shader validation on macOS 26.5 (an M5 Pro) gets its output wrong.
template <ushort TileN, ushort Simdgroups, PrefillQ4Epilogue Epilogue>
__attribute__((always_inline)) inline void
q4_prefill(device bfloat *input, device uchar *weights, device bfloat *scales,
           device bfloat *biases, device bfloat *auxiliary,
           device bfloat *output, device const float *sums,
           device float *output_sums, constant Q4Params &params, uint2 group,
           uint simd_lane, uint simd_group, threadgroup float *input_sums) {
  constexpr ushort TileM = 32;
  constexpr bool UpSiluSums = Epilogue == PrefillQ4Epilogue::UpSiluSums;
  const ulong input_offset = ulong(group.x) * TileM * params.input_size;
  const ulong output_offset = ulong(group.x) * TileM * params.output_size;
  sums += ulong(group.x) * TileM * (params.input_size / 64);
  if constexpr (UpSiluSums)
    output_sums += ulong(group.x) * TileM * (params.output_size / 64);
  q4_mpp_prefill_tile<TileM, TileN, Simdgroups,
                      Epilogue == PrefillQ4Epilogue::Residual, UpSiluSums>(
      input + input_offset, weights, scales, biases, output + output_offset,
      auxiliary + output_offset, params.output_size, params.input_size, sums,
      group.y * TileN, simd_lane, simd_group, input_sums);
  if constexpr (UpSiluSums)
    q4_prefill_write_output_sums<TileM, TileN, Simdgroups>(
        output + output_offset, output_sums, params.output_size,
        group.y * TileN, simd_lane, simd_group);
}

// Each epilogue's bindings after the input, weights, scales and biases
// (0-3), and the arguments they give q4_prefill; the plain projection reads no
// auxiliary rows and passes its output in their place.
#define PREFILL_Q4_BUFFERS_Plain                                               \
  device bfloat *output [[buffer(4)]], device const float *sums [[buffer(5)]], \
      constant Q4Params &params [[buffer(6)]]
#define PREFILL_Q4_ARGUMENTS_Plain output, output, sums, nullptr
#define PREFILL_Q4_BUFFERS_Residual                                            \
  device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],  \
      device const float *sums [[buffer(6)]],                                  \
      constant Q4Params &params [[buffer(7)]]
#define PREFILL_Q4_ARGUMENTS_Residual residual, output, sums, nullptr
#define PREFILL_Q4_BUFFERS_UpSiluSums                                          \
  device bfloat *gate [[buffer(4)]], device bfloat *output [[buffer(5)]],      \
      device const float *sums [[buffer(6)]],                                  \
      device float *output_sums [[buffer(7)]],                                 \
      constant Q4Params &params [[buffer(8)]]
#define PREFILL_Q4_ARGUMENTS_UpSiluSums gate, output, sums, output_sums
// Eight simdgroups stage the row sums in threadgroup memory; four read them
// from device memory.
#define PREFILL_Q4_INPUT_SUMS_8 threadgroup float input_sums[32 * PrefillSumBatch]
#define PREFILL_Q4_INPUT_SUMS_4 threadgroup float *const input_sums = nullptr
#define PREFILL_Q4(Name, TileN, Simdgroups, Epilogue)                          \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   PREFILL_Q4_BUFFERS_##Epilogue,                              \
                   uint2 group [[threadgroup_position_in_grid]],               \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    PREFILL_Q4_INPUT_SUMS_##Simdgroups;                                        \
    q4_prefill<TileN, Simdgroups, PrefillQ4Epilogue::Epilogue>(                \
        input, weights, scales, biases, PREFILL_Q4_ARGUMENTS_##Epilogue,       \
        params, group, simd_lane, simd_group, input_sums);                     \
  }
PREFILL_Q4(prefill_linear_q4_n128, 128, 8, Plain)
PREFILL_Q4(prefill_linear_q4_n256, 256, 8, Plain)
PREFILL_Q4(prefill_linear_q4_n128_residual, 128, 8, Residual)
PREFILL_Q4(prefill_linear_q4_n256_residual, 256, 8, Residual)
PREFILL_Q4(prefill_linear_q4_n256_up_silu_sums, 256, 8, UpSiluSums)
PREFILL_Q4(prefill_linear_q4_n128_sg4, 128, 4, Plain)
PREFILL_Q4(prefill_linear_q4_n128_residual_sg4, 128, 4, Residual)
PREFILL_Q4(prefill_linear_q4_n128_up_silu_sums_sg4, 128, 4, UpSiluSums)

// The parameters (a scale, a bias and 64 weights per column and group) that
// a view of the leading input_size inputs of rows of plane_input_size leaves
// unread in the 256-column tiles before the one holding `column`.
inline ulong q4_unread_parameters(constant Q4PrefillLeadingParams &params,
                                  uint column) {
  return ulong(column / kQ4StorageColumns) *
         (params.plane_input_size / 64 - params.matrix.input_size / 64) *
         kQ4StorageColumns;
}

// The residual kernels over a view of the leading inputs of wider weight rows,
// for the Neural Engine FFN split's down projection. Each runs the tile of its
// residual kernel on the planes advanced past the parameters the view leaves
// unread before its column tile, which the tile then addresses as rows of
// input_size inputs.
#define PREFILL_Q4_LEADING(Name, TileN, Simdgroups)                            \
  kernel void Name(device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *residual [[buffer(4)]],                      \
                   device bfloat *output [[buffer(5)]],                        \
                   device const float *sums [[buffer(6)]],                     \
                   constant Q4PrefillLeadingParams &params [[buffer(7)]],      \
                   uint2 group [[threadgroup_position_in_grid]],               \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    PREFILL_Q4_INPUT_SUMS_##Simdgroups;                                        \
    const ulong unread = q4_unread_parameters(params, group.y * TileN);        \
    q4_prefill<TileN, Simdgroups, PrefillQ4Epilogue::Residual>(                \
        input, weights + unread * 64 / 2, scales + unread, biases + unread,    \
        residual, output, sums, nullptr, params.matrix, group, simd_lane,      \
        simd_group, input_sums);                                               \
  }
PREFILL_Q4_LEADING(prefill_linear_q4_n128_residual_leading_inputs, 128, 8)
PREFILL_Q4_LEADING(prefill_linear_q4_n256_residual_leading_inputs, 256, 8)
PREFILL_Q4_LEADING(prefill_linear_q4_n128_residual_sg4_leading_inputs, 128, 4)
#undef PREFILL_Q4_LEADING
#undef PREFILL_Q4
#undef PREFILL_Q4_INPUT_SUMS_4
#undef PREFILL_Q4_INPUT_SUMS_8
#undef PREFILL_Q4_ARGUMENTS_UpSiluSums
#undef PREFILL_Q4_BUFFERS_UpSiluSums
#undef PREFILL_Q4_ARGUMENTS_Residual
#undef PREFILL_Q4_BUFFERS_Residual
#undef PREFILL_Q4_ARGUMENTS_Plain
#undef PREFILL_Q4_BUFFERS_Plain
