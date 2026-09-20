#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/q8_mpp_tiles.h"

kernel void prefill_linear_q8_sums32(device const bfloat *input [[buffer(0)]],
                              device float *sums [[buffer(1)]],
                              constant Q4PrefillParams &params [[buffer(2)]],
                              uint tile [[threadgroup_position_in_grid]],
                              uint simd_lane [[thread_index_in_simdgroup]],
                              uint simd_group
                              [[simdgroup_index_in_threadgroup]]) {
  constexpr uint TileM = 32;
  uint quant_groups = params.input_size / 64;
  input += ulong(tile) * TileM * params.input_size;
  sums += ulong(tile) * TileM * quant_groups;
  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    for (uint row = simd_group; row < TileM; row += 8) {
      uint origin = row * params.input_size + quant_group * 64 + simd_lane;
      float sum = simd_sum(float(input[origin]) + float(input[origin + 32]));
      if (simd_lane == 0) {
        sums[quant_group * TileM + row] = sum;
      }
    }
  }
}

constant constexpr ushort PrefillSumBatch = 256;

template <ushort TileM, ushort TileN, ushort Simdgroups, bool AddResidual,
          bool MultiplySiluGate>
inline void q8_mpp_prefill_tile(device bfloat *input, device uchar *weights,
                                device bfloat *scales, device bfloat *biases,
                                device bfloat *output, device bfloat *auxiliary,
                                uint output_size, uint input_size,
                                device const float *precomputed_sums,
                                uint output_origin, uint simd_lane,
                                uint simd_group,
                                threadgroup float *input_sums = nullptr) {
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
  constexpr ushort WeightTileN = 256; // weights are stored in 256-column tiles
  uint tile = output_origin / WeightTileN;
  uint tile_column = output_origin % WeightTileN;
  device uchar *tile_weights =
      weights +
      (ulong(tile) * quant_groups * WeightTileN + tile_column) * 64;
  tensor<device uint8_t, dextents<int, 2>, tensor_inline> first_b(
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
        tile_weights + ulong(quant_group) * WeightTileN * 64;
    tensor<device uint8_t, dextents<int, 2>, tensor_inline> b(
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
          (ulong(tile) * quant_groups + quant_group) * WeightTileN +
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
      value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * value;
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

kernel void prefill_linear_q8_n128(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]], device const float *sums [[buffer(5)]],
    constant Q4PrefillParams &params [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort TileM = 32, TileN = 128;
  threadgroup float input_sums[TileM * PrefillSumBatch];
  uint row_tile = group.x;
  uint output_tile = group.y;
  sums += ulong(row_tile) * TileM * (params.input_size / 64);
  input += ulong(row_tile) * TileM * params.input_size;
  output += ulong(row_tile) * TileM * params.output_size;
  q8_mpp_prefill_tile<TileM, TileN, 8, false, false>(
      input, weights, scales, biases, output, output, params.output_size,
      params.input_size, sums, output_tile * TileN, simd_lane, simd_group,
      input_sums);
}

kernel void prefill_linear_q8_n256(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]], device const float *sums [[buffer(5)]],
    constant Q4PrefillParams &params [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort TileM = 32, TileN = 256;
  threadgroup float input_sums[TileM * PrefillSumBatch];
  uint row_tile = group.x;
  uint output_tile = group.y;
  sums += ulong(row_tile) * TileM * (params.input_size / 64);
  input += ulong(row_tile) * TileM * params.input_size;
  output += ulong(row_tile) * TileM * params.output_size;
  q8_mpp_prefill_tile<TileM, TileN, 8, false, false>(
      input, weights, scales, biases, output, output, params.output_size,
      params.input_size, sums, output_tile * TileN, simd_lane, simd_group,
      input_sums);
}

kernel void prefill_linear_q8_n128_residual(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]],
    constant Q4PrefillParams &params [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort TileM = 32, TileN = 128;
  threadgroup float input_sums[TileM * PrefillSumBatch];
  uint row_tile = group.x;
  uint output_tile = group.y;
  ulong input_offset = ulong(row_tile) * TileM * params.input_size;
  ulong output_offset = ulong(row_tile) * TileM * params.output_size;
  sums += ulong(row_tile) * TileM * (params.input_size / 64);
  q8_mpp_prefill_tile<TileM, TileN, 8, true, false>(
      input + input_offset, weights, scales, biases, output + output_offset,
      residual + output_offset, params.output_size, params.input_size, sums,
      output_tile * TileN, simd_lane, simd_group, input_sums);
}

kernel void prefill_linear_q8_n256_residual(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]],
    constant Q4PrefillParams &params [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort TileM = 32, TileN = 256;
  threadgroup float input_sums[TileM * PrefillSumBatch];
  uint row_tile = group.x;
  uint output_tile = group.y;
  ulong input_offset = ulong(row_tile) * TileM * params.input_size;
  ulong output_offset = ulong(row_tile) * TileM * params.output_size;
  sums += ulong(row_tile) * TileM * (params.input_size / 64);
  q8_mpp_prefill_tile<TileM, TileN, 8, true, false>(
      input + input_offset, weights, scales, biases, output + output_offset,
      residual + output_offset, params.output_size, params.input_size, sums,
      output_tile * TileN, simd_lane, simd_group, input_sums);
}

template <ushort TileM, ushort TileN, ushort Simdgroups>
inline void q8_prefill_write_output_sums(device const bfloat *output,
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

kernel void prefill_linear_q8_n256_up_silu_sums(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *gate [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]],
    device float *output_sums [[buffer(7)]],
    constant Q4PrefillParams &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort TileM = 32, TileN = 256;
  threadgroup float input_sums[TileM * PrefillSumBatch];
  uint row_tile = group.x;
  uint output_tile = group.y;
  ulong input_offset = ulong(row_tile) * TileM * params.input_size;
  ulong output_offset = ulong(row_tile) * TileM * params.output_size;
  sums += ulong(row_tile) * TileM * (params.input_size / 64);
  output_sums += ulong(row_tile) * TileM * (params.output_size / 64);
  q8_mpp_prefill_tile<TileM, TileN, 8, false, true>(
      input + input_offset, weights, scales, biases, output + output_offset,
      gate + output_offset, params.output_size, params.input_size, sums,
      output_tile * TileN, simd_lane, simd_group, input_sums);
  q8_prefill_write_output_sums<TileM, TileN, 8>(
      output + output_offset, output_sums, params.output_size,
      output_tile * TileN, simd_lane, simd_group);
}

kernel void prefill_linear_q8_n128_sg4(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]], device const float *sums [[buffer(5)]],
    constant Q4PrefillParams &params [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort TileM = 32, TileN = 128;
  uint row_tile = group.x;
  uint output_tile = group.y;
  sums += ulong(row_tile) * TileM * (params.input_size / 64);
  input += ulong(row_tile) * TileM * params.input_size;
  output += ulong(row_tile) * TileM * params.output_size;
  q8_mpp_prefill_tile<TileM, TileN, 4, false, false>(
      input, weights, scales, biases, output, output, params.output_size,
      params.input_size, sums, output_tile * TileN, simd_lane, simd_group);
}

kernel void prefill_linear_q8_n128_residual_sg4(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]],
    constant Q4PrefillParams &params [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort TileM = 32, TileN = 128;
  uint row_tile = group.x;
  uint output_tile = group.y;
  ulong input_offset = ulong(row_tile) * TileM * params.input_size;
  ulong output_offset = ulong(row_tile) * TileM * params.output_size;
  sums += ulong(row_tile) * TileM * (params.input_size / 64);
  q8_mpp_prefill_tile<TileM, TileN, 4, true, false>(
      input + input_offset, weights, scales, biases, output + output_offset,
      residual + output_offset, params.output_size, params.input_size, sums,
      output_tile * TileN, simd_lane, simd_group);
}

kernel void prefill_linear_q8_n128_up_silu_sums_sg4(
    device bfloat *input [[buffer(0)]], device uchar *weights [[buffer(1)]],
    device bfloat *scales [[buffer(2)]], device bfloat *biases [[buffer(3)]],
    device bfloat *gate [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]],
    device float *output_sums [[buffer(7)]],
    constant Q4PrefillParams &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort TileM = 32, TileN = 128;
  uint row_tile = group.x;
  uint output_tile = group.y;
  ulong input_offset = ulong(row_tile) * TileM * params.input_size;
  ulong output_offset = ulong(row_tile) * TileM * params.output_size;
  sums += ulong(row_tile) * TileM * (params.input_size / 64);
  output_sums += ulong(row_tile) * TileM * (params.output_size / 64);
  q8_mpp_prefill_tile<TileM, TileN, 4, false, true>(
      input + input_offset, weights, scales, biases, output + output_offset,
      gate + output_offset, params.output_size, params.input_size, sums,
      output_tile * TileN, simd_lane, simd_group);
  q8_prefill_write_output_sums<TileM, TileN, 4>(
      output + output_offset, output_sums, params.output_size,
      output_tile * TileN, simd_lane, simd_group);
}
