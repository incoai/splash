#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/q4_mpp_tiles.h"

// Q8 (8-bit, group 64, StorageN=256) tiles for dense projections.
// A threadgroup computes Rows x TileN outputs with fp32 accumulation and a
// per-quant-group scale/bias epilogue.
// Weights are stored as uint8_t (1 byte per element), with group size 64
// and StorageN=256 tile layout.

template <ushort TileN, bool GateUp, bool AddResidual,
          ushort StorageN = TileN, bool Pipelined = false>
inline void q8_mpp_tile(device bfloat *input, device uchar *weights_0,
                        device bfloat *scales_0, device bfloat *biases_0,
                        device bfloat *output_0, device uchar *weights_1,
                        device bfloat *scales_1, device bfloat *biases_1,
                        device bfloat *residual, uint output_size,
                        uint input_size, threadgroup float *input_sums,
                        uint output_origin, uint simd_lane, uint simd_group) {
  auto a = tensor(input, dextents<int, 2>{int(input_size), 8},
                  array<int, 2>{1, int(input_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(8, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<8>> operation;
  auto a0 = a.slice<64, 8>(0, 0);
  uint quant_groups = input_size / 64;
  uint tile = output_origin / StorageN;
  uint tile_offset = output_origin % StorageN;
  device uchar *tile_weights_0 =
      weights_0 + ulong(tile) * quant_groups * StorageN * 64;
  device uchar *tile_weights_1 =
      weights_1 + ulong(tile) * quant_groups * StorageN * 64;
  tensor<device uint8_t, dextents<int, 2>, tensor_inline> first_b0(
      tile_weights_0 + tile_offset * 64, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  tensor<device uint8_t, dextents<int, 2>, tensor_inline> first_b1(
      tile_weights_1 + tile_offset * 64, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  auto b00 = first_b0.slice<64, TileN>(0, 0);
  auto b10 = first_b1.slice<64, TileN>(0, 0);
  auto accumulated_0 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b00), float>();
  auto accumulated_1 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b10), float>();
  const bool fullyOccupied =
      uint(accumulated_0.get_capacity()) * (8u * 32u) == 8u * TileN;
  const auto traversal = fullyOccupied ? Q4Traversal::All
                                       : q4_traversal(accumulated_0);
  q4_visit(accumulated_0, traversal, [&](ushort i) {
    accumulated_0[i] = 0.0f;
    if constexpr (GateUp)
      accumulated_1[i] = 0.0f;
  });

  q4_store_input_sums(input, input_size, 0, input_sums, 0, simd_lane,
                      simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto run_group = [&](uint quant_group,
                       thread decltype(accumulated_0) &partial_0,
                       thread decltype(accumulated_1) &partial_1) {
    uint input_origin = quant_group * 64;
    auto a_slice = a.slice<64, 8>(input_origin, 0);
    device uchar *group_weights_0 =
        tile_weights_0 + (ulong(quant_group) * StorageN + tile_offset) * 64;
    tensor<device uint8_t, dextents<int, 2>, tensor_inline> b0(
        group_weights_0, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b0_slice = b0.slice<64, TileN>(0, 0);
    operation.run(a_slice, b0_slice, partial_0);
    device uchar *group_weights_1 =
        tile_weights_1 + (ulong(quant_group) * StorageN + tile_offset) * 64;
    tensor<device uint8_t, dextents<int, 2>, tensor_inline> b1(
        group_weights_1, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b1_slice = b1.slice<64, TileN>(0, 0);
    if constexpr (GateUp)
      operation.run(a_slice, b1_slice, partial_1);
  };
  auto finish_group = [&](uint quant_group,
                          thread decltype(accumulated_0) &partial_0,
                          thread decltype(accumulated_1) &partial_1) {
    q4_visit(accumulated_0, traversal,
             [&](ushort i) __attribute__((always_inline)) {
      auto index = accumulated_0.get_multidimensional_index(i);
      uint row = index[1];
      ulong parameter = (ulong(tile) * quant_groups + quant_group) * StorageN +
                        tile_offset + index[0];
      uint sum_offset = ((quant_group >> 2) & 1) * 32 + (quant_group & 3) * 8;
      accumulated_0[i] +=
          partial_0[i] * float(scales_0[parameter]) +
          input_sums[sum_offset + row] * float(biases_0[parameter]);
      if constexpr (GateUp) {
        accumulated_1[i] +=
            partial_1[i] * float(scales_1[parameter]) +
            input_sums[sum_offset + row] * float(biases_1[parameter]);
      }
    });
    if ((quant_group & 3) == 3 && quant_group + 1 < quant_groups) {
      uint next_group = (quant_group + 1) >> 2;
      q4_store_input_sums(input, input_size, quant_group * 64 + 64, input_sums,
                          (next_group & 1) * 32, simd_lane, simd_group);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  };
  if constexpr (Pipelined) {
    uint quant_group = 0;
    for (; quant_group + 1 < quant_groups; quant_group += 2) {
      decltype(accumulated_0) first_0, second_0;
      decltype(accumulated_1) first_1, second_1;
      run_group(quant_group, first_0, first_1);
      run_group(quant_group + 1, second_0, second_1);
      finish_group(quant_group, first_0, first_1);
      finish_group(quant_group + 1, second_0, second_1);
    }
    if (quant_group < quant_groups) {
      decltype(accumulated_0) partial_0;
      decltype(accumulated_1) partial_1;
      run_group(quant_group, partial_0, partial_1);
      finish_group(quant_group, partial_0, partial_1);
    }
  } else {
    for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
      decltype(accumulated_0) partial_0;
      decltype(accumulated_1) partial_1;
      run_group(quant_group, partial_0, partial_1);
      finish_group(quant_group, partial_0, partial_1);
    }
  }

  q4_visit(accumulated_0, traversal, [&](ushort i) {
    auto index = accumulated_0.get_multidimensional_index(i);
    uint output_index = index[1] * output_size + output_origin + index[0];
    float value;
    if constexpr (GateUp) {
      float gate = float(bfloat(accumulated_0[i]));
      float up = float(bfloat(accumulated_1[i]));
      value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * up;
    } else {
      value = float(bfloat(accumulated_0[i]));
    }
    if constexpr (AddResidual)
      value += float(residual[output_index]);
    output_0[output_index] = bfloat(value);
  });
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <ushort Rows, ushort TileN, bool GateUp, bool AddResidual,
          ushort StorageN = TileN, bool MultiplySiluGate = false,
          ushort Simdgroups = 8>
inline void q8_mpp_tile_batched(
    device bfloat *input, device uchar *weights_0, device bfloat *scales_0,
    device bfloat *biases_0, device bfloat *output_0, device uchar *weights_1,
    device bfloat *scales_1, device bfloat *biases_1, device bfloat *residual,
    uint output_size, uint input_size, threadgroup float *input_sums,
    uint output_origin, uint simd_lane, uint simd_group) {
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows},
                  array<int, 2>{1, int(input_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(Rows, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  uint quant_groups = input_size / 64;
  uint tile = output_origin / StorageN;
  uint tile_offset = output_origin % StorageN;
  device uchar *tile_weights_0 =
      weights_0 + ulong(tile) * quant_groups * StorageN * 64;
  device uchar *tile_weights_1 =
      weights_1 + ulong(tile) * quant_groups * StorageN * 64;
  tensor<device uint8_t, dextents<int, 2>, tensor_inline> first_b0(
      tile_weights_0 + tile_offset * 64, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  tensor<device uint8_t, dextents<int, 2>, tensor_inline> first_b1(
      tile_weights_1 + tile_offset * 64, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  auto a0 = a.slice<64, Rows>(0, 0);
  auto b00 = first_b0.slice<64, TileN>(0, 0);
  auto b10 = first_b1.slice<64, TileN>(0, 0);
  auto accumulated_0 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b00), float>();
  auto accumulated_1 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b10), float>();
  const bool fullyOccupied =
      uint(accumulated_0.get_capacity()) * (uint(Simdgroups) * 32u) ==
      uint(Rows) * TileN;
  const auto traversal = fullyOccupied ? Q4Traversal::All
                                       : q4_traversal(accumulated_0);
  q4_visit(accumulated_0, traversal, [&](ushort i) {
    accumulated_0[i] = 0.0f;
    if constexpr (GateUp)
      accumulated_1[i] = 0.0f;
  });
  q4_store_input_sums<Rows, Simdgroups>(input, input_size, 0, input_sums, 0, simd_lane,
                            simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    uint input_origin = quant_group * 64;
    auto a_slice = a.slice<64, Rows>(input_origin, 0);
    device uchar *group_weights_0 =
        tile_weights_0 + (ulong(quant_group) * StorageN + tile_offset) * 64;
    tensor<device uint8_t, dextents<int, 2>, tensor_inline> b0(
        group_weights_0, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b0_slice = b0.slice<64, TileN>(0, 0);
    auto partial_0 = operation.template get_destination_cooperative_tensor<
        decltype(a_slice), decltype(b0_slice), float>();
    operation.run(a_slice, b0_slice, partial_0);
    device uchar *group_weights_1 =
        tile_weights_1 + (ulong(quant_group) * StorageN + tile_offset) * 64;
    tensor<device uint8_t, dextents<int, 2>, tensor_inline> b1(
        group_weights_1, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b1_slice = b1.slice<64, TileN>(0, 0);
    auto partial_1 = operation.template get_destination_cooperative_tensor<
        decltype(a_slice), decltype(b1_slice), float>();
    if constexpr (GateUp)
      operation.run(a_slice, b1_slice, partial_1);
    q4_visit(accumulated_0, traversal,
             [&](ushort i) __attribute__((always_inline)) {
      auto index = accumulated_0.get_multidimensional_index(i);
      uint row = index[1];
      ulong parameter = (ulong(tile) * quant_groups + quant_group) * StorageN +
                        tile_offset + index[0];
      uint sum_offset =
          ((quant_group >> 2) & 1) * (4 * Rows) + (quant_group & 3) * Rows;
      accumulated_0[i] +=
          partial_0[i] * float(scales_0[parameter]) +
          input_sums[sum_offset + row] * float(biases_0[parameter]);
      if constexpr (GateUp) {
        accumulated_1[i] +=
            partial_1[i] * float(scales_1[parameter]) +
            input_sums[sum_offset + row] * float(biases_1[parameter]);
      }
    });
    if ((quant_group & 3) == 3 && quant_group + 1 < quant_groups) {
      uint next_group = (quant_group + 1) >> 2;
      q4_store_input_sums<Rows, Simdgroups>(input, input_size, input_origin + 64,
                                input_sums, (next_group & 1) * (4 * Rows),
                                simd_lane, simd_group);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  q4_visit(accumulated_0, traversal, [&](ushort i) {
    auto index = accumulated_0.get_multidimensional_index(i);
    uint output_index = index[1] * output_size + output_origin + index[0];
    float value;
    if constexpr (GateUp) {
      float gate = float(bfloat(accumulated_0[i]));
      float up = float(bfloat(accumulated_1[i]));
      value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * up;
    } else if constexpr (MultiplySiluGate) {
      float gate = float(residual[output_index]);
      value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) *
              float(bfloat(accumulated_0[i]));
    } else {
      value = float(bfloat(accumulated_0[i]));
    }
    if constexpr (AddResidual)
      value += float(residual[output_index]);
    output_0[output_index] = bfloat(value);
  });
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
