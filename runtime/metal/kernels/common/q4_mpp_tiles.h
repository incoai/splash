#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/split_reduce.h"

// Q4 (4-bit, group 64) tiles shared by dense and MoE projections. A
// threadgroup computes Rows x TileN outputs with fp32 accumulation and a
// per-quant-group scale/bias epilogue.

// Packed Q4 stores 256-column tiles.
constant constexpr ushort kQ4StorageColumns = 256;

// Resolve cooperative fragment traversal once, outside the quant-group loop.
// The compact path is used only after every validity bit matches; a different
// compiler/device layout retains the intrinsic-checked traversal.
enum class Q4Traversal : ushort {
  All, FourOfEight, PrefixAndFourOfEight, HalfPrefix, Guarded
};

template <class Tensor>
__attribute__((always_inline)) inline Q4Traversal
q4_traversal(const thread Tensor &values) {
  const ushort capacity = values.get_capacity();
  bool all = true;
  // Apple9 M24/N128 fragments can have only a contiguous first half valid.
  bool halfPrefix = capacity != 0 && (capacity % 2) == 0;
  bool striped = capacity != 0 && (capacity % 8) == 0;
  bool prefixed = capacity != 0 && (capacity % 16) == 0;
#pragma unroll
  for (ushort i = 0; i < capacity; ++i) {
    const bool valid = values.is_valid_element(i);
    all &= valid;
    halfPrefix &= valid == (i < capacity / 2);
    striped &= valid == ((i & 7) < 4);
    prefixed &= valid == (i < capacity / 2 || ((i & 7) < 4));
  }
  return all ? Q4Traversal::All : striped ? Q4Traversal::FourOfEight
       : prefixed ? Q4Traversal::PrefixAndFourOfEight
       : halfPrefix ? Q4Traversal::HalfPrefix : Q4Traversal::Guarded;
}

template <class Tensor, class Body>
__attribute__((always_inline)) inline void
q4_visit(const thread Tensor &values, Q4Traversal traversal,
         const thread Body &body) {
  if (traversal == Q4Traversal::All) {
#pragma unroll
    for (ushort i = 0; i < values.get_capacity(); ++i) body(i);
  } else if (traversal == Q4Traversal::FourOfEight) {
#pragma unroll
    for (ushort i = 0; i < values.get_capacity() / 2; ++i)
      body(ushort((i / 4) * 8 + i % 4));
  } else if (traversal == Q4Traversal::PrefixAndFourOfEight) {
#pragma unroll
    for (ushort i = 0; i < values.get_capacity() / 2; ++i) body(i);
#pragma unroll
    for (ushort i = 0; i < values.get_capacity() / 4; ++i)
      body(ushort(values.get_capacity() / 2 + (i / 4) * 8 + i % 4));
  } else if (traversal == Q4Traversal::HalfPrefix) {
#pragma unroll
    for (ushort i = 0; i < values.get_capacity() / 2; ++i) body(i);
  } else {
#pragma unroll
    for (ushort i = 0; i < values.get_capacity(); ++i)
      if (values.is_valid_element(i)) body(i);
  }
}

template <ushort Rows = 8, ushort Simdgroups = 8>
inline void q4_store_input_sums(device const bfloat *input, uint input_size,
                                uint input_origin, threadgroup float *sums,
                                uint sum_origin, uint simd_lane,
                                uint simd_group) {
  for (uint row = simd_group; row < Rows; row += Simdgroups) {
    uint origin = row * input_size + input_origin + simd_lane;
    float first = simd_sum(float(input[origin]) + float(input[origin + 32]));
    float second =
        simd_sum(float(input[origin + 64]) + float(input[origin + 96]));
    float third =
        simd_sum(float(input[origin + 128]) + float(input[origin + 160]));
    float fourth =
        simd_sum(float(input[origin + 192]) + float(input[origin + 224]));
    if (simd_lane == 0) {
      sums[sum_origin + row] = first;
      sums[sum_origin + Rows + row] = second;
      sums[sum_origin + 2 * Rows + row] = third;
      sums[sum_origin + 3 * Rows + row] = fourth;
    }
  }
}

// The output at `index` of element i of a tile's fp32 sums, sums_0 the
// projection's (the gate's for GateUp) and sums_1 GateUp's up projection's.
// The projection rounds once to bf16, except into an fp32 destination (a
// plain projection's logits, ops::Projection::destination), which keeps the
// sum unrounded. Then GateUp takes silu(gate) times the bf16 up value,
// MultiplySiluGate multiplies silu of the bf16 gate in `auxiliary` into it,
// and AddResidual adds the residual in `auxiliary`; the result rounds to the
// destination's type Out.
template <bool GateUp, bool AddResidual, bool MultiplySiluGate, class Sums, class Out>
__attribute__((always_inline)) inline void
q4_store_output(thread Sums &sums_0, thread Sums &sums_1, ushort i,
                device bfloat *auxiliary, device Out *output, uint index) {
  float value;
  if constexpr (GateUp) {
    float gate = float(bfloat(sums_0[i]));
    float up = float(bfloat(sums_1[i]));
    value = splash_silu(gate) * up;
  } else if constexpr (MultiplySiluGate) {
    float gate = float(auxiliary[index]);
    value = splash_silu(gate) * float(bfloat(sums_0[i]));
  } else if constexpr (is_same_v<Out, float>) {
    value = sums_0[i];
  } else {
    value = float(bfloat(sums_0[i]));
  }
  if constexpr (AddResidual)
    value += float(auxiliary[index]);
  output[index] = Out(value);
}

// The fp32 sums of a Rows x TileN tile over quant groups
// [first_group, end_group): per group the MPP matmul of the input's 64 values
// with the group's weights, then its scale/bias epilogue, in group order, with
// the input sums of each 256-input block (first_group is a multiple of four)
// double-buffered in `input_sums`, 8 * Rows floats. GateUp accumulates the
// second weight stream alongside. The tile hands its sums to
// store(sums_0, sums_1, traversal).
//
// Pipelined issues two quant groups' matmuls before either epilogue. Narrow
// projections run few threadgroups and are bound by the latency of one group's
// weight load and matmul, so overlapping two hides most of it; wide
// projections lose to the extra live registers. The epilogues still apply in
// group order, so the results are bit-identical to the sequential form.
// Simdgroups is the cooperative scope of the matmul: four simdgroups halve a
// threadgroup to 128 threads so four of them fit a core at the 512-thread
// occupancy knee; the per-element arithmetic is unchanged, so every
// Simdgroups instance of one tile is bit-identical to the others.
template <ushort Rows, ushort TileN, bool GateUp, ushort Simdgroups,
          bool Pipelined, class Store>
__attribute__((always_inline)) inline void q4_mpp_tile_sums(
    device bfloat *input, device uchar *weights_0, device bfloat *scales_0,
    device bfloat *biases_0, device uchar *weights_1, device bfloat *scales_1,
    device bfloat *biases_1, uint input_size, threadgroup float *input_sums,
    uint output_origin, uint first_group, uint end_group, uint simd_lane,
    uint simd_group, const thread Store &store) {
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows},
                  array<int, 2>{1, int(input_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(Rows, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  uint quant_groups = input_size / 64;
  uint tile = output_origin / kQ4StorageColumns;
  uint tile_offset = output_origin % kQ4StorageColumns;
  device uchar *tile_weights_0 =
      weights_0 + ulong(tile) * quant_groups * kQ4StorageColumns * 64 / 2;
  device uchar *tile_weights_1 =
      weights_1 + ulong(tile) * quant_groups * kQ4StorageColumns * 64 / 2;
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b0(
      tile_weights_0 + tile_offset * 32, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b1(
      tile_weights_1 + tile_offset * 32, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  auto a0 = a.slice<64, Rows>(0, 0);
  auto b00 = first_b0.slice<64, TileN>(0, 0);
  auto b10 = first_b1.slice<64, TileN>(0, 0);
  auto accumulated_0 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b00), float>();
  auto accumulated_1 = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b10), float>();
  // MSL 2.22.3 partitions logical elements and makes capacity uniform across
  // participating threads. The descriptor's Rows x TileN destination has no
  // padding when total capacity equals that logical element count.
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
  q4_store_input_sums<Rows, Simdgroups>(input, input_size, first_group * 64,
                                        input_sums,
                                        ((first_group >> 2) & 1) * (4 * Rows),
                                        simd_lane, simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Both steps are inlined so the loop is one body in the AIR; left as calls,
  // the 16- to 32-row tiles measured 1-3% slower.
  auto run_group = [&](uint quant_group,
                       thread decltype(accumulated_0) &partial_0,
                       thread decltype(accumulated_1) &partial_1)
                       __attribute__((always_inline)) {
    uint input_origin = quant_group * 64;
    auto a_slice = a.slice<64, Rows>(input_origin, 0);
    device uchar *group_weights_0 =
        tile_weights_0 +
        (ulong(quant_group) * kQ4StorageColumns + tile_offset) * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b0(
        group_weights_0, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b0_slice = b0.slice<64, TileN>(0, 0);
    operation.run(a_slice, b0_slice, partial_0);
    device uchar *group_weights_1 =
        tile_weights_1 +
        (ulong(quant_group) * kQ4StorageColumns + tile_offset) * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b1(
        group_weights_1, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b1_slice = b1.slice<64, TileN>(0, 0);
    if constexpr (GateUp)
      operation.run(a_slice, b1_slice, partial_1);
  };
  auto finish_group = [&](uint quant_group,
                          thread decltype(accumulated_0) &partial_0,
                          thread decltype(accumulated_1) &partial_1)
                          __attribute__((always_inline)) {
    q4_visit(accumulated_0, traversal,
             [&](ushort i) __attribute__((always_inline)) {
      auto index = accumulated_0.get_multidimensional_index(i);
      uint row = index[1];
      ulong parameter =
          (ulong(tile) * quant_groups + quant_group) * kQ4StorageColumns +
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
    if ((quant_group & 3) == 3 && quant_group + 1 < end_group) {
      uint next_group = (quant_group + 1) >> 2;
      q4_store_input_sums<Rows, Simdgroups>(
          input, input_size, quant_group * 64 + 64, input_sums,
          (next_group & 1) * (4 * Rows), simd_lane, simd_group);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  };
  uint quant_group = first_group;
  if constexpr (Pipelined) {
    for (; quant_group + 1 < end_group; quant_group += 2) {
      decltype(accumulated_0) first_0, second_0;
      decltype(accumulated_1) first_1, second_1;
      run_group(quant_group, first_0, first_1);
      run_group(quant_group + 1, second_0, second_1);
      finish_group(quant_group, first_0, first_1);
      finish_group(quant_group + 1, second_0, second_1);
    }
  }
  for (; quant_group < end_group; ++quant_group) {
    decltype(accumulated_0) partial_0;
    decltype(accumulated_1) partial_1;
    run_group(quant_group, partial_0, partial_1);
    finish_group(quant_group, partial_0, partial_1);
  }
  store(accumulated_0, accumulated_1, traversal);
}

// A Rows x TileN output tile over every quant group, stored by
// q4_store_output.
template <ushort Rows, ushort TileN, bool GateUp, bool AddResidual,
          bool MultiplySiluGate = false, ushort Simdgroups = 8,
          bool Pipelined = false, class Out>
inline void q4_mpp_tile(
    device bfloat *input, device uchar *weights_0, device bfloat *scales_0,
    device bfloat *biases_0, device Out *output_0, device uchar *weights_1,
    device bfloat *scales_1, device bfloat *biases_1, device bfloat *residual,
    uint output_size, uint input_size, threadgroup float *input_sums,
    uint output_origin, uint simd_lane, uint simd_group) {
  q4_mpp_tile_sums<Rows, TileN, GateUp, Simdgroups, Pipelined>(
      input, weights_0, scales_0, biases_0, weights_1, scales_1, biases_1,
      input_size, input_sums, output_origin, 0, input_size / 64, simd_lane,
      simd_group,
      [&](thread auto &accumulated_0, thread auto &accumulated_1,
          Q4Traversal traversal) __attribute__((always_inline)) {
    q4_visit(accumulated_0, traversal, [&](ushort i) {
      auto index = accumulated_0.get_multidimensional_index(i);
      uint output_index = index[1] * output_size + output_origin + index[0];
      q4_store_output<GateUp, AddResidual, MultiplySiluGate>(
          accumulated_0, accumulated_1, i, residual, output_0, output_index);
    });
  });
  // Persistent callers run the next tile on the same scratch straight away,
  // and its prologue rewrites input-sum region 0 while a straggling simdgroup
  // may still read that region in the last quant-group block's epilogue (the
  // last block is even when K % 512 == 256). Every simdgroup finishes first.
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Split-K form of q4_mpp_tile across the grid, one weight stream: threadgroup
// `group` computes column tile group.x over K partition group.y of `splits`,
// whole 256-input blocks whose counts differ by at most one between
// partitions (every partition holds at least one, ops::LinearPlan). It
// publishes its fp32 sums as partials [split][row][column] over every row and
// column of the projection; the last partition of the tile to arrive adds
// them in split order (kernels/common/split_reduce.h) and stores the sums by
// q4_store_output, as the sequential tile does.
//
// Numerics: each partition accumulates its range with the sequential tile's
// per-group terms in group order, so the result differs from the sequential
// tile's only by the association of the fp32 sum (a few range sums added at
// the end instead of one running sum), and not at all with the batch: a row's
// arithmetic does not depend on the tile's other rows, so its bits are the
// same at every Rows. After the single bf16 rounding, cancellation can make
// the difference exceed one output ulp; the tests hold it to the
// operand-magnitude bound of dev/tuning/LinearNumerics.hpp.
template <ushort Rows, ushort TileN, bool AddResidual, bool MultiplySiluGate,
          ushort Simdgroups, class Out>
inline void q4_mpp_tile_grid_split(
    device bfloat *input, device uchar *weights, device bfloat *scales,
    device bfloat *biases, device bfloat *auxiliary, device Out *output,
    device coherent(device) float *partials, device atomic_uint *counters,
    uint output_size, uint input_size, threadgroup float *input_sums,
    threadgroup uint *arrival, uint2 group, uint splits, uint simd_lane,
    uint simd_group) {
  const uint blocks = input_size / 256;
  const uint output_origin = group.x * TileN;
  const ulong split_stride = ulong(Rows) * output_size;
  q4_mpp_tile_sums<Rows, TileN, false, Simdgroups, false>(
      input, weights, scales, biases, weights, scales, biases, input_size,
      input_sums, output_origin, group.y * blocks / splits * 4,
      (group.y + 1) * blocks / splits * 4, simd_lane, simd_group,
      [&](thread auto &accumulated, thread auto &, Q4Traversal traversal)
          __attribute__((always_inline)) {
    q4_visit(accumulated, traversal, [&](ushort i) {
      auto index = accumulated.get_multidimensional_index(i);
      partials[group.y * split_stride + index[1] * output_size +
               output_origin + index[0]] = accumulated[i];
    });
    const uint thread_index = simd_group * 32 + simd_lane;
    if (!split_arrive_last(counters + group.x, splits, thread_index, arrival))
      return;
    q4_visit(accumulated, traversal, [&](ushort i) {
      auto index = accumulated.get_multidimensional_index(i);
      const uint element = index[1] * output_size + output_origin + index[0];
      accumulated[i] = split_sum(
          float(accumulated[i]), group.y, splits,
          [&](uint split) { return partials[split * split_stride + element]; });
      q4_store_output<false, AddResidual, MultiplySiluGate>(
          accumulated, accumulated, i, auxiliary, output, element);
    });
    split_release(counters + group.x, thread_index);
  });
}
