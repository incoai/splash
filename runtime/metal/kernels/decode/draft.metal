#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/draft_context_kv.h"

template <uint Hidden>
inline void draft_conv_phase(device const bfloat *input,
                             device const bfloat *dynamic,
                             device const bfloat *base,
                             device const bfloat *residual,
                             device bfloat *output, bool finish, uint groups,
                             uint group, uint thread_index) {
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr uint ConvGroups = Hidden / 16;
  constexpr uint Dynamic = 4 * ConvGroups;
  for (uint element = group * 256 + thread_index; element < Rows * Hidden;
       element += groups * 256) {
    uint row = element / Hidden;
    uint channel = element % Hidden;
    uint conv_group = channel / 16;
    uint kind = finish ? 1 : 0;
    float value =
        float(input[element]) *
        (float(base[(kind * 2) * Hidden + channel]) +
         float(dynamic[row * Dynamic + (kind * 2) * ConvGroups + conv_group]));
    if (row > 0) {
      value +=
          float(input[(row - 1) * Hidden + channel]) *
          (float(base[(kind * 2 + 1) * Hidden + channel]) +
           float(
               dynamic[row * Dynamic + (kind * 2 + 1) * ConvGroups +
                       conv_group]));
    }
    if (finish)
      value += float(residual[element]);
    output[element] = bfloat(value);
  }
}

inline void draft_qkv_prepare_phase(
    device bfloat *proposal_qkv, device bfloat *queries,
    device const bfloat *q_norm, device const bfloat *k_norm,
    device const float *rope_cos, device const float *rope_sin,
    device bfloat *query_keys, device bfloat *query_values, uint groups,
    threadgroup float *reductions, threadgroup bfloat *head, uint group,
    uint thread_index, uint lane, uint simd_group) {
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS, QHeads = 32, KVHeads = 8,
                 HeadDim = 128;
  constexpr uint QWidth = QHeads * HeadDim, KWidth = KVHeads * HeadDim;
  constexpr uint PackedWidth = QWidth + 2 * KWidth;
  constexpr uint QueryTasks = Rows * QHeads;
  uint tasks = QueryTasks + Rows * KVHeads;

  for (uint element = group * 256 + thread_index;
       element < Rows * KVHeads * HeadDim; element += groups * 256) {
    uint row = element / (KVHeads * HeadDim);
    uint remainder = element % (KVHeads * HeadDim);
    uint attention_head = remainder / HeadDim;
    uint dim = remainder % HeadDim;
    query_values[(attention_head * HeadDim + dim) * Rows + row] =
        proposal_qkv[row * PackedWidth + QWidth + KWidth + remainder];
  }

  for (uint task = group; task < tasks; task += groups) {
    bool query = task < QueryTasks;
    uint local_task = query ? task : task - QueryTasks;
    uint heads = query ? QHeads : KVHeads;
    uint row = local_task / heads;
    uint attention_head = local_task % heads;
    device const bfloat *source =
        query ? proposal_qkv + row * PackedWidth + attention_head * HeadDim
              : proposal_qkv + row * PackedWidth + QWidth +
                    attention_head * HeadDim;
    device const bfloat *weight = query ? q_norm : k_norm;
    device bfloat *destination =
        query ? proposal_qkv + row * PackedWidth + attention_head * HeadDim
              : query_keys + (attention_head * Rows + row) * HeadDim;

    float value = thread_index < HeadDim ? float(source[thread_index]) : 0.0f;
    float square_sum = simd_sum(value * value);
    if (lane == 0)
      reductions[simd_group] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
      float total = 0.0f;
      for (uint i = 0; i < 8; ++i)
        total += reductions[i];
      reductions[0] = rsqrt(total / HeadDim + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < HeadDim) {
      head[thread_index] =
          bfloat(value * reductions[0] * float(weight[thread_index]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < HeadDim / 2) {
      float first = float(head[thread_index]);
      float second = float(head[thread_index + HeadDim / 2]);
      float cosine = rope_cos[row * (HeadDim / 2) + thread_index];
      float sine = rope_sin[row * (HeadDim / 2) + thread_index];
      bfloat first_rotated = bfloat(first * cosine - second * sine);
      bfloat second_rotated = bfloat(second * cosine + first * sine);
      destination[thread_index] = first_rotated;
      destination[thread_index + HeadDim / 2] = second_rotated;
      if (query) {
        uint query_index = (attention_head * Rows + row) * HeadDim;
        queries[query_index + thread_index] = first_rotated;
        queries[query_index + thread_index + HeadDim / 2] = second_rotated;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void draft_context_kv_commit(
    device const bfloat *context_qkv [[buffer(0)]],
    device const bfloat *k_norm [[buffer(1)]],
    device const float *rope_cos [[buffer(2)]],
    device const float *rope_sin [[buffer(3)]],
    device bfloat *keys0 [[buffer(4)]], device bfloat *keys1 [[buffer(5)]],
    device bfloat *keys2 [[buffer(6)]], device bfloat *keys3 [[buffer(7)]],
    device bfloat *values0 [[buffer(8)]],
    device bfloat *values1 [[buffer(9)]],
    device bfloat *values2 [[buffer(10)]],
    device bfloat *values3 [[buffer(11)]],
    device const uint *retained [[buffer(12)]],
    constant DraftContextBatchParams &params [[buffer(13)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint KVHeads = 8;
  constexpr uint PackedWidth = 6144;
  constexpr uint RopeLaneStride = Rows * 64;
  uint batch = group / (Rows * KVHeads);
  uint task = group % (Rows * KVHeads);
  if (batch >= params.lanes)
    return;
  device bfloat *keys =
      batch == 0 ? keys0 : (batch == 1 ? keys1 : (batch == 2 ? keys2 : keys3));
  device bfloat *values = batch == 0
                              ? values0
                              : (batch == 1 ? values1
                                            : (batch == 2 ? values2 : values3));
  DraftContextParams lane_params{Rows, params.cache_stride,
                                  params.start_position[batch], params.paged, params.pages[batch]};
  threadgroup float reductions[8];
  threadgroup bfloat normalized[128];
  draft_context_kv_phase(
      context_qkv + ulong(batch) * Rows * PackedWidth, k_norm,
      rope_cos + ulong(batch) * RopeLaneStride,
      rope_sin + ulong(batch) * RopeLaneStride, keys, values, lane_params,
      min(retained[batch], Rows), task, thread_index, lane, simd_group,
      reductions, normalized);
}

// One split of the attention of one KV head over the 32 query rows of one
// lane (four query heads x eight proposals). The live window occupies one or
// two physical ranges of the ring; the ring-aligned tiles touching them are
// dealt round-robin to the splits, so every context length spreads over all
// splits and no tile reads past the ring. Each split runs an online softmax
// over its tiles, the last split adds the eight current rows, and the
// unnormalized fp32 accumulator with its row maxima and sums lands in
// `partial` for the fixed-order reduce.
inline void draft_attention_split_phase(
    device bfloat *queries, device bfloat *keys, device bfloat *values,
    device bfloat *query_keys, device bfloat *query_values,
    device float *partial, uint cache_stride, uint cache_length, uint split,
    bool paged, constant DraftKvAddresses &pages, uint head,
    uint splits, threadgroup float *score_storage,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, uint thread_index, uint lane,
    uint simd_group) {
  constexpr ushort M = 32, N = 128, D = 128, TileK = 64;
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS, Window = SPLASH_DRAFT_SLIDING_WINDOW;
  uint common_start =
      cache_length >= Window - 1 ? cache_length - (Window - 1) : 0;
  uint old_count = cache_length - common_start;
  uint physical_start = common_start % Window;
  // Live tiles: [0, wrapped) when the window wraps, then [resume, tail_end);
  // a tile the wrap and the tail both touch is visited once.
  uint physical_end = physical_start + old_count;
  uint wrapped =
      physical_end > Window ? (physical_end - Window + N - 1) / N : 0;
  uint tail_end = (min(physical_end, Window) + N - 1) / N;
  uint resume = max(physical_start / N, wrapped);
  uint live_tiles = wrapped + tail_end - resume;
  // A split with no tile that is not the last one has nothing to add: it
  // leaves only row maxima of -inf, which the reduce skips, instead of
  // loading Q and storing a zero accumulator at short contexts.
  if (split >= live_tiles && split + 1 != splits) {
    if (thread_index < M)
      partial[M * D + thread_index] = -INFINITY;
    return;
  }
  auto qt = tensor(queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, TileK, false, true, false);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  auto q0 = qt.slice<TileK, M>(0, 0);
  auto pt = tensor(score_storage, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = pt.slice<TileK, M>(0, 0);
  auto first_v = tensor(values, dextents<int, 2>{N, D},
                        array<int, 2>{1, int(cache_stride)});
  auto first_v0 = first_v.slice<TileK, D>(0, 0);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, TileK, false, true, false);
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(p0), decltype(first_v0), float>();
  for (ushort i = 0; i < running.get_capacity(); ++i)
    running[i] = 0.0f;
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint tile = split; tile < live_tiles; tile += splits) {
    uint slot = (tile < wrapped ? tile : resume + tile - wrapped) * N;
    device bfloat *tile_keys = paged
        ? reinterpret_cast<device bfloat *>(pages.keys[slot / N]) + head * N * D
        : keys + slot * D;
    device bfloat *tile_values = paged
        ? reinterpret_cast<device bfloat *>(pages.values[slot / N]) + head * N * D
        : values + slot;
    auto kt = tensor(tile_keys, dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto k0 = kt.slice<TileK, N>(0, 0);
    auto scores =
        qk.template get_destination_cooperative_tensor<decltype(q0),
                                                       decltype(k0), float>();
    for (ushort i = 0; i < scores.get_capacity(); ++i)
      scores[i] = 0.0f;
    for (ushort chunk = 0; chunk < D / TileK; ++chunk) {
      auto qs = qt.slice<TileK, M>(chunk * TileK, 0);
      auto ks = kt.slice<TileK, N>(chunk * TileK, 0);
      auto partial_scores = qk.template get_destination_cooperative_tensor<
          decltype(qs), decltype(ks), float>();
      qk.run(qs, ks, partial_scores);
      for (ushort i = 0; i < scores.get_capacity(); ++i) {
        scores[i] += partial_scores[i];
      }
    }
    scores.store(pt.slice<N, M>(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort batch = 0; batch < M; batch += 8) {
      ushort matrix_row = batch + simd_group;
      uint proposal_row = matrix_row % Rows;
      uint query_position = cache_length + proposal_row;
      uint row_start =
          query_position >= Window - 1 ? query_position - (Window - 1) : 0;
      uint hidden_prefix = row_start - common_start;
      float local_scores[4];
      float tile_max = -INFINITY;
      for (ushort i = 0; i < 4; ++i) {
        uint column = lane + i * 32;
        uint key = slot + column;
        uint logical_key = key >= physical_start
                               ? key - physical_start
                               : key + Window - physical_start;
        bool valid = logical_key < old_count && logical_key >= hidden_prefix;
        float score = score_storage[matrix_row * N + column] * 0.08838834765f;
        local_scores[i] = valid ? score : -INFINITY;
        tile_max = max(tile_max, local_scores[i]);
      }
      tile_max = simd_max(tile_max);
      float next_max = max(row_max[matrix_row], tile_max);
      // A row with no visible key so far keeps its running state instead of
      // scaling it by exp(-inf - -inf).
      bool empty = next_max == -INFINITY;
      float scale = empty ? 1.0f : fast::exp(row_max[matrix_row] - next_max);
      float tile_sum = 0.0f;
      for (ushort i = 0; i < 4; ++i) {
        local_scores[i] = empty ? 0.0f : fast::exp(local_scores[i] - next_max);
        tile_sum += local_scores[i];
      }
      tile_sum = simd_sum(tile_sum);
      if (lane == 0) {
        previous_scale[matrix_row] = scale;
        row_sum[matrix_row] = row_sum[matrix_row] * scale + tile_sum;
        row_max[matrix_row] = next_max;
      }
      for (ushort i = 0; i < 4; ++i) {
        uint column = lane + i * 32;
        score_storage[matrix_row * N + column] = local_scores[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto vt = tensor(tile_values, dextents<int, 2>{N, D},
                     array<int, 2>{1, int(paged ? N : cache_stride)});
    auto v0 = vt.slice<TileK, D>(0, 0);
    auto partial_output =
        pv.template get_destination_cooperative_tensor<decltype(p0),
                                                       decltype(v0), float>();
    for (ushort i = 0; i < partial_output.get_capacity(); ++i) {
      partial_output[i] = 0.0f;
    }
    for (ushort chunk = 0; chunk < N / TileK; ++chunk) {
      auto ps = pt.slice<TileK, M>(chunk * TileK, 0);
      auto vs = vt.slice<TileK, D>(chunk * TileK, 0);
      auto partial_values = pv.template get_destination_cooperative_tensor<
          decltype(ps), decltype(vs), float>();
      pv.run(ps, vs, partial_values);
      for (ushort i = 0; i < partial_output.get_capacity(); ++i) {
        partial_output[i] += partial_values[i];
      }
    }
    for (ushort i = 0; i < running.get_capacity(); ++i) {
      auto index = running.get_multidimensional_index(i);
      running[i] = running[i] * previous_scale[index[1]] + partial_output[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (split + 1 == splits) {
    // Persistent query K/V contains exactly eight rows.  The K=16 padding
    // required by the MPP value product exists only in threadgroup memory.
    constexpr ushort CurrentN = Rows;
    constexpr ushort CurrentValueTile = 16;
    threadgroup float *current_raw_scores =
        score_storage + M * CurrentValueTile;
    threadgroup bfloat *current_values =
        reinterpret_cast<threadgroup bfloat *>(
            score_storage + M * CurrentValueTile + M * CurrentN);
    auto current_scores =
        tensor(current_raw_scores, dextents<int, 2>{CurrentN, M},
               array<int, 2>{1, CurrentN});
    constexpr auto current_qk_descriptor =
        matmul2d_descriptor(M, CurrentN, TileK, false, true, false);
    matmul2d<current_qk_descriptor, execution_simdgroups<8>> current_qk;
    auto current_kt = tensor(query_keys, dextents<int, 2>{D, CurrentN},
                             array<int, 2>{1, D});
    auto current_k0 = current_kt.slice<TileK, CurrentN>(0, 0);
    auto current = current_qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(current_k0), float>();
    for (ushort i = 0; i < current.get_capacity(); ++i)
      current[i] = 0.0f;
    for (ushort chunk = 0; chunk < D / TileK; ++chunk) {
      auto qs = qt.slice<TileK, M>(chunk * TileK, 0);
      auto ks = current_kt.slice<TileK, CurrentN>(chunk * TileK, 0);
      auto partial_scores =
          current_qk.template get_destination_cooperative_tensor<
              decltype(qs), decltype(ks), float>();
      current_qk.run(qs, ks, partial_scores);
      for (ushort i = 0; i < current.get_capacity(); ++i) {
        current[i] += partial_scores[i];
      }
    }
    current.store(current_scores.slice<CurrentN, M>(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort batch = 0; batch < M; batch += 8) {
      ushort matrix_row = batch + simd_group;
      float score = lane < CurrentN
                        ? current_raw_scores[matrix_row * CurrentN + lane] *
                              0.08838834765f
                        : -INFINITY;
      float tile_max = simd_max(score);
      float next_max = max(row_max[matrix_row], tile_max);
      float scale = fast::exp(row_max[matrix_row] - next_max);
      float probability = fast::exp(score - next_max);
      float tile_sum = simd_sum(probability);
      if (lane == 0) {
        previous_scale[matrix_row] = scale;
        row_sum[matrix_row] = row_sum[matrix_row] * scale + tile_sum;
        row_max[matrix_row] = next_max;
      }
      if (lane < CurrentValueTile) {
        score_storage[matrix_row * CurrentValueTile + lane] =
            lane < CurrentN ? probability : 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto current_probabilities =
        tensor(score_storage, dextents<int, 2>{CurrentValueTile, M},
               array<int, 2>{1, CurrentValueTile});
    auto current_p0 = current_probabilities.slice<CurrentValueTile, M>(0, 0);
    for (uint index = thread_index; index < D * CurrentValueTile;
         index += 256) {
      uint dimension = index / CurrentValueTile;
      uint token = index % CurrentValueTile;
      current_values[index] = token < CurrentN
                                  ? query_values[dimension * CurrentN + token]
                                  : bfloat(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto current_vt =
        tensor(current_values, dextents<int, 2>{CurrentValueTile, D},
               array<int, 2>{1, int(CurrentValueTile)});
    auto current_v0 = current_vt.slice<CurrentValueTile, D>(0, 0);
    constexpr auto current_pv_descriptor =
        matmul2d_descriptor(M, D, CurrentValueTile, false, true, false);
    matmul2d<current_pv_descriptor, execution_simdgroups<8>> current_pv;
    auto current_output =
        current_pv.template get_destination_cooperative_tensor<
            decltype(current_p0), decltype(current_v0), float>();
    current_pv.run(current_p0, current_v0, current_output);
    for (ushort i = 0; i < running.get_capacity(); ++i) {
      auto index = running.get_multidimensional_index(i);
      running[i] = running[i] * previous_scale[index[1]] + current_output[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  auto accumulator =
      tensor(partial, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  running.store(accumulator.slice<D, M>(0, 0));
  if (thread_index < M) {
    partial[M * D + thread_index] = row_max[thread_index];
    partial[M * D + M + thread_index] = row_sum[thread_index];
  }
}

// Fixed-order combine of one (lane, head) pair's split partials into the
// normalized bf16 rows, so the result does not depend on which split
// finished first or on how many lanes ran.
inline void draft_attention_reduce_phase(device const float *partials,
                                         device bfloat *output, uint splits,
                                         uint thread_index) {
  constexpr uint M = 32, D = 128, Stride = M * D + 2 * M;
  constexpr uint Chunk = D / 8;
  uint row = thread_index / 8;
  uint dim = (thread_index % 8) * Chunk;
  float maximum = -INFINITY;
  for (uint split = 0; split < splits; ++split)
    maximum = max(maximum, partials[split * Stride + M * D + row]);
  float total = 0.0f;
  float accumulated[Chunk];
  for (uint i = 0; i < Chunk; ++i)
    accumulated[i] = 0.0f;
  for (uint split = 0; split < splits; ++split) {
    device const float *partial = partials + split * Stride;
    float split_max = partial[M * D + row];
    // A split with no visible key contributes nothing rather than
    // exp(-inf - max) times an all-zero accumulator.
    if (split_max == -INFINITY)
      continue;
    float scale = fast::exp(split_max - maximum);
    total += scale * partial[M * D + M + row];
    for (uint i = 0; i < Chunk; ++i)
      accumulated[i] += scale * partial[row * D + dim + i];
  }
  for (uint i = 0; i < Chunk; ++i)
    output[row * D + dim + i] = bfloat(accumulated[i] / total);
}

inline void draft_attention_reorder_phase(device const bfloat *grouped,
                                          device bfloat *row_major, uint groups,
                                          uint group, uint thread_index) {
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS, QHeads = 32, HeadDim = 128;
  for (uint element = group * 256 + thread_index;
       element < Rows * QHeads * HeadDim; element += groups * 256) {
    uint row = element / (QHeads * HeadDim);
    uint remainder = element % (QHeads * HeadDim);
    uint query_head = remainder / HeadDim;
    uint dim = remainder % HeadDim;
    uint grouped_index = (query_head * Rows + row) * HeadDim + dim;
    row_major[element] = grouped[grouped_index];
  }
}

template <uint Hidden>
inline void draft_conv_decode_batch_impl(
    device const bfloat *input, device const bfloat *dynamic,
    device const bfloat *base, device const bfloat *residual,
    device bfloat *output, constant DraftConvBatchParams &params, uint2 group,
    uint thread_index) {
  constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr ulong Dynamic = Hidden / 4;
  uint batch = group.y;
  if (batch >= params.lanes)
    return;
  draft_conv_phase<Hidden>(input + batch * Rows * Hidden,
                           dynamic + batch * Rows * Dynamic, base,
                           residual + batch * Rows * Hidden,
                           output + batch * Rows * Hidden,
                           params.finish != 0, params.groups, group.x,
                           thread_index);
}

kernel void draft_conv(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *dynamic [[buffer(1)]],
    device const bfloat *base [[buffer(2)]],
    device const bfloat *residual [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant DraftConvBatchParams &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  draft_conv_decode_batch_impl<5120>(input, dynamic, base, residual, output,
                                     params, group, thread_index);
}

kernel void draft_conv_h2048(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *dynamic [[buffer(1)]],
    device const bfloat *base [[buffer(2)]],
    device const bfloat *residual [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant DraftConvBatchParams &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  draft_conv_decode_batch_impl<2048>(input, dynamic, base, residual, output,
                                     params, group, thread_index);
}

kernel void draft_attention_qkv(
    device bfloat *proposal_qkv [[buffer(0)]],
    device bfloat *queries [[buffer(1)]],
    device const bfloat *q_norm [[buffer(2)]],
    device const bfloat *k_norm [[buffer(3)]],
    device const float *rope_cos [[buffer(4)]],
    device const float *rope_sin [[buffer(5)]],
    device bfloat *query_keys [[buffer(6)]],
    device bfloat *query_values [[buffer(7)]],
    constant DraftQkvBatchParams &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr ulong Packed = 6144;
  constexpr ulong Attention = 4096;
  constexpr ulong KVHeads = 8;
  constexpr ulong HeadDim = 128;
  constexpr ulong RopeStride = Rows * 64;
  uint batch = group.y;
  if (batch >= params.lanes)
    return;
  threadgroup float reductions[8];
  threadgroup bfloat head[128];
  draft_qkv_prepare_phase(
      proposal_qkv + batch * Rows * Packed,
      queries + batch * Rows * Attention, q_norm, k_norm,
      rope_cos + batch * RopeStride, rope_sin + batch * RopeStride,
      query_keys + batch * KVHeads * Rows * HeadDim,
      query_values + batch * KVHeads * HeadDim * Rows, params.groups,
      reductions, head, group.x, thread_index, lane, simd_group);
}

// Grid {kv heads, lanes, splits}: every split streams its share of the live
// ring tiles of its head and the last one adds the eight current rows.
// Partials follow the grouped queries in the same allocation, one 32 x 130
// fp32 block per (lane, head, split).
kernel void draft_attention_bf16_split(
    device bfloat *queries [[buffer(0)]],
    device bfloat *keys0 [[buffer(1)]], device bfloat *keys1 [[buffer(2)]],
    device bfloat *keys2 [[buffer(3)]], device bfloat *keys3 [[buffer(4)]],
    device bfloat *values0 [[buffer(5)]],
    device bfloat *values1 [[buffer(6)]],
    device bfloat *values2 [[buffer(7)]],
    device bfloat *values3 [[buffer(8)]],
    device bfloat *query_keys [[buffer(9)]],
    device bfloat *query_values [[buffer(10)]],
    constant DraftAttentionBatchParams &params [[buffer(11)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint AttentionM = 32, AttentionN = 128, KVHeads = 8;
  constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr ulong Attention = 4096;
  constexpr ulong HeadDim = 128;
  constexpr ulong PartialFloats = AttentionM * HeadDim + 2 * AttentionM;
  uint batch = group.y;
  if (batch >= params.lanes)
    return;
  device bfloat *keys =
      batch == 0 ? keys0 : (batch == 1 ? keys1 : (batch == 2 ? keys2 : keys3));
  device bfloat *values = batch == 0
                              ? values0
                              : (batch == 1 ? values1
                                            : (batch == 2 ? values2 : values3));
  device float *partials =
      reinterpret_cast<device float *>(queries +
                                       params.lanes * Rows * Attention) +
      ((batch * KVHeads + group.x) * params.splits + group.z) * PartialFloats;
  threadgroup float workspace[AttentionM * AttentionN + 3 * AttentionM];
  draft_attention_split_phase(
      queries + batch * Rows * Attention + group.x * AttentionM * HeadDim,
      keys + (params.paged ? 0 : group.x * params.cache_stride * HeadDim),
      values + (params.paged ? 0 : group.x * params.cache_stride * HeadDim),
      query_keys + batch * KVHeads * Rows * HeadDim + group.x * Rows * HeadDim,
      query_values + batch * KVHeads * HeadDim * Rows +
          group.x * Rows * HeadDim,
      partials, params.cache_stride, params.cache_length[batch], group.z,
      params.paged, params.pages[batch], group.x, params.splits, workspace, workspace + AttentionM * AttentionN,
      workspace + AttentionM * AttentionN + AttentionM,
      workspace + AttentionM * AttentionN + 2 * AttentionM, thread_index,
      lane, simd_group);
}

kernel void draft_attention_bf16_reduce(
    device bfloat *queries [[buffer(0)]],
    constant DraftAttentionBatchParams &params [[buffer(1)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr uint AttentionM = 32, KVHeads = 8;
  constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr ulong Attention = 4096;
  constexpr ulong HeadDim = 128;
  constexpr ulong PartialFloats = AttentionM * HeadDim + 2 * AttentionM;
  uint batch = group.y;
  if (batch >= params.lanes)
    return;
  device const float *partials =
      reinterpret_cast<device const float *>(queries +
                                             params.lanes * Rows * Attention) +
      (batch * KVHeads + group.x) * params.splits * PartialFloats;
  draft_attention_reduce_phase(
      partials,
      queries + batch * Rows * Attention + group.x * AttentionM * HeadDim,
      params.splits, thread_index);
}

kernel void draft_attention_reorder(
    device const bfloat *grouped [[buffer(0)]],
    device bfloat *row_major [[buffer(1)]],
    constant DraftQkvBatchParams &params [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr ulong Attention = 4096;
  uint batch = group.y;
  if (batch >= params.lanes)
    return;
  draft_attention_reorder_phase(grouped + batch * Rows * Attention,
                                row_major + batch * Rows * Attention,
                                params.groups, group.x, thread_index);
}
