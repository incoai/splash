#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/draft_context_kv.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/rms_inverse.h"

// The draft's kernels are compiled per geometry, as the target's attention
// is: the shipped drafts' (32 query heads over 8 KV heads of 128 dimensions,
// a 2048-slot ring, two-tap dynamic convolutions, four split threadgroups)
// keep their pipeline names, and a draft of another compiled geometry runs
// the suffix-named instantiations (Ripple's: 80 query heads, a 4096-slot
// ring, four-tap convolutions). The query heads one threadgroup attends are
// `HeadsPerGroup` of its KV head's, so a KV head's rows are shared among
// `HeadGroups` threadgroups, each holding HeadsPerGroup x 8 rows; the head
// groups hold disjoint query heads, so only the split order a reduce
// combines is rounding-relevant.

template <uint Taps>
inline void draft_conv_phase(device const bfloat *input,
                             device const bfloat *dynamic,
                             device const bfloat *base,
                             device const bfloat *residual,
                             device bfloat *output, bool finish, uint hidden,
                             uint group, uint thread_index) {
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS;
  static_assert(SPLASH_DRAFT_CONVOLUTION_STAGES == 2,
                "a stage prepares or finishes; its taps weigh the row and "
                "the rows before it, which the block holds for its rows");
  const uint conv_groups = hidden / SPLASH_DRAFT_CONVOLUTION_GROUP;
  const uint dynamic_width =
      SPLASH_DRAFT_CONVOLUTION_STAGES * Taps * conv_groups;
  const uint element = group * 256 + thread_index;
  if (element >= Rows * hidden)
    return;
  uint row = element / hidden;
  uint channel = element % hidden;
  uint conv_group = channel / SPLASH_DRAFT_CONVOLUTION_GROUP;
  uint kind = finish ? 1 : 0;
  float value = 0.0f;
  for (uint tap = 0; tap < Taps; ++tap) {
    if (row >= tap) {
      value += float(input[(row - tap) * hidden + channel]) *
               (float(base[(kind * Taps + tap) * hidden + channel]) +
                float(dynamic[row * dynamic_width +
                              (kind * Taps + tap) * conv_groups +
                              conv_group]));
    }
  }
  if (finish)
    value += float(residual[element]);
  output[element] = bfloat(value);
}

template <uint QueryHeads>
inline void draft_qkv_prepare_phase(
    device const bfloat *proposal_qkv, device bfloat *queries,
    device const bfloat *q_norm, device const bfloat *k_norm,
    device const float *rope_cos, device const float *rope_sin,
    device bfloat *query_keys, device bfloat *query_values,
    threadgroup float *reductions, threadgroup bfloat *head, uint group,
    uint thread_index, uint lane, uint simd_group) {
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS,
                 KVHeads = SPLASH_DRAFT_KV_HEADS,
                 HeadDim = SPLASH_DRAFT_HEAD_DIMENSION;
  constexpr uint QWidth = QueryHeads * HeadDim;
  constexpr uint KWidth = KVHeads * HeadDim;
  constexpr uint PackedWidth = QWidth + 2 * KWidth;
  constexpr uint QueryTasks = Rows * QueryHeads;
  constexpr uint Tasks = QueryTasks + Rows * KVHeads;

  const uint element = group * 256 + thread_index;
  if (element < Rows * KVHeads * HeadDim) {
    uint row = element / (KVHeads * HeadDim);
    uint remainder = element % (KVHeads * HeadDim);
    uint attention_head = remainder / HeadDim;
    uint dim = remainder % HeadDim;
    query_values[(attention_head * HeadDim + dim) * Rows + row] =
        proposal_qkv[row * PackedWidth + QWidth + KWidth + remainder];
  }

  // The group's normalization task; the condition is uniform across the
  // threadgroup, so every thread meets its barriers.
  if (group < Tasks) {
    const uint task = group;
    bool query = task < QueryTasks;
    uint local_task = query ? task : task - QueryTasks;
    uint heads = query ? QueryHeads : KVHeads;
    uint row = local_task / heads;
    uint attention_head = local_task % heads;
    device const bfloat *source =
        query ? proposal_qkv + row * PackedWidth + attention_head * HeadDim
              : proposal_qkv + row * PackedWidth + QWidth +
                    attention_head * HeadDim;
    device const bfloat *weight = query ? q_norm : k_norm;
    device bfloat *destination = (query ? queries : query_keys) +
                                 (attention_head * Rows + row) * HeadDim;

    float value = thread_index < HeadDim ? float(source[thread_index]) : 0.0f;
    const float inverse = rms_inverse_of_sums(value * value, HeadDim,
                                              reductions, thread_index, lane,
                                              simd_group);
    if (thread_index < HeadDim)
      head[thread_index] = bfloat(value * inverse * float(weight[thread_index]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < HeadDim / 2) {
      float first = float(head[thread_index]);
      float second = float(head[thread_index + HeadDim / 2]);
      float cosine = rope_cos[row * SPLASH_DRAFT_ROPE_PAIRS + thread_index];
      float sine = rope_sin[row * SPLASH_DRAFT_ROPE_PAIRS + thread_index];
      destination[thread_index] = bfloat(first * cosine - second * sine);
      destination[thread_index + HeadDim / 2] =
          bfloat(second * cosine + first * sine);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <uint Window>
inline void draft_context_kv_commit_phase(
    device const bfloat *context_kv, device const bfloat *k_norm,
    device const float *rope_cos, device const float *rope_sin,
    device bfloat *keys0, device bfloat *keys1, device bfloat *keys2,
    device bfloat *keys3, device bfloat *values0, device bfloat *values1,
    device bfloat *values2, device bfloat *values3,
    device const uint *retained, constant DraftContextBatchParams &params,
    uint group, uint thread_index, uint lane, uint simd_group,
    threadgroup float *reductions, threadgroup bfloat *normalized) {
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint KVHeads = SPLASH_DRAFT_KV_HEADS;
  // A context row's keys and values (draft_context_kv_phase).
  constexpr uint RowWidth = 2 * KVHeads * SPLASH_DRAFT_HEAD_DIMENSION;
  constexpr uint RopeLaneStride = Rows * SPLASH_DRAFT_ROPE_PAIRS;
  uint batch = group / (Rows * KVHeads);
  uint task = group % (Rows * KVHeads);
  device bfloat *keys = SPLASH_LANE_BINDING(batch, keys0, keys1, keys2, keys3);
  device bfloat *values = SPLASH_LANE_BINDING(batch, values0, values1, values2, values3);
  draft_context_kv_phase<Window>(
      context_kv + ulong(batch) * Rows * RowWidth, k_norm,
      rope_cos + ulong(batch) * RopeLaneStride,
      rope_sin + ulong(batch) * RopeLaneStride, keys, values,
      params.start_position[batch], min(retained[batch], Rows), task,
      thread_index, lane, simd_group, reductions, normalized);
}

// One split of the attention of one (lane, KV head, head group): the group's
// query rows over the live ring tiles of its head. The live window occupies
// one or two physical ranges of the ring; the ring-aligned tiles touching
// them are dealt round-robin to the splits, so every context length spreads
// over all splits and no tile reads past the ring. Each split runs an online
// softmax over its tiles, the last split adds the eight current rows, and
// the unnormalized fp32 accumulator with its row maxima and sums lands in
// `partial` for the fixed-order reduce.
template <uint HeadsPerGroup, uint Splits, uint Window>
inline void draft_attention_split_phase(
    device bfloat *queries, device bfloat *keys, device bfloat *values,
    device bfloat *query_keys, device bfloat *query_values,
    device float *partial, uint value_stride, uint cache_length, uint split,
    threadgroup float *score_storage, threadgroup float *row_max,
    threadgroup float *row_sum, threadgroup float *previous_scale,
    uint thread_index, uint lane, uint simd_group) {
  constexpr ushort M = HeadsPerGroup * SPLASH_DRAFT_QUERY_ROWS, N = 128,
                   D = SPLASH_DRAFT_HEAD_DIMENSION, TileK = 64;
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS;
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
  if (split >= live_tiles && split + 1 != Splits) {
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
                        array<int, 2>{1, int(value_stride)});
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

  for (uint tile = split; tile < live_tiles; tile += Splits) {
    uint slot = (tile < wrapped ? tile : resume + tile - wrapped) * N;
    auto kt =
        tensor(keys + slot * D, dextents<int, 2>{D, N}, array<int, 2>{1, D});
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

    auto vt = tensor(values + slot, dextents<int, 2>{N, D},
                     array<int, 2>{1, int(value_stride)});
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

  if (split + 1 == Splits) {
    // Persistent query K/V contains exactly eight rows, the current rows of
    // the group. The MPP matmul asks M or N to be a multiple of sixteen,
    // which the eight real rows give only at M a multiple of sixteen: there
    // the scores are the matmul's, and elsewhere they are plain fp32 dot
    // products, at most M x 8 x 128 MACs beside a ring tile's M x 128 x 128.
    // The K=16 padding required by the MPP value product exists only in
    // threadgroup memory.
    constexpr ushort CurrentN = Rows;
    constexpr ushort CurrentValueTile = 16;
    threadgroup float *current_raw_scores =
        score_storage + M * CurrentValueTile;
    threadgroup bfloat *current_values =
        reinterpret_cast<threadgroup bfloat *>(
            score_storage + M * CurrentValueTile + M * CurrentN);
    if constexpr (M % 16 == 0) {
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
    } else {
      for (uint task = thread_index; task < M * CurrentN; task += 256) {
        const uint row = task / CurrentN;
        const uint token = task % CurrentN;
        float dot = 0.0f;
        for (uint d = 0; d < D; ++d)
          dot += float(queries[row * D + d]) *
                 float(query_keys[token * D + d]);
        current_raw_scores[task] = dot;
      }
    }
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

// Fixed-order combine of one (lane, head, head group)'s split partials into
// the normalized bf16 rows, so the result does not depend on which split
// finished first or on how many lanes ran.
template <uint M, uint Splits>
inline void draft_attention_reduce_phase(device const float *partials,
                                         device bfloat *output,
                                         uint thread_index) {
  constexpr uint D = SPLASH_DRAFT_HEAD_DIMENSION,
                 Stride = M * D + 2 * M;
  constexpr uint Chunk = D / 8;
  for (uint task = thread_index; task < M * 8; task += 256) {
    uint row = task / 8;
    uint dim = (task % 8) * Chunk;
    float maximum = -INFINITY;
    for (uint split = 0; split < Splits; ++split)
      maximum = max(maximum, partials[split * Stride + M * D + row]);
    float total = 0.0f;
    float accumulated[Chunk];
    for (uint i = 0; i < Chunk; ++i)
      accumulated[i] = 0.0f;
    for (uint split = 0; split < Splits; ++split) {
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
}

template <uint QueryHeads>
inline void draft_attention_reorder_phase(device const bfloat *grouped,
                                          device bfloat *row_major, uint group,
                                          uint thread_index) {
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS,
                 HeadDim = SPLASH_DRAFT_HEAD_DIMENSION;
  const uint element = group * 256 + thread_index;
  if (element >= Rows * QueryHeads * HeadDim)
    return;
  uint row = element / (QueryHeads * HeadDim);
  uint remainder = element % (QueryHeads * HeadDim);
  uint query_head = remainder / HeadDim;
  uint dim = remainder % HeadDim;
  uint grouped_index = (query_head * Rows + row) * HeadDim + dim;
  row_major[element] = grouped[grouped_index];
}

#define SPLASH_DRAFT_CONV_KERNEL(name, taps)                                   \
  kernel void name(                                                            \
      device const bfloat *input [[buffer(0)]],                                \
      device const bfloat *dynamic [[buffer(1)]],                              \
      device const bfloat *base [[buffer(2)]],                                 \
      device const bfloat *residual [[buffer(3)]],                             \
      device bfloat *output [[buffer(4)]],                                     \
      constant DraftConvBatchParams &params [[buffer(5)]],                     \
      uint2 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;                            \
    const ulong hidden = params.hidden;                                        \
    const ulong dynamic_width =                                                \
        SPLASH_DRAFT_CONVOLUTION_STAGES * (taps) *                             \
        (hidden / SPLASH_DRAFT_CONVOLUTION_GROUP);                             \
    uint batch = group.y;                                                      \
    draft_conv_phase<(taps)>(                                                  \
        input + batch * Rows * hidden,                                         \
        dynamic + batch * Rows * dynamic_width, base,                          \
        residual + batch * Rows * hidden,                                      \
        output + batch * Rows * hidden, params.finish != 0, params.hidden,     \
        group.x, thread_index);                                                \
  }

#define SPLASH_DRAFT_QKV_KERNEL(name, queryHeads)                              \
  kernel void name(                                                            \
      device const bfloat *proposal_qkv [[buffer(0)]],                         \
      device bfloat *queries [[buffer(1)]],                                    \
      device const bfloat *q_norm [[buffer(2)]],                               \
      device const bfloat *k_norm [[buffer(3)]],                               \
      device const float *rope_cos [[buffer(4)]],                              \
      device const float *rope_sin [[buffer(5)]],                              \
      device bfloat *query_keys [[buffer(6)]],                                 \
      device bfloat *query_values [[buffer(7)]],                               \
      uint2 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;                            \
    constexpr ulong KVHeads = SPLASH_DRAFT_KV_HEADS;                           \
    constexpr ulong HeadDim = SPLASH_DRAFT_HEAD_DIMENSION;                     \
    constexpr ulong Packed =                                                   \
        (queryHeads)*HeadDim + 2 * KVHeads * HeadDim;                          \
    constexpr ulong Attention = (queryHeads)*HeadDim;                          \
    constexpr ulong RopeStride = Rows * SPLASH_DRAFT_ROPE_PAIRS;               \
    uint batch = group.y;                                                      \
    threadgroup float reductions[8];                                           \
    threadgroup bfloat head[SPLASH_DRAFT_HEAD_DIMENSION];                      \
    draft_qkv_prepare_phase<(queryHeads)>(                                     \
        proposal_qkv + batch * Rows * Packed,                                  \
        queries + batch * Rows * Attention, q_norm, k_norm,                    \
        rope_cos + batch * RopeStride, rope_sin + batch * RopeStride,          \
        query_keys + batch * KVHeads * Rows * HeadDim,                         \
        query_values + batch * KVHeads * HeadDim * Rows, reductions, head,     \
        group.x, thread_index, lane, simd_group);                              \
  }

#define SPLASH_DRAFT_COMMIT_KERNEL(name, window)                               \
  kernel void name(                                                            \
      device const bfloat *context_kv [[buffer(0)]],                           \
      device const bfloat *k_norm [[buffer(1)]],                               \
      device const float *rope_cos [[buffer(2)]],                              \
      device const float *rope_sin [[buffer(3)]],                              \
      device bfloat *keys0 [[buffer(4)]], device bfloat *keys1 [[buffer(5)]],  \
      device bfloat *keys2 [[buffer(6)]], device bfloat *keys3 [[buffer(7)]],  \
      device bfloat *values0 [[buffer(8)]],                                    \
      device bfloat *values1 [[buffer(9)]],                                    \
      device bfloat *values2 [[buffer(10)]],                                   \
      device bfloat *values3 [[buffer(11)]],                                   \
      device const uint *retained [[buffer(12)]],                              \
      constant DraftContextBatchParams &params [[buffer(13)]],                 \
      uint group [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    threadgroup float splash_reductions[8];                                   \
    threadgroup bfloat splash_normalized[SPLASH_DRAFT_HEAD_DIMENSION];         \
    draft_context_kv_commit_phase<(window)>(                                   \
        context_kv, k_norm, rope_cos, rope_sin, keys0, keys1, keys2, keys3,    \
        values0, values1, values2, values3, retained, params, group,           \
        thread_index, lane, simd_group, splash_reductions, splash_normalized); \
  }

// Grid {kv heads x head groups, lanes, splits}: every split streams its
// share of the live ring tiles of its head and the last one adds the eight
// current rows. Partials follow the grouped queries in the same allocation,
// one M x (128 + 2) fp32 block per (lane, head, head group, split).
#define SPLASH_DRAFT_SPLIT_KERNEL(name, queryHeads, headsPerGroup, splits,     \
                                  window)                                      \
  kernel void name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device bfloat *keys0 [[buffer(1)]], device bfloat *keys1 [[buffer(2)]],  \
      device bfloat *keys2 [[buffer(3)]], device bfloat *keys3 [[buffer(4)]],  \
      device bfloat *values0 [[buffer(5)]],                                    \
      device bfloat *values1 [[buffer(6)]],                                    \
      device bfloat *values2 [[buffer(7)]],                                    \
      device bfloat *values3 [[buffer(8)]],                                    \
      device bfloat *query_keys [[buffer(9)]],                                 \
      device bfloat *query_values [[buffer(10)]],                              \
      constant DraftAttentionBatchParams &params [[buffer(11)]],               \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    constexpr uint HeadGroups =                                                \
        (queryHeads) / SPLASH_DRAFT_KV_HEADS / (headsPerGroup);                \
    constexpr uint M = (headsPerGroup)*SPLASH_DRAFT_QUERY_ROWS;                \
    constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;                            \
    constexpr ulong HeadDim = SPLASH_DRAFT_HEAD_DIMENSION;                     \
    constexpr ulong Attention = (queryHeads)*HeadDim;                          \
    constexpr ulong HeadRows =                                                 \
        ((queryHeads) / SPLASH_DRAFT_KV_HEADS) * Rows;                         \
    constexpr ulong PartialFloats = M * HeadDim + 2 * M;                       \
    static_assert(queryHeads % (SPLASH_DRAFT_KV_HEADS * (headsPerGroup)) == 0, \
                  "head groups hold whole query heads");                       \
    static_assert(M * HeadDim + 3 * M <= 8192,                                 \
                  "the split's threadgroup workspace fits the device's "       \
                  "32 KB static threadgroup memory");                          \
    uint batch = group.y;                                                      \
    uint head = group.x / HeadGroups;                                          \
    uint head_group = group.x % HeadGroups;                                    \
    device bfloat *keys = SPLASH_LANE_BINDING(batch, keys0, keys1, keys2, keys3); \
    device bfloat *values =                                                    \
        SPLASH_LANE_BINDING(batch, values0, values1, values2, values3);        \
    device float *partials =                                                   \
        reinterpret_cast<device float *>(queries +                             \
                                         params.lanes * Rows * Attention) +    \
        (((batch * SPLASH_DRAFT_KV_HEADS + head) * HeadGroups + head_group) *  \
             (splits) +                                                        \
         group.z) *                                                            \
            PartialFloats;                                                     \
    threadgroup float workspace[M * 128 + 3 * M];                              \
    draft_attention_split_phase<(headsPerGroup), (splits), (window)>(          \
        queries + batch * Rows * Attention +                                   \
            head * HeadRows * HeadDim + head_group * M * HeadDim,              \
        keys + head * (window)*HeadDim,                                        \
        values + head * (window)*HeadDim,                                      \
        query_keys +                                                           \
            batch * SPLASH_DRAFT_KV_HEADS * Rows * HeadDim +                   \
            head * Rows * HeadDim,                                             \
        query_values +                                                         \
            batch * SPLASH_DRAFT_KV_HEADS * HeadDim * Rows +                  \
            head * Rows * HeadDim,                                            \
        partials, params.value_stride, params.cache_length[batch], group.z,    \
        workspace, workspace + M * 128, workspace + M * 128 + M,               \
        workspace + M * 128 + 2 * M, thread_index, lane, simd_group);          \
  }

#define SPLASH_DRAFT_REDUCE_KERNEL(name, queryHeads, headsPerGroup, splits)    \
  kernel void name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      constant DraftAttentionBatchParams &params [[buffer(1)]],                \
      uint2 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint HeadGroups =                                                \
        (queryHeads) / SPLASH_DRAFT_KV_HEADS / (headsPerGroup);                \
    constexpr uint M = (headsPerGroup)*SPLASH_DRAFT_QUERY_ROWS;                \
    constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;                            \
    constexpr ulong HeadDim = SPLASH_DRAFT_HEAD_DIMENSION;                     \
    constexpr ulong Attention = (queryHeads)*HeadDim;                          \
    constexpr ulong HeadRows =                                                 \
        ((queryHeads) / SPLASH_DRAFT_KV_HEADS) * Rows;                         \
    constexpr ulong PartialFloats = M * HeadDim + 2 * M;                       \
    uint batch = group.y;                                                      \
    uint head = group.x / HeadGroups;                                          \
    uint head_group = group.x % HeadGroups;                                    \
    device const float *partials =                                             \
        reinterpret_cast<device const float *>(                                \
            queries + params.lanes * Rows * Attention) +                       \
        ((batch * SPLASH_DRAFT_KV_HEADS + head) * HeadGroups + head_group) *   \
            (splits)*PartialFloats;                                            \
    draft_attention_reduce_phase<M, (splits)>(                                 \
        partials,                                                              \
        queries + batch * Rows * Attention + head * HeadRows * HeadDim +       \
            head_group * M * HeadDim,                                          \
        thread_index);                                                         \
  }

#define SPLASH_DRAFT_REORDER_KERNEL(name, queryHeads)                          \
  kernel void name(                                                            \
      device const bfloat *grouped [[buffer(0)]],                              \
      device bfloat *row_major [[buffer(1)]],                                  \
      uint2 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;                            \
    constexpr ulong Attention = (queryHeads)*SPLASH_DRAFT_HEAD_DIMENSION;      \
    uint batch = group.y;                                                      \
    draft_attention_reorder_phase<(queryHeads)>(                               \
        grouped + batch * Rows * Attention,                                    \
        row_major + batch * Rows * Attention, group.x, thread_index);          \
  }

// The shipped drafts' kernels.
SPLASH_DRAFT_CONV_KERNEL(draft_conv, SPLASH_DRAFT_CONVOLUTION_TAPS)
SPLASH_DRAFT_QKV_KERNEL(draft_attention_qkv, SPLASH_DRAFT_QUERY_HEADS)
SPLASH_DRAFT_COMMIT_KERNEL(draft_context_kv_commit, SPLASH_DRAFT_SLIDING_WINDOW)
SPLASH_DRAFT_SPLIT_KERNEL(draft_attention_bf16_split, SPLASH_DRAFT_QUERY_HEADS,
                          SPLASH_DRAFT_QUERY_HEADS / SPLASH_DRAFT_KV_HEADS,
                          SPLASH_DRAFT_ATTENTION_SPLITS,
                          SPLASH_DRAFT_SLIDING_WINDOW)
SPLASH_DRAFT_REDUCE_KERNEL(draft_attention_bf16_reduce, SPLASH_DRAFT_QUERY_HEADS,
                           SPLASH_DRAFT_QUERY_HEADS / SPLASH_DRAFT_KV_HEADS,
                           SPLASH_DRAFT_ATTENTION_SPLITS)
SPLASH_DRAFT_REORDER_KERNEL(draft_attention_reorder, SPLASH_DRAFT_QUERY_HEADS)

// Ripple's draft's kernels: 80 query heads over the 8 KV heads, a 4096-slot
// ring, four-tap convolutions, and head groups of five query heads (two
// threadgroups per KV head, 40 rows each).
SPLASH_DRAFT_CONV_KERNEL(draft_conv_kv8_h5_d128_w4096_s8, 4)
SPLASH_DRAFT_QKV_KERNEL(draft_attention_qkv_kv8_h5_d128_w4096_s8, 80)
SPLASH_DRAFT_COMMIT_KERNEL(draft_context_kv_commit_kv8_h5_d128_w4096_s8, 4096)
SPLASH_DRAFT_SPLIT_KERNEL(draft_attention_bf16_split_kv8_h5_d128_w4096_s4, 80, 5,
                          4, 4096)
SPLASH_DRAFT_REDUCE_KERNEL(draft_attention_bf16_reduce_kv8_h5_d128_s4, 80, 5, 4)
SPLASH_DRAFT_SPLIT_KERNEL(draft_attention_bf16_split_kv8_h5_d128_w4096_s8, 80, 5,
                          8, 4096)
SPLASH_DRAFT_REDUCE_KERNEL(draft_attention_bf16_reduce_kv8_h5_d128_w4096_s8, 80, 5, 8)
SPLASH_DRAFT_REORDER_KERNEL(draft_attention_reorder_kv8_h5_d128_w4096_s8, 80)
