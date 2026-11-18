#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/draft_context_kv.h"
#include "metal/kernels/common/draft_context_window.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/rms_inverse.h"

inline void draft_conv_phase(device const bfloat *input,
                             device const bfloat *dynamic,
                             device const bfloat *base,
                             device const bfloat *residual,
                             device bfloat *output, bool finish, uint hidden,
                             uint group, uint thread_index) {
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS,
                 Taps = SPLASH_DRAFT_CONVOLUTION_TAPS;
  static_assert(SPLASH_DRAFT_CONVOLUTION_STAGES == 2 && Taps == 2,
                "a stage prepares or finishes; its taps are a row and the "
                "row before");
  const uint conv_groups = hidden / SPLASH_DRAFT_CONVOLUTION_GROUP;
  const uint dynamic_width = draft_dynamic_width(hidden);
  const uint element = group * 256 + thread_index;
  if (element >= Rows * hidden)
    return;
  uint row = element / hidden;
  uint channel = element % hidden;
  uint conv_group = channel / SPLASH_DRAFT_CONVOLUTION_GROUP;
  uint kind = finish ? 1 : 0;
  float value =
      float(input[element]) *
      (float(base[(kind * Taps) * hidden + channel]) +
       float(dynamic[row * dynamic_width + (kind * Taps) * conv_groups +
                     conv_group]));
  if (row > 0) {
    value +=
        float(input[(row - 1) * hidden + channel]) *
        (float(base[(kind * Taps + 1) * hidden + channel]) +
         float(
             dynamic[row * dynamic_width + (kind * Taps + 1) * conv_groups +
                     conv_group]));
  }
  if (finish)
    value += float(residual[element]);
  output[element] = bfloat(value);
}

inline void draft_qkv_prepare_phase(
    device const bfloat *proposal_qkv, device bfloat *queries,
    device const bfloat *q_norm, device const bfloat *k_norm,
    device const float *rope_cos, device const float *rope_sin,
    device bfloat *query_keys, device bfloat *query_values,
    threadgroup float *reductions, threadgroup bfloat *head, uint group,
    uint thread_index, uint lane, uint simd_group) {
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS,
                 QHeads = SPLASH_DRAFT_QUERY_HEADS,
                 KVHeads = SPLASH_DRAFT_KV_HEADS,
                 HeadDim = SPLASH_DRAFT_HEAD_DIMENSION;
  constexpr uint QWidth = SPLASH_DRAFT_ATTENTION_WIDTH, KWidth = KVHeads * HeadDim;
  constexpr uint PackedWidth = SPLASH_DRAFT_QKV_WIDTH;
  constexpr uint QueryTasks = Rows * QHeads;
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
    uint heads = query ? QHeads : KVHeads;
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

kernel void draft_context_kv_commit(
    device const bfloat *context_kv [[buffer(0)]],
    device const bfloat *k_norm [[buffer(1)]],
    device const float *rope_cos [[buffer(2)]],
    device const float *rope_sin [[buffer(3)]],
    device bfloat *keys0 [[buffer(4)]], device bfloat *keys1 [[buffer(5)]],
    device bfloat *keys2 [[buffer(6)]], device bfloat *keys3 [[buffer(7)]],
    device bfloat *values0 [[buffer(8)]],
    device bfloat *values1 [[buffer(9)]],
    device bfloat *values2 [[buffer(10)]],
    device bfloat *values3 [[buffer(11)]],
    device const uint *retained_rows [[buffer(12)]],
    device const SplashLaneRows *lane_rows [[buffer(13)]],
    constant DraftContextBatchParams &params [[buffer(14)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint KVHeads = SPLASH_DRAFT_KV_HEADS;
  // A context row's keys and values (draft_context_kv_phase).
  constexpr uint RowWidth = 2 * KVHeads * SPLASH_DRAFT_HEAD_DIMENSION;
  constexpr uint RopeLaneStride =
      SPLASH_TARGET_VERIFY_ROWS * SPLASH_DRAFT_ROPE_PAIRS;
  // A grid of (path row, KV head) tasks of a lane's block by lanes: path row
  // i is the i-th row the lane retained, its i-th set bit.
  const uint batch = group.y, row = group.x / KVHeads;
  uint path = retained_rows[batch];
  if (row >= min(uint(popcount(path)), lane_rows[batch].rows))
    return;
  for (uint earlier = 0; earlier < row; ++earlier)
    path &= path - 1;
  device bfloat *keys = SPLASH_LANE_BINDING(batch, keys0, keys1, keys2, keys3);
  device bfloat *values = SPLASH_LANE_BINDING(batch, values0, values1, values2, values3);
  const ulong rope = ulong(batch) * RopeLaneStride + row * SPLASH_DRAFT_ROPE_PAIRS;
  threadgroup float reductions[8];
  threadgroup bfloat normalized[SPLASH_DRAFT_HEAD_DIMENSION];
  draft_context_kv_phase(
      context_kv + (ulong(lane_rows[batch].first) + ctz(path)) * RowWidth,
      k_norm, rope_cos + rope, rope_sin + rope, keys, values,
      params.start_position[batch] + row, group.x % KVHeads, thread_index,
      lane, simd_group, reductions, normalized);
}

// The rows a lane retained of its rows of a decode step (`retained_rows`, its
// acceptance's mask, among its rows `lane_rows`), in path order, to the slots
// of their positions in its context window, which stays current with its
// rings: the path's i-th, its i-th set bit, at the lane's start position plus
// i.
kernel void draft_context_window_commit(
    device const bfloat *context [[buffer(0)]], device uchar *window [[buffer(1)]],
    device const uint *retained_rows [[buffer(2)]],
    device const SplashLaneRows *lane_rows [[buffer(3)]],
    constant DraftContextWindowParams &params [[buffer(4)]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  DraftContextGroup at;
  if (!draft_context_group(params, threadgroup_index, simd_group, at))
    return;
  uint path = retained_rows[0];
  if (at.row >= min(uint(popcount(path)), lane_rows[0].rows))
    return;
  for (uint earlier = 0; earlier < at.row; ++earlier)
    path &= path - 1;
  // The store reads row at.row of its rows: here the lane's path row, at or
  // past it.
  draft_context_store(context + (ulong(lane_rows[0].first) + ctz(path) - at.row) * params.width, window, params,
                      at, lane);
}

// One split of the attention of one KV head over the 32 query rows of one
// lane (four query heads x eight proposals). The live window occupies one or
// two physical ranges of the ring; the ring-aligned tiles touching them are
// dealt round-robin to the splits, so every context length spreads over all
// splits and no tile reads past the ring. Each split runs an online softmax
// over its tiles, the last split adds the eight current rows, and the
// unnormalized fp32 accumulator with its row maxima and sums lands in
// `partial` for the fixed-order reduce. A row sees the keys of the last
// Visible positions, its own among them; Visible is at most the ring's
// window (a LiLiCorr draft sees 1024 of a 2048-slot ring).
template <uint Visible>
inline void draft_attention_split_phase(
    device bfloat *queries, device bfloat *keys, device bfloat *values,
    device bfloat *query_keys, device bfloat *query_values,
    device float *partial, uint value_stride, uint cache_length, uint split,
    threadgroup float *score_storage, threadgroup float *row_max,
    threadgroup float *row_sum, threadgroup float *previous_scale,
    uint thread_index, uint lane, uint simd_group) {
  constexpr ushort M = SPLASH_DRAFT_ATTENTION_ROWS, N = 128,
                   D = SPLASH_DRAFT_HEAD_DIMENSION, TileK = 64;
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS, Window = SPLASH_DRAFT_SLIDING_WINDOW,
                 Splits = SPLASH_DRAFT_ATTENTION_SPLITS;
  static_assert(Visible <= Window, "a row sees at most the ring's window");
  uint common_start =
      cache_length >= Visible - 1 ? cache_length - (Visible - 1) : 0;
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
          query_position >= Visible - 1 ? query_position - (Visible - 1) : 0;
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
// finished first or on how many lanes ran. The head's 32 rows are its four
// query heads' eight proposal rows; each goes to its place in the lane's
// [proposal row][query head][dim] rows, the output projection's input, which
// take the place of the grouped query rows the splits have read.
inline void draft_attention_reduce_phase(device const float *partials,
                                         device bfloat *output, uint kv_head,
                                         uint thread_index) {
  constexpr uint M = SPLASH_DRAFT_ATTENTION_ROWS, D = SPLASH_DRAFT_HEAD_DIMENSION,
                 Rows = SPLASH_DRAFT_QUERY_ROWS,
                 QHeads = SPLASH_DRAFT_QUERY_HEADS, Stride = M * D + 2 * M,
                 Splits = SPLASH_DRAFT_ATTENTION_SPLITS;
  constexpr uint Chunk = D / 8;
  uint row = thread_index / 8;
  uint dim = (thread_index % 8) * Chunk;
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
  uint proposal_row = row % Rows;
  uint query_head = kv_head * (M / Rows) + row / Rows;
  device bfloat *destination =
      output + (proposal_row * QHeads + query_head) * D + dim;
  for (uint i = 0; i < Chunk; ++i)
    destination[i] = bfloat(accumulated[i] / total);
}

// draft_attention_reduce_phase with each query head's sink logit joining its
// rows' softmax denominators as a key of no value (a LiLiCorr draft's);
// `sinks` holds one logit per query head. A function of its own, so the
// reduce without sinks keeps its AIR.
inline void draft_attention_reduce_sinks_phase(device const float *partials,
                                               device bfloat *output,
                                               device const bfloat *sinks,
                                               uint kv_head,
                                               uint thread_index) {
  constexpr uint M = SPLASH_DRAFT_ATTENTION_ROWS, D = SPLASH_DRAFT_HEAD_DIMENSION,
                 Rows = SPLASH_DRAFT_QUERY_ROWS,
                 QHeads = SPLASH_DRAFT_QUERY_HEADS, Stride = M * D + 2 * M,
                 Splits = SPLASH_DRAFT_ATTENTION_SPLITS;
  constexpr uint Chunk = D / 8;
  uint row = thread_index / 8;
  uint dim = (thread_index % 8) * Chunk;
  uint proposal_row = row % Rows;
  uint query_head = kv_head * (M / Rows) + row / Rows;
  const float sink = float(sinks[query_head]);
  float maximum = sink;
  for (uint split = 0; split < Splits; ++split)
    maximum = max(maximum, partials[split * Stride + M * D + row]);
  float total = fast::exp(sink - maximum);
  float accumulated[Chunk];
  for (uint i = 0; i < Chunk; ++i)
    accumulated[i] = 0.0f;
  for (uint split = 0; split < Splits; ++split) {
    device const float *partial = partials + split * Stride;
    float split_max = partial[M * D + row];
    if (split_max == -INFINITY)
      continue;
    float scale = fast::exp(split_max - maximum);
    total += scale * partial[M * D + M + row];
    for (uint i = 0; i < Chunk; ++i)
      accumulated[i] += scale * partial[row * D + dim + i];
  }
  device bfloat *destination =
      output + (proposal_row * QHeads + query_head) * D + dim;
  for (uint i = 0; i < Chunk; ++i)
    destination[i] = bfloat(accumulated[i] / total);
}

kernel void draft_conv(device const bfloat *input [[buffer(0)]],
                       device const bfloat *dynamic [[buffer(1)]],
                       device const bfloat *base [[buffer(2)]],
                       device const bfloat *residual [[buffer(3)]],
                       device bfloat *output [[buffer(4)]],
                       constant DraftConvBatchParams &params [[buffer(5)]],
                       uint2 group [[threadgroup_position_in_grid]],
                       uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;
  const ulong hidden = params.hidden, dynamic_width = draft_dynamic_width(params.hidden);
  uint batch = group.y;
  draft_conv_phase(input + batch * Rows * hidden,
                   dynamic + batch * Rows * dynamic_width, base,
                   residual + batch * Rows * hidden,
                   output + batch * Rows * hidden, params.finish != 0,
                   params.hidden, group.x, thread_index);
}

kernel void draft_attention_qkv(
    device const bfloat *proposal_qkv [[buffer(0)]],
    device bfloat *queries [[buffer(1)]],
    device const bfloat *q_norm [[buffer(2)]],
    device const bfloat *k_norm [[buffer(3)]],
    device const float *rope_cos [[buffer(4)]],
    device const float *rope_sin [[buffer(5)]],
    device bfloat *query_keys [[buffer(6)]],
    device bfloat *query_values [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr ulong Packed = SPLASH_DRAFT_QKV_WIDTH;
  constexpr ulong Attention = SPLASH_DRAFT_ATTENTION_WIDTH;
  constexpr ulong KVHeads = SPLASH_DRAFT_KV_HEADS;
  constexpr ulong HeadDim = SPLASH_DRAFT_HEAD_DIMENSION;
  // A lane's RoPE rows hold its verify block's positions, the first Rows of
  // which its draft's rows take.
  constexpr ulong RopeStride = SPLASH_TARGET_VERIFY_ROWS * SPLASH_DRAFT_ROPE_PAIRS;
  uint batch = group.y;
  threadgroup float reductions[8];
  threadgroup bfloat head[SPLASH_DRAFT_HEAD_DIMENSION];
  draft_qkv_prepare_phase(
      proposal_qkv + batch * Rows * Packed,
      queries + batch * Rows * Attention, q_norm, k_norm,
      rope_cos + batch * RopeStride, rope_sin + batch * RopeStride,
      query_keys + batch * KVHeads * Rows * HeadDim,
      query_values + batch * KVHeads * HeadDim * Rows, reductions, head,
      group.x, thread_index, lane, simd_group);
}

// Grid {kv heads, lanes, splits}: every split streams its share of the live
// ring tiles of its head and the last one adds the eight current rows.
// Partials follow the grouped queries in the same allocation, one 32 x 130
// fp32 block per (lane, head, split).
template <uint Visible>
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
  constexpr uint AttentionM = SPLASH_DRAFT_ATTENTION_ROWS, AttentionN = 128,
                 KVHeads = SPLASH_DRAFT_KV_HEADS;
  constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr ulong Attention = SPLASH_DRAFT_ATTENTION_WIDTH;
  constexpr ulong HeadDim = SPLASH_DRAFT_HEAD_DIMENSION;
  constexpr ulong Window = SPLASH_DRAFT_SLIDING_WINDOW;
  constexpr ulong PartialFloats = AttentionM * HeadDim + 2 * AttentionM;
  uint batch = group.y;
  device bfloat *keys = SPLASH_LANE_BINDING(batch, keys0, keys1, keys2, keys3);
  device bfloat *values = SPLASH_LANE_BINDING(batch, values0, values1, values2, values3);
  device float *partials =
      reinterpret_cast<device float *>(queries +
                                       params.lanes * Rows * Attention) +
      ((batch * KVHeads + group.x) * SPLASH_DRAFT_ATTENTION_SPLITS + group.z) *
          PartialFloats;
  threadgroup float workspace[AttentionM * AttentionN + 3 * AttentionM];
  draft_attention_split_phase<Visible>(
      queries + batch * Rows * Attention + group.x * AttentionM * HeadDim,
      keys + group.x * Window * HeadDim, values + group.x * Window * HeadDim,
      query_keys + batch * KVHeads * Rows * HeadDim + group.x * Rows * HeadDim,
      query_values + batch * KVHeads * HeadDim * Rows +
          group.x * Rows * HeadDim,
      partials, params.value_stride, params.cache_length[batch], group.z,
      workspace, workspace + AttentionM * AttentionN,
      workspace + AttentionM * AttentionN + AttentionM,
      workspace + AttentionM * AttentionN + 2 * AttentionM, thread_index,
      lane, simd_group);
}

template [[host_name("draft_attention_bf16_split")]] kernel void
draft_attention_bf16_split<SPLASH_DRAFT_SLIDING_WINDOW>(
    device bfloat *, device bfloat *, device bfloat *, device bfloat *,
    device bfloat *, device bfloat *, device bfloat *, device bfloat *,
    device bfloat *, device bfloat *, device bfloat *,
    constant DraftAttentionBatchParams &, uint3, uint, uint, uint);
template [[host_name("draft_attention_bf16_split_window1024")]] kernel void
draft_attention_bf16_split<1024>(
    device bfloat *, device bfloat *, device bfloat *, device bfloat *,
    device bfloat *, device bfloat *, device bfloat *, device bfloat *,
    device bfloat *, device bfloat *, device bfloat *,
    constant DraftAttentionBatchParams &, uint3, uint, uint, uint);

kernel void draft_attention_bf16_reduce(
    device bfloat *queries [[buffer(0)]],
    constant DraftAttentionBatchParams &params [[buffer(1)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr uint AttentionM = SPLASH_DRAFT_ATTENTION_ROWS, KVHeads = SPLASH_DRAFT_KV_HEADS;
  constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr ulong Attention = SPLASH_DRAFT_ATTENTION_WIDTH;
  constexpr ulong HeadDim = SPLASH_DRAFT_HEAD_DIMENSION;
  constexpr ulong PartialFloats = AttentionM * HeadDim + 2 * AttentionM;
  uint batch = group.y;
  device const float *partials =
      reinterpret_cast<device const float *>(queries +
                                             params.lanes * Rows * Attention) +
      (batch * KVHeads + group.x) * SPLASH_DRAFT_ATTENTION_SPLITS *
          PartialFloats;
  draft_attention_reduce_phase(partials, queries + batch * Rows * Attention,
                               group.x, thread_index);
}

kernel void draft_attention_bf16_reduce_sinks(
    device bfloat *queries [[buffer(0)]],
    device const bfloat *sinks [[buffer(1)]],
    constant DraftAttentionBatchParams &params [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr uint AttentionM = SPLASH_DRAFT_ATTENTION_ROWS, KVHeads = SPLASH_DRAFT_KV_HEADS;
  constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr ulong Attention = SPLASH_DRAFT_ATTENTION_WIDTH;
  constexpr ulong HeadDim = SPLASH_DRAFT_HEAD_DIMENSION;
  constexpr ulong PartialFloats = AttentionM * HeadDim + 2 * AttentionM;
  uint batch = group.y;
  device const float *partials =
      reinterpret_cast<device const float *>(queries +
                                             params.lanes * Rows * Attention) +
      (batch * KVHeads + group.x) * SPLASH_DRAFT_ATTENTION_SPLITS *
          PartialFloats;
  draft_attention_reduce_sinks_phase(partials, queries + batch * Rows * Attention,
                                     sinks, group.x, thread_index);
}
