// Experimental kernels; the production split arithmetic stays unchanged.
#include "metal/kernels/decode/draft.metal"

template <class Pointer>
inline void muse_reduce_row_major(Pointer partials, device bfloat *output,
                                  uint thread_index) {
  constexpr uint M = SPLASH_DRAFT_ATTENTION_ROWS,
                 D = SPLASH_DRAFT_HEAD_DIMENSION, Stride = M * D + 2 * M,
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
    auto partial = partials + split * Stride;
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
  const uint token = row % SPLASH_DRAFT_QUERY_ROWS;
  const uint head = row / SPLASH_DRAFT_QUERY_ROWS;
  for (uint i = 0; i < Chunk; ++i)
    output[token * SPLASH_DRAFT_ATTENTION_WIDTH + head * D + dim + i] =
        bfloat(accumulated[i] / total);
}

inline void
muse_head_prepare(device const bfloat *packed, device bfloat *queries,
                  device const bfloat *qnorm, device const bfloat *knorm,
                  device const float *cosine, device const float *sine,
                  device bfloat *qkeys, device bfloat *qvalues,
                  threadgroup float *sums, threadgroup bfloat *normalized,
                  uint head, uint tid, uint lane, uint sg) {
  constexpr uint R = SPLASH_DRAFT_QUERY_ROWS, D = SPLASH_DRAFT_HEAD_DIMENSION;
  constexpr uint H = SPLASH_DRAFT_ATTENTION_ROWS / R,
                 QW = SPLASH_DRAFT_ATTENTION_WIDTH;
  constexpr uint KW = SPLASH_DRAFT_KV_HEADS * D, P = SPLASH_DRAFT_QKV_WIDTH;
  for (uint i = tid; i < R * D; i += 256) {
    const uint row = i / R, token = i % R;
    qvalues[i] = packed[token * P + QW + KW + head * D + row];
  }
  for (uint task = 0; task < (H + 1) * R; ++task) {
    const bool query = task < H * R;
    const uint token = query ? task % R : task - H * R;
    const uint qh = query ? head * H + task / R : head;
    device const bfloat *input = packed + token * P + (query ? 0 : QW) + qh * D;
    device bfloat *dest = query ? queries + task * D : qkeys + token * D;
    device const bfloat *weight = query ? qnorm : knorm;
    const float x = tid < D ? float(input[tid]) : 0.0f;
    const float inv = rms_inverse_of_sums(x * x, D, sums, tid, lane, sg);
    if (tid < D)
      normalized[tid] = bfloat(x * inv * float(weight[tid]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < D / 2) {
      const float a = float(normalized[tid]),
                  b = float(normalized[tid + D / 2]);
      const float c = cosine[token * SPLASH_DRAFT_ROPE_PAIRS + tid],
                  s = sine[token * SPLASH_DRAFT_ROPE_PAIRS + tid];
      dest[tid] = bfloat(a * c - b * s);
      dest[tid + D / 2] = bfloat(b * c + a * s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}

kernel void muse_attention_persistent(
    device bfloat *queries [[buffer(0)]], device bfloat *keys0 [[buffer(1)]],
    device bfloat *keys1 [[buffer(2)]], device bfloat *keys2 [[buffer(3)]],
    device bfloat *keys3 [[buffer(4)]], device bfloat *values0 [[buffer(5)]],
    device bfloat *values1 [[buffer(6)]], device bfloat *values2 [[buffer(7)]],
    device bfloat *values3 [[buffer(8)]], device bfloat *qkeys [[buffer(9)]],
    device bfloat *qvalues [[buffer(10)]], device bfloat *output [[buffer(11)]],
    constant DraftAttentionBatchParams &p [[buffer(12)]],

    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint R = SPLASH_DRAFT_QUERY_ROWS, D = SPLASH_DRAFT_HEAD_DIMENSION;
  constexpr uint M = SPLASH_DRAFT_ATTENTION_ROWS, N = 128,
                 H = SPLASH_DRAFT_KV_HEADS;
  constexpr ulong A = SPLASH_DRAFT_ATTENTION_WIDTH,
                  W = SPLASH_DRAFT_SLIDING_WINDOW;
  constexpr ulong F = M * (D + 2), S = SPLASH_DRAFT_ATTENTION_SPLITS;
  const uint batch = group.y, head = group.x;
  device bfloat *q = queries + batch * R * A + head * M * D;
  device bfloat *k = qkeys + (batch * H + head) * R * D;
  device bfloat *v = qvalues + (batch * H + head) * D * R;
  device bfloat *keys =
      SPLASH_LANE_BINDING(batch, keys0, keys1, keys2, keys3) + head * W * D;
  device bfloat *values =
      SPLASH_LANE_BINDING(batch, values0, values1, values2, values3) +
      head * W * D;
  device float *partials =
      reinterpret_cast<device float *>(queries + p.lanes * R * A) +
      (batch * H + head) * S * F;
  threadgroup float scratch[M * N + 3 * M];

  for (uint split = 0; split < S; ++split) {
    draft_attention_split_phase(q, keys, values, k, v, partials + split * F,
                                p.value_stride, p.cache_length[batch], split,
                                scratch, scratch + M * N, scratch + M * N + M,
                                scratch + M * N + 2 * M, tid, lane, sg);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
  muse_reduce_row_major(partials, output + batch * R * A + head * (M / R) * D,
                        tid);
}

kernel void muse_attention_all(
    device bfloat *queries [[buffer(0)]], device bfloat *keys0 [[buffer(1)]],
    device bfloat *keys1 [[buffer(2)]], device bfloat *keys2 [[buffer(3)]],
    device bfloat *keys3 [[buffer(4)]], device bfloat *values0 [[buffer(5)]],
    device bfloat *values1 [[buffer(6)]], device bfloat *values2 [[buffer(7)]],
    device bfloat *values3 [[buffer(8)]], device bfloat *qkeys [[buffer(9)]],
    device bfloat *qvalues [[buffer(10)]], device bfloat *output [[buffer(11)]],
    constant DraftAttentionBatchParams &p [[buffer(17)]],

    device const bfloat *packed [[buffer(12)]],
    device const bfloat *qnorm [[buffer(13)]],
    device const bfloat *knorm [[buffer(14)]],
    device const float *cosine [[buffer(15)]],
    device const float *sine [[buffer(16)]],

    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint R = SPLASH_DRAFT_QUERY_ROWS, D = SPLASH_DRAFT_HEAD_DIMENSION;
  constexpr uint M = SPLASH_DRAFT_ATTENTION_ROWS, N = 128,
                 H = SPLASH_DRAFT_KV_HEADS;
  constexpr ulong A = SPLASH_DRAFT_ATTENTION_WIDTH,
                  W = SPLASH_DRAFT_SLIDING_WINDOW;
  constexpr ulong F = M * (D + 2), S = SPLASH_DRAFT_ATTENTION_SPLITS;
  const uint batch = group.y, head = group.x;
  device bfloat *q = queries + batch * R * A + head * M * D;
  device bfloat *k = qkeys + (batch * H + head) * R * D;
  device bfloat *v = qvalues + (batch * H + head) * D * R;
  device bfloat *keys =
      SPLASH_LANE_BINDING(batch, keys0, keys1, keys2, keys3) + head * W * D;
  device bfloat *values =
      SPLASH_LANE_BINDING(batch, values0, values1, values2, values3) +
      head * W * D;
  device float *partials =
      reinterpret_cast<device float *>(queries + p.lanes * R * A) +
      (batch * H + head) * S * F;
  threadgroup float scratch[M * N + 3 * M];

  threadgroup bfloat normalized[D];
  muse_head_prepare(packed + batch * R * SPLASH_DRAFT_QKV_WIDTH, q, qnorm,
                    knorm, cosine + batch * R * SPLASH_DRAFT_ROPE_PAIRS,
                    sine + batch * R * SPLASH_DRAFT_ROPE_PAIRS, k, v, scratch,
                    normalized, head, tid, lane, sg);

  for (uint split = 0; split < S; ++split) {
    draft_attention_split_phase(q, keys, values, k, v, partials + split * F,
                                p.value_stride, p.cache_length[batch], split,
                                scratch, scratch + M * N, scratch + M * N + M,
                                scratch + M * N + 2 * M, tid, lane, sg);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
  muse_reduce_row_major(partials, output + batch * R * A + head * (M / R) * D,
                        tid);
}

inline void muse_compact_prepare_phase(
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
  constexpr uint QWidth = SPLASH_DRAFT_ATTENTION_WIDTH,
                 KWidth = KVHeads * HeadDim;
  constexpr uint PackedWidth = SPLASH_DRAFT_QKV_WIDTH;
  constexpr uint QueryTasks = Rows * QHeads;
  constexpr uint Tasks = QueryTasks + Rows * KVHeads;

  const uint element = group * 128 + thread_index;
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
    const float inverse = rms_inverse_of_sums<4>(
        value * value, HeadDim, reductions, thread_index, lane, simd_group);
    if (thread_index < HeadDim)
      head[thread_index] =
          bfloat(value * inverse * float(weight[thread_index]));
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

kernel void muse_prepare_compact(device const bfloat *packed [[buffer(0)]],
                                 device bfloat *queries [[buffer(1)]],
                                 device const bfloat *qn [[buffer(2)]],
                                 device const bfloat *kn [[buffer(3)]],
                                 device const float *cosine [[buffer(4)]],
                                 device const float *sine [[buffer(5)]],
                                 device bfloat *qkeys [[buffer(6)]],
                                 device bfloat *qvalues [[buffer(7)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr ulong R = SPLASH_DRAFT_QUERY_ROWS, A = SPLASH_DRAFT_ATTENTION_WIDTH,
                  H = SPLASH_DRAFT_KV_HEADS, D = SPLASH_DRAFT_HEAD_DIMENSION,
                  P = SPLASH_DRAFT_QKV_WIDTH, T = R * SPLASH_DRAFT_ROPE_PAIRS;
  threadgroup float sums[4];
  threadgroup bfloat normalized[D];
  muse_compact_prepare_phase(
      packed + group.y * R * P, queries + group.y * R * A, qn, kn,
      cosine + group.y * T, sine + group.y * T, qkeys + group.y * H * R * D,
      qvalues + group.y * H * D * R, sums, normalized, group.x, tid, lane, sg);
}
kernel void muse_prepare_serial(device const bfloat *packed [[buffer(0)]],
                                device bfloat *queries [[buffer(1)]],
                                device const bfloat *qn [[buffer(2)]],
                                device const bfloat *kn [[buffer(3)]],
                                device const float *cosine [[buffer(4)]],
                                device const float *sine [[buffer(5)]],
                                device bfloat *qkeys [[buffer(6)]],
                                device bfloat *qvalues [[buffer(7)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr ulong R = SPLASH_DRAFT_QUERY_ROWS, A = SPLASH_DRAFT_ATTENTION_WIDTH,
                  H = SPLASH_DRAFT_KV_HEADS, D = SPLASH_DRAFT_HEAD_DIMENSION,
                  P = SPLASH_DRAFT_QKV_WIDTH, T = R * SPLASH_DRAFT_ROPE_PAIRS,
                  M = SPLASH_DRAFT_ATTENTION_ROWS;
  threadgroup float sums[8];
  threadgroup bfloat normalized[D];
  muse_head_prepare(packed + group.y * R * P,
                    queries + group.y * R * A + group.x * M * D, qn, kn,
                    cosine + group.y * T, sine + group.y * T,
                    qkeys + (group.y * H + group.x) * R * D,
                    qvalues + (group.y * H + group.x) * D * R, sums, normalized,
                    group.x, tid, lane, sg);
}
kernel void muse_reduce_parallel(device bfloat *queries [[buffer(0)]],
                                 device bfloat *output [[buffer(1)]],
                                 constant DraftAttentionBatchParams &p
                                 [[buffer(2)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  constexpr ulong R = SPLASH_DRAFT_QUERY_ROWS, A = SPLASH_DRAFT_ATTENTION_WIDTH,
                  H = SPLASH_DRAFT_KV_HEADS, D = SPLASH_DRAFT_HEAD_DIMENSION,
                  M = SPLASH_DRAFT_ATTENTION_ROWS,
                  S = SPLASH_DRAFT_ATTENTION_SPLITS, F = M * (D + 2);
  device float *partials =
      reinterpret_cast<device float *>(queries + p.lanes * R * A) +
      (group.y * H + group.x) * S * F;
  muse_reduce_row_major(partials,
                        output + group.y * R * A + group.x * (M / R) * D, tid);
}

inline void muse_split_coherent(
    device bfloat *queries, device bfloat *keys, device bfloat *values,
    device bfloat *query_keys, device bfloat *query_values,
    device coherent(device) float *partial, uint value_stride,
    uint cache_length, uint split, threadgroup float *score_storage,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, uint thread_index, uint lane,
    uint simd_group) {
  constexpr ushort M = SPLASH_DRAFT_ATTENTION_ROWS, N = 128,
                   D = SPLASH_DRAFT_HEAD_DIMENSION, TileK = 64;
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS,
                 Window = SPLASH_DRAFT_SLIDING_WINDOW,
                 Splits = SPLASH_DRAFT_ATTENTION_SPLITS;
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
      auto partial_scores =
          qk.template get_destination_cooperative_tensor<decltype(qs),
                                                         decltype(ks), float>();
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
      auto partial_values =
          pv.template get_destination_cooperative_tensor<decltype(ps),
                                                         decltype(vs), float>();
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
    threadgroup bfloat *current_values = reinterpret_cast<threadgroup bfloat *>(
        score_storage + M * CurrentValueTile + M * CurrentN);
    auto current_scores =
        tensor(current_raw_scores, dextents<int, 2>{CurrentN, M},
               array<int, 2>{1, CurrentN});
    constexpr auto current_qk_descriptor =
        matmul2d_descriptor(M, CurrentN, TileK, false, true, false);
    matmul2d<current_qk_descriptor, execution_simdgroups<8>> current_qk;
    auto current_kt =
        tensor(query_keys, dextents<int, 2>{D, CurrentN}, array<int, 2>{1, D});
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

  // MPP's store rejects coherent tensor element types. Publish each fragment
  // directly, preserving its already computed FP32 values and index mapping.
  for (ushort i = 0; i < running.get_capacity(); ++i) {
    auto index = running.get_multidimensional_index(i);
    partial[index[1] * D + index[0]] = running[i];
  }
  if (thread_index < M) {
    partial[M * D + thread_index] = row_max[thread_index];
    partial[M * D + M + thread_index] = row_sum[thread_index];
  }
}

kernel void muse_attention_last_arriver(
    device bfloat *queries [[buffer(0)]], device bfloat *keys0 [[buffer(1)]],
    device bfloat *keys1 [[buffer(2)]], device bfloat *keys2 [[buffer(3)]],
    device bfloat *keys3 [[buffer(4)]], device bfloat *values0 [[buffer(5)]],
    device bfloat *values1 [[buffer(6)]], device bfloat *values2 [[buffer(7)]],
    device bfloat *values3 [[buffer(8)]], device bfloat *qkeys [[buffer(9)]],
    device bfloat *qvalues [[buffer(10)]], device bfloat *output [[buffer(11)]],
    device atomic_uint *finished [[buffer(12)]],
    constant DraftAttentionBatchParams &p [[buffer(13)]],

    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint R = SPLASH_DRAFT_QUERY_ROWS, D = SPLASH_DRAFT_HEAD_DIMENSION;
  constexpr uint M = SPLASH_DRAFT_ATTENTION_ROWS, N = 128,
                 H = SPLASH_DRAFT_KV_HEADS;
  constexpr ulong A = SPLASH_DRAFT_ATTENTION_WIDTH,
                  W = SPLASH_DRAFT_SLIDING_WINDOW;
  constexpr ulong F = M * (D + 2), S = SPLASH_DRAFT_ATTENTION_SPLITS;
  const uint batch = group.y, head = group.x;
  device bfloat *q = queries + batch * R * A + head * M * D;
  device bfloat *k = qkeys + (batch * H + head) * R * D;
  device bfloat *v = qvalues + (batch * H + head) * D * R;
  device bfloat *keys =
      SPLASH_LANE_BINDING(batch, keys0, keys1, keys2, keys3) + head * W * D;
  device bfloat *values =
      SPLASH_LANE_BINDING(batch, values0, values1, values2, values3) +
      head * W * D;
  device coherent(device) float *partials =
      reinterpret_cast<device coherent(device) float *>(queries +
                                                        p.lanes * R * A) +
      (batch * H + head) * S * F;
  threadgroup float scratch[M * N + 3 * M];

  const uint split = group.z;
  muse_split_coherent(q, keys, values, k, v, partials + split * F,
                      p.value_stride, p.cache_length[batch], split, scratch,
                      scratch + M * N, scratch + M * N + M,
                      scratch + M * N + 2 * M, tid, lane, sg);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  threadgroup uint final_group = 0;
  if (tid == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);
    final_group = atomic_fetch_add_explicit(finished + batch * H + head, 1u,
                                            memory_order_relaxed) +
                      1 ==
                  S;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!final_group)
    return;
  if (tid == 0)
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  muse_reduce_row_major(partials, output + batch * R * A + head * (M / R) * D,
                        tid);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (tid == 0)
    atomic_store_explicit(finished + batch * H + head, 0u,
                          memory_order_relaxed);
}

inline float muse_four_partial_rms(threadgroup float *parts) {
#pragma clang fp reassociate(off)
  float sum = 0;
  for (uint i = 0; i < 4; ++i)
    sum += parts[i];
  return rsqrt(sum / SPLASH_DRAFT_HEAD_DIMENSION + SPLASH_RMS_EPSILON);
}
template <uint Pack>
inline void muse_prepare_packed_phase(
    device const bfloat *packed, device bfloat *queries,
    device const bfloat *qn, device const bfloat *kn,
    device const float *cosine, device const float *sine, device bfloat *qkeys,
    device bfloat *qvalues, threadgroup float *parts,
    threadgroup bfloat *normalized, uint group, uint tid, uint lane, uint sg) {
  constexpr uint R = SPLASH_DRAFT_QUERY_ROWS, H = SPLASH_DRAFT_KV_HEADS,
                 D = SPLASH_DRAFT_HEAD_DIMENSION,
                 A = SPLASH_DRAFT_ATTENTION_WIDTH, P = SPLASH_DRAFT_QKV_WIDTH,
                 Threads = 128 * Pack;
  static_assert(D == 128);
  const uint element = group * Threads + tid;
  if (element < R * H * D) {
    const uint row = element / (H * D), rest = element % (H * D);
    qvalues[(rest / D * D + rest % D) * R + row] =
        packed[row * P + A + H * D + rest];
  }
  const uint sub = tid / 128, local = tid % 128, task = group * Pack + sub;
  const bool query = task < R * SPLASH_DRAFT_QUERY_HEADS;
  const uint item = query ? task : task - R * SPLASH_DRAFT_QUERY_HEADS;
  const uint heads = query ? SPLASH_DRAFT_QUERY_HEADS : H, row = item / heads,
             head = item % heads;
  device const bfloat *source = packed + row * P + (query ? 0 : A) + head * D;
  device const bfloat *weight = query ? qn : kn;
  device bfloat *dest = (query ? queries : qkeys) + (head * R + row) * D;
  const float x = float(source[local]);
  const float subtotal = simd_sum(x * x);
  if (lane == 0)
    parts[sub * 4 + sg % 4] = subtotal;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local == 0)
    parts[sub * 4] = muse_four_partial_rms(parts + sub * 4);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalized[sub * D + local] =
      bfloat(x * parts[sub * 4] * float(weight[local]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local < D / 2) {
    const float a = float(normalized[sub * D + local]),
                b = float(normalized[sub * D + local + D / 2]);
    const float c = cosine[row * SPLASH_DRAFT_ROPE_PAIRS + local],
                s = sine[row * SPLASH_DRAFT_ROPE_PAIRS + local];
    dest[local] = bfloat(a * c - b * s);
    dest[local + D / 2] = bfloat(b * c + a * s);
  }
}

kernel void muse_prepare_pack2(device const bfloat *packed [[buffer(0)]],
                               device bfloat *queries [[buffer(1)]],
                               device const bfloat *qn [[buffer(2)]],
                               device const bfloat *kn [[buffer(3)]],
                               device const float *cosine [[buffer(4)]],
                               device const float *sine [[buffer(5)]],
                               device bfloat *qkeys [[buffer(6)]],
                               device bfloat *qvalues [[buffer(7)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr ulong R = SPLASH_DRAFT_QUERY_ROWS, A = SPLASH_DRAFT_ATTENTION_WIDTH,
                  H = SPLASH_DRAFT_KV_HEADS, D = SPLASH_DRAFT_HEAD_DIMENSION,
                  P = SPLASH_DRAFT_QKV_WIDTH, T = R * SPLASH_DRAFT_ROPE_PAIRS;
  threadgroup float parts[2 * 4];
  threadgroup bfloat normalized[2 * D];
  muse_prepare_packed_phase<2>(
      packed + group.y * R * P, queries + group.y * R * A, qn, kn,
      cosine + group.y * T, sine + group.y * T, qkeys + group.y * H * R * D,
      qvalues + group.y * H * D * R, parts, normalized, group.x, tid, lane, sg);
}

kernel void muse_prepare_pack4(device const bfloat *packed [[buffer(0)]],
                               device bfloat *queries [[buffer(1)]],
                               device const bfloat *qn [[buffer(2)]],
                               device const bfloat *kn [[buffer(3)]],
                               device const float *cosine [[buffer(4)]],
                               device const float *sine [[buffer(5)]],
                               device bfloat *qkeys [[buffer(6)]],
                               device bfloat *qvalues [[buffer(7)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr ulong R = SPLASH_DRAFT_QUERY_ROWS, A = SPLASH_DRAFT_ATTENTION_WIDTH,
                  H = SPLASH_DRAFT_KV_HEADS, D = SPLASH_DRAFT_HEAD_DIMENSION,
                  P = SPLASH_DRAFT_QKV_WIDTH, T = R * SPLASH_DRAFT_ROPE_PAIRS;
  threadgroup float parts[4 * 4];
  threadgroup bfloat normalized[4 * D];
  muse_prepare_packed_phase<4>(
      packed + group.y * R * P, queries + group.y * R * A, qn, kn,
      cosine + group.y * T, sine + group.y * T, qkeys + group.y * H * R * D,
      qvalues + group.y * H * D * R, parts, normalized, group.x, tid, lane, sg);
}

kernel void muse_prepare_pack8(device const bfloat *packed [[buffer(0)]],
                               device bfloat *queries [[buffer(1)]],
                               device const bfloat *qn [[buffer(2)]],
                               device const bfloat *kn [[buffer(3)]],
                               device const float *cosine [[buffer(4)]],
                               device const float *sine [[buffer(5)]],
                               device bfloat *qkeys [[buffer(6)]],
                               device bfloat *qvalues [[buffer(7)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr ulong R = SPLASH_DRAFT_QUERY_ROWS, A = SPLASH_DRAFT_ATTENTION_WIDTH,
                  H = SPLASH_DRAFT_KV_HEADS, D = SPLASH_DRAFT_HEAD_DIMENSION,
                  P = SPLASH_DRAFT_QKV_WIDTH, T = R * SPLASH_DRAFT_ROPE_PAIRS;
  threadgroup float parts[8 * 4];
  threadgroup bfloat normalized[8 * D];
  muse_prepare_packed_phase<8>(
      packed + group.y * R * P, queries + group.y * R * A, qn, kn,
      cosine + group.y * T, sine + group.y * T, qkeys + group.y * H * R * D,
      qvalues + group.y * H * D * R, parts, normalized, group.x, tid, lane, sg);
}

kernel void muse_split_serial(
    device bfloat *queries [[buffer(0)]], device bfloat *keys0 [[buffer(1)]],
    device bfloat *keys1 [[buffer(2)]], device bfloat *keys2 [[buffer(3)]],
    device bfloat *keys3 [[buffer(4)]], device bfloat *values0 [[buffer(5)]],
    device bfloat *values1 [[buffer(6)]], device bfloat *values2 [[buffer(7)]],
    device bfloat *values3 [[buffer(8)]], device bfloat *qkeys [[buffer(9)]],
    device bfloat *qvalues [[buffer(10)]], device bfloat *output [[buffer(11)]],
    constant DraftAttentionBatchParams &p [[buffer(12)]],

    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint R = SPLASH_DRAFT_QUERY_ROWS, D = SPLASH_DRAFT_HEAD_DIMENSION;
  constexpr uint M = SPLASH_DRAFT_ATTENTION_ROWS, N = 128,
                 H = SPLASH_DRAFT_KV_HEADS;
  constexpr ulong A = SPLASH_DRAFT_ATTENTION_WIDTH,
                  W = SPLASH_DRAFT_SLIDING_WINDOW;
  constexpr ulong F = M * (D + 2), S = SPLASH_DRAFT_ATTENTION_SPLITS;
  const uint batch = group.y, head = group.x;
  device bfloat *q = queries + batch * R * A + head * M * D;
  device bfloat *k = qkeys + (batch * H + head) * R * D;
  device bfloat *v = qvalues + (batch * H + head) * D * R;
  device bfloat *keys =
      SPLASH_LANE_BINDING(batch, keys0, keys1, keys2, keys3) + head * W * D;
  device bfloat *values =
      SPLASH_LANE_BINDING(batch, values0, values1, values2, values3) +
      head * W * D;
  device float *partials =
      reinterpret_cast<device float *>(queries + p.lanes * R * A) +
      (batch * H + head) * S * F;
  threadgroup float scratch[M * N + 3 * M];

  for (uint split = 0; split < S; ++split) {
    draft_attention_split_phase(q, keys, values, k, v, partials + split * F,
                                p.value_stride, p.cache_length[batch], split,
                                scratch, scratch + M * N, scratch + M * N + M,
                                scratch + M * N + 2 * M, tid, lane, sg);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
  (void)output;
}
