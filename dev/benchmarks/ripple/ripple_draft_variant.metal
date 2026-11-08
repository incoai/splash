// Ripple draft-kernel geometry prototypes (step 1): the production draft
// attention and convolution of runtime/metal/kernels/decode/draft.metal with
// their compiled-in geometry as per-instantiation parameters, so a second
// compiled draft variant is measured beside the shipped one before the engine
// grows it. Not a production source: dev/benchmarks/ripple builds this into
// its own metallib and ripple_draft_sweep.mm drives it. The phases are the
// production ones with M (the query rows one threadgroup attends),
// HeadGroups (how many threadgroups share a KV head's query heads), Splits
// (the ring-tile deal-out) and Window (the ring length) as template
// parameters; everything else (8 proposal rows, 8 KV heads, 128-dimension
// heads, the online-softmax tile loop and the fixed split-order reduce) is
// theirs, so a baseline instantiation must time and check like production.
// The head groups of one (lane, KV head) hold disjoint query heads, so only
// the split order a reduce combines is rounding-relevant, as in production.
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/lane_bindings.h"

template <uint HeadsPerGroup, uint Splits, uint Window>
inline void ripple_draft_attention_split_phase(
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
  uint physical_end = physical_start + old_count;
  uint wrapped =
      physical_end > Window ? (physical_end - Window + N - 1) / N : 0;
  uint tail_end = (min(physical_end, Window) + N - 1) / N;
  uint resume = max(physical_start / N, wrapped);
  uint live_tiles = wrapped + tail_end - resume;
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
    // Persistent query K/V contains exactly eight rows. The MPP matmul needs
    // M or N to be a multiple of sixteen, which the eight-token scores of a
    // head group of five or ten query heads cannot give (and one padded
    // sixteen-wide tile miscomputed rows of the shipped geometry, measured
    // 2026-11-05), so the eight scores of every row are plain fp32 dot
    // products instead: at most M x 8 x 128 MACs, beside a ring tile's
    // M x 128 x 128. The K=16 padding the MPP value product needs exists only
    // in threadgroup memory, as in production.
    constexpr ushort CurrentN = Rows;
    constexpr ushort CurrentValueTile = 16;
    threadgroup float *current_raw_scores =
        score_storage + M * CurrentValueTile;
    threadgroup bfloat *current_values =
        reinterpret_cast<threadgroup bfloat *>(
            score_storage + M * CurrentValueTile + M * CurrentN);
    for (uint task = thread_index; task < M * CurrentN; task += 256) {
      const uint row = task / CurrentN;
      const uint token = task % CurrentN;
      float dot = 0.0f;
      for (uint d = 0; d < D; ++d)
        dot += float(queries[row * D + d]) *
               float(query_keys[token * D + d]);
      current_raw_scores[task] = dot;
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

// The production reduce reads one row per 8 threads, which holds 32 rows in
// its 256-thread group; the prototype loops the tasks instead, so any M runs.
template <uint M, uint Splits>
inline void ripple_draft_attention_reduce_phase(
    device const float *partials, device bfloat *output, uint thread_index) {
  constexpr uint D = SPLASH_DRAFT_HEAD_DIMENSION,
                 Stride = M * D + 2 * M, Chunk = D / 8;
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

// The production convolution, its taps a template parameter: tap t of row r
// weighs the row t before it, which row r >= t holds in the block, exactly as
// the production guard `row > 0` holds its one history row.
template <uint Taps>
inline void ripple_draft_conv_phase(device const bfloat *input,
                                    device const bfloat *dynamic,
                                    device const bfloat *base,
                                    device const bfloat *residual,
                                    device bfloat *output, bool finish,
                                    uint hidden, uint group,
                                    uint thread_index) {
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS;
  static_assert(SPLASH_DRAFT_CONVOLUTION_STAGES == 2,
                "a stage prepares or finishes");
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

#define RIPPLE_DRAFT_SPLIT_KERNEL(name, queryHeads, headsPerGroup, splits,     \
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
    constexpr uint M = (headsPerGroup) * SPLASH_DRAFT_QUERY_ROWS;              \
    constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;                            \
    constexpr ulong Attention = (queryHeads) * SPLASH_DRAFT_HEAD_DIMENSION;    \
    constexpr ulong HeadDim = SPLASH_DRAFT_HEAD_DIMENSION;                     \
    constexpr ulong Window = (window);                                         \
    constexpr ulong PartialFloats = M * HeadDim + 2 * M;                       \
    static_assert(queryHeads % (SPLASH_DRAFT_KV_HEADS * (headsPerGroup)) == 0, \
                  "head groups hold whole query heads");                       \
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
    ripple_draft_attention_split_phase<(headsPerGroup), (splits), (window)>(   \
        queries + batch * Rows * Attention +                                   \
            head * (((queryHeads) / SPLASH_DRAFT_KV_HEADS) * Rows * HeadDim) +                       \
            head_group * M * HeadDim,                                          \
        keys + head * Window * HeadDim, values + head * Window * HeadDim,      \
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

#define RIPPLE_DRAFT_REDUCE_KERNEL(name, queryHeads, headsPerGroup, splits)    \
  kernel void name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      constant DraftAttentionBatchParams &params [[buffer(1)]],                \
      uint2 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint HeadGroups =                                                \
        (queryHeads) / SPLASH_DRAFT_KV_HEADS / (headsPerGroup);                \
    constexpr uint M = (headsPerGroup) * SPLASH_DRAFT_QUERY_ROWS;              \
    constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;                            \
    constexpr ulong Attention = (queryHeads) * SPLASH_DRAFT_HEAD_DIMENSION;    \
    constexpr ulong HeadDim = SPLASH_DRAFT_HEAD_DIMENSION;                     \
    constexpr ulong PartialFloats = M * HeadDim + 2 * M;                       \
    uint batch = group.y;                                                      \
    uint head = group.x / HeadGroups;                                          \
    uint head_group = group.x % HeadGroups;                                    \
    device const float *partials =                                             \
        reinterpret_cast<device const float *>(                                \
            queries + params.lanes * Rows * Attention) +                       \
        ((batch * SPLASH_DRAFT_KV_HEADS + head) * HeadGroups + head_group) *   \
            (splits)*PartialFloats;                                            \
    ripple_draft_attention_reduce_phase<M, (splits)>(                          \
        partials,                                                              \
        queries + batch * Rows * Attention +                                   \
            head * (((queryHeads) / SPLASH_DRAFT_KV_HEADS) * Rows * HeadDim) +                       \
            head_group * M * HeadDim,                                          \
        thread_index);                                                         \
  }

#define RIPPLE_DRAFT_CONV_KERNEL(name, taps)                                   \
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
    uint batch = group.y;                                                      \
    ripple_draft_conv_phase<(taps)>(                                           \
        input + batch * Rows * hidden,                                         \
        dynamic + batch * Rows *                                               \
                      (SPLASH_DRAFT_CONVOLUTION_STAGES * (taps) *              \
                       (hidden / SPLASH_DRAFT_CONVOLUTION_GROUP)),             \
        base, residual + batch * Rows * hidden,                                \
        output + batch * Rows * hidden, params.finish != 0, params.hidden,     \
        group.x, thread_index);                                                \
  }

// The shipped geometry, as a harness cross-check against the production
// metallib's draft_attention_bf16_split/reduce.
RIPPLE_DRAFT_SPLIT_KERNEL(ripple_split_q32_h4_s4_w2048, 32, 4, 4, 2048)
RIPPLE_DRAFT_REDUCE_KERNEL(ripple_reduce_q32_h4_s4, 32, 4, 4)
// Ripple's draft: 80 query heads over 8 KV heads, so a head group of five
// query heads (40 rows) shares each KV head between two threadgroups.
RIPPLE_DRAFT_SPLIT_KERNEL(ripple_split_q80_h5_s4_w2048, 80, 5, 4, 2048)
RIPPLE_DRAFT_REDUCE_KERNEL(ripple_reduce_q80_h5_s4, 80, 5, 4)
RIPPLE_DRAFT_SPLIT_KERNEL(ripple_split_q80_h5_s8_w2048, 80, 5, 8, 2048)
RIPPLE_DRAFT_REDUCE_KERNEL(ripple_reduce_q80_h5_s8, 80, 5, 8)
// Ripple's draft ring holds 4096 positions (its config's sliding_window).
RIPPLE_DRAFT_SPLIT_KERNEL(ripple_split_q80_h5_s4_w4096, 80, 5, 4, 4096)
RIPPLE_DRAFT_REDUCE_KERNEL(ripple_reduce_q80_h5_s4_w4096, 80, 5, 4)
RIPPLE_DRAFT_SPLIT_KERNEL(ripple_split_q80_h5_s8_w4096, 80, 5, 8, 4096)
RIPPLE_DRAFT_REDUCE_KERNEL(ripple_reduce_q80_h5_s8_w4096, 80, 5, 8)
// The shipped two-tap convolution and Ripple's four-tap one.
RIPPLE_DRAFT_CONV_KERNEL(ripple_conv_t2, 2)
RIPPLE_DRAFT_CONV_KERNEL(ripple_conv_t4, 4)
// The one-80-row-group design the m80 probe asks about: the metallib builds
// it, and only the device accepts or refuses it at pipeline creation.
RIPPLE_DRAFT_SPLIT_KERNEL(ripple_split_q80_h10_s4_w2048, 80, 10, 4, 2048)
RIPPLE_DRAFT_REDUCE_KERNEL(ripple_reduce_q80_h10_s4, 80, 10, 4)
