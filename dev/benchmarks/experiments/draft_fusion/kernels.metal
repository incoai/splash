// Experimental kernels; the production split arithmetic stays unchanged.
#include "metal/kernels/decode/draft.metal"

inline void muse_reduce_row_major(device const float *partials,
                                  device bfloat *output, uint thread_index) {
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
