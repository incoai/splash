// The fused attention of speculative verification (ops::omni::Attention::
// addVerifyFused): each sequence's rows are a lane's pending token and the
// draft's proposals (or a draft's catch-up tokens), row r seeing the tokens
// through its own, committed + r. One threadgroup owns a (KV head, row):
// group.x the KV head, group.y the sequence, group.z the row within it.
// It normalizes and rotates its row's two query heads and the key head of
// every row of the sequence through its own (a row sees the keys of the
// rows before it, which their own threadgroups store too: the same bits
// into the same slots), stores those keys and values into their page
// slots, then runs omni_attention_decode_fused's page loop and merge for
// its two query vectors, so a verified row's attention is a decode step's.
// A threadgroup per row rather than per sequence keeps the page loop at a
// decode step's two vectors: the rows run side by side on an otherwise
// idle GPU (at one lane, six rows cost 0.34 ms over 28 layers against a
// decode step's 0.30; dev/benchmarks/omni_asr_verify_bench.mm).
// A forward of up to four rows runs a threadgroup per (query head, row)
// instead, one vector each (omni_attention_verify_fused_head; Omni.cpp has
// the measurements).

#include "metal/abi/Omni.h"
#include "metal/kernels/common/omni_paged_attention.h"
#include <metal_stdlib>

using namespace metal;

// Q query vectors per threadgroup: a KV head's two (group.x the KV head), or
// one query head's (group.x the query head; both threadgroups of a KV head
// store its keys and values, the same bits into the same slots), which
// doubles the threadgroups of a forward of few rows. A vector's arithmetic
// is the same either way.
template <uint Q>
inline void omni_attention_verify_body(device const bfloat *qkv, device const bfloat *q_norm, device const bfloat *k_norm,
                                       device const float *inverse_frequencies,
                                       constant OmniSequenceParams *sequences, device const uint *page_table,
                                       device uchar *pages, device bfloat *output, device float2 *rotations,
                                       constant OmniQkvParams &p, threadgroup float (*query)[OMNI_DECODER_HEAD_DIM],
                                       threadgroup float (*group_output)[Q][OMNI_DECODER_HEAD_DIM],
                                       threadgroup float (*group_max)[Q], threadgroup float (*group_sum)[Q], uint3 group,
                                       uint lane, uint sg) {
  constexpr uint D = kHeadDim, N = kPageTokens, M = Q;
  const uint kv_head = group.x * Q / kGroup, first_head = kv_head * kGroup + (Q == 1 ? group.x % kGroup : 0);
  const uint sequence = group.y, own = group.z;
  if (kv_head >= kKvHeads || sequence >= p.sequences) return;
  constant OmniSequenceParams &s = sequences[sequence];
  if (own >= s.rows) return;
  // Simdgroup tasks: the row's query heads, then the key head and the
  // values of every row through it; lane l holds dims l, l + 32, l + 64,
  // l + 96.
  for (uint task = sg; task < M + 2 * (own + 1); task += 8) {
    const bool is_query = task < M;
    const uint r = is_query ? own : (task - M) / 2;
    const uint row = s.row_offset + r, position = s.position_offset + r, token = s.committed_tokens + r;
    device const bfloat *source = qkv + ulong(row) * kQkvWidth;
    const uint page_index = omni_page_of(s, page_table, p.layer, token);
    if (!is_query && (task - M) % 2 == 1) {
      device bfloat *stored_values = omni_page_values(pages, page_index, kv_head) + (token % N) * D;
#pragma unroll
      for (uint j = 0; j < 4; ++j) stored_values[lane + 32 * j] = source[(kQueryHeads + kKvHeads + kv_head) * D + lane + 32 * j];
      continue;
    }
    const uint head = is_query ? first_head + task : kQueryHeads + kv_head;
    device const bfloat *weight = is_query ? q_norm : k_norm;
    float x[4];
#pragma unroll
    for (uint j = 0; j < 4; ++j) x[j] = float(source[head * D + lane + 32 * j]);
    // A later layer loads the row's rotations alongside the row; the first
    // stores them once, from the row's own threadgroup of query head 0.
    device float2 *row_rotations = rotations + ulong(row) * (D / 2);
    float2 rotation[2];
    if (p.rotations == OMNI_ROTATIONS_LOAD) {
      rotation[0] = row_rotations[lane];
      rotation[1] = row_rotations[lane + 32];
    }
    const float sum = simd_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
    const float inverse = rsqrt(sum / float(D) + p.rms_epsilon);
    float n[4];
#pragma unroll
    for (uint j = 0; j < 4; ++j) n[j] = float(bfloat(x[j] * inverse * float(weight[lane + 32 * j])));
    if (p.rotations != OMNI_ROTATIONS_LOAD) {
      const float a0 = float(position) * inverse_frequencies[lane];
      rotation[0] = float2(precise::cos(a0), precise::sin(a0));
      const float a1 = float(position) * inverse_frequencies[lane + 32];
      rotation[1] = float2(precise::cos(a1), precise::sin(a1));
      if (p.rotations == OMNI_ROTATIONS_STORE && first_head == 0 && task == 0) {
        row_rotations[lane] = rotation[0];
        row_rotations[lane + 32] = rotation[1];
      }
    }
    // The rotation as omni_attention_decode_fused spells it out, so every
    // mode gives the same bits.
    float rotated[4];
#pragma unroll
    for (uint j = 0; j < 2; ++j) {
      const float c = rotation[j].x, s = rotation[j].y;
      rotated[j] = fma(n[j], c, -(n[j + 2] * s));
      rotated[j + 2] = fma(n[j], s, n[j + 2] * c);
    }
    if (is_query) {
#pragma unroll
      for (uint j = 0; j < 4; ++j) query[task][lane + 32 * j] = float(bfloat(rotated[j]));
    } else {
      device bfloat *stored_keys = omni_page_keys(pages, page_index, kv_head) + (token % N) * D;
#pragma unroll
      for (uint j = 0; j < 4; ++j) stored_keys[lane + 32 * j] = bfloat(rotated[j]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  // The row sees tokens 0 .. committed + own.
  const uint visible = s.committed_tokens + own + 1;
  const uint page_count = (visible + N - 1) / N;
  OmniPageState<M> state;
  state.clear();
  for (uint page = sg; page < page_count; page += 8) {
    const uint index = omni_page_of(s, page_table, p.layer, page * N);
    omni_attention_page<M>(state, omni_page_keys(pages, index, kv_head), omni_page_values(pages, index, kv_head), query,
                           page * N, lane, [&](uint) { return visible; });
  }
#pragma unroll
  for (uint m = 0; m < M; ++m) {
#pragma unroll
    for (uint j = 0; j < 8; ++j) {
      const float both = state.acc[m][j] + simd_shuffle_xor(state.acc[m][j], 16);
      if (lane < 16) group_output[sg][m][lane * 8 + j] = both;
    }
    if (lane == 0) {
      group_max[sg][m] = state.running_max[m];
      group_sum[sg][m] = state.running_sum[m];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
#pragma unroll
  for (uint m = 0; m < M; ++m) {
    float maximum = -INFINITY;
    for (uint g = 0; g < 8; ++g) maximum = max(maximum, group_max[g][m]);
    float total = 0.0f, merged[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint g = 0; g < 8; ++g) {
      const float weight = group_max[g][m] == -INFINITY ? 0.0f : fast::exp(group_max[g][m] - maximum);
      total += weight * group_sum[g][m];
#pragma unroll
      for (uint j = 0; j < 4; ++j) merged[j] += weight * group_output[g][m][lane + 32 * j];
    }
    device bfloat *out = output + ulong(s.row_offset + own) * kAttentionWidth + (first_head + m) * D;
#pragma unroll
    for (uint j = 0; j < 4; ++j) out[lane + 32 * j] = bfloat(total > 0.0f ? merged[j] / total : 0.0f);
  }
}

#define OMNI_VERIFY_KERNEL(Name, Q)                                                                                \
  kernel void Name(device const bfloat *qkv [[buffer(0)]], device const bfloat *q_norm [[buffer(1)]],             \
                   device const bfloat *k_norm [[buffer(2)]], device const float *inverse_frequencies [[buffer(3)]], \
                   constant OmniSequenceParams *sequences [[buffer(4)]], device const uint *page_table [[buffer(5)]], \
                   device uchar *pages [[buffer(6)]], device bfloat *output [[buffer(7)]],                         \
                   device float2 *rotations [[buffer(8)]], constant OmniQkvParams &p [[buffer(9)]],                \
                   uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],           \
                   uint sg [[simdgroup_index_in_threadgroup]]) {                                                    \
    threadgroup float query[Q][kHeadDim];                                                                         \
    threadgroup float group_output[8][Q][kHeadDim];                                                               \
    threadgroup float group_max[8][Q], group_sum[8][Q];                                                           \
    omni_attention_verify_body<Q>(qkv, q_norm, k_norm, inverse_frequencies, sequences, page_table, pages, output,  \
                                  rotations, p, query, group_output, group_max, group_sum, group, lane, sg);       \
  }
OMNI_VERIFY_KERNEL(omni_attention_verify_fused, 2)
OMNI_VERIFY_KERNEL(omni_attention_verify_fused_head, 1)
#undef OMNI_VERIFY_KERNEL
