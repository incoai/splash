#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/draft_context_kv.h"
#include "metal/kernels/common/draft_context_window.h"

kernel void
prefill_draft_context_kv(device const bfloat *context_kv [[buffer(0)]],
                         device const bfloat *k_norm [[buffer(1)]],
                         device const float *rope_cos [[buffer(2)]],
                         device const float *rope_sin [[buffer(3)]],
                         device bfloat *keys [[buffer(4)]],
                         device bfloat *values [[buffer(5)]],
                         constant DraftContextParams &params [[buffer(6)]],
                         uint task [[threadgroup_position_in_grid]],
                         uint thread_index [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float reductions[8];
  threadgroup bfloat normalized[SPLASH_DRAFT_HEAD_DIMENSION];
  draft_context_kv_phase(context_kv, k_norm, rope_cos, rope_sin, keys, values,
                         params.start_position, params.tokens, task,
                         thread_index, lane, simd_group, reductions,
                         normalized);
}

kernel void
prefill_draft_context_store(device const bfloat *context [[buffer(0)]],
                            device uchar *window [[buffer(1)]],
                            constant DraftContextWindowParams &params
                            [[buffer(2)]],
                            uint threadgroup_index
                            [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint simd_group
                            [[simdgroup_index_in_threadgroup]]) {
  DraftContextGroup at;
  if (draft_context_group(params, threadgroup_index, simd_group, at))
    draft_context_store(context, window, params, at, lane);
}

kernel void
prefill_draft_context_load(device const uchar *window [[buffer(0)]],
                           device bfloat *context [[buffer(1)]],
                           constant DraftContextWindowParams &params
                           [[buffer(2)]],
                           uint threadgroup_index
                           [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simd_group
                           [[simdgroup_index_in_threadgroup]]) {
  DraftContextGroup at;
  if (!draft_context_group(params, threadgroup_index, simd_group, at))
    return;
  device const half2 *groups = reinterpret_cast<device const half2 *>(
      window + draft_context_codes_bytes(params.width));
  const half2 group = groups[ulong(at.slot) * at.groups + at.group];
  const uchar byte = window[ulong(at.slot) * (params.width / 2) +
                            at.group * (SPLASH_DRAFT_CONTEXT_GROUP / 2) + lane];
  const float scale = float(group.x), minimum = float(group.y);
  device bfloat *values = context + ulong(at.row) * params.width +
                          at.group * SPLASH_DRAFT_CONTEXT_GROUP + 2 * lane;
  values[0] = bfloat(fma(float(byte & 15), scale, minimum));
  values[1] = bfloat(fma(float(byte >> 4), scale, minimum));
}
