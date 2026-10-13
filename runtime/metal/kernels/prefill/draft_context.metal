#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/draft_context_kv.h"

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

// One simdgroup per row and group of the context window: lane l holds the
// group's values 2l and 2l + 1, which share one byte of codes. The code of x
// is round((x - minimum) / scale) with the scale and minimum as stored, so
// loading reads back minimum + code * scale.
struct DraftContextGroup {
  uint row;
  uint group;
  uint groups;
  uint slot;
};

inline bool draft_context_group(constant DraftContextWindowParams &params,
                                uint threadgroup_index, uint simd_group,
                                thread DraftContextGroup &result) {
  result.groups = params.width / SPLASH_DRAFT_CONTEXT_GROUP;
  const uint task = threadgroup_index * SPLASH_DRAFT_CONTEXT_SIMDGROUPS + simd_group;
  result.row = task / result.groups;
  result.group = task % result.groups;
  result.slot =
      (params.start_position + result.row) % SPLASH_DRAFT_SLIDING_WINDOW;
  return result.row < params.rows;
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
  if (!draft_context_group(params, threadgroup_index, simd_group, at))
    return;
  device const bfloat *values = context + ulong(at.row) * params.width +
                                at.group * SPLASH_DRAFT_CONTEXT_GROUP + 2 * lane;
  const float2 x = float2(float(values[0]), float(values[1]));
  const float low = simd_min(min(x.x, x.y));
  const float high = simd_max(max(x.x, x.y));
  const half scale = half(precise::divide(high - low, 15.0f));
  const half minimum = half(low);
  uint2 code = uint2(0);
  if (scale > half(0)) {
    const float2 steps = float2(precise::divide(x.x - float(minimum), float(scale)),
                                precise::divide(x.y - float(minimum), float(scale)));
    code = uint2(clamp(rint(steps), 0.0f, 15.0f));
  }
  device uchar *codes = window + ulong(at.slot) * (params.width / 2);
  codes[at.group * (SPLASH_DRAFT_CONTEXT_GROUP / 2) + lane] =
      uchar(code.x | (code.y << 4));
  if (lane == 0) {
    device half2 *groups = reinterpret_cast<device half2 *>(
        window + draft_context_codes_bytes(params.width));
    groups[ulong(at.slot) * at.groups + at.group] = half2(scale, minimum);
  }
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
