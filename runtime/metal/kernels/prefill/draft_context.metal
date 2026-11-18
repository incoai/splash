#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/draft_context_kv.h"
#include "metal/kernels/common/draft_context_window.h"

// The draft's context kernels take the ring length of the draft's geometry:
// the shipped drafts commit into a 2048-slot ring, Ripple's into a 4096-slot
// one, and every compiled variant's kernels are suffix-named for it.

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
prefill_draft_context_kv_kv8_h2_d128_w4096_s8(device const bfloat *context_kv [[buffer(0)]],
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
  draft_context_kv_phase<4096>(context_kv, k_norm, rope_cos, rope_sin, keys,
                               values, params.start_position, params.tokens,
                               task, thread_index, lane, simd_group,
                               reductions, normalized);
}

template <uint Window>
inline void draft_context_store_phase(device const bfloat *context, device uchar *window,
                                      constant DraftContextWindowParams &params, uint group, uint lane, uint simd) {
  DraftContextGroup at;
  if (draft_context_group<Window>(params, group, simd, at))
    draft_context_store<Window>(context, window, params, at, lane);
}

template <uint Window>
inline void draft_context_load_phase(
    device const uchar *window, device bfloat *context,
    constant DraftContextWindowParams &params, uint threadgroup_index,
    uint lane, uint simd_group) {
  DraftContextGroup at;
  if (!draft_context_group<Window>(params, threadgroup_index, simd_group, at))
    return;
  device const half2 *groups = reinterpret_cast<device const half2 *>(
      window + draft_context_codes_bytes(params.width, Window));
  const half2 group = groups[ulong(at.slot) * at.groups + at.group];
  const uchar byte = window[ulong(at.slot) * (params.width / 2) +
                            at.group * (SPLASH_DRAFT_CONTEXT_GROUP / 2) + lane];
  const float scale = float(group.x), minimum = float(group.y);
  device bfloat *values = context + ulong(at.row) * params.width +
                          at.group * SPLASH_DRAFT_CONTEXT_GROUP + 2 * lane;
  values[0] = bfloat(fma(float(byte & 15), scale, minimum));
  values[1] = bfloat(fma(float(byte >> 4), scale, minimum));
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
  draft_context_store_phase<SPLASH_DRAFT_SLIDING_WINDOW>(
      context, window, params, threadgroup_index, lane, simd_group);
}

kernel void
prefill_draft_context_store_kv8_h2_d128_w4096_s8(
    device const bfloat *context [[buffer(0)]],
    device uchar *window [[buffer(1)]],
    constant DraftContextWindowParams &params [[buffer(2)]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  draft_context_store_phase<4096>(context, window, params, threadgroup_index,
                                  lane, simd_group);
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
  draft_context_load_phase<SPLASH_DRAFT_SLIDING_WINDOW>(
      window, context, params, threadgroup_index, lane, simd_group);
}

kernel void
prefill_draft_context_load_kv8_h2_d128_w4096_s8(
    device const uchar *window [[buffer(0)]],
    device bfloat *context [[buffer(1)]],
    constant DraftContextWindowParams &params [[buffer(2)]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  draft_context_load_phase<4096>(window, context, params, threadgroup_index,
                                 lane, simd_group);
}


kernel void
prefill_draft_context_kv_kv8_h2_d128_w16384_s8(device const bfloat *context_kv [[buffer(0)]],
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
  draft_context_kv_phase<16384>(context_kv, k_norm, rope_cos, rope_sin, keys,
                               values, params.start_position, params.tokens,
                               task, thread_index, lane, simd_group,
                               reductions, normalized);
}

kernel void
prefill_draft_context_kv_kv8_h2_d128_w4096_s8_nope(device const bfloat *context_kv [[buffer(0)]],
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
  draft_context_kv_phase<4096, false>(context_kv, k_norm, rope_cos, rope_sin, keys,
                               values, params.start_position, params.tokens,
                               task, thread_index, lane, simd_group,
                               reductions, normalized);
}

kernel void
prefill_draft_context_kv_kv8_h2_d128_w16384_s8_nope(device const bfloat *context_kv [[buffer(0)]],
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
  draft_context_kv_phase<16384, false>(context_kv, k_norm, rope_cos, rope_sin, keys,
                               values, params.start_position, params.tokens,
                               task, thread_index, lane, simd_group,
                               reductions, normalized);
}

kernel void
prefill_draft_context_store_kv8_h2_d128_w16384_s8(
    device const bfloat *context [[buffer(0)]],
    device uchar *window [[buffer(1)]],
    constant DraftContextWindowParams &params [[buffer(2)]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  draft_context_store_phase<16384>(context, window, params, threadgroup_index,
                                  lane, simd_group);
}

kernel void
prefill_draft_context_load_kv8_h2_d128_w16384_s8(
    device const uchar *window [[buffer(0)]],
    device bfloat *context [[buffer(1)]],
    constant DraftContextWindowParams &params [[buffer(2)]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  draft_context_load_phase<16384>(window, context, params, threadgroup_index,
                                 lane, simd_group);
}

kernel void
prefill_draft_context_kv_nope(device const bfloat *context_kv [[buffer(0)]],
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
  draft_context_kv_phase<SPLASH_DRAFT_SLIDING_WINDOW, false>(context_kv, k_norm, rope_cos, rope_sin, keys, values,
                         params.start_position, params.tokens, task,
                         thread_index, lane, simd_group, reductions,
                         normalized);
}
