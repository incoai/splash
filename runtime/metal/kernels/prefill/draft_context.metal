#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/draft_context_kv.h"

kernel void prefill_gather_last_hidden_rows8(
    device const bfloat *input [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant LastHiddenRowsParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  uint kept = min(params.rows, 8u);
  uint start = params.rows - kept;
  for (uint element = index; element < kept * params.width;
       element += grid_size) {
    uint row = element / params.width;
    uint dim = element % params.width;
    output[element] = input[(start + row) * params.width + dim];
  }
}

kernel void
prefill_draft_context_kv(device const bfloat *context_qkv [[buffer(0)]],
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
  threadgroup bfloat normalized[128];
  draft_context_kv_phase(context_qkv, k_norm, rope_cos, rope_sin, keys, values,
                         params.cache_stride, params.start_position, params.paged,
                         params.pages, params.tokens, task, thread_index, lane,
                         simd_group, reductions, normalized);
}
