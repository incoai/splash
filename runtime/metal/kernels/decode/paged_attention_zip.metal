#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/paged_attention_zip_tile.h"

// ZBF16 (abi/KvZip.h) verify store and split. The reduce is the BF16 one.

// Store: per lane, one threadgroup per (tensor, KV head, page) of the two
// pages its eight rows can touch; a lane whose rows stay in one page leaves
// the second page's threadgroups idle.
#define PAGED_VERIFY_ZIP_STORE(Name, Heads)                                    \
  kernel void Name(                                                            \
      device const bfloat *chunk_keys [[buffer(0)]],                           \
      device const bfloat *chunk_values [[buffer(1)]],                         \
      device const SplashKvPage *page_table0 [[buffer(2)]],                    \
      device const SplashKvPage *page_table1 [[buffer(3)]],                    \
      device const SplashKvPage *page_table2 [[buffer(4)]],                    \
      device const SplashKvPage *page_table3 [[buffer(5)]],                    \
      device uchar *codec [[buffer(6)]],                                       \
      constant SplashChunkedPrefillParams *params [[buffer(7)]],               \
      uint group [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint simd_lane [[thread_index_in_simdgroup]],                            \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    threadgroup uint partial[8];                                               \
    constexpr uint SlabsPerLane = 2 * 2 * Heads;                               \
    const uint batch = group / SlabsPerLane;                                   \
    constant SplashChunkedPrefillParams &lane_params = params[batch];          \
    if (!splash_chunk_contract_valid(lane_params) ||                           \
        lane_params.chunk_tokens != SPLASH_TARGET_VERIFY_ROWS ||               \
        lane_params.chunk_stride != SPLASH_VERIFY_CHUNK_STRIDE ||              \
        thread_index >= SplashKvHeadDimension)                                 \
      return;                                                                  \
    device const SplashKvPage *page_table = SPLASH_LANE_BINDING(               \
        batch, page_table0, page_table1, page_table2, page_table3);            \
    constexpr ulong lane_tensor_stride =                                       \
        ulong(Heads) * SPLASH_VERIFY_CHUNK_STRIDE * SplashKvHeadDimension;     \
    splash_kvzip_store_chunk_slab<Heads>(                                      \
        chunk_keys + batch * lane_tensor_stride,                               \
        chunk_values + batch * lane_tensor_stride, page_table, codec,          \
        lane_params, group % SlabsPerLane, partial, thread_index, simd_lane,   \
        simd_group);                                                           \
  }
PAGED_VERIFY_ZIP_STORE(verify_attention_zip_store, 4)
PAGED_VERIFY_ZIP_STORE(verify_attention_zip_store_kv2_g8, 2)
#undef PAGED_VERIFY_ZIP_STORE

// Split: the BF16 verify split's grid and slots over decoded pages.
#define PAGED_VERIFY_ZIP_SPLIT(Name, Heads, Group)                             \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table0 [[buffer(3)]],                    \
      device const SplashKvPage *page_table1 [[buffer(4)]],                    \
      device const SplashKvPage *page_table2 [[buffer(5)]],                    \
      device const SplashKvPage *page_table3 [[buffer(6)]],                    \
      device const uchar *codec [[buffer(7)]],                                 \
      constant SplashVerifyAttentionParams *params [[buffer(8)]],              \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint M = Group * SPLASH_TARGET_VERIFY_ROWS;                      \
    constexpr uint N = SplashKvPageTokens;                                     \
    constexpr uint D = SplashKvHeadDimension;                                  \
    static_assert(M * N * sizeof(float) <= N * D * sizeof(bfloat),             \
                  "a page's scores fit the decoded tile they share");          \
    alignas(16) threadgroup bfloat probabilities[M * N];                       \
    alignas(16) threadgroup bfloat kv_tile[N * D];                             \
    threadgroup float row_max[M];                                              \
    threadgroup float row_sum[M];                                              \
    threadgroup float previous_scale[M];                                       \
    threadgroup atomic_uint rescale;                                           \
    const uint kv_head = group.x, split = group.y, batch = group.z;            \
    constant SplashVerifyAttentionParams &lane_params = params[batch];         \
    if (!splash_verify_attention_contract_valid(lane_params) ||                \
        kv_head >= Heads || split >= lane_params.split_count)                  \
      return;                                                                  \
    constexpr ulong group_stride =                                             \
        ulong(SPLASH_VERIFY_CHUNK_STRIDE) * Group * D;                         \
    device const SplashKvPage *page_table = SPLASH_LANE_BINDING(               \
        batch, page_table0, page_table1, page_table2, page_table3);            \
    const ulong slot =                                                         \
        (ulong(batch) * Heads + kv_head) * lane_params.slot_splits + split;    \
    splash_paged_attention_zip_tile<Heads, Group, SPLASH_TARGET_VERIFY_ROWS>(  \
        queries + (ulong(batch) * Heads + kv_head) * group_stride, page_table, \
        lane_params.kv, codec, kv_head, lane_params.committed_tokens,          \
        SPLASH_TARGET_VERIFY_ROWS, lane_params.split_count, split, partials,   \
        statistics, slot, probabilities, row_max, row_sum, previous_scale,     \
        &rescale, kv_tile, thread_index);                                      \
  }
PAGED_VERIFY_ZIP_SPLIT(verify_attention_zip_split, 4, 6)
PAGED_VERIFY_ZIP_SPLIT(verify_attention_zip_split_kv2_g8, 2, 8)
#undef PAGED_VERIFY_ZIP_SPLIT
