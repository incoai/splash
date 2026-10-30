#include "metal/kernels/common/paged_attention_zip_tile.h"

// ZBF16 (abi/KvZip.h) prefill store, history expansion and split. A long
// chunk attends through the BF16 split and reduce, over the BF16 scratch the
// expansion fills; a short one's split decodes the pages itself, through the
// BF16 reduce.

// Store: one threadgroup per (tensor, KV head, page the chunk touches)
// encodes that page's chunk rows in order.
#define PAGED_PREFILL_ZIP_STORE(Name, Heads)                                   \
  kernel void Name(                                                            \
      device const bfloat *chunk_keys [[buffer(0)]],                           \
      device const bfloat *chunk_values [[buffer(1)]],                         \
      device const SplashKvPage *page_table [[buffer(2)]],                     \
      device uchar *codec [[buffer(3)]],                                       \
      constant SplashChunkedPrefillParams &params [[buffer(4)]],               \
      uint group [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint simd_lane [[thread_index_in_simdgroup]],                            \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    threadgroup uint partial[8];                                               \
    if (!splash_chunk_contract_valid(params) ||                                \
        thread_index >= SPLASH_KVZIP_DIMENSIONS)                                 \
      return;                                                                  \
    splash_kvzip_store_chunk_slab<Heads>(chunk_keys, chunk_values, page_table, \
                                         codec, params, group, partial,        \
                                         thread_index, simd_lane, simd_group); \
  }
PAGED_PREFILL_ZIP_STORE(prefill_attention_zip_store, 4)
PAGED_PREFILL_ZIP_STORE(prefill_attention_zip_store_kv2_g8, 2)
#undef PAGED_PREFILL_ZIP_STORE

// Expand: one threadgroup per (tensor, KV head, visible page) writes the
// page's rows, history and chunk, into the BF16 scratch the prefill attends.
#define PAGED_PREFILL_ZIP_EXPAND(Name, Heads)                                  \
  kernel void Name(                                                            \
      device const SplashKvPage *page_table [[buffer(0)]],                     \
      device const SplashKvPage *scratch_table [[buffer(1)]],                  \
      device const uchar *codec [[buffer(2)]],                                 \
      constant SplashKvZipExpandParams &params [[buffer(3)]],                  \
      uint group [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    if (!splash_chunk_contract_valid(params.chunk) ||                          \
        params.scratch.extent_pages == 0)                                      \
      return;                                                                  \
    splash_kvzip_expand_history<Heads>(page_table, scratch_table, codec,       \
                                       params, group, thread_index);           \
  }
PAGED_PREFILL_ZIP_EXPAND(prefill_attention_zip_expand, 4)
PAGED_PREFILL_ZIP_EXPAND(prefill_attention_zip_expand_kv2_g8, 2)
#undef PAGED_PREFILL_ZIP_EXPAND

// Split: the BF16 prefill split's grid and slots over decoded pages, each
// tile decoding every page it reads.
#define PAGED_PREFILL_ZIP_SPLIT(Name, Heads, Group, OwnScores)                 \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table [[buffer(3)]],                     \
      device const uchar *codec [[buffer(4)]],                                 \
      constant SplashPrefillAttentionParams &params [[buffer(5)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint M = Group * SPLASH_PREFILL_ATTENTION_TILE_ROWS;             \
    constexpr uint N = SplashKvPageTokens;                                     \
    constexpr uint D = SPLASH_KVZIP_DIMENSIONS;                                  \
    static_assert(SPLASH_ATTENTION_TILE_PARTS(Group) == 1, "one part a tile");  \
    static_assert(OwnScores || M * N * sizeof(float) <= N * D * sizeof(bfloat), \
                  "shared scores fit the decoded tile");                       \
    alignas(16) threadgroup float scores[OwnScores ? M * N : 1];               \
    alignas(16) threadgroup bfloat probabilities[M * N];                       \
    alignas(16) threadgroup bfloat kv_tile[N * D];                             \
    threadgroup float row_max[M];                                              \
    threadgroup float row_sum[M];                                              \
    threadgroup float previous_scale[M];                                       \
    threadgroup atomic_uint rescale;                                           \
    const uint kv_head = group.x, split = group.z;                             \
    const SplashAttentionTile tile =                                           \
        splash_prefill_attention_tile<Heads, Group, D>(params, kv_head, group.y); \
    if (!tile.active || split >= tile.splits)                                  \
      return;                                                                  \
    splash_paged_attention_zip_tile<Heads, Group,                              \
                                    SPLASH_PREFILL_ATTENTION_TILE_ROWS,        \
                                    OwnScores>(                                \
        queries + tile.offset, page_table, params.kv, codec, kv_head,          \
        tile.first_position, tile.active_rows, tile.splits, split, partials,   \
        statistics, tile.first_slot + split, scores, probabilities, row_max,   \
        row_sum, previous_scale, &rescale, kv_tile, thread_index);             \
  }
PAGED_PREFILL_ZIP_SPLIT(prefill_attention_zip_split, 4, 6, true)
PAGED_PREFILL_ZIP_SPLIT(prefill_attention_zip_split_kv2_g8, 2, 8, false)
#undef PAGED_PREFILL_ZIP_SPLIT
