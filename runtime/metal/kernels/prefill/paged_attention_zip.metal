#include "metal/kernels/common/paged_attention_zip_tile.h"

// ZBF16 (abi/KvZip.h) prefill store and history expansion.
// Prefill attends through the BF16 split and reduce, over the BF16 scratch
// the expansion fills.

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
        thread_index >= SplashKvHeadDimension)                                 \
      return;                                                                  \
    splash_kvzip_store_chunk_slab<Heads>(chunk_keys, chunk_values, page_table, \
                                         codec, params, group, partial,        \
                                         thread_index, simd_lane, simd_group); \
  }
PAGED_PREFILL_ZIP_STORE(prefill_attention_zip_store, 4)
PAGED_PREFILL_ZIP_STORE(prefill_attention_zip_store_kv2_g8, 2)
#undef PAGED_PREFILL_ZIP_STORE

// Expand: one threadgroup per (tensor, KV head, history page) writes the
// page's committed rows into the BF16 scratch the prefill attends.
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
