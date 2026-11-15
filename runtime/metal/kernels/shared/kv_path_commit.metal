#include "metal/kernels/common/attention_sequence.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/lane_bindings.h"

// A decode step's q/k projection writes every row it holds into its page
// slot (splash_store_kv_slot, attention_qkv_prepare.h), a lane's at the
// slots of its rows of its block; its commit here moves the accepted path's
// rows to the slots of their depths, and the rows past them stay unreachable
// until the next command starting at the same logical position overwrites
// them.

// Moves a lane's accepted path's K/V of one KV head of one layer region
// (SplashKvPathCommitParams): its row at depth k from the slot of its row to
// that of its depth, each thread one dimension of every row it moves, a
// threadgroup of HeadDim threads per (region and KV head, lane), and thread 0
// an INT8 page's scales. Rows move in order of depth: a row's target is a
// slot only an earlier row read (rows 0, 2, 3 move 2 to 1, then 3 to 2), as
// every later row sits past it. A row at its depth stays: a chain's path
// moves nothing.
template <uint KVHeads, uint HeadDim, typename CacheElement, bool ValuesTokenMajor>
inline void splash_path_commit_phase(device const SplashKvPage *page_table, uint path, uint committed,
                                     SplashKvLayer kv, uint head, uint dimension) {
  const SplashKvAddressing<KVHeads, HeadDim, CacheElement> addressing(kv, head);
  for (uint depth = 0, rows = path; rows; ++depth, rows &= rows - 1) {
    const uint row = ctz(rows);
    if (row == depth)
      continue;
    const uint from = committed + row, to = committed + depth;
    const uint source_place = from % SplashKvPageTokens, target_place = to % SplashKvPageTokens;
    const SplashKvPageTensors<CacheElement> source = addressing.page(page_table[from / SplashKvPageTokens]);
    const SplashKvPageTensors<CacheElement> target = addressing.page(page_table[to / SplashKvPageTokens]);
    target.keys[splash_kv_key_index<KVHeads, HeadDim>(0, 0, target_place, dimension)] =
        source.keys[splash_kv_key_index<KVHeads, HeadDim>(0, 0, source_place, dimension)];
    target.values[splash_kv_value_index<KVHeads, HeadDim, ValuesTokenMajor>(0, 0, target_place, dimension)] =
        source.values[splash_kv_value_index<KVHeads, HeadDim, ValuesTokenMajor>(0, 0, source_place, dimension)];
    if constexpr (is_same<CacheElement, int8_t>::value) {
      if (dimension == 0) {
        target.key_scales[splash_q8_scale_index<KVHeads>(0, 0, target_place)] =
            source.key_scales[splash_q8_scale_index<KVHeads>(0, 0, source_place)];
        target.value_scales[splash_q8_scale_index<KVHeads>(0, 0, target_place)] =
            source.value_scales[splash_q8_scale_index<KVHeads>(0, 0, source_place)];
      }
    }
  }
}

// Grid {regions x KV heads, lanes}; paths: each lane's accepted path, as
// acceptance wrote it.
#define PAGED_ATTENTION_PATH_COMMIT(Name, Heads, HeadDim, CacheElement, ValuesTokenMajor)                         \
  kernel void Name(device const SplashKvPage *page_table0 [[buffer(0)]],                                         \
                   device const SplashKvPage *page_table1 [[buffer(1)]],                                         \
                   device const SplashKvPage *page_table2 [[buffer(2)]],                                         \
                   device const SplashKvPage *page_table3 [[buffer(3)]],                                         \
                   device const uint *paths [[buffer(4)]],                                                       \
                   constant SplashKvPathCommitParams &params [[buffer(5)]],                                      \
                   uint2 group [[threadgroup_position_in_grid]],                                                 \
                   uint thread_index [[thread_index_in_threadgroup]]) {                                          \
    using Bytes = SplashKvPageBytes<Heads, HeadDim, CacheElement>;                                               \
    const uint lane = group.y;                                                                                   \
    splash_path_commit_phase<Heads, HeadDim, CacheElement, ValuesTokenMajor>(                                    \
        SPLASH_LANE_BINDING(lane, page_table0, page_table1, page_table2, page_table3), paths[lane],              \
        params.committed_tokens[lane],                                                                           \
        splash_kv_layer(params.extent_pages, Bytes::Data, Bytes::Scale, group.x / Heads), group.x % Heads,       \
        thread_index);                                                                                           \
  }
PAGED_ATTENTION_PATH_COMMIT(attention_q8_commit, 4, 256, int8_t, false)
PAGED_ATTENTION_PATH_COMMIT(attention_q8_commit_kv2_g8, 2, 256, int8_t, false)
PAGED_ATTENTION_PATH_COMMIT(attention_bf16_commit, 4, 256, bfloat, false)
PAGED_ATTENTION_PATH_COMMIT(attention_bf16_commit_kv2_g8, 2, 256, bfloat, false)
PAGED_ATTENTION_PATH_COMMIT(attention_q8_commit_kv4_d128, 4, 128, int8_t, false)
PAGED_ATTENTION_PATH_COMMIT(attention_bf16_commit_kv4_d128, 4, 128, bfloat, false)
PAGED_ATTENTION_PATH_COMMIT(attention_q8_commit_kv2_vt, 2, 256, int8_t, true)
PAGED_ATTENTION_PATH_COMMIT(attention_bf16_commit_kv2_vt, 2, 256, bfloat, true)
#undef PAGED_ATTENTION_PATH_COMMIT
