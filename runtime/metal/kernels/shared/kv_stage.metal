#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/kv_extent.h"

// Stages one layer of a prefill's history (SplashKvStageParams,
// ops::PagedAttention::addStageLayer): threadgroup g copies page first_page +
// g of the pool's layer, each tensor's run of the page in turn, to the same
// tensor of the staged page, whose storage holds that one layer. It moves
// bytes, so one kernel serves either format and value order; every run is a
// whole number of 16-byte chunks, since a page's 32 tokens take 128 bytes of
// scales a head. A page whose pool entry is SPLASH_KV_PAGE_ABSENT is left to
// the reads that fill it.
kernel void paged_kv_stage_layer(device const SplashKvPage *pool_table [[buffer(0)]],
                                 device const SplashKvPage *staged_table [[buffer(1)]],
                                 constant SplashKvStageParams &p [[buffer(2)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint thread_index [[thread_index_in_threadgroup]],
                                 uint threads [[threads_per_threadgroup]]) {
  const uint page = p.first_page + group;
  const SplashKvPage pool = pool_table[page];
  if (pool == SPLASH_KV_PAGE_ABSENT)
    return;
  const SplashKvPage staged = staged_table[page];
  const uint pool_index = splash_kv_page_index(pool), staged_index = splash_kv_page_index(staged);
  device const uchar *from = splash_kv_extent(pool, pool_index) + p.pool.offset;
  device uchar *to = splash_kv_extent(staged, staged_index) + p.staged.offset;
  for (uint tensor = SPLASH_KV_KEYS; tensor <= SPLASH_KV_VALUE_SCALES; ++tensor) {
    const uint chunks = splash_kv_page_bytes(p.data_bytes, p.scale_bytes, tensor) / 16;
    device const uint4 *source = reinterpret_cast<device const uint4 *>(
        from + splash_kv_offset(p.pool.extent_pages, p.data_bytes, p.scale_bytes, 0, tensor, pool_index));
    device uint4 *target = reinterpret_cast<device uint4 *>(
        to + splash_kv_offset(p.staged.extent_pages, p.data_bytes, p.scale_bytes, 0, tensor, staged_index));
    for (uint chunk = thread_index; chunk < chunks; chunk += threads)
      target[chunk] = source[chunk];
  }
}
