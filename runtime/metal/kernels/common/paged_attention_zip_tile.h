#pragma once

#include "metal/kernels/common/kv_zip.h"
#include "metal/kernels/common/paged_attention_tile.h"
#include "metal/kernels/common/paged_store_row.h"

// The paged attention tile over ZBF16 pages. It runs the BF16 tile's page
// loop (splash_paged_attention_tile) with the same MPP products, softmax and
// partials; each page's keys, then its values, are first decoded into one
// BF16 threadgroup tile, token-major, so a ZBF16 page attends exactly as the
// BF16 page it decodes to (PV reads the values untransposed, bit for bit as
// the BF16 loop's transposed read). The page's scores share the tile: they
// are written after the keys' last read and read before the values' first
// write. Threadgroup scratch beyond the BF16 loop's probabilities and
// statistics: the 16 KiB tile.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile>
inline void splash_paged_attention_zip_tile(
    device bfloat *tile_queries, device const SplashKvPage *page_table,
    SplashKvLayer kv, device const uchar *codec, uint kv_head, uint committed_tokens,
    uint active_rows, uint splits, uint split, device float *partials,
    device float *statistics, ulong slot, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum, threadgroup float *previous_scale,
    threadgroup atomic_uint *rescale, threadgroup bfloat *kv_tile, uint thread_index) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr ushort N = SplashKvPageTokens;
  constexpr ushort D = SplashKvHeadDimension;
  threadgroup float *scores = reinterpret_cast<threadgroup float *>(kv_tile);
  threadgroup ushort *bits = reinterpret_cast<threadgroup ushort *>(kv_tile);
  uint visible_tokens = committed_tokens + active_rows;
  uint pages = splash_attention_pages(visible_tokens);
  uint per_split = splash_attention_pages_per_split(pages, splits);
  uint page_begin = split * per_split;
  if (page_begin >= pages)
    return;
  uint page_end = min(pages, page_begin + per_split);
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto qt = tensor(tile_queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto kt = tensor(kv_tile, dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto vt = tensor(kv_tile, dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto p0 = pt.slice<N, M>(0, 0);
  auto q0 = qt.slice<D, M>(0, 0);
  auto k0 = kt.slice<D, N>(0, 0);
  auto v0 = vt.slice<D, N>(0, 0);
  const SplashKvZipAddressing<KVHeads> addressing(kv);
  const SplashKvZipLaneBases key_bases =
      splash_kvzip_lane_bases(addressing.bases(codec, kv_head, false), thread_index);
  const SplashKvZipLaneBases value_bases =
      splash_kvzip_lane_bases(addressing.bases(codec, kv_head, true), thread_index);
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, D, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, N, false, false, true,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(p0), decltype(v0), float>();
  const bool running_full =
      uint(running.get_capacity()) * (8u * 32u) == uint(M) * D;
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index) {
    if (running_full || running.is_valid_element(index))
      running[index] = 0.0f;
  }

  for (uint page = page_begin; page < page_end; ++page) {
    const SplashKvPage entry = page_table[page];
    uint token_start = page * N;
    // Rows past the visible tokens hold stale bytes: they decode to zeros,
    // which the softmax masks as the BF16 loop masks its stale rows.
    const uint rows = min(uint(N), visible_tokens - token_start);
    const SplashKvZipSlab keys = addressing.slab(entry, kv_head, false);
    const SplashKvZipSlab values = addressing.slab(entry, kv_head, true);
    splash_kvzip_decode_slab(splash_kvzip_load(keys, rows, thread_index), keys, key_bases, false,
                             bits, thread_index);
    auto page_scores = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(k0), float>();
    qk.run(q0, k0, page_scores);
    // Every simdgroup has read the keys before the scores overwrite them.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    page_scores.store(st.slice<N, M>(0, 0));
    if (thread_index == 0)
      atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    splash_attention_page_softmax<QueryHeadsPerKVHead, RowsPerTile, false>(
        scores, probabilities, row_max, row_sum, previous_scale, rescale, nullptr,
        nullptr, token_start, visible_tokens, committed_tokens, active_rows,
        thread_index);
    // The scores are read before the values overwrite them.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    splash_kvzip_decode_slab(splash_kvzip_load(values, rows, thread_index), values, value_bases,
                             false, bits, thread_index);
    if (atomic_load_explicit(rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (!running_full && !running.is_valid_element(index))
          continue;
        auto coordinates = running.get_multidimensional_index(index);
        running[index] *= previous_scale[coordinates[1]];
      }
    }
    pv.run(p0, v0, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  auto target = tensor(partials + slot * M * D, dextents<int, 2>{D, M},
                       array<int, 2>{1, D});
  running.store(target.slice<D, M>(0, 0));
  if (thread_index < M) {
    statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
    statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
  }
}

// A chunk row's element of one dimension, by the slab's row.
struct SplashKvZipChunkRows {
  device const bfloat *chunk_keys;
  device const bfloat *chunk_values;
  uint stride;
  uint head;
  uint chunk_offset;
  uint dimension;
  bool value_tensor;

  bfloat operator()(uint row) const {
    const uint chunk_token = chunk_offset + row;
    return value_tensor
               ? chunk_values[splash_current_value_index(stride, head, chunk_token, dimension)]
               : chunk_keys[splash_current_key_index(stride, head, chunk_token, dimension)];
  }
};

// Stores a chunk's (or a verify lane's) current rows: threadgroup `slab`
// of 2 x KVHeads x pages, keys first, encodes one KV head's rows of one page
// in order. `pages` is the number of pages the rows touch.
template <uint KVHeads>
inline void splash_kvzip_store_chunk_slab(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const SplashKvPage *page_table, device uchar *codec,
    constant SplashChunkedPrefillParams &params, uint slab_index,
    threadgroup uint *partial, uint dimension, uint simd_lane, uint simd_group) {
  const uint first = params.committed_tokens;
  const uint last = params.committed_tokens + params.chunk_tokens - 1;
  const uint first_page = first / SplashKvPageTokens;
  const uint pages = last / SplashKvPageTokens - first_page + 1;
  if (slab_index >= 2 * KVHeads * pages)
    return;
  const bool value_tensor = slab_index >= KVHeads * pages;
  const uint rest = value_tensor ? slab_index - KVHeads * pages : slab_index;
  const uint head = rest / pages;
  const uint page = first_page + rest % pages;
  const uint page_first = page * SplashKvPageTokens;
  const uint row_begin = max(first, page_first) - page_first;
  const uint row_end = min(last + 1, page_first + SplashKvPageTokens) - page_first;
  const SplashKvZipAddressing<KVHeads> addressing(params.kv);
  const SplashKvZipSlab slab = addressing.slab(page_table[page], head, value_tensor);
  const SplashKvZipChunkRows source{chunk_keys, chunk_values, params.chunk_stride, head,
                                    page_first - first, dimension, value_tensor};
  splash_kvzip_store_rows(slab, addressing.bases(codec, head, value_tensor),
                          reinterpret_cast<device atomic_uint *>(codec), row_begin, row_end,
                          source, partial, dimension, simd_lane, simd_group);
}
