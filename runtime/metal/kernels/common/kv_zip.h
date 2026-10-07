#pragma once

#include "metal/abi/KvZip.h"
#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/kv_paging.h"
#include <metal_stdlib>

using namespace metal;

// ZBF16 pages (abi/KvZip.h): the store kernels encode current rows into
// their slabs, the attention kernels decode a slab into a BF16 tile. Both
// run 256 threads per threadgroup.

// One page's bytes of keys (or values) and of their aux in one layer.
template <uint KVHeads> struct SplashKvZipBytes {
  static constexpr constant uint Data = KVHeads * SPLASH_KVZIP_DATA_BYTES_PER_HEAD;
  static constexpr constant uint Aux = KVHeads * SPLASH_KVZIP_AUX_BYTES_PER_HEAD;
};

// One KV head's slab of a page's keys or values in one layer.
struct SplashKvZipSlab {
  device uchar *data;     // sm[32][256], then codes[32][128]
  device uint *count;     // escapes, in the aux header
  device uint *escapes;   // [256]: position | exponent << 16
};

template <uint KVHeads> struct SplashKvZipAddressing {
  using Bytes = SplashKvZipBytes<KVHeads>;
  uint layer_offset;
  uint layer;
  ulong key_aux_offset;
  ulong values_offset;
  ulong value_aux_offset;

  explicit SplashKvZipAddressing(SplashKvLayer kv)
      : layer_offset(kv.offset),
        layer(kv.offset / uint(kv.extent_pages * 2u * (Bytes::Data + Bytes::Aux))),
        key_aux_offset(splash_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Aux, 0,
                                        SPLASH_KV_KEY_SCALES, 0)),
        values_offset(splash_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Aux, 0,
                                       SPLASH_KV_VALUES, 0)),
        value_aux_offset(splash_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Aux, 0,
                                          SPLASH_KV_VALUE_SCALES, 0)) {}

  SplashKvZipSlab slab(SplashKvPage entry, uint head, bool value_tensor) const {
    const uint index = splash_kv_page_index(entry);
    device uchar *region = splash_kv_extent(entry, index) + layer_offset;
    const ulong data = (value_tensor ? values_offset : 0) + ulong(index) * Bytes::Data +
                       head * SPLASH_KVZIP_DATA_BYTES_PER_HEAD;
    const ulong aux = (value_tensor ? value_aux_offset : key_aux_offset) +
                      ulong(index) * Bytes::Aux + head * SPLASH_KVZIP_AUX_BYTES_PER_HEAD;
    SplashKvZipSlab result;
    result.data = region + data;
    result.count = reinterpret_cast<device uint *>(region + aux);
    result.escapes = reinterpret_cast<device uint *>(region + aux + SPLASH_KVZIP_ESCAPE_HEADER_BYTES);
    return result;
  }

  // The head's window bases (base4), 256 bytes, 16-byte aligned.
  device const uchar *bases(device const uchar *codec, uint head, bool value_tensor) const {
    return codec + splash_kvzip_base_offset(KVHeads, layer, value_tensor ? 1u : 0u, head) +
           SPLASH_KVZIP_DIMENSIONS;
  }
};

// Encodes rows [row_begin, row_end) of one slab, in order, one thread per
// dimension. `source(row)` returns the row's element of this thread's
// dimension. Rows before row_begin are committed and keep their escapes;
// the escapes of rows from row_begin on are replaced. An escape past the
// table is dropped and the slab counted. Threadgroup scratch: 9 words.
template <typename Source>
inline void splash_kvzip_store_rows(SplashKvZipSlab slab, device const uchar *bases,
                                    device atomic_uint *overflow_slabs, uint row_begin,
                                    uint row_end, Source source, threadgroup uint *partial,
                                    uint dimension, uint simd_lane, uint simd_group) {
  const uint base = bases[dimension];
  // The escapes of the committed rows: those before the first entry of a
  // row at row_begin or past it, by a threadgroup minimum over the table.
  uint count = *slab.count;
  if (row_begin) {
    uint first = count;
    if (dimension < count && (slab.escapes[dimension] & 0xFFFFu) / SPLASH_KVZIP_DIMENSIONS >= row_begin)
      first = dimension;
    first = simd_min(first);
    if (simd_lane == 0)
      partial[simd_group] = first;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = 0; index < 8; ++index)
      first = min(first, partial[index]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    count = min(count, first);
  } else {
    count = 0;
  }
  bool dropped = false;
  for (uint row = row_begin; row < row_end; ++row) {
    const uint bits = as_type<ushort>(source(row));
    const uint exponent = (bits >> 7) & 0xFFu;
    const uint sm = ((bits >> 8) & 0x80u) | (bits & 0x7Fu);
    const int delta = int(exponent) - int(base);
    const bool escapes = delta < 0 || delta >= 16;
    const uint code = escapes ? 0u : uint(delta);
    slab.data[row * SPLASH_KVZIP_DIMENSIONS + dimension] = uchar(sm);
    // Two codes to a byte: the odd dimension's lane holds the pair.
    const uint pair = code | (simd_shuffle_xor(code, 1) << 4);
    if (dimension & 1u)
      slab.data[SPLASH_KVZIP_SM_BYTES + row * SPLASH_KVZIP_CODE_BYTES_PER_ROW + dimension / 2u] =
          uchar((pair >> 4) | ((pair & 0xFu) << 4));
    // This row's escapes, in dimension order, after the rows before it.
    const ulong votes = ulong(static_cast<simd_vote::vote_t>(simd_ballot(escapes)));
    const uint before_in_simd = popcount(uint(votes & ((1ul << simd_lane) - 1ul)));
    const uint in_simd = popcount(uint(votes));
    if (simd_lane == 0)
      partial[simd_group] = in_simd;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = count, total = 0;
    for (uint index = 0; index < 8; ++index) {
      before += index < simd_group ? partial[index] : 0u;
      total += partial[index];
    }
    if (escapes) {
      const uint at = before + before_in_simd;
      if (at < SPLASH_KVZIP_ESCAPES)
        slab.escapes[at] = (row * SPLASH_KVZIP_DIMENSIONS + dimension) | (exponent << 16);
      else
        dropped = true;
    }
    count += total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (dimension == 0)
    *slab.count = min(count, SPLASH_KVZIP_ESCAPES);
  const bool lost = simd_any(dropped);
  if (simd_lane == 0)
    partial[simd_group] = lost ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (dimension == 0) {
    uint any = 0;
    for (uint index = 0; index < 8; ++index)
      any |= partial[index];
    if (any)
      atomic_fetch_add_explicit(overflow_slabs, 1u, memory_order_relaxed);
  }
}

// One thread's 32 dimensions of a head's window bases, four to a word.
struct SplashKvZipLaneBases {
  uint words[8];
};

inline SplashKvZipLaneBases splash_kvzip_lane_bases(device const uchar *bases, uint thread_index) {
  device const uint4 *vectors =
      reinterpret_cast<device const uint4 *>(bases + (thread_index & 7u) * 32u);
  const uint4 low = vectors[0], high = vectors[1];
  return {{low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w}};
}

// Four elements' BF16 bits, as two words of two, from their sign/mantissa
// bytes and exponent bytes (four to a word each).
inline uint2 splash_kvzip_pack4(uint sm, uint exponent) {
  const uint high = (sm & 0x80808080u) | ((exponent >> 1) & 0x7F7F7F7Fu);
  const uint low = ((exponent & 0x01010101u) << 7) | (sm & 0x7F7F7F7Fu);
  return uint2((low & 0xFFu) | ((high & 0xFFu) << 8) | ((low & 0xFF00u) << 8) |
                   ((high & 0xFF00u) << 16),
               ((low >> 16) & 0xFFu) | (((high >> 16) & 0xFFu) << 8) |
                   ((low & 0xFF000000u) >> 8) | (high & 0xFF000000u));
}

// A thread's 32 decoded elements, contiguous in a token-major row.
inline void splash_kvzip_store_row(threadgroup ushort *bits, thread const uint *out) {
  threadgroup uint4 *vectors = reinterpret_cast<threadgroup uint4 *>(bits);
  for (uint vector = 0; vector < 4; ++vector)
    vectors[vector] = uint4(out[4 * vector], out[4 * vector + 1], out[4 * vector + 2], out[4 * vector + 3]);
}
inline void splash_kvzip_store_row(device ushort *bits, thread const uint *out) {
  device uint4 *vectors = reinterpret_cast<device uint4 *>(bits);
  for (uint vector = 0; vector < 4; ++vector)
    vectors[vector] = uint4(out[4 * vector], out[4 * vector + 1], out[4 * vector + 2], out[4 * vector + 3]);
}

// One thread's share of a slab, loaded ahead of its decode in one round
// trip: its row's 32 sign/mantissa bytes and 16 code bytes from dimension
// 32 (t % 8) of row t / 8, the escape count and the first four escapes.
struct SplashKvZipLoaded {
  uint4 sm_low, sm_high, codes, escapes;
  uint count;
  bool valid;
};

inline SplashKvZipLoaded splash_kvzip_load(SplashKvZipSlab slab, uint rows, uint thread_index) {
  constexpr uint D = SPLASH_KVZIP_DIMENSIONS;
  const uint row = thread_index >> 3;
  const uint first = (thread_index & 7u) * 32u;
  SplashKvZipLoaded loaded{};
  loaded.valid = row < rows;
  if (loaded.valid) {
    device const uint4 *sm = reinterpret_cast<device const uint4 *>(slab.data + row * D + first);
    loaded.sm_low = sm[0], loaded.sm_high = sm[1];
    loaded.codes = *reinterpret_cast<device const uint4 *>(
        slab.data + SPLASH_KVZIP_SM_BYTES + row * SPLASH_KVZIP_CODE_BYTES_PER_ROW + first / 2u);
  }
  loaded.count = min(*slab.count, SPLASH_KVZIP_ESCAPES);
  loaded.escapes = *reinterpret_cast<device const uint4 *>(slab.escapes);
  return loaded;
}

// Replaces the exponent of an escaped element among a thread's 32 decoded
// elements (two to a word, from dimension `first` of row `row`).
inline void splash_kvzip_patch(uint escape, uint row, uint first, thread uint *out) {
  constexpr uint D = SPLASH_KVZIP_DIMENSIONS;
  const uint position = escape & 0xFFFFu;
  const uint dimension = position % D;
  if (position / D != row || dimension < first || dimension >= first + 32u)
    return;
  const uint word = (dimension - first) >> 1, shift = 16u * (dimension & 1u);
  out[word] = (out[word] & ~(0x7F80u << shift)) | (((escape >> 16) & 0xFFu) << (7u + shift));
}

// Decodes the loaded slab into a BF16 slab, token-major ([token][dimension])
// in a threadgroup tile or a BF16 page's key slab, or dimension-major
// ([dimension][token]) in a BF16 page's value slab. Rows the load found past
// the visible ones become zeros. Escapes past the first four are read here,
// at uniform addresses. Ends with a threadgroup barrier.
template <typename Bits>
inline void splash_kvzip_decode_slab(thread const SplashKvZipLoaded &loaded, SplashKvZipSlab slab,
                                     thread const SplashKvZipLaneBases &bases,
                                     bool dimension_major, Bits bits, uint thread_index) {
  constexpr uint D = SPLASH_KVZIP_DIMENSIONS;
  constexpr uint N = SPLASH_KVZIP_ROWS;
  const uint row = thread_index >> 3;
  const uint first = (thread_index & 7u) * 32u;
  const uint sm_words[8] = {loaded.sm_low.x,  loaded.sm_low.y,  loaded.sm_low.z,  loaded.sm_low.w,
                            loaded.sm_high.x, loaded.sm_high.y, loaded.sm_high.z, loaded.sm_high.w};
  const uint code_words[4] = {loaded.codes.x, loaded.codes.y, loaded.codes.z, loaded.codes.w};
  uint out[16];
#pragma unroll
  for (uint word = 0; word < 8; ++word) {
    const uint nibbles = (code_words[word >> 1] >> (16u * (word & 1u))) & 0xFFFFu;
    const uint code4 = (nibbles & 0xFu) | ((nibbles & 0xF0u) << 4) | ((nibbles & 0xF00u) << 8) |
                       ((nibbles & 0xF000u) << 12);
    const uint2 packed = splash_kvzip_pack4(sm_words[word], bases.words[word] + code4);
    out[2 * word] = packed.x;
    out[2 * word + 1] = packed.y;
  }
  const uint head[4] = {loaded.escapes.x, loaded.escapes.y, loaded.escapes.z, loaded.escapes.w};
  for (uint index = 0; index < min(loaded.count, 4u); ++index)
    splash_kvzip_patch(head[index], row, first, out);
  for (uint index = 4; index < loaded.count; ++index)
    splash_kvzip_patch(slab.escapes[index], row, first, out);
  if (!loaded.valid) {
#pragma unroll
    for (uint word = 0; word < 16; ++word)
      out[word] = 0;
  }
  if (dimension_major) {
#pragma unroll
    for (uint word = 0; word < 16; ++word) {
      bits[(first + 2 * word) * N + row] = ushort(out[word]);
      bits[(first + 2 * word + 1) * N + row] = ushort(out[word] >> 16);
    }
  } else {
    splash_kvzip_store_row(bits + row * D + first, out);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Expands one slab of a chunk's committed history into the BF16 scratch:
// threadgroup `slab` of 2 x KVHeads x history pages, keys first.
template <uint KVHeads>
inline void splash_kvzip_expand_history(device const SplashKvPage *page_table,
                                        device const SplashKvPage *scratch_table,
                                        device const uchar *codec,
                                        constant SplashKvZipExpandParams &params, uint slab_index,
                                        uint thread_index) {
  const uint committed = params.chunk.committed_tokens;
  const uint pages = (committed + SplashKvPageTokens - 1) / SplashKvPageTokens;
  if (slab_index >= 2 * KVHeads * pages)
    return;
  const bool value_tensor = slab_index >= KVHeads * pages;
  const uint rest = value_tensor ? slab_index - KVHeads * pages : slab_index;
  const uint head = rest / pages;
  const uint page = rest % pages;
  const uint rows = min(uint(SplashKvPageTokens), committed - page * SplashKvPageTokens);
  const SplashKvZipAddressing<KVHeads> addressing(params.chunk.kv);
  const SplashKvPage entry = page_table[page];
  const SplashKvPageTensors<bfloat> target =
      SplashKvAddressing<KVHeads, bfloat>(params.scratch, head).page(scratch_table[page]);
  device ushort *bits =
      reinterpret_cast<device ushort *>(value_tensor ? target.values : target.keys);
  const SplashKvZipLaneBases bases =
      splash_kvzip_lane_bases(addressing.bases(codec, head, value_tensor), thread_index);
  const SplashKvZipSlab slab = addressing.slab(entry, head, value_tensor);
  splash_kvzip_decode_slab(splash_kvzip_load(slab, rows, thread_index), slab, bases, value_tensor,
                           bits, thread_index);
}
