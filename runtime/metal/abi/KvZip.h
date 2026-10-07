#pragma once

// ZBF16: lossless BF16 KV pages. Every stored bit decodes back to the BF16
// value the projection wrote; a page is smaller than BF16 because the
// exponents of one (layer, KV head, dimension) sit in a narrow window, which
// a 4-bit code covers, as SplitZip's codebooks do (Guo and Joshi, 2026).
//
// A page keeps the BF16 extent geometry (abi/KvExtent.h): the data region of
// keys and of values holds each KV head's sign/mantissa bytes and codes, and
// the "scale" region holds each head's escapes. Keys and values are both
// stored token-major here; the attention kernels decode a page into the
// BF16 tile layout they consume.
//
// Per (page, layer, tensor, KV head) slab of 32 tokens x 256 dimensions:
//   data: sm[32][256]      sign << 7 | 7 mantissa bits, one byte per element
//         codes[32][128]   one 4-bit code per element, element d of a row in
//                          byte d / 2, the even element in the low nibble:
//                          exponent = base[dimension] + code, for the
//                          elements whose exponent lies in the window
//   aux:  count            uint32: the slab's escapes, then 12 bytes unused
//         escapes[252]     the elements whose exponent lies outside the
//                          window, in row order: uint16 position
//                          (row * 256 + dimension), uint8 exponent, one
//                          byte unused; their code is 0
// base is per (layer, tensor, KV head, dimension), the 16-binade window
// calibrated per model family (base4 in the codec buffer; base3 is kept for
// the calibration format but unused).
//
// Real KV needs under one escape per slab on average and at most a few
// dozen; prompts of one repeated token need up to ~200. A slab with more
// escapes than the table holds stores the rest with code 0 and no escape,
// so they decode to the window's lowest exponent (never infinite or NaN),
// and the store counts the slab in the codec buffer's counter, which the
// runtime reports and logs.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/PagedAttention.h"

#define SPLASH_KVZIP_ROWS 32u
#define SPLASH_KVZIP_DIMENSIONS 256u
#define SPLASH_KVZIP_CODE_BYTES_PER_ROW (SPLASH_KVZIP_DIMENSIONS / 2u)

#define SPLASH_KVZIP_SM_BYTES (SPLASH_KVZIP_ROWS * SPLASH_KVZIP_DIMENSIONS)
#define SPLASH_KVZIP_CODE_BYTES (SPLASH_KVZIP_ROWS * SPLASH_KVZIP_CODE_BYTES_PER_ROW)
#define SPLASH_KVZIP_DATA_BYTES_PER_HEAD (SPLASH_KVZIP_SM_BYTES + SPLASH_KVZIP_CODE_BYTES)

#define SPLASH_KVZIP_ESCAPE_HEADER_BYTES 16u
#define SPLASH_KVZIP_ESCAPES 252u
#define SPLASH_KVZIP_ESCAPE_BYTES 4u
#define SPLASH_KVZIP_AUX_BYTES_PER_HEAD \
  (SPLASH_KVZIP_ESCAPE_HEADER_BYTES + SPLASH_KVZIP_ESCAPES * SPLASH_KVZIP_ESCAPE_BYTES)

// The codec buffer: a header, then per attention layer and tensor (keys,
// values) and KV head, base3[256] and base4[256]; ZBF16 uses base4.
struct SplashKvZipHeader {
  // Slabs that stored escapes past the table, counted by the store kernels.
  uint32_t overflow_slabs;
  uint32_t layers;
  uint32_t kv_heads;
  uint32_t reserved[13];
};

#define SPLASH_KVZIP_HEADER_BYTES 64u
#define SPLASH_KVZIP_BASE_BYTES_PER_HEAD (2u * SPLASH_KVZIP_DIMENSIONS)

#ifndef __METAL_VERSION__
static_assert(sizeof(SplashKvZipHeader) == SPLASH_KVZIP_HEADER_BYTES,
              "the codec header is 64 bytes on both sides");
#endif

// Where one head's bases begin in the codec buffer, in bytes: base3, then
// base4 at SPLASH_KVZIP_DIMENSIONS.
inline uint32_t splash_kvzip_base_offset(uint32_t kv_heads, uint32_t layer,
                                         uint32_t value_tensor, uint32_t head) {
  return SPLASH_KVZIP_HEADER_BYTES +
         ((layer * 2u + value_tensor) * kv_heads + head) * SPLASH_KVZIP_BASE_BYTES_PER_HEAD;
}

inline uint32_t splash_kvzip_codec_bytes(uint32_t layers, uint32_t kv_heads) {
  return SPLASH_KVZIP_HEADER_BYTES + layers * 2u * kv_heads * SPLASH_KVZIP_BASE_BYTES_PER_HEAD;
}

// Prefill attends ZBF16 history through the BF16 kernels: before a chunk's
// attention, its committed history is expanded into a BF16 scratch of one
// layer, whose pages the scratch table names in logical order. chunk is the
// store's parameters in the ZBF16 pool; scratch places the layer's region in
// the scratch's single extent.
struct SplashKvZipExpandParams {
  SplashChunkedPrefillParams chunk;
  SplashKvLayer scratch;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(SplashKvZipExpandParams) == 32,
              "ZBF16 expand parameters are 32 bytes on both sides");
#endif
