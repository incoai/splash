#pragma once

// INT8-activation prefill (kernels/shared/gguf_prefill_a8.metal): parameters and tile geometry shared by
// ops/LinearGguf.cpp and the kernels. A tile is WM x WN simdgroups, each 32 rows by 32 columns; the 4- and 5-bit
// kernels take the tiles oMLX's oQ A8 kernel tuned for them on M5 (its variants 806 and 800).
#include "metal/abi/QuantFormat.h"

#define PREFILL_A8_Q4_WM 1u
#define PREFILL_A8_Q5_WM 2u
#define PREFILL_A8_WN 2u
#define PREFILL_A8_COLUMNS (32u * PREFILL_A8_WN)
#define PREFILL_A8_ROWS(wm) (32u * (wm))
#define PREFILL_A8_THREADS(wm) (32u * (wm) * PREFILL_A8_WN)
// prefill_a8_quantize: one threadgroup of this many threads per row.
#define PREFILL_A8_QUANTIZE_THREADS 256u

// prefill_a8_quantize: grid (rows).
struct PrefillA8QuantizeParams {
  uint32_t input_size;    // K
  uint32_t rows;          // rows of the chunk
  uint32_t storage_rows;  // the row stride of the group sums
  uint32_t pad;
};
static_assert(sizeof(PrefillA8QuantizeParams) == 16, "prefill A8 quantize parameters are 16 bytes on both sides");

// gguf_prefill_a8_<format>_<a|r|g>: grid (column tiles of one segment, row tiles).
struct PrefillA8Params {
  uint32_t input_size;    // K
  uint32_t rows;
  uint32_t storage_rows;
  uint32_t out_stride;
  uint32_t out_offset;    // first destination column of the segment
  uint32_t pad[3];
};
static_assert(sizeof(PrefillA8Params) == 32, "prefill A8 parameters are 32 bytes on both sides");

// Scratch of a chunk of `storage` rows of K inputs: the int8 rows in the table; in the sums buffer the fp32 row
// scales, then the int16 sums of each row's quantized 64-input groups, [K / 64][storage].
#define PREFILL_A8_SUMS_BYTES(k, storage) (uint64_t(storage) * 4u + uint64_t((k) / 64u) * uint64_t(storage) * 2u)
