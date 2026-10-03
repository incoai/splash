#pragma once

// Quantized weight formats of the MDGG0001 image, shared by the host planner
// and reader, the load-time repack, the GEMM kernels and the tests. A [N, K]
// tensor with G = K / 32 groups per row is stored in tiles of T =
// QUANT_TILE_ROWS rows as
//   plane0 [N / T][G][T][plane0_bytes]
//   plane1 [N / T][G][T][plane1_bytes]   (none when plane1_bytes is 0)
//   meta   [N / T][G / meta_groups][T][meta_bytes]
// A meta unit is one native block and holds its scale fields.
//
// Its contents are hashed into SPLASH_GGUF_PREPARATION_ID
// (dev/tools/weight_preparation_identity.py): it holds what defines prepared
// bytes, plus each format's kernel name token, which the host reads.
// Editing this file re-prepares every GGUF model.
// dev/tests/test_gguf_metadata.py reads the GGUF type of each kQuantFormats
// row with a regex on the row's leading number. The decode-only value tables
// are in metal/abi/QuantTables.h.
//
// Inside a group of 32 the elements are in lane-owned chunk order: chunk c
// (0..3) holds elements 4c..4c+3 and 16+4c..16+4c+3 as pairs p = 0..3, pair p
// being elements e0 and e0 + 1 with e0 = 16 (p >> 1) + 4c + 2 (p & 1).
// Element e is at slot quant_slot(e) = 8c + 2p + (e & 1). The planes pack the
// slots as follows.
//   4-bit linear codes (Q4_K, Q4_0, Q4_1, the low bits of Q5_K and Q6_K):
//     word c holds slots 8c..8c+7, pair p's e0 at bits 4p and e1 at bits
//     16 + 4p.
//   Every other field is a little-endian bit string of the 32 slots: IQ4 and
//     MXFP4 indices (4 bits, so byte p of word c is pair p), Q8_0 values (8),
//     Q6_K high, Q3_K low, Q2_K and PQ2_0 bits (2), Q5_K fifth and Q3_K hmask
//     bits (1).
//   IQ3_S word c: bits 0..7 and 8..15 the low grid index bits of elements
//     4c..4c+3 and 16+4c..16+4c+3, 16..23 the sign bits of slots 8c..8c+7,
//     24 and 25 the two ninth index bits, 26..29 the group's scale.
//   IQ3_XXS, IQ2 and IQ1 keep the group's native bytes, whose grid entries
//     span four or eight elements and so several chunks: its block_*'s fields
//     for the group's 32 elements, split between the planes as its row below
//     states. kernels/common/quant_formats.h decodes a chunk from them.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define QUANT_CONSTANT constant constexpr
#else
#include <stdint.h>
#define QUANT_CONSTANT inline constexpr
#endif

// Format ids: kQuantFormats index, GgufRepackParams.fmt and the run-time format
// of the fused and MoE expert kernels.
#define GGUF_FMT_Q4K 0u
#define GGUF_FMT_IQ4XS 1u
#define GGUF_FMT_IQ4NL 2u
#define GGUF_FMT_Q5K 3u
#define GGUF_FMT_Q6K 4u
#define GGUF_FMT_Q3K 5u
#define GGUF_FMT_Q80 6u
#define GGUF_FMT_IQ3S 7u
#define GGUF_FMT_Q2K 8u
#define GGUF_FMT_IQ3XXS 9u
#define GGUF_FMT_IQ2XXS 10u
#define GGUF_FMT_IQ2XS 11u
#define GGUF_FMT_IQ2S 12u
#define GGUF_FMT_IQ1S 13u
#define GGUF_FMT_IQ1M 14u
#define GGUF_FMT_Q40 15u
#define GGUF_FMT_Q41 16u
#define GGUF_FMT_MXFP4 17u
#define GGUF_FMT_PQ20 18u
#define GGUF_FMT_COUNT 19u

struct QuantFormat {
  uint32_t ggml_type;      // GGUF tensor type
  uint32_t block_elements; // elements per native block
  uint32_t block_bytes;    // bytes per native block
  uint32_t plane0_bytes;   // per row and group of 32
  uint32_t plane1_bytes;   // per row and group of 32
  uint32_t meta_bytes;     // per row and meta unit
  uint32_t meta_groups;    // groups of 32 per meta unit
  char name[8];            // kernel name suffix
};

QUANT_CONSTANT QuantFormat kQuantFormats[GGUF_FMT_COUNT] = {
    {12, 256, 144, 16, 0, 16, 8, "q4k"},  // meta: d, dmin, 12 scale bytes
    {23, 256, 136, 16, 0, 8, 8, "iq4xs"}, // meta: d, scales_h, scales_l
    {20, 32, 18, 16, 0, 2, 1, "iq4nl"},   // meta: d
    {13, 256, 176, 16, 4, 16, 8, "q5k"},  // plane1: fifth bits; meta as Q4_K
    {14, 256, 210, 16, 8, 20, 8, "q6k"},  // plane1: high 2 bits; meta: 16 int8 scales, d, 0, 0
    {11, 256, 110, 8, 4, 16, 8, "q3k"},   // plane1: hmask bits; meta: d, 0, 0, 12 scale bytes
    {8, 32, 34, 32, 0, 2, 1, "q80"},      // meta: d
    {21, 256, 110, 16, 0, 2, 8, "iq3s"},  // plane0 also holds signs, qh and the scale; meta: d
    {10, 256, 84, 8, 0, 20, 8, "q2k"},    // meta: d, dmin, the 16 scale bytes
    {18, 256, 98, 8, 4, 2, 8, "iq3xxs"},  // plane0: qs; plane1: the sign and scale word; meta: d
    {16, 256, 66, 8, 0, 2, 8, "iq2xxs"},  // plane0: the grid index and the sign and scale words; meta: d
    {17, 256, 74, 8, 1, 2, 8, "iq2xs"},   // plane0: the four qs; plane1: the scales byte; meta: d
    {22, 256, 82, 8, 2, 2, 8, "iq2s"},    // plane0: qs, then signs; plane1: qh, then the scales byte; meta: d
    {19, 256, 50, 4, 2, 2, 8, "iq1s"},    // plane0: qs; plane1: qh; meta: d
    {29, 256, 56, 4, 2, 8, 8, "iq1m"},    // plane0: qs; plane1: qh; meta: the 8 scale bytes, which hold d
    {2, 32, 18, 16, 0, 2, 1, "q40"},      // meta: d
    {3, 32, 20, 16, 0, 4, 1, "q41"},      // meta: d, m
    {39, 32, 17, 16, 0, 1, 1, "mxfp4"},   // meta: e
    {142, 128, 34, 8, 0, 2, 4, "pq20"},   // Prism's block_pq2_0, one d per 128 elements; meta: d
};

// The format that stores a GGUF tensor type; GGUF_FMT_COUNT when none does.
inline constexpr uint32_t gguf_format_of(uint32_t ggml_type) {
  for (uint32_t format = 0; format < GGUF_FMT_COUNT; ++format)
    if (kQuantFormats[format].ggml_type == ggml_type) return format;
  return GGUF_FMT_COUNT;
}

// The chunk order slot of element e (0..31) of a group.
inline constexpr uint32_t quant_slot(uint32_t e) { return 8 * ((e >> 2) & 3) + 4 * (e >> 4) + (e & 3); }

#define QUANT_TILE_ROWS 256u

// Index of (row, block) in a [rows / T][blocks][T] plane (blocks = G) or meta
// plane (blocks = meta units per row).
inline constexpr uint64_t quant_tile_index(uint32_t row, uint32_t block, uint32_t blocks) {
  return (uint64_t(row / QUANT_TILE_ROWS) * blocks + block) * QUANT_TILE_ROWS + row % QUANT_TILE_ROWS;
}

#undef QUANT_CONSTANT
