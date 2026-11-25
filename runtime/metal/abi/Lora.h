#pragma once

// Parameter layouts of the kernels that add a projection's LoRA update
// (ops::ProjectionLora; kernels/shared/gguf_linear.metal).
#include "metal/abi/Gguf.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The corrections of a staged decode tile (gguf_decode_lora_*,
// gguf_decode_fused_lora_m*; kernels/shared/gguf_linear.metal), which adds
// B t to its sums as further steps of its accumulation: up to three, a
// tensor's own, a gate/up pair's gate and up, or a fused projection's per
// segment of its dispatch. t holds bf16 rows of t_stride values, each
// correction's `rank` values from its t_offset, and each B bf16 rows
// [column][rank].
struct GgufLora {
  uint32_t rank;         // inputs of a correction, a multiple of GGUF_STAGED_STEP
  uint32_t t_stride;     // values of a row of t
  uint32_t flags;        // bit i: correction i is present
  uint32_t t_offset[3];  // each correction's first value in a row of t
};
static_assert(sizeof(GgufLora) == 24, "LoRA tile parameters are 24 bytes on both sides");
struct GgufDecodeLoraParams {
  GgufDecodeParams base;
  GgufLora lora;
};
static_assert(sizeof(GgufDecodeLoraParams) == 44, "GGUF LoRA decode parameters are 44 bytes on both sides");
struct GgufDecodeFusedLoraParams {
  GgufDecodeFusedParams base;
  GgufLora lora;
};
static_assert(sizeof(GgufDecodeFusedLoraParams) == 76, "GGUF fused LoRA decode parameters are 76 bytes on both sides");

// A chunk's tile kernels (lora_tile, lora_swiglu_tile), which add B t to
// the sums a plan without the update wrote: rows of row_stride values, the
// update's segment from `column`. lora_tile takes one correction and its
// flags, lora_swiglu_tile the gate's (0) and the up's (1).
#define LORA_TILE_RESIDUAL 1u  // lora_tile adds the residual rows
#define LORA_TILE_DITHER 2u    // lora_tile rounds with a dither of `seed`
struct LoraTileParams {
  uint32_t rows;
  uint32_t row_stride;
  uint32_t column;
  uint32_t flags;
  uint32_t seed;
  struct GgufLora lora;
};
static_assert(sizeof(LoraTileParams) == 44, "LoRA chunk tile parameters are 44 bytes on both sides");
