#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/DraftAttention.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Rotary pairs in a row of the RoPE tables (ops::RoPE), which the attention
// kernels read: the target rotates 64 of its 256 head dimensions (Qwen3.5's
// partial rotary factor 1/4), the draft its whole head.
#define SPLASH_TARGET_ROPE_PAIRS 32u
#define SPLASH_DRAFT_ROPE_PAIRS (SPLASH_DRAFT_HEAD_DIMENSION / 2u)
#define SPLASH_DRAFT4_ROPE_PAIRS (SPLASH_DRAFT4_HEAD_DIMENSION / 2u)

struct RopeTableParams {
  uint32_t target_rows;
  uint32_t draft_rows;
  // What the target table's cosines and sines are scaled by: YaRN's
  // attention factor, 1 without YaRN (model::RopeScaling).
  float target_scale;
  // The draft rotates every head dimension.
  uint32_t draft_pairs = SPLASH_DRAFT_ROPE_PAIRS;
};

static_assert(sizeof(RopeTableParams) == 16,
              "RoPE table parameters are 16 bytes on both sides");

// Draft rows of consecutive positions from start_position, the rows a
// restore computes its rings from again (ops::RoPE::addDraftRangeTables).
struct RopeRangeParams {
  uint32_t rows;
  uint32_t start_position;
  uint32_t draft_pairs = SPLASH_DRAFT_ROPE_PAIRS;
};

static_assert(sizeof(RopeRangeParams) == 12,
              "RoPE range parameters are 12 bytes on both sides");
