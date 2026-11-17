#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/DraftAttention.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Rotary pairs in a row of the RoPE tables (ops::RoPE), which the attention
// kernels read: Qwen3.5's target rotates 64 of its 256 head dimensions (its
// partial rotary factor 1/4), the draft its whole head. A target attention
// geometry that rotates other pairs reads tables of its own pairs
// (ops::RoPE::addTables, ops::PagedAttention's compiled geometries).
#define SPLASH_TARGET_ROPE_PAIRS 32u
#define SPLASH_DRAFT_ROPE_PAIRS (SPLASH_DRAFT_HEAD_DIMENSION / 2u)

struct RopeTableParams {
  uint32_t target_rows;
  uint32_t draft_rows;
  // What the target table's cosines and sines are scaled by: YaRN's
  // attention factor, 1 without YaRN (model::RopeScaling).
  float target_scale;
};

static_assert(sizeof(RopeTableParams) == 12,
              "RoPE table parameters are 12 bytes on both sides");

// Draft rows of consecutive positions from start_position, the rows a
// restore computes its rings from again (ops::RoPE::addDraftRangeTables).
struct RopeRangeParams {
  uint32_t rows;
  uint32_t start_position;
};

static_assert(sizeof(RopeRangeParams) == 8,
              "RoPE range parameters are 8 bytes on both sides");

// The target rows of a decode step's `rows` packed rows over `lanes` lanes
// (ops::RoPE::addTreeTables), which its selector lays out on the device: lane
// l's rows (its SplashLaneRows) each at the RoPE position anchors[l], its
// anchor's, plus the row's depth in the lane's tree (its SplashTreeRow), the
// table scaled by target_scale as RopeTableParams' is.
struct RopeTreeParams {
  uint32_t lanes;
  uint32_t rows;
  float target_scale;
  uint32_t anchors[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(RopeTreeParams) == 12 + 4 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "RoPE tree parameters are the same size on both sides");
