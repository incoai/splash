#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The expert slots of the routing kernels (ops::MoE): a row's router scores
// take this many floats, the block's experts first, and the select and
// grouping kernels run one thread per slot, so a MoE block routes at most
// this many experts.
#define SPLASH_MOE_EXPERT_SLOTS 256u

struct MoeRouteParams {
  uint32_t rows;
  uint32_t input_size;
  uint32_t experts;
  uint32_t top_k;
};

static_assert(sizeof(MoeRouteParams) == 16,
              "MoE routing parameters are 16 bytes on both sides");

// Every row carries top_k routed experts followed by the shared expert, whose
// id is `experts` and whose routing weight is the sigmoid of its scalar gate.
// Routes are sorted by expert: tile t covers grouped rows [t * tile_rows,
// (t + 1) * tile_rows) of one expert; padding rows carry the route ~0u. In
// place (MoeGroupParams::in_place) an expert's one tile is instead the plan's
// rows in order, every one live.
//
// Written by grouping kernels; the host uses this layout to size storage.
struct MoeTileDescriptor {
  uint32_t expert;
  uint32_t rows;
};

static_assert(sizeof(MoeTileDescriptor) == 8,
              "MoE tile descriptors are 8 bytes on both sides");

// The column tiles of the gate/up and down passes, whose grids the grouping
// writes (MoeTileCount). With `in_place` (ops::MoePlan::rowsInPlace: a plan
// whose rows fill one 8-row tile on the staged tile), every expert's tile is
// the plan's rows in place, and the gate/up pass reads them without a gather.
struct MoeGroupParams {
  uint32_t rows;
  uint32_t top_k;
  uint32_t tile_rows;
  uint32_t experts;
  uint32_t gate_up_columns;
  uint32_t down_columns;
  uint32_t in_place;
};

static_assert(sizeof(MoeGroupParams) == 28,
              "MoE grouping parameters are 28 bytes on both sides");

// What the grouping writes besides the tiles: their count, which every pass
// over them reads first, and the grids (column tiles, tiles, 1) of the
// gate/up and down passes, which they read as an indirect dispatch
// (MTLDispatchThreadgroupsIndirectArguments) to launch the live tiles alone.
struct MoeTileCount {
  uint32_t tiles;
  uint32_t gate_up_grid[3];
  uint32_t down_grid[3];
};

static_assert(sizeof(MoeTileCount) == 28,
              "the MoE tile count is 28 bytes on both sides");

struct MoeGatherParams {
  uint32_t tile_rows;
  uint32_t input_size;
  uint32_t routes_per_row;
};

static_assert(sizeof(MoeGatherParams) == 12,
              "MoE gather parameters are 12 bytes on both sides");

// A GGUF expert pass (ops/MoE.cpp): every routed expert of the projection
// is one image segment of experts * output_size rows, expert e's planes
// starting at tile e * output_size / QUANT_TILE_ROWS; the shared expert (id `experts`)
// has a segment of its own. Formats are GGUF_FMT_* (metal/abi/QuantFormat.h).
// With `in_place` (MoeGroupParams::in_place), the staged expert kernels read
// every tile's rows from the plan's rows instead of its grouped rows; the
// register expert kernels, which no such plan runs, leave it unread.
struct MoeGgufExpertParams {
  uint32_t input_size;
  uint32_t output_size; // per expert
  uint32_t experts;
  uint32_t routed_format;
  uint32_t shared_format;
  uint32_t in_place;
};

static_assert(sizeof(MoeGgufExpertParams) == 24,
              "MoE GGUF expert parameters are 24 bytes on both sides");

struct MoeCombineParams {
  uint32_t rows;
  uint32_t hidden_size;
  uint32_t routes_per_row;
};

static_assert(sizeof(MoeCombineParams) == 12,
              "MoE combine parameters are 12 bytes on both sides");
