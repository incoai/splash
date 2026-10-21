#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The expert slots of the routing kernels (ops::MoE): a row's router scores
// take this many floats, the block's experts first, and the select and
// grouping kernels run one thread per slot. A MoE block of at most
// SPLASH_MOE_EXPERT_SLOTS experts routes on their instances of that many
// slots, one of more on their wide instances (_e512), so a block routes at
// most SPLASH_MOE_WIDE_EXPERT_SLOTS experts.
#define SPLASH_MOE_EXPERT_SLOTS 256u
#define SPLASH_MOE_WIDE_EXPERT_SLOTS 512u

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
// (t + 1) * tile_rows) of one expert; padding rows carry the route ~0u.
//
// Written by grouping kernels; the host uses this layout to size storage.
struct MoeTileDescriptor {
  uint32_t expert;
  uint32_t rows;
};

static_assert(sizeof(MoeTileDescriptor) == 8,
              "MoE tile descriptors are 8 bytes on both sides");

// The staged GGUF expert passes (kernels/shared/moe_gguf.metal) run an
// expert's only tile of at most this many live rows on the column tile, which
// reads those rows alone, and every other tile on the staged tile's matmuls
// (moe_column_tile): a route's output depends on its expert's rows in the
// step.
#define MOE_GGUF_COLUMN_ROWS 2u

// Whether the staged expert passes run tile t of `tiles`, the grouping's
// MoeTileDescriptors (on the GPU in device memory), on the column tile: at
// most MOE_GGUF_COLUMN_ROWS live rows, its expert's only tile. The kernels,
// the MoE test's expected tile kinds and the MoE benchmark's share of column
// tiles all ask this.
template <typename Tiles>
inline bool moe_column_tile(Tiles tiles, uint32_t t) {
  return tiles[t].rows <= MOE_GGUF_COLUMN_ROWS && (t == 0 || tiles[t - 1].expert != tiles[t].expert);
}

// The column tiles of the gate/up and down passes, whose grids the grouping
// writes (MoeTileCount). A verify step's grouping
// (moe_group_routes_verified*) routes only each lane's verify rows (the draft
// selector's, SelectorBatchParams): row r is row r % lane_rows of lane
// r / lane_rows, and the routed routes of a row past its lane's verify rows
// stay out of every tile with the route row ~0u, which the combine skips. The
// other groupings route every row and ignore lane_rows.
struct MoeGroupParams {
  uint32_t rows;
  uint32_t top_k;
  uint32_t tile_rows;
  uint32_t experts;
  uint32_t gate_up_columns;
  uint32_t down_columns;
  uint32_t lane_rows;
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

// A GGUF expert pass (ops/MoE.cpp): the routed experts of the projection are
// one segment of slots * output_size rows, routed expert e's planes starting
// at tile slots[e] * output_size / QUANT_TILE_ROWS of it, where slots is the
// block's slot table (ops::BlockMoeWeights); the shared expert (id `experts`)
// has a segment of its own. Formats are GGUF_FMT_* (metal/abi/QuantFormat.h).
struct MoeGgufExpertParams {
  uint32_t input_size;
  uint32_t output_size; // per expert
  uint32_t experts;
  uint32_t routed_format;
  uint32_t shared_format;
};

static_assert(sizeof(MoeGgufExpertParams) == 20,
              "MoE GGUF expert parameters are 20 bytes on both sides");

// The gate and up passes in one (moe_expert_gguf_m<rows>_gate_up[_mixed]): their
// sizes and expert count, as MoeGgufExpertParams holds them, and each pass's
// routed and shared formats.
struct MoeGgufGateUpParams {
  uint32_t input_size;
  uint32_t output_size; // per expert
  uint32_t experts;
  uint32_t gate_routed_format;
  uint32_t gate_shared_format;
  uint32_t up_routed_format;
  uint32_t up_shared_format;
};

static_assert(sizeof(MoeGgufGateUpParams) == 28,
              "MoE GGUF gate/up parameters are 28 bytes on both sides");

// One projection (gate, up or down) of the experts a load writes into their
// slots (moe_expert_load, model/ExpertStore.cpp): rows x columns in `format`,
// job j's native rows of row_bytes each staged from wherever the job's source
// names, and its plane0, plane1 and meta planes in the arena from planes[i]
// on, slot_bytes[i] bytes per slot.
struct MoeExpertLoadProjection {
  uint32_t format;
  uint32_t rows;
  uint32_t columns;
  uint32_t row_bytes;
  uint64_t planes[3];
  uint64_t slot_bytes[3];
};

static_assert(sizeof(MoeExpertLoadProjection) == 64,
              "MoE expert load projections are 64 bytes on both sides");

// The gate, up and down projections of one load.
struct MoeExpertLoadParams {
  struct MoeExpertLoadProjection projections[3];
};

static_assert(sizeof(MoeExpertLoadParams) == 192,
              "MoE expert load parameters are 192 bytes on both sides");

// The threads of a moe_expert_load threadgroup, each of which repacks one
// (row, 32-wide K group) pair.
#define MOE_EXPERT_LOAD_THREADS 256u

struct MoeCombineParams {
  uint32_t rows;
  uint32_t hidden_size;
  uint32_t routes_per_row;
};

static_assert(sizeof(MoeCombineParams) == 12,
              "MoE combine parameters are 12 bytes on both sides");
