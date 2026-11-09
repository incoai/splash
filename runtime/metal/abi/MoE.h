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
// slots, one of more on the softmax routing's wide instances (_e512)
// (ops::kCompiledMoeRoutings), so a block routes at most
// SPLASH_MOE_WIDE_EXPERT_SLOTS experts.
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

// Every row carries top_k routed experts followed by the slices of the shared
// expert, ids `experts` on, whose routing weight is the sigmoid of the
// block's scalar gate or 1 (ops::MoeShape). Routes are sorted by expert: tile
// t covers grouped rows [t * tile_rows, (t + 1) * tile_rows) of one expert;
// padding rows carry the route ~0u.
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
// other groupings route every row and ignore lane_rows. Only a streamed
// grouping, of a step whose passes run around its load, reads the block's
// slot table (MoeTileCount) and copies it for the passes beside the load.
struct MoeGroupParams {
  uint32_t rows;
  uint32_t top_k;
  uint32_t tile_rows;
  uint32_t experts;
  uint32_t gate_up_columns;
  uint32_t down_columns;
  uint32_t lane_rows;
  uint32_t streamed;
};

static_assert(sizeof(MoeGroupParams) == 32,
              "MoE grouping parameters are 32 bytes on both sides");

// What the grouping writes besides the tiles: their count, which every pass
// over them reads first, the tiles of experts resident before the step's
// load, and the grids (column tiles, tiles, 1) of the gate/up and down
// passes over all of them, over the resident ones and over the others
// (MOE_TILES_*), which they read as an indirect dispatch
// (MTLDispatchThreadgroupsIndirectArguments) to launch the live tiles alone.
// Tiles [0, resident) are the resident experts' and then the shared slices',
// and tiles [resident, tiles) those of the experts whose slot-table entry is
// MOE_SLOT_ABSENT, which a load brings.
struct MoeTileCount {
  uint32_t tiles;
  uint32_t resident;
  uint32_t gate_up_grid[3];
  uint32_t down_grid[3];
  uint32_t resident_gate_up_grid[3];
  uint32_t resident_down_grid[3];
  uint32_t loaded_gate_up_grid[3];
  uint32_t loaded_down_grid[3];
};

static_assert(sizeof(MoeTileCount) == 80,
              "the MoE tile count is 80 bytes on both sides");

// The tiles an expert pass runs (MoeTileCount): all of them, the resident
// experts' and shared slices', or the others; tile moe_first_tile + y runs at
// y of the pass's grid, below moe_end_tile.
#define MOE_TILES_ALL 0u
#define MOE_TILES_RESIDENT 1u
#define MOE_TILES_LOADED 2u
// A branch, not a select: the other ranges' tile loads do not wait for a
// load of the count.
template <typename Count>
inline uint32_t moe_first_tile(Count count, uint32_t tiles) {
  uint32_t first = 0;
  if (tiles == MOE_TILES_LOADED) first = count[1];
  return first;
}
template <typename Count>
inline uint32_t moe_end_tile(Count count, uint32_t tiles) {
  return count[tiles == MOE_TILES_RESIDENT ? 1 : 0];
}

// A slot-table entry (ops::BlockMoeWeights::slotTable) of an expert no slot
// holds, which only the grouping reads.
#define MOE_SLOT_ABSENT 0x80000000u

struct MoeGatherParams {
  uint32_t tile_rows;
  uint32_t input_size;
  uint32_t routes_per_row;
};

static_assert(sizeof(MoeGatherParams) == 12,
              "MoE gather parameters are 12 bytes on both sides");

// The routed experts of a MoE block sit in slots, which chunks hold: the
// floor chunk, chunk 0, holds the first floor_slots slots, a store size's
// floor, and each elastic chunk after it chunk_slots slots (the last may hold
// fewer). Slot s < floor_slots is local slot s of chunk 0; any other is local
// slot e - (c - 1) * chunk_slots of chunk c = 1 + e / chunk_slots, where
// e = s - floor_slots, and slots is the block's slot table
// (ops::BlockMoeWeights). A resident block, and a store size without an
// elastic chunk, is its floor chunk alone, of floor_slots = chunk_slots = its
// slots. The expert kernels reach the chunks through a table of their GPU
// addresses (metal::AddressTable), in the constant address space: chunk c's
// MOE_EXPERT_CHUNK_ADDRESSES entries from c * MOE_EXPERT_CHUNK_ADDRESSES on
// are the gate's (z = 0), up's (z = 1) and down's (z = 2) plane0, plane1 and
// meta regions, at z * 3 + 0, 1 and 2; a format without plane1 has its meta
// region's address in plane1's entry. A region holds its chunk's slots in
// order, each the projection's rows, so local slot l starts at row l * rows
// of it. A table holds at most MOE_EXPERT_MAX_CHUNKS chunks, which setBytes
// passes (4 KiB).
#define MOE_EXPERT_CHUNK_ADDRESSES 9u
#define MOE_EXPERT_MAX_CHUNKS 48u

// A GGUF expert pass of one projection (the down pass,
// moe_expert_gguf_m<rows>_down; kernels/common/moe_gguf_expert_tile.h):
// routed expert e's planes start at row local * output_size of the
// projection's regions in the chunk that holds slot slots[e] (local is the
// slot's place in it); the shared expert's slices (ids `experts` on) have a
// segment of their own, slice s's planes from tile
// s * output_size / QUANT_TILE_ROWS on. Formats are GGUF_FMT_*
// (metal/abi/QuantFormat.h).
struct MoeGgufExpertParams {
  uint32_t input_size;
  uint32_t output_size; // per expert
  uint32_t experts;
  uint32_t routed_format;
  uint32_t shared_format;
  uint32_t floor_slots;
  uint32_t chunk_slots;
  uint32_t tiles; // MOE_TILES_*
};

static_assert(sizeof(MoeGgufExpertParams) == 32,
              "MoE GGUF expert parameters are 32 bytes on both sides");

// The gate and up passes in one (moe_expert_gguf_m<rows>_gate_up[_mixed]):
// their sizes, expert count, floor and chunk slots, as MoeGgufExpertParams
// holds them, and each pass's routed and shared formats. Gate reads
// projection 0 and up projection 1 of the same chunk.
struct MoeGgufGateUpParams {
  uint32_t input_size;
  uint32_t output_size; // per expert
  uint32_t experts;
  uint32_t gate_routed_format;
  uint32_t gate_shared_format;
  uint32_t up_routed_format;
  uint32_t up_shared_format;
  uint32_t floor_slots;
  uint32_t chunk_slots;
  uint32_t tiles; // MOE_TILES_*
};

static_assert(sizeof(MoeGgufGateUpParams) == 40,
              "MoE GGUF gate/up parameters are 40 bytes on both sides");

// One projection (gate, up or down) of the experts a load writes into their
// slots (moe_expert_load, model/ExpertStore.cpp): rows x columns in `format`,
// job j's native rows of row_bytes each staged from wherever the job's source
// names, and its plane0, plane1 and meta planes slot_bytes[i] bytes per slot
// in the regions of the slot's chunk.
struct MoeExpertLoadProjection {
  uint32_t format;
  uint32_t rows;
  uint32_t columns;
  uint32_t row_bytes;
  uint64_t slot_bytes[3];
};

static_assert(sizeof(MoeExpertLoadProjection) == 40,
              "MoE expert load projections are 40 bytes on both sides");

// The gate, up and down projections of one load, and the slots of the floor
// chunk and of each elastic chunk.
struct MoeExpertLoadParams {
  struct MoeExpertLoadProjection projections[3];
  uint32_t floor_slots;
  uint32_t chunk_slots;
};

static_assert(sizeof(MoeExpertLoadParams) == 128,
              "MoE expert load parameters are 128 bytes on both sides");

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
