#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

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

struct MoeGroupParams {
  uint32_t rows;
  uint32_t top_k;
  uint32_t tile_rows;
  uint32_t experts;
};

static_assert(sizeof(MoeGroupParams) == 16,
              "MoE grouping parameters are 16 bytes on both sides");

struct MoeGatherParams {
  uint32_t tile_rows;
  uint32_t input_size;
  uint32_t routes_per_row;
};

static_assert(sizeof(MoeGatherParams) == 12,
              "MoE gather parameters are 12 bytes on both sides");

struct MoeExpertParams {
  uint32_t input_size;
  uint32_t output_size;
  uint32_t experts;
  uint32_t reserved0;
  uint64_t expert_stride_bytes_0;
  uint64_t expert_stride_bytes_1;
};

static_assert(sizeof(MoeExpertParams) == 32,
              "MoE expert parameters are 32 bytes on both sides");

// A GGUF expert pass (ops/MoE.cpp): every routed expert of the projection
// is one image segment of experts * output_size rows, expert e's planes
// starting at tile e * output_size / QUANT_TILE_ROWS; the shared expert (id `experts`)
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

struct MoeCombineParams {
  uint32_t rows;
  uint32_t hidden_size;
  uint32_t routes_per_row;
};

static_assert(sizeof(MoeCombineParams) == 12,
              "MoE combine parameters are 12 bytes on both sides");
