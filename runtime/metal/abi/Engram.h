#pragma once

// Parameters of the Engram gate (kernels/shared/engram.metal, ops/Engram.cpp),
// DeepSeek-V4.1's: a row's table rows, projected into a key per stream and
// one value, gate the value into each stream of the manifold hyper-connected
// residual (metal/abi/ManifoldHyperConnection.h) by the sigmoid of the signed
// square root of the stream's normalized, weighted product with its key.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/ManifoldHyperConnection.h"

#define ENGRAM_THREADS 256u

struct EngramParams {
  uint32_t rows;
  uint32_t hidden;
  // The RMS norms' epsilon, and the floor of the product's magnitude under
  // its square root.
  float norm_epsilon;
  float magnitude_floor;
};

static_assert(sizeof(EngramParams) == 16, "Engram parameters are 16 bytes on both sides");

// The Engram hash of a verify step's rows (engram_hash; model/EngramHash.hpp
// on the host): each row's token and the ngram - 1 before it, its lane's
// rows before it and then its lane's prior tokens (the pad token's before
// the sequence's start), give each Engram layer's (ngram - 1) * heads rows,
// each plus its layer's first row in the store of their tables, layer by
// layer ([layers][rows][(ngram - 1) * heads]). A lane is `lane_rows` rows.
#define ENGRAM_MAX_LAYERS 4u
#define ENGRAM_MAX_NGRAM 8u
struct EngramHashParams {
  uint32_t rows;
  uint32_t lane_rows;
  uint32_t layers;
  uint32_t ngram;
  uint32_t heads;
  uint32_t pad;
  uint32_t first_rows[ENGRAM_MAX_LAYERS];
};

static_assert(sizeof(EngramHashParams) == 40, "Engram hash parameters are 40 bytes on both sides");
