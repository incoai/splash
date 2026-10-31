#pragma once

// Parameters of the compressor and indexer kernels (kernels/shared/latent_index.metal,
// ops/LatentIndex.cpp), DeepSeek-V4.1's: a KV source layer compresses every
// `ratio` rows of its latents into one, each value the rows' values weighted
// by the softmax of their gate scores; its indexer scores each compressed
// group a query row reaches by the sum over its heads of the head's weight
// times the ReLU of the head's product with the group's index key.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/KvExtent.h"

#define LATENT_INDEX_THREADS 256u
// The indexer's queries a threadgroup keeps: heads of LATENT_INDEX_DIM values.
#define LATENT_INDEX_MAX_HEADS 32u
#define LATENT_INDEX_DIM 128u
// The most rows a compressed group sums.
#define LATENT_COMPRESS_MAX_RATIO 8u

struct LatentCompressParams {
  uint32_t groups;
  uint32_t width;
  uint32_t ratio;
};

static_assert(sizeof(LatentCompressParams) == 12,
              "latent compression parameters are 12 bytes on both sides");

struct LatentIndexParams {
  uint32_t rows;
  uint32_t heads;
  // The compressed groups the keys hold, and the scores a row's line holds
  // room for.
  uint32_t groups;
  uint32_t stride;
  // A row at position p reaches the groups below (p + 1) / ratio.
  uint32_t ratio;
  // The factor of the heads' weights (index_dim^-0.5 index_heads^-0.5).
  float weight_scale;
  // A paged pass's keys' layer region in every extent: a sequence's pages
  // hold group g's key at the place of position (g + 1) * ratio - 1, its
  // first LATENT_INDEX_DIM values (latent_index_store).
  SplashKvLayer kv;
};

static_assert(sizeof(LatentIndexParams) == 32,
              "latent index parameters are 32 bytes on both sides");

// A KV source's compressed groups into a sequence's pages
// (latent_index_store; kv::Layout of one head of LATENT_STORE_WIDTH values,
// keys and values token-major): group first + i's latent as the key of the
// place of position (first + i + 1) * ratio - 1, its index key as the first
// LATENT_INDEX_DIM values of the place's value.
#define LATENT_STORE_WIDTH 512u
struct LatentStoreParams {
  uint32_t groups;
  uint32_t first;
  uint32_t ratio;
  SplashKvLayer kv;
};

static_assert(sizeof(LatentStoreParams) == 20,
              "latent store parameters are 20 bytes on both sides");

// A row's top `k` of `count` scores (latent_index_select): every score above
// the k-th highest, and of those equal to it the lowest, ascending, then
// LATENT_INDEX_NONE where fewer than k scores are above -inf. A threadgroup of
// LATENT_INDEX_SELECT_THREADS threads per row.
#define LATENT_INDEX_SELECT_THREADS 1024u
#define LATENT_INDEX_NONE 0xFFFFFFFFu

struct LatentSelectParams {
  uint32_t rows;
  uint32_t count;
  uint32_t k;
  // The scores a row's line holds room for, and the indices of its output.
  uint32_t stride;
  uint32_t out_stride;
};

static_assert(sizeof(LatentSelectParams) == 20,
              "latent selection parameters are 20 bytes on both sides");

// The candidate blocks of `block` groups (latent_index_blocks and
// latent_index_mask): a block's score is its groups' highest, the block of
// the last group a row reaches +inf; a later layer's scores keep the groups
// of the row's selected blocks alone.
struct LatentBlockParams {
  uint32_t rows;
  uint32_t groups;
  uint32_t stride;
  uint32_t block;
  uint32_t blocks;
  uint32_t ratio;
  // The block indices a row's list holds room for.
  uint32_t list_stride;
};

static_assert(sizeof(LatentBlockParams) == 28,
              "latent block parameters are 28 bytes on both sides");
