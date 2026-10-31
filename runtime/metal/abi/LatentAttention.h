#pragma once

// Parameters of the latent attention kernels (kernels/shared/latent_attention.metal,
// ops/LatentAttention.cpp), DeepSeek-V4.1's: every query head attends to the
// same keys, latents of LATENT_ATTENTION_DIM values that are each a key and
// its value, plus a sink logit of its own. A row's keys are a list of
// indices of bf16 latents: the window's, an index with
// LATENT_ATTENTION_WINDOW_KEY set, of the step's own latents or of a ring of
// the last ones before the step; and the compressed ones, any other, of a
// pool or, in a paged pass, of a sequence's pages (kv::Layout of one head of
// LATENT_ATTENTION_DIM, keys token-major), there the position of their place.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/KvExtent.h"

// The latent width the kernels are compiled for (head_dim).
#define LATENT_ATTENTION_DIM 512u
// The bit of a key index that names the window's pool.
#define LATENT_ATTENTION_WINDOW_KEY 0x80000000u
#define LATENT_ATTENTION_THREADS 256u
// The keys a scores threadgroup takes, and the heads a values threadgroup
// sums, each of its threads four values of each head.
#define LATENT_ATTENTION_KEY_BLOCK 32u
#define LATENT_ATTENTION_HEAD_BLOCK 8u
#define LATENT_ATTENTION_VALUE_THREADS (LATENT_ATTENTION_DIM / 4u)
// The matrix passes of a step of more rows than LATENT_ATTENTION_MAX_SPLITS,
// whose values pass splits no keys: a scores threadgroup takes
// LATENT_ATTENTION_MMA_KEYS keys of a row for every head, a values
// threadgroup LATENT_ATTENTION_MMA_VALUES of a row's latent values for every
// head, each in LATENT_ATTENTION_MMA_THREADS threads by 8x8 simdgroup
// matrices.
#define LATENT_ATTENTION_MMA_THREADS 128u
#define LATENT_ATTENTION_MMA_KEYS 64u
#define LATENT_ATTENTION_MMA_VALUES 64u
// The most parts a values pass splits a row's keys into, so that a decode's
// few rows fill the GPU: each part's sums apart, then added in part order
// (latent_attention_sum).
#define LATENT_ATTENTION_MAX_SPLITS 16u
// The most heads a row has: a scores lane holds two of them.
#define LATENT_ATTENTION_MAX_HEADS 64u

struct LatentAttentionParams {
  uint32_t rows;
  uint32_t heads;
  // The indices a row's list holds room for, and the scores it may write.
  uint32_t key_stride;
  // The scores' scale, head_dim^-0.5.
  float scale;
  // The parts the values pass splits each row's keys into.
  uint32_t splits;
  // The window keys below it name a ring of the window's last latents before
  // the step, those at or past it the step's own latents from slot `ring`
  // on; with 0 every window key names a step latent.
  uint32_t ring;
  // A paged pass's compressed keys' layer region in every extent.
  SplashKvLayer kv;
};

static_assert(sizeof(LatentAttentionParams) == 32,
              "latent attention parameters are 32 bytes on both sides");

// A step's key lists (latent_attention_keys): each row's window, the
// `window` positions that end at its own, then its selected groups, the
// first entries of its line of the indexer's selection (`selected` per row,
// ascending, LATENT_INDEX_NONE past its last; none with `selected` 0). A
// window position p is slot p % ring of one pool; or, `split`, of the ring
// before the step below `committed` (the sequence's tokens before it) and
// step latent p - committed past it, slot ring + p - committed. A selected
// group g is its index; or, with `ratio`, the position (g + 1) * ratio - 1
// of its place in a sequence's pages.
struct LatentAttentionKeysParams {
  uint32_t rows;
  uint32_t window;
  uint32_t ring;
  uint32_t selected;
  uint32_t key_stride;
  uint32_t committed;
  uint32_t split;
  uint32_t ratio;
  // With `limited`, a row from limits[lane] on (a verify lane's rows past
  // the ones it verifies) has no keys.
  uint32_t limited;
  uint32_t lane;
};

static_assert(sizeof(LatentAttentionKeysParams) == 40,
              "latent attention key list parameters are 40 bytes on both sides");

// A sequence's ring of its window's last latents after a step
// (latent_ring_update): each slot s of `ring` holds the latent of the last
// position before `end` with p % ring == s, the step's latent p - committed
// where p is at or past `committed`, else the ring's before the step. `end`
// is committed + rows, or, `retained`, committed plus the step's retained
// rows of lane `lane` (a buffer of them, at most `rows`).
struct LatentRingParams {
  uint32_t committed;
  uint32_t rows;
  uint32_t ring;
  uint32_t retained;
  uint32_t lane;
};

static_assert(sizeof(LatentRingParams) == 20, "latent ring parameters are 20 bytes on both sides");

// The open group of a paired KV source after a verify's retained rows
// (latent_open_commit): with committed + retained odd, the fp32 kv and gate
// projections of position committed + retained - 1, row `retained` of the
// step's, whose row 0 is the position before the step's first; left as they
// were otherwise.
struct LatentOpenParams {
  uint32_t committed;
  uint32_t width;
  uint32_t lane;
};

static_assert(sizeof(LatentOpenParams) == 12, "latent open group parameters are 12 bytes on both sides");
