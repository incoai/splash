#pragma once

// Latent attention (kernels/shared/latent_attention.metal), DeepSeek-V4.1's:
// every query head of a row attends to the row's keys, latents of
// LATENT_ATTENTION_DIM values each its own value, which all heads share,
// beside a sink logit per head. A row's keys are indices of bf16 latents
// (metal/abi/LatentAttention.h): its window's, the layer's own latents of the
// last positions (the step's, and a ring of those before it), and the
// compressed latents its indexer selected (a pool's, or a sequence's pages').

#include "metal/CommandGraph.hpp"
#include "metal/abi/LatentAttention.h"

#include <cstdint>

namespace splash::ops {

struct LatentAttention final {
  // The window latents a step's keys name: the step's own (`step`), and,
  // with ringSlots, a ring of the last ones before the step (`ring`).
  struct WindowLatents final {
    metal::MetalBuffer step;
    metal::MetalBuffer ring;
    uint32_t ringSlots = 0;
  };
  // The compressed latents a step's keys name: a pool of them (`pool`, a
  // key its index-th), or a sequence's pages (kv::Layout of one head of
  // LATENT_ATTENTION_DIM, keys token-major) through its page table, a key the
  // latent at the place of the position it names, in the region `kv`.
  struct CompressedKeys final {
    metal::MetalBuffer pool;
    metal::MetalBuffer pageTable;
    SplashKvLayer kv{};
  };
  // out = fp32 [rows][heads][LATENT_ATTENTION_DIM]: each head of q (bf16
  // [rows][heads][LATENT_ATTENTION_DIM]) over its row's keys, the first
  // counts[row] of indices [rows][keyStride] (uint32) into `window` or
  // `compressed`, its scores scaled by `scale`, and its sink of `sinks`
  // (fp32 [heads]). `scores` (fp32 [rows][heads][keyStride]) holds the
  // weights between the passes, and `parts` (partBytes, none for many rows)
  // the sums over parts of the keys.
  static void add(metal::CommandGraph &graph, metal::MetalBuffer q, const WindowLatents &window,
                  const CompressedKeys &compressed, metal::MetalBuffer indices, metal::MetalBuffer counts,
                  metal::MetalBuffer sinks, metal::MetalBuffer scores, metal::MetalBuffer parts,
                  metal::MetalBuffer out, uint32_t heads, uint32_t keyStride, float scale, uint32_t rows);
  // The same with a pool of window latents and a pool of compressed ones.
  static void add(metal::CommandGraph &graph, metal::MetalBuffer q, metal::MetalBuffer window,
                  metal::MetalBuffer compressed, metal::MetalBuffer indices, metal::MetalBuffer counts,
                  metal::MetalBuffer sinks, metal::MetalBuffer scores, metal::MetalBuffer parts,
                  metal::MetalBuffer out, uint32_t heads, uint32_t keyStride, float scale, uint32_t rows);
  // A step's key lists for add: indices (uint32 [rows][keyStride]) and
  // counts (uint32 [rows]) of each row's `window` window keys that end at
  // positions[row] (uint32 [rows]), as slots of a pool of `ring` latents
  // (position p at slot p % ring), then of its selected groups, those of its
  // line of `selected` (uint32 [rows][selectedStride], ascending, then
  // LATENT_INDEX_NONE) before the first LATENT_INDEX_NONE; selectedStride 0
  // reads none.
  static void addKeys(metal::CommandGraph &graph, metal::MetalBuffer positions, metal::MetalBuffer selected,
                      metal::MetalBuffer indices, metal::MetalBuffer counts, uint32_t window, uint32_t ring,
                      uint32_t selectedStride, uint32_t keyStride, uint32_t rows);
  // The key lists of a sequence's step whose window keys name WindowLatents:
  // a position below `committed` (the sequence's tokens before the step) its
  // slot p % ring of the ring, one at or past it the step's latent p -
  // committed (slot ring + p - committed); and, with `ratio`, whose selected
  // groups are a sequence's pages' (CompressedKeys::pageTable), group g the
  // position (g + 1) * ratio - 1 of its place. With `limits` (uint32 per
  // lane), its rows from limits[lane] on have no keys: a verify lane's rows
  // past the ones it verifies, which attend nothing.
  struct SplitKeys final {
    uint32_t window = 0;
    uint32_t ring = 0;
    uint32_t committed = 0;
    uint32_t ratio = 0;
    metal::MetalBuffer limits{};
    uint32_t lane = 0;
  };
  static void addKeys(metal::CommandGraph &graph, metal::MetalBuffer positions, metal::MetalBuffer selected,
                      metal::MetalBuffer indices, metal::MetalBuffer counts, SplitKeys keys, uint32_t selectedStride,
                      uint32_t keyStride, uint32_t rows);
  // A sequence's ring of its window's last latents after a step
  // (LatentRingParams), each of `ring` slots of LATENT_ATTENTION_DIM bf16
  // values: of ringIn, the ring before the step, and the step's latents
  // `step` of positions from `committed` on, up to `rows` of them or, with
  // `retained` (uint32 per lane), as many as lane `lane` retained.
  static void addRingUpdate(metal::CommandGraph &graph, metal::MetalBuffer ringIn, metal::MetalBuffer step,
                            metal::MetalBuffer ringOut, metal::MetalBuffer retained, uint32_t committed,
                            uint32_t rows, uint32_t ring, uint32_t lane = 0);
  // A paired KV source's open group after a verify's retained rows
  // (LatentOpenParams): `open` (fp32 [2][width], kv then gate) from the
  // step's kv and gate projections (fp32 [rows][width], row 0 the position
  // before the step's first) where lane `lane`'s retained rows leave a
  // group open.
  static void addOpenCommit(metal::CommandGraph &graph, metal::MetalBuffer kv, metal::MetalBuffer gate,
                            metal::MetalBuffer open, metal::MetalBuffer retained, uint32_t committed, uint32_t width,
                            uint32_t lane);
  // The parts the values pass splits each of `rows` rows' keys into.
  [[nodiscard]] static uint32_t splits(uint32_t rows) noexcept;
  // The bytes of the sums over parts of the keys `rows` rows take.
  [[nodiscard]] static uint64_t partBytes(uint32_t heads, uint32_t rows) noexcept;
};

} // namespace splash::ops
