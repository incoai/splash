#pragma once

// DeepSeek-V4.1's compressor and indexer (kernels/shared/latent_index.metal):
// a KV source layer's compressed latents, every `ratio` rows' latents
// weighted by the softmax of their gate scores (addCompress, before the
// norm, rotation and FP4 rounding of ops::LatentRotation), and the indexer's
// score of each compressed group a query row reaches (addScores), and the
// row's top groups its attention reads (addSelect), among the candidate
// blocks the first indexer of the candidates chose (addBlocks, then
// addSelect of the blocks) where a later layer keeps them alone (addMask).

#include "metal/CommandGraph.hpp"
#include "metal/abi/LatentIndex.h"

#include <cstdint>

namespace splash::ops {

struct LatentIndex final {
  // out = fp32 [groups][width]: group g's latent from rows g * ratio to (g +
  // 1) * ratio - 1 of kv and gate ([groups * ratio][width], bf16 or, where
  // floatInput, fp32).
  static void addCompress(metal::CommandGraph &graph, metal::MetalBuffer kv, metal::MetalBuffer gate,
                          metal::MetalBuffer out, uint32_t groups, uint32_t width, uint32_t ratio,
                          bool floatInput = false);
  // scores = fp32 [rows][stride]: each row's score of each group below
  // stride: the sum over its heads of weights (fp32 [rows][heads]) times
  // weightScale times the ReLU of the head's query (bf16 [rows][heads][128])
  // with the group's key (bf16 [groups][128]), -inf past the groups the row
  // reaches at positions[row] (uint32 [rows]) and `ratio`.
  // A paged pass's keys: a sequence's pages, through its page table, in
  // the region `kv` (LatentIndexParams::kv).
  struct PagedKeys final {
    metal::MetalBuffer pageTable;
    SplashKvLayer kv{};
  };
  static void addScores(metal::CommandGraph &graph, metal::MetalBuffer q, metal::MetalBuffer weights, PagedKeys keys,
                        metal::MetalBuffer positions, metal::MetalBuffer scores, uint32_t heads, uint32_t groups,
                        uint32_t stride, uint32_t ratio, float weightScale, uint32_t rows);
  // Groups `first` to first + groups - 1 of a KV source into a sequence's
  // pages (LatentStoreParams): their latents (bf16 [groups][512]) and index
  // keys (bf16 [groups][LATENT_INDEX_DIM]).
  static void addStore(metal::CommandGraph &graph, metal::MetalBuffer latents, metal::MetalBuffer indexKeys,
                       PagedKeys pages, uint32_t first, uint32_t groups, uint32_t ratio);
  static void addScores(metal::CommandGraph &graph, metal::MetalBuffer q, metal::MetalBuffer weights,
                        metal::MetalBuffer keys, metal::MetalBuffer positions, metal::MetalBuffer scores,
                        uint32_t heads, uint32_t groups, uint32_t stride, uint32_t ratio, float weightScale,
                        uint32_t rows);
  // selected = uint32 [rows][outStride]: each row's top k of the first
  // `count` scores of its line (fp32 [rows][stride]), every one above the
  // k-th highest and of those equal to it the lowest, ascending, then
  // LATENT_INDEX_NONE where fewer than k are above -inf.
  static void addSelect(metal::CommandGraph &graph, metal::MetalBuffer scores, metal::MetalBuffer selected,
                        uint32_t count, uint32_t k, uint32_t stride, uint32_t outStride, uint32_t rows);
  // blocks = fp32 [rows][ceil(groups / block)]: the highest score of each
  // block of `block` groups a row reaches at positions[row] and `ratio`,
  // -inf past them, +inf for the block of the last of them.
  static void addBlocks(metal::CommandGraph &graph, metal::MetalBuffer scores, metal::MetalBuffer positions,
                        metal::MetalBuffer blocks, uint32_t groups, uint32_t stride, uint32_t block, uint32_t ratio,
                        uint32_t rows);
  // The first `groups` scores of each row's line, in place, -inf outside the
  // blocks of `block` groups its list of kept blocks (uint32 [rows]
  // [listStride], addSelect's) names.
  static void addMask(metal::CommandGraph &graph, metal::MetalBuffer scores, metal::MetalBuffer kept,
                      uint32_t groups, uint32_t stride, uint32_t block, uint32_t listStride, uint32_t rows);

  // An indexer scores `rows` rows of `groups` groups in pieces of pieceRows
  // rows: as many as their scores of every group fit kScoreBytes, at least
  // one. A piece's blocks of `block` groups (addBlocks), of pieces of up to
  // `rows` rows, fit blockBytes: a block's share of the scores, and a float
  // a row for its last block, which may be partial.
  static constexpr uint64_t kScoreBytes = 64ull << 20;
  [[nodiscard]] static uint32_t pieceRows(uint32_t groups, uint32_t rows);
  [[nodiscard]] static uint64_t blockBytes(uint32_t block, uint32_t rows);
};

} // namespace splash::ops
