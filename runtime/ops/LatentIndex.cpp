#include "ops/LatentIndex.hpp"

#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace splash::ops {

void LatentIndex::addCompress(metal::CommandGraph &graph, metal::MetalBuffer kv, metal::MetalBuffer gate,
                              metal::MetalBuffer out, uint32_t groups, uint32_t width, uint32_t ratio,
                              bool floatInput) {
  if (!groups || !width || !ratio || ratio > LATENT_COMPRESS_MAX_RATIO)
    throw std::invalid_argument("invalid latent compression shape");
  const uint64_t rows = uint64_t{groups} * ratio;
  const uint64_t element = floatInput ? sizeof(float) : sizeof(uint16_t);
  requireBytes(kv, rows * width * element, "latent compression latents");
  requireBytes(gate, rows * width * element, "latent compression gate");
  requireBytes(out, uint64_t{groups} * width * sizeof(float), "latent compression output");
  graph.add(floatInput ? "latent_compress_f32" : "latent_compress_bf16", {std::move(kv), std::move(gate), std::move(out)},
            LatentCompressParams{groups, width, ratio},
            {groups, (width + LATENT_INDEX_THREADS - 1) / LATENT_INDEX_THREADS, 1}, {LATENT_INDEX_THREADS, 1, 1});
}

void LatentIndex::addScores(metal::CommandGraph &graph, metal::MetalBuffer q, metal::MetalBuffer weights,
                            metal::MetalBuffer keys, metal::MetalBuffer positions, metal::MetalBuffer scores,
                            uint32_t heads, uint32_t groups, uint32_t stride, uint32_t ratio, float weightScale,
                            uint32_t rows) {
  if (!rows || !heads || heads > LATENT_INDEX_MAX_HEADS || !stride || !ratio)
    throw std::invalid_argument("invalid latent index shape");
  requireBytes(q, uint64_t{rows} * heads * LATENT_INDEX_DIM * sizeof(uint16_t), "latent index queries");
  requireBytes(weights, uint64_t{rows} * heads * sizeof(float), "latent index weights");
  requireBytes(keys, uint64_t{groups} * LATENT_INDEX_DIM * sizeof(uint16_t), "latent index keys");
  requireBytes(positions, uint64_t{rows} * sizeof(uint32_t), "latent index positions");
  requireBytes(scores, uint64_t{rows} * stride * sizeof(float), "latent index scores");
  graph.add("latent_index_scores",
            {std::move(q), std::move(weights), std::move(keys), std::move(positions), std::move(scores)},
            LatentIndexParams{rows, heads, groups, stride, ratio, weightScale, {}},
            {rows, (stride + LATENT_INDEX_THREADS - 1) / LATENT_INDEX_THREADS, 1}, {LATENT_INDEX_THREADS, 1, 1});
}

void LatentIndex::addScores(metal::CommandGraph &graph, metal::MetalBuffer q, metal::MetalBuffer weights,
                            PagedKeys keys, metal::MetalBuffer positions, metal::MetalBuffer scores, uint32_t heads,
                            uint32_t groups, uint32_t stride, uint32_t ratio, float weightScale, uint32_t rows) {
  if (!rows || !heads || heads > LATENT_INDEX_MAX_HEADS || !stride || !ratio || !keys.pageTable ||
      !keys.kv.extent_pages)
    throw std::invalid_argument("invalid latent index shape");
  requireBytes(q, uint64_t{rows} * heads * LATENT_INDEX_DIM * sizeof(uint16_t), "latent index queries");
  requireBytes(weights, uint64_t{rows} * heads * sizeof(float), "latent index weights");
  requireBytes(positions, uint64_t{rows} * sizeof(uint32_t), "latent index positions");
  requireBytes(scores, uint64_t{rows} * stride * sizeof(float), "latent index scores");
  graph.add("latent_index_scores_paged",
            {std::move(q), std::move(weights), std::move(keys.pageTable), std::move(positions), std::move(scores)},
            LatentIndexParams{rows, heads, groups, stride, ratio, weightScale, keys.kv},
            {rows, (stride + LATENT_INDEX_THREADS - 1) / LATENT_INDEX_THREADS, 1}, {LATENT_INDEX_THREADS, 1, 1});
}

void LatentIndex::addStore(metal::CommandGraph &graph, metal::MetalBuffer latents, metal::MetalBuffer indexKeys,
                           PagedKeys pages, uint32_t first, uint32_t groups, uint32_t ratio) {
  if (!groups || !ratio || !pages.pageTable || !pages.kv.extent_pages)
    throw std::invalid_argument("invalid latent store");
  requireBytes(latents, uint64_t{groups} * LATENT_STORE_WIDTH * sizeof(uint16_t), "latent store latents");
  requireBytes(indexKeys, uint64_t{groups} * LATENT_INDEX_DIM * sizeof(uint16_t), "latent store index keys");
  graph.add("latent_index_store", {std::move(latents), std::move(indexKeys), std::move(pages.pageTable)},
            LatentStoreParams{groups, first, ratio, pages.kv}, {groups, 1, 1}, {LATENT_STORE_WIDTH / 4, 1, 1});
}

void LatentIndex::addSelect(metal::CommandGraph &graph, metal::MetalBuffer scores, metal::MetalBuffer selected,
                            uint32_t count, uint32_t k, uint32_t stride, uint32_t outStride, uint32_t rows) {
  if (!rows || !count || !k || count > stride || k > outStride) throw std::invalid_argument("invalid latent selection");
  requireBytes(scores, uint64_t{rows} * stride * sizeof(float), "latent selection scores");
  requireBytes(selected, uint64_t{rows} * outStride * sizeof(uint32_t), "latent selection output");
  graph.add("latent_index_select", {std::move(scores), std::move(selected)},
            LatentSelectParams{rows, count, k, stride, outStride}, {rows, 1, 1}, {LATENT_INDEX_SELECT_THREADS, 1, 1});
}

void LatentIndex::addBlocks(metal::CommandGraph &graph, metal::MetalBuffer scores, metal::MetalBuffer positions,
                            metal::MetalBuffer blocks, uint32_t groups, uint32_t stride, uint32_t block, uint32_t ratio,
                            uint32_t rows) {
  if (!rows || !groups || groups > stride || !block || !ratio) throw std::invalid_argument("invalid latent blocks");
  const uint32_t count = (groups + block - 1) / block;
  requireBytes(scores, uint64_t{rows} * stride * sizeof(float), "latent block scores input");
  requireBytes(positions, uint64_t{rows} * sizeof(uint32_t), "latent block positions");
  requireBytes(blocks, uint64_t{rows} * count * sizeof(float), "latent block scores");
  graph.add("latent_index_blocks", {std::move(scores), std::move(positions), std::move(blocks)},
            LatentBlockParams{rows, groups, stride, block, count, ratio, 0},
            {rows, (count + LATENT_INDEX_THREADS - 1) / LATENT_INDEX_THREADS, 1}, {LATENT_INDEX_THREADS, 1, 1});
}

void LatentIndex::addMask(metal::CommandGraph &graph, metal::MetalBuffer scores, metal::MetalBuffer kept,
                          uint32_t groups, uint32_t stride, uint32_t block, uint32_t listStride, uint32_t rows) {
  if (!rows || !groups || groups > stride || !block || !listStride) throw std::invalid_argument("invalid latent mask");
  requireBytes(scores, uint64_t{rows} * stride * sizeof(float), "latent masked scores");
  requireBytes(kept, uint64_t{rows} * listStride * sizeof(uint32_t), "latent kept blocks");
  const uint32_t count = (groups + block - 1) / block;
  graph.add("latent_index_mask", {std::move(scores), std::move(kept)},
            LatentBlockParams{rows, groups, stride, block, count, 1, listStride},
            {rows, (groups + LATENT_INDEX_THREADS - 1) / LATENT_INDEX_THREADS, 1}, {LATENT_INDEX_THREADS, 1, 1});
}

uint32_t LatentIndex::pieceRows(uint32_t groups, uint32_t rows) {
  if (!groups || !rows) throw std::invalid_argument("invalid latent index piece");
  return static_cast<uint32_t>(std::clamp<uint64_t>(kScoreBytes / (uint64_t{groups} * sizeof(float)), 1, rows));
}

uint64_t LatentIndex::blockBytes(uint32_t block, uint32_t rows) {
  if (!block || !rows) throw std::invalid_argument("invalid latent block scratch");
  // A piece's m rows of g groups hold m * g <= kScoreBytes / 4 scores, and
  // m * ceil(g / block) <= (m * g + m * (block - 1)) / block blocks: under
  // a block's share of the scores plus one a row.
  return kScoreBytes / block + uint64_t{rows} * sizeof(float);
}

} // namespace splash::ops
