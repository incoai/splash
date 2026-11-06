#include "ops/LatentAttention.hpp"

#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace splash::ops {

uint32_t LatentAttention::splits(uint32_t rows) noexcept {
  // A decode row's 8 head blocks in 16 parts take 128 threadgroups.
  return std::clamp<uint32_t>(LATENT_ATTENTION_MAX_SPLITS / std::max(rows, 1u), 1, LATENT_ATTENTION_MAX_SPLITS);
}

uint64_t LatentAttention::partBytes(uint32_t heads, uint32_t rows) noexcept {
  const uint32_t parts = splits(rows);
  return parts > 1 ? uint64_t{rows} * parts * heads * LATENT_ATTENTION_DIM * sizeof(float) : 0;
}

namespace {

void addKeyLists(metal::CommandGraph &graph, metal::MetalBuffer positions, metal::MetalBuffer selected,
                 metal::MetalBuffer indices, metal::MetalBuffer counts, LatentAttentionKeysParams params,
                 metal::MetalBuffer limits = {}) {
  const uint32_t rows = params.rows, window = params.window, selectedStride = params.selected,
                 keyStride = params.key_stride;
  if (!rows || !window || params.ring < window || keyStride < uint64_t{window} + selectedStride)
    throw std::invalid_argument("invalid latent attention key lists");
  requireBytes(positions, uint64_t{rows} * sizeof(uint32_t), "latent attention key positions");
  requireBytes(selected, uint64_t{rows} * selectedStride * sizeof(uint32_t), "latent attention selected groups");
  requireBytes(indices, uint64_t{rows} * keyStride * sizeof(uint32_t), "latent attention key indices");
  requireBytes(counts, uint64_t{rows} * sizeof(uint32_t), "latent attention key counts");
  if (params.limited) requireBytes(limits, uint64_t{params.lane + 1} * sizeof(uint32_t), "latent attention row limits");
  // A selection of none, and no limits, bind the positions in their place,
  // unread.
  metal::MetalBuffer groups = selectedStride ? std::move(selected) : positions;
  if (!params.limited) limits = positions;
  graph.add("latent_attention_keys",
            {std::move(positions), std::move(groups), std::move(indices), std::move(counts), std::move(limits)},
            params, {rows, 1, 1}, {LATENT_ATTENTION_THREADS, 1, 1});
}

} // namespace

void LatentAttention::addKeys(metal::CommandGraph &graph, metal::MetalBuffer positions, metal::MetalBuffer selected,
                              metal::MetalBuffer indices, metal::MetalBuffer counts, uint32_t window, uint32_t ring,
                              uint32_t selectedStride, uint32_t keyStride, uint32_t rows) {
  addKeyLists(graph, std::move(positions), std::move(selected), std::move(indices), std::move(counts),
              {rows, window, ring, selectedStride, keyStride, 0, 0, 0, 0, 0});
}

void LatentAttention::addKeys(metal::CommandGraph &graph, metal::MetalBuffer positions, metal::MetalBuffer selected,
                              metal::MetalBuffer indices, metal::MetalBuffer counts, SplitKeys keys,
                              uint32_t selectedStride, uint32_t keyStride, uint32_t rows) {
  const bool limited = static_cast<bool>(keys.limits);
  addKeyLists(graph, std::move(positions), std::move(selected), std::move(indices), std::move(counts),
              {rows, keys.window, keys.ring, selectedStride, keyStride, keys.committed, 1, keys.ratio, limited ? 1u : 0u,
               keys.lane},
              keys.limits);
}

void LatentAttention::add(metal::CommandGraph &graph, metal::MetalBuffer q, metal::MetalBuffer window,
                          metal::MetalBuffer compressed, metal::MetalBuffer indices, metal::MetalBuffer counts,
                          metal::MetalBuffer sinks, metal::MetalBuffer scores, metal::MetalBuffer parts,
                          metal::MetalBuffer out, uint32_t heads, uint32_t keyStride, float scale, uint32_t rows) {
  add(graph, std::move(q), WindowLatents{.step = std::move(window), .ring = {}, .ringSlots = 0},
      CompressedKeys{.pool = std::move(compressed), .pageTable = {}, .kv = {}}, std::move(indices), std::move(counts),
      std::move(sinks), std::move(scores), std::move(parts), std::move(out), heads, keyStride, scale, rows);
}

void LatentAttention::add(metal::CommandGraph &graph, metal::MetalBuffer q, const WindowLatents &window,
                          const CompressedKeys &compressed, metal::MetalBuffer indices, metal::MetalBuffer counts,
                          metal::MetalBuffer sinks, metal::MetalBuffer scores, metal::MetalBuffer parts,
                          metal::MetalBuffer out, uint32_t heads, uint32_t keyStride, float scale, uint32_t rows) {
  if (!rows || !heads || heads > LATENT_ATTENTION_MAX_HEADS || !keyStride || !window.step ||
      (window.ringSlots && !window.ring) || !compressed.pool == !compressed.pageTable)
    throw std::invalid_argument("invalid latent attention shape");
  constexpr uint64_t dim = LATENT_ATTENTION_DIM;
  requireBytes(q, uint64_t{rows} * heads * dim * sizeof(uint16_t), "latent attention queries");
  requireBytes(indices, uint64_t{rows} * keyStride * sizeof(uint32_t), "latent attention key indices");
  requireBytes(counts, uint64_t{rows} * sizeof(uint32_t), "latent attention key counts");
  requireBytes(sinks, uint64_t{heads} * sizeof(float), "latent attention sinks");
  requireBytes(scores, uint64_t{rows} * heads * keyStride * sizeof(float), "latent attention scores");
  requireBytes(out, uint64_t{rows} * heads * dim * sizeof(float), "latent attention output");
  requireBytes(parts, partBytes(heads, rows), "latent attention parts");
  const uint32_t split = splits(rows);
  const bool paged = static_cast<bool>(compressed.pageTable);
  const LatentAttentionParams params{rows, heads, keyStride, scale, split, window.ringSlots, compressed.kv};
  // Without a ring the step's latents stand in its binding, unread.
  const metal::MetalBuffer ring = window.ringSlots ? window.ring : window.step;
  const metal::MetalBuffer keys = paged ? compressed.pageTable : compressed.pool;
  // A step of many rows multiplies each row's heads in simdgroup matrices; a
  // decode's few rows take their keys in parts instead (splits).
  if (rows > LATENT_ATTENTION_MAX_SPLITS) {
    graph.add(paged ? "latent_attention_scores_mma_paged" : "latent_attention_scores_mma",
              {std::move(q), window.step, keys, indices, counts, scores, ring}, params,
              {rows, (keyStride + LATENT_ATTENTION_MMA_KEYS - 1) / LATENT_ATTENTION_MMA_KEYS, 1},
              {LATENT_ATTENTION_MMA_THREADS, 1, 1});
    graph.add("latent_attention_softmax", {scores, counts, std::move(sinks)}, params, {rows, heads, 1}, {32, 1, 1});
    graph.add(paged ? "latent_attention_values_mma_paged" : "latent_attention_values_mma",
              {std::move(scores), window.step, keys, std::move(indices), std::move(counts), std::move(out), ring},
              params, {rows, LATENT_ATTENTION_DIM / LATENT_ATTENTION_MMA_VALUES, 1},
              {LATENT_ATTENTION_MMA_THREADS, 1, 1});
    return;
  }
  graph.add(paged ? "latent_attention_scores_paged" : "latent_attention_scores",
            {std::move(q), window.step, keys, indices, counts, scores, ring}, params,
            {rows, (keyStride + LATENT_ATTENTION_KEY_BLOCK - 1) / LATENT_ATTENTION_KEY_BLOCK, 1},
            {LATENT_ATTENTION_THREADS, 1, 1});
  graph.add("latent_attention_softmax", {scores, counts, std::move(sinks)}, params, {rows, heads, 1}, {32, 1, 1});
  graph.add(paged ? "latent_attention_values_paged" : "latent_attention_values",
            {std::move(scores), window.step, keys, std::move(indices), std::move(counts), split > 1 ? parts : out,
             ring},
            params, {rows, (heads + LATENT_ATTENTION_HEAD_BLOCK - 1) / LATENT_ATTENTION_HEAD_BLOCK, split},
            {LATENT_ATTENTION_VALUE_THREADS, 1, 1});
  if (split > 1)
    graph.add("latent_attention_sum", {std::move(parts), std::move(out)}, params,
              {rows, (heads * dim / 4 + LATENT_ATTENTION_THREADS - 1) / LATENT_ATTENTION_THREADS, 1},
              {LATENT_ATTENTION_THREADS, 1, 1});
}

void LatentAttention::addRingUpdate(metal::CommandGraph &graph, metal::MetalBuffer ringIn, metal::MetalBuffer step,
                                    metal::MetalBuffer ringOut, metal::MetalBuffer retained, uint32_t committed,
                                    uint32_t rows, uint32_t ring, uint32_t lane) {
  if (!rows || !ring) throw std::invalid_argument("invalid latent ring update");
  constexpr uint64_t row = LATENT_ATTENTION_DIM * sizeof(uint16_t);
  requireBytes(ringIn, ring * row, "latent ring");
  requireBytes(ringOut, ring * row, "latent ring");
  requireBytes(step, rows * row, "latent ring step");
  const bool counted = static_cast<bool>(retained);
  graph.add("latent_ring_update",
            {std::move(ringIn), step, std::move(ringOut), counted ? std::move(retained) : step},
            LatentRingParams{committed, rows, ring, counted, lane}, {ring, 1, 1}, {LATENT_ATTENTION_DIM / 4, 1, 1});
}

void LatentAttention::addOpenCommit(metal::CommandGraph &graph, metal::MetalBuffer kv, metal::MetalBuffer gate,
                                    metal::MetalBuffer open, metal::MetalBuffer retained, uint32_t committed,
                                    uint32_t width, uint32_t lane) {
  if (!width) throw std::invalid_argument("invalid latent open group");
  requireBytes(open, uint64_t{2} * width * sizeof(float), "latent open group");
  graph.add("latent_open_commit", {std::move(kv), std::move(gate), std::move(open), std::move(retained)},
            LatentOpenParams{committed, width, lane}, {(width + 63) / 64, 1, 1}, {64, 1, 1});
}

} // namespace splash::ops
