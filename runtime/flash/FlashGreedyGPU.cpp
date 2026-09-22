#include "FlashGreedyGPU.hpp"

#include <array>
#include <stdexcept>
#include <string>

namespace splash::flash {
namespace {
uint32_t partitions(uint32_t vocabulary) {
  if (!vocabulary || vocabulary > 248320)
    throw std::invalid_argument("Flash GPU greedy invalid vocabulary extent");
  return (vocabulary + kFlashGreedyGPUValuesPerPartition - 1) /
      kFlashGreedyGPUValuesPerPartition;
}
void requireRows(uint32_t rows) {
  if (!rows || rows > kFlashGreedyGPUMaximumRows)
    throw std::invalid_argument("Flash GPU greedy requires 1..16 real rows");
}
void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes,
                  const char *name) {
  if (!buffer || buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash GPU greedy insufficient ") + name);
}
bool overlaps(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  if (a.sameView(b)) return true;
  const auto left = reinterpret_cast<uintptr_t>(a.contents());
  const auto right = reinterpret_cast<uintptr_t>(b.contents());
  if (!left || !right) return false;
  return left <= right ? right - left < a.sizeBytes()
                       : left - right < b.sizeBytes();
}
void requireDisjoint(std::span<const metal::MetalBuffer> buffers) {
  for (size_t a = 0; a < buffers.size(); ++a)
    for (size_t b = a + 1; b < buffers.size(); ++b)
      if (overlaps(buffers[a], buffers[b]))
        throw std::invalid_argument("Flash GPU greedy writable/read-only buffers alias");
}
FlashGreedyGPUParams validate(metal::MetalBuffer logits,
                             const FlashGreedyGPUWorkspace &w,
                             uint32_t rows, uint32_t vocabulary,
                             uint32_t stride) {
  requireRows(rows);
  const uint32_t count = partitions(vocabulary);
  requireRows(w.rowCapacity);
  if (w.partitionCapacity != partitions(w.vocabularyCapacity) ||
      rows > w.rowCapacity || vocabulary > w.vocabularyCapacity)
    throw std::invalid_argument("Flash GPU greedy scratch capacity is inconsistent");
  if (!stride) stride = vocabulary;
  if (stride < vocabulary)
    throw std::invalid_argument("Flash GPU greedy row stride is shorter than vocabulary");
  requireBytes(logits, (uint64_t{rows - 1} * stride + vocabulary) * 2, "logits");
  requireBytes(w.partials, uint64_t{w.rowCapacity} * w.partitionCapacity *
      sizeof(FlashGreedyGPURowResult), "partition scratch");
  return {rows, vocabulary, stride, count, 1, rows, 1, 0};
}
void addPartials(metal::CommandGraph &graph, metal::MetalBuffer logits,
                 const FlashGreedyGPUWorkspace &w,
                 const FlashGreedyGPUParams &params) {
  graph.add("flash_greedy_gpu_partials", {logits, w.partials}, params,
            {params.partitions, params.rows, 1}, {256, 1, 1});
}
} // namespace

FlashGreedyGPUWorkspace allocateGreedyGPUWorkspace(
    metal::MetalBackend &backend, uint32_t rows, uint32_t vocabulary,
    metal::BufferStorage storage) {
  requireRows(rows);
  const auto count = partitions(vocabulary);
  return {rows, vocabulary, count,
      backend.allocateBuffer(uint64_t{rows} * count * sizeof(FlashGreedyGPURowResult),
                             storage, "flash GPU greedy partition records")};
}

uint64_t greedyGPUWorkspacePlannedBytes(uint32_t rows, uint32_t vocabulary) {
  requireRows(rows);
  const auto count = partitions(vocabulary);
  constexpr uint64_t alignment = 16384;
  const auto rounded = [](uint64_t bytes) {
    return (bytes + alignment - 1) & ~(alignment - 1);
  };
  return rounded(uint64_t{rows} * count * sizeof(FlashGreedyGPURowResult)) +
      rounded(uint64_t{rows} * sizeof(FlashGreedyGPURowResult));
}

uint32_t greedyGPUResultToken(const FlashGreedyGPURowResult &result,
                             uint32_t vocabulary) {
  if (result.errors & kFlashGreedyGPUErrorNonfinite)
    throw std::runtime_error("non-finite Flash vocabulary logit");
  if (!vocabulary || vocabulary > 248320 || result.errors || result.reserved ||
      result.token >= vocabulary || result.rank < 0x80 || result.rank > 0xff7f ||
      result.rank == 0x7fff)
    throw std::runtime_error("Flash MTP GPU greedy result has invalid status or extent");
  return result.token;
}

void addGreedyGPU(metal::CommandGraph &graph, metal::MetalBuffer logits,
                  const FlashGreedyGPUWorkspace &w, metal::MetalBuffer results,
                  uint32_t rows, uint32_t vocabulary, uint32_t stride) {
  const auto params = validate(logits, w, rows, vocabulary, stride);
  requireBytes(results, uint64_t{rows} * sizeof(FlashGreedyGPURowResult), "row results");
  const std::array buffers{logits, w.partials, results};
  requireDisjoint(buffers);
  addPartials(graph, logits, w, params);
  graph.add("flash_greedy_gpu_finish", {w.partials, results}, params,
            {1, 1, 1}, {256, 1, 1});
}

void addGreedyGPUPrefix(metal::CommandGraph &graph, metal::MetalBuffer logits,
                        const FlashGreedyGPUWorkspace &w,
                        metal::MetalBuffer inputs, metal::MetalBuffer remaining,
                        metal::MetalBuffer results, uint32_t lanes,
                        uint32_t rowsPerLane, uint32_t vocabulary,
                        uint32_t activeLaneMask, uint32_t stride) {
  if (!lanes || lanes > kFlashGreedyGPUMaximumLanes || !rowsPerLane ||
      rowsPerLane > kFlashGreedyGPUMaximumRows ||
      uint64_t{lanes} * rowsPerLane > kFlashGreedyGPUMaximumRows ||
      activeLaneMask >> lanes)
    throw std::invalid_argument("Flash GPU greedy invalid real prefix geometry");
  auto params = validate(logits, w, lanes * rowsPerLane, vocabulary, stride);
  params.lanes = lanes;
  params.rows_per_lane = rowsPerLane;
  params.active_lane_mask = activeLaneMask;
  requireBytes(inputs, uint64_t{params.rows} * 4, "real input tokens");
  requireBytes(remaining, uint64_t{lanes} * 4, "remaining output budgets");
  requireBytes(results, uint64_t{lanes} * sizeof(FlashGreedyGPUPrefixResult), "prefix results");
  const std::array buffers{logits, w.partials, inputs, remaining, results};
  requireDisjoint(buffers);
  addPartials(graph, logits, w, params);
  graph.add("flash_greedy_gpu_prefix", {w.partials, inputs, remaining, results},
            params, {lanes, 1, 1}, {256, 1, 1});
}
} // namespace splash::flash
