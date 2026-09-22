#include "FlashMoEBuckets.hpp"

#include "metal/abi/FlashMoEBuckets.h"

#include <array>
#include <stdexcept>
#include <string>

namespace splash::flash {
namespace {

constexpr uint32_t kExperts = 512;
constexpr uint32_t kWidth = 2560;
constexpr uint32_t kThreads = 256;

void requireGeometry(uint32_t rows, uint32_t selections) {
  if (!rows || rows > kFlashMoEBucketMaximumRows || !selections ||
      selections > kFlashMoEBucketMaximumSelections)
    throw std::invalid_argument("Flash MoE buckets unsupported row/selection geometry");
}

void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes,
                  const char *name) {
  if (!buffer || buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash MoE buckets insufficient ") + name);
}

std::array<metal::MetalBuffer, 8> scratchBuffers(
    const FlashMoEBucketScratch &scratch) {
  return {scratch.counts, scratch.offsets, scratch.routeMap,
          scratch.canonicalToPacked, scratch.packedInputs,
          scratch.jobOffsets, scratch.jobCount, scratch.tileJobs};
}

// Public buffer metadata currently identifies exact views only. Shared views
// additionally expose their mapped address, permitting overlap checks without
// reading any dynamic GPU contents. Callers must keep Private views disjoint.
bool overlaps(const metal::MetalBuffer &left, const metal::MetalBuffer &right) {
  if (left.sameView(right)) return true;
  const auto leftBegin = reinterpret_cast<uintptr_t>(left.contents());
  const auto rightBegin = reinterpret_cast<uintptr_t>(right.contents());
  if (!leftBegin || !rightBegin) return false;
  // Subtraction avoids overflowing the address end on an arbitrary mapping.
  if (leftBegin <= rightBegin)
    return rightBegin - leftBegin < left.sizeBytes();
  return leftBegin - rightBegin < right.sizeBytes();
}

void requireScratch(const FlashMoEBucketScratch &scratch, uint32_t rows,
                    uint32_t selections) {
  requireGeometry(rows, selections);
  requireGeometry(scratch.rowCapacity, scratch.selectionCapacity);
  if (rows > scratch.rowCapacity || selections > scratch.selectionCapacity ||
      scratch.routeCapacity != scratch.rowCapacity * scratch.selectionCapacity ||
      scratch.jobCapacity <
          moEBucketJobCapacity(scratch.rowCapacity, scratch.selectionCapacity, 8))
    throw std::invalid_argument("Flash MoE buckets scratch capacity is inconsistent");
  requireBytes(scratch.counts, kExperts * 4, "expert counts");
  requireBytes(scratch.offsets, (kExperts + 1) * 4, "route offsets");
  requireBytes(scratch.routeMap, uint64_t{scratch.routeCapacity} * 4, "route map");
  requireBytes(scratch.canonicalToPacked, uint64_t{scratch.routeCapacity} * 4,
                "inverse route map");
  requireBytes(scratch.packedInputs,
                uint64_t{scratch.routeCapacity} * kWidth * 2, "packed inputs");
  requireBytes(scratch.jobOffsets, (kExperts + 1) * 4, "job offsets");
  requireBytes(scratch.jobCount, 4, "job count");
  requireBytes(scratch.tileJobs,
                uint64_t{scratch.jobCapacity} * sizeof(FlashMoEBucketJob), "tile jobs");
  const auto buffers = scratchBuffers(scratch);
  for (size_t first = 0; first < buffers.size(); ++first)
    for (size_t second = first + 1; second < buffers.size(); ++second)
      if (overlaps(buffers[first], buffers[second]))
        throw std::invalid_argument("Flash MoE buckets scratch buffers alias");
}

void requireDiagnostics(const FlashMoEBucketScratch &scratch,
                        const metal::MetalBuffer &diagnostics) {
  requireBytes(diagnostics, 4, "diagnostics");
  for (const auto &buffer : scratchBuffers(scratch))
    if (overlaps(buffer, diagnostics))
      throw std::invalid_argument("Flash MoE buckets scratch aliases diagnostics");
}

FlashMoEBucketParams packParams(uint32_t rows, uint32_t selections) {
  return {rows, selections, kWidth, kExperts, rows * selections, 0, 0, 0};
}

} // namespace

uint32_t moEBucketJobCapacity(uint32_t rows, uint32_t selections,
                            uint32_t tileRows) {
  requireGeometry(rows, selections);
  if (tileRows != 8 && tileRows != 16 && tileRows != 32 && tileRows != 64)
    throw std::invalid_argument("Flash MoE buckets unsupported matrix tile rows");
  // Sum ceil(count[e]/M) <= ceil(sum(count[e])/M) + E - 1.
  // Counts sum to at most rows*selections, including duplicate legal IDs.
  return (rows * selections + tileRows - 1) / tileRows + kExperts - 1;
}

FlashMoEBucketScratch allocateMoEBucketScratch(metal::MetalBackend &backend,
                                              uint32_t rows,
                                              uint32_t selections,
                                              metal::BufferStorage storage) {
  requireGeometry(rows, selections);
  FlashMoEBucketScratch scratch;
  scratch.rowCapacity = rows;
  scratch.selectionCapacity = selections;
  scratch.routeCapacity = rows * selections;
  scratch.jobCapacity = moEBucketJobCapacity(rows, selections, 8);
  scratch.counts = backend.allocateBuffer(kExperts * 4, storage,
                                         "flash MoE bucket counts");
  scratch.offsets = backend.allocateBuffer((kExperts + 1) * 4, storage,
                                          "flash MoE bucket offsets");
  scratch.routeMap = backend.allocateBuffer(uint64_t{scratch.routeCapacity} * 4,
                                           storage, "flash MoE stable route map");
  scratch.canonicalToPacked = backend.allocateBuffer(
      uint64_t{scratch.routeCapacity} * 4, storage,
      "flash MoE canonical to packed route map");
  scratch.packedInputs = backend.allocateBuffer(
      uint64_t{scratch.routeCapacity} * kWidth * 2, storage,
      "flash MoE packed hidden rows");
  scratch.jobOffsets = backend.allocateBuffer((kExperts + 1) * 4, storage,
                                             "flash MoE bucket job offsets");
  scratch.jobCount = backend.allocateBuffer(4, storage,
                                           "flash MoE bucket job count");
  scratch.tileJobs = backend.allocateBuffer(
      uint64_t{scratch.jobCapacity} * sizeof(FlashMoEBucketJob), storage,
      "flash MoE bucket matrix jobs");
  return scratch;
}

void addMoEBucketPack(metal::CommandGraph &graph, metal::MetalBuffer input,
                      metal::MetalBuffer expertIDs,
                      const FlashMoEBucketScratch &scratch,
                      metal::MetalBuffer diagnostics, uint32_t rows,
                      uint32_t selections) {
  requireScratch(scratch, rows, selections);
  requireDiagnostics(scratch, diagnostics);
  requireBytes(input, uint64_t{rows} * kWidth * 2, "hidden input");
  requireBytes(expertIDs, uint64_t{rows} * selections * sizeof(int64_t), "expert IDs");
  if (overlaps(diagnostics, input) || overlaps(diagnostics, expertIDs))
    throw std::invalid_argument("Flash MoE buckets diagnostics aliases input");
  for (const auto &buffer : scratchBuffers(scratch))
    if (overlaps(buffer, input) || overlaps(buffer, expertIDs))
      throw std::invalid_argument("Flash MoE buckets scratch aliases input");
  const auto params = packParams(rows, selections);
  graph.add("flash_moe_bucket_histogram",
            {expertIDs, scratch.counts, scratch.canonicalToPacked, diagnostics}, params,
            {kExperts, 1, 1}, {kThreads, 1, 1});
  graph.add("flash_moe_bucket_prefix",
            {scratch.counts, scratch.offsets, diagnostics}, params,
            {1, 1, 1}, {kThreads, 1, 1});
  graph.add("flash_moe_bucket_stable_map",
            {expertIDs, scratch.counts, scratch.offsets, scratch.routeMap,
             scratch.canonicalToPacked, diagnostics}, params,
            {kExperts, 1, 1}, {kThreads, 1, 1});
  graph.add("flash_moe_bucket_pack",
            {input, scratch.offsets, scratch.routeMap, scratch.packedInputs,
             diagnostics}, params, {rows * selections, 1, 1},
            {kThreads, 1, 1});
}

void addMoEBucketJobs(metal::CommandGraph &graph,
                      const FlashMoEBucketScratch &scratch,
                      metal::MetalBuffer diagnostics, uint32_t rows,
                      uint32_t tileRows, uint32_t selections) {
  requireScratch(scratch, rows, selections);
  requireDiagnostics(scratch, diagnostics);
  const uint32_t capacity = moEBucketJobCapacity(rows, selections, tileRows);
  if (capacity > scratch.jobCapacity)
    throw std::invalid_argument("Flash MoE buckets insufficient job capacity");
  auto params = packParams(rows, selections);
  params.tile_rows = tileRows;
  params.job_capacity = capacity;
  graph.add("flash_moe_bucket_job_prefix",
            {scratch.counts, scratch.offsets, scratch.jobOffsets,
             scratch.jobCount, diagnostics}, params,
            {1, 1, 1}, {kThreads, 1, 1});
  graph.add("flash_moe_bucket_jobs",
            {scratch.offsets, scratch.jobOffsets, scratch.jobCount,
             scratch.tileJobs, diagnostics}, params,
            {(capacity + kThreads - 1) / kThreads, 1, 1},
            {kThreads, 1, 1});
}

} // namespace splash::flash
