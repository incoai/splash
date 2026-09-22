#include "FlashMoEBlocked.hpp"

#include "metal/abi/FlashMoEBlocked.h"
#include "metal/abi/FlashMoEBuckets.h"
#include "metal/abi/FlashMoEDirectA.h"

#include <cstdlib>
#include <array>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::flash {
namespace {

bool q4x8Enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_FLASH_MOE_Q4X8");
    if (!value || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument("SPLASH_FLASH_MOE_Q4X8 must be 0 or 1");
  }();
  return enabled;
}

bool m64Enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_FLASH_MOE_M64");
    if (!value || std::string_view(value) == "0") return false;
    if (std::string_view(value) != "1")
      throw std::invalid_argument("SPLASH_FLASH_MOE_M64 must be 0 or 1");
    if (!q4x8Enabled())
      throw std::invalid_argument("SPLASH_FLASH_MOE_M64 requires SPLASH_FLASH_MOE_Q4X8=1");
    return true;
  }();
  return enabled;
}

bool directAEnabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_FLASH_MOE_DIRECT_A");
    if (!value || std::string_view(value) == "0") return false;
    if (std::string_view(value) != "1")
      throw std::invalid_argument("SPLASH_FLASH_MOE_DIRECT_A must be 0 or 1");
    if (!q4x8Enabled())
      throw std::invalid_argument("SPLASH_FLASH_MOE_DIRECT_A requires SPLASH_FLASH_MOE_Q4X8=1");
    return true;
  }();
  return enabled;
}

bool alignedQ4Projection(const FlashAffineProjection &p) noexcept {
  const void *address = p.weights ? p.weights->buffer.contents() : nullptr;
  return p.experts == 512 && p.bits == 4 && p.groupSize == 64 &&
      ((p.inputSize == 2560 && p.outputSize == 640) ||
       (p.inputSize == 640 && p.outputSize == 2560)) &&
      p.weightRowStrideBytes % 4 == 0 && p.weightExpertStrideBytes % 4 == 0 &&
      address && reinterpret_cast<uintptr_t>(address) % 4 == 0;
}

uint64_t multiply(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::invalid_argument("Flash blocked MoE byte extent overflows");
  return a * b;
}

uint64_t plus(uint64_t a, uint64_t b) {
  if (a > std::numeric_limits<uint64_t>::max() - b)
    throw std::invalid_argument("Flash blocked MoE byte extent overflows");
  return a + b;
}

uint32_t tileRows(FlashMoEBlockedTile tile) {
  const auto rows = static_cast<uint32_t>(tile);
  if (rows != 8 && rows != 16 && rows != 32 && rows != 64)
    throw std::invalid_argument("Flash blocked MoE unsupported matrix tile");
  return rows;
}

void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes,
                  const char *name) {
  if (!buffer || !bytes || buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash blocked MoE insufficient ") + name);
}

void requireGeometry(const FlashMoEBlockedScratch &scratch, uint32_t rows,
                      uint32_t selections, FlashMoEBlockedTile tile) {
  const uint32_t m = tileRows(tile);
  if (m == 64 && (!m64Enabled() || rows < 1024))
    throw std::invalid_argument("Flash blocked MoE M64 requires enabled policy and at least1024 rows");
  if (!rows || rows > kFlashMoEBucketMaximumRows || !selections ||
      selections > kFlashMoEBucketMaximumSelections ||
      rows > scratch.buckets.rowCapacity ||
      selections > scratch.buckets.selectionCapacity ||
      uint64_t{rows} * selections > scratch.buckets.routeCapacity ||
      moEBucketJobCapacity(rows, selections, m) > scratch.buckets.jobCapacity)
    throw std::invalid_argument("Flash blocked MoE unsupported scratch geometry");
  const uint64_t routes = multiply(rows, selections);
  requireBytes(scratch.buckets.counts, 512 * 4, "counts");
  requireBytes(scratch.buckets.offsets, 513 * 4, "offsets");
  requireBytes(scratch.buckets.routeMap, routes * 4, "stable map");
  requireBytes(scratch.buckets.canonicalToPacked, routes * 4, "inverse map");
  const uint64_t operandRows = plus(routes, directAEnabled() ? kFlashMoEDirectAPaddingRows : 0);
  requireBytes(scratch.buckets.packedInputs, multiply(operandRows, 2560 * 2), "packed input");
  requireBytes(scratch.buckets.jobCount, 4, "job count");
  requireBytes(scratch.buckets.tileJobs,
                multiply(moEBucketJobCapacity(rows, selections, m), 8), "matrix jobs");
  requireBytes(scratch.packedActivated, multiply(operandRows, 640 * 2), "packed activation");
  requireBytes(scratch.scatteredDown, multiply(routes, 2560 * 2), "scattered down");
}

void requireProjection(const FlashAffineProjection &p, bool down) {
  const uint32_t n = down ? 2560 : 640, k = down ? 640 : 2560;
  if (p.experts != 512 || p.inputSize != k || p.outputSize != n ||
      p.bits != 4 || p.groupSize != 64 || p.parameterRowStrideBytes % 2 ||
      p.parameterExpertStrideBytes % 2 || !p.weights || !p.scales || !p.biases ||
      p.weights->dtype != FlashDType::U32 || p.scales->dtype != FlashDType::BF16 ||
      p.biases->dtype != FlashDType::BF16)
    throw std::invalid_argument("Flash blocked MoE unsupported original projection");
  const uint64_t weightMatrix = plus(multiply(n - 1, p.weightRowStrideBytes), k / 2);
  const uint64_t parameterMatrix = plus(multiply(n - 1, p.parameterRowStrideBytes), k / 32);
  if (p.weightRowStrideBytes < k / 2 || p.parameterRowStrideBytes < k / 32 ||
      p.weightExpertStrideBytes < weightMatrix ||
      p.parameterExpertStrideBytes < parameterMatrix)
    throw std::invalid_argument("Flash blocked MoE overlapping source strides");
  const uint64_t weights = plus(multiply(511, p.weightExpertStrideBytes), weightMatrix);
  const uint64_t parameters = plus(multiply(511, p.parameterExpertStrideBytes), parameterMatrix);
  if (p.weights->logicalBytes < weights || p.scales->logicalBytes < parameters ||
      p.biases->logicalBytes < parameters)
    throw std::invalid_argument("Flash blocked MoE inconsistent source logical bytes");
  requireBytes(p.weights->buffer, weights, "weights");
  requireBytes(p.scales->buffer, parameters, "scales");
  requireBytes(p.biases->buffer, parameters, "biases");
}

void requireOutputDisjoint(const metal::MetalBuffer &out,
                            std::initializer_list<metal::MetalBuffer> inputs) {
  for (const auto &input : inputs)
    if (out.sameView(input))
      throw std::invalid_argument("Flash blocked MoE unsupported output alias");
}

std::string pipeline(const char *stem, FlashMoEBlockedTile tile) {
  const uint32_t m = tileRows(tile);
  return std::string(stem) + "_m" + std::to_string(m) + "_n64" + (m == 64 ? "_sg8" : "");
}

} // namespace

const char *flashMoEBlockedRouteSemantics() {
  if (directAEnabled())
    return m64Enabled() ? kFlashMoEBlockedDirectAM64Semantics : kFlashMoEBlockedDirectASemantics;
  if (m64Enabled()) return kFlashMoEBlockedQ4x8M64Semantics;
  return q4x8Enabled() ? kFlashMoEBlockedQ4x8Semantics : kFlashMoEBlockedSemantics;
}

bool flashMoEDirectAEnabled() { return directAEnabled(); }

uint64_t flashMoEDirectAWorkspaceExtraBytes(uint32_t routeCapacity,
                                          uint64_t alignment) {
  if (!routeCapacity || routeCapacity > kFlashMoEBucketMaximumRows * 10 ||
      !alignment || (alignment & (alignment - 1)))
    throw std::invalid_argument("Flash direct A workspace geometry/alignment is invalid");
  if (!directAEnabled()) return 0;
  const auto rounded = [alignment](uint64_t value) {
    return plus(value, alignment - 1) & ~(alignment - 1);
  };
  uint64_t total = 0;
  for (const uint64_t width : {2560u, 640u}) {
    const uint64_t original = multiply(routeCapacity, width * 2);
    const uint64_t padded = multiply(uint64_t(routeCapacity) + kFlashMoEDirectAPaddingRows,
                                     width * 2);
    total = plus(total, rounded(padded) - rounded(original));
  }
  return total;
}

uint64_t flashMoEBlockedWorkspacePlannedBytes(uint32_t rows, uint32_t selections,
                                            uint64_t alignment) {
  const uint32_t jobs = moEBucketJobCapacity(rows, selections, 8);
  if (!alignment || (alignment & (alignment - 1)))
    throw std::invalid_argument("Flash blocked workspace alignment must be a power of two");
  const uint64_t routes = multiply(rows, selections);
  const uint64_t operandRows = plus(routes, directAEnabled() ? kFlashMoEDirectAPaddingRows : 0);
  const std::array<uint64_t, 10> sizes{512 * 4, 513 * 4, routes * 4, routes * 4,
      multiply(operandRows, 2560 * 2), 513 * 4, 4, uint64_t(jobs) * 8,
      multiply(operandRows, 640 * 2), multiply(routes, 2560 * 2)};
  uint64_t total = 0;
  for (const auto size : sizes)
    total = plus(total, plus(size, alignment - 1) & ~(alignment - 1));
  return total;
}

FlashMoEBlockedTile flashMoEBlockedTile(uint32_t rows, bool hasHotExpertCache) {
  if (!rows || rows > kFlashMoEBucketMaximumRows)
    throw std::invalid_argument("Flash blocked MoE tile row extent unsupported");
  const bool wide = m64Enabled();
  if (rows >= 1024)
    return wide && rows >= 4096 && !hasHotExpertCache ? FlashMoEBlockedTile::M64N64
                                     : FlashMoEBlockedTile::M32N64;
  return FlashMoEBlockedTile::M16N64;
}

FlashMoEBlockedScratch allocateMoEBlockedScratch(metal::MetalBackend &backend,
                                                uint32_t rows,
                                                uint32_t selections,
                                                metal::BufferStorage storage) {
  FlashMoEBlockedScratch scratch;
  if (!directAEnabled()) {
    scratch.buckets = allocateMoEBucketScratch(backend, rows, selections, storage);
  } else {
    // Allocate the correct extents directly, avoiding a temporary duplicate of
    // the large packedInputs allocation during governed startup.
    auto &b = scratch.buckets;
    b.jobCapacity = moEBucketJobCapacity(rows, selections, 8);
    b.rowCapacity = rows; b.selectionCapacity = selections;
    b.routeCapacity = rows * selections;
    b.counts = backend.allocateBuffer(512 * 4, storage, "flash MoE bucket counts");
    b.offsets = backend.allocateBuffer(513 * 4, storage, "flash MoE bucket offsets");
    b.routeMap = backend.allocateBuffer(uint64_t(b.routeCapacity) * 4, storage,
                                      "flash MoE stable route map");
    b.canonicalToPacked = backend.allocateBuffer(uint64_t(b.routeCapacity) * 4, storage,
                                      "flash MoE canonical to packed route map");
    b.packedInputs = backend.allocateBuffer(
        uint64_t(b.routeCapacity + kFlashMoEDirectAPaddingRows) * 2560 * 2,
        storage, "flash MoE direct A packed hidden rows");
    b.jobOffsets = backend.allocateBuffer(513 * 4, storage, "flash MoE bucket job offsets");
    b.jobCount = backend.allocateBuffer(4, storage, "flash MoE bucket job count");
    b.tileJobs = backend.allocateBuffer(uint64_t(b.jobCapacity) * 8, storage,
                                      "flash MoE bucket matrix jobs");
  }
  const uint64_t routes = multiply(rows, selections);
  const uint64_t operandRows = plus(routes, directAEnabled() ? kFlashMoEDirectAPaddingRows : 0);
  scratch.packedActivated = backend.allocateBuffer(multiply(operandRows, 640 * 2), storage,
                                                    "Flash packed expert activation");
  scratch.scatteredDown = backend.allocateBuffer(multiply(routes, 2560 * 2), storage,
                                                  "Flash canonical expert down");
  return scratch;
}

void addMoEBlockedPack(metal::CommandGraph &graph, metal::MetalBuffer input,
                       metal::MetalBuffer expertIDs,
                       const FlashMoEBlockedScratch &scratch,
                       metal::MetalBuffer diagnostics, uint32_t rows,
                       FlashMoEBlockedTile tile, uint32_t selections) {
  requireGeometry(scratch, rows, selections, tile);
  requireBytes(diagnostics, 4, "diagnostics");
  if (!directAEnabled()) {
    addMoEBucketPack(graph, input, expertIDs, scratch.buckets, diagnostics, rows, selections);
  } else {
    metal::CommandGraph validated;
    addMoEBucketPack(validated, input, expertIDs, scratch.buckets, diagnostics, rows, selections);
    const FlashMoEBucketParams params{rows, selections, 2560, 512, rows * selections, 0, 0, 0};
    for (const auto &dispatch : validated.dispatches()) {
      std::vector<metal::MetalBuffer> buffers;
      for (const auto &binding : dispatch.buffers) buffers.push_back(binding.buffer);
      auto groups = dispatch.threadgroups;
      auto name = dispatch.pipelineName;
      if (name == "flash_moe_bucket_pack") {
        name = "flash_moe_direct_a_pack";
        groups.x += kFlashMoEDirectAPaddingRows;
      }
      graph.add(std::move(name), std::move(buffers), params,
                groups, dispatch.threadsPerThreadgroup);
    }
  }
  addMoEBucketJobs(graph, scratch.buckets, diagnostics, rows, tileRows(tile), selections);
}

void addMoEBlockedGateUp(metal::CommandGraph &graph,
                         const FlashAffineProjection &gate,
                         const FlashAffineProjection &up,
                         const FlashMoEBlockedScratch &scratch,
                         metal::MetalBuffer diagnostics, uint32_t rows,
                         FlashMoEBlockedTile tile, uint32_t selections) {
  requireGeometry(scratch, rows, selections, tile);
  requireProjection(gate, false); requireProjection(up, false);
  requireBytes(diagnostics, 4, "diagnostics");
  requireOutputDisjoint(scratch.packedActivated,
                        {scratch.buckets.packedInputs, gate.weights->buffer,
                         gate.scales->buffer, gate.biases->buffer, up.weights->buffer,
                         up.scales->buffer, up.biases->buffer, scratch.buckets.offsets,
                         scratch.buckets.tileJobs, scratch.buckets.jobCount, diagnostics});
  const uint32_t m = tileRows(tile);
  const FlashMoEBlockedGateParams p{
      {rows, selections, 2560, 640, 512, 0, 0, 0,
       gate.weightRowStrideBytes, gate.weightExpertStrideBytes,
       gate.parameterRowStrideBytes, gate.parameterExpertStrideBytes,
       up.weightRowStrideBytes, up.weightExpertStrideBytes,
       up.parameterRowStrideBytes, up.parameterExpertStrideBytes},
      rows * selections, moEBucketJobCapacity(rows, selections, m), m, 0};
  const bool vectorized = q4x8Enabled() && alignedQ4Projection(gate) && alignedQ4Projection(up);
  if (m == 64 && !vectorized)
    throw std::invalid_argument("Flash blocked MoE M64 requires aligned Q4x8 gate/up sources");
  if (directAEnabled() && !vectorized)
    throw std::invalid_argument("Flash direct A requires aligned Q4x8 gate/up sources");
  graph.add(pipeline(directAEnabled() ? "flash_moe_direct_a_gate_up" :
                    vectorized ? "flash_moe_q4x8_gate_up" : "flash_moe_blocked_gate_up", tile),
            {scratch.buckets.packedInputs, gate.weights->buffer, gate.scales->buffer,
             gate.biases->buffer, up.weights->buffer, up.scales->buffer, up.biases->buffer,
             scratch.buckets.offsets, scratch.buckets.tileJobs, scratch.buckets.jobCount,
             scratch.packedActivated, diagnostics}, p, {10, p.job_capacity, 1}, {m == 64 ? 256u : 128u, 1, 1});
}

void addMoEBlockedDownScatter(metal::CommandGraph &graph,
                              const FlashAffineProjection &down,
                              const FlashMoEBlockedScratch &scratch,
                              metal::MetalBuffer diagnostics, uint32_t rows,
                              FlashMoEBlockedTile tile, uint32_t selections) {
  requireGeometry(scratch, rows, selections, tile); requireProjection(down, true);
  requireBytes(diagnostics, 4, "diagnostics");
  requireOutputDisjoint(scratch.scatteredDown,
                        {scratch.packedActivated, down.weights->buffer, down.scales->buffer,
                         down.biases->buffer, scratch.buckets.routeMap, scratch.buckets.offsets,
                         scratch.buckets.tileJobs, scratch.buckets.jobCount, diagnostics});
  const uint32_t m = tileRows(tile);
  const FlashMoEBlockedDownParams p{
      {rows, selections, 640, 2560, 512, 0, 0, 0,
       down.weightRowStrideBytes, down.weightExpertStrideBytes,
       down.parameterRowStrideBytes, down.parameterExpertStrideBytes},
      rows * selections, moEBucketJobCapacity(rows, selections, m), m, 0};
  const bool vectorized = q4x8Enabled() && alignedQ4Projection(down);
  if (m == 64 && !vectorized)
    throw std::invalid_argument("Flash blocked MoE M64 requires aligned Q4x8 down source");
  // Excluded invalid routes do not appear in a bucket. Poison only those
  // canonical rows so the complete down/combine chain retains ID+NaN flags.
  graph.add("flash_moe_blocked_poison_excluded_routes",
            {scratch.buckets.canonicalToPacked, scratch.scatteredDown, diagnostics},
            p, {10, p.route_capacity, 1}, {256, 1, 1});
  if (directAEnabled() && !vectorized)
    throw std::invalid_argument("Flash direct A requires aligned Q4x8 down source");
  if (directAEnabled()) {
    requireOutputDisjoint(scratch.packedActivated,
        {down.weights->buffer, down.scales->buffer, down.biases->buffer,
         scratch.buckets.routeMap, scratch.buckets.offsets,
         scratch.buckets.tileJobs, scratch.buckets.jobCount, diagnostics});
    graph.add("flash_moe_direct_a_prepare_down",
        {scratch.packedActivated, scratch.buckets.offsets,
         scratch.packedActivated, diagnostics},
        FlashMoEDirectAPrepareParams{p.route_capacity, 640, kFlashMoEDirectAPaddingRows, 0},
        {p.route_capacity + kFlashMoEDirectAPaddingRows, 1, 1}, {256, 1, 1});
  }
  graph.add(pipeline(directAEnabled() ? "flash_moe_direct_a_down_scatter" :
                    vectorized ? "flash_moe_q4x8_down_scatter" : "flash_moe_blocked_down_scatter", tile),
            {scratch.packedActivated, down.weights->buffer, down.scales->buffer,
             down.biases->buffer, scratch.buckets.offsets, scratch.buckets.tileJobs,
             scratch.buckets.jobCount, scratch.buckets.routeMap, scratch.scatteredDown,
             diagnostics}, p, {40, p.job_capacity, 1}, {m == 64 ? 256u : 128u, 1, 1});
}

} // namespace splash::flash
