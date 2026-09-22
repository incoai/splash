#include "FlashExpertDenseCache.hpp"
#include "metal/abi/FlashExpertDenseCache.h"

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace splash::flash {
namespace {
constexpr uint64_t kAlignment = 16384, kGuardBytes = 64;
constexpr uint8_t kGuard = 0x5a;
uint64_t mul(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::invalid_argument("expert cache byte extent overflows");
  return a * b;
}
uint64_t plus(uint64_t a, uint64_t b) {
  if (a > std::numeric_limits<uint64_t>::max() - b)
    throw std::invalid_argument("expert cache byte extent overflows");
  return a + b;
}
uint64_t rounded(uint64_t n) { return plus(n, kAlignment - 1) & ~(kAlignment - 1); }
std::vector<uint32_t> selected(std::span<const uint32_t> ids) {
  if (ids.empty() || ids.size() > 128)
    throw std::invalid_argument("expert cache selection requires1..128 fixed IDs");
  std::vector<uint32_t> result(ids.begin(), ids.end());
  std::sort(result.begin(), result.end());
  if (result.back() >= 512 || std::adjacent_find(result.begin(), result.end()) != result.end())
    throw std::invalid_argument("expert cache IDs must be unique and below512");
  return result;
}
void bytes(const metal::MetalBuffer &b, uint64_t n) {
  if (!b || !n || b.sizeBytes() < n || !b.contents())
    throw std::invalid_argument("expert cache requires sufficient Shared buffer views");
}
bool overlaps(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  const auto aa = reinterpret_cast<uintptr_t>(a.contents());
  const auto bb = reinterpret_cast<uintptr_t>(b.contents());
  if (!aa || !bb) throw std::invalid_argument("expert cache requires Shared addressable views");
  return aa <= bb ? uint64_t(bb - aa) < a.sizeBytes() : uint64_t(aa - bb) < b.sizeBytes();
}
void disjoint(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  if (overlaps(a, b)) throw std::invalid_argument("expert cache output/input buffers overlap");
}
void projection(const FlashAffineProjection &p, bool down) {
  const uint32_t n = down ? 2560 : 640, k = down ? 640 : 2560;
  if (p.experts != 512 || p.outputSize != n || p.inputSize != k || p.bits != 4 ||
      p.groupSize != 64 || p.weightRowStrideBytes < k / 2 || p.weightRowStrideBytes % 4 ||
      p.weightExpertStrideBytes % 4 || p.parameterRowStrideBytes < k / 32 ||
      p.parameterRowStrideBytes % 2 || p.parameterExpertStrideBytes % 2 ||
      !p.weights || !p.scales || !p.biases || p.weights->dtype != FlashDType::U32 ||
      p.scales->dtype != FlashDType::BF16 || p.biases->dtype != FlashDType::BF16)
    throw std::invalid_argument("expert cache requires aligned original E512 Q4/G64 H2560/I640 projections");
  const auto extent = [&](uint64_t rowStride, uint64_t expertStride, uint64_t rowBytes) {
    const uint64_t matrix = plus(mul(n - 1, rowStride), rowBytes);
    if (expertStride < matrix) throw std::invalid_argument("expert cache source expert strides overlap");
    return plus(mul(511, expertStride), matrix);
  };
  const uint64_t wb = extent(p.weightRowStrideBytes, p.weightExpertStrideBytes, k / 2);
  const uint64_t sb = extent(p.parameterRowStrideBytes, p.parameterExpertStrideBytes, k / 32);
  if (p.weights->logicalBytes < wb || p.scales->logicalBytes < sb || p.biases->logicalBytes < sb)
    throw std::invalid_argument("expert cache source logical bytes are insufficient");
  bytes(p.weights->buffer, wb); bytes(p.scales->buffer, sb); bytes(p.biases->buffer, sb);
  if (reinterpret_cast<uintptr_t>(p.weights->buffer.contents()) % 4 ||
      reinterpret_cast<uintptr_t>(p.scales->buffer.contents()) % 2 ||
      reinterpret_cast<uintptr_t>(p.biases->buffer.contents()) % 2)
    throw std::invalid_argument("expert cache source bases are unaligned");
}
std::string hash(const std::string &text) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  if (text.size() > std::numeric_limits<CC_LONG>::max() ||
      !CC_SHA256(text.data(), CC_LONG(text.size()), digest.data()))
    throw std::runtime_error("expert cache identity hash failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (auto v : digest) { result += hex[v >> 4]; result += hex[v & 15]; }
  return result;
}
std::string name(const char *phase, FlashMoEBlockedTile tile) {
  const uint32_t m = static_cast<uint32_t>(tile);
  if (m != 8 && m != 16 && m != 32) throw std::invalid_argument("unsupported expert cache tile");
  return std::string("flash_expert_cache_") + phase + "_m" + std::to_string(m) + "_n64";
}
} // namespace

struct FlashExpertDenseCache::Impl final {
  metal::MetalBackend &backend;
  std::vector<uint32_t> ids;
  std::array<FlashTensor, 9> sourceTensors;
  std::array<FlashAffineProjection, 3> source;
  std::array<FlashTensor, 3> cached;
  std::array<metal::MetalBuffer, 3> allocations;
  metal::MetalBuffer rankMap, selectedIDs, diagnostics;
  std::string identity;
  uint64_t allocatedBytes = 0;
  metal::CommandTiming initialization;

  Impl(metal::MetalBackend &b, const FlashWeights &w, std::string prefix,
       std::span<const uint32_t> selection) : backend(b), ids(selected(selection)) {
    (void)FlashExpertDenseCache::plannedBytes(w, prefix, ids);
    for (uint32_t i = 0; i < 3; ++i) {
      source[i] = w.projection(prefix + (i == 0 ? ".gate_proj" : i == 1 ? ".up_proj" : ".down_proj"));
      sourceTensors[i * 3] = *source[i].weights;
      sourceTensors[i * 3 + 1] = *source[i].scales;
      sourceTensors[i * 3 + 2] = *source[i].biases;
      source[i].weights = &sourceTensors[i * 3];
      source[i].scales = &sourceTensors[i * 3 + 1];
      source[i].biases = &sourceTensors[i * 3 + 2];
    }
    std::string record = std::string(kFlashExpertDenseCacheOperandFormat) + '\n';
    const auto field = [&](std::string_view key, std::string_view value) {
      record += key; record += ':'; record += std::to_string(value.size()); record += ':';
      record += value; record += '\n';
    };
    field("source", w.sourceIdentity()); field("manifest", w.manifestFingerprint());
    field("prefix", prefix); field("execution", kFlashExpertDenseCacheSemantics);
    for (uint32_t id : ids) field("expert", std::to_string(id));
    for (const auto &p : source) field("geometry", std::to_string(p.outputSize) + "," +
        std::to_string(p.inputSize) + ",4,64," + std::to_string(p.weightRowStrideBytes) + "," +
        std::to_string(p.weightExpertStrideBytes) + "," + std::to_string(p.parameterRowStrideBytes) + "," +
        std::to_string(p.parameterExpertStrideBytes));
    identity = hash(record);
    const uint64_t before = backend.memoryStats().allocatedBytes;
    rankMap = backend.allocateBuffer(kAlignment, metal::BufferStorage::Shared, "hot expert ID to rank");
    selectedIDs = backend.allocateBuffer(kAlignment, metal::BufferStorage::Shared, "hot expert selection");
    diagnostics = backend.allocateBuffer(kAlignment, metal::BufferStorage::Shared, "hot expert conversion diagnostics");
    std::memset(rankMap.contents(), 0xff, kAlignment);
    std::memset(selectedIDs.contents(), 0xff, kAlignment);
    std::memset(diagnostics.contents(), 0, kAlignment);
    std::memcpy(selectedIDs.contents(), ids.data(), ids.size() * 4);
    auto *ranks = static_cast<uint32_t *>(rankMap.contents());
    for (uint32_t rank = 0; rank < ids.size(); ++rank) ranks[ids[rank]] = rank;
    metal::CommandGraph graph;
    for (uint32_t i = 0; i < 3; ++i) {
      const auto &p = source[i];
      const uint64_t logical = mul(mul(mul(ids.size(), p.outputSize), p.inputSize), 2);
      allocations[i] = backend.allocateBuffer(rounded(plus(logical, kGuardBytes)),
          metal::BufferStorage::Shared, "hot expert immutable BF16 coefficients");
      std::memset(allocations[i].contents(), 0xa5, logical);
      std::memset(static_cast<uint8_t *>(allocations[i].contents()) + logical, kGuard, kGuardBytes);
      cached[i] = {backend.view(allocations[i], 0, logical), FlashDType::BF16,
          {ids.size(), p.outputSize, p.inputSize}, logical};
      const FlashExpertCacheConvertParams params{{1, 1, p.inputSize, p.outputSize, 512, 4, 64, 0,
          p.weightRowStrideBytes, p.weightExpertStrideBytes, p.parameterRowStrideBytes,
          p.parameterExpertStrideBytes}, uint32_t(ids.size()), 0, 0, 0};
      graph.add("flash_expert_cache_convert_q4x8", {p.weights->buffer, p.scales->buffer,
          p.biases->buffer, selectedIDs, cached[i].buffer, diagnostics}, params,
          {(uint64_t{p.outputSize} * p.inputSize + 2047) / 2048, ids.size(), 1});
    }
    initialization = backend.submitCommand(graph.dispatches());
    uint32_t status = 0; std::memcpy(&status, diagnostics.contents(), 4);
    if (status || !guards()) throw std::runtime_error("hot expert coefficient conversion failed");
    const uint64_t after = backend.memoryStats().allocatedBytes;
    if (after < before) throw std::logic_error("expert cache allocation ledger regressed");
    allocatedBytes = after - before;
    if (allocatedBytes > FlashExpertDenseCache::plannedBytes(w, prefix, ids))
      throw std::runtime_error("expert cache exceeded planned allocation");
  }
  bool guards() const noexcept {
    for (uint32_t i = 0; i < 3; ++i) {
      const auto *base = static_cast<const uint8_t *>(allocations[i].contents());
      if (!base || !std::all_of(base + cached[i].logicalBytes,
          base + cached[i].logicalBytes + kGuardBytes, [](uint8_t v) { return v == kGuard; })) return false;
    }
    return true;
  }
  void cacheDisjoint(const metal::MetalBuffer &output) const {
    for (const auto &plane : cached) disjoint(output, plane.buffer);
    for (const auto &plane : sourceTensors) disjoint(output, plane.buffer);
    disjoint(output, rankMap); disjoint(output, selectedIDs);
  }
};

FlashExpertDenseCache::FlashExpertDenseCache(metal::MetalBackend &b, const FlashWeights &w,
    std::string prefix, std::span<const uint32_t> ids) : impl_(std::make_unique<Impl>(b, w, std::move(prefix), ids)) {}
FlashExpertDenseCache::~FlashExpertDenseCache() = default;
FlashExpertDenseCache::FlashExpertDenseCache(FlashExpertDenseCache &&) noexcept = default;
FlashExpertDenseCache &FlashExpertDenseCache::operator=(FlashExpertDenseCache &&) noexcept = default;
uint64_t FlashExpertDenseCache::plannedBytes(const FlashWeights &w, std::string_view prefix,
                                           std::span<const uint32_t> ids) {
  const auto selection = selected(ids);
  uint64_t total = 3 * kAlignment;
  for (uint32_t i = 0; i < 3; ++i) {
    const auto &p = w.projection(std::string(prefix) + (i == 0 ? ".gate_proj" : i == 1 ? ".up_proj" : ".down_proj"));
    projection(p, i == 2);
    total = plus(total, rounded(plus(mul(mul(mul(selection.size(), p.outputSize), p.inputSize), 2), kGuardBytes)));
  }
  return total;
}
const std::string &FlashExpertDenseCache::identitySha256() const { return impl_->identity; }
uint64_t FlashExpertDenseCache::actualAllocatedBytes() const noexcept { return impl_ ? impl_->allocatedBytes : 0; }
metal::CommandTiming FlashExpertDenseCache::initializationTiming() const noexcept { return impl_ ? impl_->initialization : metal::CommandTiming{}; }
std::span<const uint32_t> FlashExpertDenseCache::selectedExpertIDs() const { return impl_->ids; }
const metal::MetalBuffer &FlashExpertDenseCache::expertRanks() const { return impl_->rankMap; }
const FlashTensor &FlashExpertDenseCache::cachedProjection(FlashExpertCachePlane plane) const {
  const uint32_t index = static_cast<uint32_t>(plane);
  if (index > 2) throw std::invalid_argument("invalid expert cache plane");
  return impl_->cached[index];
}
std::vector<metal::MetalBuffer> FlashExpertDenseCache::immutableWeightBuffers() const {
  return {impl_->cached[0].buffer, impl_->cached[1].buffer, impl_->cached[2].buffer, impl_->rankMap, impl_->selectedIDs};
}
bool FlashExpertDenseCache::canariesIntact() const noexcept { return impl_ && impl_->guards(); }

void FlashExpertDenseCache::addGateUp(metal::CommandGraph &graph, const FlashMoEBlockedScratch &s,
    metal::MetalBuffer diagnostics, uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) const {
  metal::CommandGraph validated;
  addMoEBlockedGateUp(validated, impl_->source[0], impl_->source[1], s, diagnostics, rows, tile, selections);
  const auto &d = validated.dispatches()[0];
  FlashExpertCacheGateParams params{};
  std::memcpy(&params.blocked, d.bytes[0].data, sizeof(params.blocked));
  params.cached_experts = uint32_t(impl_->ids.size());
  impl_->cacheDisjoint(s.packedActivated); impl_->cacheDisjoint(diagnostics);
  bytes(s.buckets.packedInputs, uint64_t{rows} * selections * 2560 * 2);
  disjoint(s.packedActivated, s.buckets.packedInputs);
  for (const auto &buffer : {s.buckets.offsets, s.buckets.tileJobs, s.buckets.jobCount})
    disjoint(s.packedActivated, buffer);
  disjoint(s.packedActivated, diagnostics);
  graph.add(name("gate_up_hit", tile), {s.buckets.packedInputs, impl_->cached[0].buffer,
      impl_->cached[1].buffer, impl_->rankMap, s.buckets.offsets, s.buckets.tileJobs,
      s.buckets.jobCount, s.packedActivated, diagnostics}, params, {10, params.blocked.job_capacity, 1}, {128, 1, 1});
  auto missBuffers = std::vector<metal::MetalBuffer>{s.buckets.packedInputs, impl_->source[0].weights->buffer,
      impl_->source[0].scales->buffer, impl_->source[0].biases->buffer, impl_->source[1].weights->buffer,
      impl_->source[1].scales->buffer, impl_->source[1].biases->buffer, s.buckets.offsets,
      s.buckets.tileJobs, s.buckets.jobCount, s.packedActivated, diagnostics, impl_->rankMap};
  graph.add(name("gate_up_miss", tile), std::move(missBuffers), params,
      {10, params.blocked.job_capacity, 1}, {128, 1, 1});
}
void FlashExpertDenseCache::addDownScatter(metal::CommandGraph &graph, const FlashMoEBlockedScratch &s,
    metal::MetalBuffer diagnostics, uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) const {
  metal::CommandGraph validated;
  addMoEBlockedDownScatter(validated, impl_->source[2], s, diagnostics, rows, tile, selections);
  const auto &d = validated.dispatches()[1];
  FlashExpertCacheDownParams params{};
  std::memcpy(&params.blocked, d.bytes[0].data, sizeof(params.blocked));
  params.cached_experts = uint32_t(impl_->ids.size());
  impl_->cacheDisjoint(s.scatteredDown); impl_->cacheDisjoint(diagnostics);
  bytes(s.packedActivated, uint64_t{rows} * selections * 640 * 2);
  disjoint(s.scatteredDown, s.packedActivated);
  for (const auto &buffer : {s.buckets.offsets, s.buckets.tileJobs, s.buckets.jobCount,
                            s.buckets.routeMap, s.buckets.canonicalToPacked})
    disjoint(s.scatteredDown, buffer);
  disjoint(s.scatteredDown, diagnostics);
  // Preserve the original excluded-route poison producer, copying its params
  // into this graph rather than retaining a pointer into the validator.
  graph.add("flash_moe_blocked_poison_excluded_routes", {s.buckets.canonicalToPacked,
      s.scatteredDown, diagnostics}, params.blocked, {10, params.blocked.route_capacity, 1});
  graph.add(name("down_hit", tile), {s.packedActivated, impl_->cached[2].buffer, impl_->rankMap,
      s.buckets.offsets, s.buckets.tileJobs, s.buckets.jobCount, s.buckets.routeMap,
      s.scatteredDown, diagnostics}, params, {40, params.blocked.job_capacity, 1}, {128, 1, 1});
  graph.add(name("down_miss", tile), {s.packedActivated, impl_->source[2].weights->buffer,
      impl_->source[2].scales->buffer, impl_->source[2].biases->buffer, s.buckets.offsets,
      s.buckets.tileJobs, s.buckets.jobCount, s.buckets.routeMap, s.scatteredDown,
      diagnostics, impl_->rankMap}, params, {40, params.blocked.job_capacity, 1}, {128, 1, 1});
}
} // namespace splash::flash
