#include "FlashInt8ExpertStore.hpp"
#include "FlashInt8ExpertStoreMetadata.hpp"
#include "metal/abi/FlashInt8ExpertStore.h"
#include "metal/abi/FlashMoEDirectA.h"

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace splash::flash {
namespace {
constexpr uint64_t kAlignment = 16384;
[[noreturn]] void fail(const char *reason) { throw std::invalid_argument(reason); }
std::string hash(const void *pointer, uint64_t count) {
  CC_SHA256_CTX context{};
  if (!CC_SHA256_Init(&context)) fail("INT8 expert SHA256 initialization failed");
  const auto *bytes = static_cast<const uint8_t *>(pointer);
  while (count) {
    const CC_LONG step = static_cast<CC_LONG>(std::min<uint64_t>(count, 1ULL << 30));
    if (!CC_SHA256_Update(&context, bytes, step)) fail("INT8 expert SHA256 update failed");
    bytes += step; count -= step;
  }
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  if (!CC_SHA256_Final(digest.data(), &context)) fail("INT8 expert SHA256 finalization failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (uint8_t byte : digest) { result += hex[byte >> 4]; result += hex[byte & 15]; }
  return result;
}
class Mapping final {
public:
  Mapping(const std::filesystem::path &path, uint64_t expected) : bytes_(expected) {
    const int file = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (file < 0) fail("cannot open readonly INT8 expert payload");
    struct stat before{};
    if (::fstat(file, &before) || !S_ISREG(before.st_mode) || before.st_size <= 0 ||
        uint64_t(before.st_size) != expected || expected % kAlignment ||
        expected > std::numeric_limits<size_t>::max()) {
      ::close(file); fail("INT8 expert payload size/alignment changed");
    }
    address_ = ::mmap(nullptr, expected, PROT_READ, MAP_SHARED, file, 0);
    const int saved = errno;
    ::close(file);
    if (address_ == MAP_FAILED) { address_ = nullptr; errno = saved; fail("INT8 expert readonly mapping failed"); }
  }
  ~Mapping() { if (address_) ::munmap(address_, bytes_); }
  void *address() const noexcept { return address_; }
private:
  void *address_ = nullptr;
  uint64_t bytes_;
};
void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes) {
  if (!buffer || buffer.storage() != metal::BufferStorage::Shared ||
      !buffer.contents() || !bytes || buffer.sizeBytes() < bytes)
    fail("INT8 expert store requires sufficient Shared buffer views");
}
void disjoint(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  const auto aa = reinterpret_cast<uintptr_t>(a.contents());
  const auto bb = reinterpret_cast<uintptr_t>(b.contents());
  if (!aa || !bb) fail("INT8 expert store requires addressable buffer views");
  if (aa <= bb ? uint64_t(bb - aa) < a.sizeBytes() : uint64_t(aa - bb) < b.sizeBytes())
    fail("INT8 expert writable output overlaps an input or immutable operand");
}
void sourceGeometry(const FlashWeights &weights) {
  for (uint32_t layer = 0; layer < 48; ++layer) {
    const std::string prefix = "language_model.model.layers." + std::to_string(layer) + ".mlp.switch_mlp";
    for (uint32_t plane = 0; plane < 3; ++plane) {
      const auto &p = weights.projection(prefix + (plane == 0 ? ".gate_proj" : plane == 1 ? ".up_proj" : ".down_proj"));
      const uint32_t n = plane == 2 ? 2560 : 640, k = plane == 2 ? 640 : 2560;
      if (p.experts != 512 || p.outputSize != n || p.inputSize != k || p.bits != 4 || p.groupSize != 64 ||
          p.weightRowStrideBytes != k / 2 || p.weightExpertStrideBytes != uint64_t{n} * k / 2 ||
          p.parameterRowStrideBytes != k / 32 || p.parameterExpertStrideBytes != uint64_t{n} * k / 32 ||
          !p.weights || !p.scales || !p.biases || p.weights->dtype != FlashDType::U32 ||
          p.scales->dtype != FlashDType::BF16 || p.biases->dtype != FlashDType::BF16 ||
          p.weights->logicalBytes != uint64_t{512} * n * k / 2 ||
          p.scales->logicalBytes != uint64_t{512} * n * k / 32 ||
          p.biases->logicalBytes != uint64_t{512} * n * k / 32)
        fail("saved INT8 experts require original contiguous E512 Q4/G64 H2560 I640 source planes");
    }
  }
}
std::string pipeline(const char *phase, FlashMoEBlockedTile tile) {
  const uint32_t m = uint32_t(tile);
  if (m != 16 && m != 32 && m != 64) fail("saved INT8 expert tile must be M16/M32/M64");
  return std::string("flash_int8_expert_store_") + phase + "_m" + std::to_string(m) +
      (m == 64 ? "_n64_sg8" : "_n64");
}
bool gateProducer(const metal::ComputeDispatch &d) { return d.pipelineName.find("gate_up_") != std::string::npos; }
bool downProducer(const metal::ComputeDispatch &d) { return d.pipelineName.find("down_scatter_") != std::string::npos; }
bool directProducer(const metal::ComputeDispatch &d) { return d.pipelineName.starts_with("flash_moe_direct_a_"); }
void append(metal::CommandGraph &graph, const metal::ComputeDispatch &d) {
  // Copy an already validated dispatch's inline bytes into the receiving graph;
  // no pointer to the temporary validation graph is retained.
  std::vector<metal::MetalBuffer> buffers;
  for (const auto &binding : d.buffers) {
    if (binding.index != buffers.size()) fail("INT8 expert validator bindings are not contiguous");
    buffers.push_back(binding.buffer);
  }
  if (d.bytes.size() != 1) fail("INT8 expert validator expected one parameter binding");
  if (d.pipelineName == "flash_moe_blocked_poison_excluded_routes") {
    FlashMoEBlockedDownParams params{};
    if (d.bytes[0].sizeBytes != sizeof(params)) fail("INT8 expert poison parameter extent changed");
    std::memcpy(&params, d.bytes[0].data, sizeof(params));
    graph.add(d.pipelineName, std::move(buffers), params, d.threadgroups, d.threadsPerThreadgroup);
  } else if (d.pipelineName == "flash_moe_direct_a_prepare_down") {
    FlashMoEDirectAPrepareParams params{};
    if (d.bytes[0].sizeBytes != sizeof(params)) fail("INT8 expert direct-A parameter extent changed");
    std::memcpy(&params, d.bytes[0].data, sizeof(params));
    graph.add(d.pipelineName, std::move(buffers), params, d.threadgroups, d.threadsPerThreadgroup);
  } else fail("unexpected preparatory dispatch in INT8 expert validator");
}
} // namespace

struct FlashInt8ExpertStore::Impl final {
  struct Layer final {
    metal::MetalBuffer base, ranks;
    std::array<metal::MetalBuffer, 3> codes, scales;
    std::array<FlashTensor, 9> sourceTensors;
    std::array<FlashAffineProjection, 3> source;
  };
  metal::MetalBackend &backend;
  FlashInt8ExpertStoreMetadata metadata;
  std::array<Layer, 48> layers;
  uint64_t allocated = 0;

  Impl(metal::MetalBackend &b, const FlashWeights &weights, const std::filesystem::path &directory)
      : backend(b), metadata(loadFlashInt8ExpertStoreMetadata(directory, weights.sourceIdentity(),
          weights.manifestFingerprint(), weights.normConvention())) {
    sourceGeometry(weights);
    const uint64_t before = backend.memoryStats().allocatedBytes;
    for (uint32_t index = 0; index < 48; ++index) {
      auto &layer = layers[index];
      const auto &entry = metadata.layers[index];
      if (entry.bytes > backend.capabilities().maxBufferLengthBytes) fail("INT8 expert layer exceeds Metal buffer limit");
      auto mapping = std::make_shared<Mapping>(entry.path, entry.bytes);
      const auto *data = static_cast<const uint8_t *>(mapping->address());
      if (hash(data, entry.bytes) != entry.sha256) fail("INT8 expert payload checksum differs");
      uint64_t cursor = 0;
      for (uint32_t plane = 0; plane < 3; ++plane) {
        for (const auto &range : {entry.codes[plane], entry.scales[plane]}) {
          if (!std::all_of(data + cursor, data + range.offset, [](uint8_t v) { return v == 0; }))
            fail("INT8 expert plane alignment padding is nonzero");
          if (hash(data + range.offset, range.length) != range.sha256)
            fail("INT8 expert plane checksum differs");
          cursor = range.offset + range.length;
        }
        const auto &code = entry.codes[plane];
        if (std::find(data + code.offset, data + code.offset + code.length, uint8_t{128}) !=
            data + code.offset + code.length) fail("saved symmetric INT8 contains excluded -128 code");
        const auto &scale = entry.scales[plane];
        const auto *values = reinterpret_cast<const float *>(data + scale.offset);
        if (!std::all_of(values, values + scale.length / sizeof(float),
            [](float v) { return std::isfinite(v) && v > 0.0f; }))
          fail("saved INT8 expert scales must be finite and positive");
      }
      if (!std::all_of(data + cursor, data + entry.bytes, [](uint8_t v) { return v == 0; }))
        fail("INT8 expert file tail padding is nonzero");
      layer.base = backend.wrapSharedMemory(mapping->address(), entry.bytes, mapping, "saved selected signed INT8 expert layer");
      layer.ranks = backend.allocateBuffer(kAlignment, metal::BufferStorage::Shared, "saved INT8 expert ID-to-rank map");
      std::memset(layer.ranks.contents(), 0xff, kAlignment);
      auto *ranks = static_cast<uint32_t *>(layer.ranks.contents());
      for (uint32_t rank = 0; rank < entry.selectedIDs.size(); ++rank) ranks[entry.selectedIDs[rank]] = rank;
      const auto prefix = "language_model.model.layers." + std::to_string(index) + ".mlp.switch_mlp";
      for (uint32_t plane = 0; plane < 3; ++plane) {
        layer.codes[plane] = backend.view(layer.base, entry.codes[plane].offset, entry.codes[plane].length);
        layer.scales[plane] = backend.view(layer.base, entry.scales[plane].offset, entry.scales[plane].length);
        layer.source[plane] = weights.projection(prefix + (plane == 0 ? ".gate_proj" : plane == 1 ? ".up_proj" : ".down_proj"));
        layer.sourceTensors[plane * 3] = *layer.source[plane].weights;
        layer.sourceTensors[plane * 3 + 1] = *layer.source[plane].scales;
        layer.sourceTensors[plane * 3 + 2] = *layer.source[plane].biases;
        layer.source[plane].weights = &layer.sourceTensors[plane * 3];
        layer.source[plane].scales = &layer.sourceTensors[plane * 3 + 1];
        layer.source[plane].biases = &layer.sourceTensors[plane * 3 + 2];
      }
    }
    const uint64_t after = backend.memoryStats().allocatedBytes;
    if (after < before || after - before > metadata.plannedBytes) fail("saved INT8 store exceeds reserved bytes");
    allocated = after - before;
  }
  const Layer &layer(uint32_t index) const {
    if (index >= 48) fail("saved INT8 expert layer is outside target layers");
    return layers[index];
  }
  void immutableDisjoint(const metal::MetalBuffer &output) const {
    for (const auto &layer : layers) {
      disjoint(output, layer.base); disjoint(output, layer.ranks);
      for (const auto &operand : layer.sourceTensors) disjoint(output, operand.buffer);
    }
  }
};

FlashInt8ExpertStore::FlashInt8ExpertStore(metal::MetalBackend &b, const FlashWeights &w,
    const std::filesystem::path &directory) : impl_(std::make_unique<Impl>(b, w, directory)) {}
FlashInt8ExpertStore::~FlashInt8ExpertStore() = default;
FlashInt8ExpertStore::FlashInt8ExpertStore(FlashInt8ExpertStore &&) noexcept = default;
FlashInt8ExpertStore &FlashInt8ExpertStore::operator=(FlashInt8ExpertStore &&) noexcept = default;
uint64_t FlashInt8ExpertStore::plannedBytes(const FlashWeights &weights, const std::filesystem::path &directory) {
  sourceGeometry(weights);
  return loadFlashInt8ExpertStoreMetadata(directory, weights.sourceIdentity(),
      weights.manifestFingerprint(), weights.normConvention()).plannedBytes;
}
const std::string &FlashInt8ExpertStore::identitySha256() const { return impl_->metadata.identitySha256; }
const std::string &FlashInt8ExpertStore::planSha256() const { return impl_->metadata.planSha256; }
uint64_t FlashInt8ExpertStore::mappedBytes() const noexcept { return impl_ ? impl_->metadata.totalBytes : 0; }
uint64_t FlashInt8ExpertStore::actualAllocatedBytes() const noexcept { return impl_ ? impl_->allocated : 0; }
std::span<const uint32_t> FlashInt8ExpertStore::selectedExpertIDs(uint32_t index) const {
  (void)impl_->layer(index); return impl_->metadata.layers[index].selectedIDs;
}
std::vector<metal::MetalBuffer> FlashInt8ExpertStore::immutableWeightBuffers() const {
  std::vector<metal::MetalBuffer> result;
  for (const auto &layer : impl_->layers) { result.push_back(layer.base); result.push_back(layer.ranks); }
  return result;
}

void FlashInt8ExpertStore::addGateUp(metal::CommandGraph &graph, uint32_t index,
    const FlashMoEBlockedScratch &s, metal::MetalBuffer diagnostics,
    uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) const {
  const auto &layer = impl_->layer(index);
  const auto hitPipeline = pipeline("gate_up", tile);
  (void)pipeline("gate_up_miss", tile);
  metal::CommandGraph validated;
  addMoEBlockedGateUp(validated, layer.source[0], layer.source[1], s, diagnostics, rows, tile, selections);
  const auto dispatches = validated.dispatches();
  const auto producer = std::find_if(dispatches.begin(), dispatches.end(), gateProducer);
  if (producer == dispatches.end() || producer->bytes.size() != 1) fail("saved INT8 gate validator has no unique producer");
  FlashInt8ExpertStoreGateParams miss{};
  if (producer->bytes[0].sizeBytes != sizeof(miss.blocked)) fail("saved INT8 gate parameter extent changed");
  std::memcpy(&miss.blocked, producer->bytes[0].data, sizeof(miss.blocked));
  miss.stored_experts = uint32_t(impl_->metadata.layers[index].selectedIDs.size());
  miss.flags = directProducer(*producer);
  impl_->immutableDisjoint(s.packedActivated); impl_->immutableDisjoint(s.buckets.packedInputs);
  impl_->immutableDisjoint(diagnostics);
  requireBytes(s.buckets.packedInputs, uint64_t{rows} * selections * 2560 * 2);
  disjoint(s.packedActivated, s.buckets.packedInputs); disjoint(s.packedActivated, diagnostics);
  for (const auto &b : {s.buckets.offsets, s.buckets.tileJobs, s.buckets.jobCount}) disjoint(s.packedActivated, b);
  for (const auto &b : {s.buckets.offsets, s.buckets.tileJobs, s.buckets.jobCount, s.buckets.counts,
      s.buckets.jobOffsets, s.buckets.routeMap, s.buckets.canonicalToPacked, diagnostics})
    disjoint(s.buckets.packedInputs, b);
  for (auto it = dispatches.begin(); it != producer; ++it) append(graph, *it);
  if (!miss.flags) {
    const FlashInt8ExpertStoreSanitizeParams params{rows * selections, 2560, 0, 0};
    graph.add("flash_int8_expert_store_sanitize", {s.buckets.packedInputs, s.buckets.offsets, diagnostics},
        params, {(uint64_t{rows} * selections * 2560 + 255) / 256, 1, 1});
  }
  const FlashInt8ExpertStoreParams hit{rows, selections, rows * selections,
      miss.blocked.job_capacity, uint32_t(tile), miss.stored_experts, 0, 0};
  const uint32_t threads = tile == FlashMoEBlockedTile::M64N64 ? 256 : 128;
  graph.add(hitPipeline, {s.buckets.packedInputs, layer.codes[0], layer.scales[0],
      layer.codes[1], layer.scales[1], layer.ranks, s.buckets.offsets, s.buckets.tileJobs,
      s.buckets.jobCount, s.packedActivated, diagnostics}, hit, {10, miss.blocked.job_capacity, 1}, {threads, 1, 1});
  graph.add(pipeline(miss.flags ? "gate_up_miss_direct" : "gate_up_miss_staged", tile), {s.buckets.packedInputs, layer.source[0].weights->buffer,
      layer.source[0].scales->buffer, layer.source[0].biases->buffer, layer.source[1].weights->buffer,
      layer.source[1].scales->buffer, layer.source[1].biases->buffer, s.buckets.offsets,
      s.buckets.tileJobs, s.buckets.jobCount, s.packedActivated, diagnostics, layer.ranks},
      miss, {10, miss.blocked.job_capacity, 1}, {threads, 1, 1});
}

void FlashInt8ExpertStore::addDownScatter(metal::CommandGraph &graph, uint32_t index,
    const FlashMoEBlockedScratch &s, metal::MetalBuffer diagnostics,
    uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) const {
  const auto &layer = impl_->layer(index);
  const auto hitPipeline = pipeline("down_scatter", tile);
  (void)pipeline("down_miss", tile);
  metal::CommandGraph validated;
  addMoEBlockedDownScatter(validated, layer.source[2], s, diagnostics, rows, tile, selections);
  const auto dispatches = validated.dispatches();
  const auto producer = std::find_if(dispatches.begin(), dispatches.end(), downProducer);
  if (producer == dispatches.end() || producer->bytes.size() != 1) fail("saved INT8 down validator has no unique producer");
  FlashInt8ExpertStoreDownParams miss{};
  if (producer->bytes[0].sizeBytes != sizeof(miss.blocked)) fail("saved INT8 down parameter extent changed");
  std::memcpy(&miss.blocked, producer->bytes[0].data, sizeof(miss.blocked));
  miss.stored_experts = uint32_t(impl_->metadata.layers[index].selectedIDs.size());
  miss.flags = directProducer(*producer);
  impl_->immutableDisjoint(s.scatteredDown); impl_->immutableDisjoint(s.packedActivated);
  impl_->immutableDisjoint(diagnostics);
  requireBytes(s.packedActivated, uint64_t{rows} * selections * 640 * 2);
  disjoint(s.scatteredDown, s.packedActivated); disjoint(s.scatteredDown, diagnostics);
  for (const auto &b : {s.buckets.offsets, s.buckets.tileJobs, s.buckets.jobCount,
      s.buckets.routeMap, s.buckets.canonicalToPacked}) disjoint(s.scatteredDown, b);
  for (const auto &b : {s.buckets.offsets, s.buckets.tileJobs, s.buckets.jobCount, s.buckets.counts,
      s.buckets.jobOffsets, s.buckets.routeMap, s.buckets.canonicalToPacked, diagnostics})
    disjoint(s.packedActivated, b);
  for (auto it = dispatches.begin(); it != producer; ++it) append(graph, *it);
  if (!miss.flags) {
    const FlashInt8ExpertStoreSanitizeParams params{rows * selections, 640, 0, 0};
    graph.add("flash_int8_expert_store_sanitize", {s.packedActivated, s.buckets.offsets, diagnostics},
        params, {(uint64_t{rows} * selections * 640 + 255) / 256, 1, 1});
  }
  const FlashInt8ExpertStoreParams hit{rows, selections, rows * selections,
      miss.blocked.job_capacity, uint32_t(tile), miss.stored_experts, 0, 0};
  const uint32_t threads = tile == FlashMoEBlockedTile::M64N64 ? 256 : 128;
  graph.add(hitPipeline, {s.packedActivated, layer.codes[2], layer.scales[2], layer.ranks,
      s.buckets.offsets, s.buckets.tileJobs, s.buckets.jobCount, s.buckets.routeMap,
      s.scatteredDown, diagnostics}, hit, {40, miss.blocked.job_capacity, 1}, {threads, 1, 1});
  graph.add(pipeline(miss.flags ? "down_miss_direct" : "down_miss_staged", tile), {s.packedActivated, layer.source[2].weights->buffer,
      layer.source[2].scales->buffer, layer.source[2].biases->buffer, s.buckets.offsets,
      s.buckets.tileJobs, s.buckets.jobCount, s.buckets.routeMap, s.scatteredDown,
      diagnostics, layer.ranks}, miss, {40, miss.blocked.job_capacity, 1}, {threads, 1, 1});
}
} // namespace splash::flash
