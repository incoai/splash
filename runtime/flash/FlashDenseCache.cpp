#include "FlashDenseCache.hpp"
#include "FlashOperandStore.hpp"
#include "FlashDenseTraversal.hpp"
#include "FlashPrefillDenseTiles.hpp"

#include "metal/abi/FlashDenseCache.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace splash::flash {
bool flashPrefillDenseTilesEnabled() {
  static const bool enabled = parseFlashPrefillDenseTilesFlag(std::getenv("SPLASH_FLASH_PREFILL_DENSE_TILES"));
  return enabled;
}
bool flashDenseTraversalEnabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_FLASH_DENSE_TRAVERSAL");
    if (!value || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument("SPLASH_FLASH_DENSE_TRAVERSAL must be 0 or 1");
  }();
  return enabled;
}
bool flashDenseM64OutEnabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_FLASH_DENSE_M64_OUT");
    if (!value || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument("SPLASH_FLASH_DENSE_M64_OUT must be 0 or 1");
  }();
  return enabled;
}
bool flashDenseM64OutGeometry(std::string_view prefix, uint32_t rows,
                              uint32_t outputSize, uint32_t inputSize) noexcept {
  if (rows < 512 || rows > 8192 || outputSize != 2560 || inputSize != 6144) return false;
  constexpr std::string_view layerPrefix = "language_model.model.layers.";
  if (!prefix.starts_with(layerPrefix)) return false;
  const auto tail = prefix.substr(layerPrefix.size());
  const auto split = tail.find('.');
  if (split == std::string_view::npos || !split || split > 2) return false;
  uint32_t layer = 0;
  for (char digit : tail.substr(0, split)) {
    if (digit < '0' || digit > '9') return false;
    layer = layer * 10 + uint32_t(digit - '0');
  }
  if (layer >= 48) return false;
  const auto role = tail.substr(split + 1);
  return role == "linear_attn.out_proj" || role == "self_attn.o_proj";
}
const char *flashDenseCacheExecutionSemantics() {
  if (!flashDenseM64OutEnabled() && !flashDenseTraversalEnabled() && !flashPrefillDenseTilesEnabled())
    return kFlashDenseCacheExecutionSemantics;
  static const std::string value = [] {
    std::string result = kFlashDenseCacheExecutionSemantics;
    if (flashDenseM64OutEnabled()) result += kFlashDenseM64OutSemantics;
    if (flashDenseTraversalEnabled()) result += kFlashDenseTraversalSemantics;
    if (flashPrefillDenseTilesEnabled()) result += kFlashPrefillDenseTilesSemantics;
    return result;
  }();
  return value.c_str();
}
namespace {
constexpr uint64_t kAlignment = 16384;

uint64_t product(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::invalid_argument("Flash dense cache byte extent overflows");
  return a * b;
}
uint64_t plus(uint64_t a, uint64_t b) {
  if (a > std::numeric_limits<uint64_t>::max() - b)
    throw std::invalid_argument("Flash dense cache byte extent overflows");
  return a + b;
}
uint64_t rounded(uint64_t bytes) {
  return plus(bytes, kAlignment - 1) & ~(kAlignment - 1);
}
uint64_t extent(uint32_t n, uint64_t stride, uint64_t rowBytes) {
  if (!n || stride < rowBytes)
    throw std::invalid_argument("Flash dense cache invalid source row stride");
  return plus(product(n - 1, stride), rowBytes);
}
void requireBuffer(const metal::MetalBuffer &b, uint64_t bytes,
                   const char *what) {
  if (!b || !bytes || b.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash dense cache insufficient ") + what);
}
bool overlaps(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  if (a.sameView(b)) return true;
  const auto first = reinterpret_cast<uintptr_t>(a.contents());
  const auto second = reinterpret_cast<uintptr_t>(b.contents());
  if (!first || !second)
    throw std::invalid_argument("Flash dense cache projection requires Shared buffers");
  if (first > std::numeric_limits<uintptr_t>::max() - a.sizeBytes() ||
      second > std::numeric_limits<uintptr_t>::max() - b.sizeBytes())
    throw std::invalid_argument("Flash dense cache projection address extent overflows");
  return first < second + b.sizeBytes() && second < first + a.sizeBytes();
}
void requireTensor(const FlashTensor *t, FlashDType dtype, uint64_t bytes) {
  if (!t || t->dtype != dtype || t->logicalBytes < bytes)
    throw std::invalid_argument("Flash dense cache invalid source tensor");
  requireBuffer(t->buffer, bytes, "source tensor");
}
void requireProjection(std::string_view prefix, const FlashAffineProjection &p) {
  if (prefix.empty() || prefix.find("embed_tokens") != std::string_view::npos ||
      prefix.find("ngram_embedding") != std::string_view::npos ||
      p.experts != 1 || !p.outputSize || p.outputSize % 64 || !p.inputSize ||
      p.inputSize > 32768 ||
      (p.bits != 4 && p.bits != 5 && p.bits != 6 && p.bits != 8) ||
      (p.groupSize != 32 && p.groupSize != 64 && p.groupSize != 128) ||
      p.inputSize % p.groupSize || p.parameterRowStrideBytes % 2)
    throw std::invalid_argument("Flash dense cache requires a checked dense projection");
  requireTensor(p.weights, FlashDType::U32,
      extent(p.outputSize, p.weightRowStrideBytes,
             (product(p.inputSize, p.bits) + 7) / 8));
  const uint64_t coefficientBytes = extent(p.outputSize, p.parameterRowStrideBytes,
                                          product(p.inputSize / p.groupSize, 2));
  requireTensor(p.scales, FlashDType::BF16, coefficientBytes);
  requireTensor(p.biases, FlashDType::BF16, coefficientBytes);
}
std::vector<std::string> normalize(std::span<const std::string> prefixes) {
  if (prefixes.empty())
    throw std::invalid_argument("Flash dense cache projection selection is empty");
  std::vector<std::string> names(prefixes.begin(), prefixes.end());
  std::sort(names.begin(), names.end());
  names.erase(std::unique(names.begin(), names.end()), names.end());
  return names;
}
std::string sha256(const std::string &input) {
  if (input.size() > std::numeric_limits<CC_LONG>::max())
    throw std::invalid_argument("Flash dense cache identity extent exceeds SHA256 input");
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  if (!CC_SHA256(input.data(), static_cast<CC_LONG>(input.size()), digest.data()))
    throw std::runtime_error("Flash dense cache identity hash failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2);
  for (const unsigned char value : digest) {
    result += hex[value >> 4]; result += hex[value & 15];
  }
  return result;
}
struct StringHash final {
  using is_transparent = void;
  size_t operator()(std::string_view name) const noexcept {
    return std::hash<std::string_view>{}(name);
  }
};
std::pair<uint32_t, uint32_t> tileGeometry(FlashAffineMPPTile tile) {
  switch (tile) {
  case FlashAffineMPPTile::M8N64: return {8, 64};
  case FlashAffineMPPTile::M16N64: return {16, 64};
  case FlashAffineMPPTile::M16N128: return {16, 128};
  case FlashAffineMPPTile::M32N64: return {32, 64};
  case FlashAffineMPPTile::M32N128: return {32, 128};
  case FlashAffineMPPTile::M64N64: return {64, 64};
  case FlashAffineMPPTile::M64N128: return {64, 128};
  }
  throw std::invalid_argument("Flash dense cache invalid MPP tile");
}
void addDenseBF16WholeKImpl(metal::MetalBackend &backend, metal::CommandGraph &graph,
                           metal::MetalBuffer input, const FlashTensor &weight,
                           metal::MetalBuffer output, metal::MetalBuffer diagnostics,
                           uint32_t rows, FlashAffineMPPTile tile,
                           FlashPrefillDenseTilePlan prefillPlan);
} // namespace

struct FlashDenseCache::Impl final {
  metal::MetalBackend &backend;
  std::unordered_map<std::string, FlashTensor, StringHash, std::equal_to<>> tensors;
  metal::MetalBuffer initializationDiagnostics;
  std::string identity;
  std::string operandStoreIdentity;
  uint64_t persistedTensors = 0;
  uint64_t persistedBytes = 0;
  std::vector<metal::MetalBuffer> persistedBuffers;
  uint64_t allocatedBytes = 0;
  metal::CommandTiming initialization;

  Impl(metal::MetalBackend &value, const FlashWeights &weights,
       std::span<const std::string> prefixes) : backend(value) {
    const auto names = normalize(prefixes);
    // Validate the entire selection before allocating or submitting work.
    (void)FlashDenseCache::plannedBytes(weights, names);
    const auto store = FlashOperandStore::fromEnvironment(weights);
    if (store) operandStoreIdentity = store->identitySha256();
    const uint64_t before = backend.memoryStats().allocatedBytes;
    initializationDiagnostics = backend.allocateBuffer(kAlignment,
        metal::BufferStorage::Shared, "flash-dense-cache-conversion-diagnostics");
    std::memset(initializationDiagnostics.contents(), 0, initializationDiagnostics.sizeBytes());
    std::string fingerprint = std::string(kFlashDenseCacheOperandFormat) + "\n";
    const auto field = [&](std::string_view key, std::string_view data) {
      fingerprint.append(key); fingerprint += ':';
      fingerprint += std::to_string(data.size()); fingerprint += ':';
      fingerprint.append(data); fingerprint += '\n';
    };
    field("source", weights.sourceIdentity());
    field("manifest", weights.manifestFingerprint());
    metal::CommandGraph graph;
    for (const auto &name : names) {
      const auto &p = weights.projection(name);
      const uint64_t bytes = product(product(p.outputSize, p.inputSize), 2);
      FlashTensor cached;
      const bool saved = store && store->contains(name, FlashOperandFormat::BF16);
      if (saved) {
        cached = store->mapTensor(backend, flashOperandSpec(name, FlashOperandFormat::BF16, p));
        ++persistedTensors;
        persistedBytes = plus(persistedBytes, cached.buffer.sizeBytes());
        persistedBuffers.push_back(cached.buffer);
      }
      else {
        cached.buffer = backend.allocateBuffer(rounded(bytes), metal::BufferStorage::Shared,
                                              "flash-dense-cache:" + name);
        cached.dtype = FlashDType::BF16;
        cached.shape = {p.outputSize, p.inputSize};
        cached.logicalBytes = bytes;
      }
      auto [entry, inserted] = tensors.emplace(name, std::move(cached));
      if (!inserted) throw std::logic_error("Flash dense cache normalized selection has duplicates");
      const FlashAffineParams params{1, 1, p.inputSize, p.outputSize, 1,
          p.bits, p.groupSize, 0, p.weightRowStrideBytes, p.weightExpertStrideBytes,
          p.parameterRowStrideBytes, p.parameterExpertStrideBytes};
      const uint64_t elements = product(p.outputSize, p.inputSize);
      if (!saved) graph.add("flash_dense_cache_expand_bf16",
          {p.weights->buffer, p.scales->buffer, p.biases->buffer,
           entry->second.buffer, initializationDiagnostics}, params,
          {(elements - 1) / 256 + 1, 1, 1});
      field("projection", name);
      field("geometry", std::to_string(p.outputSize) + "," + std::to_string(p.inputSize) +
          "," + std::to_string(p.bits) + "," + std::to_string(p.groupSize) + "," +
          std::to_string(p.weightRowStrideBytes) + "," + std::to_string(p.parameterRowStrideBytes));
    }
    identity = sha256(fingerprint);
    if (!graph.dispatches().empty()) initialization = backend.submitCommand(graph.dispatches());
    uint32_t status = 0;
    std::memcpy(&status, initializationDiagnostics.contents(), sizeof(status));
    if (status)
      throw std::runtime_error("Flash dense cache coefficient conversion diagnostics failed: " +
                               std::to_string(status));
    const uint64_t after = backend.memoryStats().allocatedBytes;
    if (after < before) throw std::logic_error("Flash dense cache allocation ledger regressed");
    allocatedBytes = after - before;
  }
};

FlashDenseCache::FlashDenseCache(metal::MetalBackend &backend, const FlashWeights &weights,
                               std::span<const std::string> prefixes)
    : impl_(std::make_unique<Impl>(backend, weights, prefixes)) {}
FlashDenseCache::~FlashDenseCache() = default;
FlashDenseCache::FlashDenseCache(FlashDenseCache &&) noexcept = default;
FlashDenseCache &FlashDenseCache::operator=(FlashDenseCache &&) noexcept = default;

std::vector<std::string> FlashDenseCache::defaultPrefixes(const FlashWeights &weights,
                                                        bool includeVocabularyHead) {
  std::vector<std::string> result;
  const auto add = [&](const std::string &prefix) {
    if (weights.contains(prefix + ".weight")) {
      requireProjection(prefix, weights.projection(prefix));
      result.push_back(prefix);
    }
  };
  for (uint32_t layer = 0; layer < weights.descriptor().layers; ++layer) {
    const std::string p = "language_model.model.layers." + std::to_string(layer);
    for (const char *kind : {"attn_hyper_connection", "mlp_hyper_connection"}) {
      add(p + "." + kind + ".input_mix_weight_down");
      add(p + "." + kind + ".input_mix_weight_up");
    }
    if (weights.descriptor().layerKinds[layer] == FlashLayerKind::GatedDeltaNet) {
      for (const char *kind : {"in_proj_qkv", "in_proj_z", "out_proj"})
        add(p + ".linear_attn." + kind);
    } else {
      for (const char *kind : {"q_proj", "k_proj", "v_proj", "o_proj", "indexer.index_qk_proj"})
        add(p + ".self_attn." + kind);
    }
    for (const char *kind : {"gate_proj", "up_proj", "down_proj"})
      add(p + ".mlp.shared_expert." + kind);
    add(p + ".ple.key_proj"); add(p + ".ple.value_proj");
  }
  add("language_model.model.hyper_connection_mixer.input_mix_weight_down");
  add("language_model.model.hyper_connection_mixer.input_mix_weight_up");
  if (includeVocabularyHead) add("language_model.lm_head");
  return normalize(result);
}

uint64_t FlashDenseCache::plannedBytes(const FlashWeights &weights,
                                      std::span<const std::string> prefixes) {
  uint64_t bytes = kAlignment;
  for (const auto &name : normalize(prefixes)) {
    const auto &p = weights.projection(name);
    requireProjection(name, p);
    bytes = plus(bytes, rounded(product(product(p.outputSize, p.inputSize), 2)));
  }
  return bytes;
}
bool FlashDenseCache::contains(std::string_view prefix) const noexcept {
  return impl_ && impl_->tensors.contains(prefix);
}
const FlashTensor &FlashDenseCache::tensor(std::string_view prefix) const {
  if (!impl_) throw std::logic_error("Flash dense cache is not initialized");
  const auto found = impl_->tensors.find(prefix);
  if (found == impl_->tensors.end())
    throw std::invalid_argument("Flash dense cache projection was not selected: " + std::string(prefix));
  return found->second;
}
const std::string &FlashDenseCache::identitySha256() const {
  if (!impl_) throw std::logic_error("Flash dense cache is not initialized");
  return impl_->identity;
}
uint64_t FlashDenseCache::actualAllocatedBytes() const noexcept {
  return impl_ ? impl_->allocatedBytes : 0;
}
uint64_t FlashDenseCache::persistedTensorCount() const noexcept {
  return impl_ ? impl_->persistedTensors : 0;
}
uint64_t FlashDenseCache::persistedPayloadBytes() const noexcept {
  return impl_ ? impl_->persistedBytes : 0;
}
const std::string &FlashDenseCache::operandStoreIdentitySha256() const {
  if (!impl_) throw std::logic_error("Flash dense cache is not initialized");
  return impl_->operandStoreIdentity;
}
metal::CommandTiming FlashDenseCache::initializationTiming() const noexcept {
  return impl_ ? impl_->initialization : metal::CommandTiming{};
}
std::vector<metal::MetalBuffer> FlashDenseCache::immutableWeightBuffers() const {
  std::vector<metal::MetalBuffer> result;
  if (impl_) {
    result.reserve(impl_->tensors.size());
    for (const auto &[name, tensor] : impl_->tensors) {
      (void)name; result.push_back(tensor.buffer);
    }
  }
  return result;
}

std::vector<metal::MetalBuffer> FlashDenseCache::persistedWeightBuffers() const {
  if (!impl_) return {};
  if (impl_->persistedBuffers.size() != impl_->persistedTensors ||
      (!impl_->persistedBuffers.empty() && impl_->operandStoreIdentity.empty()))
    throw std::logic_error("Flash BF16 saved operand enumeration is inconsistent");
  return impl_->persistedBuffers;
}

void FlashDenseCache::addProjection(metal::CommandGraph &graph, std::string_view prefix,
                                    metal::MetalBuffer input, metal::MetalBuffer output,
                                    metal::MetalBuffer diagnostics, uint32_t rows,
                                    FlashAffineMPPTile tile) const {
  const auto &weight = tensor(prefix);
  if (flashDenseM64OutEnabled() && weight.shape.size() == 2 &&
      flashDenseM64OutGeometry(prefix, rows, uint32_t(weight.shape[0]), uint32_t(weight.shape[1])))
    tile = FlashAffineMPPTile::M64N128;
  const auto prefillPlan = flashPrefillDenseTilesEnabled() && weight.shape.size() == 2
      ? flashPrefillDenseTilePolicy(prefix,rows,uint32_t(weight.shape[0]),uint32_t(weight.shape[1]))
      : FlashPrefillDenseTilePlan{};
  addDenseBF16WholeKImpl(impl_->backend, graph, input, weight, output, diagnostics, rows, tile,prefillPlan);
}

void addDenseBF16WholeK(metal::MetalBackend &backend, metal::CommandGraph &graph,
                        metal::MetalBuffer input, const FlashTensor &weight,
                        metal::MetalBuffer output, metal::MetalBuffer diagnostics,
                        uint32_t rows, FlashAffineMPPTile tile) {
  addDenseBF16WholeKImpl(backend,graph,input,weight,output,diagnostics,rows,tile,{});
}

namespace {
void addDenseBF16WholeKImpl(metal::MetalBackend &backend, metal::CommandGraph &graph,
                           metal::MetalBuffer input, const FlashTensor &weight,
                           metal::MetalBuffer output, metal::MetalBuffer diagnostics,
                           uint32_t rows, FlashAffineMPPTile tile,
                           FlashPrefillDenseTilePlan prefillPlan) {
  if (weight.dtype != FlashDType::BF16 || weight.shape.size() != 2 ||
      !weight.shape[0] || weight.shape[0] > UINT32_MAX || weight.shape[0] % 64 ||
      !weight.shape[1] || weight.shape[1] > 32768 || weight.shape[1] % 32 ||
      weight.logicalBytes < product(product(weight.shape[0], weight.shape[1]), 2))
    throw std::invalid_argument("Flash whole-K projection requires a checked BF16 matrix");
  requireBuffer(weight.buffer, weight.logicalBytes, "BF16 weights");
  const auto [legacyM, legacyN] = tileGeometry(tile);
  const uint32_t m = prefillPlan ? prefillPlan.tileRows : legacyM;
  const uint32_t n = prefillPlan ? prefillPlan.tileOutputs : legacyN;
  const uint32_t k = static_cast<uint32_t>(weight.shape[1]);
  const uint32_t outputs = static_cast<uint32_t>(weight.shape[0]);
  if (!rows || rows > 8192)
    throw std::invalid_argument("Flash dense cache projection rows must be 1..8192");
  requireBuffer(input, product(product(rows, k), 2), "projection input");
  requireBuffer(output, product(product(rows, outputs), 2), "projection output");
  requireBuffer(diagnostics, sizeof(uint32_t), "projection diagnostics");
  for (const auto &b : {input, output, diagnostics})
    if (overlaps(b, weight.buffer))
      throw std::invalid_argument("Flash dense cache projection aliases immutable weights");
  if (overlaps(input, output) || overlaps(diagnostics, input) || overlaps(diagnostics, output))
    throw std::invalid_argument("Flash dense cache projection buffer overlap");
  const uint32_t fullRows = rows / m * m;
  const auto traversal = prefillPlan ? prefillPlan.traversal : flashDenseTraversalEnabled()
      ? flashDenseTraversalPolicy(rows, outputs, k, m, n)
      : FlashDenseTraversal::ColumnFast;
  const auto dispatch = [&](uint32_t begin, uint32_t count, uint32_t tileN) {
    if (!fullRows || !count) return;
    const auto grid = flashDenseTraversalGrid(fullRows / m, count / tileN, traversal);
    const FlashDenseCacheParams params{fullRows, k, outputs, begin, count, m, tileN,
                                      static_cast<uint32_t>(traversal)};
    const std::string suffix = traversal == FlashDenseTraversal::ColumnFast ? "" : "_traversal";
    const std::string pipeline = prefillPlan
        ? "flash_dense_cache_prefill_m128_n64_sg" + std::to_string(prefillPlan.simdGroups)
        : "flash_dense_cache_m" + std::to_string(m) + "_n" + std::to_string(tileN) + suffix;
    graph.add(pipeline,
        {input, weight.buffer, output, diagnostics}, params,
        {grid.x, grid.y, 1}, {prefillPlan ? prefillPlan.simdGroups*32 : m == 64 ? 256u : 128u, 1, 1});
  };
  const uint32_t fullColumns = outputs / n * n;
  dispatch(0, fullColumns, n);
  if (fullColumns < outputs) dispatch(fullColumns, outputs - fullColumns, 64);
  if (fullRows < rows) {
    auto tailInput = backend.view(input, product(product(fullRows, k), 2),
                                        product(product(rows - fullRows, k), 2));
    auto tailOutput = backend.view(output, product(product(fullRows, outputs), 2),
                                         product(product(rows - fullRows, outputs), 2));
    addDenseBF16(graph, tailInput, weight, tailOutput, diagnostics, rows - fullRows);
  }
}
} // namespace

bool FlashDenseCache::supportsHCUpMix(std::string_view prefix, uint32_t rows,
                                     FlashAffineMPPTile tile) const noexcept {
  if (tile == FlashAffineMPPTile::M64N64 || tile == FlashAffineMPPTile::M64N128) return false;
  if (!impl_ || rows < 32 || rows > 8192) return false;
  const auto found = impl_->tensors.find(prefix);
  if (found == impl_->tensors.end() || found->second.shape.size() != 2 ||
      found->second.shape[0] != 10240 || found->second.shape[1] != 320)
    return false;
  uint32_t m = 0;
  switch (tile) {
  case FlashAffineMPPTile::M8N64: m = 8; break;
  case FlashAffineMPPTile::M16N64:
  case FlashAffineMPPTile::M16N128: m = 16; break;
  case FlashAffineMPPTile::M32N64:
  case FlashAffineMPPTile::M32N128: m = 32; break;
  default: return false;
  }
  return rows % m == 0;
}

void FlashDenseCache::addHCUpMix(metal::CommandGraph &graph, std::string_view prefix,
                                metal::MetalBuffer activatedDown,
                                metal::MetalBuffer normalizedHyper,
                                metal::MetalBuffer mixed, metal::MetalBuffer diagnostics,
                                uint32_t rows, FlashAffineMPPTile tile) const {
  if (!supportsHCUpMix(prefix, rows, tile))
    throw std::invalid_argument("Flash cached HC up/mix requires complete inspected row tiles");
  const auto &weight = tensor(prefix);
  const auto [m, n] = tileGeometry(tile);
  requireBuffer(activatedDown, product(product(rows, 320), 2), "HC activated down");
  requireBuffer(normalizedHyper, product(product(rows, 10240), 2), "HC normalized hyper");
  requireBuffer(mixed, product(product(rows, 2560), 2), "HC mixed output");
  requireBuffer(diagnostics, sizeof(uint32_t), "HC diagnostics");
  for (const auto &b : {activatedDown, normalizedHyper, mixed, diagnostics})
    if (overlaps(b, weight.buffer))
      throw std::invalid_argument("Flash cached HC up/mix aliases immutable weights");
  if (overlaps(mixed, activatedDown) || overlaps(mixed, normalizedHyper) ||
      overlaps(diagnostics, activatedDown) || overlaps(diagnostics, normalizedHyper) ||
      overlaps(diagnostics, mixed))
    throw std::invalid_argument("Flash cached HC up/mix buffer overlap");
  const FlashDenseCacheParams params{rows, 320, 2560, 0, 2560, m, n, 0};
  graph.add("flash_dense_cache_hc_up_mix_m" + std::to_string(m) + "_n" + std::to_string(n),
      {activatedDown, weight.buffer, normalizedHyper, mixed, diagnostics}, params,
      {2560 / n, rows / m, 1}, {128, 1, 1});
}

} // namespace splash::flash
