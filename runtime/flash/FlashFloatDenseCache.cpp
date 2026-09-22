#include "FlashFloatDenseCache.hpp"

#include "FlashDenseCache.hpp"
#include "FlashOperandStore.hpp"

#include "metal/abi/FlashFloatDenseCache.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace splash::flash {
bool flashQSAOutF32N32Enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_FLASH_QSA_OUT_F32_N32");
    if (!value || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument("SPLASH_FLASH_QSA_OUT_F32_N32 must be 0 or 1");
  }();
  return enabled;
}
bool flashQSAOutF32N32Geometry(std::string_view prefix, uint32_t rows,
    uint32_t outputSize, uint32_t inputSize, uint32_t bits, uint32_t groupSize) noexcept {
  if (rows < 4 || rows > 16 || outputSize != 2560 || inputSize != 6144 ||
      (bits != 5 && bits != 6) || groupSize != 64) return false;
  constexpr std::string_view leading = "language_model.model.layers.";
  if (!prefix.starts_with(leading)) return false;
  const auto tail = prefix.substr(leading.size());
  const auto separator = tail.find('.');
  if (separator == std::string_view::npos || !separator || separator > 2) return false;
  uint32_t layer = 0;
  for (char digit : tail.substr(0, separator)) {
    if (digit < '0' || digit > '9') return false;
    layer = layer * 10 + uint32_t(digit - '0');
  }
  return layer < 48 && layer % 4 == 3 &&
      tail.substr(separator + 1) == "self_attn.o_proj";
}
namespace {
struct FloatPolicyRow final {
  std::string_view role;
  uint32_t outputs, inputs, bits, group;
  // -1: raw, 0: M8N64, 2: M16N64. Columns are R4, R8, R16, R9..15.
  int8_t four, eight, sixteen, middle;
};
// Both source screens use SPLASH_FLASH_QMV_F32=1. Require numerical checks,
// >=15% GPU gain and no measured wall regression in each available screen;
// choose the lowest median absolute cached GPU time among eligible tiles.
// dev/benchmarks/flash-float-body-screen.md describes the frozen reports.
constexpr std::array<FloatPolicyRow, 29> kFloatPolicy{{
    {"hc_up", 10240, 320, 4, 64, 0, 0, 0, 2},
    {"hc_up", 10240, 320, 5, 64, 0, 0, 0, 2},
    {"hc_up", 10240, 320, 6, 64, 0, 0, 0, 2},
    {"hc_up", 10240, 320, 8, 64, 0, 0, 0, 2},
    {"linear_attn.in_proj_qkv", 10240, 2560, 4, 64, -1, 2, 0, 2},
    {"linear_attn.in_proj_qkv", 10240, 2560, 5, 64, -1, -1, 0, -1},
    {"linear_attn.in_proj_qkv", 10240, 2560, 6, 64, 0, 0, 0, 2},
    {"linear_attn.in_proj_z", 6144, 2560, 5, 128, -1, 0, 0, -1},
    {"linear_attn.in_proj_z", 6144, 2560, 6, 64, -1, 0, 0, 2},
    {"linear_attn.out_proj", 2560, 6144, 5, 128, -1, -1, 0, -1},
    {"mlp.shared_expert.down_proj", 2560, 640, 8, 128, -1, -1, 2, -1},
    {"ple.key_proj", 10240, 2560, 4, 64, -1, 0, 0, 2},
    {"ple.value_proj", 2560, 2560, 4, 64, 0, 0, 2, 2},
    {"self_attn.q_proj", 12288, 2560, 4, 64, -1, 2, 2, 2},
    {"self_attn.q_proj", 12288, 2560, 5, 64, 0, 0, 2, 2},
    {"self_attn.q_proj", 12288, 2560, 6, 64, 0, 2, 2, 2},
    {"self_attn.q_proj", 12288, 2560, 8, 64, 2, 2, 2, 2},
    {"self_attn.o_proj", 2560, 6144, 5, 64, 2, 2, 2, 2},
    {"self_attn.o_proj", 2560, 6144, 6, 64, 2, 2, 2, 2},
    {"self_attn.o_proj", 2560, 6144, 8, 64, 0, 0, 0, 2},
    {"self_attn.k_proj", 512, 2560, 5, 64, -1, -1, 0, -1},
    {"self_attn.k_proj", 512, 2560, 6, 64, -1, 0, 0, 2},
    {"self_attn.k_proj", 512, 2560, 8, 64, -1, 0, 0, -1},
    {"self_attn.v_proj", 512, 2560, 5, 64, -1, 0, 0, -1},
    {"self_attn.v_proj", 512, 2560, 6, 128, -1, 0, 0, -1},
    {"self_attn.v_proj", 512, 2560, 8, 64, -1, -1, 0, -1},
    {"self_attn.indexer.index_qk_proj", 640, 2560, 5, 64, -1, 0, 0, 2},
    {"self_attn.indexer.index_qk_proj", 640, 2560, 6, 64, -1, 0, 0, 2},
    {"self_attn.indexer.index_qk_proj", 640, 2560, 8, 64, -1, 0, 0, 2},
}};

std::string_view policyRole(std::string_view prefix) noexcept {
  constexpr std::string_view layerPrefix = "language_model.model.layers.";
  constexpr std::string_view mixerPrefix = "language_model.model.hyper_connection_mixer.";
  std::string_view role;
  if (prefix.starts_with(layerPrefix)) {
    const auto tail = prefix.substr(layerPrefix.size());
    const auto separator = tail.find('.');
    if (separator == std::string_view::npos || !separator || separator > 2) return {};
    uint32_t layer = 0;
    for (char digit : tail.substr(0, separator)) {
      if (digit < '0' || digit > '9') return {};
      layer = layer * 10 + uint32_t(digit - '0');
    }
    if (layer >= 48) return {};
    role = tail.substr(separator + 1);
  } else if (prefix.starts_with(mixerPrefix)) {
    role = prefix.substr(mixerPrefix.size());
    if (role != "input_mix_weight_up") return {};
    return "hc_up";
  } else return {};
  if (role == "attn_hyper_connection.input_mix_weight_up" ||
      role == "mlp_hyper_connection.input_mix_weight_up") return "hc_up";
  return role;
}
} // namespace

std::optional<FlashFloatDenseSmallRowsTile>
flashFloatDenseSmallRowsPolicy(std::string_view prefix, uint32_t rows,
                              uint32_t outputSize, uint32_t inputSize,
                              uint32_t bits, uint32_t groupSize) noexcept {
  if (!rows || rows > 16) return std::nullopt;
  if (prefix == "language_model.lm_head") {
    if (rows < 2 || outputSize != 248320 || inputSize != 2560 || bits != 8 || groupSize != 64)
      return std::nullopt;
    return rows <= 8 ? FlashFloatDenseSmallRowsTile::M8N64 : FlashFloatDenseSmallRowsTile::M16N64;
  }
  if (rows < 4) return std::nullopt;
  const auto role = policyRole(prefix);
  for (const auto &entry : kFloatPolicy) {
    if (role != entry.role || outputSize != entry.outputs || inputSize != entry.inputs ||
        bits != entry.bits || groupSize != entry.group) continue;
    const int8_t tile = rows < 8 ? entry.four : (rows == 8 ? entry.eight :
        (rows == 16 ? entry.sixteen : entry.middle));
    if (tile < 0) return std::nullopt;
    return static_cast<FlashFloatDenseSmallRowsTile>(tile);
  }
  return std::nullopt;
}

namespace {
uint64_t product(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::invalid_argument("Flash float small-row dense byte extent overflows");
  return a * b;
}
void requireBuffer(const metal::MetalBuffer &b, uint64_t bytes, const char *name) {
  if (!b || !bytes || b.sizeBytes() < bytes || !b.contents())
    throw std::invalid_argument(std::string("Flash float small-row dense invalid Shared ") + name);
}
bool overlaps(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  if (a.sameView(b)) return true;
  const uintptr_t first = reinterpret_cast<uintptr_t>(a.contents());
  const uintptr_t second = reinterpret_cast<uintptr_t>(b.contents());
  if (!first || !second || first > UINTPTR_MAX - a.sizeBytes() ||
      second > UINTPTR_MAX - b.sizeBytes())
    throw std::invalid_argument("Flash float small-row dense invalid Shared address extent");
  return first < second + b.sizeBytes() && second < first + a.sizeBytes();
}
} // namespace

FlashFloatDenseSmallRowsWorkspace::FlashFloatDenseSmallRowsWorkspace(metal::MetalBackend &backend,
                                                           uint32_t maximumInputSize)
    : backend_(&backend), maximumInputSize_(maximumInputSize) {
  if (!maximumInputSize || maximumInputSize > 32768 || maximumInputSize % 32)
    throw std::invalid_argument("Flash float small-row dense maximum K must be aligned32 and1..32768");
  const uint64_t before = backend.memoryStats().allocatedBytes;
  const uint64_t bytes = product(product(16, maximumInputSize), 2);
  paddedInput_ = backend.allocateBuffer((bytes + 16383) & ~uint64_t{16383},
      metal::BufferStorage::Shared, "flash-small-row-dense-positive-zero-padding");
  const uint64_t after = backend.memoryStats().allocatedBytes;
  if (after < before) throw std::logic_error("Flash small-row workspace allocation ledger regressed");
  allocatedBytes_ = after - before;
}

namespace {
void addFloatDenseSmallRowsImpl(metal::MetalBackend &backend, metal::CommandGraph &graph,
                           metal::MetalBuffer input, const FlashTensor &weight,
                           metal::MetalBuffer output, metal::MetalBuffer diagnostics,
                           uint32_t rows, FlashFloatDenseSmallRowsWorkspace &workspace,
                           FlashFloatDenseSmallRowsTile tile, bool qualifiedQSAOutN32) {
  if (!workspace.belongsTo(backend) || !rows || rows > 16 ||
      weight.dtype != FlashDType::F32 || weight.shape.size() != 2 ||
      !weight.shape[0] || weight.shape[0] > UINT32_MAX || weight.shape[0] % 64 ||
      !weight.shape[1] || weight.shape[1] > workspace.maximumInputSize() || weight.shape[1] % 32)
    throw std::invalid_argument("Flash float small-row dense invalid matrix/rows/workspace");
  uint32_t m = 0, n = 0;
  switch (tile) {
  case FlashFloatDenseSmallRowsTile::M8N64: m = 8; n = 64; break;
  case FlashFloatDenseSmallRowsTile::M8N128: m = 8; n = 128; break;
  case FlashFloatDenseSmallRowsTile::M16N64: m = 16; n = 64; break;
  case FlashFloatDenseSmallRowsTile::M16N128: m = 16; n = 128; break;
  default: throw std::invalid_argument("Flash float small-row dense invalid tile");
  }
  if (qualifiedQSAOutN32) {
    if (weight.shape[0] != 2560 || weight.shape[1] != 6144 || rows < 4)
      throw std::invalid_argument("Flash QSA output N32 bridge invalid qualified geometry");
    m = 8; n = 32;
  }
  const uint32_t k = static_cast<uint32_t>(weight.shape[1]);
  const uint32_t outputs = static_cast<uint32_t>(weight.shape[0]);
  const uint32_t paddedRows = (rows + m - 1) / m * m;
  const uint64_t weightBytes = product(product(outputs, k), 4);
  if (weight.logicalBytes < weightBytes)
    throw std::invalid_argument("Flash float small-row dense weight logical extent is short");
  requireBuffer(weight.buffer, weightBytes, "weights");
  requireBuffer(input, product(product(rows, k), 2), "input");
  requireBuffer(output, product(product(rows, outputs), 2), "output");
  requireBuffer(diagnostics, sizeof(uint32_t), "diagnostics");
  requireBuffer(workspace.paddedInput(), product(product(paddedRows, k), 2), "padding");
  for (const auto &b : {input, output, diagnostics, workspace.paddedInput()})
    if (overlaps(b, weight.buffer))
      throw std::invalid_argument("Flash float small-row dense aliases immutable weights");
  if (overlaps(input, output) || overlaps(input, diagnostics) || overlaps(output, diagnostics) ||
      overlaps(workspace.paddedInput(), input) || overlaps(workspace.paddedInput(), output) ||
      overlaps(workspace.paddedInput(), diagnostics))
    throw std::invalid_argument("Flash float small-row dense buffer overlap");
  FlashFloatDenseSmallRowsParams params{rows, paddedRows, k, outputs, 0, outputs, m,
      qualifiedQSAOutN32 ? 64u : n};
  graph.add("flash_float_dense_small_rows_pad", {input, workspace.paddedInput(), diagnostics}, params,
      {(product(paddedRows, k) - 1) / 256 + 1, 1, 1});
  const auto dispatch = [&](uint32_t begin, uint32_t count, uint32_t tileN) {
    if (!count) return;
    params.output_begin = begin; params.output_count = count; params.tile_outputs = tileN;
    graph.add(qualifiedQSAOutN32 ? "flash_qsa_out_f32_n32_m8_n32_s4" :
        "flash_float_dense_small_rows_m" + std::to_string(m) + "_n" + std::to_string(tileN),
        {workspace.paddedInput(), weight.buffer, output, diagnostics}, params,
        {count / tileN, paddedRows / m, 1}, {128, 1, 1});
  };
  const uint32_t fullColumns = outputs / n * n;
  dispatch(0, fullColumns, n);
  if (fullColumns < outputs) dispatch(fullColumns, outputs - fullColumns, 64);
}
} // namespace

void addFloatDenseSmallRows(metal::MetalBackend &backend, metal::CommandGraph &graph,
    metal::MetalBuffer input, const FlashTensor &weight, metal::MetalBuffer output,
    metal::MetalBuffer diagnostics, uint32_t rows,
    FlashFloatDenseSmallRowsWorkspace &workspace, FlashFloatDenseSmallRowsTile tile) {
  addFloatDenseSmallRowsImpl(backend, graph, input, weight, output, diagnostics,
      rows, workspace, tile, false);
}

namespace {
uint64_t plus(uint64_t a, uint64_t b) {
  if (a > std::numeric_limits<uint64_t>::max() - b)
    throw std::invalid_argument("Flash float dense cache extent overflows");
  return a + b;
}
uint64_t rounded(uint64_t bytes) { return plus(bytes, 16383) & ~uint64_t{16383}; }
uint64_t extent(uint32_t rows, uint64_t stride, uint64_t rowBytes) {
  if (!rows || stride < rowBytes)
    throw std::invalid_argument("Flash float dense cache source stride is short");
  return plus(product(rows - 1, stride), rowBytes);
}
const FlashAffineProjection &checkedProjection(const FlashWeights &weights,
                                               std::string_view prefix) {
  const auto &p = weights.projection(prefix);
  if (prefix.empty() || prefix.find("embed_tokens") != std::string_view::npos ||
      prefix.find("ngram_embedding") != std::string_view::npos || p.experts != 1 ||
      !p.outputSize || p.outputSize % 64 || !p.inputSize || p.inputSize > 32768 ||
      (p.bits != 4 && p.bits != 5 && p.bits != 6 && p.bits != 8) ||
      (p.groupSize != 32 && p.groupSize != 64 && p.groupSize != 128) ||
      p.inputSize % p.groupSize || !p.weights || !p.scales || !p.biases ||
      p.weights->dtype != FlashDType::U32 || p.scales->dtype != FlashDType::BF16 ||
      p.biases->dtype != FlashDType::BF16 || p.parameterRowStrideBytes % 2)
    throw std::invalid_argument("Flash float dense cache requires checked selected dense projections");
  const uint64_t weightExtent = extent(p.outputSize, p.weightRowStrideBytes,
                                       (product(p.inputSize, p.bits) + 7) / 8);
  const uint64_t parameterExtent = extent(p.outputSize, p.parameterRowStrideBytes,
                                          product(p.inputSize / p.groupSize, 2));
  if (p.weights->logicalBytes < weightExtent || p.scales->logicalBytes < parameterExtent ||
      p.biases->logicalBytes < parameterExtent)
    throw std::invalid_argument("Flash float dense cache source logical extent is short");
  requireBuffer(p.weights->buffer, weightExtent, "original weights");
  requireBuffer(p.scales->buffer, parameterExtent, "original scales");
  requireBuffer(p.biases->buffer, parameterExtent, "original biases");
  return p;
}
std::vector<std::string> normalize(std::span<const std::string> prefixes) {
  if (prefixes.empty()) throw std::invalid_argument("Flash float dense cache selection is empty");
  std::vector<std::string> names(prefixes.begin(), prefixes.end());
  std::sort(names.begin(), names.end());
  names.erase(std::unique(names.begin(), names.end()), names.end());
  return names;
}
std::string identityDigest(const std::string &value) {
  if (value.size() > std::numeric_limits<CC_LONG>::max())
    throw std::invalid_argument("Flash float dense cache identity is too large");
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  if (!CC_SHA256(value.data(), static_cast<CC_LONG>(value.size()), digest.data()))
    throw std::runtime_error("Flash float dense cache SHA256 failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (const unsigned char byte : digest) {
    result += hex[byte >> 4]; result += hex[byte & 15];
  }
  return result;
}
struct StringHash final {
  using is_transparent = void;
  size_t operator()(std::string_view name) const noexcept {
    return std::hash<std::string_view>{}(name);
  }
};
} // namespace

struct FlashFloatDenseCache::Impl final {
  metal::MetalBackend &backend;
  std::unordered_map<std::string, FlashTensor, StringHash, std::equal_to<>> tensors;
  std::unordered_map<std::string, bool, StringHash, std::equal_to<>> qsaOutN32Qualified;
  std::vector<std::string> names;
  metal::MetalBuffer diagnostics;
  std::string identity;
  std::string operandStoreIdentity;
  uint64_t persistedTensors = 0;
  uint64_t persistedBytes = 0;
  std::vector<metal::MetalBuffer> persistedBuffers;
  uint64_t allocatedBytes = 0;
  metal::CommandTiming initialization;
  mutable uint64_t qsaOutN32Dispatches = 0, qsaOutN32Rows = 0;

  Impl(metal::MetalBackend &value, const FlashWeights &source,
       std::span<const std::string> prefixes)
      : backend(value), names(normalize(prefixes)) {
    (void)flashQSAOutF32N32Enabled();
    (void)FlashFloatDenseCache::plannedBytes(source, names);
    const auto store = FlashOperandStore::fromEnvironment(source);
    if (store) operandStoreIdentity = store->identitySha256();
    const uint64_t before = backend.memoryStats().allocatedBytes;
    diagnostics = backend.allocateBuffer(16384, metal::BufferStorage::Shared,
                                         "flash-f32-dense-conversion-diagnostics");
    std::memset(diagnostics.contents(), 0, diagnostics.sizeBytes());
    std::string fingerprint = std::string(kFlashFloatDenseCacheOperandFormat) + "\n";
    const auto field = [&](std::string_view key, std::string_view value) {
      fingerprint.append(key); fingerprint += ':';
      fingerprint += std::to_string(value.size()); fingerprint += ':';
      fingerprint.append(value); fingerprint += '\n';
    };
    field("source", source.sourceIdentity()); field("manifest", source.manifestFingerprint());
    metal::CommandGraph graph;
    for (const auto &name : names) {
      const auto &p = checkedProjection(source, name);
      qsaOutN32Qualified.emplace(name, flashQSAOutF32N32Geometry(name, 4,
          p.outputSize, p.inputSize, p.bits, p.groupSize));
      const uint64_t bytes = product(product(p.outputSize, p.inputSize), 4);
      FlashTensor weights;
      const bool saved = store && store->contains(name, FlashOperandFormat::F32);
      if (saved) {
        weights = store->mapTensor(backend, flashOperandSpec(name, FlashOperandFormat::F32, p));
        ++persistedTensors;
        persistedBytes = plus(persistedBytes, weights.buffer.sizeBytes());
        persistedBuffers.push_back(weights.buffer);
      }
      else {
        weights.buffer = backend.allocateBuffer(rounded(bytes), metal::BufferStorage::Shared,
                                                "flash-f32-exact-affine-cache:" + name);
        weights.dtype = FlashDType::F32;
        weights.shape = {p.outputSize, p.inputSize}; weights.logicalBytes = bytes;
      }
      auto [entry, inserted] = tensors.emplace(name, std::move(weights));
      if (!inserted) throw std::logic_error("Flash float dense cache selection has duplicates");
      const FlashAffineParams params{1, 1, p.inputSize, p.outputSize, 1,
          p.bits, p.groupSize, 0, p.weightRowStrideBytes, p.weightExpertStrideBytes,
          p.parameterRowStrideBytes, p.parameterExpertStrideBytes};
      const uint64_t elements = product(p.outputSize, p.inputSize);
      if (!saved) graph.add("flash_float_dense_cache_expand", {p.weights->buffer, p.scales->buffer,
          p.biases->buffer, entry->second.buffer, diagnostics}, params,
          {(elements - 1) / 256 + 1, 1, 1});
      field("projection", name);
      field("geometry", std::to_string(p.outputSize) + "," + std::to_string(p.inputSize) + "," +
          std::to_string(p.bits) + "," + std::to_string(p.groupSize) + "," +
          std::to_string(p.weightRowStrideBytes) + "," + std::to_string(p.parameterRowStrideBytes));
    }
    identity = identityDigest(fingerprint);
    if (!graph.dispatches().empty()) initialization = backend.submitCommand(graph.dispatches());
    uint32_t status = 0; std::memcpy(&status, diagnostics.contents(), sizeof(status));
    if (status) throw std::runtime_error("Flash F32 coefficient conversion diagnostics failed: " +
                                         std::to_string(status));
    const uint64_t after = backend.memoryStats().allocatedBytes;
    if (after < before) throw std::logic_error("Flash F32 cache allocation ledger regressed");
    allocatedBytes = after - before;
  }
};

FlashFloatDenseCache::FlashFloatDenseCache(metal::MetalBackend &backend,
    const FlashWeights &weights, std::string_view prefix) {
  const std::array<std::string, 1> names{std::string(prefix)};
  impl_ = std::make_unique<Impl>(backend, weights, names);
}
FlashFloatDenseCache::FlashFloatDenseCache(metal::MetalBackend &backend,
    const FlashWeights &weights, std::span<const std::string> prefixes)
    : impl_(std::make_unique<Impl>(backend, weights, prefixes)) {}
FlashFloatDenseCache::~FlashFloatDenseCache() = default;
FlashFloatDenseCache::FlashFloatDenseCache(FlashFloatDenseCache &&) noexcept = default;
FlashFloatDenseCache &FlashFloatDenseCache::operator=(FlashFloatDenseCache &&) noexcept = default;
uint64_t FlashFloatDenseCache::plannedBytes(const FlashWeights &weights, std::string_view prefix) {
  const std::array<std::string, 1> names{std::string(prefix)};
  return plannedBytes(weights, names);
}
uint64_t FlashFloatDenseCache::plannedBytes(const FlashWeights &weights,
                                          std::span<const std::string> prefixes) {
  uint64_t bytes = 16384;
  for (const auto &name : normalize(prefixes)) {
    const auto &p = checkedProjection(weights, name);
    bytes = plus(bytes, rounded(product(product(p.outputSize, p.inputSize), 4)));
  }
  return bytes;
}
std::vector<std::string> FlashFloatDenseCache::defaultPrefixes(const FlashWeights &weights,
                                                            bool includeVocabularyHead) {
  return FlashDenseCache::defaultPrefixes(weights, includeVocabularyHead);
}
bool FlashFloatDenseCache::contains(std::string_view prefix) const noexcept {
  return impl_ && impl_->tensors.contains(prefix);
}
const FlashTensor &FlashFloatDenseCache::tensor(std::string_view prefix) const {
  if (!impl_) throw std::logic_error("Flash F32 coefficient cache is not initialized");
  const auto found = impl_->tensors.find(prefix);
  if (found == impl_->tensors.end())
    throw std::invalid_argument("Flash F32 projection was not selected: " + std::string(prefix));
  return found->second;
}
const std::vector<std::string> &FlashFloatDenseCache::prefixes() const {
  if (!impl_) throw std::logic_error("Flash F32 coefficient cache is not initialized");
  return impl_->names;
}
std::vector<metal::MetalBuffer> FlashFloatDenseCache::immutableWeightBuffers() const {
  std::vector<metal::MetalBuffer> result;
  if (impl_) for (const auto &name : impl_->names) result.push_back(impl_->tensors.at(name).buffer);
  return result;
}
std::vector<metal::MetalBuffer> FlashFloatDenseCache::persistedWeightBuffers() const {
  if (!impl_) return {};
  if (impl_->persistedBuffers.size() != impl_->persistedTensors ||
      (!impl_->persistedBuffers.empty() && impl_->operandStoreIdentity.empty()))
    throw std::logic_error("Flash F32 saved operand enumeration is inconsistent");
  return impl_->persistedBuffers;
}
const FlashTensor &FlashFloatDenseCache::tensor() const { return tensor(prefix()); }
const std::string &FlashFloatDenseCache::prefix() const {
  if (!impl_ || impl_->names.size() != 1)
    throw std::logic_error("Flash F32 legacy accessor requires a single selected projection");
  return impl_->names.front();
}
const std::string &FlashFloatDenseCache::identitySha256() const {
  if (!impl_) throw std::logic_error("Flash F32 coefficient cache is not initialized");
  return impl_->identity;
}
uint64_t FlashFloatDenseCache::allocatedBytes() const noexcept {
  return impl_ ? impl_->allocatedBytes : 0;
}
uint64_t FlashFloatDenseCache::persistedTensorCount() const noexcept {
  return impl_ ? impl_->persistedTensors : 0;
}
uint64_t FlashFloatDenseCache::persistedPayloadBytes() const noexcept {
  return impl_ ? impl_->persistedBytes : 0;
}
const std::string &FlashFloatDenseCache::operandStoreIdentitySha256() const {
  if (!impl_) throw std::logic_error("Flash F32 coefficient cache is not initialized");
  return impl_->operandStoreIdentity;
}
metal::CommandTiming FlashFloatDenseCache::initializationTiming() const noexcept {
  return impl_ ? impl_->initialization : metal::CommandTiming{};
}
uint64_t FlashFloatDenseCache::qsaOutF32N32Dispatches() const noexcept {
  return impl_ ? impl_->qsaOutN32Dispatches : 0;
}
uint64_t FlashFloatDenseCache::qsaOutF32N32RealRows() const noexcept {
  return impl_ ? impl_->qsaOutN32Rows : 0;
}

void FlashFloatDenseCache::addSmallRows(metal::CommandGraph &graph, metal::MetalBuffer input,
    metal::MetalBuffer output, metal::MetalBuffer diagnostics, uint32_t rows,
    FlashFloatDenseSmallRowsWorkspace &workspace, FlashFloatDenseSmallRowsTile tile) const {
  addSmallRows(graph, prefix(), input, output, diagnostics, rows, workspace, tile);
}
void FlashFloatDenseCache::addSmallRows(metal::CommandGraph &graph, std::string_view prefix,
    metal::MetalBuffer input, metal::MetalBuffer output, metal::MetalBuffer diagnostics,
    uint32_t rows, FlashFloatDenseSmallRowsWorkspace &workspace,
    FlashFloatDenseSmallRowsTile tile) const {
  const auto &weights = tensor(prefix);
  bool qsaOutN32 = false;
  if (flashQSAOutF32N32Enabled() && rows >= 4 && rows <= 16) {
    const auto source = impl_->qsaOutN32Qualified.find(prefix);
    qsaOutN32 = source != impl_->qsaOutN32Qualified.end() && source->second;
  }
  addFloatDenseSmallRowsImpl(impl_->backend, graph, input, weights, output,
      diagnostics, rows, workspace, tile, qsaOutN32);
  if (qsaOutN32) {
    ++impl_->qsaOutN32Dispatches;
    impl_->qsaOutN32Rows += rows;
  }
}

} // namespace splash::flash
