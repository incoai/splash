#include "model/LoraAdapter.hpp"

#include "metal/abi/Gguf.h"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightStore.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <string_view>

namespace splash::model {
namespace {

// The modules, in the order the attention's packed input holds q, k and v.
enum class Module : uint32_t { Query, Key, Value, AttentionOutput, Gate, Up, Down };
constexpr std::array<std::string_view, 7> kModuleNames{"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                                       "self_attn.o_proj", "mlp.gate_proj",    "mlp.up_proj",
                                                       "mlp.down_proj"};
constexpr std::string_view kA = ".lora_A.weight", kB = ".lora_B.weight";
// Each buffer view starts on a cache line.
constexpr uint64_t kAlignment = 64;

struct ModuleTensors final {
  const SourceTensor *a = nullptr, *b = nullptr;
};
// By layer and module.
using Modules = std::map<std::pair<uint32_t, uint32_t>, ModuleTensors>;

// The rank the updates run at: whole steps of the staged tile, the rows
// past the adapter's own of zeros.
uint32_t paddedRank(uint32_t rank) { return (rank + GGUF_STAGED_STEP - 1) / GGUF_STAGED_STEP * GGUF_STAGED_STEP; }
uint64_t aligned(uint64_t bytes) { return (bytes + kAlignment - 1) / kAlignment * kAlignment; }

// The checkpoint's modules: every tensor is the A or the B of a module of a
// layer, of the layout's rank, and each module has both.
Modules indexModules(const SafetensorsCheckpoint &checkpoint, const LoraAdapterLayout &layout) {
  Modules modules;
  for (const std::string &text : checkpoint.names("")) {
    const std::string_view name = text;
    const bool isA = name.ends_with(kA);
    if (!isA && !name.ends_with(kB)) throw WeightStoreError("LoRA adapter tensor is not a module's A or B: " + std::string(name));
    const std::string_view stem = name.substr(0, name.size() - kA.size());
    const size_t layers = stem.rfind("layers.");
    const size_t dot = layers == std::string_view::npos ? layers : stem.find('.', layers + 7);
    uint32_t layer = 0;
    const auto module = dot == std::string_view::npos
                            ? kModuleNames.end()
                            : std::find(kModuleNames.begin(), kModuleNames.end(), stem.substr(dot + 1));
    if (module == kModuleNames.end() ||
        std::from_chars(stem.data() + layers + 7, stem.data() + dot, layer).ptr != stem.data() + dot)
      throw WeightStoreError("LoRA adapter tensor names no module Splash applies (" +
                             std::string(kModuleNames.front()) + " ... " + std::string(kModuleNames.back()) +
                             " of a layer): " + std::string(name));
    const SourceTensor &tensor = checkpoint.require(name);
    if (tensor.shape.size() != 2 || tensor.shape[isA ? 0 : 1] != layout.rank ||
        (tensor.dtype != "F32" && tensor.dtype != "F16" && tensor.dtype != "BF16"))
      throw WeightStoreError("LoRA adapter tensor is not a float matrix of the adapter's rank: " + std::string(name));
    ModuleTensors &tensors = modules[{layer, static_cast<uint32_t>(module - kModuleNames.begin())}];
    (isA ? tensors.a : tensors.b) = &tensor;
  }
  for (const auto &[key, tensors] : modules)
    if (!tensors.a || !tensors.b)
      throw WeightStoreError("LoRA adapter module has no " + std::string(tensors.a ? "B" : "A") + ": layer " +
                             std::to_string(key.first) + " " + std::string(kModuleNames[key.second]));
  if (modules.empty()) throw WeightStoreError("LoRA adapter holds no modules");
  return modules;
}

// The projection a module updates and its segment of it; null where the
// layer has none.
std::pair<ops::Projection *, size_t> target(const LoraLayerTargets &layer, Module module) {
  switch (module) {
  case Module::Query: return {layer.attentionInput, 0};
  case Module::Key: return {layer.attentionInput, 1};
  case Module::Value: return {layer.attentionInput, 2};
  case Module::AttentionOutput: return {layer.attentionOutput, 0};
  case Module::Gate: return {layer.gate, 0};
  case Module::Up: return {layer.up, 0};
  case Module::Down: return {layer.down, 0};
  }
  return {nullptr, 0};
}

// The tensor's values.
std::vector<float> values(const SourceTensor &tensor) {
  std::vector<uint8_t> bytes(tensor.bytes);
  tensor.read(0, bytes);
  const uint64_t count = tensor.shape[0] * tensor.shape[1];
  std::vector<float> result(count);
  if (tensor.dtype == "F32") {
    std::memcpy(result.data(), bytes.data(), bytes.size());
  } else {
    const bool half = tensor.dtype == "F16";
    for (uint64_t i = 0; i < count; ++i) {
      uint16_t bits = 0;
      std::memcpy(&bits, bytes.data() + 2 * i, 2);
      result[i] = half ? static_cast<float>(std::bit_cast<_Float16>(bits)) : std::bit_cast<float>(uint32_t{bits} << 16);
    }
  }
  for (const float value : result)
    if (!std::isfinite(value)) throw WeightStoreError("LoRA adapter tensor holds a value that is not finite");
  return result;
}

// bf16, rounded to nearest even.
uint16_t bf16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

} // namespace

uint64_t loraAdapterBytes(const std::filesystem::path &directory, const LoraAdapterLayout &layout) {
  const SafetensorsCheckpoint checkpoint(directory);
  const uint64_t rank = paddedRank(layout.rank);
  uint64_t bytes = 0;
  for (const auto &[key, tensors] : indexModules(checkpoint, layout))
    bytes += aligned(rank * tensors.a->shape[1] * sizeof(uint16_t)) +
             aligned(tensors.b->shape[0] * rank * sizeof(uint16_t));
  return bytes;
}

void attachLoraAdapter(metal::MetalBackend &backend, std::span<const LoraLayerTargets> layers,
                       const std::filesystem::path &directory, const LoraAdapterLayout &layout,
                       QwenTargetWeightsBase &weights) {
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  const SafetensorsCheckpoint checkpoint(directory);
  const Modules modules = indexModules(checkpoint, layout);
  const uint32_t rank = paddedRank(layout.rank);
  // The modules by projection, each with its segment, in segment order.
  struct Update final {
    size_t segment;
    uint32_t seed;
    ModuleTensors tensors;
  };
  std::map<ops::Projection *, std::vector<Update>> updates;
  for (const auto &[key, tensors] : modules) {
    const auto [layer, module] = key;
    const std::string label = "layer " + std::to_string(layer) + " " + std::string(kModuleNames[module]);
    const auto [projection, segment] =
        layer < layers.size() ? target(layers[layer], static_cast<Module>(module)) : std::pair<ops::Projection *, size_t>{};
    if (!projection || segment >= projection->blocks().segments.size())
      throw WeightStoreError("LoRA adapter module updates a projection the target does not have: " + label);
    const ops::QuantizedSegment &s = projection->blocks().segments[segment];
    if (tensors.a->shape[1] != projection->inputSize || tensors.b->shape[0] != s.outputSize ||
        s.outputSize % GGUF_STAGED_COLUMNS || projection->inputSize % 32)
      throw WeightStoreError("LoRA adapter module does not match its projection of " + std::to_string(s.outputSize) +
                             " x " + std::to_string(projection->inputSize) + ": " + label);
    updates[projection].push_back({segment, layer * static_cast<uint32_t>(kModuleNames.size()) + module, tensors});
  }
  // One buffer of every A and B, which the CPU reads too (ops::AneFfn).
  metal::MetalBuffer buffer = backend.allocateBuffer(loraAdapterBytes(directory, layout), metal::BufferStorage::Shared,
                                                     "lora adapter");
  auto *base = static_cast<uint8_t *>(buffer.contents());
  std::memset(base, 0, buffer.sizeBytes());
  uint64_t offset = 0;
  const auto take = [&](uint64_t bytes) {
    const metal::MetalBuffer view = backend.view(buffer, offset, bytes);
    offset += aligned(bytes);
    return view;
  };
  for (auto &[projection, modulesOf] : updates) {
    const uint32_t inputSize = projection->inputSize, slots = static_cast<uint32_t>(modulesOf.size());
    ops::ProjectionLora update;
    update.rank = rank;
    update.slots = slots;
    update.seed = modulesOf.front().seed;
    update.segments.resize(projection->blocks().segments.size());
    // The modules' A one after another; the views are of whole modules, so
    // the padding of each lies between them.
    const uint64_t aBytes = uint64_t{rank} * inputSize * sizeof(uint16_t);
    const uint64_t aOffset = offset;
    offset += aligned(slots * aBytes);
    for (uint32_t slot = 0; slot < slots; ++slot) {
      const Update &module = modulesOf[slot];
      const uint32_t outputSize = static_cast<uint32_t>(module.tensors.b->shape[0]);
      std::vector<float> a = values(*module.tensors.a), b = values(*module.tensors.b);
      // B takes the adapter's scale and a power of two off A's side that
      // puts its largest value in [0.5, 1), where a tile stages it as a half.
      float peak = 0.0f;
      for (const float value : b) peak = std::max(peak, std::abs(value * layout.scale));
      int exponent = 0;
      if (peak > 0.0f) static_cast<void>(std::frexp(peak, &exponent));
      const float bScale = std::ldexp(layout.scale, -exponent), aScale = std::ldexp(1.0f, exponent);
      auto *aRows = reinterpret_cast<uint16_t *>(base + aOffset + slot * aBytes);
      for (uint64_t i = 0; i < uint64_t{layout.rank} * inputSize; ++i) aRows[i] = bf16(a[i] * aScale);
      const metal::MetalBuffer bView = take(uint64_t{outputSize} * rank * sizeof(uint16_t));
      auto *bRows = static_cast<uint16_t *>(bView.contents());
      for (uint32_t row = 0; row < outputSize; ++row)
        for (uint32_t j = 0; j < layout.rank; ++j)
          bRows[uint64_t{row} * rank + j] = bf16(b[uint64_t{row} * layout.rank + j] * bScale);
      update.segments[module.segment] = {bView, slot};
    }
    update.a = ops::QuantizedSegment::rows(GGUF_FMT_BF16, slots * rank, inputSize,
                                           backend.view(buffer, aOffset, slots * aBytes));
    projection->lora = std::move(update);
  }
  weights.manifestFingerprintSha256 =
      weightDigest(weights.manifestFingerprintSha256 + " lora " +
                   weightDigest(std::span(static_cast<const uint8_t *>(base), buffer.sizeBytes())));
  weights.actualAllocatedBytes += metal::allocationDelta(allocationBaseline, backend.memoryStats().allocatedBytes);
}

} // namespace splash::model
