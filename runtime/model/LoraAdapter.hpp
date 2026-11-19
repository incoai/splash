#pragma once

#include "metal/MetalBackend.hpp"
#include "model/ModelDescriptor.hpp"
#include "model/QwenTarget.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <variant>
#include <vector>

namespace splash::model {

// A LoRA adapter beside a target: the PEFT checkpoint in lora/ of the model's
// root (adapter_config.json, which inspectModelRoot checks into a
// LoraAdapterLayout, and its safetensors weights), applied as loaded instead
// of merged into the quantized weights. Each module, "...layers.<n>.<module>
// .lora_A.weight" of [rank][in] and ".lora_B.weight" of [out][rank], updates
// the projection of that name: y = W x + scale B (A x)
// (ops::ProjectionLora). The modules are those of the Qwen layers' attention
// (self_attn.q_proj, k_proj, v_proj, o_proj) and dense FFN (mlp.gate_proj,
// up_proj, down_proj); a tensor of any other name is refused, and so is a
// module of a layer that has no such projection.

// A layer's projections a module may update; null where the layer has none.
struct LoraLayerTargets final {
  // The attention's packed q, k and v, and its output.
  ops::Projection *attentionInput = nullptr, *attentionOutput = nullptr;
  ops::Projection *gate = nullptr, *up = nullptr, *down = nullptr;
};

// The bytes attachLoraAdapter allocates for the adapter in `directory`.
[[nodiscard]] uint64_t loraAdapterBytes(const std::filesystem::path &directory, const LoraAdapterLayout &layout);

// Attaches the adapter's updates to `layers`' projections and counts them in
// `weights`: its allocation, and its fingerprint, which then covers the
// updates' values.
void attachLoraAdapter(metal::MetalBackend &backend, std::span<const LoraLayerTargets> layers,
                       const std::filesystem::path &directory, const LoraAdapterLayout &layout,
                       QwenTargetWeightsBase &weights);

// The projections of a family's layer: its attention mixer's and its dense
// FFN's, by the members the families name them with.
template <class Layer> [[nodiscard]] LoraLayerTargets loraLayerTargets(Layer &layer) {
  LoraLayerTargets targets;
  if (auto *attention = std::get_if<QwenAttentionWeights>(&layer.mixer)) {
    targets.attentionInput = &attention->inputProjection;
    targets.attentionOutput = &attention->outputProjection;
  }
  const auto dense = [&](auto &ffn) {
    if constexpr (requires { ffn.gate.outputSize; ffn.up.outputSize; ffn.down.outputSize; }) {
      targets.gate = &ffn.gate;
      targets.up = &ffn.up;
      targets.down = &ffn.down;
    }
  };
  if constexpr (requires { layer.gateProjection; }) {
    targets.gate = &layer.gateProjection;
    targets.up = &layer.upProjection;
    targets.down = &layer.downProjection;
  } else if constexpr (requires { std::visit(dense, layer.ffn); }) {
    std::visit(dense, layer.ffn);
  }
  return targets;
}

template <class Weights>
void attachLoraAdapter(metal::MetalBackend &backend, Weights &weights, const std::filesystem::path &directory,
                       const LoraAdapterLayout &layout) {
  std::vector<LoraLayerTargets> layers;
  for (auto &layer : weights.layers) layers.push_back(loraLayerTargets(layer));
  attachLoraAdapter(backend, layers, directory, layout, weights);
}

} // namespace splash::model
