#include "tuning/TuningWorkloads.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace splash::model {
namespace {

template <class Projection>
bool sameProjection(const Projection &left, const Projection &right) noexcept {
  return left.inputSize == right.inputSize && left.outputSize == right.outputSize &&
         left.weights.sameView(right.weights) &&
         left.scales.sameView(right.scales) && left.biases.sameView(right.biases);
}

bool sameWeights(const ops::tuning::LinearTuningWeights &left,
                  const ops::tuning::LinearTuningWeights &right) noexcept {
  return sameProjection(left.projection, right.projection) &&
         left.gate.has_value() == right.gate.has_value() &&
         (!left.gate || sameProjection(*left.gate, *right.gate));
}

bool sameExpert(const ops::ExpertQ4Projection &left,
                 const ops::ExpertQ4Projection &right) noexcept {
  return left.inputSize == right.inputSize && left.outputSize == right.outputSize &&
         left.experts == right.experts &&
         left.expertStrideBytes == right.expertStrideBytes &&
         left.packed.sameView(right.packed);
}

bool sameWeights(const ops::MoeWeights &left, const ops::MoeWeights &right) noexcept {
  if (!sameProjection(left.router, right.router) ||
      !sameProjection(left.sharedExpertGate, right.sharedExpertGate))
    return false;
  for (auto field : {&ops::MoeWeights::expertGate, &ops::MoeWeights::expertUp,
                     &ops::MoeWeights::expertDown, &ops::MoeWeights::sharedGate,
                     &ops::MoeWeights::sharedUp, &ops::MoeWeights::sharedDown})
    if (!sameExpert(left.*field, right.*field))
      return false;
  return true;
}

template <class Workload, class Input, class Weights>
void appendDistinct(std::map<Workload, Input> &table, const Workload &workload,
                     const Weights &weights) {
  auto &input = table.try_emplace(workload, Input{workload, {}}).first->second;
  if (std::none_of(input.weights.begin(), input.weights.end(),
                   [&](const auto &existing) { return sameWeights(existing, weights); }))
    input.weights.push_back(weights);
}

template <size_t Maximum, class Weights>
void retainRepresentatives(std::vector<Weights> &weights) {
  static_assert(Maximum > 1);
  if (weights.size() <= Maximum) return;
  // Distinct bundles arrive in layer order. Include both ends and evenly
  // spaced interior layers; tied views never consume a sampling position.
  std::vector<Weights> selected;
  selected.reserve(Maximum);
  for (size_t index = 0; index < Maximum; ++index)
    selected.push_back(std::move(weights[index * (weights.size() - 1) / (Maximum - 1)]));
  weights = std::move(selected);
}

} // namespace

TuningWorkloads collectTuningWorkloads(
    const ModelPackage &package, std::span<const uint32_t> prefillRows,
    std::span<const uint32_t> decodeWidths) {
  using ops::LinearEpilogue;
  using ops::LinearPhase;
  for (uint32_t rows : prefillRows)
    if (!rows || rows > ExecutionLimits::prefillTokenBudget)
      throw std::invalid_argument("invalid operator prefill probe size");
  for (uint32_t width : decodeWidths)
    if (!width || width > ExecutionLimits::maximumBatchWidth)
      throw std::invalid_argument("invalid operator decode probe width");

  std::map<ops::LinearWorkload, ops::tuning::LinearTuningInput> linear;
  std::map<ops::MoeWorkload, ops::tuning::MoeTuningInput> moe;
  auto projection = [&](const ops::Q4Projection &weight, LinearPhase phase,
                         LinearEpilogue epilogue,
                         const ops::Q4Projection *gate = nullptr) {
    if (!weight.inputSize || !weight.outputSize)
      throw std::invalid_argument("operator probe projection has no geometry");
    const auto sizes = phase == LinearPhase::Prefill ? prefillRows : decodeWidths;
    for (uint32_t size : sizes) {
      const uint32_t rows = phase == LinearPhase::Prefill
          ? size : size * ExecutionLimits::targetVerifyRows;
      ops::LinearWorkload workload{{weight.outputSize, weight.inputSize},
                                    rows, phase, epilogue};
      const ops::tuning::LinearTuningWeights representative{
          weight, gate ? std::optional{*gate} : std::nullopt};
      appendDistinct(linear, workload, representative);
    }
  };
  auto bothPhases = [&](const ops::Q4Projection &weight,
                         LinearEpilogue epilogue = LinearEpilogue::None) {
    projection(weight, LinearPhase::Prefill, epilogue);
    projection(weight, LinearPhase::Decode, epilogue);
  };

  TuningWorkloads result;
  std::visit([&](const auto &target) {
    using T = std::remove_cvref_t<decltype(target)>;
    const auto geometry = qwenTargetGeometry(target);
    result.targetAttention = {geometry.attentionQueryHeads,
                              geometry.attentionKvHeads,
                              geometry.attentionHeadDimension};
    if constexpr (std::is_same_v<T, Qwen3_8Q8Weights>) {
      return;
    } else {
      if (target.layers.empty())
        throw std::invalid_argument("operator probes require target layers");
      for (const auto &layer : target.layers) {
        std::visit([&](const auto &mixer) {
          using M = std::remove_cvref_t<decltype(mixer)>;
          if constexpr (!std::is_same_v<M, QwenGdnQ8Weights> &&
                        !std::is_same_v<M, QwenAttentionQ8Weights>) {
            bothPhases(mixer.inputProjection);
            bothPhases(mixer.outputProjection, LinearEpilogue::Residual);
          }
        }, layer.mixer);
        if constexpr (requires { layer.gateProjection; }) {
          projection(layer.gateProjection, LinearPhase::Prefill,
                       LinearEpilogue::None);
          projection(layer.upProjection, LinearPhase::Prefill,
                       LinearEpilogue::UpWithGate);
          projection(layer.upProjection, LinearPhase::Decode,
                       LinearEpilogue::GateUp, &layer.gateProjection);
          bothPhases(layer.downProjection, LinearEpilogue::Residual);
        } else {
          for (uint32_t rows : prefillRows) {
            ops::MoeWorkload workload{geometry.moe, rows, ops::MoePhase::Prefill};
            appendDistinct(moe, workload, layer.ffn);
          }
          for (uint32_t width : decodeWidths) {
            ops::MoeWorkload workload{geometry.moe,
                width * ExecutionLimits::targetVerifyRows, ops::MoePhase::Decode};
            appendDistinct(moe, workload, layer.ffn);
          }
        }
      }
      projection(target.logitsProjection, LinearPhase::Decode,
                   LinearEpilogue::None);
    }
  }, package.target);

  const auto &draft = package.draft;
  if (draft.layers.empty())
    throw std::invalid_argument("operator probes require draft layers");
  result.draftAttention = draft.layout.attentionShape();
  bothPhases(draft.contextProjection);
  for (const auto &layer : draft.layers) {
    bothPhases(layer.qkvProjection);
    for (const auto *weight : {&layer.attentionDynamic, &layer.mlpDynamic,
                               &layer.outputProjection, &layer.downProjection})
      projection(*weight, LinearPhase::Decode, LinearEpilogue::None);
    projection(layer.upProjection, LinearPhase::Decode,
                 LinearEpilogue::GateUp, &layer.gateProjection);
  }
  projection(draft.selectorProjection, LinearPhase::Decode,
               LinearEpilogue::None);

  result.linear.reserve(linear.size());
  for (auto &[key, input] : linear) {
    retainRepresentatives<ops::tuning::kMaximumLinearTuningRepresentatives>(input.weights);
    result.linear.push_back(std::move(input));
  }
  result.moe.reserve(moe.size());
  for (auto &[key, input] : moe) {
    retainRepresentatives<ops::tuning::kMaximumMoeTuningRepresentatives>(input.weights);
    result.moe.push_back(std::move(input));
  }
  return result;
}

} // namespace splash::model
