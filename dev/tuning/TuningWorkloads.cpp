#include "tuning/TuningWorkloads.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <type_traits>

namespace splash::model {
namespace {

using ops::tuning::LinearTuningInput;
using ops::tuning::LinearTuningWeights;
using ops::tuning::kMaximumLinearTuningRepresentatives;

bool sameProjection(const ops::Projection &left, const ops::Projection &right) {
  const auto &l = left.affine(), &r = right.affine();
  return left.inputSize == right.inputSize && left.outputSize == right.outputSize &&
         l.weights.sameView(r.weights) && l.scales.sameView(r.scales) && l.biases.sameView(r.biases);
}

bool sameWeights(const LinearTuningWeights &left, const LinearTuningWeights &right) {
  return sameProjection(left.projection, right.projection) &&
         left.gate.has_value() == right.gate.has_value() &&
         (!left.gate || sameProjection(*left.gate, *right.gate));
}

void appendDistinct(std::map<ops::LinearWorkload, LinearTuningInput> &table,
                    const ops::LinearWorkload &workload, const LinearTuningWeights &weights) {
  auto &input = table.try_emplace(workload, LinearTuningInput{workload, {}}).first->second;
  if (std::none_of(input.weights.begin(), input.weights.end(),
                   [&](const auto &existing) { return sameWeights(existing, weights); }))
    input.weights.push_back(weights);
}

void retainRepresentatives(std::vector<LinearTuningWeights> &weights) {
  constexpr size_t maximum = kMaximumLinearTuningRepresentatives;
  static_assert(maximum > 1);
  if (weights.size() <= maximum) return;
  // Distinct bundles arrive in layer order. Include both ends and evenly
  // spaced interior layers; tied views never consume a sampling position.
  std::vector<LinearTuningWeights> selected;
  selected.reserve(maximum);
  for (size_t index = 0; index < maximum; ++index)
    selected.push_back(std::move(weights[index * (weights.size() - 1) / (maximum - 1)]));
  weights = std::move(selected);
}

} // namespace

std::vector<LinearTuningInput> collectTuningWorkloads(
    const LoadedModel &model, std::span<const uint32_t> prefillRows,
    std::span<const uint32_t> decodeWidths) {
  using ops::LinearEpilogue;
  using ops::LinearPhase;
  for (uint32_t rows : prefillRows)
    if (!rows || rows > ExecutionLimits::prefillTokenBudget)
      throw std::invalid_argument("invalid operator prefill probe size");
  for (uint32_t width : decodeWidths)
    if (!width || width > ExecutionLimits::maximumBatchWidth)
      throw std::invalid_argument("invalid operator decode probe width");

  std::map<ops::LinearWorkload, LinearTuningInput> linear;
  auto projection = [&](const ops::Projection &weight, LinearPhase phase,
                         LinearEpilogue epilogue,
                         const ops::Projection *gate = nullptr) {
    if (!weight.inputSize || !weight.outputSize)
      throw std::invalid_argument("operator probe projection has no geometry");
    // Block projections are not tuned: they run the device policy's tiles.
    if (weight.layout() != ops::WeightLayout::Affine64 ||
        (gate && gate->layout() != ops::WeightLayout::Affine64)) return;
    const auto sizes = phase == LinearPhase::Prefill ? prefillRows : decodeWidths;
    for (uint32_t size : sizes) {
      const uint32_t rows = phase == LinearPhase::Prefill
          ? size : size * ExecutionLimits::targetVerifyRows;
      ops::LinearWorkload workload{{weight.outputSize, weight.inputSize},
                                    rows, phase, epilogue};
      const LinearTuningWeights representative{
          weight, gate ? std::optional{*gate} : std::nullopt};
      appendDistinct(linear, workload, representative);
    }
  };
  auto bothPhases = [&](const ops::Projection &weight,
                         LinearEpilogue epilogue = LinearEpilogue::None) {
    projection(weight, LinearPhase::Prefill, epilogue);
    projection(weight, LinearPhase::Decode, epilogue);
  };

  std::visit([&](const auto &target) {
    if (target.layers.empty())
      throw std::invalid_argument("operator probes require target layers");
    for (const auto &layer : target.layers) {
      std::visit([&](const auto &mixer) {
        bothPhases(mixer.inputProjection);
        bothPhases(mixer.outputProjection, LinearEpilogue::Residual);
      }, layer.mixer);
      if constexpr (std::is_same_v<std::remove_cvref_t<decltype(layer)>, Qwen3_8LayerWeights>) {
        projection(layer.gateProjection, LinearPhase::Prefill,
                     LinearEpilogue::None);
        projection(layer.upProjection, LinearPhase::Prefill,
                     LinearEpilogue::UpWithGate);
        projection(layer.upProjection, LinearPhase::Decode,
                     LinearEpilogue::GateUp, &layer.gateProjection);
        bothPhases(layer.downProjection, LinearEpilogue::Residual);
      }
    }
    projection(target.logitsProjection, LinearPhase::Decode,
                 LinearEpilogue::None);
  }, model.target);

  const auto &draft = model.draft;
  if (draft.layers.empty())
    throw std::invalid_argument("operator probes require draft layers");
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

  std::vector<LinearTuningInput> result;
  result.reserve(linear.size());
  for (auto &[key, input] : linear) {
    retainRepresentatives(input.weights);
    result.push_back(std::move(input));
  }
  return result;
}

} // namespace splash::model
