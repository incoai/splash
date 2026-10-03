#include "TestChecks.hpp"
#include "tuning/TuningWorkloads.hpp"

#include "metal/BackendInstrumentation.hpp"
#include "metal/abi/QuantFormat.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <set>
#include <stdexcept>
#include <type_traits>

namespace {

using namespace splash::model;
using namespace splash::ops;
using splash::metal::BackendInstrumentation;

static_assert(!tuning::kPrefillProbeRows.empty() &&
              tuning::kPrefillProbeRows.back() == ExecutionLimits::prefillTokenBudget);
static_assert(tuning::kDecodeProbeWidths == std::array<uint32_t, 4>{1, 2, 3, 4});

using splash::test::require;

template <class Function> void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid tuning inventory was accepted");
}

// The collector only borrows projection metadata. Empty immutable Metal
// handles make any accidental allocation, weight access or dispatch fail;
// these tests never construct a MetalBackend or load a model package.
Projection projection(uint32_t output, uint32_t input) {
  return {output, input, AffineWeights{}};
}

template <class Weights, class Layout> Weights targetWeights(Layout layout) {
  Weights target;
  target.layout = layout;
  target.logitsProjection = projection(layout.vocabularySize, layout.hiddenSize);
  for (uint32_t index = 0; index < layout.layers; ++index) {
    auto &layer = target.layers.emplace_back();
    if (layout.isFullAttentionLayer(index)) {
      QwenAttentionWeights mixer;
      mixer.inputProjection = projection(layout.packedFullWidth, layout.hiddenSize);
      mixer.outputProjection = projection(layout.hiddenSize, layout.attentionWidth);
      layer.mixer = mixer;
    } else {
      QwenGdnWeights mixer;
      mixer.inputProjection = projection(layout.packedGdnWidth, layout.hiddenSize);
      mixer.outputProjection = projection(layout.hiddenSize, layout.attentionWidth);
      layer.mixer = mixer;
    }
    if constexpr (std::is_same_v<Weights, Qwen3_8Weights>) {
      layer.gateProjection = projection(layout.intermediateSize, layout.hiddenSize);
      layer.upProjection = layer.gateProjection;
      layer.downProjection = projection(layout.hiddenSize, layout.intermediateSize);
    } else {
      const Q8Projection router{{}, 256, layout.hiddenSize};
      const ExpertProjection up{{}, layout.experts, layout.expertIntermediateSize,
                                  layout.hiddenSize, 16'384};
      const ExpertProjection down{{}, layout.experts, layout.hiddenSize,
                                    layout.expertIntermediateSize, 16'384};
      auto sharedUp = up;
      auto sharedDown = down;
      sharedUp.experts = sharedDown.experts = 1;
      layer.ffn = AffineMoeWeights{router, up, up, down, sharedUp, sharedUp, sharedDown, router};
    }
  }
  return target;
}

DFlashDraftWeights draftWeights(DFlashDraftLayout layout) {
  DFlashDraftWeights draft;
  draft.layout = layout;
  draft.contextProjection = projection(layout.hiddenSize, layout.targetHiddenSize);
  draft.selectorProjection = projection(layout.selectorRank, layout.hiddenSize);
  for (uint32_t index = 0; index < layout.layers; ++index) {
    auto &layer = draft.layers.emplace_back();
    layer.attentionDynamic = projection(layout.dynamicSize, layout.hiddenSize);
    layer.mlpDynamic = layer.attentionDynamic;
    layer.qkvProjection = projection(layout.qkvSize, layout.hiddenSize);
    layer.outputProjection = projection(layout.hiddenSize, layout.attentionSize);
    layer.gateProjection = projection(layout.intermediateSize, layout.hiddenSize);
    layer.upProjection = layer.gateProjection;
    layer.downProjection = projection(layout.hiddenSize, layout.intermediateSize);
  }
  return draft;
}

std::set<LinearWorkload> linearKeys(const std::vector<tuning::LinearTuningInput> &inventory) {
  std::set<LinearWorkload> result;
  for (const auto &input : inventory) {
    require(result.insert(input.workload).second, "duplicate linear workload");
    require(!input.weights.empty() &&
                input.weights.size() <= tuning::kMaximumLinearTuningRepresentatives,
            "linear representative count is empty or unbounded");
    for (const auto &weights : input.weights) {
      require(weights.projection.inputSize == input.workload.matrix.inputSize &&
                  weights.projection.outputSize == input.workload.matrix.outputSize,
              "linear representative has a different projection geometry");
      require(weights.gate.has_value() ==
                  (input.workload.epilogue == LinearEpilogue::GateUp),
              "gate weight view attached to wrong semantic epilogue");
      if (weights.gate)
        require(weights.gate->inputSize == weights.projection.inputSize &&
                    weights.gate->outputSize == weights.projection.outputSize,
                "fused gate/up lost its matching gate projection");
    }
  }
  require(std::is_sorted(inventory.begin(), inventory.end(),
                        [](const auto &a, const auto &b) {
                          return a.workload < b.workload;
                        }),
          "linear inventory order is nondeterministic");
  return result;
}

// This operation list follows the semantic production calls, independently
// of the collector's layer visitor. In particular, draft residuals are in
// convolution, not linear; target down/mixer residuals are fused linear ops.
std::set<LinearWorkload> expectedLinear(
    const ModelPackage &package, std::span<const uint32_t> prefill,
    std::span<const uint32_t> decode) {
  const auto target = std::visit([](const auto &weights) {
    return qwenTargetGeometry(weights);
  }, package.target);
  const auto &draft = package.draft.layout;
  std::set<LinearWorkload> result;
  auto add = [&](LinearMatrix matrix, LinearPhase phase, LinearEpilogue epilogue) {
    for (uint32_t size : phase == LinearPhase::Prefill ? prefill : decode)
      result.insert({matrix, phase == LinearPhase::Prefill ? size : size * 8,
                      phase, epilogue});
  };
  for (auto phase : {LinearPhase::Prefill, LinearPhase::Decode}) {
    for (auto matrix : {LinearMatrix{target.packedGdnWidth, target.hiddenSize},
                         LinearMatrix{target.packedFullWidth, target.hiddenSize},
                         LinearMatrix{draft.hiddenSize, draft.targetHiddenSize},
                         LinearMatrix{draft.qkvSize, draft.hiddenSize}})
      add(matrix, phase, LinearEpilogue::None);
    add({target.hiddenSize, target.attentionWidth}, phase, LinearEpilogue::Residual);
    if (target.ffnKind == QwenFfnKind::Dense)
      add({target.hiddenSize, target.denseIntermediateSize}, phase,
           LinearEpilogue::Residual);
  }
  if (target.ffnKind == QwenFfnKind::Dense) {
    const LinearMatrix up{target.denseIntermediateSize, target.hiddenSize};
    add(up, LinearPhase::Prefill, LinearEpilogue::None);
    add(up, LinearPhase::Prefill, LinearEpilogue::UpWithGate);
    add(up, LinearPhase::Decode, LinearEpilogue::GateUp);
  }
  for (auto matrix : {LinearMatrix{target.vocabularySize, target.hiddenSize},
                       LinearMatrix{draft.dynamicSize, draft.hiddenSize},
                       LinearMatrix{draft.hiddenSize, draft.attentionSize},
                       LinearMatrix{draft.hiddenSize, draft.intermediateSize},
                       LinearMatrix{draft.selectorRank, draft.hiddenSize}})
    add(matrix, LinearPhase::Decode, LinearEpilogue::None);
  add({draft.intermediateSize, draft.hiddenSize}, LinearPhase::Decode,
       LinearEpilogue::GateUp);
  return result;
}

void checkPair(ModelPackage package) {
  const auto startup = collectTuningWorkloads(
      package, tuning::kPrefillProbeRows, tuning::kDecodeProbeWidths);
  require(linearKeys(startup) == expectedLinear(
              package, tuning::kPrefillProbeRows, tuning::kDecodeProbeWidths),
          "startup inventory differs from fixed 2048 prefill and B1-B4 decode");
  // The metadata collector still describes exact ragged rows for correctness
  // and dependency checks; these are not additional calibration workloads.
  constexpr std::array prefill{2048U, 17U, 2048U};
  constexpr std::array decode{4U, 1U, 3U, 2U, 4U};
  const auto inventory = collectTuningWorkloads(package, prefill, decode);
  const auto keys = linearKeys(inventory);
  for (const auto &input : inventory) {
    require(input.weights.size() == 1, "tied empty views were not deduplicated");
    const auto &weight = input.weights.front().projection;
    require(!weight.affine().weights && !weight.affine().scales && !weight.affine().biases,
            "inventory created weight backing");
  }
  require(keys == expectedLinear(package, prefill, decode),
          "inventory differs from production operation/phase/epilogue set");

  package.descriptor.name = "unseen-paired-model-with-identical-operators";
  package.manifestFingerprintSha256 = "different-weight-identity";
  // An extra same-shape layer and duplicate probe sizes must not multiply the
  // offline sweep. Neither sessions nor manifest names are workload keys.
  std::visit([](auto &target) {
    target.layers.push_back(target.layers.front());
  }, package.target);
  package.draft.layers.push_back(package.draft.layers.front());
  const auto renamed = collectTuningWorkloads(package, prefill, decode);
  require(linearKeys(renamed) == keys,
          "dedup depends on layer count or model/weight names");
  const auto noDecode = collectTuningWorkloads(package, prefill, {});
  require(linearKeys(noDecode) == expectedLinear(package, prefill, {}),
          "prefill-only sweep introduced decode work");
  const auto empty = collectTuningWorkloads(package, {}, {});
  require(empty.empty(), "empty probe sets created work");
  rejects([&] { (void)collectTuningWorkloads(package, std::array{0U}, decode); });
  rejects([&] { (void)collectTuningWorkloads(package, std::array{2049U}, decode); });
  rejects([&] { (void)collectTuningWorkloads(package, prefill, std::array{0U}); });
  rejects([&] { (void)collectTuningWorkloads(package, prefill, std::array{5U}); });
  package.draft.contextProjection.inputSize = 0;
  rejects([&] { (void)collectTuningWorkloads(package, prefill, decode); });
}

// A GGUF target is not tuned: the collector takes none of its projections,
// only the affine draft's.
void blockTarget() {
  ModelPackage package;
  const Qwen3_6MoeLayout layout;
  auto target = targetWeights<Qwen3_6MoeWeights>(layout);
  const auto block = [](const Projection &p) {
    return Projection(p.outputSize, p.inputSize,
                      BlockWeights{{QuantizedSegment::planes(GGUF_FMT_Q4K, p.outputSize, p.inputSize, {}, {}, {})}});
  };
  target.logitsProjection = block(target.logitsProjection);
  for (auto &layer : target.layers) {
    std::visit([&](auto &mixer) {
      mixer.inputProjection = block(mixer.inputProjection);
      mixer.outputProjection = block(mixer.outputProjection);
    }, layer.mixer);
    layer.ffn = BlockMoeWeights{};
  }
  package.target = std::move(target);
  DFlashDraftLayout draft;
  draft.layers = 6;
  draft.hiddenSize = 2048;
  draft.dynamicSize = 512;
  draft.intermediateSize = 6144;
  draft.targetHiddenSize = 16384;
  package.draft = draftWeights(draft);
  const auto inventory = collectTuningWorkloads(package, tuning::kPrefillProbeRows,
                                                tuning::kDecodeProbeWidths);
  for (const auto &input : linearKeys(inventory))
    require(input.epilogue != LinearEpilogue::Residual &&
                input.matrix != LinearMatrix{layout.vocabularySize, layout.hiddenSize} &&
                input.matrix != LinearMatrix{layout.packedGdnWidth, layout.hiddenSize} &&
                input.matrix != LinearMatrix{layout.packedFullWidth, layout.hiddenSize},
            "a GGUF target's projections were collected for tuning");
}

void run() {
  ModelPackage dense;
  dense.target = targetWeights<Qwen3_8Weights>(Qwen3_8Layout{});
  dense.draft = draftWeights(DFlashDraftLayout{});
  checkPair(dense);

  ModelPackage sparse;
  sparse.target = targetWeights<Qwen3_6MoeWeights>(Qwen3_6MoeLayout{});
  DFlashDraftLayout smallerDraft;
  smallerDraft.layers = 6;
  smallerDraft.hiddenSize = 2048;
  smallerDraft.dynamicSize = 512;
  smallerDraft.intermediateSize = 6144;
  smallerDraft.targetHiddenSize = 16384;
  sparse.draft = draftWeights(smallerDraft);
  checkPair(sparse);

  Qwen3_8Layout alternateDense;
  alternateDense.hiddenSize = 4096;
  alternateDense.intermediateSize = 14336;
  DFlashDraftLayout alternateDraft;
  alternateDraft.hiddenSize = 4096;
  alternateDraft.dynamicSize = 1024;
  alternateDraft.intermediateSize = 12288;
  alternateDraft.targetHiddenSize = 20480;
  dense.target = targetWeights<Qwen3_8Weights>(alternateDense);
  dense.draft = draftWeights(alternateDraft);
  checkPair(dense);

  Qwen3_6MoeLayout alternateSparse;
  alternateSparse.hiddenSize = 1024;
  alternateSparse.gdnKeyHeads = 8;
  alternateSparse.gdnValueHeads = 16;
  alternateSparse.convolutionDimension = 4096;
  alternateSparse.attentionWidth = 2048;
  alternateSparse.attentionQueryHeads = 8;
  alternateSparse.packedGdnWidth = 6400;
  alternateSparse.packedFullWidth = 5120;
  alternateSparse.experts = 32;
  alternateSparse.expertsPerToken = 4;
  alternateSparse.expertIntermediateSize = 1024;
  smallerDraft.hiddenSize = 1024;
  smallerDraft.dynamicSize = 256;
  smallerDraft.intermediateSize = 4096;
  smallerDraft.targetHiddenSize = 8192;
  sparse.target = targetWeights<Qwen3_6MoeWeights>(alternateSparse);
  sparse.draft = draftWeights(smallerDraft);
  checkPair(sparse);

  dense.draft.layers.clear();
  rejects([&] { (void)collectTuningWorkloads(dense, std::array{32U}, std::array{1U}); });
  std::get<Qwen3_6MoeWeights>(sparse.target).layers.clear();
  rejects([&] { (void)collectTuningWorkloads(sparse, std::array{32U}, std::array{1U}); });
}

void metadataViews(const char *metallib) {
  using namespace splash::metal;
  MetalBackend backend(metallib);
  const uint64_t submissions = BackendInstrumentation::submittedCommands(backend);
  // Metadata-only test: one small allocation, no data access or GPU command.
  // Components are intentionally tiny because no projection is executed.
  const auto backing = backend.allocateBuffer(32 * 1024, BufferStorage::Shared,
                                               "tuning-inventory-view-test");
  ModelPackage package;
  package.target = targetWeights<Qwen3_8Weights>(Qwen3_8Layout{});
  package.draft = draftWeights(DFlashDraftLayout{});
  auto &target = std::get<Qwen3_8Weights>(package.target);
  const LinearWorkload key{{target.layout.intermediateSize, target.layout.hiddenSize},
                           8, LinearPhase::Decode, LinearEpilogue::GateUp};
  auto view = [&](uint32_t index, bool gate) {
    const uint64_t offset = uint64_t{index * 6 + (gate ? 3U : 0U)} * 128;
    return Projection(key.matrix.outputSize, key.matrix.inputSize,
                      AffineWeights{backend.view(backing, offset, 128),
                                    backend.view(backing, offset + 128, 128),
                                    backend.view(backing, offset + 256, 128)});
  };
  enum class Variation { Distinct, GateOnly, ScaleOnly, Tied };
  for (const auto variation : {Variation::Distinct, Variation::GateOnly,
                                Variation::ScaleOnly, Variation::Tied}) {
    for (uint32_t index = 0; index < target.layers.size(); ++index) {
      // Repeat each pair through freshly created view objects. Metadata, not
      // wrapper-object identity, must determine whether a tensor is tied.
      const uint32_t representative = index / 2;
      auto &layer = target.layers[index];
      layer.upProjection = view(variation == Variation::Distinct ? representative : 0, false);
      layer.gateProjection = view(variation == Variation::Distinct ||
                                     variation == Variation::GateOnly ? representative : 0, true);
      if (variation == Variation::ScaleOnly) {
        AffineWeights planes = layer.upProjection.affine();
        planes.scales = view(representative, false).affine().scales;
        layer.upProjection = Projection(key.matrix.outputSize, key.matrix.inputSize, planes);
      }
    }
    // Target and draft execute the same GateUp shape in this pair. They also
    // share a representative here, so this must not add another measurement.
    for (auto &layer : package.draft.layers) {
      layer.upProjection = target.layers.front().upProjection;
      layer.gateProjection = target.layers.front().gateProjection;
    }
    const auto bytes = backend.memoryStats().allocatedBytes;
    const auto inventory = collectTuningWorkloads(package, std::array{32U}, std::array{1U});
    (void)linearKeys(inventory);
    const auto found = std::find_if(inventory.begin(), inventory.end(),
                                    [&](const auto &input) { return input.workload == key; });
    require(found != inventory.end(), "GateUp inventory disappeared");
    const size_t expected = variation == Variation::Tied
        ? 1 : tuning::kMaximumLinearTuningRepresentatives;
    require(found->weights.size() == expected,
            "distinct/tied weight view bound or dedup is wrong");
    // 64 layers have 32 distinct paired views: keep the endpoints and six
    // evenly spaced interior bundles, rather than the first eight bundles.
    constexpr std::array<size_t, 8> sourceLayers{0, 8, 16, 26, 34, 44, 52, 62};
    for (size_t index = 0; index < expected; ++index) {
      const auto &actual = found->weights[index];
      const auto &source = target.layers[sourceLayers[index]];
      require(actual.projection.affine().weights.sameView(source.upProjection.affine().weights) &&
                  actual.projection.affine().scales.sameView(source.upProjection.affine().scales) &&
                  actual.projection.affine().biases.sameView(source.upProjection.affine().biases) &&
                  actual.gate &&
                  actual.gate->affine().weights.sameView(source.gateProjection.affine().weights) &&
                  actual.gate->affine().scales.sameView(source.gateProjection.affine().scales) &&
                  actual.gate->affine().biases.sameView(source.gateProjection.affine().biases),
              "representatives missed layer depth or lost gate/up pairing");
    }
    require(backend.memoryStats().allocatedBytes == bytes,
            "collecting representative views copied weight backing");
  }

  require(BackendInstrumentation::submittedCommands(backend) == submissions,
          "collecting representative metadata submitted GPU work");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc > 2)
      throw std::invalid_argument("usage: tuning-workloads [metallib]");
    run();
    blockTarget();
    if (argc == 2)
      metadataViews(argv[1]);
    std::cout << "Tuning workload inventory tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Tuning workload inventory failed: " << error.what() << '\n';
    return 1;
  }
}
