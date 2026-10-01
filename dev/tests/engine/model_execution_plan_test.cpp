#include "model/ModelFactory.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/RuntimeArenas.hpp"

#include "metal/abi/QuantFormat.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

using namespace splash;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Weights>
model::ModelPackage package() {
  model::ModelPackage result;
  Weights target;
  model::DFlashDraftLayout draft;
  if constexpr (std::is_same_v<Weights, model::Qwen3_6MoeWeights>) {
    draft.layers = 6;
    draft.hiddenSize = 2048;
    draft.dynamicSize = 512;
    draft.intermediateSize = 6144;
    draft.targetHiddenSize = target.layout.capturedHiddenSize();
  }
  const auto projection = [](uint32_t n, uint32_t k) {
    return ops::Projection(n, k, ops::AffineWeights{});
  };
  const auto &layout = target.layout;
  target.logitsProjection = projection(layout.vocabularySize, layout.hiddenSize);
  target.layers.resize(layout.layers);
  for (uint32_t i = 0; i < layout.layers; ++i) {
    auto &layer = target.layers[i];
    if (layout.isFullAttentionLayer(i)) {
      model::QwenAttentionWeights attention;
      attention.inputProjection = projection(layout.packedFullWidth, layout.hiddenSize);
      attention.outputProjection = projection(layout.hiddenSize, layout.attentionWidth);
      layer.mixer = std::move(attention);
    } else {
      model::QwenGdnWeights gdn;
      gdn.inputProjection = projection(layout.packedGdnWidth, layout.hiddenSize);
      gdn.outputProjection = projection(layout.hiddenSize, layout.attentionWidth);
      layer.mixer = std::move(gdn);
    }
    if constexpr (std::is_same_v<Weights, model::Qwen3_8Weights>) {
      layer.gateProjection = projection(layout.intermediateSize, layout.hiddenSize);
      layer.upProjection = layer.gateProjection;
      layer.downProjection = projection(layout.hiddenSize, layout.intermediateSize);
    }
  }
  ops::VisionLayout vision;
  vision.outputHiddenSize = target.layout.hiddenSize;
  result.descriptor = model::makeModelDescriptor(
      "operator workspace test", target.layout, draft, vision);
  result.target = std::move(target);
  result.draft.layout = draft;
  return result;
}

void checkPackage(const model::ModelPackage &package, uint32_t family) {
  DeviceCapabilities device;
  device.appleGpuFamily = family;
  ops::ExecutionPlans baseline(device);
  const auto before = model::plannedRuntimeMemory(device, package, baseline);
  const auto geometry = std::visit([](const auto &weights) {
    return model::qwenTargetGeometry(weights);
  }, package.target);
  const ops::AttentionShape attention{geometry.attentionQueryHeads,
                                      geometry.attentionKvHeads,
                                      geometry.attentionHeadDimension};
  ops::OperatorChoices choices;
  choices.prefillAttention.push_back(
      {{attention}, {ops::PrefillSplitMultiplier::Two}});
  choices.draftAttention.push_back(
      {{package.draft.layout.attentionShape(), 3}, {80}});
  if (geometry.ffnKind == model::QwenFfnKind::SparseMoe)
    choices.moe.push_back({{geometry.moeShape(), 24, ops::MoePhase::Decode},
                           {ops::MoeExpertTile::M32}});
  ops::ExecutionPlans selected(device);
  selected.install(choices);
  const auto after = model::plannedRuntimeMemory(device, package, selected);
  const auto prefillBefore = baseline.prefillAttentionWorkspace(
      2048, attention.queryHeads, geometry.kvLayout);
  const auto prefillAfter = selected.prefillAttentionWorkspace(
      2048, attention.queryHeads, geometry.kvLayout);
  const uint64_t prefillGrowth =
      model::alignArena(prefillAfter.partialsBytes) - model::alignArena(prefillBefore.partialsBytes) +
      model::alignArena(prefillAfter.statisticsBytes) - model::alignArena(prefillBefore.statisticsBytes);
  // The selected split count and the fallback baseline share an arena whose
  // governed bound includes the larger candidate's exact scratch requirement.
  require(prefillGrowth > 0 &&
              after.sharedPrefillPlannedAllocatedBytes ==
                  before.sharedPrefillPlannedAllocatedBytes + prefillGrowth,
          "runtime prefill allocation lost the selected split workspace bound");
  const auto selectedPrefill = selected.prefillAttention(
      2048, attention.queryHeads, geometry.kvLayout, 131072);
  require(selectedPrefill.configuration.splitMultiplier == ops::PrefillSplitMultiplier::Two &&
              selectedPrefill.workspace.partialsBytes ==
                  2 * baseline.prefillAttention(2048, attention.queryHeads,
                                             geometry.kvLayout, 131072)
                      .workspace.partialsBytes,
          "runtime did not install the selected prefill split plan");

  uint64_t decodeGrowth = 0;
  if (geometry.ffnKind == model::QwenFfnKind::SparseMoe) {
    const auto oldMoe = baseline.moeDecodeWorkspacePerLane(geometry.moeShape());
    const auto newMoe = selected.moeDecodeWorkspacePerLane(geometry.moeShape());
    for (const ops::MoeScratchField &field : ops::kMoeScratchFields)
      decodeGrowth += model::alignArena(model::kLaneCount * (newMoe.*field.bytes)) -
                      model::alignArena(model::kLaneCount * (oldMoe.*field.bytes));
    require(decodeGrowth > 0, "M24 expert plan did not reserve larger scratch");
  }
  require(after.sharedDecodePlannedAllocatedBytes ==
              before.sharedDecodePlannedAllocatedBytes + decodeGrowth,
          "runtime decode allocation does not use all selected width bounds");
  require(after.activeStateCellPlannedAllocatedBytes ==
              before.activeStateCellPlannedAllocatedBytes &&
              after.pipelineReserveBytes == before.pipelineReserveBytes &&
              after.runtimeOverheadReserveBytes == before.runtimeOverheadReserveBytes,
          "kernel selection changed state or unrelated memory reserves");
  require(selected.draftAttention(package.draft.layout.attentionShape(), 3)
                  .configuration().groups == 80,
          "paired draft did not use the same selection owner");
  selected.install({});
  const auto reset = model::plannedRuntimeMemory(device, package, selected);
  require(reset.sharedPrefillPlannedAllocatedBytes ==
              before.sharedPrefillPlannedAllocatedBytes &&
              reset.sharedDecodePlannedAllocatedBytes ==
              before.sharedDecodePlannedAllocatedBytes,
          "reset left stale selected workspace");
}

void checkMixedLayouts() {
  auto mixed = package<model::Qwen3_8Weights>();
  auto &target = std::get<model::Qwen3_8Weights>(mixed.target);
  auto &up = target.layers.front().upProjection;
  up = ops::Projection(up.outputSize, up.inputSize,
                       ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_Q4K, up.outputSize,
                                                                        up.inputSize, {}, {}, {})}});
  bool mismatchRejected = false;
  try { static_cast<void>(model::qwenTargetGeometry(target)); }
  catch (const model::WeightStoreError &) { mismatchRejected = true; }
  require(mismatchRejected, "incompatible fused gate/up layouts reached execution");
  target.layers.front().gateProjection = up;
  require(target.logitsProjection.layout() == ops::WeightLayout::Affine64,
          "mixed fixture must keep an affine vocabulary head");
  for (uint32_t family : {9U, 10U, 11U}) {
    DeviceCapabilities device;
    device.appleGpuFamily = family;
    device.gpuCoreCount = 16;
    ops::ExecutionPlans plans(device);
    const auto geometry = model::RuntimeGeometry::from(mixed);
    const auto head = target.logitsProjection.shape();
    const auto containsHead = [&](const auto &shapes) {
      return std::find(shapes.begin(), shapes.end(), head) != shapes.end();
    };
    require(!containsHead(geometry.target.prefillProjections) &&
                containsHead(geometry.target.decodeProjections),
            "vocabulary head must reserve workspace only in decode");
    const auto scratch = model::DecodeArena::linearScratchSize(geometry, plans);
    for (uint32_t lanes = 1; lanes <= model::kLaneCount; ++lanes) {
      const auto plan = plans.linear().plan({{up.outputSize, up.inputSize}, lanes * model::kDecodeRows,
          ops::LinearPhase::Decode, ops::LinearEpilogue::GateUp}, up);
      const auto required = plan.scratchSize();
      require(scratch.input >= required.input && scratch.sums >= required.sums &&
                  scratch.partials >= required.partials && scratch.counters >= required.counters,
              "affine head hid a block-quantized layer's scratch requirement");
      require(model::DecodeArena::gateScratchBytes(geometry, plans) >= plan.gateScratchBytes(),
              "mixed gate/up workspace is too small");
    }
    const auto sizes = model::prefillTensorBytes(geometry, plans);
    for (uint32_t rows : {1U, 8U, 17U, 32U}) {
      const auto required = plans.linear().plan({{up.outputSize, up.inputSize}, rows,
          ops::LinearPhase::Prefill, ops::LinearEpilogue::None}, up).scratchSize();
      require(sizes[uint32_t(model::PrefillTensor::LinearPartials)] >= required.partials &&
                  sizes[uint32_t(model::PrefillTensor::LinearCounters)] >= required.counters,
              "mixed short-prefill split scratch is too small");
    }
  }
  // Every MoE block of a target shares one layout, which the geometry's one
  // MoE shape records: no source mixes them.
  auto sparse = package<model::Qwen3_6MoeWeights>();
  auto &moe = std::get<model::Qwen3_6MoeWeights>(sparse.target);
  require(model::qwenTargetGeometry(moe).moeShape().weightLayout == ops::WeightLayout::Affine64,
          "the MoE shape lost the blocks' layout");
  for (auto &layer : moe.layers) layer.ffn = ops::BlockMoeWeights{};
  require(model::qwenTargetGeometry(moe).moeShape().weightLayout == ops::WeightLayout::Block32 &&
              moe.logitsProjection.layout() == ops::WeightLayout::Affine64,
          "the MoE shape must follow the expert layers, not the head");
  moe.layers.back().ffn = ops::AffineMoeWeights{};
  bool mixedRejected = false;
  try { static_cast<void>(model::qwenTargetGeometry(moe)); }
  catch (const model::WeightStoreError &) { mixedRejected = true; }
  require(mixedRejected, "a target mixing MoE layouts reached execution");
}

// One decode arena serves every lane count, and on Apple10 and later a
// Split128 plan's partials grow with the rows. The arena must hold every
// lane's plan of every affine target and draft projection, including an
// installed choice at eight splits, at the measured core counts.
void checkLaneScratch(const model::ModelPackage &package) {
  const auto geometry = model::RuntimeGeometry::from(package);
  const auto &d = geometry.draft;
  std::vector<ops::LinearMatrix> matrices{
      {d.dynamicSize, d.hiddenSize}, {d.qkvSize, d.hiddenSize}, {d.hiddenSize, d.attentionSize},
      {d.intermediateSize, d.hiddenSize}, {d.hiddenSize, d.intermediateSize},
      {d.selectorRank, d.hiddenSize}, {d.hiddenSize, d.targetHiddenSize}};
  for (const auto &p : geometry.target.decodeProjections)
    if (p.layout == ops::WeightLayout::Affine64) matrices.push_back({p.outputSize, p.inputSize});
  const ops::LinearWorkload chosen{{d.hiddenSize, d.targetHiddenSize}, model::kLaneCount * model::kDecodeRows,
                                   ops::LinearPhase::Decode, ops::LinearEpilogue::None};
  for (uint32_t family : {10U, 11U})
    for (uint32_t cores : {12U, 20U, 40U}) {
      DeviceCapabilities device;
      device.appleGpuFamily = family;
      device.gpuCoreCount = cores;
      ops::ExecutionPlans plans(device);
      ops::OperatorChoices choices;
      choices.linear.push_back({chosen, {ops::LinearTile::Split128, d.hiddenSize / 128,
                                         ops::LinearSimdgroups::Eight, 8}});
      plans.install(choices);
      const auto scratch = model::DecodeArena::linearScratchSize(geometry, plans);
      for (const auto matrix : matrices)
        for (uint32_t lanes = 1; lanes <= model::kLaneCount; ++lanes)
          for (auto epilogue : {ops::LinearEpilogue::None, ops::LinearEpilogue::Residual,
                                ops::LinearEpilogue::GateUp}) {
            const auto need = plans.linear().plan({matrix, lanes * model::kDecodeRows,
                ops::LinearPhase::Decode, epilogue}).scratchSize();
            require(scratch.partials >= need.partials && scratch.counters >= need.counters,
                    "decode arena scratch below a lane's affine plan");
          }
      require(plans.linear().plan(chosen).configuration().splits == 8 &&
                  scratch.partials >= plans.linear().plan(chosen).scratchSize().partials,
              "decode arena scratch lost the installed split choice");
    }
}

// Arenas are sized from the projections the weights hold, so each must have
// sizes; an empty one would drop its workspace from the bound silently.
void checkUnsizedProjection() {
  auto broken = package<model::Qwen3_8Weights>();
  std::get<model::Qwen3_8Weights>(broken.target).layers.back().downProjection = ops::Projection();
  bool rejected = false;
  try { static_cast<void>(model::RuntimeGeometry::from(broken)); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "a target projection without sizes reached arena sizing");
}

// The GDN value rows are sized with attentionWidth, so a layout whose value
// heads span another width is refused before loading and at arena sizing.
// The packed GDN rows must also hold the two gates of every value head.
void checkGdnWidths() {
  const auto sparse = package<model::Qwen3_6MoeWeights>();
  const auto layoutRejected = [](const model::Qwen3_6MoeLayout &layout) {
    try { model::requireQwenLayout(layout); }
    catch (const model::WeightStoreError &) { return true; }
    return false;
  };
  const auto geometryRejected = [&](const model::Qwen3_6MoeLayout &layout) {
    auto broken = sparse;
    std::get<model::Qwen3_6MoeWeights>(broken.target).layout = layout;
    try { static_cast<void>(model::RuntimeGeometry::from(broken)); }
    catch (const std::invalid_argument &) { return true; }
    return false;
  };
  const model::Qwen3_6MoeLayout shipped;
  require(!layoutRejected(shipped) && !geometryRejected(shipped),
          "the shipped sparse layout was refused");
  auto narrowValues = shipped;
  narrowValues.gdnValueHeads = narrowValues.gdnKeyHeads;
  narrowValues.convolutionDimension = 3 * narrowValues.gdnKeyHeads * narrowValues.gdnHeadDimension;
  require(layoutRejected(narrowValues) && geometryRejected(narrowValues),
          "a GDN value width other than attentionWidth was accepted");
  auto withoutGates = shipped;
  withoutGates.packedGdnWidth = shipped.convolutionDimension + shipped.attentionWidth;
  require(geometryRejected(withoutGates), "packed GDN rows without the gates reached arena sizing");
}

} // namespace

int main() {
  try {
    checkUnsizedProjection();
    checkGdnWidths();
    checkMixedLayouts();
    const auto dense = package<model::Qwen3_8Weights>();
    const auto sparse = package<model::Qwen3_6MoeWeights>();
    for (uint32_t family : {9U, 10U, 11U}) {
      checkPackage(dense, family);
      checkPackage(sparse, family);
    }
    checkLaneScratch(dense);
    checkLaneScratch(sparse);
    std::cout << "model execution plans: PASS (two paired geometries)\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
