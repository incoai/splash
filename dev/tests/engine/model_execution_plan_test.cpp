#include "TestChecks.hpp"
#include "TestDrafts.hpp"
#include "TestModel.hpp"
#include "model/ModelDescriptor.hpp"
#include "model/ModelFactory.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/RuntimeArenas.hpp"

#include "metal/abi/QuantFormat.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace splash;

using splash::test::rejects;
using splash::test::require;

template <class Weights>
model::DFlashDraftLayout draftOf() {
  if constexpr (std::is_same_v<Weights, model::Qwen3_6MoeWeights>)
    return test::qwen3_6MoeDraft();
  else if constexpr (std::is_same_v<Weights, model::Qwen4ExpWeights>)
    return test::flashNextDraft();
  else
    return test::qwen3_8Draft();
}

template <class Weights>
model::LoadedModel loadedModel() {
  model::LoadedModel result;
  Weights target;
  const model::DFlashDraftLayout draft = draftOf<Weights>();
  const auto projection = [](uint32_t n, uint32_t k) {
    return ops::Projection(n, k, ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_Q4K, n, k, {}, {}, {})}});
  };
  const auto &layout = target.layout;
  // Qwen3.8-Flash-Next's hyper-connection mixes, QSA indexers and PLE layer.
  const auto hyper = [&] {
    return model::Qwen4HyperWeights{.norm = {}, .down = projection(layout.hyperRank, layout.streamWidth()),
                                    .up = projection(layout.streamWidth(), layout.hyperRank), .inject = {}};
  };
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
    if constexpr (std::is_same_v<Weights, model::Qwen4ExpWeights>) {
      layer.mixerHyper = hyper();
      layer.ffnHyper = hyper();
      if (layout.isFullAttentionLayer(i))
        layer.indexer = model::Qwen4IndexerWeights{
            .query = projection(layout.indexerHeads * layout.indexerHeadDimension, layout.hiddenSize),
            .key = projection(layout.indexerHeadDimension, layout.hiddenSize), .queryNorm = {}, .keyNorm = {}};
      if (i == layout.pleLayer)
        layer.ple = model::Qwen4PleWeights{.key = projection(layout.hiddenSize, layout.pleWidth()),
                                           .value = projection(layout.hiddenSize, layout.pleWidth()),
                                           .keyNorm = {}, .queryNorm = {}, .convolutionNorm = {},
                                           .convolution = {}};
    }
  }
  if constexpr (std::is_same_v<Weights, model::Qwen4ExpWeights>)
    target.finalHyper = hyper();
  ops::VisionLayout vision;
  vision.outputHiddenSize = target.layout.hiddenSize;
  result.descriptor = model::makeModelDescriptor(
      "operator workspace test", target.layout, draft, vision,
      model::TargetSource::Safetensors, model::VisionSource::Safetensors);
  result.target = std::move(target);
  result.draft.layout = draft;
  return result;
}

// A layer's gate and up projections run as one gate/up plan, so their shapes
// must match, though not their formats: a pair with an F16 tensor, which the
// Apple9 register tile does not take, stages. The vocabulary head reserves
// workspace only in decode, and the arenas hold a gate/up layer's decode and
// short-prefill plans and a mixed pair's decode plans at the rows they store.
void checkGateUpLayers() {
  auto dense = loadedModel<model::Qwen3_8Weights>();
  auto &target = std::get<model::Qwen3_8Weights>(dense.target);
  const ops::Projection up = target.layers.front().upProjection;
  target.layers.front().upProjection = ops::Projection(
      up.outputSize, up.inputSize + 256,
      ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_Q4K, up.outputSize, up.inputSize + 256, {}, {}, {})}});
  rejects([&] { static_cast<void>(model::qwenTargetGeometry(target, dense.draft.layout.targetLayerIds())); },
          "fused gate/up projections must have matching shapes", "mismatched fused gate/up shapes reached execution");
  target.layers.front().upProjection = up;
  // The last layer's gate in F16 and its up in Q8_0.
  const auto planes = [&](uint32_t format) {
    return ops::Projection(
        up.outputSize, up.inputSize,
        ops::BlockWeights{{ops::QuantizedSegment::planes(format, up.outputSize, up.inputSize, {}, {}, {})}});
  };
  auto &mixed = target.layers.back();
  mixed.gateProjection = planes(GGUF_FMT_F16);
  mixed.upProjection = planes(GGUF_FMT_Q80);
  for (uint32_t family : {9U, 10U, 11U}) {
    DeviceCapabilities device;
    device.appleGpuFamily = family;
    device.gpuCoreCount = 16;
    ops::ExecutionPlans plans(device);
    const auto geometry = model::RuntimeGeometry::from(dense, kv::Format::Int8, ops::GdnStateFormat::Float32);
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
              "the decode arena is below a gate/up layer's scratch");
      require(model::DecodeArena::gateScratchBytes(geometry, plans) >= plan.gateScratchBytes(),
              "the gate/up workspace is too small");
      // The F16 and Q8_0 pair stages, at rows the decode storage of the
      // target's projections holds (QwenTarget::decodeStorageLanes).
      const auto pair = plans.linear().decodePlan(mixed.upProjection, lanes, ops::LinearEpilogue::GateUp,
                                                  &mixed.gateProjection);
      const auto paired = pair.scratchSize();
      uint32_t storage = 0;
      for (const auto &shape : geometry.target.decodeProjections)
        storage = std::max(storage, plans.linear().decodeStorageRows(lanes * model::kDecodeRows, shape));
      require(pair.configuration().tile == ops::LinearTile::GgufStaged && pair.storageRows() <= storage,
              "a gate/up pair with an F16 gate did not stage within the decode storage");
      require(scratch.input >= paired.input && scratch.sums >= paired.sums && scratch.partials >= paired.partials &&
                  scratch.counters >= paired.counters &&
                  model::DecodeArena::gateScratchBytes(geometry, plans) >= pair.gateScratchBytes(),
              "the arenas are below a gate/up pair in two formats");
    }
    const auto sizes = model::prefillTensorBytes(geometry, plans);
    for (uint32_t rows : {1U, 8U, 17U, 32U}) {
      const auto required = plans.linear().plan({{up.outputSize, up.inputSize}, rows,
          ops::LinearPhase::Prefill, ops::LinearEpilogue::None}, up).scratchSize();
      require(sizes[uint32_t(model::PrefillTensor::LinearPartials)] >= required.partials &&
                  sizes[uint32_t(model::PrefillTensor::LinearCounters)] >= required.counters,
              "the short-prefill split scratch is too small");
    }
  }
}

// One decode arena serves every lane count, and a staged plan's partials
// grow with its rows. The arena must hold every lane's plan of every target
// and draft projection at the measured core counts.
void checkLaneScratch(const model::LoadedModel &loaded) {
  const auto geometry = model::RuntimeGeometry::from(loaded, kv::Format::Int8, ops::GdnStateFormat::Float32);
  const auto &d = geometry.draft;
  std::vector<ops::LinearMatrix> matrices{
      {d.dynamicSize, d.hiddenSize}, {d.qkvSize, d.hiddenSize}, {d.contextKvSize(), d.hiddenSize},
      {d.hiddenSize, d.attentionSize},
      {d.intermediateSize, d.hiddenSize}, {d.hiddenSize, d.intermediateSize},
      {d.selectorRank, d.hiddenSize}, {d.hiddenSize, d.targetHiddenSize()}};
  for (const auto &p : geometry.target.decodeProjections) matrices.push_back({p.outputSize, p.inputSize});
  for (uint32_t family : {9U, 10U, 11U})
    for (uint32_t cores : {12U, 20U, 40U}) {
      DeviceCapabilities device;
      device.appleGpuFamily = family;
      device.gpuCoreCount = cores;
      const ops::ExecutionPlans plans(device);
      const auto scratch = model::DecodeArena::linearScratchSize(geometry, plans);
      for (const auto matrix : matrices)
        for (uint32_t lanes = 1; lanes <= model::kLaneCount; ++lanes)
          for (auto epilogue : {ops::LinearEpilogue::None, ops::LinearEpilogue::Residual,
                                ops::LinearEpilogue::GateUp}) {
            const auto need = plans.linear().plan({matrix, lanes * model::kDecodeRows,
                ops::LinearPhase::Decode, epilogue}).scratchSize();
            require(scratch.input >= need.input && scratch.sums >= need.sums && scratch.partials >= need.partials &&
                        scratch.counters >= need.counters,
                    "decode arena scratch below a lane's plan");
          }
    }
}

// A model served with YaRN reaches the runtime's geometry with its scaling
// and the window it serves, which everything sized by the context follows; a
// descriptor whose window is not the one its scaling serves is refused.
void checkYarnGeometry() {
  auto loaded = loadedModel<model::Qwen3_6MoeWeights>();
  loaded.descriptor = model::servedFor(std::move(loaded.descriptor), 524'288);
  const auto geometry = model::RuntimeGeometry::from(loaded, kv::Format::Int8, ops::GdnStateFormat::Float32);
  require(geometry.target.ropeScaling == loaded.descriptor.ropeScaling &&
              geometry.target.contextTokens() == 524'288 && geometry.target.nativeContextTokens == 262'144,
          "the runtime's geometry does not serve YaRN's window");
  loaded.descriptor.capabilities.maximumContextTokens = 262'144;
  rejects([&] {
    static_cast<void>(model::RuntimeGeometry::from(loaded, kv::Format::Int8, ops::GdnStateFormat::Float32));
  }, "invalid model runtime geometry", "a geometry serving another window than the model's was accepted");
}

// A decode tensor's bytes per lane in a runtime of `geometry`, on an M5's
// plans.
uint64_t decodeBytes(const model::RuntimeGeometry &geometry, model::DecodeTensor tensor) {
  DeviceCapabilities device;
  device.appleGpuFamily = 10;
  device.gpuCoreCount = 20;
  return model::decodeTensorBytes(geometry, ops::ExecutionPlans(device))[static_cast<uint32_t>(tensor)];
}

// A block-sparse target selects its blocks through index tables of their
// own, a lane's beside its page table (kv::PageStorage::writeIndexEntries);
// a target without a block index has none, and its arenas are as they were.
void checkIndexTables() {
  const auto geometry = [](const model::LoadedModel &loaded) {
    return model::RuntimeGeometry::from(loaded, kv::Format::Int8, ops::GdnStateFormat::Float32);
  };
  require(decodeBytes(geometry(loadedModel<model::Qwen4ExpWeights>()), model::DecodeTensor::IndexTable) ==
                  model::kMaximumPageTableEntries * sizeof(SplashKvPage) &&
              !decodeBytes(geometry(loadedModel<model::Qwen3_8Weights>()), model::DecodeTensor::IndexTable) &&
              !decodeBytes(geometry(loadedModel<model::Qwen3_6MoeWeights>()), model::DecodeTensor::IndexTable),
          "a lane's index table is not its block index's alone");
}

// A runtime whose K/V spills (one given the readers of spilled K/V,
// model::RuntimeContext) runs its prefill commands as groups, which read an
// attention layer's spilled history once each, and its decode arena holds
// where its verify's listed blocks are: a lane's list, 8 rows of 512 blocks
// and the block of each row's tail. The K/V of a target without a block
// index cannot spill.
void checkSpilledKvGeometry() {
  const auto flash = loadedModel<model::Qwen4ExpWeights>();
  const auto resident = model::RuntimeGeometry::from(flash, kv::Format::Int8, ops::GdnStateFormat::Float32);
  const auto spilled = model::RuntimeGeometry::from(flash, kv::Format::Int8, ops::GdnStateFormat::Float32,
                                                    model::StreamedParts::loaded(flash, true));
  require(resident.prefillChunks == 1 && !resident.spilledKv &&
              spilled.prefillChunks == model::kPrefillGroupChunks && spilled.spilledKv == test::sparseKvShape(),
          "a runtime's prefill groups do not follow whether its K/V spills");
  require(!decodeBytes(resident, model::DecodeTensor::Locations) &&
              decodeBytes(spilled, model::DecodeTensor::Locations) == uint64_t{8} * (512 + 1) * sizeof(uint32_t),
          "a runtime's decode arena does not hold its spilled blocks' locations exactly when its K/V spills");
  for (const model::LoadedModel &loaded :
       {loadedModel<model::Qwen3_8Weights>(), loadedModel<model::Qwen3_6MoeWeights>()})
    rejects([&] {
      static_cast<void>(model::RuntimeGeometry::from(loaded, kv::Format::Int8, ops::GdnStateFormat::Float32,
                                                     model::StreamedParts::loaded(loaded, true)));
    }, "invalid model runtime geometry", "the K/V of a target without a block index was let spill");
}

// Arenas are sized from the projections the weights hold, so each must have
// sizes; an empty one would drop its workspace from the bound silently.
void checkUnsizedProjection() {
  auto broken = loadedModel<model::Qwen3_8Weights>();
  std::get<model::Qwen3_8Weights>(broken.target).layers.back().downProjection = ops::Projection();
  rejects([&] {
    static_cast<void>(model::RuntimeGeometry::from(broken, kv::Format::Int8, ops::GdnStateFormat::Float32));
  }, "invalid model runtime geometry", "a target projection without sizes reached arena sizing");
}

// The target layers the draft reads are the target's, ascending: arena
// sizing refuses a draft reading past the last one or one twice.
void checkCaptureLayers() {
  const auto refuses = [](uint32_t index, uint32_t layer, const std::string &what) {
    auto model = loadedModel<model::Qwen3_8Weights>();
    model.draft.layout.targetLayers[index] = layer;
    rejects([&] {
      static_cast<void>(model::RuntimeGeometry::from(model, kv::Format::Int8, ops::GdnStateFormat::Float32));
    }, "invalid model runtime geometry", what + " reached arena sizing");
  };
  refuses(4, 64, "a draft reading past the target's last layer");
  refuses(1, 5, "a draft reading a layer twice");
}

// The GDN value rows are sized with attentionWidth, so a layout whose value
// heads span another width is refused before loading. Arena sizing checks
// the GDN shape, whose packed rows must also hold the two gates of every
// value head.
void checkGdnWidths() {
  const auto sparse = loadedModel<model::Qwen3_6MoeWeights>();
  const auto sizeArenas = [&](const model::Qwen3_6MoeLayout &layout) {
    auto candidate = sparse;
    std::get<model::Qwen3_6MoeWeights>(candidate.target).layout = layout;
    static_cast<void>(model::RuntimeGeometry::from(candidate, kv::Format::Int8, ops::GdnStateFormat::Float32));
  };
  // The shipped sparse layout passes both checks.
  const model::Qwen3_6MoeLayout shipped;
  model::requireQwenLayout(shipped);
  sizeArenas(shipped);
  auto narrowValues = shipped;
  narrowValues.gdnValueHeads = narrowValues.gdnKeyHeads;
  narrowValues.convolutionDimension = 3 * narrowValues.gdnKeyHeads * narrowValues.gdnHeadDimension;
  rejects([&] { model::requireQwenLayout(narrowValues); }, "Qwen target layout is inconsistent",
          "a GDN value width other than attentionWidth was accepted");
  auto withoutGates = shipped;
  withoutGates.packedGdnWidth = shipped.convolutionDimension + shipped.attentionWidth;
  rejects([&] { sizeArenas(withoutGates); }, "invalid model runtime geometry",
          "packed GDN rows without the gates reached arena sizing");
  // The dense layout is checked like the sparse one: its convolution width
  // against its GDN heads.
  const model::Qwen3_8Layout dense;
  model::requireQwenLayout(dense);
  auto convolutionMismatch = dense;
  convolutionMismatch.convolutionDimension += dense.gdnHeadDimension;
  rejects([&] { model::requireQwenLayout(convolutionMismatch); }, "Qwen target layout is inconsistent",
          "an inconsistent dense layout was accepted");
}

} // namespace

int main() {
  try {
    checkUnsizedProjection();
    checkCaptureLayers();
    checkGdnWidths();
    checkGateUpLayers();
    checkYarnGeometry();
    checkIndexTables();
    checkSpilledKvGeometry();
    const auto dense = loadedModel<model::Qwen3_8Weights>();
    const auto sparse = loadedModel<model::Qwen3_6MoeWeights>();
    checkLaneScratch(dense);
    checkLaneScratch(sparse);
    std::cout << "model execution plans: PASS (two paired geometries)\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
