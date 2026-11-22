#include "TestChecks.hpp"
#include "TestDrafts.hpp"
#include "TestModel.hpp"
#include "model/ModelDescriptor.hpp"
#include "model/ModelFactory.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/RuntimeArenas.hpp"

#include "metal/abi/Gguf.h"
#include "metal/abi/QuantFormat.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
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

// A target of Weights' architecture at the sizes of `family`, its first
// family's by default, its projections in `format`, paired with `draft` --
// an empty layout installs no draft (--draft-model none).
template <class Weights>
model::LoadedModel loadedModel(const decltype(Weights::layout) &family = {},
                               const model::DFlashDraftLayout &draft = draftOf<Weights>(),
                               uint32_t format = GGUF_FMT_Q4K) {
  model::LoadedModel result;
  Weights target;
  target.layout = family;
  const auto projection = [format](uint32_t n, uint32_t k) {
    return ops::Projection(n, k, ops::BlockWeights{{ops::QuantizedSegment::planes(format, n, k, {}, {}, {})}});
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
    ops::ExecutionPlans plans(device, ops::Int8Activations::On);
    for (const uint32_t runtimeLanes : {model::ExecutionLimits::defaultLanes, model::kLaneCount}) {
      const auto geometry = model::RuntimeGeometry::from(dense, kv::Format::Int8, ops::GdnStateFormat::Float32,
                                                         model::StreamedParts::loaded(dense), runtimeLanes);
      const auto head = target.logitsProjection.shape();
      const auto containsHead = [&](const auto &shapes) {
        return std::find(shapes.begin(), shapes.end(), head) != shapes.end();
      };
      require(!containsHead(geometry.target.prefillProjections) &&
                  containsHead(geometry.target.decodeProjections),
              "vocabulary head must reserve workspace only in decode");
      const auto scratch = model::DecodeArena::linearScratchSize(geometry, plans);
      for (uint32_t lanes = 1; lanes <= runtimeLanes; ++lanes) {
        // A step's rows, a verify block a lane: a gate/up pair on the decode
        // tiles, past them the up-with-gate pass after the gate's
        // (Linear::addSwiGlu), each the plan it runs.
        const uint32_t rows = model::ExecutionLimits::stepRows(lanes);
        const auto plan = plans.decodeLinear().plan({{up.outputSize, up.inputSize}, rows,
            rows <= ops::kMaximumDecodeTileRows ? ops::LinearEpilogue::GateUp : ops::LinearEpilogue::UpWithGate}, up);
        const uint64_t gate = plan.gateScratchBytes();
        const auto required = plan.scratchSize();
        require(scratch.input >= required.input && scratch.sums >= required.sums &&
                    scratch.partials >= required.partials && scratch.counters >= required.counters,
                "the decode arena is below a gate/up layer's scratch");
        require(model::DecodeArena::gateScratchBytes(geometry, plans) >= gate, "the gate/up workspace is too small");
        // The F16 and Q8_0 pair stages, at rows the storage of the target's
        // projections over a step's rows holds (ops::Linear::storageRows).
        if (rows > ops::kMaximumDecodeTileRows) continue;
        const auto pair = plans.decodeLinear().plan(mixed.upProjection, rows,
                                                    ops::LinearEpilogue::GateUp, &mixed.gateProjection);
        const auto paired = pair.scratchSize();
        uint32_t storage = 0;
        for (const auto &shape : geometry.target.decodeProjections)
          storage = std::max(storage, plans.decodeLinear().storageRows(rows, shape));
        require(pair.configuration().tile == ops::LinearTile::GgufStaged && pair.storageRows() <= storage,
                "a gate/up pair with an F16 gate did not stage within the decode storage");
        require(scratch.input >= paired.input && scratch.sums >= paired.sums && scratch.partials >= paired.partials &&
                    scratch.counters >= paired.counters &&
                    model::DecodeArena::gateScratchBytes(geometry, plans) >= pair.gateScratchBytes(),
                "the arenas are below a gate/up pair in two formats");
      }
      const auto sizes = model::prefillTensorBytes(geometry, plans);
      for (uint32_t rows : {1U, 8U, 17U, 32U, 33U, 48U, 64U}) {
        const auto required = plans.linear().plan({{up.outputSize, up.inputSize}, rows,
            ops::LinearEpilogue::None}, up).scratchSize();
        require(sizes[uint32_t(model::PrefillTensor::LinearPartials)] >= required.partials &&
                    sizes[uint32_t(model::PrefillTensor::LinearCounters)] >= required.counters,
                "the short-prefill split scratch is too small");
      }
    }
  }
}

// The prefill arena holds the int8 prefill tile's codes and scales of a full
// chunk of the target's widest projection the tile takes, wherever the family
// runs it with int8 activations on, and none for the draft, which prefills on
// the bf16 tiles, beside the tables of its plans of up to 32 rows on the
// decode tiles (Apple9's register tile): a GGUF target holds those tables
// alone, and Qwen3.8-27B MLX 4-bit its down projection's codes (17408 inputs),
// not its draft's context projection's (25600).
void checkPrefillInt8Scratch() {
  for (uint32_t family : {9U, 10U, 11U})
    for (const ops::Int8Activations setting : {ops::Int8Activations::On, ops::Int8Activations::Off}) {
      DeviceCapabilities device;
      device.appleGpuFamily = family;
      device.gpuCoreCount = 20;
      const ops::ExecutionPlans plans(device, setting);
      // The arena's codes and scales for `loaded`, and the tables of its plans
      // on the decode tiles.
      const auto int8Scratch = [&](const model::LoadedModel &loaded) {
        const auto sizes = model::prefillTensorBytes(
            model::RuntimeGeometry::from(loaded, kv::Format::Int8, ops::GdnStateFormat::Float32), plans);
        return std::pair{sizes[uint32_t(model::PrefillTensor::LinearTable)],
                         sizes[uint32_t(model::PrefillTensor::LinearSums)]};
      };
      const auto tables = [&](const model::LoadedModel &loaded) {
        const auto geometry = model::RuntimeGeometry::from(loaded, kv::Format::Int8, ops::GdnStateFormat::Float32);
        ops::LinearScratchSize bound;
        for (const auto &shape : geometry.target.prefillProjections)
          bound.include(plans.linear().scratchSize(shape, ops::kMaximumDecodeTileRows));
        const auto &draft = geometry.draft;
        for (const ops::ProjectionShape shape : {ops::ProjectionShape{draft.hiddenSize, draft.targetHiddenSize()},
                                                 {draft.contextKvSize(), draft.hiddenSize}})
          bound.include(plans.draftLinear().scratchSize(shape, ops::kMaximumDecodeTileRows));
        return std::pair{bound.input, bound.sums};
      };
      for (const model::LoadedModel &gguf :
           {loadedModel<model::Qwen3_8Weights>(), loadedModel<model::Qwen3_6MoeWeights>()})
        require(int8Scratch(gguf) == tables(gguf), "the prefill arena of a GGUF target holds int8 prefill scratch");
      const uint32_t down = model::Qwen3_8Layout{}.intermediateSize;
      const model::LoadedModel mlx = loadedModel<model::Qwen3_8Weights>({}, test::qwen3_8Draft(), GGUF_FMT_AF4G64);
      const auto [table, sums] = int8Scratch(mlx);
      const auto [decodeTable, decodeSums] = tables(mlx);
      const bool int8 = family >= 10 && setting == ops::Int8Activations::On;
      require(table == std::max<uint64_t>(decodeTable, int8 ? gguf_prefill_int8_table_bytes(down, SPLASH_PREFILL_TOKEN_BUDGET) : 0) &&
                  sums == std::max<uint64_t>(decodeSums, int8 ? gguf_prefill_int8_scale_bytes(down, SPLASH_PREFILL_TOKEN_BUDGET) : 0),
              "the prefill arena of an MLX target holds other int8 prefill scratch than its down projection's");
    }
}

// A decode arena, of four lanes or eight, serves every lane count up to its
// own, and a staged plan's partials grow with its rows. It must hold every lane's plan of every target
// projection and the draft's context projections, over a lane's most rows,
// and of the draft's own and of its head over its subset of the vocabulary
// (DFlashDraft), over its block's, at the measured core counts and an M3
// Ultra's 80: there the register tile splits K for the subset head's 2,322
// column tiles and not for the whole head's 3,880, whose plan takes no
// partials.
void checkLaneScratch(const model::LoadedModel &loaded) {
  for (const uint32_t runtimeLanes : {model::ExecutionLimits::defaultLanes, model::kLaneCount}) {
  const auto geometry = model::RuntimeGeometry::from(loaded, kv::Format::Int8, ops::GdnStateFormat::Float32,
                                                     model::StreamedParts::loaded(loaded), runtimeLanes);
  const auto &d = geometry.draft;
  const auto &target = geometry.target;
  // Each matrix with the rows a lane gives it: the draft's, when the model
  // has one, its head over its subset of the vocabulary among them.
  std::vector<std::pair<ops::LinearMatrix, uint32_t>> matrices;
  if (d.present())
    matrices = {
        {{d.dynamicSize, d.hiddenSize}, model::kDraftRows},       {{d.qkvSize, d.hiddenSize}, model::kDraftRows},
        {{d.hiddenSize, d.attentionSize}, model::kDraftRows},     {{d.intermediateSize, d.hiddenSize}, model::kDraftRows},
        {{d.hiddenSize, d.intermediateSize}, model::kDraftRows},  {{d.selectorRank, d.hiddenSize}, model::kDraftRows},
        {{d.contextKvSize(), d.hiddenSize}, model::kDecodeRows}, {{d.hiddenSize, d.targetHiddenSize()}, model::kDecodeRows},
        {{target.draftVocabulary.columns(target.vocabularySize), target.hiddenSize}, model::kDraftRows}};
  for (const auto &p : target.decodeProjections)
    matrices.push_back({{p.outputSize, p.inputSize}, model::kDecodeRows});
  for (uint32_t family : {9U, 10U, 11U})
    for (uint32_t cores : {12U, 20U, 40U, 80U}) {
      DeviceCapabilities device;
      device.appleGpuFamily = family;
      device.gpuCoreCount = cores;
      const ops::ExecutionPlans plans(device, ops::Int8Activations::On);
      const auto scratch = model::DecodeArena::linearScratchSize(geometry, plans);
      for (const auto &[matrix, laneRows] : matrices)
        for (uint32_t lanes = 1; lanes <= runtimeLanes; ++lanes)
          for (auto epilogue : {ops::LinearEpilogue::None, ops::LinearEpilogue::Residual,
                                ops::LinearEpilogue::GateUp}) {
            const uint32_t rows = std::min(lanes * laneRows, geometry.target.stepRows());
            if (epilogue == ops::LinearEpilogue::GateUp && rows > ops::kMaximumDecodeTileRows) continue;
            const auto need = plans.decodeLinear().plan({matrix, rows, epilogue}).scratchSize();
            require(scratch.input >= need.input && scratch.sums >= need.sums && scratch.partials >= need.partials &&
                        scratch.counters >= need.counters,
                    "decode arena scratch below a lane's plan");
          }
    }
  }
}

// A decode step of any rows its lanes may give it, up to 128 at eight lanes,
// finds its MoE plan's workspace in the decode arena: past 32 rows the 35B's
// and the 122B's plans take 32-row expert tiles, whose 274 tiles at 64 rows a
// bound of four lanes of 8 rows did not hold.
void checkDecodeStepMoe(const model::LoadedModel &loaded) {
  for (const auto &[lanes, stepRows] : {std::pair{4U, 64U}, {8U, 128U}})
  for (uint32_t family : {9U, 10U, 11U}) {
    const auto geometry = model::RuntimeGeometry::from(loaded, kv::Format::Int8, ops::GdnStateFormat::Float32,
                                                       model::StreamedParts::loaded(loaded), lanes);
    DeviceCapabilities device;
    device.appleGpuFamily = family;
    device.gpuCoreCount = 20;
    const ops::ExecutionPlans plans(device, ops::Int8Activations::On);
    const model::TargetRows step = model::decodeTargetRows(geometry, plans);
    require(step.rows == stepRows, "the decode arena does not hold a step's rows");
    for (uint32_t rows = 1; rows <= step.rows; ++rows) {
      const ops::MoeWorkspace need =
          plans.moe(geometry.target.moeBlockShape(), geometry.target.moeRouterFormat, rows, ops::StepKind::Decode).workspace();
      for (const auto field : ops::kMoeWorkspaceFields)
        require(step.moe.*field >= need.*field,
                "the decode arena does not hold the MoE plan of a step of " + std::to_string(rows) + " rows");
    }
  }
}

// A decode step holds a verify block a lane (ExecutionLimits::stepRows), 64 rows
// at four lanes; past one decode tile's 64 its projections run the staged
// tile's two row groups, which store 96 rows at five and six lanes and 128 at
// seven and eight, so the decode arena's row tensors and its target tensors
// hold the storage of every plan over the step's rows on a decode step's
// plans: the target's projections, the vocabulary head and the draft's context
// projections, on every device family.
void checkDecodeStepRows(const model::LoadedModel &loaded) {
  const model::StreamedParts parts = model::StreamedParts::loaded(loaded);
  for (uint32_t family : {9U, 10U, 11U}) {
    DeviceCapabilities device;
    device.appleGpuFamily = family;
    device.gpuCoreCount = 20;
    const ops::ExecutionPlans plans(device, ops::Int8Activations::On);
    for (uint32_t lanes = 1; lanes <= model::kLaneCount; ++lanes) {
      const auto geometry = model::RuntimeGeometry::from(loaded, kv::Format::Int8, ops::GdnStateFormat::Float32, parts,
                                                         lanes);
      const uint32_t rows = model::ExecutionLimits::stepRows(lanes);
      const uint32_t held = model::decodeTargetRows(geometry, plans).rows;
      require(held == model::decodeTensorLanes(model::DecodeTensor::CapturedTargetHidden, lanes) * model::kDecodeRows &&
                  (lanes > 4 || held == rows),
              "a decode step of " + std::to_string(lanes) + " lanes holds other rows than its verify blocks'");
      uint32_t storage = 0;
      for (const auto &p : geometry.target.decodeProjections)
        storage = std::max(storage, plans.decodeLinear().storageRows(rows, p));
      const auto &d = geometry.draft;
      for (const ops::ProjectionShape p : {ops::ProjectionShape{d.hiddenSize, d.targetHiddenSize()},
                                           ops::ProjectionShape{d.contextKvSize(), d.hiddenSize}})
        storage = std::max(storage, plans.draftDecodeLinear().storageRows(rows, p));
      require(storage <= held, "the decode arena does not hold the storage of a step of " + std::to_string(lanes) +
                                   " lanes' plans on Apple" + std::to_string(family));
    }
  }
}

// A MoE plan runs its experts' int8 passes only where the block's experts
// take them (MoeShape::int8Experts), int8 activations are on, the device's
// dense prefill runs the int8 tile and the chunk has more rows than the
// decode tiles' 64. The kernel-selection golden pins the plan lines of each
// device, format set and row count with them on; this checks what it does
// not hold: experts streamed take no int8 passes, and `--activations bf16`
// keeps a chunk the passes would take on its staged experts' exact
// activations, with no int8 scratch.
void checkMoeInt8Passes(const model::LoadedModel &loaded) {
  const auto geometry = model::RuntimeGeometry::from(loaded, kv::Format::Int8, ops::GdnStateFormat::Float32);
  // Experts in the int8 tile's formats take the passes as the start keeps
  // them resident and not as it streams them, whether the geometry comes
  // from the files or from the model as it loaded: both go through
  // RuntimeGeometry::from with the start's streamed parts.
  for (const bool streamed : {false, true}) {
    model::QwenTargetGeometry target = geometry.target;
    target.moeInt8Experts = true;
    const auto planned = model::RuntimeGeometry::from(loaded.descriptor, loaded.draft.layout, target,
                                                      kv::Format::Int8, ops::GdnStateFormat::Float32,
                                                      {.experts = streamed});
    require(planned.target.moeShape().int8Experts == !streamed,
            streamed ? "a start that streams its experts plans their int8 passes"
                     : "a start that keeps its experts resident plans no int8 passes");
  }
  for (uint32_t family : {10U, 11U}) {
    DeviceCapabilities device;
    device.appleGpuFamily = family;
    device.gpuCoreCount = 20;
    const ops::ExecutionPlans plans(device, ops::Int8Activations::Off), int8(device, ops::Int8Activations::On);
    ops::MoeShape shape = geometry.target.moeBlockShape();
    shape.int8Experts = true;
    // Every chunk the passes take with int8 activations on, the largest
    // among them.
    uint32_t taken = 0;
    for (uint32_t rows = 1; rows <= SPLASH_PREFILL_TOKEN_BUDGET; ++rows) {
      if (!int8.moe(shape, geometry.target.moeRouterFormat, rows, ops::StepKind::Prefill).configuration().int8Passes) continue;
      ++taken;
      const ops::MoePlan plan = plans.moe(shape, geometry.target.moeRouterFormat, rows, ops::StepKind::Prefill);
      require(!plan.configuration().int8Passes && plan.workspace().groupedCodesBytes == 0,
              "a MoE plan of Apple" + std::to_string(family) + " with bf16 activations of " +
                  std::to_string(rows) + " rows runs int8 passes");
    }
    require(taken && int8.moe(shape, geometry.target.moeRouterFormat, SPLASH_PREFILL_TOKEN_BUDGET, ops::StepKind::Prefill)
                         .configuration()
                         .int8Passes,
            "no chunk of Apple" + std::to_string(family) + " takes the int8 passes with int8 activations on");
  }
}

// The decode arena's plan counts each buffer it allocates in whole host
// pages, as Metal allocates them and the memory ledger charges them: one
// request's KV capacity rests on it, and a plan of the scratch fields' sum
// left Qwen3.8-Flash-Next's context a KV extent more than a request could
// use (RuntimeArenas.cpp, decodeArenaBytes).
void checkDecodeArenaPages(const model::LoadedModel &loaded) {
  const auto geometry = model::RuntimeGeometry::from(loaded, kv::Format::Int8, ops::GdnStateFormat::Float32);
  for (uint32_t family : {9U, 10U, 11U}) {
    DeviceCapabilities device;
    device.appleGpuFamily = family;
    device.gpuCoreCount = 20;
    const ops::ExecutionPlans plans(device, ops::Int8Activations::On);
    const ops::LinearScratchSize linear = model::DecodeArena::linearScratchSize(geometry, plans);
    uint64_t pages = model::decodeArenaBaseBytes(geometry, plans);
    for (const uint64_t scratch : {model::DecodeArena::gateScratchBytes(geometry, plans), linear.input, linear.sums,
                                   linear.partials, linear.counters, linear.rotated})
      pages += alignUp(scratch);
    require(model::decodeArenaBytes(geometry, plans) == pages,
            "the decode arena's plan does not count each of its buffers in whole host pages");
  }
}

// A runtime's arenas hold its lanes: eight lanes' more than the default's
// four, each lane's tensors, its GDN tape and its logits most, and a step's
// rows to a verify block a lane (ExecutionLimits::stepRows); a runtime of more
// lanes than a step holds is refused.
//
// A default runtime's arenas are pinned to the byte: `decode` and `prefill`,
// Apple9's and Apple10's (Apple11 plans as Apple10), are what
// model::plannedDecodeBytes and plannedPrefillBytes give a runtime of four
// lanes, which the planning of more lanes leaves as they are. A dense
// target's prefill arena holds its two half chunks' group cell, one GDN
// layer's state (3.26 MB on Qwen3.8-27B).
void checkArenasHoldTheirLanes(const char *name, const model::LoadedModel &loaded, std::array<uint64_t, 2> decode,
                               std::array<uint64_t, 2> prefill) {
  const model::StreamedParts parts = model::StreamedParts::loaded(loaded);
  const auto four = model::RuntimeGeometry::from(loaded, kv::Format::Int8, ops::GdnStateFormat::Float32, parts);
  const auto eight = model::RuntimeGeometry::from(loaded, kv::Format::Int8, ops::GdnStateFormat::Float32, parts, 8);
  DeviceCapabilities device;
  device.appleGpuFamily = 10;
  device.gpuCoreCount = 20;
  const ops::ExecutionPlans plans(device, ops::Int8Activations::On);
  require(four.target.lanes == model::ExecutionLimits::defaultLanes &&
              model::plannedDecodeBytes(eight, plans) > model::plannedDecodeBytes(four, plans) &&
              model::plannedPrefillBytes(eight, plans) >= model::plannedPrefillBytes(four, plans),
          std::string(name) + "'s eight lanes' arenas hold no more than four's");
  rejects([&] { static_cast<void>(model::RuntimeGeometry::from(loaded, kv::Format::Int8, ops::GdnStateFormat::Float32,
                                                               parts, model::ExecutionLimits::maximumBatchWidth + 1)); },
          "invalid model runtime lanes", "a runtime of more lanes than a step holds was planned");
  for (uint32_t family : {9U, 10U, 11U}) {
    device.appleGpuFamily = family;
    const ops::ExecutionPlans familyPlans(device, ops::Int8Activations::On);
    const size_t index = family == 9 ? 0 : 1;
    require(model::plannedDecodeBytes(four, familyPlans) == decode[index] &&
                model::plannedPrefillBytes(four, familyPlans) == prefill[index],
            std::string(name) + "'s default arenas are not the base's on Apple" + std::to_string(family));
  }
  // One to eight lanes on every device family: arenas that grow with the
  // lanes, decode tensors of at least the lanes (the default's four at four),
  // and a step of a verify block a lane.
  for (uint32_t family : {9U, 10U, 11U}) {
    device.appleGpuFamily = family;
    const ops::ExecutionPlans familyPlans(device, ops::Int8Activations::On);
    uint64_t fewer = 0;
    for (uint32_t lanes = 1; lanes <= model::ExecutionLimits::maximumBatchWidth; ++lanes) {
      const auto geometry =
          model::RuntimeGeometry::from(loaded, kv::Format::Int8, ops::GdnStateFormat::Float32, parts, lanes);
      const uint64_t decode = model::plannedDecodeBytes(geometry, familyPlans);
      const uint32_t held = model::decodeArenaLanes(geometry, familyPlans);
      require(decode >= fewer && held >= lanes && held <= model::ExecutionLimits::maximumBatchWidth &&
                  (lanes != model::ExecutionLimits::defaultLanes || held == lanes) &&
                  model::decodeTargetRows(geometry, familyPlans).rows >= geometry.target.stepRows(),
              std::string(name) + "'s arenas of " + std::to_string(lanes) + " lanes on Apple" +
                  std::to_string(family) + " do not hold their lanes");
      fewer = decode;
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
  return model::decodeTensorBytes(geometry, ops::ExecutionPlans(device, ops::Int8Activations::On))[static_cast<uint32_t>(tensor)];
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

// A model without a draft has a geometry without one: it captures no target
// layer, and its decode arena holds none of a draft's rows, its context's
// nor its RoPE, nor its gate and linear scratch, while the trees and host
// chains its cycles lay their rows out from, and the rows the target
// verifies, stay as they are with a draft.
template <class Weights>
void checkWithoutDraft() {
  const auto drafted = model::RuntimeGeometry::from(loadedModel<Weights>(), kv::Format::Int8,
                                                    ops::GdnStateFormat::Float32);
  const auto undrafted = model::RuntimeGeometry::from(loadedModel<Weights>({}, model::DFlashDraftLayout{}), kv::Format::Int8,
                                                      ops::GdnStateFormat::Float32);
  require(undrafted.draft == model::DFlashDraftLayout{} && !undrafted.target.captureLayerCount &&
              drafted.target.captureLayerCount,
          "a geometry without a draft captured target layers");
  for (const auto tensor : {model::DecodeTensor::DraftHidden0, model::DecodeTensor::DraftProposalQkv,
                            model::DecodeTensor::DraftFinalHidden, model::DecodeTensor::DraftRopeCos,
                            model::DecodeTensor::DraftQueryKeys, model::DecodeTensor::ContextKv,
                            model::DecodeTensor::CapturedTargetHidden, model::DecodeTensor::SelectorHidden})
    require(decodeBytes(drafted, tensor) && !decodeBytes(undrafted, tensor),
            "a decode arena without a draft holds a draft's rows");
  for (const auto tensor : {model::DecodeTensor::InputTokens, model::DecodeTensor::TreeRows,
                            model::DecodeTensor::VerifySources, model::DecodeTensor::Candidates,
                            model::DecodeTensor::ChainTokens, model::DecodeTensor::ChainRates,
                            model::DecodeTensor::DraftTreeTokens, model::DecodeTensor::DraftTrees,
                            model::DecodeTensor::DraftTreeScores, model::DecodeTensor::DraftTreeDraws,
                            model::DecodeTensor::DraftTreeCaps, model::DecodeTensor::DraftTreeSizes})
    require(decodeBytes(undrafted, tensor) && decodeBytes(undrafted, tensor) == decodeBytes(drafted, tensor),
            "a decode arena without a draft lays out or verifies otherwise");
  DeviceCapabilities device;
  device.appleGpuFamily = 10;
  device.gpuCoreCount = 20;
  const ops::ExecutionPlans plans(device, ops::Int8Activations::On);
  require(model::decodeArenaBytes(undrafted, plans) < model::decodeArenaBytes(drafted, plans) &&
              model::plannedPrefillBytes(undrafted, plans) < model::plannedPrefillBytes(drafted, plans),
          "the arenas of a model without a draft are not smaller than with one");
}

// A runtime whose K/V spills (one given the readers of spilled K/V,
// model::RuntimeContext) runs its prefill commands as groups, which read an
// attention layer's spilled history once each, and its decode arena holds
// where its verify's listed blocks are: a lane's list, a verify block's 16
// rows of 512 blocks and the block of each row's tail. The K/V of a target
// without a block index cannot spill.
void checkSpilledKvGeometry() {
  const auto flash = loadedModel<model::Qwen4ExpWeights>();
  const auto resident = model::RuntimeGeometry::from(flash, kv::Format::Int8, ops::GdnStateFormat::Float32);
  const auto spilled = model::RuntimeGeometry::from(flash, kv::Format::Int8, ops::GdnStateFormat::Float32,
                                                    model::StreamedParts::loaded(flash, true));
  require(resident.prefillChunks == 1 && !resident.spilledKv &&
              spilled.prefillChunks == model::kPrefillGroupChunks && spilled.spilledKv == test::sparseKvShape(),
          "a runtime's prefill groups do not follow whether its K/V spills");
  require(!decodeBytes(resident, model::DecodeTensor::Locations) &&
              decodeBytes(spilled, model::DecodeTensor::Locations) == uint64_t{16} * (512 + 1) * sizeof(uint32_t),
          "a runtime's decode arena does not hold its spilled blocks' locations exactly when its K/V spills");
  for (const model::LoadedModel &loaded :
       {loadedModel<model::Qwen3_8Weights>(), loadedModel<model::Qwen3_6MoeWeights>()})
    rejects([&] {
      static_cast<void>(model::RuntimeGeometry::from(loaded, kv::Format::Int8, ops::GdnStateFormat::Float32,
                                                     model::StreamedParts::loaded(loaded, true)));
    }, "invalid model runtime geometry", "the K/V of a target without a block index was let spill");
}

// A target without GDN layers (model::GdnStateLayout without layers) has no
// GDN scratch: its arenas leave out exactly the GDN layers' tensors, the
// decode tape among them, and a prefill group's scratch cell, whose cell
// holds nothing, and size every other tensor as a hybrid target's.
void checkArenasWithoutGdnLayers() {
  const auto sparse = loadedModel<model::Qwen3_6MoeWeights>();
  // Streamed experts make a prefill command a group, which a hybrid target's
  // scratch cell serves.
  model::StreamedParts parts = model::StreamedParts::loaded(sparse);
  parts.experts = true;
  const auto hybrid = model::RuntimeGeometry::from(sparse, kv::Format::Int8, ops::GdnStateFormat::Float32, parts);
  auto attentionOnly = hybrid;
  attentionOnly.target.stateLayout = {};
  require(hybrid.prefillChunks > 1 && attentionOnly.target.stateLayout.valid() &&
              !attentionOnly.target.stateLayout.cellBytes(),
          "the arenas' geometries are not a prefill group's with and without GDN layers");
  DeviceCapabilities device;
  device.appleGpuFamily = 10;
  device.gpuCoreCount = 20;
  const ops::ExecutionPlans plans(device, ops::Int8Activations::On);
  // Whether `bytes` are the hybrid's but for the GDN tensors, which only the
  // hybrid's arena holds.
  const auto withoutGdn = [](const auto &hybridBytes, const auto &bytes, const auto &gdnTensors) {
    using Tensor = typename std::decay_t<decltype(gdnTensors)>::value_type;
    for (uint32_t index = 0; index < hybridBytes.size(); ++index) {
      const bool gdn = std::ranges::find(gdnTensors, static_cast<Tensor>(index)) != gdnTensors.end();
      if (gdn ? !hybridBytes[index] || bytes[index] : bytes[index] != hybridBytes[index]) return false;
    }
    return true;
  };
  using model::DecodeTensor;
  using model::PrefillTensor;
  using model::TargetTensor;
  constexpr std::array gdnTarget{TargetTensor::GdnPacked, TargetTensor::GdnQueries, TargetTensor::GdnKeys,
                                 TargetTensor::GdnValues, TargetTensor::GdnDecay,   TargetTensor::GdnBeta,
                                 TargetTensor::Recurrent, TargetTensor::GdnHidden,  TargetTensor::GdnOutput};
  constexpr std::array gdnPrefill{PrefillTensor::GroupCell};
  constexpr std::array gdnDecode{DecodeTensor::GdnTape};
  for (const auto rows : {model::prefillTargetRows, model::decodeTargetRows})
    require(withoutGdn(model::targetTensorBytes(hybrid.target, rows(hybrid, plans)),
                       model::targetTensorBytes(attentionOnly.target, rows(attentionOnly, plans)), gdnTarget),
            "the target step tensors without GDN layers are not a hybrid's without its GDN scratch");
  require(withoutGdn(model::prefillTensorBytes(hybrid, plans), model::prefillTensorBytes(attentionOnly, plans),
                     gdnPrefill),
          "the prefill arena without GDN layers is not a hybrid's without its GDN scratch");
  require(withoutGdn(model::decodeTensorBytes(hybrid, plans), model::decodeTensorBytes(attentionOnly, plans),
                     gdnDecode),
          "the decode arena without GDN layers is not a hybrid's without its GDN scratch");
}

// A dense target's prefill commands run in two chunks of half the rows, the steps its Neural Engine split pipelines
// (ops::AneFfn::firstStepRows): a command still holds kPrefillRows rows, its group tensors hold no more than one chunk
// of them would, and its group cell holds one GDN layer's state and the auxiliary, which the second chunk reads within
// the layer the first wrote it (PrefillArena::groupCell). A MoE target's commands stay one chunk.
void checkDenseHalves() {
  const auto dense = model::RuntimeGeometry::from(loadedModel<model::Qwen3_8Weights>(), kv::Format::Int8,
                                                  ops::GdnStateFormat::Float32);
  const auto moe = model::RuntimeGeometry::from(loadedModel<model::Qwen3_6MoeWeights>(), kv::Format::Int8,
                                                ops::GdnStateFormat::Float32);
  require(dense.prefillChunks == 2 && dense.prefillChunkRows == model::kPrefillRows / 2 &&
              model::commandLimits(dense, 0).commandRows == model::kPrefillRows && moe.prefillChunks == 1 &&
              model::commandLimits(moe, 0).commandRows == model::kPrefillRows,
          "a dense target's commands are not two chunks of half the rows, or a MoE target's not one");
  DeviceCapabilities device;
  device.appleGpuFamily = 10;
  device.gpuCoreCount = 20;
  const ops::ExecutionPlans plans(device, ops::Int8Activations::On);
  auto single = dense;
  single.prefillChunks = 1;
  single.prefillChunkRows = model::kPrefillRows;
  const auto halves = model::targetTensorBytes(dense.target, model::prefillTargetRows(dense, plans)),
             whole = model::targetTensorBytes(single.target, model::prefillTargetRows(single, plans));
  for (uint32_t index = 0; index < halves.size(); ++index)
    require(halves[index] >= whole[index] && halves[index] - whole[index] < 2 * kHostPageBytes,
            "a dense target's tensor " + std::to_string(index) + " holds more than a chunk of its command's rows");
  const auto prefill = model::prefillTensorBytes(dense, plans);
  const model::GdnStateLayout &state = dense.target.stateLayout;
  require(prefill[static_cast<uint32_t>(model::PrefillTensor::GroupCell)] ==
              state.convolutionLayerBytes() + state.recurrentLayerBytes() + alignUp(state.auxiliaryBytes),
          "a dense target's group cell does not hold one GDN layer's state");
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
  // Every family's layout, the 122B's among them, passes the layout check.
  model::forEachFamily([](const auto &family) { model::requireQwenLayout(family); });
  // A layout without GDN layers sets none of the GDN dimensions.
  auto strayGdn = model::FlintLayout{};
  strayGdn.gdnKeyHeads = 16;
  rejects([&] { model::requireQwenLayout(strayGdn); }, "Qwen target layout is inconsistent",
          "a layout without GDN layers was accepted with a GDN dimension");
  // The dense layout is checked like the sparse one: its convolution width
  // against its GDN heads.
  const model::Qwen3_8Layout dense;
  model::requireQwenLayout(dense);
  auto convolutionMismatch = dense;
  convolutionMismatch.convolutionDimension += dense.gdnHeadDimension;
  rejects([&] { model::requireQwenLayout(convolutionMismatch); }, "Qwen target layout is inconsistent",
          "an inconsistent dense layout was accepted");
}

// A family's drafts' subset of its vocabulary (draftVocabulary) lies in it,
// whole tiles of its head's rows, or the layout is refused before loading,
// rather than its logits width wrapping: the Qwen subset over a vocabulary
// of 1,024 tokens, a part of half a tile and parts that cross. The whole
// vocabulary fits any.
void checkDraftVocabulary() {
  const model::Qwen3_8Layout shipped;
  auto small = shipped;
  small.vocabularySize = 1024;
  rejects([&] { model::requireQwenLayout(small); }, "Qwen target layout is inconsistent",
          "a subset past the vocabulary was accepted");
  small.draftVocabulary = {};
  model::requireQwenLayout(small);
  auto halfTile = shipped;
  halfTile.draftVocabulary.leading += QUANT_TILE_ROWS / 2;
  rejects([&] { model::requireQwenLayout(halfTile); }, "Qwen target layout is inconsistent",
          "a subset of part of a tile was accepted");
  auto crossed = shipped;
  crossed.draftVocabulary = {512, 448};
  rejects([&] { model::requireQwenLayout(crossed); }, "Qwen target layout is inconsistent",
          "a subset whose parts cross was accepted");
}

} // namespace

// A prefill chunk's attention plan follows the history its chunk committed:
// a short chunk after a long cached prefix shares its tiles' pages out over
// more splits than the same chunk after none.
void checkPrefillAttentionHistory() {
  const auto geometry = model::RuntimeGeometry::from(loadedModel<model::Qwen3_6MoeWeights>(), kv::Format::Int8,
                                                     ops::GdnStateFormat::Float32);
  DeviceCapabilities device;
  device.appleGpuFamily = 10;
  device.gpuCoreCount = 20;
  const ops::ExecutionPlans plans(device, ops::Int8Activations::On);
  const uint32_t heads = geometry.target.attentionQueryHeads;
  const kv::Layout &layout = geometry.target.kvLayout;
  const auto splitsAfter = [&](uint32_t history) {
    const std::array chunk{ops::PagedAttention::chunkParams(history, 300, 320, (history + 300 + 31) / 32)};
    return plans.attention(chunk, heads, layout, plans.sequenceAttentionWorkspace(heads, layout))
        .sequences()[0]
        .splits;
  };
  require(splitsAfter(65'536) > splitsAfter(0), "a chunk's attention plan does not follow its history");
}

int main() {
  try {
    checkPrefillAttentionHistory();
    checkUnsizedProjection();
    checkCaptureLayers();
    checkGdnWidths();
    checkDraftVocabulary();
    checkGateUpLayers();
    checkYarnGeometry();
    checkIndexTables();
    checkSpilledKvGeometry();
    checkArenasWithoutGdnLayers();
    checkDenseHalves();
    checkPrefillInt8Scratch();
    checkWithoutDraft<model::Qwen3_8Weights>();
    checkWithoutDraft<model::Qwen3_6MoeWeights>();
    checkWithoutDraft<model::Qwen4ExpWeights>();
    checkLaneScratch(loadedModel<model::Qwen3_8Weights>());
    checkLaneScratch(loadedModel<model::Qwen3_6MoeWeights>());
    checkLaneScratch(loadedModel<model::Qwen3_6MoeWeights>(model::kQwen3_5_122BLayout, test::qwen3_5_122BDraft()));
    checkLaneScratch(loadedModel<model::Qwen3_8Weights>({}, model::DFlashDraftLayout{}));
    checkArenasHoldTheirLanes("Qwen3.8-27B", loadedModel<model::Qwen3_8Weights>(), {450'576'384, 434'864'128},
                              {749'715'456, 733'609'984});
    checkArenasHoldTheirLanes("Qwen3.6-35B-A3B", loadedModel<model::Qwen3_6MoeWeights>(), {279'281'664, 273'874'944},
                              {539'017'216, 533'446'656});
    checkArenasHoldTheirLanes("Qwen3.8-Flash-Next", loadedModel<model::Qwen4ExpWeights>(),
                              {409'796'608, 403'963'904}, {1'124'614'144, 1'118'781'440});
    checkDecodeArenaPages(loadedModel<model::Qwen3_8Weights>());
    checkDecodeArenaPages(loadedModel<model::Qwen3_6MoeWeights>());
    checkDecodeArenaPages(loadedModel<model::Qwen3_8Weights>({}, model::DFlashDraftLayout{}));
    checkDecodeStepRows(loadedModel<model::Qwen3_8Weights>());
    checkDecodeStepRows(loadedModel<model::Qwen3_6MoeWeights>());
    checkDecodeStepMoe(loadedModel<model::Qwen3_6MoeWeights>());
    checkMoeInt8Passes(loadedModel<model::Qwen3_6MoeWeights>());
    checkDecodeStepMoe(loadedModel<model::Qwen3_6MoeWeights>(model::kQwen3_5_122BLayout, test::qwen3_5_122BDraft()));
    checkDecodeStepMoe(loadedModel<model::Qwen4ExpWeights>());
    std::cout << "model execution plans: PASS (three paired geometries, three without a draft)\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
