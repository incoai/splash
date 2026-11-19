#pragma once

#include "ops/Vision.hpp"
#include "DFlashDraft.hpp"
#include "ExpertChunkLayout.hpp"
#include "ModelDescriptor.hpp"
#include "Qwen3_6Moe.hpp"
#include "Qwen3_8.hpp"
#include "QwenVision.hpp"
#include "ops/PageStorage.hpp"
#include "ops/ExecutionPlans.hpp"

#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace splash::ops {
class AneFfn;
} // namespace splash::ops

namespace splash::model {

class KvBlockCache;
class KvHistoryStream;
class QwenStateStorage;
class SsdReaders;
class VisionLoader;

using TargetWeights = std::variant<Qwen3_8Weights, Qwen3_6MoeWeights, Qwen4ExpWeights, FlintWeights>;

// How a model loads beside its descriptor: with streamedTableBytes, the rows
// of a table the target's GGUF holds (model/ModelSizing.hpp) stream from its
// file through a row cache of that many bytes instead of loading into memory
// (model/SparseRowStore.hpp, --ple-memory). With expertMemoryBytes, a MoE
// target's routed experts stream from their files through a store of that
// many bytes at its cap, which starts at its floor and gives memory back
// down to it: expertFloorBytes, or where that is zero expertFloorShare of its
// slots, the share the model's plan sets (model/ExpertStore.hpp,
// ExpertStoreOptions, --expert-memory). Other models load as without them.
// Streamed weights read on `readers`, the engine's one pool of SSD readers,
// which KV reads share.
struct ModelLoadOptions final {
  uint64_t streamedTableBytes = 0;
  uint64_t expertMemoryBytes = 0;
  uint64_t expertFloorBytes = 0;
  double expertFloorShare = kExpertFloorShare;
  std::shared_ptr<SsdReaders> readers;
};

struct LoadedModel final {
  ModelDescriptor descriptor;
  // The memory of every image the weights below are views of.
  std::shared_ptr<WeightImages> images;
  TargetWeights target;
  DFlashDraftWeights draft;
  QwenVisionWeights vision;
  std::string manifestFingerprintSha256;

  [[nodiscard]] const std::string &name() const noexcept {
    return descriptor.name;
  }
  [[nodiscard]] kv::Layout targetKvLayout(kv::Format format) const noexcept {
    auto layout = descriptor.targetKvLayout;
    layout.format = format;
    return layout;
  }
  [[nodiscard]] CompositeStateLayout stateLayout(ops::GdnStateFormat format, bool cachedCodes = true) const noexcept {
    auto layout = descriptor.stateLayout;
    layout.target.format = format;
    layout.target.cachedCodes = cachedCodes;
    return layout;
  }
  [[nodiscard]] uint32_t maximumContextTokens() const noexcept {
    return descriptor.capabilities.maximumContextTokens;
  }
  [[nodiscard]] uint64_t targetActualAllocatedBytes() const noexcept {
    return std::visit([](const auto &weights) {
      return weights.actualAllocatedBytes;
    }, target);
  }
  [[nodiscard]] const std::string &targetManifestFingerprint() const noexcept {
    return std::visit([](const auto &weights) -> const std::string & {
      return weights.manifestFingerprintSha256;
    }, target);
  }
  // The store streamed routed experts are read through, or null.
  [[nodiscard]] ExpertStore *expertStore() const noexcept {
    return std::visit([](const auto &weights) -> ExpertStore * { return weights.experts.get(); }, target);
  }
  // The store a streamed table is read through, or null.
  [[nodiscard]] SparseRowStore *streamedTable() const noexcept {
    return std::visit([](const auto &weights) -> SparseRowStore * { return weights.streamedTable.get(); }, target);
  }
  [[nodiscard]] std::span<const WeightFileRecord> targetFiles() const noexcept {
    return std::visit([](const auto &weights) ->
                          std::span<const WeightFileRecord> {
      return weights.files;
    }, target);
  }
};

// How a decode cycle chooses its rows (--verify-rows), never past a lane's
// retention limit: the tokens its output budget leaves, at most a verify
// block's rows.
enum class VerifyRows : uint8_t {
  // The step costs': a step's rows by the costs the start measures
  // (StepCosts::choose), every row of which a lane holds verified.
  Auto,
  // Every row of a draft's block a lane, so that a greedy or seeded request
  // repeats its output.
  All,
};

// Model execution resources, which the engine assembles. What a request's
// start allocates is admitted through the state storage.
struct RuntimeContext final {
  metal::MetalBackend &backend;
  const LoadedModel &model;
  kv::PageStorage &kvPages;
  QwenStateStorage &stateStorage;
  const ops::ExecutionPlans &operators;
  // The prefill FFN's Neural Engine split (engine::startAneFfn), if any.
  ops::AneFfn *aneFfn = nullptr;
  // The readers of a spill-capable target's spilled K/V, both or neither:
  // the hot block cache its verifies resolve through, and the stream of its
  // prefills' history (StreamedParts::kv).
  KvBlockCache *blockCache = nullptr;
  KvHistoryStream *historyStream = nullptr;
  VerifyRows verifyRows = VerifyRows::Auto;
  // What a start keeps of the step costs its decode warmups measure on this
  // Mac (Runtime::warmupDecodeBatch, StepCosts::text): the line a lane
  // count's were remembered as, if any, which the runtime reads when it is
  // made, remembering one and forgetting one; none where every start
  // measures them.
  std::function<std::optional<std::string>(uint32_t lanes)> recallStepCosts{};
  std::function<void(uint32_t lanes, std::string_view line)> rememberStepCosts{};
  std::function<void(uint32_t lanes)> forgetStepCosts{};
};

// What the step costs a start's decode warmups measure follow
// (RuntimeContext::recallStepCosts), a key to keep them under but for their
// lane count: the Mac and its macOS, the build `buildId`, the models'
// manifests, the KV and state formats, the activations, the expert store's
// cap and floor, and the bytes of a streamed table's row cache.
[[nodiscard]] std::string stepCostsKey(const LoadedModel &model, const kv::PageStorage &kvPages,
                                       const QwenStateStorage &stateStorage, const ops::ExecutionPlans &operators,
                                       const metal::MetalBackend &backend, std::string_view buildId);

// Validates only the interface between independently defined target and draft
// architectures. Each architecture validates its own tensor and state layout.
void requireCompatibleModel(const LoadedModel &model);

// The rows of a family's streamed table that a step stages and prefetches,
// beside Qwen4Exp.hpp's: the Qwen3.5 families and Flint have no table to
// stream.
[[nodiscard]] SparseRowStoreOptions streamedTableRows(const Qwen3_8Layout &layout);
[[nodiscard]] SparseRowStoreOptions streamedTableRows(const Qwen3_6MoeLayout &layout);
[[nodiscard]] SparseRowStoreOptions streamedTableRows(const FlintLayout &layout);

// The vision role's source, planned; null for a model without vision.
[[nodiscard]] std::unique_ptr<VisionLoader> planVisionLoader(const std::filesystem::path &root,
                                                             const ModelDescriptor &descriptor);
// The vision role written by `loader`; empty weights for a model without
// vision (no loader).
[[nodiscard]] QwenVisionWeights loadVisionWeights(metal::MetalBackend &backend, WeightImages &images,
                                                  const VisionLoader *loader);

// Production loading is selected by the validated descriptor. There
// is one shared engine and DFlash controller; only model execution differs.
[[nodiscard]] LoadedModel
loadModel(metal::MetalBackend &backend,
                 const std::filesystem::path &root,
                 const ModelDescriptor &descriptor,
                 const ModelLoadOptions &options = {});
// The geometry of the target in `root` (qwenTargetGeometry) as its files'
// headers give it, before any weight loads: the target's loader reads its
// images reserved, of no memory and unwritten (WeightImages loaded while
// released), so every projection has its formats and sizes as a load gives
// them. Its routed experts read with their layers' images, which make no
// store and stay far below the largest buffer, which a reservation is held
// to too (Flash-Next Q2_0's largest, 743 MiB, of an M5 Max's 21 GiB); with
// `streamedTableBytes`, its table streams through a row cache of that many
// bytes on `readers`, as its whole image may exceed that buffer (Flash-Next's
// 27,466 MiB).
[[nodiscard]] QwenTargetGeometry plannedTargetGeometry(metal::MetalBackend &backend,
                                                       const std::filesystem::path &root,
                                                       const ModelDescriptor &descriptor,
                                                       uint64_t streamedTableBytes,
                                                       std::shared_ptr<SsdReaders> readers);

// The FFN layers of a target whose every layer's FFN is dense (denseFfn),
// which the prefill FFN's Neural Engine split (ops::AneFfn) may take; none
// for another target.
[[nodiscard]] std::vector<ops::SwiGluProjections> aneFfnLayers(const LoadedModel &model);
// Runs `use` on a prefill arena allocated for the call: on the dense FFN's
// buffers of a full chunk and the hidden rows layers alternate between.
void withPrefillArena(
    metal::MetalBackend &backend, const LoadedModel &model, const ops::ExecutionPlans &operators, kv::Format format,
    const std::function<void(const ops::PrefillFfnBuffers &, const std::array<metal::MetalBuffer, 2> &)> &use);
// What of a model streams from the SSD, which its arenas rest on: a target
// whose routed experts stream runs its prefill commands as groups
// (RuntimeGeometry::prefillChunks), the step of one whose table streams
// stages its rows, of tableRowBytes each (zero: the table loads), and a
// runtime whose K/V spills (RuntimeContext's readers) runs groups too, which
// read an attention layer's spilled history once each, and holds where its
// verifies' listed blocks are. loaded() is what streams of `model` as it
// loaded, and its K/V when `kv` says it spills.
struct StreamedParts final {
  bool experts = false;
  uint64_t tableRowBytes = 0;
  bool kv = false;

  [[nodiscard]] static StreamedParts loaded(const LoadedModel &model, bool kv = false) noexcept;
};
// The lane state and arenas of `model`'s geometry with `parts` streaming,
// whatever streams of it as it loaded.
[[nodiscard]] ModelMemoryPlan
plannedRuntimeMemory(const LoadedModel &model,
                     const ops::ExecutionPlans &operators,
                     kv::Format kvFormat, ops::GdnStateFormat stateFormat, StreamedParts parts);
// Those of a model of `descriptor` whose target's files give `target`
// (plannedTargetGeometry) with `parts` streaming, before it loads: they plan
// the stores of every part that may stream (engine::planStreamedMemory).
[[nodiscard]] ModelMemoryPlan plannedRuntimeMemory(const ModelDescriptor &descriptor,
                                                   const QwenTargetGeometry &target,
                                                   const ops::ExecutionPlans &operators, kv::Format kvFormat,
                                                   ops::GdnStateFormat stateFormat, StreamedParts parts);
[[nodiscard]] std::unique_ptr<RuntimeModel>
createRuntime(RuntimeContext context);

} // namespace splash::model
