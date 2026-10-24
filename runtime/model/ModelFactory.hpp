#pragma once

#include "ops/Vision.hpp"
#include "DFlashDraft.hpp"
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
#include <string>
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
// many bytes (model/ExpertStore.hpp, --expert-memory). Other models load as
// without them. Streamed weights read on `readers`, the engine's one pool of
// SSD readers, which KV reads share.
struct ModelLoadOptions final {
  uint64_t streamedTableBytes = 0;
  uint64_t expertMemoryBytes = 0;
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
  [[nodiscard]] CompositeStateLayout stateLayout(ops::GdnStateFormat format) const noexcept {
    auto layout = descriptor.stateLayout;
    layout.target.format = format;
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

// How a decode cycle chooses the rows of the draft's block the target
// verifies for each lane (--verify-rows), never past the lane's retention
// limit: the tokens its output budget leaves, at most the block's rows.
enum class VerifyRows : uint8_t {
  // The verify budget's rows on a MoE target, every row on a dense one.
  Auto,
  // Every row, so that a greedy or seeded request repeats its output.
  All,
};

// Whether the decode cycles of `model` verify the rows the verify budget
// chooses (model::VerifyBudget, ops::DraftSelector::addVerifyRows) under
// `rows`. Each row a MoE target verifies routes its own experts, which a
// rejected row reads for nothing; a dense target reads its weights once for
// all of a cycle's rows.
[[nodiscard]] bool budgetsVerifyRows(const LoadedModel &model, VerifyRows rows) noexcept;

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
};

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

// The FFN layers of a dense target, which the prefill FFN's Neural Engine
// split (ops::AneFfn) may take; none for another target.
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
// whatever streams of it as it loaded: one load plans them for every part
// that may stream (engine::planStreamedMemory).
[[nodiscard]] ModelMemoryPlan
plannedRuntimeMemory(const LoadedModel &model,
                     const ops::ExecutionPlans &operators,
                     kv::Format kvFormat, ops::GdnStateFormat stateFormat, StreamedParts parts);
[[nodiscard]] std::unique_ptr<RuntimeModel>
createRuntime(RuntimeContext context);

} // namespace splash::model
