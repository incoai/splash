#include "flash/FlashBatchMTPForward.hpp"
#include "flash/FlashAffine.hpp"
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashDenseSmallRows.hpp"
#include "flash/FlashBF16Q8Head.hpp"
#include "flash/FlashGreedyGPU.hpp"
#include "flash/FlashHC.hpp"
#include "flash/FlashHCFused.hpp"
#include "flash/FlashMoE.hpp"
#include "flash/FlashQSA.hpp"
#include "flash/FlashQSAFast.hpp"
#include "flash/FlashQSAMPP.hpp"
#include "flash/FlashMTPStateInternal.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashForward.h"
#include "metal/abi/FlashMTP.h"
#include "metal/abi/FlashBatchMTP.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

namespace splash::flash {
namespace {
constexpr uint32_t kWidth = 2560, kHyper = 10240, kSelections = 10;
constexpr uint64_t kAlignment = 16384;
bool enabled(const char *name) {
  const char *value = std::getenv(name);
  if (!value) return false;
  if (std::string_view(value) == "1") return true;
  if (std::string_view(value) == "0") return false;
  throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}
uint64_t rounded(uint64_t bytes) {
  if (!bytes || bytes > std::numeric_limits<uint64_t>::max() - kAlignment + 1)
    throw std::overflow_error("Flash batch MTP allocation extent overflows");
  return (bytes + kAlignment - 1) & ~(kAlignment - 1);
}
void validate(uint32_t capacity, uint32_t lanes, uint32_t rowsPerLane) {
  if (!capacity || capacity > 262144 || !lanes || lanes > 4 ||
      !rowsPerLane || rowsPerLane > 128)
    throw std::invalid_argument("Flash batch MTP requires capacity1..262144, lanes1..4, rows/lane1..128");
}
enum class Scratch : size_t {
  TokenIDs, Embedding, EmbeddingNormalized, EmbeddingProjected,
  HiddenNormalized, HiddenProjected, Hyper, HCNormalized, HCDown, HCUp,
  HCRawInjection, HCInjectionWeights, Mixed, Branch, QProjection, KProjection,
  VProjection, IndexProjection, AttentionOutput, Router, ExpertIDs, RouteWeights,
  ExpertGate, ExpertUp, ExpertIntermediate, ExpertDown, SharedGate, SharedUp,
  SharedIntermediate, SharedDown, SharedGateLogit, HeadLogits, Diagnostics, LaneOffsets, HeadInput, Count,
};
constexpr size_t kScratchCount = static_cast<size_t>(Scratch::Count);
} // namespace

struct FlashBatchMTPForward::Impl final {
  FlashMTPForward &owner;
  metal::MetalBackend &backend;
  const FlashWeights &weights;
  const FlashDescriptor descriptor;
  const uint32_t capacity, maximumLanes, maximumRowsPerLane, maximumRows;
  const bool fuseHC;
  const bool gpuGreedy;
  const FlashDenseCache *denseCache;
  std::optional<FlashTensor> cachedVocabulary;
  std::unique_ptr<FlashDenseSmallRowsWorkspace> vocabularyWorkspace;
  std::unique_ptr<FlashBF16Q8Head> registerVocabulary;
  std::optional<FlashGreedyGPUWorkspace> greedyWorkspace;
  metal::MetalBuffer greedyResults;
  std::array<metal::MetalBuffer, kScratchCount> scratch;
  FlashQSAWorkspace qsaWorkspace;
  FlashQSAFastWorkspace qsaFastWorkspace;
  uint32_t minimumLogicalRows = 0;
  uint32_t maximumLogicalRows = 0;
  uint64_t workspaceBytes = 0;
  uint64_t registerVocabularyCommands = 0, registerVocabularyRows = 0;
  Impl(FlashMTPForward &head, uint32_t lanes, uint32_t rowsPerLane,
       const FlashTensor *vocabulary)
      : owner(head), backend(head.batchBackend()), weights(head.batchWeights()),
        descriptor(weights.descriptor()), capacity(head.batchCapacity()),
        maximumLanes(lanes), maximumRowsPerLane(rowsPerLane),
        maximumRows(lanes * rowsPerLane), fuseHC(head.batchFuseHC()),
        gpuGreedy(enabled("SPLASH_FLASH_GPU_GREEDY")),
        denseCache(head.batchDenseCache()) {
    validate(capacity, lanes, rowsPerLane);
    if (vocabulary) {
      const uint64_t bytes = uint64_t{descriptor.vocabularySize} * kWidth * 2;
      if (vocabulary->dtype != FlashDType::BF16 ||
          vocabulary->shape != std::vector<uint64_t>{descriptor.vocabularySize, kWidth} ||
          vocabulary->logicalBytes != bytes || !vocabulary->buffer ||
          vocabulary->buffer.sizeBytes() < bytes)
        throw std::invalid_argument("Flash batch head shared vocabulary must be BF16[248320,2560]");
      // Retain the existing immutable allocation. This is a view/reference,
      // not a second weight cache or a second allocation-ledger charge.
      cachedVocabulary = *vocabulary;
    }
    const uint64_t before = backend.memoryStats().allocatedBytes;
    if (gpuGreedy) {
      const uint32_t greedyRows = std::min<uint32_t>(maximumRows, kFlashGreedyGPUMaximumRows);
      greedyWorkspace = allocateGreedyGPUWorkspace(backend, greedyRows,
          descriptor.vocabularySize, metal::BufferStorage::Private);
      greedyResults = backend.allocateBuffer(
          rounded(uint64_t{greedyRows} * sizeof(FlashGreedyGPURowResult)),
          metal::BufferStorage::Shared, "flash-batch-mtp-greedy-row-results");
    }
    if (cachedVocabulary && maximumLanes > 1)
      vocabularyWorkspace = std::make_unique<FlashDenseSmallRowsWorkspace>(backend, kWidth);
    if (flashBF16Q8HeadEnabled() && vocabularyWorkspace)
      registerVocabulary = std::make_unique<FlashBF16Q8Head>(backend, weights,
          vocabularyWorkspace->paddedInput());
    const auto allocate = [&](Scratch slot, uint64_t bytes, const char *label) {
      scratch[static_cast<size_t>(slot)] = backend.allocateBuffer(
          rounded(bytes), metal::BufferStorage::Shared, label);
    };
    const auto bf = [&](Scratch slot, uint32_t width, const char *label) {
      allocate(slot, uint64_t{maximumRows} * width * 2, label);
    };
    allocate(Scratch::TokenIDs, uint64_t{maximumRows} * 8, "flash-batch-mtp-token-ids");
    bf(Scratch::Embedding, kWidth, "flash-batch-mtp-embedding");
    bf(Scratch::EmbeddingNormalized, kWidth, "flash-batch-mtp-embedding-normalized");
    bf(Scratch::EmbeddingProjected, kWidth, "flash-batch-mtp-embedding-projected");
    bf(Scratch::HiddenNormalized, kHyper, "flash-batch-mtp-hidden-normalized");
    bf(Scratch::HiddenProjected, kHyper, "flash-batch-mtp-hidden-projected");
    bf(Scratch::Hyper, kHyper, "flash-batch-mtp-hyper-state");
    bf(Scratch::HCNormalized, kHyper, "flash-batch-mtp-hc-normalized");
    bf(Scratch::HCDown, 320, "flash-batch-mtp-hc-down");
    bf(Scratch::HCUp, kHyper, "flash-batch-mtp-hc-up");
    bf(Scratch::HCRawInjection, 4, "flash-batch-mtp-hc-raw-injection");
    bf(Scratch::HCInjectionWeights, 4, "flash-batch-mtp-hc-injection-weights");
    bf(Scratch::Mixed, kWidth, "flash-batch-mtp-mixed-input");
    bf(Scratch::Branch, kWidth, "flash-batch-mtp-branch");
    bf(Scratch::QProjection, 12288, "flash-batch-mtp-q-projection");
    bf(Scratch::KProjection, 512, "flash-batch-mtp-k-projection");
    bf(Scratch::VProjection, 512, "flash-batch-mtp-v-projection");
    bf(Scratch::IndexProjection, 640, "flash-batch-mtp-index-projection");
    bf(Scratch::AttentionOutput, 6144, "flash-batch-mtp-attention-output");
    bf(Scratch::Router, 512, "flash-batch-mtp-router");
    allocate(Scratch::ExpertIDs, uint64_t{maximumRows} * kSelections * 8, "flash-batch-mtp-expert-ids");
    bf(Scratch::RouteWeights, kSelections, "flash-batch-mtp-route-weights");
    bf(Scratch::ExpertGate, kSelections * 640, "flash-batch-mtp-expert-gate");
    bf(Scratch::ExpertUp, kSelections * 640, "flash-batch-mtp-expert-up");
    bf(Scratch::ExpertIntermediate, kSelections * 640, "flash-batch-mtp-expert-intermediate");
    bf(Scratch::ExpertDown, kSelections * kWidth, "flash-batch-mtp-expert-down");
    bf(Scratch::SharedGate, 640, "flash-batch-mtp-shared-gate");
    bf(Scratch::SharedUp, 640, "flash-batch-mtp-shared-up");
    bf(Scratch::SharedIntermediate, 640, "flash-batch-mtp-shared-intermediate");
    bf(Scratch::SharedDown, kWidth, "flash-batch-mtp-shared-down");
    bf(Scratch::SharedGateLogit, 1, "flash-batch-mtp-shared-gate-logit");
    allocate(Scratch::HeadLogits,
        uint64_t{std::min(maximumRows, uint32_t{16})} * descriptor.vocabularySize * 2,
        "flash-batch-mtp-head-logits");
    allocate(Scratch::Diagnostics, 4, "flash-batch-mtp-diagnostics");
    qsaWorkspace = allocateQSAWorkspace(backend, maximumRowsPerLane, capacity);
    if (head.batchQSAF32())
      qsaFastWorkspace = head.batchQSAMPP()
          ? allocateQSAOnlineMPPWorkspace(backend, maximumRowsPerLane, 32)
          : allocateQSAFastWorkspace(backend, maximumRowsPerLane, 4);
    allocate(Scratch::LaneOffsets, uint64_t{maximumLanes + 1} * 4, "flash-batch-mtp-lane-offsets");
    allocate(Scratch::HeadInput, uint64_t{maximumLanes} * kWidth * 2, "flash-batch-mtp-last-head-input");
    const uint64_t after = backend.memoryStats().allocatedBytes;
    if (after < before) throw std::logic_error("Flash batch MTP allocation ledger regressed");
    workspaceBytes = after - before;
  }
  metal::MetalBuffer view(Scratch slot, uint64_t bytes) {
    return backend.view(scratch[static_cast<size_t>(slot)], 0, bytes);
  }
  metal::MetalBuffer bf(Scratch slot, uint32_t rows, uint32_t width) {
    return view(slot, uint64_t{rows} * width * 2);
  }
  void project(metal::CommandGraph &graph, const std::string &prefix,
               const metal::MetalBuffer &input, const metal::MetalBuffer &output,
               const metal::MetalBuffer &diagnostics, uint32_t rows,
               uint32_t logicalRows = 0) {
    // Concatenating independent lanes does not turn a four-pair decode fold
    // into a sixteen-pair prefill. Preserve the sequential head's operand/
    // reduction policy; physical rows still cover every compact real input.
    const uint32_t eligibility = minimumLogicalRows ? minimumLogicalRows : logicalRows;
    owner.batchProject(graph, prefix, input, output, diagnostics, rows, eligibility,
        maximumLogicalRows >= 1 && maximumLogicalRows <= 8);
  }
  void hc(metal::CommandGraph &graph, const std::string &prefix,
          uint32_t rows, bool injection, metal::MetalBuffer diagnostics,
          uint32_t firstHyperRow = 0, bool normalizedReady = false) {
    const FlashHCGeometry geometry{rows, kWidth, 4, static_cast<float>(descriptor.normEpsilon)};
    const auto normalized = normalizedReady
        ? backend.view(scratch[static_cast<size_t>(Scratch::HCNormalized)],
            uint64_t{firstHyperRow} * kHyper * 2, uint64_t{rows} * kHyper * 2)
        : bf(Scratch::HCNormalized, rows, kHyper);
    const auto down = bf(Scratch::HCDown, rows, 320);
    const auto up = bf(Scratch::HCUp, rows, kHyper);
    const auto norm = prefix + ".hc_norm.weight";
    const auto hyperInput = backend.view(scratch[static_cast<size_t>(Scratch::Hyper)],
        uint64_t{firstHyperRow} * kHyper * 2, uint64_t{rows} * kHyper * 2);
    if (!normalizedReady)
      addHCGroupedNorm(graph, hyperInput, weights.tensor(norm),
          normalized, geometry, weights.normConvention(norm));
    const auto &downProjection = weights.projection(prefix + ".input_mix_weight_down");
    const auto &upProjection = weights.projection(prefix + ".input_mix_weight_up");
    const auto *injectionProjection = injection
        ? &weights.projection(prefix + ".block_inject_weight") : nullptr;
    if (fuseHC && supportsHCFused(downProjection, upProjection, injectionProjection, geometry)) {
      addHCFusedDown(graph, normalized, downProjection, injectionProjection, down,
          injection ? bf(Scratch::HCInjectionWeights, rows, 4) : metal::MetalBuffer{},
          diagnostics, geometry);
      addHCFusedUpMix(graph, normalized, down, upProjection,
          bf(Scratch::Mixed, rows, kWidth), diagnostics, geometry);
      return;
    }
    project(graph, prefix + ".input_mix_weight_down", normalized, down, diagnostics, rows);
    const FlashForwardActivationParams activation{rows, 320, 4};
    graph.add("flash_forward_hc_silu", {down, down, diagnostics}, activation,
        {(uint64_t{rows} * 320 + 255) / 256, 1, 1});
    const auto tile = rows >= 128 ? FlashAffineMPPTile::M32N128
                                 : FlashAffineMPPTile::M16N64;
    if (!injection && denseCache && minimumLogicalRows >= 32 &&
        denseCache->supportsHCUpMix(prefix + ".input_mix_weight_up", rows, tile)) {
      denseCache->addHCUpMix(graph, prefix + ".input_mix_weight_up", down, normalized,
          bf(Scratch::Mixed, rows, kWidth), diagnostics, rows, tile);
      return;
    }
    project(graph, prefix + ".input_mix_weight_up", down, up, diagnostics, rows);
    if (injection) {
      const auto raw = bf(Scratch::HCRawInjection, rows, 4);
      project(graph, prefix + ".block_inject_weight", normalized, raw, diagnostics, rows);
      addHCMixWithInjection(graph, normalized, up, raw, bf(Scratch::Mixed, rows, kWidth),
          bf(Scratch::HCInjectionWeights, rows, 4), geometry);
    } else {
      addHCMix(graph, normalized, up, bf(Scratch::Mixed, rows, kWidth), geometry);
    }
  }
};

FlashBatchMTPForward::FlashBatchMTPForward(FlashMTPForward &owner,
    uint32_t maximumLanes, uint32_t maximumRowsPerLane,
    const FlashTensor *cachedVocabulary) {
  std::lock_guard lock(owner.batchMutex());
  impl_ = std::make_unique<Impl>(owner, maximumLanes, maximumRowsPerLane, cachedVocabulary);
}
FlashBatchMTPForward::~FlashBatchMTPForward() = default;
FlashBatchMTPForward::FlashBatchMTPForward(FlashBatchMTPForward &&) noexcept = default;
FlashBatchMTPForward &FlashBatchMTPForward::operator=(FlashBatchMTPForward &&) noexcept = default;
uint64_t FlashBatchMTPForward::workspaceBytes() const noexcept {
  return impl_ ? impl_->workspaceBytes : 0;
}
const char *FlashBatchMTPForward::attentionRouteSemantics() const noexcept {
  return impl_ ? impl_->owner.attentionRouteSemantics() : "mtp-qsa-uninitialized";
}
const char *FlashBatchMTPForward::projectionRouteSemantics() const noexcept {
  return impl_ ? impl_->owner.projectionRouteSemantics() : "mtp-proposal-uninitialized";
}
bool FlashBatchMTPForward::vocabularyRegisterEnabled() const noexcept {
  return impl_ && bool(impl_->registerVocabulary);
}
const char *FlashBatchMTPForward::vocabularyRouteSemantics() const noexcept {
  if (!impl_) return "mtp-vocabulary-uninitialized";
  if (vocabularyRegisterEnabled()) return kFlashBF16Q8HeadSemantics;
  return impl_->cachedVocabulary ? "original-bf16-cached-vocabulary-m8-n128-v1"
      : "original-affine-vocabulary-owner-projection-v1";
}
uint64_t FlashBatchMTPForward::vocabularyRegisterCommands() const noexcept {
  return impl_ ? impl_->registerVocabularyCommands : 0;
}
uint64_t FlashBatchMTPForward::vocabularyRegisterRows() const noexcept {
  return impl_ ? impl_->registerVocabularyRows : 0;
}
uint64_t FlashBatchMTPForward::workspacePlannedBytes(uint32_t capacity,
    uint32_t maximumLanes, uint32_t maximumRowsPerLane,
    bool includeCachedVocabulary) {
  validate(capacity, maximumLanes, maximumRowsPerLane);
  const uint32_t rows = maximumLanes * maximumRowsPerLane;
  constexpr std::array<uint32_t, 30> widths{
      kWidth, kWidth, kWidth, kHyper, kHyper, kHyper, kHyper, 320, kHyper,
      4, 4, kWidth, kWidth, 12288, 512, 512, 640, 6144, 512, kSelections,
      kSelections * 640, kSelections * 640, kSelections * 640,
      kSelections * kWidth, 640, 640, 640, kWidth, 1, 248320};
  uint64_t bytes = 0;
  for (uint32_t width : widths)
    bytes += rounded(uint64_t{width == 248320 ? std::min(rows, uint32_t{16}) : rows} * width * 2);
  for (uint64_t extent : std::array<uint64_t, 5>{uint64_t{rows} * 8,
      uint64_t{rows} * kSelections * 8, 4, uint64_t{maximumLanes + 1} * 4,
      uint64_t{maximumLanes} * kWidth * 2}) bytes += rounded(extent);
  const uint64_t blocks = (uint64_t{capacity} + 3) / 4;
  for (uint64_t width : std::array<uint64_t, 6>{6144 * 2, 512 * 2, blocks * 4,
      512 * 4, uint64_t{24} * kFlashQSATokenWidth * 4,
      uint64_t{24} * kFlashQSATokenWidth * 2})
    bytes += rounded(uint64_t{maximumRowsPerLane} * width);
  if (includeCachedVocabulary && maximumLanes > 1)
    bytes += rounded(uint64_t{16} * kWidth * 2);
  bytes += FlashMTPForward::attentionWorkspacePlannedBytes(maximumRowsPerLane);
  if (enabled("SPLASH_FLASH_GPU_GREEDY"))
    bytes += greedyGPUWorkspacePlannedBytes(
        std::min<uint32_t>(rows, kFlashGreedyGPUMaximumRows), 248320);
  return bytes;
}

FlashBatchMTPResult FlashBatchMTPForward::forward(std::span<FlashMTPState *const> states,
    metal::MetalBuffer previousHidden, std::span<const uint32_t> nextTokens,
    std::span<const uint32_t> laneCounts, FlashMTPLogits mode) {
  if (!impl_) throw std::logic_error("Flash batch MTP head is not initialized");
  std::lock_guard lock(impl_->owner.batchMutex());
  if (states.empty() || states.size() > impl_->maximumLanes || states.size() != laneCounts.size())
    throw std::invalid_argument("Flash batch MTP states/counts have invalid lane geometry");
  if (mode != FlashMTPLogits::None && mode != FlashMTPLogits::Last && mode != FlashMTPLogits::All)
    throw std::invalid_argument("Flash batch MTP logits mode is invalid");
  std::vector<uint32_t> offsets{0};
  std::vector<uint64_t> lengths;
  for (uint32_t lane = 0; lane < states.size(); ++lane) {
    if (!states[lane] || !impl_->owner.ownsState(*states[lane]) ||
        !laneCounts[lane] || laneCounts[lane] > impl_->maximumRowsPerLane ||
        states[lane]->impl_->length > impl_->capacity ||
        laneCounts[lane] > impl_->capacity - states[lane]->impl_->length)
      throw std::invalid_argument("Flash batch MTP requires healthy owned states and capacity-bounded real spans");
    for (uint32_t prior = 0; prior < lane; ++prior)
      if (states[prior] == states[lane])
        throw std::invalid_argument("Flash batch MTP cannot contain duplicate state pointers");
    offsets.push_back(offsets.back() + laneCounts[lane]);
    lengths.push_back(states[lane]->impl_->length + laneCounts[lane]);
  }
  const uint32_t rows = offsets.back();
  if (mode == FlashMTPLogits::All && rows > 16)
    throw std::invalid_argument("Flash batch MTP All logits is bounded to16 real rows");
  if (nextTokens.size() != rows || !previousHidden ||
      previousHidden.sizeBytes() < uint64_t{rows} * kHyper * 2)
    throw std::invalid_argument("Flash batch MTP requires compact token/features with exactly the real row count");
  for (uint32_t token : nextTokens)
    if (token >= impl_->descriptor.vocabularySize)
      throw std::invalid_argument("Flash batch MTP next token is outside vocabulary");
  impl_->minimumLogicalRows = *std::min_element(laneCounts.begin(), laneCounts.end());
  impl_->maximumLogicalRows = *std::max_element(laneCounts.begin(), laneCounts.end());
  const auto tokens = impl_->view(Scratch::TokenIDs, uint64_t{rows} * 8);
  auto *hostTokens = static_cast<int64_t *>(tokens.contents());
  for (uint32_t row = 0; row < rows; ++row) hostTokens[row] = nextTokens[row];
  const auto diag = impl_->view(Scratch::Diagnostics, 4);
  std::memset(diag.contents(), 0, diag.sizeBytes());
  const auto laneOffsets = impl_->view(Scratch::LaneOffsets, uint64_t{states.size() + 1} * 4);
  std::memcpy(laneOffsets.contents(), offsets.data(), offsets.size() * 4);
  const auto bf = [&](Scratch slot, uint32_t width) { return impl_->bf(slot, rows, width); };
  const auto affine = [&](metal::CommandGraph &graph, const std::string &prefix,
      const metal::MetalBuffer &input, const metal::MetalBuffer &output) {
    impl_->project(graph, prefix, input, output, diag, rows);
  };
  const auto gathered = [&](metal::CommandGraph &graph, const FlashAffineProjection &projection,
      const metal::MetalBuffer &input, const metal::MetalBuffer &expertIDs,
      const metal::MetalBuffer &output, bool inputPerSelection = false) {
    // The selected-expert QMV arithmetic is qualified only for up to16 real
    // rows. Compact wider ragged folds must retain each sequential lane's
    // eligibility rather than switching every short lane to the control
    // reduction because unrelated peers increased the physical batch size.
    bool perLane = false;
    if (flashAffineExpertEnabled() && rows > 16)
      for (uint32_t count : laneCounts) perLane |= count <= 16;
    if (!perLane) {
      addGatheredAffine(graph, input, projection, expertIDs, output, diag,
          rows, kSelections, inputPerSelection);
      return;
    }
    const uint64_t inputWidth = uint64_t{projection.inputSize} *
        (inputPerSelection ? kSelections : 1);
    const uint64_t outputWidth = uint64_t{projection.outputSize} * kSelections;
    for (uint32_t lane = 0; lane < states.size(); ++lane) {
      const uint32_t first = offsets[lane], count = laneCounts[lane];
      addGatheredAffine(graph,
          impl_->backend.view(input, uint64_t{first} * inputWidth * 2,
              uint64_t{count} * inputWidth * 2), projection,
          impl_->backend.view(expertIDs, uint64_t{first} * kSelections * 8,
              uint64_t{count} * kSelections * 8),
          impl_->backend.view(output, uint64_t{first} * outputWidth * 2,
              uint64_t{count} * outputWidth * 2), diag,
          count, kSelections, inputPerSelection);
    }
  };
  metal::CommandGraph graph;
  const auto hyper = bf(Scratch::Hyper, kHyper);
  const auto mixed = bf(Scratch::Mixed, kWidth);
  const auto branch = bf(Scratch::Branch, kWidth);
  const FlashHCGeometry hc{rows, kWidth, 4, static_cast<float>(impl_->descriptor.normEpsilon)};
  const FlashHCGeometry embeddingNorm{rows, kWidth, 1, hc.epsilon};
  const FlashHCGeometry hiddenNorm{rows, kHyper, 1, hc.epsilon};
  addAffineEmbedding(graph, impl_->weights.projection("language_model.model.embed_tokens"),
      tokens, bf(Scratch::Embedding, kWidth), diag, rows);
  for (const auto &[name, input, output, geometry] : {
      std::tuple{"mtp.pre_fc_norm_embedding.weight", bf(Scratch::Embedding, kWidth),
          bf(Scratch::EmbeddingNormalized, kWidth), embeddingNorm},
      std::tuple{"mtp.pre_fc_norm_hidden.weight", previousHidden,
          bf(Scratch::HiddenNormalized, kHyper), hiddenNorm}}) {
    addHCGroupedNorm(graph, input, impl_->weights.tensor(name), output, geometry,
        impl_->weights.normConvention(name));
  }
  affine(graph, "mtp.fc_embedding", bf(Scratch::EmbeddingNormalized, kWidth),
      bf(Scratch::EmbeddingProjected, kWidth));
  // Treat every globally normalized stream as a separate 2560-vector for the
  // same trained fc_hidden; its output is again contiguous [rows,4,2560].
  impl_->project(graph, "mtp.fc_hidden", bf(Scratch::HiddenNormalized, kHyper),
      bf(Scratch::HiddenProjected, kHyper), diag, rows * 4, rows);
  const FlashMTPFuseParams fuse{rows, kWidth, 4};
  graph.add("flash_mtp_fuse_inputs", {bf(Scratch::EmbeddingProjected, kWidth),
      bf(Scratch::HiddenProjected, kHyper), hyper, diag}, fuse,
      {(uint64_t{rows} * kHyper + 255) / 256, 1, 1});

  const std::string prefix = "mtp.layers.0";
  impl_->hc(graph, prefix + ".attn_hyper_connection", rows, true, diag);
  const std::string attention = prefix + ".self_attn";
  const auto q = bf(Scratch::QProjection, 12288), k = bf(Scratch::KProjection, 512);
  const auto v = bf(Scratch::VProjection, 512), index = bf(Scratch::IndexProjection, 640);
  affine(graph, attention + ".q_proj", mixed, q);
  affine(graph, attention + ".k_proj", mixed, k);
  affine(graph, attention + ".v_proj", mixed, v);
  affine(graph, attention + ".indexer.index_qk_proj", mixed, index);
  const auto qNorm = attention + ".q_norm.weight", kNorm = attention + ".k_norm.weight";
  const auto iqNorm = attention + ".indexer.q_layernorm.weight";
  const auto ikNorm = attention + ".indexer.k_layernorm.weight";
  const auto attentionOutput = bf(Scratch::AttentionOutput, 6144);
  for (uint32_t lane = 0; lane < states.size(); ++lane) {
    auto &state = *states[lane]->impl_;
    const uint32_t first = offsets[lane], count = laneCounts[lane];
    const auto slice = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return impl_->backend.view(buffer, uint64_t{first} * width * 2, uint64_t{count} * width * 2);
    };
    FlashQSAFastInputs inputs;
    inputs.qProjection = slice(q, 12288); inputs.kProjection = slice(k, 512);
    inputs.vProjection = slice(v, 512); inputs.indexProjection = slice(index, 640);
    inputs.qNorm = &impl_->weights.tensor(qNorm);
    inputs.kNorm = &impl_->weights.tensor(kNorm);
    inputs.indexQNorm = &impl_->weights.tensor(iqNorm);
    inputs.indexKNorm = &impl_->weights.tensor(ikNorm);
    inputs.output = slice(attentionOutput, 6144); inputs.diagnostics = diag;
    inputs.qConvention = impl_->weights.normConvention(qNorm);
    inputs.kConvention = impl_->weights.normConvention(kNorm);
    inputs.indexQConvention = impl_->weights.normConvention(iqNorm);
    inputs.indexKConvention = impl_->weights.normConvention(ikNorm);
    inputs.epsilon = impl_->descriptor.normEpsilon;
    inputs.theta = impl_->descriptor.rotaryTheta;
    // Eligibility follows each real lane's count, exactly as sequential head
    // calls do. Compact batch rows cannot change attention arithmetic policy.
    impl_->owner.batchAddQSA(graph, inputs, state.qsa, impl_->qsaWorkspace,
        impl_->qsaFastWorkspace, static_cast<uint32_t>(state.length), count);
  }
  affine(graph, attention + ".o_proj", attentionOutput, branch);
  const auto mlpNorm = prefix + ".mlp_hyper_connection.hc_norm.weight";
  const bool mlpNormalizedReady = impl_->fuseHC && rows <= 32;
  if (mlpNormalizedReady) {
    addHCFusedInjectNorm(graph, hyper, branch, bf(Scratch::HCInjectionWeights, 4),
        impl_->weights.tensor(mlpNorm), hyper, bf(Scratch::HCNormalized, kHyper), diag,
        hc, impl_->weights.normConvention(mlpNorm));
  } else {
    addHCInject(graph, hyper, branch, bf(Scratch::HCInjectionWeights, 4), hyper, hc);
  }
  impl_->hc(graph, prefix + ".mlp_hyper_connection", rows, true, diag, 0, mlpNormalizedReady);

  const std::string mlp = prefix + ".mlp";
  const auto ids = impl_->view(Scratch::ExpertIDs, uint64_t{rows} * kSelections * 8);
  const auto route = bf(Scratch::RouteWeights, kSelections);
  if (impl_->denseCache && impl_->minimumLogicalRows >= 16)
    addDenseBF16WholeK(impl_->backend, graph, mixed, impl_->weights.tensor(mlp + ".gate.weight"),
        bf(Scratch::Router, 512), diag, rows, FlashAffineMPPTile::M16N64);
  else
    addDenseBF16(graph, mixed, impl_->weights.tensor(mlp + ".gate.weight"), bf(Scratch::Router, 512), diag, rows);
  addRoute(graph, bf(Scratch::Router, 512), ids, route, diag, rows, 512, kSelections);
  gathered(graph, impl_->weights.projection(mlp + ".switch_mlp.gate_proj"),
      mixed, ids, bf(Scratch::ExpertGate, kSelections * 640));
  gathered(graph, impl_->weights.projection(mlp + ".switch_mlp.up_proj"),
      mixed, ids, bf(Scratch::ExpertUp, kSelections * 640));
  addSiLUMultiply(graph, bf(Scratch::ExpertGate, kSelections * 640), bf(Scratch::ExpertUp, kSelections * 640),
      bf(Scratch::ExpertIntermediate, kSelections * 640), diag, rows, 640, kSelections);
  gathered(graph, impl_->weights.projection(mlp + ".switch_mlp.down_proj"),
      bf(Scratch::ExpertIntermediate, kSelections * 640), ids,
      bf(Scratch::ExpertDown, kSelections * kWidth), true);
  affine(graph, mlp + ".shared_expert.gate_proj", mixed, bf(Scratch::SharedGate, 640));
  affine(graph, mlp + ".shared_expert.up_proj", mixed, bf(Scratch::SharedUp, 640));
  addSiLUMultiply(graph, bf(Scratch::SharedGate, 640), bf(Scratch::SharedUp, 640),
      bf(Scratch::SharedIntermediate, 640), diag, rows, 640);
  affine(graph, mlp + ".shared_expert.down_proj", bf(Scratch::SharedIntermediate, 640), bf(Scratch::SharedDown, kWidth));
  affine(graph, mlp + ".shared_expert_gate", mixed, bf(Scratch::SharedGateLogit, 1));
  addCombine(graph, bf(Scratch::ExpertDown, kSelections * kWidth), ids, route,
      bf(Scratch::SharedDown, kWidth), bf(Scratch::SharedGateLogit, 1), branch, diag,
      rows, kWidth, 512, kSelections);
  bool finalNormalizedReady = false;
  // Adjacent normalization can cover all newly injected rows. Last consumes
  // only its final normalized row; the complete pre-mixer hyper is preserved.
  if (impl_->fuseHC && rows <= 32 &&
      mode != FlashMTPLogits::None) {
    const std::string finalNorm = "mtp.hyper_connection_mixer.hc_norm.weight";
    addHCFusedInjectNorm(graph, hyper, branch, bf(Scratch::HCInjectionWeights, 4),
        impl_->weights.tensor(finalNorm), hyper, bf(Scratch::HCNormalized, kHyper), diag,
        hc, impl_->weights.normConvention(finalNorm));
    finalNormalizedReady = true;
  } else {
    addHCInject(graph, hyper, branch, bf(Scratch::HCInjectionWeights, 4), hyper, hc);
  }

  uint32_t logitRows = 0;
  metal::MetalBuffer logits;
  uint32_t greedyRows = 0;
  metal::MetalBuffer greedyResults;
  if (mode != FlashMTPLogits::None) {
    // The stateless final mixer is independent per real row. Last compacts
    // its lane-final outputs before the shared vocabulary projection.
    impl_->hc(graph, "mtp.hyper_connection_mixer", rows, false, diag, 0, finalNormalizedReady);
    metal::MetalBuffer headInput = mixed;
    if (mode == FlashMTPLogits::Last) {
      logitRows = static_cast<uint32_t>(states.size());
      headInput = impl_->bf(Scratch::HeadInput, logitRows, kWidth);
      const FlashBatchMTPGatherParams gather{logitRows, kWidth, rows};
      graph.add("flash_batch_mtp_gather_last", {mixed, laneOffsets, headInput, diag}, gather,
          {(uint64_t{logitRows} * kWidth + 255) / 256, 1, 1});
    } else logitRows = rows;
    logits = impl_->bf(Scratch::HeadLogits, logitRows, impl_->descriptor.vocabularySize);
    if (mode == FlashMTPLogits::Last && logitRows >= 2 && logitRows <= 4 &&
        impl_->registerVocabulary) {
      impl_->registerVocabulary->addProjection(graph, headInput, logits, diag, logitRows);
    } else if (mode == FlashMTPLogits::Last && logitRows >= 2 &&
        impl_->cachedVocabulary && impl_->vocabularyWorkspace) {
      addDenseBF16SmallRows(impl_->backend, graph, headInput,
          *impl_->cachedVocabulary, logits, diag, logitRows,
          *impl_->vocabularyWorkspace, FlashDenseSmallRowsTile::M8N128);
    } else {
      impl_->project(graph, "language_model.lm_head", headInput, logits, diag, logitRows);
    }
    if (impl_->gpuGreedy && logitRows <= kFlashGreedyGPUMaximumRows) {
      greedyRows = logitRows;
      greedyResults = impl_->backend.view(impl_->greedyResults, 0,
          uint64_t{greedyRows} * sizeof(FlashGreedyGPURowResult));
      addGreedyGPU(graph, logits, *impl_->greedyWorkspace, greedyResults,
          greedyRows, impl_->descriptor.vocabularySize);
    }
  }
  metal::CommandTiming timing;
  try {
    timing = impl_->backend.submitCommand(graph.dispatches());
    uint32_t status = 0;
    std::memcpy(&status, diag.contents(), sizeof(status));
    if (status) throw std::runtime_error("Flash batch MTP sticky diagnostics failed: " + std::to_string(status));
  } catch (...) {
    for (auto *state : states) state->impl_->poisoned = true;
    throw;
  }
  for (uint32_t lane = 0; lane < states.size(); ++lane)
    states[lane]->impl_->length = lengths[lane];
  if (mode == FlashMTPLogits::Last && logitRows >= 2 && logitRows <= 4 &&
      impl_->registerVocabulary) {
    ++impl_->registerVocabularyCommands;
    impl_->registerVocabularyRows += logitRows;
  }
  return {timing, logits, logitRows, static_cast<uint32_t>(states.size()), hyper,
      std::move(offsets), std::move(lengths), greedyResults, greedyRows};
}
} // namespace splash::flash
