#include "flash/FlashQSABulk.hpp"
#include "flash/FlashForward.hpp"

#include "flash/FlashAffine.hpp"
#include "flash/FlashBatchPrefill.hpp"
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashSharedExpertFused.hpp"
#include "flash/FlashDenseSmallRows.hpp"
#include "flash/FlashFloatDenseCache.hpp"
#include "flash/FlashInt8Head.hpp"
#include "flash/FlashExpertDenseCache.hpp"
#include "flash/FlashInt8ExpertStore.hpp"
#include "flash/FlashExpertCachePlan.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "flash/FlashGDN.hpp"
#include "flash/FlashGDNFused.hpp"
#include "flash/FlashGDNLazyRollback.hpp"
#include "flash/FlashGDNStaged.hpp"
#include "flash/FlashHC.hpp"
#include "flash/FlashHCFused.hpp"
#include "flash/FlashMoE.hpp"
#include "flash/FlashMTPWindow.hpp"
#include "flash/FlashPrefillDenseTiles.hpp"
#include "flash/FlashPLE.hpp"
#include "flash/FlashPLEFused.hpp"
#include "flash/FlashPLESSD.hpp"
#include "flash/FlashGreedyGPU.hpp"
#include "flash/FlashQSA.hpp"
#include "flash/FlashQSAFast.hpp"
#include "flash/FlashQSAMPP.hpp"
#include "flash/FlashRequestStateInternal.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashForward.h"

#include <array>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::flash {
namespace {
constexpr uint64_t kAlignment = 16384;
constexpr uint32_t kWidth = 2560;
constexpr uint32_t kHyper = 10240;
constexpr uint32_t kSelections = 10;

bool fusionEnabled(const char *name) {
  const char *value = std::getenv(name);
  if (!value) return false;
  if (std::string_view(value) == "1") return true;
  if (std::string_view(value) == "0") return false;
  throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}

std::optional<FlashExpertCachePlan> hotExpertPlan(const FlashWeights &weights) {
  const char *path = std::getenv("SPLASH_FLASH_HOT_EXPERT_PLAN");
  if (!path) return std::nullopt;
  if (!*path) throw std::invalid_argument("SPLASH_FLASH_HOT_EXPERT_PLAN must name a local plan");
  return loadFlashExpertCachePlan(path, weights.sourceIdentity());
}
std::optional<std::filesystem::path> int8ExpertDirectory() {
  const char *path = std::getenv("SPLASH_FLASH_INT8_EXPERT_STORE");
  if (!path || std::string_view(path) == "0") return std::nullopt;
  if (!*path) throw std::invalid_argument("SPLASH_FLASH_INT8_EXPERT_STORE must name a saved local store");
  if (std::getenv("SPLASH_FLASH_HOT_EXPERT_PLAN"))
    throw std::invalid_argument("choose one selected-expert operand store: INT8 or BF16 hot cache");
  return std::filesystem::path(path);
}

uint64_t checkedMultiply(uint64_t left, uint64_t right) {
  if (left && right > std::numeric_limits<uint64_t>::max() / left)
    throw std::overflow_error("Flash forward byte count overflows");
  return left * right;
}
uint64_t roundAllocation(uint64_t bytes) {
  if (!bytes || bytes > std::numeric_limits<uint64_t>::max() - (kAlignment - 1))
    throw std::overflow_error("Flash forward allocation extent is invalid");
  return (bytes + kAlignment - 1) & ~(kAlignment - 1);
}
void validateCapacity(uint32_t capacity) {
  if (!capacity || capacity > 262144)
    throw std::invalid_argument("Flash request capacity must be 1..262144");
}
void clear(const metal::MetalBuffer &buffer) {
  if (!buffer || buffer.storage() != metal::BufferStorage::Shared || !buffer.contents())
    throw std::logic_error("Flash cold state requires CPU-accessible shared storage");
  std::memset(buffer.contents(), 0, buffer.sizeBytes());
}

enum class Scratch : size_t {
  TokenIDs, Embedding, Hyper, HCNormalized, HCUp, HCDown, HCRawInjection,
  HCInjectionWeights, Mixed, Branch, QProjection, KProjection, VProjection,
  IndexProjection, GDNMixed, GDNZ, GDNA, GDNB, GDNDecay, GDNBeta,
  GDNRecurrentRows, AttentionOutput, Router, ExpertIDs, RouteWeights,
  ExpertGate, ExpertUp, ExpertIntermediate, ExpertDown, SharedGate, SharedUp,
  SharedIntermediate, SharedDown, SharedGateLogit, NgramIDs, PLEEmbedding,
  PLEKey, PLEValue, PLENormalizedKeys, PLENormalizedQueries, PLEGated,
  PLENormalizedConvolution, PLEOutput, HeadLogits, Diagnostics, Count,
};
constexpr size_t kScratchCount = static_cast<size_t>(Scratch::Count);

} // namespace

struct FlashForward::Impl final {
  metal::MetalBackend &backend;
  const FlashWeights &weights;
  const FlashDescriptor descriptor;
  const uint32_t capacity;
  const uint32_t maximumRows;
  const uint32_t maximumLogitRows;
  const uint32_t maximumVerifyRows;
  const bool fuseHC = fusionEnabled("SPLASH_FLASH_FUSE_HC");
  const bool hcUpF32 = flashHCUpF32MPPEnabled();
  FlashHCUpEncodedCounters hcUpCounters;
  const bool fuseGDN = fusionEnabled("SPLASH_FLASH_FUSE_GDN");
  const bool stagedGDN = fusionEnabled("SPLASH_FLASH_GDN_STAGED");
  const bool cacheDense = fusionEnabled("SPLASH_FLASH_DENSE_CACHE");
  const bool fuseSharedExpert = flashSharedExpertFusedFlag(std::getenv("SPLASH_FLASH_SHARED_EXPERT_FUSED"));
  const bool blockMoE = fusionEnabled("SPLASH_FLASH_BLOCKED_MOE");
  const bool onlineQSA = fusionEnabled("SPLASH_FLASH_QSA_F32");
  const bool mppQSA = fusionEnabled("SPLASH_FLASH_QSA_MPP");
  const bool bulkQSAPrefill = qsaBulkPrefillEnabled();
  const bool bulkQSAPrefillSG8 = qsaBulkPrefillSG8Enabled(bulkQSAPrefill);
  const bool smallDense = fusionEnabled("SPLASH_FLASH_DENSE_SMALL_ROWS");
  const bool cacheFloat = fusionEnabled("SPLASH_FLASH_FLOAT_DENSE_CACHE");
  const bool codeHead = fusionEnabled("SPLASH_FLASH_INT8_HEAD");
  const bool selectiveFloat = fusionEnabled("SPLASH_FLASH_FLOAT_DENSE_SELECTIVE");
  const bool captureRoutes = fusionEnabled("SPLASH_FLASH_CAPTURE_EXPERT_IDS");
  const bool fusePLELookup = fusionEnabled("SPLASH_FLASH_PLE_LOOKUP_FUSED");
  const bool gpuGreedy = fusionEnabled("SPLASH_FLASH_GPU_GREEDY");
  const bool lazyGDN = flashGDNLazyRollbackEnabled();
  std::shared_ptr<const uint8_t> owner = std::make_shared<const uint8_t>(0);
  std::array<metal::MetalBuffer, kScratchCount> scratch;
  FlashQSAWorkspace qsaWorkspace;
  FlashQSAFastWorkspace qsaFastWorkspace;
  std::optional<FlashQSABulkWorkspace> bulkQSAWorkspace;
  FlashQSABulkCounters bulkQSACounters;
  FlashPLEWeights pleWeights;
  std::unique_ptr<FlashPLEFused> pleLookup;
  std::unique_ptr<FlashPLESSD> pleSSD;
  FlashGreedyGPUWorkspace greedyWorkspace;
  metal::MetalBuffer greedyResults;
  std::unique_ptr<FlashDenseCache> denseCache;
  std::unique_ptr<FlashDenseSmallRowsWorkspace> smallDenseWorkspace;
  std::unique_ptr<FlashFloatDenseCache> floatDenseCache;
  std::unique_ptr<FlashFloatDenseSmallRowsWorkspace> floatDenseWorkspace;
  std::unique_ptr<FlashInt8Head> int8Head;
  std::array<std::unique_ptr<FlashExpertDenseCache>, 48> expertCaches;
  std::unique_ptr<FlashInt8ExpertStore> int8ExpertStore;
  std::string expertPlanIdentity;
  metal::MetalBuffer capturedRoutes;
  uint32_t capturedRows = 0;
  FlashMoEBlockedScratch blockedScratch;
  metal::MetalBuffer verifyRecurrent;
  metal::MetalBuffer verifyConvolution;
  std::array<std::unique_ptr<FlashGDNLazyRollback>, 48> lazyGDNRecords;
  std::array<uint64_t, 48> lazyGDNTickets{};
  metal::MetalBuffer beforePLEHistory;
  metal::MetalBuffer beforePLEConvolution;
  metal::MetalBuffer retainedCount;
  metal::MetalBuffer beforeGDNConvolution;
  std::array<uint32_t, 48> gdnSlots{};
  std::weak_ptr<const uint8_t> pendingIdentity;
  uint64_t pendingBegin = 0;
  uint32_t pendingRows = 0;
  uint64_t workspaceBytes = 0;
  std::mutex mutex;

  Impl(metal::MetalBackend &value, const FlashWeights &model, uint32_t context,
       uint32_t rows, uint32_t verifyRows)
      : backend(value), weights(model), descriptor(model.descriptor()),
        capacity(context), maximumRows(rows), maximumLogitRows(std::min(rows, uint32_t{128})),
        maximumVerifyRows(verifyRows),
        pleWeights(FlashPLEWeights::fromWeights(model)) {
    descriptor.validate();
    if (!descriptor.pleParametersLoaded)
      throw std::invalid_argument("Flash forward requires checked stored PLE parameters");
    if (mppQSA && !onlineQSA)
      throw std::invalid_argument("SPLASH_FLASH_QSA_MPP requires SPLASH_FLASH_QSA_F32=1");
    if (selectiveFloat && (!cacheFloat || !flashAffineFastEnabled()))
      throw std::invalid_argument("SPLASH_FLASH_FLOAT_DENSE_SELECTIVE requires FLOAT_DENSE_CACHE=1 and QMV_F32=1");
    if (fuseSharedExpert && !cacheDense)
      throw std::invalid_argument("SPLASH_FLASH_SHARED_EXPERT_FUSED requires SPLASH_FLASH_DENSE_CACHE=1");
    validateCapacity(capacity);
    if (!maximumRows || maximumRows > 2048 || maximumRows > kFlashGDNMaximumRows || maximumRows > kFlashMoEMaxRows)
      throw std::invalid_argument("Flash forward maximum rows must be 1..2048");
    if (maximumVerifyRows > kFlashSingletonMaximumVerifyRows || maximumVerifyRows > maximumRows)
      throw std::invalid_argument("Flash target verification rows must be 0..16 and fit scratch");
    const uint64_t before = backend.memoryStats().allocatedBytes;
    if (weights.pleSSDStreamingEnabled())
      pleSSD = std::make_unique<FlashPLESSD>(backend, weights.pleSSDStore(),
          pleWeights, 1, maximumRows);
    else if (fusePLELookup && backend.supportsArgumentBuffersTier2())
      pleLookup = std::make_unique<FlashPLEFused>(backend, pleWeights);
    if (gpuGreedy) {
      const auto greedyRows = std::min(maximumLogitRows, uint32_t{kFlashGreedyGPUMaximumRows});
      greedyWorkspace = allocateGreedyGPUWorkspace(backend, greedyRows, descriptor.vocabularySize);
      greedyResults = backend.allocateBuffer(uint64_t{greedyRows} * sizeof(FlashGreedyGPURowResult),
          metal::BufferStorage::Shared, "flash-forward compact GPU greedy records");
    }
    if (cacheFloat) {
      if (smallDense) throw std::invalid_argument("choose one small-row coefficient policy: BF16 or F32");
      const auto prefixes = FlashFloatDenseCache::defaultPrefixes(weights, !codeHead);
      floatDenseCache = std::make_unique<FlashFloatDenseCache>(backend, weights, prefixes);
      floatDenseWorkspace = std::make_unique<FlashFloatDenseSmallRowsWorkspace>(backend);
    }
    if (codeHead)
      int8Head = std::make_unique<FlashInt8Head>(backend, weights,
          floatDenseWorkspace ? floatDenseWorkspace->paddedInput() : metal::MetalBuffer{});
    if (const auto plan = hotExpertPlan(weights)) {
      if (!blockMoE) throw std::invalid_argument("hot-expert cache requires blocked prefill enabled");
      expertPlanIdentity = plan->planSha256;
      for (uint32_t layer = 0; layer < descriptor.layers; ++layer)
        expertCaches[layer] = std::make_unique<FlashExpertDenseCache>(backend, weights,
            "language_model.model.layers." + std::to_string(layer) + ".mlp.switch_mlp",
            plan->selectedExperts[layer]);
    }
    if (const auto directory = int8ExpertDirectory()) {
      if (!blockMoE) throw std::invalid_argument("saved INT8 experts require blocked prefill enabled");
      int8ExpertStore = std::make_unique<FlashInt8ExpertStore>(backend, weights, *directory);
    }
    if (cacheDense) {
      const auto prefixes = FlashDenseCache::defaultPrefixes(weights, true);
      denseCache = std::make_unique<FlashDenseCache>(backend, weights, prefixes);
      if (smallDense)
        smallDenseWorkspace = std::make_unique<FlashDenseSmallRowsWorkspace>(backend);
    }
    if (captureRoutes)
      capturedRoutes = backend.allocateBuffer(
          roundAllocation(uint64_t{descriptor.layers} * maximumRows * kSelections * 8),
          metal::BufferStorage::Shared, "flash-forward-expert-route-capture");
    const auto allocate = [&](Scratch slot, uint64_t bytes, const char *label) {
      scratch[static_cast<size_t>(slot)] = backend.allocateBuffer(
          roundAllocation(bytes), metal::BufferStorage::Shared, label);
    };
    const auto bf = [&](Scratch slot, uint64_t width, const char *label) {
      allocate(slot, checkedMultiply(checkedMultiply(maximumRows, width), 2), label);
    };
    allocate(Scratch::TokenIDs, uint64_t{maximumRows} * 8, "flash-forward-token-ids");
    bf(Scratch::Embedding, kWidth, "flash-forward-embedding");
    bf(Scratch::Hyper, kHyper, "flash-forward-hyper-state");
    bf(Scratch::HCNormalized, kHyper, "flash-forward-hc-normalized");
    bf(Scratch::HCUp, kHyper, "flash-forward-hc-up");
    bf(Scratch::HCDown, 320, "flash-forward-hc-down");
    bf(Scratch::HCRawInjection, 4, "flash-forward-hc-raw-injection");
    bf(Scratch::HCInjectionWeights, 4, "flash-forward-hc-injection-weights");
    bf(Scratch::Mixed, kWidth, "flash-forward-mixed-input");
    bf(Scratch::Branch, kWidth, "flash-forward-branch");
    bf(Scratch::QProjection, 12288, "flash-forward-q-projection");
    bf(Scratch::KProjection, 512, "flash-forward-k-projection");
    bf(Scratch::VProjection, 512, "flash-forward-v-projection");
    bf(Scratch::IndexProjection, 640, "flash-forward-index-projection");
    bf(Scratch::GDNMixed, 10240, "flash-forward-gdn-convolved");
    bf(Scratch::GDNZ, 6144, "flash-forward-gdn-z");
    bf(Scratch::GDNA, 48, "flash-forward-gdn-a");
    bf(Scratch::GDNB, 48, "flash-forward-gdn-b");
    allocate(Scratch::GDNDecay, uint64_t{maximumRows} * 48 * 4, "flash-forward-gdn-decay");
    bf(Scratch::GDNBeta, 48, "flash-forward-gdn-beta");
    bf(Scratch::GDNRecurrentRows, 6144, "flash-forward-gdn-recurrent-rows");
    bf(Scratch::AttentionOutput, 6144, "flash-forward-attention-output");
    bf(Scratch::Router, 512, "flash-forward-router-logits");
    allocate(Scratch::ExpertIDs, uint64_t{maximumRows} * kSelections * 8, "flash-forward-expert-ids");
    bf(Scratch::RouteWeights, kSelections, "flash-forward-route-weights");
    bf(Scratch::ExpertGate, kSelections * 640, "flash-forward-expert-gate");
    bf(Scratch::ExpertUp, kSelections * 640, "flash-forward-expert-up");
    bf(Scratch::ExpertIntermediate, kSelections * 640, "flash-forward-expert-intermediate");
    bf(Scratch::ExpertDown, kSelections * kWidth, "flash-forward-expert-down");
    bf(Scratch::SharedGate, 640, "flash-forward-shared-gate");
    bf(Scratch::SharedUp, 640, "flash-forward-shared-up");
    bf(Scratch::SharedIntermediate, 640, "flash-forward-shared-intermediate");
    bf(Scratch::SharedDown, kWidth, "flash-forward-shared-down");
    bf(Scratch::SharedGateLogit, 1, "flash-forward-shared-gate-logit");
    allocate(Scratch::NgramIDs, uint64_t{maximumRows} * 16 * 8, "flash-forward-ngram-ids");
    bf(Scratch::PLEEmbedding, kWidth, "flash-forward-ple-embedding");
    bf(Scratch::PLEKey, kHyper, "flash-forward-ple-key");
    bf(Scratch::PLEValue, kWidth, "flash-forward-ple-value");
    bf(Scratch::PLENormalizedKeys, kHyper, "flash-forward-ple-normalized-keys");
    bf(Scratch::PLENormalizedQueries, kHyper, "flash-forward-ple-normalized-queries");
    bf(Scratch::PLEGated, kHyper, "flash-forward-ple-gated-values");
    bf(Scratch::PLENormalizedConvolution, kHyper, "flash-forward-ple-normalized-convolution");
    bf(Scratch::PLEOutput, kHyper, "flash-forward-ple-output");
    allocate(Scratch::HeadLogits, uint64_t{maximumLogitRows} * descriptor.vocabularySize * 2,
              "flash-forward-head-logits");
    allocate(Scratch::Diagnostics, 4, "flash-forward-diagnostics");
    qsaWorkspace = allocateQSAWorkspace(backend, std::min(maximumRows, uint32_t{128}), capacity);
    if (onlineQSA)
      qsaFastWorkspace = mppQSA
          ? allocateQSAOnlineMPPWorkspace(backend, std::min(maximumRows, uint32_t{128}), 32)
          : allocateQSAFastWorkspace(backend, std::min(maximumRows, uint32_t{128}), 4);
    if (blockMoE && maximumRows >= 256)
      blockedScratch = allocateMoEBlockedScratch(backend, maximumRows, kSelections);
    if (bulkQSAPrefill && maximumRows >= 2048) {
      bulkQSAWorkspace.emplace(allocateQSABulkWorkspace(backend));
      const auto &bulk = *bulkQSAWorkspace;
      const uint64_t bulkPlaneBytes = bulk.prepared.queries.sizeBytes() +
          bulk.prepared.indexQueries.sizeBytes() + bulk.prepared.selectedBlocks.sizeBytes() +
          bulk.partials.partitionStatistics.sizeBytes() + bulk.partials.partitionValues.sizeBytes();
      if (bulkPlaneBytes != qsaBulkWorkspacePlannedBytes())
        throw std::logic_error("Flash bulk QSA five-plane allocation/admission mismatch");
    }
    uint32_t gdnSlot = 0;
    for (uint32_t layer = 0; layer < descriptor.layers; ++layer)
      if (descriptor.layerKinds[layer] == FlashLayerKind::GatedDeltaNet)
        gdnSlots[layer] = gdnSlot++;
    if (maximumVerifyRows > 1) {
      const uint64_t prefixes = maximumVerifyRows - 1;
      if (lazyGDN) {
        for (uint32_t layer = 0; layer < descriptor.layers; ++layer)
          if (descriptor.layerKinds[layer] == FlashLayerKind::GatedDeltaNet)
            lazyGDNRecords[layer] = std::make_unique<FlashGDNLazyRollback>(backend, maximumVerifyRows, 1);
      } else {
        verifyRecurrent = backend.allocateBuffer(36 * prefixes * roundAllocation(flashGDNRecurrentLaneBytes()),
            metal::BufferStorage::Shared, "flash-verify-recurrent-prefixes");
        verifyConvolution = backend.allocateBuffer(36 * prefixes * roundAllocation(flashGDNConvolutionLaneBytes()),
            metal::BufferStorage::Shared, "flash-verify-convolution-prefixes");
      }
      beforePLEHistory = backend.allocateBuffer(roundAllocation(2 * sizeof(int64_t)),
          metal::BufferStorage::Shared, "flash-verify-ple-before-history");
      beforePLEConvolution = backend.allocateBuffer(roundAllocation(uint64_t{9} * kHyper * 2),
          metal::BufferStorage::Shared, "flash-verify-ple-before-convolution");
      retainedCount = backend.allocateBuffer(roundAllocation(sizeof(uint32_t)),
          metal::BufferStorage::Shared, "flash-verify-retained-count");
      if (fuseGDN && !lazyGDN)
        beforeGDNConvolution = backend.allocateBuffer(roundAllocation(flashGDNConvolutionLaneBytes()),
            metal::BufferStorage::Shared, "flash-verify-gdn-before-convolution");
    }
    const uint64_t after = backend.memoryStats().allocatedBytes;
    if (after < before) throw std::logic_error("Flash workspace allocation ledger regressed");
    workspaceBytes = after - before;
    if (bulkQSAWorkspace && workspaceBytes < qsaBulkWorkspacePlannedBytes())
      throw std::logic_error("Flash bulk QSA planes omitted from allocation ledger");
  }

  metal::MetalBuffer view(Scratch slot, uint64_t bytes) {
    return backend.view(scratch[static_cast<size_t>(slot)], 0, bytes);
  }
  metal::MetalBuffer bf(Scratch slot, uint32_t rows, uint32_t width) {
    return view(slot, uint64_t{rows} * width * 2);
  }

  void copy(metal::CommandGraph &graph, const metal::MetalBuffer &input,
            const metal::MetalBuffer &output, uint64_t bytes) {
    if (!bytes || bytes % 4 || input.sizeBytes() < bytes || output.sizeBytes() < bytes || input.sameView(output))
      throw std::invalid_argument("Flash verification raw-copy extent/alias is invalid");
    const uint64_t words = bytes / 4;
    graph.add("flash_forward_copy_words", {input, output}, FlashForwardCopyParams{words},
              {(words + 255) / 256, 1, 1});
  }

  metal::MetalBuffer prefixTape(bool recurrence, uint32_t layer, uint32_t row) {
    const uint64_t bytes = recurrence ? flashGDNRecurrentLaneBytes() : flashGDNConvolutionLaneBytes();
    const uint64_t cell = uint64_t{gdnSlots[layer]} * (maximumVerifyRows - 1) + row;
    return backend.view(recurrence ? verifyRecurrent : verifyConvolution,
                         cell * roundAllocation(bytes), bytes);
  }

  void abortLazyRecords() noexcept {
    for (uint32_t layer = 0; layer < descriptor.layers; ++layer) {
      auto &record = lazyGDNRecords[layer];
      if (record && record->pending()) {
        try { record->abort(lazyGDNTickets[layer]); } catch (...) {}
      }
      lazyGDNTickets[layer] = 0;
    }
  }
  void finishPending() noexcept {
    abortLazyRecords(); pendingIdentity.reset(); pendingRows = 0; pendingBegin = 0;
  }

  void project(metal::CommandGraph &graph, const std::string &prefix,
               const metal::MetalBuffer &input, const metal::MetalBuffer &output,
               const metal::MetalBuffer &diagnostics, uint32_t rows) {
    if (int8Head && prefix == "language_model.lm_head") {
      if (rows >= 2 && rows <= 16) {
        int8Head->addProjection(graph, input, output, diagnostics, rows);
        return;
      }
      if (rows == 1) {
        addAffine(graph, input, weights.projection(prefix), output, diagnostics, rows);
        return;
      }
    }
    if (floatDenseCache && rows >= 2 && rows <= 16 && floatDenseCache->contains(prefix)) {
      if (selectiveFloat) {
        const auto &projection = weights.projection(prefix);
        const auto tile = flashFloatDenseSmallRowsPolicy(prefix, rows,
            projection.outputSize, projection.inputSize, projection.bits, projection.groupSize);
        if (tile)
          floatDenseCache->addSmallRows(graph, prefix, input, output, diagnostics, rows,
              *floatDenseWorkspace, *tile);
        else
          addAffine(graph, input, projection, output, diagnostics, rows);
        return;
      }
      floatDenseCache->addSmallRows(graph, prefix, input, output, diagnostics, rows,
          *floatDenseWorkspace, rows > 8 ? FlashFloatDenseSmallRowsTile::M16N64
                                       : FlashFloatDenseSmallRowsTile::M8N64);
    } else if (denseCache && smallDenseWorkspace && rows < 16 && denseCache->contains(prefix)) {
      const auto &projection = weights.projection(prefix);
      addDenseBF16SmallRows(backend, graph, input, denseCache->tensor(prefix), output,
          diagnostics, rows, *smallDenseWorkspace,
          projection.outputSize >= 1024 ? FlashDenseSmallRowsTile::M8N128
                                       : FlashDenseSmallRowsTile::M8N64);
    } else if (denseCache && rows >= 16 && denseCache->contains(prefix)) {
      const auto &projection = weights.projection(prefix);
      const auto tile = rows >= 128 && projection.outputSize >= 1024
          ? FlashAffineMPPTile::M32N128 : FlashAffineMPPTile::M16N64;
      denseCache->addProjection(graph, prefix, input, output, diagnostics, rows, tile);
    } else {
      addAffine(graph, input, weights.projection(prefix), output, diagnostics, rows);
    }
  }

  void hc(metal::CommandGraph &graph, const std::string &prefix,
          uint32_t rows, bool injection, bool normalizedReady = false) {
    const FlashHCGeometry geometry{rows, kWidth, 4, static_cast<float>(descriptor.normEpsilon)};
    const auto normalized = bf(Scratch::HCNormalized, rows, kHyper);
    const auto down = bf(Scratch::HCDown, rows, 320);
    const auto up = bf(Scratch::HCUp, rows, kHyper);
    const std::string norm = prefix + ".hc_norm.weight";
    if (!normalizedReady)
      addHCGroupedNorm(graph, bf(Scratch::Hyper, rows, kHyper), weights.tensor(norm),
                       normalized, geometry, weights.normConvention(norm));
    const auto &downProjection = weights.projection(prefix + ".input_mix_weight_down");
    const auto &upProjection = weights.projection(prefix + ".input_mix_weight_up");
    const auto *injectionProjection = injection
        ? &weights.projection(prefix + ".block_inject_weight") : nullptr;
    if (fuseHC && supportsHCFused(downProjection, upProjection, injectionProjection, geometry)) {
      const auto diagnostics = view(Scratch::Diagnostics, 4);
      addHCFusedDown(graph, normalized, downProjection, injectionProjection, down,
          injection ? bf(Scratch::HCInjectionWeights, rows, 4) : metal::MetalBuffer{},
          diagnostics, geometry);
      if (!cachedHCUp(graph, prefix + ".input_mix_weight_up", normalized, down,
          bf(Scratch::Mixed, rows, kWidth), diagnostics, rows))
        addHCFusedUpMix(graph, normalized, down, upProjection,
            bf(Scratch::Mixed, rows, kWidth), diagnostics, geometry);
      return;
    }
    project(graph, prefix + ".input_mix_weight_down", normalized, down,
        view(Scratch::Diagnostics, 4), rows);
    const FlashForwardActivationParams activation{rows, 320, 4};
    graph.add("flash_forward_hc_silu", {down, down, view(Scratch::Diagnostics, 4)},
              activation, {uint64_t{rows} * 320 / 256 + (uint64_t{rows} * 320 % 256 != 0), 1, 1});
    project(graph, prefix + ".input_mix_weight_up", down, up,
        view(Scratch::Diagnostics, 4), rows);
    if (injection) {
      const auto raw = bf(Scratch::HCRawInjection, rows, 4);
      project(graph, prefix + ".block_inject_weight", normalized, raw,
          view(Scratch::Diagnostics, 4), rows);
      addHCMixWithInjection(graph, normalized, up, raw, bf(Scratch::Mixed, rows, kWidth),
                             bf(Scratch::HCInjectionWeights, rows, 4), geometry);
    } else {
      addHCMix(graph, normalized, up, bf(Scratch::Mixed, rows, kWidth), geometry);
    }
  }

  bool cachedHCUp(metal::CommandGraph &graph, const std::string &prefix,
      const metal::MetalBuffer &normalized, const metal::MetalBuffer &activated,
      const metal::MetalBuffer &mixed, const metal::MetalBuffer &diagnostics,
      uint32_t rows) {
    hcUpCounters.recordAttempt(rows);
    if (!flashHCUpF32MPPGeometry(prefix, rows, 10240, 320)) {
      ++hcUpCounters.skippedUnsupportedGeometry;
      return false;
    }
    hcUpCounters.recordEligible(rows);
    if (!hcUpF32 || !fuseHC || !floatDenseCache || !floatDenseWorkspace) {
      ++hcUpCounters.skippedDependenciesOff;
      return false;
    }
    if (!floatDenseCache->contains(prefix)) {
      ++hcUpCounters.skippedMissingOperand;
      return false;
    }
    addHCFusedUpMixF32Cache(backend, graph, normalized, activated,
        floatDenseCache->tensor(prefix), mixed, diagnostics,
        {rows, kWidth, 4, static_cast<float>(descriptor.normEpsilon)}, *floatDenseWorkspace);
    hcUpCounters.recordCachedEncoded(rows);
    return true;
  }
};

FlashRequestState::FlashRequestState() = default;
FlashRequestState::~FlashRequestState() = default;
FlashRequestState::FlashRequestState(FlashRequestState &&) noexcept = default;
FlashRequestState &FlashRequestState::operator=(FlashRequestState &&) noexcept = default;
uint64_t FlashRequestState::logicalLength() const noexcept { return impl_ ? impl_->length : 0; }
uint32_t FlashRequestState::capacity() const noexcept { return impl_ ? impl_->capacity : 0; }
bool FlashRequestState::poisoned() const noexcept { return !impl_ || impl_->poisoned; }

FlashForward::FlashForward(metal::MetalBackend &backend, const FlashWeights &weights,
                           uint32_t capacity, uint32_t maximumRows, uint32_t maximumVerifyRows)
    : impl_(std::make_unique<Impl>(backend, weights, capacity, maximumRows, maximumVerifyRows)) {}
FlashForward::~FlashForward() = default;
FlashForward::FlashForward(FlashForward &&) noexcept = default;
FlashForward &FlashForward::operator=(FlashForward &&) noexcept = default;

uint64_t FlashForward::requestStateBytes(uint32_t capacity) {
  validateCapacity(capacity);
  uint64_t bytes = 36 * (roundAllocation(flashGDNConvolutionLaneBytes()) +
                          roundAllocation(flashGDNRecurrentLaneBytes()));
  const uint64_t blocks = (uint64_t{capacity} + 3) / 4;
  const uint64_t qsa = 2 * roundAllocation(uint64_t{capacity} * 512 * 2) +
      roundAllocation(uint64_t{capacity} * 128 * 2) + roundAllocation(blocks * 128 * 2) +
      roundAllocation(uint64_t{capacity} * 8);
  bytes += 12 * qsa;
  bytes += roundAllocation(2 * sizeof(int64_t));
  bytes += roundAllocation(uint64_t{9} * kHyper * 2);
  return bytes;
}

uint64_t FlashForward::workspaceBytes() const noexcept { return impl_ ? impl_->workspaceBytes : 0; }
uint64_t FlashForward::pleSSDStagingBytes() const noexcept {
  return impl_ && impl_->pleSSD ? impl_->pleSSD->allocatedBytes() : 0;
}
uint64_t FlashForward::verificationGDNStorageBytes() const noexcept {
  if (!impl_) return 0;
  uint64_t bytes = impl_->verifyRecurrent.sizeBytes() + impl_->verifyConvolution.sizeBytes();
  for (const auto &record : impl_->lazyGDNRecords) if (record) bytes += record->allocationBytes();
  return bytes;
}
bool FlashForward::lazyGDNRollbackEnabled() const noexcept { return impl_ && impl_->lazyGDN; }
FlashGDNLazyRollbackCounters FlashForward::lazyGDNRollbackCounters() const noexcept {
  FlashGDNLazyRollbackCounters result;
  if (impl_) for (const auto &record : impl_->lazyGDNRecords)
    if (record) result.add(record->counters());
  return result;
}

uint64_t FlashForward::workspacePlannedBytes(uint32_t capacity, uint32_t maximumRows,
                                            uint32_t maximumVerifyRows) {
  validateCapacity(capacity);
  if (!maximumRows || maximumRows > 2048 || maximumVerifyRows > kFlashSingletonMaximumVerifyRows || maximumVerifyRows > maximumRows)
    throw std::invalid_argument("Flash workspace admission geometry is unsupported");
  // The39 BF16 row buffers retain their current allocation geometry. Logits
  // and QSA have separately capped row extents below.
  constexpr std::array<uint32_t, 39> widths{
      kWidth, kHyper, kHyper, kHyper, 320, 4, 4, kWidth, kWidth, 12288,
      512, 512, 640, 10240, 6144, 48, 48, 48, 6144, 6144, 512, kSelections,
      kSelections * 640, kSelections * 640, kSelections * 640, kSelections * kWidth,
      640, 640, 640, kWidth, 1, kWidth, kHyper, kWidth, kHyper, kHyper,
      kHyper, kHyper, kHyper};
  uint64_t total = 0;
  for (uint32_t width : widths) total += roundAllocation(uint64_t{maximumRows} * width * 2);
  for (uint64_t extent : std::array<uint64_t, 5>{uint64_t{maximumRows} * 8,
      uint64_t{maximumRows} * 48 * 4, uint64_t{maximumRows} * kSelections * 8,
      uint64_t{maximumRows} * 16 * 8, 4}) total += roundAllocation(extent);
  const uint64_t cappedRows = std::min(maximumRows, uint32_t{128});
  total += roundAllocation(cappedRows * 248320 * 2);
  const uint64_t blocks = (uint64_t{capacity} + 3) / 4;
  for (uint64_t width : std::array<uint64_t, 6>{6144 * 2, 512 * 2, blocks * 4,
      512 * 4, uint64_t{24} * kFlashQSATokenWidth * 4, uint64_t{24} * kFlashQSATokenWidth * 2})
    total += roundAllocation(cappedRows * width);
  // Admit the largest optional online scratch before GPU construction. Scalar
  // F32 attention uses the same strided layout with only four active partitions.
  total += roundAllocation(cappedRows * 24 * 32 * 2 * 4);
  total += roundAllocation(cappedRows * 24 * 32 * 256 * 4);
  total += verificationWorkspaceBytes(maximumVerifyRows);
  // Conservative allowance for optional padding and diagnostic route capture.
  total += roundAllocation(uint64_t{16} * 32768 * 2);
  total += roundAllocation(uint64_t{48} * maximumRows * kSelections * 8);
  if (maximumVerifyRows > 1 && !flashGDNLazyRollbackEnabled())
    total += roundAllocation(flashGDNConvolutionLaneBytes());
  if (flashPLESSDStreamingValue(std::getenv("SPLASH_FLASH_PLE_SSD_STREAMING")))
    total += FlashPLESSD::plannedBytes(1, maximumRows);
  else if (fusionEnabled("SPLASH_FLASH_PLE_LOOKUP_FUSED"))
    total += roundAllocation(kFlashPLEFusedMaximumArgumentBytes);
  if (fusionEnabled("SPLASH_FLASH_GPU_GREEDY"))
    total += greedyGPUWorkspacePlannedBytes(
        std::min(maximumRows, uint32_t{kFlashGreedyGPUMaximumRows}), 248320);
  const bool bulkPrefillEnabled = qsaBulkPrefillEnabled();
  (void)qsaBulkPrefillSG8Enabled(bulkPrefillEnabled);
  if (bulkPrefillEnabled && maximumRows >= 2048)
    total += qsaBulkWorkspacePlannedBytes();
  return total;
}

std::string FlashForward::kernelRoutes() const {
  if (!impl_) throw std::logic_error("Flash forward is not initialized");
  return std::string(flashAffineSemantics()) +
      (impl_->bulkQSAWorkspace ? kFlashQSABulkPrefillRoute : "") +
      (impl_->bulkQSAWorkspace && impl_->bulkQSAPrefillSG8 ? kFlashQSABulkPrefillSG8Route : "") +
      (impl_->fuseHC ? ";hc-fused-literal-sg4" : ";hc-separate") +
      (impl_->fuseGDN ? ";gdn-decode-persistent512-verify-capture512" : ";gdn-separate") +
      (impl_->lazyGDN ? kFlashGDNLazyRollbackRoute : "") +
      (impl_->stagedGDN ? ";gdn-prefill-staged-v16-t16" : "") +
      (impl_->denseCache ? ";dense-cache-bf16-whole-k:" + impl_->denseCache->identitySha256() : ";dense-raw") +
      (impl_->denseCache && flashDenseM64OutEnabled() ? std::string(kFlashDenseM64OutSemantics) : "") +
      (impl_->denseCache && flashPrefillDenseTilesEnabled() ? std::string(kFlashPrefillDenseTilesSemantics) : "") +
      (impl_->smallDenseWorkspace ? ";dense-all-rows-bf16-static-operands-padded-m8" : "") +
      (impl_->floatDenseCache ? ";dense-f32-original-coefficients-multirow-mpp:" + impl_->floatDenseCache->identitySha256() : "") +
      (impl_->int8Head ? std::string(";") + kFlashInt8HeadSemantics + ":" + impl_->int8Head->identitySha256() +
          ";" + impl_->int8Head->codeStorageSemantics() : "") +
      (impl_->selectiveFloat ? std::string(";") + kFlashFloatDenseSmallRowsPolicySemantics : "") +
      (impl_->hcUpF32 && impl_->fuseHC && impl_->floatDenseCache
          ? std::string(";") + kFlashHCUpF32MPPSemantics : "") +
      (impl_->fuseSharedExpert ? std::string(";") + kFlashSharedExpertFusedSemantics : "") +
      (impl_->expertPlanIdentity.empty() ? "" : ";hot-expert-bf16-whole-k:" + impl_->expertPlanIdentity) +
      (impl_->int8ExpertStore ? std::string(";") + kFlashInt8ExpertStoreSemantics + ":" +
          impl_->int8ExpertStore->identitySha256() : "") +
      (impl_->pleSSD ? kFlashPLESSDRoute
                    : impl_->pleLookup ? ";ple-exact-i64-hash-direct128-q4g32-tier2"
                       : impl_->fusePLELookup ? ";ple-gather8-tier2-unavailable"
                                              : ";ple-source-gather8") +
      ";" + flashPLEPostRouteSemantics() +
      (impl_->gpuGreedy ? ";greedy-gpu-bf16-argmax-inline" : ";greedy-host-vocabulary") +
      (impl_->blockMoE ? std::string(";") + flashMoEBlockedRouteSemantics() : ";moe-vector") +
      (impl_->onlineQSA ? (impl_->mppQSA ? qsaOnlineMPPRouteSemantics()
                                       : ";qsa-fused-prep-f32-online-partitions4")
                        : ";qsa-bf16-probabilities");
}

FlashHCUpEncodedCounters FlashForward::hcUpEncodedCounters() const {
  if (!impl_) return {};
  std::lock_guard lock(impl_->mutex);
  return impl_->hcUpCounters;
}

uint64_t FlashForward::qsaOutF32N32EncodedCalls() const {
  if (!impl_) return 0;
  std::lock_guard lock(impl_->mutex);
  return impl_->floatDenseCache ? impl_->floatDenseCache->qsaOutF32N32Dispatches() : 0;
}

uint64_t FlashForward::qsaOutF32N32EncodedRealRows() const {
  if (!impl_) return 0;
  std::lock_guard lock(impl_->mutex);
  return impl_->floatDenseCache ? impl_->floatDenseCache->qsaOutF32N32RealRows() : 0;
}

FlashQSABulkCounters FlashForward::qsaBulkPrefillCounters() const {
  if (!impl_) return {};
  std::lock_guard lock(impl_->mutex);
  return impl_->bulkQSACounters;
}

uint64_t FlashForward::expertCachePlannedBytes(const FlashWeights &weights) {
  if (const auto directory = int8ExpertDirectory())
    return FlashInt8ExpertStore::plannedBytes(weights, *directory);
  const auto plan = hotExpertPlan(weights);
  if (!plan) return 0;
  uint64_t total = 0;
  for (uint32_t layer = 0; layer < 48; ++layer) {
    const auto bytes = FlashExpertDenseCache::plannedBytes(weights,
        "language_model.model.layers." + std::to_string(layer) + ".mlp.switch_mlp",
        plan->selectedExperts[layer]);
    if (bytes > UINT64_MAX - total) throw std::overflow_error("hot-expert plan bytes overflow");
    total += bytes;
  }
  return total;
}

uint64_t FlashForward::floatDenseCachePlannedBytes(const FlashWeights &weights) {
  if (!fusionEnabled("SPLASH_FLASH_FLOAT_DENSE_CACHE")) return 0;
  return FlashFloatDenseCache::plannedBytes(weights,
      FlashFloatDenseCache::defaultPrefixes(weights, !fusionEnabled("SPLASH_FLASH_INT8_HEAD")));
}

uint64_t FlashForward::int8HeadPlannedBytes(const FlashWeights &weights) {
  if (!fusionEnabled("SPLASH_FLASH_INT8_HEAD")) return 0;
  return FlashInt8Head::plannedBytes(weights, fusionEnabled("SPLASH_FLASH_FLOAT_DENSE_CACHE"));
}

const FlashTensor *FlashForward::cachedVocabulary() const {
  return impl_ && impl_->denseCache ? &impl_->denseCache->tensor("language_model.lm_head") : nullptr;
}

const FlashTensor *FlashForward::cachedFloatVocabulary() const {
  return impl_ && impl_->floatDenseCache && impl_->floatDenseCache->contains("language_model.lm_head")
      ? &impl_->floatDenseCache->tensor("language_model.lm_head") : nullptr;
}

FlashPersistedOperandStatus FlashForward::persistedOperandStatus() const {
  FlashPersistedOperandStatus result;
  if (!impl_) return result;
  if (impl_->denseCache) {
    result.bf16Tensors = impl_->denseCache->persistedTensorCount();
    result.bf16PayloadBytes = impl_->denseCache->persistedPayloadBytes();
    result.storeManifestSha256 = impl_->denseCache->operandStoreIdentitySha256();
  }
  if (impl_->floatDenseCache) {
    result.f32Tensors = impl_->floatDenseCache->persistedTensorCount();
    result.f32PayloadBytes = impl_->floatDenseCache->persistedPayloadBytes();
    const auto &identity = impl_->floatDenseCache->operandStoreIdentitySha256();
    if (!result.storeManifestSha256.empty() && !identity.empty() && result.storeManifestSha256 != identity)
      throw std::logic_error("Flash dense caches were loaded from different operand stores");
    if (!identity.empty()) result.storeManifestSha256 = identity;
  }
  return result;
}

std::vector<metal::MetalBuffer> FlashForward::cachedOperandsOnly() const {
  if (!impl_) return {};
  // Also checks that selected BF16/F32 caches came from the same saved store.
  (void)persistedOperandStatus();
  std::vector<metal::MetalBuffer> result;
  const auto append = [&](std::vector<metal::MetalBuffer> selected) {
    for (auto &buffer : selected) {
      if (!buffer || buffer.storage() != metal::BufferStorage::Shared ||
          !buffer.contents() || !buffer.sizeBytes())
        throw std::logic_error("Flash saved residency operand is not a mapped Shared buffer");
      result.push_back(std::move(buffer));
    }
  };
  if (impl_->denseCache) append(impl_->denseCache->persistedWeightBuffers());
  if (impl_->floatDenseCache) append(impl_->floatDenseCache->persistedWeightBuffers());
  // This store enumerates its derived base/rank allocations, never the raw
  // source projections retained for the Q4 miss route.
  if (impl_->int8ExpertStore) append(impl_->int8ExpertStore->immutableWeightBuffers());
  return result;
}

metal::MetalBuffer FlashForward::capturedExpertIDs(uint32_t &rows, uint32_t &stride) const {
  rows = impl_ ? impl_->capturedRows : 0;
  stride = impl_ ? impl_->maximumRows : 0;
  return impl_ ? impl_->capturedRoutes : metal::MetalBuffer{};
}

void FlashForward::batchProject(metal::CommandGraph &graph, const std::string &prefix,
                               const metal::MetalBuffer &input, const metal::MetalBuffer &output,
                               const metal::MetalBuffer &diagnostics, uint32_t rows) {
  impl_->project(graph, prefix, input, output, diagnostics, rows);
}

bool FlashForward::batchDenseSmallRowsEnabled() const noexcept {
  return impl_ && (bool(impl_->smallDenseWorkspace) || bool(impl_->floatDenseCache) || bool(impl_->int8Head));
}

bool FlashForward::batchSharedExpertFused(metal::CommandGraph &graph,
    const std::string &prefix, const metal::MetalBuffer &input,
    const metal::MetalBuffer &activated, const metal::MetalBuffer &diagnostics,
    uint32_t rows, const metal::MetalBuffer &tailGate,
    const metal::MetalBuffer &tailUp) {
  if (!impl_ || !flashSharedExpertFusedEligible(impl_->fuseSharedExpert,
      bool(impl_->denseCache), rows)) return false;
  const auto gate = prefix + ".gate_proj", up = prefix + ".up_proj";
  if (!impl_->denseCache->contains(gate) || !impl_->denseCache->contains(up)) return false;
  addSharedExpertFusedPrefill(impl_->backend, graph, impl_->denseCache->tensor(gate),
      impl_->denseCache->tensor(up), input, activated, diagnostics, rows,
      {tailGate, tailUp});
  return true;
}

bool FlashForward::batchFloatDenseEnabled() const noexcept {
  return impl_ && bool(impl_->floatDenseCache);
}

bool FlashForward::batchHCFusedUpMixF32(metal::CommandGraph &graph,
    const std::string &prefix, const metal::MetalBuffer &normalized,
    const metal::MetalBuffer &activated, const metal::MetalBuffer &mixed,
    const metal::MetalBuffer &diagnostics, uint32_t rows) {
  return impl_ && impl_->cachedHCUp(graph, prefix, normalized, activated, mixed, diagnostics, rows);
}

bool FlashForward::batchInt8HeadEnabled() const noexcept {
  return impl_ && bool(impl_->int8Head);
}

const FlashExpertDenseCache *FlashForward::batchExpertCache(uint32_t layer) const {
  if (!impl_ || layer >= 48) throw std::invalid_argument("invalid hot-expert layer index");
  return impl_->expertCaches[layer].get();
}
const FlashInt8ExpertStore *FlashForward::batchInt8ExpertStore() const noexcept {
  return impl_ ? impl_->int8ExpertStore.get() : nullptr;
}

const FlashPLEFused *FlashForward::batchPLELookup() const noexcept {
  return impl_ ? impl_->pleLookup.get() : nullptr;
}

void FlashForward::batchValidateExternalDestination(const metal::MetalBuffer &destination) const {
  if (!impl_ || !impl_->owner || !destination ||
      destination.storage() != metal::BufferStorage::Shared)
    throw std::invalid_argument("Flash source feature destination requires initialized Shared ownership");
  validateFlashBatchPrefillCopyRange(destination.contents(), destination.sizeBytes(), destination.sizeBytes());
  const auto reject = [&](const metal::MetalBuffer &buffer) {
    // A Private allocation cannot acquire a Shared view through backend.view.
    if (flashBatchPrefillCopyRangesOverlap(destination.contents(), destination.sizeBytes(),
                                          buffer.contents(), buffer.sizeBytes()))
      throw std::invalid_argument("Flash feature destination overlaps source workspace or immutable weights");
  };
  for (const auto &buffer : impl_->scratch) reject(buffer);
  for (const auto &buffer : {impl_->qsaWorkspace.queries, impl_->qsaWorkspace.indexQueries,
      impl_->qsaWorkspace.blockScores, impl_->qsaWorkspace.selectedBlocks,
      impl_->qsaWorkspace.attentionScores, impl_->qsaWorkspace.probabilities,
      impl_->qsaFastWorkspace.partitionStatistics, impl_->qsaFastWorkspace.partitionValues,
      impl_->greedyWorkspace.partials, impl_->greedyResults, impl_->capturedRoutes,
      impl_->verifyRecurrent, impl_->verifyConvolution, impl_->beforePLEHistory,
      impl_->beforePLEConvolution, impl_->retainedCount, impl_->beforeGDNConvolution}) reject(buffer);
  if (impl_->bulkQSAWorkspace) {
    const auto &bulk = *impl_->bulkQSAWorkspace;
    for (const auto &buffer : {bulk.prepared.queries, bulk.prepared.indexQueries,
        bulk.prepared.selectedBlocks, bulk.partials.partitionStatistics,
        bulk.partials.partitionValues}) reject(buffer);
  }
  const auto &blocked = impl_->blockedScratch;
  for (const auto &buffer : {blocked.buckets.counts, blocked.buckets.offsets, blocked.buckets.routeMap,
      blocked.buckets.canonicalToPacked, blocked.buckets.packedInputs, blocked.buckets.jobOffsets,
      blocked.buckets.jobCount, blocked.buckets.tileJobs, blocked.packedActivated, blocked.scatteredDown})
    reject(buffer);
  if (impl_->smallDenseWorkspace) reject(impl_->smallDenseWorkspace->paddedInput());
  if (impl_->floatDenseWorkspace) reject(impl_->floatDenseWorkspace->paddedInput());
  if (impl_->int8Head)
    for (const auto &buffer : impl_->int8Head->scratchBuffers()) reject(buffer);
  if (impl_->pleLookup) reject(impl_->pleLookup->sources_);
  if (impl_->pleSSD)
    for (const auto &buffer : impl_->pleSSD->scratchBuffers()) reject(buffer);
  for (const auto &buffer : impl_->weights.immutableWeightBuffers()) reject(buffer);
  for (const auto &record : impl_->lazyGDNRecords)
    if (record) for (const auto &buffer : record->arenaBuffers()) reject(buffer);
  if (impl_->denseCache)
    for (const auto &buffer : impl_->denseCache->immutableWeightBuffers()) reject(buffer);
  if (impl_->floatDenseCache)
    for (const auto &buffer : impl_->floatDenseCache->immutableWeightBuffers()) reject(buffer);
  if (impl_->int8Head)
    for (const auto &buffer : impl_->int8Head->immutableWeightBuffers()) reject(buffer);
  for (const auto &cache : impl_->expertCaches)
    if (cache) for (const auto &buffer : cache->immutableWeightBuffers()) reject(buffer);
  if (impl_->int8ExpertStore)
    for (const auto &buffer : impl_->int8ExpertStore->immutableWeightBuffers()) reject(buffer);
}

bool FlashForward::ownsState(const FlashRequestState &state) const noexcept {
  return impl_ && state.impl_ && state.impl_->owner == impl_->owner &&
      !state.impl_->poisoned && !state.impl_->pendingVerification &&
      state.impl_->capacity == impl_->capacity;
}

metal::MetalBackend &FlashForward::batchBackend() const {
  if (!impl_) throw std::logic_error("Flash trunk is not initialized"); return impl_->backend;
}
const FlashWeights &FlashForward::batchWeights() const {
  if (!impl_) throw std::logic_error("Flash trunk is not initialized"); return impl_->weights;
}
uint32_t FlashForward::batchCapacity() const {
  if (!impl_) throw std::logic_error("Flash trunk is not initialized"); return impl_->capacity;
}
std::mutex &FlashForward::batchMutex() const {
  if (!impl_) throw std::logic_error("Flash trunk is not initialized"); return impl_->mutex;
}

uint64_t FlashForward::verificationWorkspaceBytes(uint32_t maximumVerifyRows) {
  if (maximumVerifyRows > kFlashSingletonMaximumVerifyRows)
    throw std::invalid_argument("Flash verification maximum rows exceeds16");
  if (maximumVerifyRows <= 1) return 0;
  const uint64_t gdn = flashGDNLazyRollbackEnabled()
      ? uint64_t{36} * FlashGDNLazyRollback::plannedBytes(maximumVerifyRows, 1)
      : uint64_t{36} * (maximumVerifyRows - 1) *
          (roundAllocation(flashGDNRecurrentLaneBytes()) + roundAllocation(flashGDNConvolutionLaneBytes()));
  return gdn +
      roundAllocation(2 * sizeof(int64_t)) + roundAllocation(uint64_t{9} * kHyper * 2) +
      roundAllocation(sizeof(uint32_t));
}

FlashRequestState FlashForward::createState() {
  if (!impl_) throw std::logic_error("Flash forward is not initialized");
  std::lock_guard lock(impl_->mutex);
  FlashRequestState result;
  result.impl_ = std::make_unique<FlashRequestState::Impl>();
  auto &state = *result.impl_;
  state.owner = impl_->owner;
  state.capacity = impl_->capacity;
  for (uint32_t layer = 0; layer < impl_->descriptor.layers; ++layer) {
    if (impl_->descriptor.layerKinds[layer] == FlashLayerKind::GatedDeltaNet) {
      state.gdn[layer].convolution = impl_->backend.allocateBuffer(
          roundAllocation(flashGDNConvolutionLaneBytes()), metal::BufferStorage::Shared,
          "flash-request-gdn-convolution");
      state.gdn[layer].recurrent = impl_->backend.allocateBuffer(
          roundAllocation(flashGDNRecurrentLaneBytes()), metal::BufferStorage::Shared,
          "flash-request-gdn-recurrent");
      clear(state.gdn[layer].convolution); clear(state.gdn[layer].recurrent);
    } else {
      state.qsa[layer] = allocateQSAState(impl_->backend, state.capacity);
      clear(state.qsa[layer].keys); clear(state.qsa[layer].values);
      clear(state.qsa[layer].rawIndexKeys); clear(state.qsa[layer].pooledKeys);
      clear(state.qsa[layer].indexPositions);
    }
  }
  state.pleHistory = impl_->backend.allocateBuffer(roundAllocation(2 * sizeof(int64_t)),
      metal::BufferStorage::Shared, "flash-request-ple-history");
  state.pleConvolution = impl_->backend.allocateBuffer(roundAllocation(uint64_t{9} * kHyper * 2),
      metal::BufferStorage::Shared, "flash-request-ple-convolution");
  clear(state.pleHistory); clear(state.pleConvolution);
  auto *history = static_cast<int64_t *>(state.pleHistory.contents());
  history[0] = history[1] = impl_->descriptor.pleHistoryEos;
  return result;
}

FlashForwardResult FlashForward::forward(FlashRequestState &request,
                                         std::span<const uint32_t> tokens,
                                         bool returnAllLogits, bool captureHidden) {
  return forwardImpl(request, tokens, returnAllLogits, captureHidden, false);
}

FlashForwardResult FlashForward::verify(FlashRequestState &request,
                                        std::span<const uint32_t> tokens) {
  return forwardImpl(request, tokens, true, true, true);
}

FlashForwardResult FlashForward::forwardImpl(FlashRequestState &request,
                                             std::span<const uint32_t> tokens,
                                             bool returnAllLogits, bool captureHidden,
                                             bool verification) {
  if (!impl_) throw std::logic_error("Flash forward is not initialized");
  std::lock_guard lock(impl_->mutex);
  if (!request.impl_ || request.impl_->owner != impl_->owner || request.impl_->poisoned)
    throw std::invalid_argument("Flash forward requires its own healthy request state");
  auto &state = *request.impl_;
  if (state.pendingVerification && !impl_->pendingRows)
    throw std::logic_error("resolve the pending batch verification before a trunk call");
  if (impl_->pendingRows && impl_->pendingIdentity.expired())
    impl_->finishPending(); // The cancelled/destroyed request cannot be resumed.
  if (impl_->pendingRows)
    throw std::logic_error("resolve the previous Flash verification before another trunk call");
  if (verification && (!impl_->maximumVerifyRows || tokens.size() > impl_->maximumVerifyRows))
    throw std::invalid_argument("Flash target verification window exceeds provisioned prefix tape");
  if (tokens.empty() || tokens.size() > impl_->maximumRows || state.length > state.capacity ||
      tokens.size() > state.capacity - state.length)
    throw std::invalid_argument("Flash forward token window exceeds rows/context capacity");
  if (returnAllLogits && tokens.size() > impl_->maximumLogitRows)
    throw std::invalid_argument("Flash all-logits window exceeds bounded head storage");
  const uint32_t rows = static_cast<uint32_t>(tokens.size());
  const uint32_t begin = static_cast<uint32_t>(state.length);
  impl_->capturedRows = 0;
  auto tokenIDs = impl_->view(Scratch::TokenIDs, uint64_t{rows} * 8);
  auto *hostIDs = static_cast<int64_t *>(tokenIDs.contents());
  if (!hostIDs) throw std::logic_error("Flash token input is not shared");
  for (uint32_t index = 0; index < rows; ++index) {
    if (tokens[index] >= impl_->descriptor.vocabularySize)
      throw std::invalid_argument("Flash input contains an out-of-vocabulary token");
    hostIDs[index] = tokens[index];
  }
  auto diag = impl_->view(Scratch::Diagnostics, 4);
  clear(diag);
  const auto bf = [&](Scratch slot, uint32_t width) { return impl_->bf(slot, rows, width); };
  const auto affine = [&](metal::CommandGraph &graph, const std::string &prefix,
                          const metal::MetalBuffer &input, const metal::MetalBuffer &output) {
    impl_->project(graph, prefix, input, output, diag, rows);
  };
  metal::CommandGraph graph;
  uint32_t bulkQSALayerCalls = 0;
  struct LazyTrialGuard {
    Impl *source;
    bool keep = false;
    ~LazyTrialGuard() { if (!keep) source->abortLazyRecords(); }
  } lazyGuard{impl_.get(), !verification};
  const auto hyper = bf(Scratch::Hyper, kHyper);
  const auto mixed = bf(Scratch::Mixed, kWidth);
  const auto branch = bf(Scratch::Branch, kWidth);
  const auto embedding = bf(Scratch::Embedding, kWidth);
  const FlashHCGeometry hcGeometry{rows, kWidth, 4, static_cast<float>(impl_->descriptor.normEpsilon)};
  const FlashPLEGeometry pleGeometry{1, rows, kWidth, 4, impl_->descriptor.pleHistoryEos,
      impl_->descriptor.vocabularySize, static_cast<float>(impl_->descriptor.normEpsilon)};
  if (impl_->pleSSD) {
    const auto *history = static_cast<const int64_t *>(state.pleHistory.contents());
    if (!history) throw std::logic_error("Flash PLE SSD request history is not Shared");
    impl_->pleSSD->prepare({hostIDs, rows}, {history, 2}, impl_->pleWeights, pleGeometry);
  }
  addAffineEmbedding(graph, impl_->weights.projection("language_model.model.embed_tokens"),
                       tokenIDs, embedding, diag, rows);
  addHCExpand(graph, embedding, hyper, hcGeometry);
  bool normalizedReady = false;
  for (uint32_t layer = 0; layer < impl_->descriptor.layers; ++layer) {
    const std::string prefix = "language_model.model.layers." + std::to_string(layer);
    if (layer == impl_->descriptor.pleLayerIndices.front()) {
      if (verification && rows > 1) {
        impl_->copy(graph, state.pleHistory, impl_->beforePLEHistory, 2 * sizeof(int64_t));
        impl_->copy(graph, state.pleConvolution, impl_->beforePLEConvolution, uint64_t{9} * kHyper * 2);
      }
      const auto ngram = impl_->view(Scratch::NgramIDs, uint64_t{rows} * 16 * 8);
      if (impl_->pleSSD) {
        impl_->pleSSD->addHashGather(graph, impl_->pleWeights, tokenIDs,
            state.pleHistory, ngram, bf(Scratch::PLEEmbedding, kWidth), diag, pleGeometry);
      } else if (!impl_->pleLookup || !impl_->pleLookup->addHashGather(graph, tokenIDs,
          state.pleHistory, ngram, bf(Scratch::PLEEmbedding, kWidth), diag, pleGeometry)) {
        addPLENgramIDs(graph, impl_->pleWeights, tokenIDs, state.pleHistory, ngram, diag, pleGeometry);
        addPLEGather(graph, impl_->pleWeights, ngram, bf(Scratch::PLEEmbedding, kWidth), diag, pleGeometry);
      }
      affine(graph, prefix + ".ple.key_proj", bf(Scratch::PLEEmbedding, kWidth), bf(Scratch::PLEKey, kHyper));
      affine(graph, prefix + ".ple.value_proj", bf(Scratch::PLEEmbedding, kWidth), bf(Scratch::PLEValue, kWidth));
      const FlashPLEPostScratch post{bf(Scratch::PLENormalizedKeys, kHyper),
          bf(Scratch::PLENormalizedQueries, kHyper), bf(Scratch::PLEGated, kHyper),
          bf(Scratch::PLENormalizedConvolution, kHyper)};
      addPLEPostProjectAndInject(graph, impl_->pleWeights, hyper, bf(Scratch::PLEKey, kHyper),
          bf(Scratch::PLEValue, kWidth), post, state.pleConvolution,
          bf(Scratch::PLEOutput, kHyper), diag, pleGeometry);
      normalizedReady = false;
    }
    impl_->hc(graph, prefix + ".attn_hyper_connection", rows, true, normalizedReady);
    normalizedReady = false;
    const auto attentionOutput = bf(Scratch::AttentionOutput, 6144);
    if (impl_->descriptor.layerKinds[layer] == FlashLayerKind::GatedDeltaNet) {
      const std::string attention = prefix + ".linear_attn";
      const auto qkv = bf(Scratch::QProjection, 10240);
      affine(graph, attention + ".in_proj_qkv", mixed, qkv);
      affine(graph, attention + ".in_proj_z", mixed, bf(Scratch::GDNZ, 6144));
      affine(graph, attention + ".in_proj_a", mixed, bf(Scratch::GDNA, 48));
      affine(graph, attention + ".in_proj_b", mixed, bf(Scratch::GDNB, 48));
      const FlashGDNWeights weights{&impl_->weights.tensor(attention + ".conv1d.weight"),
          &impl_->weights.tensor(attention + ".A_log"), &impl_->weights.tensor(attention + ".dt_bias"),
          &impl_->weights.tensor(attention + ".norm.weight")};
      const FlashGDNBuffers buffers{qkv, bf(Scratch::GDNZ, 6144), bf(Scratch::GDNA, 48),
          bf(Scratch::GDNB, 48), bf(Scratch::GDNMixed, 10240),
          impl_->view(Scratch::GDNDecay, uint64_t{rows} * 48 * 4), bf(Scratch::GDNBeta, 48),
          bf(Scratch::GDNRecurrentRows, 6144), attentionOutput, diag};
      if (verification && rows > 1) {
        if (impl_->lazyGDN) {
          impl_->lazyGDNTickets[layer] = impl_->lazyGDNRecords[layer]->begin(graph, weights, buffers,
              state.gdn[layer], rows, 1, static_cast<float>(impl_->descriptor.normEpsilon));
        } else if (impl_->fuseGDN) {
          impl_->copy(graph, state.gdn[layer].convolution, impl_->beforeGDNConvolution,
              flashGDNConvolutionLaneBytes());
          const uint64_t recurrentStride = roundAllocation(flashGDNRecurrentLaneBytes());
          const uint64_t firstCell = uint64_t{impl_->gdnSlots[layer]} * (impl_->maximumVerifyRows - 1);
          const auto tape = impl_->backend.view(impl_->verifyRecurrent,
              firstCell * recurrentStride, uint64_t{rows - 1} * recurrentStride);
          const FlashGDNCapture capture{tape, recurrentStride,
              uint64_t{rows - 1} * recurrentStride, rows - 1, 512};
          addGDNFusedCaptured(graph, weights, buffers, state.gdn[layer], capture, rows, 1,
              static_cast<float>(impl_->descriptor.normEpsilon));
          const uint64_t historyRowBytes = uint64_t{10240} * 2;
          for (uint32_t kept = 1; kept < rows; ++kept) {
            const auto destination = impl_->prefixTape(false, layer, kept - 1);
            const auto history = *flashMTPConvolutionPrefix(kept);
            if (history.oldRows) {
              const uint64_t bytes = uint64_t{history.oldRows} * historyRowBytes;
              impl_->copy(graph, impl_->backend.view(impl_->beforeGDNConvolution,
                  uint64_t{history.oldBegin} * historyRowBytes, bytes),
                  impl_->backend.view(destination, 0, bytes), bytes);
            }
            const uint64_t bytes = uint64_t{history.inputRows} * historyRowBytes;
            impl_->copy(graph, impl_->backend.view(qkv,
                uint64_t{history.inputBegin} * historyRowBytes, bytes),
                impl_->backend.view(destination, uint64_t{history.destinationInputBegin} * historyRowBytes, bytes), bytes);
          }
        } else {
        const auto slice = [&](const metal::MetalBuffer &buffer, uint32_t row, uint32_t width, uint32_t elementBytes = 2) {
          const uint64_t extent = uint64_t{width} * elementBytes;
          return impl_->backend.view(buffer, uint64_t{row} * extent, extent);
        };
        for (uint32_t row = 0; row < rows; ++row) {
          const FlashGDNBuffers single{slice(buffers.qkv, row, 10240), slice(buffers.z, row, 6144),
              slice(buffers.a, row, 48), slice(buffers.b, row, 48), slice(buffers.mixed, row, 10240),
              slice(buffers.decay, row, 48, 4), slice(buffers.beta, row, 48),
              slice(buffers.recurrentRows, row, 6144), slice(buffers.output, row, 6144), diag};
          addGDN(graph, weights, single, state.gdn[layer], 1, 1,
                   static_cast<float>(impl_->descriptor.normEpsilon));
          if (row + 1 < rows) {
            impl_->copy(graph, state.gdn[layer].convolution, impl_->prefixTape(false, layer, row),
                          flashGDNConvolutionLaneBytes());
            impl_->copy(graph, state.gdn[layer].recurrent, impl_->prefixTape(true, layer, row),
                          flashGDNRecurrentLaneBytes());
          }
        }
        }
      } else {
        if (impl_->stagedGDN && rows >= 64)
          addGDNStagedPrefill(graph, weights, buffers, state.gdn[layer], rows, 1,
              FlashGDNStageTile::Values16Time16,
              static_cast<float>(impl_->descriptor.normEpsilon));
        else if (impl_->fuseGDN)
          addGDNFused(graph, weights, buffers, state.gdn[layer], rows, 1,
              rows == 1 ? FlashGDNFusion::PersistentHead512 : FlashGDNFusion::Prepare,
              static_cast<float>(impl_->descriptor.normEpsilon));
        else
          addGDN(graph, weights, buffers, state.gdn[layer], rows, 1,
                   static_cast<float>(impl_->descriptor.normEpsilon));
      }
      affine(graph, attention + ".out_proj", attentionOutput, branch);
    } else {
      const std::string attention = prefix + ".self_attn";
      const auto q = bf(Scratch::QProjection, 12288);
      const auto k = bf(Scratch::KProjection, 512);
      const auto v = bf(Scratch::VProjection, 512);
      const auto index = bf(Scratch::IndexProjection, 640);
      affine(graph, attention + ".q_proj", mixed, q);
      affine(graph, attention + ".k_proj", mixed, k);
      affine(graph, attention + ".v_proj", mixed, v);
      affine(graph, attention + ".indexer.index_qk_proj", mixed, index);
      const std::string qNorm = attention + ".q_norm.weight", kNorm = attention + ".k_norm.weight";
      const std::string iqNorm = attention + ".indexer.q_layernorm.weight";
      const std::string ikNorm = attention + ".indexer.k_layernorm.weight";
      // QSA keeps its qualified128-row scratch geometry. Each complete chunk
      // appends/normalizes/pools/selects/attends before that scratch is reused.
      const auto slice = [&](const metal::MetalBuffer &buffer, uint32_t offset,
                              uint32_t count, uint32_t width) {
        return impl_->backend.view(buffer, uint64_t{offset} * width * 2,
                                     uint64_t{count} * width * 2);
      };
      if (impl_->bulkQSAWorkspace && qsaBulkPrefillGeometry(begin, rows, state.capacity, verification)) {
        const FlashQSAFastInputs bulkInputs{q, k, v, index,
            &impl_->weights.tensor(qNorm), &impl_->weights.tensor(kNorm),
            &impl_->weights.tensor(iqNorm), &impl_->weights.tensor(ikNorm),
            attentionOutput, diag, {},
            impl_->weights.normConvention(qNorm), impl_->weights.normConvention(kNorm),
            impl_->weights.normConvention(iqNorm), impl_->weights.normConvention(ikNorm),
            impl_->descriptor.normEpsilon, impl_->descriptor.rotaryTheta};
        addQSABulkPrefill(impl_->backend, graph, bulkInputs, state.qsa[layer],
            impl_->qsaWorkspace, impl_->qsaFastWorkspace, *impl_->bulkQSAWorkspace, begin, rows,
            impl_->bulkQSAPrefillSG8);
        ++bulkQSALayerCalls;
      } else {
      for (uint32_t offset = 0; offset < rows;) {
        const uint32_t count = std::min(rows - offset, impl_->qsaWorkspace.maximumRows);
        if (impl_->onlineQSA) {
          const FlashQSAFastInputs inputs{slice(q, offset, count, 12288),
              slice(k, offset, count, 512), slice(v, offset, count, 512),
              slice(index, offset, count, 640), &impl_->weights.tensor(qNorm),
              &impl_->weights.tensor(kNorm), &impl_->weights.tensor(iqNorm),
              &impl_->weights.tensor(ikNorm), slice(attentionOutput, offset, count, 6144), diag, {},
              impl_->weights.normConvention(qNorm), impl_->weights.normConvention(kNorm),
              impl_->weights.normConvention(iqNorm), impl_->weights.normConvention(ikNorm),
              impl_->descriptor.normEpsilon, impl_->descriptor.rotaryTheta};
          const uint32_t partitions = impl_->mppQSA
              ? qsaOnlineMPPRoutePartitions(begin + offset, count) : 0;
          if (partitions)
            addQSAOnlineMPP(graph, inputs, state.qsa[layer], impl_->qsaWorkspace,
                impl_->qsaFastWorkspace, begin + offset, count, partitions, true);
          else
            addQSAFast(graph, inputs, state.qsa[layer], impl_->qsaWorkspace,
                impl_->qsaFastWorkspace, begin + offset, count,
                FlashQSAFastMode::PartitionedF32Probabilities, 4, true);
        } else {
        addQSA(graph, slice(q, offset, count, 12288), slice(k, offset, count, 512),
            slice(v, offset, count, 512), slice(index, offset, count, 640),
            impl_->weights.tensor(qNorm), impl_->weights.tensor(kNorm),
            impl_->weights.tensor(iqNorm), impl_->weights.tensor(ikNorm), state.qsa[layer],
            impl_->qsaWorkspace, slice(attentionOutput, offset, count, 6144), diag, begin + offset, count,
            impl_->weights.normConvention(qNorm), impl_->weights.normConvention(kNorm),
            impl_->weights.normConvention(iqNorm), impl_->weights.normConvention(ikNorm),
            impl_->descriptor.normEpsilon, impl_->descriptor.rotaryTheta);
        }
        offset += count;
      }
      }
      affine(graph, attention + ".o_proj", attentionOutput, branch);
    }
    const std::string mlpNorm = prefix + ".mlp_hyper_connection.hc_norm.weight";
    if (impl_->fuseHC && rows <= 32) {
      addHCFusedInjectNorm(graph, hyper, branch, bf(Scratch::HCInjectionWeights, 4),
          impl_->weights.tensor(mlpNorm), hyper, bf(Scratch::HCNormalized, kHyper), diag,
          hcGeometry, impl_->weights.normConvention(mlpNorm));
      normalizedReady = true;
    } else {
      addHCInject(graph, hyper, branch, bf(Scratch::HCInjectionWeights, 4), hyper, hcGeometry);
    }
    impl_->hc(graph, prefix + ".mlp_hyper_connection", rows, true, normalizedReady);
    const std::string mlp = prefix + ".mlp";
    const auto router = bf(Scratch::Router, 512);
    const auto ids = impl_->view(Scratch::ExpertIDs, uint64_t{rows} * kSelections * 8);
    const auto route = bf(Scratch::RouteWeights, kSelections);
    if (impl_->denseCache && rows >= 16)
      addDenseBF16WholeK(impl_->backend, graph, mixed, impl_->weights.tensor(mlp + ".gate.weight"),
          router, diag, rows, FlashAffineMPPTile::M16N64);
    else
      addDenseBF16(graph, mixed, impl_->weights.tensor(mlp + ".gate.weight"), router, diag, rows);
    addRoute(graph, router, ids, route, diag, rows, 512, kSelections);
    if (impl_->capturedRoutes)
      impl_->copy(graph, ids, impl_->backend.view(impl_->capturedRoutes,
          uint64_t{layer} * impl_->maximumRows * kSelections * 8,
          uint64_t{rows} * kSelections * 8), uint64_t{rows} * kSelections * 8);
    const bool blocked = impl_->blockMoE && rows >= 256;
    if (blocked) {
      const auto tile = flashMoEBlockedTile(rows, bool(impl_->expertCaches[layer]));
      addMoEBlockedPack(graph, mixed, ids, impl_->blockedScratch, diag, rows, tile);
      if (impl_->int8ExpertStore) {
        impl_->int8ExpertStore->addGateUp(graph, layer, impl_->blockedScratch, diag, rows, tile);
        impl_->int8ExpertStore->addDownScatter(graph, layer, impl_->blockedScratch, diag, rows, tile);
      } else if (impl_->expertCaches[layer]) {
        impl_->expertCaches[layer]->addGateUp(graph, impl_->blockedScratch, diag, rows, tile);
        impl_->expertCaches[layer]->addDownScatter(graph, impl_->blockedScratch, diag, rows, tile);
      } else {
        addMoEBlockedGateUp(graph, impl_->weights.projection(mlp + ".switch_mlp.gate_proj"),
            impl_->weights.projection(mlp + ".switch_mlp.up_proj"), impl_->blockedScratch,
            diag, rows, tile);
        addMoEBlockedDownScatter(graph, impl_->weights.projection(mlp + ".switch_mlp.down_proj"),
            impl_->blockedScratch, diag, rows, tile);
      }
    } else {
    addGatheredAffine(graph, mixed, impl_->weights.projection(mlp + ".switch_mlp.gate_proj"), ids,
        bf(Scratch::ExpertGate, kSelections * 640), diag, rows, kSelections);
    addGatheredAffine(graph, mixed, impl_->weights.projection(mlp + ".switch_mlp.up_proj"), ids,
        bf(Scratch::ExpertUp, kSelections * 640), diag, rows, kSelections);
    addSiLUMultiply(graph, bf(Scratch::ExpertGate, kSelections * 640),
        bf(Scratch::ExpertUp, kSelections * 640), bf(Scratch::ExpertIntermediate, kSelections * 640),
        diag, rows, 640, kSelections);
    addGatheredAffine(graph, bf(Scratch::ExpertIntermediate, kSelections * 640),
        impl_->weights.projection(mlp + ".switch_mlp.down_proj"), ids,
        bf(Scratch::ExpertDown, kSelections * kWidth), diag, rows, kSelections, true);
    }
    if (!batchSharedExpertFused(graph, mlp + ".shared_expert", mixed,
        bf(Scratch::SharedIntermediate, 640), diag, rows,
        bf(Scratch::SharedGate, 640), bf(Scratch::SharedUp, 640))) {
      affine(graph, mlp + ".shared_expert.gate_proj", mixed, bf(Scratch::SharedGate, 640));
      affine(graph, mlp + ".shared_expert.up_proj", mixed, bf(Scratch::SharedUp, 640));
      addSiLUMultiply(graph, bf(Scratch::SharedGate, 640), bf(Scratch::SharedUp, 640),
          bf(Scratch::SharedIntermediate, 640), diag, rows, 640);
    }
    affine(graph, mlp + ".shared_expert.down_proj", bf(Scratch::SharedIntermediate, 640),
        bf(Scratch::SharedDown, kWidth));
    affine(graph, mlp + ".shared_expert_gate", mixed, bf(Scratch::SharedGateLogit, 1));
    addCombine(graph, blocked ? impl_->blockedScratch.scatteredDown : bf(Scratch::ExpertDown, kSelections * kWidth), ids, route,
        bf(Scratch::SharedDown, kWidth), bf(Scratch::SharedGateLogit, 1), branch,
        diag, rows, kWidth, 512, kSelections);
    const bool nextHasPLE = layer + 1 == impl_->descriptor.pleLayerIndices.front();
    normalizedReady = impl_->fuseHC && rows <= 32 && !nextHasPLE;
    if (normalizedReady) {
      const std::string nextHC = layer + 1 == impl_->descriptor.layers
          ? "language_model.model.hyper_connection_mixer"
          : "language_model.model.layers." + std::to_string(layer + 1) + ".attn_hyper_connection";
      const auto nextNorm = nextHC + ".hc_norm.weight";
      addHCFusedInjectNorm(graph, hyper, branch, bf(Scratch::HCInjectionWeights, 4),
          impl_->weights.tensor(nextNorm), hyper, bf(Scratch::HCNormalized, kHyper), diag,
          hcGeometry, impl_->weights.normConvention(nextNorm));
    } else {
      addHCInject(graph, hyper, branch, bf(Scratch::HCInjectionWeights, 4), hyper, hcGeometry);
    }
  }
  impl_->hc(graph, "language_model.model.hyper_connection_mixer", rows, false, normalizedReady);
  const uint32_t logitRows = returnAllLogits ? rows : 1;
  const auto headInput = returnAllLogits ? mixed : impl_->backend.view(mixed,
      uint64_t{rows - 1} * kWidth * 2, uint64_t{kWidth} * 2);
  const auto logits = impl_->bf(Scratch::HeadLogits, logitRows, impl_->descriptor.vocabularySize);
  impl_->project(graph, "language_model.lm_head", headInput, logits, diag, logitRows);
  metal::MetalBuffer compactGreedy;
  if (impl_->gpuGreedy && logitRows <= kFlashGreedyGPUMaximumRows) {
    compactGreedy = impl_->backend.view(impl_->greedyResults, 0,
        uint64_t{logitRows} * sizeof(FlashGreedyGPURowResult));
    addGreedyGPU(graph, logits, impl_->greedyWorkspace, compactGreedy,
        logitRows, impl_->descriptor.vocabularySize);
  }
  metal::CommandTiming timing;
  try {
    timing = impl_->backend.submitCommand(graph.dispatches());
    uint32_t status = 0;
    std::memcpy(&status, diag.contents(), sizeof(status));
    if (status) throw std::runtime_error("Flash forward sticky diagnostics failed: " + std::to_string(status));
  } catch (...) {
    state.poisoned = true;
    throw;
  }
  state.length += rows;
  if (bulkQSALayerCalls) {
    ++impl_->bulkQSACounters.completedPrefillCalls;
    impl_->bulkQSACounters.completedPrefillTokens += rows;
    impl_->bulkQSACounters.completedLayerCalls += bulkQSALayerCalls;
    if (impl_->bulkQSAPrefillSG8)
      impl_->bulkQSACounters.completedSG8LayerCalls += bulkQSALayerCalls;
  }
  impl_->capturedRows = impl_->capturedRoutes ? rows : 0;
  if (verification) {
    state.pendingVerification = true;
    impl_->pendingIdentity = state.identity;
    impl_->pendingBegin = begin;
    impl_->pendingRows = rows;
  }
  lazyGuard.keep = true;
  return {timing, logits, logitRows, state.length, state.capacity,
      captureHidden ? hyper : metal::MetalBuffer{}, compactGreedy,
      compactGreedy ? logitRows : 0};
}

metal::CommandTiming FlashForward::commitVerify(FlashRequestState &request, uint32_t retained) {
  if (!impl_) throw std::logic_error("Flash forward is not initialized");
  std::lock_guard lock(impl_->mutex);
  if (!request.impl_ || request.impl_->owner != impl_->owner || request.impl_->poisoned ||
      impl_->pendingIdentity.lock() != request.impl_->identity || !impl_->pendingRows ||
      !retained || retained > impl_->pendingRows)
    throw std::invalid_argument("Flash verification commit has no matching healthy prefix");
  auto &state = *request.impl_;
  if (state.length != impl_->pendingBegin + impl_->pendingRows)
    throw std::logic_error("Flash verification logical length changed before commit");
  struct LazyCommitGuard {
    Impl *source;
    FlashRequestState::Impl *state;
    bool keep = false;
    ~LazyCommitGuard() {
      if (!keep && source->lazyGDN) {
        state->poisoned = true; state->pendingVerification = false; source->finishPending();
      }
    }
  } lazyGuard{impl_.get(), &state};
  if (retained == impl_->pendingRows) {
    if (impl_->lazyGDN && impl_->pendingRows > 1) {
      metal::CommandGraph noWork;
      for (uint32_t layer = 0; layer < impl_->descriptor.layers; ++layer)
        if (impl_->lazyGDNRecords[layer])
          impl_->lazyGDNRecords[layer]->commit(noWork, impl_->lazyGDNTickets[layer],
              std::span<const uint32_t>(&retained, 1));
      if (!noWork.empty()) throw std::logic_error("fully accepted lazy GDN must not replay");
    }
    state.pendingVerification = false; impl_->finishPending(); lazyGuard.keep = true; return {};
  }
  auto diag = impl_->view(Scratch::Diagnostics, 4);
  clear(diag);
  std::memcpy(impl_->retainedCount.contents(), &retained, sizeof(retained));
  metal::CommandGraph graph;
  for (uint32_t layer = 0; layer < impl_->descriptor.layers; ++layer) {
    if (impl_->descriptor.layerKinds[layer] != FlashLayerKind::GatedDeltaNet) continue;
    if (impl_->lazyGDN) {
      impl_->lazyGDNRecords[layer]->commit(graph, impl_->lazyGDNTickets[layer],
          std::span<const uint32_t>(&retained, 1));
    } else {
      impl_->copy(graph, impl_->prefixTape(false, layer, retained - 1), state.gdn[layer].convolution,
                    flashGDNConvolutionLaneBytes());
      impl_->copy(graph, impl_->prefixTape(true, layer, retained - 1), state.gdn[layer].recurrent,
                    flashGDNRecurrentLaneBytes());
    }
  }
  const uint32_t rows = impl_->pendingRows;
  const FlashPLEGeometry geometry{1, rows, kWidth, 4, impl_->descriptor.pleHistoryEos,
      impl_->descriptor.vocabularySize, static_cast<float>(impl_->descriptor.normEpsilon)};
  addPLERestorePrefix(graph, impl_->beforePLEHistory, impl_->view(Scratch::TokenIDs, uint64_t{rows} * 8),
      impl_->beforePLEConvolution, impl_->bf(Scratch::PLENormalizedConvolution, rows, kHyper),
      impl_->retainedCount, state.pleHistory, state.pleConvolution, diag, geometry);
  metal::CommandTiming timing;
  try {
    timing = impl_->backend.submitCommand(graph.dispatches());
    uint32_t status = 0; std::memcpy(&status, diag.contents(), sizeof(status));
    if (status) throw std::runtime_error("Flash prefix restore diagnostics failed: " + std::to_string(status));
  } catch (...) {
    state.poisoned = true; state.pendingVerification = false; impl_->finishPending(); throw;
  }
  state.length = impl_->pendingBegin + retained;
  state.pendingVerification = false;
  impl_->finishPending();
  lazyGuard.keep = true;
  return timing;
}

void FlashForward::abortVerify(FlashRequestState &request) noexcept {
  if (!impl_) return;
  std::lock_guard lock(impl_->mutex);
  if (request.impl_ && impl_->pendingIdentity.lock() == request.impl_->identity) {
    request.impl_->poisoned = true; request.impl_->pendingVerification = false; impl_->finishPending();
  }
}

} // namespace splash::flash
