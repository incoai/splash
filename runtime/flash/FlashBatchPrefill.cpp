#include "flash/FlashBatchPrefill.hpp"

#include "flash/FlashAffine.hpp"
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashExpertDenseCache.hpp"
#include "flash/FlashGDN.hpp"
#include "flash/FlashGDNFused.hpp"
#include "flash/FlashGDNStaged.hpp"
#include "flash/FlashGDNBatchILP.hpp"
#include "flash/FlashHC.hpp"
#include "flash/FlashHCFused.hpp"
#include "flash/FlashMoE.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "flash/FlashInt8ExpertStore.hpp"
#include "flash/FlashPLE.hpp"
#include "flash/FlashPLEFused.hpp"
#include "flash/FlashPLESSD.hpp"
#include "flash/FlashGreedyGPU.hpp"
#include "flash/FlashQSA.hpp"
#include "flash/FlashQSAFast.hpp"
#include "flash/FlashQSAMPP.hpp"
#include "flash/FlashRequestStateInternal.hpp"
#include "metal/abi/FlashForward.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace splash::flash {
namespace {
constexpr uint64_t kAlignment = 16384, kGuardBytes = kAlignment;
constexpr uint8_t kGuard = 0xa7;
constexpr uint32_t kWidth = 2560, kHyper = 10240, kSelections = 10;
constexpr uint64_t kPLEConvolutionBytes = uint64_t{9} * kHyper * 2;
bool enabled(const char *name) {
  const char *value = std::getenv(name);
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}
uint64_t rounded(uint64_t bytes) {
  if (!bytes || bytes > std::numeric_limits<uint64_t>::max() - (kAlignment - 1))
    throw std::overflow_error("Flash batch prefill allocation extent is invalid");
  return (bytes + kAlignment - 1) & ~(kAlignment - 1);
}
void geometry(uint32_t capacity, uint32_t lanes, uint32_t rows) {
  if (!capacity || capacity > 262144 || !lanes || lanes > 4 || !rows ||
      rows > kFlashBatchPrefillMaximumRowsPerLane ||
      uint64_t{lanes} * rows > kFlashBatchPrefillMaximumPhysicalRows)
    throw std::invalid_argument("Flash batch prefill requires capacity 1..262144, lanes 1..4, rows 1..2048, total <=8192");
}
enum class Slot : size_t {
  Tokens, Embedding, Hyper, Normalized, Up, Down, RawInjection, Injection,
  Mixed, Branch, Q, K, V, Index, GDNMixed, Z, A, B, Decay, Beta, RecurrentRows,
  Attention, Router, ExpertIDs, Routes, ExpertGate, ExpertUp, Intermediate,
  ExpertDown, SharedGate, SharedUp, SharedIntermediate, SharedDown, SharedGateLogit,
  Ngrams, PLEEmbedding, PLEKey, PLEValue, PLENormKeys, PLENormQueries, PLEGated,
  PLENormConvolution, PLEOutput, HeadInput, Logits, Diagnostics,
  PackedPLEHistory, PackedPLEConvolution, Count,
};
constexpr size_t kSlots = static_cast<size_t>(Slot::Count);
std::array<uint64_t, kSlots> sizes(uint32_t lanes, uint32_t rows) {
  std::array<uint64_t, kSlots> result{};
  const uint64_t flat = uint64_t{lanes} * rows;
  const auto put = [&](Slot slot, uint64_t bytes) { result[static_cast<size_t>(slot)] = bytes; };
  const auto bf = [&](Slot slot, uint64_t width) { put(slot, flat * width * 2); };
  put(Slot::Tokens, flat * 8);
  bf(Slot::Embedding, kWidth); bf(Slot::Hyper, kHyper); bf(Slot::Normalized, kHyper);
  bf(Slot::Up, kHyper); bf(Slot::Down, 320); bf(Slot::RawInjection, 4); bf(Slot::Injection, 4);
  bf(Slot::Mixed, kWidth); bf(Slot::Branch, kWidth); bf(Slot::Q, 12288);
  bf(Slot::K, 512); bf(Slot::V, 512); bf(Slot::Index, 640);
  bf(Slot::GDNMixed, 10240); bf(Slot::Z, 6144); bf(Slot::A, 48); bf(Slot::B, 48);
  put(Slot::Decay, flat * 48 * 4); bf(Slot::Beta, 48);
  bf(Slot::RecurrentRows, 6144); bf(Slot::Attention, 6144); bf(Slot::Router, 512);
  put(Slot::ExpertIDs, flat * kSelections * 8); bf(Slot::Routes, kSelections);
  bf(Slot::ExpertGate, kSelections * 640); bf(Slot::ExpertUp, kSelections * 640);
  bf(Slot::Intermediate, kSelections * 640); bf(Slot::ExpertDown, kSelections * kWidth);
  bf(Slot::SharedGate, 640); bf(Slot::SharedUp, 640); bf(Slot::SharedIntermediate, 640);
  bf(Slot::SharedDown, kWidth); bf(Slot::SharedGateLogit, 1);
  put(Slot::Ngrams, flat * 16 * 8); bf(Slot::PLEEmbedding, kWidth);
  bf(Slot::PLEKey, kHyper); bf(Slot::PLEValue, kWidth); bf(Slot::PLENormKeys, kHyper);
  bf(Slot::PLENormQueries, kHyper); bf(Slot::PLEGated, kHyper);
  bf(Slot::PLENormConvolution, kHyper); bf(Slot::PLEOutput, kHyper);
  put(Slot::HeadInput, uint64_t{lanes} * kWidth * 2);
  put(Slot::Logits, uint64_t{lanes} * 248320 * 2); put(Slot::Diagnostics, 4);
  put(Slot::PackedPLEHistory, uint64_t{lanes} * 16);
  put(Slot::PackedPLEConvolution, uint64_t{lanes} * kPLEConvolutionBytes);
  return result;
}
uint64_t blockedPlanned(uint32_t rows) {
  return flashMoEBlockedWorkspacePlannedBytes(rows, kSelections, kAlignment);
}
} // namespace

struct FlashBatchPrefill::Impl final {
  metal::MetalBackend &backend;
  const FlashWeights &weights;
  FlashForward &trunk;
  const void *trunkIdentity;
  const std::string weightIdentity, sourceRoutes;
  const FlashDescriptor descriptor;
  const uint32_t capacity, maximumLanes, maximumRows;
  const bool fuseHC = enabled("SPLASH_FLASH_FUSE_HC");
  const bool fuseGDN = enabled("SPLASH_FLASH_FUSE_GDN");
  const bool stagedGDN = enabled("SPLASH_FLASH_GDN_STAGED");
  const bool batchGDNILP = flashGDNBatchILPEnabled();
  const bool cachedDense = enabled("SPLASH_FLASH_DENSE_CACHE");
  const bool blockMoE = enabled("SPLASH_FLASH_BLOCKED_MOE");
  const bool onlineQSA = enabled("SPLASH_FLASH_QSA_F32");
  const bool mppQSA = enabled("SPLASH_FLASH_QSA_MPP");
  const bool gpuGreedy = enabled("SPLASH_FLASH_GPU_GREEDY");
  FlashGreedyGPUWorkspace greedyWorkspace;
  metal::MetalBuffer greedyResults;
  std::array<metal::MetalBuffer, kSlots> scratch, guardedBases;
  const std::array<uint64_t, kSlots> extents;
  FlashPLEWeights ple;
  std::unique_ptr<FlashPLESSD> pleSSD;
  FlashQSAWorkspace qsa;
  FlashQSAFastWorkspace qsaFast;
  FlashMoEBlockedScratch blocked;
  uint64_t allocatedBytes = 0;
  std::mutex mutex;

  Impl(metal::MetalBackend &value, const FlashWeights &model, FlashForward &source,
       uint32_t context, uint32_t lanes, uint32_t rows, const void *identity)
      : backend(value), weights(model), trunk(source), trunkIdentity(identity),
        weightIdentity(model.manifestFingerprint()), sourceRoutes(source.kernelRoutes()),
        descriptor(model.descriptor()), capacity(context), maximumLanes(lanes), maximumRows(rows),
        extents(sizes(lanes, rows)), ple(FlashPLEWeights::fromWeights(model)) {
    descriptor.validate();
    if (!descriptor.pleParametersLoaded)
      throw std::invalid_argument("Flash batch prefill requires loaded PLE parameters");
    const auto route = [&](std::string_view tag) { return sourceRoutes.find(tag) != std::string::npos; };
    if (fuseHC != route(";hc-fused-literal-sg4") ||
        fuseGDN != route(";gdn-decode-persistent512-verify-capture512") ||
        stagedGDN != route(";gdn-prefill-staged-v16-t16") ||
        cachedDense != route(";dense-cache-bf16-whole-k:") ||
        blockMoE != !route(";moe-vector") ||
        onlineQSA != !route(";qsa-bf16-probabilities") ||
        (onlineQSA && mppQSA != route(kFlashQSAOnlineMPPRoute)) || (mppQSA && !onlineQSA))
      throw std::invalid_argument("Flash batch prefill route flags changed after source trunk construction");
    const uint64_t before = backend.memoryStats().allocatedBytes;
    if (weights.pleSSDStreamingEnabled())
      pleSSD = std::make_unique<FlashPLESSD>(backend, weights.pleSSDStore(),
          ple, maximumLanes, maximumRows);
    if (gpuGreedy) {
      greedyWorkspace = allocateGreedyGPUWorkspace(backend, maximumLanes, descriptor.vocabularySize);
      greedyResults = backend.allocateBuffer(uint64_t{maximumLanes} * sizeof(FlashGreedyGPURowResult),
          metal::BufferStorage::Shared, "flash-batch-prefill compact GPU greedy records");
    }
    for (size_t index = 0; index < kSlots; ++index) {
      const uint64_t bytes = rounded(extents[index]);
      guardedBases[index] = backend.allocateBuffer(bytes + 2 * kGuardBytes,
          metal::BufferStorage::Shared, "flash-batch-prefill-scratch-" + std::to_string(index));
      auto *base = static_cast<uint8_t *>(guardedBases[index].contents());
      if (!base) throw std::logic_error("Flash batch prefill scratch is not Shared");
      std::memset(base, kGuard, kGuardBytes);
      std::memset(base + kGuardBytes + bytes, kGuard, kGuardBytes);
      scratch[index] = backend.view(guardedBases[index], kGuardBytes, extents[index]);
    }
    const uint32_t qsaRows = std::min(maximumRows, uint32_t{128});
    qsa = allocateQSAWorkspace(backend, qsaRows, capacity);
    if (onlineQSA) qsaFast = mppQSA ? allocateQSAOnlineMPPWorkspace(backend, qsaRows, 32)
                                  : allocateQSAFastWorkspace(backend, qsaRows, 4);
    if (blockMoE && maximumLanes * maximumRows >= 256)
      blocked = allocateMoEBlockedScratch(backend, maximumLanes * maximumRows);
    allocatedBytes = metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
  }
  metal::MetalBuffer view(Slot slot, uint64_t bytes) {
    if (!bytes || bytes > extents[static_cast<size_t>(slot)])
      throw std::invalid_argument("Flash batch prefill scratch view exceeds its plane");
    return backend.view(scratch[static_cast<size_t>(slot)], 0, bytes);
  }
  metal::MetalBuffer bf(Slot slot, uint32_t rows, uint32_t width) {
    return view(slot, uint64_t{rows} * width * 2);
  }
  metal::MetalBuffer lane(const metal::MetalBuffer &buffer, uint32_t index, uint64_t bytes) {
    return backend.view(buffer, uint64_t{index} * bytes, bytes);
  }
  void copy(metal::CommandGraph &graph, const metal::MetalBuffer &input,
            const metal::MetalBuffer &output, uint64_t bytes) {
    if (!bytes || bytes % 4 || input.sizeBytes() < bytes || output.sizeBytes() < bytes || input.sameView(output))
      throw std::invalid_argument("Flash batch prefill copy extent or alias is invalid");
    const uint64_t words = bytes / 4;
    graph.add("flash_forward_copy_words", {input, output}, FlashForwardCopyParams{words},
        {(words + 255) / 256, 1, 1});
  }
  void project(metal::CommandGraph &graph, const std::string &prefix,
      const metal::MetalBuffer &input, const metal::MetalBuffer &output,
      const metal::MetalBuffer &diag, uint32_t rows) {
    trunk.batchProject(graph, prefix, input, output, diag, rows);
  }
  void hc(metal::CommandGraph &graph, const std::string &prefix, uint32_t rows,
      bool injection, bool normalizedReady = false) {
    const FlashHCGeometry shape{rows, kWidth, 4, static_cast<float>(descriptor.normEpsilon)};
    const auto normalized = bf(Slot::Normalized, rows, kHyper), down = bf(Slot::Down, rows, 320);
    const auto up = bf(Slot::Up, rows, kHyper), diag = view(Slot::Diagnostics, 4);
    const auto norm = prefix + ".hc_norm.weight";
    if (!normalizedReady)
      addHCGroupedNorm(graph, bf(Slot::Hyper, rows, kHyper), weights.tensor(norm), normalized,
          shape, weights.normConvention(norm));
    const auto &dp = weights.projection(prefix + ".input_mix_weight_down");
    const auto &upProjection = weights.projection(prefix + ".input_mix_weight_up");
    const auto *ip = injection ? &weights.projection(prefix + ".block_inject_weight") : nullptr;
    if (fuseHC && supportsHCFused(dp, upProjection, ip, shape)) {
      addHCFusedDown(graph, normalized, dp, ip, down,
          injection ? bf(Slot::Injection, rows, 4) : metal::MetalBuffer{}, diag, shape);
      addHCFusedUpMix(graph, normalized, down, upProjection, bf(Slot::Mixed, rows, kWidth), diag, shape);
      return;
    }
    project(graph, prefix + ".input_mix_weight_down", normalized, down, diag, rows);
    graph.add("flash_forward_hc_silu", {down, down, diag}, FlashForwardActivationParams{rows, 320, 4},
        {(uint64_t{rows} * 320 + 255) / 256, 1, 1});
    project(graph, prefix + ".input_mix_weight_up", down, up, diag, rows);
    if (injection) {
      const auto raw = bf(Slot::RawInjection, rows, 4);
      project(graph, prefix + ".block_inject_weight", normalized, raw, diag, rows);
      addHCMixWithInjection(graph, normalized, up, raw, bf(Slot::Mixed, rows, kWidth),
          bf(Slot::Injection, rows, 4), shape);
    } else addHCMix(graph, normalized, up, bf(Slot::Mixed, rows, kWidth), shape);
  }
};

FlashBatchPrefill::FlashBatchPrefill(metal::MetalBackend &backend, const FlashWeights &weights,
    FlashForward &trunk, uint32_t capacity, uint32_t maximumLanes, uint32_t maximumRows) {
  geometry(capacity, maximumLanes, maximumRows);
  if (&backend != &trunk.batchBackend() || &weights != &trunk.batchWeights() || capacity != trunk.batchCapacity())
    throw std::invalid_argument("Flash batch prefill backend, model and context must match its trunk");
  impl_ = std::make_unique<Impl>(backend, weights, trunk, capacity, maximumLanes, maximumRows, trunk.impl_.get());
}
FlashBatchPrefill::~FlashBatchPrefill() = default;
FlashBatchPrefill::FlashBatchPrefill(FlashBatchPrefill &&) noexcept = default;
FlashBatchPrefill &FlashBatchPrefill::operator=(FlashBatchPrefill &&) noexcept = default;
uint64_t FlashBatchPrefill::workspaceBytes() const noexcept { return impl_ ? impl_->allocatedBytes : 0; }
uint64_t FlashBatchPrefill::pleSSDStagingBytes() const noexcept {
  return impl_ && impl_->pleSSD ? impl_->pleSSD->allocatedBytes() : 0;
}
std::string FlashBatchPrefill::kernelRoutes() const {
  if (!impl_) throw std::logic_error("Flash batch prefill is not initialized");
  return std::string(kFlashBatchPrefillSemantics) + ":" + impl_->sourceRoutes +
      (impl_->batchGDNILP ? kFlashGDNBatchILPRoute : "");
}
uint64_t FlashBatchPrefill::workspacePlannedBytes(uint32_t capacity, uint32_t lanes, uint32_t rows) {
  geometry(capacity, lanes, rows);
  uint64_t total = 0;
  for (uint64_t extent : sizes(lanes, rows)) total += rounded(extent) + 2 * kGuardBytes;
  const uint64_t q = std::min(rows, uint32_t{128}), blocks = (uint64_t{capacity} + 3) / 4;
  for (uint64_t extent : std::array<uint64_t, 8>{q * 6144 * 2, q * 512 * 2,
      q * blocks * 4, q * 512 * 4, q * 24 * kFlashQSATokenWidth * 4,
      q * 24 * kFlashQSATokenWidth * 2, q * 24 * 32 * 2 * 4, q * 24 * 32 * 256 * 4})
    total += rounded(extent);
  if (lanes * rows >= 256) total += blockedPlanned(lanes * rows);
  if (enabled("SPLASH_FLASH_GPU_GREEDY"))
    total += greedyGPUWorkspacePlannedBytes(lanes, 248320);
  if (flashPLESSDStreamingValue(std::getenv("SPLASH_FLASH_PLE_SSD_STREAMING")))
    total += FlashPLESSD::plannedBytes(lanes, rows);
  return total;
}

FlashBatchPrefillResult FlashBatchPrefill::forwardBatch(std::span<FlashRequestState *const> requests,
    std::span<const uint32_t> tokens, uint32_t rows, bool captureHidden,
    metal::MetalBuffer hiddenDestination) {
  if (!impl_) throw std::logic_error("Flash batch prefill is not initialized");
  std::scoped_lock lock(impl_->mutex, impl_->trunk.batchMutex());
  if (impl_->trunk.impl_.get() != impl_->trunkIdentity ||
      &impl_->backend != &impl_->trunk.batchBackend() || &impl_->weights != &impl_->trunk.batchWeights() ||
      impl_->capacity != impl_->trunk.batchCapacity() ||
      impl_->weights.manifestFingerprint() != impl_->weightIdentity || impl_->trunk.kernelRoutes() != impl_->sourceRoutes)
    throw std::invalid_argument("Flash batch prefill source trunk or model was replaced");
  if (requests.empty() || requests.size() > impl_->maximumLanes || !rows || rows > impl_->maximumRows ||
      requests.size() * rows != tokens.size())
    throw std::invalid_argument("Flash batch prefill requires uniform real rows within its per-lane arena");
  const uint32_t lanes = static_cast<uint32_t>(requests.size()), flat = lanes * rows;
  std::array<FlashRequestState::Impl *, 4> states{};
  std::vector<uint64_t> lengths(lanes);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (!requests[lane] || !impl_->trunk.ownsState(*requests[lane]))
      throw std::invalid_argument("Flash batch prefill requires its trunk's healthy, nonpending request states");
    states[lane] = requests[lane]->impl_.get();
    if (states[lane]->capacity != impl_->capacity || states[lane]->length > impl_->capacity ||
        rows > impl_->capacity - states[lane]->length)
      throw std::invalid_argument("Flash batch prefill request exceeds context capacity");
    for (uint32_t previous = 0; previous < lane; ++previous)
      if (states[lane]->identity == states[previous]->identity)
        throw std::invalid_argument("Flash batch prefill includes the same request twice");
  }
  for (uint32_t token : tokens)
    if (token >= impl_->descriptor.vocabularySize)
      throw std::invalid_argument("Flash batch prefill contains an out-of-vocabulary token");
  if (hiddenDestination) {
    if (!captureHidden || hiddenDestination.storage() != metal::BufferStorage::Shared)
      throw std::invalid_argument("Flash batch prefill feature destination requires captured Shared features");
    const uint64_t bytes = flashBatchPrefillHiddenCopyBytes(lanes, rows);
    validateFlashBatchPrefillCopyRange(hiddenDestination.contents(), hiddenDestination.sizeBytes(), bytes);
    try {
      // Checks backend allocation identity on the CPU. Retain only the actual
      // written prefix; a larger caller-owned capacity tail remains untouched.
      hiddenDestination = impl_->backend.view(hiddenDestination, 0, bytes);
    } catch (const metal::MetalBackendError &) {
      throw std::invalid_argument("Flash batch prefill feature destination belongs to another backend");
    }
    impl_->trunk.batchValidateExternalDestination(hiddenDestination);
    const auto reject = [&](const metal::MetalBuffer &buffer) {
      if (flashBatchPrefillCopyRangesOverlap(hiddenDestination.contents(), bytes,
                                            buffer.contents(), buffer.sizeBytes()))
        throw std::invalid_argument("Flash batch prefill feature destination overlaps batch or request storage");
    };
    for (const auto &buffer : impl_->guardedBases) reject(buffer);
    if (impl_->pleSSD)
      for (const auto &buffer : impl_->pleSSD->scratchBuffers()) reject(buffer);
    for (const auto &buffer : {impl_->qsa.queries, impl_->qsa.indexQueries, impl_->qsa.blockScores,
        impl_->qsa.selectedBlocks, impl_->qsa.attentionScores, impl_->qsa.probabilities,
        impl_->qsaFast.partitionStatistics, impl_->qsaFast.partitionValues,
        impl_->greedyWorkspace.partials, impl_->greedyResults}) reject(buffer);
    const auto &blocked = impl_->blocked;
    for (const auto &buffer : {blocked.buckets.counts, blocked.buckets.offsets, blocked.buckets.routeMap,
        blocked.buckets.canonicalToPacked, blocked.buckets.packedInputs, blocked.buckets.jobOffsets,
        blocked.buckets.jobCount, blocked.buckets.tileJobs, blocked.packedActivated, blocked.scatteredDown})
      reject(buffer);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const auto &state = *states[lane];
      reject(state.pleHistory); reject(state.pleConvolution);
      for (uint32_t layer = 0; layer < impl_->descriptor.layers; ++layer) {
        reject(state.gdn[layer].convolution); reject(state.gdn[layer].recurrent);
        const auto &qsa = state.qsa[layer];
        for (const auto &buffer : {qsa.keys, qsa.values, qsa.rawIndexKeys,
                                  qsa.pooledKeys, qsa.indexPositions}) reject(buffer);
      }
    }
  }
  const auto ids = impl_->view(Slot::Tokens, uint64_t{flat} * 8);
  auto *hostTokens = static_cast<int64_t *>(ids.contents());
  for (uint32_t row = 0; row < flat; ++row) hostTokens[row] = tokens[row];
  const auto diag = impl_->view(Slot::Diagnostics, 4);
  std::memset(diag.contents(), 0, 4);
  const auto bf = [&](Slot slot, uint32_t width) { return impl_->bf(slot, flat, width); };
  const auto affine = [&](metal::CommandGraph &graph, const std::string &prefix,
      const metal::MetalBuffer &input, const metal::MetalBuffer &output) {
    impl_->project(graph, prefix, input, output, diag, flat);
  };
  const auto hyper = bf(Slot::Hyper, kHyper), mixed = bf(Slot::Mixed, kWidth);
  const auto branch = bf(Slot::Branch, kWidth), attentionOutput = bf(Slot::Attention, 6144);
  const FlashHCGeometry hcGeometry{flat, kWidth, 4, static_cast<float>(impl_->descriptor.normEpsilon)};
  const FlashPLEGeometry pleGeometry{lanes, rows, kWidth, 4, impl_->descriptor.pleHistoryEos,
      impl_->descriptor.vocabularySize, static_cast<float>(impl_->descriptor.normEpsilon)};
  if (impl_->pleSSD) {
    std::array<int64_t, 8> histories{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const auto *history = static_cast<const int64_t *>(states[lane]->pleHistory.contents());
      if (!history) throw std::logic_error("Flash PLE SSD batch history is not Shared");
      histories[lane * 2] = history[0]; histories[lane * 2 + 1] = history[1];
    }
    impl_->pleSSD->prepare({hostTokens, flat}, {histories.data(), lanes * 2}, impl_->ple, pleGeometry);
  }
  metal::CommandGraph graph;
  addAffineEmbedding(graph, impl_->weights.projection("language_model.model.embed_tokens"), ids,
      bf(Slot::Embedding, kWidth), diag, flat);
  addHCExpand(graph, bf(Slot::Embedding, kWidth), hyper, hcGeometry);
  bool normalizedReady = false;
  for (uint32_t layer = 0; layer < impl_->descriptor.layers; ++layer) {
    const std::string prefix = "language_model.model.layers." + std::to_string(layer);
    if (layer == impl_->descriptor.pleLayerIndices.front()) {
      const auto history = impl_->view(Slot::PackedPLEHistory, uint64_t{lanes} * 16);
      const auto convolution = impl_->view(Slot::PackedPLEConvolution, uint64_t{lanes} * kPLEConvolutionBytes);
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        impl_->copy(graph, states[lane]->pleHistory, impl_->lane(history, lane, 16), 16);
        impl_->copy(graph, states[lane]->pleConvolution,
            impl_->lane(convolution, lane, kPLEConvolutionBytes), kPLEConvolutionBytes);
      }
      const auto ngrams = impl_->view(Slot::Ngrams, uint64_t{flat} * 16 * 8);
      const auto *lookup = impl_->trunk.batchPLELookup();
      if (impl_->pleSSD) {
        impl_->pleSSD->addHashGather(graph, impl_->ple, ids, history, ngrams,
            bf(Slot::PLEEmbedding, kWidth), diag, pleGeometry);
      } else if (!lookup || !lookup->addHashGather(graph, ids, history, ngrams,
          bf(Slot::PLEEmbedding, kWidth), diag, pleGeometry)) {
        addPLENgramIDs(graph, impl_->ple, ids, history, ngrams, diag, pleGeometry);
        addPLEGather(graph, impl_->ple, ngrams, bf(Slot::PLEEmbedding, kWidth), diag, pleGeometry);
      }
      affine(graph, prefix + ".ple.key_proj", bf(Slot::PLEEmbedding, kWidth), bf(Slot::PLEKey, kHyper));
      affine(graph, prefix + ".ple.value_proj", bf(Slot::PLEEmbedding, kWidth), bf(Slot::PLEValue, kWidth));
      const FlashPLEPostScratch post{bf(Slot::PLENormKeys, kHyper), bf(Slot::PLENormQueries, kHyper),
          bf(Slot::PLEGated, kHyper), bf(Slot::PLENormConvolution, kHyper)};
      addPLEPostProjectAndInject(graph, impl_->ple, hyper, bf(Slot::PLEKey, kHyper), bf(Slot::PLEValue, kWidth),
          post, convolution, bf(Slot::PLEOutput, kHyper), diag, pleGeometry);
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        impl_->copy(graph, impl_->lane(history, lane, 16), states[lane]->pleHistory, 16);
        impl_->copy(graph, impl_->lane(convolution, lane, kPLEConvolutionBytes),
            states[lane]->pleConvolution, kPLEConvolutionBytes);
      }
      normalizedReady = false;
    }
    impl_->hc(graph, prefix + ".attn_hyper_connection", flat, true, normalizedReady);
    normalizedReady = false;
    if (impl_->descriptor.layerKinds[layer] == FlashLayerKind::GatedDeltaNet) {
      const auto attention = prefix + ".linear_attn";
      const auto qkv = bf(Slot::Q, 10240);
      affine(graph, attention + ".in_proj_qkv", mixed, qkv);
      affine(graph, attention + ".in_proj_z", mixed, bf(Slot::Z, 6144));
      affine(graph, attention + ".in_proj_a", mixed, bf(Slot::A, 48));
      affine(graph, attention + ".in_proj_b", mixed, bf(Slot::B, 48));
      const FlashGDNWeights weights{&impl_->weights.tensor(attention + ".conv1d.weight"),
          &impl_->weights.tensor(attention + ".A_log"), &impl_->weights.tensor(attention + ".dt_bias"),
          &impl_->weights.tensor(attention + ".norm.weight")};
      if (impl_->batchGDNILP && flashGDNBatchILPEligible(lanes, rows)) {
        std::array<FlashGDNBatchILPLane, 4> recurrenceLanes;
        for (uint32_t lane = 0; lane < lanes; ++lane)
          recurrenceLanes[lane] = {states[lane]->gdn[layer], rows};
        const FlashGDNBuffers buffers{qkv, bf(Slot::Z, 6144), bf(Slot::A, 48), bf(Slot::B, 48),
            bf(Slot::GDNMixed, 10240), impl_->view(Slot::Decay, uint64_t{flat} * 48 * 4),
            bf(Slot::Beta, 48), bf(Slot::RecurrentRows, 6144), attentionOutput, diag};
        addGDNBatchILP(impl_->backend, graph, weights, buffers,
            std::span<const FlashGDNBatchILPLane>(recurrenceLanes.data(), lanes), rows,
            FlashGDNBatchILPTile{}, static_cast<float>(impl_->descriptor.normEpsilon));
      } else for (uint32_t lane = 0; lane < lanes; ++lane) {
        const auto slice = [&](Slot slot, uint32_t width, uint32_t bytes = 2) {
          return impl_->lane(impl_->view(slot, uint64_t{flat} * width * bytes), lane,
              uint64_t{rows} * width * bytes);
        };
        const FlashGDNBuffers buffers{impl_->lane(qkv, lane, uint64_t{rows} * 10240 * 2),
            slice(Slot::Z, 6144), slice(Slot::A, 48), slice(Slot::B, 48), slice(Slot::GDNMixed, 10240),
            slice(Slot::Decay, 48, 4), slice(Slot::Beta, 48), slice(Slot::RecurrentRows, 6144),
            impl_->lane(attentionOutput, lane, uint64_t{rows} * 6144 * 2), diag};
        if (impl_->stagedGDN && rows >= 64)
          addGDNStagedPrefill(graph, weights, buffers, states[lane]->gdn[layer], rows, 1,
              FlashGDNStageTile::Values16Time16, static_cast<float>(impl_->descriptor.normEpsilon));
        else if (impl_->fuseGDN)
          addGDNFused(graph, weights, buffers, states[lane]->gdn[layer], rows, 1,
              rows == 1 ? FlashGDNFusion::PersistentHead512 : FlashGDNFusion::Prepare,
              static_cast<float>(impl_->descriptor.normEpsilon));
        else addGDN(graph, weights, buffers, states[lane]->gdn[layer], rows, 1,
            static_cast<float>(impl_->descriptor.normEpsilon));
      }
      affine(graph, attention + ".out_proj", attentionOutput, branch);
    } else {
      const auto attention = prefix + ".self_attn";
      const auto q = bf(Slot::Q, 12288), k = bf(Slot::K, 512), v = bf(Slot::V, 512);
      const auto index = bf(Slot::Index, 640);
      affine(graph, attention + ".q_proj", mixed, q); affine(graph, attention + ".k_proj", mixed, k);
      affine(graph, attention + ".v_proj", mixed, v);
      affine(graph, attention + ".indexer.index_qk_proj", mixed, index);
      const auto qNorm = attention + ".q_norm.weight", kNorm = attention + ".k_norm.weight";
      const auto iqNorm = attention + ".indexer.q_layernorm.weight", ikNorm = attention + ".indexer.k_layernorm.weight";
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        for (uint32_t offset = 0; offset < rows;) {
          const uint32_t count = std::min(rows - offset, impl_->qsa.maximumRows);
          const auto slice = [&](const metal::MetalBuffer &buffer, uint32_t width) {
            return impl_->backend.view(buffer, uint64_t{lane * rows + offset} * width * 2,
                uint64_t{count} * width * 2);
          };
          const uint32_t begin = static_cast<uint32_t>(states[lane]->length) + offset;
          if (impl_->onlineQSA) {
            const FlashQSAFastInputs inputs{slice(q, 12288), slice(k, 512), slice(v, 512), slice(index, 640),
                &impl_->weights.tensor(qNorm), &impl_->weights.tensor(kNorm),
                &impl_->weights.tensor(iqNorm), &impl_->weights.tensor(ikNorm),
                slice(attentionOutput, 6144), diag, {}, impl_->weights.normConvention(qNorm),
                impl_->weights.normConvention(kNorm), impl_->weights.normConvention(iqNorm),
                impl_->weights.normConvention(ikNorm), impl_->descriptor.normEpsilon, impl_->descriptor.rotaryTheta};
            const uint32_t partitions = impl_->mppQSA ? qsaOnlineMPPRoutePartitions(begin, count) : 0;
            if (partitions)
              addQSAOnlineMPP(graph, inputs, states[lane]->qsa[layer], impl_->qsa, impl_->qsaFast,
                  begin, count, partitions, true);
            else addQSAFast(graph, inputs, states[lane]->qsa[layer], impl_->qsa, impl_->qsaFast,
                begin, count, FlashQSAFastMode::PartitionedF32Probabilities, 4, true);
          } else addQSA(graph, slice(q, 12288), slice(k, 512), slice(v, 512), slice(index, 640),
              impl_->weights.tensor(qNorm), impl_->weights.tensor(kNorm), impl_->weights.tensor(iqNorm),
              impl_->weights.tensor(ikNorm), states[lane]->qsa[layer], impl_->qsa, slice(attentionOutput, 6144),
              diag, begin, count, impl_->weights.normConvention(qNorm), impl_->weights.normConvention(kNorm),
              impl_->weights.normConvention(iqNorm), impl_->weights.normConvention(ikNorm),
              impl_->descriptor.normEpsilon, impl_->descriptor.rotaryTheta);
          offset += count;
        }
      }
      affine(graph, attention + ".o_proj", attentionOutput, branch);
    }
    const auto mlpNorm = prefix + ".mlp_hyper_connection.hc_norm.weight";
    if (impl_->fuseHC && flat <= 32) {
      addHCFusedInjectNorm(graph, hyper, branch, bf(Slot::Injection, 4), impl_->weights.tensor(mlpNorm),
          hyper, bf(Slot::Normalized, kHyper), diag, hcGeometry, impl_->weights.normConvention(mlpNorm));
      normalizedReady = true;
    } else addHCInject(graph, hyper, branch, bf(Slot::Injection, 4), hyper, hcGeometry);
    impl_->hc(graph, prefix + ".mlp_hyper_connection", flat, true, normalizedReady);
    const auto mlp = prefix + ".mlp";
    const auto router = bf(Slot::Router, 512);
    const auto expertIDs = impl_->view(Slot::ExpertIDs, uint64_t{flat} * kSelections * 8);
    const auto routes = bf(Slot::Routes, kSelections);
    if (impl_->cachedDense && flat >= 16)
      addDenseBF16WholeK(impl_->backend, graph, mixed, impl_->weights.tensor(mlp + ".gate.weight"),
          router, diag, flat, FlashAffineMPPTile::M16N64);
    else addDenseBF16(graph, mixed, impl_->weights.tensor(mlp + ".gate.weight"), router, diag, flat);
    addRoute(graph, router, expertIDs, routes, diag, flat, 512, kSelections);
    const bool useBlocked = impl_->blockMoE && flat >= 256;
    if (useBlocked) {
      const auto tile = flashMoEBlockedTile(flat, impl_->trunk.batchExpertCache(layer) != nullptr);
      addMoEBlockedPack(graph, mixed, expertIDs, impl_->blocked, diag, flat, tile);
      if (const auto *store = impl_->trunk.batchInt8ExpertStore()) {
        store->addGateUp(graph, layer, impl_->blocked, diag, flat, tile);
        store->addDownScatter(graph, layer, impl_->blocked, diag, flat, tile);
      } else if (const auto *cache = impl_->trunk.batchExpertCache(layer)) {
        cache->addGateUp(graph, impl_->blocked, diag, flat, tile);
        cache->addDownScatter(graph, impl_->blocked, diag, flat, tile);
      } else {
        addMoEBlockedGateUp(graph, impl_->weights.projection(mlp + ".switch_mlp.gate_proj"),
            impl_->weights.projection(mlp + ".switch_mlp.up_proj"), impl_->blocked, diag, flat, tile);
        addMoEBlockedDownScatter(graph, impl_->weights.projection(mlp + ".switch_mlp.down_proj"),
            impl_->blocked, diag, flat, tile);
      }
    } else {
      addGatheredAffine(graph, mixed, impl_->weights.projection(mlp + ".switch_mlp.gate_proj"), expertIDs,
          bf(Slot::ExpertGate, kSelections * 640), diag, flat, kSelections);
      addGatheredAffine(graph, mixed, impl_->weights.projection(mlp + ".switch_mlp.up_proj"), expertIDs,
          bf(Slot::ExpertUp, kSelections * 640), diag, flat, kSelections);
      addSiLUMultiply(graph, bf(Slot::ExpertGate, kSelections * 640), bf(Slot::ExpertUp, kSelections * 640),
          bf(Slot::Intermediate, kSelections * 640), diag, flat, 640, kSelections);
      addGatheredAffine(graph, bf(Slot::Intermediate, kSelections * 640),
          impl_->weights.projection(mlp + ".switch_mlp.down_proj"), expertIDs,
          bf(Slot::ExpertDown, kSelections * kWidth), diag, flat, kSelections, true);
    }
    if (!impl_->trunk.batchSharedExpertFused(graph, mlp + ".shared_expert", mixed,
        bf(Slot::SharedIntermediate, 640), diag, flat,
        bf(Slot::SharedGate, 640), bf(Slot::SharedUp, 640))) {
      affine(graph, mlp + ".shared_expert.gate_proj", mixed, bf(Slot::SharedGate, 640));
      affine(graph, mlp + ".shared_expert.up_proj", mixed, bf(Slot::SharedUp, 640));
      addSiLUMultiply(graph, bf(Slot::SharedGate, 640), bf(Slot::SharedUp, 640),
          bf(Slot::SharedIntermediate, 640), diag, flat, 640);
    }
    affine(graph, mlp + ".shared_expert.down_proj", bf(Slot::SharedIntermediate, 640), bf(Slot::SharedDown, kWidth));
    affine(graph, mlp + ".shared_expert_gate", mixed, bf(Slot::SharedGateLogit, 1));
    addCombine(graph, useBlocked ? impl_->blocked.scatteredDown : bf(Slot::ExpertDown, kSelections * kWidth),
        expertIDs, routes, bf(Slot::SharedDown, kWidth), bf(Slot::SharedGateLogit, 1), branch, diag,
        flat, kWidth, 512, kSelections);
    const bool nextHasPLE = layer + 1 == impl_->descriptor.pleLayerIndices.front();
    normalizedReady = impl_->fuseHC && flat <= 32 && !nextHasPLE;
    if (normalizedReady) {
      const auto nextHC = layer + 1 == impl_->descriptor.layers ? "language_model.model.hyper_connection_mixer"
          : "language_model.model.layers." + std::to_string(layer + 1) + ".attn_hyper_connection";
      const auto nextNorm = nextHC + ".hc_norm.weight";
      addHCFusedInjectNorm(graph, hyper, branch, bf(Slot::Injection, 4), impl_->weights.tensor(nextNorm),
          hyper, bf(Slot::Normalized, kHyper), diag, hcGeometry, impl_->weights.normConvention(nextNorm));
    } else addHCInject(graph, hyper, branch, bf(Slot::Injection, 4), hyper, hcGeometry);
  }
  impl_->hc(graph, "language_model.model.hyper_connection_mixer", flat, false, normalizedReady);
  const auto headInput = impl_->bf(Slot::HeadInput, lanes, kWidth);
  for (uint32_t lane = 0; lane < lanes; ++lane)
    impl_->copy(graph, impl_->backend.view(mixed, uint64_t{lane * rows + rows - 1} * kWidth * 2,
        uint64_t{kWidth} * 2), impl_->lane(headInput, lane, uint64_t{kWidth} * 2), uint64_t{kWidth} * 2);
  const auto logits = impl_->bf(Slot::Logits, lanes, impl_->descriptor.vocabularySize);
  impl_->project(graph, "language_model.lm_head", headInput, logits, diag, lanes);
  metal::MetalBuffer compactGreedy;
  if (impl_->gpuGreedy) {
    compactGreedy = impl_->backend.view(impl_->greedyResults, 0,
        uint64_t{lanes} * sizeof(FlashGreedyGPURowResult));
    addGreedyGPU(graph, logits, impl_->greedyWorkspace, compactGreedy,
        lanes, impl_->descriptor.vocabularySize);
  }
  if (hiddenDestination)
    impl_->copy(graph, hyper, hiddenDestination, flashBatchPrefillHiddenCopyBytes(lanes, rows));
  metal::CommandTiming timing;
  try {
    timing = impl_->backend.submitCommand(graph.dispatches());
    uint32_t status = 0; std::memcpy(&status, diag.contents(), sizeof(status));
    if (status) throw std::runtime_error("Flash batch prefill sticky diagnostics failed: " + std::to_string(status));
  } catch (...) {
    for (uint32_t lane = 0; lane < lanes; ++lane) states[lane]->poisoned = true;
    throw;
  }
  for (uint32_t lane = 0; lane < lanes; ++lane) lengths[lane] = states[lane]->length += rows;
  return {timing, logits, hiddenDestination ? hiddenDestination : captureHidden ? hyper : metal::MetalBuffer{},
      std::move(lengths), lanes, rows, impl_->capacity,
      compactGreedy, compactGreedy ? lanes : 0, bool(hiddenDestination)};
}

std::vector<FlashBatchPrefillStatePlane> FlashBatchPrefill::inspectState(const FlashRequestState &request) const {
  if (!impl_) throw std::logic_error("Flash batch prefill is not initialized");
  std::scoped_lock lock(impl_->mutex, impl_->trunk.batchMutex());
  if (!impl_->trunk.ownsState(request)) throw std::invalid_argument("Flash batch prefill inspector requires its healthy state");
  const auto &state = *request.impl_;
  std::vector<FlashBatchPrefillStatePlane> result;
  for (uint32_t layer = 0; layer < impl_->descriptor.layers; ++layer) {
    const auto name = "layer." + std::to_string(layer) + ".";
    if (impl_->descriptor.layerKinds[layer] == FlashLayerKind::GatedDeltaNet) {
      result.push_back({name + "conv", state.gdn[layer].convolution, FlashDType::BF16, flashGDNConvolutionLaneBytes()});
      result.push_back({name + "recurrent", state.gdn[layer].recurrent, FlashDType::F32, flashGDNRecurrentLaneBytes()});
    } else {
      const auto &qsa = state.qsa[layer];
      result.push_back({name + "keys", qsa.keys, FlashDType::BF16, state.length * 512 * 2});
      result.push_back({name + "values", qsa.values, FlashDType::BF16, state.length * 512 * 2});
      result.push_back({name + "raw_index", qsa.rawIndexKeys, FlashDType::BF16, state.length * 128 * 2});
      result.push_back({name + "pooled", qsa.pooledKeys, FlashDType::BF16, (state.length / 4) * 128 * 2});
      result.push_back({name + "positions", qsa.indexPositions, FlashDType::I64, state.length * 8});
    }
  }
  result.push_back({"ple.history", state.pleHistory, FlashDType::I64, 16});
  result.push_back({"ple.conv", state.pleConvolution, FlashDType::BF16, kPLEConvolutionBytes});
  return result;
}
bool FlashBatchPrefill::canariesIntact() const noexcept {
  if (!impl_) return false;
  for (size_t index = 0; index < kSlots; ++index) {
    const auto *base = static_cast<const uint8_t *>(impl_->guardedBases[index].contents());
    if (!base) return false;
    const uint64_t tail = kGuardBytes + rounded(impl_->extents[index]);
    for (uint64_t byte = 0; byte < kGuardBytes; ++byte)
      if (base[byte] != kGuard || base[tail + byte] != kGuard) return false;
  }
  return true;
}
} // namespace splash::flash
