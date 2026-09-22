#include "flash/FlashBatchVerify.hpp"

#include "flash/FlashAffine.hpp"
#include "flash/FlashGDN.hpp"
#include "flash/FlashGDNFused.hpp"
#include "flash/FlashBatchVerifyGDN.hpp"
#include "flash/FlashGDNLazyRollback.hpp"
#include "flash/FlashHC.hpp"
#include "flash/FlashHCFused.hpp"
#include "flash/FlashMoE.hpp"
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
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace splash::flash {
namespace {
constexpr uint64_t kAlignment = 16384;
constexpr uint32_t kWidth = 2560, kHyper = 10240, kSelections = 10;
constexpr uint64_t kPLEConvolutionBytes = uint64_t{9} * kHyper * 2;

bool fusionEnabled(const char *name) {
  const char *value = std::getenv(name);
  if (!value) return false;
  if (std::string_view(value) == "1") return true;
  if (std::string_view(value) == "0") return false;
  throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}

uint64_t rounded(uint64_t bytes) {
  if (!bytes || bytes > std::numeric_limits<uint64_t>::max() - (kAlignment - 1))
    throw std::overflow_error("Flash batch allocation extent is invalid");
  return (bytes + kAlignment - 1) & ~(kAlignment - 1);
}
void validateGeometry(uint32_t capacity, uint32_t lanes, uint32_t rows) {
  if (!capacity || capacity > 262144 || !lanes || lanes > 4 || !rows || rows > 4)
    throw std::invalid_argument("Flash batch verify requires capacity1..262144, lanes1..4, rows1..4");
}

enum class Slot : size_t {
  Tokens, Embedding, Hyper, Normalized, Up, Down, RawInjection, Injection,
  Mixed, Branch, Q, K, V, Index, GDNMixed, Z, A, B, Decay, Beta, RecurrentRows,
  Attention, Router, ExpertIDs, Routes, ExpertGate, ExpertUp, Intermediate,
  ExpertDown, SharedGate, SharedUp, SharedIntermediate, SharedDown, SharedGateLogit,
  Ngrams, PLEEmbedding, PLEKey, PLEValue, PLENormKeys, PLENormQueries, PLEGated,
  PLENormConvolution, PLEOutput, Logits, Diagnostics, PackedGDNConvolution,
  PackedGDNRecurrent, PackedPLEHistory, PackedPLEConvolution, BeforePLEHistory,
  BeforePLEConvolution, RetainedCounts, Count,
};
constexpr size_t kSlots = static_cast<size_t>(Slot::Count);

std::array<uint64_t, kSlots> sizes(uint32_t lanes, uint32_t rows) {
  std::array<uint64_t, kSlots> result{};
  const auto put = [&](Slot slot, uint64_t bytes) { result[static_cast<size_t>(slot)] = bytes; };
  const uint64_t flattened = uint64_t{lanes} * rows;
  const auto bf = [&](Slot slot, uint64_t width) { put(slot, flattened * width * 2); };
  put(Slot::Tokens, flattened * 8);
  bf(Slot::Embedding, kWidth); bf(Slot::Hyper, kHyper); bf(Slot::Normalized, kHyper);
  bf(Slot::Up, kHyper); bf(Slot::Down, 320); bf(Slot::RawInjection, 4); bf(Slot::Injection, 4);
  bf(Slot::Mixed, kWidth); bf(Slot::Branch, kWidth); bf(Slot::Q, 12288);
  bf(Slot::K, 512); bf(Slot::V, 512); bf(Slot::Index, 640);
  bf(Slot::GDNMixed, 10240); bf(Slot::Z, 6144); bf(Slot::A, 48); bf(Slot::B, 48);
  put(Slot::Decay, flattened * 48 * 4); bf(Slot::Beta, 48);
  bf(Slot::RecurrentRows, 6144); bf(Slot::Attention, 6144); bf(Slot::Router, 512);
  put(Slot::ExpertIDs, flattened * kSelections * 8); bf(Slot::Routes, kSelections);
  bf(Slot::ExpertGate, kSelections * 640); bf(Slot::ExpertUp, kSelections * 640);
  bf(Slot::Intermediate, kSelections * 640); bf(Slot::ExpertDown, kSelections * kWidth);
  bf(Slot::SharedGate, 640); bf(Slot::SharedUp, 640); bf(Slot::SharedIntermediate, 640);
  bf(Slot::SharedDown, kWidth); bf(Slot::SharedGateLogit, 1);
  put(Slot::Ngrams, flattened * 16 * 8); bf(Slot::PLEEmbedding, kWidth);
  bf(Slot::PLEKey, kHyper); bf(Slot::PLEValue, kWidth); bf(Slot::PLENormKeys, kHyper);
  bf(Slot::PLENormQueries, kHyper); bf(Slot::PLEGated, kHyper);
  bf(Slot::PLENormConvolution, kHyper); bf(Slot::PLEOutput, kHyper);
  bf(Slot::Logits, 248320); put(Slot::Diagnostics, 4);
  put(Slot::PackedGDNConvolution, uint64_t{lanes} * rounded(flashGDNConvolutionLaneBytes()));
  put(Slot::PackedGDNRecurrent, uint64_t{lanes} * flashGDNRecurrentLaneBytes());
  put(Slot::PackedPLEHistory, uint64_t{lanes} * 16);
  put(Slot::PackedPLEConvolution, uint64_t{lanes} * kPLEConvolutionBytes);
  put(Slot::BeforePLEHistory, uint64_t{lanes} * 16);
  put(Slot::BeforePLEConvolution, uint64_t{lanes} * kPLEConvolutionBytes);
  put(Slot::RetainedCounts, uint64_t{lanes} * 4);
  return result;
}

} // namespace

struct FlashBatchVerify::Impl final {
  metal::MetalBackend &backend;
  const FlashWeights &weights;
  FlashForward &trunk;
  const void *trunkIdentity;
  const std::string weightIdentity;
  const FlashDescriptor descriptor;
  const uint32_t capacity, maximumLanes, maximumRows;
  const bool sharedDenseRoutes;
  std::array<metal::MetalBuffer, kSlots> scratch;
  FlashPLEWeights ple;
  std::unique_ptr<FlashPLESSD> pleSSD;
  FlashQSAWorkspace qsa;
  FlashQSAFastWorkspace qsaFast;
  const bool fuseHC = fusionEnabled("SPLASH_FLASH_FUSE_HC");
  const bool mppQSA = fusionEnabled("SPLASH_FLASH_QSA_MPP");
  const bool gpuGreedy = fusionEnabled("SPLASH_FLASH_GPU_GREEDY");
  FlashGreedyGPUWorkspace greedyWorkspace;
  metal::MetalBuffer greedyResults;
  bool onlineQSA = false;
  uint64_t allocatedBytes = 0;
  std::mutex mutex;
  metal::MetalBuffer convolutionTape, recurrentTape;
  std::array<uint32_t, 48> gdnSlots{};
  const bool lazyGDN = flashGDNLazyRollbackEnabled();
  std::array<std::unique_ptr<FlashGDNLazyRollback>, 48> lazyGDNLayers;
  std::array<uint64_t, 48> lazyGDNTickets{};
  struct PendingLane {
    std::weak_ptr<const uint8_t> identity;
    FlashRequestState::Impl *state = nullptr;
    uint64_t begin = 0;
  };
  std::vector<PendingLane> pendingLanes;
  uint32_t pendingRows = 0;

  Impl(metal::MetalBackend &value, const FlashWeights &model, FlashForward &source,
       uint32_t context, uint32_t lanes, uint32_t rows, const void *sourceIdentity,
       bool sharedRoutes)
      : backend(value), weights(model), trunk(source), trunkIdentity(sourceIdentity),
        weightIdentity(model.manifestFingerprint()), descriptor(model.descriptor()),
        capacity(context), maximumLanes(lanes), maximumRows(rows), sharedDenseRoutes(sharedRoutes),
        ple(FlashPLEWeights::fromWeights(model)) {
    if (const char *value = std::getenv("SPLASH_FLASH_QSA_F32")) {
      if (std::string_view(value) != "0" && std::string_view(value) != "1")
        throw std::invalid_argument("SPLASH_FLASH_QSA_F32 must be 0 or 1");
      onlineQSA = std::string_view(value) == "1";
    }
    if (mppQSA && !onlineQSA)
      throw std::invalid_argument("SPLASH_FLASH_QSA_MPP requires SPLASH_FLASH_QSA_F32=1");
    const auto sourceRoutes = trunk.kernelRoutes();
    if (onlineQSA != (sourceRoutes.find(";qsa-bf16-probabilities") == std::string::npos) ||
        (onlineQSA && mppQSA != (sourceRoutes.find(kFlashQSAOnlineMPPRoute) != std::string::npos)))
      throw std::invalid_argument("Flash batch verify QSA route flags changed after source trunk construction");
    if (lazyGDN != (sourceRoutes.find(kFlashGDNLazyRollbackRoute) != std::string::npos))
      throw std::invalid_argument("Flash batch verify lazy GDN flag changed after source trunk construction");
    validateGeometry(capacity, maximumLanes, maximumRows);
    descriptor.validate();
    if (!descriptor.pleParametersLoaded) throw std::invalid_argument("Flash batch requires loaded PLE parameters");
    const uint64_t before = backend.memoryStats().allocatedBytes;
    if (weights.pleSSDStreamingEnabled())
      pleSSD = std::make_unique<FlashPLESSD>(backend, weights.pleSSDStore(),
          ple, maximumLanes, maximumRows);
    if (gpuGreedy) {
      const auto flattened = maximumLanes * maximumRows;
      greedyWorkspace = allocateGreedyGPUWorkspace(backend, flattened, descriptor.vocabularySize);
      greedyResults = backend.allocateBuffer(uint64_t{flattened} * sizeof(FlashGreedyGPURowResult),
          metal::BufferStorage::Shared, "flash-batch-verify compact GPU greedy records");
    }
    const auto extents = sizes(maximumLanes, maximumRows);
    for (size_t index = 0; index < extents.size(); ++index)
      scratch[index] = backend.allocateBuffer(rounded(extents[index]), metal::BufferStorage::Shared,
          "flash-batch-verify-scratch-" + std::to_string(index));
    qsa = allocateQSAWorkspace(backend, maximumRows, capacity);
    if (onlineQSA) qsaFast = mppQSA ? allocateQSAOnlineMPPWorkspace(backend, maximumRows, 32)
                                  : allocateQSAFastWorkspace(backend, maximumRows, 4);
    uint32_t slot = 0;
    for (uint32_t layer = 0; layer < descriptor.layers; ++layer) {
      if (descriptor.layerKinds[layer] == FlashLayerKind::GatedDeltaNet) gdnSlots[layer] = slot++;
    }
    if (slot != 36) throw std::logic_error("Flash batch verify GDN layer count differs");
    if (lazyGDN) {
      for (uint32_t layer = 0; layer < descriptor.layers; ++layer)
        if (descriptor.layerKinds[layer] == FlashLayerKind::GatedDeltaNet)
          lazyGDNLayers[layer] = std::make_unique<FlashGDNLazyRollback>(
              backend, maximumRows, maximumLanes);
    } else if (maximumRows > 1) {
      convolutionTape = backend.allocateBuffer(uint64_t{slot} * maximumLanes * (maximumRows - 1) *
          rounded(flashGDNConvolutionLaneBytes()), metal::BufferStorage::Shared, "flash-batch-verify-convolution-prefixes");
      recurrentTape = backend.allocateBuffer(uint64_t{slot} * maximumLanes * (maximumRows - 1) *
          rounded(flashGDNRecurrentLaneBytes()), metal::BufferStorage::Shared, "flash-batch-verify-recurrent-prefixes");
    }
    const uint64_t after = backend.memoryStats().allocatedBytes;
    if (after < before) throw std::logic_error("Flash batch allocation ledger regressed");
    allocatedBytes = after - before;
  }

  metal::MetalBuffer view(Slot slot, uint64_t bytes) {
    return backend.view(scratch[static_cast<size_t>(slot)], 0, bytes);
  }
  metal::MetalBuffer bf(Slot slot, uint32_t lanes, uint32_t width) {
    return view(slot, uint64_t{lanes} * width * 2);
  }
  metal::MetalBuffer lane(const metal::MetalBuffer &buffer, uint32_t row, uint64_t bytes,
                          uint64_t stride = 0) {
    return backend.view(buffer, uint64_t{row} * (stride ? stride : bytes), bytes);
  }
  void copy(metal::CommandGraph &graph, const metal::MetalBuffer &input,
            const metal::MetalBuffer &output, uint64_t bytes) {
    if (!bytes || bytes % 4 || input.sizeBytes() < bytes || output.sizeBytes() < bytes || input.sameView(output))
      throw std::invalid_argument("Flash batch raw state-copy extent/alias is invalid");
    const uint64_t words = bytes / 4;
    graph.add("flash_forward_copy_words", {input, output}, FlashForwardCopyParams{words},
                {(words + 255) / 256, 1, 1});
  }
  metal::MetalBuffer prefixLayer(bool recurrent, uint32_t layer) {
    if (maximumRows <= 1 || lazyGDN) return {};
    const uint64_t rowStride = rounded(recurrent ? flashGDNRecurrentLaneBytes() : flashGDNConvolutionLaneBytes());
    const uint64_t bytes = uint64_t{maximumLanes} * (maximumRows - 1) * rowStride;
    return backend.view(recurrent ? recurrentTape : convolutionTape, uint64_t{gdnSlots[layer]} * bytes, bytes);
  }
  metal::MetalBuffer prefixLane(bool recurrent, uint32_t layer, uint32_t laneIndex, uint32_t prefixIndex) {
    const uint64_t rowStride = rounded(recurrent ? flashGDNRecurrentLaneBytes() : flashGDNConvolutionLaneBytes());
    const uint64_t bytes = recurrent ? flashGDNRecurrentLaneBytes() : flashGDNConvolutionLaneBytes();
    const uint64_t offset = (uint64_t{laneIndex} * (maximumRows - 1) + prefixIndex) * rowStride;
    return backend.view(prefixLayer(recurrent, layer), offset, bytes);
  }

  void abortLazyGDN() noexcept {
    for (uint32_t layer = 0; layer < descriptor.layers; ++layer) {
      if (!lazyGDNTickets[layer]) continue;
      try {
        if (lazyGDNLayers[layer] && lazyGDNLayers[layer]->pending())
          lazyGDNLayers[layer]->abort(lazyGDNTickets[layer]);
      } catch (...) {
        // Terminal cleanup cannot recover or promote request state.
      }
      lazyGDNTickets[layer] = 0;
    }
  }

  void hc(metal::CommandGraph &graph, const std::string &prefix, uint32_t rows,
          bool injection, bool normalizedReady = false) {
    const FlashHCGeometry geometry{rows, kWidth, 4, static_cast<float>(descriptor.normEpsilon)};
    const auto normed = bf(Slot::Normalized, rows, kHyper), down = bf(Slot::Down, rows, 320);
    const auto up = bf(Slot::Up, rows, kHyper), diag = view(Slot::Diagnostics, 4);
    const std::string norm = prefix + ".hc_norm.weight";
    if (!normalizedReady)
      addHCGroupedNorm(graph, bf(Slot::Hyper, rows, kHyper), weights.tensor(norm), normed,
          geometry, weights.normConvention(norm));
    const auto &downProjection = weights.projection(prefix + ".input_mix_weight_down");
    const auto &upProjection = weights.projection(prefix + ".input_mix_weight_up");
    const auto *injectionProjection = injection
        ? &weights.projection(prefix + ".block_inject_weight") : nullptr;
    if (fuseHC && supportsHCFused(downProjection, upProjection, injectionProjection, geometry)) {
      addHCFusedDown(graph, normed, downProjection, injectionProjection, down,
          injection ? bf(Slot::Injection, rows, 4) : metal::MetalBuffer{}, diag, geometry);
      if (!trunk.batchHCFusedUpMixF32(graph, prefix + ".input_mix_weight_up", normed, down,
          bf(Slot::Mixed, rows, kWidth), diag, rows))
        addHCFusedUpMix(graph, normed, down, upProjection, bf(Slot::Mixed, rows, kWidth),
            diag, geometry);
      return;
    }
    project(graph, prefix + ".input_mix_weight_down", normed, down, diag, rows);
    graph.add("flash_forward_hc_silu", {down, down, diag}, FlashForwardActivationParams{rows, 320, 4},
        {(uint64_t{rows} * 320 + 255) / 256, 1, 1});
    project(graph, prefix + ".input_mix_weight_up", down, up, diag, rows);
    if (injection) {
      const auto raw = bf(Slot::RawInjection, rows, 4);
      project(graph, prefix + ".block_inject_weight", normed, raw, diag, rows);
      addHCMixWithInjection(graph, normed, up, raw, bf(Slot::Mixed, rows, kWidth),
          bf(Slot::Injection, rows, 4), geometry);
    } else addHCMix(graph, normed, up, bf(Slot::Mixed, rows, kWidth), geometry);
  }
  void project(metal::CommandGraph &graph, const std::string &prefix,
      const metal::MetalBuffer &input, const metal::MetalBuffer &output,
      const metal::MetalBuffer &diag, uint32_t rows) {
    if (sharedDenseRoutes || (prefix == "language_model.lm_head" && trunk.batchInt8HeadEnabled()))
      trunk.batchProject(graph, prefix, input, output, diag, rows);
    else addAffine(graph, input, weights.projection(prefix), output, diag, rows);
  }
};

FlashBatchVerify::FlashBatchVerify(metal::MetalBackend &backend, const FlashWeights &weights,
                                     FlashForward &trunk, uint32_t capacity, uint32_t maximumLanes,
                                     uint32_t maximumRows, bool sharedDenseRoutes) {
  validateGeometry(capacity, maximumLanes, maximumRows);
  if (&backend != &trunk.batchBackend() || &weights != &trunk.batchWeights() || capacity != trunk.batchCapacity())
    throw std::invalid_argument("Flash batch backend/model/context must match its source trunk");
  if (sharedDenseRoutes && !trunk.batchDenseSmallRowsEnabled())
    throw std::invalid_argument("shared verifier dense routes require coherent all-small-row target policy");
  impl_ = std::make_unique<Impl>(backend, weights, trunk, capacity, maximumLanes, maximumRows,
                               trunk.impl_.get(), sharedDenseRoutes);
}
FlashBatchVerify::~FlashBatchVerify() { abortBatch(); }
FlashBatchVerify::FlashBatchVerify(FlashBatchVerify &&) noexcept = default;
FlashBatchVerify &FlashBatchVerify::operator=(FlashBatchVerify &&other) noexcept {
  if (this != &other) { abortBatch(); impl_ = std::move(other.impl_); }
  return *this;
}
uint64_t FlashBatchVerify::workspaceBytes() const noexcept { return impl_ ? impl_->allocatedBytes : 0; }
uint64_t FlashBatchVerify::pleSSDStagingBytes() const noexcept {
  return impl_ && impl_->pleSSD ? impl_->pleSSD->allocatedBytes() : 0;
}
uint64_t FlashBatchVerify::verificationGDNStorageBytes() const noexcept {
  if (!impl_) return 0;
  if (!impl_->lazyGDN)
    return impl_->convolutionTape.sizeBytes() + impl_->recurrentTape.sizeBytes();
  uint64_t total = 0;
  for (const auto &layer : impl_->lazyGDNLayers)
    if (layer) total += layer->allocationBytes();
  return total;
}
bool FlashBatchVerify::lazyGDNRollbackEnabled() const noexcept {
  return impl_ && impl_->lazyGDN;
}
FlashGDNLazyRollbackCounters FlashBatchVerify::lazyGDNRollbackCounters() const noexcept {
  FlashGDNLazyRollbackCounters result;
  if (impl_) for (const auto &record : impl_->lazyGDNLayers)
    if (record) result.add(record->counters());
  return result;
}

uint64_t FlashBatchVerify::workspacePlannedBytes(uint32_t capacity, uint32_t maximumLanes,
                                               uint32_t maximumRows) {
  validateGeometry(capacity, maximumLanes, maximumRows);
  uint64_t total = 0;
  for (uint64_t extent : sizes(maximumLanes, maximumRows)) total += rounded(extent);
  const uint64_t blocks = (uint64_t{capacity} + 3) / 4;
  for (uint64_t extent : std::array<uint64_t, 6>{uint64_t{maximumRows} * 6144 * 2,
           uint64_t{maximumRows} * 512 * 2, uint64_t{maximumRows} * blocks * 4,
           uint64_t{maximumRows} * 512 * 4, uint64_t{maximumRows} * 24 * kFlashQSATokenWidth * 4,
           uint64_t{maximumRows} * 24 * kFlashQSATokenWidth * 2})
    total += rounded(extent);
  total += rounded(uint64_t{maximumRows} * 24 * 32 * 2 * 4);
  total += rounded(uint64_t{maximumRows} * 24 * 32 * 256 * 4);
  if (flashGDNLazyRollbackEnabled())
    total += uint64_t{36} * FlashGDNLazyRollback::plannedBytes(maximumRows, maximumLanes);
  else if (maximumRows > 1)
    total += uint64_t{36} * maximumLanes * (maximumRows - 1) *
        (rounded(flashGDNConvolutionLaneBytes()) + rounded(flashGDNRecurrentLaneBytes()));
  if (fusionEnabled("SPLASH_FLASH_GPU_GREEDY"))
    total += greedyGPUWorkspacePlannedBytes(maximumLanes * maximumRows, 248320);
  if (flashPLESSDStreamingValue(std::getenv("SPLASH_FLASH_PLE_SSD_STREAMING")))
    total += FlashPLESSD::plannedBytes(maximumLanes, maximumRows);
  return total;
}

FlashBatchVerifyResult FlashBatchVerify::verifyBatch(std::span<FlashRequestState *const> requests,
                                                 std::span<const uint32_t> tokens, uint32_t rows) {
  if (!impl_) throw std::logic_error("Flash batch is not initialized");
  std::scoped_lock lock(impl_->mutex, impl_->trunk.batchMutex());
  if (impl_->trunk.impl_.get() != impl_->trunkIdentity ||
      &impl_->backend != &impl_->trunk.batchBackend() ||
      &impl_->weights != &impl_->trunk.batchWeights() || impl_->capacity != impl_->trunk.batchCapacity() ||
      impl_->weights.manifestFingerprint() != impl_->weightIdentity)
    throw std::invalid_argument("Flash batch source trunk/model was replaced after construction");
  if (impl_->pendingRows)
    throw std::logic_error("resolve the previous Flash batch verification before a new trial");
  if (requests.empty() || requests.size() > impl_->maximumLanes || !rows ||
      rows > impl_->maximumRows || requests.size() * rows != tokens.size())
    throw std::invalid_argument("Flash batch verify requires uniform real rows1..4 per lane");
  const uint32_t lanes = static_cast<uint32_t>(requests.size());
  std::array<FlashRequestState::Impl *, 4> states{};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (!requests[lane] || !impl_->trunk.ownsState(*requests[lane]))
      throw std::invalid_argument("Flash batch contains a foreign/pending/unhealthy state or invalid token");
    states[lane] = requests[lane]->impl_.get();
    if (states[lane]->length > impl_->capacity || rows > impl_->capacity - states[lane]->length)
      throw std::invalid_argument("Flash batch request has no remaining context capacity");
    for (uint32_t prior = 0; prior < lane; ++prior)
      if (states[lane]->identity == states[prior]->identity)
        throw std::invalid_argument("Flash batch includes the same request more than once");
  }
  for (uint32_t token : tokens)
    if (token >= impl_->descriptor.vocabularySize)
      throw std::invalid_argument("Flash batch verify contains an out-of-vocabulary token");
  const uint32_t flattened = lanes * rows;
  std::vector<uint64_t> lengths(lanes);
  std::vector<Impl::PendingLane> pendingLanes;
  pendingLanes.reserve(lanes);
  for (uint32_t lane = 0; lane < lanes; ++lane)
    pendingLanes.push_back({states[lane]->identity, states[lane], states[lane]->length});
  struct LazyTrialCleanup final {
    Impl &owner;
    bool keepPending = false;
    ~LazyTrialCleanup() { if (!keepPending) owner.abortLazyGDN(); }
  } lazyCleanup{*impl_};
  const auto ids = impl_->view(Slot::Tokens, uint64_t{flattened} * 8);
  auto *hostTokens = static_cast<int64_t *>(ids.contents());
  if (!hostTokens) throw std::logic_error("Flash batch token input is not shared");
  for (uint32_t row = 0; row < flattened; ++row) hostTokens[row] = tokens[row];
  const auto diag = impl_->view(Slot::Diagnostics, 4);
  std::memset(diag.contents(), 0, 4);
  const auto bf = [&](Slot slot, uint32_t width) { return impl_->bf(slot, flattened, width); };
  const auto affine = [&](metal::CommandGraph &graph, const std::string &prefix,
                          const metal::MetalBuffer &input, const metal::MetalBuffer &output) {
    impl_->project(graph, prefix, input, output, diag, flattened);
  };
  const auto hyper = bf(Slot::Hyper, kHyper), mixed = bf(Slot::Mixed, kWidth);
  const auto branch = bf(Slot::Branch, kWidth), attentionOutput = bf(Slot::Attention, 6144);
  const FlashHCGeometry hcGeometry{flattened, kWidth, 4, static_cast<float>(impl_->descriptor.normEpsilon)};
  const FlashPLEGeometry pleGeometry{lanes, rows, kWidth, 4, impl_->descriptor.pleHistoryEos,
      impl_->descriptor.vocabularySize, static_cast<float>(impl_->descriptor.normEpsilon)};
  if (impl_->pleSSD) {
    std::array<int64_t, 8> histories{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const auto *history = static_cast<const int64_t *>(states[lane]->pleHistory.contents());
      if (!history) throw std::logic_error("Flash PLE SSD batch history is not Shared");
      histories[lane * 2] = history[0]; histories[lane * 2 + 1] = history[1];
    }
    impl_->pleSSD->prepare({hostTokens, flattened}, {histories.data(), lanes * 2},
        impl_->ple, pleGeometry);
  }
  metal::CommandGraph graph;
  addAffineEmbedding(graph, impl_->weights.projection("language_model.model.embed_tokens"), ids,
      bf(Slot::Embedding, kWidth), diag, flattened);
  addHCExpand(graph, bf(Slot::Embedding, kWidth), hyper, hcGeometry);
  bool normalizedReady = false;
  for (uint32_t layer = 0; layer < impl_->descriptor.layers; ++layer) {
    const std::string prefix = "language_model.model.layers." + std::to_string(layer);
    if (layer == impl_->descriptor.pleLayerIndices.front()) {
      const auto history = impl_->view(Slot::PackedPLEHistory, uint64_t{lanes} * 16);
      const auto convolution = impl_->view(Slot::PackedPLEConvolution, uint64_t{lanes} * kPLEConvolutionBytes);
      const auto beforeHistory = impl_->view(Slot::BeforePLEHistory, uint64_t{lanes} * 16);
      const auto beforeConvolution = impl_->view(Slot::BeforePLEConvolution, uint64_t{lanes} * kPLEConvolutionBytes);
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        impl_->copy(graph, states[lane]->pleHistory, impl_->lane(history, lane, 16), 16);
        impl_->copy(graph, states[lane]->pleHistory, impl_->lane(beforeHistory, lane, 16), 16);
        impl_->copy(graph, states[lane]->pleConvolution,
            impl_->lane(convolution, lane, kPLEConvolutionBytes), kPLEConvolutionBytes);
        impl_->copy(graph, states[lane]->pleConvolution,
            impl_->lane(beforeConvolution, lane, kPLEConvolutionBytes), kPLEConvolutionBytes);
      }
      const auto ngrams = impl_->view(Slot::Ngrams, uint64_t{flattened} * 16 * 8);
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
    impl_->hc(graph, prefix + ".attn_hyper_connection", flattened, true, normalizedReady);
    normalizedReady = false;
    if (impl_->descriptor.layerKinds[layer] == FlashLayerKind::GatedDeltaNet) {
      const auto attention = prefix + ".linear_attn";
      const auto qkv = bf(Slot::Q, 10240);
      affine(graph, attention + ".in_proj_qkv", mixed, qkv);
      affine(graph, attention + ".in_proj_z", mixed, bf(Slot::Z, 6144));
      affine(graph, attention + ".in_proj_a", mixed, bf(Slot::A, 48));
      affine(graph, attention + ".in_proj_b", mixed, bf(Slot::B, 48));
      const uint64_t convStride = rounded(flashGDNConvolutionLaneBytes());
      const auto packedConv = impl_->view(Slot::PackedGDNConvolution, uint64_t{lanes} * convStride);
      const auto packedRec = impl_->view(Slot::PackedGDNRecurrent,
          uint64_t{lanes} * flashGDNRecurrentLaneBytes());
      const FlashGDNWeights weights{&impl_->weights.tensor(attention + ".conv1d.weight"),
          &impl_->weights.tensor(attention + ".A_log"), &impl_->weights.tensor(attention + ".dt_bias"),
          &impl_->weights.tensor(attention + ".norm.weight")};
      const FlashGDNBuffers buffers{qkv, bf(Slot::Z, 6144), bf(Slot::A, 48), bf(Slot::B, 48),
          bf(Slot::GDNMixed, 10240), impl_->view(Slot::Decay, uint64_t{flattened} * 48 * 4),
          bf(Slot::Beta, 48), bf(Slot::RecurrentRows, 6144), attentionOutput, diag};
      const FlashGDNState packed{packedConv, packedRec, convStride, flashGDNRecurrentLaneBytes()};
      std::array<FlashGDNState, 4> requestStates;
      for (uint32_t lane = 0; lane < lanes; ++lane) requestStates[lane] = states[lane]->gdn[layer];
      addBatchVerifyGDN(graph, impl_->backend, weights, buffers,
          std::span<const FlashGDNState>(requestStates.data(), lanes), packed,
          impl_->prefixLayer(false, layer), impl_->prefixLayer(true, layer), rows,
          impl_->maximumRows, static_cast<float>(impl_->descriptor.normEpsilon),
          impl_->lazyGDNLayers[layer].get(),
          impl_->lazyGDN ? &impl_->lazyGDNTickets[layer] : nullptr);
      affine(graph, attention + ".out_proj", attentionOutput, branch);
    } else {
      const auto attention = prefix + ".self_attn";
      const auto q = bf(Slot::Q, 12288), k = bf(Slot::K, 512), v = bf(Slot::V, 512);
      const auto index = bf(Slot::Index, 640);
      affine(graph, attention + ".q_proj", mixed, q); affine(graph, attention + ".k_proj", mixed, k);
      affine(graph, attention + ".v_proj", mixed, v);
      affine(graph, attention + ".indexer.index_qk_proj", mixed, index);
      const std::string qNorm = attention + ".q_norm.weight", kNorm = attention + ".k_norm.weight";
      const std::string iqNorm = attention + ".indexer.q_layernorm.weight", ikNorm = attention + ".indexer.k_layernorm.weight";
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        if (impl_->onlineQSA) {
          if (!impl_->qsaFast.maximumRows)
            throw std::logic_error("Flash batch online workspace was not allocated");
          const FlashQSAFastInputs inputs{impl_->lane(q, lane, uint64_t{rows} * 12288 * 2),
              impl_->lane(k, lane, uint64_t{rows} * 512 * 2), impl_->lane(v, lane, uint64_t{rows} * 512 * 2),
              impl_->lane(index, lane, uint64_t{rows} * 640 * 2), &impl_->weights.tensor(qNorm),
              &impl_->weights.tensor(kNorm), &impl_->weights.tensor(iqNorm), &impl_->weights.tensor(ikNorm),
              impl_->lane(attentionOutput, lane, uint64_t{rows} * 6144 * 2), diag, {},
              impl_->weights.normConvention(qNorm), impl_->weights.normConvention(kNorm),
              impl_->weights.normConvention(iqNorm), impl_->weights.normConvention(ikNorm),
              impl_->descriptor.normEpsilon, impl_->descriptor.rotaryTheta};
          const uint32_t begin = static_cast<uint32_t>(states[lane]->length);
          const uint32_t partitions = impl_->mppQSA ? qsaOnlineMPPRoutePartitions(begin, rows) : 0;
          if (partitions)
            addQSAOnlineMPP(graph, inputs, states[lane]->qsa[layer], impl_->qsa,
                impl_->qsaFast, begin, rows, partitions, true);
          else
            addQSAFast(graph, inputs, states[lane]->qsa[layer], impl_->qsa, impl_->qsaFast,
                begin, rows, FlashQSAFastMode::PartitionedF32Probabilities, 4, true);
        } else {
        addQSA(graph, impl_->lane(q, lane, uint64_t{rows} * 12288 * 2),
            impl_->lane(k, lane, uint64_t{rows} * 512 * 2),
            impl_->lane(v, lane, uint64_t{rows} * 512 * 2),
            impl_->lane(index, lane, uint64_t{rows} * 640 * 2),
            impl_->weights.tensor(qNorm), impl_->weights.tensor(kNorm), impl_->weights.tensor(iqNorm),
            impl_->weights.tensor(ikNorm), states[lane]->qsa[layer], impl_->qsa,
            impl_->lane(attentionOutput, lane, uint64_t{rows} * 6144 * 2), diag,
            static_cast<uint32_t>(states[lane]->length), rows,
            impl_->weights.normConvention(qNorm), impl_->weights.normConvention(kNorm),
            impl_->weights.normConvention(iqNorm), impl_->weights.normConvention(ikNorm),
            impl_->descriptor.normEpsilon, impl_->descriptor.rotaryTheta);
        }
      }
      affine(graph, attention + ".o_proj", attentionOutput, branch);
    }
    const std::string mlpNorm = prefix + ".mlp_hyper_connection.hc_norm.weight";
    if (impl_->fuseHC) {
      addHCFusedInjectNorm(graph, hyper, branch, bf(Slot::Injection, 4),
          impl_->weights.tensor(mlpNorm), hyper, bf(Slot::Normalized, kHyper), diag,
          hcGeometry, impl_->weights.normConvention(mlpNorm));
      normalizedReady = true;
    } else {
      addHCInject(graph, hyper, branch, bf(Slot::Injection, 4), hyper, hcGeometry);
    }
    impl_->hc(graph, prefix + ".mlp_hyper_connection", flattened, true, normalizedReady);
    const auto mlp = prefix + ".mlp";
    const auto expertIDs = impl_->view(Slot::ExpertIDs, uint64_t{flattened} * kSelections * 8);
    const auto routes = bf(Slot::Routes, kSelections);
    addDenseBF16(graph, mixed, impl_->weights.tensor(mlp + ".gate.weight"), bf(Slot::Router, 512), diag, flattened);
    addRoute(graph, bf(Slot::Router, 512), expertIDs, routes, diag, flattened, 512, kSelections);
    addGatheredAffine(graph, mixed, impl_->weights.projection(mlp + ".switch_mlp.gate_proj"),
        expertIDs, bf(Slot::ExpertGate, kSelections * 640), diag, flattened, kSelections);
    addGatheredAffine(graph, mixed, impl_->weights.projection(mlp + ".switch_mlp.up_proj"),
        expertIDs, bf(Slot::ExpertUp, kSelections * 640), diag, flattened, kSelections);
    addSiLUMultiply(graph, bf(Slot::ExpertGate, kSelections * 640), bf(Slot::ExpertUp, kSelections * 640),
        bf(Slot::Intermediate, kSelections * 640), diag, flattened, 640, kSelections);
    addGatheredAffine(graph, bf(Slot::Intermediate, kSelections * 640),
        impl_->weights.projection(mlp + ".switch_mlp.down_proj"), expertIDs,
        bf(Slot::ExpertDown, kSelections * kWidth), diag, flattened, kSelections, true);
    affine(graph, mlp + ".shared_expert.gate_proj", mixed, bf(Slot::SharedGate, 640));
    affine(graph, mlp + ".shared_expert.up_proj", mixed, bf(Slot::SharedUp, 640));
    addSiLUMultiply(graph, bf(Slot::SharedGate, 640), bf(Slot::SharedUp, 640),
        bf(Slot::SharedIntermediate, 640), diag, flattened, 640);
    affine(graph, mlp + ".shared_expert.down_proj", bf(Slot::SharedIntermediate, 640), bf(Slot::SharedDown, kWidth));
    affine(graph, mlp + ".shared_expert_gate", mixed, bf(Slot::SharedGateLogit, 1));
    addCombine(graph, bf(Slot::ExpertDown, kSelections * kWidth), expertIDs, routes,
        bf(Slot::SharedDown, kWidth), bf(Slot::SharedGateLogit, 1), branch, diag,
        flattened, kWidth, 512, kSelections);
    const bool nextHasPLE = layer + 1 == impl_->descriptor.pleLayerIndices.front();
    normalizedReady = impl_->fuseHC && !nextHasPLE;
    if (normalizedReady) {
      const std::string nextHC = layer + 1 == impl_->descriptor.layers
          ? "language_model.model.hyper_connection_mixer"
          : "language_model.model.layers." + std::to_string(layer + 1) + ".attn_hyper_connection";
      const std::string nextNorm = nextHC + ".hc_norm.weight";
      addHCFusedInjectNorm(graph, hyper, branch, bf(Slot::Injection, 4),
          impl_->weights.tensor(nextNorm), hyper, bf(Slot::Normalized, kHyper), diag,
          hcGeometry, impl_->weights.normConvention(nextNorm));
    } else {
      addHCInject(graph, hyper, branch, bf(Slot::Injection, 4), hyper, hcGeometry);
    }
  }
  impl_->hc(graph, "language_model.model.hyper_connection_mixer", flattened, false, normalizedReady);
  const auto logits = bf(Slot::Logits, impl_->descriptor.vocabularySize);
  impl_->project(graph, "language_model.lm_head", mixed, logits, diag, flattened);
  metal::MetalBuffer compactGreedy;
  if (impl_->gpuGreedy) {
    compactGreedy = impl_->backend.view(impl_->greedyResults, 0,
        uint64_t{flattened} * sizeof(FlashGreedyGPURowResult));
    addGreedyGPU(graph, logits, impl_->greedyWorkspace, compactGreedy,
        flattened, impl_->descriptor.vocabularySize);
  }
  metal::CommandTiming timing;
  try {
    timing = impl_->backend.submitCommand(graph.dispatches());
    uint32_t status = 0; std::memcpy(&status, diag.contents(), sizeof(status));
    if (status) throw std::runtime_error("Flash batch sticky diagnostics failed: " + std::to_string(status));
  } catch (...) {
    for (uint32_t lane = 0; lane < lanes; ++lane) states[lane]->poisoned = true;
    throw;
  }
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    lengths[lane] = states[lane]->length += rows;
    states[lane]->pendingVerification = true;
  }
  impl_->pendingLanes = std::move(pendingLanes);
  impl_->pendingRows = rows;
  lazyCleanup.keepPending = true;
  return {timing, logits, hyper, std::move(lengths), lanes, rows, impl_->capacity,
      compactGreedy, compactGreedy ? flattened : 0};
}

metal::CommandTiming FlashBatchVerify::commitBatch(
    std::span<FlashRequestState *const> requests, std::span<const uint32_t> retained) {
  if (!impl_) throw std::logic_error("Flash batch verifier is not initialized");
  std::scoped_lock lock(impl_->mutex, impl_->trunk.batchMutex());
  if (impl_->trunk.impl_.get() != impl_->trunkIdentity ||
      impl_->weights.manifestFingerprint() != impl_->weightIdentity)
    throw std::invalid_argument("Flash batch verifier source trunk/model was replaced");
  if (!impl_->pendingRows || requests.size() != impl_->pendingLanes.size() || retained.size() != requests.size())
    throw std::invalid_argument("Flash batch verifier commit has no matching pending lane set");
  const uint32_t lanes = static_cast<uint32_t>(requests.size());
  const uint32_t rows = impl_->pendingRows;
  bool needsRestore = false;
  std::array<FlashRequestState::Impl *, 4> states{};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const auto &pending = impl_->pendingLanes[lane];
    if (retained[lane] > rows)
      throw std::invalid_argument("Flash batch verifier prefix exceeds incoming rows");
    const auto identity = pending.identity.lock();
    if (!identity) {
      if (!retained[lane] && !requests[lane]) continue;
      throw std::invalid_argument("expired Flash batch lane must be terminally aborted with null pointer");
    }
    if (!retained[lane] && !requests[lane]) {
      if (!pending.state || pending.state->identity != identity ||
          pending.state->length != pending.begin + rows || !pending.state->pendingVerification)
        throw std::invalid_argument("Flash batch verifier terminal lane has changed");
      states[lane] = pending.state;
      continue;
    }
    if (!requests[lane] || !requests[lane]->impl_ ||
        requests[lane]->impl_.get() != pending.state ||
        identity != requests[lane]->impl_->identity ||
        requests[lane]->impl_->poisoned || !requests[lane]->impl_->pendingVerification ||
        requests[lane]->impl_->length != pending.begin + rows)
      throw std::invalid_argument("Flash batch verifier commit has foreign/stale/unhealthy state or invalid prefix");
    states[lane] = requests[lane]->impl_.get();
    needsRestore |= retained[lane] && retained[lane] < rows;
  }
  struct LazyCommitCleanup final {
    Impl &owner;
    std::array<FlashRequestState::Impl *, 4> &states;
    bool resolved = false;
    ~LazyCommitCleanup() {
      if (!owner.lazyGDN || resolved) return;
      for (auto *state : states)
        if (state) { state->poisoned = true; state->pendingVerification = false; }
      owner.abortLazyGDN();
      owner.pendingLanes.clear(); owner.pendingRows = 0;
    }
  } lazyCleanup{*impl_, states};
  metal::CommandTiming timing;
  if (needsRestore) {
    const auto diag = impl_->view(Slot::Diagnostics, 4);
    std::memset(diag.contents(), 0, 4);
    const auto counts = impl_->view(Slot::RetainedCounts, uint64_t{lanes} * 4);
    // Zero lanes restore only this arena's scratch snapshots. Their original
    // state storage is never copied or promoted after terminal cancellation.
    std::memcpy(counts.contents(), retained.data(), uint64_t{lanes} * 4);
    metal::CommandGraph graph;
    for (uint32_t layer = 0; layer < impl_->descriptor.layers; ++layer) {
      if (impl_->descriptor.layerKinds[layer] != FlashLayerKind::GatedDeltaNet) continue;
      if (impl_->lazyGDN) {
        impl_->lazyGDNLayers[layer]->commit(graph, impl_->lazyGDNTickets[layer], retained);
        impl_->lazyGDNTickets[layer] = 0;
      }
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        if (!retained[lane] || retained[lane] == rows) continue;
        // Each lazy layer replays into the same packed scratch. Scatter its
        // partial live lanes before the next layer overwrites that scratch.
        const auto convolution = impl_->lazyGDN
            ? impl_->lane(impl_->scratch[static_cast<size_t>(Slot::PackedGDNConvolution)],
                          lane, flashGDNConvolutionLaneBytes(),
                          kFlashBatchVerifyGDNConvolutionRowStrideBytes)
            : impl_->prefixLane(false, layer, lane, retained[lane] - 1);
        const auto recurrent = impl_->lazyGDN
            ? impl_->lane(impl_->scratch[static_cast<size_t>(Slot::PackedGDNRecurrent)],
                          lane, flashGDNRecurrentLaneBytes(),
                          kFlashBatchVerifyGDNRecurrentRowStrideBytes)
            : impl_->prefixLane(true, layer, lane, retained[lane] - 1);
        impl_->copy(graph, convolution,
            states[lane]->gdn[layer].convolution, flashGDNConvolutionLaneBytes());
        impl_->copy(graph, recurrent,
            states[lane]->gdn[layer].recurrent, flashGDNRecurrentLaneBytes());
      }
    }
    const auto history = impl_->view(Slot::PackedPLEHistory, uint64_t{lanes} * 16);
    const auto convolution = impl_->view(Slot::PackedPLEConvolution, uint64_t{lanes} * kPLEConvolutionBytes);
    const FlashPLEGeometry geometry{lanes, rows, kWidth, 4, impl_->descriptor.pleHistoryEos,
        impl_->descriptor.vocabularySize, static_cast<float>(impl_->descriptor.normEpsilon)};
    addPLERestorePrefix(graph, impl_->view(Slot::BeforePLEHistory, uint64_t{lanes} * 16),
        impl_->view(Slot::Tokens, uint64_t{lanes} * rows * 8),
        impl_->view(Slot::BeforePLEConvolution, uint64_t{lanes} * kPLEConvolutionBytes),
        impl_->bf(Slot::PLENormConvolution, lanes * rows, kHyper), counts,
        history, convolution, diag, geometry);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      if (!retained[lane]) continue;
      impl_->copy(graph, impl_->lane(history, lane, 16), states[lane]->pleHistory, 16);
      impl_->copy(graph, impl_->lane(convolution, lane, kPLEConvolutionBytes),
          states[lane]->pleConvolution, kPLEConvolutionBytes);
    }
    try {
      timing = impl_->backend.submitCommand(graph.dispatches());
      uint32_t status = 0; std::memcpy(&status, diag.contents(), sizeof(status));
      if (status) throw std::runtime_error("Flash batch verifier prefix restore diagnostics failed: " + std::to_string(status));
    } catch (...) {
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        if (states[lane]) { states[lane]->poisoned = true; states[lane]->pendingVerification = false; }
      }
      impl_->pendingLanes.clear(); impl_->pendingRows = 0;
      throw;
    }
  } else if (impl_->lazyGDN) {
    // Full and terminal lanes need no replay, but each layer must release its
    // pending ticket so the next verification can start without a GPU command.
    metal::CommandGraph graph;
    for (uint32_t layer = 0; layer < impl_->descriptor.layers; ++layer) {
      if (impl_->descriptor.layerKinds[layer] != FlashLayerKind::GatedDeltaNet) continue;
      impl_->lazyGDNLayers[layer]->commit(graph, impl_->lazyGDNTickets[layer], retained);
      impl_->lazyGDNTickets[layer] = 0;
    }
    if (!graph.dispatches().empty())
      throw std::logic_error("Flash lazy GDN full/terminal commit unexpectedly requires replay");
  }
  // Promote every logical prefix together only after the entire restore graph
  // and sticky diagnostics succeed. QSA stale rows stay beyond the new length.
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (!states[lane]) continue;
    if (!retained[lane]) {
      states[lane]->poisoned = true; states[lane]->pendingVerification = false;
      continue;
    }
    states[lane]->length = impl_->pendingLanes[lane].begin + retained[lane];
    states[lane]->pendingVerification = false;
  }
  impl_->pendingLanes.clear(); impl_->pendingRows = 0;
  lazyCleanup.resolved = true;
  return timing;
}

void FlashBatchVerify::abortBatch() noexcept {
  if (!impl_) return;
  const auto discard = [&] {
  impl_->abortLazyGDN();
  for (const auto &pending : impl_->pendingLanes) {
    // Identity has no external strong owner: a live token proves the stable
    // heap implementation still exists, even after wrapper moves.
    if (const auto identity = pending.identity.lock()) {
      if (pending.state && pending.state->identity == identity) {
        pending.state->poisoned = true; pending.state->pendingVerification = false;
      }
    }
  }
  impl_->pendingLanes.clear(); impl_->pendingRows = 0;
  };
  if (impl_->trunk.impl_.get() == impl_->trunkIdentity) {
    std::scoped_lock lock(impl_->mutex, impl_->trunk.batchMutex());
    discard();
  } else {
    std::lock_guard lock(impl_->mutex);
    discard();
  }
}
bool FlashBatchVerify::pending() const noexcept { return impl_ && impl_->pendingRows; }

} // namespace splash::flash
