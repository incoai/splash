#include "flash/FlashBatchForward.hpp"

#include "flash/FlashAffine.hpp"
#include "flash/FlashGDN.hpp"
#include "flash/FlashGDNFused.hpp"
#include "flash/FlashGDNSeparate.hpp"
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
void validateGeometry(uint32_t capacity, uint32_t lanes) {
  if (!capacity || capacity > 262144 || !lanes || lanes > 4)
    throw std::invalid_argument("Flash batch requires capacity1..262144 and lanes1..4");
}

enum class Slot : size_t {
  Tokens, Embedding, Hyper, Normalized, Up, Down, RawInjection, Injection,
  Mixed, Branch, Q, K, V, Index, GDNMixed, Z, A, B, Decay, Beta, RecurrentRows,
  Attention, Router, ExpertIDs, Routes, ExpertGate, ExpertUp, Intermediate,
  ExpertDown, SharedGate, SharedUp, SharedIntermediate, SharedDown, SharedGateLogit,
  Ngrams, PLEEmbedding, PLEKey, PLEValue, PLENormKeys, PLENormQueries, PLEGated,
  PLENormConvolution, PLEOutput, Logits, Diagnostics, PackedGDNConvolution,
  PackedGDNRecurrent, PackedPLEHistory, PackedPLEConvolution, Count,
};
constexpr size_t kSlots = static_cast<size_t>(Slot::Count);

std::array<uint64_t, kSlots> sizes(uint32_t lanes) {
  std::array<uint64_t, kSlots> result{};
  const auto put = [&](Slot slot, uint64_t bytes) { result[static_cast<size_t>(slot)] = bytes; };
  const auto bf = [&](Slot slot, uint64_t width) { put(slot, uint64_t{lanes} * width * 2); };
  put(Slot::Tokens, uint64_t{lanes} * 8);
  bf(Slot::Embedding, kWidth); bf(Slot::Hyper, kHyper); bf(Slot::Normalized, kHyper);
  bf(Slot::Up, kHyper); bf(Slot::Down, 320); bf(Slot::RawInjection, 4); bf(Slot::Injection, 4);
  bf(Slot::Mixed, kWidth); bf(Slot::Branch, kWidth); bf(Slot::Q, 12288);
  bf(Slot::K, 512); bf(Slot::V, 512); bf(Slot::Index, 640);
  bf(Slot::GDNMixed, 10240); bf(Slot::Z, 6144); bf(Slot::A, 48); bf(Slot::B, 48);
  put(Slot::Decay, uint64_t{lanes} * 48 * 4); bf(Slot::Beta, 48);
  bf(Slot::RecurrentRows, 6144); bf(Slot::Attention, 6144); bf(Slot::Router, 512);
  put(Slot::ExpertIDs, uint64_t{lanes} * kSelections * 8); bf(Slot::Routes, kSelections);
  bf(Slot::ExpertGate, kSelections * 640); bf(Slot::ExpertUp, kSelections * 640);
  bf(Slot::Intermediate, kSelections * 640); bf(Slot::ExpertDown, kSelections * kWidth);
  bf(Slot::SharedGate, 640); bf(Slot::SharedUp, 640); bf(Slot::SharedIntermediate, 640);
  bf(Slot::SharedDown, kWidth); bf(Slot::SharedGateLogit, 1);
  put(Slot::Ngrams, uint64_t{lanes} * 16 * 8); bf(Slot::PLEEmbedding, kWidth);
  bf(Slot::PLEKey, kHyper); bf(Slot::PLEValue, kWidth); bf(Slot::PLENormKeys, kHyper);
  bf(Slot::PLENormQueries, kHyper); bf(Slot::PLEGated, kHyper);
  bf(Slot::PLENormConvolution, kHyper); bf(Slot::PLEOutput, kHyper);
  bf(Slot::Logits, 248320); put(Slot::Diagnostics, 4);
  put(Slot::PackedGDNConvolution, uint64_t{lanes} * rounded(flashGDNConvolutionLaneBytes()));
  put(Slot::PackedGDNRecurrent, uint64_t{lanes} * flashGDNRecurrentLaneBytes());
  put(Slot::PackedPLEHistory, uint64_t{lanes} * 16);
  put(Slot::PackedPLEConvolution, uint64_t{lanes} * kPLEConvolutionBytes);
  return result;
}

} // namespace

struct FlashBatchForward::Impl final {
  metal::MetalBackend &backend;
  const FlashWeights &weights;
  FlashForward &trunk;
  const void *trunkIdentity;
  const std::string weightIdentity;
  const FlashDescriptor descriptor;
  const uint32_t capacity, maximumLanes;
  std::array<metal::MetalBuffer, kSlots> scratch;
  FlashPLEWeights ple;
  std::unique_ptr<FlashPLESSD> pleSSD;
  FlashQSAWorkspace qsa;
  FlashQSAFastWorkspace qsaFast;
  const bool fuseHC = fusionEnabled("SPLASH_FLASH_FUSE_HC");
  const bool fuseGDN = fusionEnabled("SPLASH_FLASH_FUSE_GDN");
  const bool mppQSA = fusionEnabled("SPLASH_FLASH_QSA_MPP");
  const bool gpuGreedy = fusionEnabled("SPLASH_FLASH_GPU_GREEDY");
  FlashGreedyGPUWorkspace greedyWorkspace;
  metal::MetalBuffer greedyResults;
  bool onlineQSA = false;
  uint64_t allocatedBytes = 0;
  std::mutex mutex;

  Impl(metal::MetalBackend &value, const FlashWeights &model, FlashForward &source,
       uint32_t context, uint32_t lanes, const void *sourceIdentity)
      : backend(value), weights(model), trunk(source), trunkIdentity(sourceIdentity),
        weightIdentity(model.manifestFingerprint()), descriptor(model.descriptor()),
        capacity(context), maximumLanes(lanes), ple(FlashPLEWeights::fromWeights(model)) {
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
      throw std::invalid_argument("Flash batch QSA route flags changed after source trunk construction");
    validateGeometry(capacity, maximumLanes);
    descriptor.validate();
    if (!descriptor.pleParametersLoaded) throw std::invalid_argument("Flash batch requires loaded PLE parameters");
    const uint64_t before = backend.memoryStats().allocatedBytes;
    if (weights.pleSSDStreamingEnabled())
      pleSSD = std::make_unique<FlashPLESSD>(backend, weights.pleSSDStore(), ple, maximumLanes, 1);
    if (gpuGreedy) {
      greedyWorkspace = allocateGreedyGPUWorkspace(backend, maximumLanes, descriptor.vocabularySize);
      greedyResults = backend.allocateBuffer(uint64_t{maximumLanes} * sizeof(FlashGreedyGPURowResult),
          metal::BufferStorage::Shared, "flash-batch compact GPU greedy records");
    }
    const auto extents = sizes(maximumLanes);
    for (size_t index = 0; index < extents.size(); ++index)
      scratch[index] = backend.allocateBuffer(rounded(extents[index]), metal::BufferStorage::Shared,
          "flash-batch-shared-scratch-" + std::to_string(index));
    qsa = allocateQSAWorkspace(backend, 1, capacity);
    if (onlineQSA) qsaFast = mppQSA ? allocateQSAOnlineMPPWorkspace(backend, 1, 32)
                                  : allocateQSAFastWorkspace(backend, 1, 4);
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
      addHCFusedUpMix(graph, normed, down, upProjection, bf(Slot::Mixed, rows, kWidth),
          diag, geometry);
      return;
    }
    trunk.batchProject(graph, prefix + ".input_mix_weight_down", normed, down, diag, rows);
    graph.add("flash_forward_hc_silu", {down, down, diag}, FlashForwardActivationParams{rows, 320, 4},
        {(uint64_t{rows} * 320 + 255) / 256, 1, 1});
    trunk.batchProject(graph, prefix + ".input_mix_weight_up", down, up, diag, rows);
    if (injection) {
      const auto raw = bf(Slot::RawInjection, rows, 4);
      trunk.batchProject(graph, prefix + ".block_inject_weight", normed, raw, diag, rows);
      addHCMixWithInjection(graph, normed, up, raw, bf(Slot::Mixed, rows, kWidth),
          bf(Slot::Injection, rows, 4), geometry);
    } else addHCMix(graph, normed, up, bf(Slot::Mixed, rows, kWidth), geometry);
  }
};

FlashBatchForward::FlashBatchForward(metal::MetalBackend &backend, const FlashWeights &weights,
                                     FlashForward &trunk, uint32_t capacity, uint32_t maximumLanes) {
  validateGeometry(capacity, maximumLanes);
  if (&backend != &trunk.batchBackend() || &weights != &trunk.batchWeights() || capacity != trunk.batchCapacity())
    throw std::invalid_argument("Flash batch backend/model/context must match its source trunk");
  impl_ = std::make_unique<Impl>(backend, weights, trunk, capacity, maximumLanes, trunk.impl_.get());
}
FlashBatchForward::~FlashBatchForward() = default;
FlashBatchForward::FlashBatchForward(FlashBatchForward &&) noexcept = default;
FlashBatchForward &FlashBatchForward::operator=(FlashBatchForward &&) noexcept = default;
uint64_t FlashBatchForward::workspaceBytes() const noexcept { return impl_ ? impl_->allocatedBytes : 0; }
uint64_t FlashBatchForward::pleSSDStagingBytes() const noexcept {
  return impl_ && impl_->pleSSD ? impl_->pleSSD->allocatedBytes() : 0;
}

uint64_t FlashBatchForward::workspacePlannedBytes(uint32_t capacity, uint32_t maximumLanes) {
  validateGeometry(capacity, maximumLanes);
  uint64_t total = 0;
  for (uint64_t extent : sizes(maximumLanes)) total += rounded(extent);
  const uint64_t blocks = (uint64_t{capacity} + 3) / 4;
  for (uint64_t extent : std::array<uint64_t, 6>{6144 * 2, 512 * 2, blocks * 4,
           512 * 4, uint64_t{24} * kFlashQSATokenWidth * 4, uint64_t{24} * kFlashQSATokenWidth * 2})
    total += rounded(extent);
  total += rounded(uint64_t{24} * 32 * 2 * 4);
  total += rounded(uint64_t{24} * 32 * 256 * 4);
  if (fusionEnabled("SPLASH_FLASH_GPU_GREEDY"))
    total += greedyGPUWorkspacePlannedBytes(maximumLanes, 248320);
  if (flashPLESSDStreamingValue(std::getenv("SPLASH_FLASH_PLE_SSD_STREAMING")))
    total += FlashPLESSD::plannedBytes(maximumLanes, 1);
  return total;
}

FlashBatchResult FlashBatchForward::forwardBatch(std::span<FlashRequestState *const> requests,
                                                 std::span<const uint32_t> tokens) {
  if (!impl_) throw std::logic_error("Flash batch is not initialized");
  std::scoped_lock lock(impl_->mutex, impl_->trunk.batchMutex());
  if (impl_->trunk.impl_.get() != impl_->trunkIdentity ||
      &impl_->backend != &impl_->trunk.batchBackend() ||
      &impl_->weights != &impl_->trunk.batchWeights() || impl_->capacity != impl_->trunk.batchCapacity() ||
      impl_->weights.manifestFingerprint() != impl_->weightIdentity)
    throw std::invalid_argument("Flash batch source trunk/model was replaced after construction");
  if (requests.empty() || requests.size() > impl_->maximumLanes || requests.size() != tokens.size())
    throw std::invalid_argument("Flash batch requires one token per active request");
  const uint32_t lanes = static_cast<uint32_t>(requests.size());
  std::array<FlashRequestState::Impl *, 4> states{};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (!requests[lane] || !impl_->trunk.ownsState(*requests[lane]) ||
        tokens[lane] >= impl_->descriptor.vocabularySize)
      throw std::invalid_argument("Flash batch contains a foreign/pending/unhealthy state or invalid token");
    states[lane] = requests[lane]->impl_.get();
    if (states[lane]->length >= impl_->capacity)
      throw std::invalid_argument("Flash batch request has no remaining context capacity");
    for (uint32_t prior = 0; prior < lane; ++prior)
      if (states[lane]->identity == states[prior]->identity)
        throw std::invalid_argument("Flash batch includes the same request more than once");
  }
  std::vector<uint64_t> lengths(lanes);
  const auto ids = impl_->view(Slot::Tokens, uint64_t{lanes} * 8);
  auto *hostTokens = static_cast<int64_t *>(ids.contents());
  if (!hostTokens) throw std::logic_error("Flash batch token input is not shared");
  for (uint32_t lane = 0; lane < lanes; ++lane) hostTokens[lane] = tokens[lane];
  const auto diag = impl_->view(Slot::Diagnostics, 4);
  std::memset(diag.contents(), 0, 4);
  const auto bf = [&](Slot slot, uint32_t width) { return impl_->bf(slot, lanes, width); };
  const auto affine = [&](metal::CommandGraph &graph, const std::string &prefix,
                          const metal::MetalBuffer &input, const metal::MetalBuffer &output) {
    impl_->trunk.batchProject(graph, prefix, input, output, diag, lanes);
  };
  const auto hyper = bf(Slot::Hyper, kHyper), mixed = bf(Slot::Mixed, kWidth);
  const auto branch = bf(Slot::Branch, kWidth), attentionOutput = bf(Slot::Attention, 6144);
  const FlashHCGeometry hcGeometry{lanes, kWidth, 4, static_cast<float>(impl_->descriptor.normEpsilon)};
  const FlashPLEGeometry pleGeometry{lanes, 1, kWidth, 4, impl_->descriptor.pleHistoryEos,
      impl_->descriptor.vocabularySize, static_cast<float>(impl_->descriptor.normEpsilon)};
  if (impl_->pleSSD) {
    std::array<int64_t, 8> histories{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const auto *history = static_cast<const int64_t *>(states[lane]->pleHistory.contents());
      if (!history) throw std::logic_error("Flash PLE SSD batch history is not Shared");
      histories[lane * 2] = history[0]; histories[lane * 2 + 1] = history[1];
    }
    impl_->pleSSD->prepare({hostTokens, lanes}, {histories.data(), lanes * 2}, impl_->ple, pleGeometry);
  }
  metal::CommandGraph graph;
  addAffineEmbedding(graph, impl_->weights.projection("language_model.model.embed_tokens"), ids,
      bf(Slot::Embedding, kWidth), diag, lanes);
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
      const auto ngrams = impl_->view(Slot::Ngrams, uint64_t{lanes} * 16 * 8);
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
    impl_->hc(graph, prefix + ".attn_hyper_connection", lanes, true, normalizedReady);
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
      for (uint32_t lane = 0; !impl_->fuseGDN && lane < lanes; ++lane) {
        impl_->copy(graph, states[lane]->gdn[layer].convolution,
            impl_->lane(packedConv, lane, flashGDNConvolutionLaneBytes(), convStride),
            flashGDNConvolutionLaneBytes());
        impl_->copy(graph, states[lane]->gdn[layer].recurrent,
            impl_->lane(packedRec, lane, flashGDNRecurrentLaneBytes()), flashGDNRecurrentLaneBytes());
      }
      const FlashGDNWeights weights{&impl_->weights.tensor(attention + ".conv1d.weight"),
          &impl_->weights.tensor(attention + ".A_log"), &impl_->weights.tensor(attention + ".dt_bias"),
          &impl_->weights.tensor(attention + ".norm.weight")};
      const FlashGDNBuffers buffers{qkv, bf(Slot::Z, 6144), bf(Slot::A, 48), bf(Slot::B, 48),
          bf(Slot::GDNMixed, 10240), impl_->view(Slot::Decay, uint64_t{lanes} * 48 * 4),
          bf(Slot::Beta, 48), bf(Slot::RecurrentRows, 6144), attentionOutput, diag};
      const FlashGDNState packed{packedConv, packedRec, convStride, flashGDNRecurrentLaneBytes()};
      if (impl_->fuseGDN) {
        std::array<FlashGDNState, 4> directStates;
        for (uint32_t lane = 0; lane < lanes; ++lane) directStates[lane] = states[lane]->gdn[layer];
        addGDNFusedSeparateStates(graph, weights, buffers, directStates, lanes,
            static_cast<float>(impl_->descriptor.normEpsilon));
      }
      else
        addGDN(graph, weights, buffers, packed, 1, lanes,
            static_cast<float>(impl_->descriptor.normEpsilon));
      for (uint32_t lane = 0; !impl_->fuseGDN && lane < lanes; ++lane) {
        impl_->copy(graph, impl_->lane(packedConv, lane, flashGDNConvolutionLaneBytes(), convStride),
            states[lane]->gdn[layer].convolution, flashGDNConvolutionLaneBytes());
        impl_->copy(graph, impl_->lane(packedRec, lane, flashGDNRecurrentLaneBytes()),
            states[lane]->gdn[layer].recurrent, flashGDNRecurrentLaneBytes());
      }
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
          const FlashQSAFastInputs inputs{impl_->lane(q, lane, 12288 * 2),
              impl_->lane(k, lane, 512 * 2), impl_->lane(v, lane, 512 * 2),
              impl_->lane(index, lane, 640 * 2), &impl_->weights.tensor(qNorm),
              &impl_->weights.tensor(kNorm), &impl_->weights.tensor(iqNorm), &impl_->weights.tensor(ikNorm),
              impl_->lane(attentionOutput, lane, 6144 * 2), diag, {},
              impl_->weights.normConvention(qNorm), impl_->weights.normConvention(kNorm),
              impl_->weights.normConvention(iqNorm), impl_->weights.normConvention(ikNorm),
              impl_->descriptor.normEpsilon, impl_->descriptor.rotaryTheta};
          const uint32_t begin = static_cast<uint32_t>(states[lane]->length);
          const uint32_t partitions = impl_->mppQSA ? qsaOnlineMPPRoutePartitions(begin, 1) : 0;
          if (partitions)
            addQSAOnlineMPP(graph, inputs, states[lane]->qsa[layer], impl_->qsa,
                impl_->qsaFast, begin, 1, partitions, true);
          else
            addQSAFast(graph, inputs, states[lane]->qsa[layer], impl_->qsa, impl_->qsaFast,
                begin, 1, FlashQSAFastMode::PartitionedF32Probabilities, 4, true);
        } else {
        addQSA(graph, impl_->lane(q, lane, 12288 * 2), impl_->lane(k, lane, 512 * 2),
            impl_->lane(v, lane, 512 * 2), impl_->lane(index, lane, 640 * 2),
            impl_->weights.tensor(qNorm), impl_->weights.tensor(kNorm), impl_->weights.tensor(iqNorm),
            impl_->weights.tensor(ikNorm), states[lane]->qsa[layer], impl_->qsa,
            impl_->lane(attentionOutput, lane, 6144 * 2), diag,
            static_cast<uint32_t>(states[lane]->length), 1,
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
    impl_->hc(graph, prefix + ".mlp_hyper_connection", lanes, true, normalizedReady);
    const auto mlp = prefix + ".mlp";
    const auto expertIDs = impl_->view(Slot::ExpertIDs, uint64_t{lanes} * kSelections * 8);
    const auto routes = bf(Slot::Routes, kSelections);
    addDenseBF16(graph, mixed, impl_->weights.tensor(mlp + ".gate.weight"), bf(Slot::Router, 512), diag, lanes);
    addRoute(graph, bf(Slot::Router, 512), expertIDs, routes, diag, lanes, 512, kSelections);
    addGatheredAffine(graph, mixed, impl_->weights.projection(mlp + ".switch_mlp.gate_proj"),
        expertIDs, bf(Slot::ExpertGate, kSelections * 640), diag, lanes, kSelections);
    addGatheredAffine(graph, mixed, impl_->weights.projection(mlp + ".switch_mlp.up_proj"),
        expertIDs, bf(Slot::ExpertUp, kSelections * 640), diag, lanes, kSelections);
    addSiLUMultiply(graph, bf(Slot::ExpertGate, kSelections * 640), bf(Slot::ExpertUp, kSelections * 640),
        bf(Slot::Intermediate, kSelections * 640), diag, lanes, 640, kSelections);
    addGatheredAffine(graph, bf(Slot::Intermediate, kSelections * 640),
        impl_->weights.projection(mlp + ".switch_mlp.down_proj"), expertIDs,
        bf(Slot::ExpertDown, kSelections * kWidth), diag, lanes, kSelections, true);
    affine(graph, mlp + ".shared_expert.gate_proj", mixed, bf(Slot::SharedGate, 640));
    affine(graph, mlp + ".shared_expert.up_proj", mixed, bf(Slot::SharedUp, 640));
    addSiLUMultiply(graph, bf(Slot::SharedGate, 640), bf(Slot::SharedUp, 640),
        bf(Slot::SharedIntermediate, 640), diag, lanes, 640);
    affine(graph, mlp + ".shared_expert.down_proj", bf(Slot::SharedIntermediate, 640), bf(Slot::SharedDown, kWidth));
    affine(graph, mlp + ".shared_expert_gate", mixed, bf(Slot::SharedGateLogit, 1));
    addCombine(graph, bf(Slot::ExpertDown, kSelections * kWidth), expertIDs, routes,
        bf(Slot::SharedDown, kWidth), bf(Slot::SharedGateLogit, 1), branch, diag,
        lanes, kWidth, 512, kSelections);
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
  impl_->hc(graph, "language_model.model.hyper_connection_mixer", lanes, false, normalizedReady);
  const auto logits = bf(Slot::Logits, impl_->descriptor.vocabularySize);
  impl_->trunk.batchProject(graph, "language_model.lm_head", mixed, logits, diag, lanes);
  metal::MetalBuffer compactGreedy;
  if (impl_->gpuGreedy) {
    compactGreedy = impl_->backend.view(impl_->greedyResults, 0,
        uint64_t{lanes} * sizeof(FlashGreedyGPURowResult));
    addGreedyGPU(graph, logits, impl_->greedyWorkspace, compactGreedy,
        lanes, impl_->descriptor.vocabularySize);
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
  for (uint32_t lane = 0; lane < lanes; ++lane) lengths[lane] = ++states[lane]->length;
  return {timing, logits, hyper, std::move(lengths), lanes, impl_->capacity,
      compactGreedy, compactGreedy ? lanes : 0};
}

} // namespace splash::flash
