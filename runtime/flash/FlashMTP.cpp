#include "flash/FlashMTP.hpp"

#include "flash/FlashAffine.hpp"
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashGreedyGPU.hpp"
#include "flash/FlashHC.hpp"
#include "flash/FlashHCFused.hpp"
#include "flash/FlashMoE.hpp"
#include "flash/FlashMTPQMVPolicy.hpp"
#include "flash/FlashQSA.hpp"
#include "flash/FlashQSAFast.hpp"
#include "flash/FlashQSAMPP.hpp"
#include "flash/FlashMTPStateInternal.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashForward.h"
#include "metal/abi/FlashMTP.h"
#include "metal/abi/FlashAffine.h"
#include "metal/abi/FlashQSAFast.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace splash::flash {
namespace {
constexpr uint32_t kWidth = 2560;
constexpr uint32_t kHyper = 10240;
constexpr uint32_t kSelections = 10;
constexpr uint64_t kAlignment = 16384;

bool enabled(const char *name) {
  const char *value = std::getenv(name);
  if (!value) return false;
  if (std::string_view(value) == "1") return true;
  if (std::string_view(value) == "0") return false;
  throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}
struct AttentionPolicy final {
  const bool f32 = enabled("SPLASH_FLASH_MTP_QSA_F32");
  const bool mpp = enabled("SPLASH_FLASH_MTP_QSA_MPP");
  AttentionPolicy() {
    if (mpp && !f32)
      throw std::invalid_argument("SPLASH_FLASH_MTP_QSA_MPP=1 requires SPLASH_FLASH_MTP_QSA_F32=1");
  }
  [[nodiscard]] uint32_t maximumPartitions() const noexcept { return mpp ? 32 : 4; }
  [[nodiscard]] const char *semantics() const noexcept {
    return !f32 ? "mtp-qsa-bf16-probabilities-v1" : mpp
        ? kFlashQSAOnlineMPPRoute + 1
        : "mtp-qsa-fused-prep-f32-online-partitions4-v1";
  }
};
void addHeadQSA(metal::CommandGraph &graph, const FlashQSAFastInputs &input,
    FlashQSAState &state, FlashQSAWorkspace &workspace,
    FlashQSAFastWorkspace &fastWorkspace, uint32_t begin, uint32_t rows,
    const AttentionPolicy &policy) {
  if (!policy.f32) {
    addQSA(graph, input.qProjection, input.kProjection, input.vProjection,
        input.indexProjection, *input.qNorm, *input.kNorm, *input.indexQNorm,
        *input.indexKNorm, state, workspace, input.output, input.diagnostics,
        begin, rows, input.qConvention, input.kConvention,
        input.indexQConvention, input.indexKConvention, input.epsilon, input.theta);
  } else {
    const uint32_t mppPartitions = policy.mpp ? qsaOnlineMPPRoutePartitions(begin, rows) : 0;
    if (mppPartitions)
      addQSAOnlineMPP(graph, input, state, workspace, fastWorkspace,
          begin, rows, mppPartitions, true);
    else
      addQSAFast(graph, input, state, workspace, fastWorkspace, begin, rows,
          FlashQSAFastMode::PartitionedF32Probabilities, 4, true);
  }
}
void addTeacherQSACache(metal::CommandGraph &graph, const FlashQSAFastInputs &input,
    FlashQSAState &state, FlashQSAWorkspace &workspace,
    FlashQSAFastWorkspace &fastWorkspace, uint32_t begin, uint32_t rows,
    const AttentionPolicy &policy) {
  // Copy the authoritative cache-writing prefix. Selection and attention
  // write only scratch; none is consumed by a later independent teacher pair.
  metal::CommandGraph qualified;
  addHeadQSA(qualified, input, state, workspace, fastWorkspace, begin, rows, policy);
  bool appendSeen = false, stopped = false;
  for (const auto &dispatch : qualified.dispatches()) {
    const auto &name = dispatch.pipelineName;
    if (name == "flash_qsa_index_scores" || name == "flash_qsa_select_blocks") {
      stopped = true;
      break;
    }
    const bool preparation = name == "flash_qsa_fast_prepare" ||
        name.starts_with("flash_qsa_norm_rope_") || name == "flash_qsa_append_aux" ||
        name.starts_with("flash_qsa_pool_rope_");
    if (!preparation || dispatch.bytes.size() != 1 ||
        dispatch.bytes[0].index != dispatch.buffers.size())
      throw std::logic_error("Flash teacher QSA cache prefix changed");
    appendSeen |= name == "flash_qsa_fast_prepare" || name == "flash_qsa_append_aux";
    std::vector<metal::MetalBuffer> buffers;
    buffers.reserve(dispatch.buffers.size());
    for (uint32_t index = 0; index < dispatch.buffers.size(); ++index) {
      if (dispatch.buffers[index].index != index)
        throw std::logic_error("Flash teacher QSA cache binding order changed");
      buffers.push_back(dispatch.buffers[index].buffer);
    }
    if (dispatch.bytes[0].sizeBytes == sizeof(FlashQSAFastParams)) {
      FlashQSAFastParams params;
      std::memcpy(&params, dispatch.bytes[0].data, sizeof(params));
      graph.add(name, std::move(buffers), params,
          dispatch.threadgroups, dispatch.threadsPerThreadgroup);
    } else if (dispatch.bytes[0].sizeBytes == sizeof(FlashQSAParams)) {
      FlashQSAParams params;
      std::memcpy(&params, dispatch.bytes[0].data, sizeof(params));
      graph.add(name, std::move(buffers), params,
          dispatch.threadgroups, dispatch.threadsPerThreadgroup);
    } else {
      throw std::logic_error("Flash teacher QSA cache parameter ABI changed");
    }
  }
  if (!appendSeen || !stopped)
    throw std::logic_error("Flash teacher QSA cache prefix is incomplete");
}
std::vector<std::string> denseHeadPrefixes() {
  // Cache only trained head matrices. Shared embedding/vocabulary matrices
  // and 512-expert banks remain in their original storage without expansion.
  std::vector<std::string> names{"mtp.fc_embedding", "mtp.fc_hidden"};
  for (const auto *hc : {"mtp.layers.0.attn_hyper_connection",
       "mtp.layers.0.mlp_hyper_connection", "mtp.hyper_connection_mixer"}) {
    names.emplace_back(std::string(hc) + ".input_mix_weight_down");
    names.emplace_back(std::string(hc) + ".input_mix_weight_up");
  }
  for (const auto *projection : {"q_proj", "k_proj", "v_proj", "o_proj",
       "indexer.index_qk_proj"})
    names.emplace_back(std::string("mtp.layers.0.self_attn.") + projection);
  for (const auto *projection : {"gate_proj", "up_proj", "down_proj"})
    names.emplace_back(std::string("mtp.layers.0.mlp.shared_expert.") + projection);
  return names;
}

uint64_t rounded(uint64_t bytes) {
  if (!bytes || bytes > std::numeric_limits<uint64_t>::max() - (kAlignment - 1))
    throw std::overflow_error("Flash MTP allocation extent overflows");
  return (bytes + kAlignment - 1) & ~(kAlignment - 1);
}
void validateCapacity(uint32_t capacity) {
  if (!capacity || capacity > 262144)
    throw std::invalid_argument("Flash MTP capacity must be 1..262144");
}
void zero(const metal::MetalBuffer &buffer) {
  if (!buffer || !buffer.contents() || buffer.storage() != metal::BufferStorage::Shared)
    throw std::logic_error("Flash MTP cold state requires shared storage");
  std::memset(buffer.contents(), 0, buffer.sizeBytes());
}
enum class Scratch : size_t {
  TokenIDs, Embedding, EmbeddingNormalized, EmbeddingProjected,
  HiddenNormalized, HiddenProjected, Hyper, HCNormalized, HCDown, HCUp,
  HCRawInjection, HCInjectionWeights, Mixed, Branch, QProjection, KProjection,
  VProjection, IndexProjection, AttentionOutput, Router, ExpertIDs, RouteWeights,
  ExpertGate, ExpertUp, ExpertIntermediate, ExpertDown, SharedGate, SharedUp,
  SharedIntermediate, SharedDown, SharedGateLogit, HeadLogits, Diagnostics, Count,
};
constexpr size_t kScratchCount = static_cast<size_t>(Scratch::Count);
} // namespace

struct FlashMTPForward::Impl final {
  metal::MetalBackend &backend;
  const FlashWeights &weights;
  FlashDescriptor descriptor;
  uint32_t capacity;
  uint32_t maximumRows;
  const bool cacheDense = enabled("SPLASH_FLASH_DENSE_CACHE");
  const bool fuseHC = enabled("SPLASH_FLASH_FUSE_HC");
  const bool gpuGreedy = enabled("SPLASH_FLASH_GPU_GREEDY");
  const bool qmvF32Proposals = flashMTPQMVF32Switch(
      std::getenv("SPLASH_FLASH_MTP_QMV_F32"), flashAffineFastEnabled());
  const AttentionPolicy attentionPolicy;
  std::string routeMetadata;
  std::shared_ptr<const uint8_t> owner = std::make_shared<const uint8_t>(0);
  std::array<metal::MetalBuffer, kScratchCount> scratch;
  FlashQSAWorkspace qsaWorkspace;
  FlashQSAFastWorkspace qsaFastWorkspace;
  std::unique_ptr<FlashDenseCache> denseCache;
  FlashGreedyGPUWorkspace greedyWorkspace;
  metal::MetalBuffer greedyResults;
  uint64_t workspaceBytes = 0;
  std::mutex mutex;

  Impl(metal::MetalBackend &metalBackend, const FlashWeights &model,
       uint32_t context, uint32_t rows)
      : backend(metalBackend), weights(model), descriptor(model.descriptor()),
        capacity(context), maximumRows(rows) {
    validateCapacity(capacity);
    descriptor.validate();
    if (descriptor.mtpLayers != 1 || !maximumRows || maximumRows > 128)
      throw std::invalid_argument("Flash MTP requires one trained layer and 1..128 rows");
    // Resolve required trained input projections before allocating scratch.
    for (const auto *name : {"mtp.fc_embedding", "mtp.fc_hidden"}) {
      const auto &projection = weights.projection(name);
      if (projection.experts != 1 || projection.inputSize != kWidth ||
          projection.outputSize != kWidth)
        throw std::invalid_argument("Flash MTP trained input projection has invalid shape");
    }
    if (qmvF32Proposals) {
      const auto &p = weights.projection("mtp.fc_hidden");
      if (!flashMTPQMVF32Geometry(p.experts, p.outputSize, p.inputSize, p.bits,
              p.groupSize, p.weightRowStrideBytes, p.parameterRowStrideBytes) ||
          !p.weights || !p.scales || !p.biases || p.weights->dtype != FlashDType::U32 ||
          p.scales->dtype != FlashDType::BF16 || p.biases->dtype != FlashDType::BF16 ||
          p.weights->shape != std::vector<uint64_t>{2560, 320} ||
          p.scales->shape != std::vector<uint64_t>{2560, 40} || p.biases->shape != p.scales->shape ||
          p.weights->logicalBytes < uint64_t{2560} * 1280 ||
          p.scales->logicalBytes < uint64_t{2560} * 80 || p.biases->logicalBytes < uint64_t{2560} * 80 ||
          !p.weights->buffer || p.weights->buffer.sizeBytes() < uint64_t{2560} * 1280 ||
          !p.scales->buffer || p.scales->buffer.sizeBytes() < uint64_t{2560} * 80 ||
          !p.biases->buffer || p.biases->buffer.sizeBytes() < uint64_t{2560} * 80)
        throw std::invalid_argument("MTP QMV F32 proposal requires original contiguous Q4/G64 fc_hidden");
    }
    routeMetadata = attentionPolicy.semantics();
    if (qmvF32Proposals) { routeMetadata += ';'; routeMetadata += kFlashMTPQMVF32ProposalSemantics; }
    const uint64_t before = backend.memoryStats().allocatedBytes;
    if (cacheDense) {
      const auto prefixes = denseHeadPrefixes();
      denseCache = std::make_unique<FlashDenseCache>(backend, weights, prefixes);
    }
    const auto allocate = [&](Scratch slot, uint64_t bytes, const char *label) {
      scratch[static_cast<size_t>(slot)] = backend.allocateBuffer(
          rounded(bytes), metal::BufferStorage::Shared, label);
    };
    const auto bf = [&](Scratch slot, uint32_t width, const char *label) {
      allocate(slot, uint64_t{maximumRows} * width * 2, label);
    };
    allocate(Scratch::TokenIDs, uint64_t{maximumRows} * 8, "flash-mtp-token-ids");
    bf(Scratch::Embedding, kWidth, "flash-mtp-embedding");
    bf(Scratch::EmbeddingNormalized, kWidth, "flash-mtp-embedding-normalized");
    bf(Scratch::EmbeddingProjected, kWidth, "flash-mtp-embedding-projected");
    bf(Scratch::HiddenNormalized, kHyper, "flash-mtp-hidden-normalized");
    bf(Scratch::HiddenProjected, kHyper, "flash-mtp-hidden-projected");
    bf(Scratch::Hyper, kHyper, "flash-mtp-hyper-state");
    bf(Scratch::HCNormalized, kHyper, "flash-mtp-hc-normalized");
    bf(Scratch::HCDown, 320, "flash-mtp-hc-down");
    bf(Scratch::HCUp, kHyper, "flash-mtp-hc-up");
    bf(Scratch::HCRawInjection, 4, "flash-mtp-hc-raw-injection");
    bf(Scratch::HCInjectionWeights, 4, "flash-mtp-hc-injection-weights");
    bf(Scratch::Mixed, kWidth, "flash-mtp-mixed-input");
    bf(Scratch::Branch, kWidth, "flash-mtp-branch");
    bf(Scratch::QProjection, 12288, "flash-mtp-q-projection");
    bf(Scratch::KProjection, 512, "flash-mtp-k-projection");
    bf(Scratch::VProjection, 512, "flash-mtp-v-projection");
    bf(Scratch::IndexProjection, 640, "flash-mtp-index-projection");
    bf(Scratch::AttentionOutput, 6144, "flash-mtp-attention-output");
    bf(Scratch::Router, 512, "flash-mtp-router");
    allocate(Scratch::ExpertIDs, uint64_t{maximumRows} * kSelections * 8, "flash-mtp-expert-ids");
    bf(Scratch::RouteWeights, kSelections, "flash-mtp-route-weights");
    bf(Scratch::ExpertGate, kSelections * 640, "flash-mtp-expert-gate");
    bf(Scratch::ExpertUp, kSelections * 640, "flash-mtp-expert-up");
    bf(Scratch::ExpertIntermediate, kSelections * 640, "flash-mtp-expert-intermediate");
    bf(Scratch::ExpertDown, kSelections * kWidth, "flash-mtp-expert-down");
    bf(Scratch::SharedGate, 640, "flash-mtp-shared-gate");
    bf(Scratch::SharedUp, 640, "flash-mtp-shared-up");
    bf(Scratch::SharedIntermediate, 640, "flash-mtp-shared-intermediate");
    bf(Scratch::SharedDown, kWidth, "flash-mtp-shared-down");
    bf(Scratch::SharedGateLogit, 1, "flash-mtp-shared-gate-logit");
    bf(Scratch::HeadLogits, descriptor.vocabularySize, "flash-mtp-head-logits");
    allocate(Scratch::Diagnostics, 4, "flash-mtp-diagnostics");
    if (gpuGreedy) {
      const auto greedyRows = std::min(maximumRows, uint32_t{kFlashGreedyGPUMaximumRows});
      greedyWorkspace = allocateGreedyGPUWorkspace(backend, greedyRows, descriptor.vocabularySize);
      greedyResults = backend.allocateBuffer(rounded(uint64_t{greedyRows} * sizeof(FlashGreedyGPURowResult)),
          metal::BufferStorage::Shared, "flash-mtp-compact-greedy-results");
    }
    qsaWorkspace = allocateQSAWorkspace(backend, maximumRows, capacity);
    if (attentionPolicy.f32)
      qsaFastWorkspace = attentionPolicy.mpp
          ? allocateQSAOnlineMPPWorkspace(backend, maximumRows, attentionPolicy.maximumPartitions())
          : allocateQSAFastWorkspace(backend, maximumRows, attentionPolicy.maximumPartitions());
    const uint64_t after = backend.memoryStats().allocatedBytes;
    if (after < before) throw std::logic_error("Flash MTP allocation ledger regressed");
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
               uint32_t logicalRows = 0, bool allowQmvProposals = true) {
    const uint32_t headRows = logicalRows ? logicalRows : rows;
    if (allowQmvProposals && qmvF32Proposals && prefix == "mtp.fc_hidden" && flashMTPQMVF32Window(rows, headRows)) {
      const auto &p = weights.projection(prefix);
      // Retain the established source/view extent validation; only this
      // trained proposal role selects the existing contiguous F32-sum kernel.
      metal::CommandGraph validated;
      addAffine(validated, input, p, output, diagnostics, rows);
      const FlashAffineParams qmv{rows, 1, p.inputSize, p.outputSize, 1, p.bits, p.groupSize, 0,
          p.weightRowStrideBytes, p.weightExpertStrideBytes,
          p.parameterRowStrideBytes, p.parameterExpertStrideBytes};
      graph.add("flash_affine_mlx_qmv_f32xsum_v1_q4_g64", {input, p.weights->buffer,
          p.scales->buffer, p.biases->buffer, input, output, diagnostics}, qmv,
          {(p.outputSize + 7) / 8, rows, 1}, {64, 1, 1});
      return;
    }
    if (denseCache && headRows >= 16 && denseCache->contains(prefix)) {
      const auto &projection = weights.projection(prefix);
      const auto tile = rows >= 128 && projection.outputSize >= 1024
          ? FlashAffineMPPTile::M32N128 : FlashAffineMPPTile::M16N64;
      denseCache->addProjection(graph, prefix, input, output, diagnostics, rows, tile);
    } else {
      addAffine(graph, input, weights.projection(prefix), output, diagnostics, rows);
    }
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
    if (!injection && denseCache &&
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

FlashMTPState::FlashMTPState() = default;
FlashMTPState::~FlashMTPState() = default;
FlashMTPState::FlashMTPState(FlashMTPState &&) noexcept = default;
FlashMTPState &FlashMTPState::operator=(FlashMTPState &&) noexcept = default;
uint64_t FlashMTPState::logicalLength() const noexcept { return impl_ ? impl_->length : 0; }
uint32_t FlashMTPState::capacity() const noexcept { return impl_ ? impl_->qsa.capacity : 0; }
bool FlashMTPState::poisoned() const noexcept { return !impl_ || impl_->poisoned; }

FlashMTPForward::FlashMTPForward(metal::MetalBackend &backend, const FlashWeights &weights,
    uint32_t capacity, uint32_t maximumRows)
    : impl_(std::make_unique<Impl>(backend, weights, capacity, maximumRows)) {}
FlashMTPForward::~FlashMTPForward() = default;
FlashMTPForward::FlashMTPForward(FlashMTPForward &&) noexcept = default;
FlashMTPForward &FlashMTPForward::operator=(FlashMTPForward &&) noexcept = default;
uint64_t FlashMTPForward::workspaceBytes() const noexcept { return impl_ ? impl_->workspaceBytes : 0; }
const char *FlashMTPForward::attentionRouteSemantics() const noexcept {
  return impl_ ? impl_->routeMetadata.c_str() : "mtp-qsa-uninitialized";
}
const char *FlashMTPForward::projectionRouteSemantics() const noexcept {
  return !impl_ ? "mtp-proposal-uninitialized" : impl_->qmvF32Proposals
      ? kFlashMTPQMVF32ProposalSemantics
      : impl_->cacheDense ? "mtp-proposal-raw-smallrows-bf16-cache-large-v1"
                          : "mtp-proposal-raw-original-coefficients-v1";
}

metal::MetalBackend &FlashMTPForward::batchBackend() const {
  if (!impl_) throw std::logic_error("Flash MTP head is not initialized");
  return impl_->backend;
}
const FlashWeights &FlashMTPForward::batchWeights() const {
  if (!impl_) throw std::logic_error("Flash MTP head is not initialized");
  return impl_->weights;
}
uint32_t FlashMTPForward::batchCapacity() const {
  if (!impl_) throw std::logic_error("Flash MTP head is not initialized");
  return impl_->capacity;
}
std::mutex &FlashMTPForward::batchMutex() const {
  if (!impl_) throw std::logic_error("Flash MTP head is not initialized");
  return impl_->mutex;
}
const FlashDenseCache *FlashMTPForward::batchDenseCache() const {
  if (!impl_) throw std::logic_error("Flash MTP head is not initialized");
  return impl_->denseCache.get();
}
std::vector<metal::MetalBuffer> FlashMTPForward::cachedOperandsOnly() const {
  return impl_ && impl_->denseCache ? impl_->denseCache->persistedWeightBuffers()
                                  : std::vector<metal::MetalBuffer>{};
}
bool FlashMTPForward::batchFuseHC() const {
  if (!impl_) throw std::logic_error("Flash MTP head is not initialized");
  return impl_->fuseHC;
}
bool FlashMTPForward::batchQSAF32() const {
  if (!impl_) throw std::logic_error("Flash MTP head is not initialized");
  return impl_->attentionPolicy.f32;
}
bool FlashMTPForward::batchQSAMPP() const {
  if (!impl_) throw std::logic_error("Flash MTP head is not initialized");
  return impl_->attentionPolicy.mpp;
}
void FlashMTPForward::batchAddQSA(metal::CommandGraph &graph,
    const FlashQSAFastInputs &inputs, FlashQSAState &state,
    FlashQSAWorkspace &workspace, FlashQSAFastWorkspace &fastWorkspace,
    uint32_t begin, uint32_t rows) {
  if (!impl_) throw std::logic_error("Flash MTP head is not initialized");
  addHeadQSA(graph, inputs, state, workspace, fastWorkspace, begin, rows,
      impl_->attentionPolicy);
}
bool FlashMTPForward::ownsState(const FlashMTPState &state) const noexcept {
  return impl_ && state.impl_ && state.impl_->owner == impl_->owner &&
      state.impl_->qsa.capacity == impl_->capacity && !state.impl_->poisoned;
}
void FlashMTPForward::batchProject(metal::CommandGraph &graph, const std::string &prefix,
    const metal::MetalBuffer &input, const metal::MetalBuffer &output,
    const metal::MetalBuffer &diagnostics, uint32_t rows, uint32_t logicalRows, bool allowQmvProposals) {
  if (!impl_) throw std::logic_error("Flash MTP head is not initialized");
  impl_->project(graph, prefix, input, output, diagnostics, rows, logicalRows, allowQmvProposals);
}

uint64_t FlashMTPForward::denseCachePlannedBytes(const FlashWeights &weights) {
  const auto prefixes = denseHeadPrefixes();
  return FlashDenseCache::plannedBytes(weights, prefixes);
}
uint64_t FlashMTPForward::attentionWorkspacePlannedBytes(uint32_t maximumRows) {
  if (!maximumRows || maximumRows > 128)
    throw std::invalid_argument("Flash MTP attention admission rows must be 1..128");
  const AttentionPolicy policy;
  if (!policy.f32) return 0;
  if (policy.mpp) {
    // Match the online MPP allocator's two buffers and retain the admission
    // ledger's conservative 16 KiB rounding for each allocation.
    const uint64_t logicalBytes = qsaOnlineMPPWorkspacePlannedBytes(maximumRows, 32);
    const uint64_t statistics = uint64_t{maximumRows} * 24 * 32 * 2 * 4;
    return rounded(statistics) + rounded(logicalBytes - statistics);
  }
  const uint64_t groups = uint64_t{maximumRows} * 24 * policy.maximumPartitions();
  return rounded(groups * 2 * 4) + rounded(groups * 256 * 4);
}

uint64_t FlashMTPForward::workspacePlannedBytes(uint32_t capacity, uint32_t maximumRows) {
  validateCapacity(capacity);
  if (!maximumRows || maximumRows > 128)
    throw std::invalid_argument("Flash MTP workspace admission rows must be 1..128");
  // Exact original head scratch shapes; optional dense-cache storage is
  // separately returned by denseCachePlannedBytes() for the active policy.
  constexpr std::array<uint32_t, 30> widths{
      kWidth, kWidth, kWidth, kHyper, kHyper, kHyper, kHyper, 320, kHyper,
      4, 4, kWidth, kWidth, 12288, 512, 512, 640, 6144, 512, kSelections,
      kSelections * 640, kSelections * 640, kSelections * 640,
      kSelections * kWidth, 640, 640, 640, kWidth, 1, 248320};
  uint64_t bytes = 0;
  for (uint32_t width : widths) bytes += rounded(uint64_t{maximumRows} * width * 2);
  for (uint64_t extent : std::array<uint64_t, 3>{
      uint64_t{maximumRows} * 8, uint64_t{maximumRows} * kSelections * 8, 4})
    bytes += rounded(extent);
  const uint64_t blocks = (uint64_t{capacity} + 3) / 4;
  for (uint64_t width : std::array<uint64_t, 6>{
      6144 * 2, 512 * 2, blocks * 4, 512 * 4,
      uint64_t{24} * kFlashQSATokenWidth * 4,
      uint64_t{24} * kFlashQSATokenWidth * 2})
    bytes += rounded(uint64_t{maximumRows} * width);
  bytes += attentionWorkspacePlannedBytes(maximumRows);
  if (enabled("SPLASH_FLASH_GPU_GREEDY"))
    bytes += greedyGPUWorkspacePlannedBytes(
        std::min(maximumRows, uint32_t{kFlashGreedyGPUMaximumRows}), 248320);
  return bytes;
}

uint64_t FlashMTPForward::requestStateBytes(uint32_t capacity) {
  validateCapacity(capacity);
  const uint64_t blocks = (uint64_t{capacity} + 3) / 4;
  return 2 * rounded(uint64_t{capacity} * 512 * 2) +
      rounded(uint64_t{capacity} * 128 * 2) + rounded(blocks * 128 * 2) +
      rounded(uint64_t{capacity} * 8);
}

FlashMTPState FlashMTPForward::createState() {
  if (!impl_) throw std::logic_error("Flash MTP is not initialized");
  std::lock_guard lock(impl_->mutex);
  FlashMTPState result;
  result.impl_ = std::make_unique<FlashMTPState::Impl>();
  result.impl_->owner = impl_->owner;
  result.impl_->qsa = allocateQSAState(impl_->backend, impl_->capacity);
  for (const auto &buffer : {result.impl_->qsa.keys, result.impl_->qsa.values,
      result.impl_->qsa.rawIndexKeys, result.impl_->qsa.pooledKeys,
      result.impl_->qsa.indexPositions}) zero(buffer);
  return result;
}

void FlashMTPForward::truncate(FlashMTPState &request, uint64_t retainedLength) {
  if (!impl_) throw std::logic_error("Flash MTP is not initialized");
  std::lock_guard lock(impl_->mutex);
  if (!request.impl_ || request.impl_->owner != impl_->owner ||
      request.impl_->poisoned || retainedLength > request.impl_->length)
    throw std::invalid_argument("Flash MTP truncate requires its healthy state and a retained prefix");
  request.impl_->length = retainedLength;
}

FlashMTPResult FlashMTPForward::forward(FlashMTPState &request,
    metal::MetalBuffer previousHidden, std::span<const uint32_t> nextTokens,
    FlashMTPLogits mode) {
  return forwardImpl(request, std::move(previousHidden), nextTokens, mode, false);
}

metal::CommandTiming FlashMTPForward::primeTeacherCache(FlashMTPState &request,
    metal::MetalBuffer previousHidden, std::span<const uint32_t> nextTokens) {
  return forwardImpl(request, std::move(previousHidden), nextTokens,
      FlashMTPLogits::None, true).timing;
}

FlashMTPResult FlashMTPForward::forwardImpl(FlashMTPState &request,
    metal::MetalBuffer previousHidden, std::span<const uint32_t> nextTokens,
    FlashMTPLogits mode, bool teacherCacheOnly) {
  if (!impl_) throw std::logic_error("Flash MTP is not initialized");
  std::lock_guard lock(impl_->mutex);
  if (!request.impl_ || request.impl_->owner != impl_->owner || request.impl_->poisoned)
    throw std::invalid_argument("Flash MTP requires its own healthy request state");
  auto &state = *request.impl_;
  if (nextTokens.empty() || nextTokens.size() > impl_->maximumRows ||
      state.length > state.qsa.capacity || nextTokens.size() > state.qsa.capacity - state.length)
    throw std::invalid_argument("Flash MTP pair window exceeds rows/context capacity");
  if (mode != FlashMTPLogits::None && mode != FlashMTPLogits::Last && mode != FlashMTPLogits::All)
    throw std::invalid_argument("Flash MTP logits mode is invalid");
  const uint32_t rows = static_cast<uint32_t>(nextTokens.size());
  if (!previousHidden || previousHidden.sizeBytes() < uint64_t{rows} * kHyper * 2)
    throw std::invalid_argument("Flash MTP requires BF16[rows,10240] premixer hidden");
  const auto tokens = impl_->view(Scratch::TokenIDs, uint64_t{rows} * 8);
  auto *hostTokens = static_cast<int64_t *>(tokens.contents());
  for (uint32_t row = 0; row < rows; ++row) {
    if (nextTokens[row] >= impl_->descriptor.vocabularySize)
      throw std::invalid_argument("Flash MTP next token is out of vocabulary");
    hostTokens[row] = nextTokens[row];
  }
  const auto diag = impl_->view(Scratch::Diagnostics, 4);
  zero(diag);
  const auto bf = [&](Scratch slot, uint32_t width) { return impl_->bf(slot, rows, width); };
  const auto affine = [&](metal::CommandGraph &graph, const std::string &prefix,
      const metal::MetalBuffer &input, const metal::MetalBuffer &output) {
    impl_->project(graph, prefix, input, output, diag, rows);
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
  FlashQSAFastInputs attentionInputs;
  attentionInputs.qProjection = q; attentionInputs.kProjection = k;
  attentionInputs.vProjection = v; attentionInputs.indexProjection = index;
  attentionInputs.qNorm = &impl_->weights.tensor(qNorm);
  attentionInputs.kNorm = &impl_->weights.tensor(kNorm);
  attentionInputs.indexQNorm = &impl_->weights.tensor(iqNorm);
  attentionInputs.indexKNorm = &impl_->weights.tensor(ikNorm);
  attentionInputs.output = attentionOutput; attentionInputs.diagnostics = diag;
  attentionInputs.qConvention = impl_->weights.normConvention(qNorm);
  attentionInputs.kConvention = impl_->weights.normConvention(kNorm);
  attentionInputs.indexQConvention = impl_->weights.normConvention(iqNorm);
  attentionInputs.indexKConvention = impl_->weights.normConvention(ikNorm);
  attentionInputs.epsilon = impl_->descriptor.normEpsilon;
  attentionInputs.theta = impl_->descriptor.rotaryTheta;
  if (teacherCacheOnly) {
    addTeacherQSACache(graph, attentionInputs, state.qsa, impl_->qsaWorkspace,
        impl_->qsaFastWorkspace, static_cast<uint32_t>(state.length), rows,
        impl_->attentionPolicy);
    metal::CommandTiming timing;
    try {
      timing = impl_->backend.submitCommand(graph.dispatches());
      uint32_t status = 0;
      std::memcpy(&status, diag.contents(), sizeof(status));
      if (status) throw std::runtime_error("Flash teacher cache sticky diagnostics failed: " + std::to_string(status));
    } catch (...) {
      state.poisoned = true;
      throw;
    }
    state.length += rows;
    return {timing, {}, 0, {}, 0, state.length, {}, 0};
  }
  addHeadQSA(graph, attentionInputs, state.qsa, impl_->qsaWorkspace,
      impl_->qsaFastWorkspace, static_cast<uint32_t>(state.length), rows,
      impl_->attentionPolicy);
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
  if (impl_->denseCache && rows >= 16)
    addDenseBF16WholeK(impl_->backend, graph, mixed, impl_->weights.tensor(mlp + ".gate.weight"),
        bf(Scratch::Router, 512), diag, rows, FlashAffineMPPTile::M16N64);
  else
    addDenseBF16(graph, mixed, impl_->weights.tensor(mlp + ".gate.weight"), bf(Scratch::Router, 512), diag, rows);
  addRoute(graph, bf(Scratch::Router, 512), ids, route, diag, rows, 512, kSelections);
  addGatheredAffine(graph, mixed, impl_->weights.projection(mlp + ".switch_mlp.gate_proj"),
      ids, bf(Scratch::ExpertGate, kSelections * 640), diag, rows, kSelections);
  addGatheredAffine(graph, mixed, impl_->weights.projection(mlp + ".switch_mlp.up_proj"),
      ids, bf(Scratch::ExpertUp, kSelections * 640), diag, rows, kSelections);
  addSiLUMultiply(graph, bf(Scratch::ExpertGate, kSelections * 640), bf(Scratch::ExpertUp, kSelections * 640),
      bf(Scratch::ExpertIntermediate, kSelections * 640), diag, rows, 640, kSelections);
  addGatheredAffine(graph, bf(Scratch::ExpertIntermediate, kSelections * 640),
      impl_->weights.projection(mlp + ".switch_mlp.down_proj"), ids,
      bf(Scratch::ExpertDown, kSelections * kWidth), diag, rows, kSelections, true);
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
  metal::MetalBuffer greedyResults;
  uint32_t greedyRows = 0;
  if (mode != FlashMTPLogits::None) {
    logitRows = mode == FlashMTPLogits::All ? rows : 1;
    // The final mixer has no state. Last needs only its last row, while the
    // returned pre-mixer hidden still contains every folded history pair.
    impl_->hc(graph, "mtp.hyper_connection_mixer", logitRows, false, diag,
        mode == FlashMTPLogits::All ? 0 : rows - 1, finalNormalizedReady);
    const auto headInput = impl_->bf(Scratch::Mixed, logitRows, kWidth);
    logits = impl_->bf(Scratch::HeadLogits, logitRows, impl_->descriptor.vocabularySize);
    addAffine(graph, headInput, impl_->weights.projection("language_model.lm_head"), logits, diag, logitRows);
    if (impl_->gpuGreedy && logitRows <= kFlashGreedyGPUMaximumRows) {
      greedyResults = impl_->backend.view(impl_->greedyResults, 0,
          uint64_t{logitRows} * sizeof(FlashGreedyGPURowResult));
      addGreedyGPU(graph, logits, impl_->greedyWorkspace, greedyResults,
          logitRows, impl_->descriptor.vocabularySize);
      greedyRows = logitRows;
    }
  }
  metal::CommandTiming timing;
  try {
    timing = impl_->backend.submitCommand(graph.dispatches());
    uint32_t status = 0;
    std::memcpy(&status, diag.contents(), sizeof(status));
    if (status) throw std::runtime_error("Flash MTP sticky diagnostics failed: " + std::to_string(status));
  } catch (...) {
    state.poisoned = true;
    throw;
  }
  state.length += rows;
  return {timing, logits, logitRows, hyper, rows, state.length, greedyResults, greedyRows};
}

} // namespace splash::flash
