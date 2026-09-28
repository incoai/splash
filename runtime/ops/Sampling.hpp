#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/MetalBackend.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace splash::ops {

// Target sampling keeps this many top candidates per row; requests may not
// ask for a larger top-k.
inline constexpr uint32_t kTargetSamplingCandidates = 32;

struct SamplingPolicy final {
  uint32_t topK = 1;
  float temperature = 1.0F;
  float topP = 1.0F;
  bool constrained = false;
  // ignore_eos: the lane may not select a stop token, so it runs to its
  // output budget.
  bool suppressStop = false;

  [[nodiscard]] bool samples() const noexcept { return temperature > 0.0F; }
};

struct SamplingWorkspace final {
  uint64_t argmaxValuesBytes = 0;
  uint64_t argmaxIndicesBytes = 0;
  uint64_t partialIdsBytes = 0;
  uint64_t partialValuesBytes = 0;
  uint64_t topIdsBytes = 0;
  uint64_t topProbabilitiesBytes = 0;
};

struct DraftSelectorWorkspace final {
  uint64_t partialIdsBytes = 0;
  uint64_t partialValuesBytes = 0;
  uint64_t candidatesBytes = 0;
  uint64_t unaryBytes = 0;
  uint64_t proposalProbabilitiesBytes = 0;
};

struct SamplingBuffers final {
  // fp32 [rows][vocabulary].
  metal::MetalBuffer logits;
  metal::MetalBuffer partialIds;
  metal::MetalBuffer partialValues;
  metal::MetalBuffer topIds;
  metal::MetalBuffer topProbabilities;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer constraintMasks;
  metal::MetalBuffer outputTokens;
  metal::MetalBuffer argmaxValues;
  metal::MetalBuffer argmaxIndices;
};

struct DraftSelectorBuffers final {
  // fp32 [rows][vocabulary].
  metal::MetalBuffer logits;
  metal::MetalBuffer partialIds;
  metal::MetalBuffer partialValues;
  metal::MetalBuffer candidates;
  metal::MetalBuffer unary;
  metal::MetalBuffer selectorHidden;
  metal::MetalBuffer predecessorCodebook;
  metal::MetalBuffer successorCodebook;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer proposedTokens;
  metal::MetalBuffer proposalProbabilities;
};

struct AcceptanceBuffers final {
  metal::MetalBuffer proposedTokens;
  metal::MetalBuffer candidates;
  metal::MetalBuffer proposalProbabilities;
  metal::MetalBuffer targetTopIds;
  metal::MetalBuffer targetTopProbabilities;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer outputTokens;
  metal::MetalBuffer retainedCounts;
  metal::MetalBuffer nextAnchors;
  metal::MetalBuffer acceptedCounts;
};

// Target token policy. This operator owns top-k/top-p, constrained selection
// and greedy argmax pipeline ABIs; the model only supplies policy and buffers.
class Sampling final {
public:
  // rowsPerLane is the kernels' SPLASH_TARGET_VERIFY_ROWS. stopTokens are
  // the ids a suppressStop policy may not select; a Sampling built without
  // them rejects such a policy.
  Sampling(metal::MetalBackend &backend, uint32_t vocabulary,
           uint32_t rowsPerLane,
           std::array<uint32_t, 2> stopTokens = {kNoStopToken, kNoStopToken});

  // Exact scratch/output bytes for the fixed precompiled sampling ABI.
  // Counts may cover one lane or a packed batch; the operator owns sharding.
  [[nodiscard]] static SamplingWorkspace workspace(uint32_t rows);
  [[nodiscard]] static DraftSelectorWorkspace draftWorkspace(uint32_t positions);

  void addInitial(metal::CommandGraph &graph, const SamplingPolicy &policy,
                  SamplingBuffers buffers, uint32_t rowOffset) const;
  void addVerify(metal::CommandGraph &graph,
                 std::span<const SamplingPolicy> policies,
                 SamplingBuffers buffers) const;
  // proposalTokens is the kernels' SPLASH_DRAFT_PROPOSAL_TOKENS.
  void addDraftSelector(
      metal::CommandGraph &graph, DraftSelectorBuffers buffers,
      std::span<const uint32_t> anchors,
      std::span<const SamplingPolicy> policies, uint32_t proposalTokens) const;
  void addAcceptance(
      metal::CommandGraph &graph, AcceptanceBuffers buffers,
      std::span<const uint32_t> maximumRetained,
      std::span<const SamplingPolicy> policies, uint32_t stopToken0,
      uint32_t stopToken1) const;
  void addVerifyInput(metal::CommandGraph &graph,
                      metal::MetalBuffer draftInputTokens,
                      metal::MetalBuffer proposedTokens,
                      metal::MetalBuffer verifyInputTokens,
                      uint32_t lanes) const;

private:
  static constexpr uint32_t kNoStopToken = UINT32_MAX;

  void addStopSuppression(metal::CommandGraph &graph, metal::MetalBuffer logits,
                          uint32_t rowOffset, uint32_t rows,
                          uint32_t laneMask) const;

  metal::MetalBackend &backend_;
  uint32_t vocabulary_ = 0;
  uint32_t rowsPerLane_ = 0;
  uint32_t maskWords_ = 0;
  std::array<uint32_t, 2> stopTokens_{};
};

} // namespace splash::ops
