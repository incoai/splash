#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/MetalBackend.hpp"

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
  // The lane ignores end-of-sequence: the target never selects a stop token,
  // though the draft may still propose one.
  bool excludesStopTokens = false;

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

// How a greedy lane picks its proposals from the selector's candidates and
// edge tables. Greedy takes the best-scoring candidate position by position.
// Lookahead takes the path with the highest expected accepted length under
// the selector's own conditional probabilities. Both use the same buffers and
// workspace, and sampling lanes draw ancestrally under either. Verification
// is unchanged, so the emitted tokens are the same; only acceptance differs.
enum class DraftSelection : uint8_t { Greedy, Lookahead };

// Target token policy. This operator owns top-k/top-p, constrained selection,
// stop-token exclusion and greedy argmax pipeline ABIs; the model only
// supplies policy, buffers and its stop tokens.
class Sampling final {
public:
  // rowsPerLane is the kernels' SPLASH_TARGET_VERIFY_ROWS.
  Sampling(metal::MetalBackend &backend, uint32_t vocabulary,
           uint32_t rowsPerLane);

  // Exact scratch/output bytes for the fixed precompiled sampling ABI.
  // Counts may cover one lane or a packed batch; the operator owns sharding.
  [[nodiscard]] static SamplingWorkspace workspace(uint32_t rows);
  [[nodiscard]] static DraftSelectorWorkspace draftWorkspace(uint32_t positions);

  void addInitial(metal::CommandGraph &graph, const SamplingPolicy &policy,
                  SamplingBuffers buffers, uint32_t rowOffset,
                  uint32_t stopToken0, uint32_t stopToken1) const;
  void addVerify(metal::CommandGraph &graph,
                 std::span<const SamplingPolicy> policies,
                 SamplingBuffers buffers, uint32_t stopToken0,
                 uint32_t stopToken1) const;
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

  // Applies to graphs encoded after the call. Greedy is the shipped default.
  void setDraftSelection(DraftSelection selection) noexcept {
    draftSelection_ = selection;
  }
  [[nodiscard]] DraftSelection draftSelection() const noexcept {
    return draftSelection_;
  }

private:
  metal::MetalBackend &backend_;
  uint32_t vocabulary_ = 0;
  uint32_t rowsPerLane_ = 0;
  uint32_t maskWords_ = 0;
  DraftSelection draftSelection_ = DraftSelection::Greedy;
};

} // namespace splash::ops
