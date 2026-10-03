#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/Sampling.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// Greedy walks the highest-scoring candidate at each position. Lookahead
// maximizes expected accepted draft length under the selector probabilities.
// Sampling lanes draw ancestrally in either mode; verification is unchanged.
enum class DraftSelection : uint8_t { Greedy, Lookahead };

struct DraftSelectorWorkspace final {
  uint64_t partialIdsBytes = 0;
  uint64_t partialValuesBytes = 0;
  uint64_t candidatesBytes = 0;
  uint64_t unaryBytes = 0;
  uint64_t proposalProbabilitiesBytes = 0;
};

struct DraftSelectorBuffers final {
  // fp32 [rows][vocabulary].
  metal::MetalBuffer logits;
  metal::MetalBuffer partialIds;
  metal::MetalBuffer partialValues;
  metal::MetalBuffer candidates;
  metal::MetalBuffer unary;
  metal::MetalBuffer selectorHidden;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer proposedTokens;
  metal::MetalBuffer proposalProbabilities;
};

// The draft's selector codebooks, which score the edge from a proposal
// position's predecessor candidate to each of its candidates.
struct DraftCodebooks final {
  metal::MetalBuffer predecessor;
  metal::MetalBuffer successor;
};

// The DFlash draft's proposal policy (draft_select_* in
// metal/kernels/decode/sampling.metal): each lane keeps the
// SPLASH_DRAFT_CANDIDATES most likely draft tokens of every proposal
// position, scores each candidate with its edge from the previous position's
// choice, and walks the SPLASH_DRAFT_PROPOSAL_TOKENS positions greedily or
// with lookahead. A sampling lane draws at its temperature in either mode.
class DraftSelector final {
public:
  explicit DraftSelector(uint32_t vocabulary,
                         DraftSelection selection = DraftSelection::Greedy);

  [[nodiscard]] DraftSelection draftSelection() const noexcept {
    return selection_;
  }

  // Exact scratch/output bytes for that many proposal positions.
  [[nodiscard]] static DraftSelectorWorkspace workspace(uint32_t positions);

  void add(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers,
           const DraftCodebooks &codebooks, std::span<const uint32_t> anchors,
           std::span<const SamplingPolicy> policies) const;

private:
  uint32_t vocabulary_ = 0;
  DraftSelection selection_ = DraftSelection::Greedy;
};

} // namespace splash::ops
