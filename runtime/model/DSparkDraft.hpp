#pragma once

// DSpark (model/DSpark.hpp) as the runtime decodes with it, beside
// DeepSeek-V4.1-Flash's target (DeepSeek41Target), which captures the mean
// of its four streams at the input of each of DSpark's target layers.
//
// A lane's state is DFlash2's (DraftStateLayout of one KV head of
// LATENT_ATTENTION_DIM values): each layer's ring of the latents of its
// window, the last ring-slot positions, in the first slots of the layer's
// ring keys (its values hold nothing); and the context window, the 4-bit
// rows of DSpark's own view of the target's rows (main_norm(fc(captured)))
// that a cached state keeps and a restore computes the rings from again.
//
// A decode cycle drafts each lane's block, its anchor and the noise rows
// after it (the target's embedding of its mask token, DSpark's noise token):
// its layers on the decode plans of the step's lanes, the target's head, and
// the Markov head's chain with its confidences (dspark_markov_step), a
// proposal each, which the target verifies as a DFlash2 draft's. Its scratch
// is its own (scratchBytes).

#include "model/DFlashDraft.hpp"
#include "model/DSpark.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/ManifoldHyperConnection.hpp"
#include "ops/MoE.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace splash::model {

// What a decode cycle's draft binds of the runtime's arenas.
struct DSparkDecodeBuffers final {
  // bf16 [lanes][SPLASH_DRAFT_QUERY_ROWS][hidden]: each lane's block rows'
  // embeddings, its anchor's first; uint32 the rows' positions.
  metal::MetalBuffer input;
  metal::MetalBuffer positions;
  // fp32 [lanes][SPLASH_DRAFT_QUERY_ROWS][vocabulary]: the rows' logits.
  metal::MetalBuffer logits;
  // Each layer's ring keys of each lane.
  std::vector<std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>> rings;
};

// What a decode cycle's proposals take (ops::DraftSelectorBuffers'): the
// logits addDecode wrote, and the proposals, their confidences and the
// draft distribution acceptance reads.
struct DSparkSelectionBuffers final {
  metal::MetalBuffer logits;
  metal::MetalBuffer proposedTokens;
  metal::MetalBuffer confidence;
  metal::MetalBuffer candidates;
  metal::MetalBuffer proposalProbabilities;
};

// What a decode cycle's context commit binds: the verify's captured rows
// (bf16 [lanes][SPLASH_TARGET_VERIFY_ROWS][target layers * hidden]), the
// rows their projection and norm pass through, the rows' positions, each
// lane's retained rows and its rings.
struct DSparkCommitBuffers final {
  metal::MetalBuffer captured;
  metal::MetalBuffer projected;
  metal::MetalBuffer hidden;
  metal::MetalBuffer positions;
  metal::MetalBuffer retainedCounts;
  std::vector<std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>> rings;
};

class DSparkDraft final {
public:
  // DSpark of `weights` beside a target whose head is `head`.
  DSparkDraft(const DSparkWeights &weights, const ops::Projection &head, metal::MetalBackend &backend,
              const ops::ExecutionPlans &operators);
  ~DSparkDraft();
  DSparkDraft(const DSparkDraft &) = delete;
  DSparkDraft &operator=(const DSparkDraft &) = delete;

  // The scratch a DSpark of `weights` allocates beside a target whose head
  // is `head`.
  [[nodiscard]] static uint64_t scratchBytes(const DSparkWeights &weights, const ops::Projection &head,
                                             const ops::ExecutionPlans &operators);
  [[nodiscard]] uint64_t allocatedScratchBytes() const noexcept { return allocated_; }

  // The context of `rows` captured rows (buffers.capturedTargetHidden) at
  // positions (uint32 [rows]): DSpark's view of each, into each span's
  // window and the windows of the states the command captures, and its
  // layers' latents of each into each span's rings (DFlashDraft's
  // addContextPrefill).
  void addContextPrefill(metal::CommandGraph &graph, const DFlashPrefillBuffers &buffers,
                         const metal::MetalBuffer &positions, uint32_t rows, std::span<const DFlashPrefillSpan> spans,
                         std::span<const DFlashWindowCapture> windowCaptures = {}) const;
  // The rings of lane `lane` from its context window's rows of the `rows`
  // positions from startPosition on, the last ring-slot of them.
  void addContextRebuild(metal::CommandGraph &graph, const DFlashPrefillBuffers &buffers,
                         const metal::MetalBuffer &window, std::span<const DFlashDraftRingLayer> ring, uint32_t rows,
                         uint32_t startPosition, uint32_t lane) const;
  // Each lane's rings after its retained rows of a verify, from position
  // startPositions[lane] on.
  void addContextCommit(metal::CommandGraph &graph, const DSparkCommitBuffers &buffers,
                        std::span<const uint32_t> startPositions) const;

  // Lanes of storage its projections bind of the arenas in a decode step or
  // context commit of `lanes` lanes (ops::Linear::decodeStorageRows).
  [[nodiscard]] uint32_t decodeStorageLanes(uint32_t lanes) const;
  // Each lane's block through the layers and the target's head, its first
  // cacheLengths[lane] positions in its rings.
  void addDecode(metal::CommandGraph &graph, const DSparkDecodeBuffers &buffers,
                 std::span<const uint32_t> cacheLengths) const;
  // After addDecode: each lane's proposals from its anchor.
  void addSelection(metal::CommandGraph &graph, const DSparkSelectionBuffers &buffers,
                    std::span<const uint32_t> anchors) const;

  // A lane's ring keys hold the ring of `ringSlots` latents in their first
  // slots.
  [[nodiscard]] uint32_t ringSlots() const noexcept { return weights_.layout.layers.slidingWindow; }

private:
  struct Scratch;

  // Each layer's latents of the `rows` rows of DSpark's view in `hidden` at
  // `positions`, into each span's rings.
  void addContextLatents(metal::CommandGraph &graph, const metal::MetalBuffer &hidden,
                         const metal::MetalBuffer &positions, uint32_t rows,
                         std::span<const DFlashPrefillSpan> spans) const;
  // A projection of a decode step's rows on the decode plans of `lanes`.
  void project(metal::CommandGraph &graph, const metal::MetalBuffer &input, const ops::Projection &projection,
               const metal::MetalBuffer &output, uint32_t lanes) const;
  [[nodiscard]] metal::MetalBuffer ring(const metal::MetalBuffer &keys) const;

  const DSparkWeights &weights_;
  const ops::Projection &head_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
  ops::ManifoldHyperShape mhc_{};
  ops::MoeShape moeShape_{};
  std::unique_ptr<Scratch> scratch_;
  uint64_t allocated_ = 0;
};

} // namespace splash::model
