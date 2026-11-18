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
// its layers over the lanes' blocks, the target's head, and the Markov head's
// chain with its confidences (dspark_markov_step), a proposal each, which is
// the lane's tree, a chain the target verifies (dspark_tree). Its scratch is
// its own (scratchBytes).

#include "model/DFlashDraft.hpp"
#include "model/DSpark.hpp"
#include "ops/DraftSelector.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/LaneRows.hpp"
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
  // embeddings, its anchor's first.
  metal::MetalBuffer input;
  // fp32 [lanes][SPLASH_DRAFT_QUERY_ROWS][vocabulary]: the rows' logits.
  metal::MetalBuffer logits;
  // Each layer's ring keys of each lane.
  std::vector<std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>> rings;
};

// What a decode cycle's context commit binds: the step's captured rows,
// each lane's from row lane * deepSeek41LaneStride(step) on (bf16 [rows]
// [target layers * hidden], DeepSeek41Target's layout), the rows their
// projection and norm pass through, each lane's retained rows (a chain's row
// mask, QwenTargetCommitBuffers::retainedRows) and its rings.
struct DSparkCommitBuffers final {
  metal::MetalBuffer captured;
  metal::MetalBuffer projected;
  metal::MetalBuffer hidden;
  metal::MetalBuffer retainedRows;
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
  // Each lane's rings after its retained rows of a decode step `step`, from
  // position startPositions[lane] on (the positions addDecode wrote).
  void addContextCommit(metal::CommandGraph &graph, const DSparkCommitBuffers &buffers,
                        std::span<const uint32_t> startPositions, const ops::StepRows &step) const;

  // Lanes of storage of SPLASH_DRAFT_QUERY_ROWS its projections bind of the
  // arenas in a decode step of `lanes` lanes (ops::Linear::storageRows).
  [[nodiscard]] uint32_t decodeStorageLanes(uint32_t lanes) const;
  // Each lane's block through the layers and the target's head, its first
  // cacheLengths[lane] positions in its rings; it writes the positions of
  // its blocks' rows, and of the commit's rows of the cycle's step `step`.
  void addDecode(metal::CommandGraph &graph, const DSparkDecodeBuffers &buffers,
                 std::span<const uint32_t> cacheLengths, const ops::StepRows &step) const;
  // After addDecode: each lane's tree (`lanes`, each an anchor, its depth,
  // and the calibration of its draft rows' estimates; DSparkTreeParams) in
  // the selector's tree buffers: the chain of its proposals, or where
  // `children` is none the chain of its first `chain` host chain tokens
  // (`tree.chainTokens` and `chainRates`), which a grammar or a thinking end
  // forces or a lookup proposes in its place.
  void addSelection(metal::CommandGraph &graph, const ops::DraftSelectorBuffers &tree,
                    std::span<const ops::DraftTreeLane> lanes) const;

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
  uint32_t moeRouterFormat_ = ops::QuantizedSegment::kFloat32;
  std::unique_ptr<Scratch> scratch_;
  uint64_t allocated_ = 0;
};

} // namespace splash::model
