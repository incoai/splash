#pragma once

// Nemotron-H's target (model/NemotronH.hpp): its single-block layers, each
// behind its own RMS norm (epsilon 1e-5), added to one residual. Its Mamba-2
// state lives in the runtime's GDN state cells (GdnStateLayout of Mamba-2
// geometry) and folds a step's retained rows from its tape as GDN's does; it
// has no auxiliary state. Its blocks run through ops::Mamba2,
// the Qwen families' attention mixers and ops::MoE, on the runtime's arena and
// the target's own decode scratch (scratchBytes). A prefill and a decode step
// run layer by layer as the Qwen families' do; a prompt's Mamba-2 blocks run
// each sequence's rows a verify block at a time, between two one-layer state
// cells of its own, and fold the last rows into the sequence's state at its
// end. Its decode verifies chains alone, and captures no state in decode
// (Runtime's capturesInDecode); its cached states are exact.

#include "model/NemotronH.hpp"
#include "model/QwenTarget.hpp"

namespace splash::model {

class NemotronHTarget final : public QwenTarget {
public:
  NemotronHTarget(const NemotronHWeights &weights, const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
                  const ops::ExecutionPlans &operators);

  // The decode and prefill scratch the constructor allocates.
  [[nodiscard]] static QwenTargetScratchBytes scratchBytes(const QwenTargetGeometry &geometry);
  [[nodiscard]] QwenTargetScratchBytes allocatedScratchBytes() const noexcept override;

  // A prompt chunk's or a decode step's layers (QwenTarget::addPrefillLayers):
  // a decode step's Mamba-2 blocks scan each lane's chain of rows from its
  // state (ops::Mamba2::addDecode), a prompt sequence's run its rows in
  // slices (addPrefillMamba).
  [[nodiscard]] std::vector<metal::MetalBuffer> addPrefill(metal::CommandGraph &graph,
                                                           std::span<const QwenTargetPrefillChunk> chunks,
                                                           const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                           QwenPrefillLayers layers, ops::AneFfn *aneFfn) override;
  void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer finalHidden,
                    metal::MetalBuffer logits, uint32_t rows, ops::LinearScratch scratch) const override;

private:
  friend class QwenTarget;

  // Rows a block's work runs on: a decode step's (a full batch's verify
  // rows) or a prefill chunk's and the slices past it.
  struct Rows final {
    metal::MetalBuffer packed, y, gated, latent, moeLatent, sharedMid;
  };
  // A prefill's two one-layer state cells and its one-layer tape of two slots.
  struct PrefillStates final {
    std::array<metal::MetalBuffer, 2> convolution, recurrent;
    metal::MetalBuffer tape;
  };

  // The shared layer loop's (QwenTarget::addPrefillLayers).
  void addPrefillBegin(PrefillStep &) const {}
  [[nodiscard]] bool attends(uint32_t index) const;
  void addPrefillLayer(PrefillStep &step, uint32_t index) const;
  [[nodiscard]] const ops::BlockMoeWeights *moe(uint32_t index) const;
  void addPrefillExperts(PrefillStep &step, uint32_t index) const;
  [[nodiscard]] metal::MetalBuffer addPrefillEnd(PrefillStep &step) const;

  [[nodiscard]] const Rows &rows(const PrefillStep &step) const { return step.decode ? decodeRows_ : prefillRows_; }
  void addProjection(PrefillStep &step, const ops::Projection &projection, metal::MetalBuffer input,
                     metal::MetalBuffer output, metal::MetalBuffer residual = {}) const;
  void addBlock(PrefillStep &step, const NemotronMambaWeights &block, metal::MetalBuffer input,
                metal::MetalBuffer output) const;
  void addBlock(PrefillStep &step, const QwenAttentionWeights &block, metal::MetalBuffer input,
                metal::MetalBuffer output) const;
  void addBlock(PrefillStep &step, const NemotronMoeWeights &block, metal::MetalBuffer input,
                metal::MetalBuffer output) const;
  void addPrefillMamba(PrefillStep &step, const NemotronMambaWeights &block, uint32_t layer,
                       const QwenTargetPrefillSequence &sequence) const;

  const NemotronHWeights &weights_;
  metal::MetalBuffer decode_, prefill_;
  Rows decodeRows_, prefillRows_;
  PrefillStates prefillStates_;
};

} // namespace splash::model
