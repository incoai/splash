#pragma once

// Inkling 2 Mini's target (model/Inkling.hpp): its 48 layers over the blocks
// every Qwen target shares (model/QwenTarget.hpp). Its residual is kept in
// fp32 states (InklingSteps: the embedding's normed rows and the streams of
// its depth attention's blocks), from which each layer mixes its inputs: its
// attention's (one, or the query's, key's and value's on a global layer),
// gated on the loop's passes after its first by messages mixed from the pass
// before, each normed into the bf16 rows of its side projection; its
// queries' side, its keys and values after their short convolutions and its
// relative states attend through the relative geometry
// (ops::PagedAttention::addRelativeProjection), whose output projection and
// convolution replace a block's stream or add to it; its MoE's input from
// the blocks and that stream, its output convolved into the stream. The head
// reads the norm of the final depth mix, its logits capped. Each layer keeps
// its K/V and its four convolutions' histories of its own, in run order:
// a prompt carries the histories from chunk to chunk and captures them at
// a cached state's row, a verify writes its rows' inputs to a tape whose
// accepted path its commit writes into the lane's next state.

#include "model/InklingSteps.hpp"
#include "model/QwenTarget.hpp"
#include "ops/LoopControl.hpp"
#include "ops/ShortConvolution.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace splash::model {

class InklingTarget final : public QwenTarget {
public:
  // A target whose prefill commands run up to `prefillChunks` chunks.
  InklingTarget(const InklingWeights &weights, const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
                const ops::ExecutionPlans &operators, uint32_t prefillChunks);

  [[nodiscard]] QwenTargetScratchBytes allocatedScratchBytes() const noexcept override;
  [[nodiscard]] std::vector<metal::MetalBuffer> addPrefill(metal::CommandGraph &graph,
                                                           std::span<const QwenTargetPrefillChunk> chunks,
                                                           const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                           QwenPrefillLayers layers, ops::AneFfn *aneFfn) override;
  // The history kernels a state captured inside a prompt's rows runs, of
  // its convolutions' two input types: layer 0's key and attention output
  // histories after the first row of the first chunk's scratch.
  void addInsideCapture(metal::CommandGraph &graph, metal::MetalBuffer auxiliaryIn,
                        metal::MetalBuffer captured) const override;
  // The hidden rows are the final mix's norm, which the head reads as they
  // are.
  void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer finalHidden,
                    metal::MetalBuffer logits, uint32_t rows, ops::LinearScratch scratch) const override;
  // Each lane's convolution histories of every layer after its accepted
  // path, from the inputs its verify wrote to the tape, in one dispatch.
  void addStateCommit(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers,
                      const ops::StepRows &step) const override;

private:
  // The shared prefill's layer loop runs the family's (QwenTarget::addPrefillLayers).
  friend class QwenTarget;

  // A step's family state: the shared one, its scratch, and its rows as its
  // layout holds them, which its convolutions take.
  struct PrefillStep : QwenTarget::PrefillStep {
    InklingScratch scratch{};
    ops::LayoutRows layoutRows{};
  };
  // The family's prefill layers: the embedding's norm into the embedding
  // state; each layer's attention through its routing (addPrefillLayer), and
  // its routed experts and their convolution (addPrefillExperts); the head's
  // depth mix and norm into buffers.hidden[1], a decode step's into its
  // final hidden, then its head (addPrefillEnd).
  void addPrefillBegin(PrefillStep &step) const;
  [[nodiscard]] bool attends(uint32_t) const noexcept { return true; }
  void addPrefillLayer(PrefillStep &step, uint32_t index) const;
  [[nodiscard]] const ops::BlockMoeWeights *moe(uint32_t index) const;
  void addPrefillExperts(PrefillStep &step, uint32_t index) const;
  [[nodiscard]] metal::MetalBuffer addPrefillEnd(PrefillStep &step) const;

  // Layer `layer`'s attention input of each stream (query, key, value)
  // normed into the bf16 rows its side projection reads: the step's
  // normalized rows, and its key and value inputs where they differ.
  [[nodiscard]] std::array<metal::MetalBuffer, 3> addAttentionInputs(PrefillStep &step, uint32_t layer,
                                                                     const InklingLayerStates &states) const;
  // The convolution of layer `layer` over the step's rows of `buffers`, its
  // input of `shape`: a prompt's from each sequence's history into its next
  // one, and into each state captured inside its rows; a verify's along each
  // row's path, its inputs to the tape.
  void addConvolution(PrefillStep &step, uint32_t layer, InklingConvolution convolution,
                      const ops::ShortConvolutionBuffers &buffers, ops::ShortConvolutionShape shape) const;
  // A state's rows of a step, as the depth attention reads them.
  [[nodiscard]] ops::StridedRows stateRows(const PrefillStep &step, uint32_t state) const;

  const InklingWeights &weights_;
  InklingSteps steps_;
  // Each core weight layer's gates of its query, key, value and FFN inputs,
  // as the loop control reads them.
  std::vector<std::array<ops::LoopControlWeights, InklingLayout::loopMessages>> loopGates_;
};

} // namespace splash::model
