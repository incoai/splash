#pragma once

// Inkling-2-Mini's target (model/Inkling2Loop.hpp): the interface the
// runtime drives (model/QwenTarget.hpp) over the family's own operators. A
// step runs the layers its schedule orders, one per application of a weight
// layer (Inkling2LoopSchedule): each mixes the block stack and its input rows
// into its attention's inputs (ops/BlockResidualMix.hpp), gated by the pass
// before's messages in the loop's later passes (ops/MessageGate.hpp); attends
// with its relative bias, row sinks and gate (ops/RelativeAttention.hpp);
// convolves its keys, values and output (ops/ShortConvolution.hpp); mixes the
// stack and its residual into its MoE's input; and convolves the MoE's output
// onto its residual into its output rows. The head reads the output norm of
// the final mix and finishes its logits (ops/LogitFinish.hpp). Its DFlash2
// draft reads the rows passed into its captured applications and the final
// norm's, as its producer captured them, and its token rows through the
// table's norm.
// Its step scratch is Inkling2LoopSteps'; its convolutions' state is the
// auxiliary state of a lane's cell, which a decode step's commit writes.

#include "model/Inkling2LoopSteps.hpp"
#include "model/QwenTarget.hpp"
#include "ops/RelativeAttention.hpp"
#include "ops/ShortConvolution.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace splash::model {

class Inkling2LoopTarget final : public QwenTarget {
public:
  // A target whose prefill commands run up to `prefillChunks` chunks.
  Inkling2LoopTarget(const Inkling2LoopWeights &weights, const QwenTargetGeometry &geometry,
                     metal::MetalBackend &backend, const ops::ExecutionPlans &operators, uint32_t prefillChunks);

  [[nodiscard]] QwenTargetScratchBytes allocatedScratchBytes() const noexcept override;
  [[nodiscard]] std::vector<metal::MetalBuffer> addPrefill(metal::CommandGraph &graph,
                                                           std::span<const QwenTargetPrefillChunk> chunks,
                                                           const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                           QwenPrefillLayers layers, ops::AneFfn *aneFfn) override;
  // The hidden rows are the output norm of the final mix, which the head
  // reads as they are; its logits take the family's finish.
  void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer finalHidden,
                    metal::MetalBuffer logits, uint32_t rows, ops::LinearScratch scratch) const override;
  // The token table's rows through its norm, as the first application reads
  // them (addPrefillBegin), in place.
  void addDraftEmbedding(metal::CommandGraph &graph, metal::MetalBuffer tokens, metal::MetalBuffer hidden,
                         uint32_t rows) const override;
  // Each lane's convolution state after its accepted path, from the inputs
  // its verify wrote of every application's convolutions, in one dispatch.
  void addStateCommit(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers,
                      const ops::StepRows &step) const override;
  // The relative attention's commit of the K/V it stores.
  void addKvPathCommit(metal::CommandGraph &graph, kv::Layout layout, uint32_t extentPages,
                       std::span<const metal::MetalBuffer> pageTables, std::span<const uint32_t> committed,
                       uint32_t laneRows, const metal::MetalBuffer &paths) const override;

private:
  // The shared prefill's layer loop runs the family's
  // (QwenTarget::addPrefillLayers), a "layer" of it an application.
  friend class QwenTarget;

  // A step's family state: the shared one, its scratch, its relative
  // attention's plans (the full layers' and the sliding layers'), and its rows
  // as the convolutions take them: a prompt chunk's sequences', or a decode
  // step's lanes' along their trees.
  struct PrefillStep : QwenTarget::PrefillStep {
    Inkling2LoopScratch scratch{};
    std::optional<ops::RelativeAttentionPlan> fullAttention{};
    std::optional<ops::RelativeAttentionPlan> slidingAttention{};
    ops::LayoutRows layoutRows{};
  };

  // With the first application, the embedding's rows through the token
  // table's norm into the stack's first plane, the first application's input.
  void addPrefillBegin(PrefillStep &step) const;
  // Every application attends, with its own KV layer.
  [[nodiscard]] bool attends(uint32_t) const { return true; }
  // An application's mixer and its FFN's mix, norm and routing, after the
  // draft's capture of the rows it reads.
  void addPrefillLayer(PrefillStep &step, uint32_t application) const;
  // The MoE block of the application's weight layer, which every pass of a
  // loop layer loads again.
  [[nodiscard]] const ops::BlockMoeWeights *moe(uint32_t application) const;
  // Its routed work and the FFN's convolution onto the residual into its
  // output rows.
  void addPrefillExperts(PrefillStep &step, uint32_t application) const;
  // The final mix and output norm into buffers.hidden[1], a decode step's
  // into its final hidden rows, which the draft captures at id
  // applications() and its head and finish then read.
  [[nodiscard]] metal::MetalBuffer addPrefillEnd(PrefillStep &step) const;

  // The bf16 rows a schedule entry names in a step.
  [[nodiscard]] metal::MetalBuffer rows(const PrefillStep &step, Inkling2LoopRows where) const;
  // Before a pass restarts: the block states the pass before pushed after
  // the loop's first, into the pass before's planes beside its output.
  void addRestart(const PrefillStep &step) const;
  // The relative attention's plans of a step's sequences.
  void planRelativeAttention(PrefillStep &step) const;
  // The application's q, k and v inputs, normalized: q into
  // buffers.normalized, k and v into the scratch's, or the q input's own
  // rows where the three are one row.
  [[nodiscard]] std::array<metal::MetalBuffer, 3> addMixerInputs(PrefillStep &step, uint32_t application,
                                                                metal::MetalBuffer input) const;
  // The application's attention of its normalized inputs into
  // buffers.attentionOutput, its output projection's rows.
  void addAttention(PrefillStep &step, uint32_t application, const std::array<metal::MetalBuffer, 3> &inputs) const;
  // The message gate `message` (0 to 3: q, k, v, FFN) of the application's
  // fp32 mix rows of `queries` queries, query `column`'s, by its message, into
  // `output` through `norm`.
  void addGate(const PrefillStep &step, uint32_t application, uint32_t queries, uint32_t column, uint32_t message,
               const ops::NormWeights &norm, metal::MetalBuffer output) const;
  // One of the application's convolutions over the step's rows: its state
  // columns in each lane's matrix, the sequences' states in and out, the
  // states captured inside them, and a decode step's tree and tape.
  void addConvolution(const PrefillStep &step, uint32_t application, Inkling2LoopConvolution convolution,
                      const ops::ShortConvBuffers &buffers, ops::ShortConvShape shape,
                      ops::ShortConvOutput output) const;

  const Inkling2LoopWeights &weights_;
  Inkling2LoopSchedule schedule_;
  Inkling2LoopSteps steps_;
};

} // namespace splash::model
