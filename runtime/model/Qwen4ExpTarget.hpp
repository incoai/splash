#pragma once

// Qwen3.8-Flash-Next's target (model/Qwen4Exp.hpp): its layers over the
// blocks every Qwen target shares (model/QwenTarget.hpp). Every block reads
// the mix of the fp32 residual streams and adds its output back to each
// stream; the PLE layer adds its gated n-gram values before its mixer's mix;
// the full-attention layers attend through the blocks their QSA indexer
// selects; and the head reads the final mix without a norm. Its step scratch
// and the inputs it makes of a step's tokens are Qwen4Steps'.

#include "model/Qwen4Steps.hpp"
#include "model/QwenTarget.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace splash::model {

struct Qwen4ExpWeights;
struct Qwen4ExpLayerWeights;
struct Qwen4HyperWeights;
struct Qwen4IndexerWeights;

class Qwen4ExpTarget final : public QwenTarget {
public:
  // A target whose prefill commands run up to `prefillChunks` chunks.
  Qwen4ExpTarget(const Qwen4ExpWeights &weights, const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
                 const ops::ExecutionPlans &operators, uint32_t prefillChunks);

  [[nodiscard]] QwenTargetScratchBytes allocatedScratchBytes() const noexcept override;
  [[nodiscard]] std::vector<metal::MetalBuffer> addPrefill(metal::CommandGraph &graph,
                                                           std::span<const QwenTargetPrefillChunk> chunks,
                                                           const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                           QwenPrefillLayers layers, ops::AneFfn *aneFfn) override;
  void addInsideCapture(metal::CommandGraph &graph, metal::MetalBuffer auxiliaryIn,
                        metal::MetalBuffer captured) const override;
  void prepareVerifyLane(uint32_t lane, uint64_t position, const QwenTargetRequest &request) override;
  // The hidden rows are the final mix, which the head reads as they are.
  void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer finalHidden,
                    metal::MetalBuffer logits, uint32_t rows, ops::LinearScratch scratch) const override;
  // The GPU hashes the verify rows' PLE rows, and when the table streams a
  // host step of the command stages them, between the hash and the verify
  // that gathers them.
  void addVerifyInput(metal::CommandGraph &graph, const QwenVerifyInput &input) const override;
  // Each lane's PLE history and QSA pending keys after its accepted path, and
  // in every attention layer the blocks the path completes into the lane's
  // QSA index, from the convolution inputs, raw keys and block keys its
  // verify wrote of its block.
  void addStateCommit(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers,
                      const ops::StepRows &step) const override;

  // The PLE rows of a lane's rows of the last step, each row's heads' in
  // order.
  [[nodiscard]] std::vector<uint32_t> verifyTableRows(const ops::LaneRows &rows) const override;

private:
  // The shared prefill's layer loop runs the family's (QwenTarget::addPrefillLayers).
  friend class QwenTarget;

  // A step's family state: the shared one, its scratch, the QSA blocks each
  // prompt sequence's rows complete or each decode lane's rows may complete,
  // and where each attention layer's QSA index sits in the extents the index
  // tables reach.
  struct PrefillStep : QwenTarget::PrefillStep {
    Qwen4Scratch scratch{};
    std::vector<Qwen4Blocks> blocks{};
    Qwen4LaneBlocks laneBlocks{};
    std::span<const uint32_t> indexOffsets{};
  };
  // The family's prefill layers. The streams start as the embedding's rows;
  // every MoE adds its rows to zeros into the branch the next mix adds to the
  // streams; the final mix goes into buffers.hidden[1].
  void addPrefillBegin(PrefillStep &step) const;
  [[nodiscard]] bool attends(uint32_t index) const;
  void addPrefillLayer(PrefillStep &step, uint32_t index) const;
  // The PLE history a state captured `capture.row` rows into a sequence
  // holds: the history of `auxiliaryIn` advanced by those rows of the
  // sequence's convolution inputs from `rowBegin` of the chunk's scratch.
  void addCaptureHistory(metal::CommandGraph &graph, const Qwen4Scratch &scratch, uint32_t rowBegin,
                         metal::MetalBuffer auxiliaryIn, const QwenTargetStateCapture &capture) const;
  [[nodiscard]] const ops::BlockMoeWeights *moe(uint32_t index) const;
  void addPrefillExperts(PrefillStep &step, uint32_t index) const;
  [[nodiscard]] metal::MetalBuffer addPrefillEnd(PrefillStep &step) const;

  // A step's mix of the streams through `hyper` into `output` and, when it
  // injects, the injection weights into the step's scratch. Its projections
  // read the tables its norm and low step write as their plans take them,
  // and with `next`, the projection that reads `output` first takes the
  // table the mix writes (the returned input).
  ops::PreparedInput addMix(PrefillStep &step, const Qwen4HyperWeights &hyper, metal::MetalBuffer output,
                            const ops::Projection *next = nullptr) const;
  // The QSA of the attention layer the step reaches next: the indexer's
  // projections of the step's rows; a prompt sequence's blocks its rows
  // complete into its index and its pending keys after them, or a verify
  // lane's block key of each block a row's path completes, which the commit
  // writes into the index once the lane accepts that path (addStateCommit);
  // a sequence or lane whose rows attend more blocks than they keep then
  // selects theirs, and returns the block lists its attention attends
  // through (none for rows that keep every block).
  [[nodiscard]] std::vector<ops::AttentionBlocks> addPrefillQsa(PrefillStep &step,
                                                                const Qwen4IndexerWeights &indexer) const;
  [[nodiscard]] std::vector<ops::AttentionBlocks> addDecodeQsa(PrefillStep &step,
                                                               const Qwen4IndexerWeights &indexer) const;
  // The PLE layer's `count` rows into scratch.pleRows: from the table by
  // scratch.pleIndices, or, when the table streams, from the rows staged by
  // scratch.pleSlots, with the table's own gather.
  void addPleGather(metal::CommandGraph &graph, const Qwen4Scratch &scratch, uint32_t count) const;

  const Qwen4ExpWeights &weights_;
  // The step scratch and the inputs the target makes of a step's tokens.
  Qwen4Steps steps_;
};

} // namespace splash::model
