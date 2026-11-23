#pragma once

// A DSpark draft (Inkling 2 Mini's, released as a GGUF of architecture
// dflash): three dense layers over a block of the anchor's normed embedding
// and seven mask rows, which attend to the context keys and values of the
// committed positions before the anchor, each layer's projected from its
// own softmax mix of four target features of each position (three target
// layers' inputs and the target's final normed row, its captured rows); the
// draft's normed rows through the target's head give each row's base
// logits, which a recurrent Markov head corrects step by step over the whole
// vocabulary, each step following the token the step before chose.
//
// Its context keys and values live in each lane's rings (DFlashDraftRing),
// its layers' normed context rows in the lane's context window, three
// windows of a row each (DraftStateLayout::contextWidth), from which a
// restored lane rebuilds its rings, and the Markov head's context, the
// f_rnn row of the lane's last position, beside its rings
// (DFlashDraftRingLayer::auxiliary), which a prompt's last row and each
// decode commit write. Its rings hold SPLASH_DRAFT_SLIDING_WINDOW positions:
// its windowed layers see the slidingWindow before the anchor as trained,
// its global layer the ring's last SPLASH_DRAFT_SLIDING_WINDOW - 1.
//
// A cycle's proposals are each lane's chain of the Markov head
// (ops::DSpark::addMarkov), whose steps' top candidates and their logits
// become the selector's lattice along the chain (ops::DraftSelector::
// addChain): a greedy lane proposes the chain of its most likely tokens, a
// sampled lane draws each step's token from its candidates at its
// temperature, which its verify rows' rejection sampling tests.

#include "model/DFlashDraft.hpp"
#include "model/GgufFile.hpp"
#include "model/GgufImage.hpp"
#include "model/WeightImages.hpp"
#include "ops/DSpark.hpp"

#include <array>
#include <filesystem>
#include <memory>
#include <vector>

namespace splash::model {

struct DSparkLayerWeights final {
  ops::NormWeights inputNorm;
  // q (its query heads'), k and v rows of the layer's own block.
  ops::Projection qkv;
  metal::MetalBuffer queryNorm;
  metal::MetalBuffer keyNorm;
  ops::Projection output;
  ops::NormWeights ffnNorm;
  ops::Projection gate;
  ops::Projection up;
  ops::Projection down;
  // Its context: the mix of the target features and its norm, then its keys
  // and values (k then v rows) and its keys' norm.
  ops::DSparkMix context;
  ops::Projection contextKv;
  metal::MetalBuffer contextKeyNorm;
};

struct DSparkDraftWeights final {
  std::vector<DSparkLayerWeights> layers;
  ops::NormWeights finalNorm;
  // The Markov head's context mix and norm.
  ops::DSparkMix headContext;
  // The mask rows' embedding (bf16).
  metal::MetalBuffer maskEmbedding;
  // The Markov head's weights; its folded W2 and b2 are filled by the fold
  // (DSparkDraft::addSetup) from its output projection's columns and bias,
  // which the fold's rows hold: bf16 [rank + 2][hidden], the projection's
  // column c as row c, then its F32 bias as two bf16 rows that sum to it.
  ops::DSparkMarkovWeights markov;
  std::vector<uint16_t> foldRows;
  metal::MetalBuffer foldInput;
  // Each step's Markov workspace, and the lanes' f_rnn rows, gathered from
  // their rings for the chain. The chain writes the selector's lattice
  // (DFlashDecodeBuffers::candidates), not a lattice of its own.
  metal::MetalBuffer markovJoint;
  metal::MetalBuffer markovState;
  metal::MetalBuffer markovPrevious;
  metal::MetalBuffer markovLogits;
  metal::MetalBuffer markovPartialValues;
  metal::MetalBuffer markovPartialIds;
  metal::MetalBuffer frnnRows;
};

// The header of the DSpark GGUF in `directory` (its one .gguf), or of the
// GGUF `file`: what its metadata states of the draft, checked against what
// the kernels take (DSparkHeader).
[[nodiscard]] DSparkHeader readDSparkHeader(const GgufFile &file);
[[nodiscard]] DSparkHeader readDSparkHeader(const std::filesystem::path &directory);

// The images of a DSpark draft of `layout` from its GGUF: each layer's, then
// model.bin's. Throws GgufError for a missing tensor or one of another shape
// or type.
[[nodiscard]] std::vector<gguf::Image> planDSparkImages(const GgufFile &file, const DFlashDraftLayout &layout);

// The bytes a DSpark draft of `layout` in `directory` loads: its images, its
// folded Markov weights and its workspace.
[[nodiscard]] uint64_t dsparkDraftBytes(const std::filesystem::path &directory, const DFlashDraftLayout &layout);

// Reads a DSpark draft of `layout` from its GGUF in `directory` into images
// of `images`.
[[nodiscard]] DFlashDraftWeights loadDSparkDraftWeights(metal::MetalBackend &backend, WeightImages &images,
                                                        const std::filesystem::path &directory,
                                                        const DFlashDraftLayout &layout);

// A DSpark draft over its target's vocabulary head `head` and its embedding
// norm `anchorNorm`, which its block's anchor row takes (the target's own
// embedding of a token is that row's norm).
class DSparkDraft final : public DraftModel {
public:
  DSparkDraft(const DFlashDraftWeights &weights, metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
              const ops::Projection &head, const ops::NormWeights &anchorNorm);

  [[nodiscard]] DFlashDraftStep step(std::span<const uint32_t> cacheLengths,
                                     std::span<const std::span<const uint32_t>> tokens,
                                     std::span<const ops::DraftTreeLane> lanes) const override;
  // Each layer's context rows of the captured rows, into each span's
  // window, and their keys and values into its rings; each span's last
  // row's f_rnn into its lane's.
  void addContextPrefill(metal::CommandGraph &graph, DFlashPrefillBuffers buffers, uint32_t rows,
                         std::span<const DFlashPrefillSpan> spans,
                         std::span<const DFlashWindowCapture> windowCaptures) const override;
  void addContextRebuild(metal::CommandGraph &graph, DFlashPrefillBuffers buffers, const metal::MetalBuffer &window,
                         std::span<const DFlashDraftRingLayer> ring, uint32_t rows,
                         uint32_t startPosition) const override;
  // The block's rows (the anchor's row normed, the mask rows), its layers,
  // its final norm and its base logits over the whole vocabulary, then each
  // lane's Markov chain from its f_rnn into the selector's lattice
  // (buffers.candidates and unary), whose candidates and probabilities the
  // target's sampled verify rows test the chain's draws against.
  void addDecode(metal::CommandGraph &graph, DFlashDecodeBuffers buffers, const DFlashDraftStep &step) const override;
  // Each lane's tree of its chain and its host chain, over the lattice
  // addDecode wrote.
  void addSelection(metal::CommandGraph &graph, const ops::DraftSelectorBuffers &buffers,
                    const DFlashDraftStep &step) const override;
  // Nothing: the block's rows are its own (addDecode).
  void addMaskRows(metal::CommandGraph &, metal::MetalBuffer, uint32_t) const override {}
  void addContextCommit(metal::CommandGraph &graph, DFlashContextBuffers buffers,
                        std::span<const uint32_t> startPositions, const ops::StepRows &step) const override;
  // The fold of the Markov head's output projection into the vocabulary
  // head: W2 = head x W_out and b2 = head x b_out, through the head's
  // decode plans in blocks of `stepRows` rows into `rows`.
  void addSetup(metal::CommandGraph &graph, ops::LinearScratch scratch, metal::MetalBuffer rows,
                uint32_t stepRows) const override;

private:
  // Each layer's context rows of `rows` rows of buffers.hidden ([layer][row]),
  // their keys and values into the spans' rings.
  void addContextKv(metal::CommandGraph &graph, const DFlashPrefillBuffers &buffers, uint32_t rows,
                    std::span<const DFlashPrefillSpan> spans) const;
  // Layer `layer`'s context window, of a lane's or a state's three.
  [[nodiscard]] metal::MetalBuffer layerWindow(const metal::MetalBuffer &window, uint32_t layer) const;
  [[nodiscard]] ops::DraftAttentionWindow attentionWindow(uint32_t layer) const;

  const DFlashDraftLayout &layout_;
  const DSparkDraftWeights &weights_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
  ops::DraftSelector selector_;
  const ops::Projection &head_;
  const ops::NormWeights &anchorNorm_;
  // The fold's rows (DSparkDraftWeights::foldRows) on the device.
  metal::MetalBuffer foldRows_;
};

// The draft a runtime drives of `weights` (DraftModel): a DFlash2 draft over
// the head's rows of `subset`, or a DSpark draft whose anchor takes
// `embeddingNorm`, its target's.
[[nodiscard]] std::unique_ptr<DraftModel> makeDraftModel(const DFlashDraftWeights &weights,
                                                         metal::MetalBackend &backend,
                                                         const ops::ExecutionPlans &operators,
                                                         const ops::Projection &head, ops::DraftVocabulary subset,
                                                         const ops::NormWeights *embeddingNorm);

} // namespace splash::model
