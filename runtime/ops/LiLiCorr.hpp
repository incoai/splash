#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/DraftSelector.hpp"
#include "ops/Embedding.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace splash::ops {

// The LiLiCorr selector of NVIDIA Nemotron 3.5 Bolt's draft
// (metal/abi/LiLiCorr.h, llama.cpp's build_lilicorr_scores): each proposal
// position's eight best draft candidates are the nodes of a small
// transformer, 56 to a lane, whose factor heads score every edge from a
// position's candidate to the next one's, and from the lane's anchor to the
// first position's. Its scores take the place of the codebook edges in the
// walk DraftSelector takes. Every bias and small table is bf16.
struct LiLiCorrLayerWeights final {
  NormWeights attentionNorm;
  Projection inProjection;
  metal::MetalBuffer inBias;
  Projection outProjection;
  metal::MetalBuffer outBias;
  NormWeights mlpNorm;
  Projection upProjection;
  metal::MetalBuffer upBias;
  Projection downProjection;
  metal::MetalBuffer downBias;
};

struct LiLiCorrWeights final {
  // A candidate's target embedding, its position's draft hidden row and the
  // lane's anchor context row, each into the selector's width.
  Projection tokenProjection;
  metal::MetalBuffer tokenBias;
  Projection passProjection;
  metal::MetalBuffer passBias;
  Projection contextProjection;
  metal::MetalBuffer contextBias;
  // The features' layer norm (weight, bias) and their MLP: [width][5] up.
  metal::MetalBuffer featureNormWeight;
  metal::MetalBuffer featureNormBias;
  metal::MetalBuffer featureUp;
  metal::MetalBuffer featureUpBias;
  Projection featureDown;
  metal::MetalBuffer featureDownBias;
  // [positions][width], [candidates][width], [heads][nodes][nodes].
  metal::MetalBuffer slotEmbedding;
  metal::MetalBuffer rankEmbedding;
  metal::MetalBuffer attentionBias;
  std::vector<LiLiCorrLayerWeights> layers;
  NormWeights outputNorm;
  NormWeights anchorNorm;
  Projection factorProjection;
  metal::MetalBuffer factorBias;
  Projection outHead;
  metal::MetalBuffer outHeadBias;
  Projection inHead;
  metal::MetalBuffer inHeadBias;
  Projection anchorOutHead;
  metal::MetalBuffer anchorOutHeadBias;

  // Every projection, the ones each layer holds last.
  [[nodiscard]] std::vector<const Projection *> projections() const;
};

// The selector's scratch for a batch of up to SPLASH_MAXIMUM_BATCH_WIDTH
// lanes (LiLiCorr::scratchBytes); its projections' inputs and outputs hold
// their tiles' rows.
struct LiLiCorrScratch final {
  metal::MetalBuffer lse, ids, features, featureDown, embedded, token, pass, x, normalized, wide, attention,
      projected, mlp, anchorInput, anchorProjected, anchorState, anchorRows, anchorOut, outHead, inHead;
  LinearScratch linear;
};

// What a lane's selection reads besides the codebook selector's buffers: the
// draft's final hidden rows (its query rows, the draft's hidden width), the
// target's token table and each lane's anchor context row.
struct LiLiCorrInputs final {
  metal::MetalBuffer finalHidden;
  const EmbeddingWeights &tokenEmbedding;
  std::span<const metal::MetalBuffer> anchorRows;
};

class LiLiCorr final {
public:
  // The bytes of each of LiLiCorrScratch's buffers but linear, in order, for
  // a draft of `hidden` width.
  [[nodiscard]] static std::vector<uint64_t> scratchBytes(uint32_t hidden);
  // The most rows a selector's projection runs: a full batch's nodes or
  // pass rows.
  [[nodiscard]] static uint32_t maximumRows() noexcept;

  // Each lane's proposals: DraftSelector's candidates, the selector's edge
  // tables in their place in buffers.partialValues, then DraftSelector's
  // trees over them.
  static void add(metal::CommandGraph &graph, const Linear &linear, const DraftSelector &selector,
                  const DraftSelectorBuffers &buffers, const LiLiCorrWeights &weights,
                  const LiLiCorrScratch &scratch, const LiLiCorrInputs &inputs, DraftVocabulary vocabulary,
                  std::span<const DraftTreeLane> lanes, uint32_t hidden);
  // Copies row `row` of `rows` (or with `counts`, row counts[0] - 1) of
  // `hidden` bf16 values into row `anchorRow` of `anchor`: a lane's anchor
  // context row.
  static void addStoreAnchor(metal::CommandGraph &graph, const metal::MetalBuffer &rows, uint32_t row,
                             const metal::MetalBuffer &anchor, uint32_t hidden, uint32_t anchorRow = 0);
  // A decode step's lane `lane`'s anchor: its deepest accepted row of `rows`
  // (the highest bit of its path mask in `paths`, from its first row in
  // `laneRows`, SplashLaneRows).
  static void addStoreAcceptedAnchor(metal::CommandGraph &graph, const metal::MetalBuffer &rows,
                                     const metal::MetalBuffer &paths, const metal::MetalBuffer &laneRows,
                                     uint32_t lane, const metal::MetalBuffer &anchor, uint32_t hidden);
};

} // namespace splash::ops
