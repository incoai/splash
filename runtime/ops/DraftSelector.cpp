#include "ops/DraftSelector.hpp"

#include "metal/abi/Sampling.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace splash::ops {
namespace {

constexpr uint32_t kShards = SPLASH_DRAFT_SAMPLING_SHARDS;
// Each position's group scores its 16 x 16 edge table eight edges per
// simdgroup task; eight simdgroups balance the seven-group B1 dispatch
// against the 28 groups of B4 (wider groups speed up B1 and slow down B4).
constexpr uint32_t kEdgeThreads = 256;

} // namespace

DraftSelector::DraftSelector(uint32_t vocabulary, uint32_t candidates, uint32_t rows)
    : vocabulary_(vocabulary), candidates_(candidates), rows_(rows) {
  if ((candidates != 16 && candidates != 32) || (rows != 8 && rows != 16) || (candidates == 16 && rows != 8))
    throw std::invalid_argument("invalid draft selector geometry");
  if (!vocabulary)
    throw std::invalid_argument("invalid draft selector vocabulary");
}

DraftSelectorWorkspace DraftSelector::workspace(uint32_t positions, uint32_t candidateCount, uint32_t rows) {
  if (candidateCount != 16 && candidateCount != 32)
    throw std::invalid_argument("invalid draft selector candidate count");
  if (!positions || (rows != 8 && rows != 16))
    throw std::invalid_argument("invalid draft selector workspace position count");
  const uint64_t candidates = uint64_t{positions} * candidateCount;
  // The partial values are followed by each position's 16 x 16 edge table.
  return {candidates * kShards * sizeof(uint32_t),
          candidates * (kShards + candidateCount) * sizeof(float) +
              ((positions + rows - 2) / (rows - 1)) *
                  draft_tree_lattice_bytes(candidateCount),
          candidates * sizeof(uint32_t), candidates * sizeof(float)};
}

void DraftSelector::add(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers,
                        const DraftCodebooks &codebooks, DraftVocabulary vocabulary,
                        std::span<const DraftTreeLane> lanes) const {
  if (lanes.empty() || lanes.size() > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid draft selector batch");
  if (vocabulary.leading > vocabulary.trailing || vocabulary.trailing > vocabulary_ || !vocabulary.columns(vocabulary_))
    throw std::invalid_argument("a draft's tokens lie in its vocabulary");
  const uint32_t count = static_cast<uint32_t>(lanes.size());
  const uint32_t columns = vocabulary.columns(vocabulary_);
  const uint32_t kPositions = positions();
  SelectorBatchParams params{};
  params.rows = rows_;
  params.positions = kPositions;
  params.lanes = count;
  params.vocabulary = vocabulary_;
  params.columns = columns;
  params.leading = vocabulary.leading;
  params.trailing = vocabulary.trailing;
  DraftTreeParams tree{};
  tree.lanes = count;
  tree.positions = kPositions;
  for (uint32_t lane = 0; lane < count; ++lane) {
    const DraftTreeLane &l = lanes[lane];
    // A tree's rows hold its anchor and every row's path below it, a verify
    // block's rows.
    if (l.depth >= SPLASH_TARGET_VERIFY_ROWS || l.chain > SPLASH_TREE_CHAIN_TOKENS ||
        l.children > SPLASH_TREE_CHILDREN)
      throw std::invalid_argument("invalid draft tree lane");
    if (!std::isfinite(l.calibration.slope) ||
        !std::ranges::all_of(l.calibration.intercepts, [](float value) { return std::isfinite(value); }))
      throw std::invalid_argument("invalid draft tree calibration");
    if (!(l.temperature >= 0.0F && std::isfinite(l.temperature)))
      throw std::invalid_argument("invalid draft tree temperature");
    params.anchor[lane] = tree.anchor[lane] = l.anchor;
    tree.depth[lane] = l.depth;
    tree.chain[lane] = l.chain;
    tree.children[lane] = l.children;
    tree.slope[lane] = l.calibration.slope;
    std::ranges::copy(l.calibration.intercepts, tree.intercept[lane]);
    tree.temperature[lane] = l.temperature;
    tree.noise[lane][0] = static_cast<uint32_t>(l.noise);
    tree.noise[lane][1] = static_cast<uint32_t>(l.noise >> 32);
  }
  // A lane's proposal positions take rows 1-7 of its eight query rows of the
  // logits and the selector hidden rows; its tree takes a verify block's rows.
  const DraftSelectorWorkspace workspace = DraftSelector::workspace(kPositions, candidates_, rows_);
  const uint64_t queryRows = uint64_t{count} * rows_;
  const uint64_t codebook = uint64_t{vocabulary_} * SPLASH_DRAFT_SELECTOR_RANK * 2;
  const uint64_t treeRows = uint64_t{count} * SPLASH_TARGET_VERIFY_ROWS;
  requireBytes(buffers.logits, queryRows * columns * sizeof(float), "draft logits");
  requireBytes(buffers.partialIds, workspace.partialIdsBytes * count, "draft selector partial id");
  requireBytes(buffers.partialValues, workspace.partialValuesBytes * count, "draft selector partial value");
  requireBytes(buffers.candidates, workspace.candidatesBytes * count, "draft candidate");
  requireBytes(buffers.unary, workspace.unaryBytes * count, "draft candidate score");
  requireBytes(buffers.selectorHidden, queryRows * SPLASH_DRAFT_SELECTOR_RANK * 2, "draft selector hidden");
  requireBytes(codebooks.predecessor, codebook, "draft predecessor codebook");
  requireBytes(codebooks.successor, codebook, "draft successor codebook");
  requireBytes(buffers.chainTokens, uint64_t{count} * SPLASH_TREE_CHAIN_TOKENS * sizeof(uint32_t),
               "draft tree chain token");
  requireBytes(buffers.chainRates,
               uint64_t{count} * SPLASH_TREE_CHAIN_TOKENS * SPLASH_TREE_AGREEMENTS * sizeof(float),
               "draft tree chain rate");
  requireBytes(buffers.treeTokens, treeRows * sizeof(uint32_t), "draft tree token");
  requireBytes(buffers.trees, treeRows * sizeof(TreeRow), "draft tree row");
  requireBytes(buffers.treeScores, treeRows * 2 * sizeof(float), "draft tree score");
  requireBytes(buffers.treeSizes, uint64_t{count} * sizeof(uint32_t), "draft tree size");
  requireBytes(buffers.treeDraws, treeRows * SPLASH_TREE_CHILDREN, "draft tree draw");
  requireBytes(buffers.treeNext, treeRows * candidates_ * sizeof(float), "draft tree next");
  requireBytes(buffers.treeCaps, treeRows * sizeof(float), "draft tree cap");
  const std::string suffix = candidates_ == 32 ? "_k32" : "";
  graph.add("draft_select_top16_sharded" + suffix,
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            params, {uint64_t{count} * kPositions * kShards, 1, 1});
  graph.add("draft_select_edges" + suffix,
            {buffers.partialIds, buffers.partialValues, buffers.candidates,
             buffers.unary, buffers.selectorHidden, codebooks.predecessor,
             codebooks.successor},
            params, {uint64_t{count} * kPositions, 1, 1},
            {kEdgeThreads, 1, 1});
  // A group's threads take the lattice's rows, one a position and
  // predecessor, before its first thread grows the tree.
  graph.add("draft_select_tree" + suffix,
            {buffers.candidates, buffers.unary, buffers.partialValues, buffers.chainTokens, buffers.chainRates,
             buffers.treeTokens, buffers.trees, buffers.treeScores, buffers.treeSizes, buffers.treeDraws,
             buffers.treeNext, buffers.treeCaps},
            tree, {count, 1, 1}, {kPositions * candidates_, 1, 1});
}

void DraftSelector::addVerifyRows(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers,
                                  const StepRows &step, float floor) {
  requireStepRows(step);
  if (!(floor >= 0.0F && std::isfinite(floor)))
    throw std::invalid_argument("invalid verify rows floor");
  const VerifyRowsParams params{.lanes = step.lanes, .rows = step.rows, .lane_cap = step.laneCap, .floor = floor};
  // It reads each lane's tree and writes the step's rows.
  const uint64_t treeRows = uint64_t{step.lanes} * SPLASH_TARGET_VERIFY_ROWS;
  requireBytes(buffers.treeTokens, treeRows * sizeof(uint32_t), "draft tree token");
  requireBytes(buffers.trees, treeRows * sizeof(TreeRow), "draft tree row");
  requireBytes(buffers.treeScores, treeRows * 2 * sizeof(float), "draft tree score");
  requireBytes(buffers.treeSizes, uint64_t{step.lanes} * sizeof(uint32_t), "draft tree size");
  requireBytes(buffers.inputTokens, uint64_t{step.rows} * sizeof(uint32_t), "verify input token");
  requireBytes(buffers.treeRows, uint64_t{step.rows} * sizeof(TreeRow), "verify tree row");
  requireBytes(buffers.laneRows, uint64_t{step.lanes} * sizeof(LaneRows), "verify lane rows");
  requireBytes(buffers.verifySources, uint64_t{step.rows} * sizeof(uint32_t), "verify source");
  graph.add("draft_verify_rows",
            {buffers.treeTokens, buffers.trees, buffers.treeScores, buffers.treeSizes, buffers.inputTokens,
             buffers.treeRows, buffers.laneRows, buffers.verifySources},
            params, {1, 1, 1}, {1, 1, 1});
}

} // namespace splash::ops
