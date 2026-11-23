#include "ops/DraftSelector.hpp"

#include "metal/abi/Sampling.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace splash::ops {
namespace {

constexpr uint32_t kShards = SPLASH_DRAFT_SAMPLING_SHARDS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
// Each position's group scores its 16 x 16 edge table eight edges per
// simdgroup task; eight simdgroups balance the seven-group B1 dispatch
// against the 28 groups of B4 (wider groups speed up B1 and slow down B4).
constexpr uint32_t kEdgeThreads = 256;

} // namespace

DraftSelector::DraftSelector(uint32_t vocabulary) : vocabulary_(vocabulary) {
  if (!vocabulary)
    throw std::invalid_argument("invalid draft selector vocabulary");
}

DraftSelectorWorkspace DraftSelector::workspace(uint32_t positions) {
  if (!positions)
    throw std::invalid_argument("invalid draft selector workspace position count");
  const uint64_t candidates = uint64_t{positions} * SPLASH_DRAFT_CANDIDATES;
  // The partial values are followed by each position's 16 x 16 edge table.
  return {candidates * kShards * sizeof(uint32_t),
          candidates * (kShards + SPLASH_DRAFT_CANDIDATES) * sizeof(float),
          candidates * sizeof(uint32_t), candidates * sizeof(float)};
}

namespace {

// The tree parameters of `lanes`, each checked.
DraftTreeParams treeParams(std::span<const DraftTreeLane> lanes) {
  if (lanes.empty() || lanes.size() > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid draft selector batch");
  DraftTreeParams tree{};
  tree.lanes = static_cast<uint32_t>(lanes.size());
  for (uint32_t lane = 0; lane < tree.lanes; ++lane) {
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
    tree.anchor[lane] = l.anchor;
    tree.depth[lane] = l.depth;
    tree.chain[lane] = l.chain;
    tree.children[lane] = l.children;
    tree.slope[lane] = l.calibration.slope;
    std::ranges::copy(l.calibration.intercepts, tree.intercept[lane]);
    tree.temperature[lane] = l.temperature;
    tree.noise[lane][0] = static_cast<uint32_t>(l.noise);
    tree.noise[lane][1] = static_cast<uint32_t>(l.noise >> 32);
  }
  return tree;
}

// Throws unless `buffers` hold `lanes` lanes' trees and host chains.
void requireTrees(const DraftSelectorBuffers &buffers, uint32_t lanes) {
  const uint64_t treeRows = uint64_t{lanes} * SPLASH_TARGET_VERIFY_ROWS;
  requireBytes(buffers.chainTokens, uint64_t{lanes} * SPLASH_TREE_CHAIN_TOKENS * sizeof(uint32_t),
               "draft tree chain token");
  requireBytes(buffers.chainRates,
               uint64_t{lanes} * SPLASH_TREE_CHAIN_TOKENS * SPLASH_TREE_AGREEMENTS * sizeof(float),
               "draft tree chain rate");
  requireBytes(buffers.treeTokens, treeRows * sizeof(uint32_t), "draft tree token");
  requireBytes(buffers.trees, treeRows * sizeof(TreeRow), "draft tree row");
  requireBytes(buffers.treeScores, treeRows * 2 * sizeof(float), "draft tree score");
  requireBytes(buffers.treeSizes, uint64_t{lanes} * sizeof(uint32_t), "draft tree size");
  requireBytes(buffers.treeDraws, treeRows * SPLASH_TREE_CHILDREN, "draft tree draw");
  requireBytes(buffers.treeNext, treeRows * SPLASH_DRAFT_CANDIDATES * sizeof(float), "draft tree next");
  requireBytes(buffers.treeCaps, treeRows * sizeof(float), "draft tree cap");
}

// The tree kernel over the lattice of `buffers`' candidates and scores.
void addTree(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers, const DraftTreeParams &tree) {
  // A group's threads take the lattice's rows, one a position and
  // predecessor, before its first thread grows the tree.
  graph.add("draft_select_tree",
            {buffers.candidates, buffers.unary, buffers.partialValues, buffers.chainTokens, buffers.chainRates,
             buffers.treeTokens, buffers.trees, buffers.treeScores, buffers.treeSizes, buffers.treeDraws,
             buffers.treeNext, buffers.treeCaps},
            tree, {tree.lanes, 1, 1}, {kPositions * SPLASH_DRAFT_CANDIDATES, 1, 1});
}

} // namespace

void DraftSelector::addChain(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers,
                             std::span<const DraftTreeLane> lanes) const {
  DraftTreeParams tree = treeParams(lanes);
  tree.markov = (1u << tree.lanes) - 1;
  const uint64_t lattice = uint64_t{tree.lanes} * kPositions * SPLASH_DRAFT_CANDIDATES;
  requireBytes(buffers.candidates, lattice * sizeof(uint32_t), "draft candidate");
  requireBytes(buffers.unary, lattice * sizeof(float), "draft candidate score");
  requireBytes(buffers.partialValues, sizeof(float), "draft selector partial value");
  requireTrees(buffers, tree.lanes);
  addTree(graph, buffers, tree);
}

void DraftSelector::add(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers,
                        const DraftCodebooks &codebooks, DraftVocabulary vocabulary,
                        std::span<const DraftTreeLane> lanes) const {
  const DraftTreeParams tree = treeParams(lanes);
  if (vocabulary.leading > vocabulary.trailing || vocabulary.trailing > vocabulary_ || !vocabulary.columns(vocabulary_))
    throw std::invalid_argument("a draft's tokens lie in its vocabulary");
  const uint32_t count = static_cast<uint32_t>(lanes.size());
  const uint32_t columns = vocabulary.columns(vocabulary_);
  SelectorBatchParams params{};
  params.lanes = count;
  params.vocabulary = vocabulary_;
  params.columns = columns;
  params.leading = vocabulary.leading;
  params.trailing = vocabulary.trailing;
  std::ranges::copy(std::span(tree.anchor).first(count), params.anchor);
  // A lane's proposal positions take rows 1-7 of its eight query rows of the
  // logits and the selector hidden rows; its tree takes a verify block's rows.
  const DraftSelectorWorkspace workspace = DraftSelector::workspace(count * kPositions);
  const uint64_t queryRows = uint64_t{count} * SPLASH_DRAFT_QUERY_ROWS;
  const uint64_t codebook = uint64_t{vocabulary_} * SPLASH_DRAFT_SELECTOR_RANK * 2;
  requireBytes(buffers.logits, queryRows * columns * sizeof(float), "draft logits");
  requireBytes(buffers.partialIds, workspace.partialIdsBytes, "draft selector partial id");
  requireBytes(buffers.partialValues, workspace.partialValuesBytes, "draft selector partial value");
  requireBytes(buffers.candidates, workspace.candidatesBytes, "draft candidate");
  requireBytes(buffers.unary, workspace.unaryBytes, "draft candidate score");
  requireBytes(buffers.selectorHidden, queryRows * SPLASH_DRAFT_SELECTOR_RANK * 2, "draft selector hidden");
  requireBytes(codebooks.predecessor, codebook, "draft predecessor codebook");
  requireBytes(codebooks.successor, codebook, "draft successor codebook");
  requireTrees(buffers, count);
  graph.add("draft_select_top16_sharded",
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            params, {uint64_t{count} * kPositions * kShards, 1, 1});
  graph.add("draft_select_edges",
            {buffers.partialIds, buffers.partialValues, buffers.candidates,
             buffers.unary, buffers.selectorHidden, codebooks.predecessor,
             codebooks.successor},
            params, {uint64_t{count} * kPositions, 1, 1},
            {kEdgeThreads, 1, 1});
  addTree(graph, buffers, tree);
}

void DraftSelector::writeChain(const DraftSelectorBuffers &buffers, uint32_t lane, uint32_t anchor,
                               std::span<const uint32_t> chain, std::span<const float> rates) {
  if (lane >= SPLASH_MAXIMUM_BATCH_WIDTH || chain.size() > SPLASH_TREE_CHAIN_TOKENS || rates.size() != chain.size() ||
      !std::ranges::all_of(rates, [](float rate) { return rate >= 0.0F && rate <= 1.0F; }))
    throw std::invalid_argument("invalid host chain tree");
  const auto rows = static_cast<uint32_t>(chain.size()) + 1;
  const uint64_t first = uint64_t{lane} * SPLASH_TARGET_VERIFY_ROWS, end = first + rows;
  requireBytes(buffers.treeTokens, end * sizeof(uint32_t), "draft tree token");
  requireBytes(buffers.trees, end * sizeof(TreeRow), "draft tree row");
  requireBytes(buffers.treeScores, end * 2 * sizeof(float), "draft tree score");
  requireBytes(buffers.treeSizes, (uint64_t{lane} + 1) * sizeof(uint32_t), "draft tree size");
  requireBytes(buffers.treeDraws, end * SPLASH_TREE_CHILDREN, "draft tree draw");
  requireBytes(buffers.treeCaps, end * sizeof(float), "draft tree cap");
  auto *tokens = static_cast<uint32_t *>(buffers.treeTokens.contents());
  auto *tree = static_cast<TreeRow *>(buffers.trees.contents());
  auto *scores = static_cast<float *>(buffers.treeScores.contents());
  auto *sizes = static_cast<uint32_t *>(buffers.treeSizes.contents());
  auto *draws = static_cast<uint8_t *>(buffers.treeDraws.contents());
  auto *caps = static_cast<float *>(buffers.treeCaps.contents());
  if (!tokens || !tree || !scores || !sizes || !draws || !caps)
    throw std::invalid_argument("a host chain tree's buffers are not CPU-visible");
  float estimate = 1.0F;
  for (uint32_t row = 0; row < rows; ++row) {
    if (row)
      estimate *= rates[row - 1];
    tokens[first + row] = row ? chain[row - 1] : anchor;
    // Rows 0 to row: its ancestors, itself the last.
    tree[first + row] = {(2U << row) - 1, static_cast<uint16_t>(row ? row - 1 : SPLASH_TREE_ROOT),
                         static_cast<uint16_t>(row)};
    scores[2 * (first + row)] = row ? -1.0F : 1.0F;
    scores[2 * (first + row) + 1] = estimate;
    caps[first + row] = 1.0F;
  }
  std::fill(draws + first * SPLASH_TREE_CHILDREN, draws + end * SPLASH_TREE_CHILDREN, uint8_t{0xff});
  sizes[lane] = rows;
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
