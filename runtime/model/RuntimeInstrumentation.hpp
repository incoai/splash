#pragma once

#include "metal/abi/KvExtent.h"
#include "model/Runtime.hpp"
#include "model/StepCosts.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace splash::model {

// What tests impose on a Runtime beyond its serving API. Only the
// instrumented build of Runtime.cpp (SPLASH_RUNTIME_INSTRUMENTATION) defines
// these; production binaries never link it.
class RuntimeInstrumentation final {
public:
  // The PLE rows the last decode step gathered for batch lane `lane`'s rows
  // of the step, each row's heads' in order (Qwen3.8-Flash-Next), until the
  // next decode step; none for a target without them.
  [[nodiscard]] static std::vector<uint32_t> lastTableRows(const Runtime &runtime, uint32_t lane);
  // The fp32 logits the last command that selected left in batch lane
  // `lane`'s rows, each the target's vocabulary wide, until the next command:
  // a decode step's rows as its selector laid them out, or the row of the
  // final position of the prefill chunk that ended a prompt.
  [[nodiscard]] static std::span<const float> lastLogits(const Runtime &runtime, uint32_t lane);
  // The target rows the last prefill command captured for the draft in its
  // chunk `chunk`'s buffer, until the next prefill: bf16 rows of each target
  // layer id the draft reads (its target_layer_ids, hiddenSize values each,
  // in their order; QwenTargetDimensions::draftReadsLayerInputs names which
  // rows), the captured rows in the order the draft's context plan captured
  // them (a cold prompt's first command: its rows).
  [[nodiscard]] static std::span<const uint16_t> lastCapturedRows(const Runtime &runtime, uint32_t chunk);
  // Has every later prefill command's draft read `rows`, the captured rows of
  // positions 0 .. their count - 1 (capturedHiddenSize bf16 values each), in
  // place of those its target captures at those positions, so that a
  // reference's draft and Splash's read the same rows; no rows restores the
  // target's.
  static void replaceCapturedRows(Runtime &runtime, std::span<const uint16_t> rows);
  // The draft's lattice of batch lane `lane` of the last decode step of
  // `lanes` lanes, until the next decode step, as ops::DraftSelector left
  // it: each proposal position's candidates, their unary scores (the
  // draft's logits of them, its head's finish but its softcap applied) and
  // the edges to them from each candidate of the position before (from the
  // anchor alone, predecessor 0, for the first position), each
  // [position][predecessor][candidate]; and the probabilities of the first
  // position's candidates after the anchor, which its tree's first row holds.
  struct DraftLattice final {
    std::vector<uint32_t> candidates;
    std::vector<float> unary;
    std::vector<float> edges;
    std::vector<float> first;
  };
  [[nodiscard]] static DraftLattice lastDraftLattice(const Runtime &runtime, uint32_t lane, uint32_t lanes);
  // The rows batch lane `lane` of the last decode step verified, its rows of
  // the step as its selector laid them out (ops::DraftSelector::
  // addVerifyRows), until the next command that selects.
  [[nodiscard]] static uint32_t lastVerifyRows(const Runtime &runtime, uint32_t lane);
  // The path of the rows batch lane `lane` of the last decode step retained,
  // a bit a row of its rows from its first (metal/abi/VerifyTree.h): a chain
  // of r rows is its first r bits; until the next command that selects.
  [[nodiscard]] static uint32_t lastRetainedRows(const Runtime &runtime, uint32_t lane);
  // Every later decode step of a runtime takes `rows` rows, at most a verify
  // block's a lane and at least a row a lane, each of them verified but
  // those past an anchor estimated below `floor` (VerifyRowsParams), in
  // place of its cycles' choice (StepCosts::choose), which such steps do
  // not teach; none gives the choice back. For measuring steps' costs by
  // their rows, and for steps whose lanes take fewer rows than they hold.
  static void forceStepRows(Runtime &runtime, std::optional<uint32_t> rows, float floor = 0.0F);
  // The next decode cycle of constrained request `requestId` gives each row
  // `row` of its lane's tree past the anchor the token `token` of `tokens`
  // in place of its own: written into the cycle's verify input once its
  // draft laid out the step's rows (ops::DraftSelector::addVerifyRows),
  // before its mask request and the target's forward, which both read it;
  // the tree's rows and their places stay. For checking that a row computes
  // what its path holds whatever its tree's other rows hold.
  static void replaceTreeTokens(Runtime &runtime, uint64_t requestId,
                                std::vector<std::pair<uint32_t, uint32_t>> tokens);
  // The next decode cycle of request `requestId` draws its tree's children
  // from its draft at `temperature` with `noise` (ops::DraftTreeLane), in
  // place of its own: other draws of a sampled lane, draws in place of a
  // greedy lane's most likely candidates. Its verify selects by its own
  // policy and uniforms. For checking what a cycle's own draft decides.
  static void redrawTree(Runtime &runtime, uint64_t requestId, float temperature, uint64_t noise);
  // Whether a streamed-expert target's later cycles each plan a draft's
  // block a lane within its store's round, at their floor, whatever class
  // their floor passes the lanes' last chances into (CostLine::choose): for
  // measuring what the larger class's kernels cost a cycle.
  static void planBlocks(Runtime &runtime, bool enabled);
  // A decode cycle's choice of its step (StepCosts::choose) and what it
  // weighed: its rows and lane cap, the chances its lanes' curves held
  // (VerifyBudget::laneChances), the expected accepted tokens and its cost
  // (none before its lane count is measured, nor where a floor sizes the
  // cycle); and once it completed, the rows its lanes verified (fewer than
  // the step's where its trees held fewer or its floor left rows out), the
  // tokens they retained and the cycle's wall and GPU times.
  struct StepChoice final {
    uint32_t lanes = 0;
    uint32_t rows = 0;
    uint32_t laneCap = 0;
    uint32_t chances = 0;
    double expected = 0.0;
    double cost = 0.0;
    uint32_t verified = 0;
    uint32_t retained = 0;
    double wallMilliseconds = 0.0;
    double gpuMilliseconds = 0.0;
    // The cost of a pass over a thousand tokens of history it weighed
    // (StepCosts::contextCost).
    double contextCost = 0.0;
    // The floor its lanes took their rows at (StepCosts::Choice), and the
    // time a streamed-expert target's line took the cycle at
    // (CostLine::observe; none where no line did).
    double floor = 0.0;
    double sampleMilliseconds = 0.0;
  };
  // The choices of the decode cycles the step costs sized since the last
  // take, which it clears.
  [[nodiscard]] static std::vector<StepChoice> takeStepChoices(Runtime &runtime);
  // The classes of `lanes` lanes' steps and their measured costs
  // (StepCosts::points); none without --verify-rows auto.
  [[nodiscard]] static std::vector<StepCosts::Point> stepPoints(const Runtime &runtime, uint32_t lanes);
  // The level check of the tables its start recalled: their largest step of
  // the first warmup's lanes over that step's median now (none without one).
  [[nodiscard]] static std::optional<double> levelRatio(const Runtime &runtime);
  // While capturing, each command that selects first tokens or verifies
  // copies the rows lastLogits returns, as the target's forward wrote them
  // before the policy rewrites them in place: for each of its batch lanes,
  // row 0 for a selection or the lane's rows (lastVerifyRows) for a verify.
  // Only a capturing command adds the host step that copies them.
  // takeTargetLogits() reads and clears the copies, one per batch lane of the
  // last such command since the last take; none if none ran.
  static void captureTargetLogits(Runtime &runtime, bool enabled);
  [[nodiscard]] static std::vector<std::vector<float>> takeTargetLogits(Runtime &runtime);
  // The anchor request `requestId`'s last selection left pending, which its
  // next decode step emits first; none until its first token is selected.
  [[nodiscard]] static std::optional<uint32_t> pendingToken(const Runtime &runtime, uint64_t requestId);
  // The tokens request `requestId`'s lookup index holds (model::LookupIndex),
  // until its next command or suspension.
  [[nodiscard]] static std::span<const uint32_t> lookupTokens(const Runtime &runtime, uint64_t requestId);
  // The lookup chain request `requestId`'s latest decode cycle proposed beside
  // its draft's tree, until its next; none where it proposed none.
  [[nodiscard]] static std::span<const uint32_t> lookupChain(const Runtime &runtime, uint64_t requestId);
  // The GPU tables of resident request `requestId`'s lane as its last
  // command left them, every entry the lane holds: its page table and, for
  // a layout with a block index, its index table (none without).
  struct LaneTables final {
    std::span<const SplashKvPage> pages;
    std::span<const SplashKvPage> index;
  };
  [[nodiscard]] static LaneTables laneTables(const Runtime &runtime, uint64_t requestId);
  // The rows resident request `requestId`'s latest decode step accepted,
  // which its GDN cell holds only once its next step folds them in
  // (QwenLaneMetadata::pendingPath): every GDN layer's tape rows
  // (metal/abi/GDN.h) of that path in its order, field by field, layer by
  // layer.
  [[nodiscard]] static std::vector<uint8_t> pendingGdnRows(const Runtime &runtime, uint64_t requestId);
  // The entries of the registry through which the placements of an image
  // share its rows: an entry for each image whose rows something holds, and
  // for each whose rows went since the last start with images.
  [[nodiscard]] static size_t imageEntries(const Runtime &runtime);
};

} // namespace splash::model
