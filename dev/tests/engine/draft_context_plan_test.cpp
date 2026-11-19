#include "TestChecks.hpp"
#include "DraftRingView.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "model/Model.hpp"
#include "benchmarks/PrefillWork.hpp"
#include "engine/Engine.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace splash;
using benchmark::draftContextRows;

namespace {

using splash::test::rejects;
using splash::test::require;

DraftContextPlan activePlan(uint32_t replayBegin, uint32_t replayEnd) {
  return planDraftContext(replayBegin, replayEnd, {}, SPLASH_DRAFT_SLIDING_WINDOW);
}

DraftContextPlan
cachedPlan(uint32_t replayBegin, uint32_t replayEnd,
           std::initializer_list<uint32_t> materializationBoundaries) {
  return planDraftContext(
      replayBegin, replayEnd,
      {materializationBoundaries.begin(), materializationBoundaries.size()},
      SPLASH_DRAFT_SLIDING_WINDOW);
}

// The rows a plan captures for the boundaries of one purpose.
uint64_t purposeRows(const DraftContextPlan &plan,
                     DraftBoundaryPurpose purpose) {
  uint64_t rows = 0;
  for (const DraftBoundaryPlan &boundary : plan.boundaries) {
    if (boundary.purpose == purpose)
      rows += boundary.boundary - boundary.captureBegin;
  }
  return rows;
}

// The capture spans that start a new draft window.
uint32_t resets(const DraftContextPlan &plan) {
  return static_cast<uint32_t>(std::count_if(
      plan.captureSpans.begin(), plan.captureSpans.end(),
      [](const DraftCaptureSpan &span) { return span.resetDraftState; }));
}

void testColdLengths() {
  constexpr std::array<uint32_t, 12> lengths{0,  1,    7,    8,    31,   32,
                                             33, 2047, 2048, 2049, 4096, 10000};
  for (uint32_t length : lengths) {
    const auto plan = activePlan(0, length);
    require(draftContextRows(plan) == std::min<uint32_t>(length, 2048),
            "cold draft window was not clipped to 2048 rows");
    require(resets(plan) == (length ? 1U : 0U),
            "cold draft state reset count is wrong");
  }
}

void testPartialHit() {
  {
    const auto plan = activePlan(4096, 5000);
    require(plan.captureSpans.size() == 1, "short suffix lost capture span");
    require(plan.captureSpans[0].begin == 4096 &&
                plan.captureSpans[0].end == 5000 &&
                !plan.captureSpans[0].resetDraftState,
            "short suffix did not continue restored draft state");
    require(draftContextRows(plan) == 904,
            "short suffix processed more than actual suffix rows");
    require(plan.restoresDraftState,
            "short suffix incorrectly skipped restored state");
  }
  {
    const auto plan = activePlan(4096, 8192);
    require(plan.captureSpans.size() == 1, "long suffix lost capture span");
    require(plan.captureSpans[0].begin == 6144 &&
                plan.captureSpans[0].end == 8192 &&
                plan.captureSpans[0].resetDraftState,
            "long suffix did not rebuild only its final window");
    require(draftContextRows(plan) == 2048,
            "long suffix did not clip draft work");
    require(!plan.restoresDraftState,
            "long suffix copied a state it necessarily overwrites");
  }
}

void testFinalFullBlockBound() {
  const auto plan = cachedPlan(0, 10000, {9984});
  require(plan.captureSpans.size() == 1,
          "nearby final boundaries should share one capture span");
  require(plan.captureSpans[0].begin == 7936 &&
              plan.captureSpans[0].end == 10000,
          "10K final/full-block window is wrong");
  require(draftContextRows(plan) == 2064,
          "10K final/full-block work exceeded the planned window");
  require(draftContextRows(plan) <= 2079,
          "ordinary 10K draft work exceeded W+31 rows");

  // When prompt end itself is Page32-aligned, exact lookup must back off one
  // whole input block. Preserving both that cache state and the active prompt
  // state needs W+32 rows; dropping one row would corrupt one of the rings.
  const auto aligned = cachedPlan(0, 4096, {4064});
  require(draftContextRows(aligned) == 2080,
          "Page32-aligned prompt lost a required draft-context row");
}

void testBenchmarkWorkIncludesRecoveryPoints() {
  const uint32_t interval = engine::kPrefillCheckpointTokens;
  using benchmark::expectedDraftContextRows;
  // No checkpoint lies within one prefill chunk of an end: not 8192 before
  // the 10K prompt's replay boundary at 9984, nor 12288 before the 14K
  // prompt's at 14080.
  require(expectedDraftContextRows(10000, interval) == 4112,
          "10K cold benchmark omitted rolling recovery windows");
  require(expectedDraftContextRows(14096, interval, 9984) == 2064,
          "4K suffix benchmark planned a checkpoint within a chunk of an end");
  require(expectedDraftContextRows(14096, interval) == 6160,
          "partial-hit benchmark confused cold work with restored work");
  require(expectedDraftContextRows(10000, 0) == 2064 &&
              expectedDraftContextRows(2048, interval) == 2048 &&
              expectedDraftContextRows(10000, interval, 9984) == 16,
          "benchmark changed disabled, short, or exact-hit draft work");
}

void testWideBenchmarkWork() {
  using benchmark::defaultCheckpointTokens;
  using benchmark::expectedDraftContextRows;
  require(defaultCheckpointTokens(2048) == 4096 &&
              defaultCheckpointTokens(16384) == 32768,
          "benchmark checkpoint interval ignored model retention");
  require(expectedDraftContextRows(10000, 32768, 0, 16384) == 10000 &&
              expectedDraftContextRows(14096, 32768, 9984, 16384) == 4112,
          "wide-window benchmark clipped a short prompt or restored suffix");
  // 16K at 32768, another 16K at 49984, then 16 rows to prompt end.
  require(expectedDraftContextRows(50000, 32768, 0, 16384) == 32784 &&
              expectedDraftContextRows(50000, 0, 0, 16384) == 16400,
          "wide-window benchmark miscounted recovery or final replay rows");
  const std::array<uint32_t, 1> replay{14080};
  require(planDraftContext(9984, 14096, replay, 16384).restoresDraftState &&
              !planDraftContext(9984, 14096, replay, 2048).restoresDraftState,
          "4K benchmark suffix ignored whether restored draft rows are still needed");
}

void testLayerRingViews() {
  constexpr uint32_t heads = 2, dimensions = 3;
  const auto encoded = [](uint64_t position, uint32_t head, uint32_t dimension) {
    return static_cast<uint16_t>((position * 17 + head * 7 + dimension) % 65521);
  };
  // Fill each physical ring by writing an entire logical history. The view
  // must return only the latest rows, even when another layer retains more.
  for (uint32_t window : {2048U, 4096U, 16384U}) {
    for (uint32_t end : {0U, 17U, window, window + 37, 2 * window + 11, 50000U}) {
      std::vector<uint16_t> keys(heads * dimensions * window), values(keys.size());
      for (uint32_t position = 0; position < end; ++position)
        for (uint32_t head = 0; head < heads; ++head)
          for (uint32_t dimension = 0; dimension < dimensions; ++dimension) {
            const auto value = encoded(position, head, dimension);
            keys[(head * window + position % window) * dimensions + dimension] = value;
            values[(head * dimensions + dimension) * window + position % window] = value ^ 0xffff;
          }
      for (uint32_t retained : {64U, 2048U, 16384U}) {
        const uint32_t count = std::min({end, retained, window});
        const uint32_t base = end - std::min(end, retained);
        const test::DraftRingView view(keys, values, heads, dimensions, window, base, end);
        require(view.size() == uint64_t{heads} * count * dimensions,
                "layer ring exposed overwritten or unfilled rows");
        uint64_t index = 0;
        for (uint32_t head = 0; head < heads; ++head)
          for (uint32_t position = end - count; position < end; ++position)
            for (uint32_t dimension = 0; dimension < dimensions; ++dimension, ++index) {
              const auto value = encoded(position, head, dimension);
              require(view.position(index) == position && view.key(index) == value &&
                          view.value(index) == (value ^ 0xffff),
                      "layer ring view changed logical order or physical layout");
            }
        rejects([&] { (void)view.key(view.size()); }, "outside retained rows",
                "layer ring accepted an out-of-range logical index");
      }
      rejects([&] {
        test::DraftRingView view(std::span(keys).first(keys.size() - 1), values,
                                 heads, dimensions, window, 0, end);
      }, "smaller than its layer window", "undersized key ring was accepted");
      rejects([&] {
        test::DraftRingView view(keys, std::span(values).first(values.size() - 1),
                                 heads, dimensions, window, 0, end);
      }, "smaller than its layer window", "undersized value ring was accepted");
    }
  }
}

void testJunctionDistancesAndReservation() {
  {
    const auto plan = cachedPlan(4096, 7000, {5000});
    require(
        plan.restoresDraftState,
        "near first junction discarded draft state needed for continuation");
  }
  {
    const auto plan = cachedPlan(0, 6000, {4096});
    require(plan.captureSpans.size() == 1 && plan.captureSpans[0].begin == 2048 &&
                plan.captureSpans[0].end == 6000,
            "near junction did not continue its materialized state");
    require(purposeRows(plan, DraftBoundaryPurpose::Materialization) == 2048 &&
                purposeRows(plan, DraftBoundaryPurpose::Active) == 1904,
            "near junction attribution is wrong");
  }
  {
    const auto plan = cachedPlan(0, 7000, {4096});
    require(plan.captureSpans.size() == 2,
            "far junction did not create a fresh final window");
    require(plan.captureSpans[0].begin == 2048 &&
                plan.captureSpans[0].end == 4096 &&
                plan.captureSpans[1].begin == 4952 &&
                plan.captureSpans[1].end == 7000,
            "far junction capture ranges are wrong");
  }
  {
    const auto plan = activePlan(0, 7000);
    require(plan.captureSpans.size() == 1 && plan.captureSpans[0].begin == 4952,
            "denied junction performed hidden draft work");
    require(purposeRows(plan, DraftBoundaryPurpose::Materialization) == 0,
            "denied junction was counted as materialization work");
  }
}

void testJunctionAndLatestReplayAreBothMaterialized() {
  const auto plan = cachedPlan(0, 10000, {4096, 9984});
  require(plan.boundaries.size() == 3,
          "two cache states and active end were not retained");
  require(
      plan.boundaries[0].boundary == 4096 &&
          plan.boundaries[0].purpose == DraftBoundaryPurpose::Materialization &&
          plan.boundaries[1].boundary == 9984 &&
          plan.boundaries[1].purpose == DraftBoundaryPurpose::Materialization &&
          plan.boundaries[2].boundary == 10000 &&
          plan.boundaries[2].purpose == DraftBoundaryPurpose::Active,
      "cache-state boundaries are not ordered before active prompt end");
  require(plan.captureSpans.size() == 2 && plan.captureSpans[0].begin == 2048 &&
              plan.captureSpans[0].end == 4096 &&
              plan.captureSpans[1].begin == 7936 &&
              plan.captureSpans[1].end == 10000,
          "junction/latest draft windows were not planned minimally");
  require(purposeRows(plan, DraftBoundaryPurpose::Materialization) == 4096 &&
              purposeRows(plan, DraftBoundaryPurpose::Active) == 16 &&
              draftContextRows(plan) == 4112,
          "two-boundary draft work attribution is wrong");
}

void testCompactDispatchCaptures() {
  const auto plan = cachedPlan(0, 7000, {4096});
  const auto first = draftCaptureSpansForDispatch(plan, 2000, 4000);
  require(first.size() == 1 && first[0].absoluteBegin == 2048 &&
              first[0].absoluteEnd == 4000 &&
              first[0].compactDestinationRow == 0 && first[0].resetDraftState &&
              first[0].activeRows == 0 && first[0].materializationRows == 1952,
          "first compact capture span is wrong");
  const auto middle = draftCaptureSpansForDispatch(plan, 4000, 6048);
  require(
      middle.size() == 2 && middle[0].absoluteBegin == 4000 &&
          middle[0].absoluteEnd == 4096 && !middle[0].resetDraftState &&
          middle[0].activeRows == 0 && middle[0].materializationRows == 96 &&
          middle[1].absoluteBegin == 4952 && middle[1].absoluteEnd == 6048 &&
          middle[1].compactDestinationRow == 96 && middle[1].resetDraftState &&
          middle[1].activeRows == 1096 && middle[1].materializationRows == 0,
      "two-span compact capture layout is wrong");
}

void testFourRaggedLanes() {
  constexpr std::array<uint32_t, 4> lengths{33, 2047, 2049, 10000};
  uint64_t capturedRows = 0;
  for (uint32_t length : lengths)
    capturedRows += draftContextRows(activePlan(0, length));
  require(capturedRows == 33 + 2047 + 2048 + 2048,
          "ragged lanes shared or padded draft capture rows");
}

void testSparseCheckpointWindowsAndShortResume() {
  const auto plan =
      cachedPlan(0, 33001, {8192, 16384, 24576, 32768, 32992});
  require(plan.captureSpans.size() == 4 && plan.boundaries.size() == 6,
          "sparse checkpoints were truncated to the old fixed plan size");
  require(draftContextRows(plan) == 4 * 2048 + 233 &&
              purposeRows(plan, DraftBoundaryPurpose::Materialization) ==
                  4 * 2048 + 224 &&
              purposeRows(plan, DraftBoundaryPurpose::Active) == 9,
          "sparse checkpoints rebuilt draft across uncaptured prompt gaps");
  uint64_t dispatchedRows = 0;
  for (uint32_t begin = 0; begin < 33001; begin += 1376) {
    const auto captures = draftCaptureSpansForDispatch(
        plan, begin, std::min<uint32_t>(begin + 1376, 33001));
    for (const auto &capture : captures)
      dispatchedRows += capture.absoluteEnd - capture.absoluteBegin;
  }
  require(dispatchedRows == draftContextRows(plan),
          "ragged dispatch lost or repeated a sparse draft capture");

  for (uint32_t suffix : {1U, 31U, 32U, 2047U, 2048U, 2049U}) {
    const auto resumed = activePlan(16384, 16384 + suffix);
    require(draftContextRows(resumed) == std::min(suffix, 2048U) &&
                resumed.restoresDraftState == (suffix < 2048),
            "checkpoint resume did not preserve the short-suffix draft window");
  }
}

void testCheckpointWindowsAcrossSmallDispatches() {
  const std::array plans{
      cachedPlan(0, 12321, {4096, 8192, 12288, 12320}),
      cachedPlan(8192, 14337, {8224, 10272, 14336}),
      cachedPlan(0, 65, {32, 64}),
  };
  constexpr uint32_t window = model::ExecutionLimits::draftContextTokens;
  for (const auto &plan : plans) {
    for (uint32_t budget : {64U, 256U, 1376U, 2048U}) {
      std::vector<uint32_t> state;
      for (uint32_t row = plan.replayBegin - std::min(plan.replayBegin, window);
           row < plan.replayBegin; ++row)
        state.push_back(row);

      uint32_t begin = plan.replayBegin;
      for (const auto &boundary : plan.boundaries) {
        while (begin < boundary.boundary) {
          const uint32_t end = std::min(begin + budget, boundary.boundary);
          for (const auto &capture : draftCaptureSpansForDispatch(plan, begin,
                                                                end)) {
            if (capture.resetDraftState)
              state.clear();
            for (uint32_t row = capture.absoluteBegin;
                 row < capture.absoluteEnd; ++row)
              state.push_back(row);
          }
          begin = end;
        }
        const uint32_t rows = std::min(boundary.boundary, window);
        require(state.size() >= rows,
                "small dispatches dropped a checkpoint's draft context");
        for (uint32_t row = 0; row < rows; ++row)
          require(state[state.size() - rows + row] ==
                      boundary.boundary - rows + row,
                  "checkpoint draft window contains stale or reordered rows");
      }
    }
  }
}

// A restored plan continues the restored draft ring, and its first capture
// does not reset it, exactly when its first boundary is within one window of
// the restore; a cold plan restores none.
void testRestoresDraftStateFlag() {
  constexpr uint32_t window = model::ExecutionLimits::draftContextTokens;
  for (const auto &[plan, restores] :
       {std::pair{activePlan(4096, 4096 + window - 1), true},
        std::pair{activePlan(4096, 4096 + window), false},
        std::pair{cachedPlan(4096, 9000, {4096 + window - 1}), true},
        std::pair{cachedPlan(4096, 9000, {4096 + window}), false}}) {
    require(plan.restoresDraftState == restores &&
                plan.captureSpans.front().resetDraftState == !restores,
            "restored plan misjudged whether its draft ring continues");
  }
  for (const auto &plan : {activePlan(0, 1), activePlan(0, window - 1),
                           cachedPlan(0, 3 * window, {window})})
    require(!plan.restoresDraftState, "cold plan claimed a restored draft ring");
}

void testInvalidInputs() {
  const auto plan = activePlan(0, 4096);
  rejects([&] { (void)draftCaptureSpansForDispatch(plan, 0, 2049); },
          "invalid target-prefill dispatch range", "oversized dispatch was accepted");
  rejects([] { (void)cachedPlan(0, 4096, {2048, 1024}); },
          "draft materialization boundaries are not sorted and unique",
          "unsorted cache-state boundaries were accepted");
}

} // namespace

int main() {
  try {
    testColdLengths();
    testPartialHit();
    testFinalFullBlockBound();
    testBenchmarkWorkIncludesRecoveryPoints();
    testWideBenchmarkWork();
    testLayerRingViews();
    testJunctionDistancesAndReservation();
    testJunctionAndLatestReplayAreBothMaterialized();
    testCompactDispatchCaptures();
    testFourRaggedLanes();
    testSparseCheckpointWindowsAndShortResume();
    testCheckpointWindowsAcrossSmallDispatches();
    testRestoresDraftStateFlag();
    testInvalidInputs();
    std::cout << "draft context plan tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "draft context plan tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
