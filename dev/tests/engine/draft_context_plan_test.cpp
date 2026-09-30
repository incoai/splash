#include "benchmarks/PrefillWork.hpp"
#include "engine/Engine.hpp"
#include "model/Model.hpp"
#include "model/WindowPlan.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace splash;

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

DraftContextPlan activePlan(uint32_t replayBegin, uint32_t replayEnd,
                            std::optional<uint32_t> restored = std::nullopt) {
  return planDraftContext(replayBegin, replayEnd, restored, {});
}

DraftContextPlan
cachedPlan(uint32_t replayBegin, uint32_t replayEnd,
           std::optional<uint32_t> restored,
           std::initializer_list<uint32_t> materializationBoundaries) {
  return planDraftContext(
      replayBegin, replayEnd, restored,
      {materializationBoundaries.begin(), materializationBoundaries.size()});
}

void testDemandAwareWindowCoverage() {
  const std::array<uint32_t, 2> optional{5120, 6144};
  const auto absent = planWindow(4096, 8192, 2048, 4096, optional);
  require(absent.usable && absent.loadBegin == 4096 &&
              absent.boundaries.size() == 2 &&
              absent.boundaries.front().tokens == 6144,
          "missing old pages rejected a long suffix or retained an impossible "
          "checkpoint");
  const auto present = planWindow(4096, 8192, 2048, 3072, optional);
  require(present.usable && present.loadBegin == 3072 &&
              present.boundaries.size() == 3,
          "available old suffix did not preserve the early checkpoint");
  require(planWindow(4096, 5120, 2048, 3072).usable &&
              !planWindow(4096, 5120, 2048, 3104).usable,
          "short suffix coverage ignored a needed old page");
  // Independent row coverage oracle across aligned availability and arbitrary
  // P.
  for (uint32_t b : {0u, 32u, 2048u, 4096u})
    for (uint32_t a = b - std::min(b, 2048u); a <= b; a += 32)
      for (uint32_t suffix = 0; suffix <= 2304; suffix += 127) {
        const uint32_t p = b + suffix;
        const auto plan = planWindow(b, p, 2048, a);
        const bool expected = suffix >= 2048 || a <= p - std::min(p, 2048u);
        require(plan.usable == expected,
                "window match disagrees with row coverage");
        if (!plan.usable)
          continue;
        for (uint32_t row = p - std::min(p, 2048u); row < p; ++row) {
          bool covered = row >= plan.loadBegin && row < plan.loadEnd;
          for (const auto &capture : plan.captures)
            covered |= row >= capture.begin && row < capture.end;
          require(covered, "planned boundary reads an unavailable row");
        }
      }
}

void testColdLengths() {
  constexpr std::array<uint32_t, 12> lengths{0,  1,    7,    8,    31,   32,
                                             33, 2047, 2048, 2049, 4096, 10000};
  for (uint32_t length : lengths) {
    const auto plan = activePlan(0, length);
    require(plan.targetPrefillRows == length, "cold target row count changed");
    require(plan.draftContextRows() == std::min<uint32_t>(length, 2048),
            "cold draft window was not clipped to 2048 rows");
    require(plan.draftContextRowsAvoided ==
                length - std::min<uint32_t>(length, 2048),
            "cold avoided-row accounting is wrong");
    require(plan.draftStateResets == (length ? 1U : 0U),
            "cold draft state reset count is wrong");
  }
}

void testPartialHit() {
  {
    const auto plan = activePlan(4096, 5000, 4096);
    require(plan.captureSpans.size() == 1, "short suffix lost capture span");
    require(plan.captureSpans[0].begin == 4096 &&
                plan.captureSpans[0].end == 5000 &&
                !plan.captureSpans[0].resetDraftState,
            "short suffix did not continue restored draft state");
    require(plan.draftContextRows() == 904,
            "short suffix processed more than actual suffix rows");
    require(plan.draftStateRestoreSkipped == 0,
            "short suffix incorrectly skipped restored state");
  }
  {
    const auto plan = activePlan(4096, 8192, 4096);
    require(plan.captureSpans.size() == 1, "long suffix lost capture span");
    require(plan.captureSpans[0].begin == 6144 &&
                plan.captureSpans[0].end == 8192 &&
                plan.captureSpans[0].resetDraftState,
            "long suffix did not rebuild only its final window");
    require(plan.draftContextRows() == 2048,
            "long suffix did not clip draft work");
    require(plan.draftStateRestoreSkipped == 1,
            "long suffix copied a state it necessarily overwrites");
  }
}

void testFinalFullBlockBound() {
  const auto plan = cachedPlan(0, 10000, std::nullopt, {9984});
  require(plan.captureSpans.size() == 1,
          "nearby final boundaries should share one capture span");
  require(plan.captureSpans[0].begin == 7936 &&
              plan.captureSpans[0].end == 10000,
          "10K final/full-block window is wrong");
  require(plan.draftContextRows() == 2064,
          "10K final/full-block work exceeded the planned window");
  require(plan.draftContextRows() <= 2079,
          "ordinary 10K draft work exceeded W+31 rows");

  // When prompt end itself is Page32-aligned, exact lookup must back off one
  // whole input block. Preserving both that cache state and the active prompt
  // state needs W+32 rows; dropping one row would corrupt one of the rings.
  const auto aligned = cachedPlan(0, 4096, std::nullopt, {4064});
  require(aligned.draftContextRows() == 2080,
          "Page32-aligned prompt lost a required draft-context row");
}

void testBenchmarkWorkIncludesRecoveryPoints() {
  const uint32_t interval = engine::EngineConfig{}.prefillCheckpointTokens;
  using benchmark::expectedDraftContextRows;
  require(expectedDraftContextRows(10000, interval) == 5904,
          "10K cold benchmark omitted rolling recovery windows");
  require(expectedDraftContextRows(14096, interval, 9984) == 3856,
          "4K suffix benchmark omitted its intermediate recovery window");
  require(expectedDraftContextRows(14096, interval) == 7952,
          "partial-hit benchmark confused cold work with restored work");
  require(expectedDraftContextRows(10000, 0) == 2064 &&
              expectedDraftContextRows(2048, interval) == 2048 &&
              expectedDraftContextRows(10000, interval, 9984) == 16,
          "benchmark changed disabled, short, or exact-hit draft work");
}

void testJunctionDistancesAndReservation() {
  {
    const auto plan = cachedPlan(4096, 7000, 4096, {5000});
    require(
        plan.draftStateRestoreSkipped == 0,
        "near first junction discarded draft state needed for continuation");
  }
  {
    const auto plan = cachedPlan(0, 6000, std::nullopt, {4096});
    require(plan.captureSpans.size() == 1 && plan.captureSpans[0].begin == 2048 &&
                plan.captureSpans[0].end == 6000,
            "near junction did not continue its materialized state");
    require(plan.draftContextRowsMaterialization == 2048 &&
                plan.draftContextRowsActive == 1904,
            "near junction attribution is wrong");
  }
  {
    const auto plan = cachedPlan(0, 7000, std::nullopt, {4096});
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
    require(plan.draftContextRowsMaterialization == 0,
            "denied junction was counted as materialization work");
  }
}

void testJunctionAndLatestReplayAreBothMaterialized() {
  const auto plan = cachedPlan(0, 10000, std::nullopt, {4096, 9984});
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
  require(plan.draftContextRowsMaterialization == 4096 &&
              plan.draftContextRowsActive == 16 &&
              plan.draftContextRows() == 4112,
          "two-boundary draft work attribution is wrong");
}

void testDispatchPacking() {
  const auto plan = cachedPlan(0, 7000, std::nullopt, {4096});
  const auto first = draftCaptureSpansForDispatch(plan, 2000, 4000);
  require(first.size() == 1 && first[0].absoluteBegin == 2048 &&
              first[0].absoluteEnd == 4000 &&
              first[0].compactDestinationRow == 0 && first[0].resetDraftState &&
              first[0].activeRows == 0 && first[0].materializationRows == 1952,
          "first packed capture span is wrong");
  const auto middle = draftCaptureSpansForDispatch(plan, 4000, 6048);
  require(
      middle.size() == 2 && middle[0].absoluteBegin == 4000 &&
          middle[0].absoluteEnd == 4096 && !middle[0].resetDraftState &&
          middle[0].activeRows == 0 && middle[0].materializationRows == 96 &&
          middle[1].absoluteBegin == 4952 && middle[1].absoluteEnd == 6048 &&
          middle[1].compactDestinationRow == 96 && middle[1].resetDraftState &&
          middle[1].activeRows == 1096 && middle[1].materializationRows == 0,
      "two-span packed capture layout is wrong");
}

void testFourRaggedLanes() {
  constexpr std::array<uint32_t, 4> lengths{33, 2047, 2049, 10000};
  uint64_t packedRows = 0;
  for (uint32_t length : lengths)
    packedRows += activePlan(0, length).draftContextRows();
  require(packedRows == 33 + 2047 + 2048 + 2048,
          "ragged lanes shared or padded draft capture rows");
}

void testSparseCheckpointWindowsAndShortResume() {
  const auto plan = cachedPlan(0, 33001, std::nullopt,
                               {8192, 16384, 24576, 32768, 32992});
  require(plan.captureSpans.size() == 4 && plan.boundaries.size() == 6,
          "sparse checkpoints were truncated to the old fixed plan size");
  require(plan.draftContextRows() == 4 * 2048 + 233 &&
              plan.draftContextRowsMaterialization == 4 * 2048 + 224 &&
              plan.draftContextRowsActive == 9,
          "sparse checkpoints rebuilt draft across uncaptured prompt gaps");
  uint64_t dispatchedRows = 0;
  for (uint32_t begin = 0; begin < 33001; begin += 1376) {
    const auto captures = draftCaptureSpansForDispatch(
        plan, begin, std::min<uint32_t>(begin + 1376, 33001));
    for (const auto &capture : captures)
      dispatchedRows += capture.absoluteEnd - capture.absoluteBegin;
  }
  require(dispatchedRows == plan.draftContextRows(),
          "ragged dispatch lost or repeated a sparse draft capture");

  for (uint32_t suffix : {1U, 31U, 32U, 2047U, 2048U, 2049U}) {
    const auto resumed = activePlan(16384, 16384 + suffix, 16384);
    require(resumed.draftContextRows() == std::min(suffix, 2048U) &&
                resumed.draftStateRestoreSkipped == (suffix >= 2048),
            "checkpoint resume did not preserve the short-suffix draft window");
  }
}

void testCheckpointWindowsAcrossSmallDispatches() {
  const std::array plans{
      cachedPlan(0, 12321, std::nullopt, {4096, 8192, 12288, 12320}),
      cachedPlan(8192, 14337, 8192, {8224, 10272, 14336}),
      cachedPlan(0, 65, std::nullopt, {32, 64}),
  };
  constexpr uint32_t window = model::ExecutionLimits::draftContextTokens;
  for (const auto &plan : plans) {
    for (uint32_t budget : {64U, 256U, 1376U, 2048U}) {
      std::vector<uint32_t> state;
      for (uint32_t row = plan.replayBegin - std::min(plan.replayBegin, window);
           row < plan.replayBegin; ++row)
        state.push_back(row);

      uint32_t begin = plan.replayBegin;
      for (const auto &boundary : plan.plannedBoundaries()) {
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

void testInvalidInputs() {
  bool threw = false;
  try {
    (void)activePlan(32, 64);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "KV-only replay without restore state was accepted");

  threw = false;
  try {
    const auto plan = activePlan(0, 4096);
    (void)draftCaptureSpansForDispatch(plan, 0, 2049);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "oversized packed dispatch was accepted");

  threw = false;
  try {
    (void)cachedPlan(0, 4096, std::nullopt, {2048, 1024});
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "unsorted cache-state boundaries were accepted");
}

} // namespace

int main() {
  try {
    testDemandAwareWindowCoverage();
    testColdLengths();
    testPartialHit();
    testFinalFullBlockBound();
    testBenchmarkWorkIncludesRecoveryPoints();
    testJunctionDistancesAndReservation();
    testJunctionAndLatestReplayAreBothMaterialized();
    testDispatchPacking();
    testFourRaggedLanes();
    testSparseCheckpointWindowsAndShortResume();
    testCheckpointWindowsAcrossSmallDispatches();
    testInvalidInputs();
    std::cout << "draft context plan tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "draft context plan tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
