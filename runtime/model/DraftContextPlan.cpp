#include "model/Model.hpp"

#include <algorithm>
#include <stdexcept>

namespace splash {
DraftContextPlan
planDraftContext(uint32_t replayBegin, uint32_t replayEnd,
                 std::span<const uint32_t> materializationBoundaries) {
  if (replayEnd < replayBegin) {
    throw std::invalid_argument("draft replay range is reversed");
  }

  uint32_t previousMaterialization = 0;
  bool havePreviousMaterialization = false;
  for (uint32_t boundary : materializationBoundaries) {
    if (boundary <= replayBegin || boundary > replayEnd) {
      throw std::invalid_argument(
          "draft materialization boundary is outside replay range");
    }
    if (havePreviousMaterialization && boundary <= previousMaterialization) {
      throw std::invalid_argument(
          "draft materialization boundaries are not sorted and unique");
    }
    previousMaterialization = boundary;
    havePreviousMaterialization = true;
  }

  DraftContextPlan result;
  result.replayBegin = replayBegin;
  result.replayEnd = replayEnd;

  constexpr uint32_t window = model::ExecutionLimits::draftContextTokens;
  uint32_t stateBoundary = replayBegin;
  bool haveState = replayBegin != 0;

  const auto addBoundary = [&](uint32_t boundary,
                               DraftBoundaryPurpose purpose) {
    const uint32_t distance = boundary - stateBoundary;
    const bool useRestored = haveState && distance == 0;
    uint32_t captureBegin = boundary;

    if (!useRestored) {
      const bool continueState = haveState && distance < window;
      captureBegin =
          continueState ? stateBoundary : boundary - std::min(boundary, window);
      const uint32_t rows = boundary - captureBegin;

      if (rows != 0 && !result.captureSpans.empty() && continueState &&
          result.captureSpans.back().end ==
              captureBegin) {
        result.captureSpans.back().end = boundary;
      } else if (rows != 0) {
        result.captureSpans.push_back({captureBegin, boundary, !continueState});
      }
    }

    result.boundaries.push_back({boundary, purpose, captureBegin});
    stateBoundary = boundary;
    haveState = true;
  };

  // A cache state at prompt end is the active state itself; do not describe or
  // build the same physical state twice.
  for (uint32_t boundary : materializationBoundaries) {
    if (boundary < replayEnd)
      addBoundary(boundary, DraftBoundaryPurpose::Materialization);
  }
  addBoundary(replayEnd, DraftBoundaryPurpose::Active);

  const uint32_t firstBoundary =
      !materializationBoundaries.empty() &&
              materializationBoundaries.front() < replayEnd
          ? materializationBoundaries.front()
          : replayEnd;
  result.restoresDraftState =
      replayBegin != 0 && firstBoundary - replayBegin < window;
  return result;
}

DispatchDraftCapturePlan
draftCaptureSpansForDispatch(const DraftContextPlan &plan,
                             uint32_t dispatchBegin, uint32_t dispatchEnd) {
  if (dispatchEnd < dispatchBegin || dispatchBegin < plan.replayBegin ||
      dispatchEnd > plan.replayEnd ||
      dispatchEnd - dispatchBegin >
          model::ExecutionLimits::prefillTokenBudget) {
    throw std::invalid_argument("invalid target-prefill dispatch range");
  }

  DispatchDraftCapturePlan result;
  uint32_t compactRow = 0;
  for (const DraftCaptureSpan &span : plan.captureSpans) {
    const uint32_t begin = std::max(span.begin, dispatchBegin);
    const uint32_t end = std::min(span.end, dispatchEnd);
    if (begin >= end)
      continue;
    if (result.count == result.values.size()) {
      throw std::logic_error(
          "more than two draft capture spans intersect one dispatch");
    }
    DispatchDraftCaptureSpan capture{
        begin, end, compactRow, span.resetDraftState && begin == span.begin,
        0,     0};
    for (const DraftBoundaryPlan &boundary : plan.boundaries) {
      const uint32_t segmentBegin = std::max(begin, boundary.captureBegin);
      const uint32_t segmentEnd = std::min(end, boundary.boundary);
      if (segmentBegin >= segmentEnd)
        continue;
      uint32_t &rows = boundary.purpose == DraftBoundaryPurpose::Active
                           ? capture.activeRows
                           : capture.materializationRows;
      rows += segmentEnd - segmentBegin;
    }
    if (capture.activeRows + capture.materializationRows != end - begin) {
      throw std::logic_error("draft capture accounting is incomplete");
    }
    result.values[result.count++] = capture;
    compactRow += end - begin;
  }
  return result;
}

} // namespace splash
