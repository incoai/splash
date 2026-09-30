#include "model/Model.hpp"
#include "model/WindowPlan.hpp"

#include <algorithm>
#include <stdexcept>

namespace splash {
DraftContextPlan
planDraftContext(uint32_t replayBegin, uint32_t replayEnd,
                 std::optional<uint32_t> restoredDraftBoundary,
                 std::span<const uint32_t> materializationBoundaries,
                 uint32_t availableBegin) {
  if (replayEnd < replayBegin) {
    throw std::invalid_argument("draft replay range is reversed");
  }
  if (restoredDraftBoundary && *restoredDraftBoundary != replayBegin) {
    throw std::invalid_argument(
        "restored draft state must coincide with the replay boundary");
  }
  if (!restoredDraftBoundary && replayBegin != 0) {
    throw std::invalid_argument("nonzero replay requires a restored state");
  }

  const auto window = planWindow(replayBegin, replayEnd,
                                 model::ExecutionLimits::draftContextTokens,
                                 availableBegin, materializationBoundaries);
  if (!window.usable)
    throw std::invalid_argument("restored window cannot reach prompt end");
  DraftContextPlan result;
  result.replayBegin = replayBegin;
  result.replayEnd = replayEnd;
  result.restoredDraftBoundary = restoredDraftBoundary;
  result.oldWindowBegin = window.loadBegin;
  result.targetPrefillRows = replayEnd - replayBegin;
  result.draftStateRestoreSkipped =
      replayBegin && window.loadBegin == replayBegin;
  for (const auto &capture : window.captures) {
    result.captureSpans.push_back({capture.begin, capture.end, capture.reset});
    result.draftStateResets += capture.reset;
  }
  for (const auto &boundary : window.boundaries) {
    const bool active = boundary.tokens == replayEnd;
    result.boundaries.push_back({boundary.tokens,
                                 active ? DraftBoundaryPurpose::Active
                                        : DraftBoundaryPurpose::Materialization,
                                 boundary.captureBegin});
    (active ? result.draftContextRowsActive
            : result.draftContextRowsMaterialization) +=
        boundary.tokens - boundary.captureBegin;
  }
  result.draftContextRowsAvoided =
      result.targetPrefillRows > result.draftContextRows()
          ? result.targetPrefillRows - result.draftContextRows()
          : 0;
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
  for (const DraftCaptureSpan &span : plan.captures()) {
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
    for (const DraftBoundaryPlan &boundary : plan.plannedBoundaries()) {
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
