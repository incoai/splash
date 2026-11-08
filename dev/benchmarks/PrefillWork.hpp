#pragma once

#include "engine/Engine.hpp"
#include "model/Model.hpp"
#include "ops/PagedKv.hpp"

#include <cstdint>
#include <vector>

namespace splash::benchmark {

// The draft context rows a plan captures.
inline uint64_t draftContextRows(const DraftContextPlan &plan) {
  uint64_t rows = 0;
  for (const DraftCaptureSpan &span : plan.captureSpans)
    rows += span.end - span.begin;
  return rows;
}

// Work expected by cold and prefix-hit benchmark requests, including the
// engine's rolling recovery points, its tail checkpoint and final reusable
// replay state.
inline uint64_t expectedDraftContextRows(uint32_t promptTokens,
                                         uint32_t checkpointTokens,
                                         uint32_t restoredTokens = 0) {
  if (!promptTokens)
    return 0;
  const uint32_t replayBoundary =
      (promptTokens - 1) / kv::kPageTokens * kv::kPageTokens;
  std::vector<uint32_t> boundaries =
      engine::plannedCheckpoints(restoredTokens, replayBoundary, checkpointTokens);
  // As Engine::addTailCheckpoint plans it for a request with no junction: a
  // prefill chunk or more past the state before it.
  const uint32_t tail = replayBoundary > engine::kTrimmedTailTokens
                            ? replayBoundary - engine::kTrimmedTailTokens
                            : 0;
  const uint32_t before = boundaries.empty() ? restoredTokens : boundaries.back();
  if (checkpointTokens && tail > restoredTokens &&
      tail >= uint64_t{before} + model::ExecutionLimits::prefillTokenBudget)
    boundaries.push_back(tail);
  if (replayBoundary > restoredTokens)
    boundaries.push_back(replayBoundary);
  return draftContextRows(
      planDraftContext(restoredTokens, promptTokens, boundaries));
}

} // namespace splash::benchmark
