#include "EngineTestSupport.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestMetalMemory.hpp"
#include "engine/Engine.hpp"
#include "engine/RuntimeMetrics.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace splash;
using namespace splash::engine;

using splash::test::rejects;
using splash::test::require;

bool close(double left, double right) { return std::abs(left - right) < 1e-9; }

void testLatencyWindowAndThroughput() {
  RuntimeMetrics metrics(3);
  for (uint64_t id = 1; id <= 4; ++id) {
    double submitted = double(id) * 100.0;
    const double first = submitted + double(id) * 10.0;
    metrics.tokens(submitted, std::nullopt, 1, first);
    metrics.tokens(submitted, first, 2, first + 8.0);
  }

  metrics.batchCompleted(WorkKind::Prefill, 1, 1000, 0, 0, 0, 100.0, 110.0);
  metrics.batchCompleted(WorkKind::Decode, 4, 0, 8, 28, 4, 40.0, 45.0);
  metrics.capacityFailed();
  metrics.metalFailed();

  RuntimeMetricsSnapshot snapshot = metrics.snapshot();
  require(close(snapshot.ttftP50Milliseconds, 30.0) &&
              close(snapshot.ttftP95Milliseconds, 40.0) &&
              snapshot.ttftSamples == 3,
          "bounded TTFT percentile window is incorrect");
  require(close(snapshot.itlP50Milliseconds, 4.0) &&
              close(snapshot.itlP95Milliseconds, 4.0) &&
              snapshot.itlSamples == 3,
          "per-token ITL samples are incorrect");
  require(close(snapshot.prefillTokensPerSecond, 10000.0) &&
              close(snapshot.decodeTokensPerSecond, 200.0) &&
              snapshot.prefillInputTokens == 1000 &&
              close(snapshot.prefillWallMilliseconds, 100.0) &&
              snapshot.decodeOutputTokens == 8 &&
              close(snapshot.decodeWallMilliseconds, 40.0),
          "batch throughput metrics are incorrect");
  require(close(snapshot.decodeCycleMilliseconds, 45.0),
          "the decode cycle did not count decode commands alone");
  require(snapshot.draftedTokens == 28 && snapshot.acceptedDraftTokens == 4 &&
              close(snapshot.draftAcceptanceRate, 1.0 / 7.0) &&
              snapshot.capacityFailures == 1 && snapshot.metalFailures == 1,
          "lifetime runtime counters are incorrect");
  require(snapshot.currentPrefillBatch.valid &&
              snapshot.currentPrefillBatch.width == 1 &&
              snapshot.currentPrefillBatch.inputTokens == 1000 &&
              close(snapshot.currentPrefillBatch.tokensPerSecond, 10000.0) &&
              snapshot.currentDecodeBatch.valid &&
              snapshot.currentDecodeBatch.width == 4 &&
              snapshot.currentDecodeBatch.outputTokens == 8 &&
              snapshot.currentDecodeBatch.draftedTokens == 28 &&
              snapshot.currentDecodeBatch.acceptedDraftTokens == 4 &&
              close(snapshot.currentDecodeBatch.tokensPerSecond, 200.0),
          "current prefill/decode batch samples are incomplete");
}

// A request's gap between two token events is the whole interval, however
// many tokens the later event carries; its ITL divides the interval by them.
// The first event is its TTFT, not a gap.
void testDecodeGapIsTheWholeInterval() {
  RuntimeMetrics metrics;
  metrics.tokens(0.0, std::nullopt, 1, 500.0);
  metrics.tokens(0.0, 500.0, 4, 1500.0);
  const RuntimeMetricsSnapshot snapshot = metrics.snapshot();
  require(snapshot.decodeGap.count == 1 && close(snapshot.decodeGap.maximumMilliseconds, 1000.0) &&
              close(snapshot.decodeGap.totalMilliseconds, 1000.0),
          "the decode gap is not the whole interval between token events");
  require(snapshot.itlSamples == 1 && close(snapshot.itlP50Milliseconds, 250.0),
          "the ITL does not divide the interval by the event's tokens");
}

// Each tick that submits a command records its host time, from its first
// planning call to the submission; a tick that only takes a command's
// completion plans nothing and records none.
void testTickHostTimeIsRecorded() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  Cache cache(pool, nullptr, nullptr);
  test::Executor model;
  test::Events events;
  Engine engine(test::engineConfig(), cache, model, events);
  test::guardReleases(storage, engine);
  engine.submit(test::request(1, std::vector<uint32_t>(64, 7)));
  static_cast<void>(engine.tick(1));
  require(engine.commandInFlight() && engine.snapshot().tickHostMilliseconds.count == 1,
          "the tick that submitted a command did not record its host time");
  test::runUntilIdle(engine);
  require(engine.snapshot().tickHostMilliseconds.count == events.cycles.size(),
          "the ticks' host times are not one per command submitted");
}

// Percentiles over a window that holds no sample are refused at construction.
void testEmptyLatencyWindowIsRefused() {
  rejects([] { RuntimeMetrics metrics(0); }, "the latency window must hold a sample",
          "metrics with an empty latency window were built");
}

void testValidation() {
  RuntimeMetrics metrics;
  rejects([&] { metrics.tokens(10.0, 20.0, 1, 19.0); }, "metrics token clock moved backwards",
          "backwards token metrics were accepted");
}

} // namespace

int main() {
  try {
    testLatencyWindowAndThroughput();
    testDecodeGapIsTheWholeInterval();
    testTickHostTimeIsRecorded();
    testEmptyLatencyWindowIsRefused();
    testValidation();
    std::cout << "runtime metrics tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "runtime metrics test failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
