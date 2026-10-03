#include "ScopedTestConfig.hpp"
#include "TestChecks.hpp"
#include "engine/Status.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

using namespace splash;
using namespace splash::engine;

using splash::test::require;

bool close(double left, double right) { return std::abs(left - right) < 1e-9; }

void testLatencyWindowAndThroughput() {
  const test::ScopedTestConfig seam({.metricsLatencyWindow = 3});
  RuntimeMetrics metrics;
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

void testValidation() {
  RuntimeMetrics metrics;
  bool threw = false;
  try {
    metrics.tokens(10.0, 20.0, 1, 19.0);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "backwards token metrics were accepted");
}

} // namespace

int main() {
  try {
    testLatencyWindowAndThroughput();
    testValidation();
    std::cout << "runtime metrics tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "runtime metrics test failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
