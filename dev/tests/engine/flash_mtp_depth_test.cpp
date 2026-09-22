#include "flash/FlashMTPDepth.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace {
using namespace splash::flash;
uint64_t checks = 0;
void require(bool value, const char *message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}
void near(double actual, double expected, const char *message) {
  require(std::abs(actual - expected) < 1e-10, message);
}
template <class Operation> void invalid(Operation operation, const char *message) {
  bool failed = false;
  try { operation(); } catch (const std::invalid_argument &) { failed = true; }
  require(failed, message);
}
struct Trajectory {
  std::array<uint64_t, 4> total{}, tail{};
  double elapsed = 0.0;
  uint64_t emitted = 0, probes = 0;
};
Trajectory simulate(FlashMTPDepthController &controller, uint32_t cycles,
    const std::array<double, 3> &probabilities, const std::array<double, 4> &milliseconds,
    uint64_t seed = 7) {
  std::mt19937_64 random(seed);
  Trajectory result;
  for (uint32_t cycle = 0; cycle < cycles; ++cycle) {
    const uint32_t depth = controller.select();
    uint32_t matched = 0;
    for (uint32_t position = 0; position < depth; ++position) {
      if (std::generate_canonical<double, 53>(random) >= probabilities[position]) break;
      ++matched;
    }
    ++result.total[depth];
    if (cycle >= cycles / 2) ++result.tail[depth];
    result.probes += controller.snapshot().probing;
    result.elapsed += milliseconds[depth];
    result.emitted += 1 + matched;
    controller.observe(depth, matched, milliseconds[depth]);
  }
  return result;
}
double expectedOutput(const std::array<double, 3> &probabilities, uint32_t depth) {
  double result = 1.0, prefix = 1.0;
  for (uint32_t position = 0; position < depth; ++position) {
    prefix *= probabilities[position]; result += prefix;
  }
  return result;
}
void calibrated(FlashMTPDepthController &controller, const std::array<double, 4> &costs) {
  std::array<std::optional<double>, 4> values;
  for (size_t depth = 0; depth < values.size(); ++depth) values[depth] = costs[depth];
  controller.seedCosts(values);
}

void guardsAndCensoring() {
  FlashMTPDepthConfig unsupported;
  unsupported.maximumDepth = 4;
  invalid([&] { FlashMTPDepthController controller(unsupported); }, "depth4 must reject");
  FlashMTPDepthConfig badAlpha;
  badAlpha.acceptanceAlpha = 0.0;
  invalid([&] { FlashMTPDepthController controller(badAlpha); }, "zero alpha must reject");
  FlashMTPDepthController controller;
  invalid([&] { controller.observe(3, 4, 10.0); }, "matched cannot exceed proposed");
  invalid([&] { controller.observe(4, 0, 10.0); }, "proposed cannot exceed head cap");
  invalid([&] { controller.observe(1, 1, 0.0); }, "measured zero cost must reject");
  invalid([&] { controller.observe(1, 1, std::numeric_limits<double>::quiet_NaN()); }, "NaN cost must reject");
  invalid([&] { controller.seedCosts({20.0, 30.0, INFINITY, 40.0}); }, "infinite seed must reject");
  require(controller.snapshot().cycles == 0, "failed observations must not mutate counters");
  FlashMTPDepthConfig fast;
  fast.acceptanceAlpha = 0.5;
  FlashMTPDepthController conditional(fast);
  conditional.observe(3, 1, 40.0);
  auto state = conditional.snapshot();
  near(state.conditionalAcceptance[0], 0.8, "accepted first proposal updates p1");
  near(state.conditionalAcceptance[1], 0.3, "first rejected proposal updates p2");
  near(state.conditionalAcceptance[2], 0.6, "unverified suffix must remain censored");
  near(conditional.expectedOutputs(3), 2.184, "prefix probability product must match expected emissions");
  conditional.observe(3, 0, 40.0, false);
  state = conditional.snapshot();
  near(state.conditionalAcceptance[1], 0.3, "p2 censored after first rejection");
  near(state.conditionalAcceptance[2], 0.6, "p3 censored after first rejection");
  near(*state.cycleMilliseconds[3], 40.0, "excluded maintenance cost leaves estimate unchanged");
  require(conditional.select(0) == 0 && conditional.select(1) <= 1, "remaining quota must cap selection");
}

void warmupAndBackoff() {
  FlashMTPDepthController controller;
  const std::array<uint32_t, 6> sweep{3, 2, 1, 0, 0, 0};
  const std::array<double, 6> cost{41.0, 34.0, 29.0, 30.0, 22.0, 27.0};
  for (size_t index = 0; index < sweep.size(); ++index) {
    require(controller.select() == sweep[index], "warmup measures all native shapes and repeated baseline");
    controller.observe(sweep[index], sweep[index], cost[index]);
  }
  require(controller.snapshot().calibrated, "warmup must finish");
  near(*controller.snapshot().cycleMilliseconds[0], 22.0, "fastest startup baseline guards warm-shape bias");
  FlashMTPDepthController excluded;
  excluded.observe(3, 3, 300.0, false);
  require(excluded.select() == 3 && !excluded.snapshot().cycleMilliseconds[3],
      "maintenance must not advance calibration or seed inflated cost");
  FlashMTPDepthController partial;
  partial.seedCosts({22.0, 29.0, {}, 41.0});
  require(partial.select() == 2, "partial seeding calibrates only absent costs");
  partial.observe(2, 2, 34.0);
  require(partial.snapshot().calibrated, "seeded baseline avoids redundant startup steps");

  FlashMTPDepthConfig high;
  high.initialAcceptance = 0.95;
  FlashMTPDepthController changed(high);
  calibrated(changed, {25.0, 28.0, 32.0, 38.0});
  require(changed.select() == 3, "predictable text initially chooses deep");
  changed.observe(3, 0, 38.0);
  changed.observe(3, 0, 38.0);
  require(changed.select() == 1, "two repeated first failures immediately stop deep waste");
  changed.observe(1, 0, 28.0);
  require(changed.select() == 1, "shallow backoff persists until a match or baseline probe");
  changed.observe(1, 0, 28.0);
  require(changed.select() == 0, "four repeated first failures force a real baseline");
  require(changed.snapshot().conditionalAcceptance[0] < 0.15,
      "change detector must retire stale high p1 confidence");
  changed.observe(0, 0, 25.0);
  require(changed.select() == 0, "measured baseline wins after sustained rejection");
}

void spikesAndDrift() {
  FlashMTPDepthConfig high;
  high.initialAcceptance = 0.95;
  FlashMTPDepthController controller(high);
  calibrated(controller, {20.0, 24.0, 28.0, 32.0});
  const auto chosen = controller.select();
  controller.observe(chosen, chosen, 1000.0);
  require(*controller.snapshot().cycleMilliseconds[chosen] < 40.0,
      "one driver/contention pause must not poison a dormant depth for seconds");
  require(controller.snapshot().costAgeMilliseconds[0] >= 1000.0,
      "real elapsed time still ages stale cost measurements despite clipping");
  // True persistent load changes must still converge; spikes are damped, not
  // ignored forever. Direct observations are a valid external measured stream.
  for (uint32_t cycle = 0; cycle < 100; ++cycle) controller.observe(chosen, chosen, 96.0);
  require(*controller.snapshot().cycleMilliseconds[chosen] > 90.0,
      "sustained slower native cycles must be learned");

  FlashMTPDepthConfig moderate;
  moderate.initialAcceptance = 0.9;
  FlashMTPDepthController stale(moderate);
  calibrated(stale, {22.0, 28.0, 45.0, 65.0});
  require(stale.select() == 1, "old inflated deep cost starts shallow");
  const auto adapted = simulate(stale, 4000, {0.99, 0.99, 0.99}, {22.0, 28.0, 30.0, 32.0}, 77);
  require(adapted.tail[3] > 1500, "bidirectional stale probes must discover a newly cheap deeper shape");
  require(adapted.probes < 800, "heavy stale exploration must remain duty bounded");
}

void initialCostConfirmation() {
  FlashMTPDepthConfig high;
  high.initialAcceptance = 0.95;
  FlashMTPDepthController cold(high);
  const std::array<uint32_t, 6> sweep{3, 2, 1, 0, 0, 0};
  const std::array<double, 6> coldCosts{1000.0, 34.0, 29.0, 22.0, 22.0, 22.0};
  for (size_t index = 0; index < sweep.size(); ++index) {
    require(cold.select() == sweep[index], "cold outlier leaves ordinary warmup order intact");
    cold.observe(sweep[index], sweep[index], coldCosts[index]);
  }
  require(cold.select() == 3 && !cold.snapshot().calibrated,
      "extreme speculative startup cost must receive immediate confirmation before sharing costs");
  cold.observe(3, 3, 700.0, false);
  require(cold.select() == 3 && !cold.snapshot().calibrated,
      "excluded cycle must not complete startup confirmation");
  near(*cold.snapshot().cycleMilliseconds[3], 1000.0,
      "excluded startup confirmation must leave its timing unchanged");
  cold.observe(3, 3, 41.0);
  require(cold.snapshot().calibrated && cold.snapshot().cycles <= 12 && cold.select() == 3,
      "unseeded cold outlier must recover the profitable deep shape within twelve cycles");
  near(*cold.snapshot().cycleMilliseconds[3], 41.0,
      "confirmation replaces cold startup timing with minimum of both samples");
  const auto recovered = simulate(cold, 32, {1.0, 1.0, 1.0}, {22.0, 29.0, 34.0, 41.0}, 81);
  require(recovered.tail[3] == 16,
      "short predictable request must retain profitable deep drafting after cold confirmation");

  FlashMTPDepthController expensive(high);
  for (size_t index = 0; index < sweep.size(); ++index)
    expensive.observe(sweep[index], sweep[index], index == 0 ? 120.0 : coldCosts[index]);
  require(expensive.select() == 3 && !expensive.snapshot().calibrated,
      "legitimate expensive shape must receive the same single confirmation");
  expensive.observe(3, 3, 130.0);
  near(*expensive.snapshot().cycleMilliseconds[3], 120.0,
      "confirming a truly expensive shape preserves its measured cost");
  require(expensive.snapshot().calibrated && expensive.select() == 2,
      "confirmed expensive shape must lose to the profitable shallower depth");
  const auto stable = simulate(expensive, 16, {1.0, 1.0, 1.0}, {22.0, 29.0, 34.0, 120.0}, 82);
  require(stable.total[3] == 0,
      "confirmed expensive shape must not trigger another immediate startup confirmation");

  FlashMTPDepthController seeded(high);
  calibrated(seeded, {22.0, 29.0, 34.0, 1000.0});
  require(seeded.snapshot().calibrated && seeded.select() == 2,
      "external measured costs must bypass startup confirmation");
  FlashMTPDepthController partial(high);
  partial.seedCosts({22.0, 29.0, {}, 41.0});
  partial.observe(2, 2, 1000.0);
  require(!partial.snapshot().calibrated && partial.select() == 2,
      "unseeded slot must compare its cold timing against seeded speculative shapes");
  partial.observe(2, 2, 34.0);
  require(partial.snapshot().calibrated,
      "partial seeding must finish after the missing shape is confirmed");
  near(*partial.snapshot().cycleMilliseconds[2], 34.0,
      "partial seeding confirmation must retire its cold initial sample");
  FlashMTPDepthConfig invalidRatio;
  invalidRatio.initialCostConfirmationRatio = 1.0;
  invalid([&] { FlashMTPDepthController controller(invalidRatio); },
      "startup confirmation ratio must exceed one");
  FlashMTPDepthConfig plain;
  plain.maximumDepth = 0;
  FlashMTPDepthController zero(plain);
  for (uint32_t repetition = 0; repetition < 3; ++repetition) zero.observe(0, 0, 22.0);
  require(zero.snapshot().calibrated && zero.select() == 0,
      "depth-zero-only controller must calibrate without speculative comparisons");
  FlashMTPDepthConfig one;
  one.maximumDepth = 1;
  FlashMTPDepthController onlyOne(one);
  onlyOne.observe(1, 1, 1000.0);
  for (uint32_t repetition = 0; repetition < 3; ++repetition) onlyOne.observe(0, 0, 22.0);
  require(onlyOne.snapshot().calibrated && onlyOne.select() == 0,
      "plain baseline shape must not be used to invent a speculative startup comparison");
  FlashMTPDepthController twoCold(high);
  const std::array<double, 6> twoColdCosts{1000.0, 600.0, 29.0, 22.0, 22.0, 22.0};
  for (size_t index = 0; index < sweep.size(); ++index)
    twoCold.observe(sweep[index], sweep[index], twoColdCosts[index]);
  require(twoCold.select() == 3 && !twoCold.snapshot().calibrated,
      "largest initial cost ratio must be confirmed first");
  twoCold.observe(3, 3, 41.0);
  require(twoCold.select() == 2 && !twoCold.snapshot().calibrated,
      "another extreme speculative startup cost must receive its own one-time confirmation");
  twoCold.observe(2, 2, 34.0);
  require(twoCold.snapshot().calibrated && twoCold.select() == 3,
      "calibration must finish once all extreme speculative startup costs are confirmed");
}

void contentChanges() {
  FlashMTPDepthController controller;
  const std::array<double, 4> costs{22.0, 29.0, 34.0, 41.0};
  const auto initial = simulate(controller, 2000, {0.97, 0.95, 0.92}, costs, 55);
  require(initial.tail[3] > 700, "predictable prefix should first settle deep");
  const auto unexpected = simulate(controller, 2000, {0.02, 0.04, 0.01}, costs, 56);
  require(unexpected.tail[0] > 850, "same request must stop waste when content turns unpredictable");
  const auto recovered = simulate(controller, 4000, {0.97, 0.95, 0.92}, costs, 57);
  require(recovered.tail[3] > 1400, "periodic stale probes must restore deep drafts when code returns");
  const auto before = controller.snapshot();
  controller.observe(0, 0, 1.0, false); // Quota permits no speculative row.
  const auto after = controller.snapshot();
  require(after.conditionalAcceptance == before.conditionalAcceptance &&
      after.cycleMilliseconds == before.cycleMilliseconds,
      "quota-clamped terminal row must not poison probabilities or measured costs");
}

Trajectory regime(uint32_t optimal, const std::array<double, 3> &probabilities,
    const std::array<double, 4> &costs, uint64_t seed) {
  uint32_t exactBest = 0;
  double exactScore = 0.0;
  for (uint32_t depth = 0; depth <= 3; ++depth) {
    const double candidate = expectedOutput(probabilities, depth) / costs[depth];
    if (candidate > exactScore) { exactScore = candidate; exactBest = depth; }
  }
  require(exactBest == optimal, "fixture optimum must follow independent expected throughput calculation");
  FlashMTPDepthController controller;
  const auto result = simulate(controller, 6000, probabilities, costs, seed);
  require(result.tail[optimal] > 2100, "controller must spend over70% of settled cycles at profitable depth");
  const double realized = result.emitted / result.elapsed;
  require(realized > exactScore * 0.85, "adaptive realized throughput must stay close to known regime optimum");
  if (optimal == 0 || optimal == 1) {
    const double fixed3 = expectedOutput(probabilities, 3) / costs[3];
    require(realized > fixed3 * 1.2, "adaptive policy must materially beat wasteful fixeddepth3 on rejected text");
  }
  return result;
}
void writeTrajectory(const char *name, const Trajectory &result) {
  std::cout << ",\"" << name << "\":{\"depth_cycles\":[";
  for (size_t i = 0; i < result.total.size(); ++i) { if (i) std::cout << ','; std::cout << result.total[i]; }
  std::cout << "],\"emitted\":" << result.emitted << ",\"cycle_ms\":" << result.elapsed
            << ",\"simulated_tokens_per_second\":" << 1000.0 * result.emitted / result.elapsed
            << ",\"probe_cycles\":" << result.probes << '}';
}
} // namespace

int main() {
  try {
    guardsAndCensoring();
    warmupAndBackoff();
    spikesAndDrift();
    initialCostConfirmation();
    contentChanges();
    const auto baseline = regime(0, {0.08, 0.04, 0.01}, {22.0, 34.0, 46.0, 58.0}, 101);
    const auto prose = regime(1, {0.82, 0.22, 0.1}, {22.0, 29.0, 38.0, 47.0}, 102);
    const auto two = regime(2, {0.9, 0.88, 0.12}, {22.0, 26.0, 29.0, 34.0}, 103);
    const auto code = regime(3, {0.97, 0.95, 0.92}, {22.0, 29.0, 34.0, 41.0}, 104);
    std::cout << "{\"pass\":true,\"gpu_work\":false,\"checks\":" << checks
              << ",\"semantics\":\"" << kFlashMTPDepthSemantics << '"';
    writeTrajectory("baseline", baseline);
    writeTrajectory("prose", prose);
    writeTrajectory("depth2", two);
    writeTrajectory("code", code);
    std::cout << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "flash-mtp-depth-test: " << error.what() << '\n'; return 1;
  }
}
