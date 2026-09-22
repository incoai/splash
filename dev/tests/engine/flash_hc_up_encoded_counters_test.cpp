// CPU only: graph-construction accounting does not imply GPU submission,
// completion, generated tokens, or numerical acceptance.
#include "flash/FlashForward.hpp"
#include "flash/FlashHCFused.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using splash::flash::FlashHCUpEncodedCounters;
using splash::flash::flashHCUpF32MPPGeometry;
using Counters = FlashHCUpEncodedCounters;
constexpr std::string_view mainUp =
    "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up";
uint64_t checks = 0;

static_assert(noexcept(Counters{}.recordAttempt(0)));
static_assert(noexcept(Counters{}.recordEligible(0)));
static_assert(noexcept(Counters{}.recordCachedEncoded(0)));

void require(bool condition, const char *message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

uint64_t total(const std::array<uint64_t, 18> &histogram) {
  return std::accumulate(histogram.begin(), histogram.end(), uint64_t{0});
}

enum class Outcome { UnsupportedGeometry, DependenciesOff, MissingOperand, CachedEncoded };

// Outcomes are supplied event fixtures. The actual geometry helper classifies
// eligibility; this test does not instantiate or simulate a Metal backend.
void recordEvent(Counters &c, std::string_view prefix, uint32_t rows, Outcome outcome) {
  c.recordAttempt(rows);
  const bool eligible = flashHCUpF32MPPGeometry(prefix, rows, 10240, 320);
  require(eligible == (outcome != Outcome::UnsupportedGeometry),
          "event fixture disagrees with production geometry");
  if (eligible) c.recordEligible(rows);
  switch (outcome) {
  case Outcome::UnsupportedGeometry: ++c.skippedUnsupportedGeometry; break;
  case Outcome::DependenciesOff: ++c.skippedDependenciesOff; break;
  case Outcome::MissingOperand: ++c.skippedMissingOperand; break;
  case Outcome::CachedEncoded: c.recordCachedEncoded(rows); break;
  }
}

void completedAccounting(const Counters &c) {
  require(c.graphBuildAttempts == c.geometryEligibleAttempts + c.skippedUnsupportedGeometry,
          "attempts are not partitioned into eligible and unsupported events");
  require(c.geometryEligibleAttempts == c.cachedEncodedCalls + c.skippedDependenciesOff +
          c.skippedMissingOperand, "eligible events are not partitioned into encoded and skipped events");
  require(c.cachedEncodedRealRows <= c.geometryEligibleRealRows &&
          c.geometryEligibleRealRows <= c.graphBuildRealRows, "real-row accounting lost stage ordering");
  require(total(c.graphBuildCallsByRows) == c.graphBuildAttempts,
          "attempt histogram total disagrees with graph builds");
  require(total(c.cachedEncodedCallsByRows) == c.cachedEncodedCalls,
          "encoded histogram total disagrees with encoded calls");
}

void mixedEventSequence() {
  Counters c;
  recordEvent(c, mainUp, 1, Outcome::UnsupportedGeometry);
  recordEvent(c, mainUp, 4, Outcome::DependenciesOff);
  recordEvent(c, mainUp, 8, Outcome::MissingOperand);
  recordEvent(c, mainUp, 16, Outcome::CachedEncoded);
  recordEvent(c, mainUp, 64, Outcome::UnsupportedGeometry);
  require(c.graphBuildAttempts == 5 && c.graphBuildRealRows == 93,
          "mixed event graph-build totals are wrong");
  require(c.geometryEligibleAttempts == 3 && c.geometryEligibleRealRows == 28,
          "mixed event eligible totals are wrong");
  require(c.cachedEncodedCalls == 1 && c.cachedEncodedRealRows == 16,
          "mixed event encoded totals are wrong");
  require(c.skippedDependenciesOff == 1 && c.skippedMissingOperand == 1 &&
          c.skippedUnsupportedGeometry == 2, "mixed event skip reasons are wrong");
  std::array<uint64_t, 18> expectedAttempts{};
  for (uint32_t bucket : {1u, 4u, 8u, 16u, 17u}) expectedAttempts[bucket] = 1;
  std::array<uint64_t, 18> expectedEncoded{};
  expectedEncoded[16] = 1;
  require(c.graphBuildCallsByRows == expectedAttempts, "mixed event attempt buckets are wrong");
  require(c.cachedEncodedCallsByRows == expectedEncoded, "mixed event encoded buckets are wrong");
  completedAccounting(c);
}

void stagesDoNotAdvanceImplicitly() {
  Counters c;
  c.recordAttempt(16);
  require(c.graphBuildAttempts == 1 && c.graphBuildRealRows == 16,
          "attempt did not record real graph rows");
  require(c.geometryEligibleAttempts == 0 && c.geometryEligibleRealRows == 0 &&
          c.cachedEncodedCalls == 0 && c.cachedEncodedRealRows == 0,
          "an attempt implicitly counted eligibility or encoding");
  require(total(c.cachedEncodedCallsByRows) == 0, "an attempt updated the encoded histogram");
  c.recordEligible(16);
  require(c.graphBuildAttempts == 1 && c.graphBuildRealRows == 16 &&
          c.geometryEligibleAttempts == 1 && c.geometryEligibleRealRows == 16,
          "eligibility altered graph-build accounting");
  require(c.cachedEncodedCalls == 0 && c.cachedEncodedRealRows == 0 &&
          total(c.cachedEncodedCallsByRows) == 0, "eligibility implicitly counted an encoded call");
  c.recordCachedEncoded(16);
  require(c.graphBuildAttempts == 1 && c.geometryEligibleAttempts == 1 &&
          c.cachedEncodedCalls == 1 && c.cachedEncodedRealRows == 16,
          "encoding altered a prior stage or lost its own rows");
  completedAccounting(c);
}

void completeMainHCGraph() {
  Counters c;
  for (uint32_t layer = 0; layer < 48; ++layer)
    for (std::string_view role : {".attn_hyper_connection.input_mix_weight_up",
                                 ".mlp_hyper_connection.input_mix_weight_up"}) {
      const std::string prefix = "language_model.model.layers." + std::to_string(layer) +
                                 std::string(role);
      recordEvent(c, prefix, 16, Outcome::CachedEncoded);
    }
  recordEvent(c, "language_model.model.hyper_connection_mixer.input_mix_weight_up",
              16, Outcome::CachedEncoded);
  require(c.graphBuildAttempts == 97 && c.geometryEligibleAttempts == 97 &&
          c.cachedEncodedCalls == 97, "one main HC graph did not account for 97 blocks");
  require(c.graphBuildRealRows == 1552 && c.geometryEligibleRealRows == 1552 &&
          c.cachedEncodedRealRows == 1552, "main HC graph confused block rows with request rows");
  require(c.skippedDependenciesOff == 0 && c.skippedMissingOperand == 0 &&
          c.skippedUnsupportedGeometry == 0, "fully encoded HC graph has a skipped event");
  std::array<uint64_t, 18> expected{};
  expected[16] = 97;
  require(c.graphBuildCallsByRows == expected && c.cachedEncodedCallsByRows == expected,
          "main HC graph histogram is not exactly 97 R16 calls");
  completedAccounting(c);
}

void unsupportedRoleAndLargeRows() {
  Counters c;
  recordEvent(c, "language_model.model.layers.0.linear_attn.out_proj", 16,
              Outcome::UnsupportedGeometry);
  constexpr uint32_t maximum = std::numeric_limits<uint32_t>::max();
  recordEvent(c, mainUp, maximum, Outcome::UnsupportedGeometry);
  require(c.graphBuildAttempts == 2 && c.graphBuildRealRows == uint64_t{maximum} + 16,
          "unsupported event lost rows or overflowed 32-bit accounting");
  require(c.geometryEligibleAttempts == 0 && c.cachedEncodedCalls == 0 &&
          c.skippedUnsupportedGeometry == 2, "unsupported role/rows reached encoding");
  require(c.graphBuildCallsByRows[16] == 1 && c.graphBuildCallsByRows[17] == 1,
          "unsupported role and large-row event used the wrong buckets");
  completedAccounting(c);
}

void pureHistogramRange() {
  // These methods count supplied metadata independently of geometry. Exercising
  // their full uint32 input range does not describe an eligible GPU encoding.
  Counters attempts, encoded;
  for (uint32_t rows = 0; rows <= 16; ++rows) {
    attempts.recordAttempt(rows);
    encoded.recordCachedEncoded(rows);
  }
  constexpr uint32_t maximum = std::numeric_limits<uint32_t>::max();
  for (uint32_t rows : {17u, 64u, maximum}) {
    attempts.recordAttempt(rows);
    encoded.recordCachedEncoded(rows);
  }
  for (uint32_t bucket = 0; bucket <= 16; ++bucket)
    require(attempts.graphBuildCallsByRows[bucket] == 1 &&
            encoded.cachedEncodedCallsByRows[bucket] == 1, "exact-row histogram bucket is wrong");
  require(attempts.graphBuildCallsByRows[17] == 3 && encoded.cachedEncodedCallsByRows[17] == 3,
          "larger uint32 row counts did not use the bounded overflow bucket");
  const uint64_t realRows = uint64_t{136} + 17 + 64 + maximum;
  require(attempts.graphBuildAttempts == 20 && attempts.graphBuildRealRows == realRows &&
          encoded.cachedEncodedCalls == 20 && encoded.cachedEncodedRealRows == realRows,
          "histogram range totals lost supplied real rows");
  require(total(attempts.graphBuildCallsByRows) == 20 &&
          total(encoded.cachedEncodedCallsByRows) == 20, "histogram range lost calls");
}
} // namespace

int main() {
  try {
    mixedEventSequence(); stagesDoNotAdvanceImplicitly(); completeMainHCGraph();
    unsupportedRoleAndLargeRows(); pureHistogramRange();
    std::cout << "{\"pass\":true,\"cpu_checks\":" << checks << ",\"gpu_commands\":0}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "flash_hc_up_encoded_counters_test: " << error.what() << '\n';
    return 1;
  }
}
