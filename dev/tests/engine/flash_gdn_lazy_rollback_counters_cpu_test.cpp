// Counter value tests only: no MetalBackend, device, graph or model instance.
#include "flash/FlashGDNLazyRollback.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace {
using C = splash::flash::FlashGDNLazyRollbackCounters;
uint64_t checks = 0, commitPatterns = 0;
constexpr uint64_t stateBytes = 48ULL * 128 * 128 * sizeof(float);
constexpr uint64_t historyBytes = 3ULL * 10240 * sizeof(uint16_t);
constexpr uint64_t rawRowBytes = 10240ULL * sizeof(uint16_t);
constexpr uint64_t preparedRowBytes = rawRowBytes + 48 * sizeof(float) + 48 * sizeof(uint16_t);
constexpr uint64_t replayRowBytes = 8192ULL * sizeof(uint16_t) + 48 * sizeof(float) + 48 * sizeof(uint16_t);

static_assert(sizeof(splash::metal::CommandTiming) == 200);
static_assert(std::is_same_v<decltype(C::logical_verify_record_write_bytes_avoided), int64_t>);
static_assert(noexcept(C{}.add(C{})));

constexpr auto unsignedFields = std::to_array<uint64_t C::*>({
    &C::layer_trial_graphs_built, &C::trial_lane_rows_planned,
    &C::r1_bypass_trial_graphs, &C::commit_calls,
    &C::no_replay_commit_fastpaths, &C::full_accept_commit_fastpaths,
    &C::full_accepted_lanes, &C::partial_replay_graphs_built,
    &C::partial_replay_layer_lanes_planned, &C::partial_replay_rows_planned,
    &C::terminal_lanes_discarded, &C::all_terminal_commit_calls,
    &C::aborted_trial_calls, &C::aborted_trial_lanes,
    &C::logical_eager_prefix_record_bytes_reference,
    &C::logical_initial_state_snapshot_bytes, &C::logical_initial_history_snapshot_bytes,
    &C::logical_raw_qkv_copy_bytes, &C::logical_saved_prework_footprint_bytes,
    &C::logical_incremental_verify_record_write_bytes,
    &C::logical_replay_initial_state_read_bytes, &C::logical_replay_state_write_bytes,
    &C::logical_replay_history_write_bytes, &C::logical_replay_operand_footprint_bytes});

void require(bool value, const char *message) {
  ++checks; if (!value) throw std::runtime_error(message);
}
void equal(const C &actual, const C &expected) {
  for (auto field : unsignedFields)
    require(actual.*field == expected.*field, "counter field differs");
  require(actual.logical_verify_record_write_bytes_avoided ==
      expected.logical_verify_record_write_bytes_avoided, "signed avoided bytes differ");
  require(actual.layer_trial_graphs_by_rows_and_lanes ==
      expected.layer_trial_graphs_by_rows_and_lanes, "trial geometry histogram differs");
}
template <class Operation> void rejectsUnchanged(C &value, Operation operation) {
  const auto before = value;
  bool rejected = false;
  try { operation(); } catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "invalid counter metadata accepted"); equal(value, before);
}
void trials() {
  C cumulative;
  uint64_t expectedTrialRows = 0;
  for (uint32_t rows = 1; rows <= 16; ++rows)
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      C actual, expected;
      actual.recordTrial(rows, lanes);
      expected.layer_trial_graphs_built = 1;
      expected.trial_lane_rows_planned = uint64_t(rows) * lanes;
      expected.layer_trial_graphs_by_rows_and_lanes[(rows - 1) * 4 + lanes - 1] = 1;
      if (rows == 1) expected.r1_bypass_trial_graphs = 1;
      else {
        const uint64_t state = lanes * stateBytes, history = lanes * historyBytes;
        const uint64_t raw = uint64_t(rows) * lanes * rawRowBytes;
        const uint64_t eager = uint64_t(rows - 1) * lanes * (stateBytes + historyBytes);
        expected.logical_eager_prefix_record_bytes_reference = eager;
        expected.logical_initial_state_snapshot_bytes = state;
        expected.logical_initial_history_snapshot_bytes = history;
        expected.logical_raw_qkv_copy_bytes = raw;
        expected.logical_saved_prework_footprint_bytes = uint64_t(rows) * lanes * preparedRowBytes;
        expected.logical_incremental_verify_record_write_bytes = state + history + raw;
        expected.logical_verify_record_write_bytes_avoided = int64_t(eager) - int64_t(state + history + raw);
      }
      equal(actual, expected);
      cumulative.add(actual); expectedTrialRows += uint64_t(rows) * lanes;
    }
  require(cumulative.layer_trial_graphs_built == 64, "not all trial geometries counted");
  require(cumulative.trial_lane_rows_planned == expectedTrialRows, "cumulative actual row count differs");
  require(cumulative.r1_bypass_trial_graphs == 4, "R1 bypass graph count differs");
  for (auto count : cumulative.layer_trial_graphs_by_rows_and_lanes)
    require(count == 1, "trial histogram index duplicated or omitted");
  C r2; r2.recordTrial(2, 1);
  require(r2.logical_verify_record_write_bytes_avoided == -40960,
          "R2 overhead was clamped or made unsigned");
  C r16; r16.recordTrial(16, 4);
  require(r16.logical_eager_prefix_record_bytes_reference == 192430080,
          "R16/B4 eager logical reference changed");
  require(r16.logical_incremental_verify_record_write_bytes == 14139392,
          "R16/B4 incremental recording bytes changed");
  require(r16.logical_verify_record_write_bytes_avoided == 178290688,
          "R16/B4 signed savings changed");
}
void checkCommit(uint32_t rows, std::span<const uint32_t> retained) {
  C actual, expected;
  actual.recordCommit(rows, retained);
  uint64_t partialLanes = 0, partialRows = 0, fullLanes = 0, terminalLanes = 0;
  for (auto kept : retained) {
    terminalLanes += kept == 0;
    fullLanes += kept == rows;
    if (kept && kept < rows) { ++partialLanes; partialRows += kept; }
  }
  expected.commit_calls = 1;
  expected.full_accepted_lanes = fullLanes;
  expected.terminal_lanes_discarded = terminalLanes;
  expected.full_accept_commit_fastpaths = fullLanes == retained.size();
  expected.all_terminal_commit_calls = terminalLanes == retained.size();
  expected.no_replay_commit_fastpaths = partialLanes == 0;
  expected.partial_replay_graphs_built = partialLanes != 0;
  expected.partial_replay_layer_lanes_planned = partialLanes;
  expected.partial_replay_rows_planned = partialRows;
  expected.logical_replay_initial_state_read_bytes = partialLanes * stateBytes;
  expected.logical_replay_state_write_bytes = partialLanes * stateBytes;
  expected.logical_replay_history_write_bytes = partialLanes * historyBytes;
  expected.logical_replay_operand_footprint_bytes = partialRows * replayRowBytes;
  equal(actual, expected); ++commitPatterns;
}
void commits() {
  std::array<uint32_t, 4> kept{};
  for (uint32_t rows : {1u, 2u, 4u, 8u, 16u})
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      uint32_t patterns = 1;
      for (uint32_t lane = 0; lane < lanes; ++lane) patterns *= rows + 1;
      for (uint32_t encoded = 0; encoded < patterns; ++encoded) {
        auto remaining = encoded;
        for (uint32_t lane = 0; lane < lanes; ++lane) {
          kept[lane] = remaining % (rows + 1); remaining /= rows + 1;
        }
        checkCommit(rows, std::span(kept.data(), lanes));
      }
    }
  C mixed; const std::array<uint32_t, 4> choices{16, 0, 1, 15};
  mixed.recordCommit(16, choices);
  require(mixed.partial_replay_rows_planned == 16 &&
      mixed.partial_replay_layer_lanes_planned == 2 &&
      mixed.logical_replay_operand_footprint_bytes == 266752,
      "mixed full/terminal/partial accounting replayed discarded rows");
}
void aggregationAndAborts() {
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    C actual, expected;
    actual.recordAbort(lanes);
    expected.aborted_trial_calls = 1; expected.aborted_trial_lanes = lanes;
    equal(actual, expected);
  }
  C first, second, expected;
  uint64_t value = 1;
  for (auto field : unsignedFields) {
    first.*field = value; second.*field = 2 * value; expected.*field = 3 * value;
    ++value;
  }
  for (size_t index = 0; index < 64; ++index) {
    first.layer_trial_graphs_by_rows_and_lanes[index] = index + 1;
    second.layer_trial_graphs_by_rows_and_lanes[index] = 3 * (index + 1);
    expected.layer_trial_graphs_by_rows_and_lanes[index] = 4 * (index + 1);
  }
  first.logical_verify_record_write_bytes_avoided = -40960;
  second.logical_verify_record_write_bytes_avoided = 178290688;
  expected.logical_verify_record_write_bytes_avoided = 178249728;
  const auto other = second;
  first.add(second); equal(first, expected); equal(second, other);
  first.add(C{}); equal(first, expected);
  C doubled = expected;
  doubled.add(doubled);
  for (auto field : unsignedFields) require(doubled.*field == 2 * (expected.*field), "self-add lost field");
  for (size_t index = 0; index < 64; ++index)
    require(doubled.layer_trial_graphs_by_rows_and_lanes[index] ==
        2 * expected.layer_trial_graphs_by_rows_and_lanes[index], "self-add lost geometry bin");
  require(doubled.logical_verify_record_write_bytes_avoided ==
      2 * expected.logical_verify_record_write_bytes_avoided, "self-add lost signed savings");
}
void invalidMetadataIsAtomic() {
  C value; value.recordTrial(16, 4);
  const std::array<uint32_t, 4> mixed{16, 0, 1, 15}; value.recordCommit(16, mixed); value.recordAbort(3);
  for (const auto &[rows, lanes] : std::array<std::pair<uint32_t, uint32_t>, 6>{{
      {0, 1}, {17, 1}, {1, 0}, {1, 5}, {UINT32_MAX, 1}, {1, UINT32_MAX}}})
    rejectsUnchanged(value, [&] { value.recordTrial(rows, lanes); });
  for (uint32_t lanes : {0u, 5u, UINT32_MAX})
    rejectsUnchanged(value, [&] { value.recordAbort(lanes); });
  rejectsUnchanged(value, [&] { value.recordCommit(4, {}); });
  const std::array<uint32_t, 5> five{0, 1, 2, 3, 4};
  rejectsUnchanged(value, [&] { value.recordCommit(4, five); });
  for (uint32_t rows : {0u, 17u, UINT32_MAX})
    rejectsUnchanged(value, [&] { value.recordCommit(rows, mixed); });
  for (uint32_t slot = 0; slot < 4; ++slot) {
    auto choices = mixed; choices[slot] = 17;
    rejectsUnchanged(value, [&] { value.recordCommit(16, choices); });
    choices[slot] = UINT32_MAX;
    rejectsUnchanged(value, [&] { value.recordCommit(16, choices); });
  }
}
} // namespace

int main() {
  try {
    C defaultInitialized; equal(defaultInitialized, C{});
    trials(); commits(); aggregationAndAborts(); invalidMetadataIsAtomic();
    std::cout << "{\"pass\":true,\"cpu_checks\":" << checks
        << ",\"trial_geometries\":64,\"commit_patterns\":" << commitPatterns
        << ",\"semantics\":\"CPU layer graph planning and logical footprints\""
           ",\"command_timing_size_bytes\":200,\"metal_backend_constructions\":0,\"gpu_commands\":0}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "flash_gdn_lazy_rollback_counters_cpu_test: " << error.what() << '\n'; return 1;
  }
}
