#include "tuning/AttentionTuning.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using namespace splash;
using namespace splash::ops;
using namespace splash::ops::tuning;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
template <typename Function> void rejects(Function function) {
  bool rejected = false;
  try { function(); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "invalid attention fixture was accepted");
}
uint64_t align(uint64_t bytes) { return (bytes + 16383) / 16384 * 16384; }

// Synthetic CPU timings exercise the selector; they are never submitted,
// persisted, or reported as measured GPU evidence.
MeasurementResult measured(CandidateId candidate, double gpuGain, double wallGain) {
  MeasurementResult result;
  result.candidate = candidate;
  result.workload = {0};
  result.status = MeasurementStatus::Completed;
  result.pairCount = kMinPairedSamples;
  for (size_t pair = 0; pair < result.pairCount; ++pair) {
    result.gpuPairs[pair] = {1.0, 1.0 - gpuGain, measurementOrder(pair), false};
    result.wallPairs[pair] = {2.0, 2.0 * (1.0 - wallGain), measurementOrder(pair), false};
  }
  result.gpuAssessment = evaluate(result.rawGpuSamples());
  result.wallAssessment = evaluate(result.rawWallSamples());
  if (!result.gpuAssessment.qualified() || !result.wallAssessment.qualified())
    result.status = MeasurementStatus::Rejected;
  return result;
}

std::vector<PrefillAttentionTuningResult> prefillEvidence(AttentionShape shape,
    double moreGpu = 0.20, double moreWall = 0.20) {
  std::vector<PrefillAttentionTuningResult> probes;
  for (const auto workload : prefillAttentionPolicyWorkloads(shape)) {
    PrefillAttentionTuningResult probe{{workload, {}}, {}, true, {}};
    const kv::Layout layout{1, shape.kvHeads, shape.headDimension};
    const auto baseline = PagedAttention::prefillPlan(workload.rows, shape.queryHeads, layout,
                                                     workload.historyTokens);
    const auto configs = PagedAttention::prefillCandidates();
    for (uint32_t candidate = 1; candidate < configs.size(); ++candidate) {
      const auto plan = PagedAttention::prefillPlan(workload.rows, shape.queryHeads, layout,
                                                  workload.historyTokens, configs[candidate]);
      if (baseline.sameExecutionAs(plan))
        probe.equivalentCandidates.push_back({candidate});
      else
        probe.measurements.push_back(measured({candidate}, candidate == 1 ? moreGpu : 0,
                                                candidate == 1 ? moreWall : 0));
    }
    probes.push_back(std::move(probe));
  }
  return probes;
}

void planExecutionTests() {
  for (auto shape : {AttentionShape{24, 4, 256}, AttentionShape{16, 2, 256}}) {
    const kv::Layout layout{1, shape.kvHeads, shape.headDimension};
    for (uint32_t rows : {1U, 7U, 8U}) {
      const auto baseline = PagedAttention::prefillPlan(rows, shape.queryHeads, layout, 33);
      const auto alias = PagedAttention::prefillPlan(rows, shape.queryHeads, layout, 33,
                                                    {PrefillSplitMultiplier::Two});
      require(baseline.configuration != alias.configuration &&
                  baseline.sameExecutionAs(alias) && alias.sameExecutionAs(baseline) &&
                  alias.sameExecutionAs(alias),
              "resolved prefill policy alias was treated as different execution");
      const auto cooperative = PagedAttention::prefillPlan(
          rows, shape.queryHeads, layout, 33,
          {PrefillSplitMultiplier::One, AttentionScalePlacement::Cooperative});
      require(!baseline.sameExecutionAs(cooperative) &&
                  !cooperative.sameExecutionAs(baseline) &&
                  cooperative.workspace.partialsBytes == baseline.workspace.partialsBytes &&
                  cooperative.workspace.statisticsBytes == baseline.workspace.statisticsBytes,
              "scale placement was aliased or changed prefill workspace");
    }
    for (uint32_t rows : {65U, 512U, 2048U}) {
      const auto baseline = PagedAttention::prefillPlan(rows, shape.queryHeads, layout, 131072);
      const auto changed = PagedAttention::prefillPlan(rows, shape.queryHeads, layout, 131072,
                                                      {PrefillSplitMultiplier::Two});
      require(!baseline.sameExecutionAs(changed) && !changed.sameExecutionAs(baseline),
              "different prefill execution was treated as a policy alias");
    }
    const auto first = PagedAttention::prefillPlan(1, shape.queryHeads, layout, 33);
    const auto second = PagedAttention::prefillPlan(2, shape.queryHeads, layout, 33);
    require(first.splits == second.splits &&
                !first.sameExecutionAs(second),
            "different active rows were ignored by prefill execution comparison");
    require(!first.sameExecutionAs(PagedAttention::prefillPlan(1, shape.queryHeads, layout, 34)),
            "different logical history was ignored by execution comparison");
  }
  require(!PagedAttention::prefillPlan(512, 24, {1, 4, 256}, 0).sameExecutionAs(
              PagedAttention::prefillPlan(512, 16, {1, 2, 256}, 0)),
          "different attention pipelines/geometry were treated as identical");
}

void verifyExecutionTests() {
  for (auto shape : {AttentionShape{24, 4, 256}, AttentionShape{16, 2, 256}}) {
    const kv::Layout layout{1, shape.kvHeads, shape.headDimension};
    const auto baseline = VerifyAttentionConfig{};
    auto alternative = baseline;
    alternative.splitCount = VerifySplitCount::Eight;
    const auto configs = verifyAttentionTuningCandidates(baseline);
    const CandidateId alternativeId{static_cast<uint32_t>(
        std::find(configs.begin(), configs.end(), alternative) - configs.begin())};
    for (uint32_t lanes : {3U, 4U}) {
      const auto workloads = verifyAttentionPolicyWorkloads({shape, lanes});
      std::vector<VerifyAttentionTuningResult> probes;
      for (const auto &workload : workloads) {
        const auto base = PagedAttention::verifyPlan(
            lanes, shape.queryHeads, layout, workload.historyTokens, baseline);
        const auto candidate = PagedAttention::verifyPlan(
            lanes, shape.queryHeads, layout, workload.historyTokens, alternative);
        require(base.sameExecutionAs(base), "verify execution is not reflexive");
        if (workload == workloads[3]) {
          // At 128K all lanes saturate at 128 splits, regardless of the
          // base split count. These different configurations really alias.
          require(base.configuration != candidate.configuration &&
                      base.sameExecutionAs(candidate) && candidate.sameExecutionAs(base),
                  "identical resolved verify execution was not recognized");
          auto cooperative = alternative;
          cooperative.scalePlacement = AttentionScalePlacement::Cooperative;
          require(!base.sameExecutionAs(PagedAttention::verifyPlan(
                      lanes, shape.queryHeads, layout, workload.historyTokens, cooperative)),
                  "verify scale placement was treated as equivalent");
        }
        if (workload == workloads[4] || workload == workloads[5]) {
          require(base.splits == candidate.splits &&
                      base.splitGroups.x == candidate.splitGroups.x &&
                      base.splitGroups.y == candidate.splitGroups.y &&
                      base.splitGroups.z == candidate.splitGroups.z &&
                      base.laneSplits != candidate.laneSplits &&
                      !base.sameExecutionAs(candidate) && !candidate.sameExecutionAs(base),
                  "mixed/reversed verify histories hid a different lane partition");
        }
        VerifyAttentionTuningResult probe{{workload, baseline}, {}, true, {}};
        for (uint32_t id = 1; id < configs.size(); ++id) {
          const auto plan = PagedAttention::verifyPlan(
              lanes, shape.queryHeads, layout, workload.historyTokens, configs[id]);
          if (base.sameExecutionAs(plan))
            probe.equivalentCandidates.push_back({id});
          else
            probe.measurements.push_back(measured({id}, id == alternativeId.value ? 0.2 : -0.1,
                                                     id == alternativeId.value ? 0.2 : -0.1));
        }
        probes.push_back(std::move(probe));
      }
      require(selectVerifyAttentionPolicy(probes, baseline) == alternative,
              "valid verify aliases prevented a qualified policy selection");
      for (size_t mixed : {4U, 5U}) {
        auto forged = probes;
        std::erase_if(forged[mixed].measurements, [&](const auto &measurement) {
          return measurement.candidate == alternativeId;
        });
        forged[mixed].equivalentCandidates.push_back(alternativeId);
        require(selectVerifyAttentionPolicy(forged, baseline) == baseline,
                "verify selector accepted unmeasured mixed-lane execution as an alias");
      }
    }
  }
  constexpr std::array histories{131072U, 131072U, 131072U, 131072U};
  const auto base = PagedAttention::verifyPlan(4, 24, {1, 4, 256}, histories);
  require(!base.sameExecutionAs(PagedAttention::verifyPlan(3, 24, {1, 4, 256}, histories)) &&
              !base.sameExecutionAs(PagedAttention::verifyPlan(4, 16, {1, 2, 256}, histories)),
          "different verify batch width or attention geometry was treated as equivalent");
}

void policyTests() {
  static_assert(!std::is_convertible_v<PrefillAttentionTuningResult::ProbeChoice,
                                       PrefillAttentionChoice>);
  static_assert(!std::is_convertible_v<VerifyAttentionTuningResult::ProbeChoice,
                                       VerifyAttentionChoice>);
  for (auto shape : {AttentionShape{24, 4, 256}, AttentionShape{16, 2, 256}}) {
    auto probes = prefillEvidence(shape);
    constexpr std::array histories{0U, 2048U, 16384U, 131072U};
    require(probes.size() == histories.size() &&
                PagedAttention::prefillCandidates().size() == 4,
            "prefill calibration is not a fixed bounded comparison");
    for (size_t i = 0; i < probes.size(); ++i)
      require(probes[i].choice.workload.rows == 2048 &&
                  probes[i].choice.workload.historyTokens == histories[i] &&
                  probes[i].measurements.size() == 3 && probes[i].equivalentCandidates.empty(),
              "prefill calibration changed rows, histories or distinct candidate count");
    require(selectPrefillAttentionPolicy(probes).splitMultiplier == PrefillSplitMultiplier::Two,
            "complete fixed-chunk evidence did not select the qualified candidate");
    for (uint32_t winner = 2; winner < PagedAttention::prefillCandidates().size(); ++winner) {
      auto cooperative = probes;
      for (auto &probe : cooperative)
        for (auto &measurement : probe.measurements)
          measurement = measured(measurement.candidate,
              measurement.candidate.value == winner ? 0.2 : 0,
              measurement.candidate.value == winner ? 0.2 : 0);
      require(selectPrefillAttentionPolicy(cooperative) == PagedAttention::prefillCandidates()[winner],
              "qualified cooperative-scale prefill candidate could not be selected");
    }
    auto broken = probes;
    broken.pop_back();
    require(selectPrefillAttentionPolicy(broken) == PrefillAttentionConfig{},
            "partial policy suite selected an override");
    broken = probes;
    broken[3].complete = false;
    require(selectPrefillAttentionPolicy(broken) == PrefillAttentionConfig{},
            "incomplete required probe selected an override");
    broken = probes;
    broken[0].measurements[0] = measured({1}, -0.1, 0.2);
    require(selectPrefillAttentionPolicy(broken) == PrefillAttentionConfig{},
            "one regressed required workload was hidden by policy averaging");
    broken = probes;
    broken[0].measurements.erase(broken[0].measurements.begin());
    require(selectPrefillAttentionPolicy(broken) == PrefillAttentionConfig{},
            "missing timing was treated as unchanged evidence");
    broken[0].equivalentCandidates.push_back({1});
    require(selectPrefillAttentionPolicy(broken) == PrefillAttentionConfig{},
            "forged equivalence bypassed typed plan validation");
    broken = probes;
    broken.back().measurements.push_back(measured({1}, 0, 0));
    require(selectPrefillAttentionPolicy(broken) == PrefillAttentionConfig{},
            "duplicate candidate timings were accepted");
    broken = probes;
    broken[2].choice.workload.historyTokens += 1;
    require(selectPrefillAttentionPolicy(broken) == PrefillAttentionConfig{},
            "wrong representative was accepted into the required suite");
    broken = probes;
    broken[2].choice.workload.rows = 2047;
    require(selectPrefillAttentionPolicy(broken) == PrefillAttentionConfig{},
            "ragged rows were accepted into fixed-chunk calibration");
    broken = prefillEvidence(shape, 0.25, -0.08);
    require(selectPrefillAttentionPolicy(broken) == PrefillAttentionConfig{},
            "GPU/wall policy winner disagreement was ignored");
    const auto operators = PagedAttention::verifyCandidates();
    require(operators.size() == 8 &&
                operators[2] == VerifyAttentionConfig{VerifySplitCount::Eight},
            "verify calibration omitted a split/scale combination");
    const auto baseline = VerifyAttentionConfig{};
    const auto candidates = verifyAttentionTuningCandidates(baseline);
    std::vector<VerifyAttentionConfig> behind(operators.begin(), operators.end());
    std::erase(behind, baseline);
    require(candidates.size() == operators.size() && candidates.front() == baseline &&
                std::equal(candidates.begin() + 1, candidates.end(),
                           behind.begin(), behind.end()),
            "verify tuning order did not lead with the default");
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const auto workloads = verifyAttentionPolicyWorkloads({shape, lanes});
      require(workloads.size() == (lanes == 1 ? 4U : 6U), "verify policy suite is not bounded");
      std::vector<VerifyAttentionTuningResult> verify;
      for (const auto workload : workloads) {
        VerifyAttentionTuningResult probe{{workload, baseline}, {}, true, {}};
        for (uint32_t candidate = 1; candidate < candidates.size(); ++candidate)
          probe.measurements.push_back(measured({candidate}, candidate == 2 ? 0.15 : -0.1,
                                                 candidate == 2 ? 0.15 : -0.1));
        verify.push_back(std::move(probe));
      }
      require(selectVerifyAttentionPolicy(verify, baseline) == candidates[2] &&
                  candidates[2].splitCount == VerifySplitCount::Eight,
              "verify policy ignored complete multi-history evidence");
      for (uint32_t winner = 1; winner < candidates.size(); ++winner) {
        auto cooperative = verify;
        for (auto &probe : cooperative)
          for (auto &measurement : probe.measurements)
            measurement = measured(measurement.candidate,
                measurement.candidate.value == winner ? 0.2 : 0,
                measurement.candidate.value == winner ? 0.2 : 0);
        require(selectVerifyAttentionPolicy(cooperative, baseline) == candidates[winner],
                "qualified verify candidate could not be selected");
      }
      if (lanes > 1) {
        require(workloads[4].historyTokens[0] != workloads[4].historyTokens[lanes - 1] &&
                    workloads[5].historyTokens[0] == workloads[4].historyTokens[lanes - 1],
                "verify policy omitted mixed or reversed lane histories");
        verify.back().measurements[1] = measured({2}, 0.15, -0.1);
        require(selectVerifyAttentionPolicy(verify, baseline) == baseline,
                "mixed-lane wall regression was hidden by homogeneous timings");
      }
      verify.pop_back();
      require(selectVerifyAttentionPolicy(verify, baseline) == baseline,
              "verify policy selected before every required history finished");
    }
  }
  rejects([] { (void)verifyAttentionTuningCandidates({VerifySplitCount(0)}); });
  rejects([] {
    (void)verifyAttentionTuningCandidates({VerifySplitCount::ThirtyTwo,
                                         AttentionScalePlacement(2)});
  });
}

uint64_t expectedBytes(AttentionShape shape, uint32_t rows, uint32_t lanes,
                       const std::array<uint32_t, 4> &histories, bool verify) {
  uint32_t pages = 0;
  uint64_t tables = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const uint32_t lanePages = (histories[lane] + rows + 31) / 32;
    pages += lanePages;
    tables += align(uint64_t{lanePages} * 4);
  }
  const uint32_t physicalPages = pages + 1 + pages % 2;
  const uint64_t data = uint64_t{physicalPages} * shape.kvHeads * 32 * 256;
  const uint64_t scales = uint64_t{physicalPages} * shape.kvHeads * 32 * 4;
  const uint32_t stride = (rows + 31) / 32 * 32;
  const uint64_t chunks = uint64_t{lanes} * shape.kvHeads * stride * 256 * 2;
  const uint64_t queries = uint64_t{lanes} * shape.queryHeads * stride * 256 * 2;
  // Verify scratch covers the maximum history-scaled split count.
  uint32_t slots = kv::kQ8VerifyMaximumSplits;
  if (!verify) {
    const uint32_t tiles = (rows + 7) / 8;
    const uint32_t baselineSplits = std::clamp(32U / tiles, 1U, 32U);
    slots = tiles * std::min(32U, baselineSplits * 2);
  }
  const uint64_t partials = uint64_t{lanes} * slots * 8 *
                             shape.queryHeads * 256 * 4;
  const uint64_t statistics = uint64_t{lanes} * slots * 8 *
                               shape.queryHeads * 2 * 4;
  return 2 * align(data) + 2 * align(scales) + 2 * align(chunks) +
         3 * align(queries) + align(partials) + align(statistics) + tables;
}

void cpuTests() {
  require(measurementBatchRepetitions(0.0001) == 16 &&
              measurementBatchRepetitions(0.001) == 5 &&
              measurementBatchRepetitions(0.006) == 1,
          "whole-graph repetition count did not use the bounded GPU pilot policy");
  for (auto shape : {AttentionShape{24, 4, 256}, AttentionShape{16, 2, 256}}) {
    for (uint32_t rows : {1U, 7U, 8U, 9U, 17U, 33U, 2048U})
      for (uint32_t history : {0U, 1U, 31U, 32U, 33U, 2048U, 8192U,
                               kv::kMaximumPhysicalTokens - 2048}) {
        const auto actual = prefillAttentionTuningFixtureBytes({shape, rows, history});
        require(actual == expectedBytes(shape, rows, 1, {history, 0, 0, 0}, false),
                "prefill fixture admission differs from exact candidate tensor bound");
        require(actual % 16384 == 0, "fixture allocation is not page aligned");
      }
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      std::array<uint32_t, 4> histories{};
      for (uint32_t lane = 0; lane < lanes; ++lane) histories[lane] = lane * 137 + 33;
      require(verifyAttentionTuningFixtureBytes({shape, lanes, histories}) ==
                  expectedBytes(shape, 8, lanes, histories, true),
              "verify fixture did not preserve heterogeneous exact lane histories");
      for (uint32_t lane = 0; lane < lanes; ++lane)
        histories[lane] = kv::kMaximumPhysicalTokens - 8;
      require(verifyAttentionTuningFixtureBytes({shape, lanes, histories}) ==
                  expectedBytes(shape, 8, lanes, histories, true),
              "maximum physical history fixture overflowed");
    }
    rejects([&] { (void)prefillAttentionTuningFixtureBytes({shape, 0, 0}); });
    rejects([&] { (void)prefillAttentionTuningFixtureBytes({shape, 2049, 0}); });
    rejects([&] { (void)prefillAttentionTuningFixtureBytes({shape, 1, UINT32_MAX}); });
    rejects([&] { (void)verifyAttentionTuningFixtureBytes({shape, 0, {}}); });
    rejects([&] { (void)verifyAttentionTuningFixtureBytes({shape, 5, {}}); });
    rejects([&] { (void)verifyAttentionTuningFixtureBytes({shape, UINT32_MAX, {}}); });
    rejects([&] { (void)verifyAttentionTuningFixtureBytes({shape, 1, {0, 1, 0, 0}}); });
    rejects([&] { (void)verifyAttentionTuningFixtureBytes({shape, 1, {UINT32_MAX, 0, 0, 0}}); });
  }
  rejects([] { (void)prefillAttentionTuningFixtureBytes({{32, 4, 256}, 1, 0}); });
  rejects([] { (void)verifyAttentionTuningFixtureBytes({{24, 4, 128}, 1, {}}); });
  rejects([] { (void)verifyAttentionTuningFixtureBytes({{24, 0, 256}, 1, {}}); });
  rejects([] { (void)verifyAttentionTuningFixtureBytes({{}, 1, {}}); });
}

// candidates is the tuning order the result's IDs index: the baseline first.
template <typename Result, typename Candidates>
void completed(const Result &result, const MeasurementOptions &options,
               const Candidates &candidates) {
  if (result.failure) std::rethrow_exception(result.failure);
  require(result.complete && !result.measurements.empty(),
          "attention tuner did not finish its bounded sweep");
  require(result.repetitions >= 1 && result.repetitions <= 16,
          "attention tuner omitted its whole-graph batch repetition metadata");
  for (const auto &m : result.measurements) {
    require(m.candidate != kBaseline && m.candidate.value < candidates.size(),
            "attention tuner measured an ID outside its baseline-first order");
    require((m.status == MeasurementStatus::Completed || m.status == MeasurementStatus::Rejected) &&
                m.pairCount == kMinPairedSamples &&
                m.warmup.returnedCalls == 2 && m.measurement.returnedCalls == 24,
            "attention tuner omitted raw production command timing pairs");
    for (size_t pair = 0; pair < m.pairCount; ++pair)
      require(m.gpuPairs[pair].first == measurementOrder(pair) &&
                  m.wallPairs[pair].first == measurementOrder(pair),
              "attention tuner changed paired measurement order");
  }
  std::vector<WorkloadMeasurements> gpu, wall;
  gpu.reserve(result.measurements.size());
  wall.reserve(result.measurements.size());
  for (const auto &measurement : result.measurements) {
    gpu.push_back({{0}, measurement.rawGpuSamples()});
    wall.push_back({{0}, measurement.rawWallSamples()});
  }
  std::vector<CandidateMeasurements> gpuCandidates, wallCandidates;
  for (size_t i = 0; i < result.measurements.size(); ++i) {
    gpuCandidates.push_back({result.measurements[i].candidate, {&gpu[i], 1}});
    wallCandidates.push_back({result.measurements[i].candidate, {&wall[i], 1}});
  }
  constexpr std::array required{WorkloadId{0}};
  const auto gpuWinner = selectCandidate(gpuCandidates, required, options.policy);
  const auto wallWinner = selectCandidate(wallCandidates, required, options.policy);
  auto expected = candidates.front();
  if (gpuWinner.verdict == SelectionVerdict::Selected &&
      wallWinner.verdict == SelectionVerdict::Selected &&
      gpuWinner.candidate == wallWinner.candidate)
    expected = candidates[gpuWinner.candidate.value];
  require(result.choice.configuration == expected,
          "attention selected without independent GPU/wall winner agreement");
}

// Explicit opt-in only: CPU sizing tests do not create a Metal backend.
// This path validates production candidate numerics through the tuner itself;
// timing rejection is acceptable and is never a performance assertion.
void metalTests(const char *metallib) {
  metal::MetalBackend backend(metallib);
  MeasurementOptions options;
  options.warmupPairs = 1;
  options.maximumWallSeconds = 60;
  size_t admissionCalls = 0;
  uint64_t admittedBytes = 0;
  const metal::AllocationAdmission admit = [&](uint64_t bytes, const auto &allocate) {
    ++admissionCalls;
    admittedBytes = bytes;
    allocate();
    return true;
  };
  const metal::AllocationAdmission deny = [&](uint64_t, const auto &) {
    ++admissionCalls;
    return false;
  };
  // The sweep's verify IDs index the tuning order of this device's baseline.
  const auto verifyBaseline = VerifyAttentionConfig{};
  const auto verifyCandidates = verifyAttentionTuningCandidates(verifyBaseline);
  for (auto format : {kv::Format::Int8, kv::Format::BFloat16})
  for (auto shape : {AttentionShape{24, 4, 256, format}, AttentionShape{16, 2, 256, format}}) {
    for (uint32_t history : {0U, 2048U}) {
      const PrefillAttentionWorkload workload{shape, 2048, history};
      const size_t calls = admissionCalls;
      const auto result = tunePrefillAttention(backend, admit, workload, options);
      completed(result, options, PagedAttention::prefillCandidates());
      require(admissionCalls == calls + 1 &&
                  admittedBytes == prefillAttentionTuningFixtureBytes(workload),
              "prefill fixtures were not allocated once under exact admission");
    }
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      std::array<uint32_t, 4> histories{};
      for (uint32_t lane = 0; lane < lanes; ++lane) histories[lane] = 33 + lane * 137;
      const VerifyAttentionWorkload workload{shape, lanes, histories};
      const size_t calls = admissionCalls;
      const auto result = tuneVerifyAttention(backend, admit, workload, options);
      completed(result, options, verifyCandidates);
      require(admissionCalls == calls + 1 &&
                  admittedBytes == verifyAttentionTuningFixtureBytes(workload),
              "verify fixtures were not allocated once under exact admission");
    }
  }
  const VerifyAttentionWorkload workload{{24, 4, 256}, 3, {1, 32, 2049, 0}};
  const auto denied = tuneVerifyAttention(backend, deny, workload, options);
  require(!denied.complete && denied.measurements.empty() && !denied.failure &&
              denied.choice.configuration == verifyBaseline,
          "denied admission did not preserve baseline without execution");
  const auto deniedPolicy = tuneVerifyAttentionPolicy(backend, deny, {workload.shape, workload.lanes}, options);
  require(!deniedPolicy.complete && !deniedPolicy.failure && deniedPolicy.probes.size() == 1 &&
              deniedPolicy.choice.configuration == verifyBaseline,
          "policy admission denial was recorded as complete evidence");
  const size_t beforeInvalidPrefill = admissionCalls;
  const auto invalidPrefill = tunePrefillAttentionPolicy(
      backend, admit, {workload.shape, 2047}, options);
  require(!invalidPrefill.complete && invalidPrefill.failure &&
              invalidPrefill.probes.empty() && admissionCalls == beforeInvalidPrefill,
          "ragged prefill policy allocated a calibration fixture");
  const auto invalidAdmission = tuneVerifyAttention(backend,
      [](uint64_t, const auto &) { return true; }, workload, options);
  require(!invalidAdmission.complete && invalidAdmission.failure &&
              invalidAdmission.measurements.empty(),
          "successful admission without an allocation was not reported");
  size_t calls = admissionCalls;
  const auto stopped = tuneVerifyAttention(backend, admit, workload, options, {}, [] { return true; });
  const auto pressured = tuneVerifyAttention(backend, admit, workload, options, [] { return true; });
  require(!stopped.complete && !pressured.complete && admissionCalls == calls,
          "stop/pressure did not prevent fixture allocation and GPU work");
  auto invalid = options;
  invalid.samplePairs = 1;
  const auto rejected = tuneVerifyAttention(backend, admit, workload, invalid);
  require(!rejected.complete && rejected.measurements.empty() && admissionCalls == calls,
          "invalid measurement options consumed fixture resources");
  const auto cancelledAfterAdmission = tuneVerifyAttention(
      backend, admit, workload, options, {}, [&] { return admissionCalls != calls; });
  require(!cancelledAfterAdmission.complete && cancelledAfterAdmission.measurements.empty(),
          "cancellation after allocation submitted a command");
}
} // namespace

int main(int argc, char **argv) {
  try {
    cpuTests();
    planExecutionTests();
    verifyExecutionTests();
    policyTests();
    if (argc == 3 && std::string_view(argv[1]) == "--metal") metalTests(argv[2]);
    else require(argc == 1, "usage: attention-tuning [--metal production.metallib]");
    std::cout << "PASS attention tuning: " << (argc == 1 ? "CPU fixture bounds" : "CPU + native candidate qualification")
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL attention tuning: " << error.what() << '\n';
    return 1;
  }
}
