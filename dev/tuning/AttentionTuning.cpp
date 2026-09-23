#include "tuning/AttentionTuning.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace splash::ops::tuning {
namespace {

using Clock = std::chrono::steady_clock;
constexpr uint64_t kAlignment = 16 * 1024;
constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kDimension = 256;
constexpr uint32_t kPageRows = kv::kPageTokens;
constexpr uint32_t kVerifyRows = SPLASH_TARGET_VERIFY_ROWS;

enum class Tensor : uint32_t {
  Keys, KeyScales, Values, ValueScales, ChunkKeys, ChunkValues,
  Queries, Output, Reference, Partials, Statistics, Table0, Table1, Table2,
  Table3, Count
};
constexpr size_t tensorIndex(Tensor tensor) { return static_cast<size_t>(tensor); }
constexpr size_t kTensorCount = tensorIndex(Tensor::Count);

uint64_t aligned(uint64_t bytes) {
  if (bytes > std::numeric_limits<uint64_t>::max() - kAlignment + 1)
    throw std::invalid_argument("attention tuning fixture size overflow");
  return (bytes + kAlignment - 1) & ~(kAlignment - 1);
}
uint16_t bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return uint16_t(bits >> 16);
}
float fp32(uint16_t value) {
  return std::bit_cast<float>(uint32_t{value} << 16);
}

struct FixturePlan final {
  AttentionShape shape;
  uint32_t lanes = 0;
  uint32_t rows = 0;
  uint32_t stride = 0;
  uint32_t physicalPages = 0;
  std::array<uint32_t, kMaximumLanes> histories{};
  std::array<uint32_t, kMaximumLanes> pages{};
  std::array<uint64_t, kTensorCount> sizes{};
  uint64_t bytes = 0;

  kv::Layout layout() const { return {1, shape.kvHeads, shape.headDimension, shape.format}; }
  void size(Tensor tensor, uint64_t bytes) { sizes[tensorIndex(tensor)] = bytes; }
  uint64_t queryIndex(uint32_t lane, uint32_t head, uint32_t row,
                      uint32_t dimension) const {
    const uint32_t group = shape.queryHeads / shape.kvHeads;
    return (((uint64_t{lane} * shape.kvHeads + head / group) * stride + row) *
                group + head % group) * kDimension + dimension;
  }
};

template <typename Workload> FixturePlan fixturePlan(Workload workload) {
  FixturePlan plan;
  plan.shape = workload.shape;
  AttentionWorkspace scratch;
  if constexpr (std::is_same_v<Workload, PrefillAttentionWorkload>) {
    plan.lanes = 1;
    plan.rows = workload.rows;
    plan.histories[0] = workload.historyTokens;
    for (auto config : PagedAttention::prefillCandidates()) {
      const auto candidate = PagedAttention::prefillPlan(
          plan.rows, plan.shape.queryHeads, plan.layout(), workload.historyTokens, config);
      scratch.partialsBytes = std::max(scratch.partialsBytes,
                                       candidate.workspace.partialsBytes);
      scratch.statisticsBytes = std::max(scratch.statisticsBytes,
                                         candidate.workspace.statisticsBytes);
    }
  } else {
    plan.lanes = workload.lanes;
    plan.rows = kVerifyRows;
    plan.histories = workload.historyTokens;
    for (auto config : PagedAttention::verifyCandidates()) {
      const auto candidate = PagedAttention::verifyPlan(
          plan.lanes, plan.shape.queryHeads, plan.layout(), plan.histories, config);
      scratch.partialsBytes = std::max(scratch.partialsBytes,
                                       candidate.workspace.partialsBytes);
      scratch.statisticsBytes = std::max(scratch.statisticsBytes,
                                         candidate.workspace.statisticsBytes);
    }
  }
  plan.stride = (plan.rows + kPageRows - 1) / kPageRows * kPageRows;
  uint32_t pages = 0;
  for (uint32_t lane = 0; lane < kMaximumLanes; ++lane) {
    if (lane >= plan.lanes) {
      if (plan.histories[lane])
        throw std::invalid_argument("inactive tuning history must be zero");
      continue;
    }
    const uint64_t tokens = uint64_t{plan.histories[lane]} + plan.rows;
    if (tokens > kv::kMaximumPhysicalTokens)
      throw std::invalid_argument("attention tuning history exceeds context");
    plan.pages[lane] = uint32_t((tokens + kPageRows - 1) / kPageRows);
    pages += plan.pages[lane];
    plan.size(static_cast<Tensor>(tensorIndex(Tensor::Table0) + lane),
              uint64_t{plan.pages[lane]} * sizeof(uint32_t));
  }
  // An odd physical pool permits an injective stride-two page permutation.
  // Only one/two spare pages are needed, even for long exact histories.
  plan.physicalPages = pages + 1 + (pages % 2);
  const uint64_t dataBytes = plan.physicalPages * plan.layout().dataBytesPerLayerPage();
  const uint64_t scaleBytes = plan.physicalPages * plan.layout().scaleBytesPerLayerPage();
  plan.size(Tensor::Keys, dataBytes);
  plan.size(Tensor::Values, dataBytes);
  plan.size(Tensor::KeyScales, scaleBytes);
  plan.size(Tensor::ValueScales, scaleBytes);
  const uint64_t chunks = uint64_t{plan.lanes} * plan.shape.kvHeads *
                           plan.stride * kDimension * sizeof(uint16_t);
  const uint64_t queries = uint64_t{plan.lanes} * plan.shape.queryHeads *
                            plan.stride * kDimension * sizeof(uint16_t);
  plan.size(Tensor::ChunkKeys, chunks);
  plan.size(Tensor::ChunkValues, chunks);
  plan.size(Tensor::Queries, queries);
  plan.size(Tensor::Output, queries);
  plan.size(Tensor::Reference, queries);
  plan.size(Tensor::Partials, scratch.partialsBytes);
  plan.size(Tensor::Statistics, scratch.statisticsBytes);
  for (uint64_t size : plan.sizes) {
    const uint64_t allocation = aligned(size);
    if (allocation > std::numeric_limits<uint64_t>::max() - plan.bytes)
      throw std::invalid_argument("attention tuning fixture size overflow");
    plan.bytes += allocation;
  }
  return plan;
}

struct NumericalMismatch final : std::runtime_error {
  NumericalMismatch() : std::runtime_error("attention tuning candidate failed output qualification") {}
};

struct Interrupted final {
  MeasurementStatus status;
};

class Fixture final {
public:
  Fixture(metal::MetalBackend &backend, FixturePlan plan)
      : plan_(std::move(plan)), base_(backend.allocateBuffer(
            plan_.bytes, metal::BufferStorage::Shared, "attention-tuning-fixture")) {
    uint64_t offset = 0;
    for (size_t i = 0; i < plan_.sizes.size(); ++i) {
      if (plan_.sizes[i]) buffers_[i] = backend.view(base_, offset, plan_.sizes[i]);
      offset += aligned(plan_.sizes[i]);
    }
    layer_ = {get(Tensor::Keys), get(Tensor::KeyScales), get(Tensor::Values),
               get(Tensor::ValueScales), plan_.shape.format};
    for (uint32_t lane = 0; lane < plan_.lanes; ++lane) {
      tables_[lane] = get(static_cast<Tensor>(tensorIndex(Tensor::Table0) + lane));
      stores_[lane] = {plan_.histories[lane], plan_.rows, plan_.stride,
                        plan_.pages[lane], plan_.physicalPages, 0, 0, 0};
      attention_[lane] = kv::q8VerifyAttentionParams(
          plan_.histories[lane], kVerifyRows, plan_.stride, plan_.pages[lane],
          plan_.physicalPages);
    }
    for (uint32_t lane = plan_.lanes; lane < kMaximumLanes; ++lane) {
      tables_[lane] = tables_[0];
      stores_[lane] = stores_[0];
      attention_[lane] = attention_[0];
    }
  }

  // Called once after admission. Long histories periodically consult the same
  // sweep control callback; no GPU command has been submitted at this point.
  bool initialize(const MeasurementStop &stop) {
    std::memset(base_.contents(), 0, plan_.bytes);
    auto *keys = data<int8_t>(Tensor::Keys);
    auto *values = data<int8_t>(Tensor::Values);
    auto *keyScales = data<float>(Tensor::KeyScales);
    auto *valueScales = data<float>(Tensor::ValueScales);
    auto *chunkKeys = data<uint16_t>(Tensor::ChunkKeys);
    auto *chunkValues = data<uint16_t>(Tensor::ChunkValues);
    auto *queries = data<uint16_t>(Tensor::Queries);
    uint32_t firstPage = 0;
    for (uint32_t lane = 0; lane < plan_.lanes; ++lane) {
      auto *table = static_cast<uint32_t *>(tables_[lane].contents());
      for (uint32_t page = 0; page < plan_.pages[lane]; ++page)
        table[page] = (2 * (firstPage + page) + 1) % plan_.physicalPages;
      firstPage += plan_.pages[lane];
      for (uint32_t token = 0; token < plan_.histories[lane]; ++token) {
        if (token % 256 == 0 && stop && stop()) return false;
        for (uint32_t head = 0; head < plan_.shape.kvHeads; ++head) {
          const uint64_t scale = scaleIndex(lane, head, token);
          if (plan_.shape.format == kv::Format::Int8) {
            keyScales[scale] = 0.006f;
            valueScales[scale] = 0.007f;
          }
          for (uint32_t d = 0; d < kDimension; ++d) {
            const int key = int((uint64_t{token} * 37 + head * 101 + d * 17 +
                     uint64_t{token} * d * 3 + lane * 7) % 255) - 127;
            const int value = int((uint64_t{token} * 53 + head * 79 + d * 29 +
                     uint64_t{token} * d * 5 + lane * 19) % 255) - 127;
            if (plan_.shape.format == kv::Format::Int8) {
              keys[scale * kDimension + d] = key;
              values[valueIndex(scale, d)] = value;
            } else {
              data<uint16_t>(Tensor::Keys)[scale * kDimension + d] = bf16(key * 0.006f);
              data<uint16_t>(Tensor::Values)[valueIndex(scale, d)] = bf16(value * 0.007f);
            }
          }
        }
      }
      for (uint32_t row = 0; row < plan_.rows; ++row) {
        for (uint32_t head = 0; head < plan_.shape.kvHeads; ++head) {
          const uint64_t base = (uint64_t{lane} * plan_.shape.kvHeads + head) *
                                 plan_.stride * kDimension;
          for (uint32_t d = 0; d < kDimension; ++d) {
            chunkKeys[base + row * kDimension + d] =
                bf16(float(int((row * 37 + head * 101 + d * 17 + lane * 7) % 255) - 127) * 0.006f);
            chunkValues[base + uint64_t{d} * plan_.stride + row] =
                bf16(float(int((row * 53 + head * 79 + d * 29 + lane * 19) % 255) - 127) * 0.007f);
          }
        }
        for (uint32_t head = 0; head < plan_.shape.queryHeads; ++head)
          for (uint32_t d = 0; d < kDimension; ++d)
            queries[plan_.queryIndex(lane, head, row, d)] =
                bf16(float(int((row * 43 + head * 67 + d * 11 + head * d * 7 +
                                lane * 29) % 1019) - 509) / 1018.0f);
      }
    }
    return true;
  }

  template <typename Config> metal::CommandGraph graph(Config config,
                                                       uint32_t repetitions = 1) const {
    metal::CommandGraph result;
    const auto attentionPlan = [&] {
      if constexpr (std::is_same_v<Config, PrefillAttentionConfig>)
        return PagedAttention::prefillPlan(plan_.rows, plan_.shape.queryHeads,
                                           plan_.layout(), plan_.histories[0], config);
      else
        return PagedAttention::verifyPlan(plan_.lanes, plan_.shape.queryHeads,
                                          plan_.layout(), plan_.histories, config);
    }();
    // Each repeated subgraph begins with the deterministic KV store and ends
    // with split reduction. It never consumes the previous attention output:
    // queries/chunk K/V/history remain unchanged and current KV slots are
    // overwritten with identical values. Do not repeat individual dispatches.
    for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
      if constexpr (std::is_same_v<Config, PrefillAttentionConfig>) {
        PagedAttention::addPrefillStore(result, layer_, get(Tensor::ChunkKeys),
                                        get(Tensor::ChunkValues), tables_[0], stores_[0],
                                        plan_.layout());
        PagedAttention::addPrefill(result, layer_, get(Tensor::Queries), get(Tensor::Output),
                                   get(Tensor::Partials), get(Tensor::Statistics), tables_[0],
                                   stores_[0], attentionPlan);
      } else {
        PagedAttention::addVerify(
            result, layer_, {get(Tensor::ChunkKeys), get(Tensor::ChunkValues),
                             get(Tensor::Queries), get(Tensor::Partials),
                             get(Tensor::Statistics), get(Tensor::Output), tables_},
            stores_, attention_, attentionPlan);
      }
    }
    return result;
  }

  void reset() {
    for (Tensor tensor : {Tensor::Output, Tensor::Partials, Tensor::Statistics})
      std::memset(get(tensor).contents(), 0, get(tensor).sizeBytes());
    // Only current-row cache slots are mutated by the production store.
    // Restore those exact slots; immutable history, input and tables remain.
    auto *keys = data<int8_t>(Tensor::Keys);
    auto *values = data<int8_t>(Tensor::Values);
    auto *keyScales = data<float>(Tensor::KeyScales);
    auto *valueScales = data<float>(Tensor::ValueScales);
    for (uint32_t lane = 0; lane < plan_.lanes; ++lane)
      for (uint32_t row = 0; row < plan_.rows; ++row)
        for (uint32_t head = 0; head < plan_.shape.kvHeads; ++head) {
          const uint64_t scale = scaleIndex(lane, head, plan_.histories[lane] + row);
          if (plan_.shape.format == kv::Format::Int8) {
            keyScales[scale] = 0;
            valueScales[scale] = 0;
            std::memset(keys + scale * kDimension, 0, kDimension);
            for (uint32_t d = 0; d < kDimension; ++d)
              values[valueIndex(scale, d)] = 0;
          } else {
            std::memset(data<uint16_t>(Tensor::Keys) + scale * kDimension,
                        0, kDimension * sizeof(uint16_t));
            for (uint32_t d = 0; d < kDimension; ++d)
              data<uint16_t>(Tensor::Values)[valueIndex(scale, d)] = 0;
          }
        }
  }

  void qualify(bool baseline) {
    auto *reference = data<uint16_t>(Tensor::Reference);
    const auto *output = data<uint16_t>(Tensor::Output);
    double dot = 0, refSquared = 0, outSquared = 0;
    float maximumError = 0;
    for (uint32_t lane = 0; lane < plan_.lanes; ++lane)
      for (uint32_t head = 0; head < plan_.shape.queryHeads; ++head)
        for (uint32_t row = 0; row < plan_.rows; ++row)
          for (uint32_t d = 0; d < kDimension; ++d) {
            const uint64_t index = plan_.queryIndex(lane, head, row, d);
            const float right = fp32(output[index]);
            if (!std::isfinite(right)) throw NumericalMismatch();
            if (!haveReference_) continue;
            const float left = fp32(reference[index]);
            maximumError = std::max(maximumError, std::abs(left - right));
            dot += double(left) * right;
            refSquared += double(left) * left;
            outSquared += double(right) * right;
          }
    if (!haveReference_) {
      if (!baseline) throw std::logic_error("attention baseline was not measured first");
      std::memcpy(reference, output, get(Tensor::Output).sizeBytes());
      haveReference_ = true;
    } else if (!(maximumError < 0.02f) || !(refSquared > 0) || !(outSquared > 0) ||
               !(dot / std::sqrt(refSquared * outSquared) > 0.9995)) {
      throw NumericalMismatch();
    }
  }

private:
  metal::MetalBuffer get(Tensor tensor) const { return buffers_[tensorIndex(tensor)]; }
  template <typename T> T *data(Tensor tensor) const {
    return static_cast<T *>(get(tensor).contents());
  }
  uint64_t scaleIndex(uint32_t lane, uint32_t head, uint32_t token) const {
    const auto *table = static_cast<const uint32_t *>(tables_[lane].contents());
    return (uint64_t{table[token / kPageRows]} * plan_.shape.kvHeads + head) *
               kPageRows + token % kPageRows;
  }
  static uint64_t valueIndex(uint64_t scale, uint32_t dimension) {
    return (scale / kPageRows * kDimension + dimension) * kPageRows + scale % kPageRows;
  }
  FixturePlan plan_;
  metal::MetalBuffer base_;
  std::array<metal::MetalBuffer, kTensorCount> buffers_{};
  kv::LayerStorage layer_;
  std::array<metal::MetalBuffer, kMaximumLanes> tables_{};
  std::array<kv::Q8ChunkedPrefillParams, kMaximumLanes> stores_{};
  std::array<kv::Q8VerifyAttentionParams, kMaximumLanes> attention_{};
  bool haveReference_ = false;
};

template <typename Workload, typename Config>
bool equivalentToBaseline(const Workload &workload, Config baseline, Config config) {
  const kv::Layout layout{1, workload.shape.kvHeads, workload.shape.headDimension, workload.shape.format};
  const auto plan = [&](Config selected) {
    if constexpr (std::is_same_v<Config, PrefillAttentionConfig>)
      return PagedAttention::prefillPlan(workload.rows, workload.shape.queryHeads,
                                         layout, workload.historyTokens, selected);
    else
      return PagedAttention::verifyPlan(workload.lanes, workload.shape.queryHeads, layout,
                                        workload.historyTokens, selected);
  };
  const auto base = plan(baseline), candidate = plan(config);
  return base.sameExecutionAs(candidate);
}

// configurations is the tuning order: its front is the baseline every other
// entry is measured against, and candidate IDs index it.
template <typename Result, typename Workload, typename Config>
Result tune(metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
            Workload workload, std::span<const Config> configurations,
            const MeasurementOptions &options, const MeasurementStop &pressure,
            const MeasurementStop &stop) {
  if (configurations.empty()) throw std::logic_error("attention tuning has no baseline");
  const Config baseline = configurations.front();
  Result result{{workload, baseline}, {}, false, {}};
  const auto started = Clock::now();
  auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - started).count(); };
  auto interrupted = [&] {
    return (stop && stop()) || (pressure && pressure()) || elapsed() >= options.maximumWallSeconds;
  };
  try {
    const auto plan = fixturePlan(workload);
    if (!admit || !validMeasurementOptions(options) || interrupted() ||
        plan.bytes > backend.capabilities().maxBufferLengthBytes)
      return result;
    std::unique_ptr<Fixture> fixture;
    bool invoked = false;
    const auto admitted = admit(plan.bytes, [&] {
      if (invoked) throw std::logic_error("attention fixture admission invoked twice");
      invoked = true;
      fixture = std::make_unique<Fixture>(backend, plan);
    });
    if (!admitted) {
      if (fixture) throw std::logic_error("attention admission denied after retaining allocation");
      return result;
    }
    if (!invoked || !fixture)
      throw std::logic_error("attention admission succeeded without allocation");
    if (interrupted() || !fixture->initialize(interrupted))
      return result;
    const auto requireControl = [&] {
      if (stop && stop()) throw Interrupted{MeasurementStatus::Cancelled};
      if (pressure && pressure()) throw Interrupted{MeasurementStatus::UnderPressure};
      if (elapsed() >= options.maximumWallSeconds)
        throw Interrupted{MeasurementStatus::BudgetExceeded};
    };
    const auto qualificationRun = [&](Config config, uint32_t repetitions) {
      requireControl();
      fixture->reset();
      requireControl();
      const auto graph = fixture->graph(config, repetitions);
      const auto timing = backend.submitCommand(graph.dispatches());
      requireControl();
      if (!std::isfinite(timing.gpuSeconds) || timing.gpuSeconds <= 0 ||
          !std::isfinite(timing.wallSeconds) || timing.wallSeconds <= 0)
        throw std::runtime_error("invalid attention qualification timing");
      return timing;
    };
    // Reuse the ordinary single-baseline qualification as the GPU pilot.
    // The resulting count is fixed for the entire workload, never tuned per
    // candidate or inferred from the faster side of a measured pair.
    const auto baselinePilot = qualificationRun(baseline, 1);
    fixture->qualify(true);
    result.repetitions = measurementBatchRepetitions(baselinePilot.gpuSeconds);
    if (result.repetitions > 1) {
      (void)qualificationRun(baseline, result.repetitions);
      // Repeated complete graphs use the same numerical contract as a
      // single candidate; normal floating-point differences are permitted.
      fixture->qualify(false);
    }
    for (size_t candidate = 1; candidate < configurations.size(); ++candidate) {
      const Config config = configurations[candidate];
      if (equivalentToBaseline(workload, baseline, config)) {
        result.equivalentCandidates.push_back({static_cast<uint32_t>(candidate)});
        continue;
      }
      if (interrupted()) return result;
      (void)qualificationRun(config, 1);
      fixture->qualify(false);
      if (result.repetitions > 1) {
        (void)qualificationRun(config, result.repetitions);
        fixture->qualify(false);
      }
      if (interrupted()) return result;
      auto remainingOptions = options;
      remainingOptions.maximumWallSeconds = options.maximumWallSeconds - elapsed();
      const CandidateId id{static_cast<uint32_t>(candidate)};
      auto measurement = measureWorkload(
          id, {0}, [&](CandidateId selected) {
            if (pressure && pressure()) return RunTiming{0, 0, true};
            fixture->reset();
            if (pressure && pressure()) return RunTiming{0, 0, true};
            if (stop && stop()) throw Interrupted{MeasurementStatus::Cancelled};
            if (elapsed() >= options.maximumWallSeconds)
              throw Interrupted{MeasurementStatus::BudgetExceeded};
            const auto wallStart = Clock::now();
            const auto graph = fixture->graph(selected == kBaseline ? baseline : config,
                                               result.repetitions);
            const auto timing = backend.submitCommand(graph.dispatches());
            const double wallSeconds =
                std::chrono::duration<double>(Clock::now() - wallStart).count();
            return RunTiming{timing.gpuSeconds, wallSeconds,
                              pressure && pressure()};
          }, remainingOptions, stop);
      if (measurement.status == MeasurementStatus::RunFailed && measurement.failure) {
        try { std::rethrow_exception(measurement.failure); }
        catch (const Interrupted &interruption) {
          measurement.status = interruption.status;
          measurement.failure = {};
        }
        catch (...) {}
      }
      const auto status = measurement.status;
      result.measurements.push_back(std::move(measurement));
      if (status != MeasurementStatus::Completed && status != MeasurementStatus::Rejected) {
        result.failure = result.measurements.back().failure;
        return result;
      }
    }
    if (interrupted()) return result;
    std::vector<WorkloadMeasurements> gpuWorkloads, wallWorkloads;
    gpuWorkloads.reserve(result.measurements.size());
    wallWorkloads.reserve(result.measurements.size());
    for (const auto &measurement : result.measurements) {
      // Each metric independently sees every finished candidate, including
      // one rejected by the OTHER metric. Filtering those records could hide
      // disagreement between their winners.
      gpuWorkloads.push_back({{0}, measurement.rawGpuSamples()});
      wallWorkloads.push_back({{0}, measurement.rawWallSamples()});
    }
    std::vector<CandidateMeasurements> gpuCandidates, wallCandidates;
    for (size_t i = 0; i < result.measurements.size(); ++i) {
      gpuCandidates.push_back({result.measurements[i].candidate, {&gpuWorkloads[i], 1}});
      wallCandidates.push_back({result.measurements[i].candidate, {&wallWorkloads[i], 1}});
    }
    constexpr std::array required{WorkloadId{0}};
    const auto gpu = selectCandidate(gpuCandidates, required, options.policy);
    const auto wall = selectCandidate(wallCandidates, required, options.policy);
    if (gpu.verdict == SelectionVerdict::Selected &&
        wall.verdict == SelectionVerdict::Selected && gpu.candidate == wall.candidate)
      result.choice.configuration = configurations[gpu.candidate.value];
    result.complete = true;
  } catch (const Interrupted &) {
    // Prequalification control boundaries have no partial timing sample to
    // report. They still stop the entire sweep without selecting a winner.
  } catch (...) {
    result.failure = std::current_exception();
  }
  return result;
}

template <typename Config, typename Probe, typename Workload>
Config selectPolicy(std::span<const Probe> probes, std::span<const Workload> required,
                     std::span<const Config> configurations, const Policy &policy) {
  if (configurations.empty()) throw std::logic_error("attention tuning has no baseline");
  const Config baseline = configurations.front();
  if (probes.size() != required.size() || required.empty()) return baseline;
  const size_t candidates = configurations.size() - 1;
  std::vector<WorkloadId> ids;
  std::vector<std::vector<WorkloadMeasurements>> gpu(candidates), wall(candidates);
  for (size_t i = 0; i < probes.size(); ++i) {
    const auto &probe = probes[i];
    if (!probe.complete || probe.failure || probe.choice.workload != required[i] ||
        probe.measurements.size() + probe.equivalentCandidates.size() != candidates)
      return baseline;
    ids.push_back({static_cast<uint32_t>(i)});
    for (size_t c = 1; c < configurations.size(); ++c) {
      const CandidateId id{static_cast<uint32_t>(c)};
      const auto equivalent = std::count(probe.equivalentCandidates.begin(),
                                         probe.equivalentCandidates.end(), id);
      const MeasurementResult *measured = nullptr;
      for (const auto &measurement : probe.measurements) {
        if (measurement.candidate != id) continue;
        if (measured) return baseline;
        measured = &measurement;
      }
      if (equivalent > 1 || (equivalent != 0) == (measured != nullptr)) return baseline;
      if (equivalent) {
        // Re-check structural evidence at the typed-plan boundary. Never
        // infer equivalence from an absent record or a measured zero gain.
        if (!equivalentToBaseline(required[i], baseline, configurations[c])) return baseline;
        gpu[c - 1].push_back({ids.back(), {}, true});
        wall[c - 1].push_back({ids.back(), {}, true});
      } else {
        if (measured->failure || (measured->status != MeasurementStatus::Completed &&
            measured->status != MeasurementStatus::Rejected)) return baseline;
        gpu[c - 1].push_back({ids.back(), measured->rawGpuSamples()});
        wall[c - 1].push_back({ids.back(), measured->rawWallSamples()});
      }
    }
  }
  std::vector<CandidateMeasurements> gpuCandidates, wallCandidates;
  for (size_t c = 0; c < candidates; ++c) {
    gpuCandidates.push_back({{static_cast<uint32_t>(c + 1)}, gpu[c]});
    wallCandidates.push_back({{static_cast<uint32_t>(c + 1)}, wall[c]});
  }
  const auto selectedGpu = selectCandidate(gpuCandidates, ids, policy);
  const auto selectedWall = selectCandidate(wallCandidates, ids, policy);
  if (selectedGpu.verdict == SelectionVerdict::Selected &&
      selectedWall.verdict == SelectionVerdict::Selected &&
      selectedGpu.candidate == selectedWall.candidate)
    return configurations[selectedGpu.candidate.value];
  return baseline;
}

template <typename Result, typename PolicyKey, typename Config, typename Workloads,
          typename Run, typename Select>
Result tunePolicy(PolicyKey key, Config baseline, const Workloads &workloads,
                    const MeasurementOptions &options, const MeasurementStop &pressure,
                    const MeasurementStop &stop, Run run, Select select) {
  Result result{{key, baseline}, {}, false, {}};
  const auto started = Clock::now();
  const auto remaining = [&] {
    return options.maximumWallSeconds - std::chrono::duration<double>(Clock::now() - started).count();
  };
  const auto interrupted = [&] {
    return (stop && stop()) || (pressure && pressure()) || remaining() <= 0;
  };
  try {
    if (!validMeasurementOptions(options)) return result;
    for (const auto &workload : workloads) {
      if (interrupted()) return result;
      auto probeOptions = options;
      probeOptions.maximumWallSeconds = remaining();
      result.probes.push_back(run(workload, probeOptions));
      const auto &probe = result.probes.back();
      if (!probe.complete || probe.failure) {
        result.failure = probe.failure;
        return result;
      }
    }
    if (interrupted()) return result;
    result.choice.configuration = select(result.probes, options.policy);
    result.complete = true;
  } catch (...) {
    result.failure = std::current_exception();
  }
  return result;
}

} // namespace

std::array<PrefillAttentionWorkload, 4>
prefillAttentionPolicyWorkloads(AttentionShape shape) {
  constexpr uint32_t rows = SPLASH_PREFILL_TOKEN_BUDGET;
  (void)PagedAttention::prefillPlan(rows, shape.queryHeads,
                                   {1, shape.kvHeads, shape.headDimension, shape.format}, 0);
  return {{{shape, rows, 0}, {shape, rows, 2048},
           {shape, rows, 16384}, {shape, rows, 131072}}};
}

std::vector<VerifyAttentionWorkload>
verifyAttentionPolicyWorkloads(VerifyAttentionPolicy policy) {
  const auto shape = policy.shape;
  const std::array<uint32_t, kMaximumLanes> histories{};
  (void)PagedAttention::verifyPlan(policy.lanes, shape.queryHeads,
                                    {1, shape.kvHeads, shape.headDimension, shape.format}, histories);
  std::vector<VerifyAttentionWorkload> result;
  for (uint32_t history : {1U, 25U, 2048U, 131072U}) {
    VerifyAttentionWorkload workload{shape, policy.lanes, {}};
    std::fill_n(workload.historyTokens.begin(), policy.lanes, history);
    result.push_back(workload);
  }
  if (policy.lanes > 1) {
    constexpr std::array anchors{25U, 2049U, 8191U, 131072U};
    VerifyAttentionWorkload mixed{shape, policy.lanes, {}};
    for (uint32_t lane = 0; lane < policy.lanes; ++lane)
      mixed.historyTokens[lane] = anchors[lane * 3 / (policy.lanes - 1)];
    result.push_back(mixed);
    std::reverse(mixed.historyTokens.begin(), mixed.historyTokens.begin() + policy.lanes);
    result.push_back(mixed);
  }
  return result;
}

std::vector<VerifyAttentionConfig>
verifyAttentionTuningCandidates(VerifyAttentionConfig baseline) {
  const auto operators = PagedAttention::verifyCandidates();
  if (std::count(operators.begin(), operators.end(), baseline) != 1)
    throw std::invalid_argument("verify attention baseline is not a precompiled candidate");
  std::vector<VerifyAttentionConfig> result{baseline};
  for (const auto config : operators)
    if (config != baseline) result.push_back(config);
  return result;
}

PrefillAttentionConfig selectPrefillAttentionPolicy(
    std::span<const PrefillAttentionTuningResult> probes, const Policy &policy) {
  if (probes.empty()) return {};
  const auto workloads = prefillAttentionPolicyWorkloads(probes.front().choice.workload.shape);
  return selectPolicy(probes, std::span<const PrefillAttentionWorkload>(workloads),
                        PagedAttention::prefillCandidates(), policy);
}
VerifyAttentionConfig selectVerifyAttentionPolicy(
    std::span<const VerifyAttentionTuningResult> probes, VerifyAttentionConfig baseline,
    const Policy &policy) {
  const auto candidates = verifyAttentionTuningCandidates(baseline);
  if (probes.empty()) return baseline;
  const auto &first = probes.front().choice.workload;
  const auto workloads = verifyAttentionPolicyWorkloads({first.shape, first.lanes});
  return selectPolicy(probes, std::span<const VerifyAttentionWorkload>(workloads),
                        std::span<const VerifyAttentionConfig>(candidates), policy);
}

PrefillAttentionPolicyResult tunePrefillAttentionPolicy(
    metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
    PrefillAttentionPolicy policy, const MeasurementOptions &options,
    const MeasurementStop &pressure, const MeasurementStop &stop) {
  try {
    if (policy.rows != SPLASH_PREFILL_TOKEN_BUDGET)
      throw std::invalid_argument("prefill attention calibration requires the fixed chunk size");
    return tunePolicy<PrefillAttentionPolicyResult>(policy, PrefillAttentionConfig{},
        prefillAttentionPolicyWorkloads(policy.shape), options, pressure, stop,
        [&](auto workload, const auto &probeOptions) {
          return tunePrefillAttention(backend, admit, workload, probeOptions, pressure, stop);
        }, selectPrefillAttentionPolicy);
  } catch (...) {
    return {{policy, {}}, {}, false, std::current_exception()};
  }
}
VerifyAttentionPolicyResult tuneVerifyAttentionPolicy(
    metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
    VerifyAttentionPolicy policy, const MeasurementOptions &options,
    const MeasurementStop &pressure, const MeasurementStop &stop) {
  const auto baseline = VerifyAttentionConfig{};
  try {
    return tunePolicy<VerifyAttentionPolicyResult>(policy, baseline,
        verifyAttentionPolicyWorkloads(policy), options, pressure, stop,
        [&](auto workload, const auto &probeOptions) {
          return tuneVerifyAttention(backend, admit, workload, probeOptions, pressure, stop);
        },
        [&](std::span<const VerifyAttentionTuningResult> probes, const Policy &selection) {
          return selectVerifyAttentionPolicy(probes, baseline, selection);
        });
  } catch (...) {
    return {{policy, baseline}, {}, false, std::current_exception()};
  }
}

uint64_t prefillAttentionTuningFixtureBytes(PrefillAttentionWorkload workload) {
  return fixturePlan(workload).bytes;
}
uint64_t verifyAttentionTuningFixtureBytes(VerifyAttentionWorkload workload) {
  return fixturePlan(workload).bytes;
}
PrefillAttentionTuningResult tunePrefillAttention(
    metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
    PrefillAttentionWorkload workload, const MeasurementOptions &options,
    const MeasurementStop &underPressure, const MeasurementStop &shouldStop) {
  return tune<PrefillAttentionTuningResult>(backend, admit, workload,
                                           PagedAttention::prefillCandidates(),
                                           options, underPressure, shouldStop);
}
VerifyAttentionTuningResult tuneVerifyAttention(
    metal::MetalBackend &backend, const metal::AllocationAdmission &admit,
    VerifyAttentionWorkload workload, const MeasurementOptions &options,
    const MeasurementStop &underPressure, const MeasurementStop &shouldStop) {
  const auto candidates = verifyAttentionTuningCandidates(VerifyAttentionConfig{});
  return tune<VerifyAttentionTuningResult>(backend, admit, workload,
                                          std::span<const VerifyAttentionConfig>(candidates),
                                          options, underPressure, shouldStop);
}

} // namespace splash::ops::tuning
