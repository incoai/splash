#include "engine/Engine.hpp"
#include "engine/Bootstrap.hpp"
#include "engine/Json.hpp"
#include "engine/RuntimeResources.hpp"
#include "model/Runtime.hpp"
#include "PrefillWork.hpp"

#include <algorithm>
#include <iterator>
#include <array>
#include <charconv>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifndef SPLASH_BUILD_ID
#error "backend_benchmark requires the production SPLASH_BUILD_ID"
#endif

using namespace splash;
using namespace splash::engine;
using splash::benchmark::expectedDraftContextRows;

namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point value) {
  return std::chrono::duration<double, std::milli>(value.time_since_epoch())
      .count();
}

struct Observation final {
  std::string cacheStatus;
  uint32_t matchedTokens = 0;
  std::vector<uint32_t> outputTokens;
  std::optional<Clock::time_point> firstToken;
  bool completed = false;
  std::string failure;
};

struct BatchObservation final {
  WorkKind kind = WorkKind::Decode;
  uint32_t width = 0;
  uint32_t inputTokens = 0;
  uint32_t outputTokens = 0;
  uint32_t draftedTokens = 0;
  uint32_t acceptedDraftTokens = 0;
};

class Events final : public EngineEventSink {
public:
  void batchCompleted(WorkKind kind, uint32_t width, uint32_t inputTokens,
                      uint32_t outputTokens,
                      uint32_t draftedTokens, uint32_t acceptedDraftTokens,
                      double, double) override {
    lastBatch_ = {kind, width, inputTokens, outputTokens, draftedTokens,
                  acceptedDraftTokens};
    ++batchSequence_;
  }

  void started(uint64_t requestId, uint32_t matchedTokens,
               uint32_t) override {
    Observation &value = observations_[requestId];
    value.cacheStatus = matchedTokens ? "prefix_hit" : "miss";
    value.matchedTokens = matchedTokens;
  }

  void tokens(uint64_t requestId, std::span<const uint32_t> values) override {
    Observation &value = observations_[requestId];
    if (!value.firstToken)
      value.firstToken = Clock::now();
    value.outputTokens.insert(value.outputTokens.end(), values.begin(),
                              values.end());
  }

  void maskRequested(uint64_t, std::span<const uint32_t>) override {
    throw std::logic_error("unconstrained benchmark requested a token mask");
  }

  void completed(uint64_t requestId, EngineFinishReason, uint32_t,
                 uint32_t, std::span<const float>) override {
    observations_[requestId].completed = true;
  }

  void failed(uint64_t requestId, LaneOutcome outcome,
              std::string message) override {
    observations_[requestId].failure =
        std::string(laneOutcomeWire(outcome).code) + ":" + message;
  }

  [[nodiscard]] const Observation &get(uint64_t requestId) const {
    auto found = observations_.find(requestId);
    if (found == observations_.end())
      throw std::logic_error("request produced no events");
    return found->second;
  }

  [[nodiscard]] uint64_t batchSequence() const noexcept {
    return batchSequence_;
  }

  [[nodiscard]] const BatchObservation &lastBatch() const noexcept {
    return lastBatch_;
  }

private:
  std::unordered_map<uint64_t, Observation> observations_;
  BatchObservation lastBatch_;
  uint64_t batchSequence_ = 0;
};

struct Measurement final {
  std::string scenario;
  uint32_t sample = 0;
  uint32_t promptTokens = 0;
  std::string cacheStatus;
  uint32_t matchedTokens = 0;
  double wallMilliseconds = 0.0;
  double ttftMilliseconds = 0.0;
  double prefillGpuMilliseconds = 0.0;
  double decodeGpuMilliseconds = 0.0;
  uint64_t targetPrefillRows = 0;
  uint64_t draftContextRows = 0;
  uint64_t draftContextRowsAvoided = 0;
  uint64_t draftStateRestoreSkipped = 0;
  uint64_t junctionMaterializations = 0;
  std::vector<uint32_t> outputTokens;
  std::optional<bool> coldOutputMatch;
};

struct DecodeThroughputMeasurement final {
  uint32_t width = 0;
  uint32_t sample = 0;
  uint32_t outputTokens = 0;
  uint64_t decodeOutputTokens = 0;
  uint64_t decodeBatches = 0;
  uint64_t draftedTokens = 0;
  uint64_t acceptedDraftTokens = 0;
  double requestWallMilliseconds = 0.0;
  double prefillWallMilliseconds = 0.0;
  double prefillGpuMilliseconds = 0.0;
  double decodeWallMilliseconds = 0.0;
  double decodeGpuMilliseconds = 0.0;
  double aggregateRequestWallTokensPerSecond = 0.0;
  double aggregateDecodeWallTokensPerSecond = 0.0;
  double aggregateGpuTokensPerSecond = 0.0;
  double outputTokensPerLaneBatch = 0.0;
  double draftAcceptanceRate = 0.0;
  std::vector<uint32_t> firstLaneOutputTokens;
};

uint64_t tokenSequenceHash(std::span<const uint32_t> tokens) noexcept {
  uint64_t hash = 1469598103934665603ULL;
  for (uint32_t token : tokens) {
    for (uint32_t shift = 0; shift != 32; shift += 8) {
      hash ^= static_cast<uint8_t>(token >> shift);
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

// JSONL is intentional: every flushed line is independently readable, so a
// crash during a long 128K run preserves all completed samples.
class ProgressJournal final {
public:
  ProgressJournal(const std::filesystem::path &path, uint32_t samples)
      : output_(path, std::ios::out | std::ios::trunc) {
    if (!output_)
      throw std::runtime_error("cannot open benchmark progress journal");
    output_ << std::setprecision(10)
            << "{\"event\":\"started\",\"build_id\":\""
            << SPLASH_BUILD_ID << "\",\"samples_requested\":" << samples
            << "}\n";
    flush();
  }

  void identity(std::string_view value) {
    output_ << "{\"event\":\"identity\",\"identity\":" << value << "}\n";
    flush();
  }

  void begin(std::string_view phase, std::string_view scenario, uint32_t sample,
             uint32_t promptTokens = 0, uint32_t width = 0,
             std::optional<uint64_t> inputTokenHash = std::nullopt) {
    output_ << "{\"event\":\"begin\",\"phase\":\"" << phase
            << "\",\"scenario\":\"" << scenario << "\",\"sample\":"
            << sample << ",\"prompt_tokens\":" << promptTokens
            << ",\"width\":" << width;
    if (inputTokenHash)
      output_ << ",\"input_token_hash\":\"" << *inputTokenHash << '"';
    output_ << "}\n";
    flush();
    std::cerr << "backend-benchmark: start phase=" << phase
              << " scenario=" << scenario << " sample=" << sample
              << " prompt_tokens=" << promptTokens << " width=" << width
              << '\n';
  }

  void complete(std::string_view kind, std::string_view scenario,
                uint32_t sample, uint32_t promptTokens, uint32_t width,
                double wallMilliseconds, double gpuMilliseconds,
                double ttftMilliseconds, std::string_view cacheStatus = {},
                uint32_t matchedTokens = 0, uint64_t targetRows = 0,
                uint64_t draftRows = 0,
                std::optional<uint64_t> decodeBatches = std::nullopt,
                std::optional<uint64_t> draftedTokens = std::nullopt,
                std::optional<uint64_t> acceptedDraftTokens = std::nullopt,
                std::optional<uint64_t> inputTokenHash = std::nullopt,
                std::span<const uint32_t> outputTokens = {},
                std::span<const uint32_t> prefillBatchRows = {}) {
    output_ << "{\"event\":\"complete\",\"kind\":\"" << kind
            << "\",\"scenario\":\"" << scenario << "\",\"sample\":"
            << sample << ",\"prompt_tokens\":" << promptTokens
            << ",\"width\":" << width << ",\"wall_ms\":"
            << wallMilliseconds << ",\"gpu_ms\":" << gpuMilliseconds
            << ",\"ttft_ms\":" << ttftMilliseconds
            << ",\"cache_status\":\"" << cacheStatus
            << "\",\"matched_tokens\":" << matchedTokens
            << ",\"target_prefill_rows\":" << targetRows
            << ",\"draft_context_rows\":" << draftRows;
    if (decodeBatches)
      output_ << ",\"decode_batches\":" << *decodeBatches;
    if (draftedTokens)
      output_ << ",\"drafted_tokens\":" << *draftedTokens;
    if (acceptedDraftTokens)
      output_ << ",\"accepted_draft_tokens\":" << *acceptedDraftTokens;
    if (inputTokenHash) {
      output_ << ",\"input_token_hash\":\"" << *inputTokenHash
              << "\",\"output_token_hash\":\"" << tokenSequenceHash(outputTokens)
              << "\",\"output_tokens\":[";
      for (size_t index = 0; index < outputTokens.size(); ++index) {
        if (index)
          output_ << ',';
        output_ << outputTokens[index];
      }
      output_ << "],\"prefill_batch_rows\":[";
      for (size_t index = 0; index < prefillBatchRows.size(); ++index) {
        if (index)
          output_ << ',';
        output_ << prefillBatchRows[index];
      }
      output_ << ']';
    }
    output_ << "}\n";
    flush();
    std::cerr << "backend-benchmark: done kind=" << kind
              << " scenario=" << scenario << " sample=" << sample
              << " wall_ms=" << wallMilliseconds
              << " gpu_ms=" << gpuMilliseconds
              << " ttft_ms=" << ttftMilliseconds << '\n';
  }

  void finish(size_t failureCount) {
    output_ << "{\"event\":\"finished\",\"result\":\""
            << (failureCount ? "fail" : "pass")
            << "\",\"performance_failure_count\":" << failureCount
            << "}\n";
    flush();
  }

  void fail() noexcept {
    try {
      output_ << "{\"event\":\"finished\",\"result\":\"fail\"}\n";
      flush();
    } catch (...) {
    }
  }

private:
  void flush() {
    output_.flush();
    if (!output_)
      throw std::runtime_error("cannot write benchmark progress journal");
  }

  std::ofstream output_;
};

class Driver final {
public:
  Driver(engine::Engine &engine, Events &events)
      : engine_(engine), events_(events) {
    // A failing scenario destroys Driver before Engine drains its outstanding
    // command. The completion callback must keep its own wake state alive.
    engine_.setCompletionNotifier([wake = wake_] {
      {
        std::lock_guard lock(wake->mutex);
        wake->notified = true;
      }
      wake->condition.notify_one();
    });
  }

  void untilIdle(
      const std::function<void(const BatchObservation &)> &observer =
          {}) {
    const auto deadline = Clock::now() + std::chrono::hours(2);
    uint64_t observedBatchSequence = events_.batchSequence();
    while (!drained()) {
      if (engine_.tick(milliseconds(Clock::now()))) {
        const uint64_t sequence = events_.batchSequence();
        if (sequence != observedBatchSequence) {
          if (sequence != observedBatchSequence + 1)
            throw std::logic_error("benchmark skipped a completed batch");
          observedBatchSequence = sequence;
          if (observer)
            observer(events_.lastBatch());
        }
        continue;
      }
      if (Clock::now() >= deadline)
        throw std::runtime_error("backend benchmark timed out");
      auto wakeup = deadline;
      if (auto next = engine_.nextWakeupMilliseconds()) {
        wakeup = std::min(wakeup, Clock::time_point{
            std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double, std::milli>(*next))});
      }
      std::unique_lock lock(wake_->mutex);
      wake_->condition.wait_until(lock, wakeup, [this] {
        if (!wake_->notified)
          return false;
        wake_->notified = false;
        return true;
      });
    }
  }

private:
  struct WakeState final {
    std::mutex mutex;
    std::condition_variable condition;
    bool notified = false;
  };

  // Every submitted request has ended and no command is in flight.
  [[nodiscard]] bool drained() const {
    const engine::EngineSnapshot counts = engine_.snapshot();
    return counts.submitted ==
               counts.completed + counts.cancelled + counts.failed &&
           !engine_.commandInFlight();
  }

  engine::Engine &engine_;
  Events &events_;
  std::shared_ptr<WakeState> wake_ = std::make_shared<WakeState>();
};

std::vector<uint32_t> prompt(uint32_t length, uint64_t salt) {
  std::vector<uint32_t> result(length);
  uint64_t state = salt ^ 0x9e3779b97f4a7c15ULL;
  for (uint32_t &token : result) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    token = 100 + static_cast<uint32_t>((state >> 17) % 200000);
  }
  return result;
}

// What each lookup of a cache check found, for its failure: under host
// memory pressure the engine evicts the cached prefixes the checks reuse.
std::string lookups(
    std::initializer_list<std::reference_wrapper<const Measurement>> requests,
    uint32_t expectedTokens) {
  std::string found;
  for (const Measurement &request : requests)
    found += request.scenario + "=" + request.cacheStatus + "/" +
             std::to_string(request.matchedTokens) + " ";
  return found + "(expected a " + std::to_string(expectedTokens) +
         "-token hit)";
}

Measurement runRequest(engine::Engine &engine, Driver &driver,
                       model::RuntimeModel &executor, Events &events,
                       ProgressJournal *progress, uint64_t requestId,
                       std::string scenario, uint32_t sample,
                       std::vector<uint32_t> tokens,
                       uint32_t maxNewTokens = 1) {
  const uint32_t promptTokens = static_cast<uint32_t>(tokens.size());
  const uint64_t inputTokenHash = tokenSequenceHash(tokens);
  if (progress)
    progress->begin("prefill", scenario, sample, promptTokens, 0, inputTokenHash);
  EngineRequest request;
  request.id = requestId;
  request.prompt = std::move(tokens);
  request.maxNewTokens = maxNewTokens;
  request.sampling.temperature = 0.0F;
  request.sampling.topP = 1.0F;
  request.deadlineMilliseconds =
      milliseconds(Clock::now() + std::chrono::hours(2));

  const model::ModelTelemetry before = executor.telemetry();
  const engine::EngineSnapshot engineBefore = engine.snapshot();
  std::vector<uint32_t> prefillBatchRows;
  prefillBatchRows.reserve(
      promptTokens / model::ExecutionLimits::prefillTokenBudget + 3);
  const auto began = Clock::now();
  engine.submit(std::move(request));
  driver.untilIdle([&](const BatchObservation &batch) {
    if (batch.kind == WorkKind::Prefill)
      prefillBatchRows.push_back(batch.inputTokens);
  });
  const auto ended = Clock::now();
  const model::ModelTelemetry after = executor.telemetry();
  const engine::EngineSnapshot engineAfter = engine.snapshot();
  const Observation &observed = events.get(requestId);
  if (!observed.completed || !observed.failure.empty()) {
    throw std::runtime_error("request failed: " + observed.failure);
  }

  Measurement result;
  result.scenario = std::move(scenario);
  result.sample = sample;
  result.promptTokens = promptTokens;
  result.cacheStatus = observed.cacheStatus;
  result.matchedTokens = observed.matchedTokens;
  result.wallMilliseconds = milliseconds(ended) - milliseconds(began);
  result.ttftMilliseconds =
      observed.firstToken
          ? milliseconds(*observed.firstToken) - milliseconds(began)
          : result.wallMilliseconds;
  result.prefillGpuMilliseconds =
      (after.totalPrefillGpuSeconds - before.totalPrefillGpuSeconds) * 1000.0;
  result.decodeGpuMilliseconds =
      (after.totalDecodeGpuSeconds - before.totalDecodeGpuSeconds) * 1000.0;
  result.targetPrefillRows = after.targetPrefillRows - before.targetPrefillRows;
  result.draftContextRows =
      after.draftContextRowsActive + after.draftContextRowsMaterialization -
      before.draftContextRowsActive - before.draftContextRowsMaterialization;
  result.draftContextRowsAvoided =
      after.draftContextRowsAvoided - before.draftContextRowsAvoided;
  result.draftStateRestoreSkipped =
      after.draftStateRestoreSkipped - before.draftStateRestoreSkipped;
  result.junctionMaterializations = engineAfter.junctionMaterializations -
                                    engineBefore.junctionMaterializations;
  result.outputTokens = observed.outputTokens;
  if (progress) {
    progress->complete(
        "request", result.scenario, result.sample, result.promptTokens, 0,
        result.wallMilliseconds,
        result.prefillGpuMilliseconds + result.decodeGpuMilliseconds,
        result.ttftMilliseconds, result.cacheStatus, result.matchedTokens,
        result.targetPrefillRows, result.draftContextRows,
        std::nullopt, std::nullopt, std::nullopt, inputTokenHash,
        result.outputTokens, prefillBatchRows);
  }
  return result;
}

// One reclaim pass evicts every unpinned entry and releases the extents it
// empties.
void evictAllCache(engine::Cache &resources) {
  static_cast<void>(resources.evictAll());
  const engine::CacheSnapshot snapshot = resources.snapshot();
  if (snapshot.stateCache.entries || snapshot.pool.pagesPrefix)
    throw std::logic_error("native benchmark cache did not drain");
}

// Evicts every cached state, one at a time, and leaves the KV graph intact.
void evictAllCompositeState(engine::Cache &resources) {
  while (resources.snapshot().stateCache.entries) {
    if (!resources.reclaimStateForLane(engine::ReclaimClass::InUse).madeProgress)
      throw std::logic_error("native benchmark state cache made no progress");
  }
}

DecodeThroughputMeasurement
runDecodeThroughput(engine::Engine &engine, Driver &driver,
                    model::RuntimeModel &executor, Events &events,
                    ProgressJournal *progress, uint64_t &requestId,
                    uint32_t width, uint32_t sample,
                    std::span<const uint32_t> tokens) {
  if (!width || width > model::ExecutionLimits::maximumBatchWidth) {
    throw std::invalid_argument("invalid decode throughput width");
  }
  if (progress)
    progress->begin("decode_throughput", "decode", sample, tokens.size(), width);

  // The lanes share one prompt. The first request to reach a Page32 boundary
  // publishes the composite state there and splits its prefill around it, so
  // it would start decoding one command after the deduplicated lanes. Warm
  // the prefix with a one-token request; every lane then resumes from the
  // published prefix in a single ragged prefill and decodes in lockstep.
  {
    EngineRequest warm;
    warm.id = requestId++;
    warm.prompt.assign(tokens.begin(), tokens.end());
    warm.maxNewTokens = 1;
    warm.sampling.temperature = 0.0F;
    warm.sampling.topP = 1.0F;
    warm.deadlineMilliseconds =
        milliseconds(Clock::now() + std::chrono::hours(2));
    engine.submit(std::move(warm));
    driver.untilIdle();
    const Observation &observed = events.get(requestId - 1);
    if (!observed.completed || !observed.failure.empty()) {
      throw std::runtime_error("decode throughput warm-up failed: " +
                               observed.failure);
    }
  }

  const model::ModelTelemetry telemetryBefore =
      executor.telemetry();
  const engine::SchedulerSnapshot schedulerBefore = engine.snapshot().scheduler;
  std::vector<uint64_t> requestIds;
  requestIds.reserve(width);
  uint64_t decodeBatches = 0;
  uint64_t decodeOutputTokens = 0;
  uint64_t draftedTokens = 0;
  uint64_t acceptedDraftTokens = 0;
  const auto began = Clock::now();
  for (uint32_t lane = 0; lane < width; ++lane) {
    EngineRequest request;
    request.id = requestId++;
    request.prompt.assign(tokens.begin(), tokens.end());
    request.maxNewTokens = 64;
    request.sampling.temperature = 0.0F;
    request.sampling.topP = 1.0F;
    request.deadlineMilliseconds =
        milliseconds(began + std::chrono::hours(2));
    requestIds.push_back(request.id);
    engine.submit(std::move(request));
  }
  driver.untilIdle([&](const BatchObservation &batch) {
    if (batch.kind != WorkKind::Decode)
      return;
    if (batch.width != width) {
      throw std::runtime_error(
          "decode throughput batch used the wrong physical width");
    }
    ++decodeBatches;
    decodeOutputTokens += batch.outputTokens;
    draftedTokens += batch.draftedTokens;
    acceptedDraftTokens += batch.acceptedDraftTokens;
  });
  const auto ended = Clock::now();
  const model::ModelTelemetry telemetryAfter = executor.telemetry();
  const engine::SchedulerSnapshot schedulerAfter = engine.snapshot().scheduler;

  uint32_t completedTokens = 0;
  const Observation *reference = nullptr;
  for (uint64_t id : requestIds) {
    const Observation &observed = events.get(id);
    if (!observed.completed || !observed.failure.empty() ||
        observed.outputTokens.size() != 64) {
      throw std::runtime_error(
          "decode throughput request did not produce exactly 64 tokens");
    }
    if (reference && reference->outputTokens != observed.outputTokens) {
      throw std::runtime_error("decode throughput lanes diverged");
    }
    reference = &observed;
    completedTokens += static_cast<uint32_t>(observed.outputTokens.size());
  }

  for (uint32_t candidate = 1;
       candidate <= model::ExecutionLimits::maximumBatchWidth; ++candidate) {
    const uint64_t delta =
        schedulerAfter.decodeBatchesByWidth[candidate - 1] -
        schedulerBefore.decodeBatchesByWidth[candidate - 1];
    if ((candidate == width && !delta) || (candidate != width && delta)) {
      throw std::runtime_error(
          "decode throughput request used a different physical batch width");
    }
  }

  DecodeThroughputMeasurement result;
  result.width = width;
  result.sample = sample;
  result.outputTokens = completedTokens;
  result.decodeOutputTokens = decodeOutputTokens;
  result.decodeBatches = decodeBatches;
  result.draftedTokens = draftedTokens;
  result.acceptedDraftTokens = acceptedDraftTokens;
  result.requestWallMilliseconds = milliseconds(ended) - milliseconds(began);
  result.prefillWallMilliseconds =
      (telemetryAfter.totalPrefillWallSeconds -
       telemetryBefore.totalPrefillWallSeconds) *
      1000.0;
  result.prefillGpuMilliseconds =
      (telemetryAfter.totalPrefillGpuSeconds -
       telemetryBefore.totalPrefillGpuSeconds) *
      1000.0;
  result.decodeWallMilliseconds =
      (telemetryAfter.totalDecodeWallSeconds -
       telemetryBefore.totalDecodeWallSeconds) *
      1000.0;
  result.decodeGpuMilliseconds =
      (telemetryAfter.totalDecodeGpuSeconds -
       telemetryBefore.totalDecodeGpuSeconds) *
      1000.0;
  if (!(result.requestWallMilliseconds > 0.0) ||
      !(result.prefillWallMilliseconds > 0.0) ||
      !(result.prefillGpuMilliseconds > 0.0) ||
      !(result.decodeWallMilliseconds > 0.0) ||
      !(result.decodeGpuMilliseconds > 0.0)) {
    throw std::runtime_error("decode throughput timing is not positive");
  }
  result.aggregateRequestWallTokensPerSecond =
      static_cast<double>(completedTokens) * 1000.0 /
      result.requestWallMilliseconds;
  result.aggregateDecodeWallTokensPerSecond =
      static_cast<double>(decodeOutputTokens) * 1000.0 /
      result.decodeWallMilliseconds;
  result.aggregateGpuTokensPerSecond =
      static_cast<double>(decodeOutputTokens) * 1000.0 /
      result.decodeGpuMilliseconds;
  if (!result.decodeBatches || !decodeOutputTokens ||
      decodeOutputTokens > completedTokens) {
    throw std::runtime_error("decode throughput work accounting is invalid");
  }
  if (draftedTokens != uint64_t{width} * decodeBatches *
                           model::ExecutionLimits::draftProposalTokens ||
      acceptedDraftTokens > draftedTokens) {
    throw std::runtime_error("decode throughput DFlash accounting is invalid");
  }
  result.outputTokensPerLaneBatch =
      static_cast<double>(decodeOutputTokens) /
      static_cast<double>(width * result.decodeBatches);
  result.draftAcceptanceRate =
      result.draftedTokens
          ? static_cast<double>(result.acceptedDraftTokens) /
                static_cast<double>(result.draftedTokens)
          : 0.0;
  result.firstLaneOutputTokens = reference->outputTokens;
  if (progress) {
    progress->complete("decode_throughput", "B" + std::to_string(width),
                       sample, tokens.size(), width,
                       result.requestWallMilliseconds,
                       result.prefillGpuMilliseconds +
                           result.decodeGpuMilliseconds,
                       0.0, {}, 0, 0, 0,
                       result.decodeBatches, result.draftedTokens,
                       result.acceptedDraftTokens);
  }
  return result;
}

void emitMeasurement(const Measurement &value, bool first) {
  if (!first)
    std::cout << ',';
  std::cout << "{\"scenario\":\"" << value.scenario
            << "\",\"sample\":" << value.sample
            << ",\"prompt_tokens\":" << value.promptTokens
            << ",\"target_prefill_rows\":" << value.targetPrefillRows
            << ",\"cache_status\":\"" << value.cacheStatus
            << "\",\"matched_tokens\":" << value.matchedTokens
            << ",\"wall_ms\":" << value.wallMilliseconds
            << ",\"ttft_ms\":" << value.ttftMilliseconds
            << ",\"prefill_gpu_ms\":" << value.prefillGpuMilliseconds
            << ",\"decode_gpu_ms\":" << value.decodeGpuMilliseconds
            << ",\"draft_context_rows\":" << value.draftContextRows
            << ",\"draft_context_rows_avoided\":"
            << value.draftContextRowsAvoided
            << ",\"draft_state_restore_skipped\":"
            << value.draftStateRestoreSkipped
            << ",\"junction_materializations\":"
            << value.junctionMaterializations << ",\"output_tokens\":[";
  for (size_t index = 0; index < value.outputTokens.size(); ++index) {
    if (index)
      std::cout << ',';
    std::cout << value.outputTokens[index];
  }
  std::cout << ']';
  if (value.coldOutputMatch)
    std::cout << ",\"cold_output_match\":"
              << (*value.coldOutputMatch ? "true" : "false");
  std::cout << '}';
}

uint32_t parseSamples(std::string_view value) {
  uint64_t parsed = 0;
  for (char character : value) {
    if (character < '0' || character > '9')
      throw std::invalid_argument("samples must be a positive integer");
    parsed = parsed * 10 + static_cast<uint64_t>(character - '0');
    if (parsed > 100)
      throw std::invalid_argument("samples must not exceed 100");
  }
  if (!parsed)
    throw std::invalid_argument("samples must be a positive integer");
  return static_cast<uint32_t>(parsed);
}

// A context limit, as serve's --max-context: a positive token count.
uint32_t parseMaxContext(std::string_view value) {
  uint32_t tokens = 0;
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), tokens);
  if (error != std::errc{} || end != value.data() + value.size() || !tokens)
    throw std::invalid_argument("--max-context takes a positive token count");
  return tokens;
}

// --ane-ffn-minimum-rows: the least rows of a chunk the prefill FFN's Neural
// Engine split takes, a positive count.
uint32_t parseAneFfnMinimumRows(std::string_view value) {
  uint32_t rows = 0;
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), rows);
  if (error != std::errc{} || end != value.data() + value.size() || !rows)
    throw std::invalid_argument("--ane-ffn-minimum-rows takes a positive row count");
  return rows;
}

// --ane-ffn-share: the prefill FFN's Neural Engine share, in [0, 1).
double parseAneFfnShare(std::string_view value) {
  double share = 0.0;
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), share);
  if (error != std::errc{} || end != value.data() + value.size() || !(share >= 0.0 && share < 1.0))
    throw std::invalid_argument("--ane-ffn-share takes a share in [0, 1)");
  return share;
}

double median(std::vector<double> values) {
  if (values.empty())
    throw std::invalid_argument("cannot take the median of no samples");
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() & 1 ? values[middle]
                           : (values[middle - 1] + values[middle]) / 2.0;
}

// The aggregate wall decode throughput of each sample at one width.
std::vector<double> decodeWallThroughputs(
    std::span<const DecodeThroughputMeasurement> measurements,
    uint32_t width) {
  std::vector<double> values;
  for (const DecodeThroughputMeasurement &measurement : measurements) {
    if (measurement.width == width)
      values.push_back(measurement.aggregateDecodeWallTokensPerSecond);
  }
  if (values.empty())
    throw std::invalid_argument("no decode throughput samples at this width");
  return values;
}

// The scenarios one run measures: decode, partial and context by default, or
// the comma-separated set --scenario names. exact measures the context
// lengths as repeated restores instead, so it excludes context.
struct BenchmarkScenarios final {
  bool decode = true;
  bool partial = true;
  // The short scenario; short is a keyword.
  bool shortPrompts = false;
  bool context = true;
  bool exact = false;
};

BenchmarkScenarios parseScenarios(std::string_view value) {
  BenchmarkScenarios selected{false, false, false, false, false};
  for (;;) {
    const size_t comma = value.find(',');
    const std::string_view name = value.substr(0, comma);
    bool *scenario = name == "decode"    ? &selected.decode
                     : name == "partial" ? &selected.partial
                     : name == "short"   ? &selected.shortPrompts
                     : name == "context" ? &selected.context
                     : name == "exact"   ? &selected.exact
                                         : nullptr;
    if (!scenario)
      throw std::invalid_argument("unknown benchmark scenario");
    if (*scenario)
      throw std::invalid_argument("benchmark scenario named twice");
    *scenario = true;
    if (comma == std::string_view::npos)
      break;
    value.remove_prefix(comma + 1);
  }
  if (selected.context && selected.exact)
    throw std::invalid_argument("exact and context scenarios exclude each other");
  return selected;
}

} // namespace

int main(int argc, char **argv) {
  std::unique_ptr<ProgressJournal> progress;
  try {
    if (argc < 3) {
      std::cerr << "usage: backend-benchmark METALLIB MODEL_ROOT "
                   "[--samples COUNT] [--progress PATH] "
                   "[--scenario NAME[,NAME...]] [--ane-ffn-share SHARE "
                   "[--ane-ffn-minimum-rows ROWS]] [--max-context TOKENS] "
                   "[--kv-format int8|bf16|zbf16]\n"
                   "  NAME: decode, partial, short, context or exact "
                   "(default: decode,partial,context)\n"
                   "  SHARE: the prefill FFN's Neural Engine share in [0, 1) "
                   "to run instead of calibrating one (0: GPU alone)\n"
                   "  ROWS: the least rows of a chunk that share takes "
                   "(default: 512)\n"
                   "  TOKENS: the context the engine serves, as serve's "
                   "--max-context (default: what memory holds)\n";
      return 2;
    }
    uint32_t samples = 1;
    BenchmarkScenarios selected;
    std::optional<std::filesystem::path> progressPath;
    std::optional<double> aneFfnShare;
    std::optional<uint32_t> aneFfnMinimumRows;
    uint32_t maxContext = 0;
    kv::Format kvFormat = kv::Format::Int8;
    for (int index = 3; index < argc; index += 2) {
      if (index + 1 >= argc)
        throw std::invalid_argument("benchmark option requires a value");
      const std::string_view option(argv[index]);
      if (option == "--samples") {
        samples = parseSamples(argv[index + 1]);
      } else if (option == "--progress") {
        progressPath = std::filesystem::path(argv[index + 1]);
      } else if (option == "--scenario") {
        selected = parseScenarios(argv[index + 1]);
      } else if (option == "--ane-ffn-share") {
        aneFfnShare = parseAneFfnShare(argv[index + 1]);
      } else if (option == "--ane-ffn-minimum-rows") {
        aneFfnMinimumRows = parseAneFfnMinimumRows(argv[index + 1]);
      } else if (option == "--max-context") {
        maxContext = parseMaxContext(argv[index + 1]);
      } else if (option == "--kv-format") {
        const std::string_view value(argv[index + 1]);
        if (value != "int8" && value != "bf16" && value != "zbf16")
          throw std::invalid_argument("--kv-format takes int8, bf16 or zbf16");
        kvFormat = value == "int8"   ? kv::Format::Int8
                   : value == "bf16" ? kv::Format::BFloat16
                                     : kv::Format::ZipBFloat16;
      } else {
        throw std::invalid_argument("unknown benchmark option");
      }
    }
    if (progressPath)
      progress = std::make_unique<ProgressJournal>(*progressPath, samples);

    engine::RuntimeBootstrapConfig bootstrapConfig;
    engine::RuntimeMetrics metrics;
    bootstrapConfig.nativeLoop.metrics = &metrics;
    auto &config = bootstrapConfig.resources;
    config.metallibPath = std::filesystem::path(argv[1]);
    config.modelRoot = std::filesystem::path(argv[2]);
    config.model = model::inspectModelRoot(config.modelRoot);
    config.buildId = SPLASH_BUILD_ID;
    config.kvFormat = kvFormat;
    config.aneFfn = engine::AneFfnSetting::fromGiven(aneFfnShare, aneFfnMinimumRows);
    bootstrapConfig.nativeLoop.engine.maxContext = maxContext;
    const std::string modelRoot = config.modelRoot.string();
    const auto &capabilities = config.model.capabilities;
    // Complete production warmup and memory audit before measuring. Retry
    // host-capacity refusals while memory from the previous engine settles.
    std::unique_ptr<engine::RuntimeBootstrap> bootstrap;
    const auto memoryDeadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(180);
    bool waitingAnnounced = false;
    while (!bootstrap) {
      try {
        bootstrap = engine::RuntimeBootstrap::start(
            bootstrapConfig, [](std::span<const uint8_t>) {},
            []() -> std::string {
              throw std::logic_error("benchmark does not accept native status requests");
            });
      } catch (const engine::RuntimeBootstrapError &error) {
        if (error.report().resourceFailure !=
                engine::RuntimeResourceFailure::HostCapacity ||
            std::chrono::steady_clock::now() >= memoryDeadline)
          throw;
        if (!waitingAnnounced) {
          std::cerr << "backend-benchmark: waiting for host memory to settle "
                       "before loading the model\n";
          waitingAnnounced = true;
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
      }
    }
    // start() returns once the loop has announced Ready.
    if (!bootstrap->nativeLoop().engineHealthy() ||
        bootstrap->nativeLoop().commandInFlight())
      throw std::runtime_error("benchmark production bootstrap did not finish idle and ready");
    // Non-owning borrows. This scope never feeds or ticks the bootstrap loop;
    // its Engine stays empty. The later benchmark Engine is the sole request
    // driver and is destroyed before the bootstrap owner/model/shared cache.
    auto *resources = &bootstrap->resources();
    // The Neural Engine split startup calibrated, or the one given.
    const double ranAneFfnShare = resources->aneFfnShare();
    const uint32_t ranAneFfnMinimumRows = resources->aneFfnMinimumRows();
    auto *executor = &bootstrap->modelRuntime();
    const auto &cacheIdentity = resources->cacheIdentity();
    const std::string identity =
        "{\"model_root\":" + json::quote(modelRoot) +
        ",\"loaded_model_layout_sha256\":" +
        json::quote(cacheIdentity.modelLayoutSha256) + ",\"device\":" +
        json::quote(resources->backend().capabilities().deviceName) + "}";
    if (progress)
      progress->identity(identity);

    // The warmups run on the startup runway, pages [0, warmupKvPages). After
    // startup the pool keeps one warm extent, which holds the runway only
    // when one extent covers it (INT8 and ZBF16 at their extent sizes, not
    // BF16's): otherwise the warmup timings are skipped and reported as 0.
    bool runway = true;
    for (uint32_t page = 0; page < model::ExecutionLimits::warmupKvPages; ++page)
      runway = runway && resources->kvPages().isAllocated(page);
    if (!runway)
      std::cerr << "backend-benchmark: the startup runway was released; skipping warmup timings\n";
    if (progress)
      progress->begin("warmup", "prefill_2048", 0,
                      model::ExecutionLimits::prefillTokenBudget);
    const model::WarmupStepResult prefillWarmup =
        runway ? executor->warmupPrefill(model::ExecutionLimits::prefillTokenBudget)
               : model::WarmupStepResult{};
    const double prefillWarmupGpuMilliseconds =
        runway ? executor->telemetry().lastPrefillGpuSeconds * 1000.0 : 0.0;
    if (progress) {
      progress->complete("prefill_warmup", "prefill_2048", 0,
                         model::ExecutionLimits::prefillTokenBudget, 0,
                         prefillWarmup.wallSeconds * 1000.0,
                         prefillWarmupGpuMilliseconds, 0.0);
    }
    std::array<double, model::ExecutionLimits::maximumBatchWidth>
        decodeWarmupWall{};
    std::array<double, model::ExecutionLimits::maximumBatchWidth>
        decodeWarmupGpu{};
    std::vector<std::string> performanceFailures;
    for (uint32_t width = 1; runway && width <= decodeWarmupWall.size(); ++width) {
      decodeWarmupWall[width - 1] =
          executor->warmupDecodeBatch(width).wallSeconds * 1000.0;
      decodeWarmupGpu[width - 1] =
          executor->telemetry().lastDecodeGpuSeconds * 1000.0;
    }

    Events events;
    engine::EngineConfig engineConfig;
    engineConfig.maxContext = bootstrap->nativeLoop().snapshot().maximumContextTokens;
    engineConfig.vocabularySize = capabilities.vocabularySize;
    engine::connectToGovernor(engineConfig, resources->memoryGovernor());
    engine::Engine engine(engineConfig, resources->cache(),
                                  *executor, events);
    Driver driver(engine, events);

    // All four widths receive the same production-token payload. Different
    // batch kernels may round differently; report their actual output hashes
    // and acceptance counts rather than require a batch-invariant transcript.
    // Historical throughput is intentionally not embedded here: frequency
    // state changes it materially even on one Mac.
    // This executable characterizes one source identity and checks internal
    // invariants; a source-regression claim requires a separately captured
    // baseline report and an explicit comparison outside this process.
    constexpr std::array<uint32_t, 39> decodeThroughputPrompt{
        248045, 846,   198,   2427, 38453, 494,   220,   16,
        11,     4237,  1754,  1324, 321,   1141,  6163,  803,
        383,    264,   491,   1500, 13,    14569, 2980,  488,
        5372,   220,   17,    15,   15,    13,    248046, 198,
        248045, 74455, 198,   248068, 271,  248069, 271};
    constexpr std::string_view decodeThroughputPromptSha256 =
        "8c9bac848ac2e727f235beda84f06bac1a9320cc7a014d72da24263d07a3d982";
    std::vector<DecodeThroughputMeasurement> decodeThroughput;
    decodeThroughput.reserve(model::ExecutionLimits::maximumBatchWidth * samples);
    uint64_t requestId = 1;
    if (selected.decode) {
      for (uint32_t sample = 0; sample < samples; ++sample) {
        for (uint32_t offset = 0;
             offset < model::ExecutionLimits::maximumBatchWidth; ++offset) {
          const uint32_t width =
              1 + (sample + offset) % model::ExecutionLimits::maximumBatchWidth;
          evictAllCache(resources->cache());
          decodeThroughput.push_back(runDecodeThroughput(
              engine, driver, *executor, events, progress.get(), requestId,
              width, sample, decodeThroughputPrompt));
        }
      }
      // A wider batch must not lose aggregate throughput. A dense model can
      // fill the GPU by width 3, leaving B3 within B2's noise, so the gate
      // fails only when every B3 sample falls below every B2 sample; how far
      // widths scale is for a comparison with a baseline build.
      const std::vector<double> b2 = decodeWallThroughputs(decodeThroughput, 2);
      const std::vector<double> b3 = decodeWallThroughputs(decodeThroughput, 3);
      if (*std::max_element(b3.begin(), b3.end()) <
          *std::min_element(b2.begin(), b2.end())) {
        performanceFailures.push_back(
            "B3 aggregate decode throughput fell below B2");
      }
      // Nor may a B3 step take longer than a B1 step and a B2 step, which
      // would decode the three lanes sooner apart. Every width is timed over
      // its own run of steps here. A warmup command is not comparable across
      // widths: it follows one setup prefill per lane, so where sustained
      // load lowers the GPU clock (an M3 Max in Low Power Mode) a wider one
      // runs at a lower clock.
      const auto stepGpuMilliseconds = [&](uint32_t width) {
        std::vector<double> values;
        for (const DecodeThroughputMeasurement &measurement : decodeThroughput) {
          if (measurement.width == width)
            values.push_back(measurement.decodeGpuMilliseconds /
                             static_cast<double>(measurement.decodeBatches));
        }
        return median(std::move(values));
      };
      if (stepGpuMilliseconds(3) >
          stepGpuMilliseconds(1) + stepGpuMilliseconds(2)) {
        performanceFailures.push_back(
            "a B3 decode step takes longer than a B1 step and a B2 step");
      }
    }

    constexpr std::array<uint32_t, 4> lengths{2048, 10000, 50000, 128000};
    // The rolling request, a length's longest, appends this many tokens.
    constexpr uint32_t rollingSuffixTokens = 256;
    std::vector<Measurement> measurements;
    // Lengths the memory plan cannot hold, on a Mac with less memory.
    std::vector<uint32_t> skippedLengths;
    for (uint32_t length : selected.context || selected.exact
                               ? std::span<const uint32_t>{lengths}
                               : std::span<const uint32_t>{}) {
      // Every request generates at least one token beyond its prompt.
      if (length + (selected.exact ? 0 : rollingSuffixTokens) >=
          engineConfig.maxContext) {
        skippedLengths.push_back(length);
        continue;
      }
      // Repeat short contexts for timing; run the costly 50K/128K cases once.
      const uint32_t lengthSamples =
          !selected.exact && length <= 10000 ? samples : 1;
      for (uint32_t sample = 0; sample < lengthSamples; ++sample) {
        evictAllCache(resources->cache());
        // Every cache prompt ends with the chat-formatted decode prompt so the
        // answer request below produces a real multi-block greedy answer.
        std::vector<uint32_t> cold = prompt(
            length - static_cast<uint32_t>(decodeThroughputPrompt.size()),
            uint64_t{length} << 32 | sample);
        cold.insert(cold.end(), decodeThroughputPrompt.begin(),
                    decodeThroughputPrompt.end());
        Measurement coldResult =
            runRequest(engine, driver, *executor, events, progress.get(),
                       requestId++, "cold", sample, cold);
        Measurement exactResult =
            runRequest(engine, driver, *executor, events, progress.get(),
                       requestId++, "exact", sample, cold);
        const uint32_t expectedBoundary =
            ((length - 1) / kv::kPageTokens) *
            kv::kPageTokens;
        if (coldResult.cacheStatus != "miss" ||
            exactResult.cacheStatus != "prefix_hit" ||
            exactResult.matchedTokens != expectedBoundary) {
          throw std::runtime_error("cold/exact cache oracle failed: " +
                                   lookups({coldResult, exactResult},
                                           expectedBoundary));
        }
        // Cache reuse changes chunking and reduction order. Output agreement
        // is diagnostic; the runtime/Metal oracles validate state and numerics.
        exactResult.coldOutputMatch = coldResult.outputTokens == exactResult.outputTokens;
        if (selected.exact) {
          // Retain one cold seed and every hit, including the first. Repeated
          // long-context restore timing must not require repeated cold prefill.
          for (uint32_t hit = 0; hit < samples; ++hit) {
            if (hit)
              exactResult = runRequest(engine, driver, *executor, events,
                                       progress.get(), requestId++, "exact", hit, cold);
            if (exactResult.cacheStatus != "prefix_hit" ||
                exactResult.matchedTokens != expectedBoundary ||
                exactResult.targetPrefillRows != length - expectedBoundary)
              throw std::runtime_error("repeated exact cache oracle failed: " +
                                       lookups({exactResult}, expectedBoundary));
            exactResult.coldOutputMatch = coldResult.outputTokens == exactResult.outputTokens;
            if (exactResult.ttftMilliseconds * 2 >= coldResult.ttftMilliseconds)
              performanceFailures.push_back("repeated exact TTFT did not improve twofold");
            measurements.push_back(std::move(exactResult));
          }
          measurements.push_back(std::move(coldResult));
          continue;
        }
        // An agent turn appends the previous answer and a new message. The
        // continuation must reuse the prompt; its actual restoration boundary
        // and agreement with a cold evaluation are reported separately.
        constexpr uint32_t answerTokens = 96;
        Measurement answerResult = runRequest(
            engine, driver, *executor, events, progress.get(), requestId++,
            "answer", sample, cold, answerTokens);
        std::vector<uint32_t> continuation = cold;
        continuation.insert(continuation.end(),
                            answerResult.outputTokens.begin(),
                            answerResult.outputTokens.end());
        std::vector<uint32_t> turn =
            prompt(64, (uint64_t{length} << 32 | sample) ^ 0x5a5a5a5aULL);
        continuation.insert(continuation.end(), turn.begin(), turn.end());
        Measurement continuationResult =
            runRequest(engine, driver, *executor, events, progress.get(),
                       requestId++, "continuation", sample, continuation);
        if (answerResult.cacheStatus != "prefix_hit" ||
            answerResult.outputTokens.size() <=
                kv::kPageTokens) {
          throw std::runtime_error(
              "answer did not reuse its prompt or stopped after " +
              std::to_string(answerResult.outputTokens.size()) + " tokens");
        }
        std::vector<uint32_t> rolling = cold;
        std::vector<uint32_t> suffix =
            prompt(rollingSuffixTokens,
                   (uint64_t{length} << 32 | sample) ^ 0xa5a5a5a5ULL);
        rolling.insert(rolling.end(), suffix.begin(), suffix.end());
        Measurement rollingResult =
            runRequest(engine, driver, *executor, events, progress.get(),
                       requestId++, "rolling", sample, std::move(rolling));
        evictAllCache(resources->cache());
        std::vector<uint32_t> rollingReference = cold;
        rollingReference.insert(rollingReference.end(), suffix.begin(),
                                suffix.end());
        Measurement rollingCold = runRequest(
            engine, driver, *executor, events, progress.get(), requestId++,
            "rolling_cold", sample, std::move(rollingReference));
        if (rollingResult.cacheStatus != "prefix_hit" ||
            rollingCold.cacheStatus != "miss" ||
            rollingResult.matchedTokens != expectedBoundary) {
          throw std::runtime_error("rolling cache oracle failed: " +
                                   lookups({rollingResult, rollingCold},
                                           expectedBoundary));
        }
        rollingResult.coldOutputMatch = rollingResult.outputTokens == rollingCold.outputTokens;
        evictAllCache(resources->cache());
        Measurement continuationCold = runRequest(
            engine, driver, *executor, events, progress.get(), requestId++,
            "continuation_cold", sample, std::move(continuation));
        if (continuationResult.cacheStatus != "prefix_hit" ||
            continuationCold.cacheStatus != "miss") {
          throw std::runtime_error(
              "continuation cache status oracle failed: hit=" +
              continuationResult.cacheStatus +
              " cold=" + continuationCold.cacheStatus);
        }
        continuationResult.coldOutputMatch =
            continuationResult.outputTokens == continuationCold.outputTokens;
        if (coldResult.draftContextRows != expectedDraftContextRows(
                length, engine::kPrefillCheckpointTokens)) {
          throw std::runtime_error(
              "cold prefill performed unnecessary draft-context work");
        }
        if (exactResult.ttftMilliseconds * 2.0 >=
            coldResult.ttftMilliseconds) {
          performanceFailures.push_back(
              std::to_string(length) + "-token exact sample " +
              std::to_string(sample) +
              " did not improve TTFT by at least two times");
        }
        measurements.push_back(std::move(coldResult));
        measurements.push_back(std::move(exactResult));
        measurements.push_back(std::move(rollingResult));
        measurements.push_back(std::move(rollingCold));
        measurements.push_back(std::move(answerResult));
        measurements.push_back(std::move(continuationResult));
        measurements.push_back(std::move(continuationCold));
      }
    }

    // A 4K suffix rebuilds the windows required by its recovery boundaries.
    // Compare against the same prompt evaluated cold, then recreate its 10K
    // prefix so the second evaluation is a real partial state-backed hit.
    if (selected.partial) {
      std::vector<uint32_t> partialBase = prompt(10000, 0x5041525449414cULL);
      std::vector<uint32_t> partialPrompt = partialBase;
      std::vector<uint32_t> partialSuffix = prompt(4096, 0x535546464958ULL);
      partialPrompt.insert(partialPrompt.end(), partialSuffix.begin(),
                           partialSuffix.end());
      for (uint32_t sample = 0; sample < samples; ++sample) {
        evictAllCache(resources->cache());
        Measurement partialCold =
            runRequest(engine, driver, *executor, events, progress.get(),
                       requestId++, "partial_4k_cold", sample, partialPrompt);
        evictAllCache(resources->cache());
        Measurement partialSeed =
            runRequest(engine, driver, *executor, events, progress.get(),
                       requestId++, "partial_4k_seed", sample, partialBase);
        Measurement partialHit =
            runRequest(engine, driver, *executor, events, progress.get(),
                       requestId++, "partial_4k_hit", sample, partialPrompt);
        const uint32_t partialBoundary =
            ((partialBase.size() - 1) / kv::kPageTokens) *
            kv::kPageTokens;
        const uint64_t expectedPartialRows = expectedDraftContextRows(
            partialPrompt.size(), engine::kPrefillCheckpointTokens,
            partialBoundary);
        if (partialSeed.cacheStatus != "miss" ||
            partialHit.cacheStatus != "prefix_hit" ||
            partialHit.matchedTokens != partialBoundary ||
            partialHit.draftStateRestoreSkipped != 1 ||
            partialHit.draftContextRows != expectedPartialRows) {
          throw std::runtime_error(
              "4K partial-hit draft-window oracle failed: seed_status=" +
              partialSeed.cacheStatus + " hit_status=" +
              partialHit.cacheStatus +
              " matched=" + std::to_string(partialHit.matchedTokens) +
              " expected_matched=" + std::to_string(partialBoundary) +
              " restore_skipped=" +
              std::to_string(partialHit.draftStateRestoreSkipped) +
              " draft_rows=" + std::to_string(partialHit.draftContextRows) +
              " expected_draft_rows=" +
              std::to_string(expectedPartialRows));
        }
        partialHit.coldOutputMatch = partialCold.outputTokens == partialHit.outputTokens;
        measurements.push_back(std::move(partialCold));
        measurements.push_back(std::move(partialSeed));
        measurements.push_back(std::move(partialHit));
      }
    }

    // Cold prefills whose first chunk holds these rows, around where the
    // prefill FFN's Neural Engine split starts and where its program's
    // functions step (ops::AneFfn): a cold prompt prefills up to its replay
    // point, the last 32-token page boundary before its last token, then the
    // rest, so a prompt of rows + 1 tokens runs a chunk of those rows and one
    // of a row. Each prompt is unique and runs on an empty cache.
    if (selected.shortPrompts) {
      constexpr std::array<uint32_t, 8> shortRows{480, 512, 544, 640, 672, 1024, 1536, 2016};
      for (uint32_t sample = 0; sample < samples; ++sample) {
        for (uint32_t rows : shortRows) {
          evictAllCache(resources->cache());
          Measurement result = runRequest(
              engine, driver, *executor, events, progress.get(), requestId++, "short", sample,
              prompt(rows + 1, (uint64_t{rows} << 32 | sample) ^ 0x53484f5254ULL));
          if (result.cacheStatus != "miss")
            throw std::runtime_error("short prompt of a " + std::to_string(rows) +
                                     "-row chunk was not a cold miss: " + result.cacheStatus);
          measurements.push_back(std::move(result));
        }
      }
    }

    // A request lazily materializes a KV junction where its match ends past
    // its state at a branch point: another branch goes on below and holds a
    // state there, and the junction lies a draft window or more past the
    // state the request resumes from. Two prompts share a 7K prefix and then
    // diverge. State eviction deliberately leaves the Page32 graph intact.
    if (selected.context) {
      evictAllCache(resources->cache());
      const std::vector<uint32_t> lazyShared = prompt(7168, 0x4c415a594b56ULL);
      const auto lazyBranch = [&](uint64_t salt) {
        std::vector<uint32_t> branch = lazyShared;
        const std::vector<uint32_t> tail = prompt(2048, salt);
        branch.insert(branch.end(), tail.begin(), tail.end());
        return branch;
      };
      const std::vector<uint32_t> lazyPrompt = lazyBranch(0x4c415a5941ULL);
      Measurement lazySeed =
          runRequest(engine, driver, *executor, events, progress.get(),
                     requestId++, "lazy_seed", 0, lazyPrompt);
      evictAllCompositeState(resources->cache());
      // The repeat's match ends where its own chain does, so it plans no
      // junction: it replays the prompt and rebuilds its replay point, which
      // the next repeat restores.
      Measurement lazyChainEnd =
          runRequest(engine, driver, *executor, events, progress.get(),
                     requestId++, "lazy_chain_end", 0, lazyPrompt);
      Measurement lazyRepeat =
          runRequest(engine, driver, *executor, events, progress.get(),
                     requestId++, "lazy_repeat", 0, lazyPrompt);
      const uint32_t lazyReplayBoundary =
          ((lazyPrompt.size() - 1) / kv::kPageTokens) * kv::kPageTokens;
      if (lazySeed.cacheStatus != "miss" ||
          lazyChainEnd.cacheStatus != "miss" ||
          lazyChainEnd.junctionMaterializations != 0 ||
          lazyRepeat.matchedTokens != lazyReplayBoundary ||
          lazyRepeat.junctionMaterializations != 0) {
        throw std::runtime_error(
            "chain-end KV junction oracle failed: junctions=" +
            std::to_string(lazyChainEnd.junctionMaterializations) + "," +
            std::to_string(lazyRepeat.junctionMaterializations) + " " +
            lookups({lazySeed, lazyChainEnd, lazyRepeat}, lazyReplayBoundary));
      }
      // The other branch's match ends at the shared prefix, above the first
      // branch's replay point: it materializes the junction there, and a
      // third branch off the prefix restores from it.
      Measurement lazyMaterialize =
          runRequest(engine, driver, *executor, events, progress.get(),
                     requestId++, "lazy_materialize", 0,
                     lazyBranch(0x4c415a5942ULL));
      Measurement lazyReuse =
          runRequest(engine, driver, *executor, events, progress.get(),
                     requestId++, "lazy_reuse", 0, lazyBranch(0x4c415a5943ULL));
      const auto lazyBranchPoint = static_cast<uint32_t>(lazyShared.size());
      if (lazyMaterialize.junctionMaterializations != 1 ||
          lazyReuse.matchedTokens != lazyBranchPoint ||
          lazyReuse.junctionMaterializations != 0) {
        throw std::runtime_error(
            "branch KV junction oracle failed: junctions=" +
            std::to_string(lazyMaterialize.junctionMaterializations) + "," +
            std::to_string(lazyReuse.junctionMaterializations) + " " +
            lookups({lazyMaterialize, lazyReuse}, lazyBranchPoint));
      }
      lazyChainEnd.coldOutputMatch = lazySeed.outputTokens == lazyChainEnd.outputTokens;
      lazyRepeat.coldOutputMatch = lazySeed.outputTokens == lazyRepeat.outputTokens;
      measurements.push_back(std::move(lazySeed));
      measurements.push_back(std::move(lazyChainEnd));
      measurements.push_back(std::move(lazyRepeat));
      measurements.push_back(std::move(lazyMaterialize));
      measurements.push_back(std::move(lazyReuse));
    }

    // Shortest form that reads back as the same share.
    std::array<char, 32> share{};
    const auto written = std::to_chars(share.data(), share.data() + share.size(), ranAneFfnShare);
    std::cout << "{\"schema_version\":2,\"build_id\":\"" << SPLASH_BUILD_ID
              << "\",\"identity\":" << identity
              << ",\"ane_ffn_share\":" << std::string_view(share.data(), written.ptr - share.data())
              << ",\"ane_ffn_minimum_rows\":" << ranAneFfnMinimumRows
              << ",\"geometry\":{\"prefill_rows\":"
              << model::ExecutionLimits::prefillTokenBudget
              << ",\"verify_rows\":" << model::ExecutionLimits::targetVerifyRows
              << ",\"batch_width\":" << model::ExecutionLimits::maximumBatchWidth
              << ",\"kv_block_tokens\":"
              << kv::kPageTokens
              << "},\"warmup\":{\"prefill_2048_wall_ms\":"
              << prefillWarmup.wallSeconds * 1000.0
              << ",\"prefill_2048_gpu_ms\":"
              << prefillWarmupGpuMilliseconds
              << ",\"decode_wall_ms\":[";
    for (size_t index = 0; index < decodeWarmupWall.size(); ++index) {
      if (index)
        std::cout << ',';
      std::cout << decodeWarmupWall[index];
    }
    std::cout << "],\"decode_gpu_ms\":[";
    for (size_t index = 0; index < decodeWarmupGpu.size(); ++index) {
      if (index)
        std::cout << ',';
      std::cout << decodeWarmupGpu[index];
    }
    std::cout << "]},\"decode_throughput\":{\"prompt_tokens\":"
              << decodeThroughputPrompt.size()
              << ",\"prompt_sha256\":\"" << decodeThroughputPromptSha256
              << "\",\"output_tokens_per_lane\":64,\"samples\":[";
    for (size_t index = 0; index < decodeThroughput.size(); ++index) {
      if (index)
        std::cout << ',';
      const DecodeThroughputMeasurement &value = decodeThroughput[index];
      std::cout << "{\"width\":" << value.width
                << ",\"sample\":" << value.sample
                << ",\"output_tokens\":" << value.outputTokens
                << ",\"decode_output_tokens\":" << value.decodeOutputTokens
                << ",\"decode_batches\":" << value.decodeBatches
                << ",\"drafted_tokens\":" << value.draftedTokens
                << ",\"accepted_draft_tokens\":"
                << value.acceptedDraftTokens
                << ",\"output_tokens_per_lane_batch\":"
                << value.outputTokensPerLaneBatch
                << ",\"draft_acceptance_rate\":"
                << value.draftAcceptanceRate
                << ",\"output_token_hash\":\""
                << tokenSequenceHash(value.firstLaneOutputTokens) << '"'
                << ",\"request_wall_ms\":"
                << value.requestWallMilliseconds
                << ",\"prefill_wall_ms\":"
                << value.prefillWallMilliseconds
                << ",\"prefill_gpu_ms\":"
                << value.prefillGpuMilliseconds
                << ",\"decode_wall_ms\":" << value.decodeWallMilliseconds
                << ",\"decode_gpu_ms\":"
                << value.decodeGpuMilliseconds
                << ",\"aggregate_request_wall_tokens_per_second\":"
                << value.aggregateRequestWallTokensPerSecond
                << ",\"aggregate_decode_wall_tokens_per_second\":"
                << value.aggregateDecodeWallTokensPerSecond
                << ",\"aggregate_gpu_tokens_per_second\":"
                << value.aggregateGpuTokensPerSecond << '}';
    }
    // The context the engine served, the automatic one whatever split the
    // round ran (engine::AneFfnOutcome::context) unless --max-context gave
    // it, and the one its memory plan holds without the Neural Engine split,
    // and its elastic state/KV budget and the split's.
    const engine::EngineMemoryBreakdown &plan = resources->memoryPlan().breakdown();
    std::cout << "]},\"max_context_tokens\":" << engineConfig.maxContext
              << ",\"no_ane_context_tokens\":" << resources->aneFfnOutcome().contextWithout
              << ",\"dynamic_budget_bytes\":" << plan.dynamicBudgetBytes
              << ",\"ane_ffn_bytes\":" << plan.aneFfnBytes
              << ",\"skipped_context_lengths\":[";
    for (size_t index = 0; index < skippedLengths.size(); ++index)
      std::cout << (index ? "," : "") << skippedLengths[index];
    std::cout << "],\"measurements\":[";
    for (size_t index = 0; index < measurements.size(); ++index)
      emitMeasurement(measurements[index], index == 0);
    const engine::EngineSnapshot snapshot = engine.snapshot();
    std::cout << "],\"performance_pass\":"
              << (performanceFailures.empty() ? "true" : "false")
              << ",\"performance_failures\":[";
    for (size_t index = 0; index < performanceFailures.size(); ++index) {
      if (index)
        std::cout << ',';
      std::cout << '"' << performanceFailures[index] << '"';
    }
    std::cout << "],\"final\":{\"cache_hits\":" << snapshot.cacheHits
              << ",\"cold_misses\":" << snapshot.coldMisses
              << ",\"reused_tokens\":" << snapshot.reusedTokens
              << ",\"kv_pages_cache\":" << snapshot.resources.pool.pagesPrefix
              << ",\"state_entries\":" << snapshot.resources.stateCache.entries
              << "}}\n";
    if (progress)
      progress->finish(performanceFailures.size());
    return performanceFailures.empty() ? 0 : 1;
  } catch (const std::exception &error) {
    if (progress)
      progress->fail();
    std::cerr << "backend-benchmark: " << error.what() << '\n';
    return 1;
  }
}
