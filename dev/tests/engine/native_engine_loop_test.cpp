#include "TestImmediateTicket.hpp"
#include "engine/Cache.hpp"
#include "engine/NativeRuntime.hpp"
#include "metal/CommandWatchdog.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

using namespace splash;
using namespace splash::engine;

namespace {

class Backing final : public KvBacking {
public:
  explicit Backing(uint32_t pages) : resident_(pages, true) {}
  uint32_t pageCount() const noexcept override { return resident_.size(); }
  uint64_t bytesPerPage() const noexcept override { return 4096; }
  bool isResident(uint32_t page) const override { return resident_.at(page); }
  splash::metal::AllocationResult ensureResident(uint32_t page) override {
    resident_.at(page) = true;
    return true;
  }
  bool releaseBackingForPage(uint32_t page) override {
    resident_.at(page) = false;
    return true;
  }
  uint32_t extentFirstPage(uint32_t page) const override {
    return page - page % 4;
  }
  uint32_t extentPageCount(uint32_t page) const override {
    return std::min<uint32_t>(4, resident_.size() - extentFirstPage(page));
  }
private:
  std::vector<bool> resident_;
};

class State final : public CompositeState {
public:
  uint64_t bytes() const noexcept override { return 64; }
};

class HeldTicket final : public ModelBatchTicket {
public:
  HeldTicket(std::vector<ModelStepResult> results, std::shared_ptr<bool> ready)
      : results_(std::move(results)), ready_(std::move(ready)) {}
  bool ready() const noexcept override { return *ready_; }
  std::vector<ModelStepResult> wait() override { return std::move(results_); }
  double wallMilliseconds() const noexcept override { return 0.0; }

private:
  std::vector<ModelStepResult> results_;
  std::shared_ptr<bool> ready_;
};

class Executor final : public model::Model {
public:
  std::shared_ptr<bool> ticketReady;
  std::function<void()> onSubmit;
  std::function<void()> onHealthCheck;
  bool pendingHealth = false;
  // Score requests whose final prompt chunk reports a per-lane model failure.
  std::unordered_set<uint64_t> invalidScores;
  // Prefill chunks each request received, to prove a failure was isolated to
  // the last one rather than to a prefill that never chunked.
  std::unordered_map<uint64_t, uint32_t> prefillChunks;
  uint32_t widestBatch = 0;
  void checkHealth() override {
    if (onHealthCheck)
      onHealthCheck();
  }
  bool needsHealthCheck() const noexcept override { return pendingHealth; }
  StateAdmission begin(const ModelRequest &request) override {
    for (uint32_t slot = 0; slot < model::ExecutionLimits::maximumBatchWidth;
         ++slot) {
      const bool used = std::any_of(
          requests_.begin(), requests_.end(),
          [slot](const auto &entry) { return entry.second.slot == slot; });
      if (!used) {
        requests_.emplace(
            request.id,
            Active{slot, static_cast<uint32_t>(request.prompt.size()),
                   {request.scoreTokens.begin(), request.scoreTokens.end()}});
        return {slot, StateFailure::None};
      }
    }
    return {{}, StateFailure::ConcurrencyLimit};
  }
  void suspend(uint64_t) override {}
  StateAdmission resume(const ModelRequest &) override {
    return {{}, StateFailure::ConcurrencyLimit};
  }
  void restore(uint64_t, uint32_t length,
                     std::shared_ptr<const CompositeState> state,
                     bool) override {
    if (!state)
      throw std::runtime_error("missing composite state");
    restored_ += length;
  }
  void setDraftContextPlan(uint64_t, DraftContextPlan) override {}
  std::vector<ModelStepResult>
  prefill(const BatchPlan &, std::span<const ModelBatchItem> items) {
    std::vector<ModelStepResult> results;
    for (const auto &item : items) {
      ++prefillChunks[item.requestId];
      auto found = requests_.find(item.requestId);
      const bool last =
          found != requests_.end() &&
          item.logicalPosition + item.tokenCount == found->second.promptTokens;
      const bool scoring =
          found != requests_.end() && !found->second.scoreTokens.empty();
      std::vector<float> logits;
      std::string failure;
      if (scoring && last) {
        if (invalidScores.contains(item.requestId)) {
          failure = "score logit is not finite";
        } else {
          logits.reserve(found->second.scoreTokens.size());
          for (size_t index = 0; index < found->second.scoreTokens.size();
               ++index) {
            logits.push_back(static_cast<float>(index) + 0.5f);
          }
        }
      }
      results.push_back({item.requestId, item.tokenCount, {}, scoring && last,
                         DecodeStage::Regular, 0, 0, 0, std::move(logits),
                         std::move(failure)});
    }
    return results;
  }
  std::vector<ModelStepResult>
  decode(const BatchPlan &, std::span<const ModelBatchItem> items) {
    std::vector<ModelStepResult> results;
    for (const auto &item : items) {
      results.push_back({item.requestId,
                         0,
                         std::vector<uint32_t>(stepTokens, 42),
                         true,
                         DecodeStage::Regular,
                         7,
                         7});
    }
    return results;
  }
  // Tokens one decode step emits; the last one is the terminal anchor.
  uint32_t stepTokens = 1;
  std::unique_ptr<ModelBatchTicket>
  submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
              std::function<void()> completion) override {
    widestBatch = std::max(widestBatch, static_cast<uint32_t>(items.size()));
    if (onSubmit)
      onSubmit();
    auto result = plan.kind == WorkKind::Prefill ? prefill(plan, items)
                                               : decode(plan, items);
    if (ticketReady)
      return std::make_unique<HeldTicket>(std::move(result), ticketReady);
    return test::immediateTicket(std::move(result), completion);
  }
  std::shared_ptr<const CompositeState> snapshot(uint64_t) override {
    return std::make_shared<State>();
  }
  uint64_t reclaimIdleState() noexcept override { return 0; }
  void provideMask(uint64_t, std::span<const uint32_t>) override {}
  void end(uint64_t id) override { requests_.erase(id); }
  uint32_t restored() const noexcept { return restored_; }
  bool holdsSlot(uint64_t id) const { return requests_.contains(id); }

private:
  struct Active {
    uint32_t slot = 0;
    uint32_t promptTokens = 0;
    std::vector<uint32_t> scoreTokens;
  };
  std::unordered_map<uint64_t, Active> requests_;
  uint32_t restored_ = 0;
};

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

std::vector<protocol::Message> decodeMessages(std::span<const uint8_t> bytes) {
  protocol::FrameParser parser;
  std::vector<protocol::Message> result;
  size_t offset = 0;
  while (offset < bytes.size()) {
    auto step = parser.consume(bytes.subspan(offset));
    offset += step.consumedBytes;
    if (step.issue)
      throw std::runtime_error(step.issue->describe());
    if (!step.frame)
      continue;
    auto decoded = protocol::decodeFrame(*step.frame);
    if (!decoded)
      throw std::runtime_error(decoded.issue->describe());
    result.push_back(std::move(*decoded.value));
  }
  return result;
}

protocol::RequestFrame request(uint64_t id, uint32_t maxOutputTokens = 1) {
  protocol::RequestFrame result;
  result.requestId = id;
  result.priority = protocol::RequestPriority::Foreground;
  result.cohort = protocol::Cohort::Greedy;
  result.constraint = protocol::ConstraintMode::None;
  result.absoluteDeadlineUnixMicros = 2'000'000;
  result.remainingDeadlineMicros = 1'000'000;
  result.logicalMaxOutputTokens = maxOutputTokens;
  result.promptTokens.resize(65);
  for (uint32_t i = 0; i < result.promptTokens.size(); ++i) {
    result.promptTokens[i] = i + 1;
  }
  return result;
}

void runUntilIdle(engine::NativeRuntime &loop) {
  for (uint32_t step = 0; step < 32 && !loop.idle(); ++step) {
    static_cast<void>(loop.tick());
  }
  require(loop.idle(), "native loop did not become idle");
}

void testPromptProgress() {
  Backing backing(512);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  std::vector<uint8_t> output;
  double monotonic = 100.0;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 8192;
  engine::NativeRuntime loop(
      config, resources, executor,
      [&](std::span<const uint8_t> bytes) {
        output.insert(output.end(), bytes.begin(), bytes.end());
      },
      [] { return std::string("{\"schema_version\":5,\"ready\":true}"); },
      {[] { return uint64_t{1'000'000}; }, [&] { return monotonic += 0.25; }});
  loop.announceReady();
  auto submit = [&](uint64_t id, bool enabled) {
    auto input = request(id);
    input.returnProgress = enabled;
    input.promptTokens.resize(4097);
    for (uint32_t i = 0; i < input.promptTokens.size(); ++i)
      input.promptTokens[i] = i + 1;
    auto encoded = protocol::serializeMessage(protocol::Message{input});
    require(encoded && loop.receive(*encoded.value), "progress request failed");
  };

  executor.ticketReady = std::make_shared<bool>(false);
  submit(1, true);
  require(loop.tick() && loop.commandInFlight(), "prefill was not submitted");
  for (int i = 0; i < 3; ++i)
    static_cast<void>(loop.tick());
  uint32_t count = 0;
  for (const auto &message : decodeMessages(output)) {
    if (const auto *event =
            std::get_if<protocol::PromptProgressEvent>(&message)) {
      require(event->processedTokens == 0,
              "unfinished command advanced progress");
      ++count;
    }
  }
  require(count == 1, "missing initial progress or repeated pending progress");
  *executor.ticketReady = true;
  runUntilIdle(loop);
  submit(2, true);
  runUntilIdle(loop);
  submit(3, false);
  runUntilIdle(loop);

  std::unordered_map<uint64_t, uint32_t> processed;
  std::unordered_map<uint64_t, uint64_t> elapsed;
  std::unordered_map<uint64_t, bool> tokensSeen;
  uint32_t coldUpdates = 0;
  for (const auto &message : decodeMessages(output)) {
    if (const auto *event =
            std::get_if<protocol::PromptProgressEvent>(&message)) {
      require(event->requestId != 3, "default path emitted progress");
      require(!tokensSeen[event->requestId],
              "progress arrived after generation");
      const bool first = !processed.contains(event->requestId);
      require(first || event->processedTokens > processed[event->requestId],
              "progress repeated or moved backwards");
      require(event->processedTokens <= 4097 &&
                  event->elapsedMicros >= elapsed[event->requestId],
              "progress overflow or elapsed time regression");
      if (first)
        require(event->processedTokens == (event->requestId == 1 ? 0 : 4096),
                "initial progress did not include the cache hit");
      processed[event->requestId] = event->processedTokens;
      elapsed[event->requestId] = event->elapsedMicros;
      coldUpdates += event->requestId == 1;
    } else if (const auto *event =
                   std::get_if<protocol::TokensEvent>(&message)) {
      if (event->requestId != 3)
        require(processed[event->requestId] == 4097,
                "generation started before prompt progress completed");
      tokensSeen[event->requestId] = true;
    }
  }
  require(coldUpdates >= 3 && tokensSeen[1] && tokensSeen[2] && tokensSeen[3],
          "missing incremental progress or terminal output");

  output.clear();
  *executor.ticketReady = false;
  submit(4, true);
  require(loop.tick() && loop.commandInFlight(),
          "cancel test needs pending work");
  auto cancel =
      protocol::serializeMessage(protocol::Message{protocol::CancelFrame{4}});
  require(cancel && loop.receive(*cancel.value), "cancel request failed");
  *executor.ticketReady = true;
  runUntilIdle(loop);
  count = 0;
  bool cancelled = false;
  for (const auto &message : decodeMessages(output)) {
    if (std::holds_alternative<protocol::PromptProgressEvent>(message))
      ++count;
    if (const auto *event = std::get_if<protocol::DoneEvent>(&message))
      cancelled = event->reason == protocol::FinishReason::Cancelled;
  }
  require(count == 1 && cancelled,
          "cancelled prefill published further progress");
}

void testWireLifecycleAndCacheHit() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  std::vector<uint8_t> output;
  double monotonic = 100.0;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  engine::NativeRuntime loop(
      config, resources, executor,
      [&](std::span<const uint8_t> bytes) {
        output.insert(output.end(), bytes.begin(), bytes.end());
      },
      [] { return std::string("{\"schema_version\":5,\"ready\":true}"); },
      {[] { return uint64_t{1'000'000}; }, [&] { return monotonic += 0.25; }});

  loop.announceReady();
  auto first = protocol::serializeMessage(protocol::Message{request(1)});
  require(first && loop.receive(*first.value), "cold request wire failed");
  require(loop.tick() && loop.commandInFlight(),
          "status regression requires a pending command");
  auto status = protocol::serializeMessage(
      protocol::Message{protocol::StatusRequestFrame{77}});
  require(status && loop.receive(*status.value), "in-flight status request failed");
  const auto pendingMessages = decodeMessages(output);
  require(loop.commandInFlight() &&
              std::any_of(pendingMessages.begin(), pendingMessages.end(),
                          [](const auto &message) {
                            const auto *event =
                                std::get_if<protocol::StatusJsonEvent>(&message);
                            return event && event->correlationId == 77;
                          }),
          "status waited for or drained the pending command");
  runUntilIdle(loop);
  auto second = protocol::serializeMessage(protocol::Message{request(2)});
  require(second && loop.receive(*second.value), "junction request wire failed");
  runUntilIdle(loop);
  auto third = protocol::serializeMessage(protocol::Message{request(3)});
  require(third && loop.receive(*third.value), "restored request wire failed");
  runUntilIdle(loop);
  auto messages = decodeMessages(output);
  uint32_t misses = 0;
  uint32_t hits = 0;
  uint32_t tokens = 0;
  uint32_t done = 0;
  bool statusSeen = false;
  for (const auto &message : messages) {
    if (const auto *start = std::get_if<protocol::StartEvent>(&message)) {
      if (start->cacheDisposition == protocol::CacheDisposition::Miss) {
        ++misses;
      } else {
        ++hits;
        require(start->matchedPromptTokens == 64,
                "prefix hit did not replay one token");
      }
    } else if (const auto *emitted =
                   std::get_if<protocol::TokensEvent>(&message)) {
      tokens += emitted->tokens.size();
    } else if (std::holds_alternative<protocol::DoneEvent>(message)) {
      ++done;
    } else if (const auto *reported =
                   std::get_if<protocol::StatusJsonEvent>(&message)) {
      statusSeen = reported->correlationId == 77 &&
                   reported->schemaVersion == protocol::kStatusSchemaVersion;
    }
  }
  require(misses == 1 && hits == 2 && executor.restored() == 128,
          "cold/latest-replay/hit classification is wrong");
  require(tokens == 3 && done == 3 && statusSeen,
          "native lifecycle events are incomplete");
}

void testFatalFramingClosesConnection() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  std::vector<uint8_t> output;
  engine::NativeRuntime loop(
      {}, resources, executor,
      [&](std::span<const uint8_t> bytes) {
        output.insert(output.end(), bytes.begin(), bytes.end());
      },
      [] { return std::string("{}"); });
  const std::array<uint8_t, 24> invalid{};
  require(!loop.receive(invalid), "bad frame did not close connection");
  require(loop.connectionMustClose() && loop.engineHealthy(),
          "protocol failure was misclassified as engine failure");
}

// A request rejected while it is decoded is a request-scoped error: the
// frames behind it in the same read, whole or cut by the read boundary, must
// still be processed.
void testRequestErrorKeepsFraming() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  std::vector<uint8_t> output;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  protocol::ProtocolLimits limits;
  limits.maxLogicalOutputTokens = 1;
  engine::NativeRuntime loop(
      config, resources, executor,
      [&](std::span<const uint8_t> bytes) {
        output.insert(output.end(), bytes.begin(), bytes.end());
      },
      [] { return std::string("{\"schema_version\":5,\"ready\":true}"); },
      {[] { return uint64_t{1'000'000}; }, [] { return 100.0; }}, limits);

  loop.announceReady();
  const auto wire = [](protocol::Message message) {
    auto encoded = protocol::serializeMessage(message);
    require(static_cast<bool>(encoded), "test message wire encoding failed");
    return *encoded.value;
  };
  // Errors, completions and status answers so far; every error must be the
  // rejected request's own.
  const auto events = [&] {
    std::array<uint32_t, 3> counts{};
    for (const protocol::Message &message : decodeMessages(output)) {
      if (const auto *error = std::get_if<protocol::ErrorEvent>(&message)) {
        require(error->failureClass == protocol::FailureClass::RequestError &&
                    error->requestId == 9,
                "rejected request was not reported as its own error");
      }
      counts[0] += std::holds_alternative<protocol::ErrorEvent>(message);
      counts[1] += std::holds_alternative<protocol::DoneEvent>(message);
      counts[2] += std::holds_alternative<protocol::StatusJsonEvent>(message);
    }
    return counts;
  };

  const std::vector<uint8_t> rejected = wire(request(9, 2));
  const std::vector<uint8_t> first = wire(request(1));
  const std::vector<uint8_t> status = wire(protocol::StatusRequestFrame{77});
  std::vector<uint8_t> read = rejected;
  read.insert(read.end(), first.begin(), first.end());
  read.insert(read.end(), status.begin(), status.end());
  require(loop.receive(read), "request-scoped error closed the connection");
  runUntilIdle(loop);
  require(events() == std::array<uint32_t, 3>{1, 1, 1},
          "frames behind a rejected request were dropped");

  const std::vector<uint8_t> second = wire(request(2));
  const size_t half = second.size() / 2;
  read = rejected;
  read.insert(read.end(), second.begin(), second.begin() + half);
  require(loop.receive(read) &&
              loop.receive(std::span<const uint8_t>(second).subspan(half)),
          "cut frame behind a rejected request closed the connection");
  runUntilIdle(loop);
  require(events() == std::array<uint32_t, 3>{2, 2, 1} && loop.engineHealthy(),
          "cut frame behind a rejected request was not reassembled");
}

void testCapacityFailureHasOneTerminalFrame() {
  Backing backing(1);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  std::vector<uint8_t> output;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  engine::NativeRuntime loop(
      config, resources, executor,
      [&](std::span<const uint8_t> bytes) {
        output.insert(output.end(), bytes.begin(), bytes.end());
      },
      [] { return std::string("{\"schema_version\":5,\"ready\":true}"); },
      {[] { return uint64_t{1'000'000}; }, [] { return 100.0; }});

  loop.announceReady();
  auto encoded = protocol::serializeMessage(protocol::Message{request(3)});
  require(encoded && loop.receive(*encoded.value),
          "capacity request wire failed");
  runUntilIdle(loop);

  uint32_t capacity = 0;
  uint32_t errors = 0;
  uint32_t done = 0;
  for (const protocol::Message &message : decodeMessages(output)) {
    capacity += std::holds_alternative<protocol::CapacityExhaustedEvent>(message);
    errors += std::holds_alternative<protocol::ErrorEvent>(message);
    done += std::holds_alternative<protocol::DoneEvent>(message);
  }
  require(capacity == 1 && errors == 0 && done == 0,
          "capacity failure emitted more than one terminal frame");
  require(loop.engineHealthy(),
          "request-scoped capacity failure made the engine unhealthy");
}

// A verify step can retain every row and add the terminal anchor; the wire
// limit the production binary derives from ExecutionLimits must carry that
// step in one TokensEvent, and an event that cannot be encoded must surface
// as an engine error rather than a silently shorter stream.
void testCommandWatchdogAndPendingHealthWake() {
  metal::CommandWatchdog generations;
  generations.start(1, 0.0);
  generations.complete(1);
  require(!generations.expired(1000.0), "completed command retained a deadline");
  generations.start(2, 10.0);
  generations.complete(1);
  require(!generations.expired(129.999) && generations.expired(130.0),
          "stale completion cleared a newer command deadline");

  for (bool gpuCompleted : {false, true}) {
    Backing backing(32);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor;
    executor.ticketReady = std::make_shared<bool>(false);
    double now = 0.0;
    metal::CommandWatchdog watchdog;
    executor.onSubmit = [&] { watchdog.start(1, now / 1000.0); };
    executor.onHealthCheck = [&] {
      if (watchdog.expired(now / 1000.0))
        throw metal::MetalBackendError("test command completion timeout");
    };
    std::vector<uint8_t> output;
    engine::NativeRuntime loop(
        {}, resources, executor,
        [&](std::span<const uint8_t> bytes) {
          output.insert(output.end(), bytes.begin(), bytes.end());
        },
        [] { return std::string("{\"schema_version\":5,\"ready\":true}"); },
        {[] { return uint64_t{1'000'000}; }, [&] { return now; }});
    loop.announceReady();
    auto input = request(1);
    input.absoluteDeadlineUnixMicros = 601'000'000;
    input.remainingDeadlineMicros = 600'000'000;
    const auto wire = protocol::serializeMessage(protocol::Message{input});
    require(wire && loop.receive(*wire.value) && loop.tick(),
            "watchdog fixture did not submit its command");
    const auto cancel = protocol::serializeMessage(
        protocol::Message{protocol::CancelFrame{1}});
    require(cancel && loop.receive(*cancel.value) &&
                loop.millisecondsUntilNextWakeup() == 1000.0,
            "cancelled in-flight command lost its bounded health wake");
    now = 119'999.0;
    require(!loop.tick() && loop.engineHealthy() && loop.commandInFlight(),
            "watchdog failed a command before its safety deadline");
    if (gpuCompleted)
      watchdog.complete(1);
    now = 120'000.0;
    require(!loop.tick(), "held command unexpectedly completed");
    if (gpuCompleted) {
      require(loop.engineHealthy() && loop.commandInFlight(),
              "a model-side wait was mistaken for a pending GPU command");
      *executor.ticketReady = true;
      require(loop.tick() && loop.idle(), "completed GPU ownership did not drain");
    } else {
      uint32_t errors = 0;
      for (const auto &message : decodeMessages(output)) {
        if (const auto *error = std::get_if<protocol::ErrorEvent>(&message)) {
          require(error->requestId == 0 &&
                      error->failureClass == protocol::FailureClass::EngineUnhealthy &&
                      error->code == "metal_execution_failed",
                  "watchdog timeout lost structured engine failure");
          ++errors;
        }
      }
      require(errors == 1 && loop.connectionMustClose() &&
                  loop.commandInFlight() && !loop.idle(),
              "watchdog released command ownership or emitted duplicate failures");
    }
  }

  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  executor.pendingHealth = true; // A sparse unmap can outlive all requests.
  engine::NativeRuntime loop({}, resources, executor,
      [](std::span<const uint8_t>) {}, [] { return std::string("{}"); },
      {[] { return uint64_t{1'000'000}; }, [] { return 0.0; }});
  require(!loop.tick() && loop.idle() &&
              loop.millisecondsUntilNextWakeup() == 1000.0,
          "an idle outstanding backend operation lost its health wake");
  executor.pendingHealth = false;
  require(!loop.millisecondsUntilNextWakeup(),
          "fully idle engine retained a polling wake");
}

void testDuplicateLiveRequestClosesWithoutAmbiguousError() {
  for (bool malformed : {false, true}) {
    Backing backing(32);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor;
    std::vector<uint8_t> output;
    protocol::ProtocolLimits limits;
    limits.maxLogicalOutputTokens = 1;
    engine::NativeRuntime loop(
        {}, resources, executor,
        [&](std::span<const uint8_t> bytes) {
          output.insert(output.end(), bytes.begin(), bytes.end());
        },
        [] { return std::string("{\"schema_version\":5,\"ready\":true}"); },
        {[] { return uint64_t{1'000'000}; }, [] { return 100.0; }}, limits);
    loop.announceReady();
    const auto first = protocol::serializeMessage(protocol::Message{request(1)});
    require(first && loop.receive(*first.value) && loop.tick(),
            "live duplicate fixture did not start");
    const auto duplicate = protocol::serializeMessage(
        protocol::Message{request(1, malformed ? 2 : 1)});
    require(duplicate && !loop.receive(*duplicate.value) &&
                loop.connectionMustClose() && loop.snapshot().submitted == 1,
            "duplicate live request was accepted");
    uint32_t errors = 0;
    for (const auto &message : decodeMessages(output)) {
      if (const auto *error = std::get_if<protocol::ErrorEvent>(&message)) {
        require(error->requestId == 0 &&
                    error->failureClass == protocol::FailureClass::ProtocolFatal,
                "duplicate id produced an ambiguous error for the active request");
        ++errors;
      }
    }
    require(errors == 1, "duplicate id produced multiple terminal errors");
  }
}

void testControlFailureUsesExecutionBoundary() {
  for (bool metalFailure : {false, true}) {
    Backing backing(32);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor;
    std::vector<uint8_t> output;
    engine::NativeRuntime loop(
        {}, resources, executor,
        [&](std::span<const uint8_t> bytes) {
          output.insert(output.end(), bytes.begin(), bytes.end());
        },
        [] { return std::string("{\"schema_version\":5,\"ready\":true}"); });
    loop.announceReady();
    require(loop.runControl([] { return true; }) && loop.engineHealthy(),
            "ordinary deferred control work failed");
    require(!loop.runControl([&]() -> bool {
              if (metalFailure)
                throw metal::MetalBackendError("control test");
              throw std::runtime_error("control test");
            }), "failed control work requested another retry");
    uint32_t errors = 0;
    for (const auto &message : decodeMessages(output)) {
      if (const auto *error = std::get_if<protocol::ErrorEvent>(&message)) {
        require(error->requestId == 0 &&
                    error->failureClass == protocol::FailureClass::EngineUnhealthy &&
                    error->code == (metalFailure ? "metal_execution_failed"
                                                : "engine_execution_failed"),
                "control exception lost structured failure classification");
        ++errors;
      }
    }
    require(errors == 1 && !loop.engineHealthy() && loop.connectionMustClose(),
            "control exception did not terminate the unhealthy engine");
  }
}

void testInvalidPromptTokensStayRequestScoped() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  std::vector<uint8_t> output;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.engine.vocabularySize = 128;
  engine::NativeRuntime loop(
      config, resources, executor,
      [&](std::span<const uint8_t> bytes) {
        output.insert(output.end(), bytes.begin(), bytes.end());
      },
      [] { return std::string("{\"schema_version\":5,\"ready\":true}"); },
      {[] { return uint64_t{1'000'000}; }, [] { return 100.0; }});
  loop.announceReady();
  for (uint32_t token : {128U, std::numeric_limits<uint32_t>::max()}) {
    auto invalid = request(9);
    invalid.promptTokens.back() = token;
    const auto wire = protocol::serializeMessage(protocol::Message{invalid});
    require(wire && loop.receive(*wire.value),
            "invalid prompt token closed the native connection");
  }
  auto valid = request(1);
  valid.promptTokens.back() = 127;
  const auto wire = protocol::serializeMessage(protocol::Message{valid});
  require(wire && loop.receive(*wire.value),
          "valid request after invalid tokens was rejected");
  runUntilIdle(loop);
  uint32_t errors = 0, done = 0;
  for (const auto &message : decodeMessages(output)) {
    if (const auto *error = std::get_if<protocol::ErrorEvent>(&message)) {
      require(error->requestId == 9 && error->code == "invalid_request" &&
                  error->failureClass == protocol::FailureClass::RequestError,
              "invalid prompt token did not produce its own request error");
      ++errors;
    }
    done += std::holds_alternative<protocol::DoneEvent>(message);
  }
  require(errors == 2 && done == 1 && loop.engineHealthy() &&
              loop.snapshot().submitted == 1,
          "invalid tokens reached admission or prevented subsequent completion");
}

void testStepTokensFitTheWire() {
  for (uint32_t limit : {model::ExecutionLimits::maximumStepTokens, 1U}) {
    Backing backing(32);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor;
    executor.stepTokens = model::ExecutionLimits::maximumStepTokens;
    std::vector<uint8_t> output;
    engine::NativeLoopConfig config;
    config.engine.maxContext = 1024;
    protocol::ProtocolLimits limits;
    limits.maxTokenBatch = limit;
    engine::NativeRuntime loop(
        config, resources, executor,
        [&](std::span<const uint8_t> bytes) {
          output.insert(output.end(), bytes.begin(), bytes.end());
        },
        [] { return std::string("{\"schema_version\":5,\"ready\":true}"); },
        {[] { return uint64_t{1'000'000}; }, [] { return 100.0; }}, limits);

    loop.announceReady();
    auto encoded = protocol::serializeMessage(
        protocol::Message{request(5, executor.stepTokens)});
    require(encoded && loop.receive(*encoded.value), "step request wire failed");
    runUntilIdle(loop);

    uint32_t streamed = 0;
    std::optional<uint32_t> completion;
    uint32_t encodeErrors = 0;
    for (const protocol::Message &message : decodeMessages(output)) {
      if (const auto *emitted = std::get_if<protocol::TokensEvent>(&message)) {
        streamed += emitted->tokens.size();
      } else if (const auto *done = std::get_if<protocol::DoneEvent>(&message)) {
        completion = done->completionTokens;
      } else if (const auto *error = std::get_if<protocol::ErrorEvent>(&message)) {
        encodeErrors += error->code == "protocol_encode_failed" &&
                        error->failureClass ==
                            protocol::FailureClass::EngineUnhealthy;
      }
    }
    if (limit >= executor.stepTokens) {
      require(streamed == executor.stepTokens && completion == streamed &&
                  !encodeErrors && loop.engineHealthy(),
              "a full step with its terminal anchor did not fit one event");
    } else {
      require(!streamed && !completion && encodeErrors == 1 &&
                  !loop.engineHealthy() && loop.connectionMustClose(),
              "an unencodable event was not reported as an engine error");
    }
  }
}

protocol::RequestFrame scoreRequest(uint64_t id, uint32_t promptTokens) {
  protocol::RequestFrame result = request(id, 0);
  result.promptTokens.resize(promptTokens);
  for (uint32_t i = 0; i < result.promptTokens.size(); ++i)
    result.promptTokens[i] = i + 1;
  result.scoreTokens = {10, 20, 30};
  return result;
}

void testScoreRequestCompletesAfterFullPrompt() {
  Backing backing(512);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  std::vector<uint8_t> output;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 8192;
  engine::NativeRuntime loop(
      config, resources, executor,
      [&](std::span<const uint8_t> bytes) {
        output.insert(output.end(), bytes.begin(), bytes.end());
      },
      [] { return std::string("{\"schema_version\":5,\"ready\":true}"); },
      {[] { return uint64_t{1'000'000}; }, [] { return 100.0; }});
  loop.announceReady();
  auto encoded = protocol::serializeMessage(
      protocol::Message{scoreRequest(9, 3000)});
  require(encoded && loop.receive(*encoded.value), "score request failed");
  runUntilIdle(loop);

  uint32_t tokensEvents = 0;
  uint32_t doneCount = 0;
  for (const protocol::Message &message : decodeMessages(output)) {
    if (std::holds_alternative<protocol::TokensEvent>(message))
      ++tokensEvents;
    if (const auto *done = std::get_if<protocol::DoneEvent>(&message)) {
      ++doneCount;
      require(done->requestId == 9, "score done id mismatch");
      require(done->completionTokens == 0, "score done emitted completion");
      require(done->decodeMicros == 0, "score done reported decode time");
      require(done->promptTokens == 3000, "score done prompt count mismatch");
      require(done->optionLogits.size() == 3, "score done logit count mismatch");
      require(done->optionLogits[0] == 0.5f && done->optionLogits[1] == 1.5f &&
                  done->optionLogits[2] == 2.5f,
              "score done logits are not in request order");
      require(done->reason == protocol::FinishReason::Stop,
              "score done reason is not stop");
    }
  }
  require(doneCount == 1 && tokensEvents == 0,
          "score request did not complete without generating tokens");
}

void testCancelledScoreReturnsEmptyLogits() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  executor.ticketReady = std::make_shared<bool>(false);
  std::vector<uint8_t> output;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  engine::NativeRuntime loop(
      config, resources, executor,
      [&](std::span<const uint8_t> bytes) {
        output.insert(output.end(), bytes.begin(), bytes.end());
      },
      [] { return std::string("{\"schema_version\":5,\"ready\":true}"); },
      {[] { return uint64_t{1'000'000}; }, [] { return 100.0; }});
  loop.announceReady();
  auto encoded = protocol::serializeMessage(
      protocol::Message{scoreRequest(11, 65)});
  require(encoded && loop.receive(*encoded.value), "cancel-score request failed");
  require(loop.tick() && loop.commandInFlight(), "score prefill was not held");
  protocol::CancelFrame cancel{11};
  auto cancelWire = protocol::serializeMessage(protocol::Message{cancel});
  require(cancelWire && loop.receive(*cancelWire.value), "score cancel failed");
  *executor.ticketReady = true;
  runUntilIdle(loop);

  uint32_t doneCount = 0;
  for (const protocol::Message &message : decodeMessages(output)) {
    if (const auto *done = std::get_if<protocol::DoneEvent>(&message)) {
      ++doneCount;
      require(done->reason == protocol::FinishReason::Cancelled,
              "cancelled score did not report Cancelled");
      require(done->optionLogits.empty(),
              "cancelled score returned logits");
      require(done->completionTokens == 0, "cancelled score emitted tokens");
    }
  }
  require(doneCount == 1, "cancelled score did not emit Done");
}

struct ScoreBesideChat final {
  uint32_t publishedBlocks = 0;
  uint32_t failures = 0;
  uint32_t completions = 0;
  uint32_t tokens = 0;
  uint32_t scoreChunks = 0;
  uint32_t widestBatch = 0;
  uint64_t failedRequest = 0;
  std::string failureCode;
  bool requestScoped = false;
  bool scoreCompleted = false;
  bool scoreEmittedTokens = false;
  bool healthy = false;
  bool slotsReleased = false;
};

// Runs one score request beside one ordinary chat request in a single batch,
// then a third request afterwards. The score's final prompt chunk either
// returns logits or reports a non-finite one.
ScoreBesideChat runScoreBesideChat(bool invalidScore) {
  Backing backing(512);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  if (invalidScore)
    executor.invalidScores.insert(21);
  std::vector<uint8_t> output;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 8192;
  engine::NativeRuntime loop(
      config, resources, executor,
      [&](std::span<const uint8_t> bytes) {
        output.insert(output.end(), bytes.begin(), bytes.end());
      },
      [] { return std::string("{\"schema_version\":5,\"ready\":true}"); },
      {[] { return uint64_t{1'000'000}; }, [] { return 100.0; }});
  loop.announceReady();

  // Whole KV pages, so the last prompt chunk is the one that publishes the
  // final block. The chat lane deliberately does not share this prefix.
  auto scored = scoreRequest(21, 4 * KvCache::pageTokens);
  for (uint32_t index = 0; index < scored.promptTokens.size(); ++index)
    scored.promptTokens[index] = 1000 + index;
  auto scoredWire = protocol::serializeMessage(protocol::Message{scored});
  require(scoredWire && loop.receive(*scoredWire.value),
          "score request wire failed");
  auto chatWire = protocol::serializeMessage(protocol::Message{request(22)});
  require(chatWire && loop.receive(*chatWire.value),
          "batched chat request wire failed");
  runUntilIdle(loop);

  ScoreBesideChat result;
  result.publishedBlocks = resources.snapshot().kvCache.blocks;
  result.healthy = loop.engineHealthy() && !loop.connectionMustClose();
  result.slotsReleased = !executor.holdsSlot(21) && !executor.holdsSlot(22);
  result.scoreChunks = executor.prefillChunks[21];
  result.widestBatch = executor.widestBatch;

  auto laterWire = protocol::serializeMessage(protocol::Message{request(23)});
  require(laterWire && loop.receive(*laterWire.value),
          "post-batch request wire failed");
  runUntilIdle(loop);

  for (const protocol::Message &message : decodeMessages(output)) {
    if (const auto *error = std::get_if<protocol::ErrorEvent>(&message)) {
      ++result.failures;
      result.failedRequest = error->requestId;
      result.failureCode = error->code;
      result.requestScoped =
          error->failureClass == protocol::FailureClass::RequestError;
    } else if (const auto *emitted =
                   std::get_if<protocol::TokensEvent>(&message)) {
      result.scoreEmittedTokens |= emitted->requestId == 21;
      result.tokens += static_cast<uint32_t>(emitted->tokens.size());
    } else if (const auto *done = std::get_if<protocol::DoneEvent>(&message)) {
      ++result.completions;
      result.scoreCompleted |= done->requestId == 21;
    }
  }
  return result;
}

void testInvalidScoreFailsOneRequestAndKeepsTheBatch() {
  const ScoreBesideChat healthy = runScoreBesideChat(false);
  const ScoreBesideChat failed = runScoreBesideChat(true);

  require(healthy.widestBatch == 2 && failed.widestBatch == 2,
          "the score and the chat never shared one batch");
  require(healthy.scoreChunks >= 2 && failed.scoreChunks >= 2,
          "the score prefill never chunked, so nothing isolates its last step");
  require(healthy.failures == 0 && healthy.completions == 3 &&
              !healthy.scoreEmittedTokens,
          "the baseline score batch did not complete cleanly");

  require(failed.healthy, "a non-finite score logit took the whole engine down");
  require(failed.failures == 1 && failed.requestScoped &&
              failed.failedRequest == 21 &&
              failed.failureCode == "model_result_invalid",
          "the non-finite score logit was not a per-request failure");
  require(!failed.scoreCompleted && !failed.scoreEmittedTokens,
          "the failed score still produced a result");
  require(failed.completions == 2 && failed.tokens == healthy.tokens,
          "the failed score took its batch partner or the next request along");
  require(failed.slotsReleased, "the failed score kept its model slot");
  // Chunks before the failing one keep their blocks, exactly as they do for a
  // cancelled request. Only the failed step must publish nothing.
  require(failed.publishedBlocks + 1 == healthy.publishedBlocks,
          "the failed step published its KV block");
}

} // namespace

int main() {
  try {
    testWireLifecycleAndCacheHit();
    testPromptProgress();
    testCapacityFailureHasOneTerminalFrame();
    testFatalFramingClosesConnection();
    testRequestErrorKeepsFraming();
    testCommandWatchdogAndPendingHealthWake();
    testDuplicateLiveRequestClosesWithoutAmbiguousError();
    testControlFailureUsesExecutionBoundary();
    testInvalidPromptTokensStayRequestScoped();
    testStepTokensFitTheWire();
    testScoreRequestCompletesAfterFullPrompt();
    testCancelledScoreReturnsEmptyLogits();
    testInvalidScoreFailsOneRequestAndKeepsTheBatch();
    std::cout << "native KV-first loop tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "native KV-first loop tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
