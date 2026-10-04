#include "AllocationFailure.hpp"
#include "ProtocolPeer.hpp"
#include "ScopedTestConfig.hpp"
#include "TestChecks.hpp"
#include "TestImmediateTicket.hpp"
#include "TestKvPool.hpp"
#include "TestMetalMemory.hpp"
#include "TestStatus.hpp"
#include "engine/Cache.hpp"
#include "engine/MemoryControl.hpp"
#include "engine/NativeRuntime.hpp"
#include "metal/CommandWatchdog.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace splash;
using namespace splash::engine;

namespace {

class State final : public CompositeState {
public:
  uint64_t bytes() const noexcept override { return 64; }
};

class Executor final : public model::Model {
public:
  std::shared_ptr<std::atomic<bool>> ticketReady;
  std::function<void()> onBegin;
  std::function<void()> onSubmit;
  std::function<void()> onHealthCheck;
  // Score requests whose final prompt chunk reports a per-lane model failure.
  std::unordered_set<uint64_t> invalidScores;
  // The request flags and sampling each request began with.
  std::unordered_map<uint64_t, uint32_t> beganFlags;
  std::unordered_map<uint64_t, SamplingParameters> beganSampling;
  // Prefill chunks each request received, to prove a failure was isolated to
  // the last one rather than to a prefill that never chunked.
  std::unordered_map<uint64_t, uint32_t> prefillChunks;
  uint32_t widestBatch = 0;
  void checkHealth() override {
    if (onHealthCheck)
      onHealthCheck();
  }
  StateAdmission begin(const ModelRequest &request) override {
    if (onBegin)
      onBegin();
    beganFlags[request.id] = request.flags;
    beganSampling[request.id] = request.sampling;
    for (uint32_t slot = 0; slot < model::ExecutionLimits::maximumBatchWidth;
         ++slot) {
      const bool used = std::any_of(
          requests_.begin(), requests_.end(),
          [slot](const auto &entry) { return entry.second.slot == slot; });
      if (!used) {
        requests_.emplace(
            request.id,
            Active{slot, static_cast<uint32_t>(request.prompt.size()),
                   {request.scoreTokens.begin(), request.scoreTokens.end()},
                   request.constraint == ConstraintMode::TokenMask});
        return {slot, StateFailure::None};
      }
    }
    return {{}, StateFailure::ConcurrencyLimit};
  }
  void suspend(uint64_t) override {}
  StateAdmission resume(const ModelRequest &) override {
    return {{}, StateFailure::ConcurrencyLimit};
  }
  std::unique_ptr<StateRestore> beginRestore(uint64_t, uint32_t length,
                                             std::shared_ptr<const CompositeState> state,
                                             bool, std::function<void()>) override {
    if (!state)
      throw std::runtime_error("missing composite state");
    restored_ += length;
    return {};
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
      // A constrained prompt's end asks for the first token's mask.
      const bool awaitsMask =
          found != requests_.end() && found->second.constrained && last;
      results.push_back({item.requestId, item.tokenCount, {}, scoring && last,
                         awaitsMask ? DecodeStage::ApplyInitialMask
                                    : DecodeStage::Regular,
                         0, 0, 0, std::move(logits), std::move(failure)});
    }
    return results;
  }
  std::vector<ModelStepResult>
  decode(const BatchPlan &plan, std::span<const ModelBatchItem> items) {
    std::vector<ModelStepResult> results;
    for (const auto &item : items) {
      // The first token's selection under the mask emits nothing here.
      if (plan.decodeStage == DecodeStage::ApplyInitialMask) {
        results.push_back(
            {item.requestId, 0, {}, false, DecodeStage::Regular, 0, 0});
        continue;
      }
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
      return std::make_unique<test::HeldTicket>(std::move(result), ticketReady,
                                                0.0);
    return test::immediateTicket(std::move(result), completion);
  }
  uint64_t snapshotBytes() const noexcept override { return 64; }
  std::shared_ptr<const CompositeState> snapshot(uint64_t) override {
    return std::make_shared<State>();
  }
  uint64_t reclaimIdleState(bool, model::IdleMemory) noexcept override { return 0; }
  std::optional<std::string>
  provideMask(uint64_t, std::span<const uint32_t> words) override {
    // As in the model, a mask row must permit some token.
    if (std::none_of(words.begin(), words.end(),
                     [](uint32_t word) { return word != 0; }))
      return "token mask row permits no vocabulary token";
    ++providedMasks;
    return std::nullopt;
  }
  uint32_t providedMasks = 0;
  void end(uint64_t id) override { requests_.erase(id); }
  uint32_t restored() const noexcept { return restored_; }
  bool holdsSlot(uint64_t id) const { return requests_.contains(id); }

private:
  struct Active {
    uint32_t slot = 0;
    uint32_t promptTokens = 0;
    std::vector<uint32_t> scoreTokens;
    bool constrained = false;
  };
  std::unordered_map<uint64_t, Active> requests_;
  uint32_t restored_ = 0;
};

using splash::test::require;

// Every submitted request has ended and no command is in flight.
bool idle(const engine::NativeRuntime &loop) {
  const EngineSnapshot counts = loop.snapshot();
  return counts.submitted == counts.completed + counts.cancelled + counts.failed &&
         !loop.commandInFlight();
}

protocol::RequestFrame request(uint64_t id, uint32_t maxOutputTokens = 1) {
  protocol::RequestFrame result;
  result.requestId = id;
  result.priority = RequestPriority::Foreground;
  result.constraint = ConstraintMode::None;
  result.absoluteDeadlineUnixMicros = 2'000'000;
  result.remainingDeadlineMicros = 1'000'000;
  result.logicalMaxOutputTokens = maxOutputTokens;
  result.promptTokens.resize(65);
  for (uint32_t i = 0; i < result.promptTokens.size(); ++i) {
    result.promptTokens[i] = i + 1;
  }
  return result;
}

protocol::RequestFrame scoreRequest(uint64_t id, uint32_t promptTokens) {
  protocol::RequestFrame result = request(id, 0);
  result.promptTokens.resize(promptTokens);
  for (uint32_t i = 0; i < result.promptTokens.size(); ++i)
    result.promptTokens[i] = i + 1;
  result.scoreTokens = {10, 20, 30};
  return result;
}

void runUntilIdle(engine::NativeRuntime &loop) {
  for (uint32_t step = 0; step < 32 && !idle(loop); ++step) {
    static_cast<void>(loop.tick());
  }
  require(idle(loop), "native loop did not become idle");
}

// The KV pool under the loop: `pages` pages in extents of `extentPages`, of
// which the first `runwayPages` are allocated up front.
struct PoolShape {
  uint32_t pages = 32;
  uint32_t extentPages = 4;
  uint32_t runwayPages = pages;
};

// How long the loop keeps weights without a request (NativeLoopConfig).
constexpr double kWeightKeepAliveSeconds = 2.0;

// A native loop over the fake model, collecting the bytes it writes. Its
// clocks come from the test seam: the unix clock stands still and the steady
// clock reads `monotonic`, advanced by `clockStep` on every read.
struct LoopFixture {
  explicit LoopFixture(NativeLoopConfig config = {}, PoolShape shape = {},
                       protocol::ProtocolLimits limits = {})
      : seam({.residencyKeepAliveSeconds = kWeightKeepAliveSeconds,
              .unixMicros = [] { return uint64_t{1'000'000}; },
              .monotonicMilliseconds = [this] { return monotonic += clockStep; }}),
        storage(shape.pages, 4096, shape.extentPages),
        pool(storage, shape.runwayPages), cache(pool, nullptr, nullptr),
        loop(
            std::move(config), cache, executor,
            [this](std::span<const uint8_t> bytes) {
              if (writeFailure)
                throw *writeFailure;
              output.insert(output.end(), bytes.begin(), bytes.end());
            },
            [this] { return status(); }, limits) {
    storage.commandInFlight = [this] { return loop.commandInFlight(); };
  }

  // Every event the loop has written.
  std::vector<protocol::EngineEvent> events() const {
    return protocol::peer::decodeEvents(output);
  }

  test::ScopedTestConfig seam;
  test::TestKvStorage storage;
  KvPool pool;
  engine::Cache cache;
  Executor executor;
  std::vector<uint8_t> output;
  // Thrown by every write while set, as by a closed output pipe.
  std::optional<std::system_error> writeFailure;
  // What the loop answers a status request with.
  std::function<std::string()> status = test::readyStatusJson;
  double monotonic = 100.0;
  double clockStep = 0.0;
  engine::NativeRuntime loop;
};

void testPromptProgress() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 8192;
  LoopFixture fixture(config, {.pages = 512});
  engine::NativeRuntime &loop = fixture.loop;
  Executor &executor = fixture.executor;
  fixture.clockStep = 0.25;
  loop.announceReady();
  auto submit = [&](uint64_t id, bool enabled) {
    auto input = request(id);
    input.returnProgress = enabled;
    input.promptTokens.resize(4097);
    for (uint32_t i = 0; i < input.promptTokens.size(); ++i)
      input.promptTokens[i] = i + 1;
    require(loop.receive(protocol::peer::serialize(input)),
            "progress request failed");
  };

  executor.ticketReady = std::make_shared<std::atomic<bool>>(false);
  submit(1, true);
  require(loop.tick() && loop.commandInFlight(), "prefill was not submitted");
  for (int i = 0; i < 3; ++i)
    static_cast<void>(loop.tick());
  uint32_t count = 0;
  for (const auto &message : fixture.events()) {
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
  for (const auto &message : fixture.events()) {
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

  fixture.output.clear();
  *executor.ticketReady = false;
  submit(4, true);
  require(loop.tick() && loop.commandInFlight(),
          "cancel test needs pending work");
  require(loop.receive(protocol::peer::serialize(protocol::CancelFrame{4})),
          "cancel request failed");
  *executor.ticketReady = true;
  runUntilIdle(loop);
  count = 0;
  bool cancelled = false;
  for (const auto &message : fixture.events()) {
    if (std::holds_alternative<protocol::PromptProgressEvent>(message))
      ++count;
    if (const auto *event = std::get_if<protocol::DoneEvent>(&message))
      cancelled = event->reason == EngineFinishReason::Cancelled;
  }
  require(count == 1 && cancelled,
          "cancelled prefill published further progress");
}

void testWireLifecycleAndCacheHit() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  Executor &executor = fixture.executor;
  fixture.clockStep = 0.25;

  loop.announceReady();
  require(loop.receive(protocol::peer::serialize(request(1))),
          "cold request wire failed");
  require(loop.tick() && loop.commandInFlight(),
          "status regression requires a pending command");
  require(
      loop.receive(protocol::peer::serialize(protocol::StatusRequestFrame{77})),
      "in-flight status request failed");
  const auto pendingMessages = fixture.events();
  require(loop.commandInFlight() &&
              std::any_of(pendingMessages.begin(), pendingMessages.end(),
                          [](const auto &message) {
                            const auto *event =
                                std::get_if<protocol::StatusJsonEvent>(&message);
                            return event && event->correlationId == 77;
                          }),
          "status waited for or drained the pending command");
  runUntilIdle(loop);
  require(loop.receive(protocol::peer::serialize(request(2))),
          "junction request wire failed");
  runUntilIdle(loop);
  require(loop.receive(protocol::peer::serialize(request(3))),
          "restored request wire failed");
  runUntilIdle(loop);
  auto messages = fixture.events();
  uint32_t misses = 0;
  uint32_t hits = 0;
  uint32_t tokens = 0;
  uint32_t done = 0;
  bool statusSeen = false;
  for (const auto &message : messages) {
    if (const auto *start = std::get_if<protocol::StartEvent>(&message)) {
      if (!start->matchedPromptTokens) {
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
      statusSeen = reported->correlationId == 77;
    }
  }
  require(misses == 1 && hits == 2 && executor.restored() == 128,
          "cold/latest-replay/hit classification is wrong");
  require(tokens == 3 && done == 3 && statusSeen,
          "native lifecycle events are incomplete");
}

// The request's generation prompt reaches the engine: its replay state, which
// an identical retry resumes from, ends before it.
void testGenerationPromptBoundsTheReplayState() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  fixture.clockStep = 0.25;
  loop.announceReady();
  for (uint64_t id : {1, 2}) {
    auto input = request(id);
    input.generationPromptTokens = 2;
    require(loop.receive(protocol::peer::serialize(input)),
            "generation prompt request wire failed");
    runUntilIdle(loop);
  }
  std::vector<uint32_t> matched;
  for (const auto &message : fixture.events()) {
    if (const auto *start = std::get_if<protocol::StartEvent>(&message))
      matched.push_back(start->matchedPromptTokens);
  }
  require(matched == std::vector<uint32_t>{0, 32},
          "the replay state did not end before the generation prompt");
}

// A request's flags reach the model with the rest of its request.
void testRequestFlagsReachTheModel() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  Executor &executor = fixture.executor;
  fixture.clockStep = 0.25;
  loop.announceReady();
  for (uint64_t id : {1, 2}) {
    auto input = request(id);
    input.flags = id == 2 ? RequestIgnoreEndOfSequence : 0;
    require(loop.receive(protocol::peer::serialize(input)),
            "flagged request wire failed");
    runUntilIdle(loop);
  }
  require(executor.beganFlags ==
              std::unordered_map<uint64_t, uint32_t>{
                  {1, 0}, {2, RequestIgnoreEndOfSequence}},
          "request flags did not reach the model");
}

// The penalties cross the wire to the model with the rest of the sampling;
// greedy requests carry them too.
void testSamplingReachesTheModel() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  Executor &executor = fixture.executor;
  fixture.clockStep = 0.25;
  loop.announceReady();
  auto greedy = request(1);
  greedy.sampling = {0.0f, 1.0f, 0, 1.5f, 0.0f, 1.1f, 0.2f};
  auto sampled = request(2);
  sampled.sampling = {0.7f, 0.8f, 20, -0.5f, 2.0f, 0.9f, 0.05f};
  sampled.sampling.seed = 77;
  for (const auto &input : {greedy, sampled}) {
    require(loop.receive(protocol::peer::serialize(input)),
            "sampled request wire failed");
    runUntilIdle(loop);
  }
  require(executor.beganSampling.at(1) == greedy.sampling &&
              executor.beganSampling.at(2) == sampled.sampling,
          "a penalty, min_p or the seed did not reach the model");
}

// A header without the magic, and an event the engine sends itself: neither
// is a client frame, so framing is lost and the connection closes.
void testFatalFramingClosesConnection() {
  const auto event = protocol::serializeEvent(protocol::StatusJsonEvent{1, "{}"},
                                              protocol::ProtocolLimits{});
  require(static_cast<bool>(event), "status event encoding failed");
  for (const auto &[invalid, code] :
       {std::pair{std::vector<uint8_t>(24), "bad_magic"},
        std::pair{*event.value, "unknown_frame_type"}}) {
    LoopFixture fixture({}, {.pages = 8});
    engine::NativeRuntime &loop = fixture.loop;
    require(!loop.receive(invalid), "bad frame did not close connection");
    require(loop.connectionMustClose() && loop.engineHealthy(),
            "protocol failure was misclassified as engine failure");
    const auto events = fixture.events();
    const auto *error = events.size() == 1
                            ? std::get_if<protocol::ErrorEvent>(&events.front())
                            : nullptr;
    require(error &&
                error->failureClass == protocol::FailureClass::ProtocolFatal &&
                error->code == code,
            "a frame that is not a client frame was not reported as fatal");
  }
}

// A request rejected while it is decoded is a request-scoped error: the
// frames behind it in the same read, whole or cut by the read boundary, must
// still be processed.
void testRequestErrorKeepsFraming() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  protocol::ProtocolLimits limits;
  limits.maxLogicalOutputTokens = 1;
  LoopFixture fixture(config, {}, limits);
  engine::NativeRuntime &loop = fixture.loop;

  loop.announceReady();
  // Errors, completions and status answers so far; every error must be the
  // rejected request's own.
  const auto events = [&] {
    std::array<uint32_t, 3> counts{};
    for (const protocol::EngineEvent &message : fixture.events()) {
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

  const auto rejected = protocol::peer::serialize(request(9, 2));
  const auto first = protocol::peer::serialize(request(1));
  const auto status =
      protocol::peer::serialize(protocol::StatusRequestFrame{77});
  std::vector<uint8_t> read = rejected;
  read.insert(read.end(), first.begin(), first.end());
  read.insert(read.end(), status.begin(), status.end());
  require(loop.receive(read), "request-scoped error closed the connection");
  runUntilIdle(loop);
  require(events() == std::array<uint32_t, 3>{1, 1, 1},
          "frames behind a rejected request were dropped");

  const auto second = protocol::peer::serialize(request(2));
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
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config, {.pages = 4, .extentPages = 1, .runwayPages = 1});
  engine::NativeRuntime &loop = fixture.loop;
  fixture.storage.budgetPages = 1;

  loop.announceReady();
  require(loop.receive(protocol::peer::serialize(request(3))),
          "capacity request wire failed");
  runUntilIdle(loop);

  uint32_t capacity = 0;
  uint32_t errors = 0;
  uint32_t done = 0;
  for (const protocol::EngineEvent &message : fixture.events()) {
    if (const auto *error = std::get_if<protocol::ErrorEvent>(&message)) {
      ++errors;
      capacity += error->failureClass == protocol::FailureClass::RequestError &&
                  !error->retryable &&
                  error->code ==
                      laneOutcomeWire(LaneOutcome::CapacityExhausted).code;
    }
    done += std::holds_alternative<protocol::DoneEvent>(message);
  }
  require(capacity == 1 && errors == 1 && done == 0,
          "capacity failure emitted more than one terminal frame");
  require(loop.engineHealthy(),
          "request-scoped capacity failure made the engine unhealthy");
}

void testCommandWatchdogAndPendingHealthWake() {
  metal::CommandWatchdog generations(120.0);
  generations.start(1, 0.0);
  generations.complete(1);
  require(!generations.expired(1000.0), "completed command retained a deadline");
  generations.start(2, 10.0);
  generations.complete(1);
  require(!generations.expired(129.999) && generations.expired(130.0),
          "stale completion cleared a newer command deadline");

  for (bool gpuCompleted : {false, true}) {
    LoopFixture fixture;
    engine::NativeRuntime &loop = fixture.loop;
    Executor &executor = fixture.executor;
    double &now = fixture.monotonic;
    now = 0.0;
    executor.ticketReady = std::make_shared<std::atomic<bool>>(false);
    metal::CommandWatchdog watchdog(120.0);
    executor.onSubmit = [&] { watchdog.start(1, now / 1000.0); };
    executor.onHealthCheck = [&] {
      if (watchdog.expired(now / 1000.0))
        throw metal::MetalBackendError("test command completion timeout");
    };
    loop.announceReady();
    auto input = request(1);
    input.absoluteDeadlineUnixMicros = 601'000'000;
    input.remainingDeadlineMicros = 600'000'000;
    require(loop.receive(protocol::peer::serialize(input)) && loop.tick(),
            "watchdog fixture did not submit its command");
    require(loop.receive(protocol::peer::serialize(protocol::CancelFrame{1})) &&
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
      require(loop.tick() && idle(loop), "completed GPU ownership did not drain");
    } else {
      uint32_t errors = 0;
      for (const auto &message : fixture.events()) {
        if (const auto *error = std::get_if<protocol::ErrorEvent>(&message)) {
          require(error->requestId == 0 &&
                      error->failureClass == protocol::FailureClass::EngineUnhealthy &&
                      error->code == "metal_execution_failed",
                  "watchdog timeout lost structured engine failure");
          ++errors;
        }
      }
      require(errors == 1 && loop.connectionMustClose() &&
                  loop.commandInFlight() && !idle(loop),
              "watchdog released command ownership or emitted duplicate failures");
    }
  }

  LoopFixture fixture;
  engine::NativeRuntime &loop = fixture.loop;
  require(!loop.tick() && idle(loop) && !loop.millisecondsUntilNextWakeup(),
          "fully idle engine retained a polling wake");
}

void testDuplicateLiveRequestClosesWithoutAmbiguousError() {
  for (bool malformed : {false, true}) {
    protocol::ProtocolLimits limits;
    limits.maxLogicalOutputTokens = 1;
    LoopFixture fixture({}, {}, limits);
    engine::NativeRuntime &loop = fixture.loop;
    loop.announceReady();
    require(loop.receive(protocol::peer::serialize(request(1))) && loop.tick(),
            "live duplicate fixture did not start");
    require(!loop.receive(
                protocol::peer::serialize(request(1, malformed ? 2 : 1))) &&
                loop.connectionMustClose() && loop.snapshot().submitted == 1,
            "duplicate live request was accepted");
    uint32_t errors = 0;
    for (const auto &message : fixture.events()) {
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

// The protocol frees an id when its request ends. A cancel and a new request
// for that id in one input start the new request while the engine still
// holds the cancelled one's finished entry.
void testCancelledIdIsReusableInTheSameInput() {
  LoopFixture fixture;
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady();
  require(loop.receive(protocol::peer::serialize(request(7, 4))) &&
              loop.tick() && loop.tick() && !loop.commandInFlight(),
          "the first request did not start");
  std::vector<uint8_t> input =
      protocol::peer::serialize(protocol::CancelFrame{7});
  const std::vector<uint8_t> again = protocol::peer::serialize(request(7));
  input.insert(input.end(), again.begin(), again.end());
  require(loop.receive(input),
          "a cancel and a request for one id closed the connection");
  runUntilIdle(loop);
  std::vector<EngineFinishReason> done;
  uint32_t errors = 0;
  for (const auto &message : fixture.events()) {
    if (const auto *event = std::get_if<protocol::DoneEvent>(&message))
      done.push_back(event->reason);
    errors += std::holds_alternative<protocol::ErrorEvent>(message);
  }
  require(errors == 0 &&
              done == std::vector<EngineFinishReason>{
                          EngineFinishReason::Cancelled,
                          EngineFinishReason::Stop},
          "a cancelled id was not reusable in the same input");
}

void testControlFailureUsesExecutionBoundary() {
  for (bool metalFailure : {false, true}) {
    LoopFixture fixture;
    engine::NativeRuntime &loop = fixture.loop;
    loop.announceReady();
    require(loop.runControl([] { return true; }) && loop.engineHealthy(),
            "ordinary deferred control work failed");
    require(!loop.runControl([&]() -> bool {
              if (metalFailure)
                throw metal::MetalBackendError("control test");
              throw std::runtime_error("control test");
            }), "failed control work requested another retry");
    uint32_t errors = 0;
    for (const auto &message : fixture.events()) {
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

// A frame whose handling throws stops the engine through the same boundary
// as tick(): one EngineUnhealthy event, whatever was thrown, with a Metal
// failure named and counted.
void testFrameFailureUsesExecutionBoundary() {
  enum class Thrown { Standard, Metal, Foreign };
  for (const Thrown thrown : {Thrown::Standard, Thrown::Metal, Thrown::Foreign}) {
    RuntimeMetrics metrics;
    engine::NativeLoopConfig config;
    config.metrics = &metrics;
    LoopFixture fixture(config, {.pages = 8});
    engine::NativeRuntime &loop = fixture.loop;
    fixture.status = [thrown]() -> std::string {
      if (thrown == Thrown::Standard)
        throw std::runtime_error("status test");
      if (thrown == Thrown::Metal)
        throw metal::MetalBackendError("status test");
      throw 42;
    };
    loop.announceReady();
    require(!loop.receive(protocol::peer::serialize(
                protocol::StatusRequestFrame{77})) &&
                !loop.engineHealthy() && loop.connectionMustClose(),
            "a failed status frame did not stop the engine");
    uint32_t errors = 0;
    for (const auto &message : fixture.events()) {
      if (const auto *error = std::get_if<protocol::ErrorEvent>(&message)) {
        require(error->requestId == 0 &&
                    error->failureClass == protocol::FailureClass::EngineUnhealthy &&
                    error->code == (thrown == Thrown::Metal
                                        ? "metal_execution_failed"
                                        : "engine_execution_failed"),
                "a frame exception lost its engine failure classification");
        ++errors;
      }
    }
    require(errors == 1 && metrics.snapshot().metalFailures ==
                               (thrown == Thrown::Metal ? 1U : 0U),
            "a frame exception was reported or counted more than once");
  }
}

// The loop checks its protocol limits once, when it is built; the codec and
// the parser rely on them.
void testInvalidLimitsAreRejectedAtConstruction() {
  protocol::ProtocolLimits limits;
  limits.maxMaskWords = 0;
  bool refused = false;
  try {
    LoopFixture fixture({}, {.pages = 8}, limits);
  } catch (const std::invalid_argument &error) {
    refused = std::string(error.what()).find("limit_exceeded") !=
              std::string::npos;
  }
  require(refused, "the loop accepted invalid protocol limits");
}

// An exception while the engine admits a request is engine-fatal: nothing
// below the engine rolls back, and the loop reports it once.
void testAdmissionExceptionStopsTheEngineOnce() {
  LoopFixture fixture;
  engine::NativeRuntime &loop = fixture.loop;
  fixture.executor.onBegin = [] { throw std::runtime_error("begin failed"); };
  loop.announceReady();
  require(loop.receive(protocol::peer::serialize(request(1))),
          "admission fixture was refused");
  require(!loop.tick() && !loop.engineHealthy() && loop.connectionMustClose(),
          "an admission exception did not stop the engine");
  uint32_t errors = 0;
  for (const auto &message : fixture.events()) {
    if (const auto *error = std::get_if<protocol::ErrorEvent>(&message)) {
      require(error->requestId == 0 &&
                  error->failureClass == protocol::FailureClass::EngineUnhealthy &&
                  error->code == "engine_execution_failed" &&
                  error->message == "begin failed",
              "an admission exception lost its engine failure");
      ++errors;
    }
  }
  require(errors == 1, "an admission exception was reported more than once");
}

// Every path that stops the engine keeps its reason for the exit log, not
// only the ones that report through engineError().
void testEngineFailureNamesItsReason() {
  {
    LoopFixture fixture({}, {.pages = 8});
    engine::NativeRuntime &loop = fixture.loop;
    const std::system_error closed(EPIPE, std::generic_category(),
                                   "write(native output)");
    loop.announceReady();
    fixture.writeFailure = closed;
    require(!loop.receive(
                protocol::peer::serialize(protocol::StatusRequestFrame{77})) &&
                !loop.engineHealthy() && loop.connectionMustClose(),
            "a failed output write did not stop the engine");
    require(loop.engineFailure() ==
                std::string("output_write_failed: ") + closed.what(),
            "a failed output write left the engine failure unnamed");
  }
  {
    LoopFixture fixture({}, {.pages = 8});
    engine::NativeRuntime &loop = fixture.loop;
    loop.announceReady();
    const auto frame = protocol::peer::serialize(request(1));
    // A header and one payload byte: the parser's first allocation is the
    // payload buffer, and it fails.
    allocationFailureAfter = 0;
    const bool received = loop.receive(std::span<const uint8_t>(
        frame.data(), protocol::kFrameHeaderBytes + 1));
    allocationFailureAfter = -1;
    uint32_t errors = 0;
    for (const auto &message : fixture.events()) {
      if (const auto *error = std::get_if<protocol::ErrorEvent>(&message)) {
        require(error->failureClass ==
                        protocol::FailureClass::EngineUnhealthy &&
                    error->code == "allocation_failure",
                "an inbound allocation failure lost its classification");
        ++errors;
      }
    }
    require(!received && errors == 1 && !loop.engineHealthy() &&
                loop.connectionMustClose(),
            "an inbound allocation failure did not stop the engine");
    require(loop.engineFailure() ==
                "allocation_failure: allocation failed while receiving frame "
                "payload",
            "an inbound allocation failure left the engine failure unnamed");
  }
}

void testOutOfVocabularyTokensStayRequestScoped() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.engine.vocabularySize = 128;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady();
  std::vector<protocol::RequestFrame> invalid;
  for (uint32_t token : {128U, std::numeric_limits<uint32_t>::max()}) {
    invalid.push_back(request(9));
    invalid.back().promptTokens.back() = token;
  }
  invalid.push_back(scoreRequest(9, 65));
  invalid.back().scoreTokens.back() = 128;
  for (const protocol::RequestFrame &frame : invalid) {
    require(loop.receive(protocol::peer::serialize(frame)),
            "out-of-vocabulary token closed the native connection");
  }
  auto valid = request(1);
  valid.promptTokens.back() = 127;
  require(loop.receive(protocol::peer::serialize(valid)),
          "valid request after invalid tokens was rejected");
  runUntilIdle(loop);
  uint32_t errors = 0, done = 0;
  for (const auto &message : fixture.events()) {
    if (const auto *error = std::get_if<protocol::ErrorEvent>(&message)) {
      require(error->requestId == 9 && error->code == "invalid_request" &&
                  error->failureClass == protocol::FailureClass::RequestError,
              "out-of-vocabulary token did not produce its own request error");
      ++errors;
    }
    done += std::holds_alternative<protocol::DoneEvent>(message);
  }
  require(errors == invalid.size() && done == 1 && loop.engineHealthy() &&
              loop.snapshot().submitted == 1,
          "invalid tokens reached admission or prevented subsequent completion");
}

// Whether Ready announces vision for an engine admitting images of up to
// `maxImagePatches` patches.
bool announcedVision(uint32_t maxImagePatches) {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.engine.maxImagePatches = maxImagePatches;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady();
  const auto announced = fixture.events();
  const auto *ready = announced.size() == 1
                          ? std::get_if<protocol::ReadyEvent>(&announced.front())
                          : nullptr;
  require(ready, "announceReady did not send exactly one Ready event");
  return ready->vision;
}

void testReadyAnnouncesVisionWhenImagesAreAdmitted() {
  require(announcedVision(ops::kMaximumImagePatches),
          "Ready did not announce vision for an engine that admits images");
  require(!announcedVision(0),
          "Ready announced vision for an engine serving without it");
}

// Without vision, or past the server's per-image patch cap, an image request
// fails by itself and the engine keeps serving.
void testImageRequestBeyondVisionStaysRequestScoped() {
  // The image below has 64 patches.
  const std::pair<uint32_t, std::string_view> cases[] = {
      {0, "this model is serving without vision"},
      {32, "image has more patches than the server's pixel cap allows"}};
  for (const auto &[maxImagePatches, expected] : cases) {
    engine::NativeLoopConfig config;
    config.engine.maxContext = 1024;
    config.engine.maxImagePatches = maxImagePatches;
    LoopFixture fixture(config);
    engine::NativeRuntime &loop = fixture.loop;
    loop.announceReady();
    auto image = request(9);
    image.imageSpans = {{8, 16, 8, 8, 1, 2}};
    image.imagePixels.assign(image.imageSpans[0].pixelBytes(), 1);
    for (const protocol::RequestFrame &frame : {image, request(1)}) {
      require(loop.receive(protocol::peer::serialize(frame)),
              "image request closed the native connection");
    }
    runUntilIdle(loop);
    uint32_t errors = 0, done = 0;
    for (const auto &message : fixture.events()) {
      if (const auto *error = std::get_if<protocol::ErrorEvent>(&message)) {
        require(error->requestId == 9 && error->code == "invalid_request" &&
                    error->failureClass ==
                        protocol::FailureClass::RequestError &&
                    error->message == expected,
                "image request did not produce its own vision error");
        ++errors;
      }
      done += std::holds_alternative<protocol::DoneEvent>(message);
    }
    require(errors == 1 && done == 1 && loop.engineHealthy() &&
                loop.snapshot().submitted == 1,
            "image request reached admission or stopped the engine");
  }
}

// A verify step can retain every row and add the terminal anchor; the wire
// limit the production binary derives from ExecutionLimits must carry that
// step in one TokensEvent, and an event that cannot be encoded must surface
// as an engine error rather than a silently shorter stream.
void testStepTokensFitTheWire() {
  for (uint32_t limit : {model::ExecutionLimits::maximumStepTokens, 1U}) {
    engine::NativeLoopConfig config;
    config.engine.maxContext = 1024;
    protocol::ProtocolLimits limits;
    limits.maxTokenBatch = limit;
    LoopFixture fixture(config, {}, limits);
    engine::NativeRuntime &loop = fixture.loop;
    Executor &executor = fixture.executor;
    executor.stepTokens = model::ExecutionLimits::maximumStepTokens;

    loop.announceReady();
    require(loop.receive(
                protocol::peer::serialize(request(5, executor.stepTokens))),
            "step request wire failed");
    runUntilIdle(loop);

    uint32_t streamed = 0;
    std::optional<uint32_t> completion;
    uint32_t encodeErrors = 0;
    for (const protocol::EngineEvent &message : fixture.events()) {
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
      require(loop.engineFailure().starts_with("protocol_encode_failed: "),
              "an unencodable event left the engine failure unnamed");
    }
  }
}

void testScoreRequestCompletesAfterFullPrompt() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 8192;
  LoopFixture fixture(config, {.pages = 512});
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady();
  require(loop.receive(protocol::peer::serialize(scoreRequest(9, 3000))),
          "score request failed");
  runUntilIdle(loop);

  uint32_t tokensEvents = 0;
  uint32_t doneCount = 0;
  for (const protocol::EngineEvent &message : fixture.events()) {
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
      require(done->reason == EngineFinishReason::Stop,
              "score done reason is not stop");
    }
  }
  require(doneCount == 1 && tokensEvents == 0,
          "score request did not complete without generating tokens");
}

void testCancelledScoreReturnsEmptyLogits() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  Executor &executor = fixture.executor;
  executor.ticketReady = std::make_shared<std::atomic<bool>>(false);
  loop.announceReady();
  require(loop.receive(protocol::peer::serialize(scoreRequest(11, 65))),
          "cancel-score request failed");
  require(loop.tick() && loop.commandInFlight(), "score prefill was not held");
  protocol::CancelFrame cancel{11};
  require(loop.receive(protocol::peer::serialize(cancel)),
          "score cancel failed");
  *executor.ticketReady = true;
  runUntilIdle(loop);

  uint32_t doneCount = 0;
  for (const protocol::EngineEvent &message : fixture.events()) {
    if (const auto *done = std::get_if<protocol::DoneEvent>(&message)) {
      ++doneCount;
      require(done->reason == EngineFinishReason::Cancelled,
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
  engine::NativeLoopConfig config;
  config.engine.maxContext = 8192;
  LoopFixture fixture(config, {.pages = 512});
  engine::NativeRuntime &loop = fixture.loop;
  Executor &executor = fixture.executor;
  if (invalidScore)
    executor.invalidScores.insert(21);
  loop.announceReady();

  // Whole KV pages, so the last prompt chunk is the one that publishes the
  // final block. The chat lane deliberately does not share this prefix.
  auto scored = scoreRequest(21, 4 * KvCache::pageTokens);
  for (uint32_t index = 0; index < scored.promptTokens.size(); ++index)
    scored.promptTokens[index] = 1000 + index;
  require(loop.receive(protocol::peer::serialize(scored)),
          "score request wire failed");
  require(loop.receive(protocol::peer::serialize(request(22))),
          "batched chat request wire failed");
  runUntilIdle(loop);

  ScoreBesideChat result;
  result.publishedBlocks = fixture.cache.snapshot().pool.pagesPrefix;
  result.healthy = loop.engineHealthy() && !loop.connectionMustClose();
  result.slotsReleased = !executor.holdsSlot(21) && !executor.holdsSlot(22);
  result.scoreChunks = executor.prefillChunks[21];
  result.widestBatch = executor.widestBatch;

  require(loop.receive(protocol::peer::serialize(request(23))),
          "post-batch request wire failed");
  runUntilIdle(loop);

  for (const protocol::EngineEvent &message : fixture.events()) {
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

// A constrained request's initial token mask crosses the native protocol.
// Only the response to the pending mask request, with one row of the
// configured width, reaches the model. Any other response fails that
// request alone, and one that arrives after the request ended, cancelled or
// timed out waiting for it, is ignored.
void testConstrainedMaskExchange() {
  enum class Reply {
    Valid,
    WrongMaskId,
    WrongWordCount,
    EmptyRow,
    Malformed,
    AfterCancel,
    AfterTimeout
  };
  for (Reply reply :
       {Reply::Valid, Reply::WrongMaskId, Reply::WrongWordCount,
        Reply::EmptyRow, Reply::Malformed, Reply::AfterCancel,
        Reply::AfterTimeout}) {
    engine::NativeLoopConfig config;
    config.engine.maxContext = 1024;
    // Three mask words per token; the prompt's tokens stay in vocabulary.
    config.engine.vocabularySize = 96;
    LoopFixture fixture(config);
    engine::NativeRuntime &loop = fixture.loop;
    Executor &executor = fixture.executor;
    double &now = fixture.monotonic;
    loop.announceReady();
    const auto send = [&](const protocol::ClientMessage &message) {
      require(loop.receive(protocol::peer::serialize(message)),
              "mask exchange message closed the connection");
    };
    auto constrained = request(7);
    constrained.priority = RequestPriority::Background;
    constrained.constraint = ConstraintMode::TokenMask;
    constrained.absoluteDeadlineUnixMicros = 601'000'000;
    constrained.remainingDeadlineMicros = 600'000'000;
    send(constrained);
    while (loop.tick()) {
    }
    std::optional<protocol::MaskRequestEvent> asked;
    for (const auto &message : fixture.events()) {
      if (const auto *event = std::get_if<protocol::MaskRequestEvent>(&message))
        asked = *event;
    }
    require(asked && asked->requestId == 7 && asked->maskRequestId &&
                asked->wordsPerMask == 3 && asked->simulationTokens.empty(),
            "constrained request did not ask for one initial mask row");

    protocol::MaskResponseFrame response{7, asked->maskRequestId, {1, 0, 0}};
    if (reply == Reply::WrongMaskId)
      ++response.maskRequestId;
    if (reply == Reply::WrongWordCount)
      response.maskWords.push_back(0);
    if (reply == Reply::EmptyRow)
      response.maskWords = {0, 0, 0};
    if (reply == Reply::AfterCancel) {
      send(protocol::CancelFrame{7});
      runUntilIdle(loop);
    }
    if (reply == Reply::AfterTimeout) {
      now += 5000.0;
      runUntilIdle(loop);
    }
    if (reply == Reply::Malformed) {
      // The frame claims one more mask word than it carries.
      auto wire = protocol::peer::serialize(response);
      ++wire[protocol::kFrameHeaderBytes + 16];
      require(loop.receive(wire),
              "malformed mask response closed the connection");
    } else {
      send(response);
    }
    runUntilIdle(loop);

    std::optional<EngineFinishReason> done;
    std::vector<std::string> errors;
    std::string errorMessage;
    uint32_t maskRequests = 0;
    for (const auto &message : fixture.events()) {
      if (const auto *event = std::get_if<protocol::DoneEvent>(&message))
        done = event->reason;
      if (const auto *error = std::get_if<protocol::ErrorEvent>(&message)) {
        require(error->requestId == 7 &&
                    error->failureClass == protocol::FailureClass::RequestError,
                "mask response failure was not the request's own error");
        errors.push_back(error->code);
        errorMessage = error->message;
      }
      maskRequests += std::holds_alternative<protocol::MaskRequestEvent>(message);
    }
    require(maskRequests == 1 && loop.engineHealthy() &&
                !loop.connectionMustClose() && !executor.holdsSlot(7),
            "mask exchange stopped the engine or kept the request's slot");
    if (reply == Reply::Valid) {
      require(done == EngineFinishReason::Stop && errors.empty() &&
                  executor.providedMasks == 1,
              "valid initial mask did not let the request finish");
    } else if (reply == Reply::AfterCancel) {
      require(done == EngineFinishReason::Cancelled && errors.empty() &&
                  executor.providedMasks == 0,
              "mask response after cancellation was not ignored");
    } else if (reply == Reply::AfterTimeout) {
      require(!done && errors == std::vector<std::string>{"mask_timeout"} &&
                  executor.providedMasks == 0,
              "mask response after its timeout was not ignored");
    } else {
      require(!done &&
                  errors == std::vector<std::string>{"invalid_mask_response"} &&
                  executor.providedMasks == 0,
              "mismatched mask response did not fail only its request");
      require(reply != Reply::Malformed ||
                  errorMessage.starts_with("invalid_payload_length: "),
              "a malformed mask response lost its decoding issue");
    }
  }
}

// Between commands the control pass runs what the host's pressure asks for:
// a paced pass toward the recovery margin, which keeps the resume point and
// the empty runway extent; nothing once the host recovers; and under critical
// pressure every cache entry and extent. None waits for a transfer.
void testControlPassReclaimsUnderHostPressure() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  const engine::Cache &resources = fixture.cache;
  const KvPool &pool = fixture.pool;
  loop.announceReady();
  // Two finished requests with different prompts leave two replay states.
  for (uint64_t id : {1, 2}) {
    auto input = request(id);
    for (uint32_t &token : input.promptTokens)
      token += static_cast<uint32_t>(100 * id);
    require(loop.receive(protocol::peer::serialize(input)),
            "control pass request failed");
    runUntilIdle(loop);
  }
  require(resources.snapshot().stateCache.entries == 2,
          "the fixture did not cache two replay states");

  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  constexpr uint64_t hostReserve = 2ULL << 30;
  std::optional<uint64_t> available = hostReserve + kHostWarningMarginBytes / 2;
  MemoryGovernor governor(backend, 40ULL << 30, hostReserve,
                          [&available] { return available; }, 0);
  MemoryControl control(governor, backend, loop);
  require(!control.run(MemoryPressure::Normal), "a paced pass waited for a transfer");
  const auto paced = resources.snapshot();
  require(paced.stateCache.entries == 1 && paced.stateCache.evictions == 1 &&
              paced.pool.reclaimableBytes == 4 * 4096,
          "host pressure did not evict toward its target, keeping the resume "
          "point and the runway extent");

  available = hostReserve + kHostRecoveryMarginBytes + kHostWarningMarginBytes;
  const uint64_t releases = pool.snapshot().extentReleases;
  // The host's recovery lifts the pressure, so no pass runs: the untargeted
  // pass the settle window would otherwise run finds nothing to take either.
  require(!control.run(MemoryPressure::Normal) &&
              governor.snapshot().pressure == MemoryPressure::Normal &&
              resources.snapshot().stateCache.entries == 1 &&
              pool.snapshot().extentReleases == releases,
          "a recovered host still reclaimed");

  require(!control.run(MemoryPressure::Critical), "a critical pass waited for a transfer");
  const auto critical = resources.snapshot();
  require(critical.stateCache.entries == 0 && critical.pool.pagesPrefix == 0 &&
              critical.pool.pagesAllocated == 0,
          "critical pressure did not empty the cache and release every extent");
}

// Weights of three images that count what the loop does with them.
class Weights final : public model::WeightMemory {
public:
  static constexpr uint32_t kImages = 3;
  [[nodiscard]] bool released() const noexcept override {
    return restored_ < kImages;
  }
  void release() override {
    require(!released(), "weights were released twice");
    restored_ = 0;
  }
  bool restore() override {
    if (failRestore)
      throw std::runtime_error("weights restore test");
    require(released(), "held weights were restored");
    ++restores;
    return ++restored_ == kImages;
  }
  bool failRestore = false;
  // The images written back.
  uint32_t restores = 0;

private:
  uint32_t restored_ = kImages;
};

bool answeredStatus(const LoopFixture &fixture, uint64_t correlationId) {
  const auto events = fixture.events();
  return std::any_of(events.begin(), events.end(), [&](const auto &message) {
    const auto *event = std::get_if<protocol::StatusJsonEvent>(&message);
    return event && event->correlationId == correlationId;
  });
}

// The control pass between commands releases the weights once the engine has
// held no request for the keep-alive, counted from the end of the last one
// however long it ran, never while one is there. A request waits while they
// are written back, an image per tick, so the loop answers frames between
// them, and the images all come back even when it is cancelled. A restore
// that fails stops the engine.
void testIdleWeightsAreReleasedAndRestored() {
  constexpr double keepAlive = 1000.0 * kWeightKeepAliveSeconds;
  Weights weights;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.weights = &weights;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  fixture.executor.onBegin = [&] {
    require(!weights.released(), "the model began a request on released weights");
  };
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  MemoryGovernor governor(backend, 40ULL << 30, 2ULL << 30,
                          [] { return std::optional<uint64_t>{64ULL << 30}; }, 0);
  MemoryControl control(governor, backend, loop);
  const auto idlePass = [&](double elapsed) {
    fixture.monotonic += elapsed;
    static_cast<void>(control.run(MemoryPressure::Normal));
    return weights.released();
  };
  loop.announceReady();
  require(!idlePass(keepAlive - 1), "weights were released before the keep-alive passed");
  require(idlePass(1), "idle weights were kept past the keep-alive");

  require(loop.receive(protocol::peer::serialize(request(1))) && weights.restores == 0,
          "weights were written back before a tick");
  require(loop.tick() && weights.restores == 1 && weights.released(),
          "a tick did not write back one image");
  require(loop.receive(protocol::peer::serialize(protocol::StatusRequestFrame{77})) &&
              answeredStatus(fixture, 77),
          "the loop did not answer status while it restored the weights");
  while (weights.released())
    require(loop.tick(), "a tick did not write back an image");
  require(weights.restores == Weights::kImages, "an image was written back twice");
  require(!idlePass(2 * keepAlive), "weights were released under a request");
  runUntilIdle(loop);

  require(loop.receive(protocol::peer::serialize(request(2))), "a second request failed");
  // It runs past the keep-alive, without a control pass between commands.
  fixture.clockStep = keepAlive;
  runUntilIdle(loop);
  fixture.clockStep = 0;
  require(!idlePass(0), "weights were released as soon as a long request ended");
  require(idlePass(keepAlive), "weights were kept past the keep-alive after a request");

  require(loop.receive(protocol::peer::serialize(request(3))) && loop.tick() &&
              loop.receive(protocol::peer::serialize(protocol::CancelFrame{3})),
          "a request cancelled while the weights came back failed");
  while (weights.released())
    require(loop.tick(), "a cancelled request stopped its weights coming back");
  require(weights.restores == 2 * Weights::kImages, "an image was written back twice");
  // The transport ticks again after a tick that progressed: the engine then
  // lets the cancelled request go.
  static_cast<void>(loop.tick());

  require(idlePass(keepAlive), "weights were kept past the keep-alive after a cancel");
  weights.failRestore = true;
  require(loop.receive(protocol::peer::serialize(request(4))) && !loop.tick(),
          "a failed restore kept the engine running");
  const auto events = fixture.events();
  require(std::any_of(events.begin(), events.end(),
                      [](const auto &message) {
                        const auto *error = std::get_if<protocol::ErrorEvent>(&message);
                        return error &&
                               error->failureClass == protocol::FailureClass::EngineUnhealthy &&
                               error->message.find("weights restore test") != std::string::npos;
                      }) &&
              !loop.engineHealthy() && loop.connectionMustClose(),
          "a failed restore did not stop the engine");
}

// The reporter logs a change in what requests wait for, never a retry or the
// depth of a queue.
void testMemoryStatusReporterLogsTransitionsOnly() {
  ResourceWaitSnapshot wait{.memory = 2, .concurrency = 1, .heldBehindRefusal = 4,
                            .restoring = 1, .suspended = 1,
                            .oldestWaitMilliseconds = 1250.0, .draining = true};
  MemoryStatusReporter reporter;
  require(reporter.update({}, true).empty(), "healthy idle engine logged pressure");
  require(!reporter.update(wait, false).empty(), "pressure transition was silent");
  ++wait.memory;
  wait.oldestWaitMilliseconds += 1000;
  require(reporter.update(wait, false).empty(), "pressure retries flooded the log");
  wait = {};
  wait.concurrency = 4;
  require(!reporter.update(wait, true).empty(), "end of memory wait was silent");
  require(reporter.update(wait, true).empty(), "concurrency queue logged pressure");
  // The refused request is deferred for scheduling, out of the memory
  // count, while it still holds the others back.
  wait.heldBehindRefusal = 2;
  const std::string held = reporter.update(wait, true);
  require(held.find("held=2") != std::string::npos &&
              held.find("cleared") == std::string::npos,
          "requests held behind a refusal were reported as no wait");
}

} // namespace

int main() {
  try {
    testWireLifecycleAndCacheHit();
    testGenerationPromptBoundsTheReplayState();
    testRequestFlagsReachTheModel();
    testSamplingReachesTheModel();
    testPromptProgress();
    testCapacityFailureHasOneTerminalFrame();
    testFatalFramingClosesConnection();
    testInvalidLimitsAreRejectedAtConstruction();
    testRequestErrorKeepsFraming();
    testCommandWatchdogAndPendingHealthWake();
    testDuplicateLiveRequestClosesWithoutAmbiguousError();
    testCancelledIdIsReusableInTheSameInput();
    testControlFailureUsesExecutionBoundary();
    testAdmissionExceptionStopsTheEngineOnce();
    testFrameFailureUsesExecutionBoundary();
    testEngineFailureNamesItsReason();
    testOutOfVocabularyTokensStayRequestScoped();
    testReadyAnnouncesVisionWhenImagesAreAdmitted();
    testImageRequestBeyondVisionStaysRequestScoped();
    testStepTokensFitTheWire();
    testScoreRequestCompletesAfterFullPrompt();
    testCancelledScoreReturnsEmptyLogits();
    testInvalidScoreFailsOneRequestAndKeepsTheBatch();
    testConstrainedMaskExchange();
    testControlPassReclaimsUnderHostPressure();
    testIdleWeightsAreReleasedAndRestored();
    testMemoryStatusReporterLogsTransitionsOnly();
    std::cout << "native KV-first loop tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "native KV-first loop tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
