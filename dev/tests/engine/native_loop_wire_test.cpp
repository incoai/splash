#include "AllocationFailure.hpp"
#include "LoopFixture.hpp"
#include "ProtocolPeer.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestRegistry.hpp"
#include "TestStatus.hpp"
#include "engine/Cache.hpp"
#include "engine/NativeRuntime.hpp"
#include "metal/CommandWatchdog.hpp"

#include <algorithm>
#include <cerrno>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::Executor;
using splash::test::idle;
using splash::test::LoopFixture;
using splash::test::rejects;
using splash::test::request;
using splash::test::require;
using splash::test::runUntilIdle;
using splash::test::Weights;

protocol::RequestFrame scoreRequest(uint64_t id, uint32_t promptTokens) {
  protocol::RequestFrame result = request(id, 0);
  result.promptTokens.resize(promptTokens);
  for (uint32_t i = 0; i < result.promptTokens.size(); ++i)
    result.promptTokens[i] = i + 1;
  result.scoreTokens = {10, 20, 30};
  return result;
}

void testPromptProgress() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 8192;
  LoopFixture fixture(config, {.pages = 512});
  engine::NativeRuntime &loop = fixture.loop;
  Executor &executor = fixture.executor;
  fixture.clockStep = 0.25;
  loop.announceReady(protocol::peer::kTestReady);
  auto submit = [&](uint64_t id, bool enabled) {
    auto input = request(id);
    input.flags = enabled ? protocol::kReturnProgressFlag : 0;
    input.promptTokens.resize(4097);
    for (uint32_t i = 0; i < input.promptTokens.size(); ++i)
      input.promptTokens[i] = i + 1;
    require(loop.receive(protocol::peer::serialize(input)),
            "progress request failed");
  };

  const auto commandsReady = executor.holdCommands();
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
  *commandsReady = true;
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
  *commandsReady = false;
  submit(4, true);
  require(loop.tick() && loop.commandInFlight(),
          "cancel test needs pending work");
  require(loop.receive(protocol::peer::serialize(protocol::CancelFrame{4})),
          "cancel request failed");
  *commandsReady = true;
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

// /status active_requests: a live request's phase, prompt progress, output
// budget and age since its arrival, until it finishes.
void testSnapshotReportsLiveRequests() {
  engine::NativeLoopConfig config;
  LoopFixture fixture(config, {.pages = 512});
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady(protocol::peer::kTestReady);
  const auto commandsReady = fixture.executor.holdCommands();
  require(loop.receive(protocol::peer::serialize(request(7, 4))), "request failed");
  require(loop.tick() && loop.commandInFlight(), "prefill was not submitted");
  fixture.monotonic += 250.0;
  const std::vector<engine::ActiveRequestSnapshot> active = loop.statusSnapshot().activeRequests;
  require(active.size() == 1 && active[0].id == 7 && active[0].phase == engine::Phase::Prefill &&
              active[0].priority == RequestPriority::Foreground && active[0].promptTokens == 65 &&
              active[0].promptProcessed == 0 && active[0].generatedTokens == 0 &&
              active[0].maxNewTokens == 4 && active[0].ageMilliseconds == 250.0,
          "a prefilling request's progress or age is wrong");
  *commandsReady = true;
  runUntilIdle(loop);
  require(loop.statusSnapshot().activeRequests.empty(), "a finished request is still reported");
}

void testWireLifecycleAndCacheHit() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  Executor &executor = fixture.executor;
  fixture.clockStep = 0.25;

  loop.announceReady(protocol::peer::kTestReady);
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
  require(misses == 1 && hits == 2 && executor.restored == 128,
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
  loop.announceReady(protocol::peer::kTestReady);
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

// A request's cache salt reaches the engine: the same prompt under another
// salt starts cold and under its own salt resumes. A Done event reports the
// tokens a request read from blocks another request computed, and a salt no
// other request uses reads none.
void testCacheSaltAndDeduplicationCrossTheWire() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  fixture.clockStep = 0.25;
  loop.announceReady(protocol::peer::kTestReady);
  const auto send = [&](protocol::RequestFrame input) {
    require(loop.receive(protocol::peer::serialize(input)), "salted request wire failed");
    runUntilIdle(loop);
  };
  for (const auto &[id, salt] : {std::pair<uint64_t, uint64_t>{1, 7}, {2, 8}, {3, 7}}) {
    auto input = request(id);
    input.cacheSalt = {salt, 0};
    send(input);
  }
  // An unsalted longer prompt, then two with its first three blocks' tokens
  // and no state there: one unsalted, one under a salt no other request uses.
  auto longer = request(4, 8);
  longer.promptTokens.resize(129);
  for (uint32_t i = 0; i < longer.promptTokens.size(); ++i)
    longer.promptTokens[i] = 1000 + i;
  send(longer);
  for (const auto &[id, salt] : {std::pair<uint64_t, uint64_t>{5, 0}, {6, 99}}) {
    auto branch = longer;
    branch.requestId = id;
    branch.cacheSalt = {salt, 0};
    std::fill(branch.promptTokens.begin() + 100, branch.promptTokens.end(), 7);
    send(branch);
  }
  std::unordered_map<uint64_t, uint32_t> matched, deduplicated;
  for (const auto &message : fixture.events()) {
    if (const auto *start = std::get_if<protocol::StartEvent>(&message))
      matched[start->requestId] = start->matchedPromptTokens;
    if (const auto *done = std::get_if<protocol::DoneEvent>(&message))
      deduplicated[done->requestId] = done->deduplicatedTokens;
  }
  require(matched.at(1) == 0 && matched.at(2) == 0 && matched.at(3) > 0,
          "a request's cache salt did not reach the engine");
  require(matched.at(5) == 0 && deduplicated.at(5) > 0 && matched.at(6) == 0 &&
              deduplicated.at(6) == 0,
          "a Done event did not carry the tokens read from another request's blocks");
}

// A request's flags reach the model with the rest of its request.
void testRequestFlagsReachTheModel() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  Executor &executor = fixture.executor;
  fixture.clockStep = 0.25;
  loop.announceReady(protocol::peer::kTestReady);
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
  loop.announceReady(protocol::peer::kTestReady);
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
       {std::pair{std::vector<uint8_t>(protocol::kFrameHeaderBytes), "bad_magic"},
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

  loop.announceReady(protocol::peer::kTestReady);
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

  loop.announceReady(protocol::peer::kTestReady);
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
  // A command's progress (its host steps) moves its deadline a timeout on,
  // and another command's moves nothing.
  metal::CommandWatchdog progressing(120.0);
  progressing.start(3, 0.0);
  progressing.progress(3, 100.0);
  progressing.progress(2, 200.0);
  require(!progressing.expired(219.999) && progressing.expired(220.0),
          "a command's progress did not move its deadline, or another's did");

  for (bool gpuCompleted : {false, true}) {
    LoopFixture fixture;
    engine::NativeRuntime &loop = fixture.loop;
    Executor &executor = fixture.executor;
    double &now = fixture.monotonic;
    now = 0.0;
    const auto commandReady = executor.holdCommands();
    metal::CommandWatchdog watchdog(120.0);
    executor.submitObserver = [&] { watchdog.start(1, now / 1000.0); };
    executor.healthCheck = [&] {
      if (watchdog.expired(now / 1000.0))
        throw metal::MetalBackendError("test command completion timeout");
    };
    loop.announceReady(protocol::peer::kTestReady);
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
      *commandReady = true;
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
    loop.announceReady(protocol::peer::kTestReady);
    require(loop.receive(protocol::peer::serialize(request(1))) && loop.tick(),
            "live duplicate fixture did not start");
    require(!loop.receive(
                protocol::peer::serialize(request(1, malformed ? 2 : 1))) &&
                loop.connectionMustClose() && loop.statusSnapshot().submitted == 1,
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
  loop.announceReady(protocol::peer::kTestReady);
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
    loop.announceReady(protocol::peer::kTestReady);
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
// failure named and counted, and the engine's failure told once.
void testFrameFailureUsesExecutionBoundary() {
  enum class Thrown { Standard, Metal, Foreign };
  for (const Thrown thrown : {Thrown::Standard, Thrown::Metal, Thrown::Foreign}) {
    engine::NativeLoopConfig config;
    uint32_t told = 0;
    config.engineFailed = [&told] { ++told; };
    LoopFixture fixture(config, {.pages = 8});
    engine::NativeRuntime &loop = fixture.loop;
    fixture.status = [thrown]() -> std::string {
      if (thrown == Thrown::Standard)
        throw std::runtime_error("status test");
      if (thrown == Thrown::Metal)
        throw metal::MetalBackendError("status test");
      throw 42;
    };
    loop.announceReady(protocol::peer::kTestReady);
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
    require(errors == 1 && told == 1 &&
                loop.metricsSnapshot().metalFailures == (thrown == Thrown::Metal ? 1U : 0U),
            "a frame exception was reported, told or counted other than once");
  }
}

// The loop checks its protocol limits once, when it is built; the codec and
// the parser rely on them.
void testInvalidLimitsAreRejectedAtConstruction() {
  protocol::ProtocolLimits limits;
  limits.maxMaskWords = 0;
  rejects([&] { LoopFixture fixture({}, {.pages = 8}, limits); }, "limit_exceeded",
          "the loop accepted invalid protocol limits");
}

// The loop refuses a config without the weights the bootstrap always gives
// it.
void testLoopNeedsItsComponents() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 8);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  NativeLoopConfig config{.engine = test::engineConfig()};
  rejects([&] {
    engine::NativeRuntime(std::move(config), cache, executor, [](std::span<const uint8_t>) {},
                          protocol::ProtocolLimits{});
  }, "the native engine loop lacks a component it needs", "a loop without its weights was built");
}

// The loop refuses an empty status source.
void testStatusNeedsASource() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 8);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Weights weights;
  engine::NativeRuntime loop(
      {.engine = test::engineConfig(
           {.weights = WeightResidencyConfig{.weights = &weights, .admit = test::admitAll}})},
      cache, executor, [](std::span<const uint8_t>) {}, protocol::ProtocolLimits{});
  rejects([&] { loop.serveStatus({}); }, "no status source", "an empty status source was connected");
}

// An exception while the engine admits a request is engine-fatal: nothing
// below the engine rolls back, and the loop reports it once.
void testAdmissionExceptionStopsTheEngineOnce() {
  LoopFixture fixture;
  engine::NativeRuntime &loop = fixture.loop;
  fixture.executor.beginObserver = [] { throw std::runtime_error("begin failed"); };
  loop.announceReady(protocol::peer::kTestReady);
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

// A failed output write closes the connection without failing the engine:
// the transport decides whether the peer closed or the write failed.
void testOutputFailureClosesTheConnection() {
  uint32_t told = 0;
  engine::NativeLoopConfig config;
  config.engineFailed = [&told] { ++told; };
  LoopFixture fixture(config, {.pages = 8});
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady(protocol::peer::kTestReady);
  fixture.writeFailure = std::system_error(EPIPE, std::generic_category(),
                                           "write(native output)");
  require(!loop.receive(protocol::peer::serialize(protocol::StatusRequestFrame{77})) &&
              loop.connectionMustClose() && loop.engineHealthy() &&
              loop.engineFailure().empty() && !told,
          "a failed output write failed the engine");
}

// A Ready the connection cannot carry fails the start, which serve-native
// reports before it exits, instead of counting as announced.
void testReadyThatCannotBeSentThrows() {
  LoopFixture fixture({}, {.pages = 8});
  fixture.writeFailure = std::system_error(EPIPE, std::generic_category(),
                                           "write(native output)");
  try {
    fixture.loop.announceReady(protocol::peer::kTestReady);
  } catch (const std::runtime_error &) {
    return;
  }
  throw std::runtime_error("a Ready that was not sent counted as announced");
}

// Every path that stops the engine keeps its reason for the exit log, not
// only the ones that report through engineError().
void testEngineFailureNamesItsReason() {
  {
    LoopFixture fixture({}, {.pages = 8});
    engine::NativeRuntime &loop = fixture.loop;
    loop.announceReady(protocol::peer::kTestReady);
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
  loop.announceReady(protocol::peer::kTestReady);
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
              loop.statusSnapshot().submitted == 1,
          "invalid tokens reached admission or prevented subsequent completion");
}

// The loop announces the Ready its start built (engine::readyEventFor), every
// field as given, once.
void testAnnounceReadySendsTheReadyItIsGiven() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config);
  const protocol::ReadyEvent ready{4, 1000, true, 248'320, {248'044, 248'046}, 9};
  fixture.loop.announceReady(ready);
  const auto announced = fixture.events();
  require(announced.size() == 1 && announced.front() == protocol::EngineEvent{ready},
          "announceReady did not send exactly the Ready it was given");
  rejects([&] { fixture.loop.announceReady(ready); }, "ready was already announced",
          "a second Ready was announced");
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
    loop.announceReady(protocol::peer::kTestReady);
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
                loop.statusSnapshot().submitted == 1,
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
    uint32_t told = 0;
    config.engineFailed = [&told] { ++told; };
    protocol::ProtocolLimits limits;
    limits.maxTokenBatch = limit;
    LoopFixture fixture(config, {}, limits);
    engine::NativeRuntime &loop = fixture.loop;
    Executor &executor = fixture.executor;
    executor.decodeTokens = model::ExecutionLimits::maximumStepTokens;

    loop.announceReady(protocol::peer::kTestReady);
    require(loop.receive(
                protocol::peer::serialize(request(5, executor.decodeTokens))),
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
    if (limit >= executor.decodeTokens) {
      require(streamed == executor.decodeTokens && completion == streamed &&
                  !encodeErrors && loop.engineHealthy(),
              "a full step with its terminal anchor did not fit one event");
    } else {
      require(!streamed && !completion && encodeErrors == 1 && told == 1 &&
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
  loop.announceReady(protocol::peer::kTestReady);
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
  const auto commandReady = executor.holdCommands();
  loop.announceReady(protocol::peer::kTestReady);
  require(loop.receive(protocol::peer::serialize(scoreRequest(11, 65))),
          "cancel-score request failed");
  require(loop.tick() && loop.commandInFlight(), "score prefill was not held");
  protocol::CancelFrame cancel{11};
  require(loop.receive(protocol::peer::serialize(cancel)),
          "score cancel failed");
  *commandReady = true;
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

// A Stop frame ends a request that still has output budget left with one
// Done reporting Stop, counted completed.
void testStopFrameEndsTheRequestAsStopped() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  // The fake model never ends the request itself, so only the Stop can.
  fixture.executor.decodeFinishes = false;
  loop.announceReady(protocol::peer::kTestReady);
  constexpr uint32_t kOutputBudget = 512;
  require(loop.receive(protocol::peer::serialize(request(41, kOutputBudget))), "stop request wire failed");
  const auto emitted = [&] {
    return std::ranges::any_of(fixture.events(), [](const protocol::EngineEvent &message) {
      return std::holds_alternative<protocol::TokensEvent>(message);
    });
  };
  for (uint32_t step = 0; step < 32 && !emitted(); ++step)
    static_cast<void>(loop.tick());
  require(emitted(), "stop request emitted no token");

  require(loop.receive(protocol::peer::serialize(protocol::StopFrame{41})), "stop frame failed");
  runUntilIdle(loop);
  uint32_t doneCount = 0;
  for (const protocol::EngineEvent &message : fixture.events()) {
    if (const auto *done = std::get_if<protocol::DoneEvent>(&message)) {
      ++doneCount;
      require(done->requestId == 41 && done->reason == EngineFinishReason::Stop &&
                  done->completionTokens < kOutputBudget,
              "a stop frame did not end the request early as Stop");
    }
  }
  const EngineSnapshot stopped = loop.statusSnapshot();
  require(doneCount == 1 && stopped.completed == 1 && stopped.cancelled == 0,
          "a stopped request did not end with one Done, counted completed");
}

// A request's prefill time runs from when the engine starts it, even when
// its clock reads zero then, not from its arrival.
void testPrefillTimeCountsFromTheStart() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady(protocol::peer::kTestReady);
  fixture.monotonic = -2.0;
  require(loop.receive(protocol::peer::serialize(request(1))), "request wire failed");
  fixture.monotonic = 0.0;
  require(loop.tick() && loop.statusSnapshot().submitted == 1 && loop.commandInFlight(),
          "the request did not start");
  fixture.monotonic = 3.0;
  runUntilIdle(loop);
  std::optional<protocol::DoneEvent> done;
  for (const protocol::EngineEvent &message : fixture.events())
    if (const auto *event = std::get_if<protocol::DoneEvent>(&message))
      done = *event;
  require(done && done->prefillMicros == 3000 && done->wallMicros == 5000,
          "a request started at time zero counted its prefill from its arrival");
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
  loop.announceReady(protocol::peer::kTestReady);

  // Whole KV pages, so the last prompt chunk is the one that publishes the
  // final block: a chunk and a page, whose first command runs the chunk
  // alone. The chat arrives as it runs, and its 65 rows join the page that
  // is left (Scheduler::planPrefill), so the score's last step shares its
  // command. The chat lane deliberately does not share this prefix.
  auto scored = scoreRequest(
      21, model::ExecutionLimits::prefillTokenBudget + KvCache::pageTokens);
  for (uint32_t index = 0; index < scored.promptTokens.size(); ++index)
    scored.promptTokens[index] = 1000 + index;
  require(loop.receive(protocol::peer::serialize(scored)),
          "score request wire failed");
  for (uint32_t step = 0; step < 32 && !executor.prefillChunks.contains(21); ++step)
    static_cast<void>(loop.tick());
  require(executor.prefillChunks.contains(21), "the score's first chunk did not run");
  require(loop.receive(protocol::peer::serialize(request(22))),
          "batched chat request wire failed");
  runUntilIdle(loop);

  ScoreBesideChat result;
  result.publishedBlocks = fixture.cache.snapshot().pool.pagesPrefix;
  result.healthy = loop.engineHealthy() && !loop.connectionMustClose();
  result.slotsReleased = !executor.requests.contains(21) && !executor.requests.contains(22);
  result.scoreChunks = executor.prefillChunks[21];
  result.widestBatch = std::ranges::max(executor.prefillWidths);

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
// configured width, reaches the model, and the request goes on to its first
// cycle's verify mask. Any other response fails that request alone, and one
// that arrives after the request ended, cancelled or timed out waiting for
// it, is ignored.
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
    loop.announceReady(protocol::peer::kTestReady);
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
    if (reply == Reply::Valid) {
      // Drafting starts once the first token is selected: the first cycle
      // asks for its verify mask, a row for each draft token and the anchor.
      std::optional<protocol::MaskRequestEvent> verify;
      for (uint32_t step = 0; step < 32 && !verify; ++step) {
        static_cast<void>(loop.tick());
        for (const auto &message : fixture.events()) {
          const auto *event = std::get_if<protocol::MaskRequestEvent>(&message);
          if (event && event->maskRequestId != asked->maskRequestId)
            verify = *event;
        }
      }
      require(verify && verify->requestId == 7 && verify->wordsPerMask == 3 &&
                  verify->simulationTokens.size() == 8,
              "the first constrained cycle did not ask for its verify mask");
      protocol::MaskResponseFrame rows{7, verify->maskRequestId, {}};
      for (size_t row = 0; row <= verify->simulationTokens.size(); ++row)
        rows.maskWords.insert(rows.maskWords.end(), {1, 0, 0});
      send(rows);
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
    require(maskRequests == (reply == Reply::Valid ? 2U : 1U) && loop.engineHealthy() &&
                !loop.connectionMustClose() && !executor.requests.contains(7),
            "mask exchange stopped the engine or kept the request's slot");
    if (reply == Reply::Valid) {
      require(done == EngineFinishReason::Stop && errors.empty() &&
                  executor.providedMasks == 2,
              "valid masks did not let the request finish");
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

} // namespace

std::span<const test::TestCase> wireTests() {
  static constexpr test::TestCase kTests[] = {
      {"testWireLifecycleAndCacheHit", testWireLifecycleAndCacheHit},
      {"testSnapshotReportsLiveRequests", testSnapshotReportsLiveRequests},
      {"testGenerationPromptBoundsTheReplayState", testGenerationPromptBoundsTheReplayState},
      {"testCacheSaltAndDeduplicationCrossTheWire", testCacheSaltAndDeduplicationCrossTheWire},
      {"testRequestFlagsReachTheModel", testRequestFlagsReachTheModel},
      {"testSamplingReachesTheModel", testSamplingReachesTheModel},
      {"testPromptProgress", testPromptProgress},
      {"testCapacityFailureHasOneTerminalFrame", testCapacityFailureHasOneTerminalFrame},
      {"testFatalFramingClosesConnection", testFatalFramingClosesConnection},
      {"testInvalidLimitsAreRejectedAtConstruction", testInvalidLimitsAreRejectedAtConstruction},
      {"testLoopNeedsItsComponents", testLoopNeedsItsComponents},
      {"testStatusNeedsASource", testStatusNeedsASource},
      {"testRequestErrorKeepsFraming", testRequestErrorKeepsFraming},
      {"testCommandWatchdogAndPendingHealthWake", testCommandWatchdogAndPendingHealthWake},
      {"testDuplicateLiveRequestClosesWithoutAmbiguousError", testDuplicateLiveRequestClosesWithoutAmbiguousError},
      {"testCancelledIdIsReusableInTheSameInput", testCancelledIdIsReusableInTheSameInput},
      {"testControlFailureUsesExecutionBoundary", testControlFailureUsesExecutionBoundary},
      {"testAdmissionExceptionStopsTheEngineOnce", testAdmissionExceptionStopsTheEngineOnce},
      {"testFrameFailureUsesExecutionBoundary", testFrameFailureUsesExecutionBoundary},
      {"testOutputFailureClosesTheConnection", testOutputFailureClosesTheConnection},
      {"testReadyThatCannotBeSentThrows", testReadyThatCannotBeSentThrows},
      {"testEngineFailureNamesItsReason", testEngineFailureNamesItsReason},
      {"testOutOfVocabularyTokensStayRequestScoped", testOutOfVocabularyTokensStayRequestScoped},
      {"testAnnounceReadySendsTheReadyItIsGiven", testAnnounceReadySendsTheReadyItIsGiven},
      {"testImageRequestBeyondVisionStaysRequestScoped", testImageRequestBeyondVisionStaysRequestScoped},
      {"testStepTokensFitTheWire", testStepTokensFitTheWire},
      {"testScoreRequestCompletesAfterFullPrompt", testScoreRequestCompletesAfterFullPrompt},
      {"testCancelledScoreReturnsEmptyLogits", testCancelledScoreReturnsEmptyLogits},
      {"testStopFrameEndsTheRequestAsStopped", testStopFrameEndsTheRequestAsStopped},
      {"testPrefillTimeCountsFromTheStart", testPrefillTimeCountsFromTheStart},
      {"testInvalidScoreFailsOneRequestAndKeepsTheBatch", testInvalidScoreFailsOneRequestAndKeepsTheBatch},
      {"testConstrainedMaskExchange", testConstrainedMaskExchange},
  };
  return kTests;
}
