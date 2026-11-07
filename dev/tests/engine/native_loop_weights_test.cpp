#include "ClaimScope.hpp"
#include "LoopFixture.hpp"
#include "ProtocolPeer.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestKvTier.hpp"
#include "TestMetalStatistics.hpp"
#include "TestRegistry.hpp"
#include "TestStderr.hpp"
#include "engine/MemoryControl.hpp"
#include "engine/NativeRuntime.hpp"
#include "engine/ReleasableMemory.hpp"
#include "engine/WriteBehind.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::answeredStatus;
using splash::test::Executor;
using splash::test::hostMemory;
using splash::test::idle;
using splash::test::LoopFixture;
using splash::test::rejects;
using splash::test::releaseAnswer;
using splash::test::request;
using splash::test::require;
using splash::test::runUntilIdle;
using splash::test::Weights;

// The error event the loop wrote for request `id`, if any.
std::optional<protocol::ErrorEvent> errorOf(const LoopFixture &fixture, uint64_t id) {
  for (const auto &message : fixture.events())
    if (const auto *error = std::get_if<protocol::ErrorEvent>(&message); error && error->requestId == id)
      return *error;
  return std::nullopt;
}

// Admits every allocation, or refuses each for want of host memory while
// `refuse` is set.
metal::AllocationAdmission refusing(const bool &refuse) {
  return [&refuse](uint64_t, const std::function<void()> &allocate) -> metal::AllocationResult {
    if (refuse)
      return metal::AllocationFailure::HostPressure;
    allocate();
    return {};
  };
}

// The control pass between commands releases the weights once the engine has
// held no request for the idle release, counted from Ready or from the end of
// the last request however long it ran, never while one is there. A request
// waits while they are written back, an image per tick, so the loop answers
// frames between them, and the images all come back even when it is cancelled,
// while memory admits them (testRefusedRestoreWaitsForMemory). A restore that
// fails other than for memory stops the engine. What the engine gives back
// while idle without a Neural Engine split, ReleasableMemory over the images
// alone, is the images: the test runs through it when `releasable`.
void testIdleWeightsAreReleasedAndRestored(bool releasable) {
  constexpr double kIdleReleaseSeconds = 2.0;
  constexpr double idleRelease = 1000.0 * kIdleReleaseSeconds;
  Weights weights;
  ReleasableMemory memory({&weights}, std::nullopt);
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.engine.weights = WeightResidencyConfig{
      .weights = releasable ? static_cast<model::WeightMemory *>(&memory) : &weights};
  LoopFixture fixture(config, {}, {}, kIdleReleaseSeconds);
  engine::NativeRuntime &loop = fixture.loop;
  fixture.executor.beginObserver = [&] {
    require(!weights.released(), "the model began a request on released weights");
  };
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  MemoryGovernor governor(backend, 40ULL << 30, 2ULL << 30, 2ULL << 30,
                          [] { return hostMemory(64ULL << 30); }, 0);
  MemoryControl control(governor, backend, loop);
  // The pass runs as the transport runs it, at a command-free point: one
  // asked while a command is in flight waits for its end (FdTransport).
  const auto idlePass = [&](double elapsed) {
    fixture.monotonic += elapsed;
    if (!loop.commandInFlight())
      static_cast<void>(control.run(MemoryPressure::Normal));
    return weights.released();
  };
  // Warmup before Ready is not idleness.
  fixture.monotonic += 10 * idleRelease;
  loop.announceReady(protocol::peer::kTestReady);
  require(!idlePass(idleRelease - 1), "weights were released before the idle release passed");
  require(idlePass(1), "idle weights were kept past the idle release");
  require(loop.weightsSnapshot().released && loop.weightsSnapshot().restores == 0 &&
              loop.weightsSnapshot().idleReleaseSeconds == kIdleReleaseSeconds,
          "the weights' snapshot did not report their release");

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
  require(!loop.weightsSnapshot().released && loop.weightsSnapshot().restores == 1,
          "the weights' snapshot did not count one restore");
  require(!idlePass(2 * idleRelease), "weights were released under a request");
  runUntilIdle(loop);

  require(loop.receive(protocol::peer::serialize(request(2))), "a second request failed");
  // It runs past the idle release, without a control pass between commands.
  // The clock moves before each tick alone: the snapshot idle() checks reads
  // it too, for the requests' ages.
  for (uint32_t step = 0; step < 32 && !idle(loop); ++step) {
    fixture.monotonic += idleRelease;
    static_cast<void>(loop.tick());
  }
  require(idle(loop) && !idlePass(0), "weights were released as soon as a long request ended");
  require(idlePass(idleRelease), "weights were kept past the idle release after a request");

  require(loop.receive(protocol::peer::serialize(request(3))) && loop.tick() &&
              loop.receive(protocol::peer::serialize(protocol::CancelFrame{3})),
          "a request cancelled while the weights came back failed");
  while (weights.released())
    require(loop.tick(), "a cancelled request stopped its weights coming back");
  require(weights.restores == 2 * Weights::kImages, "an image was written back twice");
  // The transport ticks again after a tick that progressed: the engine then
  // lets the cancelled request go.
  static_cast<void>(loop.tick());

  require(idlePass(idleRelease), "weights were kept past the idle release after a cancel");
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

// A store of streamed experts (model::ExpertStore), the part after the images,
// goes back with them and comes back in one restore after their last; a
// release during a restore gives back what came back.
void testReleasableMemoryRestoresTheExpertsAfterTheImages() {
  Weights weights;
  struct Experts final : model::WeightMemory {
    bool held = true;
    uint32_t restores = 0;
    [[nodiscard]] bool released() const noexcept override { return !held; }
    void release() override { held = false; }
    bool restore(const metal::AllocationAdmission &) override {
      require(!held, "a held arena was restored");
      held = true;
      ++restores;
      return true;
    }
  } experts;
  ReleasableMemory memory({&weights, &experts}, std::nullopt);
  const metal::AllocationAdmission admit = [](uint64_t, const std::function<void()> &operation) {
    operation();
    return metal::AllocationResult{};
  };
  memory.release();
  require(memory.released() && weights.released() && experts.released(),
          "the release kept the images or the arena");
  for (uint32_t image = 0; image < Weights::kImages; ++image)
    require(!memory.restore(admit) && experts.released(),
            "the arena came back before the last image");
  require(!weights.released() && memory.released(), "the restore ended without the arena");
  require(memory.restore(admit) && !memory.released() && experts.restores == 1,
          "the arena did not come back in one restore after the images");
  memory.release();
  require(!memory.restore(admit) && weights.held() == 1,
          "a restore did not write the first image back");
  memory.release();
  require(weights.held() == 0 && experts.released() && experts.restores == 1,
          "a release during a restore kept what came back");
}

// With the Neural Engine split, the idle release gives back the images, then
// the split's program. A request waits while the images are written back, an
// image per tick, and the program is loaded in one tick more, before the
// request begins. A program that does not load again stops the split, never
// the engine (ops::AneFfn::restore): the request runs. The stopped split gives
// back what a load that ran late may have left once, at the next idle
// release, and loads nothing, so that the images alone come back after that.
void testIdleReleaseRestoresTheSplitLast() {
  constexpr double kIdleReleaseSeconds = 2.0;
  constexpr double idleRelease = 1000.0 * kIdleReleaseSeconds;
  Weights weights;
  // The split's program: whether it is loaded, whether the split released
  // it, the times it was loaded again, and whether the split stopped, as one
  // whose program does not load again (`failLoad`) does.
  bool loaded = true, released = false, stopped = false, failLoad = false;
  uint32_t loads = 0;
  ReleasableMemory memory(
      {&weights}, ReleasableMemory::Split{
                   [&] {
                     require(weights.released(),
                             "the split's program was unloaded before the images were released");
                     if (released)
                       return false;
                     released = true;
                     loaded = false;
                     return true;
                   },
                   [&] {
                     require(!weights.released() && !loaded,
                             "the split's program was loaded before every image was written back");
                     if (!released || stopped)
                       return;
                     released = false;
                     ++loads;
                     if (failLoad)
                       stopped = true;
                     else
                       loaded = true;
                   }});
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.engine.weights = WeightResidencyConfig{.weights = &memory};
  LoopFixture fixture(config, {}, {}, kIdleReleaseSeconds);
  engine::NativeRuntime &loop = fixture.loop;
  fixture.executor.beginObserver = [&] {
    require(!weights.released() && (loaded || stopped),
            "the model began a request before the split's program was back");
  };
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  MemoryGovernor governor(backend, 40ULL << 30, 2ULL << 30, 2ULL << 30,
                          [] { return hostMemory(64ULL << 30); }, 0);
  MemoryControl control(governor, backend, loop);
  const auto idlePass = [&] {
    fixture.monotonic += idleRelease;
    static_cast<void>(control.run(MemoryPressure::Normal));
    return memory.released();
  };
  // Sends request `id`, which ends the idle, and ticks once per image, the
  // request waiting: true while it still waits after the last image.
  const auto imagesBack = [&](uint64_t id) {
    require(loop.receive(protocol::peer::serialize(request(id))), "a request was refused");
    for (uint32_t image = 0; image < Weights::kImages; ++image)
      require(loop.tick() && loop.statusSnapshot().completed == id - 1,
              "a request ran before every image was written back");
    return loop.weightsSnapshot().restores == id - 1;
  };
  loop.announceReady(protocol::peer::kTestReady);

  require(idlePass() && !loaded, "the idle release kept the split's program");
  require(imagesBack(1) && !loaded && !memory.released(),
          "the restore ended with the last image, before the split's program");
  // The weights are no longer released, but the restore is still under way: a
  // release request leaves it alone, and its answer reports it.
  const WeightsSnapshot answer = releaseAnswer(fixture, 1);
  require(!answer.released && answer.restoring && !memory.released() && !loaded,
          "a release request before the split's program was back was answered as a release");
  require(loop.tick() && loaded && loads == 1 && loop.weightsSnapshot().restores == 1,
          "the tick after the last image did not load the split's program");
  runUntilIdle(loop);
  require(loop.statusSnapshot().completed == 1,
          "the request did not run once the program was back");

  failLoad = true;
  require(idlePass() && !loaded, "the idle release kept the split's program");
  require(imagesBack(2) && loop.tick() && stopped && loads == 2 &&
              loop.weightsSnapshot().restores == 2 && loop.engineHealthy(),
          "a program that did not load again failed the restore");
  runUntilIdle(loop);
  require(loop.statusSnapshot().completed == 2, "the request did not run after the split stopped");

  require(idlePass() && released, "the stopped split kept what its program may hold");
  require(imagesBack(3) && loop.tick() && loads == 2 && loop.weightsSnapshot().restores == 3,
          "a stopped split's program was loaded again");
  runUntilIdle(loop);
  require(loop.statusSnapshot().completed == 3 && loop.engineHealthy(),
          "a request after the split stopped did not run");
  require(idlePass(), "the idle release kept the images once the split stopped");
  require(!imagesBack(4) && loads == 2 && loop.weightsSnapshot().restores == 4,
          "the images did not come back alone once the stopped split had given its program back");
  runUntilIdle(loop);
  require(loop.statusSnapshot().completed == 4, "a later request did not run");
}

// A restore takes the weights' memory under the governor's admission as
// memory the requests waiting need: the host's margins do not refuse it, and
// critical pressure does. A refused restore keeps the parts that came back
// and makes no progress; the loop tries again after the retry backoff, and
// every request waits for memory, as their resource wait reports, for the
// resource wait limit from the first refusal after the last part that came
// back. Memory that comes back in time lets the restore go on. Once the limit
// passes, every request waiting fails retryably and the weights give back the
// parts that came back, the split's program still released for the next
// restore to load last; the engine stays up, and the next request starts
// over, its wait counted afresh. A refused restore that no request waits for
// any more takes no memory past the host's margins: it gives up at once and
// counts no failure.
void testRefusedRestoreWaitsForMemory() {
  constexpr double kIdleReleaseSeconds = 2.0;
  constexpr double kWaitLimit = 400.0;
  constexpr uint64_t kReserve = 2ULL << 30;
  constexpr uint64_t kHostMemory = 64ULL << 30;
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  uint64_t hostAvailable = kHostMemory;
  MemoryGovernor governor(backend, 40ULL << 30, kReserve, kReserve,
                          [&] { return hostMemory(hostAvailable); }, 0);
  Weights weights;
  uint32_t programReleases = 0, programLoads = 0;
  ReleasableMemory memory({&weights}, ReleasableMemory::Split{[&] {
                                                             ++programReleases;
                                                             return true;
                                                           },
                                                           [&] { ++programLoads; }});
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.engine.resourceWaitTimeoutMilliseconds = kWaitLimit;
  engine::connectToGovernor(config.engine, governor);
  config.engine.weights =
      WeightResidencyConfig{.weights = &memory, .admit = governor.allocationAdmission()};
  LoopFixture fixture(config, {}, {}, kIdleReleaseSeconds);
  engine::NativeRuntime &loop = fixture.loop;
  MemoryControl control(governor, backend, loop);
  const auto pass = [&](MemoryPressure pressure, double elapsed = 0.0) {
    fixture.monotonic += elapsed;
    static_cast<void>(control.run(pressure));
  };
  const auto errorOf = [&](uint64_t id) { return ::errorOf(fixture, id); };
  loop.announceReady(protocol::peer::kTestReady);

  pass(MemoryPressure::Normal, 1000.0 * kIdleReleaseSeconds);
  require(memory.released() && programReleases == 1, "idle weights were kept");
  hostAvailable = kReserve + (512ULL << 20);
  {
    // Memory no request needs is a Start claim's, as the restore's is with
    // no request waiting.
    const test::ClaimScope start(governor, ClaimClass::Start);
    require(governor.allocationAdmission()(Weights::kImageBytes, [] {}).failure ==
                metal::AllocationFailure::HostPressure,
            "the host's margin admitted memory no request needs");
  }
  require(loop.receive(protocol::peer::serialize(request(1))) && loop.tick() &&
              weights.held() == 1,
          "the host's margin held back weights a request waits for");

  pass(MemoryPressure::Critical);
  require(!loop.tick() && weights.held() == 1 && weights.restores == 1,
          "critical pressure admitted the weights' memory");
  require(loop.millisecondsUntilNextWakeup() == engine::kResourceRetryBackoffMilliseconds &&
              loop.resourceWaitSnapshot().memory == 1,
          "the request behind a refused restore did not wait for memory");
  fixture.monotonic += kWaitLimit - 1.0;
  require(!loop.tick() && weights.held() == 1 && !errorOf(1) && loop.engineHealthy() &&
              loop.resourceWaitSnapshot().oldestWaitMilliseconds == kWaitLimit - 1.0,
          "a refused restore gave up before the resource wait limit");
  hostAvailable = kHostMemory;
  pass(MemoryPressure::Normal);
  runUntilIdle(loop);
  require(loop.statusSnapshot().completed == 1 && weights.restores == Weights::kImages &&
              programLoads == 1 && loop.weightsSnapshot().restores == 1,
          "the restore did not go on once memory came back");

  pass(MemoryPressure::Normal, 1000.0 * kIdleReleaseSeconds);
  require(memory.released() && programReleases == 2 &&
              loop.receive(protocol::peer::serialize(request(2))) && loop.tick() &&
              weights.held() == 1,
          "the second restore did not write back its first image");
  pass(MemoryPressure::Critical);
  require(!loop.tick(), "critical pressure admitted the second image");
  require(loop.receive(protocol::peer::serialize(request(4))) && !loop.tick() &&
              loop.resourceWaitSnapshot().memory == 2,
          "a request that came during a refused restore did not wait for memory");
  fixture.monotonic += 0.75 * kWaitLimit;
  pass(MemoryPressure::Normal);
  require(loop.tick() && weights.held() == 2, "the second image did not come back");
  pass(MemoryPressure::Critical);
  require(!loop.tick(), "critical pressure admitted the third image");
  fixture.monotonic += 0.75 * kWaitLimit;
  require(!loop.tick() && weights.held() == 2 && !errorOf(2) && !errorOf(4),
          "the resource wait limit did not restart once a part came back");
  fixture.monotonic += 0.25 * kWaitLimit;
  require(loop.tick(), "a restore past the resource wait limit made no progress");
  for (uint64_t id : {2, 4}) {
    const std::optional<protocol::ErrorEvent> error = errorOf(id);
    require(error && error->failureClass == protocol::FailureClass::RequestError &&
                error->code == "resource_timeout" && error->retryable &&
                error->message.find("macOS is short of memory") != std::string::npos,
            "a request behind a restore that gave up did not fail retryably");
  }
  require(memory.released() && weights.held() == 0 && programReleases == 2 &&
              loop.weightsSnapshot().restoreFailures == 1 &&
              loop.resourceWaitSnapshot().memory == 0 && loop.engineHealthy() &&
              !loop.connectionMustClose(),
          "a restore that gave up kept what came back, released the program again or stopped "
          "the engine");

  pass(MemoryPressure::Normal);
  static_cast<void>(loop.tick());
  require(weights.held() == 0, "a restore went on after it gave up");

  pass(MemoryPressure::Critical);
  require(loop.receive(protocol::peer::serialize(request(3))) && !loop.tick() && !errorOf(3) &&
              loop.resourceWaitSnapshot().memory == 1,
          "the next request did not wait for memory afresh");
  pass(MemoryPressure::Normal);
  runUntilIdle(loop);
  require(loop.statusSnapshot().completed == 2 && weights.restores == 2 * Weights::kImages + 2 &&
              programLoads == 2 && loop.weightsSnapshot().restores == 2,
          "the next request did not restore the weights from the first image");

  pass(MemoryPressure::Normal, 1000.0 * kIdleReleaseSeconds);
  require(memory.released() && loop.receive(protocol::peer::serialize(request(5))) &&
              loop.tick() && weights.held() == 1,
          "the last restore did not write back its first image");
  pass(MemoryPressure::Critical);
  require(!loop.tick() && loop.receive(protocol::peer::serialize(protocol::CancelFrame{5})),
          "a refused restore's request could not be cancelled");
  hostAvailable = kReserve + (512ULL << 20);
  pass(MemoryPressure::Normal);
  require(loop.tick() && memory.released() && weights.held() == 0 &&
              loop.weightsSnapshot().restoreFailures == 1 &&
              !loop.millisecondsUntilNextWakeup(),
          "a refused restore no request waits for took memory past the host's margin, went "
          "on, or counted a failure");
}

// With --idle-release off the control pass never releases the weights,
// however long the engine holds no request, but a release request still does,
// and the next request restores them.
void testIdleReleaseOffReleasesOnlyOnRequest() {
  Weights weights;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.engine.weights = WeightResidencyConfig{.weights = &weights};
  LoopFixture fixture(config, {}, {}, std::numeric_limits<double>::infinity());
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady(protocol::peer::kTestReady);
  fixture.monotonic += 1e12;
  loop.releaseIdleWeights();
  require(!weights.released(), "weights were released with the idle release off");
  const WeightsSnapshot answer = releaseAnswer(fixture, 5);
  require(weights.released() && answer.released && !answer.restoring &&
              answer.idleReleaseSeconds == std::numeric_limits<double>::infinity(),
          "a release request did not release the weights with the idle release off");
  require(loop.receive(protocol::peer::serialize(request(1))), "the request after the release failed");
  runUntilIdle(loop);
  require(!weights.released() && weights.restores == Weights::kImages,
          "the request did not restore the weights released with the idle release off");
  // An idle release that is not positive is refused.
  for (double seconds : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()}) {
    rejects([&] { LoopFixture refusedFixture(config, {}, {}, seconds); },
            "the idle release must be positive",
            "an idle release that is not positive was accepted");
  }
}

// A release request gives the weights back now, as the idle release does, when
// the engine holds no request and no restore is under way, and otherwise leaves
// them as they are. Either way its answer reports what the engine did: released
// and not restoring only once the weights are given back; a restore under way,
// which leaves them released until its last image, as restoring. The next
// request restores them as after the idle release.
void testReleaseRequestGivesTheWeightsBackWhenIdle() {
  Weights weights;
  ReleasableMemory memory({&weights}, std::nullopt);
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.engine.weights = WeightResidencyConfig{.weights = &memory};
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady(protocol::peer::kTestReady);
  require(!weights.released(), "the weights were released before any release request");

  WeightsSnapshot answer = releaseAnswer(fixture, 5);
  require(weights.released() && answer.released && !answer.restoring && answer.restores == 0,
          "an idle engine did not release its weights on request");
  // A second release request changes nothing, and is answered as the first.
  answer = releaseAnswer(fixture, 6);
  require(weights.released() && answer.released && !answer.restoring,
          "releasing released weights again failed");

  // A request restores them, an image per tick, from its arrival. A release
  // request during the restore leaves it alone, and its answer reports the
  // restore.
  require(loop.receive(protocol::peer::serialize(request(1))) && loop.weightsSnapshot().restoring,
          "a request that found the weights released did not start their restore");
  require(loop.tick() && weights.held() == 1, "the request did not start the restore");
  answer = releaseAnswer(fixture, 7);
  require(weights.held() == 1 && answer.released && answer.restoring,
          "a release request during the restore released the weights, or its answer did not "
          "report the restore");
  while (weights.released())
    require(loop.tick(), "a tick did not write back an image");
  require(weights.restores == Weights::kImages, "the restore was disturbed by a release request");

  // With a request in service the weights stay.
  require(loop.receive(protocol::peer::serialize(request(2))), "a second request failed");
  answer = releaseAnswer(fixture, 8);
  require(!weights.released() && !answer.released && !answer.restoring,
          "the weights were released under a request");
  runUntilIdle(loop);
  answer = releaseAnswer(fixture, 9);
  require(weights.released() && answer.released && !answer.restoring,
          "the engine did not release once its request ended");
}

// The loop logs each release once, as the release it was: the idle release,
// after its time without a request, or a release request.
void testEachReleaseLogsOnce() {
  constexpr double kIdleReleaseSeconds = 2.0;
  Weights weights;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.engine.weights = WeightResidencyConfig{.weights = &weights};
  LoopFixture fixture(config, {}, {}, kIdleReleaseSeconds);
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady(protocol::peer::kTestReady);
  // The lines of `log` that report a release.
  const auto releases = [](const std::string &log) {
    std::vector<std::string> lines;
    std::istringstream stream(log);
    for (std::string line; std::getline(stream, line);)
      if (line.find("Weights released") != std::string::npos)
        lines.push_back(line);
    return lines;
  };
  fixture.monotonic += 1000.0 * kIdleReleaseSeconds;
  const auto idle = releases(test::capturedStderr([&] { loop.releaseIdleWeights(); }));
  require(weights.released() && idle.size() == 1 && idle[0].find("without a request") != std::string::npos,
          "the idle release did not log itself once");
  require(loop.receive(protocol::peer::serialize(request(1))), "the request after the idle release failed");
  runUntilIdle(loop);
  const auto asked = releases(test::capturedStderr([&] { static_cast<void>(releaseAnswer(fixture, 5)); }));
  require(weights.released() && asked.size() == 1 && asked[0].find("on request") != std::string::npos,
          "a release request did not log itself once");
}

// The loop reports that it holds requests from the first it takes to the end
// of its last, the span the process keeps the Mac from idle sleep for.
void testHoldingRequestsSpansFirstToLastRequest() {
  std::vector<bool> reported;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.holdingRequests = [&](bool held) { reported.push_back(held); };
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady(protocol::peer::kTestReady);
  require(reported.empty(), "a loop without requests reported holding one");
  require(loop.receive(protocol::peer::serialize(request(1))) &&
              loop.receive(protocol::peer::serialize(request(2))),
          "two requests failed");
  require(reported == std::vector<bool>{true}, "two requests did not share one span");
  runUntilIdle(loop);
  require(reported == std::vector<bool>{true, false},
          "the span did not end with the last request");
  // A cancelled request completes; a request whose deadline passes fails.
  require(loop.receive(protocol::peer::serialize(request(3))) && loop.tick() &&
              loop.receive(protocol::peer::serialize(protocol::CancelFrame{3})),
          "a cancelled request failed");
  runUntilIdle(loop);
  require(reported == std::vector<bool>{true, false, true, false},
          "a cancelled request did not end its span");
  require(loop.receive(protocol::peer::serialize(request(4, 64))), "request 4 failed");
  fixture.monotonic += 5000.0;
  runUntilIdle(loop);
  require(reported == std::vector<bool>{true, false, true, false, true, false},
          "a request that failed did not end its span");
  // One refused on arrival is never held.
  protocol::RequestFrame expired = request(5);
  expired.absoluteDeadlineUnixMicros = 1'000'000;
  require(loop.receive(protocol::peer::serialize(expired)) && reported.size() == 6,
          "a request refused on arrival was held");
}

// A request whose deadline passes while a refused restore keeps the weights
// released ends at its deadline, DeadlineExceeded, not at the resource wait
// limit: the loop wakes for it. The restore no request waits for any more
// then gives back what came back and counts no failure.
void testDeadlineEndsARequestWaitingForTheWeights() {
  bool refuse = false;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.engine.weights = WeightResidencyConfig{.admit = refusing(refuse)};
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady(protocol::peer::kTestReady);
  require(!releaseAnswer(fixture, 1).restoring && fixture.weights.released(),
          "the weights were not released");
  constexpr double kDeadline = 200.0;
  protocol::RequestFrame shortLived = request(1);
  shortLived.remainingDeadlineMicros = static_cast<uint64_t>(1000.0 * kDeadline);
  require(loop.receive(protocol::peer::serialize(shortLived)) && loop.tick() &&
              fixture.weights.held() == 1,
          "the restore did not write back its first image");
  refuse = true;
  require(!loop.tick() && loop.resourceWaitSnapshot().memory == 1,
          "the request behind a refused restore did not wait for memory");
  fixture.monotonic += kDeadline - 1.0;
  require(!loop.tick() && !errorOf(fixture, 1) && loop.millisecondsUntilNextWakeup() == 1.0,
          "the loop did not wake for the deadline of a request waiting for the weights");
  fixture.monotonic += 1.0;
  require(loop.tick(), "a request whose deadline passed during a refused restore made no progress");
  const std::optional<protocol::ErrorEvent> error = errorOf(fixture, 1);
  require(error && error->code == "deadline_exceeded" && !error->retryable,
          "a request whose deadline passed during a refused restore did not end at its deadline");
  require(fixture.weights.released() && fixture.weights.held() == 0 &&
              loop.weightsSnapshot().restoreFailures == 0 && !loop.weightsSnapshot().restoring &&
              loop.resourceWaitSnapshot().memory == 0,
          "the restore no request waited for kept what came back or counted a failure");
}

// While the weights are being restored the engine goes on with the work that
// needs none: a transfer that lands is collected at the next tick, and the
// copies of a replay point written behind go on, while the request waits for
// memory the restore is refused.
void testTransfersLandWhileTheWeightsAreRestored() {
  test::TestKvStorage storage{72, 4096, 4};
  KvPool pool{storage, 0};
  test::TestKvTier tier(true);
  tier.transferLimit = 64;
  tier.capacity = 128;
  engine::Cache cache(pool, &tier, nullptr);
  Executor executor;
  executor.keptTier = std::make_shared<test::OffloadControl>();
  executor.keptTier->ready = true;
  Weights weights;
  bool refuse = false;
  double now = 100.0;
  std::vector<uint8_t> output;
  engine::NativeRuntime loop(
      {.engine = test::engineConfig(
           {.maxContext = 102400,
            .weights = WeightResidencyConfig{.weights = &weights, .admit = refusing(refuse)}}),
       .unixMicros = [] { return uint64_t{1'000'000}; },
       .monotonicMilliseconds = [&] { return now; }},
      cache, executor,
      [&](std::span<const uint8_t> bytes) { output.insert(output.end(), bytes.begin(), bytes.end()); },
      protocol::ProtocolLimits{});
  loop.serveStatus(test::readyStatusJson);
  storage.commandInFlight = [&] { return loop.commandInFlight(); };
  loop.announceReady(protocol::peer::kTestReady);
  // A conversation leaves its replay point, whose copies start once it has
  // waited the write-behind delay.
  protocol::RequestFrame conversation = request(1);
  conversation.promptTokens.assign(2060, 7);
  require(loop.receive(protocol::peer::serialize(conversation)), "the conversation was refused");
  runUntilIdle(loop);
  now += WriteBehind::kDelayMilliseconds;
  static_cast<void>(loop.tick());
  const uint32_t started = tier.demotions;
  require(started && tier.inFlight() == started && !cache.snapshot().kvTier.copies,
          "the replay point's copies did not start");
  require(loop.receive(protocol::peer::serialize(protocol::ReleaseRequestFrame{1})) && weights.released(),
          "the weights were not released");
  refuse = true;
  require(loop.receive(protocol::peer::serialize(request(2))), "the request was refused");
  static_cast<void>(loop.tick());
  require(weights.held() == 0 && loop.resourceWaitSnapshot().memory == 1 && tier.demotions == started &&
              !cache.snapshot().kvTier.copies,
          "the request did not wait for the refused restore");
  tier.complete();
  static_cast<void>(loop.tick());
  require(cache.snapshot().kvTier.copies == started && tier.demotions > started,
          "a transfer that landed during a restore was not collected, or the copies behind it stopped");
  require(weights.released() && !executor.requests.contains(2) && loop.resourceWaitSnapshot().memory == 1,
          "the request ran before the weights were back");
}

// Each request waiting for the weights waits for memory on its own clock, as
// a start refused memory does: the engine's resource wait counts it from its
// first refusal, and one that came later fails at its own limit, not at the
// first one's, and gets the weights once memory comes back.
void testRequestsWaitForTheWeightsEachOnItsOwnClock() {
  constexpr double kWaitLimit = 400.0;
  bool refuse = false;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.engine.resourceWaitTimeoutMilliseconds = kWaitLimit;
  config.engine.weights = WeightResidencyConfig{.admit = refusing(refuse)};
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady(protocol::peer::kTestReady);
  require(!releaseAnswer(fixture, 1).restoring && fixture.weights.released(),
          "the weights were not released");
  refuse = true;
  require(loop.receive(protocol::peer::serialize(request(1))) && !loop.tick() &&
              loop.resourceWaitSnapshot().memory == 1,
          "the first request did not wait for memory");
  fixture.monotonic += kWaitLimit / 2;
  require(loop.receive(protocol::peer::serialize(request(2))) && !loop.tick(),
          "the second request did not wait for memory");
  const ResourceWaitSnapshot both = loop.resourceWaitSnapshot();
  require(both.memory == 2 && both.oldestWaitMilliseconds == kWaitLimit / 2,
          "the engine's resource wait did not count the requests waiting for the weights");
  fixture.monotonic += kWaitLimit / 2;
  require(loop.tick(), "the first request's limit passed without progress");
  const std::optional<protocol::ErrorEvent> first = errorOf(fixture, 1);
  require(first && first->code == "resource_timeout" && first->retryable,
          "the first request did not fail at its resource wait limit");
  require(!errorOf(fixture, 2) && loop.resourceWaitSnapshot().memory == 1 &&
              loop.resourceWaitSnapshot().oldestWaitMilliseconds == kWaitLimit / 2,
          "a request that came later failed at the first one's limit");
  refuse = false;
  runUntilIdle(loop);
  require(loop.statusSnapshot().completed == 1 && !errorOf(fixture, 2) &&
              loop.weightsSnapshot().restores == 1 && loop.weightsSnapshot().restoreFailures == 0,
          "the request on its own clock did not get the weights once memory came back");
}

// A request waiting for the weights fails at its resource wait limit only
// once the restore's attempt there is refused too: memory the host freed
// since the last attempt, which nothing in the engine signals, lets the
// restore go on, and the request runs.
void testRestoreTriesAgainAtTheLimit() {
  constexpr double kWaitLimit = 400.0;
  bool refuse = false;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.engine.resourceWaitTimeoutMilliseconds = kWaitLimit;
  config.engine.weights = WeightResidencyConfig{.admit = refusing(refuse)};
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady(protocol::peer::kTestReady);
  require(!releaseAnswer(fixture, 1).restoring && fixture.weights.released(),
          "the weights were not released");
  refuse = true;
  require(loop.receive(protocol::peer::serialize(request(1))) && !loop.tick() &&
              loop.resourceWaitSnapshot().memory == 1,
          "the request did not wait for memory");
  // The last attempt before the limit is refused; the host frees memory
  // after it.
  fixture.monotonic += kWaitLimit - 50.0;
  require(!loop.tick() && !errorOf(fixture, 1) && fixture.weights.held() == 0,
          "a refused restore gave up before the resource wait limit");
  refuse = false;
  fixture.monotonic += 50.0;
  require(loop.tick() && !errorOf(fixture, 1) && fixture.weights.held() == 1,
          "a request failed at its limit before the restore tried the memory that came back");
  runUntilIdle(loop);
  require(loop.statusSnapshot().completed == 1 && loop.weightsSnapshot().restores == 1 &&
              loop.weightsSnapshot().restoreFailures == 0,
          "the request did not get the weights that came back at its limit");
}

// A part of the weights that comes back ends the waits for memory of the
// requests waiting for them: a request refused memory at its start once the
// weights are back waits the resource wait limit from that refusal, however
// long the restore kept it waiting for the last part.
void testARestoredRequestWaitsAfreshAtItsStart() {
  constexpr double kWaitLimit = 400.0;
  bool refuseWeights = false, refuseStates = false;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.engine.resourceWaitTimeoutMilliseconds = kWaitLimit;
  config.engine.weights = WeightResidencyConfig{.admit = refusing(refuseWeights)};
  LoopFixture fixture(config, {}, {}, metal::kResidencyKeepAliveSeconds,
                      test::LoopOptions{.memory = test::FakeMemory::Config{}, .control = false});
  // A lane's state buffers, refused while refuseStates is set.
  fixture.executor.laneFootprintBytes = 64ULL << 20;
  fixture.executor.admission = refusing(refuseStates);
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady(protocol::peer::kTestReady);
  require(!releaseAnswer(fixture, 1).restoring && fixture.weights.released(),
          "the weights were not released");
  require(loop.receive(protocol::peer::serialize(request(1))) && loop.tick() && loop.tick() &&
              fixture.weights.held() == Weights::kImages - 1,
          "the restore did not write back the images before the last");
  refuseWeights = true;
  require(!loop.tick() && loop.resourceWaitSnapshot().memory == 1,
          "the request did not wait for the last image");
  fixture.monotonic += kWaitLimit - 100.0;
  require(!loop.tick() && !errorOf(fixture, 1), "a refused restore gave up before the resource wait limit");
  // The last image comes back; the request's start is refused memory.
  refuseWeights = false;
  refuseStates = true;
  require(loop.tick() && !fixture.weights.released(), "the last image did not come back");
  static_cast<void>(loop.tick());
  const ResourceWaitSnapshot wait = loop.resourceWaitSnapshot();
  require(wait.memory == 1 && wait.oldestWaitMilliseconds == 0.0 && !errorOf(fixture, 1),
          "the request refused memory at its start did not wait from that refusal");
  fixture.monotonic += kWaitLimit - 1.0;
  static_cast<void>(loop.tick());
  require(!errorOf(fixture, 1),
          "a request the restore kept waiting failed before the resource wait limit from its "
          "refusal at its start");
  fixture.monotonic += 1.0;
  require(loop.tick(), "the request's resource wait limit passed without progress");
  const std::optional<protocol::ErrorEvent> error = errorOf(fixture, 1);
  require(error && error->code == "resource_timeout" && error->retryable,
          "the request refused memory at its start did not fail at its limit");
}

// The expert store as weight memory: a release gives back its floor and
// the chunks above it, and a restore takes the floor back alone in one
// admission (`floors` counts those), as model::ExpertStore::restore does.
struct StoreMemory final : model::WeightMemory {
  [[nodiscard]] bool released() const noexcept override { return out; }
  void release() override { out = true; }
  bool restore(const metal::AllocationAdmission &admit) override {
    if (const metal::AllocationResult admitted = admit(kFloorBytes, [] {}); !admitted)
      throw metal::MetalAllocationError("the store's floor was refused", admitted.failure);
    ++floors;
    out = false;
    return true;
  }
  static constexpr uint64_t kFloorBytes = 1ULL << 28;
  bool out = false;
  uint32_t floors = 0;
};

// The expert store's idle restore takes its floor back, in one admission, as
// memory the request waiting needs: the request runs at once. The chunks
// above the floor come back as the ledger grows the store's pool
// (native_loop_memory_test's testRestoreBringsTheFloorThenGrowthTheRest).
void testIdleRestoreTakesTheFloorForTheRequest() {
  StoreMemory store;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.engine.weights = WeightResidencyConfig{.weights = &store};
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady(protocol::peer::kTestReady);
  require(!releaseAnswer(fixture, 1).restoring && store.released(), "the store was not released");
  require(loop.receive(protocol::peer::serialize(request(1))), "the request failed");
  runUntilIdle(loop);
  require(loop.statusSnapshot().completed == 1 && loop.resourceWaitSnapshot().memory == 0 &&
              store.floors == 1,
          "the restore did not take the floor back in one admission, or held the request");
}

// Under the governor's admissions (Bootstrap's), the restore takes back the
// store's floor alone, whatever the host spares, and the request runs: a
// request waits for it, so it is a Floor claim's, which the host's margin
// does not refuse.
void testRestoreTakesTheFloorWhateverTheHostSpares() {
  constexpr uint64_t chunk = 1ULL << 30;
  constexpr uint64_t hostReserve = 2ULL << 30;
  for (const bool ample : {false, true}) {
    StoreMemory store;
    test::metalStatistics() = {};
    metal::MetalBackend backend("unused");
    // Short of the host's margin, or past it by the whole store.
    const uint64_t spare = ample ? 3 * chunk + kHostMarginBytes : kHostMarginBytes - 4096;
    MemoryGovernor governor(backend, 40ULL << 30, hostReserve, hostReserve,
                            [&] { return hostMemory(hostReserve + spare); }, 0);
    engine::NativeLoopConfig config;
    config.engine.maxContext = 1024;
    engine::connectToGovernor(config.engine, governor);
    config.engine.weights =
        WeightResidencyConfig{.weights = &store, .admit = governor.allocationAdmission()};
    LoopFixture fixture(config);
    fixture.loop.announceReady(protocol::peer::kTestReady);
    require(!releaseAnswer(fixture, 1).restoring && store.released(), "the store was not released");
    require(fixture.loop.receive(protocol::peer::serialize(request(1))), "the request failed");
    runUntilIdle(fixture.loop);
    require(fixture.loop.statusSnapshot().completed == 1 && store.floors == 1,
            "the restore did not take the floor back in one admission");
  }
}

} // namespace

std::span<const test::TestCase> weightsTests() {
  static constexpr test::TestCase kTests[] = {
      {"testIdleWeightsAreReleasedAndRestored(false)", [] { testIdleWeightsAreReleasedAndRestored(false); }},
      {"testIdleWeightsAreReleasedAndRestored(true)", [] { testIdleWeightsAreReleasedAndRestored(true); }},
      {"testReleasableMemoryRestoresTheExpertsAfterTheImages", testReleasableMemoryRestoresTheExpertsAfterTheImages},
      {"testIdleReleaseRestoresTheSplitLast", testIdleReleaseRestoresTheSplitLast},
      {"testRefusedRestoreWaitsForMemory", testRefusedRestoreWaitsForMemory},
      {"testIdleReleaseOffReleasesOnlyOnRequest", testIdleReleaseOffReleasesOnlyOnRequest},
      {"testReleaseRequestGivesTheWeightsBackWhenIdle", testReleaseRequestGivesTheWeightsBackWhenIdle},
      {"testEachReleaseLogsOnce", testEachReleaseLogsOnce},
      {"testHoldingRequestsSpansFirstToLastRequest", testHoldingRequestsSpansFirstToLastRequest},
      {"testDeadlineEndsARequestWaitingForTheWeights", testDeadlineEndsARequestWaitingForTheWeights},
      {"testTransfersLandWhileTheWeightsAreRestored", testTransfersLandWhileTheWeightsAreRestored},
      {"testRequestsWaitForTheWeightsEachOnItsOwnClock", testRequestsWaitForTheWeightsEachOnItsOwnClock},
      {"testRestoreTriesAgainAtTheLimit", testRestoreTriesAgainAtTheLimit},
      {"testARestoredRequestWaitsAfreshAtItsStart", testARestoredRequestWaitsAfreshAtItsStart},
      {"testIdleRestoreTakesTheFloorForTheRequest", testIdleRestoreTakesTheFloorForTheRequest},
      {"testRestoreTakesTheFloorWhateverTheHostSpares", testRestoreTakesTheFloorWhateverTheHostSpares},
  };
  return kTests;
}
