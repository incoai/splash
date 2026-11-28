#include "ProtocolPeer.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "TestStatus.hpp"
#include "engine/Cache.hpp"
#include "engine/FdTransport.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

extern char **environ;

using namespace splash;
using namespace splash::engine;

namespace {

struct Pipes final {
  std::array<int, 2> input{};
  std::array<int, 2> output{};
  Pipes() {
    if (pipe(input.data()) || pipe(output.data())) {
      throw std::runtime_error("pipe creation failed");
    }
  }
  ~Pipes() {
    for (int fd : input)
      if (fd >= 0)
        close(fd);
    for (int fd : output)
      if (fd >= 0)
        close(fd);
  }
  void closeInputWriter() {
    close(input[1]);
    input[1] = -1;
  }
};

using splash::test::rejects;
using splash::test::require;

// For a failure that leaves a thread blocked: unwinding would wait for it.
[[noreturn]] void abandon(const char *message) {
  std::cerr << "native fd transport tests failed: " << message << '\n';
  std::_Exit(EXIT_FAILURE);
}

struct Harness final {
  explicit Harness(size_t inputQueueBytes = engine::FdTransport::kInputQueueBytes,
                   int inputFd = -1, int outputFd = -1)
      : transport(inputFd < 0 ? pipes.input[0] : inputFd,
                  outputFd < 0 ? pipes.output[1] : outputFd, inputQueueBytes) {
    loop.serveStatus([this] { return status(); });
    storage.commandInFlight = [this] { return loop.commandInFlight(); };
  }
  Pipes pipes;
  test::TestKvStorage storage{8, 4096, 1};
  KvPool pool{storage, 8};
  engine::Cache resources{pool, nullptr, nullptr};
  // No request gets a lane: one waits for it until its deadline.
  test::Executor executor{0};
  engine::FdTransport transport;
  // What the loop answers a status request with.
  std::function<std::string()> status = test::readyStatusJson;
  test::Weights weights;
  engine::NativeRuntime loop{
      {.engine = test::engineConfig(
           {.weights = engine::WeightResidencyConfig{.weights = &weights, .admit = test::admitAll}})},
      resources, executor, transport.outputSink(), protocol::ProtocolLimits{}};
};

// A loop over a persistent tier, with room for a 2060-token prompt, that
// admits requests, publishes their states and runs each command at once; a
// request ends with the token its prompt's last step selects. The tier's
// copies land only when the test says so.
struct PersistentHarness final {
  PersistentHarness() {
    storage.commandInFlight = [this] { return loop.commandInFlight(); };
    tier.transferLimit = 64;
    tier.capacity = 128;
    executor.prefillAnchor = true;
    // States the tier writes, whose writes land when the test says so.
    executor.keptTier = std::make_shared<test::OffloadControl>();
  }
  Pipes pipes;
  test::TestKvStorage storage{72, 4096, 4};
  KvPool pool{storage, 0};
  test::TestKvTier tier{true};
  engine::Cache resources{pool, &tier, nullptr};
  test::Executor executor;
  engine::FdTransport transport{pipes.input[0], pipes.output[1]};
  test::Weights weights;
  engine::NativeRuntime loop{
      {.engine = test::engineConfig(
           {.weights = engine::WeightResidencyConfig{.weights = &weights, .admit = test::admitAll}})},
      resources, executor, transport.outputSink(), protocol::ProtocolLimits{}};
};

// A request for three prompt tokens and one output token: its wall-clock
// deadline is a minute away, its remaining budget `remainingMicros`.
protocol::RequestFrame requestFrame(uint64_t id, uint64_t remainingMicros) {
  protocol::RequestFrame request;
  request.requestId = id;
  request.promptTokens = {1, 2, 3};
  request.logicalMaxOutputTokens = 1;
  request.absoluteDeadlineUnixMicros =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count() +
      60'000'000;
  request.remainingDeadlineMicros = remainingMicros;
  return request;
}

// Writes every byte, waiting for room in the pipe.
void writeAll(int fd, std::span<const uint8_t> bytes) {
  while (!bytes.empty()) {
    const ssize_t count = write(fd, bytes.data(), bytes.size());
    if (count > 0) {
      bytes = bytes.subspan(static_cast<size_t>(count));
    } else if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      pollfd descriptor{fd, POLLOUT, 0};
      static_cast<void>(poll(&descriptor, 1, 100));
    } else if (!(count < 0 && errno == EINTR)) {
      throw std::runtime_error("test input write failed");
    }
  }
}

// Writes what the pipe takes without waiting; returns the bytes written.
size_t writeAvailable(int fd, std::span<const uint8_t> bytes) {
  size_t written = 0;
  while (written < bytes.size()) {
    const ssize_t count = write(fd, bytes.data() + written, bytes.size() - written);
    if (count > 0)
      written += static_cast<size_t>(count);
    else if (!(count < 0 && errno == EINTR))
      break;
  }
  return written;
}

// Reads the loop's output until it answers status request `correlationId`.
bool awaitStatus(int fd, uint64_t correlationId, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  protocol::peer::EventReader reader;
  std::array<uint8_t, 4096> buffer{};
  while (true) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    pollfd descriptor{fd, POLLIN, 0};
    if (left.count() <= 0 || poll(&descriptor, 1, static_cast<int>(left.count())) <= 0)
      return false;
    const ssize_t count = read(fd, buffer.data(), buffer.size());
    if (count <= 0)
      return false;
    for (const auto &event : reader.feed(std::span<const uint8_t>(
             buffer.data(), static_cast<size_t>(count)))) {
      const auto *status = std::get_if<protocol::StatusJsonEvent>(&event);
      if (status && status->correlationId == correlationId)
        return true;
    }
  }
}

// Holds the loop in its first control pass until released, as a long
// command or pass does.
struct BusyPass final {
  std::promise<void> entered;
  std::promise<void> release;
  std::shared_future<void> released = release.get_future().share();
  std::atomic<bool> held{false};

  void install(engine::FdTransport &transport) {
    transport.setControlHandler([this] {
      if (!held.exchange(true)) {
        entered.set_value();
        released.wait_for(std::chrono::seconds(30));
      }
      return false;
    });
    transport.controlNotifier()();
  }
};

// Runs the loop on a thread of its own.
std::future<engine::NativeProcessExit> start(Harness &harness) {
  return std::async(std::launch::async,
                    [&harness] { return harness.transport.run(harness.loop); });
}

engine::NativeProcessExit finish(std::future<engine::NativeProcessExit> &loop) {
  if (loop.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
    abandon("the loop did not end");
  return loop.get();
}

void wakeWithStatusRequest(Harness &harness, uint64_t id) {
  writeAll(harness.pipes.input[1],
           protocol::peer::serialize(protocol::StatusRequestFrame{id}));
}

// Runs the loop on a thread of its own and returns once it takes the input:
// it has answered a status request.
std::future<engine::NativeProcessExit> startTaking(Harness &harness) {
  constexpr uint64_t kTaking = 1'000;
  auto loop = start(harness);
  wakeWithStatusRequest(harness, kTaking);
  if (!awaitStatus(harness.pipes.output[0], kTaking, std::chrono::seconds(5)))
    abandon("the loop did not take its input");
  return loop;
}

// Hands `input` to a loop that takes the input, then ends the input.
engine::NativeProcessExit run(std::span<const uint8_t> input) {
  Harness harness;
  auto loop = startTaking(harness);
  writeAll(harness.pipes.input[1], input);
  harness.pipes.closeInputWriter();
  return finish(loop);
}

// A shutdown request ends run() with a clean exit while the input is still
// open, and a control handler that reports pending work is run again at the
// loop's next wake, without another control notification, until it reports
// none. Here input wakes the loop, as a landing transfer's completion does.
void testShutdownRequestAndControlContinuation() {
  {
    Harness harness;
    harness.transport.requestShutdown();
    require(harness.transport.run(harness.loop) == engine::NativeProcessExit::CleanEof,
            "shutdown request did not end the loop cleanly");
  }
  {
    Harness harness;
    int invocations = 0;
    harness.transport.setControlHandler([&] {
      if (++invocations < 3) {
        wakeWithStatusRequest(harness, invocations);
        return true;
      }
      harness.transport.requestShutdown();
      return false;
    });
    harness.transport.controlNotifier()();
    require(harness.transport.run(harness.loop) == engine::NativeProcessExit::CleanEof,
            "control-driven shutdown did not end the loop cleanly");
    require(invocations == 3,
            "control handler was not continued until it reported no pending work");
  }
}

// A control notification that arrives while a command is in flight runs
// only once the loop has consumed the command.
void testControlWaitsForTheCommandInFlight() {
  Harness harness;
  // The request takes the one lane and ends with the token its prefill
  // selects; the prefill stays in flight until the test completes it, as
  // Metal's completion handler would.
  harness.executor.maximumLanes = 1;
  harness.executor.prefillAnchor = true;
  const auto commandReady = harness.executor.holdCommands();
  std::promise<void> submitted;
  harness.executor.submitObserver = [&submitted] { submitted.set_value(); };
  std::atomic<int> passes{0};
  std::atomic<bool> passedInFlight{false};
  harness.transport.setControlHandler([&harness, &passes, &passedInFlight] {
    if (harness.loop.commandInFlight())
      passedInFlight = true;
    ++passes;
    harness.transport.requestShutdown();
    return false;
  });
  harness.loop.announceReady(protocol::peer::kTestReady);
  writeAll(harness.pipes.input[1],
           protocol::peer::serialize(requestFrame(7, 30'000'000)));
  auto loop = start(harness);
  if (submitted.get_future().wait_for(std::chrono::seconds(10)) !=
      std::future_status::ready)
    abandon("the loop did not submit the request's prefill");
  harness.transport.controlNotifier()();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const int passesInFlight = passes;
  *commandReady = true;
  harness.executor.heldCompletion();
  require(finish(loop) == engine::NativeProcessExit::CleanEof,
          "the control pass after the command did not end the loop cleanly");
  require(passesInFlight == 0,
          "a control pass ran while the command was in flight");
  require(passes == 1 && !passedInFlight,
          "the deferred control pass did not run once after the command");
}

// The loop records its longest iteration, the input it hands the loop
// included, and keeps it as later ones are shorter.
void testLoopRecordsItsLongestTick() {
  Harness harness;
  require(harness.transport.maxTickMilliseconds() == 0.0,
          "a loop that never ran reported a tick");
  // The loop answers a status request while it takes the input.
  int answers = 0;
  harness.status = [&] {
    if (++answers == 1)
      std::this_thread::sleep_for(std::chrono::milliseconds(40));
    return test::readyStatusJson();
  };
  auto loop = start(harness);
  for (uint64_t id : {1, 2}) {
    wakeWithStatusRequest(harness, id);
    require(awaitStatus(harness.pipes.output[0], id, std::chrono::seconds(5)),
            "the loop did not answer a status request");
  }
  harness.transport.requestShutdown();
  require(finish(loop) == engine::NativeProcessExit::CleanEof,
          "a shutdown did not end the loop cleanly");
  require(harness.transport.maxTickMilliseconds() >= 40.0,
          "the loop did not keep its longest iteration");
}

// With no input and no command in flight, the loop sleeps until the
// engine's next deadline and then fails the request that reached it.
void testLoopWakesForAnEngineDeadline() {
  Pipes pipes;
  test::TestKvStorage storage{8, 4096, 1};
  KvPool pool{storage, 8};
  engine::Cache resources{pool, nullptr, nullptr};
  // No request gets a lane.
  test::Executor executor{0};
  engine::FdTransport transport{pipes.input[0], pipes.output[1]};
  std::vector<protocol::ErrorEvent> errors;
  test::Weights weights;
  engine::NativeRuntime loop{
      {.engine = test::engineConfig(
           {.weights = engine::WeightResidencyConfig{.weights = &weights, .admit = test::admitAll}})},
      resources, executor,
      [&](std::span<const uint8_t> bytes) {
        for (const auto &event : protocol::peer::decodeEvents(bytes)) {
          if (const auto *error = std::get_if<protocol::ErrorEvent>(&event)) {
            errors.push_back(*error);
            transport.requestShutdown();
          }
        }
      },
      protocol::ProtocolLimits{}};
  storage.commandInFlight = [&] { return loop.commandInFlight(); };
  loop.announceReady(protocol::peer::kTestReady);
  const auto wire = protocol::peer::serialize(requestFrame(5, 20'000));
  require(write(pipes.input[1], wire.data(), wire.size()) ==
              static_cast<ssize_t>(wire.size()),
          "failed to send the request");
  // The input stays open. The alarm turns a missed wake-up into a failure
  // instead of a hang.
  alarm(10);
  const engine::NativeProcessExit exit = transport.run(loop);
  alarm(0);
  require(exit == engine::NativeProcessExit::CleanEof && errors.size() == 1 &&
              errors[0].requestId == 5 && errors[0].code == "deadline_exceeded",
          "loop did not wake for the request deadline");
}

// While a part of the weights is written on its thread, the loop answers a
// status request through the pipes, as the server's liveness probe asks it to
// within supervision::kLoopStall; the write's end, signalled from that thread
// through the loop's completion notifier, wakes the loop without any input,
// and the restore goes on at once.
void testTheLoopAnswersWhileTheWeightsAreWritten() {
  Harness harness;
  test::Weights &weights = harness.weights;
  weights.writes = true;
  harness.loop.announceReady(protocol::peer::kTestReady);
  require(harness.loop.receive(protocol::peer::serialize(protocol::ReleaseRequestFrame{1})) && weights.released(),
          "the weights were not released");
  writeAll(harness.pipes.input[1], protocol::peer::serialize(requestFrame(7, 30'000'000)));
  // Whether, within 5 s, `restores` images have been taken back, the last of
  // them still being written.
  const auto writingImage = [&weights](uint32_t restores) {
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!(weights.restores == restores && weights.writing()) && std::chrono::steady_clock::now() < limit)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return weights.restores == restores && weights.writing();
  };
  auto loop = start(harness);
  if (!writingImage(1))
    abandon("the loop did not start writing the first image");
  wakeWithStatusRequest(harness, 9);
  const bool answered = awaitStatus(harness.pipes.output[0], 9, std::chrono::seconds(5));
  const bool answeredWhileWriting = answered && weights.writing();
  bool woke = true;
  for (uint32_t image = 2; image <= test::Weights::kImages; ++image) {
    weights.endWrite();
    woke = woke && writingImage(image);
  }
  harness.transport.requestShutdown();
  require(finish(loop) == engine::NativeProcessExit::CleanEof, "the loop did not end cleanly");
  require(answeredWhileWriting, "the loop did not answer status while an image was written");
  require(woke, "the end of an image's write did not wake the loop");
}

// While the loop spends longer in one pass than the server waits for a write
// to progress, the reader takes a 1 MiB frame without stalling its writer;
// the loop then handles it and the frame after it, in order.
void testReaderReadsWhileTheLoopIsBusy() {
  Harness harness;
  BusyPass busy;
  busy.install(harness.transport);
  auto loop = start(harness);
  if (busy.entered.get_future().wait_for(std::chrono::seconds(5)) !=
      std::future_status::ready)
    abandon("the loop did not enter its control pass");
  // A mask response for a request that has ended is accepted and dropped.
  protocol::MaskResponseFrame mask;
  mask.requestId = 77;
  mask.maskRequestId = 1;
  mask.maskWords.assign(262'144, 0);
  std::vector<uint8_t> input = protocol::peer::serialize(mask);
  require(input.size() > 1024 * 1024, "the test frame is smaller than 1 MiB");
  const auto status =
      protocol::peer::serialize(protocol::StatusRequestFrame{9});
  input.insert(input.end(), status.begin(), status.end());
  auto writer = std::async(std::launch::async,
                           [&] { writeAll(harness.pipes.input[1], input); });
  const bool unstalled =
      writer.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
  busy.release.set_value();
  if (writer.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
    abandon("the writer stayed stalled after the loop resumed");
  const bool answered =
      awaitStatus(harness.pipes.output[0], 9, std::chrono::seconds(5));
  harness.transport.requestShutdown();
  const auto exit = finish(loop);
  require(unstalled, "a busy loop stalled the writer of a 1 MiB frame");
  require(answered && exit == engine::NativeProcessExit::CleanEof,
          "the frames read while the loop was busy were not handled in order");
}

// While the loop is busy the reader queues up to its bound, then stops
// reading, and the pipe stops the writer. Once the loop takes the queue the
// rest is written and every frame is handled.
void testQueueBoundStopsTheWriter() {
  constexpr size_t kBound = 256 * 1024;
  Harness harness(kBound);
  BusyPass busy;
  busy.install(harness.transport);
  auto loop = start(harness);
  if (busy.entered.get_future().wait_for(std::chrono::seconds(5)) !=
      std::future_status::ready)
    abandon("the loop did not enter its control pass");
  // Cancels of a request that does not exist are accepted and dropped.
  const auto cancel = protocol::peer::serialize(protocol::CancelFrame{99});
  std::vector<uint8_t> input;
  while (input.size() < 4 * kBound)
    input.insert(input.end(), cancel.begin(), cancel.end());
  const auto status =
      protocol::peer::serialize(protocol::StatusRequestFrame{10});
  input.insert(input.end(), status.begin(), status.end());
  const int writer = harness.pipes.input[1];
  require(fcntl(writer, F_SETFL, fcntl(writer, F_GETFL) | O_NONBLOCK) == 0,
          "could not make the test writer nonblocking");
  size_t written = 0;
  for (int idle = 0; idle < 10 && written < input.size(); ) {
    const size_t more = writeAvailable(
        writer, std::span<const uint8_t>(input).subspan(written));
    written += more;
    idle = more ? 0 : idle + 1;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  const size_t accepted = written;
  busy.release.set_value();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (written < input.size() && std::chrono::steady_clock::now() < deadline) {
    written += writeAvailable(writer, std::span<const uint8_t>(input).subspan(written));
    pollfd descriptor{writer, POLLOUT, 0};
    static_cast<void>(poll(&descriptor, 1, 100));
  }
  const bool answered =
      awaitStatus(harness.pipes.output[0], 10, std::chrono::seconds(5));
  harness.transport.requestShutdown();
  const auto exit = finish(loop);
  // A full pipe may be in flight beyond the bound.
  require(accepted >= kBound && accepted <= kBound + 256 * 1024,
          "the reader did not queue up to its bound and then stop the writer");
  require(written == input.size() && answered &&
              exit == engine::NativeProcessExit::CleanEof,
          "writing did not resume, or a frame was lost, once the loop caught up");
}

// A shutdown ends run() promptly, and the transport's end stops and joins
// the reader, whether it waits on an idle pipe or for room in a full queue.
void testShutdownAndTheTransportsEndStopTheReader() {
  for (bool full : {false, true}) {
    auto owned = std::make_unique<Harness>(64 * 1024);
    Harness &harness = *owned;
    BusyPass busy;
    if (full)
      busy.install(harness.transport);
    auto loop = start(harness);
    if (full) {
      if (busy.entered.get_future().wait_for(std::chrono::seconds(5)) !=
          std::future_status::ready)
        abandon("the loop did not enter its control pass");
      const int writer = harness.pipes.input[1];
      require(fcntl(writer, F_SETFL, fcntl(writer, F_GETFL) | O_NONBLOCK) == 0,
              "could not make the test writer nonblocking");
      const std::vector<uint8_t> filler(1024 * 1024, 0);
      for (int idle = 0; idle < 10; ) {
        idle = writeAvailable(writer, filler) ? 0 : idle + 1;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    const auto requested = std::chrono::steady_clock::now();
    harness.transport.requestShutdown();
    if (full)
      busy.release.set_value();
    const auto exit = finish(loop);
    require(exit == engine::NativeProcessExit::CleanEof &&
                std::chrono::steady_clock::now() - requested < std::chrono::seconds(2),
            "a shutdown did not end run() promptly");
    auto ended = std::async(std::launch::async, [&owned] { owned.reset(); });
    if (ended.wait_for(std::chrono::seconds(2)) != std::future_status::ready)
      abandon(full ? "the transport's end did not stop a reader waiting for room"
                   : "the transport's end did not stop a reader waiting on an idle pipe");
  }
}

// Until run() takes the input, its end is a shutdown request: a start whose
// server has gone stops (main.mm). run() then ends at once, as a shutdown.
void testTheInputsEndBeforeRunIsAShutdown() {
  Harness harness;
  harness.pipes.closeInputWriter();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!harness.transport.shutdownRequested() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  require(harness.transport.shutdownRequested(),
          "the input's end before run() did not request a shutdown");
  require(harness.transport.run(harness.loop) == engine::NativeProcessExit::CleanEof,
          "the input's end before run() did not end run() as a shutdown");
}

// The input's end and a read error reach the loop after the bytes read
// before them: a status request sent just before either is answered.
void testInputEndsAfterItsBytes() {
  const auto status =
      protocol::peer::serialize(protocol::StatusRequestFrame{11});
  {
    Harness harness;
    auto loop = startTaking(harness);
    writeAll(harness.pipes.input[1], status);
    harness.pipes.closeInputWriter();
    require(finish(loop) == engine::NativeProcessExit::CleanEof &&
                awaitStatus(harness.pipes.output[0], 11, std::chrono::seconds(1)),
            "the end of the input overtook the bytes before it");
  }
  // A loopback connection reset after its bytes reports them first, then
  // ECONNRESET.
  const int listener = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  socklen_t length = sizeof(address);
  require(listener >= 0 &&
              !bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) &&
              !listen(listener, 1) &&
              !getsockname(listener, reinterpret_cast<sockaddr *>(&address), &length),
          "could not listen on loopback");
  const int client = socket(AF_INET, SOCK_STREAM, 0);
  require(client >= 0 &&
              !connect(client, reinterpret_cast<sockaddr *>(&address), sizeof(address)),
          "could not connect on loopback");
  const int server = accept(listener, nullptr, nullptr);
  require(server >= 0, "could not accept on loopback");
  writeAll(server, status);
  const linger reset{1, 0};
  require(!setsockopt(server, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
          "could not arm the connection reset");
  close(server);
  {
    Harness harness(engine::FdTransport::kInputQueueBytes, client);
    require(harness.transport.run(harness.loop) == engine::NativeProcessExit::IoFailure &&
                awaitStatus(harness.pipes.output[0], 11, std::chrono::seconds(1)),
            "a read error overtook the bytes before it");
    require(harness.transport.failure().find(std::strerror(ECONNRESET)) !=
                std::string::npos,
            "the transport did not report the read error that ended it");
  }
  close(client);
  close(listener);
}

// Fills the output pipe, so the loop's next write blocks.
void fillOutput(Harness &harness) {
  const int output = harness.pipes.output[1];
  const int flags = fcntl(output, F_GETFL);
  require(fcntl(output, F_SETFL, flags | O_NONBLOCK) == 0,
          "could not make the output nonblocking");
  const std::vector<uint8_t> filler(64 * 1024, 0);
  while (writeAvailable(output, filler)) {
  }
  require(fcntl(output, F_SETFL, flags) == 0, "could not restore the output");
}

// The signal that requests a shutdown ends an output write the server has
// stopped draining, which would otherwise hold the loop forever: run()
// returns at once, as a shutdown, with the engine healthy.
void testShutdownInterruptsABlockedOutputWrite() {
  Harness harness;
  // The loop builds the status reply right before it writes it.
  std::promise<void> answering;
  harness.status = [&] {
    answering.set_value();
    return test::readyStatusJson();
  };
  // Like the process's own handler, without SA_RESTART: the signal interrupts
  // the write.
  struct sigaction action {};
  action.sa_handler = [](int) {};
  sigemptyset(&action.sa_mask);
  struct sigaction previous {};
  require(sigaction(SIGUSR1, &action, &previous) == 0,
          "could not install the test signal handler");
  fillOutput(harness);
  std::promise<engine::NativeProcessExit> exited;
  auto exit = exited.get_future();
  std::thread loop([&] { exited.set_value(harness.transport.run(harness.loop)); });
  wakeWithStatusRequest(harness, 12);
  if (answering.get_future().wait_for(std::chrono::seconds(5)) !=
      std::future_status::ready)
    abandon("the loop did not answer the status request");
  // The loop is past its shutdown check, on its way into the write. A signal
  // that comes before the write starts interrupts nothing, so it repeats
  // until run() returns.
  harness.transport.requestShutdown();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (exit.wait_for(std::chrono::milliseconds(10)) != std::future_status::ready) {
    if (std::chrono::steady_clock::now() >= deadline)
      abandon("a shutdown did not end a blocked output write");
    pthread_kill(loop.native_handle(), SIGUSR1);
  }
  loop.join();
  sigaction(SIGUSR1, &previous, nullptr);
  require(exit.get() == engine::NativeProcessExit::CleanEof && harness.loop.engineHealthy(),
          "an interrupted output write did not end the loop as a shutdown");
}

// No output write starts after a shutdown request: the server may have
// stopped draining the output, and no second signal would end the write.
void testNoOutputWriteStartsAfterAShutdown() {
  Harness harness;
  fillOutput(harness);
  harness.status = [&] {
    harness.transport.requestShutdown();
    return test::readyStatusJson();
  };
  auto loop = start(harness);
  wakeWithStatusRequest(harness, 14);
  require(finish(loop) == engine::NativeProcessExit::CleanEof && harness.loop.engineHealthy(),
          "a write after a shutdown request did not end the loop as a shutdown");
}

// A server that closed its end of the output ends the loop as a shutdown;
// any other write error ends it as an I/O failure that names the error. The
// engine stays healthy either way.
void testFailedOutputEndsTheLoop() {
  {
    // As in the process (main.mm), the write fails with EPIPE.
    struct sigaction ignore {};
    ignore.sa_handler = SIG_IGN;
    sigemptyset(&ignore.sa_mask);
    struct sigaction previous {};
    require(sigaction(SIGPIPE, &ignore, &previous) == 0, "could not ignore SIGPIPE");
    Harness harness;
    close(harness.pipes.output[0]);
    harness.pipes.output[0] = -1;
    wakeWithStatusRequest(harness, 13);
    const auto exit = harness.transport.run(harness.loop);
    sigaction(SIGPIPE, &previous, nullptr);
    require(exit == engine::NativeProcessExit::CleanEof && harness.loop.engineHealthy(),
            "a closed output did not end the loop as a shutdown");
  }
  Pipes unwritable;
  // A pipe's read end refuses writes with EBADF.
  Harness harness(engine::FdTransport::kInputQueueBytes, -1, unwritable.output[0]);
  wakeWithStatusRequest(harness, 15);
  require(harness.transport.run(harness.loop) == engine::NativeProcessExit::IoFailure &&
              harness.loop.engineHealthy(),
          "a failed output write did not end the loop as an I/O failure");
  require(harness.transport.failure().find(std::strerror(EBADF)) != std::string::npos,
          "the transport did not report the write error that ended it");
}

// A reader that may queue no input would never hand the loop a frame.
void testTransportNeedsRoomForInput() {
  Pipes pipes;
  rejects([&] { engine::FdTransport transport(pipes.input[0], pipes.output[1], 0); },
          "room for input", "a transport with no room for input was built");
}

void testCleanEofAndProtocolFailure() {
  require(run({}) == engine::NativeProcessExit::CleanEof,
          "empty clean input did not return clean EOF");
  const std::array<uint8_t, protocol::kFrameHeaderBytes> malformed{};
  require(run(malformed) == engine::NativeProcessExit::ProtocolFailure,
          "malformed input did not return protocol failure");
}

// A clean stop's flush ends at once with no restore point waiting, and
// stops at its deadline while the writes it started have not landed.
void testFlushEndsOrStopsAtItsDeadline() {
  using namespace std::chrono_literals;
  {
    Harness harness;
    require(harness.transport.runFlush(harness.loop, 0ms),
            "a flush with nothing waiting did not end");
  }
  PersistentHarness harness;
  harness.loop.announceReady(protocol::peer::kTestReady);
  protocol::RequestFrame frame = requestFrame(1, 30'000'000);
  frame.promptTokens.assign(2060, 7);
  require(harness.loop.receive(protocol::peer::serialize(frame)), "the request was refused");
  for (uint32_t step = 0; step < 16 && !harness.loop.statusSnapshot().writeBehind.waiting; ++step)
    static_cast<void>(harness.loop.tick());
  require(harness.loop.statusSnapshot().writeBehind.waiting == 1,
          "the replay point did not wait to be written");
  const auto started = std::chrono::steady_clock::now();
  require(!harness.transport.runFlush(harness.loop, 50ms) &&
              std::chrono::steady_clock::now() - started >= 50ms && harness.tier.demotions == 16,
          "a flush whose writes never land did not stop at its deadline");
}

// The roles this binary plays for the stop-signal test, spawned as their
// own processes: an engine that installs the stop signals and says when one
// asks it to stop, and a server that starts one and exits when told.
constexpr std::string_view kEngineRole = "stop-signals-engine";
constexpr std::string_view kServerRole = "stop-signals-server";

void writeLine(int fd, const std::string &line) {
  const std::string text = line + "\n";
  if (write(fd, text.data(), text.size()) != static_cast<ssize_t>(text.size()))
    _exit(3);
}

// Installs the stop signals over a transport whose input never ends, says
// "ready", and "stopped" once a signal asked the transport to stop; it gives
// up after 10 s.
int runStopSignalsEngine() {
  std::array<int, 2> input{};
  if (pipe(input.data()))
    return 2;
  const int output = open("/dev/null", O_WRONLY);
  FdTransport transport(input[0], output);
  StopSignals signals(transport);
  writeLine(STDOUT_FILENO, "ready");
  for (int waited = 0; waited < 5'000; ++waited) {
    if (transport.shutdownRequested()) {
      writeLine(STDOUT_FILENO, "stopped");
      _exit(0);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  _exit(1);
}

// Spawns `self` in `role` with stdin and stdout on the fds given, stderr,
// and no other fd: a pipe's end it inherited would keep the pipe open. Its
// stop signals are at their defaults and none blocked, and it has the
// variable by which a server tells its engine to stop on its signal alone
// when `underServer`.
pid_t spawnRole(const char *self, std::string_view role, int in, int out, bool underServer) {
  posix_spawn_file_actions_t files;
  posix_spawn_file_actions_init(&files);
  posix_spawn_file_actions_adddup2(&files, in, STDIN_FILENO);
  posix_spawn_file_actions_adddup2(&files, out, STDOUT_FILENO);
  posix_spawn_file_actions_addinherit_np(&files, STDERR_FILENO);
  posix_spawnattr_t attributes;
  posix_spawnattr_init(&attributes);
  sigset_t defaults, none;
  sigemptyset(&defaults);
  for (const int number : {SIGTERM, SIGINT, SIGHUP})
    sigaddset(&defaults, number);
  sigemptyset(&none);
  posix_spawnattr_setsigdefault(&attributes, &defaults);
  posix_spawnattr_setsigmask(&attributes, &none);
  posix_spawnattr_setflags(&attributes,
                           POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_CLOEXEC_DEFAULT);
  std::vector<std::string> environment;
  for (char **variable = environ; *variable; ++variable)
    if (!std::string_view(*variable).starts_with("SPLASH_ENGINE_STOPPED_BY_PARENT="))
      environment.emplace_back(*variable);
  if (underServer)
    environment.emplace_back("SPLASH_ENGINE_STOPPED_BY_PARENT=1");
  std::vector<char *> envp;
  for (std::string &variable : environment)
    envp.push_back(variable.data());
  envp.push_back(nullptr);
  std::string program(self), name(role);
  char *argv[] = {program.data(), name.data(), nullptr};
  pid_t pid = 0;
  const int error = posix_spawn(&pid, self, &files, &attributes, argv, envp.data());
  posix_spawn_file_actions_destroy(&files);
  posix_spawnattr_destroy(&attributes);
  if (error || pid <= 0)
    throw std::runtime_error("cannot spawn " + std::string(role) + ": " + std::strerror(error));
  return pid;
}

// Starts an engine as a server does and exits once its input ends, leaving
// the engine to run without it; says the engine's pid first.
int runStopSignalsServer(const char *self) {
  const pid_t engine = spawnRole(self, kEngineRole, STDIN_FILENO, STDOUT_FILENO, true);
  writeLine(STDOUT_FILENO, "engine " + std::to_string(engine));
  char byte = 0;
  while (read(STDIN_FILENO, &byte, 1) > 0) {
  }
  _exit(0);
}

// The next line a role writes on `fd` within `timeout`; empty at its end or
// when none arrived.
std::string readLine(int fd, std::chrono::milliseconds timeout) {
  std::string line;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  char byte = 0;
  while (true) {
    const auto left =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
    pollfd ready{fd, POLLIN, 0};
    if (left <= 0 || poll(&ready, 1, static_cast<int>(left)) <= 0 || read(fd, &byte, 1) != 1)
      return {};
    if (byte == '\n')
      return line;
    line += byte;
  }
}

// Sends `number` to `pid` from another process, as a stop of the whole
// process group comes from one.
void signalFromAnotherProcess(pid_t pid, const char *number) {
  // A pid of 0 or less names a process group, or every process.
  if (pid <= 1)
    throw std::logic_error("no engine to signal");
  std::string program("/bin/kill"), flag("-s"), name(number), target = std::to_string(pid);
  char *argv[] = {program.data(), flag.data(), name.data(), target.data(), nullptr};
  pid_t sender = 0;
  if (posix_spawn(&sender, "/bin/kill", nullptr, nullptr, argv, environ))
    throw std::runtime_error("cannot run /bin/kill");
  int status = 0;
  waitpid(sender, &status, 0);
  require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "/bin/kill did not signal the engine");
}

// An engine a server started stops on that server's own SIGTERM or SIGINT,
// not on one another process sends while the server lives, as a stop of the
// process group reaches it beside the server, which drains first; SIGHUP,
// any once the server is gone, and any to an engine no server started stop
// it whoever sends them. Each stop is said within 5 s; each that is not, in
// 300 ms.
void testAnEngineUnderAServerStopsOnItsServersSignalAlone(const char *self) {
  using namespace std::chrono_literals;
  struct Spawned final {
    Pipes pipes;
    pid_t pid = 0;
    Spawned(const char *self, bool underServer) {
      pid = spawnRole(self, kEngineRole, pipes.input[0], pipes.output[1], underServer);
      close(pipes.output[1]);
      pipes.output[1] = -1;
      require(readLine(pipes.output[0], 5s) == "ready", "the engine did not install its stop signals");
    }
    [[nodiscard]] bool stops(std::chrono::milliseconds within) const {
      return readLine(pipes.output[0], within) == "stopped";
    }
    ~Spawned() {
      kill(pid, SIGKILL);
      waitpid(pid, nullptr, 0);
    }
  };
  for (const char *number : {"TERM", "INT"}) {
    Spawned engine(self, true);
    signalFromAnotherProcess(engine.pid, number);
    require(!engine.stops(300ms), std::string("SIG") + number + " from another process stopped an engine its server runs");
    kill(engine.pid, std::string_view(number) == "TERM" ? SIGTERM : SIGINT);
    require(engine.stops(5s), std::string("the server's own SIG") + number + " did not stop its engine");
  }
  {
    Spawned engine(self, true);
    signalFromAnotherProcess(engine.pid, "HUP");
    require(engine.stops(5s), "SIGHUP from another process did not stop an engine its server runs");
  }
  {
    Spawned engine(self, false);
    signalFromAnotherProcess(engine.pid, "TERM");
    require(engine.stops(5s), "SIGTERM from another process did not stop an engine no server started");
  }
  // The server goes, the engine stays, and then any SIGTERM stops it.
  Pipes pipes;
  const pid_t server = spawnRole(self, kServerRole, pipes.input[0], pipes.output[1], false);
  close(pipes.output[1]);
  pipes.output[1] = -1;
  const std::string started = readLine(pipes.output[0], 5s);
  require(started.starts_with("engine ") && readLine(pipes.output[0], 5s) == "ready",
          "the server role did not start its engine");
  const pid_t engine = static_cast<pid_t>(std::stol(started.substr(7)));
  pipes.closeInputWriter();
  waitpid(server, nullptr, 0);
  signalFromAnotherProcess(engine, "TERM");
  const bool stopped = readLine(pipes.output[0], 5s) == "stopped";
  if (!stopped)
    kill(engine, SIGKILL);
  require(stopped, "SIGTERM from another process did not stop an engine whose server is gone");
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 1 && argv[1] == kEngineRole)
    return runStopSignalsEngine();
  if (argc > 1 && argv[1] == kServerRole)
    return runStopSignalsServer(argv[0]);
  try {
    testTransportNeedsRoomForInput();
    testCleanEofAndProtocolFailure();
    testShutdownRequestAndControlContinuation();
    testControlWaitsForTheCommandInFlight();
    testLoopRecordsItsLongestTick();
    testLoopWakesForAnEngineDeadline();
    testTheLoopAnswersWhileTheWeightsAreWritten();
    testReaderReadsWhileTheLoopIsBusy();
    testQueueBoundStopsTheWriter();
    testShutdownAndTheTransportsEndStopTheReader();
    testTheInputsEndBeforeRunIsAShutdown();
    testInputEndsAfterItsBytes();
    testShutdownInterruptsABlockedOutputWrite();
    testNoOutputWriteStartsAfterAShutdown();
    testFailedOutputEndsTheLoop();
    testFlushEndsOrStopsAtItsDeadline();
    testAnEngineUnderAServerStopsOnItsServersSignalAlone(argv[0]);
    std::cout << "native fd transport tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "native fd transport tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
