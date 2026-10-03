#include "ProtocolPeer.hpp"
#include "ScopedTestConfig.hpp"
#include "TestChecks.hpp"
#include "TestImmediateTicket.hpp"
#include "TestKvPool.hpp"
#include "TestStatus.hpp"
#include "engine/Cache.hpp"
#include "engine/FdTransport.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
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
#include <stdexcept>
#include <string>
#include <thread>

using namespace splash;
using namespace splash::engine;

namespace {

class Executor final : public model::Model {
public:
  // Unless admit is set, no request gets a lane: one waits for it until its
  // deadline. An admitted request takes lane 0, and each of its commands
  // stays in flight until the test sets ticketReady and calls
  // heldCompletion, as Metal's completion handler would.
  bool admit = false;
  std::shared_ptr<std::atomic<bool>> ticketReady =
      std::make_shared<std::atomic<bool>>(false);
  std::function<void()> heldCompletion;
  std::function<void()> onSubmit;

  StateAdmission begin(const ModelRequest &request) override {
    if (!admit)
      return {{}, StateFailure::ConcurrencyLimit};
    promptTokens_ = request.prompt.size();
    return {0, StateFailure::None};
  }
  void suspend(uint64_t) override {}
  StateAdmission resume(const ModelRequest &) override {
    return {0, StateFailure::None};
  }
  std::unique_ptr<StateRestore> beginRestore(uint64_t, uint32_t,
                                             std::shared_ptr<const CompositeState>, bool,
                                             std::function<void()>) override {
    return {};
  }
  void setDraftContextPlan(uint64_t, DraftContextPlan) override {}
  // Prefill consumes its rows; the final chunk selects token 42, which ends
  // the request.
  std::unique_ptr<ModelBatchTicket>
  submit(const BatchPlan &, std::span<const ModelBatchItem> items,
         std::function<void()> completion) override {
    std::vector<ModelStepResult> results;
    for (const ModelBatchItem &item : items) {
      ModelStepResult step{item.requestId, item.tokenCount, {}};
      if (item.logicalPosition + item.tokenCount == promptTokens_) {
        step.outputTokens = {42};
        step.outputTokensWithoutKv = 1;
        step.finished = true;
      }
      results.push_back(std::move(step));
    }
    heldCompletion = std::move(completion);
    if (onSubmit)
      onSubmit();
    return std::make_unique<test::HeldTicket>(std::move(results), ticketReady,
                                              0.0);
  }
  uint64_t snapshotBytes() const noexcept override { return 64; }
  std::shared_ptr<const CompositeState> snapshot(uint64_t) override {
    return {};
  }
  uint64_t reclaimIdleState(bool, model::IdleMemory) noexcept override { return 0; }
  std::optional<std::string> provideMask(uint64_t,
                                         std::span<const uint32_t>) override {
    return std::nullopt;
  }
  void end(uint64_t) override {}

private:
  uint64_t promptTokens_ = 0;
};

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

using splash::test::require;

// For a failure that leaves a thread blocked: unwinding would wait for it.
[[noreturn]] void abandon(const char *message) {
  std::cerr << "native fd transport tests failed: " << message << '\n';
  std::_Exit(EXIT_FAILURE);
}

struct Harness final {
  explicit Harness(size_t inputQueueBytes = engine::FdTransport::kInputQueueBytes,
                   int inputFd = -1)
      : seam({.transportInputQueueBytes = inputQueueBytes}),
        transport(inputFd < 0 ? pipes.input[0] : inputFd, pipes.output[1]) {
    storage.commandInFlight = [this] { return loop.commandInFlight(); };
  }
  Pipes pipes;
  test::TestKvStorage storage{8, 4096, 1};
  KvPool pool{storage, 8};
  engine::Cache resources{pool, nullptr, nullptr};
  Executor executor;
  test::ScopedTestConfig seam;
  engine::FdTransport transport;
  // What the loop answers a status request with.
  std::function<std::string()> status = test::readyStatusJson;
  engine::NativeRuntime loop{{}, resources, executor, transport.outputSink(),
                             [this] { return status(); }, protocol::ProtocolLimits{}};
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

engine::NativeProcessExit run(std::span<const uint8_t> input) {
  Harness harness;
  if (!input.empty()) {
    ssize_t count = write(harness.pipes.input[1], input.data(), input.size());
    require(count == static_cast<ssize_t>(input.size()),
            "failed to seed transport input");
  }
  harness.pipes.closeInputWriter();
  return harness.transport.run(harness.loop);
}

void wakeWithStatusRequest(Harness &harness, uint64_t id) {
  writeAll(harness.pipes.input[1],
           protocol::peer::serialize(protocol::StatusRequestFrame{id}));
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
  std::promise<void> submitted;
  harness.executor.admit = true;
  harness.executor.onSubmit = [&submitted] { submitted.set_value(); };
  std::atomic<int> passes{0};
  std::atomic<bool> passedInFlight{false};
  harness.transport.setControlHandler([&harness, &passes, &passedInFlight] {
    if (harness.loop.commandInFlight())
      passedInFlight = true;
    ++passes;
    harness.transport.requestShutdown();
    return false;
  });
  harness.loop.announceReady();
  writeAll(harness.pipes.input[1],
           protocol::peer::serialize(requestFrame(7, 30'000'000)));
  auto loop = start(harness);
  if (submitted.get_future().wait_for(std::chrono::seconds(10)) !=
      std::future_status::ready)
    abandon("the loop did not submit the request's prefill");
  harness.transport.controlNotifier()();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const int passesInFlight = passes;
  *harness.executor.ticketReady = true;
  harness.executor.heldCompletion();
  require(finish(loop) == engine::NativeProcessExit::CleanEof,
          "the control pass after the command did not end the loop cleanly");
  require(passesInFlight == 0,
          "a control pass ran while the command was in flight");
  require(passes == 1 && !passedInFlight,
          "the deferred control pass did not run once after the command");
}

// The loop records its longest control pass and tick, and keeps it as later
// ones are shorter.
void testLoopRecordsItsLongestTick() {
  Harness harness;
  require(harness.transport.maxTickMilliseconds() == 0.0,
          "a loop that never ran reported a tick");
  int invocations = 0;
  double recorded = 0.0;
  harness.transport.setControlHandler([&] {
    if (++invocations == 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(40));
      wakeWithStatusRequest(harness, 1);
      return true;
    }
    recorded = harness.transport.maxTickMilliseconds();
    harness.transport.requestShutdown();
    return false;
  });
  harness.transport.controlNotifier()();
  require(harness.transport.run(harness.loop) == engine::NativeProcessExit::CleanEof,
          "control-driven shutdown did not end the loop cleanly");
  require(recorded >= 40.0 && harness.transport.maxTickMilliseconds() >= recorded,
          "the loop did not keep its longest control pass and tick");
}

// With no input and no command in flight, the loop sleeps until the
// engine's next deadline and then fails the request that reached it.
void testLoopWakesForAnEngineDeadline() {
  Pipes pipes;
  test::TestKvStorage storage{8, 4096, 1};
  KvPool pool{storage, 8};
  engine::Cache resources{pool, nullptr, nullptr};
  Executor executor;
  engine::FdTransport transport{pipes.input[0], pipes.output[1]};
  std::vector<protocol::ErrorEvent> errors;
  engine::NativeRuntime loop{
      {}, resources, executor,
      [&](std::span<const uint8_t> bytes) {
        for (const auto &event : protocol::peer::decodeEvents(bytes)) {
          if (const auto *error = std::get_if<protocol::ErrorEvent>(&event)) {
            errors.push_back(*error);
            transport.requestShutdown();
          }
        }
      },
      test::readyStatusJson, protocol::ProtocolLimits{}};
  storage.commandInFlight = [&] { return loop.commandInFlight(); };
  loop.announceReady();
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

// Shutdown ends run() promptly and joins the reader, whether it waits on an
// idle pipe or for room in a full queue.
void testShutdownJoinsTheReader() {
  for (bool full : {false, true}) {
    Harness harness(64 * 1024);
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
            full ? "shutdown did not stop a reader waiting for room"
                 : "shutdown did not stop a reader waiting on an idle pipe");
  }
}

// The input's end and a read error reach the loop after the bytes read
// before them: a status request sent just before either is answered.
void testInputEndsAfterItsBytes() {
  const auto status =
      protocol::peer::serialize(protocol::StatusRequestFrame{11});
  {
    Harness harness;
    writeAll(harness.pipes.input[1], status);
    harness.pipes.closeInputWriter();
    require(harness.transport.run(harness.loop) == engine::NativeProcessExit::CleanEof &&
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

// The signal that requests a shutdown ends an output write the server has
// stopped draining, which would otherwise hold the loop forever: run()
// returns at once with the engine failed.
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
  // Fill the output pipe, so the reply's write blocks.
  const int output = harness.pipes.output[1];
  const int flags = fcntl(output, F_GETFL);
  require(fcntl(output, F_SETFL, flags | O_NONBLOCK) == 0,
          "could not make the output nonblocking");
  const std::vector<uint8_t> filler(64 * 1024, 0);
  while (writeAvailable(output, filler)) {
  }
  require(fcntl(output, F_SETFL, flags) == 0, "could not restore the output");
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
  require(exit.get() == engine::NativeProcessExit::EngineFailure,
          "an interrupted output write did not fail the engine");
}

void testCleanEofAndProtocolFailure() {
  require(run({}) == engine::NativeProcessExit::CleanEof,
          "empty clean input did not return clean EOF");
  const std::array<uint8_t, protocol::kFrameHeaderBytes> malformed{};
  require(run(malformed) == engine::NativeProcessExit::ProtocolFailure,
          "malformed input did not return protocol failure");
}

} // namespace

int main() {
  try {
    testCleanEofAndProtocolFailure();
    testShutdownRequestAndControlContinuation();
    testControlWaitsForTheCommandInFlight();
    testLoopRecordsItsLongestTick();
    testLoopWakesForAnEngineDeadline();
    testReaderReadsWhileTheLoopIsBusy();
    testQueueBoundStopsTheWriter();
    testShutdownJoinsTheReader();
    testInputEndsAfterItsBytes();
    testShutdownInterruptsABlockedOutputWrite();
    std::cout << "native fd transport tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "native fd transport tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
