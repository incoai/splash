#include "engine/FdTransport.hpp"
#include "AwakeClock.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <mutex>
#include <optional>
#include <poll.h>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace splash::engine {
namespace {

[[noreturn]] void throwIo(const char *operation) {
  throw std::system_error(errno, std::generic_category(), operation);
}

int pollTimeout(const NativeRuntime &loop) {
  auto delay = loop.millisecondsUntilNextWakeup();
  int timeout = -1;
  if (delay) {
    if (*delay <= 0.0)
      timeout = 0;
    else if (*delay >= double(INT_MAX))
      timeout = INT_MAX;
    else
      timeout = static_cast<int>(std::ceil(*delay));
  }
  return timeout;
}

} // namespace

struct FdTransport::LoopWake {
  int readFd;
  int writeFd;
  std::atomic<bool> controlPending{false};
  std::atomic<bool> shutdownRequested{false};
  // Set once run() takes the input, which then hands its end to the loop.
  std::atomic<bool> inputTaken{false};

  LoopWake() {
    int descriptors[2];
    if (pipe(descriptors) < 0)
      throwIo("pipe(loop wake)");
    readFd = descriptors[0];
    writeFd = descriptors[1];
    auto makeNonBlocking = [&](int fd) {
      int flags = fcntl(fd, F_GETFL);
      if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        int saved = errno;
        close(readFd);
        close(writeFd);
        errno = saved;
        throwIo("fcntl(loop wake)");
      }
    };
    makeNonBlocking(readFd);
    makeNonBlocking(writeFd);
  }

  ~LoopWake() {
    close(readFd);
    close(writeFd);
  }

  void notify() const noexcept {
    constexpr uint8_t byte = 1;
    while (true) {
      ssize_t written = write(writeFd, &byte, sizeof(byte));
      if (written > 0 ||
          (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))) {
        return;
      }
      if (written < 0 && errno == EINTR)
        continue;
      return;
    }
  }

  // Async-signal-safe.
  void requestShutdown() noexcept {
    shutdownRequested.store(true, std::memory_order_release);
    notify();
  }

  // The input ended (EOF). Once run() takes the input it hands that to the
  // loop; before, the server has gone, and it is a shutdown request.
  void inputEnded() noexcept {
    if (!inputTaken.load(std::memory_order_acquire))
      requestShutdown();
    else
      notify();
  }

  void notifyControl() noexcept {
    controlPending.store(true, std::memory_order_release);
    notify();
  }

  bool takeControl() noexcept {
    return controlPending.exchange(false, std::memory_order_acq_rel);
  }

  void drain() const noexcept {
    std::array<uint8_t, 64> bytes{};
    while (true) {
      ssize_t count = read(readFd, bytes.data(), bytes.size());
      if (count > 0)
        continue;
      if (count < 0 && errno == EINTR)
        continue;
      return;
    }
  }
};

// Reads the input on a thread of its own and queues it for the loop, which
// takes it between ticks. The bytes, then the input's end or a read error,
// reach the loop in the order they were read. Destruction stops the thread,
// whether it waits for input or for room in the queue, and joins it.
class FdTransport::InputReader final {
public:
  struct Input final {
    // What each read returned, in order.
    std::deque<std::vector<uint8_t>> chunks;
    // Set once the input has ended after these chunks: 0 at its end,
    // otherwise the error that ended reading.
    std::optional<int> end;
  };

  InputReader(int fd, size_t queueBytes, std::shared_ptr<LoopWake> wake)
      : fd_(fd), queueBytes_(queueBytes), wake_(std::move(wake)) {
    int descriptors[2];
    if (pipe(descriptors) < 0)
      throwIo("pipe(input reader stop)");
    stopRead_ = descriptors[0];
    stopWrite_ = descriptors[1];
    try {
      thread_ = std::thread([this] { read(); });
    } catch (...) {
      close(stopRead_);
      close(stopWrite_);
      throw;
    }
  }

  ~InputReader() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    room_.notify_all();
    constexpr uint8_t byte = 1;
    while (write(stopWrite_, &byte, sizeof(byte)) < 0 && errno == EINTR) {
    }
    thread_.join();
    close(stopRead_);
    close(stopWrite_);
  }

  InputReader(const InputReader &) = delete;
  InputReader &operator=(const InputReader &) = delete;

  // Everything read since the last call, with the input's end once nothing
  // read before it is left.
  [[nodiscard]] Input take() {
    Input input;
    {
      std::lock_guard lock(mutex_);
      input.chunks.swap(chunks_);
      queuedBytes_ = 0;
      input.end = end_;
    }
    room_.notify_all();
    return input;
  }

private:
  void read() noexcept {
    try {
      std::vector<uint8_t> buffer(64 * 1024);
      while (true) {
        size_t room = 0;
        {
          std::unique_lock lock(mutex_);
          room_.wait(lock, [&] { return stopping_ || queuedBytes_ < queueBytes_; });
          if (stopping_)
            return;
          room = queueBytes_ - queuedBytes_;
        }
        std::array<pollfd, 2> descriptors{pollfd{fd_, POLLIN, 0},
                                          pollfd{stopRead_, POLLIN, 0}};
        if (poll(descriptors.data(), descriptors.size(), -1) < 0) {
          if (errno == EINTR)
            continue;
          return finish(errno);
        }
        if (descriptors[1].revents)
          return;
        if (descriptors[0].revents & POLLNVAL)
          return finish(EBADF);
        if (!descriptors[0].revents)
          continue;
        // Only take() changes the room meanwhile, and only to more, so the
        // queue never passes its bound.
        const ssize_t count =
            ::read(fd_, buffer.data(), std::min(buffer.size(), room));
        if (count > 0) {
          std::vector<uint8_t> chunk(buffer.begin(), buffer.begin() + count);
          {
            std::lock_guard lock(mutex_);
            queuedBytes_ += chunk.size();
            chunks_.push_back(std::move(chunk));
          }
          wake_->notify();
          continue;
        }
        if (count == 0)
          return finish(0);
        if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
          return finish(errno);
      }
    } catch (const std::bad_alloc &) {
      finish(ENOMEM);
    }
  }

  void finish(int error) noexcept {
    {
      std::lock_guard lock(mutex_);
      end_ = error;
    }
    if (error)
      wake_->notify();
    else
      wake_->inputEnded();
  }

  int fd_ = -1;
  size_t queueBytes_ = 0;
  std::shared_ptr<LoopWake> wake_;
  int stopRead_ = -1;
  int stopWrite_ = -1;
  std::mutex mutex_;
  std::condition_variable room_;
  std::deque<std::vector<uint8_t>> chunks_;
  size_t queuedBytes_ = 0;
  std::optional<int> end_;
  bool stopping_ = false;
  std::thread thread_;
};

FdTransport::FdTransport(int inputFd, int outputFd, size_t inputQueueBytes)
    : inputFd_(inputFd), outputFd_(outputFd), wake_(std::make_shared<LoopWake>()) {
  if (inputFd_ < 0 || outputFd_ < 0 || !inputQueueBytes) {
    throw std::invalid_argument(
        "native transport requires valid fds and room for input");
  }
  // Nonblocking so that a read after a spurious readiness returns EAGAIN and
  // the reader goes back to poll, where its stop pipe can end it; without it
  // ~InputReader could wait forever in read().
  inputFlags_ = fcntl(inputFd_, F_GETFL);
  if (inputFlags_ < 0)
    throwIo("fcntl(F_GETFL)");
  if (fcntl(inputFd_, F_SETFL, inputFlags_ | O_NONBLOCK) < 0)
    throwIo("fcntl(F_SETFL)");
  try {
    reader_ = std::make_unique<InputReader>(inputFd_, inputQueueBytes, wake_);
  } catch (...) {
    static_cast<void>(fcntl(inputFd_, F_SETFL, inputFlags_));
    throw;
  }
}

FdTransport::~FdTransport() {
  reader_.reset();
  static_cast<void>(fcntl(inputFd_, F_SETFL, inputFlags_));
}

NativeRuntime::ByteSink FdTransport::outputSink() {
  return [this](std::span<const uint8_t> bytes) { writeAll(bytes); };
}

std::function<void()> FdTransport::controlNotifier() {
  std::shared_ptr<LoopWake> wake = wake_;
  return [wake] { wake->notifyControl(); };
}

void FdTransport::setControlHandler(ControlHandler handler) {
  controlHandler_ = std::move(handler);
}

NativeProcessExit FdTransport::run(NativeRuntime &loop) {
  std::shared_ptr<LoopWake> wake = wake_;
  loop.setCompletionNotifier([wake] { wake->notify(); });
  // From here the loop takes the input's end (LoopWake::inputEnded).
  wake->inputTaken.store(true, std::memory_order_release);
  bool deferredControl = false;

  while (!loop.connectionMustClose()) {
    if (shutdownRequested())
      return NativeProcessExit::CleanEof;
    const auto stepStarted = AwakeClock::now();
    {
      InputReader::Input input = reader_->take();
      for (; !input.chunks.empty(); input.chunks.pop_front())
        if (!loop.receive(input.chunks.front()))
          return loopEnded(loop);
      if (input.end) {
        if (*input.end == ENOMEM) {
          failure_ = "the native input reader ran out of memory";
          return NativeProcessExit::EngineFailure;
        }
        if (*input.end) {
          failure_ = "read(native input): " +
                     std::string(std::strerror(*input.end));
          return NativeProcessExit::IoFailure;
        }
        return loop.finishInput() ? NativeProcessExit::CleanEof
                                  : loopEnded(loop);
      }
    } // What was taken is freed before control, tick and poll.

    deferredControl = wake->takeControl() || deferredControl;
    if (deferredControl && !loop.commandInFlight()) {
      deferredControl = loop.runControl(controlHandler_);
    }

    const bool progressed = loop.tick();
    const double tickMilliseconds = millisecondsSince(stepStarted);
    if (tickMilliseconds > maxTickMilliseconds_)
      maxTickMilliseconds_ = tickMilliseconds;
    if (progressed)
      continue;
    if (loop.connectionMustClose())
      return loopEnded(loop);

    // Input, completions, control notifications and shutdown all wake the
    // loop through this pipe.
    pollfd descriptor{wake->readFd, POLLIN, 0};
    int result;
    do {
      result = poll(&descriptor, 1, pollTimeout(loop));
    } while (result < 0 && errno == EINTR);
    if (result < 0) {
      failure_ = "poll(loop wake): " + std::string(std::strerror(errno));
      return NativeProcessExit::IoFailure;
    }
    if (descriptor.revents & POLLIN)
      wake->drain();
    // A zero result is a deadline, health-check or admission-retry wake; the
    // next iteration takes input, runs control and ticks.
  }
  return loopEnded(loop);
}

bool FdTransport::runFlush(NativeRuntime &loop, std::chrono::milliseconds budget) {
  const auto deadline = AwakeClock::now() + budget;
  while (!loop.flushRestorePoints()) {
    const auto left = std::chrono::ceil<std::chrono::milliseconds>(
        deadline - AwakeClock::now());
    if (!loop.engineHealthy() || left.count() <= 0)
      return false;
    pollfd descriptor{wake_->readFd, POLLIN, 0};
    const int result = poll(&descriptor, 1, static_cast<int>(left.count()));
    if (result < 0 && errno != EINTR) {
      failure_ = "poll(loop wake): " + std::string(std::strerror(errno));
      return false;
    }
    if (result > 0 && (descriptor.revents & POLLIN))
      wake_->drain();
  }
  return true;
}

bool FdTransport::shutdownRequested() const noexcept {
  return wake_->shutdownRequested.load(std::memory_order_acquire);
}

double FdTransport::maxTickMilliseconds() const noexcept {
  return maxTickMilliseconds_;
}

const std::string &FdTransport::failure() const noexcept { return failure_; }

void FdTransport::requestShutdown() noexcept { wake_->requestShutdown(); }

NativeProcessExit FdTransport::loopEnded(const NativeRuntime &loop) {
  if (!loop.engineHealthy())
    return NativeProcessExit::EngineFailure;
  if (!outputError_)
    return NativeProcessExit::ProtocolFailure;
  // A server that closed its end or is stopping the engine ends it as a
  // shutdown, which still flushes a persistent cache.
  if (outputError_ == EPIPE || shutdownRequested())
    return NativeProcessExit::CleanEof;
  failure_ = "write(native output): " + std::string(std::strerror(outputError_));
  return NativeProcessExit::IoFailure;
}

void FdTransport::writeAll(std::span<const uint8_t> bytes) {
  size_t offset = 0;
  while (offset < bytes.size()) {
    // Once a shutdown is requested no write starts or resumes: the server
    // may have stopped draining the pipe, and its one stop signal has
    // already been spent, so nothing would end a blocked write.
    if (shutdownRequested())
      outputFailed(EINTR);
    const ssize_t count =
        write(outputFd_, bytes.data() + offset, bytes.size() - offset);
    if (count > 0) {
      offset += static_cast<size_t>(count);
      continue;
    }
    if (count < 0 && errno == EINTR)
      continue;
    outputFailed(count < 0 ? errno : EIO);
  }
}

void FdTransport::outputFailed(int error) {
  outputError_ = error;
  throw std::system_error(error, std::generic_category(), "write(native output)");
}

} // namespace splash::engine
