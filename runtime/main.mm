#include "AwakeClock.hpp"
#include "StderrLine.hpp"
#include "engine/FdTransport.hpp"
#include "engine/Bootstrap.hpp"
#include "engine/NativeArguments.hpp"
#include "engine/Status.hpp"

#include <IOKit/pwr_mgt/IOPMLib.h>
#include <dispatch/dispatch.h>
#include <mach-o/dyld.h>
#include <mach/mach_error.h>
#include <sysexits.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#ifndef SPLASH_BUILD_ID
#error "production build requires the generated BuildIdentity.hpp"
#endif

namespace splash {
namespace {

// Temporary host/driver allocation failures can recover during startup.
// Preserve the desktop reserve and bound retries; configuration and compute
// failures remain immediate and fail-closed.
constexpr auto kStartupMemoryRecoveryTimeout = std::chrono::seconds(30);
constexpr auto kStartupMemoryRecoveryPoll = std::chrono::seconds(1);

// One observer spans bootstrap and serving. The dispatch queue only records
// pressure and wakes control; all allocation/reclaim decisions stay on the
// native thread. RAII also covers failed or interrupted startup.
class MemoryPressureMonitor final {
public:
  explicit MemoryPressureMonitor(std::function<void()> notify)
      : pending_(std::make_shared<std::atomic<engine::MemoryPressure>>(
            engine::MemoryPressure::Normal)),
        queue_(dispatch_queue_create("com.splash.memory-pressure",
                                     DISPATCH_QUEUE_SERIAL)) {
    source_ = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
        DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN |
            DISPATCH_MEMORYPRESSURE_CRITICAL, queue_);
    if (!source_)
      throw std::runtime_error("unable to create memory-pressure monitor");
    const auto pending = pending_;
    const auto source = source_;
    dispatch_source_set_event_handler(source_, ^{
      const unsigned long event = dispatch_source_get_data(source);
      engine::MemoryPressure pressure = engine::MemoryPressure::Normal;
      if (event & DISPATCH_MEMORYPRESSURE_CRITICAL)
        pressure = engine::MemoryPressure::Critical;
      else if (event & DISPATCH_MEMORYPRESSURE_WARN)
        pressure = engine::MemoryPressure::Warning;
      pending->store(pressure, std::memory_order_release);
      notify();
    });
    dispatch_activate(source_);
    timer_ = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, queue_);
    if (!timer_) {
      dispatch_source_cancel(source_);
      dispatch_sync(queue_, ^{});
      throw std::runtime_error("unable to create memory-pressure timer");
    }
    // Notifications are coarse. The same safe-point control handler also
    // samples live host headroom twice a second, without dispatch-thread IO.
    dispatch_source_set_timer(
        timer_, dispatch_time(DISPATCH_TIME_NOW, 500 * NSEC_PER_MSEC),
        500 * NSEC_PER_MSEC, 100 * NSEC_PER_MSEC);
    dispatch_source_set_event_handler(timer_, ^{ notify(); });
    dispatch_activate(timer_);
  }
  ~MemoryPressureMonitor() {
    dispatch_source_cancel(timer_);
    dispatch_source_cancel(source_);
    dispatch_sync(queue_, ^{});
  }
  MemoryPressureMonitor(const MemoryPressureMonitor &) = delete;
  MemoryPressureMonitor &operator=(const MemoryPressureMonitor &) = delete;
  [[nodiscard]] engine::MemoryPressure pressure() const noexcept {
    // Notifications select individual processes and may arrive late. Sample
    // the current system level at the same safe points as host availability.
    return engine::querySystemMemoryPressure().value_or(
        pending_->load(std::memory_order_acquire));
  }

private:
  std::shared_ptr<std::atomic<engine::MemoryPressure>> pending_;
  dispatch_queue_t queue_;
  dispatch_source_t source_;
  dispatch_source_t timer_;
};

// Keeps the Mac from sleeping automatically while held, as `caffeinate -i`
// does: the display may still sleep, and the lid, the Sleep command or a low
// battery still sleep the Mac; requests continue when it wakes. Without the
// assertion the engine serves on and says so once.
class IdleSleepAssertion final {
public:
  IdleSleepAssertion() = default;
  ~IdleSleepAssertion() { hold(false); }
  IdleSleepAssertion(const IdleSleepAssertion &) = delete;
  IdleSleepAssertion &operator=(const IdleSleepAssertion &) = delete;

  void hold(bool held) noexcept {
    if (held == (assertion_ != kIOPMNullAssertionID))
      return;
    if (!held) {
      IOPMAssertionRelease(assertion_);
      assertion_ = kIOPMNullAssertionID;
      return;
    }
    const IOReturn result = IOPMAssertionCreateWithName(
        kIOPMAssertPreventUserIdleSystemSleep, kIOPMAssertionLevelOn,
        CFSTR("Splash is serving a request"), &assertion_);
    if (result == kIOReturnSuccess)
      return;
    assertion_ = kIOPMNullAssertionID;
    if (reported_)
      return;
    reported_ = true;
    logLine("Splash cannot keep the Mac awake while requests run (",
            mach_error_string(result), "); it may sleep during one.");
  }

private:
  IOPMAssertionID assertion_ = kIOPMNullAssertionID;
  bool reported_ = false;
};

void printUsage(std::string_view executable) {
  writeStderrLine(
      "usage: " + std::string(executable) +
      " serve-native MODEL_DIRECTORY"
      " MAX_CONTEXT|auto MAX_MEMORY_BYTES|auto [MAX_CACHE_DISK_BYTES]"
      " [--kv-format int8|bf16] [--decode-share SHARE]"
      " [--max-image-patches PATCHES] [--cache-dir DIRECTORY]"
      " [--idle-release SECONDS|off] [--idle-sleep prevent|allow]");
}

std::filesystem::path executablePath() {
  uint32_t size = PATH_MAX;
  std::vector<char> buffer(size);
  if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
    buffer.resize(size);
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
      throw std::runtime_error("could not resolve executable path");
    }
  }
  std::error_code error;
  std::filesystem::path path = std::filesystem::canonical(buffer.data(), error);
  if (error) {
    throw std::runtime_error("could not canonicalize executable path: " +
                             error.message());
  }
  return path;
}

engine::RuntimeBootstrapConfig
bootstrapConfig(const engine::NativeArguments &arguments) {
  engine::RuntimeBootstrapConfig config;
  config.resources.metallibPath =
      executablePath().parent_path() / "splash.metallib";
  config.resources.modelRoot = arguments.modelRoot;
  config.resources.model = arguments.model;
  config.resources.buildId = SPLASH_BUILD_ID;
  config.resources.maximumMemoryBytes = arguments.maxMemoryBytes;
  config.resources.maximumCacheDiskBytes = arguments.maxCacheDiskBytes;
  config.resources.persistentCacheRoot = arguments.persistentCacheRoot;
  config.resources.kvFormat = arguments.kvFormat;
  config.resources.maximumImagePatches = arguments.maxImagePatches;
  config.resources.idleReleaseSeconds = arguments.idleReleaseSeconds;
  config.nativeLoop.engine.maxContext = arguments.maxContext;
  config.nativeLoop.engine.decodeShare = arguments.decodeShare;
  return config;
}

// SIGTERM, SIGINT and SIGHUP end the transport loop instead of killing the
// process, so the normal destructors run. An inherited ignored SIGHUP (nohup)
// stays ignored, as it does for the server. SIGPIPE is ignored: a closed
// parent pipe surfaces as EPIPE, which the transport already reports as an
// I/O failure.
std::atomic<engine::FdTransport *> gShutdownTransport{nullptr};

void requestShutdownFromSignal(int) {
  const int savedErrno = errno;
  if (engine::FdTransport *transport =
          gShutdownTransport.load(std::memory_order_acquire)) {
    transport->requestShutdown();
  }
  errno = savedErrno;
}

// Keep shutdown idempotent through process teardown. Detach the transport before
// it is destroyed; later stop signals remain harmless until process exit.
class ShutdownSignals final {
public:
  explicit ShutdownSignals(engine::FdTransport &transport) {
    gShutdownTransport.store(&transport, std::memory_order_release);
    struct sigaction action {};
    action.sa_handler = requestShutdownFromSignal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    for (const int number : {SIGTERM, SIGINT, SIGHUP}) {
      struct sigaction inherited {};
      if (number == SIGHUP && sigaction(number, nullptr, &inherited) == 0 &&
          inherited.sa_handler == SIG_IGN)
        continue;
      sigaction(number, &action, nullptr);
    }
    std::signal(SIGPIPE, SIG_IGN);
  }
  ShutdownSignals(const ShutdownSignals &) = delete;
  ShutdownSignals &operator=(const ShutdownSignals &) = delete;
  ~ShutdownSignals() {
    gShutdownTransport.store(nullptr, std::memory_order_release);
  }
};

// The time a clean stop gives the newest restore points of a persistent
// cache to reach the disk before it closes the cache. The server waits
// longer for the engine to exit (server/runtime.py `_shutdown_grace_seconds`).
constexpr std::chrono::seconds kFlushBudget{6};

// A clean stop of a persistent cache: the newest restore points reach the
// disk, then the cache closes. An engine that fails meanwhile writes nothing
// more, and its cache counts as ending uncleanly.
void closePersistentCache(engine::FdTransport &transport,
                          engine::RuntimeBootstrap &bootstrap) {
  engine::NativeRuntime &loop = bootstrap.nativeLoop();
  const bool flushed = transport.runFlush(loop, kFlushBudget);
  if (!loop.engineHealthy()) {
    writeStderrLine("error: the engine failed while it saved its newest restore points (" +
                    loop.engineFailure() + ")");
    return;
  }
  if (!flushed)
    logLine("Persistent cache: stopping before every newest restore point reached the disk.");
  if (!bootstrap.resources().closePersistentCache())
    writeStderrLine("error: the persistent cache did not reach the disk; the next start takes it "
                    "back on probation.");
}

int runNative(const engine::NativeArguments &arguments) {
  engine::FdTransport transport(STDIN_FILENO, STDOUT_FILENO);
  ShutdownSignals signals(transport);
  MemoryPressureMonitor pressureMonitor(transport.controlNotifier());
  IdleSleepAssertion idleSleep;
  engine::RuntimeMetrics metrics;
  engine::RuntimeBootstrap *published = nullptr;
  auto statusProvider = [&] {
    return published->statusJson(
        metrics.snapshot(),
        engine::NativeLoopTiming{transport.maxTickMilliseconds()});
  };

  engine::StartupRetryWindow recovery(kStartupMemoryRecoveryTimeout);
  bool reportedRecoveryWait = false;
  std::unique_ptr<engine::RuntimeBootstrap> bootstrap;
  while (!bootstrap) {
    if (transport.shutdownRequested())
      return static_cast<int>(engine::NativeProcessExit::CleanEof);
    engine::RuntimeBootstrapConfig config = bootstrapConfig(arguments);
    config.resources.memoryPressure = [&] { return pressureMonitor.pressure(); };
    config.resources.cancelled = [&] { return transport.shutdownRequested(); };
    config.nativeLoop.metrics = &metrics;
    if (arguments.preventIdleSleep)
      config.nativeLoop.holdingRequests = [&](bool held) { idleSleep.hold(held); };
    try {
      bootstrap = engine::RuntimeBootstrap::start(
          std::move(config), transport.outputSink(), statusProvider);
    } catch (const engine::RuntimeBootstrapError &error) {
      if (transport.shutdownRequested())
        return static_cast<int>(engine::NativeProcessExit::CleanEof);
      const auto now = AwakeClock::now();
      const auto recoveryDeadline = recovery.retryUntil(error.report(), now);
      if (!recoveryDeadline)
        throw;
      if (!reportedRecoveryWait) {
        logLine("Waiting for sufficient available memory to start; "
                "the macOS reserve remains protected...");
        reportedRecoveryWait = true;
      }
      const auto resumeAt = std::min(now + kStartupMemoryRecoveryPoll, *recoveryDeadline);
      while (AwakeClock::now() < resumeAt &&
             !transport.shutdownRequested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
    }
  }
  if (transport.shutdownRequested())
    return static_cast<int>(engine::NativeProcessExit::CleanEof);
  published = bootstrap.get();

  transport.setControlHandler([&pressureMonitor, published] {
    return published->controlPass(pressureMonitor.pressure());
  });
  const auto exit = transport.run(bootstrap->nativeLoop());
  switch (exit) {
  case engine::NativeProcessExit::CleanEof:
    if (bootstrap->resources().cache().persistent())
      closePersistentCache(transport, *bootstrap);
    break;
  case engine::NativeProcessExit::ProtocolFailure:
    writeStderrLine(
        "error: native transport stopped after a protocol failure");
    break;
  case engine::NativeProcessExit::EngineFailure:
    writeStderrLine(
        "error: native transport stopped after an engine failure (" +
        (transport.failure().empty() ? bootstrap->nativeLoop().engineFailure()
                                     : transport.failure()) +
        ")");
    // A command the backend gave up on may never complete; teardown would
    // wait for it. The OS and the driver reclaim everything, as after
    // SIGKILL.
    if (!bootstrap->resources().backend().healthy()) {
      writeStderrLine(
          "error: the Metal backend is unhealthy; exiting without teardown");
      _exit(static_cast<int>(exit));
    }
    break;
  case engine::NativeProcessExit::IoFailure:
    writeStderrLine(
        "error: native transport stopped after an I/O failure (" +
        transport.failure() + ")");
    break;
  }
  return static_cast<int>(exit);
}

// The engine's device rule, which the launcher runs before any download:
// serve-native applies it only once the model is prepared.
int checkDevice() {
  const auto message = metal::probeDeviceCapabilities().validationMessage();
  if (!message)
    return 0;
  writeStderrLine("error: " + *message);
  return static_cast<int>(engine::NativeProcessExit::EngineFailure);
}

} // namespace
} // namespace splash

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string_view(argv[1]) == "device-check")
        return splash::checkDevice();
      const splash::engine::NativeArguments arguments =
          splash::engine::parseNativeArguments(argc, argv);
      return splash::runNative(arguments);
    } catch (const splash::engine::UsageError &error) {
      splash::writeStderrLine(std::string("error: ") + error.what());
      splash::printUsage(argc > 0 ? argv[0] : "splash");
      return EX_USAGE;
    } catch (const std::system_error &error) {
      splash::writeStderrLine(
          std::string("error: native runtime I/O failed: ") + error.what());
      return static_cast<int>(splash::engine::NativeProcessExit::IoFailure);
    } catch (const std::exception &error) {
      splash::writeStderrLine(std::string("error: ") + error.what());
      return static_cast<int>(
          splash::engine::NativeProcessExit::EngineFailure);
    }
  }
}
