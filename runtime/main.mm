#include "AwakeClock.hpp"
#include "LegacyWeightCache.hpp"
#include "StderrLine.hpp"
#include "Supervision.hpp"
#include "engine/FdTransport.hpp"
#include "engine/Bootstrap.hpp"
#include "engine/Json.hpp"
#include "engine/NativeArguments.hpp"
#include "model/ModelDescriptor.hpp"

#import <AVFoundation/AVFoundation.h>
#import <VideoToolbox/VideoToolbox.h>
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <dispatch/dispatch.h>
#include <mach-o/dyld.h>
#include <mach/mach_error.h>
#include <sysexits.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cerrno>
#include <csignal>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits.h>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
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
  const std::string command(executable);
  writeStderrLine(
      "usage: " + command +
      " serve-native MODEL_DIRECTORY"
      " MAX_CONTEXT|auto MAX_MEMORY_BYTES|auto [MAX_CACHE_DISK_BYTES]"
      " [--kv-format int8|bf16] [--state-format fp32|bf16] [--cached-states int8|exact] [--decode-share SHARE]"
      " [--max-image-patches PATCHES] [--persistent-cache on|off] [--cache-dir DIRECTORY]"
      " [--max-cache-disk-writes BYTES]"
      " [--idle-release SECONDS|off] [--idle-sleep prevent|allow] [--ane on|off]"
      " [--activations auto|bf16] [--relaunch-context TOKENS] [--ple-memory BYTES|auto|off]"
      " [--expert-memory BYTES|auto|off] [--sparse-kv-memory BYTES|auto]"
      " [--kv-spill always] [--kv-window-trim on|off] [--verify-rows auto|all]");
  writeStderrLine("       " + command + " model-check mlx-affine none|safetensors CONFIG [DRAFT_CONFIG]");
  writeStderrLine("       " + command + " model-check gguf none|gguf CONFIG GGUF_HEADERS [DRAFT_CONFIG]");
  writeStderrLine("       " + command + " device-check");
  writeStderrLine("       " + command + " video-frames VIDEO FPS MAX_FRAMES");
  writeStderrLine("       " + command + " families");
}

// Reports the exception in flight on stderr, as every command ends on one,
// and returns its exit status from sysexits(3): EX_USAGE, after the usage,
// for a command line; EX_DATAERR for a model root it cannot serve; EX_IOERR
// for an I/O failure; else EX_SOFTWARE.
int reportFailure(std::string_view executable) {
  try {
    throw;
  } catch (const engine::UsageError &error) {
    writeStderrLine(std::string("error: ") + error.what());
    printUsage(executable);
    return EX_USAGE;
  } catch (const engine::ModelRootError &error) {
    writeStderrLine(std::string("error: ") + error.what());
    return EX_DATAERR;
  } catch (const std::system_error &error) {
    writeStderrLine(std::string("error: native runtime I/O failed: ") + error.what());
    return static_cast<int>(engine::NativeProcessExit::IoFailure);
  } catch (const std::exception &error) {
    writeStderrLine(std::string("error: ") + error.what());
    return static_cast<int>(engine::NativeProcessExit::EngineFailure);
  }
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

// Splash's caches in the home directory; empty without one.
std::filesystem::path splashCaches() {
  const char *home = std::getenv("HOME");
  return home && *home ? std::filesystem::path(home) / "Library/Caches/Splash"
                       : std::filesystem::path();
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
  config.resources.cacheDisk = arguments.maxCacheDisk;
  config.resources.cacheDiskWritesPerDay = arguments.maxCacheDiskWritesBytes;
  config.resources.persistentCache = arguments.persistentCache;
  config.resources.cacheRoot = arguments.cacheRoot;
  if (const std::filesystem::path caches = splashCaches();
      config.resources.cacheRoot.empty() && !caches.empty())
    config.resources.cacheRoot = caches / "prefix-cache";
  config.resources.kvFormat = arguments.kvFormat;
  config.resources.stateFormat = arguments.stateFormat;
  config.resources.exactCachedStates = arguments.exactCachedStates;
  config.resources.maximumImagePatches = arguments.maxImagePatches;
  config.resources.idleReleaseSeconds = arguments.idleReleaseSeconds;
  config.resources.aneFfn.enabled = arguments.neuralEngine;
  config.resources.relaunchContext = arguments.relaunchContext;
  config.resources.pleMemory = arguments.pleMemory;
  config.resources.expertMemory = arguments.expertMemory;
  config.resources.sparseKvMemory = arguments.sparseKvMemory;
  config.resources.kvSpillAlways = arguments.kvSpillAlways;
  config.resources.kvWindowTrim = arguments.kvWindowTrim;
  config.resources.verifyRows = arguments.verifyRows;
  config.resources.int8Activations = arguments.int8Activations;
  config.nativeLoop.engine.maxContext = arguments.maxContext;
  config.nativeLoop.engine.decodeShare = arguments.decodeShare;
  return config;
}

// SIGTERM, SIGINT and SIGHUP end the transport loop instead of killing the
// process, so the normal destructors run. An inherited ignored SIGHUP (nohup)
// stays ignored, as it does for the server. SIGPIPE is ignored: a closed
// parent pipe surfaces as EPIPE, which ends the transport as a shutdown.
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

// The time the newest restore points of a persistent cache get to reach the
// disk once the transport has ended, before the cache closes; the server
// waits supervision::kStop for the engine to exit.
constexpr auto kFlushBudget = supervision::kFlush;

// A clean close of a persistent cache, however the transport ended: the
// newest restore points reach the disk, then the cache closes. An engine
// that fails meanwhile writes nothing more, and its cache counts as ending
// uncleanly.
void closePersistentCache(engine::FdTransport &transport,
                          engine::ServingRuntime &runtime) {
  engine::NativeRuntime &loop = runtime.nativeLoop();
  const bool flushed = transport.runFlush(loop, kFlushBudget);
  if (!loop.engineHealthy()) {
    writeStderrLine("error: the engine failed while it saved its newest restore points (" +
                    loop.engineFailure() + ")");
    return;
  }
  if (!flushed)
    logLine("Persistent cache: stopping before every newest restore point reached the disk.");
  if (!runtime.resources().closePersistentCache())
    writeStderrLine("error: the persistent cache did not reach the disk; the next start takes it "
                    "back on probation.");
}

// Ends a start that failed before Ready, as the exception in flight says: the
// report and exit status of any failure (reportFailure), then the failure's
// class in place of the ReadyEvent: Configuration for the command line, Model
// for the model root it names, the bootstrap's (engine::startFailedEvent),
// else Internal. The frame follows the report, so the server's line about it
// follows the report too.
int refuseStart(engine::FdTransport &transport, std::string_view executable) {
  const int exit = reportFailure(executable);
  using Kind = protocol::StartFailureKind;
  protocol::StartFailedEvent failure;
  try {
    throw;
  } catch (const engine::UsageError &error) {
    failure = {.kind = Kind::Configuration, .message = error.what()};
  } catch (const engine::ModelRootError &error) {
    failure = {.kind = Kind::Model, .message = error.what()};
  } catch (const engine::RuntimeBootstrapError &error) {
    failure = engine::startFailedEvent(error.report());
  } catch (const std::exception &error) {
    failure = {.kind = Kind::Internal, .message = error.what()};
  }
  const auto frame = protocol::serializeEvent(failure, protocol::ProtocolLimits{});
  if (!frame)
    return exit;
  try {
    transport.outputSink()(*frame.value);
  } catch (...) {
  }
  return exit;
}

// serve-native. Its transport is the server's pipes, open before the command
// line is read, so that every start that fails before Ready tells the server
// why (refuseStart), and a start whose server has gone stops: the input's end
// before run() requests a shutdown (FdTransport).
int serveNative(int argc, char **argv) {
  if (const std::filesystem::path caches = splashCaches(); !caches.empty())
    reportLegacyWeightCache(caches / "weights");
  engine::FdTransport transport(STDIN_FILENO, STDOUT_FILENO);
  ShutdownSignals signals(transport);
  std::optional<MemoryPressureMonitor> pressureMonitor;
  IdleSleepAssertion idleSleep;

  std::unique_ptr<engine::ServingRuntime> runtime;
  try {
    const engine::NativeArguments arguments = engine::parseNativeArguments(argc, argv);
    pressureMonitor.emplace(transport.controlNotifier());
    engine::StartupRetryWindow recovery(supervision::kStartRetryWindow);
    bool reportedRecoveryWait = false;
    while (!runtime) {
      if (transport.shutdownRequested())
        return static_cast<int>(engine::NativeProcessExit::CleanEof);
      engine::RuntimeBootstrapConfig config = bootstrapConfig(arguments);
      config.resources.memoryPressure = [&] { return pressureMonitor->pressure(); };
      config.resources.cancelled = [&] { return transport.shutdownRequested(); };
      if (arguments.preventIdleSleep)
        config.nativeLoop.holdingRequests = [&](bool held) { idleSleep.hold(held); };
      try {
        runtime = engine::ServingRuntime::start(std::move(config), transport.outputSink(),
                                                [&transport] { return transport.maxTickMilliseconds(); });
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
        const auto resumeAt =
            std::min(now + engine::kStartupMemoryRecoveryPoll, *recoveryDeadline);
        while (AwakeClock::now() < resumeAt &&
               !transport.shutdownRequested()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
      }
    }
  } catch (const std::exception &) {
    return refuseStart(transport, argv[0]);
  }
  // Ready follows the start, out of refuseStart's reach: the server takes a
  // StartFailed after Ready for a protocol failure, so a failure from Ready
  // on ends the process with its report alone.
  try {
    runtime->announceReady();
  } catch (const std::exception &) {
    // A stop requested meanwhile ends the process as a stop: it cuts the
    // Ready write short (FdTransport::writeAll), which is no failure.
    if (transport.shutdownRequested())
      return static_cast<int>(engine::NativeProcessExit::CleanEof);
    throw;
  }

  transport.setControlHandler([&pressureMonitor, &runtime] {
    return runtime->controlPass(pressureMonitor->pressure());
  });
  const auto exit = transport.run(runtime->nativeLoop());
  switch (exit) {
  case engine::NativeProcessExit::CleanEof:
    break;
  case engine::NativeProcessExit::ProtocolFailure:
    writeStderrLine(
        "error: native transport stopped after a protocol failure");
    break;
  case engine::NativeProcessExit::EngineFailure:
    writeStderrLine(
        "error: native transport stopped after an engine failure (" +
        (transport.failure().empty() ? runtime->nativeLoop().engineFailure()
                                     : transport.failure()) +
        ")");
    // A command the backend gave up on may never complete; teardown would
    // wait for it. The OS and the driver reclaim everything, as after
    // SIGKILL.
    if (!runtime->resources().backend().healthy()) {
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
  // Not only after CleanEof: a protocol or I/O failure leaves the engine
  // healthy, and its newest restore points still reach the disk.
  if (runtime->resources().cache().persistent() && runtime->nativeLoop().engineHealthy())
    closePersistentCache(transport, *runtime);
  return static_cast<int>(exit);
}

// The engine's device rule, which the launcher runs before any download:
// serve-native applies it only once the model is prepared.
int checkDevice(int argc) {
  if (argc != 2)
    throw engine::UsageError("device-check takes no arguments");
  const auto message = metal::probeDeviceCapabilities().validationMessage();
  if (!message)
    return 0;
  writeStderrLine("error: " + *message);
  return static_cast<int>(engine::NativeProcessExit::EngineFailure);
}

// The engine's rules for an upstream model's configuration, which the
// installer applies before any weight download: the family it describes, as
// JSON, or the refusal (model::inspectSourceConfiguration).
int checkModel(int argc, char **argv) {
  // The draft's config, when given, follows the target's: its config, and a
  // GGUF's directory of headers after it.
  const int draftIndex = argc > 2 && std::string_view(argv[2]) == "gguf" ? 6 : 5;
  if (argc != draftIndex && argc != draftIndex + 1)
    throw engine::UsageError(
        "model-check takes the source formats, the target's config, a GGUF target's headers and the draft's config");
  const auto path = [&](int index) {
    return index < argc ? std::optional<std::filesystem::path>(argv[index]) : std::nullopt;
  };
  const std::string_view family = model::inspectSourceConfiguration(
      argv[2], argv[3], argv[4], draftIndex == 6 ? path(5) : std::nullopt, path(draftIndex));
  std::printf("{\"family\":\"%.*s\"}\n", static_cast<int>(family.size()), family.data());
  return 0;
}

// The frames vLLM's Nemotron loader samples: uniform over the clip, at most
// FPS per second and MAX_FRAMES, by index into its frames in presentation
// order. The server runs this in a disposable process, since the video is
// untrusted. stdout: one JSON line (width, height, source fps, frame count,
// sampled indices), then each sampled frame's RGB rows.
// Decoding is software: the hardware decoder waits indefinitely for memory
// while a large model holds most of it wired (Super on 64 GB), and software
// decodes 1080p H.264 at about 300 frames a second.
// ponytail: ignores the track's rotation; portrait phone clips come out sideways.
int extractVideoFrames(int argc, char **argv) {
  if (argc != 5)
    throw engine::UsageError("video-frames takes the video, the sampling fps and the frame cap");
  const double fps = std::atof(argv[3]);
  const long cap = std::atol(argv[4]);
  if (!(fps > 0) || cap < 1)
    throw engine::UsageError("video-frames needs a positive fps and frame cap");
  AVURLAsset *asset = [AVURLAsset
      URLAssetWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]]
              options:nil];
  __block AVAssetTrack *track = nil;
  dispatch_semaphore_t loaded = dispatch_semaphore_create(0);
  [asset loadTracksWithMediaType:AVMediaTypeVideo
               completionHandler:^(NSArray<AVAssetTrack *> *tracks, NSError *) {
                 track = tracks.firstObject;
                 dispatch_semaphore_signal(loaded);
               }];
  dispatch_semaphore_wait(loaded, DISPATCH_TIME_FOREVER);
  AVAssetTrack *const video = track;
  if (!video)
    throw std::runtime_error("the video has no video track");
  // Its compressed samples, in decode order.
  const auto samples = [&] {
    NSError *error = nil;
    AVAssetReader *reader = [AVAssetReader assetReaderWithAsset:asset error:&error];
    AVAssetReaderTrackOutput *output =
        [AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack:video outputSettings:nil];
    if (!reader || ![reader canAddOutput:output])
      throw std::runtime_error("the video cannot be decoded");
    [reader addOutput:output];
    if (![reader startReading])
      throw std::runtime_error("the video cannot be decoded");
    return std::pair{reader, output};
  };
  // Every frame's presentation time, after the track's edits, as the decoder
  // reports it, in the track's time scale: the decoder's own may differ, and
  // the container's frame count is an estimate.
  const int32_t scale = video.naturalTimeScale > 0 ? video.naturalTimeScale : 600;
  const auto tick = [scale](CMTime time) {
    return CMTimeConvertScale(time, scale, kCMTimeRoundingMethod_RoundHalfAwayFromZero).value;
  };
  std::vector<int64_t> times;
  // The decoded frames' size, as OpenCV reads it: the track's naturalSize is
  // the display size from the track header, which can be off by a fraction
  // (568.9 for 568 coded) or scaled by the pixel aspect ratio.
  CMVideoDimensions dimensions{0, 0};
  // Per sample buffer, in decode order: whether decoding can start there, and
  // its earliest presentation time; per frame, the buffer that holds it.
  std::vector<bool> sync;
  std::vector<int64_t> earliest;
  std::map<int64_t, size_t> holder;
  {
    auto [reader, output] = samples();
    while (CMSampleBufferRef sample = [output copyNextSampleBuffer]) {
      if (CMFormatDescriptionRef format = CMSampleBufferGetFormatDescription(sample); format && !dimensions.width)
        dimensions = CMVideoFormatDescriptionGetDimensions(format);
      const CMItemCount count = CMSampleBufferGetNumSamples(sample);
      std::vector<CMSampleTimingInfo> timing(static_cast<size_t>(count));
      int64_t first = INT64_MAX;
      if (count && CMSampleBufferGetOutputSampleTimingInfoArray(sample, count, timing.data(), nullptr) == noErr)
        for (const CMSampleTimingInfo &info : timing) {
          // An hour at 240 fps is 864,000; past that, refuse before the
          // per-frame tables grow with an untrusted file.
          if (times.size() == 1'000'000)
            throw std::runtime_error("the video has too many frames");
          times.push_back(tick(info.presentationTimeStamp));
          holder.emplace(times.back(), sync.size());
          first = std::min(first, times.back());
        }
      CFArrayRef attachments = CMSampleBufferGetSampleAttachmentsArray(sample, false);
      const auto notSync = attachments && CFArrayGetCount(attachments)
          ? static_cast<CFBooleanRef>(CFDictionaryGetValue(
                static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(attachments, 0)), kCMSampleAttachmentKey_NotSync))
          : nullptr;
      sync.push_back(!(notSync && CFBooleanGetValue(notSync)));
      earliest.push_back(first);
      CFRelease(sample);
    }
    if (reader.status != AVAssetReaderStatusCompleted)
      throw std::runtime_error("the video cannot be decoded");
  }
  std::sort(times.begin(), times.end());
  // A frame is found by its time alone, so two frames may not share one.
  if (std::adjacent_find(times.begin(), times.end()) != times.end())
    throw std::runtime_error("the video has frames sharing a presentation time");
  const long total = static_cast<long>(times.size());
  const double sourceFps = video.nominalFrameRate;
  // Each side alone first: the container's size is untrusted and the
  // product could overflow.
  if (total < 1 || !(sourceFps >= 0.1) || !(dimensions.width >= 1 && dimensions.width <= 16384) ||
      !(dimensions.height >= 1 && dimensions.height <= 16384))
    throw std::runtime_error("the video has no frames, no frame rate or an unsupported size");
  const long width = dimensions.width;
  const long height = dimensions.height;
  if (width * height > 4096L * 4096L)
    throw std::runtime_error("the video has no frames, no frame rate or an unsupported size");
  // vLLM's VideoBackend: min(cap, frames, floor(duration * fps)) frames at
  // np.linspace(0, frames - 1, n, dtype=int). The rate is a float's, 30 fps
  // as 30.000002: the floor allows for it, as OpenCV's exact 30/1 would.
  // Compared as doubles: an odd rate would overflow a long.
  const long count = static_cast<long>(std::max(1.0, std::min({static_cast<double>(cap),
      static_cast<double>(total), std::floor(static_cast<double>(total) / sourceFps * fps + 1e-4)})));
  std::vector<long> indices(count);
  const double step = count > 1 ? static_cast<double>(total - 1) / static_cast<double>(count - 1) : 0;
  for (long i = 0; i < count; ++i)
    indices[i] = count > 1 && i + 1 == count ? total - 1 : static_cast<long>(static_cast<double>(i) * step);
  std::string header = "{\"width\":" + std::to_string(width) + ",\"height\":" + std::to_string(height) +
      ",\"fps\":" + std::to_string(sourceFps) + ",\"frames\":" + std::to_string(total) + ",\"indices\":[";
  for (long i = 0; i < count; ++i)
    header += (i ? "," : "") + std::to_string(indices[i]);
  header += "]}\n";
  std::fwrite(header.data(), 1, header.size(), stdout);

  // Decoded frames arrive in decode order; each sampled one waits here until
  // those before it in presentation order are written.
  struct Output {
    std::map<int64_t, size_t> wanted; // presentation time -> output position
    std::map<size_t, std::vector<uint8_t>> pending;
    size_t next = 0;
    long width = 0, height = 0;
    bool failed = false;
    // The handler may run on a decoder thread while this one reads.
    std::mutex lock;
  } out;
  for (size_t i = 0; i < indices.size(); ++i)
    out.wanted.emplace(times[static_cast<size_t>(indices[i])], i);
  // Only the buffers a sampled frame needs: from the sync buffer before it,
  // or the one before that when the frame shows earlier than its sync buffer
  // (a leading picture of an open GOP), through its own. An hour at 30 fps
  // decodes 128 short runs instead of 100,000 frames.
  std::vector<bool> needed(sync.size(), false);
  for (const auto &[time, position] : out.wanted) {
    const size_t own = holder.at(time);
    size_t start = own;
    while (start > 0 && !sync[start])
      --start;
    if (time < earliest[start] && start > 0)
      for (--start; start > 0 && !sync[start]; --start) {
      }
    for (size_t b = start; b <= own; ++b)
      needed[b] = true;
  }
  out.width = width, out.height = height;
  Output *const state = &out;
  VTDecompressionSessionRef session = nullptr;
  auto [reader, output] = samples();
  size_t buffer = 0;
  bool skipped = false;
  while (CMSampleBufferRef sample = [output copyNextSampleBuffer]) {
    // The file changed between the passes: decode everything rather than guess.
    const bool wanted = buffer >= needed.size() || needed[buffer];
    ++buffer;
    if (!wanted) {
      skipped = true;
      CFRelease(sample);
      continue;
    }
    // After a gap, let the frames the decoder holds for reordering out first.
    if (skipped && session) {
      VTDecompressionSessionFinishDelayedFrames(session);
      VTDecompressionSessionWaitForAsynchronousFrames(session);
    }
    skipped = false;
    CMFormatDescriptionRef format = CMSampleBufferGetFormatDescription(sample);
    if (format && CMSampleBufferGetNumSamples(sample)) {
      if (!session) {
        NSDictionary *software =
            @{(id)kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder : @NO};
        NSDictionary *pixels = @{(id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)};
        if (VTDecompressionSessionCreate(nullptr, format, (__bridge CFDictionaryRef)software,
                                         (__bridge CFDictionaryRef)pixels, nullptr, &session) != noErr) {
          CFRelease(sample);
          throw std::runtime_error("the video's codec is not supported");
        }
      }
      VTDecompressionSessionDecodeFrameWithOutputHandler(
          session, sample, 0, nullptr,
          ^(OSStatus status, VTDecodeInfoFlags, CVImageBufferRef image, CMTime time, CMTime) {
            const std::lock_guard guard(state->lock);
            const auto wanted = state->wanted.find(tick(time));
            if (wanted == state->wanted.end())
              return;
            if (status != noErr || !image ||
                static_cast<long>(CVPixelBufferGetWidth(image)) != state->width ||
                static_cast<long>(CVPixelBufferGetHeight(image)) != state->height) {
              state->failed = true;
              return;
            }
            std::vector<uint8_t> rgb(static_cast<size_t>(state->width * state->height * 3));
            CVPixelBufferLockBaseAddress(image, kCVPixelBufferLock_ReadOnly);
            const auto *base = static_cast<const uint8_t *>(CVPixelBufferGetBaseAddress(image));
            const size_t stride = CVPixelBufferGetBytesPerRow(image);
            for (long y = 0; y < state->height; ++y)
              for (long x = 0; x < state->width; ++x) {
                const uint8_t *bgra = base + y * stride + x * 4;
                uint8_t *pixel = rgb.data() + (y * state->width + x) * 3;
                pixel[0] = bgra[2], pixel[1] = bgra[1], pixel[2] = bgra[0];
              }
            CVPixelBufferUnlockBaseAddress(image, kCVPixelBufferLock_ReadOnly);
            state->pending.emplace(wanted->second, std::move(rgb));
            for (auto ready = state->pending.find(state->next); ready != state->pending.end();
                 ready = state->pending.find(state->next)) {
              std::fwrite(ready->second.data(), 1, ready->second.size(), stdout);
              state->pending.erase(ready);
              ++state->next;
            }
          });
    }
    CFRelease(sample);
    const std::lock_guard guard(out.lock);
    if (out.failed || out.next == indices.size())
      break;
  }
  if (session) {
    VTDecompressionSessionFinishDelayedFrames(session);
    VTDecompressionSessionWaitForAsynchronousFrames(session);
    VTDecompressionSessionInvalidate(session);
    CFRelease(session);
  }
  [reader cancelReading];
  if (out.failed || out.next != indices.size())
    throw std::runtime_error("the video could not be decoded to its counted frames");
  return std::fflush(stdout) == 0 ? 0 : static_cast<int>(engine::NativeProcessExit::IoFailure);
}

// What each family's layout gives the options a start takes
// (model::familyFacts), as JSON by family, which the installer reports to the
// app (install/families.py).
int printFamilies(int argc) {
  if (argc != 2)
    throw engine::UsageError("families takes no arguments");
  json::Writer w;
  w.object([&] {
    for (const model::FamilyFacts &facts : model::familyFacts())
      w.object(json::Key(facts.family), [&] {
        w.field("experts", facts.experts)
            .field("recurrent_state", facts.recurrentState)
            .field("ple_table", facts.pleTable)
            .field("sparse_kv", facts.sparseKv)
            .field("dense_ffn", facts.denseFfn)
            .field("context_tokens", facts.contextTokens)
            .field("max_context_tokens", facts.maximumContextTokens);
      });
  });
  std::printf("%s\n", w.str().c_str());
  return 0;
}

} // namespace
} // namespace splash

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      const std::string_view command = argc > 1 ? argv[1] : "";
      if (command == "device-check")
        return splash::checkDevice(argc);
      if (command == "model-check")
        return splash::checkModel(argc, argv);
      if (command == "families")
        return splash::printFamilies(argc);
      if (command == "video-frames")
        return splash::extractVideoFrames(argc, argv);
      if (command == "serve-native")
        return splash::serveNative(argc, argv);
      throw splash::engine::UsageError(command.empty() ? std::string("expected a command")
                                                       : "unknown command " + std::string(command));
    } catch (const std::exception &) {
      return splash::reportFailure(argc > 0 ? argv[0] : "splash");
    }
  }
}
