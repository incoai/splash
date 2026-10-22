#include "video/VideoRuntime.hpp"

#include "AwakeClock.hpp"
#include "StderrLine.hpp"
#include "engine/NativeArguments.hpp"
#include "engine/Protocol.hpp"
#include "metal/MetalBackend.hpp"
#include "video/TextEncoder.hpp"
#include "video/h3/h3.h"
#include "video/h3/model/Numpy.hpp"

#import <Foundation/Foundation.h>

#include <mach/mach.h>
#include <sysexits.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <csignal>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace splash::video {
namespace {

using protocol::VideoPhase;

// The DiT reads its block weights from SSD into a ring of this many slots:
// a 15 s clip fits in about 15 GB instead of 32 GB, at the same speed
// (runtime/video/VENDORED).
constexpr int32_t kStreamSlots = 8;

// A request stopped by its Cancel.
struct Cancelled final {};

// A failure of the vendored engine: its status and message.
class EngineFailure final : public std::runtime_error {
public:
  EngineFailure(h3_status status, const std::string &message) : std::runtime_error(message), status(status) {}
  h3_status status;
};

std::atomic<bool> gStopping{false};

void requestStop(int) { gStopping.store(true, std::memory_order_relaxed); }

// The process's peak physical footprint, Metal allocations included, as
// Activity Monitor's Memory column counts it.
uint64_t peakFootprint() {
  task_vm_info_data_t info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
    return 0;
  return std::max<uint64_t>(info.phys_footprint, info.ledger_phys_footprint_peak);
}

uint64_t microsSince(AwakeClock::time_point start) {
  return static_cast<uint64_t>(millisecondsSince(start) * 1000.0);
}

// Whole event frames on the protocol's descriptor, from either thread.
class EventWriter final {
public:
  explicit EventWriter(int fd) : fd_(fd) {}

  void send(const protocol::EngineEvent &event) {
    const auto frame = protocol::serializeEvent(event, limits_);
    if (!frame) throw std::logic_error("video event breaks the protocol: " + frame.issue->describe());
    std::lock_guard lock(mutex_);
    for (size_t written = 0; written < frame.value->size();) {
      const ssize_t count = ::write(fd_, frame.value->data() + written, frame.value->size() - written);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) throw std::system_error(errno, std::generic_category(), "video event write");
      written += static_cast<size_t>(count);
    }
  }

  void requestError(uint64_t requestId, std::string code, std::string message) {
    send(protocol::ErrorEvent{protocol::FailureClass::RequestError, requestId, false, std::move(code),
                              std::move(message)});
  }

private:
  int fd_;
  std::mutex mutex_;
  protocol::ProtocolLimits limits_;
};

struct Options {
  std::filesystem::path root;
  std::filesystem::path metallib;
  std::optional<std::vector<float>> hiddenStates;
  std::filesystem::path keep;
};

std::vector<float> readHiddenStates(const std::filesystem::path &path) {
  std::map<std::string, h3::model::NpyArray> npz;
  try {
    npz = h3::model::readNpz(path.string());
  } catch (const std::runtime_error &) {
  }
  const auto found = npz.find("hidden");
  if (found == npz.end() || found->second.dtype != "<f4" || found->second.shape.size() != 2 ||
      found->second.shape[1] != TextEncoder::kHiddenSize)
    throw std::invalid_argument(path.string() + " holds no f32 [L][5120] hidden");
  const auto *values = found->second.as<float>();
  return {values, values + found->second.count()};
}

Options parse(const std::filesystem::path &metallib, int argc, char **argv) {
  Options options{argv[2], metallib, std::nullopt, {}};
  for (int index = 3; index < argc; index += 2) {
    const std::string_view option(argv[index]);
    if (index + 1 == argc || (option != "--hidden-states" && option != "--keep"))
      throw engine::UsageError("a video model's engine takes --hidden-states NPZ and --keep DIRECTORY");
    if (option == "--hidden-states")
      options.hiddenStates = readHiddenStates(argv[index + 1]);
    else
      options.keep = argv[index + 1];
  }
  return options;
}

// The vendored engine for one request, destroyed on every path.
class H3Engine final {
public:
  H3Engine(const Options &options, const h3_request &request) {
    const std::string metallib = options.metallib.string(), weights = (options.root / "weights").string();
    const h3_config config{metallib.c_str(), weights.c_str()};
    check(h3_create(&config, &request, &engine_), "start the video engine");
  }
  ~H3Engine() { h3_destroy(engine_); }
  H3Engine(const H3Engine &) = delete;
  H3Engine &operator=(const H3Engine &) = delete;

  [[nodiscard]] h3_engine *get() const noexcept { return engine_; }
  void check(h3_status status, std::string_view what) const {
    if (status != H3_OK)
      throw EngineFailure(status, std::string(what) + ": " + (engine_ ? h3_error(engine_) : "invalid request or files"));
  }

private:
  h3_engine *engine_ = nullptr;
};

// One request's phases, each announced as it starts and as it completes.
class Request final {
public:
  Request(protocol::VideoRequestFrame frame, EventWriter &events)
      : frame(std::move(frame)), arrived(AwakeClock::now()), events_(events) {}

  void start(VideoPhase phase, uint32_t steps = 1) {
    if (cancelled.load(std::memory_order_relaxed)) throw Cancelled{};
    phase_ = phase;
    steps_ = steps;
    phaseStarted_ = AwakeClock::now();
    progress(0);
  }
  void progress(uint32_t step) {
    events_.send(protocol::VideoProgressEvent{frame.requestId, phase_, step, steps_, microsSince(phaseStarted_),
                                              peakFootprint()});
  }
  void complete() { progress(steps_); }
  void output(protocol::VideoOutputKind kind, uint32_t frames, std::vector<uint8_t> mp4) {
    events_.send(protocol::VideoOutputEvent{frame.requestId, kind, frames, microsSince(arrived), std::move(mp4)});
  }

  const protocol::VideoRequestFrame frame;
  const AwakeClock::time_point arrived;
  std::atomic<bool> cancelled{false};

private:
  EventWriter &events_;
  VideoPhase phase_ = VideoPhase::Encode;
  uint32_t steps_ = 1;
  AwakeClock::time_point phaseStarted_;
};

// An npz of named arrays (--keep).
void write(const std::filesystem::path &path, const std::vector<std::pair<std::string, h3::model::NpyArray>> &arrays) {
  std::vector<std::pair<std::string, const h3::model::NpyArray *>> members;
  for (const auto &[name, array] : arrays) members.emplace_back(name, &array);
  h3::model::writeNpz(path.string(), members);
}

void writeFile(const std::filesystem::path &path, const void *data, size_t bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
  if (!file.flush()) throw std::runtime_error("cannot write " + path.string());
}

// The mp4 of frames and, when given, the waveform, through a file of its own.
std::vector<uint8_t> mp4(const std::vector<uint8_t> &frames, int32_t count, const h3_geometry &geometry,
                         const std::vector<float> &wav, uint64_t requestId) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() /
                                     ("splash-video-" + std::to_string(getpid()) + "-" +
                                      std::to_string(requestId) + ".mp4");
  h3_mp4_stats stats{};
  const h3_status status =
      h3_write_mp4(path.c_str(), frames.data(), count, geometry.width, geometry.height,
                   wav.empty() ? nullptr : wav.data(), wav.empty() ? 0 : geometry.audio_samples, &stats);
  std::ifstream file(path, std::ios::binary);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
  if (status != H3_OK || bytes.empty()) throw EngineFailure(status, "cannot write the mp4");
  return bytes;
}

class VideoServer final {
public:
  VideoServer(Options options, EventWriter &events)
      : options_(std::move(options)), events_(events), backend_(options_.metallib.string()),
        encoder_(backend_, options_.root / "encoder") {}

  // Reads client frames from `input` until it ends or a stop signal; a
  // request runs on the worker meanwhile.
  int run(int input) {
    events_.send(protocol::ReadyEvent{1, TextEncoder::kMaximumTokens, false, true});
    std::thread worker([this] { work(); });
    const int status = read(input);
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
      if (running_) running_->cancelled.store(true, std::memory_order_relaxed);
    }
    wake_.notify_all();
    worker.join();
    return status;
  }

private:
  int read(int input) {
    protocol::FrameParser parser(limits_);
    std::array<uint8_t, 1 << 16> buffer;
    while (!gStopping.load(std::memory_order_relaxed)) {
      const ssize_t count = ::read(input, buffer.data(), buffer.size());
      if (count < 0 && errno == EINTR) continue;
      if (count < 0) return EX_IOERR;
      if (count == 0) return parser.finish() ? EX_PROTOCOL : EX_OK;
      for (std::span<const uint8_t> bytes(buffer.data(), static_cast<size_t>(count)); !bytes.empty();) {
        protocol::ParseStep step = parser.consume(bytes);
        bytes = bytes.subspan(step.consumedBytes);
        if (step.issue) return fatal(*step.issue);
        if (!step.frame) continue;
        auto message = protocol::decodeFrame(std::move(*step.frame), limits_);
        if (!message && message.issue->failureClass == protocol::FailureClass::RequestError)
          events_.requestError(message.issue->requestId, std::string(protocol::issueCodeName(message.issue->code)),
                               message.issue->message);
        else if (!message)
          return fatal(*message.issue);
        else if (auto *request = std::get_if<protocol::VideoRequestFrame>(&*message.value))
          admit(std::move(*request));
        else if (auto *cancel = std::get_if<protocol::CancelFrame>(&*message.value))
          this->cancel(cancel->requestId);
        else
          return fatal({protocol::FailureClass::ProtocolFatal, protocol::IssueCode::UnknownFrameType, 0,
                        "a video model's engine takes VideoRequest and Cancel frames"});
      }
    }
    return EX_OK;
  }

  int fatal(const protocol::ProtocolIssue &issue) {
    events_.send(protocol::ErrorEvent{issue.failureClass, 0, false, std::string(protocol::issueCodeName(issue.code)),
                                      issue.message});
    return EX_PROTOCOL;
  }

  // One request at a time: the server queues the others.
  void admit(protocol::VideoRequestFrame frame) {
    std::lock_guard lock(mutex_);
    if (pending_ || running_) {
      events_.send(protocol::ErrorEvent{protocol::FailureClass::RequestError, frame.requestId, true, "engine_busy",
                                        "the video engine runs one request at a time"});
      return;
    }
    pending_ = std::make_unique<Request>(std::move(frame), events_);
    wake_.notify_all();
  }

  void cancel(uint64_t requestId) {
    std::lock_guard lock(mutex_);
    for (Request *request : {pending_.get(), running_.get()})
      if (request && request->frame.requestId == requestId) request->cancelled.store(true, std::memory_order_relaxed);
  }

  void work() {
    while (true) {
      {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [this] { return stopping_ || pending_; });
        if (stopping_) return;
        running_ = std::move(pending_);
      }
      generate(*running_);
      std::lock_guard lock(mutex_);
      running_.reset();
    }
  }

  // Runs the request and answers it; a failure the engine cannot serve on
  // after ends the process.
  void generate(Request &request) {
    const uint64_t id = request.frame.requestId;
    try {
      try {
        @autoreleasepool {
          run(request);
        }
      } catch (const Cancelled &) {
        events_.requestError(id, "cancelled", "the video request was cancelled");
      } catch (const EngineFailure &error) {
        // A command buffer or device failure leaves the GPU in doubt.
        if (error.status == H3_ERROR_GPU || error.status == H3_ERROR_METAL) fail(error.what());
        events_.requestError(id, error.status == H3_ERROR_ARGUMENT ? "invalid_video_request" : "video_engine_failed",
                             error.what());
      } catch (const std::invalid_argument &error) {
        events_.requestError(id, "invalid_video_request", error.what());
      }
    } catch (const std::exception &error) {
      fail(error.what());
    }
  }

  // Reports the failure as the engine's, when the stream still takes it, and
  // exits: the server starts another engine.
  [[noreturn]] void fail(const std::string &message) {
    writeStderrLine("error: the video engine failed: " + message);
    try {
      events_.send(protocol::ErrorEvent{protocol::FailureClass::EngineUnhealthy, 0, false, "video_engine_failed",
                                        message});
    } catch (const std::exception &) {
    }
    _exit(EX_SOFTWARE);
  }

  void run(Request &request) {
    const protocol::VideoRequestFrame &frame = request.frame;
    h3_request clip = h3_default_request();
    clip.variant = frame.variant == protocol::VideoVariant::Turbo ? H3_VARIANT_TURBO : H3_VARIANT_DEFAULT;
    clip.frames = static_cast<int32_t>(frame.frames);
    clip.width = static_cast<int32_t>(frame.width);
    clip.height = static_cast<int32_t>(frame.height);
    clip.stream = 1;
    clip.stream_slots = kStreamSlots;
    h3_geometry geometry{};
    if (h3_geometry_of(&clip, &geometry) != H3_OK)
      throw std::invalid_argument("MiniMax-H3 renders 17 n + 5 frames of 1344 x 768");
    const std::filesystem::path keep =
        options_.keep.empty() ? std::filesystem::path() : options_.keep / std::to_string(frame.requestId);
    if (!keep.empty()) std::filesystem::create_directories(keep);

    request.start(VideoPhase::Encode);
    const std::vector<float> hidden =
        options_.hiddenStates ? *options_.hiddenStates : encoder_.encode(frame.promptTokens);
    const auto rows = static_cast<int64_t>(hidden.size() / TextEncoder::kHiddenSize);
    if (!keep.empty())
      write(keep / "hidden.npz", {{"hidden", h3::model::NpyArray::of("<f4", {rows, TextEncoder::kHiddenSize},
                                                                      hidden.data(), hidden.size() * 4)}});
    request.complete();

    H3Engine engine(options_, clip);
    std::vector<float> video(size_t(geometry.video_rows) * 96), audio(size_t(geometry.audio_rows) * 32);
    @autoreleasepool {
      request.start(VideoPhase::Prepare);
      // The keyframes arrive on the canvas (protocol::VideoRequestFrame): the video VAE encoder's input as is.
      auto keyframe = [](const std::vector<uint8_t> &rgb) { return rgb.empty() ? nullptr : rgb.data(); };
      const h3_prep_inputs inputs{hidden.data(),
                                  static_cast<int32_t>(rows),
                                  static_cast<int32_t>(frame.steps),
                                  frame.seed,
                                  nullptr,
                                  keyframe(frame.firstFrame),
                                  keyframe(frame.lastFrame)};
      h3_prepared prepared{};
      engine.check(h3_prep(engine.get(), &inputs, &prepared), "prepare the request");
      request.complete();

      request.start(VideoPhase::Denoise, static_cast<uint32_t>(prepared.steps));
      const h3_dit_inputs dit{prepared.text_rows, prepared.tags,      prepared.rope,
                              prepared.text,      prepared.cond_rows, prepared.cond};
      engine.check(h3_dit_load(engine.get(), &dit), "load the DiT");
      const h3_schedule schedule{prepared.steps,         prepared.ada_blocks,   prepared.ada_final,
                                 prepared.video_t,       prepared.audio_t,      prepared.video_sigmas,
                                 prepared.audio_sigmas};
      // A Cancel stops the loop at the next step: the step callback throws
      // through h3_denoise, which is C++ (CApi.mm), once the step's commands
      // have completed.
      const h3_step_callback onStep = [](void *user, int32_t step, const h3_forward_stats *, const float *,
                                         const float *) {
        auto &stepping = *static_cast<Request *>(user);
        stepping.progress(static_cast<uint32_t>(step + 1));
        if (stepping.cancelled.load(std::memory_order_relaxed)) throw Cancelled{};
      };
      engine.check(h3_denoise(engine.get(), &schedule, prepared.video_noise, prepared.audio_noise, video.data(),
                              audio.data(), onStep, &request),
                   "denoise");
      h3_dit_release(engine.get());
      h3_prep_release(engine.get());
    }
    if (!keep.empty())
      write(keep / "final.npz",
            {{"video", h3::model::NpyArray::of("<f4", {geometry.video_rows, 96}, video.data(), video.size() * 4)},
             {"audio", h3::model::NpyArray::of("<f4", {geometry.audio_rows, 32}, audio.data(), audio.size() * 4)}});

    if (frame.outputs & protocol::VideoOutputPreview) {
      @autoreleasepool {
        request.start(VideoPhase::Preview);
        std::vector<uint8_t> frames(size_t(geometry.width) * geometry.height * 3 * geometry.preview_frames);
        h3_preview_stats stats{};
        engine.check(h3_preview(engine.get(), video.data(), frames.data(), nullptr, &stats), "decode the preview");
        request.output(protocol::VideoOutputKind::Preview, static_cast<uint32_t>(geometry.preview_frames),
                       mp4(frames, geometry.preview_frames, geometry, {}, frame.requestId));
        request.complete();
      }
    }

    std::vector<uint8_t> frames(size_t(geometry.width) * geometry.height * 3 * geometry.decode_frames);
    @autoreleasepool {
      request.start(VideoPhase::Decode);
      h3_decode_stats stats{};
      engine.check(h3_decode(engine.get(), video.data(), frames.data(), &stats), "decode the video");
      if (!keep.empty()) writeFile(keep / "frames.rgb", frames.data(), frames.size());
      request.complete();
    }
    std::vector<float> wav;
    if (frame.outputs & protocol::VideoOutputAudio) {
      @autoreleasepool {
        request.start(VideoPhase::Audio);
        wav.resize(size_t(2) * geometry.audio_samples);
        h3_audio_stats stats{};
        engine.check(h3_decode_audio(engine.get(), audio.data(), wav.data(), &stats), "decode the audio");
        request.complete();
      }
    }
    request.start(VideoPhase::Mux);
    std::vector<uint8_t> clipMp4 = mp4(frames, geometry.decode_frames, geometry, wav, frame.requestId);
    request.complete();
    request.output(protocol::VideoOutputKind::Video, static_cast<uint32_t>(geometry.decode_frames),
                   std::move(clipMp4));
  }

  Options options_;
  EventWriter &events_;
  protocol::ProtocolLimits limits_;
  metal::MetalBackend backend_;
  TextEncoder encoder_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::unique_ptr<Request> pending_, running_;
  bool stopping_ = false;
};

} // namespace

bool servesVideo(const std::filesystem::path &modelRoot) {
  @autoreleasepool {
    NSData *data = [NSData dataWithContentsOfFile:@((modelRoot / "model.json").c_str())];
    NSDictionary *record = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    return [record isKindOfClass:[NSDictionary class]] && [record[@"target_format"] isEqual:@"h3-video"];
  }
}

int serve(const std::filesystem::path &metallib, int argc, char **argv) {
  struct sigaction action {};
  action.sa_handler = requestStop;
  sigemptyset(&action.sa_mask);
  for (const int number : {SIGTERM, SIGINT, SIGHUP}) sigaction(number, &action, nullptr);
  std::signal(SIGPIPE, SIG_IGN);
  if (metal::probeDeviceCapabilities().appleGpuFamily < 10) {
    writeStderrLine("error: the video engine runs on an Apple10 GPU (M5) or later");
    return EX_SOFTWARE;
  }
  EventWriter writer(STDOUT_FILENO);
  VideoServer server(parse(metallib, argc, argv), writer);
  return server.run(STDIN_FILENO);
}

} // namespace splash::video
