#include "video/VideoRuntime.hpp"

#include "AwakeClock.hpp"
#include "StderrLine.hpp"
#include "engine/FdTransport.hpp"
#include "engine/NativeArguments.hpp"
#include "engine/Protocol.hpp"
#include "video/model/FastH3.hpp"
#include "video/model/TextEncoder.hpp"
#include "video/ops/WindowAttention.hpp"
#include "video/stages/AudioDecodingStage.hpp"
#include "video/stages/DecodingStage.hpp"
#include "video/stages/DenoisingStage.hpp"
#include "video/stages/MuxStage.hpp"
#include "video/stages/PrepStage.hpp"
#include "video/stages/PreviewStage.hpp"
#include "video/stages/TextEncodingStage.hpp"

#import <Foundation/Foundation.h>

#include <mach/mach.h>
#include <sysexits.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <condition_variable>
#include <csignal>
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

// One request: its frame, and the events of its stages, each announced as it starts and as it completes.
class Request final : public PipelineEvents {
public:
  Request(protocol::VideoRequestFrame frame, EventWriter &events)
      : frame(std::move(frame)), arrived(AwakeClock::now()), events_(events) {}

  void start(VideoPhase phase, uint32_t steps) override {
    phase_ = phase;
    steps_ = steps;
    phaseStarted_ = AwakeClock::now();
    progress(0);
  }
  void progress(uint32_t step) override {
    step_ = step;
    events_.send(protocol::VideoProgressEvent{frame.requestId, phase_, step, steps_, microsSince(phaseStarted_),
                                              peakFootprint()});
  }
  // The phase's last step, unless the stage reported it.
  void complete() override {
    if (step_ != steps_) progress(steps_);
  }
  void output(protocol::VideoOutputKind kind, uint32_t frames, std::vector<uint8_t> mp4) override {
    events_.send(protocol::VideoOutputEvent{frame.requestId, kind, frames, microsSince(arrived), std::move(mp4)});
  }
  [[nodiscard]] bool cancelled() const override { return cancel.load(std::memory_order_relaxed); }

  // Its references move into the batch (batchOf); the rest stays as it arrived.
  protocol::VideoRequestFrame frame;
  const AwakeClock::time_point arrived;
  std::atomic<bool> cancel{false};

private:
  EventWriter &events_;
  VideoPhase phase_ = VideoPhase::Encode;
  uint32_t steps_ = 1, step_ = 0;
  AwakeClock::time_point phaseStarted_;
};

// The batch of a request, its DiT under weights and its clip's geometry checked; the frame's references move into it.
VideoBatch batchOf(protocol::VideoRequestFrame &frame, const std::filesystem::path &weights) {
  VideoBatch batch;
  batch.requestId = frame.requestId;
  ClipOptions &options = batch.options;
  options.dit = frame.dit;
  std::error_code missing;
  if (!std::filesystem::is_directory(weights / model::ditDirectory(frame.dit), missing))
    throw std::invalid_argument("the weights hold no DiT " + frame.dit);
  options.frames = static_cast<int32_t>(frame.frames);
  options.width = static_cast<int32_t>(frame.width);
  options.height = static_cast<int32_t>(frame.height);
  if (std::string refusal = ops::canvasRefusal(options.width, options.height); !refusal.empty())
    throw std::invalid_argument(refusal);
  try {
    batch.geometry = ops::VideoGeometry::of(options.frames, options.width, options.height);
  } catch (const std::invalid_argument &) {
    throw std::invalid_argument("MiniMax-H3 renders 17 n + 5 frames");
  }
  batch.promptTokens = frame.promptTokens;
  batch.steps = static_cast<int32_t>(frame.steps);
  batch.seed = frame.seed;
  batch.firstFrame = frame.firstFrame;
  batch.lastFrame = frame.lastFrame;
  if (!frame.references.empty()) {
    if (!batch.firstFrame.empty() || !batch.lastFrame.empty())
      throw std::invalid_argument("a video request takes keyframes or references, not both");
    for (protocol::VideoReference &reference : frame.references) {
      ClipReference r;
      r.kind = reference.kind == protocol::VideoReferenceKind::Video   ? ops::ReferenceShape::Kind::Video
               : reference.kind == protocol::VideoReferenceKind::Audio ? ops::ReferenceShape::Kind::Audio
                                                                         : ops::ReferenceShape::Kind::Image;
      r.frames = static_cast<int32_t>(reference.frames);
      r.width = static_cast<int32_t>(reference.width);
      r.height = static_cast<int32_t>(reference.height);
      r.rgb = std::move(reference.pixels);
      r.wav = std::move(reference.soundtrack);
      batch.references.push_back(std::move(r));
    }
    frame.references.clear();
  }
  batch.preview = (frame.outputs & protocol::VideoOutputPreview) != 0;
  batch.audio = (frame.outputs & protocol::VideoOutputAudio) != 0;
  return batch;
}

class VideoServer final {
public:
  VideoServer(const VideoRuntime &runtime, EventWriter &events, std::filesystem::path weights)
      : runtime_(runtime), events_(events), weights_(std::move(weights)) {
    // A request carries its references' frames (protocol::kAbsoluteMaxVideoRequestPayloadBytes).
    limits_.maxVideoRequestPayloadBytes = protocol::kAbsoluteMaxVideoRequestPayloadBytes;
  }

  // Reads client frames from `input` until it ends or a stop signal; a
  // request runs on the worker meanwhile.
  int run(int input) {
    // The prompt bound: a presentation of kMaximumPresentationTokens, a Ref2VA one's with its references' vision
    // blocks; the server keeps the text rows within one pass.
    events_.send(protocol::ReadyEvent{.maxConcurrentRequests = 1,
                                      .maxContextTokens = model::TextEncoder::kMaximumPresentationTokens,
                                      .video = true});
    std::thread worker([this] { work(); });
    const int status = read(input);
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
      if (running_) running_->cancel.store(true, std::memory_order_relaxed);
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
      if (request && request->frame.requestId == requestId) request->cancel.store(true, std::memory_order_relaxed);
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

  // Runs the request and answers it; a failure the engine cannot serve on after ends the process.
  void generate(Request &request) {
    const uint64_t id = request.frame.requestId;
    try {
      try {
        @autoreleasepool {
          run(request);
        }
      } catch (const Cancelled &) {
        events_.requestError(id, "cancelled", "the video request was cancelled");
      } catch (const metal::MetalBackendError &error) {
        // A command buffer or device failure leaves the GPU in doubt.
        fail(error.what());
      } catch (const std::invalid_argument &error) {
        logLine("video request failed: ", error.what());
        events_.requestError(id, "invalid_video_request", error.what());
      } catch (const std::exception &error) {
        logLine("video request failed: ", error.what());
        events_.requestError(id, "video_engine_failed", error.what());
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
    VideoBatch batch = batchOf(request.frame, weights_);
    runtime_.forward(batch, request);
    request.output(protocol::VideoOutputKind::Video,
                   static_cast<uint32_t>(ops::planVaeDecode(batch.geometry).outputFrames()), std::move(batch.mp4));
  }

  const VideoRuntime &runtime_;
  EventWriter &events_;
  const std::filesystem::path weights_;
  protocol::ProtocolLimits limits_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::unique_ptr<Request> pending_, running_;
  bool stopping_ = false;
};

} // namespace

Stages VideoRuntime::stages(metal::MetalBackend &backend, const std::filesystem::path &modelRoot,
                            const std::optional<std::vector<model::AdapterSpec>> &adapters) {
  const std::filesystem::path weights = modelRoot / "weights";
  Stages stages;
  stages.push_back(std::make_unique<TextEncodingStage>(backend, modelRoot / "encoder"));
  stages.push_back(std::make_unique<PrepStage>(backend, weights, adapters));
  stages.push_back(std::make_unique<DenoisingStage>(backend, weights, adapters));
  stages.push_back(std::make_unique<PreviewStage>(backend, weights));
  stages.push_back(std::make_unique<DecodingStage>(backend, weights));
  stages.push_back(std::make_unique<AudioDecodingStage>(backend, weights));
  stages.push_back(std::make_unique<MuxStage>());
  return stages;
}

void VideoRuntime::forward(VideoBatch &batch, PipelineEvents &events) const {
  for (const std::unique_ptr<PipelineStage> &stage : stages_) {
    if (!stage->runs(batch)) continue;
    if (events.cancelled()) throw Cancelled{};
    @autoreleasepool {
      events.start(stage->phase(), stage->steps(batch));
      stage->forward(batch, events);
      events.complete();
    }
  }
}

bool servesVideo(const std::filesystem::path &modelRoot) {
  @autoreleasepool {
    NSData *data = [NSData dataWithContentsOfFile:@((modelRoot / "model.json").c_str())];
    NSDictionary *record = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    return [record isKindOfClass:[NSDictionary class]] && [record[@"target_format"] isEqual:@"h3-video"];
  }
}

int tabulateAdaLn(const std::filesystem::path &metallib, int argc, char **argv) {
  int32_t steps = 0;
  const std::string_view stepsText = argc == 6 ? argv[5] : "";
  const auto [stepsEnd, stepsError] = std::from_chars(stepsText.data(), stepsText.data() + stepsText.size(), steps);
  if ((argc != 5 && argc != 6) || !protocol::validDitName(argv[3]) ||
      (argc == 6 && (stepsError != std::errc() || stepsEnd != stepsText.data() + stepsText.size() || steps < 1)))
    throw engine::UsageError("video-adaln takes WEIGHTS NAME (its DiT, dit-NAME) TRANSFORMER [STEPS] (a VDN-H3 "
                             "DiT's denoising steps)");
  try {
    metal::MetalBackend backend(metallib.string());
    model::Prep::tabulateAdaLn(backend, std::string(argv[2]) + "/" + model::ditDirectory(argv[3]), argv[4], steps);
    return EX_OK;
  } catch (const std::exception &error) {
    writeStderrLine(std::string("error: ") + error.what());
    return EX_SOFTWARE;
  }
}

int serve(const std::filesystem::path &metallib, const std::filesystem::path &modelRoot,
          const std::function<Stages(metal::MetalBackend &)> &compose) {
  struct sigaction action {};
  action.sa_handler = requestStop;
  sigemptyset(&action.sa_mask);
  for (const int number : {SIGTERM, SIGINT, SIGHUP}) sigaction(number, &action, nullptr);
  std::signal(SIGPIPE, SIG_IGN);
  EventWriter writer(STDOUT_FILENO);
  std::unique_ptr<metal::MetalBackend> backend;
  std::optional<VideoRuntime> runtime;
  using Kind = protocol::StartFailureKind;
  Kind kind = Kind::Device;
  try {
    if (metal::probeDeviceCapabilities().appleGpuFamily < 10)
      throw std::runtime_error("the video engine runs on an Apple10 GPU (M5) or later");
    kind = Kind::Internal;
    backend = std::make_unique<metal::MetalBackend>(metallib.string());
    ops::checkRegisterLayout(*backend);
    runtime.emplace(compose(*backend));
  } catch (const std::exception &error) {
    // A start that fails reports on stderr, then tells the server its reason in place of the ReadyEvent, as a
    // language model's engine does (main.mm refuseStart).
    writeStderrLine(std::string("error: ") + error.what());
    try {
      writer.send(protocol::StartFailedEvent{.kind = kind, .message = error.what()});
    } catch (const std::exception &) {
    }
    return static_cast<int>(engine::NativeProcessExit::EngineFailure);
  }
  VideoServer server(*runtime, writer, modelRoot / "weights");
  return server.run(STDIN_FILENO);
}

int serve(const std::filesystem::path &metallib, int argc, char **argv) {
  if (argc != 3) throw engine::UsageError("a video model's engine takes its model directory alone");
  const std::filesystem::path modelRoot = argv[2];
  return serve(metallib, modelRoot,
               [&](metal::MetalBackend &backend) { return VideoRuntime::stages(backend, modelRoot); });
}

} // namespace splash::video
