#include "flash/FlashWorker.hpp"
#include "engine/Json.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Protocol.hpp"
#include "flash/FlashForward.hpp"
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashFloatDenseCache.hpp"
#include "flash/FlashBF16Q8Head.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "flash/FlashInt8ExpertStore.hpp"
#include "flash/FlashBatchForward.hpp"
#include "flash/FlashBatchPrefill.hpp"
#include "flash/FlashGDNBatchILP.hpp"
#include "flash/FlashGDNLazyRollback.hpp"
#include "flash/FlashBatchVerify.hpp"
#include "flash/FlashBatchMTPForward.hpp"
#include "flash/FlashMTP.hpp"
#include "flash/FlashPrefillDenseTiles.hpp"
#include "flash/FlashQSABulk.hpp"
#include "flash/FlashMTPDepth.hpp"
#include "flash/FlashMTPWindow.hpp"
#include "flash/FlashGreedy.hpp"
#include "flash/FlashGreedyGPU.hpp"
#include "flash/FlashWeights.hpp"
#include "flash/FlashOriginalResidency.hpp"
#include "flash/FlashRequestCommandTrace.hpp"
#include "flash/FlashIdleResidencyMaintenance.hpp"
#include "flash/FlashPLESSDStore.hpp"
#include "flash/FlashIdleResidencyScheduler.hpp"

#include <dispatch/dispatch.h>
#include <mach-o/dyld.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_set>
#include <variant>
#include <vector>

namespace splash::flash {
namespace {
using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;
namespace wire = protocol;
constexpr uint32_t kConcurrent = 4, kHeadRows = 128, kDefaultPrefillRows = 128;
constexpr uint32_t kMTPDepth = 3, kHyper = 10240;
constexpr uint32_t kDefaultBatchPrefillRows = 512;
constexpr uint64_t kJointHiddenBytes = uint64_t{kConcurrent} * (kMTPDepth + 1) * kHyper * 2;
constexpr uint64_t kBatchHeadPrimeInputBytes = uint64_t{kConcurrent} * kHeadRows * kHyper * 2;
constexpr const char *kWorkerMTPSemantics = "native-worker5-greedy-lightning-mtp3-committed-prefix-v1";
constexpr const char *kWorkerBatchHeadPrimeSemantics =
    "native-worker5-owned-real-head-priming-equal128-or-tiny-tail-v1";
constexpr size_t kPendingBound = 64, kControlBound = 256;
constexpr auto kAdmissionPressureRetry = std::chrono::milliseconds(250);
constexpr auto kAdmissionPressureLimit = std::chrono::seconds(2);
struct SingletonMTPPolicy final {
  uint32_t maximumDepth = kMTPDepth;
  bool explicitOverride = false;
  uint32_t hiddenRows() const { return std::max(maximumDepth, kMTPDepth) + 1; }
  uint64_t hiddenBytes() const { return uint64_t{hiddenRows()} * kHyper * 2; }
};
struct SavedOperandsResidencyStatus final {
  bool requested = false;
  bool supported = false;
  uint64_t requestedViews = 0;
  uint64_t requestedViewBytes = 0;
  std::string failureReason;
  bool originalRequested = false;
  bool originalAdded = false;
  FlashOriginalTextResidencySelection originalSelection;
  engine::MemoryGovernorSnapshot originalHostSnapshot{};
  std::string originalFailureReason;
};
SingletonMTPPolicy environmentSingletonMTPPolicy() {
  const char *value = std::getenv("SPLASH_FLASH_MTP_DRAFT_DEPTH");
  if (!value) return {};
  const auto depth = flashParseSingletonMTPDepth(value);
  if (!depth)
    throw std::invalid_argument("SPLASH_FLASH_MTP_DRAFT_DEPTH must be canonical ASCII decimal 1..15");
  return {*depth, true};
}
uint32_t singletonAllowedDepth(uint32_t configured, uint32_t remaining, bool peers) {
  if (!configured || configured > kFlashSingletonMaximumMTPDepth || !remaining)
    throw std::invalid_argument("invalid singleton MTP draft budget");
  return std::min(remaining - 1, peers ? std::min(configured, kMTPDepth) : configured);
}
bool jointFoldReady(uint32_t rows) { return rows && rows <= kMTPDepth + 1; }
volatile std::sig_atomic_t signalStop = 0;
void stopFromSignal(int) { signalStop = 1; }
uint64_t unixMicros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}
uint64_t micros(Time begin, Time end) {
  return end > begin ? std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() : 0;
}
double milliseconds(Time begin, Time end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}
template <typename T> T positive(std::string_view value, const char *label) {
  T result = 0;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !result)
    throw std::invalid_argument(std::string(label) + " must be auto or a positive integer");
  return result;
}
bool environmentSwitch(const char *name) {
  const char *value = std::getenv(name);
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}
uint32_t parsePrefillRows(std::string_view value) {
  uint32_t result = 0;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  constexpr std::array<uint32_t, 7> choices{32, 64, 128, 256, 512, 1024, 2048};
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
      std::find(choices.begin(), choices.end(), result) == choices.end())
    throw std::invalid_argument("SPLASH_FLASH_PREFILL_ROWS must be 32,64,128,256,512,1024 or 2048");
  return result;
}
uint32_t environmentPrefillRows() {
  const char *value = std::getenv("SPLASH_FLASH_PREFILL_ROWS");
  return value ? parsePrefillRows(value) : kDefaultPrefillRows;
}
uint32_t parseBatchPrefillRows(std::string_view value) {
  uint32_t rows = 0;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), rows);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
      (rows != 512 && rows != 1024 && rows != 2048))
    throw std::invalid_argument("SPLASH_FLASH_BATCH_PREFILL_ROWS must be 512,1024 or 2048");
  return rows;
}
uint32_t environmentBatchPrefillRows() {
  const char *value = std::getenv("SPLASH_FLASH_BATCH_PREFILL_ROWS");
  return value ? parseBatchPrefillRows(value) : kDefaultBatchPrefillRows;
}
uint64_t batchPrefillHiddenBytes(uint32_t rows) {
  if (!rows || rows > kFlashBatchPrefillMaximumRowsPerLane)
    throw std::invalid_argument("invalid Flash batch prefill hidden capacity");
  return uint64_t{kConcurrent} * rows * kHyper * 2;
}
std::filesystem::path executablePath() {
  uint32_t size = 4096;
  std::vector<char> storage(size);
  if (_NSGetExecutablePath(storage.data(), &size)) {
    storage.resize(size);
    if (_NSGetExecutablePath(storage.data(), &size))
      throw std::runtime_error("unable to locate Flash native executable");
  }
  return std::filesystem::canonical(storage.data());
}
uint64_t physicalMemory() {
  uint64_t value = 0;
  size_t bytes = sizeof(value);
  if (sysctlbyname("hw.memsize", &value, &bytes, nullptr, 0) || !value)
    throw std::runtime_error("unable to query physical memory");
  return value;
}
const char *pressureName(engine::MemoryPressure pressure) {
  switch (pressure) {
    case engine::MemoryPressure::Normal: return "normal";
    case engine::MemoryPressure::Warning: return "warning";
    case engine::MemoryPressure::Critical: return "critical";
  }
  return "critical";
}
wire::ProtocolLimits serveLimits(uint32_t capacity, uint32_t vocabulary, bool mtp = false,
                                uint32_t singletonMaximumDepth = kMTPDepth) {
  wire::ProtocolLimits limits;
  limits.maxPromptTokens = capacity; limits.maxLogicalOutputTokens = capacity;
  // The codec requires positive limits even for disabled input modalities.
  limits.maxImageSpans = 1;
  limits.maxTokenBatch = mtp ? std::max(singletonMaximumDepth, kMTPDepth) + 1 : 1;
  limits.maxSimulationTokens = 1;
  limits.maxMaskWords = (vocabulary + 31) / 32;
  limits.maxFramePayloadBytes = uint64_t{capacity} * 4 + uint64_t{limits.maxMaskWords} * 4 + 4096;
  limits.maxStatusJsonBytes = std::min<uint64_t>(32ULL << 20, limits.maxFramePayloadBytes - 12);
  limits.maxErrorStringBytes = std::min<uint64_t>(4096, limits.maxFramePayloadBytes / 4);
  return limits;
}
class PressureMonitor final {
public:
  PressureMonitor() : value_(std::make_shared<std::atomic<engine::MemoryPressure>>(
        engine::MemoryPressure::Normal)),
      queue_(dispatch_queue_create("com.splash.flash.memory-pressure", DISPATCH_QUEUE_SERIAL)) {
    source_ = dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
        DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN |
            DISPATCH_MEMORYPRESSURE_CRITICAL, queue_);
    if (!source_) throw std::runtime_error("unable to create memory pressure monitor");
    const auto value = value_;
    const auto source = source_;
    dispatch_source_set_event_handler(source_, ^{
      const auto data = dispatch_source_get_data(source);
      value->store(data & DISPATCH_MEMORYPRESSURE_CRITICAL ? engine::MemoryPressure::Critical
          : data & DISPATCH_MEMORYPRESSURE_WARN ? engine::MemoryPressure::Warning
          : engine::MemoryPressure::Normal, std::memory_order_release);
    });
    dispatch_activate(source_);
  }
  ~PressureMonitor() { dispatch_source_cancel(source_); dispatch_sync(queue_, ^{}); }
  engine::MemoryPressure value() const { return value_->load(std::memory_order_acquire); }
private:
  std::shared_ptr<std::atomic<engine::MemoryPressure>> value_;
  dispatch_queue_t queue_;
  dispatch_source_t source_;
};
struct Incoming {
  std::variant<wire::Message, wire::ProtocolIssue> value;
  Time arrived;
  bool maskIssue = false;
};

// Only main submits GPU commands. Reader answers status from a safe-point
// snapshot while main waits. All output is framed and serialized.
class Transport final {
public:
  explicit Transport(wire::ProtocolLimits limits) : limits_(limits) {
    // A full parent pipe must not block inside write() past the timeout below.
    const int flags = fcntl(STDOUT_FILENO, F_GETFL);
    if (flags < 0 || fcntl(STDOUT_FILENO, F_SETFL, flags | O_NONBLOCK) < 0)
      throw std::system_error(errno, std::generic_category(), "native stdout flags");
  }
  ~Transport() { stop(); if (reader_.joinable()) reader_.join(); }
  void start() { reader_ = std::thread([this] { readLoop(); }); }
  bool stopping() const { return stopped_.load(std::memory_order_acquire) || signalStop; }
  bool failed() const { return failed_.load(std::memory_order_acquire); }
  void stop() { stopped_.store(true, std::memory_order_release); condition_.notify_all(); }
  void fail() { failed_.store(true, std::memory_order_release); stop(); }
  void cacheStatus(std::string json) {
    std::lock_guard lock(queueMutex_);
    status_ = std::move(json);
    statusCaptured_ = Clock::now();
  }
  std::vector<Incoming> drain() {
    std::lock_guard lock(queueMutex_);
    std::vector<Incoming> result;
    result.reserve(incoming_.size());
    while (!incoming_.empty()) {
      result.push_back(std::move(incoming_.front())); incoming_.pop_front();
    }
    return result;
  }
  void wait(std::chrono::milliseconds duration) {
    std::unique_lock lock(queueMutex_);
    condition_.wait_for(lock, duration, [&] { return !incoming_.empty() || stopping(); });
  }
  bool incomingEmpty() {
    std::lock_guard lock(queueMutex_);
    return incoming_.empty();
  }
  void send(wire::Message event, bool terminal = false) {
    const auto encoded = wire::serializeMessage(event, limits_);
    if (!encoded) throw std::runtime_error("native event serialization failed: " + encoded.issue->describe());
    std::lock_guard lock(outputMutex_);
    size_t offset = 0;
    const auto began = Clock::now();
    while (offset < encoded.value->size()) {
      if ((!terminal && stopping()) || Clock::now() - began > std::chrono::seconds(30))
        throw std::runtime_error("native stdout write deadline exceeded");
      pollfd descriptor{STDOUT_FILENO, POLLOUT, 0};
      const int polled = ::poll(&descriptor, 1, 100);
      if (polled < 0 && errno == EINTR) continue;
      if (polled < 0) throw std::system_error(errno, std::generic_category(), "native stdout poll");
      if (!polled) {
        if (stopping() || Clock::now() - began > std::chrono::seconds(30))
          throw std::runtime_error("native stdout unavailable");
        continue;
      }
      if (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL))
        throw std::runtime_error("native stdout closed");
      const auto written = ::write(STDOUT_FILENO, encoded.value->data() + offset,
                                   encoded.value->size() - offset);
      if (written < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
      if (written <= 0) throw std::system_error(errno, std::generic_category(), "native stdout write");
      offset += static_cast<size_t>(written);
    }
  }
private:
  void enqueue(Incoming message) {
    {
      std::lock_guard lock(queueMutex_);
      if (incoming_.size() >= kControlBound) throw std::runtime_error("native control queue exceeds bound");
      incoming_.push_back(std::move(message));
    }
    condition_.notify_one();
  }
  void issue(wire::ProtocolIssue value, bool maskIssue = false) {
    if (value.failureClass == wire::FailureClass::RequestError && !value.requestId)
      value.failureClass = wire::FailureClass::ProtocolFatal;
    if (wire::connectionMustClose(value.failureClass)) {
      failed_.store(true, std::memory_order_release); stop();
      send(wire::ErrorEvent{value.failureClass, 0, false,
          std::string(wire::issueCodeName(value.code)), value.message}, true);
    } else enqueue({std::move(value), Clock::now(), maskIssue});
  }
  void readLoop() {
    @autoreleasepool {
      try {
        wire::FrameParser parser(limits_);
        std::array<uint8_t, 16384> bytes;
        while (!stopping()) {
          pollfd descriptor{STDIN_FILENO, POLLIN, 0};
          const auto polled = ::poll(&descriptor, 1, 50);
          if (polled < 0 && errno == EINTR) continue;
          if (polled < 0) throw std::system_error(errno, std::generic_category(), "native stdin poll");
          if (!polled) continue;
          if (descriptor.revents & (POLLERR | POLLNVAL)) throw std::runtime_error("native stdin failed");
          const auto received = ::read(STDIN_FILENO, bytes.data(), bytes.size());
          if (received < 0 && errno == EINTR) continue;
          if (received < 0) throw std::system_error(errno, std::generic_category(), "native stdin read");
          if (!received) { if (const auto truncated = parser.finish()) issue(*truncated); stop(); return; }
          size_t offset = 0;
          while (offset < static_cast<size_t>(received) && !stopping()) {
            auto parsed = parser.consume(std::span(bytes).subspan(offset, received - offset));
            offset += parsed.consumedBytes;
            if (parsed.issue) { issue(std::move(*parsed.issue)); break; }
            if (!parsed.frame) continue;
            auto decoded = wire::decodeFrame(*parsed.frame, limits_);
            if (!decoded) {
              issue(std::move(*decoded.issue), parsed.frame->type == wire::FrameType::MaskResponse);
              continue;
            }
            if (const auto status = std::get_if<wire::StatusRequestFrame>(&*decoded.value)) {
              std::string snapshot;
              Time captured;
              { std::lock_guard lock(queueMutex_); snapshot = status_; captured = statusCaptured_; }
              const auto readAt = Clock::now();
              if (snapshot.empty() || snapshot.back() != '}')
                throw std::runtime_error("native cached safe-point status is unavailable");
              snapshot.pop_back();
              std::ostringstream freshness;
              freshness << std::setprecision(12)
                  << R"(,"status_read":{"scope":"reader timestamp; model state remains cached at native safe point","steady_seconds":)"
                  << std::chrono::duration<double>(readAt.time_since_epoch()).count()
                  << R"(,"safe_point_snapshot_age_ms":)" << milliseconds(captured, readAt)
                  << R"(,"safe_point_snapshot_current":)" << (readAt - captured <= std::chrono::milliseconds(250) ? "true" : "false")
                  << "}}";
              snapshot += freshness.str();
              send(wire::StatusJsonEvent{status->correlationId, wire::kStatusSchemaVersion, std::move(snapshot)});
            } else if (std::holds_alternative<wire::RequestFrame>(*decoded.value) ||
                std::holds_alternative<wire::CancelFrame>(*decoded.value) ||
                std::holds_alternative<wire::MaskResponseFrame>(*decoded.value)) {
              enqueue({std::move(*decoded.value), Clock::now()});
            } else issue({wire::FailureClass::ProtocolFatal, wire::IssueCode::UnknownFrameType, 0,
                "frontend sent a native event instead of control"});
          }
        }
      } catch (const std::exception &error) {
        std::cerr << "error: Flash native reader: " << error.what() << '\n';
        failed_.store(true, std::memory_order_release); stop();
      }
    }
  }
  wire::ProtocolLimits limits_;
  std::atomic<bool> stopped_{false}, failed_{false};
  std::mutex queueMutex_, outputMutex_;
  std::condition_variable condition_;
  std::deque<Incoming> incoming_;
  std::string status_;
  Time statusCaptured_{};
  std::thread reader_;
};
struct Request {
  wire::RequestFrame frame;
  Time arrived, deadline, began{};
  std::optional<Time> firstToken, lastToken;
  std::optional<FlashRequestState> state;
  std::optional<FlashMTPState> mtpState;
  metal::MetalBuffer mtpFoldHidden;
  std::vector<uint32_t> mtpFoldTokens;
  FlashMTPDepthController mtpDepth;
  uint32_t slot = 0, promptOffset = 0, emitted = 0;
  std::optional<uint32_t> pendingToken;
  std::vector<uint16_t> logits;
  std::vector<uint32_t> mask;
  uint64_t maskId = 0;
  uint64_t generation = 0;
  std::mt19937_64 random;
  bool cancelled = false;
  bool mtpPriming = false;
  Time pressureWaitBegan{}, admissionRetryAt{};
};
bool retryAdmission(metal::AllocationFailure failure, bool anyActive,
                    Time pressureBegan, Time now, Time deadline) {
  if (now >= deadline) return false;
  if (anyActive) return true; // Existing active work may free the required bytes.
  return failure == metal::AllocationFailure::HostPressure &&
      (pressureBegan == Time{} || now - pressureBegan < kAdmissionPressureLimit);
}
struct RequestCookie {
  uint64_t id = 0, generation = 0;
};
bool sameCookie(const Request *request, RequestCookie cookie) {
  return request && request->frame.requestId == cookie.id && request->generation == cookie.generation;
}
uint32_t jointDepth(std::span<const uint32_t> remaining) {
  if (remaining.empty()) throw std::invalid_argument("empty joint MTP cohort");
  uint32_t depth = kMTPDepth;
  for (uint32_t budget : remaining) {
    if (!budget) throw std::invalid_argument("joint MTP cohort contains completed output budget");
    depth = std::min(depth, budget - 1);
  }
  return depth;
}
uint32_t jointWindowRows(std::span<const uint32_t> realCounts,
                        std::span<const uint32_t> remaining) {
  if (realCounts.empty() || realCounts.size() > kConcurrent || realCounts.size() != remaining.size())
    throw std::invalid_argument("invalid joint MTP real cohort geometry");
  uint32_t rows = kMTPDepth + 1;
  for (uint32_t lane = 0; lane < realCounts.size(); ++lane) {
    if (!realCounts[lane] || realCounts[lane] > kMTPDepth + 1 || !remaining[lane])
      throw std::invalid_argument("joint MTP cohort cannot contain padded or completed lane");
    rows = std::min(rows, std::min(realCounts[lane], remaining[lane]));
  }
  return rows;
}
uint32_t compactLastRow(std::span<const uint32_t> offsets, uint32_t lane) {
  if (offsets.size() < 2 || lane + 1 >= offsets.size() ||
      offsets[lane + 1] <= offsets[lane] || offsets[lane + 1] > kConcurrent * (kMTPDepth + 1))
    throw std::invalid_argument("invalid compact joint head lane offsets");
  return offsets[lane + 1] - 1;
}
bool decodeControlReady(const Request &request, Time now) {
  return request.pendingToken && request.promptOffset == request.frame.promptTokens.size() &&
      request.emitted < request.frame.logicalMaxOutputTokens && !request.maskId &&
      request.logits.empty() && !request.cancelled && !request.mtpPriming && request.deadline > now;
}
bool prefillControlReady(const Request &request, Time now) {
  return request.promptOffset < request.frame.promptTokens.size() && !request.emitted &&
      request.frame.logicalMaxOutputTokens && !request.pendingToken && !request.maskId &&
      request.logits.empty() && !request.cancelled && !request.mtpPriming && request.deadline > now;
}
uint32_t batchPrefillWindowRows(std::span<const uint32_t> remaining,
    uint32_t maximumRows = kDefaultBatchPrefillRows) {
  if (remaining.empty() || remaining.size() > kConcurrent ||
      !maximumRows || maximumRows > kFlashBatchPrefillMaximumRowsPerLane)
    throw std::invalid_argument("invalid Flash batch prefill real cohort size");
  uint32_t rows = maximumRows;
  for (uint32_t count : remaining) {
    if (!count) throw std::invalid_argument("Flash batch prefill cannot contain a completed prompt or padding");
    rows = std::min(rows, count);
  }
  return rows;
}
uint32_t prefillPrimeRows(uint32_t promptSize, uint32_t begin, uint32_t rows) {
  if (begin >= promptSize || !rows || rows > promptSize - begin)
    throw std::invalid_argument("invalid Flash prefill real token window");
  return std::min(rows, promptSize - begin - 1);
}
uint64_t batchPrefillHiddenOffset(uint32_t lane, uint32_t rows, uint32_t row) {
  if (lane >= kConcurrent || !rows || rows > kFlashBatchPrefillMaximumRowsPerLane || row >= rows)
    throw std::invalid_argument("invalid Flash batch prefill real hidden row");
  return uint64_t{lane * rows + row} * kHyper * 2;
}
bool groupedHeadPrimeCompatible(std::span<const uint32_t> counts) {
  if (counts.empty() || counts.size() > kConcurrent)
    throw std::invalid_argument("invalid Flash grouped head priming lane count");
  const uint32_t count = counts.front();
  for (uint32_t value : counts)
    if (!value || value > kHeadRows)
      throw std::invalid_argument("Flash grouped head priming requires real rows 1..128");
  if (counts.size() < 2 ||
      std::any_of(counts.begin(), counts.end(), [&](uint32_t value) { return value != count; }))
    return false;
  // Preserve both ordinary and fc_hidden cached tiles, scalar tails, and HC
  // fusion eligibility. Wider intermediate/final folds remain sequential.
  return count == kHeadRows || (count < 16 && counts.size() * count <= 32);
}
std::vector<uint32_t> selectedHeadPrimePositions(std::span<const uint32_t> counts) {
  (void)groupedHeadPrimeCompatible(counts); // Validate every pending real span.
  std::vector<uint32_t> selected, equalCounts;
  for (uint32_t position = 0; position < counts.size(); ++position)
    if (counts[position] == counts.front()) {
      selected.push_back(position); equalCounts.push_back(counts[position]);
    }
  if (!groupedHeadPrimeCompatible(equalCounts)) selected.resize(1);
  return selected;
}
uint64_t headPrimeSourceOffset(uint32_t lane, uint32_t rows, uint32_t primedRows, uint32_t count) {
  if (!count || count > kHeadRows || primedRows >= rows || count > rows - primedRows)
    throw std::invalid_argument("Flash grouped head priming exceeds its real source window");
  return batchPrefillHiddenOffset(lane, rows, primedRows);
}
void validateBatchHeadPrimeFlags(bool enabled, bool mtp, bool batchPrefill) {
  if (enabled && (!mtp || !batchPrefill))
    throw std::invalid_argument("SPLASH_FLASH_BATCH_MTP_PREFILL=1 requires SPLASH_FLASH_MTP=1 and SPLASH_FLASH_BATCH_PREFILL=1");
}
void validateBatchPrefillGPUCopyFlags(bool enabled, bool mtp, bool batchPrefill) {
  if (enabled && (!mtp || !batchPrefill))
    throw std::invalid_argument("SPLASH_FLASH_GPU_PREFILL_COPY=1 requires SPLASH_FLASH_MTP=1 and SPLASH_FLASH_BATCH_PREFILL=1");
}
bool prefillMemberMatches(const Request *request, RequestCookie cookie, uint32_t promptOffset, Time now) {
  return sameCookie(request, cookie) && request->promptOffset == promptOffset &&
      promptOffset <= request->frame.promptTokens.size() && !request->cancelled && request->deadline > now;
}
bool needsAdmissionRecheck(uint64_t previousAdmissions, uint64_t admissions, uint32_t rechecks) {
  return admissions > previousAdmissions && rechecks < kConcurrent;
}
struct PhaseTiming {
  uint64_t batches = 0, rows = 0;
  double lastGpu = 0, lastWall = 0, gpu = 0, wall = 0, host = 0;
  metal::CommandHostTiming commandHost;
  void add(uint32_t count, metal::CommandTiming timing, double hostSeconds) {
    ++batches; rows += count; lastGpu = timing.gpuSeconds; lastWall = timing.wallSeconds;
    gpu += timing.gpuSeconds; wall += timing.wallSeconds; host += hostSeconds;
    commandHost.add(timing.host);
  }
};
class StatusTimingWindow final {
public:
  void append(double value) {
    samples_.push_back(value);
    if (samples_.size() > 4096) samples_.pop_front();
    dirty_ = true;
  }
  [[nodiscard]] size_t size() const noexcept { return samples_.size(); }
  [[nodiscard]] double p50() const { refresh(); return p50_; }
  [[nodiscard]] double p95() const { refresh(); return p95_; }
private:
  void refresh() const {
    if (!dirty_) return;
    // Both status percentiles share the exact previous sorted-window rule.
    // Command safe points do not append samples, so they reuse this result.
    std::vector<double> sorted(samples_.begin(), samples_.end());
    std::sort(sorted.begin(), sorted.end());
    p50_ = sorted.empty() ? 0 : sorted[static_cast<size_t>((sorted.size() - 1) * .5)];
    p95_ = sorted.empty() ? 0 : sorted[static_cast<size_t>((sorted.size() - 1) * .95)];
    dirty_ = false;
  }
  std::deque<double> samples_;
  mutable double p50_ = 0, p95_ = 0;
  mutable bool dirty_ = true;
};
float logit(uint16_t value) { return std::bit_cast<float>(uint32_t{value} << 16); }
bool mtpEligible(const wire::RequestFrame &frame) {
  return frame.constraint == wire::ConstraintMode::None &&
      frame.cohort == wire::Cohort::Greedy && frame.sampling.temperature == 0 &&
      frame.logicalMaxOutputTokens > 1;
}
bool stopToken(uint32_t token) { return token == 248044 || token == 248046; }
uint32_t greedyToken(std::span<const uint16_t> values) {
  return flashGreedyToken(values);
}
uint32_t greedyRow(const metal::MetalBuffer &buffer, uint32_t vocabulary, uint32_t row = 0,
                   const metal::MetalBuffer &compact = {}) {
  if (compact) {
    const uint64_t needed = (uint64_t{row} + 1) * sizeof(FlashGreedyGPURowResult);
    if (!compact.contents() || compact.sizeBytes() < needed ||
        compact.sizeBytes() % sizeof(FlashGreedyGPURowResult) ||
        compact.sizeBytes() > uint64_t{kFlashGreedyGPUMaximumRows} * sizeof(FlashGreedyGPURowResult))
      throw std::runtime_error("Flash MTP GPU greedy result has invalid storage or extent");
    FlashGreedyGPURowResult result;
    std::memcpy(&result, static_cast<const std::byte *>(compact.contents()) +
        uint64_t{row} * sizeof(result), sizeof(result));
    return greedyGPUResultToken(result, vocabulary);
  }
  if (!buffer.contents() || buffer.sizeBytes() < uint64_t{row + 1} * vocabulary * 2)
    throw std::runtime_error("Flash MTP logits have invalid storage or extent");
  return greedyToken({static_cast<const uint16_t *>(buffer.contents()) + uint64_t{row} * vocabulary, vocabulary});
}
struct GreedyVerification {
  std::vector<uint32_t> output;
  uint32_t matched = 0;
  std::optional<wire::FinishReason> finish;
  uint32_t retained() const { return static_cast<uint32_t>(output.size()); }
};
GreedyVerification verifiedGreedy(std::span<const uint32_t> inputs,
    std::span<const uint32_t> predictions, uint32_t remaining) {
  const std::array<uint32_t, 2> stops{248044, 248046};
  const auto prefix = flashMTPAcceptGreedyPrefix(inputs, predictions, remaining, stops);
  if (!prefix)
    throw std::invalid_argument("invalid greedy MTP verification window");
  GreedyVerification result;
  result.matched = prefix->matchedDrafts;
  result.output.assign(prefix->output.begin(), prefix->output.begin() + prefix->retainedRows);
  if (prefix->finish == FlashMTPPrefixFinish::Stop) result.finish = wire::FinishReason::Stop;
  if (prefix->finish == FlashMTPPrefixFinish::Length) result.finish = wire::FinishReason::Length;
  return result;
}
struct JointMTPMember {
  RequestCookie cookie;
  uint64_t targetBegin = 0, foldedLength = 0;
  uint32_t remaining = 0, nextDraft = 0;
  std::vector<uint32_t> inputs;
  std::vector<uint16_t> lastHeadHidden;
  GreedyVerification accepted;
  std::vector<uint16_t> trueTargetHidden;
};
uint32_t selectToken(Request &request) {
  if (request.logits.empty() || !std::isfinite(request.frame.sampling.temperature) ||
      request.frame.sampling.temperature < 0 || !std::isfinite(request.frame.sampling.topP) ||
      request.frame.sampling.topP <= 0 || request.frame.sampling.topP > 1 ||
      request.frame.sampling.topK > 32 ||
      (request.frame.sampling.temperature > 0 && !request.frame.sampling.topK) ||
      (!request.mask.empty() && request.mask.size() != (request.logits.size() + 31) / 32))
    throw std::invalid_argument("invalid native token-selection geometry");
  if (request.frame.sampling.temperature <= 0 && request.mask.empty())
    return greedyToken(request.logits);
  struct Candidate { float value; uint32_t token; };
  const auto better = [](const Candidate &a, const Candidate &b) {
    return a.value > b.value || (a.value == b.value && a.token < b.token);
  };
  const uint32_t topK = request.frame.sampling.temperature > 0 ? request.frame.sampling.topK : 1;
  std::vector<Candidate> candidates;
  candidates.reserve(topK + 1);
  for (uint32_t token = 0; token < request.logits.size(); ++token) {
    if (!request.mask.empty() && !(request.mask[token / 32] & (uint32_t{1} << (token % 32)))) continue;
    const float value = logit(request.logits[token]);
    if (!std::isfinite(value)) throw std::runtime_error("non-finite Flash vocabulary logit");
    const Candidate candidate{value, token};
    if (candidates.size() == topK && !better(candidate, candidates.back())) continue;
    const auto insert = std::lower_bound(candidates.begin(), candidates.end(), candidate, better);
    if (insert != candidates.end() || candidates.size() < topK) {
      candidates.insert(insert, candidate);
      if (candidates.size() > topK) candidates.pop_back();
    }
  }
  if (candidates.empty()) throw std::invalid_argument("token mask allows no vocabulary token");
  if (request.frame.sampling.temperature <= 0) return candidates.front().token;
  std::array<double, 32> probabilities{};
  double total = 0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    probabilities[i] = std::exp((double(candidates[i].value) - candidates[0].value) /
                                request.frame.sampling.temperature);
    total += probabilities[i];
  }
  double retained = 0;
  size_t count = 0;
  do { retained += probabilities[count++]; }
  while (count < candidates.size() && retained < total * request.frame.sampling.topP);
  const double uniform = std::generate_canonical<double, 53>(request.random) * retained;
  double accumulated = 0;
  for (size_t i = 0; i < count; ++i) {
    accumulated += probabilities[i];
    if (uniform < accumulated) return candidates[i].token;
  }
  return candidates[count - 1].token;
}

// Exercises native host sampling and protocol framing without constructing a
// Metal device, mapping a model, or submitting any GPU work.
void cpuSelfTest() {
  const auto require = [](bool valid, const char *what) {
    if (!valid) throw std::runtime_error(std::string("Flash CPU self-test: ") + what);
  };
  {
    StatusTimingWindow window;
    require(window.size() == 0 && window.p50() == 0 && window.p95() == 0,
            "empty status timing window has zero percentiles");
    window.append(9);
    require(window.size() == 1 && window.p50() == 9 && window.p95() == 9,
            "status percentiles refresh after append");
    window.append(1);
    require(window.p50() == 1 && window.p95() == 1,
            "status percentile indices retain lower-rank convention");
    require(window.p50() == 1 && window.p95() == 1,
            "unchanged status window retains cached exact percentiles");
    std::deque<double> prior;
    StatusTimingWindow rolling;
    const auto oldPercentile = [&](double fraction) {
      std::vector<double> sorted(prior.begin(), prior.end());
      std::sort(sorted.begin(), sorted.end());
      return sorted.empty() ? 0 : sorted[static_cast<size_t>((sorted.size() - 1) * fraction)];
    };
    for (uint32_t index = 0; index < 4130; ++index) {
      const double value = static_cast<double>((index * 7919) % 997) / 7;
      rolling.append(value); prior.push_back(value);
      if (prior.size() > 4096) prior.pop_front();
      if (index < 16 || index % 31 == 0 || index >= 4095) {
        require(rolling.size() == prior.size() && rolling.p50() == oldPercentile(.5) &&
                    rolling.p95() == oldPercentile(.95),
                "cached status percentiles match prior rule across append and eviction");
      }
    }
    StatusTimingWindow evicted;
    for (uint32_t index = 0; index < 4096; ++index) evicted.append(index);
    require(evicted.p50() == 2047 && evicted.p95() == 3890,
            "full timing window percentiles use original indices");
    evicted.append(4096);
    require(evicted.size() == 4096 && evicted.p50() == 2048 && evicted.p95() == 3891,
            "timing-window eviction invalidates cache despite unchanged sample count");
  }
  Request greedy;
  greedy.logits = {0x3f80, 0x4000, 0x4000, 0x0000}; // 1,2,2,0
  require(selectToken(greedy) == 1, "greedy tie must choose lower token id");
  greedy.mask = {uint32_t{1} << 2};
  require(selectToken(greedy) == 2, "masked argmax must obey allowed token");
  greedy.mask = {0};
  bool rejected = false;
  try { static_cast<void>(selectToken(greedy)); } catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "zero mask must reject");
  Request sampling;
  sampling.logits = {0x3f80, 0x4000, 0x4040, 0x4080}; // 1,2,3,4
  sampling.frame.sampling = {1, 1, 2};
  sampling.random.seed(1234);
  Request repeated;
  repeated.logits = sampling.logits; repeated.frame.sampling = sampling.frame.sampling;
  repeated.random.seed(1234);
  bool observedSecond = false;
  for (uint32_t i = 0; i < 256; ++i) {
    const auto token = selectToken(sampling);
    require(token == selectToken(repeated), "seeded sampling must repeat");
    require(token == 2 || token == 3, "top-k must exclude lower logits");
    observedSecond |= token == 2;
  }
  require(observedSecond, "temperature sampling must sample beyond argmax");
  sampling.frame.sampling.topP = .01;
  for (uint32_t i = 0; i < 32; ++i) require(selectToken(sampling) == 3, "top-p cutoff");
  sampling.frame.sampling.topK = 0;
  rejected = false;
  try { static_cast<void>(selectToken(sampling)); } catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "sampling needs nonzero top-k");
  wire::RequestFrame eligible;
  eligible.logicalMaxOutputTokens = 128;
  require(mtpEligible(eligible), "unmasked greedy request should be MTP eligible");
  eligible.constraint = wire::ConstraintMode::TokenMask;
  eligible.cohort = wire::Cohort::Constrained;
  require(!mtpEligible(eligible), "grammar request must stay autoregressive");
  eligible.constraint = wire::ConstraintMode::None;
  eligible.cohort = wire::Cohort::Sampling;
  eligible.sampling.temperature = 1;
  require(!mtpEligible(eligible), "sampling request must stay autoregressive");
  const std::array<uint32_t, 4> inputs{100, 101, 102, 103};
  for (uint32_t matches = 0; matches <= 3; ++matches) {
    std::array<uint32_t, 4> predictions{901, 902, 903, 904};
    for (uint32_t index = 0; index < matches; ++index) predictions[index] = inputs[index + 1];
    for (uint32_t budget = 1; budget <= 4; ++budget) {
      const auto accepted = verifiedGreedy(inputs, predictions, budget);
      require(accepted.matched == matches, "greedy verifier must stop at first mismatch");
      require(accepted.retained() == std::min(matches + 1, budget), "commit count must equal emitted prediction count");
      require(accepted.output.back() == (budget <= matches ? inputs[budget] : predictions[matches]),
              "greedy verifier correction/bonus and budget");
      require((accepted.finish == wire::FinishReason::Length) == (accepted.retained() == budget),
              "greedy verifier exact output-budget termination");
    }
  }
  const std::array<uint32_t, 3> eosInputs{100, 101, 248046};
  const std::array<uint32_t, 3> eosPredictions{101, 248046, 77};
  const auto eos = verifiedGreedy(eosInputs, eosPredictions, 2);
  require(eos.output == std::vector<uint32_t>({101, 248046}) && eos.retained() == 2 &&
          eos.finish == wire::FinishReason::Stop, "EOS must win at the output-budget boundary");
  const std::array<uint32_t, 8> deepInputs{100, 101, 102, 103, 104, 105, 106, 107};
  for (uint32_t matches = 0; matches <= 7; ++matches) {
    std::array<uint32_t, 8> predictions{901, 902, 903, 904, 905, 906, 907, 908};
    for (uint32_t row = 0; row < matches; ++row) predictions[row] = deepInputs[row + 1];
    for (uint32_t budget = 1; budget <= 8; ++budget) {
      const auto prefix = verifiedGreedy(deepInputs, predictions, budget);
      require(prefix.matched == matches && prefix.retained() == std::min(matches + 1, budget),
              "deep verification must preserve exact mismatch and output-budget prefix");
      require(prefix.output.back() == (budget <= matches ? deepInputs[budget] : predictions[matches]),
              "deep verification must emit the true correction or final bonus");
    }
  }
  for (uint32_t configured = 1; configured <= kFlashSingletonMaximumMTPDepth; ++configured) {
    require(flashParseSingletonMTPDepth(std::to_string(configured)) == configured,
            "singleton draft depth strict supported range");
    const SingletonMTPPolicy policy{configured, true};
    require(policy.hiddenRows() == std::max(configured, kMTPDepth) + 1 &&
            policy.hiddenBytes() == uint64_t{policy.hiddenRows()} * kHyper * 2,
            "singleton hidden capacity must preserve joint four-row carry and deep sixteen-row carry");
    for (uint32_t remaining = 1; remaining <= kFlashSingletonMaximumVerifyRows + 2; ++remaining) {
      require(singletonAllowedDepth(configured, remaining, false) == std::min(configured, remaining - 1),
              "singleton draft cap must honor remaining true output budget");
      require(singletonAllowedDepth(configured, remaining, true) ==
                  std::min(std::min(configured, kMTPDepth), remaining - 1),
              "ready peers or pending admission must bound singleton transition at three drafts");
    }
  }
  for (const auto value : {"", "0", "07", " 7", "7 ", "+7", "-1", "7x"})
    require(!flashParseSingletonMTPDepth(value), "invalid singleton draft depth must reject before GPU startup");
  require(!flashParseSingletonMTPDepth(std::to_string(kFlashSingletonMaximumMTPDepth + 1)),
          "singleton draft depth beyond the current supported bound must reject before GPU startup");
  require(!jointFoldReady(0) && jointFoldReady(1) && jointFoldReady(4) &&
          !jointFoldReady(5) && !jointFoldReady(8),
          "an oversized singleton fold must run scalar before joining a four-row joint head");
  const std::array<uint32_t, 4> transitionPredictions{101, 102, 103, 104};
  const auto transition = verifiedGreedy(inputs, transitionPredictions, 128);
  require(singletonAllowedDepth(7, 128, true) == 3 && jointFoldReady(transition.retained()),
          "one scalar peer-bounded cycle must leave an exact fold eligible for joint execution");
  std::array<uint32_t, kFlashSingletonMaximumVerifyRows> fullInputs{}, fullPredictions{};
  for (uint32_t row = 0; row < fullInputs.size(); ++row) fullInputs[row] = 100 + row;
  for (uint32_t matches = 0; matches < fullInputs.size(); ++matches) {
    for (uint32_t row = 0; row < fullPredictions.size(); ++row)
      fullPredictions[row] = row < matches ? fullInputs[row + 1] : 900 + row;
    for (uint32_t budget = 1; budget <= fullInputs.size(); ++budget) {
      const auto prefix = verifiedGreedy(fullInputs, fullPredictions, budget);
      require(prefix.matched == matches && prefix.retained() == std::min(matches + 1, budget),
              "sixteen-row verification must retain the exact mismatch and quota prefix");
      uint32_t consumed = 0;
      while (consumed < prefix.retained()) {
        const auto count = flashMTPCommittedFoldChunkRows(prefix.retained() - consumed);
        require(count && *count <= 8 && consumed + *count <= prefix.retained(),
                "committed head chunks must avoid the sixteen-row BF16 coefficient switch");
        consumed += *count;
      }
      require(consumed == prefix.retained(), "head chunks must consume every retained pair exactly once");
    }
  }
  Request batchReady;
  const auto now = Clock::now();
  batchReady.frame.promptTokens = {7};
  batchReady.frame.logicalMaxOutputTokens = 128;
  batchReady.promptOffset = 1;
  batchReady.emitted = 1;
  batchReady.pendingToken = 8;
  batchReady.deadline = now + std::chrono::seconds(1);
  require(decodeControlReady(batchReady, now), "completed prompt with selected incoming token may join decode batch");
  batchReady.maskId = 3;
  require(!decodeControlReady(batchReady, now), "outstanding grammar row may not join batch");
  batchReady.maskId = 0;
  batchReady.logits = {0x3f80};
  require(!decodeControlReady(batchReady, now), "already computed prediction may not be recomputed in batch");
  batchReady.logits.clear();
  batchReady.promptOffset = 0;
  require(!decodeControlReady(batchReady, now), "unfinished prefill may not join decode batch");
  batchReady.promptOffset = 1;
  batchReady.cancelled = true;
  require(!decodeControlReady(batchReady, now), "cancelled request may not join batch");
  batchReady.cancelled = false;
  batchReady.emitted = 128;
  require(!decodeControlReady(batchReady, now), "completed output budget may not join batch");
  batchReady.emitted = 1;
  require(!decodeControlReady(batchReady, batchReady.deadline), "expired request may not join batch");
  Request prefillReady;
  prefillReady.frame.requestId = 91; prefillReady.generation = 5;
  prefillReady.frame.promptTokens.resize(513, 7);
  prefillReady.frame.logicalMaxOutputTokens = 128;
  prefillReady.deadline = now + std::chrono::seconds(1);
  require(prefillControlReady(prefillReady, now), "healthy real prompt may join prefill cohort");
  prefillReady.frame.constraint = wire::ConstraintMode::TokenMask;
  prefillReady.frame.cohort = wire::Cohort::Constrained;
  require(prefillControlReady(prefillReady, now), "grammar may share real prompt prefill before mask selection");
  prefillReady.frame.constraint = wire::ConstraintMode::None;
  prefillReady.frame.cohort = wire::Cohort::Sampling;
  prefillReady.frame.sampling.temperature = 1;
  require(prefillControlReady(prefillReady, now), "sampling may share real prompt prefill before sampling");
  prefillReady.pendingToken = 8;
  require(!prefillControlReady(prefillReady, now), "selected decode token may not reenter prefill");
  prefillReady.pendingToken.reset();
  prefillReady.maskId = 1;
  require(!prefillControlReady(prefillReady, now), "pending grammar row may not join prefill");
  prefillReady.maskId = 0;
  prefillReady.logits = {0x3f80};
  require(!prefillControlReady(prefillReady, now), "owned final prediction may not be recomputed in prefill");
  prefillReady.logits.clear();
  prefillReady.mtpPriming = true;
  require(!prefillControlReady(prefillReady, now), "unfolded head prefix may not admit another main prompt window");
  prefillReady.mtpPriming = false;
  prefillReady.cancelled = true;
  require(!prefillControlReady(prefillReady, now), "cancelled request may not join prefill");
  prefillReady.cancelled = false;
  require(!prefillControlReady(prefillReady, prefillReady.deadline), "expired request may not join prefill");
  prefillReady.emitted = 1;
  require(!prefillControlReady(prefillReady, now), "emitted output may not reenter prompt prefill");
  prefillReady.emitted = 0;
  prefillReady.promptOffset = 513;
  require(!prefillControlReady(prefillReady, now), "complete prompt may not invent another prefill token");
  prefillReady.promptOffset = 512;
  const RequestCookie prefillCookie{91, 5};
  require(prefillMemberMatches(&prefillReady, prefillCookie, 512, now) &&
          !prefillMemberMatches(&prefillReady, {91, 4}, 512, now) &&
          !prefillMemberMatches(&prefillReady, prefillCookie, 511, now) &&
          !prefillMemberMatches(nullptr, prefillCookie, 512, now),
          "prefill priming must retain exact generation and independent lane offset after control drains");
  prefillReady.cancelled = true;
  require(!prefillMemberMatches(&prefillReady, prefillCookie, 512, now),
          "a cancelled cohort member must not consume another owned priming slice");
  prefillReady.cancelled = false;
  require(!prefillMemberMatches(&prefillReady, prefillCookie, 512, prefillReady.deadline),
          "deadline at a priming boundary must drop only that member");
  for (uint32_t lanes = 1; lanes <= kConcurrent; ++lanes)
  for (uint32_t count = 1; count <= kHeadRows; ++count) {
    const std::vector<uint32_t> counts(lanes, count);
    const bool expected = lanes >= 2 && (count == kHeadRows || (count < 16 && lanes * count <= 32));
    require(groupedHeadPrimeCompatible(counts) == expected,
            "grouped head priming must preserve sequential cached tiles and fused HC eligibility");
    const auto selected = selectedHeadPrimePositions(counts);
    require(selected.size() == (expected ? lanes : 1) && selected.front() == 0,
            "unqualified head tails must retain the original single-lane call");
  }
  require(selectedHeadPrimePositions(std::array<uint32_t, 4>{128, 127, 128, 128}) ==
              std::vector<uint32_t>({0, 2, 3}) &&
          selectedHeadPrimePositions(std::array<uint32_t, 4>{127, 128, 128, 128}) ==
              std::vector<uint32_t>({0}),
          "mixed final head tails must never change a peer's qualified real count");
  require(kBatchHeadPrimeInputBytes == 10485760,
          "batch head priming compact input must reserve exactly B4x128 true BF16 hyper rows");
  for (const auto &invalid : {std::vector<uint32_t>{}, std::vector<uint32_t>{0},
          std::vector<uint32_t>{128, 129}, std::vector<uint32_t>{1, 1, 1, 1, 1}}) {
    bool rejectedHead = false;
    try { (void)selectedHeadPrimePositions(invalid); } catch (const std::invalid_argument &) { rejectedHead = true; }
    require(rejectedHead, "invalid grouped head priming spans must reject before GPU work");
  }
  for (bool enabled : {false, true}) for (bool mtp : {false, true}) for (bool batchPrefill : {false, true}) {
    bool rejectedFlags = false;
    try { validateBatchHeadPrimeFlags(enabled, mtp, batchPrefill); }
    catch (const std::invalid_argument &) { rejectedFlags = true; }
    require(rejectedFlags == (enabled && (!mtp || !batchPrefill)),
            "grouped head priming must validate both required startup routes before model mapping");
    bool rejectedCopyFlags = false;
    try { validateBatchPrefillGPUCopyFlags(enabled, mtp, batchPrefill); }
    catch (const std::invalid_argument &) { rejectedCopyFlags = true; }
    require(rejectedCopyFlags == (enabled && (!mtp || !batchPrefill)),
            "GPU feature copying must validate its startup routes before model mapping");
  }
  for (uint32_t lanes = 2; lanes <= kConcurrent; ++lanes)
  for (uint32_t mainRows : {1u, 8u, 15u, 16u, 127u, 128u, 129u, 512u, 1024u, 2048u}) {
    std::array<uint32_t, kConcurrent> primeRows{}, consumed{};
    std::vector<uint32_t> owned;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      primeRows[lane] = prefillPrimeRows(1000 + mainRows + lane, 1000, mainRows);
      for (uint32_t row = 0; row < mainRows; ++row) owned.push_back(lane * 100000 + row);
    }
    PhaseTiming priming;
    while (true) {
      std::vector<uint32_t> pendingLanes, counts;
      for (uint32_t lane = 0; lane < lanes; ++lane) if (consumed[lane] < primeRows[lane]) {
        pendingLanes.push_back(lane); counts.push_back(std::min(kHeadRows, primeRows[lane] - consumed[lane]));
      }
      if (counts.empty()) break;
      std::vector<uint32_t> compact;
      uint32_t realRows = 0;
      for (uint32_t position : selectedHeadPrimePositions(counts)) {
        const uint32_t lane = pendingLanes[position], count = counts[position];
        const uint64_t source = headPrimeSourceOffset(lane, mainRows, consumed[lane], count);
        require(source == uint64_t{lane * mainRows + consumed[lane]} * kHyper * 2 &&
                source + uint64_t{count} * kHyper * 2 <= uint64_t{lanes} * mainRows * kHyper * 2 &&
                uint64_t{realRows + count} * kHyper * 2 <= kBatchHeadPrimeInputBytes,
                "grouped priming must copy only exact owned lane features into compact capacity");
        for (uint32_t row = 0; row < count; ++row) {
          const uint32_t previous = owned[lane * mainRows + consumed[lane] + row];
          const uint32_t next = 1000 + consumed[lane] + row + 1;
          require(previous == lane * 100000 + consumed[lane] + row &&
                  next < 1000 + mainRows + lane,
                  "grouped priming must pair each previous feature with exactly its real next prompt token");
          compact.push_back(previous);
        }
        consumed[lane] += count; realRows += count;
      }
      require(compact.size() == realRows, "grouped priming cannot invent padding or a final anchor pair");
      priming.add(realRows, {0.01, 0.02}, 0.03);
    }
    uint64_t expectedRows = 0;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      require(consumed[lane] == primeRows[lane], "grouped head priming must consume every adjacent real pair exactly once");
      expectedRows += primeRows[lane];
    }
    require(priming.rows == expectedRows &&
            std::abs(priming.gpu - priming.batches * 0.01) < 1e-12,
            "grouped head priming timing must count each command and sum real pair rows once");
  }
  {
    // An ordinary deadline may pass between copies while a three-member
    // cohort is being assembled. The incomplete compact graph is discarded;
    // rebuilding consumes only the two surviving members with fresh pointers.
    std::array<Request, 3> packing;
    std::array<RequestCookie, 3> cookies;
    for (uint32_t lane = 0; lane < packing.size(); ++lane) {
      packing[lane].frame.requestId = 300 + lane; packing[lane].generation = 7;
      packing[lane].frame.promptTokens.resize(256, 1); packing[lane].promptOffset = 128;
      packing[lane].deadline = now + std::chrono::seconds(lane == 1 ? 1 : 2);
      cookies[lane] = {300 + lane, 7};
    }
    Time packingNow = now;
    std::vector<Request *> collected;
    std::vector<uint32_t> compact;
    PhaseTiming commands;
    bool rebuild = false;
    for (uint32_t lane : selectedHeadPrimePositions(std::array<uint32_t, 3>{128, 128, 128})) {
      if (!prefillMemberMatches(&packing[lane], cookies[lane], 128, packingNow)) {
        rebuild = true; break;
      }
      collected.push_back(&packing[lane]);
      for (uint32_t row = 0; row < 128; ++row) compact.push_back(lane * 1000 + row);
      packingNow = now + std::chrono::seconds(1); // Deadline callback after copy1.
    }
    require(rebuild && collected.size() == 1 && compact.size() == 128 &&
            commands.batches == 0 && commands.rows == 0,
            "deadline during compact copy must discard the incomplete graph without a submission or engine failure");
    collected.clear(); compact.clear(); // No collected request pointer crosses the drain.
    packing[1].cancelled = true; // Model expiry publishing this member's terminal.
    std::vector<uint32_t> survivors, counts;
    for (uint32_t lane = 0; lane < packing.size(); ++lane)
      if (prefillMemberMatches(&packing[lane], cookies[lane], 128, packingNow)) {
        survivors.push_back(lane); counts.push_back(128);
      }
    for (uint32_t position : selectedHeadPrimePositions(counts)) {
      const uint32_t lane = survivors[position];
      collected.push_back(&packing[lane]);
      for (uint32_t row = 0; row < 128; ++row) compact.push_back(lane * 1000 + row);
    }
    commands.add(static_cast<uint32_t>(compact.size()), {0.01, 0.02}, 0.03);
    require(collected == std::vector<Request *>({&packing[0], &packing[2]}) &&
            compact.size() == 256 && compact.front() == 0 && compact[128] == 2000 &&
            commands.batches == 1 && commands.rows == 256,
            "late-deadline cohort rebuild must submit only fresh surviving pointers and real feature pairs once");
  }
  {
    std::array<Request, kConcurrent> controls;
    std::array<RequestCookie, kConcurrent> cookies;
    std::array<uint32_t, kConcurrent> consumed{};
    for (uint32_t lane = 0; lane < kConcurrent; ++lane) {
      controls[lane].frame.requestId = 200 + lane; controls[lane].generation = 5;
      controls[lane].frame.promptTokens.resize(2048, 7); controls[lane].promptOffset = 1024;
      controls[lane].deadline = now + std::chrono::seconds(1);
      cookies[lane] = {200 + lane, 5};
    }
    PhaseTiming phase;
    while (true) {
      std::vector<uint32_t> pendingLanes, counts;
      for (uint32_t lane = 0; lane < kConcurrent; ++lane)
        if (consumed[lane] < 1024 && prefillMemberMatches(&controls[lane], cookies[lane], 1024, now)) {
          pendingLanes.push_back(lane); counts.push_back(std::min(kHeadRows, 1024 - consumed[lane]));
        }
      if (counts.empty()) break;
      uint32_t realRows = 0;
      for (uint32_t position : selectedHeadPrimePositions(counts)) {
        consumed[pendingLanes[position]] += counts[position]; realRows += counts[position];
      }
      phase.add(realRows, {0.01, 0.02}, 0.03);
      if (phase.batches == 1) controls[1].generation = 6;
      if (phase.batches == 2) controls[2].cancelled = true;
      if (phase.batches == 3) controls[3].deadline = now;
    }
    require(consumed == std::array<uint32_t, 4>{1024, 128, 256, 384} &&
            phase.batches == 8 && phase.rows == 1792,
            "generation reuse, cancellation and deadline must drop only their own head priming lanes at command boundaries");
  }
  require(batchPrefillWindowRows(std::array<uint32_t, 4>{2048, 2048, 2048, 2048}) == 512 &&
          batchPrefillWindowRows(std::array<uint32_t, 3>{1024, 17, 513}) == 17 &&
          batchPrefillWindowRows(std::array<uint32_t, 2>{1, 513}) == 1,
          "uniform real prefill windows must respect the shortest member without padding");
  for (uint32_t rows : {512u, 1024u, 2048u}) {
    require(parseBatchPrefillRows(std::to_string(rows)) == rows &&
            batchPrefillWindowRows(std::array<uint32_t, 4>{4097, 4097, 4097, 4097}, rows) == rows &&
            batchPrefillHiddenBytes(rows) == uint64_t{kConcurrent} * rows * kHyper * 2 &&
            batchPrefillHiddenOffset(3, rows, rows - 1) + uint64_t{kHyper} * 2 == batchPrefillHiddenBytes(rows),
            "wide prefill row setting must bound every real row and owned hidden extent");
  }
  require(batchPrefillHiddenBytes(2048) == 167772160 &&
          batchPrefillWindowRows(std::array<uint32_t, 3>{2048, 511, 1024}, 2048) == 511 &&
          uint64_t{kFlashBatchPrefillMaximumPhysicalRows} * 10 * 2560 * 2 == 419430400,
          "largest wide prefill staging, short member, and expert plane extents must stay exact");
  for (std::string_view value : {"", "0", "128", "513", "2049", "8192", "2048x", "-1", " 512"}) {
    bool rejectedRows = false;
    try { (void)parseBatchPrefillRows(value); } catch (const std::invalid_argument &) { rejectedRows = true; }
    require(rejectedRows, "invalid wide prefill setting must reject before GPU construction");
  }
  for (const auto &invalid : {std::vector<uint32_t>{}, std::vector<uint32_t>{1, 0},
          std::vector<uint32_t>{1, 1, 1, 1, 1}}) {
    bool rejectedWindow = false;
    try { (void)batchPrefillWindowRows(invalid); } catch (const std::invalid_argument &) { rejectedWindow = true; }
    require(rejectedWindow, "empty, completed or over-wide prefill cohort must reject before GPU work");
  }
  for (uint32_t maximumRows : {512u, 1024u, 2048u})
  for (uint32_t width = 2; width <= kConcurrent; ++width) {
    for (const uint32_t prompt : {1u, 127u, 128u, 129u, 511u, 512u, 513u, 1024u, 2048u, 4097u}) {
      std::array<uint32_t, kConcurrent> sizes{}, offsets{}, folded{};
      uint64_t totalTokens = 0;
      for (uint32_t lane = 0; lane < width; ++lane) sizes[lane] = prompt + lane * 17;
      while (true) {
        std::vector<uint32_t> survivors, left;
        for (uint32_t lane = 0; lane < width; ++lane) if (offsets[lane] < sizes[lane]) {
          survivors.push_back(lane); left.push_back(sizes[lane] - offsets[lane]);
        }
        if (left.empty()) break;
        const uint32_t realRows = batchPrefillWindowRows(left, maximumRows);
        std::vector<uint32_t> borrowed;
        for (uint32_t lane : survivors) for (uint32_t row = 0; row < realRows; ++row)
          borrowed.push_back(lane * 100000 + offsets[lane] + row);
        const auto owned = borrowed;
        std::fill(borrowed.begin(), borrowed.end(), UINT32_MAX);
        for (uint32_t member = 0; member < survivors.size(); ++member) {
          const uint32_t lane = survivors[member], begin = offsets[lane];
          const uint32_t primeRows = prefillPrimeRows(sizes[lane], begin, realRows);
          for (uint32_t primeBegin = 0; primeBegin < primeRows; primeBegin += kHeadRows) {
            const uint32_t count = std::min(kHeadRows, primeRows - primeBegin);
            require(folded[lane] == begin + primeBegin && count &&
                    begin + primeBegin + count < sizes[lane],
                    "batched prefill must fold every real adjacent token pair exactly once");
            require(batchPrefillHiddenOffset(member, realRows, primeBegin) ==
                        uint64_t{member * realRows + primeBegin} * kHyper * 2 &&
                    batchPrefillHiddenOffset(member, realRows, primeBegin) + uint64_t{count} * kHyper * 2 <=
                        uint64_t{survivors.size()} * realRows * kHyper * 2 &&
                    owned[member * realRows + primeBegin] == lane * 100000 + begin + primeBegin,
                    "owned hidden slices must retain the compact cohort layout across arena overwrite");
            folded[lane] += count;
          }
          offsets[lane] += realRows;
        }
        totalTokens += uint64_t{survivors.size()} * realRows;
      }
      uint64_t expectedTokens = 0;
      for (uint32_t lane = 0; lane < width; ++lane) {
        require(offsets[lane] == sizes[lane] && folded[lane] == sizes[lane] - 1,
                "partial prefill cohorts and singleton tails must preserve full prompt/head coverage");
        expectedTokens += sizes[lane];
      }
      require(totalTokens == expectedTokens, "prefill counters must count only real consumed prompt tokens once");
    }
  }
  {
    std::array<uint32_t, 4> remaining{512, 1024, 1536, 2048};
    std::array<uint64_t, 4> widths{};
    PhaseTiming phase;
    while (true) {
      std::vector<uint32_t> live;
      for (uint32_t value : remaining) if (value) live.push_back(value);
      if (live.empty()) break;
      const uint32_t rows = batchPrefillWindowRows(live);
      ++widths[live.size() - 1]; phase.add(static_cast<uint32_t>(live.size()) * rows, {0.01, 0.02}, 0.03);
      for (auto &value : remaining) if (value) value -= rows;
    }
    require(widths == std::array<uint64_t, 4>{1, 1, 1, 1} && phase.batches == 4 &&
            phase.rows == 5120 && std::abs(phase.gpu - 0.04) < 1e-12,
            "B4 to B3 to B2 to B1 prefill counters must count one command per actual cohort");
  }
  {
    // The first drain sees one frame. Its state allocation gives the reader
    // time to enqueue the other three; one refresh must admit those peers
    // before any long singleton prefill starts.
    uint64_t previous = 0, admitted = 1;
    uint32_t rechecks = 0, readyLanes = 1;
    while (needsAdmissionRecheck(previous, admitted, rechecks)) {
      previous = admitted; ++rechecks;
      if (rechecks == 1) { admitted += 3; readyLanes += 3; }
    }
    require(readyLanes == 4 && admitted == 4 && rechecks == 2,
            "reader arrivals during state allocation must form a full real prefill cohort");
    previous = 0; admitted = 1; rechecks = 0;
    while (needsAdmissionRecheck(previous, admitted, rechecks)) {
      previous = admitted; ++rechecks; // Empty queue: no admission or waiting.
    }
    require(rechecks == 1 && admitted == 1,
            "isolated C1 must perform only one empty reader refresh without waiting");
    previous = 0; admitted = 1; rechecks = 0;
    while (needsAdmissionRecheck(previous, admitted, rechecks)) {
      previous = admitted; ++rechecks; ++admitted; // Continuous arrivals.
    }
    require(rechecks == kConcurrent && admitted == 5 &&
            !needsAdmissionRecheck(10, 10, 0) && !needsAdmissionRecheck(10, 9, 0),
            "continuous arrivals cannot turn allocation refresh into an unbounded admission loop");
    // Controls arriving during an allocation are applied before eligibility is
    // evaluated for the re-drain cohort. Cancelled/expired lanes do no work.
    Request cancelledAllocation, expiredArrival, survivorArrival;
    for (Request *request : {&cancelledAllocation, &expiredArrival, &survivorArrival}) {
      request->frame.promptTokens.resize(2048, 7);
      request->frame.logicalMaxOutputTokens = 128;
      request->deadline = now + std::chrono::seconds(1);
    }
    cancelledAllocation.cancelled = true; expiredArrival.deadline = now;
    require(!prefillControlReady(cancelledAllocation, now) &&
            !prefillControlReady(expiredArrival, now) && prefillControlReady(survivorArrival, now),
            "allocation queue refresh must apply cancellation and deadlines before launching prompt work");
  }
  for (const uint32_t rows : {32u, 64u, 128u, 256u, 512u, 1024u, 2048u}) {
    require(parsePrefillRows(std::to_string(rows)) == rows, "configured prefill geometry");
    for (const uint32_t prompt : {1u, 127u, 128u, 129u, 255u, 256u, 257u, 511u, 2047u, 2048u, 2049u, 4097u}) {
      uint32_t folded = 0;
      for (uint32_t begin = 0; begin < prompt; begin += rows) {
        const uint32_t count = std::min(rows, prompt - begin);
        const uint32_t primeRows = prefillPrimeRows(prompt, begin, count);
        for (uint32_t primeBegin = 0; primeBegin < primeRows; primeBegin += kHeadRows) {
          const uint32_t primeCount = std::min(kHeadRows, primeRows - primeBegin);
          require(primeCount && primeCount <= 128 && folded == begin + primeBegin,
                  "MTP priming slices must preserve contiguous previous-hidden pairs");
          require(begin + primeBegin + primeCount < prompt,
                  "MTP priming next-token slice must stay inside prompt");
          folded += primeCount;
        }
      }
      require(folded == prompt - 1, "large prefill priming must fold every adjacent prompt pair exactly once");
    }
  }
  for (const auto value : {"", "0", "16", "96", "129", "4096", "128x", "-1", " 128"}) {
    bool rejectedRows = false;
    try { static_cast<void>(parsePrefillRows(value)); } catch (const std::invalid_argument &) { rejectedRows = true; }
    require(rejectedRows, "unsupported prefill geometry must fail before model construction");
  }
  require(jointDepth(std::array<uint32_t, 4>{128, 4, 9, 7}) == 3 &&
          jointDepth(std::array<uint32_t, 3>{128, 2, 9}) == 1 &&
          jointDepth(std::array<uint32_t, 2>{1, 128}) == 0,
          "joint draft cap must respect every actual member's remaining budget");
  require(jointWindowRows(std::array<uint32_t, 4>{4, 3, 2, 4},
              std::array<uint32_t, 4>{128, 128, 128, 128}) == 2 &&
          jointWindowRows(std::array<uint32_t, 2>{4, 4}, std::array<uint32_t, 2>{2, 128}) == 2,
          "joint target window must shorten all lanes without padding or budget overflow");
  Request reused;
  reused.frame.requestId = 77; reused.generation = 11;
  const RequestCookie original{77, 10}, replacement{77, 11};
  require(!sameCookie(&reused, original) && sameCookie(&reused, replacement) && !sameCookie(nullptr, original),
          "reused request ids cannot revive an in-flight cohort cookie");
  std::array<const Request *, 3> liveMembers{&reused, nullptr, &reused};
  const std::array<RequestCookie, 3> slots{original, replacement, replacement};
  std::array<uint32_t, 3> commitCounts{};
  for (uint32_t slot = 0; slot < slots.size(); ++slot)
    commitCounts[slot] = sameCookie(liveMembers[slot], slots[slot]) ? 2 : 0;
  require(commitCounts == std::array<uint32_t, 3>{0, 0, 2},
          "cohort cancellation must preserve original zero/null slots and live peer retention");
  const std::array<uint32_t, 4> offsets{0, 1, 4, 8};
  require(compactLastRow(offsets, 0) == 0 && compactLastRow(offsets, 1) == 3 &&
          compactLastRow(offsets, 2) == 7, "ragged head last-feature offsets are distinct from Last logits lane indices");
  std::array<uint16_t, 8> borrowed{10, 11, 12, 13, 14, 15, 16, 17};
  std::array<uint16_t, 3> ownedLast{};
  for (uint32_t lane = 0; lane < ownedLast.size(); ++lane)
    ownedLast[lane] = borrowed[compactLastRow(offsets, lane)];
  borrowed.fill(99);
  require(ownedLast == std::array<uint16_t, 3>{10, 13, 17},
          "owned borrowed-head copies must survive arena overwrite and survivor compaction");
  const Time pressureStart = Clock::now();
  const Time requestDeadline = pressureStart + std::chrono::seconds(10);
  require(retryAdmission(metal::AllocationFailure::HostPressure, false, Time{}, pressureStart, requestDeadline) &&
          retryAdmission(metal::AllocationFailure::HostPressure, false, pressureStart,
              pressureStart + std::chrono::seconds(1), requestDeadline) &&
          !retryAdmission(metal::AllocationFailure::HostPressure, false, pressureStart,
              pressureStart + std::chrono::seconds(2), requestDeadline),
          "temporary no-active host-pressure admission retries must stop after two seconds");
  require(!retryAdmission(metal::AllocationFailure::EngineBudget, false, Time{}, pressureStart, requestDeadline),
          "hard engine budget denial must remain a capacity error");
  require(!retryAdmission(metal::AllocationFailure::HostPressure, false, pressureStart, requestDeadline, requestDeadline),
          "expired admission wait must never allocate");
  Request waiting;
  waiting.frame.requestId = 90; waiting.generation = 3;
  waiting.deadline = requestDeadline; waiting.cancelled = true;
  require(!decodeControlReady(waiting, pressureStart), "cancelled pressure-wait request remains non-executable");
  // Simulate pressure then a successful governed reservation. Success resumes
  // admission through the normal reservation branch, without bypassing checks.
  const std::array<metal::AllocationFailure, 3> verdicts{
      metal::AllocationFailure::HostPressure, metal::AllocationFailure::HostPressure, metal::AllocationFailure::None};
  uint32_t retries = 0, reservations = 0;
  for (uint32_t step = 0; step < verdicts.size(); ++step) {
    if (verdicts[step] == metal::AllocationFailure::None) ++reservations;
    else if (retryAdmission(verdicts[step], false, pressureStart,
        pressureStart + std::chrono::milliseconds(step * 250), requestDeadline)) ++retries;
  }
  require(retries == 2 && reservations == 1, "host-pressure recovery should admit exactly once after successful reservation");
  wire::ProtocolLimits limits;
  limits.maxSimulationTokens = 1; limits.maxMaskWords = 7760;
  const auto encoded = wire::serializeMessage(wire::MaskRequestEvent{5, 7, 7760, {}}, limits);
  require(bool(encoded), "one-row mask request must serialize");
  wire::FrameParser parser(limits);
  std::optional<wire::Frame> frame;
  for (const auto byte : *encoded.value) {
    const auto step = parser.consume(std::span(&byte, 1));
    require(step.consumedBytes == 1 && !step.issue, "fragmented frame parsing");
    if (step.frame) frame = step.frame;
  }
  require(frame.has_value() && !parser.finish(), "complete fragmented frame");
  const auto decoded = wire::decodeFrame(*frame, limits);
  require(bool(decoded) && std::holds_alternative<wire::MaskRequestEvent>(*decoded.value), "mask frame decode");
  require(std::get<wire::MaskRequestEvent>(*decoded.value).simulationTokens.empty(), "ordinary autoregression mask row");
  wire::FrameParser truncated(limits);
  static_cast<void>(truncated.consume(std::span(*encoded.value).first(encoded.value->size() - 1)));
  const auto issue = truncated.finish();
  require(issue && issue->failureClass == wire::FailureClass::ProtocolFatal, "truncated frame must close stream");
  for (const uint32_t context : {1u, 8192u, 262144u}) {
    const auto configured = serveLimits(context, 248320);
    require(bool(wire::serializeMessage(wire::ReadyEvent{1, 4, context, wire::kNativeFeatureBits}, configured)),
            "serve limits must encode Ready");
    require(bool(wire::serializeMessage(wire::StatusJsonEvent{1, 5, R"({"schema_version":5,"ready":true})"}, configured)),
            "serve limits must encode status");
    require(bool(wire::serializeMessage(wire::MaskRequestEvent{5, 7, 7760, {}}, configured)),
            "serve limits must encode ordinary mask request");
    require(bool(wire::serializeMessage(wire::ErrorEvent{wire::FailureClass::ProtocolFatal, 0, false,
                "bad_magic", "bad magic"}, configured)), "serve limits must encode fatal error");
    require(bool(wire::serializeMessage(wire::TokensEvent{5, 0, {1, 2, 3, 4}},
                serveLimits(context, 248320, true))), "MTP serve limits must encode four-token blocks");
    const auto deepLimits = serveLimits(context, 248320, true, 7);
    const wire::TokensEvent deepEvent{77, 9, {1, 2, 3, 4, 5, 6, 7, 8}};
    const auto deepEncoded = wire::serializeMessage(deepEvent, deepLimits);
    require(bool(deepEncoded) && deepLimits.maxTokenBatch == 8,
            "deep singleton serve limits must encode real eight-token blocks");
    wire::FrameParser deepParser(deepLimits);
    std::optional<wire::Frame> deepFrame;
    for (const auto byte : *deepEncoded.value) {
      const auto parsed = deepParser.consume(std::span(&byte, 1));
      require(parsed.consumedBytes == 1 && !parsed.issue, "deep token block fragmented parser");
      if (parsed.frame) deepFrame = parsed.frame;
    }
    require(deepFrame && !deepParser.finish(), "deep token block must complete after fragmented delivery");
    const auto deepDecoded = wire::decodeFrame(*deepFrame, deepLimits);
    require(bool(deepDecoded) && std::holds_alternative<wire::TokensEvent>(*deepDecoded.value) &&
            std::get<wire::TokensEvent>(*deepDecoded.value).tokens == deepEvent.tokens,
            "deep token block must retain every real token after framing");
    require(!wire::serializeMessage(wire::TokensEvent{77, 9, {1, 2, 3, 4, 5, 6, 7, 8, 9}}, deepLimits),
            "deep singleton serve limits must reject a ninth token instead of truncating it");
    require(!wire::serializeMessage(deepEvent, serveLimits(context, 248320, true)),
            "default three-draft serve limits remain four tokens per event");
    const auto widestLimits = serveLimits(context, 248320, true, kFlashSingletonMaximumMTPDepth);
    std::vector<uint32_t> widestTokens(kFlashSingletonMaximumVerifyRows);
    for (uint32_t row = 0; row < widestTokens.size(); ++row) widestTokens[row] = row + 1;
    const auto widestEncoded = wire::serializeMessage(wire::TokensEvent{77, 9, widestTokens}, widestLimits);
    require(bool(widestEncoded) && widestLimits.maxTokenBatch == 16,
            "fifteen-draft limits must frame sixteen actual tokens");
    wire::FrameParser widestParser(widestLimits);
    std::optional<wire::Frame> widestFrame;
    for (const auto byte : *widestEncoded.value) {
      const auto parsed = widestParser.consume(std::span(&byte, 1));
      require(parsed.consumedBytes == 1 && !parsed.issue, "sixteen-token fragmented frame parsing");
      if (parsed.frame) widestFrame = parsed.frame;
    }
    require(widestFrame && !widestParser.finish(), "sixteen-token frame must complete");
    const auto widestDecoded = wire::decodeFrame(*widestFrame, widestLimits);
    require(bool(widestDecoded) && std::holds_alternative<wire::TokensEvent>(*widestDecoded.value) &&
            std::get<wire::TokensEvent>(*widestDecoded.value).tokens == widestTokens,
            "sixteen-token framing must preserve every actual token");
    widestTokens.push_back(17);
    require(!wire::serializeMessage(wire::TokensEvent{77, 9, widestTokens}, widestLimits),
            "fifteen-draft limits must reject seventeen-token frames");
  }
  std::cout << R"({"valid":true,"gpu_work":false,"checks":["status_timing_cache","greedy","masked_argmax","seed","top_k","top_p","invalid_sampling","fragmented_frames","truncation","mtp_eligibility","mtp_prefix_commit","mtp_eos_budget","mtp_block_framing","deep_mtp_prefix_budget","deep_mtp_depth_parser","deep_mtp_joint_transition","deep_mtp_eight_token_framing","batch_control_guards","prefill_geometry","sliced_mtp_priming","batch_prefill_control_guards","batch_prefill_real_windows","batch_prefill_partial_cohorts","batch_prefill_owned_priming","batch_prefill_cookie_deadlines","batch_prefill_width_accounting","batch_mtp_prefill_eligibility","batch_mtp_prefill_owned_compaction","batch_mtp_prefill_exact_pairs","batch_mtp_prefill_command_accounting","batch_mtp_prefill_cookie_cancel_deadline","batch_mtp_prefill_late_deadline_rebuild","batch_mtp_prefill_flag_dependencies","gpu_prefill_copy_flag_dependencies","allocation_arrival_cohort_refresh","allocation_refresh_c1_no_wait","allocation_refresh_bounded","allocation_refresh_cancel_deadline","joint_budget","joint_no_padding","joint_cookie_drop","joint_borrowed_layout","admission_pressure_retry","admission_deadline_budget","admission_pressure_recovery"]})" << '\n';
}

class Worker final {
public:
  Worker(Transport &transport, metal::MetalBackend &backend, const FlashWeights &weights,
         FlashForward &forward, engine::MemoryGovernor &governor, PressureMonitor &pressure,
         uint32_t capacity, uint64_t instance, FlashMTPForward *head = nullptr,
         FlashBatchForward *batch = nullptr, uint32_t prefillRows = kDefaultPrefillRows,
         FlashBatchVerify *jointVerify = nullptr, FlashBatchMTPForward *jointHead = nullptr,
         metal::MetalBuffer jointHidden = {}, SingletonMTPPolicy singletonMTP = {},
         FlashBatchPrefill *batchPrefill = nullptr, metal::MetalBuffer batchPrefillHidden = {},
         uint32_t batchPrefillRows = kDefaultBatchPrefillRows,
         FlashBatchMTPForward *batchPrimeHead = nullptr, metal::MetalBuffer batchPrimeInput = {},
         SavedOperandsResidencyStatus savedResidency = {},
         metal::ResidencyLease savedResidencyLease = {},
         std::unique_ptr<FlashRequestCommandTrace> requestCommandTrace = {},
         std::unique_ptr<idle_maintenance::Maintenance> idleMaintenance = {},
         uint32_t idleMaintenanceInterval = idle_maintenance::kIntervalMilliseconds,
         bool idleMaintenanceRequested = false, std::string idleMaintenanceFailureReason = {})
      : transport_(transport), backend_(backend), weights_(weights), forward_(forward),
        governor_(governor), pressure_(pressure), capacity_(capacity), instance_(instance), head_(head), batch_(batch),
        prefillRows_(prefillRows),
        jointVerify_(jointVerify), jointHead_(jointHead), jointHidden_(std::move(jointHidden)),
        batchPrefill_(batchPrefill), batchPrefillRows_(batchPrefillRows),
        batchPrefillHidden_(std::move(batchPrefillHidden)),
        batchPrimeHead_(batchPrimeHead), batchPrimeInput_(std::move(batchPrimeInput)),
        savedResidency_(std::move(savedResidency)),
        savedResidencyLease_(std::move(savedResidencyLease)),
        requestCommandTrace_(std::move(requestCommandTrace)),
        idleMaintenance_(std::move(idleMaintenance)),
        idleMaintenanceRequested_(idleMaintenanceRequested),
        idleMaintenanceFailureReason_(std::move(idleMaintenanceFailureReason)),
        idleMaintenanceInterval_(idleMaintenanceInterval),
        idleMaintenanceScheduler_(idleMaintenanceInterval),
        singletonMTP_(singletonMTP),
        words_((weights.descriptor().vocabularySize + 31) / 32) {}
  [[nodiscard]] FlashRequestTraceInfo requestCommandTraceInfo() const noexcept {
    return requestCommandTrace_ ? requestCommandTrace_->info() : FlashRequestTraceInfo{};
  }
  int run() {
    publishStatus();
    transport_.send(wire::ReadyEvent{instance_, kConcurrent, capacity_, wire::kNativeFeatureBits});
    transport_.start();
    while (!transport_.stopping()) {
      drain(); expire();
      uint64_t previousAdmissions = mtpRequests_ + arRequests_;
      admit();
      if (batchPrefill_) {
        // Request frames may arrive on the reader while createState() allocates
        // and clears the first lane. Consume those already-queued controls
        // before launching a long singleton graph. No coalescing sleep is used.
        uint32_t rechecks = 0;
        while (needsAdmissionRecheck(previousAdmissions, mtpRequests_ + arRequests_, rechecks)) {
          previousAdmissions = mtpRequests_ + arRequests_;
          ++rechecks; ++allocationQueueRechecks_;
          drain(); expire(); admit();
          recheckAdmissions_ += mtpRequests_ + arRequests_ - previousAdmissions;
        }
      }
      publishStatus();
      if (transport_.stopping()) break;
      // A real prefill cohort shares projection/expert reads before the decode
      // policy runs. Each graph stays within the configured real-row arena.
      if (batchPrefill_ && tickBatchPrefill(nextSlot_ % kConcurrent)) {
        ++nextSlot_;
        continue;
      }
      bool worked = false;
      for (uint32_t attempt = 0; attempt < kConcurrent; ++attempt) {
        const uint32_t slot = nextSlot_++ % kConcurrent;
        if (!active_[slot]) continue;
        Request &request = *active_[slot];
        if (request.maskId && request.mask.empty()) continue;
        worked = true;
        if (jointHead_ && readyJointMTP(request)) {
          if (!tickJointMTP(slot)) tick(request);
          break;
        }
        if (!batch_ || !readyBatch(request) || !tickBatch(slot)) tick(request);
        break;
      }
      if (!worked) {
        tickIdleMaintenance();
        if (!transport_.stopping()) transport_.wait(std::chrono::milliseconds(50));
      }
    }
    return transport_.failed() ? 2 : 0;
  }
private:
  void tickIdleMaintenance();
  void noteUserGPUCommand(const metal::CommandTiming &timing) {
    if (timing.wallSeconds > 0 || timing.gpuSeconds > 0) {
      idleMaintenanceScheduler_.userGPUCompleted(Clock::now());
    }
  }
  void traceRequestCommand(const char *phase, const char *role, RequestCookie cookie,
      uint32_t rows, const metal::CommandTiming &timing) {
    noteUserGPUCommand(timing);
    if (!requestCommandTrace_) return;
    const std::array<FlashRequestTraceLane, 1> lanes{{{cookie.id, cookie.generation, rows}}};
    requestCommandTrace_->afterCall(backend_, instance_, phase, role, lanes, timing);
  }
  template <class CookieAt, class RowsAt>
  void traceGroupCommand(const char *phase, const char *role, uint32_t width,
      const metal::CommandTiming &timing, CookieAt cookieAt, RowsAt rowsAt) {
    noteUserGPUCommand(timing);
    if (!requestCommandTrace_) return;
    std::array<FlashRequestTraceLane, kConcurrent> lanes{};
    if (!width || width > kConcurrent) throw std::logic_error("invalid labelled trace lane count");
    for (uint32_t lane = 0; lane < width; ++lane) {
      const auto cookie = cookieAt(lane);
      lanes[lane] = {cookie.id, cookie.generation, rowsAt(lane)};
    }
    requestCommandTrace_->afterCall(backend_, instance_, phase, role,
        std::span(lanes).first(width), timing);
  }
  bool readyBatch(const Request &request) const {
    return request.state && forward_.ownsState(*request.state) &&
        request.state->logicalLength() < capacity_ &&
        decodeControlReady(request, Clock::now());
  }
  bool readyPrefill(const Request &request) const {
    return request.state && forward_.ownsState(*request.state) &&
        request.state->logicalLength() == request.promptOffset &&
        request.state->logicalLength() < capacity_ && prefillControlReady(request, Clock::now());
  }
  void recordPrefill(uint32_t width, uint32_t rows, metal::CommandTiming timing, double host) {
    if (!width || width > kConcurrent || !rows || rows > kFlashBatchPrefillMaximumPhysicalRows)
      throw std::logic_error("invalid actual Flash prefill width or token count");
    prefill_.add(rows, timing, host);
    ++prefillWidths_[width - 1];
  }
  bool tickBatchPrefill(uint32_t primarySlot);
  void recordDecode(uint32_t width, uint32_t predictions, metal::CommandTiming timing, double host) {
    if (!width || width > kConcurrent) throw std::logic_error("invalid actual Flash decoder batch width");
    decode_.add(predictions, timing, host);
    ++decodeWidths_[width - 1];
  }
  bool tickBatch(uint32_t primarySlot) {
    std::vector<Request *> requests;
    std::vector<FlashRequestState *> states;
    std::vector<uint32_t> tokens;
    std::vector<uint64_t> lengths, ids, generations;
    for (uint32_t offset = 0; offset < kConcurrent; ++offset) {
      auto &candidate = active_[(primarySlot + offset) % kConcurrent];
      if (!candidate || !readyBatch(*candidate) || (jointHead_ && candidate->mtpState)) continue;
      requests.push_back(candidate.get());
      states.push_back(&*candidate->state);
      tokens.push_back(*candidate->pendingToken);
      lengths.push_back(candidate->state->logicalLength());
      ids.push_back(candidate->frame.requestId);
      generations.push_back(candidate->generation);
    }
    if (requests.size() < 2) return false;
    // Joining a real batch permanently selects AR for this request. The exact
    // trunk state needs no replay/conversion. A head cache cannot resume after
    // AR advances without refolding its missing true-target feature pairs.
    for (Request *request : requests) if (request->mtpState) {
      request->mtpState.reset();
      request->mtpFoldHidden = {};
      request->mtpFoldTokens.clear();
      ++mtpBatchFallbacks_;
    }
    inFlight_ = true; publishStatus();
    const auto began = Clock::now();
    const auto result = batch_->forwardBatch(states, tokens);
    const uint32_t width = static_cast<uint32_t>(requests.size());
    traceGroupCommand("autoregressive_decode", "target_trunk", width, result.timing,
        [&](uint32_t lane) { return RequestCookie{ids[lane], generations[lane]}; },
        [](uint32_t) { return 1u; });
    recordDecode(width, width, result.timing,
        std::chrono::duration<double>(Clock::now() - began).count());
    if (result.lanes != width || result.capacity != capacity_ ||
        result.logicalLengths.size() != width ||
        !result.logitsBF16.contents() ||
        result.logitsBF16.sizeBytes() < uint64_t{width} * weights_.descriptor().vocabularySize * 2)
      throw std::runtime_error("Flash batch returned invalid lane logits or geometry");
    const uint64_t rowBytes = uint64_t{weights_.descriptor().vocabularySize} * 2;
    // Copy every borrowed lane before any control processing, token emission,
    // peer trunk/head call, or terminal state destruction can run.
    for (uint32_t lane = 0; lane < width; ++lane) {
      Request &request = *requests[lane];
      if (!forward_.ownsState(*request.state) ||
          result.logicalLengths[lane] != lengths[lane] + 1 ||
          request.state->logicalLength() != result.logicalLengths[lane])
        throw std::logic_error("Flash batch advanced a different request offset");
      const auto row = static_cast<const uint16_t *>(result.logitsBF16.contents()) +
          uint64_t{lane} * weights_.descriptor().vocabularySize;
      request.logits.assign(row, row + weights_.descriptor().vocabularySize);
      request.pendingToken.reset();
      batchCopiedLogitBytes_ += rowBytes;
    }
    inFlight_ = false;
    drain(); expire();
    if (transport_.stopping()) return true;
    for (uint32_t lane = 0; lane < width; ++lane) {
      Request *request = find(ids[lane]);
      if (request && request->generation == generations[lane] && !request->logits.empty())
        maskOrEmit(*request);
    }
    return true;
  }
  bool readyJointMTP(const Request &request) const {
    return jointHead_ && jointVerify_ && readyBatch(request) && request.mtpState &&
        !request.mtpState->poisoned() && mtpEligible(request.frame) &&
        jointFoldReady(static_cast<uint32_t>(request.mtpFoldTokens.size()));
  }
  Request *jointMember(const JointMTPMember &member) {
    Request *request = find(member.cookie.id);
    return sameCookie(request, member.cookie) && request->state && request->mtpState ? request : nullptr;
  }
  void jointControls() {
    inFlight_ = false;
    drain(); expire();
  }
  bool tickJointMTP(uint32_t primarySlot);
  void requestError(uint64_t id, std::string code, std::string message, bool retryable = false) {
    transport_.send(wire::ErrorEvent{wire::FailureClass::RequestError, id, retryable,
                                    std::move(code), std::move(message)});
    ++failed_; remove(id);
  }
  void remove(uint64_t id) {
    live_.erase(id);
    for (auto &request : active_) if (request && request->frame.requestId == id) {
      if (request->state) forward_.abortVerify(*request->state);
      request.reset();
    }
    std::erase_if(pending_, [id](const auto &request) { return request->frame.requestId == id; });
    // Publish a terminal safe point immediately, rather than allowing a reader
    // response to reuse the preceding "active/in-flight" snapshot until the
    // next scheduling iteration.
    if (!inFlight_) publishStatus();
  }
  Request *find(uint64_t id) {
    for (auto &request : active_) if (request && request->frame.requestId == id) return request.get();
    for (auto &request : pending_) if (request->frame.requestId == id) return request.get();
    return nullptr;
  }
  void done(Request &request, wire::FinishReason reason) {
    const auto now = Clock::now();
    const auto began = request.began == Time{} ? request.arrived : request.began;
    const auto first = request.firstToken.value_or(now);
    const auto id = request.frame.requestId;
    transport_.send(wire::DoneEvent{id, reason, static_cast<uint32_t>(request.frame.promptTokens.size()),
        request.emitted, micros(began, first), request.firstToken ? micros(first, now) : 0,
        micros(request.arrived, now)});
    if (reason == wire::FinishReason::Cancelled) ++cancelled_; else ++completed_;
    if (idleMaintenance_) idleMaintenanceScheduler_.userFinished(now, request.emitted,
        reason == wire::FinishReason::Cancelled);
    remove(id);
  }
  void drain() {
    for (auto &input : transport_.drain()) {
      if (auto issue = std::get_if<wire::ProtocolIssue>(&input.value)) {
        // A grammar callback may finish after cancellation or completion.
        if (input.maskIssue && issue->requestId && !live_.contains(issue->requestId)) continue;
        requestError(issue->requestId, std::string(wire::issueCodeName(issue->code)), issue->message);
        continue;
      }
      auto &message = std::get<wire::Message>(input.value);
      if (auto frame = std::get_if<wire::RequestFrame>(&message)) {
        if (live_.contains(frame->requestId)) {
          transport_.fail();
          transport_.send(wire::ErrorEvent{wire::FailureClass::ProtocolFatal, 0, false,
              "invalid_request_id", "request id is already active"}, true);
          throw std::runtime_error("duplicate live Flash request id");
        }
        ++submitted_;
        const uint64_t nowUnix = unixMicros();
        if (frame->absoluteDeadlineUnixMicros <= nowUnix) {
          requestError(frame->requestId, "deadline_exceeded", "request expired before admission"); continue;
        }
        if (!frame->imageSpans.empty() || !frame->imagePixels.empty()) {
          requestError(frame->requestId, "unsupported_modality", "Flash native route accepts text input"); continue;
        }
        if (frame->promptTokens.size() > capacity_ ||
            uint64_t{frame->promptTokens.size()} + frame->logicalMaxOutputTokens > capacity_) {
          requestError(frame->requestId, "context_length_exceeded", "prompt and output budget exceed native context"); continue;
        }
        if (std::any_of(frame->promptTokens.begin(), frame->promptTokens.end(), [&](uint32_t token) {
              return token >= weights_.descriptor().vocabularySize;
            })) {
          requestError(frame->requestId, "invalid_request", "prompt contains an out-of-vocabulary token"); continue;
        }
        if (pending_.size() >= kPendingBound) {
          requestError(frame->requestId, "queue_capacity_exceeded", "native admission queue is full", true); continue;
        }
        auto request = std::make_unique<Request>();
        request->arrived = input.arrived;
        const auto elapsed = micros(input.arrived, Clock::now());
        const uint64_t relative = frame->remainingDeadlineMicros > elapsed
            ? frame->remainingDeadlineMicros - elapsed : 0;
        const auto remaining = std::min(frame->absoluteDeadlineUnixMicros - nowUnix, relative);
        if (!remaining) {
          requestError(frame->requestId, "deadline_exceeded", "request expired in native transport"); continue;
        }
        // Saturate very distant sender deadlines to the steady-clock horizon.
        const auto now = Clock::now();
        const auto capped = std::min<uint64_t>(remaining, static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(Time::max() - now).count()));
        request->deadline = now + std::chrono::microseconds(capped);
        request->random.seed(frame->seed); request->frame = std::move(*frame);
        if (!nextGeneration_) throw std::runtime_error("Flash admission generation space exhausted");
        request->generation = nextGeneration_++;
        live_.insert(request->frame.requestId); pending_.push_back(std::move(request));
      } else if (auto cancel = std::get_if<wire::CancelFrame>(&message)) {
        if (auto request = find(cancel->requestId)) request->cancelled = true;
      } else if (auto mask = std::get_if<wire::MaskResponseFrame>(&message)) {
        Request *request = find(mask->requestId);
        if (!request) continue;
        if (!request->maskId || request->maskId != mask->maskRequestId ||
            !request->mask.empty() || mask->maskWords.size() != words_ ||
            std::none_of(mask->maskWords.begin(), mask->maskWords.end(), [](uint32_t word) { return word != 0; })) {
          requestError(mask->requestId, "invalid_mask_response", "mask does not match the pending vocabulary row");
        } else request->mask = std::move(mask->maskWords);
      }
    }
  }
  void expire() {
    std::vector<uint64_t> expired, cancelled;
    const auto now = Clock::now();
    for (const auto id : live_) {
      const auto request = find(id);
      if (request->cancelled) cancelled.push_back(id);
      else if (request->deadline <= now) expired.push_back(id);
    }
    for (const auto id : cancelled) if (auto request = find(id)) done(*request, wire::FinishReason::Cancelled);
    for (const auto id : expired) requestError(id, "deadline_exceeded", "native request deadline elapsed");
  }
  void admit() {
    governor_.setPressure(pressure_.value());
    for (uint32_t slot = 0; slot < kConcurrent && !pending_.empty(); ++slot) {
      if (active_[slot]) continue;
      const auto preferred = std::min_element(pending_.begin(), pending_.end(), [](const auto &a, const auto &b) {
        return a->frame.priority < b->frame.priority;
      });
      const auto admissionNow = Clock::now();
      if ((*preferred)->admissionRetryAt > admissionNow) return;
      const bool eligible = head_ && mtpEligible((*preferred)->frame);
      const auto bytes = FlashForward::requestStateBytes(capacity_) +
          (eligible ? FlashMTPForward::requestStateBytes(capacity_) + singletonMTP_.hiddenBytes() : 0);
      metal::AllocationFailure failure = metal::AllocationFailure::None;
      auto reservation = governor_.tryReserve(bytes, &failure);
      if (!reservation) {
        const bool anyActive = std::any_of(active_.begin(), active_.end(), [](const auto &request) { return bool(request); });
        lastAdmissionFailure_ = failure;
        lastAdmissionBytes_ = bytes;
        lastAdmissionAt_ = admissionNow;
        lastAdmissionMemory_ = governor_.snapshot();
        ++admissionDenied_;
        Request &waiting = **preferred;
        if (retryAdmission(failure, anyActive, waiting.pressureWaitBegan, admissionNow, waiting.deadline)) {
          if (anyActive)
            waiting.pressureWaitBegan = Time{}; // Grace starts after active work drains.
          else if (failure == metal::AllocationFailure::HostPressure && waiting.pressureWaitBegan == Time{})
            waiting.pressureWaitBegan = admissionNow;
          waiting.admissionRetryAt = admissionNow + kAdmissionPressureRetry;
        } else {
          requestError(waiting.frame.requestId, "capacity_exhausted",
              std::string(metal::allocationFailureName(failure)), true);
        }
        return;
      }
      std::optional<FlashRequestState> state;
      std::optional<FlashMTPState> headState;
      metal::MetalBuffer foldHidden;
      try {
        state.emplace(forward_.createState());
        if (eligible) {
          headState.emplace(head_->createState());
          foldHidden = backend_.allocateBuffer(singletonMTP_.hiddenBytes(), metal::BufferStorage::Shared,
                                               "flash-worker-mtp-committed-hidden");
        }
      }
      catch (const metal::MetalAllocationError &error) {
        if (!backend_.healthy()) throw;
        requestError((*preferred)->frame.requestId, "capacity_exhausted", error.what(), true);
        return;
      }
      auto request = std::move(*preferred); pending_.erase(preferred);
      request->state = std::move(state);
      request->mtpState = std::move(headState);
      request->mtpFoldHidden = std::move(foldHidden);
      if (eligible) ++mtpRequests_; else ++arRequests_;
      reservation->commit();
      request->slot = slot; request->began = Clock::now();
      const auto id = request->frame.requestId;
      active_[slot] = std::move(request);
      transport_.send(wire::StartEvent{id, wire::CacheDisposition::Miss, static_cast<int32_t>(slot), 0, capacity_});
    }
  }
  void captureLogits(Request &request, const FlashForwardResult &result) {
    const uint64_t rowBytes = uint64_t{weights_.descriptor().vocabularySize} * 2;
    if (result.logicalLength != request.state->logicalLength() ||
        result.capacity != capacity_ || request.state->poisoned() ||
        !result.logitRows || result.logitsBF16.sizeBytes() < rowBytes * result.logitRows ||
        !result.logitsBF16.contents()) throw std::runtime_error("Flash logits are not host-readable BF16 rows");
    const auto row = static_cast<const uint16_t *>(result.logitsBF16.contents()) +
        uint64_t{result.logitRows - 1} * weights_.descriptor().vocabularySize;
    // Forward's scratch is borrowed. Retain one CPU row per request so another
    // request may run during this one's constraint-mask calculation.
    request.logits.assign(row, row + weights_.descriptor().vocabularySize);
  }
  void emitTokens(Request &request, std::span<const uint32_t> tokens) {
    if (tokens.empty() || tokens.size() > request.frame.logicalMaxOutputTokens - request.emitted)
      throw std::logic_error("Flash token emission exceeds its logical output budget");
    const auto now = Clock::now();
    if (!request.firstToken) { request.firstToken = now; ttft_.append(milliseconds(request.arrived, now)); }
    else itl_.append(milliseconds(*request.lastToken, now) / tokens.size());
    request.lastToken = now;
    transport_.send(wire::TokensEvent{request.frame.requestId, request.emitted,
                                     std::vector<uint32_t>(tokens.begin(), tokens.end())});
    request.emitted += static_cast<uint32_t>(tokens.size()); emitted_ += tokens.size();
    request.mask.clear(); request.maskId = 0; request.logits.clear();
    const uint32_t last = tokens.back();
    if (stopToken(last)) done(request, wire::FinishReason::Stop);
    else if (request.emitted >= request.frame.logicalMaxOutputTokens) done(request, wire::FinishReason::Length);
    else request.pendingToken = last;
  }
  void maskOrEmit(Request &request) {
    if (request.frame.constraint == wire::ConstraintMode::TokenMask && request.mask.empty()) {
      request.maskId = nextMask_++;
      if (!request.maskId) request.maskId = nextMask_++;
      transport_.send(wire::MaskRequestEvent{request.frame.requestId, request.maskId, words_, {}}); return;
    }
    const uint32_t token = selectToken(request);
    if (request.mtpState) request.mtpFoldTokens = {token};
    emitTokens(request, std::span(&token, 1));
  }
  bool safePoint(uint64_t id, uint64_t generation) {
    inFlight_ = false;
    drain(); expire();
    const auto current = find(id);
    return !transport_.stopping() && current && current->generation == generation && current->state;
  }
  void copyHidden(Request &request, metal::MetalBuffer source, uint32_t rows, uint32_t beginRow = 0) {
    const uint64_t bytes = uint64_t{rows} * kHyper * 2;
    const uint64_t offset = uint64_t{beginRow} * kHyper * 2;
    if (!rows || rows > singletonMTP_.hiddenRows() || !source.contents() ||
        source.sizeBytes() < offset + bytes || !request.mtpFoldHidden.contents() ||
        request.mtpFoldHidden.sizeBytes() < bytes)
      throw std::runtime_error("Flash MTP committed features have invalid storage or extent");
    std::memcpy(request.mtpFoldHidden.contents(), static_cast<const uint8_t *>(source.contents()) + offset, bytes);
  }
  void tickMTP(Request &request);
  metal::CommandTiming teacherPrime(FlashMTPState &state, metal::MetalBuffer hidden,
      std::span<const uint32_t> tokens) {
    if (!head_) throw std::logic_error("Flash teacher priming has no trained head");
    if (mtpTeacherCacheOnly_) {
      const auto timing = head_->primeTeacherCache(state, std::move(hidden), tokens);
      ++teacherCachePrimeCalls_;
      return timing;
    }
    const auto result = head_->forward(state, std::move(hidden), tokens, FlashMTPLogits::None);
    if (result.logitRows || result.logitsBF16)
      throw std::logic_error("Flash teacher priming unexpectedly returned vocabulary logits");
    return result.timing;
  }
  void tick(Request &request) {
    const uint64_t id = request.frame.requestId;
    const uint64_t generation = request.generation;
    if (request.promptOffset < request.frame.promptTokens.size()) {
      const auto promptBegin = request.promptOffset;
      const auto count = std::min<size_t>(prefillRows_, request.frame.promptTokens.size() - request.promptOffset);
      inFlight_ = true; publishStatus();
      const auto began = Clock::now();
      const auto result = forward_.forward(*request.state,
          std::span(request.frame.promptTokens).subspan(request.promptOffset, count), false,
          request.mtpState.has_value());
      traceRequestCommand("prefill", "target_trunk", {id, generation}, static_cast<uint32_t>(count), result.timing);
      recordPrefill(1, static_cast<uint32_t>(count), result.timing,
          std::chrono::duration<double>(Clock::now() - began).count());
      inFlight_ = false; request.promptOffset += count;
      const bool final = request.promptOffset == request.frame.promptTokens.size();
      if (final) {
        captureLogits(request, result);
        if (request.mtpState) copyHidden(request, result.hiddenBF16, 1, static_cast<uint32_t>(count - 1));
      }
      if (!safePoint(id, generation)) return;
      if (request.mtpState) {
        const auto primeRows = prefillPrimeRows(static_cast<uint32_t>(request.frame.promptTokens.size()),
            promptBegin, static_cast<uint32_t>(count));
        request.mtpPriming = primeRows != 0;
        for (size_t primeBegin = 0; primeBegin < primeRows; primeBegin += kHeadRows) {
          const auto primeCount = std::min<size_t>(kHeadRows, primeRows - primeBegin);
          const auto hidden = backend_.view(result.hiddenBF16, uint64_t{primeBegin} * kHyper * 2,
                                             uint64_t{primeCount} * kHyper * 2);
          inFlight_ = true; publishStatus();
          const auto primeBegan = Clock::now();
          const auto timing = teacherPrime(*request.mtpState, hidden,
              std::span(request.frame.promptTokens).subspan(promptBegin + primeBegin + 1, primeCount));
          traceRequestCommand("prompt_head_priming", "mtp_head", {id, generation}, static_cast<uint32_t>(primeCount), timing);
          const auto host = std::chrono::duration<double>(Clock::now() - primeBegan).count();
          mtpPrime_.add(static_cast<uint32_t>(primeCount), timing, host);
          prefill_.gpu += timing.gpuSeconds; prefill_.wall += timing.wallSeconds;
          prefill_.lastGpu += timing.gpuSeconds; prefill_.lastWall += timing.wallSeconds;
          prefill_.host += host;
          if (!safePoint(id, generation)) return;
        }
        request.mtpPriming = false;
        if (final && request.mtpState->logicalLength() + 1 != request.frame.promptTokens.size())
          throw std::logic_error("Flash MTP prompt priming offset differs");
      }
      if (request.frame.returnProgress) transport_.send(wire::PromptProgressEvent{id,
          request.promptOffset, micros(request.began, Clock::now())});
    } else if (request.pendingToken) {
      if (request.mtpState && request.frame.logicalMaxOutputTokens - request.emitted > 1) {
        tickMTP(request);
        return;
      }
      const std::array<uint32_t, 1> token{*request.pendingToken};
      inFlight_ = true; publishStatus();
      const auto began = Clock::now();
      const auto result = forward_.forward(*request.state, token);
      traceRequestCommand("autoregressive_decode", "target_trunk", {id, generation}, 1, result.timing);
      recordDecode(1, 1, result.timing, std::chrono::duration<double>(Clock::now() - began).count());
      inFlight_ = false; request.pendingToken.reset(); captureLogits(request, result);
    }
    // Apply cancellation after the GPU wait and before publishing a token.
    drain(); expire();
    if (transport_.stopping()) return;
    if (Request *live = find(id); live && live->generation == generation && !live->logits.empty()) maskOrEmit(*live);
  }
  static void appendTiming(std::ostringstream &out, const PhaseTiming &phase) {
    out << "{\"last_gpu_ms\":" << phase.lastGpu * 1000 << ",\"last_wall_ms\":" << phase.lastWall * 1000
        << ",\"total_gpu_ms\":" << phase.gpu * 1000 << ",\"total_wall_ms\":" << phase.wall * 1000
        << ",\"forward_host_wall_ms\":" << phase.host * 1000;
    const auto &host = phase.commandHost;
    out << R"(,"host_command_subphases":{"scope":"normal_command_boundary_durations","intervals_can_overlap":true,"preparation_excluded_from_command_wall":true,"scheduled_latency_scope":"commit begin to scheduled callback arrival; not actual GPU scheduling","timed_commands":)"
        << host.timedCommands << R"(,"commit_samples":)" << host.commitSamples
        << R"(,"scheduled_callback_samples":)" << host.scheduledCallbackSamples
        << R"(,"completed_callback_samples":)" << host.completedCallbackSamples
        << R"(,"pre_commit_memory_samples":)" << host.preCommitMemorySamples
        << R"(,"post_commit_memory_samples":)" << host.postCommitMemorySamples
        << R"(,"scheduled_memory_samples":)" << host.scheduledMemorySamples
        << R"(,"completed_memory_samples":)" << host.completedMemorySamples
        << R"(,"ticket_wait_calls":)" << host.ticketWaitCalls
        << R"(,"preparation_ms":)" << host.preparationSeconds * 1000
        << R"(,"encoding_ms":)" << host.encodingSeconds * 1000
        << R"(,"encoding_end_to_commit_begin_ms":)" << host.beforeCommitSeconds * 1000
        << R"(,"sparse_dependency_wait_ms":)" << host.dependencyWaitSeconds * 1000
        << R"(,"commit_ms":)" << host.commitSeconds * 1000
        << R"(,"commit_to_scheduled_callback_ms":)" << host.commitToScheduledCallbackSeconds * 1000
        << R"(,"commit_to_completed_callback_ms":)" << host.commitToCompletedCallbackSeconds * 1000
        << R"(,"completed_callback_to_wall_end_ms":)" << host.completionCallbackBeforeWallEndSeconds * 1000
        << R"(,"submission_return_latency_ms":)" << host.submissionReturnSeconds * 1000
        << R"(,"ticket_blocking_wait_ms":)" << host.ticketBlockingWaitSeconds * 1000
        << R"(,"pre_commit_memory_query_ms":)" << host.preCommitMemorySampleSeconds * 1000
        << R"(,"post_commit_memory_query_ms":)" << host.postCommitMemorySampleSeconds * 1000
        << R"(,"scheduled_memory_query_ms":)" << host.scheduledMemorySampleSeconds * 1000
        << R"(,"completed_memory_query_ms":)" << host.completedMemorySampleSeconds * 1000 << "}}";
  }
  static void appendLazyGDNCounters(std::ostringstream &out,
                                    const FlashGDNLazyRollbackCounters &counters) {
    out << R"({"scope":"GDN layer graphs built since startup; not completed GPU commands","byte_scope":"logical record and operand footprints; not physical traffic or bandwidth","layer_trial_graphs_by_rows_and_lanes":[)";
    for (uint32_t index = 0; index < counters.layer_trial_graphs_by_rows_and_lanes.size(); ++index) {
      if (index) out << ',';
      out << counters.layer_trial_graphs_by_rows_and_lanes[index];
    }
    out << R"json(],"geometry_index":"(rows-1)*4+(lanes-1)")json";
#define SPLASH_LAZY_GDN_COUNTER(name) out << ",\"" #name "\":" << counters.name
    SPLASH_LAZY_GDN_COUNTER(layer_trial_graphs_built);
    SPLASH_LAZY_GDN_COUNTER(trial_lane_rows_planned);
    SPLASH_LAZY_GDN_COUNTER(r1_bypass_trial_graphs);
    SPLASH_LAZY_GDN_COUNTER(commit_calls);
    SPLASH_LAZY_GDN_COUNTER(no_replay_commit_fastpaths);
    SPLASH_LAZY_GDN_COUNTER(full_accept_commit_fastpaths);
    SPLASH_LAZY_GDN_COUNTER(full_accepted_lanes);
    SPLASH_LAZY_GDN_COUNTER(partial_replay_graphs_built);
    SPLASH_LAZY_GDN_COUNTER(partial_replay_layer_lanes_planned);
    SPLASH_LAZY_GDN_COUNTER(partial_replay_rows_planned);
    SPLASH_LAZY_GDN_COUNTER(terminal_lanes_discarded);
    SPLASH_LAZY_GDN_COUNTER(all_terminal_commit_calls);
    SPLASH_LAZY_GDN_COUNTER(aborted_trial_calls);
    SPLASH_LAZY_GDN_COUNTER(aborted_trial_lanes);
    SPLASH_LAZY_GDN_COUNTER(logical_eager_prefix_record_bytes_reference);
    SPLASH_LAZY_GDN_COUNTER(logical_initial_state_snapshot_bytes);
    SPLASH_LAZY_GDN_COUNTER(logical_initial_history_snapshot_bytes);
    SPLASH_LAZY_GDN_COUNTER(logical_raw_qkv_copy_bytes);
    SPLASH_LAZY_GDN_COUNTER(logical_saved_prework_footprint_bytes);
    SPLASH_LAZY_GDN_COUNTER(logical_incremental_verify_record_write_bytes);
    SPLASH_LAZY_GDN_COUNTER(logical_verify_record_write_bytes_avoided);
    SPLASH_LAZY_GDN_COUNTER(logical_replay_initial_state_read_bytes);
    SPLASH_LAZY_GDN_COUNTER(logical_replay_state_write_bytes);
    SPLASH_LAZY_GDN_COUNTER(logical_replay_history_write_bytes);
    SPLASH_LAZY_GDN_COUNTER(logical_replay_operand_footprint_bytes);
#undef SPLASH_LAZY_GDN_COUNTER
    out << '}';
  }
  static void appendHCUpCounters(std::ostringstream &out, const FlashHCUpEncodedCounters &counters) {
    out << R"({"scope":"main model HC-up graph construction since startup; not completed GPU dispatches","row_bucket_scope":"exact rows0..16; bucket17 is rows greater than16","graph_build_attempts":)"
        << counters.graphBuildAttempts << R"(,"graph_build_real_rows":)" << counters.graphBuildRealRows
        << R"(,"geometry_eligible_attempts":)" << counters.geometryEligibleAttempts
        << R"(,"geometry_eligible_real_rows":)" << counters.geometryEligibleRealRows
        << R"(,"cached_encoded_calls":)" << counters.cachedEncodedCalls
        << R"(,"cached_encoded_real_rows":)" << counters.cachedEncodedRealRows
        << R"(,"skipped_dependencies_off":)" << counters.skippedDependenciesOff
        << R"(,"skipped_missing_operand":)" << counters.skippedMissingOperand
        << R"(,"skipped_unsupported_geometry":)" << counters.skippedUnsupportedGeometry
        << R"(,"graph_build_calls_by_rows":[)";
    for (uint32_t index = 0; index < counters.graphBuildCallsByRows.size(); ++index) {
      if (index) out << ',';
      out << counters.graphBuildCallsByRows[index];
    }
    out << R"(],"cached_encoded_calls_by_rows":[)";
    for (uint32_t index = 0; index < counters.cachedEncodedCallsByRows.size(); ++index) {
      if (index) out << ',';
      out << counters.cachedEncodedCallsByRows[index];
    }
    out << "]}";
  }
  void publishStatus();
  Transport &transport_;
  metal::MetalBackend &backend_;
  const FlashWeights &weights_;
  FlashForward &forward_;
  engine::MemoryGovernor &governor_;
  PressureMonitor &pressure_;
  uint32_t capacity_;
  uint64_t instance_;
  FlashMTPForward *head_ = nullptr;
  FlashBatchForward *batch_ = nullptr;
  uint32_t prefillRows_ = kDefaultPrefillRows;
  FlashBatchVerify *jointVerify_ = nullptr;
  FlashBatchMTPForward *jointHead_ = nullptr;
  metal::MetalBuffer jointHidden_;
  FlashBatchPrefill *batchPrefill_ = nullptr;
  uint32_t batchPrefillRows_ = kDefaultBatchPrefillRows;
  metal::MetalBuffer batchPrefillHidden_;
  const bool gpuPrefillCopy_ = environmentSwitch("SPLASH_FLASH_GPU_PREFILL_COPY");
  const bool mtpTeacherCacheOnly_ = environmentSwitch("SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY");
  FlashBatchMTPForward *batchPrimeHead_ = nullptr;
  metal::MetalBuffer batchPrimeInput_;
  const SavedOperandsResidencyStatus savedResidency_;
  metal::ResidencyLease savedResidencyLease_;
  std::unique_ptr<FlashRequestCommandTrace> requestCommandTrace_;
  std::unique_ptr<idle_maintenance::Maintenance> idleMaintenance_;
  const bool idleMaintenanceRequested_;
  const std::string idleMaintenanceFailureReason_;
  const uint32_t idleMaintenanceInterval_;
  idle_maintenance::Scheduler idleMaintenanceScheduler_;
  bool idleMaintenanceInFlight_ = false;
  std::string idleMaintenanceState_ = "disabled";
  uint64_t idleMaintenanceCommands_ = 0, idleMaintenancePressureSuspensions_ = 0;
  uint64_t idleMaintenanceFailures_ = 0;
  uint64_t idleMaintenanceBusySkips_ = 0, idleMaintenanceColdMisses_ = 0;
  double idleMaintenanceGPUSeconds_ = 0, idleMaintenanceWallSeconds_ = 0;
  double idleMaintenanceLastGPUSeconds_ = 0, idleMaintenanceLastWallSeconds_ = 0;
  double idleMaintenanceLastIdleSeconds_ = 0, idleMaintenanceMaximumWallSeconds_ = 0;
  const SingletonMTPPolicy singletonMTP_;
  const bool adaptiveMTP_ = environmentSwitch("SPLASH_FLASH_MTP_ADAPTIVE") && !singletonMTP_.explicitOverride;
  uint32_t words_, nextSlot_ = 0;
  uint64_t nextMask_ = 1, nextGeneration_ = 1, submitted_ = 0, completed_ = 0, cancelled_ = 0, failed_ = 0, emitted_ = 0;
  bool inFlight_ = false;
  uint64_t mtpRequests_ = 0, arRequests_ = 0, mtpCycles_ = 0, mtpCommittedHeadCalls_ = 0;
  uint64_t teacherCachePrimeCalls_ = 0;
  uint64_t drafted_ = 0, accepted_ = 0, matched_ = 0, emittedAccepted_ = 0;
  std::array<uint64_t, kFlashSingletonMaximumVerifyRows> acceptedPrefixes_{};
  std::array<uint64_t, kFlashSingletonMaximumVerifyRows> mtpDepthCycles_{};
  std::array<uint64_t, 4> decodeWidths_{};
  std::array<uint64_t, 4> prefillWidths_{};
  uint64_t batchPrefillHiddenCopiedBytes_ = 0, batchPrefillLogitsCopiedBytes_ = 0, batchPrefillDropped_ = 0;
  uint64_t batchPrefillHiddenCPUCopyBytes_ = 0, batchPrefillHiddenGPUCopyBytes_ = 0;
  uint64_t batchHeadPrimeCopiedBytes_ = 0;
  uint32_t batchHeadPrimeMaximumWidth_ = 0;
  std::array<uint64_t, kConcurrent> batchHeadPrimeWidths_{};
  uint64_t mtpBatchFallbacks_ = 0, batchCopiedLogitBytes_ = 0;
  uint64_t jointMTPCohorts_ = 0, jointMTPPartial_ = 0, jointMTPDropped_ = 0;
  uint64_t jointMTPHiddenBytes_ = 0;
  uint64_t admissionDenied_ = 0, lastAdmissionBytes_ = 0;
  uint64_t allocationQueueRechecks_ = 0, recheckAdmissions_ = 0;
  Time lastAdmissionAt_{};
  metal::AllocationFailure lastAdmissionFailure_ = metal::AllocationFailure::None;
  engine::MemoryGovernorSnapshot lastAdmissionMemory_{};
  std::array<uint64_t, 4> jointHeadWidths_{}, jointVerifyWidths_{};
  std::unordered_set<uint64_t> live_;
  std::deque<std::unique_ptr<Request>> pending_;
  std::array<std::unique_ptr<Request>, kConcurrent> active_;
  PhaseTiming prefill_, decode_;
  PhaseTiming mtpPrime_, mtpBatchPrime_, mtpHead_, mtpVerify_, mtpCommit_;
  StatusTimingWindow ttft_, itl_;
};

bool Worker::tickBatchPrefill(uint32_t primarySlot) {
  struct Member final {
    RequestCookie cookie;
    uint32_t begin = 0, primeRows = 0, primedRows = 0;
    uint64_t targetBegin = 0;
    bool final = false, hasHead = false, dropped = false;
    std::vector<uint32_t> primeTokens;
  };
  std::vector<Request *> requests;
  std::vector<FlashRequestState *> states;
  std::vector<uint32_t> remaining;
  for (uint32_t offset = 0; offset < kConcurrent; ++offset) {
    const auto &candidate = active_[(primarySlot + offset) % kConcurrent];
    if (!candidate || !readyPrefill(*candidate)) continue;
    requests.push_back(candidate.get());
    states.push_back(&*candidate->state);
    remaining.push_back(static_cast<uint32_t>(candidate->frame.promptTokens.size()) - candidate->promptOffset);
  }
  if (requests.size() < 2) return false;
  const uint32_t width = static_cast<uint32_t>(requests.size());
  const uint32_t rows = batchPrefillWindowRows(remaining, batchPrefillRows_);
  std::vector<Member> members;
  std::vector<uint32_t> tokens;
  members.reserve(width); tokens.reserve(width * rows);
  bool captureHidden = false;
  for (uint32_t lane = 0; lane < width; ++lane) {
    const auto &request = *requests[lane];
    Member member;
    member.cookie = {request.frame.requestId, request.generation};
    member.begin = request.promptOffset;
    member.targetBegin = request.state->logicalLength();
    member.final = rows == remaining[lane];
    member.hasHead = request.mtpState.has_value();
    if (member.hasHead) {
      captureHidden = true;
      if (!head_ || request.mtpState->poisoned() || request.mtpState->logicalLength() != member.begin)
        throw std::logic_error("Flash batch prefill MTP prefix differs from its real prompt window");
      member.primeRows = prefillPrimeRows(static_cast<uint32_t>(request.frame.promptTokens.size()), member.begin, rows);
      // These adjacent next-token pairs must outlive a later control drain.
      const auto nextTokens = std::span(request.frame.promptTokens).subspan(member.begin + 1, member.primeRows);
      member.primeTokens.assign(nextTokens.begin(), nextTokens.end());
    }
    const auto incoming = std::span(request.frame.promptTokens).subspan(member.begin, rows);
    tokens.insert(tokens.end(), incoming.begin(), incoming.end());
    members.push_back(std::move(member));
  }
  inFlight_ = true; publishStatus();
  const auto began = Clock::now();
  metal::MetalBuffer hiddenDestination;
  if (gpuPrefillCopy_ && captureHidden) {
    const uint64_t bytes = flashBatchPrefillHiddenCopyBytes(width, rows);
    if (!batchPrefillHidden_.contents() || batchPrefillHidden_.sizeBytes() < bytes)
      throw std::runtime_error("Flash batch prefill GPU feature destination exceeds owned staging");
    hiddenDestination = backend_.view(batchPrefillHidden_, 0, bytes);
  }
  const auto result = batchPrefill_->forwardBatch(states, tokens, rows, captureHidden, hiddenDestination);
  traceGroupCommand("prefill", "target_trunk", width, result.timing,
      [&](uint32_t lane) { return members[lane].cookie; }, [&](uint32_t) { return rows; });
  recordPrefill(width, width * rows, result.timing, std::chrono::duration<double>(Clock::now() - began).count());
  const uint32_t vocabulary = weights_.descriptor().vocabularySize;
  const uint64_t logitBytes = uint64_t{vocabulary} * 2;
  if (result.lanes != width || result.rows != rows || result.capacity != capacity_ ||
      result.logicalLengths.size() != width || !result.logitsBF16.contents() ||
      result.logitsBF16.sizeBytes() < uint64_t{width} * logitBytes)
    throw std::runtime_error("Flash batch prefill returned invalid real lane logits or geometry");
  if (result.hiddenDeliveredToDestination != bool(hiddenDestination) ||
      (hiddenDestination && !result.hiddenBF16.sameView(hiddenDestination)))
    throw std::runtime_error("Flash batch prefill GPU feature delivery differs from its owned destination");
  if (captureHidden) {
    const uint64_t bytes = uint64_t{width} * rows * kHyper * 2;
    if (!result.hiddenBF16.contents() || result.hiddenBF16.sizeBytes() < bytes ||
        !batchPrefillHidden_.contents() || batchPrefillHidden_.sizeBytes() < bytes)
      throw std::runtime_error("Flash batch prefill true target features exceed owned staging");
    // The producer graph or CPU copy owns every real lane before cancellation
    // can destroy request storage. Head calls never borrow the prefill arena.
    if (result.hiddenDeliveredToDestination) batchPrefillHiddenGPUCopyBytes_ += bytes;
    else {
      std::memcpy(batchPrefillHidden_.contents(), result.hiddenBF16.contents(), bytes);
      batchPrefillHiddenCPUCopyBytes_ += bytes;
    }
    batchPrefillHiddenCopiedBytes_ += bytes;
  }
  for (uint32_t lane = 0; lane < width; ++lane) {
    auto &request = *requests[lane];
    const auto &member = members[lane];
    if (!forward_.ownsState(*request.state) || result.logicalLengths[lane] != member.targetBegin + rows ||
        request.state->logicalLength() != result.logicalLengths[lane] || request.promptOffset != member.begin)
      throw std::logic_error("Flash batch prefill advanced a different request offset");
    request.promptOffset += rows;
    if (member.final) {
      const auto *row = static_cast<const uint16_t *>(result.logitsBF16.contents()) + uint64_t{lane} * vocabulary;
      request.logits.assign(row, row + vocabulary);
      batchPrefillLogitsCopiedBytes_ += logitBytes;
      if (member.hasHead) copyHidden(request, batchPrefillHidden_, 1, lane * rows + rows - 1);
    }
    request.mtpPriming = member.hasHead && member.primeRows != 0;
  }
  const auto liveMember = [&](const Member &member) -> Request * {
    Request *request = find(member.cookie.id);
    if (!prefillMemberMatches(request, member.cookie, member.begin + rows, Clock::now()) ||
        !request->state || !forward_.ownsState(*request->state)) return nullptr;
    return request;
  };
  const auto drop = [&](Member &member) {
    if (!member.dropped) { member.dropped = true; ++batchPrefillDropped_; }
  };
  inFlight_ = false; drain(); expire();
  if (transport_.stopping()) return true;
  if (batchPrimeHead_) {
    while (true) {
      std::vector<uint32_t> pendingLanes, pendingCounts;
      for (uint32_t lane = 0; lane < width; ++lane) {
        auto &member = members[lane];
        if (!member.hasHead || member.primedRows == member.primeRows || member.dropped) continue;
        Request *request = liveMember(member);
        if (!request) { drop(member); continue; }
        if (!request->mtpState || request->mtpState->poisoned() ||
            request->mtpState->logicalLength() != member.begin + member.primedRows)
          throw std::logic_error("Flash grouped head priming lost its exact adjacent pair offset");
        pendingLanes.push_back(lane);
        pendingCounts.push_back(std::min(kHeadRows, member.primeRows - member.primedRows));
      }
      if (pendingLanes.empty()) break;
      // Equal-count peers retain the source's sequential arithmetic policy.
      // A lone peer or an unqualified tail uses the original head call.
      std::vector<uint32_t> selectedLanes, counts;
      for (uint32_t index : selectedHeadPrimePositions(pendingCounts)) {
        selectedLanes.push_back(pendingLanes[index]); counts.push_back(pendingCounts[index]);
      }
      const bool grouped = selectedLanes.size() > 1;
      std::vector<FlashMTPState *> headStates;
      std::vector<uint32_t> compactTokens;
      std::vector<uint64_t> expectedLengths;
      uint32_t totalRows = 0;
      bool rebuildCohort = false;
      for (uint32_t index = 0; index < selectedLanes.size(); ++index) {
        const uint32_t lane = selectedLanes[index], count = counts[index];
        const auto &member = members[lane];
        Request *request = liveMember(member);
        if (!request) {
          // Time may cross a deadline while peers' compact features are
          // copied. Drop that ordinary terminal and rebuild before using
          // any collected state pointer or submitting a partial cohort.
          drop(members[lane]); rebuildCohort = true; break;
        }
        if (!request->mtpState ||
            request->mtpState->logicalLength() != member.begin + member.primedRows)
          throw std::logic_error("Flash grouped head priming changed generation before submission");
        headStates.push_back(&*request->mtpState);
        expectedLengths.push_back(request->mtpState->logicalLength() + count);
        const auto next = std::span(member.primeTokens).subspan(member.primedRows, count);
        compactTokens.insert(compactTokens.end(), next.begin(), next.end());
        if (grouped) {
          const uint64_t sourceOffset = headPrimeSourceOffset(lane, rows, member.primedRows, count);
          const uint64_t destinationOffset = uint64_t{totalRows} * kHyper * 2;
          const uint64_t bytes = uint64_t{count} * kHyper * 2;
          if (!batchPrimeInput_.contents() || destinationOffset + bytes > batchPrimeInput_.sizeBytes() ||
              sourceOffset + bytes > batchPrefillHidden_.sizeBytes())
            throw std::runtime_error("Flash grouped head priming exceeds owned real feature storage");
          std::memcpy(static_cast<uint8_t *>(batchPrimeInput_.contents()) + destinationOffset,
              static_cast<const uint8_t *>(batchPrefillHidden_.contents()) + sourceOffset, bytes);
          batchHeadPrimeCopiedBytes_ += bytes;
        }
        totalRows += count;
      }
      if (rebuildCohort) {
        inFlight_ = false; drain(); expire();
        if (transport_.stopping()) return true;
        continue;
      }
      inFlight_ = true; publishStatus();
      const auto primeBegan = Clock::now();
      metal::CommandTiming timing;
      if (grouped) {
        const auto primed = batchPrimeHead_->forward(headStates,
            backend_.view(batchPrimeInput_, 0, uint64_t{totalRows} * kHyper * 2),
            compactTokens, counts, FlashMTPLogits::None);
        if (primed.lanes != selectedLanes.size() || primed.logitRows || primed.logitsBF16 ||
            primed.logicalLengths != expectedLengths || primed.laneOffsets.size() != selectedLanes.size() + 1 ||
            primed.laneOffsets.back() != totalRows || primed.hiddenBF16.sizeBytes() < uint64_t{totalRows} * kHyper * 2)
          throw std::runtime_error("Flash grouped head priming returned invalid real pair metadata");
        uint32_t compactOffset = 0;
        for (uint32_t index = 0; index < counts.size(); ++index) {
          if (primed.laneOffsets[index] != compactOffset)
            throw std::runtime_error("Flash grouped head priming changed its compact real lane offsets");
          compactOffset += counts[index];
        }
        timing = primed.timing;
      } else {
        const auto &member = members[selectedLanes.front()];
        const auto hidden = backend_.view(batchPrefillHidden_,
            headPrimeSourceOffset(selectedLanes.front(), rows, member.primedRows, totalRows),
            uint64_t{totalRows} * kHyper * 2);
        timing = teacherPrime(*headStates.front(), hidden, compactTokens);
        if (headStates.front()->logicalLength() != expectedLengths.front())
          throw std::runtime_error("Flash sequential head priming returned invalid real pair metadata");
      }
      const auto host = std::chrono::duration<double>(Clock::now() - primeBegan).count();
      traceGroupCommand("prompt_head_priming", "mtp_head", static_cast<uint32_t>(selectedLanes.size()), timing,
          [&](uint32_t lane) { return members[selectedLanes[lane]].cookie; },
          [&](uint32_t lane) { return counts[lane]; });
      for (uint32_t index = 0; index < selectedLanes.size(); ++index) {
        if (headStates[index]->logicalLength() != expectedLengths[index] || headStates[index]->poisoned())
          throw std::logic_error("Flash grouped head priming advanced a different request prefix");
        members[selectedLanes[index]].primedRows += counts[index];
      }
      mtpPrime_.add(totalRows, timing, host);
      if (grouped) {
        mtpBatchPrime_.add(totalRows, timing, host);
        ++batchHeadPrimeWidths_[selectedLanes.size() - 1];
        batchHeadPrimeMaximumWidth_ = std::max(batchHeadPrimeMaximumWidth_,
            static_cast<uint32_t>(selectedLanes.size()));
      }
      prefill_.gpu += timing.gpuSeconds; prefill_.wall += timing.wallSeconds;
      prefill_.lastGpu += timing.gpuSeconds; prefill_.lastWall += timing.wallSeconds;
      prefill_.host += host;
      // No request pointer survives this boundary. Every next compact cohort
      // revalidates its generation, cancellation, deadline, and exact offset.
      inFlight_ = false; drain(); expire();
      if (transport_.stopping()) return true;
    }
  }
  for (uint32_t lane = 0; lane < width; ++lane) {
    auto &member = members[lane];
    Request *request = liveMember(member);
    if (!request) { drop(member); continue; }
    if (member.hasHead) {
      for (uint32_t primeBegin = 0; !batchPrimeHead_ && primeBegin < member.primeRows; primeBegin += kHeadRows) {
        request = liveMember(member);
        if (!request) break;
        if (!request->mtpState || request->mtpState->poisoned() ||
            request->mtpState->logicalLength() != member.begin + primeBegin)
          throw std::logic_error("Flash batch prefill MTP priming lost its exact adjacent pair offset");
        const uint32_t count = std::min(kHeadRows, member.primeRows - primeBegin);
        const auto hidden = backend_.view(batchPrefillHidden_, batchPrefillHiddenOffset(lane, rows, primeBegin),
            uint64_t{count} * kHyper * 2);
        inFlight_ = true; publishStatus();
        const auto primeBegan = Clock::now();
        const auto timing = teacherPrime(*request->mtpState, hidden,
            std::span(member.primeTokens).subspan(primeBegin, count));
        traceRequestCommand("prompt_head_priming", "mtp_head", member.cookie, count, timing);
        const auto host = std::chrono::duration<double>(Clock::now() - primeBegan).count();
        mtpPrime_.add(count, timing, host);
        prefill_.gpu += timing.gpuSeconds; prefill_.wall += timing.wallSeconds;
        prefill_.lastGpu += timing.gpuSeconds; prefill_.lastWall += timing.wallSeconds;
        prefill_.host += host;
        if (!safePoint(member.cookie.id, member.cookie.generation)) {
          if (transport_.stopping()) return true;
          request = nullptr; break;
        }
      }
      request = liveMember(member);
      if (!request) { drop(member); continue; }
      if (batchPrimeHead_ && (!request->mtpState || request->mtpState->poisoned() ||
          member.primedRows != member.primeRows ||
          request->mtpState->logicalLength() != member.begin + member.primeRows))
        throw std::logic_error("Flash grouped head priming did not consume its complete real prompt pairs");
      request->mtpPriming = false;
      if (!request->mtpState || (member.final &&
          request->mtpState->logicalLength() + 1 != request->frame.promptTokens.size()))
        throw std::logic_error("Flash batch prefill final MTP prompt priming offset differs");
    }
    if (request->frame.returnProgress)
      transport_.send(wire::PromptProgressEvent{member.cookie.id, request->promptOffset,
          micros(request->began, Clock::now())});
  }
  // A cancelled/expired generation may leave the cohort at any priming
  // boundary. Surviving lanes keep their original feature offsets and state.
  drain(); expire();
  if (transport_.stopping()) return true;
  for (auto &member : members) {
    Request *request = liveMember(member);
    if (!request) { drop(member); continue; }
    if (!request->mtpPriming && !request->logits.empty()) maskOrEmit(*request);
  }
  return true;
}

bool Worker::tickJointMTP(uint32_t primarySlot) {
  std::vector<JointMTPMember> members;
  for (uint32_t offset = 0; offset < kConcurrent; ++offset) {
    const auto &request = active_[(primarySlot + offset) % kConcurrent];
    if (!request || !readyJointMTP(*request)) continue;
    JointMTPMember member;
    member.cookie = {request->frame.requestId, request->generation};
    member.targetBegin = request->state->logicalLength();
    member.remaining = request->frame.logicalMaxOutputTokens - request->emitted;
    member.inputs = {*request->pendingToken};
    members.push_back(std::move(member));
  }
  if (members.size() < 2) return false; // No GPU/control work has happened.
  ++jointMTPCohorts_;
  const auto began = Clock::now();
  metal::CommandTiming total;
  const auto accumulate = [&](metal::CommandTiming timing) {
    total.gpuSeconds += timing.gpuSeconds; total.wallSeconds += timing.wallSeconds;
    total.host.add(timing.host);
  };
  const auto partial = [&] {
    ++jointMTPPartial_;
    // No target verifier ran, so do not invent a decoder width/target batch.
    decode_.gpu += total.gpuSeconds; decode_.wall += total.wallSeconds;
    decode_.host += std::chrono::duration<double>(Clock::now() - began).count();
  };
  const auto prune = [&] {
    const auto before = members.size();
    std::erase_if(members, [&](const auto &member) { return !jointMember(member); });
    jointMTPDropped_ += before - members.size();
  };
  std::vector<uint32_t> budgets;
  for (const auto &member : members) budgets.push_back(member.remaining);
  const uint32_t depth = jointDepth(budgets);
  const auto headCall = [&](bool committedFold, bool wantLogits) {
    std::vector<FlashMTPState *> states;
    std::vector<uint32_t> tokens, counts;
    std::vector<uint64_t> expectedLengths;
    uint32_t packed = 0;
    if (!jointHidden_.contents() || jointHidden_.sizeBytes() < kJointHiddenBytes)
      throw std::logic_error("joint MTP hidden staging is not owned shared storage");
    auto *destination = static_cast<uint16_t *>(jointHidden_.contents());
    for (const auto &member : members) {
      Request *request = jointMember(member);
      if (!request) throw std::logic_error("joint MTP member disappeared outside control boundary");
      const uint32_t count = committedFold ? static_cast<uint32_t>(request->mtpFoldTokens.size()) : 1;
      if (!count || count > kMTPDepth + 1 || packed + count > kConcurrent * (kMTPDepth + 1))
        throw std::logic_error("joint MTP compact pair count exceeds owned staging");
      const uint64_t bytes = uint64_t{count} * kHyper * 2;
      if (committedFold) {
        if (!request->mtpFoldHidden.contents() || request->mtpFoldHidden.sizeBytes() < bytes ||
            request->mtpState->logicalLength() + count != member.targetBegin)
          throw std::logic_error("joint MTP committed pair window does not match its target");
        std::memcpy(destination + uint64_t{packed} * kHyper, request->mtpFoldHidden.contents(), bytes);
        tokens.insert(tokens.end(), request->mtpFoldTokens.begin(), request->mtpFoldTokens.end());
      } else {
        if (member.lastHeadHidden.size() != kHyper || member.inputs.size() < 2)
          throw std::logic_error("joint MTP chain lacks an owned head feature/token pair");
        std::memcpy(destination + uint64_t{packed} * kHyper, member.lastHeadHidden.data(), bytes);
        tokens.push_back(member.inputs.back());
      }
      packed += count;
      counts.push_back(count);
      states.push_back(&*request->mtpState);
      expectedLengths.push_back(request->mtpState->logicalLength() + count);
    }
    inFlight_ = true; publishStatus();
    const auto commandBegan = Clock::now();
    const auto result = jointHead_->forward(states,
        backend_.view(jointHidden_, 0, uint64_t{packed} * kHyper * 2), tokens, counts,
        wantLogits ? FlashMTPLogits::Last : FlashMTPLogits::None);
    traceGroupCommand(committedFold ? "committed_head_fold" : "draft_head_chain", "mtp_head",
        static_cast<uint32_t>(members.size()), result.timing,
        [&](uint32_t lane) { return members[lane].cookie; }, [&](uint32_t lane) { return counts[lane]; });
    mtpHead_.add(packed, result.timing, std::chrono::duration<double>(Clock::now() - commandBegan).count());
    ++jointHeadWidths_[members.size() - 1];
    accumulate(result.timing);
    if (result.lanes != members.size() || result.logicalLengths != expectedLengths ||
        result.laneOffsets.size() != members.size() + 1 || result.laneOffsets.front() ||
        result.laneOffsets.back() != packed || result.logitRows != (wantLogits ? members.size() : 0) ||
        !result.hiddenBF16.contents() || result.hiddenBF16.sizeBytes() < uint64_t{packed} * kHyper * 2)
      throw std::logic_error("joint MTP head returned invalid compact borrowed geometry");
    uint32_t expectedOffset = 0;
    for (uint32_t lane = 0; lane < members.size(); ++lane) {
      expectedOffset += counts[lane];
      if (result.laneOffsets[lane + 1] != expectedOffset)
        throw std::logic_error("joint MTP head compact lane offsets differ from real pair counts");
      auto &member = members[lane];
      const auto *source = static_cast<const uint16_t *>(result.hiddenBF16.contents()) +
          uint64_t{compactLastRow(result.laneOffsets, lane)} * kHyper;
      member.lastHeadHidden.assign(source, source + kHyper);
      if (wantLogits) member.nextDraft = greedyRow(result.logitsBF16,
          weights_.descriptor().vocabularySize, lane, result.greedyResultsU32);
      if (committedFold) {
        member.foldedLength = result.logicalLengths[lane];
        if (member.foldedLength != member.targetBegin)
          throw std::logic_error("joint MTP post-fold head offset differs from target");
      }
      jointMTPHiddenBytes_ += uint64_t{kHyper} * 2;
    }
    // Last logits are lane-major; hidden is compact. Both have been copied
    // BEFORE cancellation/deadline processing or the next head/trunk call.
    jointControls();
    prune();
  };

  headCall(true, depth != 0);
  if (transport_.stopping() || members.empty()) { partial(); return true; }
  for (uint32_t proposal = 0; proposal < depth; ++proposal) {
    bool anyEOS = false;
    for (auto &member : members) {
      member.inputs.push_back(member.nextDraft);
      ++drafted_;
      anyEOS |= stopToken(member.nextDraft);
    }
    // EOS is only a proposal. Include it in target verification; never invent
    // another post-EOS proposal or pad other lanes to a longer target window.
    if (anyEOS || proposal + 1 == depth) break;
    headCall(false, true);
    if (transport_.stopping() || members.empty()) { partial(); return true; }
  }
  jointControls(); prune();
  if (transport_.stopping() || members.empty()) { partial(); return true; }
  std::vector<uint32_t> realCounts, realBudgets;
  std::vector<FlashRequestState *> targetStates;
  std::vector<uint32_t> inputs;
  for (const auto &member : members) {
    realCounts.push_back(static_cast<uint32_t>(member.inputs.size()));
    realBudgets.push_back(member.remaining);
  }
  const uint32_t rows = jointWindowRows(realCounts, realBudgets);
  for (const auto &member : members) {
    Request *request = jointMember(member);
    if (!request || !forward_.ownsState(*request->state))
      throw std::logic_error("joint MTP target cohort contains a pending/foreign request");
    targetStates.push_back(&*request->state);
    inputs.insert(inputs.end(), member.inputs.begin(), member.inputs.begin() + rows);
  }
  const uint32_t width = static_cast<uint32_t>(members.size());
  inFlight_ = true; publishStatus();
  auto commandBegan = Clock::now();
  const auto verified = jointVerify_->verifyBatch(targetStates, inputs, rows);
  traceGroupCommand("target_verify", "target_trunk", static_cast<uint32_t>(members.size()), verified.timing,
      [&](uint32_t lane) { return members[lane].cookie; }, [&](uint32_t) { return rows; });
  mtpVerify_.add(width * rows, verified.timing, std::chrono::duration<double>(Clock::now() - commandBegan).count());
  accumulate(verified.timing);
  ++mtpCycles_; ++jointVerifyWidths_[width - 1];
  if (verified.lanes != width || verified.rows != rows || verified.capacity != capacity_ ||
      verified.logicalLengths.size() != width || !verified.hiddenBF16.contents() ||
      verified.hiddenBF16.sizeBytes() < uint64_t{width} * rows * kHyper * 2)
    throw std::logic_error("joint target verifier returned invalid lane-major geometry");
  // Preserve original target slots after this point. Resolution includes zero
  // retained/null canceled slots so healthy peers are never abandoned.
  for (uint32_t lane = 0; lane < width; ++lane) {
    auto &member = members[lane];
    if (verified.logicalLengths[lane] != member.targetBegin + rows)
      throw std::logic_error("joint target verifier consumed a different causal lane offset");
    std::vector<uint32_t> predictions;
    for (uint32_t row = 0; row < rows; ++row)
      predictions.push_back(greedyRow(verified.logitsBF16, weights_.descriptor().vocabularySize,
          lane * rows + row, verified.greedyResultsU32));
    member.accepted = verifiedGreedy(std::span(member.inputs).first(rows), predictions, member.remaining);
    const uint32_t retained = member.accepted.retained();
    const auto *hidden = static_cast<const uint16_t *>(verified.hiddenBF16.contents()) +
        uint64_t{lane} * rows * kHyper;
    member.trueTargetHidden.assign(hidden, hidden + uint64_t{retained} * kHyper);
    jointMTPHiddenBytes_ += uint64_t{retained} * kHyper * 2;
  }
  jointControls();
  if (transport_.stopping()) {
    jointVerify_->abortBatch();
    recordDecode(width, 0, total, std::chrono::duration<double>(Clock::now() - began).count());
    return true;
  }
  std::vector<uint32_t> retained(width);
  for (uint32_t lane = 0; lane < width; ++lane) {
    Request *request = jointMember(members[lane]);
    targetStates[lane] = request ? &*request->state : nullptr;
    retained[lane] = request ? members[lane].accepted.retained() : 0;
    if (!request) ++jointMTPDropped_;
  }
  inFlight_ = true; publishStatus();
  commandBegan = Clock::now();
  const auto committed = jointVerify_->commitBatch(targetStates, retained);
  traceGroupCommand("target_prefix_restore", "target_restore", width, committed,
      [&](uint32_t lane) { return members[lane].cookie; }, [&](uint32_t lane) { return retained[lane]; });
  if (committed.gpuSeconds > 0 || committed.wallSeconds > 0)
    mtpCommit_.add(width, committed, std::chrono::duration<double>(Clock::now() - commandBegan).count());
  accumulate(committed);
  uint32_t prepared = 0;
  for (uint32_t lane = 0; lane < width; ++lane) {
    if (!retained[lane]) continue;
    auto &member = members[lane];
    Request *request = jointMember(member);
    if (!request) throw std::logic_error("joint MTP committed member disappeared outside control boundary");
    head_->truncate(*request->mtpState, member.foldedLength);
    const uint64_t bytes = uint64_t{retained[lane]} * kHyper * 2;
    if (!request->mtpFoldHidden.contents() || request->mtpFoldHidden.sizeBytes() < bytes ||
        member.trueTargetHidden.size() * 2 != bytes)
      throw std::logic_error("joint MTP retained features do not fit owned pair storage");
    std::memcpy(request->mtpFoldHidden.contents(), member.trueTargetHidden.data(), bytes);
    request->mtpFoldTokens = member.accepted.output;
    request->pendingToken.reset();
    if (request->state->logicalLength() != member.targetBegin + retained[lane] ||
        request->state->logicalLength() + 1 != request->frame.promptTokens.size() + request->emitted + retained[lane])
      throw std::logic_error("joint MTP exact committed target prefix differs");
    prepared += retained[lane];
    matched_ += member.accepted.matched; accepted_ += retained[lane] - 1;
    ++acceptedPrefixes_[retained[lane] - 1];
  }
  recordDecode(width, prepared, total, std::chrono::duration<double>(Clock::now() - began).count());
  jointControls();
  if (transport_.stopping()) return true;
  for (uint32_t lane = 0; lane < width; ++lane) {
    if (!retained[lane]) continue;
    auto &member = members[lane];
    if (Request *request = jointMember(member)) {
      emittedAccepted_ += std::min(member.accepted.matched, retained[lane]);
      emitTokens(*request, member.accepted.output);
    }
  }
  return true;
}

void Worker::tickMTP(Request &request) {
  const uint64_t id = request.frame.requestId;
  const uint64_t generation = request.generation;
  if (!head_ || !request.mtpState || !request.pendingToken || !mtpEligible(request.frame) ||
      request.mtpFoldTokens.empty() || request.mtpFoldTokens.size() > singletonMTP_.hiddenRows() ||
      request.mtpState->poisoned() || request.state->poisoned())
    throw std::logic_error("Flash MTP request has inconsistent greedy state");
  const uint32_t remaining = request.frame.logicalMaxOutputTokens - request.emitted;
  // An oversized prior singleton fold is consumed here before that request
  // can join the unchanged four-row joint head. Once another MTP peer is
  // ready or awaiting admission, this cycle leaves at most four new pairs.
  const bool peers = !pending_.empty() || std::any_of(active_.begin(), active_.end(),
      [&](const auto &candidate) {
        return candidate && candidate.get() != &request && candidate->mtpState && readyBatch(*candidate);
      });
  const uint32_t allowedDepth = singletonAllowedDepth(singletonMTP_.maximumDepth, remaining, peers);
  const uint32_t depth = adaptiveMTP_ ? request.mtpDepth.select(allowedDepth) : allowedDepth;
  const uint64_t targetBegin = request.state->logicalLength();
  if (request.mtpState->logicalLength() + request.mtpFoldTokens.size() != targetBegin)
    throw std::logic_error("Flash MTP committed fold offset differs from target");
  metal::CommandTiming total;
  const auto accumulate = [&](metal::CommandTiming timing) {
    total.gpuSeconds += timing.gpuSeconds; total.wallSeconds += timing.wallSeconds;
    total.host.add(timing.host);
  };
  const auto began = Clock::now();
  const auto stillActive = [&] {
    if (safePoint(id, generation)) return true;
    recordDecode(1, 0, total, std::chrono::duration<double>(Clock::now() - began).count());
    return false;
  };
  FlashMTPResult folded;
  auto commandBegan = Clock::now();
  // Preserve the qualified raw head coefficient policy: a full sixteen-row
  // fold would switch dense/router projections to BF16 cached matrices.
  // Cancellation/cookie checks occur after every actual ordered chunk.
  for (uint32_t begin = 0; begin < request.mtpFoldTokens.size();) {
    const auto count = flashMTPCommittedFoldChunkRows(
        static_cast<uint32_t>(request.mtpFoldTokens.size() - begin));
    if (!count) throw std::logic_error("Flash committed head fold exceeds bounded chunks");
    const bool last = begin + *count == request.mtpFoldTokens.size();
    inFlight_ = true; publishStatus();
    commandBegan = Clock::now();
    folded = head_->forward(*request.mtpState,
        backend_.view(request.mtpFoldHidden, uint64_t{begin} * kHyper * 2, uint64_t{*count} * kHyper * 2),
        std::span(request.mtpFoldTokens).subspan(begin, *count),
        last && depth ? FlashMTPLogits::Last : FlashMTPLogits::None);
    traceRequestCommand("committed_head_fold", "mtp_head", {id, generation}, *count, folded.timing);
    mtpHead_.add(*count, folded.timing, std::chrono::duration<double>(Clock::now() - commandBegan).count());
    accumulate(folded.timing);
    ++mtpCommittedHeadCalls_;
    begin += *count;
    if (!stillActive()) return;
  }
  const uint64_t foldedLength = request.mtpState->logicalLength();
  if (foldedLength != targetBegin) throw std::logic_error("Flash MTP folded offset differs from target");
  if (!stillActive()) return;
  std::vector<uint32_t> inputs{*request.pendingToken};
  auto current = folded;
  for (uint32_t draft = 0; draft < depth; ++draft) {
    const uint32_t token = greedyRow(current.logitsBF16, weights_.descriptor().vocabularySize,
        0, current.greedyResultsU32);
    inputs.push_back(token); ++drafted_;
    if (stopToken(token)) break;
    if (draft + 1 < depth) {
      const auto hidden = backend_.view(current.hiddenBF16,
          uint64_t{current.hiddenRows - 1} * kHyper * 2, uint64_t{kHyper} * 2);
      inFlight_ = true; publishStatus();
      commandBegan = Clock::now();
      current = head_->forward(*request.mtpState, hidden, std::span(&token, 1));
      traceRequestCommand("draft_head_chain", "mtp_head", {id, generation}, 1, current.timing);
      mtpHead_.add(1, current.timing, std::chrono::duration<double>(Clock::now() - commandBegan).count());
      accumulate(current.timing);
      if (!stillActive()) return;
    }
  }
  inFlight_ = true; publishStatus();
  commandBegan = Clock::now();
  const auto verified = depth ? forward_.verify(*request.state, inputs)
                              : forward_.forward(*request.state, inputs, false, true);
  traceRequestCommand(depth ? "target_verify" : "autoregressive_decode", "target_trunk",
      {id, generation}, static_cast<uint32_t>(inputs.size()), verified.timing);
  if (depth) {
    mtpVerify_.add(static_cast<uint32_t>(inputs.size()), verified.timing,
        std::chrono::duration<double>(Clock::now() - commandBegan).count());
    ++mtpCycles_;
  }
  accumulate(verified.timing);
  // remove() aborts any live provisional tape before releasing a terminal
  // request. No other trunk can run while that tape remains unresolved.
  if (!stillActive()) {
    if (depth)
      if (auto live = find(id); live && live->generation == generation && live->state) forward_.abortVerify(*live->state);
    return;
  }
  if (verified.logitRows != inputs.size() || verified.logicalLength != targetBegin + inputs.size())
    throw std::logic_error("Flash MTP verifier returned a different window");
  std::vector<uint32_t> predictions;
  for (uint32_t row = 0; row < verified.logitRows; ++row)
    predictions.push_back(greedyRow(verified.logitsBF16, weights_.descriptor().vocabularySize,
        row, verified.greedyResultsU32));
  const auto accepted = verifiedGreedy(inputs, predictions, remaining);
  matched_ += accepted.matched;
  const uint32_t retained = accepted.retained();
  inFlight_ = true; publishStatus();
  commandBegan = Clock::now();
  const auto committed = depth ? forward_.commitVerify(*request.state, retained) : metal::CommandTiming{};
  if (depth) traceRequestCommand("target_prefix_restore", "target_restore", {id, generation}, retained, committed);
  if (committed.wallSeconds > 0 || committed.gpuSeconds > 0)
    mtpCommit_.add(retained, committed, std::chrono::duration<double>(Clock::now() - commandBegan).count());
  accumulate(committed);
  head_->truncate(*request.mtpState, foldedLength);
  // Both target and head results borrow shared arenas. Copy the true target
  // features and committed pair tokens before a round-robin peer can run.
  copyHidden(request, verified.hiddenBF16, retained);
  request.mtpFoldTokens = accepted.output;
  request.pendingToken.reset();
  if (request.state->logicalLength() != targetBegin + retained ||
      request.state->logicalLength() + 1 != request.frame.promptTokens.size() + request.emitted + retained)
    throw std::logic_error("Flash MTP exact committed target offset differs");
  accepted_ += retained - 1;
  ++acceptedPrefixes_[retained - 1];
  ++mtpDepthCycles_[inputs.size() - 1];
  const double cycleSeconds = std::chrono::duration<double>(Clock::now() - began).count();
  if (adaptiveMTP_)
    request.mtpDepth.observe(static_cast<uint32_t>(inputs.size() - 1), accepted.matched,
        cycleSeconds * 1000.0, !accepted.finish && remaining > retained && allowedDepth == singletonMTP_.maximumDepth);
  recordDecode(1, retained, total, cycleSeconds);
  if (!safePoint(id, generation)) return;
  emittedAccepted_ += std::min(accepted.matched, retained);
  emitTokens(request, accepted.output);
}

void Worker::tickIdleMaintenance() {
  if (!idleMaintenance_) {
    if (idleMaintenanceRequested_ && idleMaintenanceState_ == "disabled")
      idleMaintenanceState_ = "unavailable_immutable_owner_union";
    return;
  }
  if (!idleMaintenanceScheduler_.armed()) {
    if (!idleMaintenanceColdMisses_) idleMaintenanceState_ = "waiting_for_successful_user_request";
    return;
  }
  const auto now = Clock::now();
  if (!idleMaintenanceScheduler_.due(now)) return;
  const bool activeEmpty = std::none_of(active_.begin(), active_.end(),
      [](const auto &request) { return bool(request); });
  if (!idle_maintenance::eligible(idleMaintenanceScheduler_.armed(), activeEmpty, pending_.empty(),
      live_.empty(), inFlight_, backend_.needsHealthCheck(), transport_.stopping(),
      backend_.healthy(), transport_.incomingEmpty())) {
    idleMaintenanceState_ = "request_or_command_pending"; ++idleMaintenanceBusySkips_; return;
  }
  governor_.setPressure(pressure_.value());
  const auto host = governor_.snapshot();
  if (host.reservedBytes || !idle_maintenance::hostAllowed(host.hostMeasurementValid,
      host.growthAllowed, host.pressure == engine::MemoryPressure::Normal,
      host.systemPressure == engine::MemoryPressure::Normal,
      host.hostAvailableBytes, host.hostReserveBytes)) {
    idleMaintenanceState_ = "suspended_for_host_reserve_or_pressure";
    ++idleMaintenancePressureSuspensions_;
    idleMaintenanceScheduler_.pressureSuspended(now); return;
  }
  // A reader arrival after this last check can overlap this one synchronous
  // command. No transport mutex is held during GPU submission/wait, so the
  // reader queues controls immediately; drain runs again before model work.
  if (!transport_.incomingEmpty() || transport_.stopping()) {
    idleMaintenanceState_ = "request_arrived_before_submission"; ++idleMaintenanceBusySkips_; return;
  }
  idleMaintenanceLastIdleSeconds_ = std::chrono::duration<double>(now - idleMaintenanceScheduler_.lastUserGPUCommand()).count();
  idleMaintenanceInFlight_ = true; idleMaintenanceState_ = "maintenance_command_in_flight";
  publishStatus();
  if (!transport_.incomingEmpty() || transport_.stopping() || !backend_.healthy() || backend_.needsHealthCheck()) {
    idleMaintenanceInFlight_ = false;
    idleMaintenanceState_ = "request_or_command_arrived_before_submission";
    ++idleMaintenanceBusySkips_; return;
  }
  metal::CommandTiming timing;
  bool writingMaintenanceTrace = false;
  try {
    timing = idleMaintenance_->run(backend_);
    if (requestCommandTrace_) {
      writingMaintenanceTrace = true;
      requestCommandTrace_->afterCall(backend_, instance_, "idle_immutable_resource_maintenance",
          "immutable_resource_maintenance", {}, timing);
      writingMaintenanceTrace = false;
    }
  } catch (const std::exception &) {
    idleMaintenanceInFlight_ = false; ++idleMaintenanceFailures_; idleMaintenanceScheduler_.failed();
    idleMaintenanceState_ = "disabled_after_maintenance_failure";
    if (!backend_.healthy()) throw;
    // Drain any completed diagnostic profile before future request attribution.
    (void)backend_.takeCommandDispatchProfiles();
    if (writingMaintenanceTrace) {
      requestCommandTrace_.reset();
      backend_.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::Off);
    }
    idleMaintenance_.reset(); drain(); expire(); publishStatus(); return;
  }
  idleMaintenanceInFlight_ = false; ++idleMaintenanceCommands_;
  idleMaintenanceGPUSeconds_ += timing.gpuSeconds; idleMaintenanceWallSeconds_ += timing.wallSeconds;
  idleMaintenanceLastGPUSeconds_ = timing.gpuSeconds; idleMaintenanceLastWallSeconds_ = timing.wallSeconds;
  idleMaintenanceMaximumWallSeconds_ = std::max(idleMaintenanceMaximumWallSeconds_, timing.wallSeconds);
  if (timing.wallSeconds >= double(idleMaintenanceInterval_) / 1000.0) {
    ++idleMaintenanceColdMisses_;
    idleMaintenanceState_ = "cold_miss_suspended_until_successful_user_request";
  } else idleMaintenanceState_ = "maintaining_immutable_resource_accessibility";
  idleMaintenanceScheduler_.maintenanceCompleted(Clock::now(), timing.wallSeconds);
  drain(); expire(); publishStatus();
}

void Worker::publishStatus() {
  governor_.setPressure(pressure_.value());
  const auto memory = backend_.memoryStats();
  const auto governor = governor_.snapshot();
  uint32_t active = 0, prefilling = 0, decoding = 0, masks = 0, mtpActive = 0;
  for (const auto &request : active_) if (request) {
    ++active;
    if (request->mtpState) ++mtpActive;
    if (request->promptOffset < request->frame.promptTokens.size() || request->mtpPriming) ++prefilling;
    else if (request->maskId && request->mask.empty()) ++masks;
    else ++decoding;
  }
  const uint64_t current = std::max(memory.allocatedBytes + memory.sparseResidentBytes,
                                    memory.deviceCurrentAllocatedBytes);
  const uint64_t peak = std::max({current, memory.peakResidentBytes, memory.devicePeakAllocatedBytes});
  const bool healthy = backend_.healthy();
  const bool ready = healthy && governor.pressure != engine::MemoryPressure::Critical &&
      current <= governor.limitBytes && !transport_.stopping();
  const auto persistedOperands = forward_.persistedOperandStatus();
  const auto *persistedExperts = forward_.batchInt8ExpertStore();
  uint64_t persistedExpertCount = 0;
  if (persistedExperts)
    for (uint32_t layer = 0; layer < 48; ++layer)
      persistedExpertCount += persistedExperts->selectedExpertIDs(layer).size();
  std::ostringstream out;
  const auto pleStorage = weights_.pleSSDStorageStats();
  const auto pleStore = weights_.pleSSDStore();
  const auto pleIO = pleStore ? pleStore->statistics() : FlashPLESSDStore::Statistics{};
  out << std::setprecision(12) << R"({"schema_version":5,"ready":)" << (ready ? "true" : "false")
      << R"(,"ple_storage":{"ssd_streaming_enabled":)" << (pleStorage.enabled ? "true" : "false")
      << R"(,"table_gpu_buffers_present":)" << (pleStorage.enabled ? "false" : "true")
      << R"(,"original_payload_bytes":)" << pleStorage.originalPayloadBytes
      << R"(,"gpu_mapped_original_bytes":)" << pleStorage.gpuMappedBytes
      << R"(,"disk_only_payload_bytes":)" << pleStorage.diskOnlyPayloadBytes
      << R"(,"disk_only_logical_bytes":)" << pleStorage.diskOnlyLogicalBytes
      << R"(,"disk_tensor_count":)" << pleStorage.diskTensorCount
      << R"(,"disk_projection_count":)" << pleStorage.diskProjectionCount
      << R"(,"native_window_count":)" << pleStorage.nativeWindowCount
      << R"(,"fully_disk_payload_count":)" << pleStorage.fullyDiskPayloadCount
      << R"(,"singleton_staging_bytes":)" << forward_.pleSSDStagingBytes()
      << R"(,"batch_decode_staging_bytes":)" << (batch_ ? batch_->pleSSDStagingBytes() : 0)
      << R"(,"joint_verify_staging_bytes":)" << (jointVerify_ ? jointVerify_->pleSSDStagingBytes() : 0)
      << R"(,"batch_prefill_staging_bytes":)" << (batchPrefill_ ? batchPrefill_->pleSSDStagingBytes() : 0)
      << R"(,"cache_budget_bytes":)" << pleIO.cacheBudgetBytes
      << R"(,"cache_accounted_bytes":)" << pleIO.cacheAccountedBytes
      << R"(,"cached_rows":)" << pleIO.cachedRows
      << R"(,"read_scratch_limit_bytes":)" << pleIO.readScratchLimitBytes
      << R"(,"file_cache_bypass_enabled":)" << (pleIO.fileCacheBypassEnabled ? "true" : "false")
      << R"(,"prepared_batches":)" << pleIO.preparedBatches
      << R"(,"requested_rows":)" << pleIO.requestedRows
      << R"(,"unique_miss_rows":)" << pleIO.uniqueMissRows
      << R"(,"cache_hit_rows":)" << pleIO.cacheHitRows
      << R"(,"duplicate_miss_rows":)" << pleIO.duplicateMissRows
      << R"(,"read_requests":)" << pleIO.readRequests
      << R"(,"requested_read_bytes":)" << pleIO.requestedReadBytes
      << R"(,"completed_read_bytes":)" << pleIO.completedReadBytes
      << R"(,"cache_evictions":)" << pleIO.cacheEvictions
      << R"(,"failed_batches":)" << pleIO.failedBatches
      << R"(,"source_validation_failures":)" << pleIO.sourceValidationFailures
      << R"(,"host_read_ms":)" << double(pleIO.hostReadNanoseconds) / 1e6
      << R"(,"poisoned":)" << (pleIO.poisoned ? "true" : "false")
      << R"(,"read_byte_scope":"pread payload bytes; not measured physical SSD traffic","cache_scope":"bounded host row cache; table is not GPU-mapped in SSD mode"})"
      << R"(,"maximum_context_tokens":)" << capacity_
      << R"(,"idle_residency_maintenance":{"requested":)" << (idleMaintenanceRequested_ ? "true" : "false")
      << R"(,"available":)" << (idleMaintenance_ ? "true" : "false")
      << R"(,"failure_reason":)" << json::quote(idleMaintenanceFailureReason_)
      << R"(,"runtime_default_enabled":false,"armed_after_successful_user_request":)" << (idleMaintenanceScheduler_.armed() ? "true" : "false")
      << R"(,"state":)" << json::quote(idleMaintenanceState_)
      << R"(,"interval_ms":)" << idleMaintenanceInterval_
      << R"(,"maintenance_command_in_flight":)" << (idleMaintenanceInFlight_ ? "true" : "false")
      << R"(,"immutable_owner_count":)" << (idleMaintenance_ ? idleMaintenance_->ownerCount() : 0)
      << R"(,"immutable_owner_bytes":)" << (idleMaintenance_ ? idleMaintenance_->ownerBytes() : 0)
      << R"(,"gpu_bytes_read_per_command":)" << (idleMaintenance_ ? idleMaintenance_->ownerCount() * 4 : 0)
      << R"(,"added_weight_backing_bytes":0,"added_diagnostic_allocation_bytes":)"
      << (idleMaintenance_ ? idleMaintenance_->allocatedBytes() : 0)
      << R"(,"completed_commands":)" << idleMaintenanceCommands_
      << R"(,"pressure_suspensions":)" << idleMaintenancePressureSuspensions_
      << R"(,"maintenance_failures":)" << idleMaintenanceFailures_
      << R"(,"busy_safe_point_skips":)" << idleMaintenanceBusySkips_
      << R"(,"cold_misses_wall_at_least_interval":)" << idleMaintenanceColdMisses_
      << R"(,"gpu_ms":)" << idleMaintenanceGPUSeconds_ * 1000
      << R"(,"wall_ms":)" << idleMaintenanceWallSeconds_ * 1000
      << R"(,"last_gpu_ms":)" << idleMaintenanceLastGPUSeconds_ * 1000
      << R"(,"last_wall_ms":)" << idleMaintenanceLastWallSeconds_ * 1000
      << R"(,"maximum_wall_ms":)" << idleMaintenanceMaximumWallSeconds_ * 1000
      << R"(,"last_seconds_since_user_gpu_command":)" << idleMaintenanceLastIdleSeconds_
      << R"(,"physical_pinning_guaranteed":false,"request_arrival_can_overlap_one_maintenance_command":true})"
      << R"(,"memory_pressure":)" << json::quote(pressureName(governor.pressure))
      << R"(,"capabilities":{"input_modalities":["text"],"output_modalities":["text"],)"
      << R"("reasoning":true,"tools":true,"structured_output":true,"mtp":)" << (head_ ? "true" : "false") << ','
      << R"("decode_batching":)" << (batch_ ? "true" : "false") << ','
      << R"("prefill_batching":)" << (batchPrefill_ ? "true" : "false") << ','
      << R"("batch_mtp_prefill":)" << (batchPrimeHead_ ? "true" : "false") << ','
      << R"("batch_mtp":)" << (jointHead_ ? "true" : "false") << ','
      << R"("prefix_cache":false,"native_route":"flash-next"})"
      << R"(,"identity":{"source":)" << json::quote(weights_.sourceIdentity())
      << R"(,"loaded_model_layout_sha256":)" << json::quote(weights_.manifestFingerprint())
      << R"(,"engine_instance_id":)" << instance_
      << R"(,"forward_semantics":)" << json::quote(kFlashForwardSemantics)
      << R"(,"kernel_routes":)" << json::quote(forward_.kernelRoutes())
      << R"(,"worker_semantics":)" << json::quote(head_ && singletonMTP_.explicitOverride
                                                  ? "native-worker5-singleton-fixed-depth1..15-fold8-jointcap3-v6" :
                                                  jointHead_ ? "native-worker5-joint-greedy-mtp3-and-singleton-policy-v4" :
                                                  adaptiveMTP_ && head_ ? "native-worker5-ar-batch-or-adaptive-greedy-mtp3-v3" :
                                                  batch_ ? "native-worker5-ar-batch-or-greedy-mtp3-v2" :
                                                  head_ ? kWorkerMTPSemantics : "native-worker5-autoregression-v1")
      << R"(,"mtp_semantics":)" << (head_ ? json::quote(kFlashMTPSemantics) : "null")
      << R"(,"mtp_teacher_priming_route":)" << (head_ ? json::quote(mtpTeacherCacheOnly_
          ? kFlashMTPTeacherCacheSemantics : "mtp-full-forward-none-logits-v1") : "null")
      << R"(,"mtp_attention_route":)" << (head_ ? json::quote(head_->attentionRouteSemantics()) : "null")
      << R"(,"batch_semantics":)" << (batch_ ? json::quote(kFlashBatchForwardExecution) : "null")
      << R"(,"joint_head_semantics":)" << (jointHead_ ? json::quote(kFlashBatchMTPSemantics) : "null")
      << R"(,"joint_head_attention_route":)" << (jointHead_ ? json::quote(jointHead_->attentionRouteSemantics()) : "null")
      << R"(,"joint_head_vocabulary_register":)"
      << (jointHead_ && jointHead_->vocabularyRegisterEnabled() ? "true" : "false")
      << R"(,"joint_head_vocabulary_route":)"
      << (jointHead_ ? json::quote(jointHead_->vocabularyRouteSemantics()) : "null")
      << R"(,"joint_verifier_semantics":)" << (jointVerify_ ? json::quote(kFlashBatchVerifySemantics) : "null")
      << R"(,"batch_prefill_semantics":)" << (batchPrefill_ ? json::quote(kFlashBatchPrefillSemantics) : "null")
      << R"(,"batch_prefill_kernel_routes":)" << (batchPrefill_ ? json::quote(batchPrefill_->kernelRoutes()) : "null")
      << R"(,"gpu_prefill_feature_copy":)" << (gpuPrefillCopy_ ? "true" : "false")
      << R"(,"qsa_output_f32_n32":)" << (flashQSAOutF32N32Enabled() ? "true" : "false")
      << R"(,"qsa_output_f32_n32_semantics":)"
      << (flashQSAOutF32N32Enabled() ? json::quote(kFlashQSAOutF32N32Semantics) : "null")
      << R"(,"batch_mtp_prefill_semantics":)" << (batchPrimeHead_ ? json::quote(kWorkerBatchHeadPrimeSemantics) : "null")
      << R"(,"batch_mtp_prefill_attention_route":)" << (batchPrimeHead_ ? json::quote(batchPrimeHead_->attentionRouteSemantics()) : "null")
      << R"(,"prefill_execution":)" << json::quote(batchPrefill_
          ? "native_metal_real_uniform_batch_prefill_or_singleton" : "native_metal_single_request_prefill")
      << R"(,"execution":)" << json::quote(jointHead_ ? "native_metal_ar_or_real_joint_greedy_mtp" :
                                           batch_ && head_ ? "native_metal_ar_batch_or_exact_greedy_mtp" :
                                           batch_ ? "native_metal_ar_batch" :
                                           head_ ? "native_metal_ar_or_exact_greedy_mtp" : "native_metal_autoregression")
      << R"(,"weight_format":"mlx_affine_preconverted"})"
      << R"(,"persisted_operands":{"schema":"splash-local-affine-operands-v1","store_manifest_sha256":)"
      << (persistedOperands.storeManifestSha256.empty() ? "null" : json::quote(persistedOperands.storeManifestSha256))
      << R"(,"bf16_tensors":)" << persistedOperands.bf16Tensors
      << R"(,"f32_tensors":)" << persistedOperands.f32Tensors
      << R"(,"bf16_mapped_payload_bytes":)" << persistedOperands.bf16PayloadBytes
      << R"(,"f32_mapped_payload_bytes":)" << persistedOperands.f32PayloadBytes
      << R"(,"mapping":"readonly_verified_shared_no_copy"})"
      << R"(,"saved_operands_residency":{"requested":)" << (savedResidency_.requested ? "true" : "false")
      << R"(,"supported":)" << (savedResidency_.supported ? "true" : "false")
      << R"(,"active":)" << (savedResidencyLease_ && healthy && !transport_.stopping() ? "true" : "false")
      << R"(,"request_succeeded":)" << (savedResidencyLease_ ? "true" : "false")
      << R"(,"requested_view_count":)" << savedResidency_.requestedViews
      << R"(,"requested_view_bytes":)" << savedResidency_.requestedViewBytes
      << R"(,"registered_base_allocation_count":)" << savedResidencyLease_.bufferCount()
      << R"(,"registered_base_allocation_bytes":)" << savedResidencyLease_.byteCount()
      << R"(,"failure_reason":)" << json::quote(savedResidency_.failureReason)
      << R"(,"scope":)" << json::quote(savedResidency_.originalAdded
          ? "verified derived operands plus checked pure-text original bases; mixed PLE/vision bases excluded"
          : "verified saved dense/F32 operands and selected INT8 payload/rank allocations only; original checkpoint and PLE excluded")
      << R"(,"physical_pinning_verified":false,"backing_already_charged":true})"
      << R"(,"original_text_residency":{"requested":)" << (savedResidency_.originalRequested ? "true" : "false")
      << R"(,"added_to_union":)" << (savedResidency_.originalAdded ? "true" : "false")
      << R"(,"active":)" << (savedResidency_.originalAdded && savedResidencyLease_ && healthy && !transport_.stopping() ? "true" : "false")
      << R"(,"eligible_base_count":)" << savedResidency_.originalSelection.buffers.size()
      << R"(,"eligible_mapped_bytes":)" << savedResidency_.originalSelection.mappedBytes
      << R"(,"excluded_base_count":)" << savedResidency_.originalSelection.excludedBaseCount
      << R"(,"excluded_mapped_bytes":)" << savedResidency_.originalSelection.excludedMappedBytes
      << R"(,"excluded_ple_base_count":)" << savedResidency_.originalSelection.excludedPLEBaseCount
      << R"(,"excluded_vision_base_count":)" << savedResidency_.originalSelection.excludedVisionBaseCount
      << R"(,"excluded_unknown_base_count":)" << savedResidency_.originalSelection.excludedUnknownBaseCount
      << R"(,"paths":[)";
  for (size_t index = 0; index < savedResidency_.originalSelection.paths.size(); ++index) {
    if (index) out << ',';
    out << json::quote(savedResidency_.originalSelection.paths[index]);
  }
  out << R"(],"registered_union_base_allocation_count":)" << savedResidencyLease_.bufferCount()
      << R"(,"registered_union_base_allocation_bytes":)" << savedResidencyLease_.byteCount()
      << R"(,"host_measurement_valid":)" << (savedResidency_.originalHostSnapshot.hostMeasurementValid ? "true" : "false")
      << R"(,"host_available_bytes":)" << savedResidency_.originalHostSnapshot.hostAvailableBytes
      << R"(,"host_reserve_bytes":)" << savedResidency_.originalHostSnapshot.hostReserveBytes
      << R"(,"host_headroom_bytes":)" << savedResidency_.originalHostSnapshot.hostHeadroomBytes
      << R"(,"host_required_headroom_bytes":)" << (savedResidency_.originalRequested
          ? savedResidency_.originalSelection.mappedBytes + (2ULL << 30) : 0)
      << R"(,"failure_reason":)" << json::quote(savedResidency_.originalFailureReason)
      << R"(,"physical_pinning_verified":false,"backing_already_charged":true})"
      << R"(,"persisted_experts":{"enabled":)" << (persistedExperts ? "true" : "false")
      << R"(,"store_manifest_sha256":)"
      << (persistedExperts ? json::quote(persistedExperts->identitySha256()) : "null")
      << R"(,"selection_plan_sha256":)"
      << (persistedExperts ? json::quote(persistedExperts->planSha256()) : "null")
      << R"(,"expert_count":)" << persistedExpertCount
      << R"(,"mapped_bytes":)" << (persistedExperts ? persistedExperts->mappedBytes() : 0)
      << R"(,"numerical_alternative":)" << (persistedExperts ? "true" : "false")
      << R"(,"scope":"large-row target prefill; original Q4 misses, decode and trained MTP"})"
      << R"(,"gdn_verification_storage":{"lazy_enabled":)"
      << (forward_.lazyGDNRollbackEnabled() ? "true" : "false")
      << R"(,"singleton_bytes":)" << forward_.verificationGDNStorageBytes()
      << R"(,"joint_bytes":)" << (jointVerify_ ? jointVerify_->verificationGDNStorageBytes() : 0)
      << R"(,"joint_lazy_enabled":)"
      << (jointVerify_ && jointVerify_->lazyGDNRollbackEnabled() ? "true" : "false")
      << R"(,"singleton_graph_counters":)";
  appendLazyGDNCounters(out, forward_.lazyGDNRollbackCounters());
  out << R"(,"joint_graph_counters":)";
  appendLazyGDNCounters(out, jointVerify_ ? jointVerify_->lazyGDNRollbackCounters()
                                        : FlashGDNLazyRollbackCounters{});
  out << '}' << R"(,"hc_up_route_counters":)";
  appendHCUpCounters(out, forward_.hcUpEncodedCounters());
  out << R"(,"qsa_output_f32_n32_route_counters":{"scope":"main model graph construction since startup; not completed GPU dispatches","encoded_calls":)"
      << forward_.qsaOutF32N32EncodedCalls()
      << R"(,"encoded_real_rows":)" << forward_.qsaOutF32N32EncodedRealRows() << '}';
  const auto bulkQSACounters = forward_.qsaBulkPrefillCounters();
  out << R"(,"qsa_bulk_prefill_route_counters":{"scope":"completed main model target commands since startup","completed_prefill_calls":)"
      << bulkQSACounters.completedPrefillCalls
      << R"(,"completed_prefill_tokens":)" << bulkQSACounters.completedPrefillTokens
      << R"(,"completed_layer_calls":)" << bulkQSACounters.completedLayerCalls
      << R"(,"completed_sg8_layer_calls":)" << bulkQSACounters.completedSG8LayerCalls << '}';
  const auto requestTrace = requestCommandTraceInfo();
  out << R"(,"request_command_trace":{"enabled":)" << (requestTrace.enabled ? "true" : "false")
      << R"(,"records":)" << requestTrace.records
      << R"(,"missing_profiles":)" << requestTrace.missingProfiles
      << R"(,"unexpected_profiles":)" << requestTrace.unexpectedProfiles
      << R"(,"scope":"opt-in request-labelled command timestamps; diagnostic metadata overhead; encoder boundaries preserved"})"
      << R"(,"memory_actual":{"dense_bytes":)" << memory.allocatedBytes
      << R"(,"current_bytes":)" << current << R"(,"peak_bytes":)" << peak
      << R"(,"sparse_virtual_bytes":)" << memory.sparseVirtualBytes
      << R"(,"sparse_resident_bytes":)" << memory.sparseResidentBytes << '}'
      << R"(,"memory_governor":{"limit_bytes":)" << governor.limitBytes
      << R"(,"headroom_bytes":)" << governor.headroomBytes
      << R"(,"host_available_bytes":)" << governor.hostAvailableBytes
      << R"(,"host_reserve_bytes":)" << governor.hostReserveBytes
      << R"(,"host_measurement_valid":)" << (governor.hostMeasurementValid ? "true" : "false")
      << R"(,"system_pressure":)" << json::quote(pressureName(governor.systemPressure))
      << R"(,"growth_allowed":)" << (governor.growthAllowed ? "true" : "false")
      << R"(,"denied_reservations":)" << governor.deniedReservations << '}'
      << R"(,"memory_audit":{"valid":)" << (current <= governor.limitBytes ? "true" : "false")
      << R"(,"scope":"native_allocation_ledger","weight_bytes":)" << weights_.actualAllocatedBytes()
      << R"(,"workspace_bytes":)" << forward_.workspaceBytes()
      << R"(,"state_bytes_per_request":)" << FlashForward::requestStateBytes(capacity_)
      << R"(,"mtp_workspace_bytes":)" << (head_ ? head_->workspaceBytes() : 0)
      << R"(,"mtp_extra_state_bytes_per_eligible_request":)"
      << (head_ ? FlashMTPForward::requestStateBytes(capacity_) + singletonMTP_.hiddenBytes() : 0)
      << R"(,"batch_workspace_bytes":)" << (batch_ ? batch_->workspaceBytes() : 0)
      << R"(,"joint_verifier_workspace_bytes":)" << (jointVerify_ ? jointVerify_->workspaceBytes() : 0)
      << R"(,"joint_head_workspace_bytes":)" << (jointHead_ ? jointHead_->workspaceBytes() : 0)
      << R"(,"joint_owned_hidden_input_bytes":)" << jointHidden_.sizeBytes()
      << R"(,"batch_prefill_workspace_bytes":)" << (batchPrefill_ ? batchPrefill_->workspaceBytes() : 0)
      << R"(,"batch_prefill_owned_hidden_input_bytes":)" << batchPrefillHidden_.sizeBytes()
      << R"(,"batch_mtp_prefill_workspace_bytes":)" << (batchPrimeHead_ ? batchPrimeHead_->workspaceBytes() : 0)
      << R"(,"batch_mtp_prefill_owned_hidden_input_bytes":)" << batchPrimeInput_.sizeBytes() << '}'
      << R"(,"cache":{"enabled":false,"hits":0,"cold_misses":)" << submitted_
      << R"(,"reused_tokens":0})"
      << R"(,"admission":{"waiting":)" << pending_.size()
      << R"(,"waiting_concurrency":)" << (active == kConcurrent ? pending_.size() : 0)
      << R"(,"waiting_memory":)" << (active < kConcurrent ? pending_.size() : 0)
      << R"(,"pressure_retry_interval_ms":250,"maximum_no_active_pressure_wait_ms":2000)"
      << R"(,"denied_state_reservations":)" << admissionDenied_
      << R"(,"allocation_queue_rechecks":)" << allocationQueueRechecks_
      << R"(,"admitted_after_allocation_queue_recheck":)" << recheckAdmissions_
      << R"(,"allocation_queue_recheck_scope":"already-queued controls after new state allocation; at most four per owner loop; no waiting")"
      << R"(,"last_denial":{"failure":)" << json::quote(metal::allocationFailureName(lastAdmissionFailure_))
      << R"(,"requested_bytes":)" << lastAdmissionBytes_
      << R"(,"steady_seconds":)" << (lastAdmissionAt_ == Time{} ? 0 :
          std::chrono::duration<double>(lastAdmissionAt_.time_since_epoch()).count())
      << R"(,"snapshot_scope":"immediately after denied reservation")"
      << R"(,"system_pressure":)" << json::quote(pressureName(lastAdmissionMemory_.systemPressure))
      << R"(,"effective_pressure":)" << json::quote(pressureName(lastAdmissionMemory_.pressure))
      << R"(,"host_available_bytes":)" << lastAdmissionMemory_.hostAvailableBytes
      << R"(,"host_headroom_bytes":)" << lastAdmissionMemory_.hostHeadroomBytes
      << R"(,"engine_headroom_bytes":)" << lastAdmissionMemory_.headroomBytes << "}}"
      << R"(,"scheduler":{"mode":)"
      << json::quote(batchPrefill_ ? "real_uniform_batch_prefill_before_cooperative_decode" :
                         jointHead_ ? "cooperative_prefill_joint_greedy_mtp_and_ineligible_ar" :
                         batch_ && head_ ? "cooperative_prefill_real_decode_batches_and_singleton_mtp" :
                         batch_ ? "cooperative_prefill_and_real_decode_batches" :
                               "cooperative_single_request_commands")
      << R"(,"queued":)" << pending_.size()
      << R"(,"maximum_prefill_rows":)" << prefillRows_
      << R"(,"maximum_batch_prefill_rows_per_lane":)" << (batchPrefill_ ? batchPrefillRows_ : 0)
      << R"(,"maximum_mtp_priming_rows":)" << (head_ ? kHeadRows : 0)
      << R"(,"maximum_batch_mtp_priming_lanes":)" << (batchPrimeHead_ ? kConcurrent : 0)
      << R"(,"maximum_batch_mtp_priming_rows_per_lane":)" << (batchPrimeHead_ ? kHeadRows : 0)
      << R"(,"batch_mtp_priming_batches":)" << mtpBatchPrime_.batches
      << R"(,"batch_mtp_priming_real_rows":)" << mtpBatchPrime_.rows
      << R"(,"batch_mtp_priming_actual_maximum_lanes":)" << batchHeadPrimeMaximumWidth_
      << R"(,"batch_mtp_priming_scope":"one command; real adjacent hidden/next-token pairs summed over live lanes; None logits")"
      << R"(,"batch_mtp_priming_owned_feature_copied_bytes":)" << batchHeadPrimeCopiedBytes_
      << R"(,"batch_mtp_priming_batches_by_width":{"b1":)" << batchHeadPrimeWidths_[0]
      << R"(,"b2":)" << batchHeadPrimeWidths_[1] << R"(,"b3":)" << batchHeadPrimeWidths_[2]
      << R"(,"b4":)" << batchHeadPrimeWidths_[3] << '}'
      << R"(,"active_requests":)" << active << R"(,"prefilling":)" << prefilling
      << R"(,"mtp_active_requests":)" << mtpActive
      << R"(,"decoding":)" << decoding << R"(,"waiting_mask":)" << masks
      << R"(,"command_in_flight":)" << (inFlight_ ? "true" : "false")
      << R"(,"prefill_batches":)" << prefill_.batches << R"(,"prefill_rows":)" << prefill_.rows
      << R"(,"prefill_batches_scope":"one main-trunk command; actual consumed prompt tokens summed over real lanes")"
      << R"(,"prefill_batches_by_width":{"b1":)" << prefillWidths_[0]
      << R"(,"b2":)" << prefillWidths_[1] << R"(,"b3":)" << prefillWidths_[2]
      << R"(,"b4":)" << prefillWidths_[3] << '}'
      << R"(,"decode_batches":)" << decode_.batches
      << R"(,"decode_batches_scope":"actual AR command or joint target verifier width; B1 singleton MTP cycle")"
      << R"(,"decode_batches_by_width":{"b1":)" << decodeWidths_[0]
      << R"(,"b2":)" << decodeWidths_[1] << R"(,"b3":)" << decodeWidths_[2]
      << R"(,"b4":)" << decodeWidths_[3] << "}}"
      << R"(,"requests":{"submitted":)" << submitted_ << R"(,"completed":)" << completed_
      << R"(,"cancelled":)" << cancelled_ << R"(,"failed":)" << failed_ << '}'
      << R"(,"model_timing":{"scope":"worker_lifetime_command","prefill":)";
  appendTiming(out, prefill_);
  out << R"(,"decode":)"; appendTiming(out, decode_);
  out << R"(},"metrics":{"ttft_ms":{"p50":)" << ttft_.p50()
      << R"(,"p95":)" << ttft_.p95() << R"(,"samples":)" << ttft_.size()
      << R"(},"itl_ms":{"p50":)" << itl_.p50()
      << R"(,"p95":)" << itl_.p95() << R"(,"samples":)" << itl_.size()
      << R"(},"prefill_input_tokens":)" << prefill_.rows
      << R"(,"prefill_wall_ms":)" << prefill_.host * 1000
      << R"(,"prefill_tokens_per_second":)" << (prefill_.host > 0 ? prefill_.rows / prefill_.host : 0)
      << R"(,"decode_output_tokens":)" << decode_.rows
      << R"(,"decode_output_tokens_scope":"prepared target predictions; actual streamed count is autoregressive_output_tokens")"
      << R"(,"decode_wall_ms":)" << decode_.host * 1000
      << R"(,"decode_tokens_per_second":)" << (decode_.host > 0 ? decode_.rows / decode_.host : 0)
      << R"(,"autoregressive_output_tokens":)" << emitted_
      << R"(,"drafted_tokens":)" << drafted_ << R"(,"accepted_draft_tokens":)" << accepted_
      << R"(,"accepted_draft_tokens_scope":"committed verifier input drafts","metal_failures":0})"
      << R"(,"mtp":{"enabled":)" << (head_ ? "true" : "false")
      << R"(,"teacher_cache_only_requested":)" << (mtpTeacherCacheOnly_ ? "true" : "false")
      << R"(,"teacher_cache_only_priming_calls":)" << teacherCachePrimeCalls_
      << R"(,"teacher_cache_only_scope":"sequential prompt priming; true grouped head priming retains full None forward")"
      << R"(,"maximum_draft_tokens":)" << (head_ ? std::max(singletonMTP_.maximumDepth, jointHead_ ? kMTPDepth : 0) : 0)
      << R"(,"singleton_maximum_draft_tokens":)" << (head_ ? singletonMTP_.maximumDepth : 0)
      << R"(,"singleton_depth_override":)" << (singletonMTP_.explicitOverride ? std::to_string(singletonMTP_.maximumDepth) : "null")
      << R"(,"singleton_concurrent_draft_cap":)" << (head_ ? std::min(singletonMTP_.maximumDepth, kMTPDepth) : 0)
      << R"(,"singleton_committed_fold_maximum_rows":)" << kFlashMTPMaximumCommittedFoldRows
      << R"(,"singleton_committed_fold_policy":"ordered chunks at most 8 preserve raw decode coefficients; priming unchanged")"
      << R"(,"singleton_committed_fold_calls":)" << mtpCommittedHeadCalls_
      << R"(,"joint_maximum_draft_tokens":)" << (jointHead_ ? kMTPDepth : 0)
      << R"(,"policy":)" << json::quote(adaptiveMTP_
          ? jointHead_ ? "singleton adaptive depth0..3; joint fixedcap3; greedy unmasked; AR for ineligible requests"
                       : "adaptive depth0..3 by conditional acceptance and measured cycle cost; greedy unmasked; batch fallback AR"
          : "singleton fixed cap" + std::to_string(singletonMTP_.maximumDepth) +
              " bounded by output budget; concurrent ready MTP peers or pending admission cap singleton drafts at 3; " +
              (jointHead_ ? "joint fixedcap3; greedy unmasked; AR for ineligible requests"
                          : "greedy unmasked; requests joining real batches permanently use AR"))
      << R"(,"joint_policy":)" << (jointHead_ ? json::quote("fixed cap3 shared-budget/EOS bound; true joint head and target; survivors retain independent prefixes") : "null")
      << R"(,"depth_controller_semantics":)" << (adaptiveMTP_ ? json::quote(kFlashMTPDepthSemantics) : "null")
      << R"(,"completed_cycles_by_proposed_depth":[)";
  for (uint32_t depth = 0; depth < mtpDepthCycles_.size(); ++depth) {
    if (depth) out << ',';
    out << mtpDepthCycles_[depth];
  }
  out << ']'
      << R"(,"completed_cycles_by_proposed_depth_scope":"singleton adaptive/fixed policy only; joint cycles use separate counters")"
      << R"(,"eligible_requests":)" << mtpRequests_ << R"(,"autoregressive_requests":)" << arRequests_
      << R"(,"verification_cycles":)" << mtpCycles_ << R"(,"drafted_tokens":)" << drafted_
      << R"(,"accepted_committed_drafts":)" << accepted_ << R"(,"matched_proposals":)" << matched_
      << R"(,"emitted_accepted_proposals":)" << emittedAccepted_
      << R"(,"permanent_batch_fallback_requests":)" << mtpBatchFallbacks_
      << R"(,"accepted_prefix_histogram":[)";
  for (uint32_t prefix = 0; prefix < acceptedPrefixes_.size(); ++prefix) {
    if (prefix) out << ',';
    out << acceptedPrefixes_[prefix];
  }
  out << R"(],"head_priming":)";
  appendTiming(out, mtpPrime_);
  out << R"(,"batched_head_priming_subset":)"; appendTiming(out, mtpBatchPrime_);
  out << R"(,"head_decode":)"; appendTiming(out, mtpHead_);
  out << R"(,"target_verify":)"; appendTiming(out, mtpVerify_);
  out << R"(,"prefix_restore":)"; appendTiming(out, mtpCommit_);
  out << R"(,"joint_cohorts_attempted":)" << jointMTPCohorts_
      << R"(,"joint_partial_cohorts_before_target":)" << jointMTPPartial_
      << R"(,"joint_members_dropped_before_commit":)" << jointMTPDropped_
      << R"(,"joint_member_drop_scope":"cancellation, deadline, request error or generation loss before target commit; later emission drops are request terminals")"
      << R"(,"joint_borrowed_hidden_copied_bytes":)" << jointMTPHiddenBytes_
      << R"(,"joint_head_vocabulary_register_commands":)"
      << (jointHead_ ? jointHead_->vocabularyRegisterCommands() : 0)
      << R"(,"joint_head_vocabulary_register_rows":)"
      << (jointHead_ ? jointHead_->vocabularyRegisterRows() : 0)
      << R"(,"joint_head_vocabulary_register_counter_scope":"successfully completed joint head commands; real vocabulary rows")"
      << R"(,"joint_head_commands_by_width":[)" << jointHeadWidths_[0] << ',' << jointHeadWidths_[1]
      << ',' << jointHeadWidths_[2] << ',' << jointHeadWidths_[3] << ']'
      << R"(,"joint_target_verifiers_by_width":[)" << jointVerifyWidths_[0] << ',' << jointVerifyWidths_[1]
      << ',' << jointVerifyWidths_[2] << ',' << jointVerifyWidths_[3] << ']';
  out << R"(},"batch_decode":{"enabled":)" << (batch_ ? "true" : "false")
      << R"(,"maximum_lanes":)" << (batch_ ? kConcurrent : 0)
      << R"(,"borrowed_logits_copied_bytes":)" << batchCopiedLogitBytes_
      << R"(,"duration_scope":"one command duration per actual B2/B3/B4 group")"
      << R"(},"batch_prefill":{"enabled":)" << (batchPrefill_ ? "true" : "false")
      << R"(,"maximum_lanes":)" << (batchPrefill_ ? kConcurrent : 0)
      << R"(,"maximum_real_rows_per_lane":)" << (batchPrefill_ ? batchPrefillRows_ : 0)
      << R"(,"borrowed_logits_copied_bytes":)" << batchPrefillLogitsCopiedBytes_
      << R"(,"true_target_hidden_copied_bytes":)" << batchPrefillHiddenCopiedBytes_
      << R"(,"cpu_feature_copy_bytes":)" << batchPrefillHiddenCPUCopyBytes_
      << R"(,"gpu_owned_feature_copy_bytes":)" << batchPrefillHiddenGPUCopyBytes_
      << R"(,"gpu_feature_copy_enabled":)" << (gpuPrefillCopy_ ? "true" : "false")
      << R"(,"members_dropped_at_control_boundaries":)" << batchPrefillDropped_
      << R"(,"priming":)" << json::quote(batchPrimeHead_
          ? "ordered real adjacent pairs; equal128 or compatible tiny tails grouped over live lanes; other tails sequential"
          : "ordered per-lane head calls of at most 128 real adjacent pairs")
      << R"(,"timing_scope":"one main-trunk duration per cohort plus each actual head prime command once")"
      << R"(},"warmup":{"performed":false,"detail":"pipelines compile lazily"})"
      << R"(,"status_snapshot":{"scope":"last_native_safe_point","steady_seconds":)"
      << std::chrono::duration<double>(Clock::now().time_since_epoch()).count() << '}'
      << R"(,"metal":{"healthy":)" << (healthy ? "true" : "false")
      << R"(,"failure_reason":)" << json::quote(healthy ? std::string{} : backend_.unhealthyReason()) << "}}";
  transport_.cacheStatus(out.str());
}
} // namespace

int runFlashWorker(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string_view(argv[1]) == "--cpu-self-test") {
        cpuSelfTest();
        return 0;
      }
      if (argc != 5 || std::string_view(argv[1]) != "serve-flash-native") {
        std::cerr << "usage: splash-flash serve-flash-native DERIVED_DIRECTORY MAX_CONTEXT|auto MAX_MEMORY_BYTES|auto\n";
        return 2;
      }
      std::signal(SIGPIPE, SIG_IGN);
      std::signal(SIGINT, stopFromSignal); std::signal(SIGTERM, stopFromSignal); std::signal(SIGHUP, stopFromSignal);
      const auto directory = std::filesystem::canonical(argv[2]);
      // Fail a requested diagnostic sink before loading/model mapping. No
      // environment setting creates no file and leaves backend profiling off.
      auto requestCommandTrace = FlashRequestCommandTrace::fromEnvironment();
      // Experimental runtime-only switch stays off until root qualification.
      const bool mtpEnabled = environmentSwitch("SPLASH_FLASH_MTP");
      const bool teacherCacheOnlyEnabled = environmentSwitch("SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY");
      if (teacherCacheOnlyEnabled && !mtpEnabled)
        throw std::invalid_argument("SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY=1 requires SPLASH_FLASH_MTP=1");
      const bool prefillDenseTilesEnabled = flashPrefillDenseTilesEnabled();
      if (prefillDenseTilesEnabled && !environmentSwitch("SPLASH_FLASH_DENSE_CACHE"))
        throw std::invalid_argument("SPLASH_FLASH_PREFILL_DENSE_TILES=1 requires SPLASH_FLASH_DENSE_CACHE=1");
      const bool batchEnabled = environmentSwitch("SPLASH_FLASH_BATCH");
      const bool batchPrefillEnabled = environmentSwitch("SPLASH_FLASH_BATCH_PREFILL");
      (void)flashGDNBatchILPEnabled();
      (void)flashGDNLazyRollbackEnabled();
      (void)flashQSAOutF32N32Enabled();
      (void)qsaBulkPrefillSG8Enabled(qsaBulkPrefillEnabled());
      (void)flashBF16Q8HeadEnabled();
      const bool gpuPrefillCopyEnabled = environmentSwitch("SPLASH_FLASH_GPU_PREFILL_COPY");
      const bool batchMTPEnabled = environmentSwitch("SPLASH_FLASH_BATCH_MTP");
      const bool batchMTPPrefillEnabled = environmentSwitch("SPLASH_FLASH_BATCH_MTP_PREFILL");
      const bool savedResidencyRequested = environmentSwitch("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT");
      const bool originalTextResidencyRequested = environmentSwitch("SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT");
      const bool idleMaintenanceRequested = idle_maintenance::parseSwitch(
          std::getenv("SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE"));
      const uint32_t idleMaintenanceInterval = idle_maintenance::parseInterval(
          std::getenv("SPLASH_FLASH_IDLE_RESIDENCY_INTERVAL_MS"));
      const auto singletonMTP = environmentSingletonMTPPolicy();
      if (batchMTPEnabled && !mtpEnabled)
        throw std::invalid_argument("SPLASH_FLASH_BATCH_MTP=1 requires SPLASH_FLASH_MTP=1");
      validateBatchHeadPrimeFlags(batchMTPPrefillEnabled, mtpEnabled, batchPrefillEnabled);
      validateBatchPrefillGPUCopyFlags(gpuPrefillCopyEnabled, mtpEnabled, batchPrefillEnabled);
      const uint32_t prefillRows = environmentPrefillRows();
      const uint32_t batchPrefillRows = environmentBatchPrefillRows();
      const auto descriptor = FlashDescriptor::fromConfig(directory / "config.json");
      const uint32_t capacity = std::string_view(argv[3]) == "auto"
          ? std::min<uint32_t>(8192, descriptor.maximumContextTokens) : positive<uint32_t>(argv[3], "MAX_CONTEXT");
      if (capacity > descriptor.maximumContextTokens)
        throw std::invalid_argument("native context exceeds model architecture limit");
      const uint64_t physical = physicalMemory();
      const uint64_t reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
      const uint64_t budget = std::string_view(argv[4]) == "auto"
          ? (physical > reserve ? physical - reserve : 0) : positive<uint64_t>(argv[4], "MAX_MEMORY_BYTES");
      if (!budget || budget > physical) throw std::invalid_argument("native memory budget must fit physical RAM");
      const auto limits = serveLimits(capacity, descriptor.vocabularySize, mtpEnabled, singletonMTP.maximumDepth);
      Transport transport(limits);
      PressureMonitor pressure;
      metal::MetalBackend backend((executablePath().parent_path() / "splash.metallib").string());
      backend.setCancellationProbe([&transport] { return transport.stopping(); });
      auto weights = FlashWeights::load(backend, directory);
      engine::MemoryGovernor governor(backend, budget, reserve);
      uint64_t plannedTrunk = FlashForward::workspacePlannedBytes(
          capacity, prefillRows, mtpEnabled ? singletonMTP.maximumDepth + 1 : 0);
      plannedTrunk += FlashForward::expertCachePlannedBytes(weights);
      plannedTrunk += FlashForward::floatDenseCachePlannedBytes(weights);
      plannedTrunk += FlashForward::int8HeadPlannedBytes(weights);
      if (const char *online = std::getenv("SPLASH_FLASH_QSA_F32");
          online && std::string_view(online) == "1")
        plannedTrunk += uint64_t{std::min(prefillRows, uint32_t{128})} * 24 * 4 * (256 + 2) * 4 + (32ULL << 10);
      if (const char *cached = std::getenv("SPLASH_FLASH_DENSE_CACHE");
          cached && std::string_view(cached) == "1") {
        const auto prefixes = FlashDenseCache::defaultPrefixes(weights, true);
        plannedTrunk += FlashDenseCache::plannedBytes(weights, prefixes);
      }
      if (const char *blocked = std::getenv("SPLASH_FLASH_BLOCKED_MOE");
          blocked && std::string_view(blocked) == "1" && prefillRows >= 256)
        plannedTrunk += flashMoEBlockedWorkspacePlannedBytes(prefillRows, 10);
      metal::AllocationFailure trunkFailure = metal::AllocationFailure::None;
      auto trunkReservation = governor.tryReserve(plannedTrunk, &trunkFailure);
      if (!trunkReservation) throw metal::MetalAllocationError(
          "Flash trunk workspace cannot reserve physical memory", trunkFailure);
      FlashForward forward(backend, weights, capacity, prefillRows, mtpEnabled ? singletonMTP.maximumDepth + 1 : 0);
      if (forward.workspaceBytes() > plannedTrunk)
        throw std::runtime_error("Flash trunk workspace exceeds its reserved planned allocation");
      trunkReservation->commit();
      std::optional<FlashMTPForward> head;
      if (mtpEnabled) {
        uint64_t plannedHead = FlashMTPForward::workspacePlannedBytes(capacity, kHeadRows);
        if (const char *cached = std::getenv("SPLASH_FLASH_DENSE_CACHE");
            cached && std::string_view(cached) == "1")
          plannedHead += FlashMTPForward::denseCachePlannedBytes(weights);
        metal::AllocationFailure headFailure = metal::AllocationFailure::None;
        auto headReservation = governor.tryReserve(plannedHead, &headFailure);
        if (!headReservation) throw metal::MetalAllocationError(
            "Flash head workspace cannot reserve physical memory", headFailure);
        head.emplace(backend, weights, capacity, kHeadRows);
        if (head->workspaceBytes() > plannedHead)
          throw std::runtime_error("Flash head workspace exceeds its reserved planned allocation");
        headReservation->commit();
      }
      std::optional<FlashBatchForward> batch;
      if (batchEnabled) {
        const uint64_t planned = FlashBatchForward::workspacePlannedBytes(capacity, kConcurrent);
        metal::AllocationFailure failure = metal::AllocationFailure::None;
        auto reservation = governor.tryReserve(planned, &failure);
        if (!reservation) throw metal::MetalAllocationError(
            "Flash batch workspace cannot reserve physical memory", failure);
        batch.emplace(backend, weights, forward, capacity, kConcurrent);
        if (batch->workspaceBytes() > planned)
          throw std::runtime_error("Flash batch workspace exceeds its reserved planned allocation");
        reservation->commit();
      }
      std::optional<FlashBatchPrefill> batchPrefill;
      metal::MetalBuffer batchPrefillHidden;
      if (batchPrefillEnabled) {
        const uint64_t planned = FlashBatchPrefill::workspacePlannedBytes(capacity, kConcurrent, batchPrefillRows);
        const uint64_t hiddenBytes = head ? batchPrefillHiddenBytes(batchPrefillRows) : 0;
        metal::AllocationFailure failure = metal::AllocationFailure::None;
        auto reservation = governor.tryReserve(planned + hiddenBytes, &failure);
        if (!reservation) throw metal::MetalAllocationError(
            "Flash batch prefill workspace and true target staging cannot reserve physical memory", failure);
        const uint64_t before = backend.memoryStats().allocatedBytes;
        batchPrefill.emplace(backend, weights, forward, capacity, kConcurrent, batchPrefillRows);
        if (hiddenBytes)
          batchPrefillHidden = backend.allocateBuffer(hiddenBytes, metal::BufferStorage::Shared,
              "flash-worker-batch-prefill-owned-target-features");
        if (batchPrefill->workspaceBytes() > planned ||
            (hiddenBytes && batchPrefillHidden.sizeBytes() < hiddenBytes) ||
            metal::allocationDelta(before, backend.memoryStats().allocatedBytes) > planned + hiddenBytes)
          throw std::runtime_error("Flash batch prefill exceeds its reserved planned allocation");
        reservation->commit();
      }
      std::optional<FlashBatchVerify> jointVerify;
      std::optional<FlashBatchMTPForward> batchPrimeHead;
      metal::MetalBuffer batchPrimeInput;
      if (batchMTPPrefillEnabled) {
        const uint64_t planned = FlashBatchMTPForward::workspacePlannedBytes(
            capacity, kConcurrent, kHeadRows, false);
        metal::AllocationFailure failure = metal::AllocationFailure::None;
        auto reservation = governor.tryReserve(planned + kBatchHeadPrimeInputBytes, &failure);
        if (!reservation) throw metal::MetalAllocationError(
            "Flash batch head priming workspace and owned target staging cannot reserve physical memory", failure);
        const uint64_t before = backend.memoryStats().allocatedBytes;
        // This independent arena shares immutable trained coefficients with
        // its sequential owner. Priming never allocates a vocabulary cache or
        // computes a vocabulary row; every invocation requests None logits.
        batchPrimeHead.emplace(*head, kConcurrent, kHeadRows, nullptr);
        batchPrimeInput = backend.allocateBuffer(kBatchHeadPrimeInputBytes, metal::BufferStorage::Shared,
            "flash-worker-batch-head-priming-owned-input");
        if (batchPrimeHead->workspaceBytes() > planned ||
            batchPrimeInput.sizeBytes() < kBatchHeadPrimeInputBytes ||
            metal::allocationDelta(before, backend.memoryStats().allocatedBytes) > planned + kBatchHeadPrimeInputBytes)
          throw std::runtime_error("Flash batch head priming exceeds its reserved planned allocation");
        reservation->commit();
      }
      std::optional<FlashBatchMTPForward> jointHead;
      metal::MetalBuffer jointHidden;
      if (batchMTPEnabled) {
        const uint64_t plannedVerify = FlashBatchVerify::workspacePlannedBytes(capacity, kConcurrent, kMTPDepth + 1);
        metal::AllocationFailure verifyFailure = metal::AllocationFailure::None;
        auto verifyReservation = governor.tryReserve(plannedVerify, &verifyFailure);
        if (!verifyReservation) throw metal::MetalAllocationError(
            "Flash joint verifier workspace cannot reserve physical memory", verifyFailure);
        // Opt into the source's declared all-small-row BF coefficient policy
        // together; default raw routes preserve the qualified original policy.
        const bool sharedRoutes = environmentSwitch("SPLASH_FLASH_DENSE_SMALL_ROWS") ||
            environmentSwitch("SPLASH_FLASH_FLOAT_DENSE_CACHE");
        jointVerify.emplace(backend, weights, forward, capacity, kConcurrent, kMTPDepth + 1, sharedRoutes);
        if (jointVerify->workspaceBytes() > plannedVerify)
          throw std::runtime_error("Flash joint verifier exceeds its reserved planned allocation");
        verifyReservation->commit();
        const auto *vocabulary = forward.cachedVocabulary();
        const uint64_t plannedHead = FlashBatchMTPForward::workspacePlannedBytes(
            capacity, kConcurrent, kMTPDepth + 1, vocabulary != nullptr);
        metal::AllocationFailure jointFailure = metal::AllocationFailure::None;
        auto jointReservation = governor.tryReserve(plannedHead + kJointHiddenBytes, &jointFailure);
        if (!jointReservation) throw metal::MetalAllocationError(
            "Flash joint head and hidden staging cannot reserve physical memory", jointFailure);
        jointHead.emplace(*head, kConcurrent, kMTPDepth + 1, vocabulary);
        jointHidden = backend.allocateBuffer(kJointHiddenBytes, metal::BufferStorage::Shared,
                                              "flash-worker-joint-head-owned-input");
        if (jointHead->workspaceBytes() > plannedHead || jointHidden.sizeBytes() < kJointHiddenBytes)
          throw std::runtime_error("Flash joint head exceeds its reserved planned allocation");
        jointReservation->commit();
      }
      if (backend.memoryStats().allocatedBytes > budget) throw std::runtime_error("Flash workspace exceeds native memory budget");
      SavedOperandsResidencyStatus savedResidency;
      savedResidency.requested = savedResidencyRequested || originalTextResidencyRequested;
      savedResidency.originalRequested = originalTextResidencyRequested;
      savedResidency.supported = backend.capabilities().appleGpuFamily >= 6;
      metal::ResidencyLease savedResidencyLease;
      if (savedResidency.requested) {
        std::vector<metal::MetalBuffer> operands;
        if (savedResidencyRequested) {
          operands = forward.cachedOperandsOnly();
          if (head) {
            auto headOperands = head->cachedOperandsOnly();
            operands.insert(operands.end(), headOperands.begin(), headOperands.end());
          }
        }
        if (originalTextResidencyRequested) {
          savedResidency.originalSelection = weights.checkedOriginalTextResidency();
          governor.setPressure(pressure.value());
          savedResidency.originalHostSnapshot = governor.snapshot();
          const auto &host = savedResidency.originalHostSnapshot;
          if (flashOriginalResidencyHostAllowed(host.hostMeasurementValid, host.growthAllowed,
              host.pressure == engine::MemoryPressure::Normal && host.systemPressure == engine::MemoryPressure::Normal,
              host.hostHeadroomBytes, savedResidency.originalSelection.mappedBytes)) {
            operands.insert(operands.end(), savedResidency.originalSelection.buffers.begin(),
                savedResidency.originalSelection.buffers.end());
            savedResidency.originalAdded = true;
          } else {
            savedResidency.originalFailureReason = "host reserve protected; original residency requires mapped bytes plus 2 GiB headroom";
          }
        }
        for (const auto &buffer : operands) {
          if (!buffer || buffer.storage() != metal::BufferStorage::Shared ||
              !buffer.contents() || !buffer.sizeBytes())
            throw std::logic_error("saved operand residency selection includes an invalid mapping");
          if (buffer.sizeBytes() > UINT64_MAX - savedResidency.requestedViewBytes)
            throw std::overflow_error("saved operand residency selected byte count overflows");
          savedResidency.requestedViewBytes += buffer.sizeBytes();
        }
        savedResidency.requestedViews = operands.size();
        if (!savedResidency.supported) {
          savedResidency.failureReason = "device does not support residency sets";
        } else if (operands.empty()) {
          savedResidency.failureReason = "no verified immutable weight mappings are selected";
        } else {
          // Final union, after every constructor command has completed and
          // before requests begin. The registration retains mapped owners and
          // charges no already-accounted operand backing a second time.
          try {
            savedResidencyLease = backend.requestWeightResidency(operands,
                savedResidency.originalAdded ? "Splash verified derived and pure-text original operands"
                                              : "Splash verified saved derived operands");
          } catch (const std::exception &error) {
            savedResidency.failureReason = error.what();
          }
        }
        if (savedResidency.originalAdded && !savedResidencyLease)
          savedResidency.originalFailureReason = savedResidency.failureReason;
      }
      std::unique_ptr<idle_maintenance::Maintenance> idleMaintenance;
      std::string idleMaintenanceFailureReason;
      if (idleMaintenanceRequested) {
        const auto maintenanceStorage = weights.pleSSDStreamingEnabled()
            ? idle_maintenance::StorageMode::PLESSD : idle_maintenance::StorageMode::Raw;
        auto original = weights.immutableWeightBuffers();
        uint64_t originalBytes = 0;
        for (const auto &buffer : original) {
          if (buffer.sizeBytes() > UINT64_MAX - originalBytes) throw std::overflow_error("idle residency original bytes overflow");
          originalBytes += buffer.sizeBytes();
        }
        const uint64_t originalCount = original.size();
        auto derived = forward.cachedOperandsOnly();
        if (head) {
          auto headDerived = head->cachedOperandsOnly();
          derived.insert(derived.end(), headDerived.begin(), headDerived.end());
        }
        original.insert(original.end(), derived.begin(), derived.end());
        uint64_t ownerBytes = 0;
        for (const auto &buffer : original) {
          if (buffer.sizeBytes() > UINT64_MAX - ownerBytes) throw std::overflow_error("idle residency immutable union bytes overflow");
          ownerBytes += buffer.sizeBytes();
        }
        if (!idle_maintenance::qualifiedUnionAvailable(weights.sourceIdentity(), weights.manifestFingerprint(),
            originalCount, originalBytes, original.size(), ownerBytes,
            descriptor.layers, descriptor.experts, descriptor.hiddenSize, maintenanceStorage)) {
          idleMaintenanceFailureReason = "qualified model requires its complete verified saved operand and expert stores";
        } else {
          const uint64_t planned = idle_maintenance::Maintenance::plannedBytes(backend, maintenanceStorage);
          metal::AllocationFailure failure = metal::AllocationFailure::None;
          auto reservation = governor.tryReserve(planned, &failure);
          if (!reservation) throw metal::MetalAllocationError("idle residency diagnostics cannot reserve memory", failure);
          const uint64_t before = backend.memoryStats().allocatedBytes;
          idleMaintenance = std::make_unique<idle_maintenance::Maintenance>(backend, std::move(original), maintenanceStorage);
          if (idleMaintenance->allocatedBytes() > planned ||
              metal::allocationDelta(before, backend.memoryStats().allocatedBytes) > planned)
            throw std::runtime_error("idle residency diagnostics exceed governed planned bytes");
          reservation->commit();
        }
      }
      const uint64_t instance = (uint64_t{static_cast<uint32_t>(getpid())} << 32) ^
          static_cast<uint64_t>(Clock::now().time_since_epoch().count());
      // Constructors/startup conversion commands are never request-labelled.
      if (requestCommandTrace)
        backend.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::Command);
      Worker worker(transport, backend, weights, forward, governor, pressure, capacity,
                    instance ? instance : 1, head ? &*head : nullptr, batch ? &*batch : nullptr, prefillRows,
                    jointVerify ? &*jointVerify : nullptr, jointHead ? &*jointHead : nullptr,
                    std::move(jointHidden), singletonMTP, batchPrefill ? &*batchPrefill : nullptr,
                    std::move(batchPrefillHidden), batchPrefillRows,
                    batchPrimeHead ? &*batchPrimeHead : nullptr, std::move(batchPrimeInput),
                    std::move(savedResidency), std::move(savedResidencyLease), std::move(requestCommandTrace),
                    std::move(idleMaintenance), idleMaintenanceInterval, idleMaintenanceRequested,
                    std::move(idleMaintenanceFailureReason));
      try { return worker.run(); }
      catch (const std::exception &error) {
        if (transport.stopping()) return transport.failed() ? 2 : 0;
        if (!transport.stopping()) {
          try { transport.send(wire::ErrorEvent{wire::FailureClass::EngineUnhealthy, 0, false,
              "native_execution_failed", error.what()}); } catch (...) {}
        }
        transport.stop();
        std::cerr << "error: Flash native worker: " << error.what() << '\n';
        return 3;
      }
    } catch (const std::exception &error) {
      std::cerr << "error: Flash native startup: " << error.what() << '\n';
      return 3;
    }
  }
}
} // namespace splash::flash

#ifndef SPLASH_FLASH_NO_MAIN
int main(int argc, char **argv) { return splash::flash::runFlashWorker(argc, argv); }
#endif
