#pragma once

#include "AwakeClock.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace splash {

// Optional development instrumentation. Names have static storage; events
// contain geometry and opaque ids, never prompt text, token ids or paths.
// All instants use AwakeClock/system mach time, like Metal's GPU timestamps.
struct TraceEvent final {
  const char *name;
  uint32_t track;
  double begin;
  double end;
  uint64_t id = 0;
  uint32_t width = 0;
  uint32_t rows = 0;
  uint64_t context = 0;
  uint32_t flags = 0;
  double value = 0.0;
};

class TraceSink {
public:
  virtual ~TraceSink() = default;
  // May run on a completion thread or under the ANE handoff's lock. Do not
  // reenter the runtime or perform IO. Dropping a trace cannot fail serving.
  virtual void record(const TraceEvent &event) noexcept = 0;
};

inline double traceSeconds() noexcept {
  return std::chrono::duration<double>(AwakeClock::now().time_since_epoch()).count();
}

class TraceScope final {
public:
  TraceScope(TraceSink *sink, const char *name, uint32_t track = 1,
             uint64_t id = 0, uint32_t width = 0, uint32_t rows = 0,
             uint64_t context = 0, uint32_t flags = 0) noexcept
      : sink_(sink), event_{name, track, sink ? traceSeconds() : 0.0, 0.0,
                            id, width, rows, context, flags} {}
  ~TraceScope() {
    if (sink_) {
      event_.end = traceSeconds();
      sink_->record(event_);
    }
  }
  TraceScope(const TraceScope &) = delete;
  TraceScope &operator=(const TraceScope &) = delete;

private:
  TraceSink *sink_;
  TraceEvent event_;
};

// Preallocated, bounded storage shared by engine and handoff callbacks. IO
// happens only after the workload, so a filesystem stall cannot enter its
// measured intervals. Snapshot only after producers have quiesced.
class TraceBuffer final : public TraceSink {
public:
  explicit TraceBuffer(size_t capacity = 200000) : capacity_(capacity) {
    events_.reserve(capacity);
  }
  void record(const TraceEvent &event) noexcept override {
    try {
      std::lock_guard lock(mutex_);
      if (events_.size() < capacity_)
        events_.push_back(event);
      else
        ++dropped_;
    } catch (...) {
      ++dropped_;
    }
  }
  [[nodiscard]] std::vector<TraceEvent> snapshot() const {
    std::lock_guard lock(mutex_);
    return events_;
  }
  [[nodiscard]] uint64_t dropped() const noexcept { return dropped_.load(); }

private:
  size_t capacity_;
  mutable std::mutex mutex_;
  std::vector<TraceEvent> events_;
  std::atomic<uint64_t> dropped_{0};
};

} // namespace splash
