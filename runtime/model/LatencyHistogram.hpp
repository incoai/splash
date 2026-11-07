#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>

namespace splash::model {

// Latency quantiles: from LatencyHistogram each is the upper bound of its
// log-2 bucket of microseconds (so within a factor of two), from
// FineLatencyHistogram within a millisecond above the latency it stands for
// and never above the maximum.
struct LatencyStats final {
  uint64_t count = 0;
  double totalMilliseconds = 0;
  double maximumMilliseconds = 0;
  double p50Milliseconds = 0, p95Milliseconds = 0, p99Milliseconds = 0;
};

// Latencies in log-2 buckets of microseconds, updated from any thread.
class LatencyHistogram final {
public:
  void add(uint64_t nanoseconds) noexcept {
    const uint64_t micros = nanoseconds / 1000;
    const uint32_t bucket = std::min<uint32_t>(uint32_t(std::bit_width(micros)), kBuckets - 1);
    buckets_[bucket].fetch_add(1, std::memory_order_relaxed);
    total_.fetch_add(nanoseconds, std::memory_order_relaxed);
    uint64_t seen = maximum_.load(std::memory_order_relaxed);
    while (nanoseconds > seen && !maximum_.compare_exchange_weak(seen, nanoseconds, std::memory_order_relaxed)) {
    }
  }
  [[nodiscard]] LatencyStats snapshot() const noexcept {
    LatencyStats result;
    std::array<uint64_t, kBuckets> counts{};
    for (uint32_t b = 0; b < kBuckets; ++b) {
      counts[b] = buckets_[b].load(std::memory_order_relaxed);
      result.count += counts[b];
    }
    result.totalMilliseconds = double(total_.load(std::memory_order_relaxed)) / 1e6;
    result.maximumMilliseconds = double(maximum_.load(std::memory_order_relaxed)) / 1e6;
    const auto quantile = [&](double q) {
      if (!result.count) return 0.0;
      const uint64_t rank = uint64_t(q * double(result.count - 1)) + 1;
      uint64_t seen = 0;
      for (uint32_t b = 0; b < kBuckets; ++b) {
        seen += counts[b];
        // Bucket b holds microseconds of bit width b: below 2^b.
        if (seen >= rank) return double(uint64_t{1} << b) / 1000.0;
      }
      return result.maximumMilliseconds;
    };
    result.p50Milliseconds = quantile(0.50);
    result.p95Milliseconds = quantile(0.95);
    result.p99Milliseconds = quantile(0.99);
    return result;
  }

private:
  static constexpr uint32_t kBuckets = 40;
  std::array<std::atomic<uint64_t>, kBuckets> buckets_{};
  std::atomic<uint64_t> total_{0}, maximum_{0};
};

// Latencies in buckets of a millisecond, updated from any thread, for
// quantiles a gate holds to a bound of the same order (the scheduler's
// contended prefill units): bucket b holds (b - 1, b] ms and reads as b, or
// as the maximum where that is lower. Latencies past the buckets share the
// last one, which reads as the maximum.
class FineLatencyHistogram final {
public:
  void add(uint64_t nanoseconds) noexcept {
    const uint64_t millis = nanoseconds / kNanoseconds + (nanoseconds % kNanoseconds != 0);
    buckets_[std::min<uint64_t>(millis, kBuckets - 1)].fetch_add(1, std::memory_order_relaxed);
    total_.fetch_add(nanoseconds, std::memory_order_relaxed);
    uint64_t seen = maximum_.load(std::memory_order_relaxed);
    while (nanoseconds > seen && !maximum_.compare_exchange_weak(seen, nanoseconds, std::memory_order_relaxed)) {
    }
  }
  [[nodiscard]] LatencyStats snapshot() const noexcept {
    LatencyStats result;
    std::array<uint64_t, kBuckets> counts{};
    for (uint32_t b = 0; b < kBuckets; ++b) {
      counts[b] = buckets_[b].load(std::memory_order_relaxed);
      result.count += counts[b];
    }
    result.totalMilliseconds = double(total_.load(std::memory_order_relaxed)) / 1e6;
    result.maximumMilliseconds = double(maximum_.load(std::memory_order_relaxed)) / 1e6;
    const auto quantile = [&](double q) {
      if (!result.count) return 0.0;
      const uint64_t rank = uint64_t(q * double(result.count - 1)) + 1;
      uint64_t seen = 0;
      for (uint32_t b = 0; b + 1 < kBuckets; ++b) {
        seen += counts[b];
        if (seen >= rank) return std::min(double(b), result.maximumMilliseconds);
      }
      return result.maximumMilliseconds;
    };
    result.p50Milliseconds = quantile(0.50);
    result.p95Milliseconds = quantile(0.95);
    result.p99Milliseconds = quantile(0.99);
    return result;
  }

private:
  static constexpr uint64_t kNanoseconds = 1'000'000;
  // A millisecond's resolution to 4.094 s, eight times the 500 ms decode gap
  // units are sized to, in 32 KiB.
  static constexpr uint32_t kBuckets = 4096;
  std::array<std::atomic<uint64_t>, kBuckets> buckets_{};
  std::atomic<uint64_t> total_{0}, maximum_{0};
};

} // namespace splash::model
