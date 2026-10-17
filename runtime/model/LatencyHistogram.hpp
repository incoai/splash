#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>

namespace splash::model {

// Latency quantiles from a log-2 histogram of microseconds: each value is
// the upper bound of its bucket (so within a factor of two).
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

} // namespace splash::model
