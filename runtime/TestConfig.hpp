#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>

namespace splash {

// Values production holds constant, or measures live, that a test
// substitutes. Every field is empty in production, where the component's
// named constant or live estimate applies; a component reads its field once,
// when it is constructed. Only dev/tests writes it
// (dev/tests/engine/ScopedTestConfig.hpp).
struct TestConfig final {
  std::optional<double> commandTimeoutSeconds;     // metal::kCommandTimeoutSeconds
  std::optional<double> residencyKeepAliveSeconds; // metal::kResidencyKeepAliveSeconds
  std::optional<uint32_t> kvTierTransfers;        // engine::KvPageTier::kTransfers
  std::optional<size_t> transportInputQueueBytes; // engine::FdTransport::kInputQueueBytes
  std::optional<uint32_t> metricsLatencyWindow;   // engine::RuntimeMetrics::kLatencyWindow
  std::optional<uint32_t> prefillCheckpointTokens; // engine::kPrefillCheckpointTokens
  std::optional<double> resourceWaitTimeoutMilliseconds; // engine::kResourceWaitTimeoutMilliseconds
  std::optional<uint64_t> writeBehindHourlyBytes; // engine::WriteBehind::kHourlyBytes
  // RuntimeResources and its governor: the live vm_statistics64 estimate.
  std::function<std::optional<uint64_t>()> hostAvailableMemory;
  // NativeRuntime: the system clock in microseconds and the steady clock in
  // milliseconds.
  std::function<uint64_t()> unixMicros;
  std::function<double()> monotonicMilliseconds;
};

namespace detail {
inline TestConfig &testConfigStorage() noexcept {
  static TestConfig config;
  return config;
}
} // namespace detail

[[nodiscard]] inline const TestConfig &testConfig() noexcept {
  return detail::testConfigStorage();
}

} // namespace splash
