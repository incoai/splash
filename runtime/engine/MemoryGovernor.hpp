#pragma once

#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>

namespace splash::engine {

struct HostMemoryPages {
  uint64_t active = 0;
  uint64_t inactive = 0;
  uint64_t speculative = 0;
  uint64_t wired = 0;
  uint64_t compressor = 0;
  uint64_t fileBacked = 0;
  uint64_t purgeable = 0;
};

// Physical memory minus used pages, crediting pageable file-backed and
// purgeable pages regardless of active/inactive status. The governor also
// enforces the engine budget, host reserve and system pressure.
[[nodiscard]] uint64_t estimateHostAvailableMemory(
    const HostMemoryPages &pages, uint64_t pageSize,
    uint64_t physicalMemoryBytes) noexcept;
[[nodiscard]] std::optional<uint64_t> queryHostAvailableMemory() noexcept;

enum class MemoryPressure : uint8_t {
  Normal,
  Warning,
  Critical,
};

// One spelling of the levels for status JSON and startup diagnostics.
[[nodiscard]] inline const char *
memoryPressureName(MemoryPressure pressure) noexcept {
  switch (pressure) {
  case MemoryPressure::Normal:
    return "normal";
  case MemoryPressure::Warning:
    return "warning";
  case MemoryPressure::Critical:
    return "critical";
  }
  return "critical";
}

[[nodiscard]] std::optional<MemoryPressure> querySystemMemoryPressure() noexcept;

// Allocation and recovery use separate watermarks to avoid oscillation.
// Low availability causes paced reclaim; unavailable telemetry pauses growth;
// the OS critical signal causes full eviction of unpinned cache entries.
inline constexpr uint64_t kHostWarningMarginBytes = 1ULL << 30;
inline constexpr uint64_t kHostRecoveryMarginBytes = 2ULL << 30;

struct MemoryGovernorSnapshot {
  uint64_t limitBytes = 0;
  uint64_t observedResidentBytes = 0;
  uint64_t reservedBytes = 0;
  uint64_t headroomBytes = 0;
  MemoryPressure pressure = MemoryPressure::Normal;
  // Failed reservation attempts, including retries of the same request.
  uint64_t deniedReservations = 0;
  bool hostMeasurementValid = false;
  uint64_t hostAvailableBytes = 0;
  uint64_t hostReserveBytes = 0;
  uint64_t hostHeadroomBytes = 0;
  MemoryPressure systemPressure = MemoryPressure::Normal;
  bool growthAllowed = true;
  bool hostGrowthAllowed = true;
};

struct MemoryReclaimDirective {
  bool reclaimEmptyKvExtents = false;
  bool evictAllUnpinnedPrefixes = false;
  uint64_t targetBytes = 0;
  // Keep the newest state publication, the point a follow-up request resumes
  // from. Only a shrink that nothing is waiting for can afford to.
  bool keepResumePoint = false;
};

// Bounded shrink passes separated by a telemetry settling interval. New host
// pressure is never offset by bytes reclaimed earlier in the same episode.
class MemoryPressurePolicy final {
public:
  // requestWaiting reports whether a request cannot proceed for want of
  // memory. Without one the pass is speculative and keeps the resume point.
  [[nodiscard]] MemoryReclaimDirective
  update(const MemoryGovernorSnapshot &snapshot, double nowMilliseconds,
         bool requestWaiting) noexcept;

private:
  double nextReclaimMilliseconds_ = 0.0;
};

// The sole physical-memory admission ledger. It does not allocate, evict, or
// schedule work; it only gives a short-lived byte reservation to a caller that
// is about to commit a placement heap. That keeps policy out of MetalBackend
// and makes every growth operation transactional.
class MemoryGovernor final {
public:
  using HostAvailableMemoryProvider =
      std::function<std::optional<uint64_t>()>;

  class Reservation final {
  public:
    Reservation() = default;
    ~Reservation();
    Reservation(const Reservation &) = delete;
    Reservation &operator=(const Reservation &) = delete;
    Reservation(Reservation &&) noexcept;
    Reservation &operator=(Reservation &&) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    void commit();

  private:
    Reservation(MemoryGovernor *owner, uint64_t bytes);
    void release() noexcept;

    MemoryGovernor *owner_ = nullptr;
    uint64_t bytes_ = 0;

    friend class MemoryGovernor;
  };

  MemoryGovernor(metal::MetalBackend &backend, uint64_t limitBytes,
                 uint64_t hostReserveBytes);
  MemoryGovernor(metal::MetalBackend &backend, uint64_t limitBytes,
                 uint64_t hostReserveBytes,
                 HostAvailableMemoryProvider hostAvailableMemory);

  [[nodiscard]] std::optional<Reservation> tryReserve(
      uint64_t bytes, metal::AllocationFailure *failure = nullptr);
  // Low-level storage/model components receive only this transactional
  // callback, so physical allocation stays governed without introducing a
  // reverse dependency on engine policy.
  [[nodiscard]] metal::AllocationAdmission allocationAdmission() noexcept;
  void setPressure(MemoryPressure pressure) noexcept;
  [[nodiscard]] MemoryGovernorSnapshot snapshot() const noexcept;

private:
  [[nodiscard]] uint64_t
  observedResidentBytes(bool refreshDevice = false) const noexcept;
  [[nodiscard]] std::optional<uint64_t> sampleHostAvailable() const noexcept;
  [[nodiscard]] uint64_t
  hostHeadroomBytes(const std::optional<uint64_t> &hostAvailable,
                    uint64_t reservedBytes) const noexcept;
  [[nodiscard]] MemoryPressure updateEffectivePressure(
      const std::optional<uint64_t> &hostAvailable,
      uint64_t reservedBytes) const noexcept;
  void release(uint64_t bytes) noexcept;

  metal::MetalBackend &backend_;
  uint64_t limitBytes_ = 0;
  uint64_t hostReserveBytes_ = 0;
  HostAvailableMemoryProvider hostAvailableMemory_;
  mutable std::mutex mutex_;
  uint64_t reservedBytes_ = 0;
  uint64_t deniedReservations_ = 0;
  MemoryPressure systemPressure_ = MemoryPressure::Normal;
  mutable bool hostConstrained_ = false;
};

} // namespace splash::engine
