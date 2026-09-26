#pragma once

#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>

namespace splash::engine {

struct HostMemoryPages {
  // Mach's free_count, which includes the speculative pages.
  uint64_t free = 0;
  uint64_t speculative = 0;
  uint64_t fileBacked = 0;
  uint64_t purgeable = 0;
  // Anonymous pages outside the compressor, and the compressor's own pages
  // and the pages it holds, whose ratio is what compressing them saves.
  uint64_t anonymous = 0;
  uint64_t compressor = 0;
  uint64_t compressed = 0;
};

// The pages macOS can hand out without swapping: free pages plus pageable
// file-backed and purgeable pages, regardless of active/inactive status,
// and, with compression, what compressing the anonymous pages frees at the
// compressor's present ratio, counted at most at 2:1 (half of them; 2:1 also
// while the compressor holds nothing). Memory in no VM queue (the firmware
// carve-out, tag storage) is never available. The governor also enforces
// the engine budget, host reserve and system pressure.
[[nodiscard]] uint64_t estimateHostAvailableMemory(
    const HostMemoryPages &pages, uint64_t pageSize,
    bool compression = false) noexcept;
// The live estimate, counting compression unless macOS reports critical
// memory pressure (or none): a Mac that uses its compressor as designed keeps
// serving, the credit shrinking as the anonymous pages it counts are
// compressed, and one in critical pressure falls back to the pages it can
// hand out as they are.
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
  // Charged against the limit: the backend's resident buffers plus the
  // untracked reserve, or the device's allocation when that is larger.
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

// What a reclaim pass made of its directive's target.
enum class ReclaimOutcome : uint8_t {
  // The directive set none; the pass returned only empty backing.
  Untargeted,
  // Released, counting the pages whose copies are being written.
  Met,
  // Transfers or a release in flight hold back the rest, which a pass can
  // take once they land.
  Pending,
  // Nothing is left to release.
  Exhausted,
};

struct MemoryReclaimResult {
  uint64_t releasedBytes = 0;
  ReclaimOutcome outcome = ReclaimOutcome::Untargeted;
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
  // What the pass of `directive` achieved. The passes up to the next
  // measurement continue the part of its target that transfers held back:
  // a KV chain gives up one leaf at a time, each after its copy is written.
  void reclaimed(const MemoryReclaimDirective &directive,
                 const MemoryReclaimResult &result) noexcept;

private:
  double nextReclaimMilliseconds_ = 0.0;
  std::optional<MemoryReclaimDirective> continued_;
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
  // Metal memory outside the backend's buffers (pipelines, driver
  // allocations) is charged only beyond untrackedReserveBytes, the part of
  // the limit the caller has set aside for it.
  MemoryGovernor(metal::MetalBackend &backend, uint64_t limitBytes,
                 uint64_t hostReserveBytes,
                 HostAvailableMemoryProvider hostAvailableMemory,
                 uint64_t untrackedReserveBytes = 0);

  [[nodiscard]] std::optional<Reservation> tryReserve(
      uint64_t bytes, metal::AllocationFailure *failure = nullptr);
  // Low-level storage/model components receive only this transactional
  // callback, so physical allocation stays governed without introducing a
  // reverse dependency on engine policy.
  [[nodiscard]] metal::AllocationAdmission allocationAdmission() noexcept;
  void setPressure(MemoryPressure pressure) noexcept;
  // The outcome of the engine's last reclaim pass with a target. While one
  // finds nothing left to release, the hold for the recovery margin is
  // waived: growth that clears the warning margin proceeds, since only other
  // applications could restore the rest, and the paced passes keep looking.
  // A pass that releases or waits for memory again, or the host's recovery,
  // ends the waiver.
  void reclaimed(ReclaimOutcome outcome) noexcept;
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
  // Growth waits for the recovery margin.
  [[nodiscard]] bool hostHeld() const noexcept {
    return hostConstrained_ && !reclaimExhausted_;
  }
  void release(uint64_t bytes) noexcept;

  metal::MetalBackend &backend_;
  uint64_t limitBytes_ = 0;
  uint64_t hostReserveBytes_ = 0;
  HostAvailableMemoryProvider hostAvailableMemory_;
  uint64_t untrackedReserveBytes_ = 0;
  mutable std::mutex mutex_;
  uint64_t reservedBytes_ = 0;
  uint64_t deniedReservations_ = 0;
  MemoryPressure systemPressure_ = MemoryPressure::Normal;
  mutable bool hostConstrained_ = false;
  // Reclaim found nothing to release in this episode of host pressure.
  mutable bool reclaimExhausted_ = false;
};

} // namespace splash::engine
