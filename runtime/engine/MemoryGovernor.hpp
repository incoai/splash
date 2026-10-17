#pragma once

#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <functional>
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

// What macOS can hand out without swapping, two ways. Both count the free
// pages and the pageable file-backed and purgeable pages, regardless of
// active/inactive status; memory in no VM queue (the firmware carve-out, tag
// storage) is never counted. With compression, available adds what
// compressing the anonymous pages frees at the compressor's present ratio,
// counted at most at 2:1 (half of them; 2:1 also while the compressor holds
// nothing): what memory a request needs may take, other applications paying
// with compression. spare adds half of that credit (at most a quarter of the
// anonymous pages): what memory no request needs, the cache, may take, other
// applications paying half as much of it in compression.
struct HostMemory final {
  uint64_t available = 0;
  uint64_t spare = 0;
};
[[nodiscard]] HostMemory estimateHostMemory(const HostMemoryPages &pages,
                                            uint64_t pageSize,
                                            bool compression) noexcept;
// The live estimate, counting compression unless macOS reports critical
// memory pressure (or none): a Mac that uses its compressor as designed keeps
// serving, the credit shrinking as the anonymous pages it counts are
// compressed, and one in critical pressure falls back to the pages it can
// hand out as they are.
[[nodiscard]] std::optional<HostMemory> queryHostMemory() noexcept;

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

// Growth that no request in service needs leaves the warning margin free
// above the spare reserve, in the host's spare memory, and the paced reclaim
// returns cache to keep it so. Once the host has run short, a request that
// waits for memory has the paced reclaim free toward the recovery margin,
// and such growth waits for that margin meanwhile, so that what reclaim
// frees goes to the request; with no request waiting, a pass only restores
// the warning margin and growth keeps to it. macOS's warning asks caches to
// give memory back: while it lasts, such growth stops whatever the host's
// headroom, and every pass returns up to the warning margin's worth of cache
// until the level is normal again. Unavailable telemetry pauses growth; the
// OS critical signal causes full eviction of unpinned cache entries and stops
// all growth.
inline constexpr uint64_t kHostWarningMarginBytes = 1ULL << 30;
inline constexpr uint64_t kHostRecoveryMarginBytes = 2ULL << 30;

struct MemoryGovernorSnapshot {
  uint64_t limitBytes = 0;
  // Charged against the limit: the backend's allocated buffers plus the
  // untracked reserve, or the device's allocation when that is larger.
  uint64_t chargedBytes = 0;
  // Room under the limit beside what is charged and reserved: zero once the
  // engine's limit is reached.
  uint64_t headroomBytes = 0;
  MemoryPressure pressure = MemoryPressure::Normal;
  // Failed reservation attempts, including retries of the same request.
  uint64_t deniedReservations = 0;
  bool hostMeasurementValid = false;
  uint64_t hostAvailableBytes = 0;
  uint64_t hostReserveBytes = 0;
  uint64_t hostSpareBytes = 0;
  uint64_t hostSpareReserveBytes = 0;
  // The host's spare memory above its spare reserve, beside what is
  // reserved: the room growth that no request in service needs takes from.
  uint64_t hostHeadroomBytes = 0;
  MemoryPressure systemPressure = MemoryPressure::Normal;
  // Whether the host has room for growth that no request in service needs
  // and macOS reports normal pressure. Admission still grants what such a
  // request needs while this is false, short of critical pressure.
  // Allocation need not ask: admission names the host as the cause of every
  // refusal the host shares.
  bool hostGrowthAllowed = true;
};

// Critical pressure evicts every unpinned entry and takes what a request
// starts from, one lane's pooled state buffers and the empty KV runway
// extent, so it has no target. Every other pass keeps those for the growth
// that follows; targetBytes zero returns only empty extents. keepResumePoint keeps
// the newest state publication (else the newest checkpoint), the point a
// follow-up request resumes from; only a shrink nothing waits for can afford
// to, or one under macOS's warning, which holds a waiting request back too.
struct MemoryReclaimDirective {
  bool critical = false;
  uint64_t targetBytes = 0;
  bool keepResumePoint = false;
};

// What a reclaim pass made of its directive's target.
enum class ReclaimOutcome : uint8_t {
  // The directive set none; the pass returned only empty extents.
  Untargeted,
  // Released, counting the pages whose copies are being written.
  Met,
  // Transfers in flight hold back the rest, which a pass can take once they
  // land.
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
  // The pass to run now; none under normal pressure. requestWaiting reports
  // whether a request cannot proceed for want of memory. Without one the
  // pass is speculative: it keeps the resume point, as every pass under
  // macOS's warning does, and, unless macOS warns, only restores the warning
  // margin.
  [[nodiscard]] std::optional<MemoryReclaimDirective>
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

// The sole memory admission ledger. It does not allocate, evict, or
// schedule work; it only gives a short-lived byte reservation to a caller that
// is about to allocate Metal memory. That keeps policy out of MetalBackend
// and makes every growth operation transactional. Single-threaded: startup,
// the engine and the control pass all run on the native loop thread; nothing
// here is synchronized.
class MemoryGovernor final {
public:
  using HostMemoryProvider = std::function<std::optional<HostMemory>()>;

  // Metal memory outside the backend's buffers (pipelines, driver
  // allocations) is charged only beyond untrackedReserveBytes, the part of
  // the limit the caller has set aside for it. The host's available memory
  // must exceed hostReserveBytes, which macOS keeps; growth that no request
  // in service needs keeps hostSpareReserveBytes of its spare memory, and
  // the warning margin above them, for other applications.
  // hostHeldBytes, host memory held outside Metal (the streamed table's
  // row cache), is charged in full.
  MemoryGovernor(metal::MetalBackend &backend, uint64_t limitBytes,
                 uint64_t hostReserveBytes, uint64_t hostSpareReserveBytes,
                 HostMemoryProvider hostMemory, uint64_t untrackedReserveBytes, uint64_t hostHeldBytes = 0);

  // The only way to reserve memory. Low-level storage/model components
  // receive only this transactional callback, so allocation stays governed
  // without introducing a reverse dependency on engine policy. It reserves
  // the bytes under the limit and the host's headroom while the allocation
  // runs, or refuses them with the cause: HostPressure when the host refuses,
  // whether or not the limit does too, and EngineBudget when only the limit
  // does. The host refuses under critical pressure and, unless a request in
  // service needs the bytes (setServing), while macOS warns, inside the
  // warning margin or while it holds for the recovery margin for a waiting
  // request (setRequestWaiting).
  [[nodiscard]] metal::AllocationAdmission allocationAdmission() noexcept;
  // Marks the reservations that follow as memory a request in service needs,
  // until it is cleared, and returns the mark it replaced, which a scope
  // that sets it restores (Serving). Neither the host's margins nor macOS's
  // warning refuse those: holding them back would strand the request and
  // the memory it already has, while the cache it could give up instead is
  // what the paced reclaim returns. The limit and critical pressure refuse
  // them like any other. A start marks what it allocates so until serving
  // begins (RuntimeResources::beginServing).
  bool setServing(bool serving) noexcept;
  // Whether a request cannot proceed for want of memory, as each control pass
  // finds: only then does growth wait for the recovery margin once the host
  // has run short.
  void setRequestWaiting(bool waiting) noexcept;
  void setPressure(MemoryPressure pressure) noexcept;
  // The outcome of the engine's last reclaim pass with a target. While one
  // finds nothing left to release, the hold for the recovery margin is
  // waived: growth proceeds as with no request waiting, since only other
  // applications could restore the rest, and the paced passes keep looking.
  // A pass that releases or waits for memory again, or the host's recovery,
  // ends the waiver. Only a pass while the host is short counts, so no
  // episode starts waived by a pass that macOS's warning alone asked for.
  void reclaimed(ReclaimOutcome outcome) noexcept;
  // What the governor makes of a fresh sample of the host. Reading it
  // changes nothing: the host's hysteresis moves only with the samples a
  // reservation or the control pass (evaluate) takes, so /status and the
  // engine's growth queries do not move admission.
  [[nodiscard]] MemoryGovernorSnapshot snapshot() const noexcept;
  // The control pass's evaluation: snapshot(), with the host's hysteresis
  // moved to its sample, as a reservation's moves it.
  MemoryGovernorSnapshot evaluate() noexcept;

private:
  // Held while an admitted allocation runs; released when it ends.
  class Reservation final {
  public:
    ~Reservation();
    Reservation(const Reservation &) = delete;
    Reservation &operator=(const Reservation &) = delete;
    Reservation(Reservation &&) noexcept;

  private:
    Reservation(MemoryGovernor *owner, uint64_t bytes);
    void release() noexcept;

    MemoryGovernor *owner_ = nullptr;
    uint64_t bytes_ = 0;

    friend class MemoryGovernor;
  };

  // The host under the hysteresis at one sample of its available memory.
  struct HostState final {
    // From when its headroom falls inside the warning margin, or cannot be
    // measured, until the headroom clears the recovery margin.
    bool constrained = false;
    // Reclaim found nothing to release in this episode of host pressure.
    bool reclaimExhausted = false;

    // Short of the recovery margin with reclaim not yet exhausted.
    [[nodiscard]] bool held() const noexcept {
      return constrained && !reclaimExhausted;
    }
  };

  // Growth that no request in service needs waits for the recovery margin
  // while the host holds and a request waits for memory.
  [[nodiscard]] bool growthHeld(const HostState &host) const noexcept {
    return requestWaiting_ && host.held();
  }
  // allocationAdmission's reservation, or nothing with the cause of the
  // refusal in failure.
  [[nodiscard]] std::optional<Reservation>
  tryReserve(uint64_t bytes, metal::AllocationFailure &failure);
  [[nodiscard]] uint64_t
  chargedBytes(bool refreshDevice = false) const noexcept;
  [[nodiscard]] std::optional<HostMemory> sampleHost() const noexcept;
  [[nodiscard]] uint64_t hostHeadroomBytes(const std::optional<HostMemory> &host,
                                           uint64_t reservedBytes) const noexcept;
  // The host's state at this sample, from the one it was last evaluated at.
  [[nodiscard]] HostState hostState(const std::optional<HostMemory> &host) const noexcept;
  [[nodiscard]] MemoryPressure
  effectivePressure(const HostState &host) const noexcept;
  [[nodiscard]] MemoryGovernorSnapshot
  snapshot(const std::optional<HostMemory> &hostMemory, const HostState &host) const noexcept;
  void release(uint64_t bytes) noexcept;

  metal::MetalBackend &backend_;
  uint64_t limitBytes_ = 0;
  uint64_t hostReserveBytes_ = 0;
  uint64_t hostSpareReserveBytes_ = 0;
  HostMemoryProvider hostMemory_;
  uint64_t untrackedReserveBytes_ = 0;
  uint64_t hostHeldBytes_ = 0;
  uint64_t reservedBytes_ = 0;
  bool serving_ = false;
  bool requestWaiting_ = false;
  uint64_t deniedReservations_ = 0;
  MemoryPressure systemPressure_ = MemoryPressure::Normal;
  // As the latest reservation or control pass evaluated it.
  HostState host_;
};

} // namespace splash::engine
