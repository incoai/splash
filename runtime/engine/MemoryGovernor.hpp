#pragma once

#include "engine/MemoryLedger.hpp"
#include "engine/ReclaimClass.hpp"
#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

struct task_vm_info;
struct vm_statistics64;

namespace splash::engine {

struct EngineConfig;
struct EngineMemoryBreakdown;

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
// pages and the pageable file-backed and purgeable pages, active or
// inactive; memory in no VM queue (the firmware carve-out, tag storage) is
// never counted. With compression, available adds what compressing the
// anonymous pages frees at the compressor's present ratio, at most 2:1
// (2:1 also while the compressor holds nothing): what memory a request needs
// may take, other applications paying with compression. spare adds half of
// that credit (at most a quarter of the anonymous pages): what memory no
// request needs, the cache, may take, other applications paying half as much.
struct HostMemory final {
  uint64_t available = 0;
  uint64_t spare = 0;
};
[[nodiscard]] HostMemory estimateHostMemory(const HostMemoryPages &pages,
                                            uint64_t pageSize,
                                            bool compression) noexcept;
// The live estimate, counting compression unless macOS reports critical
// memory pressure (or none), so a Mac that uses its compressor as designed
// keeps serving: the credit shrinks as the anonymous pages it counts are
// compressed, and under critical pressure only the pages macOS can hand out
// as they are count.
[[nodiscard]] std::optional<HostMemory> queryHostMemory() noexcept;

// The host's memory as macOS counts it, reported in /status beside the
// governor's estimate and acted on by no rule, all in bytes: the process's
// physical footprint and the part of it macOS holds compressed
// (task_vm_info's phys_footprint and compressed), the host's file-backed
// pages and its inactive and speculative pages (vm_statistics64 splits
// neither queue into file and anonymous pages), and the file cache macOS
// keeps however short memory is (vm.vm_page_filecache_min).
struct HostTelemetry final {
  uint64_t physFootprintBytes = 0;
  uint64_t compressedBytes = 0;
  uint64_t fileBackedBytes = 0;
  uint64_t inactiveBytes = 0;
  uint64_t fileCacheMinimumBytes = 0;
};
// The telemetry of a task's `task` and the host's `pages` of `pageSize` bytes
// beside a file cache minimum of `fileCacheMinimumPages` pages.
[[nodiscard]] HostTelemetry hostTelemetryOf(const task_vm_info &task, const vm_statistics64 &pages,
                                            uint64_t fileCacheMinimumPages, uint64_t pageSize) noexcept;
// The live telemetry of this process and its host; none when macOS does not
// report any part of it.
[[nodiscard]] std::optional<HostTelemetry> queryHostTelemetry() noexcept;

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

// One account of the memory ledger: a consumer the ledger moves between its
// floor and its cap, named as its owner names it ("kv", "states", "fixed",
// or a model pool's name, which status renders and nothing branches on).
struct MemoryAccountSnapshot final {
  std::string name;
  uint64_t floorBytes = 0;
  uint64_t currentBytes = 0;
  // The bound its growth approaches (model::PoolState::capBytes).
  uint64_t capBytes = 0;
  uint64_t unitBytes = 0;
  // Units it took and gave back since the engine started.
  uint64_t grows = 0;
  uint64_t shrinks = 0;
};

// Bytes that transfers in flight or owed free when they land, the part of
// them owed to waiting claims, and the age of the oldest transfer.
struct OwedBytesSnapshot final {
  uint32_t transfers = 0;
  uint64_t bytes = 0;
  uint64_t toClaimsBytes = 0;
  double oldestMilliseconds = 0.0;
};

// Requests that cannot proceed for want of memory (a wait for the SSD
// cache's quota is not one): how many, the bytes they lack, and how long the
// oldest has waited.
struct MemoryClaimsSnapshot final {
  uint32_t waiting = 0;
  uint64_t bytes = 0;
  double oldestMilliseconds = 0.0;
};

struct MemoryGovernorSnapshot {
  uint64_t limitBytes = 0;
  // Charged against the limit: the backend's allocated buffers plus the
  // untracked reserve, or the device's allocation when that is larger.
  uint64_t chargedBytes = 0;
  // Room under the limit beside what is charged and reserved: zero once the
  // engine's limit is reached.
  uint64_t headroomBytes = 0;
  // macOS's critical pressure; else warning while macOS warns or the host
  // is in its band (short of its margin, or unmeasured); else normal.
  MemoryPressure pressure = MemoryPressure::Normal;
  // Failed reservation attempts, including retries of the same request.
  uint64_t deniedReservations = 0;
  bool hostMeasurementValid = false;
  uint64_t hostAvailableBytes = 0;
  uint64_t hostReserveBytes = 0;
  uint64_t hostSpareBytes = 0;
  uint64_t hostSpareReserveBytes = 0;
  // The host's spare memory above its spare reserve, beside what is
  // reserved: the room the host's rule weighs work against (h).
  uint64_t hostHeadroomBytes = 0;
  MemoryPressure systemPressure = MemoryPressure::Normal;
  // Whether the host's rule admits growth now: macOS reports normal
  // pressure, the host is outside its band and keeps its margin. A claim
  // the host refuses is told so with its cause, whatever this says.
  bool hostGrowthAllowed = true;
  // The host's memory as macOS counts it, which /status reports; none when
  // it does not (queryHostTelemetry).
  std::optional<HostTelemetry> hostTelemetry{};
  // The ledger's accounts, what transfers will return, and the claims that
  // wait.
  std::vector<MemoryAccountSnapshot> accounts{};
  OwedBytesSnapshot owed{};
  MemoryClaimsSnapshot claims{};
};

// The sole memory admission. It does not allocate, evict or schedule work;
// it gives a short-lived byte reservation to a caller about to allocate
// Metal memory, which keeps policy out of MetalBackend and makes every
// growth transactional. Single-threaded: startup, the engine and the control
// pass all run on the native loop thread; nothing here is synchronized.
class MemoryGovernor final {
public:
  using HostMemoryProvider = std::function<std::optional<HostMemory>()>;

  // Metal memory outside the backend's buffers (pipelines, driver
  // allocations) is charged only beyond untrackedReserveBytes, the part of
  // the limit the caller has set aside for it. The host's available memory
  // must exceed hostReserveBytes, which macOS keeps; the host's rule weighs
  // work against its spare memory beyond hostSpareReserveBytes, which it
  // keeps for other applications with the margin above them.
  // hostHeldBytes, host memory held outside Metal (the streamed table's
  // row cache), is charged in full.
  MemoryGovernor(metal::MetalBackend &backend, uint64_t limitBytes,
                 uint64_t hostReserveBytes, uint64_t hostSpareReserveBytes,
                 HostMemoryProvider hostMemory, uint64_t untrackedReserveBytes, uint64_t hostHeldBytes = 0);
  // The governor of an engine that `budget` plans, as a start builds it: the
  // hard budget is its limit, the plan's pipeline and runtime reserves are
  // the untracked reserve, the host keeps the reserves EngineMemoryPolicy
  // sets for the Mac's memory, and the streamed table's host memory is
  // charged in full.
  MemoryGovernor(metal::MetalBackend &backend, const EngineMemoryBreakdown &budget, HostMemoryProvider hostMemory);

  // The only way to reserve memory. Storage and model components receive
  // only this transactional callback, so allocation stays governed without
  // a reverse dependency on engine policy. Every allocation runs inside its
  // claim's Scope; one outside any is a defect (std::logic_error). It
  // reserves the bytes under the limit, beside the room claims ranked above
  // it hold, while the allocation runs, if the host's rule admits them for
  // the claim (its class's, or a lane's growth's for what the lane needs).
  // Otherwise it refuses with the cause: HostPressure when the host refuses,
  // whether or not the limit does too, EngineBudget when only the limit
  // does. A refusal records what the claim lacks (need).
  [[nodiscard]] metal::AllocationAdmission allocationAdmission() noexcept;
  void setPressure(MemoryPressure pressure) noexcept;
  // What the governor makes of a fresh sample of the host. Reading it
  // changes nothing: the host's band moves only with the samples a
  // reservation or the control pass takes, so /status does not move
  // admission.
  [[nodiscard]] MemoryGovernorSnapshot snapshot() const noexcept;
  // snapshot(), with the host's band moved to its sample, as a
  // reservation's and the control pass's move it.
  MemoryGovernorSnapshot evaluate() noexcept;

  // The accounts, claims and victim sources the governor's rule works on.
  [[nodiscard]] MemoryLedger &ledger() noexcept { return ledger_; }
  [[nodiscard]] const MemoryLedger &ledger() const noexcept { return ledger_; }
  // While one lives, admissions are for `claim`, under the host's rule for
  // its class and the room claims ranked above it hold
  // (MemoryLedger::heldBytes). It restores the scope it replaced.
  class Scope final {
  public:
    Scope(MemoryGovernor &governor, ClaimId claim) noexcept;
    ~Scope();
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

  private:
    MemoryGovernor &governor_;
    ClaimId previousClaim_;
  };
  // The need its scope's refusals recorded, with the caller's in-place part.
  [[nodiscard]] Need need(ClaimId claim, const std::optional<Need> &inPlace = {}) const;
  // MemoryLedger::heldPages beside the room admission grants in-service
  // work now: under the engine's limit, and of the host's headroom what lies
  // beyond its margin at normal pressure, read now, since pages are handed
  // out between the host rule's samples.
  [[nodiscard]] std::array<uint32_t, 3> heldPages(ClaimId claim) const;
  // Frees what the claim lacks (MemoryLedger::free). A need the host refused
  // is met in place first, from the pages and state buffers already
  // allocated; failing that, the retries of an InService claim, or
  // of an Optional one its lane needs (ClaimSpec::laneNeeds), pass the host,
  // and another claim falls short.
  [[nodiscard]] FreeResult free(ClaimId claim, const Need &need);
  // What free() would find for such a claim, up to `enough`, taking nothing
  // and leaving the chains `kept` (VictimQuery::kept).
  [[nodiscard]] uint64_t available(const ClaimSpec &spec, const Need &need, const RestoredChains *kept = nullptr) const;
  // Whether the host admits an allocation of the class, as its last sample
  // found: the paths that ask take no sample of their own.
  [[nodiscard]] bool admits(ClaimClass cls) const noexcept;
  // Whether a temporary tier's copies ahead pay (MemoryLedger::copyDemand).
  [[nodiscard]] bool copyDemand() const noexcept;
  // Whether a claim waits for memory on its way back, which copies ahead
  // would hold up.
  [[nodiscard]] bool claimWaiting() const noexcept;
  // The control pass, between commands, once the host has settled from the
  // last pass. Under macOS's critical pressure: one Immediate host claim of
  // the margin plus what host-refused claims wait for, once per settle time,
  // which may take the store above its floor, certain reuse and units in
  // use. Otherwise, with a deficit (host-refused claims count in it), the
  // host claim frees the lesser of the margin and the deficit; then the
  // pools grow. macOS's warning alone takes nothing; it holds back growth
  // from free room and elective admission. True when the host claim waits
  // for memory on its way back. The transport then runs the pass again at
  // the next command-free point, which lands transfers and grows the pools;
  // the host claim itself is judged again once the host has settled.
  [[nodiscard]] bool controlPass(double nowMilliseconds);
  // At startup, before Ready: grows the pools into the free room, by the
  // host's rule at macOS's `pressure` now.
  void growToTargets(double nowMilliseconds, MemoryPressure pressure);

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

  // The host's one rule at its latest sample. Headroom h is the spare memory
  // above the spare reserve, less what is being allocated and pool growth
  // the host's counters do not show yet. The band begins when h falls below
  // the margin M (kHostMarginBytes) or cannot be measured, and ends once h
  // clears M by the largest pool growth unit (or by what host-refused claims
  // wait for, if less: the band never holds them once h has room for them),
  // or once a host pass finds nothing to take. Floor work passes. In-service
  // work passes while h keeps M beyond it at normal pressure, and as an
  // exempt claim once nothing in place meets it. A pool's growth the growth
  // pass admitted on its net change (growthHost) passes. Anything else needs
  // h to keep M beyond it outside the band and macOS's warning. macOS's
  // critical pressure refuses everything.
  class HostRule final {
  public:
    void sample(const std::optional<HostMemory> &host, uint64_t spareReserve, uint64_t inFlight,
                MemoryPressure system, uint64_t hysteresisBytes, uint64_t refusedWaitingBytes,
                bool reclaimShort) noexcept;
    [[nodiscard]] bool admits(ClaimClass cls, uint64_t bytes, bool hostExempt) const noexcept;
    // What the host lacks of M and of what the claims it refused wait for,
    // beyond what is landing; nothing measured, nothing lacking.
    [[nodiscard]] uint64_t deficitBytes(uint64_t refusedWaitingBytes, uint64_t returningBytes) const noexcept;
    [[nodiscard]] bool constrained() const noexcept { return constrained_; }
    [[nodiscard]] MemoryPressure level() const noexcept { return system_; }

  private:
    bool measured_ = false;
    uint64_t headroom_ = 0;
    MemoryPressure system_ = MemoryPressure::Normal;
    bool constrained_ = true;
  };
  // The host claim's pass: up to `target` of what it may take.
  [[nodiscard]] FreeResult hostPass(uint64_t target, bool critical);
  // Room under the limit beside what is charged and being allocated.
  [[nodiscard]] uint64_t roomBytes() const noexcept;
  // HostRule::sample at a fresh sample of the host.
  void sampleHostRule(const std::optional<HostMemory> &host) noexcept;
  // The bytes the host's rule weighs an allocation of the class at: a pool's
  // growth leaves the host, besides the margin, what the lanes in service are
  // projected to grow into (MemoryLedger::laneDemandBytes), as it leaves them
  // that room under the limit; it never runs while a request's claim waits
  // (MemoryLedger::grow).
  [[nodiscard]] uint64_t hostBytes(ClaimClass cls, uint64_t bytes) const noexcept;
  [[nodiscard]] GrowthHost growthHost(bool paced);
  // The admission's reservation for the scope's claim, or nothing with the
  // cause of the refusal in failure.
  [[nodiscard]] std::optional<Reservation>
  tryReserve(uint64_t bytes, metal::AllocationFailure &failure);
  [[nodiscard]] uint64_t
  chargedBytes(bool refreshDevice = false) const noexcept;
  [[nodiscard]] std::optional<HostMemory> sampleHost() const noexcept;
  [[nodiscard]] uint64_t hostHeadroomBytes(const std::optional<HostMemory> &host,
                                           uint64_t reservedBytes) const noexcept;
  // Of what pools grew within kHostSettleMilliseconds, what the host's
  // reading `host` does not show yet: what the headroom has fallen since the
  // growth does not cover (a store chunk's pages count only once experts load
  // into it).
  [[nodiscard]] uint64_t unseenGrowthBytes(const std::optional<HostMemory> &host) const noexcept;
  [[nodiscard]] MemoryGovernorSnapshot
  snapshot(const std::optional<HostMemory> &hostMemory, const HostRule &rule) const noexcept;
  void release(uint64_t bytes) noexcept;

  metal::MetalBackend &backend_;
  uint64_t limitBytes_ = 0;
  uint64_t hostReserveBytes_ = 0;
  uint64_t hostSpareReserveBytes_ = 0;
  HostMemoryProvider hostMemory_;
  uint64_t untrackedReserveBytes_ = 0;
  uint64_t hostHeldBytes_ = 0;
  uint64_t reservedBytes_ = 0;
  uint64_t deniedReservations_ = 0;
  MemoryPressure systemPressure_ = MemoryPressure::Normal;
  MemoryLedger ledger_;
  // The innermost Scope's claim.
  ClaimId scope_;
  HostRule hostRule_;
  // The largest unit a pool grows by, as the last pass or evaluation found:
  // how far past the margin the band reaches. Kept, so that an admission a
  // pool's growth runs inside does not read the pool.
  uint64_t growthUnitBytes_ = 0;
  // When the host has settled from the last pass that returned memory or
  // waits for what lands, and whether the last host pass in the band found
  // nothing to take (the band's waiver).
  double nextHostPassMilliseconds_ = 0.0;
  bool bandWaived_ = false;
  // The latest control pass's clock, or the start growth's.
  double clockMilliseconds_ = 0.0;
  // Pools' growth the host's counters may not show yet (unseenGrowthBytes):
  // its bytes, the host's raw headroom before it, and when it lapses.
  uint64_t unseenGrowthBytes_ = 0;
  uint64_t unseenGrowthHeadroom_ = 0;
  double unseenGrowthUntilMilliseconds_ = 0.0;
};

// Connects an engine to the governor that admits its memory and whose ledger
// holds its requests' claims (EngineConfig::governor). Every engine that
// runs against a governor connects through it.
void connectToGovernor(EngineConfig &config, MemoryGovernor &governor);

} // namespace splash::engine
