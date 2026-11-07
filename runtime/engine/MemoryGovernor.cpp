#include "engine/MemoryGovernor.hpp"

#include "Checked.hpp"
#include "engine/Engine.hpp"
#include "engine/MemoryPlan.hpp"

#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::engine {

namespace {

// What compressing the anonymous pages frees: the share the compressor does
// not keep at its ratio of held to own pages, at most half of them.
uint64_t compressionSavings(const HostMemoryPages &pages) noexcept {
  if (!pages.compressor || pages.compressed / 2 >= pages.compressor) return pages.anonymous / 2;
  if (pages.compressed <= pages.compressor) return 0;
  return pages.anonymous -
         static_cast<uint64_t>(static_cast<unsigned __int128>(pages.anonymous) * pages.compressor / pages.compressed);
}

// The host's spare memory above its spare reserve, beside `reservedBytes`
// being allocated: the room growth no request in service needs takes from.
uint64_t spareHeadroom(const HostMemory &host, uint64_t spareReserveBytes, uint64_t reservedBytes) noexcept {
  return saturatingSubtract(saturatingSubtract(host.spare, spareReserveBytes), reservedBytes);
}

// The host's page counts and page size; false when macOS reports neither.
bool sampleHostPages(vm_statistics64_data_t &pages, vm_size_t &pageSize) noexcept {
  mach_port_t host = mach_host_self();
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  const kern_return_t pageResult = host_page_size(host, &pageSize);
  const kern_return_t pagesResult =
      pageResult == KERN_SUCCESS
          ? host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&pages), &count)
          : pageResult;
  mach_port_deallocate(mach_task_self(), host);
  return pagesResult == KERN_SUCCESS && pageSize;
}

// The class whose rule the host applies to a claim: memory a lane in
// service takes in any case (ClaimSpec::laneNeeds) it judges as the lane's
// growth.
ClaimClass hostClass(const ClaimSpec &spec) noexcept { return spec.laneNeeds ? ClaimClass::InService : spec.cls; }

} // namespace

HostMemory estimateHostMemory(const HostMemoryPages &statistics, uint64_t pageSize,
                              bool compression) noexcept {
  // free_count includes the speculative pages, which external_page_count
  // also counts; wired file pages are in neither.
  if (!pageSize || statistics.speculative > statistics.free) return {};
  uint64_t pages = statistics.free - statistics.speculative;
  for (uint64_t reclaimable : {statistics.fileBacked, statistics.purgeable})
    if (!checkedAdd(pages, reclaimable, pages)) return {};
  const uint64_t credit = compression ? compressionSavings(statistics) : 0;
  uint64_t available = 0;
  if (!checkedAdd(pages, credit, available) || !checkedMultiply(available, pageSize, available)) return {};
  return {available, (pages + credit / 2) * pageSize};
}

std::optional<HostMemory> queryHostMemory() noexcept {
  vm_size_t pageSize = 0;
  vm_statistics64_data_t statistics{};
  if (!sampleHostPages(statistics, pageSize)) return std::nullopt;

  return estimateHostMemory(
      {.free = statistics.free_count,
       .speculative = statistics.speculative_count,
       .fileBacked = statistics.external_page_count,
       .purgeable = statistics.purgeable_count,
       .anonymous = statistics.internal_page_count,
       .compressor = statistics.compressor_page_count,
       .compressed = statistics.total_uncompressed_pages_in_compressor},
      pageSize, querySystemMemoryPressure().value_or(MemoryPressure::Critical) != MemoryPressure::Critical);
}

HostTelemetry hostTelemetryOf(const task_vm_info &task, const vm_statistics64 &pages, uint64_t fileCacheMinimumPages,
                              uint64_t pageSize) noexcept {
  return {.physFootprintBytes = task.phys_footprint,
          .compressedBytes = task.compressed,
          .fileBackedBytes = uint64_t{pages.external_page_count} * pageSize,
          .inactiveBytes = (uint64_t{pages.inactive_count} + pages.speculative_count) * pageSize,
          .fileCacheMinimumBytes = fileCacheMinimumPages * pageSize};
}

std::optional<HostTelemetry> queryHostTelemetry() noexcept {
  task_vm_info_data_t task{};
  mach_msg_type_number_t taskCount = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&task), &taskCount) != KERN_SUCCESS)
    return std::nullopt;
  vm_size_t pageSize = 0;
  vm_statistics64_data_t pages{};
  uint32_t fileCacheMinimum = 0;
  size_t size = sizeof(fileCacheMinimum);
  if (!sampleHostPages(pages, pageSize) ||
      sysctlbyname("vm.vm_page_filecache_min", &fileCacheMinimum, &size, nullptr, 0) != 0 ||
      size != sizeof(fileCacheMinimum))
    return std::nullopt;
  return hostTelemetryOf(task, pages, fileCacheMinimum, pageSize);
}

std::optional<MemoryPressure> querySystemMemoryPressure() noexcept {
  uint32_t level = 0;
  size_t size = sizeof(level);
  if (sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &size,
                   nullptr, 0) != 0 || size != sizeof(level))
    return std::nullopt;
  switch (level) {
  case DISPATCH_MEMORYPRESSURE_NORMAL: return MemoryPressure::Normal;
  case DISPATCH_MEMORYPRESSURE_WARN: return MemoryPressure::Warning;
  case DISPATCH_MEMORYPRESSURE_CRITICAL: return MemoryPressure::Critical;
  default: return std::nullopt;
  }
}

MemoryGovernor::Reservation::Reservation(MemoryGovernor *owner, uint64_t bytes)
    : owner_(owner), bytes_(bytes) {}

MemoryGovernor::Reservation::~Reservation() { release(); }

MemoryGovernor::Reservation::Reservation(Reservation &&other) noexcept
    : owner_(other.owner_), bytes_(other.bytes_) {
  other.owner_ = nullptr;
  other.bytes_ = 0;
}

void MemoryGovernor::Reservation::release() noexcept {
  if (owner_ && bytes_)
    owner_->release(bytes_);
  owner_ = nullptr;
  bytes_ = 0;
}

MemoryGovernor::MemoryGovernor(metal::MetalBackend &backend, uint64_t limitBytes,
                               uint64_t hostReserveBytes, uint64_t hostSpareReserveBytes,
                               HostMemoryProvider hostMemory, uint64_t untrackedReserveBytes, uint64_t hostHeldBytes)
    : backend_(backend), limitBytes_(limitBytes), hostReserveBytes_(hostReserveBytes),
      hostSpareReserveBytes_(hostSpareReserveBytes), hostMemory_(std::move(hostMemory)),
      untrackedReserveBytes_(untrackedReserveBytes), hostHeldBytes_(hostHeldBytes) {
  if (!limitBytes_) {
    throw std::invalid_argument("memory governor limit must be positive");
  }
  if (!hostReserveBytes_) {
    throw std::invalid_argument("host memory reserve must be positive");
  }
  if (!hostMemory_) {
    throw std::invalid_argument("host memory provider must be present");
  }
  if (chargedBytes(true) > limitBytes_) {
    throw metal::MetalAllocationError(
        "existing Metal allocations exceed memory governor limit",
        metal::AllocationFailure::EngineBudget);
  }
  const std::optional<HostMemory> host = hostMemory_();
  if (!host || host->available <= hostReserveBytes_) {
    throw metal::MetalAllocationError(
        "host available memory does not satisfy the system reserve",
        metal::AllocationFailure::HostPressure);
  }
}

MemoryGovernor::MemoryGovernor(metal::MetalBackend &backend, const EngineMemoryBreakdown &budget,
                               HostMemoryProvider hostMemory)
    : MemoryGovernor(backend, budget.hardBudgetBytes,
                     EngineMemoryPolicy::hostAvailableReserveBytes(budget.physicalMemoryBytes),
                     EngineMemoryPolicy::hostSpareReserveBytes(budget.physicalMemoryBytes), std::move(hostMemory),
                     budget.pipelineReserveBytes + budget.runtimeOverheadReserveBytes, budget.streamedTableBytes) {}

uint64_t
MemoryGovernor::chargedBytes(bool refreshDevice) const noexcept {
  metal::MetalMemoryStats memory = refreshDevice
      ? backend_.refreshMemoryStats()
      : backend_.memoryStats();
  // The backend's buffers are charged in full, the rest of the device's
  // footprint only where it exceeds the untracked reserve.
  const uint64_t accounted = saturatingAdd(memory.allocatedBytes, untrackedReserveBytes_);
  return saturatingAdd(std::max(accounted, memory.deviceCurrentAllocatedBytes), hostHeldBytes_);
}

std::optional<HostMemory> MemoryGovernor::sampleHost() const noexcept {
  try {
    return hostMemory_();
  } catch (...) {
    return std::nullopt;
  }
}

uint64_t MemoryGovernor::hostHeadroomBytes(const std::optional<HostMemory> &host,
                                           uint64_t reservedBytes) const noexcept {
  return host ? spareHeadroom(*host, hostSpareReserveBytes_, reservedBytes) : 0;
}

uint64_t MemoryGovernor::unseenGrowthBytes(const std::optional<HostMemory> &host) const noexcept {
  if (!host || clockMilliseconds_ >= unseenGrowthUntilMilliseconds_)
    return 0;
  const uint64_t fallen = saturatingSubtract(unseenGrowthHeadroom_, hostHeadroomBytes(host, 0));
  return saturatingSubtract(unseenGrowthBytes_, fallen);
}

std::optional<MemoryGovernor::Reservation>
MemoryGovernor::tryReserve(uint64_t bytes, metal::AllocationFailure &failure) {
  if (!bytes) {
    throw std::invalid_argument("memory reservation must be positive");
  }
  if (!scope_)
    throw std::logic_error("memory was allocated outside a claim's scope");
  const uint64_t observed = chargedBytes(true);
  uint64_t requested = 0;
  const bool overflows = !checkedAdd(reservedBytes_, bytes, requested);
  const std::optional<HostMemory> host = sampleHost();
  sampleHostRule(host);
  const bool engineFits = !overflows && observed <= limitBytes_ && requested <= limitBytes_ - observed;
  // The room claims ranked above the scope's claim hold is not its, and the
  // host admits it by its class.
  const ClaimState &claim = ledger_.claim(scope_);
  const bool fits = engineFits && ledger_.heldBytes(scope_) <= limitBytes_ - observed - requested;
  const ClaimClass cls = hostClass(claim.spec);
  const bool hostAdmits = hostRule_.admits(cls, hostBytes(cls, bytes), claim.hostExempt);
  if (!fits || !hostAdmits) {
    // The host's refusal lifts with its pressure, the limit's only once
    // memory is freed: a refusal they share is the host's.
    failure = hostAdmits ? metal::AllocationFailure::EngineBudget : metal::AllocationFailure::HostPressure;
    if (deniedReservations_ != std::numeric_limits<uint64_t>::max())
      ++deniedReservations_;
    ledger_.refused(scope_, {.bytes = bytes, .hostRefused = !hostAdmits});
    return std::nullopt;
  }
  // A pool's growth reaches the host's counters only as it fills: until
  // they show it, or the host has settled, the host's rule counts what it
  // admitted on its margin.
  if (claim.spec.cls == ClaimClass::Growth && host) {
    unseenGrowthBytes_ = unseenGrowthBytes(host) + bytes;
    unseenGrowthHeadroom_ = hostHeadroomBytes(host, 0);
    unseenGrowthUntilMilliseconds_ = clockMilliseconds_ + kHostSettleMilliseconds;
  }
  reservedBytes_ = requested;
  return Reservation(this, bytes);
}

metal::AllocationAdmission MemoryGovernor::allocationAdmission() noexcept {
  return [this](uint64_t bytes, const std::function<void()> &allocate) -> metal::AllocationResult {
    metal::AllocationFailure failure;
    auto reservation = tryReserve(bytes, failure);
    if (!reservation)
      return failure;
    try {
      allocate();
    } catch (const metal::MetalAllocationError &error) {
      // Host headroom is an estimate; the driver can still deny the allocation.
      return error.failure();
    }
    return {};
  };
}

void MemoryGovernor::setPressure(MemoryPressure pressure) noexcept {
  systemPressure_ = pressure;
}

MemoryGovernorSnapshot MemoryGovernor::snapshot() const noexcept {
  const std::optional<HostMemory> host = sampleHost();
  // The rule at this sample, from the one the governor last moved.
  HostRule rule = hostRule_;
  rule.sample(host, hostSpareReserveBytes_, reservedBytes_ + unseenGrowthBytes(host), systemPressure_,
              growthUnitBytes_, ledger_.hostRefusedWaitingBytes(), bandWaived_);
  MemoryGovernorSnapshot result = snapshot(host, rule);
  result.hostTelemetry = queryHostTelemetry();
  return result;
}

MemoryGovernorSnapshot MemoryGovernor::evaluate() noexcept {
  const std::optional<HostMemory> host = sampleHost();
  growthUnitBytes_ = ledger_.largestGrowthUnit();
  sampleHostRule(host);
  return snapshot(host, hostRule_);
}

MemoryGovernorSnapshot
MemoryGovernor::snapshot(const std::optional<HostMemory> &hostMemory,
                         const HostRule &rule) const noexcept {
  const uint64_t observed = chargedBytes();
  const uint64_t used = saturatingAdd(observed, reservedBytes_);
  uint64_t hostHeadroom = hostHeadroomBytes(hostMemory, reservedBytes_ + unseenGrowthBytes(hostMemory));
  const MemoryPressure pressure = rule.level() == MemoryPressure::Critical ? MemoryPressure::Critical
                                  : rule.constrained()                     ? MemoryPressure::Warning
                                                                           : rule.level();
  const bool hostGrowthAllowed = rule.admits(ClaimClass::Growth, 0, false);
  MemoryGovernorSnapshot snapshot{
      limitBytes_,
      observed,
      saturatingSubtract(limitBytes_, used),
      pressure,
      deniedReservations_,
      hostMemory.has_value(),
      hostMemory ? hostMemory->available : 0,
      hostReserveBytes_,
      hostMemory ? hostMemory->spare : 0,
      hostSpareReserveBytes_,
      hostHeadroom,
      systemPressure_,
      hostGrowthAllowed,
  };
  ledger_.fill(snapshot, observed);
  return snapshot;
}

void MemoryGovernor::release(uint64_t bytes) noexcept { reservedBytes_ = saturatingSubtract(reservedBytes_, bytes); }

MemoryGovernor::Scope::Scope(MemoryGovernor &governor, ClaimId claim) noexcept
    : governor_(governor), previousClaim_(std::exchange(governor.scope_, claim)) {}

MemoryGovernor::Scope::~Scope() { governor_.scope_ = previousClaim_; }

void MemoryGovernor::HostRule::sample(const std::optional<HostMemory> &host, uint64_t spareReserve, uint64_t inFlight,
                                      MemoryPressure system, uint64_t hysteresisBytes, uint64_t refusedWaitingBytes,
                                      bool reclaimShort) noexcept {
  measured_ = host.has_value();
  headroom_ = host ? spareHeadroom(*host, spareReserve, inFlight) : 0;
  system_ = system;
  const uint64_t reach = refusedWaitingBytes ? std::min(hysteresisBytes, refusedWaitingBytes) : hysteresisBytes;
  if (!measured_ || headroom_ < kHostMarginBytes)
    constrained_ = true;
  else if (reclaimShort || headroom_ - kHostMarginBytes >= reach)
    constrained_ = false;
}

bool MemoryGovernor::HostRule::admits(ClaimClass cls, uint64_t bytes, bool hostExempt) const noexcept {
  if (system_ == MemoryPressure::Critical)
    return false;
  if (cls == ClaimClass::Floor || ((cls == ClaimClass::InService || cls == ClaimClass::Growth) && hostExempt))
    return true;
  const bool keepsMargin = measured_ && headroom_ >= bytes && headroom_ - bytes >= kHostMarginBytes;
  if (cls == ClaimClass::InService)
    return keepsMargin && system_ == MemoryPressure::Normal;
  return keepsMargin && system_ == MemoryPressure::Normal && !constrained_;
}

uint64_t MemoryGovernor::HostRule::deficitBytes(uint64_t refusedWaitingBytes, uint64_t returningBytes) const noexcept {
  // Missing telemetry holds elective work back but is no evidence that
  // memory must be given back.
  if (!measured_)
    return 0;
  const uint64_t lacking = saturatingSubtract(saturatingAdd(kHostMarginBytes, refusedWaitingBytes), headroom_);
  return saturatingSubtract(lacking, returningBytes);
}

void MemoryGovernor::sampleHostRule(const std::optional<HostMemory> &host) noexcept {
  hostRule_.sample(host, hostSpareReserveBytes_, reservedBytes_ + unseenGrowthBytes(host), systemPressure_,
                   growthUnitBytes_, ledger_.hostRefusedWaitingBytes(), bandWaived_);
  // A waiver lasts the band it ended: the next band starts unwaived.
  if (!hostRule_.constrained())
    bandWaived_ = false;
}

uint64_t MemoryGovernor::roomBytes() const noexcept {
  return saturatingSubtract(saturatingSubtract(limitBytes_, chargedBytes()), reservedBytes_);
}

Need MemoryGovernor::need(ClaimId claim, const std::optional<Need> &inPlace) const {
  // A refusal records bytes only, so every other field is the caller's.
  const Need &refused = ledger_.claim(claim).refusal;
  Need need = inPlace.value_or(Need{});
  need.bytes = refused.bytes;
  need.hostRefused = refused.hostRefused;
  return need;
}

std::array<uint32_t, 3> MemoryGovernor::heldPages(ClaimId claim) const {
  const std::optional<HostMemory> host = sampleHost();
  const uint64_t headroom = hostHeadroomBytes(host, reservedBytes_ + unseenGrowthBytes(host));
  const uint64_t hostRoom =
      host && systemPressure_ == MemoryPressure::Normal ? saturatingSubtract(headroom, kHostMarginBytes) : 0;
  return ledger_.heldPages(claim, std::min(roomBytes(), hostRoom));
}

FreeResult MemoryGovernor::free(ClaimId claim, const Need &need) {
  return ledger_.free(claim, need, roomBytes(), hostRule_.level() == MemoryPressure::Critical,
                      hostClass(ledger_.claim(claim).spec) == ClaimClass::InService);
}

uint64_t MemoryGovernor::available(const ClaimSpec &spec, const Need &need, const RestoredChains *kept) const {
  return ledger_.available(spec, need, roomBytes(), hostRule_.level() == MemoryPressure::Critical, kept);
}

bool MemoryGovernor::admits(ClaimClass cls) const noexcept { return hostRule_.admits(cls, 0, false); }

bool MemoryGovernor::copyDemand() const noexcept { return ledger_.copyDemand(); }

bool MemoryGovernor::claimWaiting() const noexcept { return ledger_.claimWaiting(); }

// The host's claim of what the pass finds, up to `target`, taken at once:
// the target is net of what lands already (HostRule::deficitBytes). What it
// may take its rights say (takeRights): under macOS's critical pressure
// units in use too, otherwise units that free later. It asks with no room,
// which returns nothing to the host. free() gives back the units that cost
// nothing and are sized only when taken (the idle state buffers) first
// either way. A pass that freed nothing now or later waits (Pending) while
// memory lands if its rights wait for writes, else is Short: it found
// nothing to take.
FreeResult MemoryGovernor::hostPass(uint64_t target, bool critical) {
  const ClaimSpec spec{.cls = ClaimClass::Host};
  const Need need{.bytes = target};
  const uint64_t found = ledger_.available(spec, need, 0, critical);
  const ClaimId claim = ledger_.open(spec);
  FreeResult result = ledger_.free(claim, {.bytes = std::min(target, found)}, 0, critical);
  ledger_.close(claim);
  const bool waits = takeRights(spec, need, critical, std::nullopt).waitsForWrites;
  if (result.outcome == FreeOutcome::Met && !result.nowBytes && !result.laterBytes)
    result.outcome = waits && ledger_.returning().landingBytes ? FreeOutcome::Pending : FreeOutcome::Short;
  return result;
}

uint64_t MemoryGovernor::hostBytes(ClaimClass cls, uint64_t bytes) const noexcept {
  if (cls != ClaimClass::Growth)
    return bytes;
  return saturatingAdd(bytes, ledger_.laneDemandBytes());
}

GrowthHost MemoryGovernor::growthHost(bool paced) {
  // Each unit at a fresh sample, which counts the units grown before it.
  return {.admits =
              [this](uint64_t bytes, uint64_t funded) {
                sampleHostRule(sampleHost());
                // A unit the cache funds whole changes the host by nothing:
                // it needs only no critical pressure and no deficit, under
                // macOS's warning too. The rest is judged as growth.
                if (funded >= bytes)
                  return hostRule_.level() != MemoryPressure::Critical &&
                         !hostRule_.deficitBytes(ledger_.hostRefusedWaitingBytes(), ledger_.returning().landingBytes);
                return hostRule_.admits(ClaimClass::Growth, hostBytes(ClaimClass::Growth, bytes - funded), false);
              },
          .paced = paced,
          .grow = [this](ClaimId claim, model::ElasticPool &pool) {
            // The pass admitted the unit on its net change (admits): its
            // allocation passes the host.
            ledger_.setHostExempt(claim, true);
            const Scope scope(*this, claim);
            return pool.grow(allocationAdmission());
          }};
}

bool MemoryGovernor::controlPass(double nowMilliseconds) {
  clockMilliseconds_ = nowMilliseconds;
  ledger_.setNow(nowMilliseconds / 1000.0);
  growthUnitBytes_ = ledger_.largestGrowthUnit();
  sampleHostRule(sampleHost());
  const bool critical = hostRule_.level() == MemoryPressure::Critical;
  std::optional<FreeResult> pass;
  if (nowMilliseconds >= nextHostPassMilliseconds_) {
    const uint64_t refused = ledger_.hostRefusedWaitingBytes();
    if (critical) {
      // One sized claim per settled second, resampled while critical
      // pressure lasts: the margin and what the host refused, in the one
      // order, the store down to its floor at most.
      pass = hostPass(kHostMarginBytes + refused, true);
      nextHostPassMilliseconds_ = nowMilliseconds + kHostSettleMilliseconds;
    } else if (const uint64_t deficit = hostRule_.deficitBytes(refused, ledger_.returning().landingBytes)) {
      pass = hostPass(std::min(kHostMarginBytes, deficit), false);
    }
  }
  // After the host's claim: growth goes only where the host admits it, so
  // it never competes with what the host takes.
  if (!critical)
    ledger_.grow(roomBytes(), growthHost(true));
  if (pass) {
    // Only a pass while the host is short can waive the band: one that found
    // nothing to take, since only other applications could return the rest.
    if (hostRule_.constrained())
      bandWaived_ = pass->outcome == FreeOutcome::Short;
    // A pass that waits for what lands is judged again once the host has
    // settled too: run again at every command-free point, it reran the
    // whole control pass each loop iteration while a landing lasted.
    if (pass->nowBytes || pass->laterBytes || pass->outcome == FreeOutcome::Pending)
      nextHostPassMilliseconds_ = nowMilliseconds + kHostSettleMilliseconds;
  }
  return pass && pass->outcome == FreeOutcome::Pending;
}

void MemoryGovernor::growToTargets(double nowMilliseconds, MemoryPressure pressure) {
  clockMilliseconds_ = nowMilliseconds;
  ledger_.setNow(nowMilliseconds / 1000.0);
  systemPressure_ = pressure;
  growthUnitBytes_ = ledger_.largestGrowthUnit();
  sampleHostRule(sampleHost());
  ledger_.grow(roomBytes(), growthHost(false));
}

void connectToGovernor(EngineConfig &config, MemoryGovernor &governor) { config.governor = &governor; }

} // namespace splash::engine
