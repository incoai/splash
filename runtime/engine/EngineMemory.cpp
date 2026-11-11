#include "engine/Engine.hpp"
#include "engine/EngineAllocate.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace splash::engine {

bool Engine::holdsBack(std::optional<uint64_t> refused) const {
  if (!refused)
    return false;
  const auto found = requests_.find(*refused);
  return found != requests_.end() && !found->second.finalized && found->second.refusedResources;
}

ResourceWaitSnapshot Engine::resourceWaitSnapshot(double now) const {
  ResourceWaitSnapshot result;
  result.draining = drainingForRecovery();
  for (const auto &[id, active] : requests_) {
    if (active.finalized)
      continue;
    // Admitted, it waits for disk reads, not for memory.
    if (active.restore) {
      ++result.restoring;
      continue;
    }
    const bool waiting = scheduler_.phase(id) == Phase::WaitingResources;
    // Requests the latest admission pass held back, in any phase, while the
    // request they wait behind is still refused.
    if (holdsBack(active.heldBehind))
      ++result.heldBehindRefusal;
    if (!waiting)
      continue;
    if (active.resourceWait.reason == StateFailure::ConcurrencyLimit)
      ++result.concurrency;
    else if (active.resourceWait.diskQuota)
      ++result.diskQuota;
    else
      ++result.memory;
    if (scheduler_.suspended(id))
      ++result.suspended;
    if (active.resourceWait.startedMilliseconds)
      result.oldestWaitMilliseconds = std::max(
          result.oldestWaitMilliseconds, now - *active.resourceWait.startedMilliseconds);
  }
  return result;
}

uint64_t Engine::projectedTokens(const Request &active, uint32_t outputLimit) noexcept {
  const uint64_t generated = active.exactTokens.size() - active.promptTokens;
  const uint64_t remaining = outputLimit - std::min<uint64_t>(generated, outputLimit);
  return active.exactTokens.size() + std::min<uint64_t>(remaining, kProjectedOutputTokens);
}

KvMode Engine::placement(const Request &active) const {
  const EngineSpillConfig &spill = config_.spill;
  if (!spill.capable)
    return KvMode::Resident;
  if (spill.always)
    return KvMode::Spill;
  const uint64_t tokens = projectedTokens(active, active.request.maxNewTokens);
  if (tokens <= spill.denseTokens)
    return KvMode::Resident;
  // Spill never takes more than Resident: its peak is its floor and, while
  // it prefills, the staging of its history, while it decodes, its lane's
  // floor of the hot block cache it reads through, priced as laneDemand
  // prices a waiting lane's: extents no lane reads are rebuild-only memory
  // a Resident start may take as well, so they count for neither.
  const SpillGeometry &geometry = spill.geometry;
  const uint64_t id = active.request.id;
  const auto pages = static_cast<uint32_t>((tokens + KvCache::pageTokens - 1) / KvCache::pageTokens);
  const uint64_t resident = geometry.residentBytes(tokens);
  const uint64_t peak = geometry.spillFloorBytes(tokens) +
                        std::max(model_.stagingShortfall(id, pages), model_.hotFloorShortfall(std::span(&id, 1)));
  if (resident <= peak)
    return KvMode::Resident;
  // Its Resident KV beside what the lanes still grow into, from what a
  // start may take: the room, what lands, the cache, the pools that cost
  // only their rebuild and the releases of Spill lanes of its priority or a
  // lower one (TakeRights::lanes), never the expert store, whose chunks a
  // Spill lane leaves in place.
  return residentRoom(active, claimFor(active, ClaimClass::Start), tokens).holds() ? KvMode::Resident
                                                                                   : KvMode::Spill;
}

Engine::ResidentRoom Engine::residentRoom(const Request &active, const ClaimSpec &spec, uint64_t tokens) const {
  const auto pages = static_cast<uint32_t>((tokens + KvCache::pageTokens - 1) / KvCache::pageTokens);
  // Its pages less the restored chain's pages in memory, which its start
  // takes in place; the claim keeps that chain for the start to reuse.
  // Counted as the claims take it: its KV and the lanes' Full pages from the
  // free pages the walk keeps, then in whole extents, plus what its lane
  // allocates, with what the claim frees counted against the pool it leaves.
  // Alone, that is what the memory plan holds one request (ContextBounds::
  // resident): its KV in whole extents beside its lane's buffers, a warmup's
  // in the pool or allocated now. No state its prefill captures, which takes
  // buffers only where memory has them, or else is written from the lane to
  // the SSD cache or not kept: counted, a Qwen3.8-Flash-Next state's 118.75 MB
  // refused lone starts the plan's context held.
  RestoredChains own;
  uint32_t held = 0;
  if (active.admissionProbe && !active.skipCache) {
    const std::span<const uint64_t> chain = active.admissionProbe->restoredChain();
    own.add(chain);
    held = cache_.residentChainPages(chain);
  }
  const LaneDemand lanes = laneDemand();
  Need need = cache_.growthNeed(pages - std::min(pages, held) + lanes.fullPages);
  const uint64_t beside = model_.startStateBytes() + lanes.bytes;
  need.bytes += beside;
  need.afterStart = true;
  const uint64_t available = governor().available(spec, need, &own);
  // Short of the need, what is there holds the restored chain's pages and
  // those the rest allocates past its state and the lanes' growth.
  uint64_t heldPages = 0;
  if (available >= beside) {
    const uint64_t allocated = cache_.growthPages(available - beside);
    heldPages = held + allocated - std::min<uint64_t>(allocated, lanes.fullPages);
  }
  return {.needed = need.bytes, .available = available, .heldTokens = heldPages * KvCache::pageTokens};
}

uint64_t Engine::laneDemandBytes() const {
  const LaneDemand lanes = laneDemand();
  return uint64_t{lanes.fullPages} * cache_.bytesPerPage(kv::PageForm::Full) + lanes.bytes;
}

Engine::LaneDemand Engine::laneDemand() const {
  const SpillGeometry &geometry = config_.spill.geometry;
  const auto pages = [](uint64_t tokens) { return (tokens + KvCache::pageTokens - 1) / KvCache::pageTokens; };
  LaneDemand demand;
  // The lanes that read spilled KV, those holding Index pages: those
  // without their floor of the hot block cache yet lack it once for them
  // all, in whole extents: floorExtents(2) for two, not 2 x floorExtents(1);
  // and one in prefill the staging of its history through its next command.
  // A Spill lane with none reads nothing spilled.
  std::vector<uint64_t> readers;
  for (const auto &[id, peer] : requests_) {
    if (!peer.lane || peer.pendingEnd)
      continue;
    const bool prefill = scheduler_.phase(id) == Phase::Prefill;
    if (cache_.holdsIndexPages(id)) {
      readers.push_back(id);
      if (prefill)
        demand.bytes += model_.stagingShortfall(id, nextCommandPages(peer));
    }
    const uint64_t projection = projectedTokens(peer);
    const uint64_t held = kvCapacity(id);
    if (peer.kvMode == KvMode::Spill) {
      // Its floor grows, up to its projection. But a prefill command writes
      // its rows to Full pages, which no release returns until their copies
      // land: its next command's pages count Full, as a Resident lane's do,
      // and the floor grows from their end. Priced at the floor, a 225-page
      // command right after a start placed Resident beside the lane took
      // the room the start counted, and the start converted itself to Spill
      // in its prefill. A decode step adds a page as its tail window lets go
      // of one whose copy landed long since, which keeps the floor's count.
      const uint64_t command =
          prefill ? std::max<uint64_t>(nextCommandPages(peer), pages(held)) - pages(held) : 0;
      const uint64_t floor = geometry.spillFloorBytes(held + command * KvCache::pageTokens);
      const uint64_t projected = geometry.spillFloorBytes(projection);
      demand.fullPages += static_cast<uint32_t>(command);
      demand.bytes += projected > floor ? projected - floor : 0;
    } else if (pages(projection) > pages(held)) {
      demand.fullPages += static_cast<uint32_t>(pages(projection) - pages(held));
    }
  }
  demand.bytes += model_.hotFloorShortfall(readers);
  return demand;
}

std::optional<RequestPriority> Engine::highestLanePriority() const {
  std::optional<RequestPriority> highest;
  for (const auto &[_, active] : requests_)
    if (active.lane && !active.finalized && (!highest || active.request.priority < *highest))
      highest = active.request.priority;
  return highest;
}

MemoryGovernor &Engine::governor() const { return *config_.governor; }

std::array<uint32_t, 3> Engine::heldPages(const Request &active) const {
  return active.claim.id ? governor().heldPages(active.claim.id) : std::array<uint32_t, 3>{};
}

void Engine::syncClaims(double now) {
  MemoryLedger &ledger = governor().ledger();
  const std::optional<RequestPriority> tier = suspendedTier();
  const bool draining = drainingForRecovery();
  for (auto &[_, active] : requests_) {
    if (!active.claim.id)
      continue;
    ResourceWait &wait = active.resourceWait;
    const bool retried = active.lane ? !belowRunnable(active) && !wait.passedOver
                                     : admissionTries(active, tier, draining) && !holdsBack(active.heldBehind);
    const double deadline = resourceDeadline(active);
    wait.holdEnded = (wait.pendingSinceMilliseconds &&
                      now >= *wait.pendingSinceMilliseconds + config_.resourceWaitTimeoutMilliseconds) ||
                     (deadline > 0.0 && now >= deadline);
    const bool waiting = !active.finalized && wait.pending && !wait.diskQuota && wait.pendingSinceMilliseconds &&
                         retried && !wait.holdEnded;
    if (ledger.claim(active.claim.id).waiting != waiting)
      ledger.setWaiting(active.claim.id, waiting);
    const bool hostRefused = !active.finalized && !wait.pending &&
                             wait.allocationFailure == metal::AllocationFailure::HostPressure && retried &&
                             (deadline <= 0.0 || now < deadline);
    ledger.setHostRefusedWaiting(active.claim.id, hostRefused);
    // A lane that sent tokens waits in its next decode gap.
    active.gapSince.claimWaited =
        active.gapSince.claimWaited || ((waiting || hostRefused) && active.lastTokenMilliseconds);
  }
}

void Engine::passOver(const BatchPlan &plan, std::span<const uint64_t> excluded) {
  // A lane the plan leaves out while ready to retry waits for its command,
  // not for memory: its claim holds no room the plan's lanes would take
  // more for, and its wait does not read as memory's. A prompt that led by
  // fewer rows left one submitted before it 31 s on its claim.
  const Phase kind = plan.kind == WorkKind::Prefill ? Phase::Prefill : Phase::Decode;
  for (auto &[id, active] : requests_) {
    ResourceWait &wait = active.resourceWait;
    if (!active.lane || !wait.pending || scheduler_.phase(id) != kind ||
        std::ranges::find(excluded, id) != excluded.end())
      continue;
    wait.passedOver = std::ranges::find(plan.items, id, &BatchItem::requestId) == plan.items.end();
    if (wait.passedOver)
      governor().ledger().setWaiting(active.claim.id, false);
  }
}

uint64_t Engine::copyDemandBytes() const {
  const MemoryGovernor &memory = governor();
  if (!memory.copyDemand())
    return 0;
  // The KV pages the lanes' next commands add: a prefill lane's next
  // command's rows, a decoding lane's verify rows.
  uint64_t pages = 0;
  for (const auto &[id, active] : requests_) {
    if (!active.lane || active.pendingEnd)
      continue;
    if (const Phase phase = scheduler_.phase(id); phase != Phase::Prefill && phase != Phase::Decode)
      continue;
    const uint64_t needed = nextCommandPages(active);
    const uint64_t held = kvCapacity(id) / KvCache::pageTokens;
    pages += needed > held ? needed - held : 0;
  }
  // Twice those, so the next command finds them copied too, or, if more,
  // what a pool's growth lacked, one growth unit at most (a 1 GiB store
  // chunk), for the next try: when idle nothing else asks for copies ahead,
  // and the cache a burst left would hold the store at its floor. Waiting
  // claims add nothing: write-behind stops whenever one waits.
  const MemoryLedger &ledger = memory.ledger();
  return std::max(2 * pages * cache_.bytesPerPage(kv::PageForm::Full),
                  std::min(ledger.growthLackBytes(), ledger.largestGrowthUnit()));
}

uint32_t Engine::nextCommandPages(const Request &active) const {
  const uint64_t id = active.request.id;
  const uint64_t end =
      scheduler_.phase(id) == Phase::Prefill
          ? std::min<uint64_t>(scheduler_.promptProcessed(id) + model_.commandLimits().commandRows, active.promptTokens)
          : active.exactTokens.size() + model::ExecutionLimits::targetVerifyRows;
  return static_cast<uint32_t>((end + KvCache::pageTokens - 1) / KvCache::pageTokens);
}

uint64_t Engine::readsShortfall(std::span<const uint64_t> lanes) const {
  uint64_t bytes = 0;
  std::vector<uint64_t> decoding;
  for (const uint64_t id : lanes) {
    if (cache_.holdsIndexPages(id))
      continue;
    if (scheduler_.phase(id) == Phase::Prefill)
      bytes += model_.stagingShortfall(id, nextCommandPages(requests_.at(id)));
    else
      decoding.push_back(id);
  }
  return bytes + model_.hotFloorShortfall(decoding);
}

bool Engine::fitsResidentAlone(const Request &active) const {
  return projectedTokens(active) <= config_.spill.geometry.residentTokens;
}

uint64_t Engine::chainEnd(const CacheLookup &lookup) noexcept {
  return lookup.state ? lookup.state->kvBlock() : 0;
}

uint64_t Engine::chainEnd(const Request &active) const {
  const uint64_t id = active.request.id;
  const uint32_t held = scheduler_.phase(id) == Phase::Prefill ? scheduler_.promptProcessed(id)
                                                                : active.publishedTokens;
  return held < KvCache::pageTokens
             ? 0
             : cache_.blockAt(id, held / KvCache::pageTokens * KvCache::pageTokens);
}

uint32_t Engine::spillPages(const Request &active, uint32_t copied) const {
  const uint64_t pages = projectedTokens(active) / KvCache::pageTokens;
  return static_cast<uint32_t>(pages - std::min<uint64_t>(pages, copied));
}

Engine::InMemory Engine::inMemory(const Request &active, const ClaimSpec &spec) const {
  if (!fitsResidentAlone(active))
    return {};
  const ResidentRoom room = residentRoom(active, spec, projectedTokens(active));
  if (room.holds())
    return {.admission = SpillAdmission::Resident};
  if (spec.cls == ClaimClass::Start)
    return {.admission = SpillAdmission::Waiting};
  // Alone, the KV memory holds bounds its output where it holds a token past
  // the prompt and the rows past it that the decode step verifies
  // (workEndOf): a lane of T tokens takes the KV of T + 7 at its last step.
  const uint64_t rows = model::ExecutionLimits::targetVerifyRows - 1;
  if (room.heldTokens > active.exactTokens.size() + rows)
    return {.admission = SpillAdmission::Resident, .tokens = room.heldTokens - rows};
  return {.room = residentRoom(active, spec, active.exactTokens.size() + rows + 1)};
}

Engine::SpillAdmission Engine::admitSpill(Request &active, bool resuming, uint64_t chainEnd, const ClaimSpec &spec,
                                          double now) {
  const SpillGeometry &geometry = config_.spill.geometry;
  // It goes on as memory says: Resident, under memory's bound where memory
  // sets one, or waiting for the lanes in service.
  const auto goInMemory = [&](const InMemory &memory) {
    if (*memory.admission == SpillAdmission::Waiting) {
      waitForResources(active, now, {.projection = true});
    } else {
      active.kvMode = KvMode::Resident;
      if (memory.tokens)
        active.outputLimit = static_cast<uint32_t>(
            std::min<uint64_t>(active.outputLimit, *memory.tokens - active.promptTokens));
    }
    return *memory.admission;
  };
  // A start refused names what memory does not hold of it, where memory was
  // asked.
  const auto shortfall = [&](const InMemory &memory) {
    return memory.room ? "; " + memoryShortfall(active, memory.room) : std::string();
  };
  // A resume goes on spilling whatever the tier and the write budget: its
  // writes are in use, and pages a closed tier does not take stay Full.
  if (!resuming) {
    if (const SpillGate gate = cache_.spillGate(); gate != SpillGate::Open) {
      const InMemory memory = inMemory(active, spec);
      if (memory.admission)
        return goInMemory(memory);
      const std::string reason = "requests past " + std::to_string(geometry.residentTokens) +
                                 " tokens keep their KV on the SSD cache, ";
      if (gate == SpillGate::WriteBudget) {
        ++counters_.spill.refusedWriteBudget;
        // It refills over hours: a retry may succeed once it covers again
        // what Spill requests committed, or sooner as they end.
        const double admitted = cache_.spillWritesAdmittedAt().value_or(now);
        settle(active, {LaneOutcome::KvWriteBudget,
                        reason + "whose daily write budget (--max-cache-disk-writes) is spent" + shortfall(memory),
                        static_cast<uint32_t>(std::lround(std::max(0.0, admitted - now) / 1000.0))});
      } else {
        ++counters_.spill.refusedClosed;
        settle(active, {LaneOutcome::CapacityExhausted, reason + "which takes no writes" + shortfall(memory)});
      }
      return SpillAdmission::Settled;
    }
  }
  // The copies on the chain it restores that no request holds count in the
  // quota once it holds them, as its commitment does; the others count
  // already (Cache::chainCopies).
  const Cache::ChainCopies copies = cache_.chainCopies(chainEnd);
  const uint64_t room = cache_.spillRoom() / geometry.slotBytes;
  if (room >= uint64_t{spillPages(active, copies.copied)} + copies.unheld)
    return SpillAdmission::Commit;
  // Short of its projection, a start goes on in memory where memory allows,
  // as without the SSD cache, unless every request spills.
  InMemory memory;
  if (!resuming && !config_.spill.always) {
    memory = inMemory(active, spec);
    if (memory.admission)
      return goInMemory(memory);
  }
  // Otherwise the quota bounds its output to the pages it holds, if they
  // hold a token past the request's tokens; pages of the restored chain
  // whose copies the quota counts already need none of it. A resume counts
  // the anchor its model holds once it selected the first token
  // (Request::suspendedWithAnchor), which the model decodes only with output
  // left past it: it emits a last budgeted anchor as it selects it.
  const uint64_t held = active.exactTokens.size() + (active.suspendedWithAnchor ? 1 : 0);
  const uint64_t counted = copies.copied - copies.unheld;
  const uint64_t needed = held / KvCache::pageTokens + 1 - counted;
  if (room >= needed) {
    active.outputLimit = static_cast<uint32_t>(std::min<uint64_t>(
        active.outputLimit, (counted + room) * KvCache::pageTokens - active.promptTokens));
    return SpillAdmission::Commit;
  }
  // What it lacks is other requests' active KV, the copies of the blocks
  // they hold and their commitments: it waits for them to end. The quota
  // cannot hold it even alone: a resume with output ends at its length, and
  // a start, or a resume with no output to return, is refused.
  const uint64_t activeLimitBytes = cache_.spillLimit();
  if (activeLimitBytes / geometry.slotBytes >= needed) {
    waitForResources(active, now, {.diskQuota = true});
    return SpillAdmission::Waiting;
  }
  if (resuming && generatedTokens(active)) {
    endAtLength(active);
  } else {
    ++counters_.spill.refusedQuota;
    settle(active, {LaneOutcome::KvDiskQuota,
                    "this request's KV needs needed_bytes=" +
                        std::to_string(needed * geometry.slotBytes) +
                        " of the SSD cache, which may hold active_limit_bytes=" +
                        std::to_string(activeLimitBytes) +
                        " of active KV now (--max-cache-disk and the volume's free space)" + shortfall(memory)});
  }
  return SpillAdmission::Settled;
}

void Engine::waitForResources(Request &active, double now, const Denial &denial) {
  active.refusedResources = true;
  scheduler_.waitForResources(active.request.id);
  deferResourceRetry(active, now, denial);
  const std::optional<RequestPriority> highest = highestLanePriority();
  if (!scheduler_.suspended(active.request.id) && !denial.pending && !denial.diskQuota &&
      (!denial.projection || (highest && *highest > active.request.priority)))
    static_cast<void>(preemptBelow(active.request.priority, now));
}

bool Engine::competesForQuota(const Request &active) const {
  if (active.lane)
    return cache_.holdsActiveKvQuota(active.request.id);
  return (active.placement && *active.placement == KvMode::Spill) ||
         (active.admissionProbe && !active.skipCache &&
          cache_.restoreTakesQuota(active.admissionProbe->restoredChain()));
}

bool Engine::convertToSpill(Request &active) {
  if (!config_.spill.capable || active.kvMode != KvMode::Resident || passCommand_.holds(active.request.id))
    return false;
  if (!cache_.convertToSpill(active.request.id, spillPages(active, cache_.chainCopies(chainEnd(active)).copied)))
    return false;
  active.kvMode = KvMode::Spill;
  active.converted = true;
  ++counters_.spill.converted;
  return true;
}

bool Engine::spillingPrompt() const {
  return std::any_of(requests_.begin(), requests_.end(), [&](const auto &entry) {
    const auto &[id, active] = entry;
    return active.lane && active.kvMode == KvMode::Spill && scheduler_.phase(id) == Phase::Prefill;
  });
}

bool Engine::laneInService() const {
  return std::any_of(requests_.begin(), requests_.end(), [](const auto &entry) { return entry.second.lane.has_value(); });
}

SpilledExtents Engine::landedExtents() const {
  // Under --kv-spill always a converted lane's moves release the Full extents
  // they empty: its conversion was for those bytes, which its hot block floor
  // needs. So do all moves under critical pressure, which refuses even the
  // floor: only those bytes pay for the Index extents the next pages move into
  // (MemoryGovernor::fund), and the host's pass takes a kept one anyway.
  const bool converting = std::ranges::any_of(
      requests_, [](const auto &entry) { return entry.second.lane && entry.second.converted; });
  const bool critical = !governor().admits(ClaimClass::Floor);
  return !converting && !critical && (spillingPrompt() || !governor().admits(ClaimClass::Start))
             ? SpilledExtents::KeepRunway
             : SpilledExtents::Released;
}

void Engine::spillLanded(bool controlPass) {
  if (!config_.spill.capable)
    return;
  // The Index runway the pages draw on refills if memory allows. Index
  // pages, the runway's included, are memory the resident Spill lanes need,
  // like their Full pages, and a fraction of what the moves free, which
  // pays for them past the engine's limit and the host
  // (MemoryGovernor::fund): with no such lane
  // in service they take only what costs nothing; with one, they are the
  // highest-ranked such lane's, at its priority, since the room its waiting
  // claim holds is the room its released pages move into.
  const SpilledExtents extents = landedExtents();
  const Request *needed = nullptr;
  for (const auto &[_, active] : requests_)
    if (active.lane && active.kvMode == KvMode::Spill && (!needed || rankOf(active) > rankOf(*needed)))
      needed = &active;
  if (!needed && !controlPass)
    return;
  const auto fund = [&](uint64_t bytes) { governor().fund(bytes); };
  static_cast<void>(allocate([&] { return cache_.spillWritten(extents, fund); },
                             {.cls = needed ? ClaimClass::InService : ClaimClass::Optional,
                              .rank = needed ? rankOf(*needed) : ClaimRank{},
                              .timing = NeedTiming::Immediate,
                              .priority = needed ? needed->request.priority : RequestPriority::Foreground},
                             Refusal::Settled));
}

void Engine::endAtLength(Request &active) {
  ++counters_.spill.lengthEnds;
  counters_.spill.convertedLengthEnds += active.converted;
  finish(active, EngineFinishReason::Length, {});
}

bool Engine::resourceRetryReady(const Request &active,
                                double now) const noexcept {
  return active.resourceWait.retryMilliseconds <= 0.0 ||
         active.resourceWait.epoch != resourceEpoch_ ||
         now >= active.resourceWait.retryMilliseconds;
}

void Engine::deferResourceRetry(Request &active, double now,
                                const Denial &denial, StateFailure reason) noexcept {
  auto &wait = active.resourceWait;
  if (!wait.startedMilliseconds)
    wait.startedMilliseconds = now;
  // Waiting for a lane is not waiting for memory. Without a lane, refused at
  // its start or resume or suspended, a request's wait counts the lanes
  // admitted before its latest refusal, its latest retry's for a start or a
  // resume, that compete for what it was refused (tick): their ends may
  // return what it waits for. A lane in service counts those admitted before
  // its first refusal, not the starts admitted after it.
  if (reason != StateFailure::ConcurrencyLimit && (!active.lane || !wait.admittedBefore))
    wait.admittedBefore = admissions_;
  noteRefusal(active, now, denial);
  wait.reason = reason;
  if (reason == StateFailure::ConcurrencyLimit)
    wait.deadlineMilliseconds = 0.0;
  else if (wait.deadlineMilliseconds <= 0.0)
    wait.deadlineMilliseconds = now + config_.resourceWaitTimeoutMilliseconds;
  wait.retryMilliseconds = now + kResourceRetryBackoffMilliseconds;
}

void Engine::noteRefusal(Request &active, double now, const Denial &denial) noexcept {
  auto &wait = active.resourceWait;
  if (wait.pending && wait.epoch != resourceEpoch_ && wait.deadlineMilliseconds > 0.0)
    wait.deadlineMilliseconds = now + config_.resourceWaitTimeoutMilliseconds;
  if (denial.pending && !wait.startedMilliseconds)
    wait.startedMilliseconds = now;
  if (denial.pending && !wait.pendingSinceMilliseconds)
    wait.pendingSinceMilliseconds = now;
  wait.allocationFailure = denial.allocationFailure;
  wait.pending = denial.pending;
  wait.diskQuota = denial.diskQuota;
  wait.epoch = resourceEpoch_;
}

void Engine::deferWait(Request &active) {
  ResourceWait kept;
  // A start refused resources keeps when its wait began, what refused it,
  // and when its claim first waited for memory on its way back, which bounds
  // what that claim holds.
  if (active.refusedResources) {
    kept.startedMilliseconds = active.resourceWait.startedMilliseconds;
    kept.diskQuota = active.resourceWait.diskQuota;
    kept.pendingSinceMilliseconds = active.resourceWait.pendingSinceMilliseconds;
    kept.holdEnded = active.resourceWait.holdEnded;
    kept.admittedBefore = active.resourceWait.admittedBefore;
  }
  active.resourceWait = kept;
  // Its claim holds no room from this pass on: the starts admission tries
  // after it in the same pass may take that room.
  if (active.claim.id)
    governor().ledger().setWaiting(active.claim.id, false);
}

double Engine::resourceDeadline(const Request &active) const noexcept {
  const ResourceWait &wait = active.resourceWait;
  if ((wait.pending && wait.epoch != resourceEpoch_) || wait.deadlineMilliseconds <= 0.0)
    return 0.0;
  // The limit restarts whenever a lane submitted before the request, or
  // admitted before the refusal ResourceWait::admittedBefore records and
  // competing for what it was refused, works, however long that takes
  // (tick). It fires once no such lane has worked for a whole limit; the
  // request's own deadline applies regardless.
  return std::max(wait.deadlineMilliseconds,
                  wait.earlierLaneWorkMilliseconds + config_.resourceWaitTimeoutMilliseconds);
}

void Engine::signalResourceProgress() noexcept {
  if (resourceEpoch_ != std::numeric_limits<uint64_t>::max())
    ++resourceEpoch_;
}

std::optional<Engine::Denied> Engine::growLane(Request &active, uint64_t workEnd, const PlanContext &plan) {
  const uint64_t id = active.request.id;
  ClaimSpec spec = claimFor(active, ClaimClass::InService);
  spec.supersedes = plan.supersedes;
  spec.kvGrowth = true;
  const auto kv = allocate(
      [&] { return cache_.ensureTokens(id, workEnd, std::numeric_limits<uint64_t>::max(), heldPages(active)); }, spec,
      Refusal::Waits);
  if (kv.admission.granted())
    return std::nullopt;
  return Denied{id, workEnd, kv.denial, kv.admission, {}};
}

std::optional<Engine::Denied> Engine::reserveReads(Request &active, const BatchItem &scheduled,
                                                   const PlanContext &plan) {
  const uint64_t id = active.request.id;
  const auto pages = static_cast<uint32_t>(
      (uint64_t{scheduled.promptOffset} + scheduled.tokenCount + KvCache::pageTokens - 1) /
      KvCache::pageTokens);
  const model::SpilledReadNeed need{plan.prefill ? WorkKind::Prefill : WorkKind::Decode, pages};
  ClaimSpec spec = claimFor(active, ClaimClass::InService);
  spec.supersedes = plan.supersedes;
  const auto memory = allocate([&] { return model_.reserveSpilledReads(id, need); }, spec, Refusal::Waits);
  if (memory.admission.granted())
    return std::nullopt;
  return Denied{id, workEndOf(active, scheduled, plan.prefill), memory.denial, std::nullopt,
                plan.prefill ? "prefill history staging" : "hot KV cache"};
}

Engine::Prepared Engine::refuse(std::span<const Denied> denied, const PlanContext &plan) {
  const bool spill = config_.spill.capable;
  // Partial admissions execute at their actual width. If no lane fits, the
  // lane to yield is chosen among all runnable residents: an unstarted peer
  // can release its lane before completed prefill is discarded.
  if (denied.empty())
    throw std::logic_error("empty resource admission result");
  // No lane runs: the copies Spill requests owe and have not started (a
  // restored chain's before its first command, or those the control pass
  // left when it collected the landed ones) start now, and the lanes wait.
  // Their next attempt finds the copies in flight, which a claim alone waits
  // for, and releases their pages once they land (EngineVictims::releases).
  const bool written = spill && cache_.writeThrough() != 0;
  // Memory on its way back arrives without anyone yielding. The lanes still
  // take a retry deadline: the transfer's completion wakes the engine, and
  // the deadline is what makes the wait end if that wake is ever missed.
  const auto wait = [&] {
    for (const Denied &entry : denied)
      deferResourceRetry(request(entry.requestId), plan.now, entry.denial);
    return Prepared::Waiting;
  };
  if (written || std::any_of(denied.begin(), denied.end(), [](const Denied &entry) { return entry.denial.pending; }))
    return wait();
  const Denied &victim = *std::min_element(
      denied.begin(), denied.end(),
      [&](const Denied &left, const Denied &right) {
        return yieldsBefore(request(left.requestId), request(right.requestId));
      });
  Request *const yielding = laneToYield();
  if (!yielding)
    throw std::logic_error("no resident lane can yield");
  Request &active = *yielding;
  const auto own = std::find_if(denied.begin(), denied.end(), [&](const Denied &entry) {
    return entry.requestId == active.request.id;
  });
  // A peer that has not failed a growth attempt resumes to its current KV
  // capacity, not to the blocked lane's requirement.
  const uint64_t resumeTarget =
      own != denied.end() ? own->workEnd : kvCapacity(active.request.id);
  switch (judge(victim.denial, active.request.id)) {
  case Verdict::Wait:
    return wait();
  case Verdict::Yield:
    return yieldLane(active, resumeTarget, victim.denial.allocationFailure, plan.now);
  case Verdict::Fail:
    break;
  }
  // Alone with nothing on its way back. A Spill lane that decodes ends at
  // its length once its KV gives no more memory back: the SSD cache takes
  // no more of it, or a release down to its floor does not cover what it
  // lacks; any other lane fails. A Spill lane, whose KV alone goes to the
  // SSD cache, names which: the SSD cache where it owes copies no write
  // started, for the quota or the tier; otherwise the release.
  Request &lone = request(victim.requestId);
  std::string spilled;
  if (lone.kvMode == KvMode::Spill)
    spilled = cache_.owesCopies(victim.requestId) ? "the SSD cache took no more of its KV"
                                                  : "even with its KV released down to its floor";
  if (!victim.kv)
    settle(lone, capacityExhausted(lone, victim.memory, victim.denial.allocationFailure, spilled));
  else if (lone.kvMode == KvMode::Spill && !plan.prefill)
    endAtLength(lone);
  else
    settle(lone, capacityExhausted(lone, "KV target", victim.denial.allocationFailure,
                                   pageShortfall(*victim.kv) + (spilled.empty() ? "" : "; " + spilled)));
  return Prepared::Yielded;
}

ReclaimClass Engine::growthClass(const Request &active) const {
  return &active == laneToYield() && anotherResident(active.request.id) ? ReclaimClass::Ordinary
                                                                        : ReclaimClass::InUse;
}

uint32_t Engine::reserveCaptures(Request &active, const BatchItem &item,
                                 std::vector<StateCapture> &states,
                                 std::vector<LaneCapture> &captures, bool supersedes) {
  const uint32_t start = item.promptOffset;
  // A request waiting for a prefix starts once the command that reaches its
  // state has ended.
  const uint32_t end = awaitedBoundary(active, start + item.tokenCount);
  const std::vector<Request::StateBoundary> &boundaries = active.stateBoundaries;
  const auto stateClass = [&](size_t index) {
    return boundaries[index].disposable                               ? StateClass::Checkpoint
           : boundaries[index].tokens == promptReplayBoundary(active) ? StateClass::InUse
                                                                      : StateClass::Ordinary;
  };
  // Takes the buffers of the state at boundaries[index]; false when refused.
  // Only the command's first state follows the lane's checkpoint, its
  // recovery point; each later one follows the state before it, since a
  // checkpoint the command passes is no recovery point.
  const auto reserve = [&](size_t index) {
    const uint32_t tokens = boundaries[index].tokens;
    const StateClass kind = stateClass(index);
    if (kind == StateClass::InUse && !states.empty() && boundaries[index - 1].disposable) {
      // The prompt's replay point keeps the checkpoint captured before it
      // beside buffers of its own, in its claim. Refused the pool and the
      // admission, it takes that checkpoint's buffers, as it takes its
      // lane's after a command: nothing is freed for it.
      std::shared_ptr<StateReservation> buffers;
      static_cast<void>(allocate(
          [&] {
            metal::AllocationFailure refusal = metal::AllocationFailure::None;
            buffers = model_.reserveState(active.request.id, tokens, refusal);
            return StateBuffers{.ok = buffers != nullptr, .allocationFailure = refusal};
          },
          stateClaim(active, kind, supersedes), Refusal::SettledOnce));
      if (buffers) {
        states.push_back({tokens, std::move(buffers)});
        captures.push_back({tokens});
      } else {
        states.back().tokens = tokens;
        captures.back().tokens = tokens;
      }
      return true;
    }
    Reservation reservation = reserveCapture(active, kind, tokens, states.empty(), supersedes);
    if (!reservation.buffers) {
      if (reservation.fromLane)
        captures.push_back({tokens, false, true});
      return false;
    }
    states.push_back({tokens, std::move(reservation.buffers)});
    captures.push_back({tokens, reservation.recycled});
    return true;
  };
  for (size_t index = active.stateBoundaryCursor;
       index < boundaries.size() && boundaries[index].tokens <= end; ++index) {
    const uint32_t tokens = boundaries[index].tokens;
    // Published, every state but the prompt's replay point retires the
    // checkpoint before it (apply): a checkpoint the command passes on its
    // way to such a state is not captured, and that state takes buffers of
    // its own class.
    const bool passed = boundaries[index].disposable && index + 1 < boundaries.size() &&
                        boundaries[index + 1].tokens <= end &&
                        stateClass(index + 1) != StateClass::InUse;
    if (!passed && !reserve(index))
      return tokens - start;
    if (tokens == end)
      break;
    // The KV up to the next boundary, as prepare() takes the first. Its
    // refusal only ends the command here, capturing a checkpoint it would
    // have passed, and the lane's next command asks again: an Optional
    // claim. The lane takes that KV anyway, so the host judges it as the
    // lane's growth: refused with nothing in place to meet it, it passes the
    // host and does not end this command (ClaimSpec::laneNeeds). So too it
    // releases the lane's own landed pages, which that next command would
    // release: a cut costs a pass more of the routed experts where they
    // stream.
    const uint32_t next = index + 1 < boundaries.size() && boundaries[index + 1].tokens < end
                              ? boundaries[index + 1].tokens
                              : end;
    const auto kv = allocate(
        [&] {
          return cache_.ensureTokens(active.request.id, next, std::numeric_limits<uint64_t>::max(),
                                     heldPages(active));
        },
        {.cls = ClaimClass::Optional,
         .requestId = active.request.id,
         .rank = rankOf(active),
         .timing = NeedTiming::Immediate,
         .supersedes = supersedes,
         .priority = active.request.priority,
         .laneNeeds = true},
        Refusal::Settled);
    if (!kv.admission.granted()) {
      if (passed)
        static_cast<void>(reserve(index));
      return tokens - start;
    }
  }
  return end - start;
}

ClaimSpec Engine::stateClaim(const Request &active, StateClass kind, bool supersedes) const {
  ClaimSpec spec = claimFor(active, kind == StateClass::InUse ? ClaimClass::InService : ClaimClass::Optional);
  spec.supersedes = supersedes;
  switch (kind) {
  case StateClass::InUse:
    // The point its next turn resumes from may wait for memory on its way
    // back, and takes what is in use last.
    spec.reach = ReclaimClass::InUse;
    break;
  case StateClass::Checkpoint:
  case StateClass::Ordinary:
    // Its work goes on without it: it takes what frees memory at once.
    spec.timing = NeedTiming::Immediate;
    break;
  }
  return spec;
}

Engine::Reservation Engine::reserveCapture(Request &active, StateClass kind, uint32_t tokens,
                                           bool followsCheckpoint, bool supersedes) {
  const uint64_t id = active.request.id;
  // The lane's checkpoint is its recovery point until a state that retires
  // it is published after a completed command (a cancelled, expired or
  // failed one leaves it to a retry). So no state takes its buffers before
  // the command, nor room a reclaim could make of it (checkpoints go first).
  // Refused the pool and the admission while the lane holds one, the state
  // ends its command, and apply then takes it from the lane: a junction or
  // checkpoint in the buffers of the checkpoint it retires
  // (publishReachedStateBoundaries). A checkpoint whose predecessor a reader
  // pins takes no room either: apply publishes it from the lane only once
  // the pin has gone, keeping the predecessor the recovery point until then.
  // Otherwise room comes from what the ledger frees for the state's claim
  // (stateClaim).
  const bool predecessorPinned =
      kind == StateClass::Checkpoint && followsCheckpoint && checkpointPinned(active);
  const bool fromLane = predecessorPinned || checkpointReclaimable(active);
  Reservation result;
  const auto attempt = [&] {
    metal::AllocationFailure refusal = metal::AllocationFailure::None;
    result.buffers = model_.reserveState(id, tokens, refusal);
    return StateBuffers{.ok = result.buffers != nullptr, .allocationFailure = refusal};
  };
  const ClaimSpec spec = stateClaim(active, kind, supersedes);
  const auto taken = allocate(attempt, spec, fromLane ? Refusal::SettledOnce : Refusal::Settled);
  result.fromLane = !result.buffers && fromLane;
  result.recycled = result.buffers && taken.freed;
  return result;
}

Engine::ClaimHandle::ClaimHandle(ClaimHandle &&other) noexcept
    : governor(other.governor), id(std::exchange(other.id, {})) {}

Engine::ClaimHandle::~ClaimHandle() { reset(); }

void Engine::ClaimHandle::reset() noexcept {
  if (id)
    governor->ledger().close(id);
  id = {};
}

ClaimRank Engine::rankOf(const Request &active) const {
  const Phase phase = scheduler_.phase(active.request.id);
  return {.priority = static_cast<uint8_t>(static_cast<uint8_t>(RequestPriority::Background) -
                                           static_cast<uint8_t>(active.request.priority)),
          // At equal priority, prefer uninterrupted streaming over less
          // replay work.
          .decoding = phase == Phase::Decode || phase == Phase::WaitingMask,
          .completedTokens = completedTokens(active),
          .earliness = std::numeric_limits<uint64_t>::max() - scheduler_.submissionOrder(active.request.id)};
}

ClaimSpec Engine::claimFor(const Request &active, ClaimClass cls) const {
  return {.cls = cls,
          .requestId = active.request.id,
          .rank = rankOf(active),
          .reach = cls == ClaimClass::InService ? growthClass(active) : ReclaimClass::Ordinary,
          .priority = active.request.priority};
}

bool Engine::yieldsBefore(const Request &a, const Request &b) const {
  const ClaimRank left = rankOf(a);
  const ClaimRank right = rankOf(b);
  return std::tie(left.priority, left.decoding, left.completedTokens) <
         std::tie(right.priority, right.decoding, right.completedTokens);
}

const Engine::Request *Engine::laneToYield() const {
  const Request *yielding = nullptr;
  for (const auto &[id, candidate] : requests_) {
    if (!candidate.lane)
      continue;
    // Requests enter the scheduler before they can acquire a lane.
    const Phase phase = scheduler_.phase(id);
    if (phase != Phase::Prefill && phase != Phase::Decode)
      continue;
    if (!yielding || rankOf(candidate) < rankOf(*yielding))
      yielding = &candidate;
  }
  return yielding;
}

uint64_t Engine::kvCapacity(uint64_t requestId) const {
  return uint64_t{cache_.pageTable(requestId).pages.size()} * KvCache::pageTokens;
}

bool Engine::preemptBelow(RequestPriority priority, double now) {
  // The lane that yields first has the lowest priority of all residents.
  Request *const victim = laneToYield();
  if (!victim || victim->request.priority <= priority)
    return false;
  if (yieldsEndedLanes(*victim))
    return true;
  // A lane of the command between its passes is of a lower priority than
  // the start: its suspension gives the command up, as the start's prompt
  // would (Scheduler::yieldsTo).
  suspendLane(*victim, kvCapacity(victim->request.id), metal::AllocationFailure::None,
              StateFailure::ConcurrencyLimit, now);
  // What the lane held is free: an earlier failure may fit now.
  allocationFailed_ = false;
  ++counters_.prioritySuspensions;
  signalResourceProgress();
  return true;
}

bool Engine::anotherResident(uint64_t requestId) const {
  return std::any_of(requests_.begin(), requests_.end(), [&](const auto &entry) {
    return entry.first != requestId && entry.second.lane;
  });
}

Engine::Verdict Engine::judge(const Denial &denial, uint64_t requestId) const {
  if (denial.pending)
    return Verdict::Wait;
  // Pages held by resident lanes come back when they finish, and memory the
  // host refuses when its pressure lifts. Alone, its claim took what the
  // order gives (ClaimSpec::lone): it fails only once no write is in flight
  // either, nothing else holding memory it could get.
  if (anotherResident(requestId) ||
      denial.allocationFailure == metal::AllocationFailure::HostPressure)
    return Verdict::Yield;
  return cache_.writesInFlight() ? Verdict::Wait : Verdict::Fail;
}

ReleaseRule Engine::EngineVictims::releaseRule(const VictimQuery &query) const {
  const Engine &engine = engine_;
  // A command between its passes reads its lanes' history as its first pass
  // cut it.
  const auto releasable = [&engine, &query](uint64_t id) {
    const auto found = engine.requests_.find(id);
    if (found == engine.requests_.end())
      return false;
    const Request &active = found->second;
    return active.lane && !active.restore && !active.finalized && !active.pendingEnd &&
           active.kvMode == KvMode::Spill && !engine.passCommand_.holds(id) &&
           query.reaches(id, active.request.priority);
  };
  // A Resident start reads its history Full, so its count as Resident
  // (VictimQuery::kept, Engine::residentRoom) and its own claims hold the
  // chain its admission probe restores in place.
  ReleaseRule rule{.takesFrom = releasable};
  const auto claimant = engine.requests_.find(query.requestId);
  if (claimant != engine.requests_.end() && claimant->second.admissionProbe && !claimant->second.skipCache &&
      (query.kept || claimant->second.kvMode == KvMode::Resident))
    rule.kept = claimant->second.admissionProbe->restoredChain();
  return rule;
}

std::vector<Victim> Engine::EngineVictims::releases(const VictimQuery &query) const {
  const Engine &engine = engine_;
  std::vector<Victim> units;
  if (query.rights.lanes == TakeRights::Lanes::None || !engine.config_.spill.capable || engine.pending_)
    return units;
  // The lanes the claim releases, the lowest-ranked first.
  const ReleaseRule rule = releaseRule(query);
  std::vector<const Request *> lanes;
  for (const auto &[id, active] : engine.requests_)
    if (rule.takesFrom(id))
      lanes.push_back(&active);
  std::ranges::sort(lanes, [&](const Request *left, const Request *right) {
    return engine.rankOf(*left) < engine.rankOf(*right);
  });
  const uint64_t pageBytes = engine.config_.spill.geometry.releasedPageBytes();
  // A lane's unit for a need of bytes, of `pages` pages: whole extents of
  // them, and an extent more where the free Full pages beside complete it
  // (Cache::releaseBeside), each page of those extents freeing its Full
  // bytes less its Index page's, less the first reads of the lanes they give
  // Index pages: a growth's extent takes two extents of pages. They free
  // only as many Full extents as compaction can empty, where the first
  // runway's moves empty one (Cache::releaseMoves). A need the host refused
  // lacks no bytes. These units add up as their takes in turn free them,
  // each moving its lane's oldest pages, so each counts only the pages,
  // first reads, extents and free pages beside those before it leave: a
  // page lanes share counts once (ReleaseCount).
  const bool forBytes = query.need.bytes && !query.need.hostRefused;
  ReleaseCount count{.beside = forBytes ? engine.cache_.releaseBeside(query) : 0};
  const uint64_t extentPages = engine.cache_.extentPages();
  const auto offer = [&](uint64_t id, uint32_t pages, uint64_t reads, bool landing) {
    const Cache::ReleaseMoves moves =
        forBytes ? engine.cache_.releaseMoves(pages, landing, count) : Cache::ReleaseMoves{};
    const Victim unit{.key = id,
                      .bytes = saturatingSubtract(moves.extents * extentPages * pageBytes, reads),
                      .form = kv::PageForm::Full,
                      .write = landing,
                      .later = landing,
                      .kind = VictimKind::Release};
    if (!unit.bytes)
      return;
    units.push_back(unit);
    engine.cache_.countRelease(id, moves, landing, rule, count);
  };
  // Each lane's units. For a need of Full pages, first the claim's pages
  // moved in place, with an extent more where they would leave the Index
  // runway short (Cache::inPlaceMoves), whose bytes it counts, where the
  // reads of the lanes they give Index pages lack nothing, since what it
  // frees is not sized for those: it moves the fewest pages, and a release
  // for bytes before it could take the pages it needs yet leave the claim
  // short of its extent, by 6.3 MB on Qwen3.8-Flash-Next. Then the unit for
  // bytes, where it frees some.
  for (const Request *lane : lanes) {
    const uint64_t id = lane->request.id;
    const Cache::ReleasablePages pages = engine.cache_.releasablePages(id, rule, count);
    const uint64_t reads = engine.readsShortfall(pages.holders);
    const uint32_t moves = engine.cache_.inPlaceMoves(query.need.pages, pages.clean);
    if (!reads && query.need.form == kv::PageForm::Full && moves)
      units.push_back({.key = id,
                       .bytes = uint64_t{moves - query.need.pages} * pageBytes,
                       .pages = query.need.pages,
                       .form = kv::PageForm::Full,
                       .kind = VictimKind::Release});
    offer(id, pages.clean, reads, false);
  }
  // Last, after every lane's, for a need of bytes the pages whose copies
  // land, which free nothing by themselves outside --kv-spill always: the
  // claim waits for them (Pending). Under it their landings count already
  // (Cache::returning).
  if (forBytes && !engine.config_.spill.always)
    for (const Request *lane : lanes) {
      const Cache::ReleasablePages pages = engine.cache_.releasablePages(lane->request.id, rule, count);
      offer(lane->request.id, pages.clean + pages.landing, engine.readsShortfall(pages.holders), true);
    }
  return units;
}

std::vector<Victim> Engine::EngineVictims::conversions(const VictimQuery &query) const {
  const Engine &engine = engine_;
  std::vector<Victim> units;
  // Only the claimant's own lane converts (TakeRights::convertLane). Its
  // claim's retries release its pages into the Index runway, so with none
  // free nothing converts.
  const uint64_t id = query.rights.convertLane;
  if (!id || !engine.config_.spill.capable || engine.cache_.spillGate() != SpillGate::Open ||
      !engine.cache_.freePages(kv::PageForm::Index))
    return units;
  // Resident, decoding or prefilling, between its commands and not between a
  // command's passes: its history is whole and its next growth is what it
  // lacks. A prefill goes on as a Spill lane's.
  const auto found = engine.requests_.find(id);
  if (found == engine.requests_.end())
    return units;
  const Request &active = found->second;
  if (!active.lane || active.restore || active.finalized || active.pendingEnd ||
      active.kvMode != KvMode::Resident || engine.passCommand_.holds(id))
    return units;
  const SpillGeometry &geometry = engine.config_.spill.geometry;
  if (!geometry.slotBytes ||
      engine.cache_.spillRoom() / geometry.slotBytes <
          engine.spillPages(active, engine.cache_.chainCopies(engine.chainEnd(active)).copied))
    return units;
  // The gate: the pages it would spill (Full pages no Resident request
  // holds) free Full - Index bytes each; less the first lane's hot block
  // floor its decode then reads through, they must cover what the claim
  // lacks. On Flash-Next a lane converts from about 46K tokens.
  const uint64_t pages = engine.cache_.convertiblePages(id);
  const uint64_t freed = pages * geometry.releasedPageBytes();
  const uint64_t reads = engine.model_.hotFloorBytes(1);
  const uint64_t missing = std::max(query.need.bytes, uint64_t{query.need.pages} * geometry.fullPageBytes);
  if (freed <= reads || freed - reads < missing)
    return units;
  units.push_back(Victim{.key = id,
                         .bytes = freed - reads,
                         .pages = static_cast<uint32_t>(pages),
                         .form = kv::PageForm::Full,
                         .write = true,
                         .later = true,
                         .kind = VictimKind::Conversion});
  return units;
}

std::optional<Victim> Engine::EngineVictims::peek(const VictimQuery &query, const Victim *after) const {
  // The idle state buffers, taken as they go, of at least the bytes the
  // pool releases short of critical pressure.
  const Engine &engine = engine_;
  if (!after)
    return Victim{.bytes = engine.model_.idleBufferBytes(0, query.need.afterStart, true), .sized = false};
  if (after->kind == VictimKind::LaneStart)
    return std::nullopt;
  // Then, for work that supersedes it, the command between its passes with
  // none in flight: the buffers its captures hold, counted as bytes for a
  // need of bytes. It costs that work nothing, so it comes before the
  // releases and the conversions, which cost their lanes the spilled reads.
  if (after->kind == VictimKind::Memory && engine.passCommand_.active() && !engine.pending_ &&
      query.rights.passCommand)
    return Victim{.stateBuffers = engine.passCommand_.captureBuffers(), .kind = VictimKind::PassCommand};
  // The releases free now, or as copies land, without a conversion's
  // writes: they come before it. A pass over the order counts them once, as
  // it reaches them, and finds the unit after each among those (reached_).
  auto next = reached_.begin();
  if (after->kind == VictimKind::Release || after->kind == VictimKind::Conversion) {
    next = std::ranges::find_if(reached_, [&](const Victim &unit) {
      return unit.kind == after->kind && unit.key == after->key && unit.later == after->later &&
             (unit.pages != 0) == (after->pages != 0);
    });
    if (next != reached_.end())
      ++next;
  } else {
    reached_ = releases(query);
    const std::vector<Victim> converting = conversions(query);
    reached_.insert(reached_.end(), converting.begin(), converting.end());
    next = reached_.begin();
  }
  if (next != reached_.end())
    return *next;
  // Last, the pooled buffers the next lane starts from.
  if (query.rights.laneStart == TakeRights::LaneStart::Last)
    return Victim{.kind = VictimKind::LaneStart};
  return std::nullopt;
}

FreeResult Engine::EngineVictims::take(const VictimQuery &query, const Victim &victim) {
  Engine &engine = engine_;
  if (engine.pending_)
    throw std::logic_error("the engine's victims taken while a command is in flight");
  // The buffers one lane starts from stay unless the claim takes them first,
  // or takes them as the unit of their own.
  const auto releaseIdle = [&](bool keepLane) {
    uint64_t bytes = 0;
    while (const uint64_t buffer = engine.model_.releaseIdleBuffer(keepLane))
      bytes += buffer;
    return bytes;
  };
  const bool keepLaneStart = query.rights.laneStart != TakeRights::LaneStart::First;
  switch (victim.kind) {
  case VictimKind::Memory:
    if (const uint64_t bytes = engine.model_.releaseIdleBuffer(keepLaneStart))
      return {FreeOutcome::Met, bytes};
    return {};
  case VictimKind::Conversion:
    // Its Full pages as peek counted them, which the claim's retries
    // release once their copies land.
    if (engine.convertToSpill(engine.request(victim.key)))
      return {.outcome = FreeOutcome::Pending,
              .laterBytes = victim.bytes,
              .pages = query.need.form == victim.form ? victim.pages : 0};
    return {};
  case VictimKind::Release: {
    // Pages whose copies land free nothing yet: the claim waits for them.
    if (victim.later)
      return {.outcome = FreeOutcome::Pending};
    const ReleaseRule rule = releaseRule(query);
    const Cache::ReleasablePages releasable = engine.cache_.releasablePages(victim.key, rule);
    const uint64_t reads = engine.readsShortfall(releasable.holders);
    // In place, the pages the claim lacks still. Else what it lacks still in
    // bytes and the first reads of the lanes the pages give Index pages
    // beside it, in whole extents (Cache::releaseMoves), each page freeing
    // its Full bytes less its Index page's: four extents of pages for a
    // 390 MiB hot block floor, the free Full pages beside completing the
    // last where the lane's pages run short (Cache::releaseBeside).
    const bool inPlace = victim.pages != 0;
    uint32_t pages = std::min(query.lackingPages, victim.pages);
    if (!inPlace) {
      const uint64_t pageBytes = engine.config_.spill.geometry.releasedPageBytes();
      const uint64_t extent = engine.cache_.extentPages();
      const uint64_t lacking = saturatingAdd(query.lackingBytes, reads);
      const uint64_t whole = ((lacking + pageBytes - 1) / pageBytes + extent - 1) / extent * extent;
      pages = static_cast<uint32_t>(std::min<uint64_t>(whole, releasable.clean));
    }
    const uint64_t before = engine.cache_.allocatedBytes();
    const auto readsFirst = [&] {
      std::vector<uint64_t> lanes;
      for (const uint64_t id : releasable.holders)
        if (!engine.cache_.holdsIndexPages(id))
          lanes.push_back(id);
      return lanes;
    }();
    const uint32_t moved = engine.cache_.releasePages(
        victim.key, pages, inPlace, rule, [&](uint64_t bytes) { engine.governor().fund(bytes); },
        engine.cache_.releaseBeside(query));
    if (!moved)
      return {};
    // Critical pressure refuses a lane's first reads at its next prepare:
    // what this release freed pays for them here (MemoryGovernor::fund), or a
    // lane that needed no memory would stop and time out.
    if (query.rights.laneStart == TakeRights::LaneStart::First)
      for (const uint64_t id : readsFirst)
        if (engine.cache_.holdsIndexPages(id))
          static_cast<void>(engine.model_.reserveSpilledReads(
              id, {engine.scheduler_.phase(id) == Phase::Prefill ? WorkKind::Prefill : WorkKind::Decode,
                   engine.nextCommandPages(engine.requests_.at(id))}));
    // What it frees net of the first reads it gave the lanes, as its count
    // is: taken gross, a claim of two lanes' releases kept a 390 MiB hot
    // block floor less for each than it counted. In place an extent more
    // may leave beside the claim's pages (inPlaceMoves).
    const uint64_t created = saturatingSubtract(reads, engine.readsShortfall(releasable.holders));
    const uint64_t freed = saturatingSubtract(saturatingSubtract(before, engine.cache_.allocatedBytes()), created);
    if (inPlace)
      return {.outcome = FreeOutcome::Met, .nowBytes = freed, .pages = std::min(moved, pages)};
    if (!freed)
      return {};
    return {FreeOutcome::Met, freed};
  }
  case VictimKind::PassCommand:
    // The buffers its captures held. Settling here would change the ledger
    // inside its own free() (settleEndedBetweenPasses).
    static_cast<void>(engine.passCommand_.yield());
    return {.outcome = FreeOutcome::Met,
            .nowBytes = releaseIdle(keepLaneStart),
            .stateBuffers = victim.stateBuffers};
  case VictimKind::LaneStart:
    if (const uint64_t bytes = releaseIdle(false))
      return {FreeOutcome::Met, bytes};
    return {};
  }
  return {};
}

std::array<uint32_t, 3> Engine::EngineVictims::movesInto(const VictimQuery &query) const {
  // Released for the claim's bytes first, those free Index pages left the
  // release only the runway, and a lone lane's growth at the wall ended at
  // its length.
  std::array<uint32_t, 3> pages{};
  if (query.need.pages && query.need.form == kv::PageForm::Full &&
      std::ranges::any_of(releases(query), [](const Victim &unit) { return unit.pages != 0; }))
    pages[static_cast<size_t>(kv::PageForm::Index)] = engine_.cache_.inPlaceIndexPages(query.need.pages);
  return pages;
}

void Engine::suspendLane(Request &active, uint64_t workEnd, metal::AllocationFailure failure,
                         StateFailure reason, double now) {
  if (!active.lane || scheduler_.suspended(active.request.id)) {
    throw std::logic_error("request cannot be suspended");
  }
  // A decoding lane holds an anchor unless its first mask has yet to select
  // one; a lane in prefill holds what it held when its replay began.
  if (scheduler_.phase(active.request.id) == Phase::Decode)
    active.suspendedWithAnchor =
        scheduler_.decodeStage(active.request.id) == DecodeStage::Regular;
  vacateLane(active, true);
  active.resumeKvTargetTokens = workEnd;
  active.replayTokens = static_cast<uint32_t>(active.exactTokens.size());
  scheduler_.suspendForResources(active.request.id);
  deferResourceRetry(active, now, {.allocationFailure = failure}, reason);
}

void Engine::suspendForGrowth(Request &active, uint64_t workEnd,
                              metal::AllocationFailure failure, double now) {
  suspendLane(active, workEnd, failure, StateFailure::MemoryPressure, now);
  // Resident lanes drain before admission resumes. Growth the host refused
  // resumes when its pressure lifts; any other limit only once memory is
  // freed, so it counts as a failure the drain waits out.
  drainEndMilliseconds_ = now + config_.resourceWaitTimeoutMilliseconds;
  allocationFailed_ = failure != metal::AllocationFailure::HostPressure;
  ++counters_.resourceSuspensions;
}

void Engine::collectLandings(double now) {
  static_cast<void>(now);
  if (pending_)
    throw std::logic_error("memory landings collected while a command is in flight");
  if (cache_.pollTransfers()) {
    signalResourceProgress();
    if (config_.spill.capable)
      static_cast<void>(cache_.writeThrough());
  }
  spillLanded(true);
  governor().ledger().setLanes(laneDemandBytes(), highestLanePriority());
  // The chains requests without a lane restore once they start: admission's
  // probes brought up to date, and for those it has not probed, probes of
  // this pass alone.
  std::vector<CacheProbe> probes;
  std::vector<std::span<const uint64_t>> chains;
  for (auto &[_, active] : requests_) {
    if (active.lane || active.finalized || active.skipCache)
      continue;
    if (active.admissionProbe) {
      cache_.refresh(*active.admissionProbe, active.exactTokens, promptIdentity(active.request));
      chains.push_back(active.admissionProbe->restoredChain());
    } else {
      probes.push_back(cache_.probe(active.exactTokens, promptIdentity(active.request)));
    }
  }
  for (const CacheProbe &probe : probes)
    chains.push_back(probe.restoredChain());
  cache_.setWantedChains(std::move(chains));
}

} // namespace splash::engine
