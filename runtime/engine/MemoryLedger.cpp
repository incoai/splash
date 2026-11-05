#include "engine/MemoryLedger.hpp"

#include "Checked.hpp"
#include "engine/MemoryGovernor.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace splash::engine {

namespace {

constexpr double kMiB = double(1ULL << 20);

const char *className(ClaimClass cls) noexcept {
  switch (cls) {
  case ClaimClass::Floor: return "floor";
  case ClaimClass::InService: return "in-service";
  case ClaimClass::Start: return "start";
  case ClaimClass::Optional: return "optional";
  case ClaimClass::Growth: return "growth";
  case ClaimClass::Host: return "host";
  }
  return "host";
}

const char *outcomeName(FreeOutcome outcome) noexcept {
  switch (outcome) {
  case FreeOutcome::Met: return "met";
  case FreeOutcome::Pending: return "pending";
  case FreeOutcome::Short: return "short";
  }
  return "short";
}

} // namespace

TakeRights takeRights(const ClaimSpec &spec, const Need &need, bool critical,
                      std::optional<RequestPriority> highestLane) {
  using Reuse = TakeRights::CertainReuse;
  using Reserves = TakeRights::Reserves;
  using LaneStart = TakeRights::LaneStart;
  if (spec.cls == ClaimClass::Floor)
    throw std::logic_error("a Floor claim takes nothing: the start and a weights restore pass only the host's rule");
  // The engine's own optional work takes no unit, so every right is the
  // narrowest.
  if (spec.cls == ClaimClass::Optional && !spec.requestId)
    return {};
  const bool host = spec.cls == ClaimClass::Host;
  const bool inService = spec.cls == ClaimClass::InService;
  const bool hostCritical = host && critical;
  const bool waits = (host ? !critical : spec.timing == NeedTiming::CanWait) && !need.hostRefused;
  const bool belowALane = highestLane && *highestLane < spec.priority;
  return {.units = true,
          .waitsForWrites = waits,
          .waitsForLandings = waits && !host,
          .inUse = (inService && waits && spec.reach == ReclaimClass::InUse) || hostCritical,
          .certainReuse = inService || hostCritical ? Reuse::All
                          : spec.cls == ClaimClass::Start || spec.cls == ClaimClass::Optional ? Reuse::ResumePoint
                                                                                              : Reuse::None,
          .reserves = critical                                        ? Reserves::All
                      : inService || spec.cls == ClaimClass::Start ? Reserves::NextExtent
                                                                     : Reserves::None,
          .rebuiltPools = !belowALane && spec.cls != ClaimClass::Growth,
          .storeAboveFloor = !belowALane && ((inService && waits) || hostCritical),
          .passCommand = inService && waits && spec.supersedes,
          .convertLane = inService && waits && spec.kvGrowth ? spec.requestId : 0,
          .laneStart = critical                         ? LaneStart::First
                       : inService && waits && need.kv ? LaneStart::Last
                                                       : LaneStart::Kept};
}

// What a claim lacks, in each measure that can meet it: device bytes beyond
// the room it has, free pages of a form in kept extents, pooled state
// buffers.
struct MemoryLedger::Lacking final {
  uint64_t bytes = 0;
  uint32_t pages = 0;
  kv::PageForm form = kv::PageForm::Full;
  uint32_t buffers = 0;

  [[nodiscard]] bool none() const noexcept { return !bytes && !pages && !buffers; }
  // Whether a unit meets any of the need, in any measure; a need of bytes
  // counts its state buffers too, and the buffers a lane starts from, once
  // they leave the model's pool.
  [[nodiscard]] bool meets(const Victim &victim) const noexcept {
    return (bytes && (victim.bytes || victim.stateBuffers || victim.kind == VictimKind::LaneStart)) ||
           (pages && victim.form == form && victim.pages) || (buffers && victim.stateBuffers);
  }
  // Whether these amounts, of pages of its form, meet it, in any measure.
  [[nodiscard]] bool covered(uint64_t freedBytes, uint32_t freedPages, uint32_t freedBuffers) const noexcept {
    return (bytes && freedBytes >= bytes) || (pages && freedPages >= pages) || (buffers && freedBuffers >= buffers);
  }
};

struct MemoryLedger::Account final {
  std::string name;
  std::function<uint64_t()> current;
  model::ElasticPool *pool = nullptr;
  uint64_t grows = 0;
  uint64_t shrinks = 0;
};

MemoryLedger::MemoryLedger() {
  accounts_.push_back(Account{.name = "fixed"});
}

MemoryLedger::~MemoryLedger() = default;

AccountId MemoryLedger::addAccount(std::string name, std::function<uint64_t()> currentBytes) {
  if (accounts_.size() > std::numeric_limits<uint8_t>::max())
    throw std::logic_error("the memory ledger has no account id left");
  if (!currentBytes)
    throw std::invalid_argument("an account the ledger pulls needs its current bytes");
  accounts_.push_back(Account{.name = std::move(name), .current = std::move(currentBytes)});
  return {static_cast<uint8_t>(accounts_.size() - 1)};
}

AccountId MemoryLedger::addPool(model::ElasticPool &pool) {
  if (accounts_.size() > std::numeric_limits<uint8_t>::max())
    throw std::logic_error("the memory ledger has no account id left");
  accounts_.push_back(Account{.name = std::string(pool.state().name), .pool = &pool});
  return {static_cast<uint8_t>(accounts_.size() - 1)};
}

void MemoryLedger::addSource(VictimSource &source) {
  if (std::ranges::find(sources_, &source) == sources_.end())
    sources_.push_back(&source);
}

void MemoryLedger::removeSource(VictimSource &source) noexcept { std::erase(sources_, &source); }

void MemoryLedger::setReturning(std::function<Returning()> all,
                                std::function<Returning(std::span<const uint64_t>)> ofRequests) {
  returning_ = std::move(all);
  returningFor_ = std::move(ofRequests);
}

void MemoryLedger::setBufferBytes(std::function<uint64_t(uint32_t, bool, bool)> bytes) {
  bufferBytes_ = std::move(bytes);
}

void MemoryLedger::setPageBytes(const std::array<uint64_t, 3> &bytes) noexcept { pageBytes_ = bytes; }

void MemoryLedger::setNow(double nowSeconds) noexcept { nowSeconds_ = nowSeconds; }

void MemoryLedger::setLanes(uint64_t demandBytes, std::optional<RequestPriority> highest) noexcept {
  laneDemandBytes_ = demandBytes;
  lanePriority_ = highest;
}

void MemoryLedger::setDecisionLog(std::function<void(std::string_view)> log) { log_ = std::move(log); }

ClaimId MemoryLedger::open(const ClaimSpec &spec) {
  const ClaimId id{nextClaim_++};
  claims_.emplace(id.value, ClaimState{.spec = spec});
  return id;
}

void MemoryLedger::update(ClaimId claim, const ClaimSpec &spec) { claims_.at(claim.value).spec = spec; }

void MemoryLedger::refused(ClaimId claim, const Need &need) {
  claims_.at(claim.value).refusal = {.bytes = need.bytes, .hostRefused = need.hostRefused};
}

void MemoryLedger::setWaiting(ClaimId claim, bool waiting) {
  ClaimState &state = claims_.at(claim.value);
  if (waiting && !state.waiting)
    state.waitingSinceSeconds = nowSeconds_;
  state.waiting = waiting;
}

void MemoryLedger::setHostRefusedWaiting(ClaimId claim, bool waiting) {
  claims_.at(claim.value).hostRefusedWaiting = waiting;
}

void MemoryLedger::setHostExempt(ClaimId claim, bool exempt) { claims_.at(claim.value).hostExempt = exempt; }

void MemoryLedger::granted(ClaimId claim) {
  ClaimState &state = claims_.at(claim.value);
  state.logged.clear();
}

void MemoryLedger::close(ClaimId claim) noexcept { claims_.erase(claim.value); }

const ClaimState &MemoryLedger::claim(ClaimId claim) const { return claims_.at(claim.value); }

uint64_t MemoryLedger::heldByClaims(const ClaimRank &rank, std::optional<uint64_t> self) const noexcept {
  uint64_t held = 0;
  for (const auto &[id, state] : claims_)
    if (state.waiting && (!self || id != *self) && state.spec.rank > rank)
      held = saturatingAdd(held, state.need.bytes);
  return held;
}

uint64_t MemoryLedger::heldFor(const ClaimSpec &spec, std::optional<uint64_t> self) const noexcept {
  // The host's claim is for memory the host lacks, which no room under the
  // engine's limit returns: nothing held there stands against it.
  if (spec.cls == ClaimClass::Host)
    return 0;
  const uint64_t lanes = spec.cls == ClaimClass::Growth ? laneDemandBytes_ : 0;
  return saturatingAdd(lanes, heldByClaims(spec.rank, self));
}

uint64_t MemoryLedger::heldBytes(ClaimId claim) const { return heldFor(claims_.at(claim.value).spec, claim.value); }

std::array<uint32_t, 3> MemoryLedger::heldPages(ClaimId claim, uint64_t roomBytes) const {
  const ClaimState &self = claims_.at(claim.value);
  std::array<uint32_t, 3> held{};
  for (const auto &[id, state] : claims_)
    if (state.waiting && id != claim.value && state.spec.rank > self.spec.rank)
      held[static_cast<size_t>(state.need.form)] += state.need.pages;
  // The bytes they lack beyond the room admission would grant them, which
  // it holds for them already, in each form's pages: a page handed out of a
  // kept extent's free pages takes room they wait for as an extent would
  // (the extents their landings complete hold free pages too). A form
  // without pages has none to hold.
  const uint64_t bytes = saturatingSubtract(heldByClaims(self.spec.rank, claim.value), roomBytes);
  for (size_t form = 0; form < held.size(); ++form)
    if (pageBytes_[form]) {
      const uint64_t pages = bytes / pageBytes_[form] + (bytes % pageBytes_[form] != 0);
      held[form] = static_cast<uint32_t>(
          std::min<uint64_t>(held[form] + pages, std::numeric_limits<uint32_t>::max()));
    }
  return held;
}

bool MemoryLedger::claimWaiting() const noexcept {
  return std::ranges::any_of(claims_, [](const auto &entry) { return entry.second.waiting; });
}

bool MemoryLedger::holdsGrowth() const noexcept {
  return std::ranges::any_of(claims_, [](const auto &entry) {
    return entry.second.spec.requestId && (entry.second.waiting || entry.second.hostRefusedWaiting);
  });
}

uint64_t MemoryLedger::hostRefusedWaitingBytes() const noexcept {
  uint64_t bytes = 0;
  for (const auto &[id, state] : claims_)
    if ((state.waiting || state.hostRefusedWaiting) && state.refusal.hostRefused)
      bytes = saturatingAdd(bytes, state.refusal.bytes);
  return bytes;
}

uint64_t MemoryLedger::poolReleases() const noexcept {
  uint64_t units = 0;
  for (const Account &account : accounts_)
    units = saturatingAdd(units, account.shrinks);
  return units;
}

uint64_t MemoryLedger::largestGrowthUnit() const {
  uint64_t unit = 0;
  for (const Account &account : accounts_)
    if (account.pool)
      unit = std::max(unit, account.pool->state().growUnitBytes);
  return unit;
}

Returning MemoryLedger::returning() const { return returning_ ? returning_() : Returning{}; }

bool MemoryLedger::copyDemand() const noexcept {
  return uncopiedReachedSeconds_ && nowSeconds_ - *uncopiedReachedSeconds_ <= kCopyDemandSeconds;
}

// What the units of a run give a claim: counted (bound) or taken (take).
struct MemoryLedger::Run final {
  // take: the units taken, by source or pool, and the bytes each freed.
  struct Took final {
    size_t source = 0;
    std::optional<AccountId> pool;
    Victim unit;
    uint64_t bytes = 0;
  };
  std::vector<Took> took;
  // Now and later, with the bytes its state buffers give once they leave
  // the model's pool.
  uint64_t bytes = 0;
  // take: what the model's pool releases now beyond what it released
  // before the run, of the buffers its states returned to it.
  uint64_t pooled = 0;
  uint32_t pages = 0;
  uint32_t buffers = 0;
  bool covered = false;
  // take: what it freed now and as the transfers it started land.
  FreeResult freed;
  // take: a unit it reached waits for a transfer in flight, for a claim
  // that can wait; one that cannot waits for nothing that lands.
  bool pending = false;
  // It reached a cached unit that frees memory only once its copy lands.
  bool uncopied = false;
  // take: it took a unit whose loss costs work later: a store chunk, the
  // command between its passes, a conversion, a unit in use, the buffers
  // the next lane starts from.
  bool costly = false;
};

// The one order units are counted and taken in: memory nothing needs that
// frees now (the cache's units that need no write, by its walk); the units
// of pools that cost only their rebuild; the cache's units that need a
// write (Victim::write), a state the idle staging buffer writes, which
// frees at once, and what frees once a write lands; the claimant's own
// conversion to Spill and the command between its passes; the other pools'
// units (the expert store's chunks, top first) down to their floors, which
// cost reads while they stay away; what another request uses, since its
// state in use costs it a re-prefill of its whole prompt while a store
// chunk costs only reads until it regrows; last the pooled buffers the next
// lane starts from, which the claim takes only once nothing else in its
// reach is left. Within a stage the source registered first goes first,
// then the pools; each source's own order is kept. What the claim may not
// take (TakeRights) is left out, by its source or here for the pools, not
// stopped at; a claim that does not wait for writes passes over what frees
// only once one lands (Victim::later).
class MemoryLedger::Order final {
public:
  struct Head final {
    std::optional<Victim> victim;
    // The unit passed over last, before it in its source's order.
    std::optional<Victim> before;
    size_t source = 0;
    std::optional<AccountId> pool;
    // A pool's reading, and the bytes above its floor a count passed.
    model::PoolState state;
    uint64_t counted = 0;
    uint32_t units = 0;
  };

  Order(const MemoryLedger &ledger, const VictimQuery &query) : ledger_(ledger), query_(query) {
    const TakeRights &rights = query.rights;
    heads.reserve(ledger.sources_.size() + ledger.accounts_.size());
    for (size_t index = 0; index < ledger.sources_.size(); ++index)
      heads.push_back(Head{.victim = ledger.sources_[index]->peek(query, nullptr), .source = index});
    for (size_t index = 1; index < ledger.accounts_.size(); ++index) {
      const Account &account = ledger.accounts_[index];
      if (!account.pool)
        continue;
      Head head{.pool = AccountId{static_cast<uint8_t>(index)}, .state = account.pool->state()};
      if (!(head.state.rebuildOnly ? rights.rebuiltPools : rights.storeAboveFloor))
        continue;
      head.victim = poolUnit(head);
      heads.push_back(std::move(head));
    }
  }

  // The order's stages. What frees without a write comes first; from
  // PassOrConversion on, a unit's loss costs work later (Run::costly), and a
  // pending unit holds those stages back.
  enum class Stage : uint8_t { FreeNow, Rebuild, NeedsWrite, PassOrConversion, StoreAboveFloor, InUse, LaneStart };
  [[nodiscard]] static Stage stage(const Head &head) {
    if (head.pool)
      return head.state.rebuildOnly ? Stage::Rebuild : Stage::StoreAboveFloor;
    if (head.victim->kind == VictimKind::LaneStart)
      return Stage::LaneStart;
    if (head.victim->kind != VictimKind::Memory)
      return Stage::PassOrConversion;
    if (head.victim->cls == ReclaimClass::InUse)
      return Stage::InUse;
    return head.victim->write ? Stage::NeedsWrite : Stage::FreeNow;
  }

  // Moves past the head's unit, which stays in its source's order (or which
  // a count took).
  void pass(Head &head) {
    if (head.pool) {
      head.counted += head.victim->bytes;
      ++head.units;
      head.victim = poolUnit(head);
      return;
    }
    head.before = head.victim;
    head.victim = ledger_.sources_[head.source]->peek(query_, &*head.before);
  }

  // The head's unit after a take, from its source's head: what a take
  // freed can bring back units passed over (a leaf whose state went, now
  // dead, is in another phase of the cache's walk).
  void again(Head &head) {
    if (head.pool) {
      head.state = ledger_.accounts_[head.pool->value].pool->state();
      head.counted = 0;
      head.units = 0;
      head.victim = poolUnit(head);
      return;
    }
    head.before.reset();
    head.victim = ledger_.sources_[head.source]->peek(query_, nullptr);
  }

  std::vector<Head> heads;

private:
  [[nodiscard]] std::optional<Victim> poolUnit(const Head &head) const {
    const uint64_t above = saturatingSubtract(head.state.currentBytes, head.state.floorBytes);
    if (!head.state.releaseUnitBytes || head.counted >= above)
      return std::nullopt;
    // Each unit at its own bytes where the pool lists them (the store's
    // chunks, top first), else at the next one's.
    const std::vector<uint64_t> &units = head.state.releaseUnits;
    const uint64_t bytes =
        std::min(head.units < units.size() ? units[head.units] : head.state.releaseUnitBytes, above - head.counted);
    return Victim{.key = head.units, .bytes = bytes};
  }

  const MemoryLedger &ledger_;
  const VictimQuery &query_;
};

uint64_t MemoryLedger::pooledBytes(const VictimQuery &query, const Lacking &lack, uint32_t states,
                                   bool keepLane) const {
  if (!lack.bytes || !bufferBytes_)
    return 0;
  return bufferBytes_(states, query.need.afterStart, keepLane);
}

// The most the order gives a claim, counted without taking: the units
// take() would take, in its order, each at what its source says it frees
// at most (VictimSource::peek), and none past the unit that covers the
// lack. The state buffers of the units counted give a need of bytes what
// the model's pool would release with them beyond what it releases now
// (pooledBytes), without keeping the buffers the next lane starts from once
// the count reaches them. A unit that meets none of the need, or that lands
// later for an Immediate claim, is skipped. A claim that takes no units
// (TakeRights::units) counts none and reaches none: no source is walked.
MemoryLedger::Run MemoryLedger::bound(const VictimQuery &query, const Lacking &lack) const {
  Run run;
  if (!query.rights.units)
    return run;
  Order order(*this, query);
  const uint64_t pooledBefore = pooledBytes(query, lack, 0);
  uint64_t unitBytes = 0;
  bool keepLane = true;
  for (;;) {
    Order::Head *best = nullptr;
    for (Order::Head &head : order.heads) {
      while (head.victim && !lack.meets(*head.victim))
        order.pass(head);
      if (head.victim && (!best || Order::stage(head) < Order::stage(*best)))
        best = &head;
    }
    if (!best)
      break;
    const Victim victim = *best->victim;
    if (victim.later && victim.kind == VictimKind::Memory)
      run.uncopied = true;
    // Kept in an Immediate claim's way until here, to note that the run
    // reached it.
    if (victim.later && !query.rights.waitsForWrites) {
      order.pass(*best);
      continue;
    }
    unitBytes = saturatingAdd(unitBytes, victim.bytes);
    if (victim.form == lack.form)
      run.pages += victim.pages;
    run.buffers += victim.stateBuffers;
    keepLane = keepLane && victim.kind != VictimKind::LaneStart;
    run.bytes = saturatingSubtract(saturatingAdd(unitBytes, pooledBytes(query, lack, run.buffers, keepLane)),
                                   pooledBefore);
    if (lack.covered(run.bytes, run.pages, run.buffers)) {
      run.covered = true;
      break;
    }
    order.pass(*best);
  }
  return run;
}

// Takes the order's units one at a time until what they freed, now and as
// the transfers they started land, covers the lack. After a take that freed
// something its source is peeked again from its head; a unit whose take
// frees nothing is passed over: a transfer in flight may hold it (pending),
// or nothing of it is left. A claim that can wait takes nothing past the
// cache's units that need a write once a unit is pending: it waits for the
// transfer rather than take what costs work later. Every take frees
// something of a finite cache or pool, or passes one of a source's
// finitely many units, so the walk ends. The state buffers of the units
// taken give a need of bytes as in bound(), read from the model's pool once
// their takes returned them to it, so that each counts once: what the pool
// releases now beyond what it released before the run, an idle buffer a
// take released being in what that take freed. An Immediate claim passes
// over what lands later.
MemoryLedger::Run MemoryLedger::take(const VictimQuery &query, const Lacking &lack) {
  Order order(*this, query);
  Run run{.freed = {.outcome = FreeOutcome::Met}};
  const uint64_t pooledBefore = pooledBytes(query, lack, 0);
  // A pending unit holds back the stages after the cache's writes.
  Order::Stage ceiling = Order::Stage::LaneStart;
  while (!lack.none() && !lack.covered(run.bytes, run.pages, run.buffers)) {
    Order::Head *best = nullptr;
    for (Order::Head &head : order.heads) {
      while (head.victim && (!lack.meets(*head.victim) ||
                             (!query.rights.waitsForWrites && head.victim->later))) {
        if (head.victim->later && head.victim->kind == VictimKind::Memory)
          run.uncopied = true;
        order.pass(head);
      }
      if (head.victim && Order::stage(head) <= ceiling && (!best || Order::stage(head) < Order::stage(*best)))
        best = &head;
    }
    if (!best)
      break;
    const Victim unit = *best->victim;
    FreeResult taken;
    if (best->pool) {
      Account &account = accounts_[best->pool->value];
      taken.nowBytes = account.pool->release(unit.bytes);
      if (taken.nowBytes) {
        taken.outcome = FreeOutcome::Met;
        ++account.shrinks;
      }
    } else {
      taken = sources_[best->source]->take(query, unit);
      if (unit.later && unit.kind == VictimKind::Memory)
        run.uncopied = true;
    }
    if (!taken.gave()) {
      if (taken.outcome == FreeOutcome::Pending && query.rights.waitsForWrites) {
        run.pending = true;
        ceiling = std::min(ceiling, std::max(Order::stage(*best), Order::Stage::NeedsWrite));
      }
      order.pass(*best);
      continue;
    }
    if (taken.outcome == FreeOutcome::Pending)
      run.freed.outcome = FreeOutcome::Pending;
    run.freed.nowBytes = saturatingAdd(run.freed.nowBytes, taken.nowBytes);
    run.freed.laterBytes = saturatingAdd(run.freed.laterBytes, taken.laterBytes);
    const uint64_t pooledNow = pooledBytes(query, lack, 0);
    run.pooled = saturatingSubtract(pooledNow, pooledBefore);
    run.bytes = saturatingSubtract(saturatingAdd(saturatingAdd(run.freed.nowBytes, run.freed.laterBytes), pooledNow), pooledBefore);
    run.pages += taken.pages;
    run.buffers += taken.stateBuffers;
    run.costly = run.costly || Order::stage(*best) > Order::Stage::NeedsWrite;
    if (!run.took.empty() && run.took.back().source == best->source && run.took.back().pool == best->pool &&
        run.took.back().unit.key == unit.key)
      run.took.back().bytes = saturatingAdd(run.took.back().bytes, saturatingAdd(taken.nowBytes, taken.laterBytes));
    else
      run.took.push_back({best->source, best->pool, unit, saturatingAdd(taken.nowBytes, taken.laterBytes)});
    order.again(*best);
  }
  run.covered = lack.covered(run.bytes, run.pages, run.buffers);
  if (run.freed.laterBytes)
    run.freed.outcome = FreeOutcome::Pending;
  return run;
}

uint64_t MemoryLedger::takeUnsized(const VictimQuery &query) {
  uint64_t bytes = 0;
  for (VictimSource *source : sources_) {
    for (std::optional<Victim> unit = source->peek(query, nullptr);
         unit && !unit->sized; unit = source->peek(query, nullptr)) {
      const FreeResult taken = source->take(query, *unit);
      bytes = saturatingAdd(bytes, taken.nowBytes);
      if (!taken.nowBytes && !taken.laterBytes)
        break;
    }
  }
  return bytes;
}

// The one rule. The units that cost nothing and cannot be sized go first. Then
// a claim that can wait (never the host's, whose pass's target is net of what
// lands) waits when what is landing, with its own lanes' owed copies, meets its
// need (or would, but for claims ranked above it). Else, when the most the
// order gives (bound) meets the rest, its units are taken one at a time until
// what they freed now and later meets it (take), and the claim waits for what
// lands unless they meet it alone; one they do not meet waits while what it
// reached or started lands, else is Short. Else nothing is taken, and it waits
// when every copy in flight or owed would meet it. A claim refused bytes lacks
// what admission asks beyond the room: its need plus the room held against it.
// A need met in place lacks no bytes. A need the host refused counts only units
// that leave memory in place. The host's pass asks with no room
// (MemoryGovernor::hostPass), since room under the engine's limit returns
// nothing to the host. State buffers the units free meet a need of bytes once they
// leave the model's pool, so the units that cost nothing go again (a need in
// place keeps them). A claim that can wait, short only by the room waiting
// claims ranked above it hold, waits as their holds end.
FreeResult MemoryLedger::free(ClaimId claimId, const Need &need, uint64_t freeBytes, bool critical, bool exempts) {
  const uint64_t id = claimId.value;
  ClaimState &claim = claims_.at(id);
  const ClaimSpec &spec = claim.spec;
  const VictimQuery query{.rights = takeRights(spec, need, critical, lanePriority_), .need = need};

  FreeResult result{.nowBytes = takeUnsized(query)};
  const uint64_t room = saturatingAdd(freeBytes, result.nowBytes);
  // Waiting claims ranked above it hold room it waits for, and the lanes'
  // projected growth room a pool's growth leaves them.
  const uint64_t held = heldFor(spec, id);
  const uint64_t forClaim = saturatingSubtract(room, held);
  const uint64_t lackingBytes =
      need.hostRefused || !need.bytes ? 0 : saturatingSubtract(saturatingAdd(need.bytes, held), room);
  const uint32_t lackingPages = need.pages;
  const bool canWait = query.rights.waitsForLandings;

  // What a claim that can wait may wait for: what lands of every transfer
  // in flight and every owed copy, and the copies its own lane owes; the
  // pages of its form waiting claims ranked above it take first.
  Returning all, own;
  const size_t form = static_cast<size_t>(need.form);
  uint64_t heldPagesAbove = 0;
  if (canWait && (lackingBytes || lackingPages)) {
    all = returning();
    if (returningFor_ && spec.requestId)
      own = returningFor_(std::span<const uint64_t>(&spec.requestId, 1));
    heldPagesAbove = heldPages(ClaimId{id}, room)[form];
  }
  const auto waits = [&](bool owedToo) {
    if (!canWait || (!lackingBytes && !lackingPages))
      return false;
    const uint64_t landing = saturatingAdd(all.landingBytes, owedToo ? all.owedBytes : own.owedBytes);
    const uint64_t landingPages =
        uint64_t{all.landingPages[form]} + (owedToo ? all.owedPages[form] : own.owedPages[form]);
    if (lackingBytes && (saturatingAdd(forClaim, landing) >= need.bytes ||
                         (!owedToo && saturatingAdd(room, all.landingBytes) >= need.bytes)))
      return true;
    return lackingPages && (saturatingSubtract(landingPages, heldPagesAbove) >= lackingPages ||
                            (!owedToo && all.landingPages[form] >= lackingPages));
  };
  // What lands and its own lanes' owed copies count before victims also in
  // part: the units cover only what they do not. The room held above the
  // claim is in what it lacks already.
  const uint64_t creditBytes = saturatingAdd(all.landingBytes, own.owedBytes);
  const uint64_t creditPages =
      saturatingSubtract(uint64_t{all.landingPages[form]} + own.owedPages[form], heldPagesAbove);

  std::optional<Run> run;
  // A refusal of bytes the room covers once the units that cost nothing
  // went is met: the fit is tried again, whatever memory allocated could
  // meet of it in place.
  if (!need.hostRefused && !lackingBytes && (need.bytes || (!lackingPages && !need.lacksStateBuffers()))) {
    result.outcome = FreeOutcome::Met;
  } else if (waits(false)) {
    result.outcome = FreeOutcome::Pending;
  } else {
    const Lacking lack{.bytes = saturatingSubtract(lackingBytes, creditBytes),
                       .pages = static_cast<uint32_t>(saturatingSubtract(lackingPages, creditPages)),
                       .form = need.form,
                       .buffers = need.stateBuffers};
    const Run most = bound(query, lack);
    if (most.uncopied)
      uncopiedReachedSeconds_ = nowSeconds_;
    if (most.covered) {
      run = take(query, lack);
      if (run->uncopied)
        uncopiedReachedSeconds_ = nowSeconds_;
      FreeResult taken = run->freed;
      // The buffers its states freed leave the model's pool unless they met
      // the need in place, so a need of bytes beside an unheldStateBuffer is
      // met by their bytes.
      const bool metInPlace = need.stateBuffers && run->buffers >= need.stateBuffers;
      if (run->pooled && !metInPlace)
        taken.nowBytes = saturatingAdd(taken.nowBytes, takeUnsized(query));
      const bool alone = (lackingBytes && run->bytes >= lackingBytes) ||
                         (lackingPages && run->pages >= lackingPages) ||
                         (need.stateBuffers && run->buffers >= need.stateBuffers);
      // Not met, it waits while what it reached or started lands, or as
      // every copy in flight or owed would meet it.
      const bool landing = run->pending || taken.laterBytes || waits(true);
      const FreeOutcome outcome = !run->covered ? (landing ? FreeOutcome::Pending : FreeOutcome::Short)
                                  : alone        ? taken.outcome
                                                 : FreeOutcome::Pending;
      result = {outcome, saturatingAdd(result.nowBytes, taken.nowBytes), taken.laterBytes, run->pages, run->buffers};
    } else {
      // Short only by the room waiting claims ranked above it hold, it waits
      // for them and takes nothing: Short would end a lone lane as if at its
      // capacity (Engine::judge).
      const bool heldAbove = canWait && lackingBytes && held &&
                             saturatingAdd(saturatingAdd(most.bytes, creditBytes), held) >= lackingBytes;
      result.outcome = waits(true) || heldAbove ? FreeOutcome::Pending : FreeOutcome::Short;
    }
  }
  // Pending: the claim waits, holding this need. Short: it waits for
  // nothing. Met: its wait stays as it was, since a need met for one of its
  // request's allocations does not end the request's wait for another.
  if (result.outcome == FreeOutcome::Pending) {
    if (!claim.waiting)
      claim.waitingSinceSeconds = nowSeconds_;
    claim.waiting = true;
    claim.need = need;
  } else if (result.outcome == FreeOutcome::Short) {
    claim.waiting = false;
  }
  // A request whose claim is not met, or that took what costs work later,
  // restarts the pools' wait to regrow: a unit regrown now would go back to
  // its next claim. One met from cache does not, or requests arriving every
  // few seconds would keep the pools from ever regrowing.
  if (spec.requestId && (result.outcome != FreeOutcome::Met || (run && run->costly)))
    quietSinceSeconds_ = nowSeconds_;
  // Met past the host on its retry, the line says so: logged short, a
  // prefill whose every extent the host refused read as short for 67 s
  // while it ran at its full rate.
  const bool exempted = exempts && need.hostRefused && result.outcome == FreeOutcome::Short;
  if (exempted) {
    claim.hostExempt = true;
    result.outcome = FreeOutcome::Met;
  }
  // One line per decision of a request's claim, with what its units took:
  // an outcome, or a need met past the host, once for each kind of need
  // since the claim's last grant. A retry back to a decision logged since
  // then logs nothing, whatever it took and however much the need still
  // lacks: a claim that flips between short and pending as copies land
  // logged 669 lines in one second, and one whose pages came a page at a
  // time a pair of lines a page (5,216 of a run's 5,812). A claim of no
  // request logs each take.
  bool logs = run.has_value();
  if (spec.requestId) {
    const auto bit = static_cast<uint8_t>(1U << (exempted ? 3U : static_cast<unsigned>(result.outcome)));
    const ClaimState::Logged::Lack lack{.bytes = need.bytes != 0,
                                        .hostRefused = need.hostRefused,
                                        .pages = need.pages != 0,
                                        .stateBuffers = need.lacksStateBuffers(),
                                        .form = need.form};
    auto found = std::ranges::find(claim.logged, lack, &ClaimState::Logged::lack);
    if (found == claim.logged.end())
      found = claim.logged.insert(claim.logged.end(), {lack});
    logs = !(found->decisions & bit);
    found->decisions |= bit;
  }
  if (logs)
    log(spec, need, run ? &*run : nullptr, result, exempted);
  return result;
}

// free() counted, taking nothing (bound): the units that cost nothing and are sized only when
// taken count as their sources size them now (peek), and for a claim that
// can wait what lands and the copies its own request owes count before
// victims, as free() counts them.
uint64_t MemoryLedger::available(const ClaimSpec &spec, const Need &need, uint64_t freeBytes, bool critical,
                                 const RestoredChains *kept) const {
  const VictimQuery query{.rights = takeRights(spec, need, critical, lanePriority_), .need = need, .kept = kept};
  const uint64_t enough = need.bytes;
  const uint64_t held = heldFor(spec, std::nullopt);
  uint64_t landing = 0;
  if (query.rights.waitsForLandings) {
    landing = returning().landingBytes;
    if (returningFor_ && spec.requestId)
      landing = saturatingAdd(landing, returningFor_(std::span<const uint64_t>(&spec.requestId, 1)).owedBytes);
  }
  const uint64_t room = saturatingAdd(freeBytes, landing);
  if (saturatingSubtract(room, held) >= enough)
    return enough;
  const Lacking lack{.bytes = saturatingSubtract(saturatingAdd(enough, held), room)};
  const uint64_t taken = bound(query, lack).bytes;
  return std::min(enough, saturatingSubtract(saturatingAdd(room, taken), held));
}

// Growth into the room beyond what waiting claims and the lanes' projected
// growth hold, a pool at a time in registration order. Nothing grows while a
// request's claim waits for memory, or its request to retry what the host
// refused: the claim takes the memory that appears. A unit takes the room it
// lacks from cache units that free memory now, in the cache's order, as an
// Immediate Growth claim: never what is in use, of certain reuse, or frees
// only once a write lands; it takes the state the idle staging buffer
// writes. Growing into free room alone, the lanes' KV cached after a burst
// that shrank the store would keep it at its floor. A count (bound) first
// finds what the cache funds at most; the host then judges the unit's net
// change (GrowthHost::admits), and a refusal takes nothing. Serving's passes
// (GrowthHost::paced) try one unit once kRegrowQuietSeconds passed with no
// request short of memory or taking what costs work later, and no try, so a
// regrown unit is not retaken soon after and the cache is walked at most
// once per quiet period; what a try lacks a temporary tier copies ahead
// (growthLackBytes) for the next. The start grows each pool as far as the
// room and the host admit.
bool MemoryLedger::grow(uint64_t freeBytes, const GrowthHost &host) {
  if (holdsGrowth()) {
    quietSinceSeconds_ = nowSeconds_;
    growthLackBytes_ = 0;
    return false;
  }
  if (host.paced && nowSeconds_ - quietSinceSeconds_ < kRegrowQuietSeconds)
    return false;
  growthLackBytes_ = 0;
  const ClaimSpec spec{.cls = ClaimClass::Growth, .timing = NeedTiming::Immediate};
  uint64_t grownBytes = 0;
  bool grew = false;
  for (size_t index = 1; index < accounts_.size(); ++index) {
    Account &account = accounts_[index];
    if (!account.pool)
      continue;
    // A unit the host refuses even when the cache funds it all is not
    // counted: the cache is not walked for it.
    for (uint64_t unit = account.pool->state().growUnitBytes; unit && host.admits(unit, unit);
         unit = account.pool->state().growUnitBytes) {
      const uint64_t room = saturatingSubtract(freeBytes, grownBytes);
      const uint64_t lacking = saturatingSubtract(saturatingAdd(unit, heldFor(spec, std::nullopt)), room);
      uint64_t funded = 0;
      bool grown = true;
      if (lacking) {
        const VictimQuery query{.rights = takeRights(spec, {.bytes = unit}, false, lanePriority_),
                                .need = {.bytes = unit}};
        const Run run = bound(query, {.bytes = lacking});
        if (run.uncopied)
          uncopiedReachedSeconds_ = nowSeconds_;
        grown = run.covered;
        funded = run.bytes;
      }
      grown = grown && host.admits(unit, funded);
      // Growth into free room alone that the host refuses is no try: the
      // cache was not walked.
      if (!grown && !lacking)
        break;
      if (grown) {
        const ClaimId claim = open(spec);
        if (lacking) {
          const FreeResult taken = free(claim, {.bytes = unit}, room, false);
          freeBytes = saturatingAdd(freeBytes, taken.nowBytes);
          grown = taken.outcome == FreeOutcome::Met;
        }
        grown = grown && host.grow(claim, *account.pool);
        close(claim);
      }
      if (grown) {
        grownBytes = saturatingAdd(grownBytes, unit);
        ++account.grows;
        grew = true;
        if (log_) {
          std::ostringstream line;
          line << std::fixed << std::setprecision(1) << "Memory: " << account.name << " grew "
               << double(unit) / kMiB << " MiB";
          log_(line.str());
        }
      } else if (lacking && funded < lacking) {
        // Only what the cache could not give: copies ahead cannot change a
        // refusal of the host's.
        growthLackBytes_ = lacking;
      }
      if (host.paced) {
        quietSinceSeconds_ = nowSeconds_;
        return grew;
      }
      if (!grown)
        break;
    }
  }
  return grew;
}

void MemoryLedger::fill(MemoryGovernorSnapshot &snapshot, uint64_t chargedBytes) const {
  snapshot.accounts.clear();
  uint64_t others = 0;
  std::vector<MemoryAccountSnapshot> accounts;
  for (size_t index = 1; index < accounts_.size(); ++index) {
    const Account &account = accounts_[index];
    MemoryAccountSnapshot row{.name = account.name, .grows = account.grows, .shrinks = account.shrinks};
    if (account.pool) {
      const model::PoolState state = account.pool->state();
      row.floorBytes = state.floorBytes;
      row.currentBytes = state.currentBytes;
      row.capBytes = state.capBytes;
      row.unitBytes = state.releaseUnitBytes ? state.releaseUnitBytes : state.growUnitBytes;
    } else {
      row.currentBytes = account.current();
      row.capBytes = row.currentBytes;
    }
    others = saturatingAdd(others, row.currentBytes);
    accounts.push_back(std::move(row));
  }
  const uint64_t fixed = saturatingSubtract(chargedBytes, others);
  snapshot.accounts.push_back({.name = accounts_.front().name, .currentBytes = fixed, .capBytes = fixed});
  for (MemoryAccountSnapshot &row : accounts)
    snapshot.accounts.push_back(std::move(row));

  const Returning back = returning();
  snapshot.owed = {.transfers = back.transfers,
                   .bytes = saturatingAdd(back.landingBytes, back.owedBytes),
                   .oldestMilliseconds = back.oldestMilliseconds};
  snapshot.claims = {};
  std::optional<double> oldest;
  for (const auto &[id, state] : claims_) {
    if (!state.waiting)
      continue;
    ++snapshot.claims.waiting;
    snapshot.claims.bytes = saturatingAdd(snapshot.claims.bytes, state.need.bytes);
    oldest = std::min(oldest.value_or(state.waitingSinceSeconds), state.waitingSinceSeconds);
  }
  if (oldest)
    snapshot.claims.oldestMilliseconds = std::max(0.0, nowSeconds_ - *oldest) * 1000.0;
  snapshot.owed.toClaimsBytes = std::min(snapshot.owed.bytes, snapshot.claims.bytes);
}

void MemoryLedger::log(const ClaimSpec &spec, const Need &need, const Run *run, const FreeResult &result,
                       bool exempted) const {
  if (!log_)
    return;
  std::ostringstream line;
  line << std::fixed << std::setprecision(1) << "Memory: " << className(spec.cls) << " claim";
  if (spec.requestId)
    line << " of request " << spec.requestId;
  line << " needs " << double(need.bytes) / kMiB << " MiB";
  if (need.pages)
    line << ", " << need.pages << " pages";
  if (need.stateBuffers)
    line << ", " << need.stateBuffers << " state buffers";
  if (need.unheldStateBuffer)
    line << ", a state buffer no cached state holds";
  line << ": " << outcomeName(result.outcome);
  if (exempted)
    line << " past the host";
  if (run) {
    line << ", took";
    for (const Run::Took &took : run->took) {
      line << ' ';
      if (took.pool)
        line << accounts_[took.pool->value].name << " unit";
      else
        line << sources_[took.source]->name()
             << (took.unit.kind == VictimKind::Conversion    ? " conversion"
                 : took.unit.kind == VictimKind::PassCommand ? " pass command"
                 : took.unit.kind == VictimKind::LaneStart   ? " lane start"
                 : took.unit.cls == ReclaimClass::InUse      ? " in use"
                                                             : "");
      line << ' ' << double(took.bytes) / kMiB;
    }
  }
  line << " (" << double(result.nowBytes) / kMiB << " MiB now, "
       << double(result.laterBytes) / kMiB << " later)";
  log_(line.str());
}

} // namespace splash::engine
