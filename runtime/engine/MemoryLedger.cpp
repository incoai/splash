#include "engine/MemoryLedger.hpp"

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

uint64_t less(uint64_t a, uint64_t b) noexcept { return a > b ? a - b : 0; }

uint64_t plus(uint64_t a, uint64_t b) noexcept {
  return a <= std::numeric_limits<uint64_t>::max() - b ? a + b : std::numeric_limits<uint64_t>::max();
}

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

// What a claim lacks, in each measure that can meet it: device bytes beyond
// the room it has, free pages of a form in kept extents, pooled state
// buffers.
struct Lacking final {
  uint64_t bytes = 0;
  uint32_t pages = 0;
  kv::PageForm form = kv::PageForm::Full;
  uint32_t buffers = 0;

  [[nodiscard]] bool none() const noexcept { return !bytes && !pages && !buffers; }
  // Whether a unit meets any of the need, in any measure.
  [[nodiscard]] bool meets(const Victim &victim) const noexcept {
    return (bytes && victim.bytes) || (pages && victim.form == form && victim.pages) ||
           (buffers && victim.stateBuffers);
  }
};

} // namespace

struct MemoryLedger::Account final {
  std::string name;
  AccountRole role = AccountRole::Fixed;
  std::function<uint64_t()> current;
  model::ElasticPool *pool = nullptr;
  uint64_t grows = 0;
  uint64_t shrinks = 0;
};

// One unit a dry run chose: a source's (by index, after the units of the
// source it passed over since the source's previous choice), or a pool's
// unit (by account). A unit the run carries to a later one of its source is
// a choice of its own, before that one.
struct MemoryLedger::Choice final {
  size_t source = 0;
  uint32_t passed = 0;
  std::optional<AccountId> pool;
  Victim victim;
};

struct MemoryLedger::DryRun final {
  std::vector<Choice> chosen;
  uint64_t bytes = 0;
  uint32_t pages = 0;
  uint32_t buffers = 0;
  // Of its bytes, those its units' state buffers give once they leave the
  // model's pool.
  uint64_t bufferBytes = 0;
  bool covered = false;
  bool later = false;
  // It reached a cached unit that frees memory only once its copy lands.
  bool uncopied = false;
  // It chose a unit whose loss costs work later: a store chunk, the command
  // between its passes, a conversion, a unit in use.
  bool costly = false;
};

MemoryLedger::MemoryLedger() {
  accounts_.push_back(Account{.name = "fixed", .role = AccountRole::Fixed});
}

MemoryLedger::~MemoryLedger() = default;

AccountId MemoryLedger::addAccount(std::string name, AccountRole role, std::function<uint64_t()> currentBytes) {
  if (accounts_.size() > std::numeric_limits<uint8_t>::max())
    throw std::logic_error("the memory ledger has no account id left");
  if (role == AccountRole::Fixed || role == AccountRole::Pool || !currentBytes)
    throw std::invalid_argument("an account the ledger pulls needs its current bytes");
  accounts_.push_back(Account{.name = std::move(name), .role = role, .current = std::move(currentBytes)});
  return {static_cast<uint8_t>(accounts_.size() - 1)};
}

AccountId MemoryLedger::addPool(model::ElasticPool &pool) {
  if (accounts_.size() > std::numeric_limits<uint8_t>::max())
    throw std::logic_error("the memory ledger has no account id left");
  accounts_.push_back(Account{.name = std::string(pool.state().name), .role = AccountRole::Pool, .pool = &pool});
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

void MemoryLedger::setBufferBytes(std::function<uint64_t(uint32_t)> bytes) { bufferBytes_ = std::move(bytes); }

void MemoryLedger::setPageBytes(const std::array<uint64_t, 3> &bytes) noexcept { pageBytes_ = bytes; }

void MemoryLedger::setNow(double nowSeconds) noexcept { nowSeconds_ = nowSeconds; }

void MemoryLedger::setLanes(uint64_t demandBytes, std::optional<RequestPriority> highest) noexcept {
  laneDemandBytes_ = demandBytes;
  lanePriority_ = highest;
}

void MemoryLedger::setDecisionLog(std::function<void(std::string_view)> log) { log_ = std::move(log); }

ClaimId MemoryLedger::open(const ClaimSpec &spec) {
  const ClaimId id{nextClaim_++};
  if (!nextClaim_)
    nextClaim_ = 1;
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
  state.outcome.reset();
  state.logged.clear();
}

void MemoryLedger::close(ClaimId claim) noexcept { claims_.erase(claim.value); }

const ClaimState &MemoryLedger::claim(ClaimId claim) const { return claims_.at(claim.value); }

uint64_t MemoryLedger::heldByClaims(const ClaimRank &rank, std::optional<uint32_t> self) const noexcept {
  uint64_t held = 0;
  for (const auto &[id, state] : claims_)
    if (state.waiting && (!self || id != *self) && state.spec.rank > rank)
      held = plus(held, state.need.bytes);
  return held;
}

uint64_t MemoryLedger::heldFor(const ClaimSpec &spec, std::optional<uint32_t> self) const noexcept {
  const uint64_t lanes = spec.cls == ClaimClass::Growth ? laneDemandBytes_ : 0;
  return plus(lanes, heldByClaims(spec.rank, self));
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
  const uint64_t bytes = less(heldByClaims(self.spec.rank, claim.value), roomBytes);
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
      bytes = plus(bytes, state.refusal.bytes);
  return bytes;
}

uint64_t MemoryLedger::poolReleases() const noexcept {
  uint64_t units = 0;
  for (const Account &account : accounts_)
    units = plus(units, account.shrinks);
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

// The rule's dry run: units in a fixed order until the need is met, the
// least lost first. Memory nothing needs now (the cache's units, by its
// walk); the units of pools that cost only their rebuild; the claimant's own
// conversion to Spill and the command between its passes; the other pools'
// units (the expert store's chunks, top first) down to their floors, which
// cost reads while they stay away; last what another request uses, since its
// state in use costs it a re-prefill of its whole prompt while a store chunk
// costs only reads until it regrows. Within a stage the source registered
// first goes first, then the pools; each source's own order is kept. A unit
// the claim's class may not take is skipped, not stopped at. A unit's state
// buffers count for a need of bytes as what the model's pool releases of
// them beyond the units chosen before (setBufferBytes). A unit whose buffers
// the pool keeps (they refill those a lane starts from) meets no need of
// bytes alone: it is carried to the next unit chosen from its source and
// counted with it, since the buffers after it count only beside its own and
// the cache weighs each unit as a step from where the units before it leave
// the cache. A unit that meets none of the need otherwise, or that lands
// later for an Immediate claim, is skipped (VictimSource: taking a later
// unit leaves it where it was).
MemoryLedger::DryRun MemoryLedger::dryRun(const ClaimSpec &spec, const VictimQuery &query, uint64_t lackingBytes,
                                          uint32_t lackingPages) const {
  const Lacking lack{.bytes = lackingBytes,
                     .pages = lackingPages,
                     .form = query.need.form,
                     .buffers = query.need.stateBuffers};
  // A need the host refused runs as an Immediate claim that reaches no
  // further than ordinary units: the host, not the engine's limit, refused
  // it, so nothing that lands later or that a request uses helps it.
  const bool immediate = spec.timing == NeedTiming::Immediate || query.need.hostRefused;
  const bool hostCritical = spec.cls == ClaimClass::Host && query.critical;
  const bool inService = spec.cls == ClaimClass::InService;
  // What each class may take. Immediate claims and host-refused needs take
  // only what frees now: no store chunk, pass, conversion or unit in use.
  // Pass and conversion only for an in-service claim that can wait: the pass
  // when it supersedes the command, the conversion of its own lane for that
  // lane's KV growth. Units in use for such a claim that reaches them, and
  // for the host at critical pressure. Certain reuse never for growth or the
  // host below critical pressure; Start, Optional and Floor take the newest
  // resume point but never a queued chain, which their walk keeps
  // (Cache::walkFor).
  const auto mayTake = [&](const Victim &victim) {
    if (victim.cls == ReclaimClass::InUse &&
        !((inService && !immediate && spec.reach == ReclaimClass::InUse) || hostCritical))
      return false;
    if (victim.certainReuse && (spec.cls == ClaimClass::Growth || (spec.cls == ClaimClass::Host && !query.critical)))
      return false;
    switch (victim.kind) {
    case VictimKind::Memory: return true;
    case VictimKind::Conversion: return inService && !immediate && spec.kvGrowth;
    case VictimKind::PassCommand: return inService && !immediate && spec.supersedes;
    }
    return false;
  };
  // Pool units above floors (the expert store's chunks) only for an
  // in-service claim that can wait and the host at critical pressure: a
  // chunk costs reads while it stays away, so nothing that only starts or
  // helps work takes it, and a host claim would hand a whole chunk to the
  // lanes' KV for a far smaller margin. The units of pools that cost only
  // their rebuild (PoolState::rebuildOnly), such as the hot block cache's
  // extents when no lane reads above its floor, for in-service work, a
  // start, optional work and the host; never for a weights restore or a
  // pool's growth. None for a claim below a lane in service.
  const bool belowALane = lanePriority_ && *lanePriority_ < spec.priority;
  const bool poolUnits = !belowALane && ((inService && !immediate) || hostCritical);
  const bool rebuiltUnits =
      poolUnits || (!belowALane && (inService || spec.cls == ClaimClass::Start || spec.cls == ClaimClass::Optional ||
                                    spec.cls == ClaimClass::Host));

  struct Head final {
    std::optional<Victim> victim;
    size_t source = 0;
    uint32_t passed = 0;  // the source's units passed over since its last choice or carried unit
    // The units it carries to its next choice.
    std::vector<Choice> carried;
    std::optional<AccountId> pool;
    // A pool's reading, and the bytes above its floor the run has taken.
    model::PoolState state;
    uint64_t taken = 0;
    uint32_t units = 0;
  };
  std::vector<Head> heads;
  heads.reserve(sources_.size() + accounts_.size());
  const auto poolUnit = [&](const Head &head) -> std::optional<Victim> {
    const uint64_t above = less(head.state.currentBytes, head.state.floorBytes);
    if (!head.state.releaseUnitBytes || head.taken >= above)
      return std::nullopt;
    // Each unit at its own bytes where the pool lists them (the store's
    // chunks, top first), else at the next one's.
    const std::vector<uint64_t> &units = head.state.releaseUnits;
    const uint64_t bytes =
        std::min(head.units < units.size() ? units[head.units] : head.state.releaseUnitBytes, above - head.taken);
    return Victim{.key = head.units, .bytes = bytes};
  };
  for (size_t index = 0; index < sources_.size(); ++index)
    heads.push_back(Head{.victim = sources_[index]->peek(query, nullptr), .source = index});
  for (size_t index = 1; index < accounts_.size(); ++index) {
    const Account &account = accounts_[index];
    if (!account.pool || !rebuiltUnits)
      continue;
    Head head{.pool = AccountId{static_cast<uint8_t>(index)}, .state = account.pool->state()};
    if (!poolUnits && !head.state.rebuildOnly)
      continue;
    head.victim = poolUnit(head);
    heads.push_back(std::move(head));
  }
  // The head's stage in the order; from kCostlyStage on, a unit whose loss
  // costs work later (DryRun::costly).
  constexpr int kCostlyStage = 2;
  const auto stage = [](const Head &head) {
    if (head.pool)
      return head.state.rebuildOnly ? 1 : 3;
    if (head.victim->kind != VictimKind::Memory)
      return 2;
    return head.victim->cls == ReclaimClass::InUse ? 4 : 0;
  };
  const auto advance = [&](Head &head) {
    if (head.pool) {
      head.taken += head.victim->bytes;
      ++head.units;
      head.victim = poolUnit(head);
    } else {
      head.victim = sources_[head.source]->peek(query, &*head.victim);
    }
  };
  const auto pass = [&](Head &head) {
    ++head.passed;
    advance(head);
  };
  const auto carry = [&](Head &head) {
    head.carried.push_back(
        Choice{.source = head.source, .passed = std::exchange(head.passed, 0), .victim = *head.victim});
    advance(head);
  };
  // The head's unit with the units it carries, as one.
  const auto withCarried = [](const Head &head) {
    Victim unit = *head.victim;
    for (const Choice &carried : head.carried) {
      const Victim &each = carried.victim;
      unit.bytes = plus(unit.bytes, each.bytes);
      if (each.form == unit.form)
        unit.pages += each.pages;
      unit.stateBuffers += each.stateBuffers;
      unit.later = unit.later || each.later;
    }
    return unit;
  };

  DryRun run;
  const auto counted = [&](const Victim &victim) {
    Victim unit = victim;
    if (lack.bytes && victim.stateBuffers && bufferBytes_)
      unit.bytes = plus(unit.bytes, less(bufferBytes_(run.buffers + victim.stateBuffers), bufferBytes_(run.buffers)));
    return unit;
  };
  for (;;) {
    Head *best = nullptr;
    for (Head &head : heads) {
      // Skip what this claim may not take, carry a unit whose buffers the
      // pool keeps, and skip what meets none of its need otherwise; keep a
      // unit that lands later in an Immediate claim's way, to note that the
      // run reached it.
      while (head.victim) {
        if (!mayTake(*head.victim))
          pass(head);
        else if (lack.meets(counted(withCarried(head))))
          break;
        else if (lack.bytes && head.victim->stateBuffers && !(immediate && head.victim->later))
          carry(head);
        else
          pass(head);
      }
      if (head.victim && (!best || stage(head) < stage(*best)))
        best = &head;
    }
    if (!best)
      break;
    const Victim units = withCarried(*best);
    const Victim victim = counted(units);
    if (victim.later && victim.kind == VictimKind::Memory)
      run.uncopied = true;
    if (victim.later && immediate) {
      pass(*best);
      continue;
    }
    if (!spec.takesUnits)
      break;
    run.bytes = plus(run.bytes, victim.bytes);
    run.bufferBytes = plus(run.bufferBytes, victim.bytes - units.bytes);
    if (victim.form == lack.form)
      run.pages += victim.pages;
    run.buffers += victim.stateBuffers;
    run.later = run.later || victim.later;
    run.chosen.insert(run.chosen.end(), best->carried.begin(), best->carried.end());
    best->carried.clear();
    run.chosen.push_back(Choice{.source = best->source, .passed = best->passed, .pool = best->pool, .victim = victim});
    run.costly = run.costly || stage(*best) >= kCostlyStage;
    best->passed = 0;
    advance(*best);
    if ((lack.bytes && run.bytes >= lack.bytes) || (lack.pages && run.pages >= lack.pages) ||
        (lack.buffers && run.buffers >= lack.buffers)) {
      run.covered = true;
      break;
    }
  }
  return run;
}

// Takes what a covered dry run chose, in its order. A source's unit (each
// unit it carried first) is found by its place in the source's current
// order, past the units the run passed over, up to a unit no longer offered
// as chosen (VictimSource). A pool's units are released by what the run's
// choices of them still lack, not choice by choice: the run sizes a pool's
// units at the bytes it lists (the store's last live chunk is a remainder,
// the one before it about 1 GiB) or else at the first unit's, so a released
// unit may be larger than its choice and already meet the next one.
FreeResult MemoryLedger::take(const DryRun &run, const VictimQuery &query, bool &whole) {
  FreeResult result{.outcome = run.later ? FreeOutcome::Pending : FreeOutcome::Met};
  whole = true;
  // Each source's last unit the run passed over, which stays in its order.
  std::vector<std::optional<Victim>> passed(sources_.size());
  // Each pool's bytes the run chose so far, and those its releases gave.
  std::vector<uint64_t> chosen(accounts_.size()), released(accounts_.size());
  for (const Choice &choice : run.chosen) {
    if (choice.pool) {
      const size_t index = choice.pool->value;
      Account &account = accounts_[index];
      chosen[index] = plus(chosen[index], choice.victim.bytes);
      if (released[index] >= chosen[index])
        continue;
      const uint64_t bytes = account.pool->release(chosen[index] - released[index]);
      if (bytes)
        ++account.shrinks;
      released[index] = plus(released[index], bytes);
      result.nowBytes = plus(result.nowBytes, bytes);
      continue;
    }
    VictimSource &source = *sources_[choice.source];
    std::optional<Victim> &before = passed[choice.source];
    std::optional<Victim> unit = source.peek(query, before ? &*before : nullptr);
    for (uint32_t over = 0; unit && over < choice.passed; ++over) {
      before = unit;
      unit = source.peek(query, &*before);
    }
    if (!unit || unit->kind != choice.victim.kind ||
        (unit->kind == VictimKind::Conversion && unit->key != choice.victim.key)) {
      whole = false;
      return result;
    }
    const FreeResult taken = source.take(query, *unit);
    result.nowBytes = plus(result.nowBytes, taken.nowBytes);
    result.laterBytes = plus(result.laterBytes, taken.laterBytes);
  }
  return result;
}

uint64_t MemoryLedger::takeUnsized(const VictimQuery &query) {
  uint64_t bytes = 0;
  for (VictimSource *source : sources_) {
    for (std::optional<Victim> unit = source->peek(query, nullptr);
         unit && !unit->sized; unit = source->peek(query, nullptr)) {
      const FreeResult taken = source->take(query, *unit);
      bytes = plus(bytes, taken.nowBytes);
      if (!taken.nowBytes && !taken.laterBytes)
        break;
    }
  }
  return bytes;
}

// The one rule. The units that cost nothing and cannot be sized go first.
// Then a claim that can wait (never the host's, whose pass's target is net
// of what lands) waits when what is landing, with its own lanes' owed
// copies, meets its need (or would, but for claims ranked above it).
// Else the units the dry run reaches first that meet the rest are taken, all
// or none, and it waits for what lands unless they meet it alone. Else it
// waits when every copy in flight or owed would meet it. A claim refused
// bytes lacks what admission asks beyond the room: its need plus the room
// held against it. A need met in place lacks no bytes. A need the host
// refused counts only units that leave memory in place, and the host's claim
// lacks its need beyond the units that cost nothing, since room under the
// engine's limit returns nothing to the host. State buffers the units free
// meet a need of bytes once they leave the model's pool, so the units that
// cost nothing go again (a need in place keeps them). A claim that can
// wait, short only by the room waiting claims ranked above it hold, waits
// as their holds end.
FreeResult MemoryLedger::free(ClaimId claimId, const Need &need, uint64_t freeBytes, bool critical, bool exempts) {
  const uint32_t id = claimId.value;
  ClaimState &claim = claims_.at(id);
  const ClaimSpec &spec = claim.spec;
  const VictimQuery query{.claim = &spec, .need = need, .critical = critical};

  FreeResult result{.nowBytes = takeUnsized(query)};
  const bool host = spec.cls == ClaimClass::Host;
  const uint64_t room = host ? 0 : plus(freeBytes, result.nowBytes);
  // Waiting claims ranked above it hold room it waits for, and the lanes'
  // projected growth room a pool's growth leaves them.
  const uint64_t held = host ? 0 : heldFor(spec, id);
  const uint64_t forClaim = less(room, held);
  const uint64_t lackingBytes = need.hostRefused || !need.bytes ? 0
                                : host ? less(need.bytes, result.nowBytes)
                                       : less(plus(need.bytes, held), room);
  const uint32_t lackingPages = need.pages;
  // The host's claim waits for nothing that lands: its pass's target is net
  // of it already (MemoryGovernor::controlPass).
  const bool canWait = spec.timing == NeedTiming::CanWait && !need.hostRefused && !host;

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
    const uint64_t landing = plus(all.landingBytes, owedToo ? all.owedBytes : own.owedBytes);
    const uint64_t landingPages =
        uint64_t{all.landingPages[form]} + (owedToo ? all.owedPages[form] : own.owedPages[form]);
    if (lackingBytes && (plus(forClaim, landing) >= need.bytes || (!owedToo && plus(room, all.landingBytes) >= need.bytes)))
      return true;
    return lackingPages && (less(landingPages, heldPagesAbove) >= lackingPages ||
                            (!owedToo && all.landingPages[form] >= lackingPages));
  };
  // What lands and its own lanes' owed copies count before victims also in
  // part: the dry run covers only what they do not. The room held above the
  // claim is in what it lacks already.
  const uint64_t creditBytes = plus(all.landingBytes, own.owedBytes);
  const uint64_t creditPages = less(uint64_t{all.landingPages[form]} + own.owedPages[form], heldPagesAbove);

  std::optional<DryRun> run;
  // A refusal of bytes the room covers once the units that cost nothing
  // went is met: the fit is tried again, whatever memory allocated could
  // meet of it in place.
  if (!need.hostRefused && !lackingBytes && (need.bytes || (!lackingPages && !need.stateBuffers))) {
    result.outcome = FreeOutcome::Met;
  } else if (waits(false)) {
    result.outcome = FreeOutcome::Pending;
  } else {
    run = dryRun(spec, query, less(lackingBytes, creditBytes), static_cast<uint32_t>(less(lackingPages, creditPages)));
    if (run->uncopied)
      uncopiedReachedSeconds_ = nowSeconds_;
    if (run->covered) {
      bool whole = true;
      FreeResult taken = take(*run, query, whole);
      // The buffers its states freed leave the model's pool unless they met
      // the need in place: a need of bytes beside buffers no state can give
      // (Model::statesToActivate with no ring idle) is met by their bytes.
      const bool metInPlace = need.stateBuffers && run->buffers >= need.stateBuffers;
      if (run->bufferBytes && !metInPlace)
        taken.nowBytes = plus(taken.nowBytes, takeUnsized(query));
      const bool alone = (lackingBytes && run->bytes >= lackingBytes) ||
                         (lackingPages && run->pages >= lackingPages) ||
                         (need.stateBuffers && run->buffers >= need.stateBuffers);
      // A take that stopped short waits for what it started to land, else
      // leaves what it freed to the claim's retry.
      const FreeOutcome outcome = !whole ? (taken.laterBytes ? FreeOutcome::Pending : FreeOutcome::Short)
                                  : alone ? taken.outcome
                                          : FreeOutcome::Pending;
      result = {outcome, plus(result.nowBytes, taken.nowBytes), taken.laterBytes};
    } else {
      // Short only by the room waiting claims ranked above it hold, it waits
      // for them and takes nothing: Short would end a lone lane as if at its
      // capacity (Engine::judge).
      const bool heldAbove = canWait && lackingBytes && held &&
                             plus(plus(run->bytes, creditBytes), held) >= lackingBytes;
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
  // One line per decision: whatever takes units, and a change of a
  // request's outcome or a need met past the host, but not a retry back to
  // a decision logged for the same need since the claim's last grant: a
  // claim that flips between short and pending as copies land logged 669
  // lines in one second.
  bool changed = spec.requestId && (claim.outcome != result.outcome || exempted);
  if (changed) {
    const auto bit = static_cast<uint8_t>(1U << (exempted ? 3U : static_cast<unsigned>(result.outcome)));
    auto found = std::ranges::find(claim.logged, need, &ClaimState::Logged::need);
    if (found == claim.logged.end())
      claim.logged.push_back({need, bit});
    else if (found->decisions & bit)
      changed = false;
    else
      found->decisions |= bit;
  }
  if ((run && run->covered) || changed)
    log(spec, need, run && run->covered ? &*run : nullptr, result, exempted);
  claim.outcome = result.outcome;
  return result;
}

// free() as a dry run: the units that cost nothing and are sized only when
// taken count as their sources size them now (peek), and for a claim that
// can wait what lands and the copies its own request owes count before
// victims, as free() counts them.
uint64_t MemoryLedger::available(const ClaimSpec &spec, const Need &need, uint64_t freeBytes, bool critical,
                                 const RestoredChains *kept) const {
  const uint64_t enough = need.bytes;
  const bool host = spec.cls == ClaimClass::Host;
  const uint64_t held = host ? 0 : heldFor(spec, std::nullopt);
  uint64_t landing = 0;
  if (!host && spec.timing == NeedTiming::CanWait) {
    landing = returning().landingBytes;
    if (returningFor_ && spec.requestId)
      landing = plus(landing, returningFor_(std::span<const uint64_t>(&spec.requestId, 1)).owedBytes);
  }
  const uint64_t room = host ? 0 : plus(freeBytes, landing);
  if (less(room, held) >= enough)
    return enough;
  const VictimQuery query{.claim = &spec, .need = need, .critical = critical, .kept = kept};
  const uint64_t taken = dryRun(spec, query, less(plus(enough, held), room), 0).bytes;
  return std::min(enough, less(plus(room, taken), held));
}

// Growth into the room beyond what waiting claims and the lanes' projected
// growth hold, a pool at a time in registration order. Nothing grows while a
// request's claim waits for memory, or its request to retry what the host
// refused: the claim takes the memory that appears. A unit takes the room it
// lacks from cache units that free memory now, in the cache's order, as an
// Immediate Growth claim, all or none: never what is in use, of certain
// reuse or needs a write. Growing into free room alone, the lanes' KV cached
// after a burst that shrank the store would keep it at its floor. A dry run
// first finds what the cache funds; the host then judges the unit's net
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
    // weighed: the cache is not walked for it.
    for (uint64_t unit = account.pool->state().growUnitBytes; unit && host.admits(unit, unit);
         unit = account.pool->state().growUnitBytes) {
      const uint64_t room = less(freeBytes, grownBytes);
      const uint64_t lacking = less(plus(unit, heldFor(spec, std::nullopt)), room);
      uint64_t funded = 0;
      bool grown = true;
      if (lacking) {
        const VictimQuery query{.claim = &spec, .need = {.bytes = unit}};
        const DryRun run = dryRun(spec, query, lacking, 0);
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
          freeBytes = plus(freeBytes, taken.nowBytes);
          grown = taken.outcome == FreeOutcome::Met;
        }
        grown = grown && host.grow(claim, *account.pool);
        close(claim);
      }
      if (grown) {
        grownBytes = plus(grownBytes, unit);
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
    others = plus(others, row.currentBytes);
    accounts.push_back(std::move(row));
  }
  const uint64_t fixed = less(chargedBytes, others);
  snapshot.accounts.push_back({.name = accounts_.front().name, .currentBytes = fixed, .capBytes = fixed});
  for (MemoryAccountSnapshot &row : accounts)
    snapshot.accounts.push_back(std::move(row));

  const Returning back = returning();
  snapshot.owed = {.transfers = back.transfers,
                   .bytes = plus(back.landingBytes, back.owedBytes),
                   .oldestMilliseconds = back.oldestMilliseconds};
  snapshot.claims = {};
  std::optional<double> oldest;
  for (const auto &[id, state] : claims_) {
    if (!state.waiting)
      continue;
    ++snapshot.claims.waiting;
    snapshot.claims.bytes = plus(snapshot.claims.bytes, state.need.bytes);
    oldest = std::min(oldest.value_or(state.waitingSinceSeconds), state.waitingSinceSeconds);
  }
  if (oldest)
    snapshot.claims.oldestMilliseconds = std::max(0.0, nowSeconds_ - *oldest) * 1000.0;
  snapshot.owed.toClaimsBytes = std::min(snapshot.owed.bytes, snapshot.claims.bytes);
}

void MemoryLedger::log(const ClaimSpec &spec, const Need &need, const DryRun *run, const FreeResult &result,
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
  line << ": " << outcomeName(result.outcome);
  if (exempted)
    line << " past the host";
  if (run) {
    line << ", took";
    for (const Choice &choice : run->chosen) {
      line << ' ';
      if (choice.pool)
        line << accounts_[choice.pool->value].name << " unit";
      else
        line << sources_[choice.source]->name()
             << (choice.victim.kind == VictimKind::Conversion    ? " conversion"
                 : choice.victim.kind == VictimKind::PassCommand ? " pass command"
                 : choice.victim.cls == ReclaimClass::InUse      ? " in use"
                                                                 : "");
      line << ' ' << double(choice.victim.bytes) / kMiB;
    }
  }
  line << " (" << double(result.nowBytes) / kMiB << " MiB now, "
       << double(result.laterBytes) / kMiB << " later)";
  log_(line.str());
}

} // namespace splash::engine
