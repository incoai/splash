#pragma once

#include "engine/ReclaimClass.hpp"
#include "engine/Types.hpp"
#include "model/ElasticPool.hpp"
#include "ops/PagedKv.hpp"

#include <array>
#include <compare>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::engine {

struct MemoryGovernorSnapshot;
class RestoredChains;

// The memory layer's policy constants. The working-set margin and the host's
// reserves, which set the limit and gate the start, are EngineMemoryPolicy's.
// What elective allocation leaves free above the host's spare reserve.
inline constexpr uint64_t kHostMarginBytes = 1ULL << 30;
// How long the host's counters take to show what a pass returned.
inline constexpr double kHostSettleMilliseconds = 1000.0;
// How long a pool waits to regrow after a request's claim last went unmet or
// took what costs work later (a store chunk, a pass, a conversion, a unit in
// use), while a claim waits, and after the pool last grew or tried to: a
// chunk regrown sooner goes back to the next claim, and each cycle costs a
// refill and a compaction. 30 s covers the 1-29 s between a regrow and its
// retake under churn. Under a macOS warning only a unit the cache funds
// whole regrows.
inline constexpr double kRegrowQuietSeconds = 30.0;
// How long a cached unit reached that frees memory only once its copy lands
// keeps a temporary tier copying ahead: claims at the engine's limit come
// in runs, and while one lasts the next claim finds such units copied
// instead of waiting for their writes. A run ends after 60 s with none.
inline constexpr double kCopyDemandSeconds = 60.0;

struct AccountId final {
  uint8_t value = 0;
  friend bool operator==(AccountId, AccountId) = default;
};
inline constexpr AccountId kFixedAccount{0};
// For status only: nothing branches on an account's role or name.
enum class AccountRole : uint8_t { Fixed, Kv, States, Pool };

// Why an allocation is made: decides the host's rule (but for what a lane
// needs, ClaimSpec::laneNeeds) and what free() may take.
enum class ClaimClass : uint8_t { Floor, InService, Start, Optional, Growth, Host };
// Whether the claimant can wait for memory a transfer returns.
enum class NeedTiming : uint8_t { CanWait, Immediate };

// Who memory goes to first, the larger first: Engine::yieldsBefore inverted,
// then the earlier submission (Engine::laneToYield's tie).
struct ClaimRank final {
  uint8_t priority = 0;          // 2 - RequestPriority: Foreground 2, Background 0
  bool decoding = false;         // at equal priority a decoding lane outranks a prefill
  uint64_t completedTokens = 0;  // then more completed work
  uint64_t earliness = 0;        // then earlier submission: UINT64_MAX - Scheduler::submissionOrder
  auto operator<=>(const ClaimRank &) const = default;
};

// Everything the caller knows about why it allocates, passed down explicitly.
struct ClaimSpec final {
  ClaimClass cls = ClaimClass::InService;
  uint64_t requestId = 0;                       // 0: the engine's own, a pool's growth, the host
  ClaimRank rank;
  ReclaimClass reach = ReclaimClass::Ordinary;  // InUse: may take what unfinished requests use
  NeedTiming timing = NeedTiming::CanWait;
  // May take the command between its passes, as the scheduler's decision
  // that the claimant's work supersedes it says.
  bool supersedes = false;
  // Its own lane's KV growth (Engine::growLane): the one claim that may
  // convert its lane to Spill.
  bool kvGrowth = false;
  // false: it takes only the units that cost nothing (the idle buffers) and
  // what lands, for memory its work can go without.
  bool takesUnits = true;
  // The tier that decides what it may take from lanes: no pool unit while a
  // lane of a higher priority is in service (pools serve every lane); the
  // rank still orders held room. A claim of no lane (the host, a pool's
  // growth) takes Foreground.
  RequestPriority priority = RequestPriority::Foreground;
  // An Optional claim for memory its lane in service takes anyway and only
  // asks for sooner: the KV past a prefill command's first state boundary,
  // which the lane's next command takes should this one end there. The host
  // judges it as the lane's InService growth (by the margin, not the band);
  // refused, it takes the in-place units its class may take and passes the
  // margin once none meets the refusal (MemoryGovernor::free).
  bool laneNeeds = false;
};

struct ClaimId final {
  uint32_t value = 0;
  explicit operator bool() const noexcept { return value != 0; }
  friend bool operator==(ClaimId, ClaimId) = default;
};

// What a refused claim lacks. bytes and hostRefused come from the admission
// refused inside its scope. The caller sets pages and stateBuffers for needs
// met in place (TokenAdmission::additionalPages, Model::statesToActivate),
// and for a KV need keptPages: the free pages of each other form its attempt
// takes, which units taken for it keep as they keep `form`'s
// (TokenAdmission::keptPages, Cache::walkFor).
struct Need final {
  uint64_t bytes = 0;
  bool hostRefused = false;
  uint32_t pages = 0;
  kv::PageForm form = kv::PageForm::Full;
  uint32_t stateBuffers = 0;
  std::array<uint32_t, 3> keptPages{};
  friend bool operator==(const Need &, const Need &) = default;
};

enum class FreeOutcome : uint8_t { Met, Pending, Short };
struct FreeResult final {
  FreeOutcome outcome = FreeOutcome::Short;
  uint64_t nowBytes = 0;     // left memory now (extents, chunks, buffers)
  uint64_t laterBytes = 0;   // leave memory when transfers it started land
  // The free KV pages of the need's form it gave, now or as its transfers
  // land, and the pooled state buffers it freed: a stepwise unit's step
  // (Victim::stepwise), and free()'s units.
  uint32_t pages = 0;
  uint32_t stateBuffers = 0;

  // It gave something: met, or memory, pages or buffers.
  [[nodiscard]] bool gave() const noexcept {
    return outcome == FreeOutcome::Met || nowBytes || laterBytes || pages || stateBuffers;
  }
};

// What taking a unit gives up besides its bytes, which the claim's class
// decides whether it may: memory nothing needs now (a cached unit, a
// buffer), a Resident lane's KV in memory (its conversion to Spill), or the
// command between its passes.
enum class VictimKind : uint8_t { Memory, Conversion, PassCommand };

// One unit a source would give up, in the source's own order.
struct Victim final {
  uint64_t key = 0;             // the source's handle (block, state block, request id, pool unit)
  uint64_t bytes = 0;           // device bytes it frees (an extent it empties, a chunk, buffers)
  uint32_t pages = 0;           // free KV pages of `form` it leaves in kept extents (in-place needs)
  kv::PageForm form = kv::PageForm::Full;
  uint32_t stateBuffers = 0;    // pooled state buffers it frees (in-place needs; bytes, setBufferBytes)
  bool later = false;           // frees on a landing (a demotion, a state write, a conversion's copies)
  bool certainReuse = false;    // in use, the newest resume point, a chain a queued request restores
  // false: a unit that costs nothing, taken before any count, whose
  // bytes are at least what peek gives (available() counts them)
  bool sized = true;
  // Taken a step at a time (the cache's reclaims): each take makes one step
  // and reports what it freed, pages and buffers too. Any other unit goes
  // whole in one take, at the pages and buffers peek gave.
  bool stepwise = false;
  ReclaimClass cls = ReclaimClass::Ordinary;
  VictimKind kind = VictimKind::Memory;
};
struct VictimQuery final {
  const ClaimSpec *claim = nullptr;
  Need need;
  bool critical = false;        // macOS critical: InUse and certain-reuse units for Host claims too
  // Chains every walk keeps: the one a start restores, which its placement
  // leaves out of the need it asks about (Engine::placement).
  const RestoredChains *kept = nullptr;
};

// A source of the units free() takes, on the engine thread between commands.
// peek changes nothing and gives the unit after `after` in the source's
// order with at most what taking it frees, never less: free() takes nothing
// unless what it peeked covers the need. take takes the unit, or one step of
// it where the source frees it a step at a time (the cache's reclaims), and
// reports what that freed: Met, or Pending on the transfers it started, when
// it took something; else Pending while a transfer in flight holds the unit,
// Short when nothing of it is left. A later unit's bytes include what goes
// only once another of its transfers lands: take starts what can start now,
// and the waiting claim takes the rest on its retries. A key may name its
// unit only by its place in the source's order; a conversion's key is its
// lane's request id. Taking a unit leaves the units before it in place and
// those after it in order: free() peeks again from the unit before it. Units
// that cost nothing and are sized only when taken come first in a source's
// order.
class VictimSource {
public:
  virtual ~VictimSource() = default;
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  [[nodiscard]] virtual std::optional<Victim> peek(const VictimQuery &query, const Victim *after) const = 0;
  virtual FreeResult take(const VictimQuery &query, const Victim &victim) = 0;
};

// What transfers free when they land (Cache::returning). Landing: in flight
// now. Owed: copies Spill requests owe and have not started. Restores free
// nothing and are not here.
struct Returning final {
  // By kv::PageForm: demotions, and Spill landings as far as free Index
  // pages take them.
  std::array<uint32_t, 3> landingPages{};
  // Full: owed copies of pages that spill once copied, all of them (the
  // Full extents they empty fund their Index pages).
  std::array<uint32_t, 3> owedPages{};
  // The extents the landing pages complete (compaction), and the state
  // write's buffers.
  uint64_t landingBytes = 0;
  uint64_t owedBytes = 0;  // the extents the owed pages complete
  uint32_t transfers = 0;
  double oldestMilliseconds = 0.0;
};

// What the ledger knows of an open claim.
struct ClaimState final {
  ClaimSpec spec;
  // What the latest refusal in its scope recorded (refused): the need of
  // the allocation it attempts now.
  Need refusal;
  // The need, in-place part included, of its latest free() that left it
  // waiting. A waiting claim holds it until the request is admitted or its
  // lane runs, a free() finds it short, or its resource wait ends
  // (Engine::syncClaims). A need met for one of the request's allocations
  // does not end its wait for another.
  Need need;
  bool waiting = false;
  double waitingSinceSeconds = 0.0;
  // Its request waits to retry an allocation the host refused
  // (Engine::syncClaims): that refusal counts in the host's deficit until
  // the wait ends. It holds no room, since the host, not the engine's
  // limit, refused it.
  bool hostRefusedWaiting = false;
  // The host refused it after in-place units could not meet it: the
  // allocation's retries pass the host with each of their reservations (a
  // Floor or InService claim, or an Optional one its lane needs) until the
  // allocation ends (Engine::allocate), since nothing left in place would
  // meet it and waiting would stall the work in service. A growth pass also
  // sets it on a unit it admitted on its net change, so that allocation
  // passes the host (MemoryGovernor::growthHost).
  bool hostExempt = false;
  // What its latest free() decided.
  std::optional<FreeOutcome> outcome;
  // The decisions its lines logged since its last grant, by need: a retry
  // back to one of them logs nothing.
  struct Logged final {
    Need need;
    uint8_t decisions = 0;
  };
  std::vector<Logged> logged;
};

// The host's side of one growth pass, from the governor. admits: whether the
// host admits a pool's growth of `bytes`, `funded` of it by cache units the
// same try takes (the Growth rule at the pass's sample, on the net change).
// paced: a serving pass (one unit after kRegrowQuietSeconds), not the
// start's (every unit the room and the host admit). grow: one unit of the
// pool under the governor's admission inside Scope{claim}.
struct GrowthHost final {
  std::function<bool(uint64_t bytes, uint64_t funded)> admits;
  bool paced = false;
  std::function<metal::AllocationResult(ClaimId claim, model::ElasticPool &pool)> grow;
};

// The books and the one rule. No Metal, no host: the governor supplies the
// charged bytes and the host's decisions.
class MemoryLedger final {
public:
  MemoryLedger();
  ~MemoryLedger();
  MemoryLedger(const MemoryLedger &) = delete;
  MemoryLedger &operator=(const MemoryLedger &) = delete;

  // Registration (startup; stable for the process). An account's current
  // bytes are pulled from its owner. Units tie by stage, then to the source
  // registered first, then to the pools.
  AccountId addAccount(std::string name, AccountRole role, std::function<uint64_t()> currentBytes);
  AccountId addPool(model::ElasticPool &pool);   // account + victim source; name = PoolState::name
  // The cache, the engine's lane victims: each source once, whoever
  // registers it again (two engines over one cache), until it is removed.
  void addSource(VictimSource &source);
  void removeSource(VictimSource &source) noexcept;
  // What transfers return when they land: all of them (Cache::returning),
  // and those of some requests' own pages, the extents their owed copies
  // complete beside every landing (Cache::returningFor).
  void setReturning(std::function<Returning()> all,
                    std::function<Returning(std::span<const uint64_t> requestIds)> ofRequests);
  // The bytes the model's pool releases of `states` more state buffers
  // (Model::idleBufferBytes beyond its value at none): what the buffers a
  // unit frees give a need of bytes once they leave the pool.
  void setBufferBytes(std::function<uint64_t(uint32_t states)> bytes);
  // The bytes of a KV page of each kv::PageForm (Cache::bytesPerPage), in
  // which the bytes waiting claims hold are held as pages (heldPages).
  void setPageBytes(const std::array<uint64_t, 3> &bytes) noexcept;
  // The control clock the ledger dates waits, copy demand and regrowth by.
  void setNow(double nowSeconds) noexcept;
  // The lanes in service: their projected growth, which pools' growth leaves
  // free, and the highest priority among them.
  void setLanes(uint64_t demandBytes, std::optional<RequestPriority> highest) noexcept;
  [[nodiscard]] uint64_t laneDemandBytes() const noexcept { return laneDemandBytes_; }
  // Where the ledger writes one line per decision: the claimant, what it
  // took, the bytes and the outcome.
  void setDecisionLog(std::function<void(std::string_view)> log);

  // Claims.
  [[nodiscard]] ClaimId open(const ClaimSpec &spec);
  void update(ClaimId claim, const ClaimSpec &spec);
  void refused(ClaimId claim, const Need &need);
  void setWaiting(ClaimId claim, bool waiting);
  void setHostRefusedWaiting(ClaimId claim, bool waiting);
  void setHostExempt(ClaimId claim, bool exempt);
  void close(ClaimId claim) noexcept;
  [[nodiscard]] const ClaimState &claim(ClaimId claim) const;
  // Every open claim by its id, for the checks of the ledger's relations.
  [[nodiscard]] const std::map<uint32_t, ClaimState> &claims() const noexcept { return claims_; }
  // What admission holds against an allocation in `claim`'s scope: waiting
  // claims ranked above it, and for a Growth claim the lanes' projected
  // growth.
  [[nodiscard]] uint64_t heldBytes(ClaimId claim) const;
  // The free KV pages of each form held against a page handed out in
  // `claim`'s scope: the pages waiting claims ranked above it lack, plus the
  // bytes they lack beyond `roomBytes` (room admission already holds for
  // them) in that form's pages, since a page from a kept extent's free pages
  // takes room they wait for.
  [[nodiscard]] std::array<uint32_t, 3> heldPages(ClaimId claim, uint64_t roomBytes) const;
  // Whether a claim waits; the bytes the host refused the claims that wait
  // or whose requests wait to retry.
  [[nodiscard]] bool claimWaiting() const noexcept;
  [[nodiscard]] uint64_t hostRefusedWaitingBytes() const noexcept;
  // The units the pools gave back since the engine started.
  [[nodiscard]] uint64_t poolReleases() const noexcept;
  // The largest unit a pool grows by next: the host's band reaches past its
  // margin by it.
  [[nodiscard]] uint64_t largestGrowthUnit() const;
  [[nodiscard]] Returning returning() const;
  // Whether, within kCopyDemandSeconds, a free() reached a cached unit that
  // frees memory only once its copy lands (took it so, or skipped it for an
  // Immediate claim, a pool's growth among them): a temporary tier's copies
  // ahead then pay.
  [[nodiscard]] bool copyDemand() const noexcept;
  // The bytes the last paced growth lacked beyond its room, which the cache
  // did not give at once; 0 once a pool grew, a request's claim waits or no
  // pool is below its cap. The copies ahead a temporary tier makes for it
  // let a later growth take that cache (Engine::copyDemandBytes).
  [[nodiscard]] uint64_t growthLackBytes() const noexcept { return growthLackBytes_; }

  // The rule: what is taken for a claim, what is there for one, what grows.
  // `critical`: macOS's critical pressure; `kept`: the chains a walk for
  // what is there keeps (VictimQuery::kept). `exempts`: a need the host
  // refused that is short is met by the host exemption (hostExempt).
  [[nodiscard]] FreeResult free(ClaimId claim, const Need &need, uint64_t freeBytes, bool critical,
                                bool exempts = false);
  [[nodiscard]] uint64_t available(const ClaimSpec &spec, const Need &need, uint64_t freeBytes, bool critical,
                                   const RestoredChains *kept = nullptr) const;
  // An allocation of the claim was granted: what it lacks next logs afresh
  // (free).
  void granted(ClaimId claim);
  // One growth pass. True when a pool grew.
  bool grow(uint64_t freeBytes, const GrowthHost &host);

  // The accounts, owed bytes and claims.
  void fill(MemoryGovernorSnapshot &snapshot, uint64_t chargedBytes) const;

private:
  struct Account;
  struct Lacking;
  struct Run;
  class Order;

  // The needs of waiting claims ranked above `rank`.
  [[nodiscard]] uint64_t heldByClaims(const ClaimRank &rank, std::optional<uint32_t> self) const noexcept;
  // What admission holds against `spec`'s claim (heldBytes).
  [[nodiscard]] uint64_t heldFor(const ClaimSpec &spec, std::optional<uint32_t> self) const noexcept;
  // Whether the pools hold off growing (grow): a request's claim waits for
  // memory, or its request to retry what the host refused it, and takes the
  // memory that appears. A claim of no request never holds it (requests
  // first).
  [[nodiscard]] bool holdsGrowth() const noexcept;
  // What `added` state buffers give a need of bytes as they leave the
  // model's pool, beyond the `before` ones counted already (setBufferBytes).
  [[nodiscard]] uint64_t bufferBytes(const Lacking &lack, uint32_t before, uint32_t added) const;
  // The most the units the claim's class may take give it, counted in the
  // one order without taking any, up to the lack.
  [[nodiscard]] Run bound(const ClaimSpec &spec, const VictimQuery &query, const Lacking &lack) const;
  // Takes those units in that order, one at a time, until what they freed
  // now and later covers the lack, or none is left.
  [[nodiscard]] Run take(const ClaimSpec &spec, const VictimQuery &query, const Lacking &lack);
  // Takes each source's units that cost nothing and are sized only when
  // taken, until one frees nothing; the bytes they freed.
  uint64_t takeUnsized(const VictimQuery &query);

  void log(const ClaimSpec &spec, const Need &need, const Run *run, const FreeResult &result,
           bool exempted) const;

  std::vector<Account> accounts_;
  std::vector<VictimSource *> sources_;
  std::map<uint32_t, ClaimState> claims_;
  uint32_t nextClaim_ = 1;
  std::function<Returning()> returning_;
  std::function<Returning(std::span<const uint64_t>)> returningFor_;
  std::function<uint64_t(uint32_t)> bufferBytes_;
  std::array<uint64_t, 3> pageBytes_{};
  uint64_t laneDemandBytes_ = 0;
  std::optional<RequestPriority> lanePriority_;
  std::function<void(std::string_view)> log_;
  double nowSeconds_ = 0.0;
  std::optional<double> uncopiedReachedSeconds_;
  // Since when no request's claim went unmet or took what costs work later,
  // and no pool grew or tried to (grow).
  double quietSinceSeconds_ = 0.0;
  uint64_t growthLackBytes_ = 0;
};

} // namespace splash::engine
