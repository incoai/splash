#pragma once

// The engine's allocation template and the helpers both of its sources use:
// Engine.cpp and EngineMemory.cpp include it, nothing else does.

#include "engine/Engine.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace splash::engine {

// A state's buffers, taken from the model's pool or one admission
// (Model::reserveState, snapshot), or the refusal's cause.
struct StateBuffers final {
  bool ok = false;
  metal::AllocationFailure allocationFailure = metal::AllocationFailure::None;
  [[nodiscard]] bool granted() const noexcept { return ok; }
};

// What the request's KV blocks are cached under beside its tokens.
inline PromptIdentity promptIdentity(const EngineRequest &request) noexcept {
  return {request.images, request.cacheSalt};
}

// A refusal for memory, which what the ledger frees or the host's recovery
// may end. A KV admission the pages on their way back cover is one too: the
// ledger decides whether the claim waits for them (MemoryLedger::free).
inline bool memoryDenied(const StateAdmission &admission) noexcept {
  return admission.failure == StateFailure::MemoryPressure;
}

inline bool memoryDenied(const TokenAdmission &admission) noexcept {
  return admission.failure != TokenAdmissionFailure::None;
}

inline bool memoryDenied(const model::ModelMemoryAdmission &admission) noexcept {
  return !admission.granted();
}

inline bool memoryDenied(const StateBuffers &buffers) noexcept { return !buffers.granted(); }

// The part of a refused need that memory already allocated meets in place:
// the pages of the refused form it lacks beyond the free ones, with those
// claims ranked above it hold (`held`) left free, as a page is handed out
// only beyond them (KvPool::acquirePages), of a KV need; the pooled state
// buffers a lane's activation lacks, one state's buffers.
inline std::optional<Need> inPlace(const TokenAdmission &admission, const std::array<uint32_t, 3> &held) {
  const uint64_t wanted = uint64_t{admission.additionalPages} + held[static_cast<size_t>(admission.form)];
  return Need{.pages = static_cast<uint32_t>(
                  std::min<uint64_t>(wanted > admission.availablePages ? wanted - admission.availablePages : 0,
                                     std::numeric_limits<uint32_t>::max())),
              .form = admission.form,
              .keptPages = admission.keptPages,
              .kv = true};
}

inline std::optional<Need> inPlace(const model::ModelMemoryAdmission &, const std::array<uint32_t, 3> &) {
  return Need{};
}

inline std::optional<Need> inPlace(const StateBuffers &, const std::array<uint32_t, 3> &) {
  return Need{.stateBuffers = 1};
}

// What a KV target lacked when it could not be allocated.
inline std::string pageShortfall(const TokenAdmission &admission) {
  return "additional_pages=" + std::to_string(admission.additionalPages) +
         ", free_pages=" + std::to_string(admission.availablePages);
}

template <class Attempt>
auto Engine::allocate(Attempt &&attempt, const ClaimSpec &spec, Refusal refusal)
    -> Allocation<std::invoke_result_t<Attempt &>> {
  using Admission = std::invoke_result_t<Attempt &>;
  MemoryGovernor &governor = this->governor();
  MemoryLedger &ledger = governor.ledger();
  // A request's allocations are its one claim's, the engine's own a claim
  // of this allocation alone.
  ClaimHandle transient;
  ClaimHandle &claim = spec.requestId ? request(spec.requestId).claim : transient;
  if (claim.id) {
    ledger.update(claim.id, spec);
  } else {
    claim.governor = &governor;
    claim.id = ledger.open(spec);
  }
  ledger.setLanes(laneDemandBytes(), highestLanePriority());
  // Each attempt runs in the claim's scope, from a need the scope's refusal
  // records afresh; what a waiting claim holds stays held.
  const auto tryOnce = [&] {
    ledger.refused(claim.id, {});
    const MemoryGovernor::Scope scope(governor, claim.id);
    return attempt();
  };
  Allocation<Admission> result{tryOnce(), {}};
  Admission &admission = result.admission;
  Denial &denial = result.denial;
  // Each refusal asks the ledger to free what the claim lacks, and the
  // attempt runs again once that is met or memory left now. The loop stops
  // at a refusal the ledger cannot help: one that lacks nothing it counts,
  // one the host refuses after exempting the claim (critical pressure), or a
  // need the ledger already freed for once in this allocation. The claim
  // then waits if the ledger's latest decision was to wait.
  std::vector<Need> met;
  const auto metBefore = [&](const Need &need) { return std::ranges::find(met, need) != met.end(); };
  while (refusal != Refusal::SettledOnce && memoryDenied(admission)) {
    std::optional<Need> part;
    if constexpr (std::is_same_v<Admission, StateAdmission>) {
      const std::optional<uint32_t> states = model_.statesToActivate();
      part = Need{.stateBuffers = states.value_or(0), .unheldStateBuffer = !states};
    } else {
      part = inPlace(admission, governor.heldPages(claim.id));
    }
    const Need need = governor.need(claim.id, part);
    if ((!need.bytes && !need.hostRefused && !need.pages && !need.lacksStateBuffers()) ||
        (need.hostRefused && ledger.claim(claim.id).hostExempt) ||
        metBefore(need))
      break;
    const bool exempt = ledger.claim(claim.id).hostExempt;
    const FreeResult freed = governor.free(claim.id, need);
    ++ledgerChanges_;
    denial.pending = freed.outcome == FreeOutcome::Pending;
    // Past its hold's limit a request's claim waits holding nothing
    // (syncClaims), whatever this retry found.
    if (denial.pending && spec.requestId && request(spec.requestId).resourceWait.holdEnded)
      ledger.setWaiting(claim.id, false);
    // What it freed now, the units that cost nothing among them, the
    // attempt may fit in.
    if (freed.outcome != FreeOutcome::Met && !freed.nowBytes) {
      // A conversion frees later, and at once the pages whose copies landed
      // before (Cache::convertToSpill), which one more attempt may fit in.
      if (freed.laterBytes)
        admission = tryOnce();
      break;
    }
    // The host's exemption takes nothing for it.
    result.freed = result.freed || exempt || !ledger.claim(claim.id).hostExempt;
    met.push_back(need);
    signalResourceProgress();
    admission = tryOnce();
  }
  // The host's exemption covers this allocation's retries alone.
  if (ledger.claim(claim.id).hostExempt)
    ledger.setHostExempt(claim.id, false);
  if (admission.granted()) {
    ledger.granted(claim.id);
  } else {
    denial.allocationFailure = admission.allocationFailure;
    if constexpr (std::is_same_v<Admission, TokenAdmission>) {
      denial.pages = admission.additionalPages;
      denial.form = admission.form;
    }
    // Memory on its way back ends the shortage without the residents, and a
    // refusal its caller settles leaves no work waiting.
    if (memoryDenied(admission) && !denial.pending && refusal == Refusal::Waits)
      allocationFailed_ = true;
  }
  // A grant leaves a request's wait as it was: it ends once the request is
  // admitted or its lane runs (admit, prepare), or its free() finds it
  // short.
  return result;
}

} // namespace splash::engine
