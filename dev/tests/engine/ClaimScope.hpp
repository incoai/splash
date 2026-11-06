#pragma once

// A claim and its scope for what a test or tool allocates through a
// MemoryGovernor outside an engine's allocation.

#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryLedger.hpp"

#include <optional>

namespace splash::test {

// A claim of `spec` in the governor's ledger, open while this lives, and its
// scope (MemoryGovernor::Scope): what is allocated through the governor
// meanwhile is for that claim, as inside an engine's allocation. Every
// allocation is for a claim; a fixture or tool that allocates through the
// governor outside an engine does it inside one: a Floor claim for what a
// start builds, a request's class for what a request allocates outside the
// engine.
class ClaimScope final {
public:
  ClaimScope(engine::MemoryGovernor &governor, const engine::ClaimSpec &spec)
      : claim_(governor, spec), scope_(governor, claim_.id) {}
  ClaimScope(engine::MemoryGovernor &governor, engine::ClaimClass cls)
      : ClaimScope(governor, engine::ClaimSpec{.cls = cls}) {}
  [[nodiscard]] engine::ClaimId id() const noexcept { return claim_.id; }

private:
  // Closed after the scope that names it ends.
  struct Claim final {
    Claim(engine::MemoryGovernor &governor, const engine::ClaimSpec &spec)
        : governor(governor), id(governor.ledger().open(spec)) {}
    ~Claim() { governor.ledger().close(id); }
    Claim(const Claim &) = delete;
    Claim &operator=(const Claim &) = delete;
    engine::MemoryGovernor &governor;
    engine::ClaimId id;
  };
  Claim claim_;
  engine::MemoryGovernor::Scope scope_;
};

// A start's Floor claim and its scope over the governor, when there is one,
// for a fixture to build its allocators in (the KV runway) before serving.
[[nodiscard]] inline std::optional<ClaimScope> startScope(std::optional<engine::MemoryGovernor> &governor) {
  if (!governor)
    return std::nullopt;
  return std::optional<ClaimScope>(std::in_place, *governor, engine::ClaimClass::Floor);
}

} // namespace splash::test
