#include "engine/MemoryControl.hpp"

#include "StderrLine.hpp"

#include <optional>
#include <sstream>

namespace splash::engine {

std::string MemoryStatusReporter::update(const ResourceWaitSnapshot &wait,
                                         bool hostGrowthAllowed) {
  const unsigned state = (!hostGrowthAllowed ? 1u : 0u) |
                         (wait.memory ? 2u : 0u) |
                         (wait.suspended ? 4u : 0u) |
                         (wait.draining ? 8u : 0u) |
                         (wait.heldBehindRefusal ? 16u : 0u);
  if (state == state_)
    return {};
  state_ = state;
  if (!state)
    return "Memory: growth available; resource wait cleared";
  std::ostringstream out;
  out << "Memory: growth " << (hostGrowthAllowed ? "available" : "paused")
      << "; waiting=" << wait.memory << "; suspended=" << wait.suspended;
  if (wait.heldBehindRefusal)
    out << "; held=" << wait.heldBehindRefusal;
  if (wait.draining)
    out << "; waiting for resident requests to finish";
  return out.str();
}

bool MemoryControl::run(MemoryPressure pressure) {
  loop_.releaseIdleWeights();
  governor_.setPressure(pressure);
  const double now = loop_.monotonicMilliseconds();
  static_cast<void>(backend_.refreshMemoryStats());
  const MemoryGovernorSnapshot memory = governor_.snapshot();
  const ResourceWaitSnapshot wait = loop_.resourceWaitSnapshot();
  const std::string diagnostic = reporter_.update(wait, memory.hostGrowthAllowed);
  if (!diagnostic.empty())
    writeStderrLine(diagnostic);
  // Requests held back by a refusal wait for memory too, the refused one
  // included while a pass defers it.
  const std::optional<MemoryReclaimDirective> directive = policy_.update(
      memory, now, wait.memory || wait.suspended || wait.heldBehindRefusal);
  if (!directive)
    return false;
  const MemoryReclaimResult reclaim = loop_.reclaimMemory(*directive);
  policy_.reclaimed(*directive, reclaim);
  governor_.reclaimed(reclaim.outcome);
  static_cast<void>(backend_.refreshMemoryStats());
  // What transfers held back continues at the next command-free point.
  return reclaim.outcome == ReclaimOutcome::Pending;
}

} // namespace splash::engine
