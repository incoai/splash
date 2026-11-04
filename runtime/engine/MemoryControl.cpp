#include "engine/MemoryControl.hpp"

#include "StderrLine.hpp"
#include "engine/NativeRuntime.hpp"

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
  loop_.collectLandings(now);
  static_cast<void>(backend_.refreshMemoryStats());
  const bool again = governor_.controlPass(now);
  static_cast<void>(backend_.refreshMemoryStats());
  // Read after the pass, at the last sample of the host it took.
  const std::string diagnostic =
      reporter_.update(loop_.resourceWaitSnapshot(), governor_.admits(ClaimClass::Growth));
  if (!diagnostic.empty())
    logLine(diagnostic);
  return again;
}

} // namespace splash::engine
