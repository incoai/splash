#include "engine/MemoryControl.hpp"

#include "StderrLine.hpp"

#include <iomanip>
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

bool MemoryControl::guardCompression(const MemoryGovernorSnapshot &memory, double now) {
  // The weights are released only here, in the control pass, so a pass
  // without a sample always comes between their release and their restore.
  std::optional<CompressionGuard::Sample> sample;
  if (!loop_.weightsSnapshot().released && memory.hostMeasurementValid) {
    if (const std::optional<uint64_t> compressed = compressorContent_()) {
      const uint64_t room = memory.hostAvailableBytes > memory.hostReserveBytes
          ? memory.hostAvailableBytes - memory.hostReserveBytes
          : 0;
      sample = CompressionGuard::Sample{*compressed, memory.chargedBytes, room};
    }
  }
  switch (guard_.update(now, sample)) {
  case CompressionGuard::Event::None:
    return false;
  case CompressionGuard::Event::Fired: {
    governor_.setHostDebt(guard_.debtBytes());
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << "Memory: macOS compressed "
        << static_cast<double>(guard_.growthBytes()) / static_cast<double>(1ULL << 30)
        << " GiB of memory in "
        << static_cast<int>(kCompressionWindowMilliseconds / 1000.0)
        << " s while the engine grew; cache growth holds for "
        << static_cast<int>(kCompressionHoldMilliseconds / 60'000.0)
        << " min and the cache gives that back";
    logLine(out.str());
    return true;
  }
  case CompressionGuard::Event::Expired:
    governor_.setHostDebt(0);
    logLine("Memory: compression hold ended");
    return true;
  }
  return false;
}

bool MemoryControl::run(MemoryPressure pressure) {
  loop_.releaseIdleWeights();
  governor_.setPressure(pressure);
  const double now = loop_.monotonicMilliseconds();
  static_cast<void>(backend_.refreshMemoryStats());
  MemoryGovernorSnapshot memory = governor_.evaluate();
  if (guardCompression(memory, now))
    memory = governor_.evaluate();
  const ResourceWaitSnapshot wait = loop_.resourceWaitSnapshot();
  const std::string diagnostic = reporter_.update(wait, memory.hostGrowthAllowed);
  if (!diagnostic.empty())
    logLine(diagnostic);
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
