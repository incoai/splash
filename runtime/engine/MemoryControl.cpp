#include "engine/MemoryControl.hpp"

#include "StderrLine.hpp"

#include <algorithm>
#include <optional>
#include <sstream>

namespace splash::engine {

namespace {

// While one lives, the governor's growth admission counts the bytes an
// exchange returned to the system as the host's already
// (MemoryGovernor::setGrowthCredit). It restores the credit it found.
class GrowthCredit final {
public:
  GrowthCredit(MemoryGovernor &governor, uint64_t bytes) noexcept
      : governor_(governor), previous_(governor.setGrowthCredit(bytes)) {}
  ~GrowthCredit() { static_cast<void>(governor_.setGrowthCredit(previous_)); }
  GrowthCredit(const GrowthCredit &) = delete;
  GrowthCredit &operator=(const GrowthCredit &) = delete;

private:
  MemoryGovernor &governor_;
  uint64_t previous_;
};

} // namespace

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
  // Requests held back by a refusal wait for memory too, the refused one
  // included while a pass defers it.
  const ResourceWaitSnapshot wait = loop_.resourceWaitSnapshot();
  const bool requestWaiting = wait.memory || wait.suspended || wait.heldBehindRefusal;
  governor_.setRequestWaiting(requestWaiting);
  static_cast<void>(backend_.refreshMemoryStats());
  const MemoryGovernorSnapshot memory = governor_.evaluate();
  const std::string diagnostic = reporter_.update(wait, memory.hostGrowthAllowed);
  if (!diagnostic.empty())
    logLine(diagnostic);
  const std::optional<MemoryReclaimDirective> directive =
      policy_.update(memory, now, requestWaiting);
  noteExpertReleases(now);
  if (!directive) {
    growExperts(memory, requestWaiting, now);
    return false;
  }
  const MemoryReclaimResult reclaim = loop_.reclaimMemory(*directive);
  noteExpertReleases(now);
  policy_.reclaimed(*directive, reclaim);
  governor_.reclaimed(reclaim.outcome);
  static_cast<void>(backend_.refreshMemoryStats());
  // What transfers held back continues at the next command-free point.
  return reclaim.outcome == ReclaimOutcome::Pending;
}

void MemoryControl::noteExpertReleases(double now) {
  const uint64_t releases = loop_.expertReleases();
  if (releases == expertReleases_)
    return;
  expertReleases_ = releases;
  if (expertGrowthMilliseconds_ && now - *expertGrowthMilliseconds_ < expertCooldownMilliseconds_)
    expertCooldownMilliseconds_ =
        std::min(2 * expertCooldownMilliseconds_, kExpertGrowthCooldownLimitMilliseconds);
  expertGrowthMilliseconds_.reset();
  expertReleaseMilliseconds_ = now;
}

void MemoryControl::growExperts(const MemoryGovernorSnapshot &memory, bool requestWaiting,
                                double now) {
  if (expertGrowthMilliseconds_ && now - *expertGrowthMilliseconds_ >= expertCooldownMilliseconds_) {
    expertGrowthMilliseconds_.reset();
    expertCooldownMilliseconds_ = kExpertGrowthCooldownMilliseconds;
  }
  if (requestWaiting || !memory.hostGrowthAllowed ||
      (expertReleaseMilliseconds_ && now - *expertReleaseMilliseconds_ < expertCooldownMilliseconds_))
    return;
  const uint64_t bytes = loop_.expertGrowthBytes();
  if (!bytes)
    return;
  // What the chunk lacks of the host's headroom beyond the recovery margin,
  // in the pass's sample, or of the room the engine's limit has free,
  // whichever lacks more: what the engine returns adds to both. An engine
  // without requests returns it first, or nothing when what leaves at once
  // falls short; one with a request returns nothing, and the chunk waits. A
  // chunk refused after all leaves what the engine returned with the
  // system.
  const uint64_t spare =
      memory.hostHeadroomBytes - std::min(memory.hostHeadroomBytes, kHostRecoveryMarginBytes);
  const uint64_t lacking = bytes - std::min({bytes, spare, memory.headroomBytes});
  const uint64_t returned = lacking ? loop_.reclaimForExperts(lacking) : 0;
  bool grown = false;
  if (returned >= lacking) {
    // The admission samples the host again, which may not show yet what the
    // engine returned: the credit counts it as the host's.
    const GrowthCredit credit(governor_, returned);
    grown = loop_.growExperts(memory.headroomBytes + returned);
  }
  if (grown)
    expertGrowthMilliseconds_ = now;
  if (grown || returned)
    static_cast<void>(backend_.refreshMemoryStats());
}

} // namespace splash::engine
