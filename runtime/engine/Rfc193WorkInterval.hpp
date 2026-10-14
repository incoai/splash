#pragma once

// RFC 193 scaffolding: P-core work-interval hint for verify dispatch.
// See docs/rfc/193-work-interval.md and
// https://github.com/incoai/splash/issues/193.

#include <cstdint>

namespace splash::engine::rfc193 {

inline constexpr bool kEnabledDefault = false;
inline constexpr const char* kEnvVar = "SPLASH_ENGINE_WORK_INTERVAL";

// Scoped scheduling hint: raised around prompt-forward submit and the
// verify-block build, renewed on promptForwardBegan/Submitted, dropped
// immediately after. Scheduling-only — never held across a blocking wait,
// so UI/audio threads cannot starve.
enum class IntervalScope : std::uint8_t {
  Idle = 0,
  PromptForwardSubmit = 1,
  VerifyBlockBuild = 2,
};

struct WorkIntervalConfig final {
  bool enabled = kEnabledDefault;
  // Renew points inside one measured window; bounded so a stuck submitter
  // cannot pin the hint on.
  std::uint32_t maxRenewsPerWindow = 2;
};

[[nodiscard]] constexpr bool renewAllowed(std::uint32_t renewsSoFar,
                                          const WorkIntervalConfig& config) noexcept {
  return config.enabled && renewsSoFar < config.maxRenewsPerWindow;
}

static_assert(!renewAllowed(0, WorkIntervalConfig{}));
static_assert(renewAllowed(0, WorkIntervalConfig{true, 2}));
static_assert(!renewAllowed(2, WorkIntervalConfig{true, 2}));

}  // namespace splash::engine::rfc193
