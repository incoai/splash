#pragma once

// RFC 190 scaffolding: deferred GDN recurrent replay.
// See docs/rfc/190-deferred-gdn.md and
// https://github.com/incoai/splash/issues/190.

#include <cstdint>

namespace splash::ops::rfc190 {

inline constexpr bool kEnabledDefault = false;
inline constexpr const char* kEnvVar = "SPLASH_DEFERRED_GDN_REPLAY";

// Deferred-replay state for one GDN layer. On partial accept the
// strict-prefix SSM is left unbuilt (Deferred) and fused into the next
// verify scan — which the existing verify_gdn_fused kernels already do —
// or lazily rebuilt (replay(keep)). Full acceptance resolves to Built
// with no work at all, so no new .metal is needed for this RFC.
enum class ReplayState : std::uint8_t { Built = 0, Deferred = 1 };

struct DeferredReplay final {
  ReplayState state = ReplayState::Built;
  // Tokens of strict-prefix input held unbuilt while Deferred.
  std::uint32_t pendingTokens = 0;
};

// Transition on verify outcome: accepted==proposed rebuilds nothing;
// accepted<proposed defers the strict prefix; accepted==0 keeps prior.
[[nodiscard]] constexpr DeferredReplay onVerify(DeferredReplay cur,
                                               std::uint32_t accepted,
                                               std::uint32_t proposed) noexcept {
  if (accepted >= proposed) return DeferredReplay{ReplayState::Built, 0};
  if (accepted == 0) return cur;
  return DeferredReplay{ReplayState::Deferred, proposed - accepted};
}

static_assert(onVerify({ReplayState::Built, 0}, 8, 8).state == ReplayState::Built);
static_assert(onVerify({ReplayState::Built, 0}, 8, 8).pendingTokens == 0);
static_assert(onVerify({ReplayState::Built, 0}, 5, 8).state == ReplayState::Deferred);
static_assert(onVerify({ReplayState::Built, 0}, 5, 8).pendingTokens == 3);
static_assert(onVerify({ReplayState::Deferred, 3}, 0, 8).state == ReplayState::Deferred);

}  // namespace splash::ops::rfc190
