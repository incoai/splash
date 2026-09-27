#pragma once

// RFC 188 scaffolding: in-place KV append + squeezed verify update.
// See docs/rfc/188-inplace-kv.md and
// https://github.com/incoai/splash/issues/188.

#include <cstdint>

namespace splash::ops::rfc188 {

inline constexpr bool kEnabledDefault = false;
inline constexpr const char* kEnvVar = "SPLASH_INPLACE_KV_APPEND";

// Squeeze batch to 1 before slice-assign on head-transposed verify slices.
inline constexpr bool kSqueezeBatchForVerify = true;

// In-place append addressing (host mirror of splash_chunk_base_for_lane
// in paged_store_row.h — keep the two in sync). With squeezed=true every
// lane reads the B=1 verify slice at lane 0's chunk base instead of its
// own lane base, killing the per-layer concat copy.
[[nodiscard]] constexpr unsigned long chunkBaseForLane(unsigned long laneStride,
                                                       std::uint32_t batch,
                                                       bool squeezed) noexcept {
  return (squeezed ? 0UL : static_cast<unsigned long>(batch)) * laneStride;
}

static_assert(chunkBaseForLane(1024, 2, false) == 2048);
static_assert(chunkBaseForLane(1024, 2, true) == 0);
static_assert(chunkBaseForLane(1024, 0, false) == 0);

}  // namespace splash::ops::rfc188
