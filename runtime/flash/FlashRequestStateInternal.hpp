#pragma once

// Internal state bridge for native trunk executors. Both scalar verification
// and real batch decode mutate the same request-owned storage; no conversion
// or duplicate request state is introduced by the batch path.
#include "flash/FlashForward.hpp"
#include "flash/FlashGDN.hpp"
#include "flash/FlashQSA.hpp"

#include <array>
#include <memory>

namespace splash::flash {

struct FlashRequestState::Impl final {
  std::shared_ptr<const uint8_t> owner;
  std::shared_ptr<const uint8_t> identity = std::make_shared<const uint8_t>(0);
  uint32_t capacity = 0;
  uint64_t length = 0;
  bool poisoned = false;
  bool pendingVerification = false;
  std::array<FlashGDNState, 48> gdn;
  std::array<FlashQSAState, 48> qsa;
  metal::MetalBuffer pleHistory;
  metal::MetalBuffer pleConvolution;
};

} // namespace splash::flash
