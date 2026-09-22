#pragma once

// Shared only by the trained single-request and compact batch head. No public
// API exposes caches or allows another model/owner to mutate them.
#include "flash/FlashMTP.hpp"
#include "flash/FlashQSA.hpp"

namespace splash::flash {
struct FlashMTPState::Impl final {
  std::shared_ptr<const uint8_t> owner;
  FlashQSAState qsa;
  uint64_t length = 0;
  bool poisoned = false;
};
} // namespace splash::flash
