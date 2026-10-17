#pragma once

// What a store of streamed routed experts reports (ExpertStore::stats,
// /status), apart from the store, whose header brings Metal and the MoE
// operators with it.

#include "model/LatencyHistogram.hpp"

#include <cstdint>
#include <string>

namespace splash::model {

struct ExpertStoreStats final {
  uint64_t arenaBytes = 0;
  uint64_t slots = 0;
  uint64_t borrowed = 0; // of them, those prefill loads borrow to stage in
  uint64_t experts = 0;  // every block's routed experts
  uint64_t resident = 0; // slots holding an expert
  // Per ops::MoePhase, at its value (uses[0] prefill, uses[1] decode): the
  // blocks loaded, the distinct experts they routed, those not resident and
  // of them those read ahead, the experts read ahead whole (routed or not),
  // the bytes read from the sources (ahead or not, routed or not), and the
  // round trips of the loads' host steps (metal::HostStepTiming: the GPU's
  // work before a step to the runner taking it up, its work, its end to the
  // GPU's work after it).
  struct Use final {
    uint64_t loads = 0;
    uint64_t routed = 0;
    uint64_t read = 0;
    uint64_t prefetched = 0;
    uint64_t readAhead = 0;
    uint64_t readBytes = 0;
    LatencyStats pickup, work, resume;
  };
  Use uses[2];
  // Why a load failed, which fails every later load too; empty while none
  // has.
  std::string failure;
};

} // namespace splash::model
