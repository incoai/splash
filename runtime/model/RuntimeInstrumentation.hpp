#pragma once

#include "metal/abi/KvExtent.h"
#include "model/Runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace splash::model {

// What tests impose on a Runtime beyond its serving API. Only the
// instrumented build of Runtime.cpp (SPLASH_RUNTIME_INSTRUMENTATION) defines
// these; production binaries never link it.
class RuntimeInstrumentation final {
public:
  // The PLE rows the last decode step gathered for batch lane `lane`'s
  // verify rows, each row's heads' in order (Qwen3.8-Flash-Next), until the
  // next decode step; none for a target without them.
  [[nodiscard]] static std::vector<uint32_t> lastTableRows(const Runtime &runtime, uint32_t lane);
  // The fp32 logits the last command left in batch lane `lane`'s rows, each
  // the target's vocabulary wide, until the next command: a decode step's
  // verify rows, or in row 0 the final position of the prefill chunk that
  // ended a prompt.
  [[nodiscard]] static std::span<const float> lastLogits(const Runtime &runtime, uint32_t lane);
  // The rows of its block batch lane `lane` of the last decode step verified
  // (ops::DraftSelector::addVerifyRows), until the next decode step.
  [[nodiscard]] static uint32_t lastVerifyRows(const Runtime &runtime, uint32_t lane);
  // The GPU tables of resident request `requestId`'s lane as its last
  // command left them, every entry the lane holds: its page table and, for
  // a layout with a block index, its index table (none without).
  struct LaneTables final {
    std::span<const SplashKvPage> pages;
    std::span<const SplashKvPage> index;
  };
  [[nodiscard]] static LaneTables laneTables(const Runtime &runtime, uint64_t requestId);
  // The rows resident request `requestId`'s latest decode step retained,
  // which its GDN cell holds only once its next step folds them in
  // (QwenLaneMetadata::pendingRows): every GDN layer's tape rows
  // (metal/abi/GDN.h) up to that count, field by field, layer by layer.
  [[nodiscard]] static std::vector<uint8_t> pendingGdnRows(const Runtime &runtime, uint64_t requestId);
  // The entries of the registry through which the placements of an image
  // share its rows: an entry for each image whose rows something holds, and
  // for each whose rows went since the last start with images.
  [[nodiscard]] static size_t imageEntries(const Runtime &runtime);
};

} // namespace splash::model
