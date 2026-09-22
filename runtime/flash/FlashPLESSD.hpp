#pragma once

#include "flash/FlashPLE.hpp"
#include "flash/FlashPLESSDLayout.hpp"
#include "flash/FlashPLESSDStore.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace splash::flash {

inline constexpr const char *kFlashPLESSDRoute =
    ";ple-ssd-original-q4g32-bounded-staging-gpu-id-checked-v1";

// Each synchronous executor owns its arena; the immutable disk store/cache is
// shared across executors. No full PLE table receives a native GPU allocation.
class FlashPLESSD final {
public:
  FlashPLESSD(metal::MetalBackend &backend,
              std::shared_ptr<FlashPLESSDStore> store,
              const FlashPLEWeights &weights, uint32_t maximumLanes,
              uint32_t maximumRows);
  ~FlashPLESSD();
  FlashPLESSD(const FlashPLESSD &) = delete;
  FlashPLESSD &operator=(const FlashPLESSD &) = delete;

  // Histories are the completed, request-owned I64[lane,2] snapshots. Only
  // private CPU copies change while forming IDs. Fetch errors precede graph
  // submission, so request history and convolution remain untouched.
  void prepare(std::span<const int64_t> tokens,
               std::span<const int64_t> histories,
               const FlashPLEWeights &weights, FlashPLEGeometry geometry);
  void addHashGather(metal::CommandGraph &graph,
                     const FlashPLEWeights &weights,
                     metal::MetalBuffer tokens, metal::MetalBuffer history,
                     metal::MetalBuffer ids, metal::MetalBuffer output,
                     metal::MetalBuffer diagnostics,
                     FlashPLEGeometry geometry);
  [[nodiscard]] uint64_t allocatedBytes() const noexcept;
  [[nodiscard]] std::vector<metal::MetalBuffer> scratchBuffers() const;
  [[nodiscard]] static uint64_t plannedBytes(uint32_t lanes, uint32_t rows);

private:
  metal::MetalBackend &backend_;
  std::shared_ptr<FlashPLESSDStore> store_;
  metal::MetalBuffer expectedIDs_, rows_;
  uint32_t maximumLanes_, maximumRows_, preparedLanes_ = 0, preparedRows_ = 0;
};

} // namespace splash::flash
