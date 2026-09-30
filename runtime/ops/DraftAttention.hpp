#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"

#include <array>
#include <compare>
#include <cstdint>
#include <span>

namespace splash::ops {

struct DraftAttentionShape final {
  uint32_t hiddenSize = 0;
  uint32_t dynamicSize = 0;
  uint32_t qkvSize = 0;
  uint32_t attentionSize = 0;
  uint32_t queryHeads = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;

  auto operator<=>(const DraftAttentionShape &) const = default;
};

// These configurations vary the surrounding convolution, QKV preparation and
// reorder phases. The compiled attention core stays M32/N128/D128 with eight
// query rows, 256 threads, a fixed number of window splits per KV head and the
// semantic 2048 window.
struct DraftAttentionConfiguration final {
  // Zero uses the full element/task grid. Nonzero selects a persistent group
  // count for the surrounding phases.
  uint32_t groups = 0;
  bool operator==(const DraftAttentionConfiguration &) const = default;
};

struct DraftAttentionWorkspace final {
  uint64_t convolutionBytes = 0;
  uint64_t qkvBytes = 0;
  uint64_t groupedQueriesBytes = 0;
  uint64_t queryKeysBytes = 0;
  uint64_t queryValuesBytes = 0;
};

// Constructed only by DraftAttention::plan so configuration, physical rows
// and workspace cannot disagree. The grouped-queries tensor also carries the
// split partials of the attention core behind the query rows, so the core
// needs no device scratch beyond these tensors.
class DraftAttentionPlan final {
public:
  [[nodiscard]] DraftAttentionShape shape() const noexcept { return shape_; }
  [[nodiscard]] uint32_t lanes() const noexcept { return lanes_; }
  [[nodiscard]] DraftAttentionConfiguration configuration() const noexcept {
    return configuration_;
  }
  [[nodiscard]] DraftAttentionWorkspace workspace() const noexcept;

private:
  DraftAttentionPlan(DraftAttentionShape shape, uint32_t lanes,
                     DraftAttentionConfiguration configuration)
      : shape_(shape), lanes_(lanes), configuration_(configuration) {}

  DraftAttentionShape shape_;
  uint32_t lanes_;
  DraftAttentionConfiguration configuration_;

  friend class DraftAttention;
};

enum class DraftConvolutionStage : uint8_t { Prepare, Residual };

struct DraftConvolutionBuffers final {
  metal::MetalBuffer input;
  metal::MetalBuffer dynamic;
  metal::MetalBuffer weights;
  metal::MetalBuffer residual;
  metal::MetalBuffer output;
};

struct DraftPrepareBuffers final {
  metal::MetalBuffer qkv;
  metal::MetalBuffer groupedQueries;
  metal::MetalBuffer queryNorm;
  metal::MetalBuffer keyNorm;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer queryKeys;
  metal::MetalBuffer queryValues;
};

// One layer of fixed-size pages: K is [head, token, dimension],
// V is [head, dimension, token], adjacent within each page.
struct DraftKvBuffers final {
  std::vector<metal::MetalBuffer> keyPages{}, valuePages{};
};

struct DraftDecodeAttentionBuffers final {
  metal::MetalBuffer groupedQueries;
  std::span<const DraftKvBuffers> context;
  metal::MetalBuffer queryKeys;
  metal::MetalBuffer queryValues;
};

class DraftAttention final {
public:
  [[nodiscard]] static std::span<const DraftAttentionConfiguration>
  candidates(DraftAttentionShape shape);
  [[nodiscard]] static DraftAttentionPlan
  plan(DraftAttentionShape shape, uint32_t lanes,
       DraftAttentionConfiguration configuration = {});

  static void captureTargetHidden(
      metal::CommandGraph &graph, metal::MetalBuffer source,
      metal::MetalBuffer captured, uint32_t rows, uint32_t captureSlot,
      uint32_t sourceStart, uint32_t destinationStart, uint32_t hiddenWidth,
      uint32_t targetWidth);
  static void gatherLastRows(metal::CommandGraph &graph,
                             metal::MetalBuffer source,
                             metal::MetalBuffer destination, uint32_t rows,
                             uint32_t width);
  static void addConvolution(metal::CommandGraph &graph,
                             DraftConvolutionBuffers buffers,
                             const DraftAttentionPlan &plan,
                             DraftConvolutionStage stage);
  static void addPrepare(metal::CommandGraph &graph,
                         DraftPrepareBuffers buffers,
                         const DraftAttentionPlan &plan);
  static void addDecode(
      metal::CommandGraph &graph, DraftDecodeAttentionBuffers buffers,
      std::span<const uint32_t> cacheLengths, uint32_t cacheStride,
      const DraftAttentionPlan &plan);
  static void addReorder(metal::CommandGraph &graph,
                         metal::MetalBuffer grouped,
                         metal::MetalBuffer packed,
                         const DraftAttentionPlan &plan);
  static void
  addContextPrefill(metal::CommandGraph &graph, metal::MetalBuffer contextQkv,
                    metal::MetalBuffer keyNorm, metal::MetalBuffer ropeCos,
                    metal::MetalBuffer ropeSin, const DraftKvBuffers &context,
                    uint32_t tokens, uint32_t cacheStride,
                    uint32_t startPosition, DraftAttentionShape shape);
  static void addContextCommit(
      metal::CommandGraph &graph, metal::MetalBuffer contextQkv,
      metal::MetalBuffer keyNorm, metal::MetalBuffer ropeCos,
      metal::MetalBuffer ropeSin, std::span<const DraftKvBuffers> context,
      metal::MetalBuffer retainedCounts,
      std::span<const uint32_t> startPositions, uint32_t cacheStride,
      DraftAttentionShape shape, uint32_t lanes);
};

} // namespace splash::ops
