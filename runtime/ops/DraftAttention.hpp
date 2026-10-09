#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"

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

  bool operator==(const DraftAttentionShape &) const = default;
};

struct DraftAttentionWorkspace final {
  uint64_t convolutionBytes = 0;
  uint64_t qkvBytes = 0;
  uint64_t groupedQueriesBytes = 0;
  uint64_t queryKeysBytes = 0;
  uint64_t queryValuesBytes = 0;
};

// Constructed only by DraftAttention::plan so the shape, physical rows and
// workspace cannot disagree. The grouped-queries tensor also carries the
// split partials of the attention core behind the query rows, so the core
// needs no device scratch beyond these tensors.
class DraftAttentionPlan final {
public:
  [[nodiscard]] DraftAttentionShape shape() const noexcept { return shape_; }
  [[nodiscard]] uint32_t lanes() const noexcept { return lanes_; }
  [[nodiscard]] DraftAttentionWorkspace workspace() const noexcept;

private:
  DraftAttentionPlan(DraftAttentionShape shape, uint32_t lanes)
      : shape_(shape), lanes_(lanes) {}

  DraftAttentionShape shape_;
  uint32_t lanes_;

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

struct DraftDecodeAttentionBuffers final {
  metal::MetalBuffer groupedQueries;
  std::span<const metal::MetalBuffer> persistentKeys;
  std::span<const metal::MetalBuffer> persistentValues;
  metal::MetalBuffer queryKeys;
  metal::MetalBuffer queryValues;
};

class DraftAttention final {
public:
  [[nodiscard]] static DraftAttentionPlan plan(DraftAttentionShape shape,
                                               uint32_t lanes);

  static void addConvolution(metal::CommandGraph &graph,
                             DraftConvolutionBuffers buffers,
                             const DraftAttentionPlan &plan,
                             DraftConvolutionStage stage);
  static void addPrepare(metal::CommandGraph &graph,
                         DraftPrepareBuffers buffers,
                         const DraftAttentionPlan &plan);
  static void addDecode(metal::CommandGraph &graph,
                        DraftDecodeAttentionBuffers buffers,
                        std::span<const uint32_t> cacheLengths,
                        const DraftAttentionPlan &plan);
  static void addReorder(metal::CommandGraph &graph,
                         metal::MetalBuffer grouped,
                         metal::MetalBuffer rowMajor,
                         const DraftAttentionPlan &plan);
  // The context writers. A row of contextKv holds its keys, then its values
  // (kvHeads * headDimension each); its normalized, rotated keys and its
  // values go to its position's slot of the ring.
  static void addContextPrefill(
      metal::CommandGraph &graph, metal::MetalBuffer contextKv,
      metal::MetalBuffer keyNorm, metal::MetalBuffer ropeCos,
      metal::MetalBuffer ropeSin, metal::MetalBuffer keys,
      metal::MetalBuffer values, uint32_t tokens, uint32_t startPosition,
      DraftAttentionShape shape);
  static void addContextCommit(
      metal::CommandGraph &graph, metal::MetalBuffer contextKv,
      metal::MetalBuffer keyNorm, metal::MetalBuffer ropeCos,
      metal::MetalBuffer ropeSin,
      std::span<const metal::MetalBuffer> persistentKeys,
      std::span<const metal::MetalBuffer> persistentValues,
      metal::MetalBuffer retainedCounts,
      std::span<const uint32_t> startPositions, DraftAttentionShape shape);

  // A context window: for each slot of a ring, the bf16 context row of
  // `width` values the slot's keys and values are computed from, in 4-bit
  // codes of 64-value groups with each group's fp16 scale and minimum
  // (metal/abi/DraftAttention.h). Positions map to slots as in the rings.
  [[nodiscard]] static uint64_t contextWindowBytes(uint32_t width);
  // Stores `rows` context rows in the slots of positions startPosition on.
  static void addWindowStore(metal::CommandGraph &graph,
                             metal::MetalBuffer context,
                             metal::MetalBuffer window, uint32_t rows,
                             uint32_t startPosition, uint32_t width);
  // Loads the rows of positions startPosition on back into `rows` rows.
  static void addWindowLoad(metal::CommandGraph &graph,
                            metal::MetalBuffer window,
                            metal::MetalBuffer context, uint32_t rows,
                            uint32_t startPosition, uint32_t width);
};

} // namespace splash::ops
