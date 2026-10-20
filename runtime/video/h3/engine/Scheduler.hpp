#pragma once

// One DiT forward: input embedding, the 50 blocks and the final layer. Every block step is encoded into its own
// command buffer and committed back to back on one queue (Metal orders the buffers through hazard tracking); the host
// waits once at the end. With weight streaming, each block's int8 matrices are bound from the slot the reader filled,
// and the slot is handed back when all of the block's command buffers have completed.
#import <Metal/Metal.h>

#include "video/h3/engine/MemoryPlan.hpp"
#include "video/h3/engine/Types.hpp"
#include "video/h3/engine/WeightStream.hpp"
#include "video/h3/engine/WindowTable.hpp"
#include "video/h3/metal/MetalBackend.hpp"
#include "video/h3/model/DitWeights.hpp"
#include "video/h3/model/Geometry.hpp"
#include "video/h3/ops/LinearBranch.hpp"
#include "video/h3/ops/WindowAttention.hpp"

#include <functional>
#include <set>
#include <vector>

namespace h3 {

enum class StepClass : uint8_t { Linear = 0, Attention = 1, Other = 2, Branch = 3 };

// Receives the steps of a block in order.
class StepSink {
public:
  using Encode = std::function<void(id<MTLComputeCommandEncoder>)>;
  virtual ~StepSink() = default;
  virtual void step(const char *name, StepClass cls, const Encode &encode) = 0;
};

// Weights bound for one block: resident buffers, or the streamed matrices of its slot.
struct BlockBinding {
  const model::DitBlockWeights *w = nullptr;
  id<MTLBuffer> qkvg = nil, out = nil, fc1 = nil, fc2 = nil, outLinear = nil;
  id<MTLBuffer> qkfixWeight = nil;  // set for QKFIX blocks
  id<MTLBuffer> qkfixSplit = nil;
};

class Scheduler {
public:
  Scheduler(metal::MetalBackend &backend, const model::SequenceLayout &layout, const MemoryPlan &plan,
            const WindowTable &table, const model::DitWeights *weights, WeightStream *stream,
            std::set<int> qkfixBlocks);

  // Encodes block `block` step by step (one call of sink.step per kernel group). Its step names and the plan views
  // are what dev/tests/BlockCheck.mm snapshots.
  void encodeBlock(int block, const BlockBinding &weights, StepSink &sink) const;
  [[nodiscard]] BlockBinding residentBinding(int block) const;

  // ada: bf16 [50][3][6][5376] and [3][2][5376] of one step; latents fp32 [Nv][96] / [Na][32].
  Status forward(const uint16_t *adaBlocks, const uint16_t *adaFinal, const float *videoIn, const float *audioIn,
                 float *videoOut, float *audioOut, ForwardStats &stats);

  [[nodiscard]] const MemoryPlan &plan() const { return plan_; }
  // Blocks per forward (default all 50; the quick synthetic check runs 1).
  void setBlocks(int blocks) { blocks_ = blocks; }

private:
  [[nodiscard]] ops::AttentionBuffers attentionBuffers() const;
  [[nodiscard]] ops::BranchBuffers branchBuffers() const;
  [[nodiscard]] ops::QkvTargets qkvTargets() const;

  metal::MetalBackend &backend_;
  model::SequenceLayout layout_;
  const MemoryPlan &plan_;
  const WindowTable &table_;
  const model::DitWeights *weights_;  // null in the block check harness (encodeBlock only)
  WeightStream *stream_;
  std::set<int> qkfixBlocks_;
  ops::LinearBranch branch_;
  int blocks_ = H3_DIT_BLOCKS;
};

} // namespace h3
