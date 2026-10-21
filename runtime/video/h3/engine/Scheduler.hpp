#pragma once

// One DiT forward: input embedding, the 50 blocks and the final layer, each block one command. The host encodes block
// b + 1 while the GPU runs block b and submits it once b completed. With weight streaming, each block's int8 matrices
// are bound from the slot the reader fills, and the slot is handed back once the block's command has completed.
#include "video/h3/engine/Encoder.hpp"
#include "video/h3/engine/MemoryPlan.hpp"
#include "video/h3/engine/Types.hpp"
#include "video/h3/engine/WeightStream.hpp"
#include "video/h3/engine/WindowTable.hpp"
#include "video/h3/model/DitWeights.hpp"
#include "video/h3/model/Geometry.hpp"
#include "video/h3/ops/LinearBranch.hpp"
#include "video/h3/ops/WindowAttention.hpp"

#include <set>

namespace h3 {

// Weights bound for one block: resident buffers, or the streamed matrices of its slot.
struct BlockBinding {
  const model::DitBlockWeights *w = nullptr;
  BufferView qkvg, out, fc1, fc2, outLinear;
  BufferView qkfixWeight;  // set for QKFIX blocks
  BufferView qkfixSplit;
};

class Scheduler {
public:
  Scheduler(metal::MetalBackend &backend, const model::SequenceLayout &layout, const MemoryPlan &plan,
            const WindowTable &table, const model::DitWeights *weights, WeightStream *stream,
            std::set<int> qkfixBlocks);

  // Encodes block `block`, its kernels in order.
  void encodeBlock(int block, const BlockBinding &weights, Encoder &e) const;
  [[nodiscard]] BlockBinding residentBinding(int block) const;

  // ada: bf16 [50][3][6][5376] and [3][2][5376] of one step; latents fp32 [Nv][96] / [Na][32].
  void forward(const uint16_t *adaBlocks, const uint16_t *adaFinal, const float *videoIn, const float *audioIn,
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
  const model::DitWeights *weights_;
  WeightStream *stream_;
  std::set<int> qkfixBlocks_;
  ops::LinearBranch branch_;
  int blocks_ = H3_DIT_BLOCKS;
};

} // namespace h3
