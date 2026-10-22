#pragma once

// The denoising loop (SGLang Diffusion's DenoisingStage): the request's DiT, loaded for the stage and freed after it,
// and the prepared schedule's rectified-flow Euler steps (eta 0), one forward each. Only the target rows are stepped:
// the keyframe conditioning rows stay at their noised latents.
//
// A forward: input embedding, the 50 blocks and the final layer, each block one command. The host encodes block b + 1
// while the GPU runs block b and submits it once b completed. With weight streaming, each block's int8 matrices are
// bound from the slot the reader fills, and the slot is handed back once the block's command has completed.
#include "video/metal/Encoder.hpp"
#include "video/model/DitWeights.hpp"
#include "video/model/MemoryPlan.hpp"
#include "video/model/WeightStream.hpp"
#include "video/ops/Geometry.hpp"
#include "video/ops/LinearBranch.hpp"
#include "video/ops/WindowAttention.hpp"
#include "video/ops/WindowTable.hpp"
#include "video/pipeline/PipelineStage.hpp"

#include <filesystem>
#include <memory>
#include <set>
#include <vector>

namespace splash::video {

struct ForwardStats {
  double wallMs = 0, embedMs = 0, blocksSpanMs = 0, blocksGpuMs = 0, finalMs = 0;
  model::StreamStats stream;
  double deviceMb = 0;
};

// The packed sequence [text | conditioning | audio | video] as the request's preparation laid it out (borrowed while
// the DiT is built; the conditioning rows are copied).
struct DitInputs {
  int32_t textRows = 0;
  const uint8_t *tags = nullptr;   // [rows] 0 video (and a keyframe's vision-block text rows), 1 text, 2 audio, 3 cond
  const uint16_t *rope = nullptr;  // bf16 [rows][96]
  const uint16_t *text = nullptr;  // bf16 [textRows][5376] refined text
  int32_t condRows = 0;            // keyframe conditioning rows (whole keyframes of 1008; 0 for text-to-video)
  const float *cond = nullptr;     // [condRows][96]: the noised keyframe latents, fixed for every step
};

// The QKFIX blocks of a clip's options.
[[nodiscard]] std::set<int> qkfixBlocks(const ClipOptions &options);

// One request's DiT: its layout, weights (resident or streamed from SSD), the window softmax's key-range table, the
// activation plan, and its forward.
class Dit final {
public:
  Dit(metal::MetalBackend &backend, const ClipOptions &options, const ops::VideoGeometry &geometry,
      const DitInputs &inputs, std::unique_ptr<model::DitWeights> weights);
  Dit(const Dit &) = delete;
  Dit &operator=(const Dit &) = delete;

  // ada: bf16 [50][3 | 4][6][5376] (4 slots with conditioning rows) and [3][2][5376] of one step; latents fp32
  // [Nv][96] / [Na][32] (the target rows).
  void forward(const uint16_t *adaBlocks, const uint16_t *adaFinal, const float *videoIn, const float *audioIn,
               float *videoOut, float *audioOut, ForwardStats &stats);
  // Blocks per forward: all 50, or the first ones only (the quick synthetic check of dev/tests runs 1).
  void setBlocks(int blocks) { blocks_ = blocks; }
  [[nodiscard]] const ops::SequenceLayout &layout() const { return layout_; }

private:
  // Weights bound for one block: resident buffers, or the streamed matrices of its slot.
  struct BlockBinding {
    const model::DitBlockWeights *w = nullptr;
    BufferView qkvg, out, fc1, fc2, outLinear;
    BufferView qkfixWeight;  // set for QKFIX blocks
    BufferView qkfixSplit;
  };

  [[nodiscard]] BlockBinding residentBinding(int block) const;
  // Encodes block `block`, its kernels in order.
  void encodeBlock(int block, const BlockBinding &weights, Encoder &e) const;
  [[nodiscard]] ops::AttentionBuffers attentionBuffers() const;
  [[nodiscard]] ops::BranchBuffers branchBuffers() const;
  [[nodiscard]] ops::QkvTargets qkvTargets() const;

  metal::MetalBackend &backend_;
  ops::SequenceLayout layout_;
  std::set<int> qkfixBlocks_;
  std::unique_ptr<model::DitWeights> weights_;
  ops::WindowTable table_;
  std::unique_ptr<model::WeightStream> stream_;
  model::MemoryPlan plan_;
  ops::LinearBranch branch_;
  int blocks_ = H3_DIT_BLOCKS;
  std::vector<float> cond_;
};

class DenoisingStage final : public PipelineStage {
public:
  // weights: the converted weights' root, of dit-turbo/ and dit-default/.
  DenoisingStage(metal::MetalBackend &backend, std::filesystem::path weights)
      : backend_(backend), weights_(std::move(weights)) {}

  [[nodiscard]] protocol::VideoPhase phase() const override { return protocol::VideoPhase::Denoise; }
  [[nodiscard]] uint32_t steps(const VideoBatch &batch) const override {
    return static_cast<uint32_t>(batch.prepared.steps);
  }
  void forward(VideoBatch &batch, StageEvents &events) override;

private:
  metal::MetalBackend &backend_;
  std::filesystem::path weights_;
};

} // namespace splash::video
