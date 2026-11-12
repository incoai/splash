#pragma once

// The denoising loop (SGLang Diffusion's DenoisingStage): the request's DiT, loaded for the stage and freed after it,
// and the prepared schedule's rectified-flow Euler steps (eta 0), one forward each, or for a contract of sampler
// "renoise" (pika-6: video/model/FastH3.hpp Sampler) Pika's consistency re-noising with the prepared fresh noise. Only the
// target rows are stepped: the keyframe conditioning rows stay at their noised latents. With a runtime adapter
// (video/model/Adapter.hpp: the DiT manifest's "adapters", or the stage's override) every adapted linear adds its LoRA.
//
// A forward: input embedding, the 50 blocks and the final layer, each block one command. The host encodes block b + 1
// while the GPU runs block b and submits it once b completed. With weight streaming, each block's int8 matrices are
// bound from the slot the reader fills, and the slot is handed back once the block's command has completed. FastH3's
// blocks (video/model/FastH3.hpp) run Video Sparse Attention on the rows in tile order (video/ops/VsaTable.hpp): the packed
// embedding is gathered into tile order before block 0 and back after block 49.
#include "video/metal/Encoder.hpp"
#include "video/model/Adapter.hpp"
#include "video/model/DitWeights.hpp"
#include "video/model/MemoryPlan.hpp"
#include "video/model/WeightStream.hpp"
#include "video/ops/Geometry.hpp"
#include "video/ops/LinearBranch.hpp"
#include "video/ops/VsaAttention.hpp"
#include "video/ops/WindowAttention.hpp"
#include "video/ops/WindowTable.hpp"
#include "video/pipeline/PipelineStage.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace splash::video {

struct ForwardStats {
  double wallMs = 0, blocksSpanMs = 0, blocksGpuMs = 0;
  model::StreamStats stream;
  double deviceMb = 0;
};

// The window softmax's key-frame chunk and radius (OpenVDN's; anchor frames first and last).
inline constexpr int32_t kWindowChunk = 5, kWindowRadius = 1;

// The packed sequence [text | conditioning | audio conditioning | audio | video] (ops::SequenceLayout, the engine order
// of video/ops/RowLayout.hpp) as the request's preparation laid it out (borrowed while the DiT is built; the conditioning
// rows are copied).
struct DitInputs {
  int32_t textRows = 0;
  // [rows] adaLN slots: 0 video (and the text segment's vision rows), 1 text, 2 audio, 3 conditioning video, 4
  // conditioning audio
  const uint8_t *tags = nullptr;
  const uint16_t *rope = nullptr;  // bf16 [rows][96]
  const uint16_t *text = nullptr;  // bf16 [textRows][5376] refined text
  // video conditioning rows: FL2VA's keyframes (whole keyframes of G_H x G_W), Ref2VA's reference video and image rows
  int32_t condRows = 0;
  const float *cond = nullptr;       // [condRows][96]: the noised latents, fixed for every step
  int32_t condAudioRows = 0;         // Ref2VA's reference audio rows
  const float *condAudio = nullptr;  // [condAudioRows][32]: clean, fixed for every step
  // the dense prefix segments (ops::SequenceLayout::segments; null: [text], [conditioning], [audio]); given, the
  // conditioning rows need not be whole frames (references)
  const std::vector<std::pair<int32_t, int32_t>> *segments = nullptr;
};

// One request's DiT: its layout, weights (streamed from SSD, or resident when they have no streamed tensors), the
// window softmax's key-range table (or, for FastH3's blocks, given vsa, the VSA tile map), the activation plan, and its
// forward.
class Dit final {
public:
  Dit(metal::MetalBackend &backend, const ops::VideoGeometry &geometry, const DitInputs &inputs,
      std::unique_ptr<model::DitWeights> weights, std::optional<ops::VsaSpec> vsa = std::nullopt,
      std::unique_ptr<model::Adapter> adapter = nullptr);
  Dit(const Dit &) = delete;
  Dit &operator=(const Dit &) = delete;

  // ada: bf16 [50][adaModalities][6][5376] and [finalSlots][2][5376] of one step (ops::SequenceLayout); latents
  // fp32 [Nv][96] / [Na][32] (the target rows).
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
  void encodeVsaBlock(int block, const BlockBinding &weights, Encoder &e) const;  // stages/FastH3Block.mm
  [[nodiscard]] ops::VsaBuffers vsaBuffers() const;
  [[nodiscard]] ops::AttentionBuffers attentionBuffers() const;
  [[nodiscard]] ops::BranchBuffers branchBuffers() const;
  [[nodiscard]] ops::QkvTargets qkvTargets() const;
  // The adapter's LoRA of one linear of a block (empty without an adapter); a: its A (default the linear's own).
  [[nodiscard]] ops::LoraBinding lora(const model::LoraLinear &linear, const metal::MetalBuffer *a = nullptr) const;
  [[nodiscard]] const model::AdapterBlock *adapterBlock(int block) const {
    return adapter_ ? &adapter_->block(block) : nullptr;
  }

  metal::MetalBackend &backend_;
  ops::SequenceLayout layout_;
  std::unique_ptr<model::DitWeights> weights_;
  ops::WindowTable table_;
  std::unique_ptr<ops::VsaTable> vsa_;  // FastH3
  std::unique_ptr<model::WeightStream> stream_;
  model::MemoryPlan plan_;
  ops::LinearBranch branch_;
  int blocks_ = H3_DIT_BLOCKS;
  // the target audio rows of the final layer's output (after the audio conditioning rows')
  [[nodiscard]] const void *audioOutRows() const {
    return static_cast<const uint8_t *>(plan_.view(model::Tensor::AudioOut).contents()) +
           size_t(layout_.condAudio) * H3_DIT_AUDIO_CHANNELS * 4;
  }
  std::vector<float> cond_, condAudio_;
  std::unique_ptr<model::Adapter> adapter_;
  metal::MetalBuffer loraT_;  // the LoRA down-projection's output, fp16 [rows rounded up to 64, + 64][3 x 128]
};

class DenoisingStage final : public PipelineStage {
public:
  // weights: the converted weights' root, of dit-<name>/ (video/model/FastH3.hpp ditDirectory); adapters: in place of the
  // DiT manifest's (model::adapterSpecs).
  DenoisingStage(metal::MetalBackend &backend, std::filesystem::path weights,
                 std::optional<std::vector<model::AdapterSpec>> adapters = std::nullopt)
      : backend_(backend), weights_(std::move(weights)), adapters_(std::move(adapters)) {}

  [[nodiscard]] protocol::VideoPhase phase() const override { return protocol::VideoPhase::Denoise; }
  [[nodiscard]] uint32_t steps(const VideoBatch &batch) const override {
    return static_cast<uint32_t>(batch.prepared.steps);
  }
  void forward(VideoBatch &batch, StageEvents &events) override;

private:
  metal::MetalBackend &backend_;
  std::filesystem::path weights_;
  std::optional<std::vector<model::AdapterSpec>> adapters_;
};

} // namespace splash::video
