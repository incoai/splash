#pragma once

// The Qwen3-ASR audio tower on the Omni kernels: log-mel frames
// ([frames][128] fp32, omni/MelSpectrogram.hpp) to audio embeddings of the
// text decoder's width (bf16 [tokens][output]), 13 tokens per 100-frame
// chunk.
//
// Per chunk: conv2d1 (direct kernel), conv2d2 and conv2d3 (im2col then
// ops::omni::DenseLinear with GELU), the channel-major flatten, conv_out,
// sinusoidal positions 0..12. Then the pre-LayerNorm layers on an fp32
// residual stream with bf16 projection inputs: biased QKV, dense attention
// of heads of 64 block-diagonal over 104-token windows, out_proj, fc1
// (GELU) and fc2; ln_post, proj1 (GELU), proj2. The chunks of a long signal
// run the convolutions in groups, so the workspace is bounded by the group
// and by the maximum frames of one encode. The convolution front end (480
// channels) is the same in every checkpoint; the transformer's width,
// depth, FFN and output differ (AudioEncoderDims).

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Omni.h"
#include "metal/abi/OmniAudioEncoder.h"
#include "ops/Omni.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace splash::model {
class SafetensorsCheckpoint;
}

namespace splash::ops::omni {

// The tower's transformer: Qwen3-ASR-1.7B's (the defaults) or 0.6B's (896
// wide, 18 layers of 14 heads, FFN 3584, 1024-dim output). Heads are 64
// wide.
struct AudioEncoderDims final {
  uint32_t hidden = OMNI_AUDIO_HIDDEN;
  uint32_t layers = 24;
  uint32_t ffn = OMNI_AUDIO_FFN;
  uint32_t output = OMNI_AUDIO_OUTPUT;
  [[nodiscard]] uint32_t heads() const noexcept { return hidden / OMNI_ENCODER_HEAD_DIM; }
  bool operator==(const AudioEncoderDims &) const = default;
};

// bf16 weight and bias of a LayerNorm over the hidden width (eps 1e-5).
struct AudioLayerNorm final {
  metal::MetalBuffer weight;
  metal::MetalBuffer bias;
};

struct AudioEncoderLayerWeights final {
  AudioLayerNorm attentionNorm; // self_attn_layer_norm
  DenseWeights qkv;             // q_proj, k_proj, v_proj stacked: [3 hidden][hidden], fp32 bias
  DenseWeights out;             // out_proj [hidden][hidden] + bias
  AudioLayerNorm ffnNorm;       // final_layer_norm
  DenseWeights fc1;             // [ffn][hidden] + bias
  DenseWeights fc2;             // [hidden][ffn] + bias
};

struct AudioEncoderWeights final {
  AudioEncoderDims dims;
  metal::MetalBuffer conv1Weight; // bf16 [480][3][3] (out, kh, kw)
  metal::MetalBuffer conv1Bias;   // fp32 [480]
  // conv2d2 and conv2d3 as GEMMs: bf16 [512][4352], row co's taps at
  // (kh * 3 + kw) * 480 + ci (the MLX [out][kh][kw][in] layout flattened),
  // zero past 4320 and past row 479; fp32 bias [512], zero past 479.
  DenseWeights conv2;
  DenseWeights conv3;
  DenseWeights convOut; // [hidden][7680], no bias
  std::vector<AudioEncoderLayerWeights> layers;
  AudioLayerNorm postNorm; // ln_post
  DenseWeights proj1;      // [hidden][hidden] + bias
  DenseWeights proj2;      // [output][hidden] + bias
};

// The audio tokens of `melFrames` mel frames: 13 per full 100-frame chunk
// plus the three stride-2 reductions of the remainder.
[[nodiscard]] uint32_t audioTokens(uint32_t melFrames) noexcept;

// The dense attention row table of `tokens` tokens: bidirectional within
// windows of 104 tokens, the last window shorter.
[[nodiscard]] std::vector<OmniDenseRow> audioAttentionRows(uint32_t tokens);

// One request's audio in a packed encode: its log-mel frames (fp32
// [melFrames][128]).
struct AudioClip final {
  metal::MetalBuffer mel;
  uint32_t melFrames = 0;
};

// Whether a clip of `melFrames` frames may share a packed encode, which
// gives it the bits of its own: its projections run the matrix tiles (more
// than four tokens), whose rows' arithmetic does not depend on the rows
// around them or on the tile's shape (omni_audio_encoder_test); up to four
// tokens run the GEMV.
[[nodiscard]] bool packableClip(uint32_t melFrames) noexcept;

class AudioEncoder final {
public:
  // A workspace for encodes of up to maximumMelFrames frames, the
  // convolutions running up to groupChunks chunks at a time.
  AudioEncoder(metal::MetalBackend &backend, AudioEncoderWeights weights, uint32_t maximumMelFrames,
               uint32_t groupChunks = 8);

  [[nodiscard]] uint32_t maximumMelFrames() const noexcept { return maximumFrames_; }
  // The most tokens of one encode, packed or not: audioTokens(maximumMelFrames).
  [[nodiscard]] uint32_t maximumTokens() const noexcept { return maximumTokens_; }
  [[nodiscard]] const AudioEncoderWeights &weights() const noexcept { return weights_; }

  // Encodes the `melFrames` frames of `mel` (fp32 [melFrames][128]) into
  // `output` (bf16 [audioTokens(melFrames)][dims.output]). The workspace serves one
  // encode at a time: the graph must complete before the next addEncode.
  void addEncode(metal::CommandGraph &graph, metal::MetalBuffer mel, uint32_t melFrames, metal::MetalBuffer output);
  // Encodes several requests' clips in one pass into `output`, clip i's
  // audioTokens rows after the rows of the clips before it, each row equal
  // bit for bit to the clip's own encode. Every clip runs its own front end
  // and positions; the norms and projections run once over the packed rows,
  // and each clip's attention runs over its own windows (one dispatch per
  // clip, whose key blocks start where the clip's rows do, as in its own
  // encode). Clips of a packed encode are packableClip; their tokens
  // together take at most maximumTokens().
  void addEncode(metal::CommandGraph &graph, std::span<const AudioClip> clips, metal::MetalBuffer output);

private:
  // A clip's rows in the packed workspace.
  struct ClipRows final {
    uint32_t offset = 0, tokens = 0;
  };
  void addConvolutions(metal::CommandGraph &graph, const metal::MetalBuffer &mel, uint32_t melFrames,
                       uint32_t tokens, const metal::MetalBuffer &flat);
  void addLayer(metal::CommandGraph &graph, const AudioEncoderLayerWeights &layer, uint32_t tokens,
                std::span<const ClipRows> clips);

  metal::MetalBackend &backend_;

  AudioEncoderWeights weights_;
  uint32_t maximumFrames_ = 0;
  uint32_t maximumTokens_ = 0;
  uint32_t paddedTokens_ = 0;
  uint32_t groupChunks_ = 0;
  metal::MetalBuffer conv1Out_;  // bf16 [group][64][50][480]
  metal::MetalBuffer im2col_;    // bf16 [group * 32 * 25 (padded)][4352]
  metal::MetalBuffer conv2Out_;  // bf16 [group * 32 * 25][512]
  metal::MetalBuffer conv3Out_;  // bf16 [group * 16 * 13][512]
  metal::MetalBuffer flat_;      // bf16 [paddedTokens][7680]
  metal::MetalBuffer hidden_;    // fp32 [paddedTokens][hidden], the residual stream
  metal::MetalBuffer normed_;    // bf16 [paddedTokens][hidden]
  metal::MetalBuffer qkv_;       // bf16 [paddedTokens][3 hidden]
  metal::MetalBuffer keys_;      // qkv_ from element hidden on
  metal::MetalBuffer values_;    // qkv_ from element 2 hidden on
  metal::MetalBuffer attention_; // bf16 [paddedTokens][hidden]
  metal::MetalBuffer ffn_;       // bf16 [paddedTokens][ffn]
  metal::MetalBuffer rows_;      // OmniDenseRow [paddedTokens]
  metal::MetalBuffer positions_; // fp32 [13][hidden]
};

// Reads audio_tower.* of a Qwen3-ASR checkpoint of the tower `dims` (bf16
// as stored, biases to fp32, the conv GEMM weights padded) into shared
// Metal buffers.
[[nodiscard]] AudioEncoderWeights loadAudioEncoderWeights(metal::MetalBackend &backend,
                                                          const splash::model::SafetensorsCheckpoint &checkpoint,
                                                          const AudioEncoderDims &dims = {});

} // namespace splash::ops::omni
