#pragma once

// The Qwen3-TTS 12 Hz speech tokenizer decoder (codes -> 24 kHz waveform)
// over the kernels of runtime/metal/kernels/omni/codec.metal: RVQ lookup
// and projection, the causal pre-convolution, the 8-layer windowed
// transformer, the two ConvNeXt upsample stages and the SnakeBeta /
// transposed-convolution decoder stack. Activations and weights are fp32;
// every weight is repacked at load into the [taps][input][output] layout
// the implicit-GEMM convolution reads, and again in fp16 for the
// half-precision kernel that runs every convolution of whole groups of 8
// output channels (all of the model's).

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/OmniCodec.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace splash::ops::omni {

// The decoder's shape (speech_tokenizer/config.json decoder_config). The
// kernels take any channel counts that are multiples of 32 and heads of 64.
struct CodecDims final {
  uint32_t hidden = 512;
  uint32_t intermediate = 1024;
  uint32_t latent = 1024;
  uint32_t codebookDim = 512; // two halves of codebookDim / 2
  uint32_t codebookSize = 2048;
  uint32_t quantizers = 16;
  uint32_t decoderDim = 1536;
  uint32_t layers = 8;
  uint32_t heads = 16;
  uint32_t window = 72;
  float rmsEpsilon = 1e-5F;
  float ropeTheta = 10000.0F;
  std::vector<uint32_t> upsampleRates{8, 5, 4, 3};
  std::vector<uint32_t> upsamplingRatios{2, 2};
  // Output samples per frame: the product of the rates and ratios (1920).
  [[nodiscard]] uint32_t samplesPerFrame() const noexcept;
};

// A convolution, transposed convolution or linear as the GEMM kernel reads
// it: fp32 weight [taps][inputChannels][outputChannels] and an optional fp32
// bias [outputChannels]. A transposed convolution of stride s and kernel
// k s stores its s phases as outputChannels = s * channels over k taps.
struct CodecConv final {
  metal::MetalBuffer weight;     // fp32 [taps][inputChannels][outputChannels]
  metal::MetalBuffer weightHalf; // the same in fp16 for omni_codec_conv_half (empty: omni_codec_conv)
  metal::MetalBuffer bias;
  uint32_t inputChannels = 0;
  uint32_t outputChannels = 0;
  uint32_t taps = 1;
  uint32_t dilation = 1;
};

// SnakeBeta's per-channel exp(alpha) and 1 / (exp(beta) + 1e-9).
struct CodecSnake final {
  metal::MetalBuffer alpha;
  metal::MetalBuffer beta;
  uint32_t channels = 0;
};

struct CodecResidualUnit final {
  CodecSnake act1;
  CodecConv conv1; // k = 7, dilation 1, 3 or 9
  CodecSnake act2;
  CodecConv conv2; // k = 1
};

struct CodecDecoderBlock final {
  CodecSnake snake;
  CodecConv upsample; // transposed, kernel 2 * rate, stride rate (2 taps)
  uint32_t rate = 1;
  std::array<CodecResidualUnit, 3> units;
};

struct CodecConvNeXt final {
  CodecConv upsample; // transposed, kernel = stride = ratio (1 tap)
  uint32_t ratio = 1;
  metal::MetalBuffer depthwiseWeight; // fp32 [taps][channels]
  metal::MetalBuffer depthwiseBias;
  uint32_t depthwiseTaps = 7;
  metal::MetalBuffer normWeight, normBias; // LayerNorm eps 1e-6
  CodecConv pw1; // channels -> 4 channels, GELU
  CodecConv pw2; // 4 channels -> channels
  metal::MetalBuffer gamma;
};

struct CodecTransformerLayer final {
  metal::MetalBuffer inputNorm;
  CodecConv qkv; // hidden -> 3 * heads * 64 (queries, keys, values)
  CodecConv o;   // heads * 64 -> hidden
  metal::MetalBuffer attentionScale;
  metal::MetalBuffer postNorm;
  CodecConv gateUp; // hidden -> 2 * intermediate (gate, up)
  CodecConv down;   // intermediate -> hidden
  metal::MetalBuffer mlpScale;
};

struct CodecDecoderWeights final {
  // Every convolution of the decoder, in decode order.
  template <class Visit> void forEachConv(Visit &&visit);
  CodecDims dims;
  metal::MetalBuffer codebooks; // fp32 [quantizers][codebookSize][codebookDim / 2]
  CodecConv rvqProjection;      // codebookDim -> codebookDim, no bias
  CodecConv preConv;            // k = 3, codebookDim -> latent
  CodecConv inputProj;          // latent -> hidden
  std::vector<CodecTransformerLayer> layers;
  metal::MetalBuffer finalNorm;
  CodecConv outputProj; // hidden -> latent
  std::vector<CodecConvNeXt> upsample;
  CodecConv initialConv; // k = 7, latent -> decoderDim
  std::vector<CodecDecoderBlock> blocks;
  CodecSnake outputSnake;
  CodecConv outputConv; // k = 7, channels -> 1
};

template <class Visit> void CodecDecoderWeights::forEachConv(Visit &&visit) {
  visit(rvqProjection);
  visit(preConv);
  visit(inputProj);
  for (CodecTransformerLayer &layer : layers) {
    visit(layer.qkv);
    visit(layer.o);
    visit(layer.gateUp);
    visit(layer.down);
  }
  visit(outputProj);
  for (CodecConvNeXt &stage : upsample) {
    visit(stage.upsample);
    visit(stage.pw1);
    visit(stage.pw2);
  }
  visit(initialConv);
  for (CodecDecoderBlock &block : blocks) {
    visit(block.upsample);
    for (CodecResidualUnit &unit : block.units) {
      visit(unit.conv1);
      visit(unit.conv2);
    }
  }
  visit(outputConv);
}

// Builders from host fp32 tensors in the checkpoint's (PyTorch) layouts.
[[nodiscard]] CodecConv makeCodecConv(metal::MetalBackend &backend, const std::vector<float> &weight /* [out][in][taps] */,
                                      const std::vector<float> *bias, uint32_t outputChannels, uint32_t inputChannels,
                                      uint32_t taps, uint32_t dilation = 1);
// kernel is a multiple of stride; the reference trims kernel - stride
// samples on the right, which this layout never produces.
[[nodiscard]] CodecConv makeCodecTransposedConv(metal::MetalBackend &backend,
                                                const std::vector<float> &weight /* [in][out][kernel] */,
                                                const std::vector<float> *bias, uint32_t inputChannels,
                                                uint32_t outputChannels, uint32_t stride, uint32_t kernel);
[[nodiscard]] CodecConv makeCodecLinear(metal::MetalBackend &backend, const std::vector<float> &weight /* [out][in] */,
                                        const std::vector<float> *bias, uint32_t outputChannels, uint32_t inputChannels);
[[nodiscard]] CodecSnake makeCodecSnake(metal::MetalBackend &backend, const std::vector<float> &alpha,
                                        const std::vector<float> &beta);
[[nodiscard]] metal::MetalBuffer makeCodecVector(metal::MetalBackend &backend, const std::vector<float> &values);

// Reads the decoder's tensors of the speech tokenizer checkpoint (the
// speech_tokenizer/model.safetensors file or its directory) into fp32 shared
// buffers; the encoder's tensors are not read.
[[nodiscard]] CodecDecoderWeights loadCodecDecoderWeights(metal::MetalBackend &backend,
                                                           const std::filesystem::path &speechTokenizerSafetensors);

// A tensor of `rows` rows of `width` contiguous fp32 elements in a buffer.
struct CodecRows final {
  metal::MetalBuffer buffer;
  uint32_t rows = 0;
  uint32_t width = 0;
  [[nodiscard]] uint64_t bytes() const noexcept { return rows ? uint64_t{rows} * width * 4 : 0; }
};

// The left context one causal convolution of a stream keeps: its last
// `length` input rows of `width` channels, in two halves the chunks
// alternate between (one read by the chunk's convolution, the other written
// with the updated history).
struct CodecHistory final {
  std::array<metal::MetalBuffer, 2> rows;
  uint32_t length = 0;
  uint32_t width = 0;
};

// The state of one request's incremental decode: the history of every
// causal convolution (transposed ones included, as their taps over input
// rows) and a ring of the last window - 1 + maximumFrames rotated keys and
// values per transformer layer, with the frame position continuing across
// chunks. Chunks of 1 .. maximumFrames frames decode, through
// CodecDecoder::addDecodeStream, to the samples a whole-sequence decode
// yields. The host writes nothing between chunks; reset() (between
// requests, with no command in flight on the stream) clears the histories
// and the position.
class CodecStream final {
public:
  CodecStream(metal::MetalBackend &backend, const CodecDecoderWeights &weights, uint32_t maximumFrames);
  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] uint32_t maximumFrames() const noexcept { return maximumFrames_; }
  // Frames decoded since the reset.
  [[nodiscard]] uint32_t position() const noexcept { return position_; }
  // The history half the next chunk reads.
  [[nodiscard]] uint32_t phase() const noexcept { return phase_; }
  // The histories of the convolution stack (the initial convolution through
  // the output convolution), in decode order.
  [[nodiscard]] std::span<const CodecHistory> stackHistories() const noexcept {
    return std::span<const CodecHistory>(histories_).subspan(frontHistories_);
  }
  void reset();

private:
  friend class CodecDecoder;
  uint32_t frontHistories_ = 0; // the pre-conv's and the ConvNeXt stages' histories, before the stack's
  uint32_t maximumFrames_ = 0;
  uint32_t capacity_ = 0; // key/value ring rows
  uint32_t position_ = 0;
  uint32_t phase_ = 0; // the history half the next chunk reads
  uint64_t bytes_ = 0;
  std::vector<CodecHistory> histories_;
  std::vector<metal::MetalBuffer> keys_, values_; // per layer, [capacity_][heads * 64]
};

// Decodes code sequences of up to maximumFrames frames (whole, or chunk by
// chunk through a CodecStream) into a workspace allocated once
// (workspaceBytes). A whole sequence is decoded in one pass, as the
// reference decodes a chunk; the caller supplies the left context and trims
// it. Several streams' chunks may be encoded into one graph.
class CodecDecoder final {
public:
  CodecDecoder(metal::MetalBackend &backend, CodecDecoderWeights weights, uint32_t maximumFrames);
  [[nodiscard]] static uint64_t workspaceBytes(const CodecDims &dims, uint32_t maximumFrames);
  [[nodiscard]] uint32_t maximumFrames() const noexcept { return maximumFrames_; }
  [[nodiscard]] const CodecDecoderWeights &weights() const noexcept { return weights_; }
  // Releases the fp32 weights of every convolution the GPU runs from its
  // fp16 copy: the Neural Engine's program reads them when it is built,
  // nothing after. Those convolutions then have their fp16 copy alone.
  void releaseFloatCopies();
  // codes: u32 [frames][quantizers]; output: fp32 [frames * samplesPerFrame].
  void addDecode(metal::CommandGraph &graph, metal::MetalBuffer codes, uint32_t frames, metal::MetalBuffer output);
  // The next `frames` (1 .. the stream's and the decoder's maximum) of the
  // stream's sequence; the stream advances when the graph is encoded, so
  // its chunks must run in encoding order.
  void addDecodeStream(metal::CommandGraph &graph, CodecStream &stream, metal::MetalBuffer codes, uint32_t frames,
                       metal::MetalBuffer output);

  // A stream's chunk in two parts, for a convolution stack that runs
  // elsewhere (CodecAneStack): addFrontStream decodes the codes through the
  // ConvNeXt upsample stages into stackInput(frames) and advances the
  // stream (its position and phase) as addDecodeStream does; addStackStream
  // runs the GPU's convolution stack over stackInput(frames) with the
  // stream's stack histories at `phase` (the stream's phase before the
  // front of those rows advanced it), reading that half and writing the
  // other.
  void addFrontStream(metal::CommandGraph &graph, CodecStream &stream, metal::MetalBuffer codes, uint32_t frames);
  void addStackStream(metal::CommandGraph &graph, CodecStream &stream, uint32_t frames, uint32_t phase,
                      metal::MetalBuffer output);
  // The rows between the front and the stack: fp32 [frames * the upsampling
  // ratios][latent] in the workspace, valid until the next decode encoded
  // in the graph.
  [[nodiscard]] CodecRows stackInput(uint32_t frames) const;

private:
  struct Encode;
  void encode(Encode &e, metal::MetalBuffer codes, metal::MetalBuffer output);
  void encodeFront(Encode &e, metal::MetalBuffer codes);
  void encodeStack(Encode &e, metal::MetalBuffer output);
  void addTransformer(Encode &e, const CodecRows &input, const CodecRows &output);
  void addConvNeXt(Encode &e, const CodecConvNeXt &stage, const CodecRows &input, const CodecRows &output,
                   const CodecRows &scratchA, const CodecRows &scratchB);
  void addBlock(Encode &e, const CodecDecoderBlock &block, const CodecRows &input, const CodecRows &output,
                const CodecRows &scratch);

  metal::MetalBackend &backend_;
  CodecDecoderWeights weights_;
  uint32_t maximumFrames_ = 0;
  uint64_t slabBytes_ = 0;
  std::array<metal::MetalBuffer, 3> slabs_;
  metal::MetalBuffer residual_, normed_, qkv_, attention_, gateUp_, hidden_;
  std::vector<metal::MetalBuffer> keys_, values_; // whole-sequence rings, [maximumFrames][heads * 64]
};

} // namespace splash::ops::omni
