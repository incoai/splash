#include "ops/OmniAudioEncoder.hpp"

#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops::omni {
namespace {

constexpr uint32_t kMelBins = OMNI_AUDIO_MEL_BINS;
constexpr uint32_t kChunkFrames = OMNI_AUDIO_CHUNK_FRAMES;
constexpr uint32_t kChunkTokens = OMNI_AUDIO_CHUNK_TOKENS;
constexpr uint32_t kChannels = OMNI_AUDIO_CONV_CHANNELS;
constexpr uint32_t kPaddedChannels = OMNI_AUDIO_CONV_PADDED_CHANNELS;
[[maybe_unused]] constexpr uint32_t kTaps = OMNI_AUDIO_CONV_TAPS;
constexpr uint32_t kPaddedTaps = OMNI_AUDIO_CONV_PADDED_TAPS;
constexpr uint32_t kHeight1 = OMNI_AUDIO_CONV1_HEIGHT;
constexpr uint32_t kHeight2 = OMNI_AUDIO_CONV2_HEIGHT;
constexpr uint32_t kHeight3 = OMNI_AUDIO_CONV3_HEIGHT;
constexpr uint32_t kConvOutInput = OMNI_AUDIO_CONV_OUT_INPUT;
constexpr uint32_t kWindow = OMNI_AUDIO_WINDOW_TOKENS;
constexpr float kLayerNormEpsilon = 1e-5F;
constexpr float kAttentionScale = 0.125F; // 1 / sqrt(64)

uint32_t roundUp(uint32_t value, uint32_t multiple) noexcept { return (value + multiple - 1) / multiple * multiple; }
// The output length of a 3-wide stride-2 pad-1 convolution over `length`.
uint32_t halved(uint32_t length) noexcept { return length ? (length - 1) / 2 + 1 : 0; }
// Conv1's output width of a chunk of `frames` frames, then conv2's, conv3's.
uint32_t width1(uint32_t frames) noexcept { return halved(frames); }
uint32_t width2(uint32_t frames) noexcept { return halved(width1(frames)); }
uint32_t width3(uint32_t frames) noexcept { return halved(width2(frames)); }

void requireLayerNorm(const AudioLayerNorm &norm, uint32_t width, const char *what) {
  requireBytes(norm.weight, uint64_t{width} * 2, std::string(what) + " weight");
  requireBytes(norm.bias, uint64_t{width} * 2, std::string(what) + " bias");
}
void requireShape(const DenseWeights &w, uint32_t outputSize, uint32_t inputSize, bool bias, const char *what) {
  if (w.outputSize != outputSize || w.inputSize != inputSize || static_cast<bool>(w.bias) != bias)
    throw std::invalid_argument(std::string(what) + ": unexpected projection shape");
}

// The 13 x hidden sinusoidal positions of a chunk: sin over the first half
// of the dimensions, cos over the last, timescales
// exp(-ln(10000) / (half - 1) * i).
std::vector<float> positionTable(uint32_t hidden) {
  const uint32_t half = hidden / 2;
  std::vector<float> table(size_t(kChunkTokens) * hidden);
  const double increment = std::log(10000.0) / double(half - 1);
  for (uint32_t t = 0; t < kChunkTokens; ++t)
    for (uint32_t i = 0; i < half; ++i) {
      // The reference forms the timescales and angles in fp32.
      const float inverse = float(std::exp(-increment * double(i)));
      const float angle = float(t) * inverse;
      table[size_t(t) * hidden + i] = std::sin(angle);
      table[size_t(t) * hidden + half + i] = std::cos(angle);
    }
  return table;
}

metal::MetalBuffer zeroed(metal::MetalBackend &backend, uint64_t bytes, const char *label) {
  metal::MetalBuffer buffer = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

} // namespace

uint32_t audioTokens(uint32_t melFrames) noexcept {
  const uint32_t remainder = melFrames % kChunkFrames;
  return (melFrames / kChunkFrames) * kChunkTokens + width3(remainder);
}

bool packableClip(uint32_t melFrames) noexcept { return audioTokens(melFrames) > 4; }

std::vector<OmniDenseRow> audioAttentionRows(uint32_t tokens) {
  std::vector<OmniDenseRow> rows(tokens);
  for (uint32_t r = 0; r < tokens; ++r) {
    const uint32_t window = r / kWindow;
    rows[r] = {window * kWindow, std::min((window + 1) * kWindow, tokens), r, 0};
  }
  return rows;
}

AudioEncoder::AudioEncoder(metal::MetalBackend &backend, AudioEncoderWeights weights, uint32_t maximumMelFrames,
                           uint32_t groupChunks)
    : backend_(backend), weights_(std::move(weights)), maximumFrames_(maximumMelFrames) {
  if (!maximumMelFrames) throw std::invalid_argument("the audio encoder takes at least one mel frame");
  if (!groupChunks) throw std::invalid_argument("the audio encoder convolves at least one chunk at a time");
  requireBytes(weights_.conv1Weight, uint64_t{kChannels} * 9 * 2, "conv1 weight");
  requireBytes(weights_.conv1Bias, uint64_t{kChannels} * 4, "conv1 bias");
  requireShape(weights_.conv2, kPaddedChannels, kPaddedTaps, true, "conv2");
  requireShape(weights_.conv3, kPaddedChannels, kPaddedTaps, true, "conv3");
  const AudioEncoderDims &d = weights_.dims;
  if (!d.hidden || d.hidden % OMNI_ENCODER_HEAD_DIM || !d.ffn || d.ffn % 64 || !d.output || d.output % 64)
    throw std::invalid_argument("the audio encoder takes whole heads of 64 and 64-wide FFN and output");
  requireShape(weights_.convOut, d.hidden, kConvOutInput, false, "conv_out");
  if (weights_.layers.size() != d.layers || weights_.layers.empty())
    throw std::invalid_argument("the audio encoder's layers do not match its dimensions");
  for (const AudioEncoderLayerWeights &layer : weights_.layers) {
    requireLayerNorm(layer.attentionNorm, d.hidden, "attention norm");
    requireLayerNorm(layer.ffnNorm, d.hidden, "ffn norm");
    requireShape(layer.qkv, 3 * d.hidden, d.hidden, true, "qkv");
    requireShape(layer.out, d.hidden, d.hidden, true, "out_proj");
    requireShape(layer.fc1, d.ffn, d.hidden, true, "fc1");
    requireShape(layer.fc2, d.hidden, d.ffn, true, "fc2");
  }
  requireLayerNorm(weights_.postNorm, d.hidden, "ln_post");
  requireShape(weights_.proj1, d.hidden, d.hidden, true, "proj1");
  requireShape(weights_.proj2, d.output, d.hidden, true, "proj2");

  const uint32_t chunks = (maximumMelFrames + kChunkFrames - 1) / kChunkFrames;
  groupChunks_ = std::min(groupChunks, chunks);
  maximumTokens_ = audioTokens(maximumMelFrames);
  // A packed clip's attention reads its last key block whole, up to 31 rows
  // past the packed rows.
  paddedTokens_ = roundUp(maximumTokens_ + OMNI_DENSE_KEY_BLOCK - 1, 64);
  const uint32_t w1 = width1(kChunkFrames), w2 = width2(kChunkFrames), w3 = width3(kChunkFrames);
  const uint64_t group = groupChunks_;
  conv1Out_ = zeroed(backend, group * kHeight1 * w1 * kChannels * 2, "audio conv1");
  im2col_ = zeroed(backend, uint64_t{roundUp(uint32_t(group * kHeight2 * w2), 64)} * kPaddedTaps * 2, "audio im2col");
  conv2Out_ = zeroed(backend, uint64_t{roundUp(uint32_t(group * kHeight2 * w2), 64)} * kPaddedChannels * 2,
                     "audio conv2");
  conv3Out_ = zeroed(backend, uint64_t{roundUp(uint32_t(group * kHeight3 * w3), 64)} * kPaddedChannels * 2,
                     "audio conv3");
  flat_ = zeroed(backend, uint64_t{paddedTokens_} * kConvOutInput * 2, "audio conv_out input");
  hidden_ = zeroed(backend, uint64_t{paddedTokens_} * d.hidden * 4, "audio hidden");
  normed_ = zeroed(backend, uint64_t{paddedTokens_} * d.hidden * 2, "audio normed");
  qkv_ = zeroed(backend, uint64_t{paddedTokens_} * 3 * d.hidden * 2, "audio qkv");
  // The keys and values of a row are its qkv row from element hidden and
  // 2 hidden on, at the qkv stride.
  keys_ = backend.view(qkv_, uint64_t{d.hidden} * 2, qkv_.sizeBytes() - uint64_t{d.hidden} * 2);
  values_ = backend.view(qkv_, uint64_t{2 * d.hidden} * 2, qkv_.sizeBytes() - uint64_t{2 * d.hidden} * 2);
  attention_ = zeroed(backend, uint64_t{paddedTokens_} * d.hidden * 2, "audio attention");
  ffn_ = zeroed(backend, uint64_t{paddedTokens_} * d.ffn * 2, "audio ffn");
  rows_ = zeroed(backend, uint64_t{paddedTokens_} * sizeof(OmniDenseRow), "audio attention rows");
  const std::vector<float> table = positionTable(d.hidden);
  positions_ = zeroed(backend, table.size() * 4, "audio positions");
  std::memcpy(positions_.contents(), table.data(), table.size() * 4);
}

// The front end of one clip of `tokens` tokens into `flat`, the clip's rows
// of the conv_out input.
void AudioEncoder::addConvolutions(metal::CommandGraph &graph, const metal::MetalBuffer &mel, uint32_t melFrames,
                                   uint32_t tokens, const metal::MetalBuffer &flat) {
  const uint32_t chunks = (melFrames + kChunkFrames - 1) / kChunkFrames;
  // A lone chunk is convolved at its own length; with more, the last chunk
  // is zero-padded to 100 frames and its surplus frames dropped after.
  const uint32_t chunkFrames = chunks == 1 ? melFrames : kChunkFrames;
  const uint32_t w1 = width1(chunkFrames), w2 = width2(chunkFrames), w3 = width3(chunkFrames);
  const uint32_t lastTokens = tokens - (chunks - 1) * kChunkTokens;
  if (!lastTokens || lastTokens > w3) throw std::logic_error("audio chunk tokens exceed the conv3 width");
  for (uint32_t first = 0; first < chunks; first += groupChunks_) {
    const uint32_t group = std::min(groupChunks_, chunks - first);
    OmniAudioConv1Params conv1{};
    conv1.first_chunk = first;
    conv1.chunks = group;
    conv1.mel_frames = melFrames;
    conv1.chunk_frames = chunkFrames;
    conv1.output_width = w1;
    // Eight channels a thread.
    const uint32_t conv1Threads = group * kHeight1 * w1 * (kChannels / 8);
    graph.add("omni_audio_conv1", {mel, weights_.conv1Weight, weights_.conv1Bias, conv1Out_}, conv1,
              {(conv1Threads + 255) / 256, 1, 1});

    const auto convolve = [&](const metal::MetalBuffer &input, uint32_t inputHeight, uint32_t inputWidth,
                              uint32_t inputStride, uint32_t outputHeight, uint32_t outputWidth,
                              const DenseWeights &weights, const metal::MetalBuffer &output) {
      const uint32_t rows = group * outputHeight * outputWidth;
      OmniAudioIm2colParams im2col{group, inputHeight, inputWidth, inputStride, outputHeight, outputWidth, rows, 0};
      graph.add("omni_audio_im2col", {input, im2col_}, im2col, {(rows * (kPaddedTaps / 8) + 255) / 256, 1, 1});
      GemmBuffers buffers{im2col_, output, {}, kPaddedTaps, kPaddedChannels};
      DenseLinear::add(graph, weights, buffers, rows, {Activation::Gelu, false, FloatOutput::BFloat16, false});
    };
    convolve(conv1Out_, kHeight1, w1, kChannels, kHeight2, w2, weights_.conv2, conv2Out_);
    convolve(conv2Out_, kHeight2, w2, kPaddedChannels, kHeight3, w3, weights_.conv3, conv3Out_);

    OmniAudioFlattenParams flatten{};
    flatten.first_chunk = first;
    flatten.chunks = group;
    flatten.width = w3;
    flatten.last_chunk = chunks - 1;
    flatten.last_chunk_tokens = lastTokens;
    const uint64_t flatElements = uint64_t{group} * w3 * kConvOutInput;
    graph.add("omni_audio_flatten", {conv3Out_, flat}, flatten, {(flatElements + 255) / 256, 1, 1});
  }
}

void AudioEncoder::addLayer(metal::CommandGraph &graph, const AudioEncoderLayerWeights &layer, uint32_t tokens,
                            std::span<const ClipRows> clips) {
  const GemmOptions residual{Activation::None, true, FloatOutput::Float32, false};
  const AudioEncoderDims &d = weights_.dims;
  Norms::addLayer(graph, hidden_, layer.attentionNorm.weight, layer.attentionNorm.bias, normed_, tokens, d.hidden,
                  kLayerNormEpsilon, false);
  DenseLinear::add(graph, layer.qkv, {normed_, qkv_, {}, 0, 0}, tokens, {.stepTiles = false});
  // Each clip's attention over its rows' views, so its key blocks start at
  // its first row as in its own encode.
  const uint64_t qkvRow = uint64_t{3 * d.hidden} * 2, row = uint64_t{d.hidden} * 2;
  for (const ClipRows &clip : clips) {
    const auto view = [&](const metal::MetalBuffer &buffer, uint64_t rowBytes) {
      return backend_.view(buffer, clip.offset * rowBytes, buffer.sizeBytes() - clip.offset * rowBytes);
    };
    DenseAttention::add(graph,
                        {view(qkv_, qkvRow), view(keys_, qkvRow), view(values_, qkvRow), view(attention_, row),
                         view(rows_, sizeof(OmniDenseRow))},
                        clip.tokens, roundUp(clip.tokens, OMNI_DENSE_KEY_BLOCK), 3 * d.hidden, DenseMask::None, 0,
                        kAttentionScale, d.heads());
  }
  DenseLinear::add(graph, layer.out, {attention_, hidden_, hidden_, 0, 0}, tokens, residual);
  Norms::addLayer(graph, hidden_, layer.ffnNorm.weight, layer.ffnNorm.bias, normed_, tokens, d.hidden,
                  kLayerNormEpsilon, false);
  DenseLinear::add(graph, layer.fc1, {normed_, ffn_, {}, 0, 0}, tokens,
                   {Activation::Gelu, false, FloatOutput::BFloat16, false});
  DenseLinear::add(graph, layer.fc2, {ffn_, hidden_, hidden_, 0, 0}, tokens, residual);
}

void AudioEncoder::addEncode(metal::CommandGraph &graph, metal::MetalBuffer mel, uint32_t melFrames,
                             metal::MetalBuffer output) {
  const AudioClip clip{std::move(mel), melFrames};
  addEncode(graph, std::span<const AudioClip>(&clip, 1), std::move(output));
}

void AudioEncoder::addEncode(metal::CommandGraph &graph, std::span<const AudioClip> clips, metal::MetalBuffer output) {
  if (clips.empty()) throw std::invalid_argument("audio encode takes at least one clip");
  std::vector<ClipRows> rows;
  uint32_t tokens = 0;
  for (const AudioClip &clip : clips) {
    if (!clip.melFrames || clip.melFrames > maximumFrames_)
      throw std::invalid_argument("audio encode takes 1.." + std::to_string(maximumFrames_) + " mel frames");
    requireBytes(clip.mel, uint64_t{clip.melFrames} * kMelBins * 4, "audio mel");
    if (clips.size() > 1 && !packableClip(clip.melFrames))
      throw std::invalid_argument("packed audio clips take more than four tokens each");
    rows.push_back({tokens, audioTokens(clip.melFrames)});
    tokens += rows.back().tokens;
  }
  if (tokens > maximumTokens_)
    throw std::invalid_argument("packed audio clips take more than " + std::to_string(maximumTokens_) + " tokens");
  requireBytes(output, uint64_t{tokens} * weights_.dims.output * 2, "audio embedding");
  // The clips' attention tables, each relative to its first row, and their
  // front ends and positions.
  const uint32_t hidden = weights_.dims.hidden;
  for (size_t i = 0; i < clips.size(); ++i) {
    const ClipRows &clip = rows[i];
    const std::vector<OmniDenseRow> table = audioAttentionRows(clip.tokens);
    std::memcpy(static_cast<OmniDenseRow *>(rows_.contents()) + clip.offset, table.data(),
                table.size() * sizeof(OmniDenseRow));
    addConvolutions(graph, clips[i].mel, clips[i].melFrames, clip.tokens,
                    backend_.view(flat_, uint64_t{clip.offset} * kConvOutInput * 2,
                                  uint64_t{clip.tokens} * kConvOutInput * 2));
  }
  DenseLinear::add(graph, weights_.convOut, {flat_, hidden_, {}, 0, 0}, tokens,
                   {Activation::None, false, FloatOutput::Float32, false});
  for (const ClipRows &clip : rows) {
    const OmniAudioPositionParams positions{clip.tokens, hidden, kChunkTokens, 0};
    graph.add("omni_audio_positions",
              {backend_.view(hidden_, uint64_t{clip.offset} * hidden * 4, uint64_t{clip.tokens} * hidden * 4),
               positions_},
              positions, {(uint64_t{clip.tokens} * hidden + 255) / 256, 1, 1});
  }
  for (const AudioEncoderLayerWeights &layer : weights_.layers) addLayer(graph, layer, tokens, rows);
  Norms::addLayer(graph, hidden_, weights_.postNorm.weight, weights_.postNorm.bias, normed_, tokens,
                  weights_.dims.hidden, kLayerNormEpsilon, false);
  DenseLinear::add(graph, weights_.proj1, {normed_, attention_, {}, 0, 0}, tokens,
                   {Activation::Gelu, false, FloatOutput::BFloat16, false});
  DenseLinear::add(graph, weights_.proj2, {attention_, std::move(output), {}, 0, 0}, tokens, {.stepTiles = false});
}

} // namespace splash::ops::omni
