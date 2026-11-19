#include "ops/OmniCodec.hpp"

#include "ops/BufferExtent.hpp"
#include "ops/Omni.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops::omni {
namespace {

constexpr uint32_t kHeadDim = OMNI_CODEC_HEAD_DIM;
constexpr uint32_t kConvDepth = OMNI_CODEC_CONV_TILE_DEPTH;

uint64_t ceilDiv(uint64_t value, uint64_t divisor) noexcept { return (value + divisor - 1) / divisor; }

// IEEE half of a float, round to nearest even (the clang vector conversion
// is unavailable in C++ here).
uint16_t floatToHalfBits(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const int32_t exponent = int32_t((bits >> 23) & 0xFFu) - 127 + 15;
  uint32_t mantissa = bits & 0x7FFFFFu;
  if (((bits >> 23) & 0xFFu) == 0xFFu) return uint16_t(sign | 0x7C00u | (mantissa ? 0x200u : 0u)); // inf / nan
  if (exponent >= 31) return uint16_t(sign | 0x7C00u);                                              // overflow
  if (exponent <= 0) {
    if (exponent < -10) return uint16_t(sign);
    mantissa |= 0x800000u;
    const uint32_t shift = uint32_t(14 - exponent);
    uint32_t half = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1u << shift) - 1u), halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (half & 1u))) ++half;
    return uint16_t(sign | half);
  }
  uint32_t half = uint32_t(exponent) << 10 | (mantissa >> 13);
  const uint32_t remainder = mantissa & 0x1FFFu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) ++half; // may carry into the exponent
  return uint16_t(sign | half);
}

metal::MetalBuffer uploadHalves(metal::MetalBackend &backend, const std::vector<float> &values, const char *label) {
  metal::MetalBuffer buffer = backend.allocateBuffer(values.size() * 2, metal::BufferStorage::Shared, label);
  auto *bits = static_cast<uint16_t *>(buffer.contents());
  for (size_t i = 0; i < values.size(); ++i) bits[i] = floatToHalfBits(values[i]);
  return buffer;
}

metal::MetalBuffer uploadFloats(metal::MetalBackend &backend, const std::vector<float> &values, const char *label) {
  if (values.empty()) throw std::invalid_argument(std::string(label) + ": an empty tensor");
  metal::MetalBuffer buffer = backend.allocateBuffer(values.size() * sizeof(float), metal::BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(float));
  return buffer;
}

void requireCount(const std::vector<float> &values, uint64_t count, const char *what) {
  if (values.size() != count)
    throw std::invalid_argument(std::string(what) + " holds " + std::to_string(values.size()) + " values, expected " +
                                std::to_string(count));
}

void requireConv(const CodecConv &conv, const char *what) {
  if (!conv.inputChannels || conv.inputChannels % kConvDepth || !conv.outputChannels || conv.outputChannels % 4 ||
      !conv.taps || !conv.dilation)
    throw std::invalid_argument(std::string(what) + ": codec convolutions take input channels in multiples of 32 and "
                                                    "output channels in multiples of 4");
  requireBytes(conv.weight, uint64_t{conv.taps} * conv.inputChannels * conv.outputChannels * 4, "codec conv weight");
  if (conv.bias) requireBytes(conv.bias, uint64_t{conv.outputChannels} * 4, "codec conv bias");
}

void requireSnake(const CodecSnake &snake, uint32_t channels) {
  if (snake.channels != channels) throw std::invalid_argument("codec snake channels do not match the input");
  requireBytes(snake.alpha, uint64_t{channels} * 4, "codec snake alpha");
  requireBytes(snake.beta, uint64_t{channels} * 4, "codec snake beta");
}

} // namespace

namespace {

// A stream's history of one convolution's input for this chunk: the half
// the convolution reads and the half the update writes.
struct HistoryUse final {
  const CodecHistory *history = nullptr;
  uint32_t phase = 0;
  [[nodiscard]] uint32_t rows() const noexcept { return history ? history->length : 0; }
};

struct ConvCall final {
  CodecRows input;
  uint32_t rows = 0; // output rows
  int32_t pad = 0;
  metal::MetalBuffer output;
  uint32_t outputStride = 0;
  metal::MetalBuffer residual;
  metal::MetalBuffer scale;
  bool gelu = false;
  HistoryUse history;
};

void requireHistory(const HistoryUse &use, uint32_t width, uint32_t length, const char *what) {
  if (!use.history) return;
  if (use.history->width != width || use.history->length != length)
    throw std::invalid_argument(std::string(what) + ": the stream history does not match the convolution");
  for (const metal::MetalBuffer &half : use.history->rows)
    requireBytes(half, uint64_t{length} * width * 4, "codec history");
}

// After a convolution read the history's current half: the new history is
// the last `length` rows of [history; input], written to the other half.
void addHistoryUpdate(metal::CommandGraph &graph, const HistoryUse &use, const CodecRows &input) {
  if (!use.history) return;
  const CodecHistory &h = *use.history;
  const OmniCodecHistoryParams params{input.rows, h.length, h.width, 0};
  graph.add("omni_codec_history", {input.buffer, h.rows[use.phase], h.rows[use.phase ^ 1]}, params,
            {ceilDiv(uint64_t{h.length} * h.width, 256), 1, 1});
}

// output[t][n] over `rows` rows: see OmniCodecConvParams.
void addConv(metal::CommandGraph &graph, const CodecConv &conv, const ConvCall &call, const char *what) {
  requireConv(conv, what);
  if (!call.rows || !call.input.rows) throw std::invalid_argument(std::string(what) + ": a convolution takes rows");
  if (call.input.width != conv.inputChannels)
    throw std::invalid_argument(std::string(what) + ": the input width is not the convolution's input channels");
  const uint32_t n = conv.outputChannels;
  const uint32_t outputStride = call.outputStride ? call.outputStride : n;
  if (outputStride < n) throw std::invalid_argument(std::string(what) + ": the output stride is narrower than the rows");
  requireBytes(call.input.buffer, call.input.bytes(), "codec conv input");
  requireBytes(call.output, rowBytes(call.rows, outputStride, n, 4), "codec conv output");
  requireHistory(call.history, conv.inputChannels, (conv.taps - 1) * conv.dilation, what);
  uint32_t flags = conv.bias ? OMNI_CODEC_CONV_FLAG_BIAS : 0u;
  if (call.residual) {
    requireBytes(call.residual, rowBytes(call.rows, outputStride, n, 4), "codec conv residual");
    flags |= OMNI_CODEC_CONV_FLAG_RESIDUAL;
  }
  if (call.scale) {
    requireBytes(call.scale, uint64_t{n} * 4, "codec conv scale");
    flags |= OMNI_CODEC_CONV_FLAG_SCALE;
  }
  if (call.gelu) flags |= OMNI_CODEC_CONV_FLAG_GELU;
  OmniCodecConvParams params{};
  params.rows = call.rows;
  params.input_rows = call.input.rows;
  params.in_channels = conv.inputChannels;
  params.out_channels = n;
  params.taps = conv.taps;
  params.dilation = conv.dilation;
  params.pad = call.pad;
  params.input_stride = call.input.width;
  params.output_stride = outputStride;
  params.flags = flags;
  params.history_rows = call.history.rows();
  const metal::MetalBuffer bias = conv.bias ? conv.bias : conv.weight;
  const metal::MetalBuffer residual = call.residual ? call.residual : call.output;
  const metal::MetalBuffer scale = call.scale ? call.scale : conv.weight;
  const metal::MetalBuffer history = call.history.history ? call.history.history->rows[call.history.phase] : call.input.buffer;
  const bool half = static_cast<bool>(conv.weightHalf);
  graph.add(half ? "omni_codec_conv_half" : "omni_codec_conv",
            {call.input.buffer, half ? conv.weightHalf : conv.weight, bias, residual, scale, call.output, history}, params,
            {ceilDiv(call.rows, OMNI_CODEC_CONV_TILE_ROWS), ceilDiv(n, OMNI_CODEC_CONV_TILE_COLUMNS), 1},
            {OMNI_CODEC_CONV_THREADS, 1, 1});
  addHistoryUpdate(graph, call.history, call.input);
}

// A causal convolution over the rows of `input` (left pad (taps - 1) *
// dilation, or the stream's history): the output has the input's rows.
void addCausalConv(metal::CommandGraph &graph, const CodecConv &conv, const CodecRows &input,
                   const CodecRows &output, metal::MetalBuffer residual, const HistoryUse &history,
                   const char *what) {
  if (output.rows != input.rows || output.width != conv.outputChannels)
    throw std::invalid_argument(std::string(what) + ": the output rows do not match the convolution");
  ConvCall call;
  call.input = input;
  call.rows = input.rows;
  call.pad = static_cast<int32_t>((conv.taps - 1) * conv.dilation);
  call.output = output.buffer;
  call.residual = std::move(residual);
  call.history = history;
  addConv(graph, conv, call, what);
}

// A causal transposed convolution of stride `stride`: input rows I become
// stride * I output rows of conv.outputChannels / stride channels.
void addTransposedConv(metal::CommandGraph &graph, const CodecConv &conv, uint32_t stride,
                       const CodecRows &input, const CodecRows &output, const HistoryUse &history,
                       const char *what) {
  if (conv.dilation != 1 || !stride || conv.outputChannels % stride)
    throw std::invalid_argument(std::string(what) + ": a transposed convolution holds taps of stride phases");
  if (output.rows != input.rows * stride || output.width * stride != conv.outputChannels)
    throw std::invalid_argument(std::string(what) + ": the output rows do not match the transposed convolution");
  ConvCall call;
  call.input = input;
  call.rows = input.rows;
  call.pad = static_cast<int32_t>(conv.taps - 1);
  call.output = output.buffer;
  call.history = history;
  addConv(graph, conv, call, what);
}

void addLinear(metal::CommandGraph &graph, const CodecConv &linear, const CodecRows &input,
               const CodecRows &output, metal::MetalBuffer residual, metal::MetalBuffer scale, bool gelu,
               const char *what) {
  if (linear.taps != 1) throw std::invalid_argument(std::string(what) + ": a linear is one tap");
  if (output.rows != input.rows || output.width != linear.outputChannels)
    throw std::invalid_argument(std::string(what) + ": the output rows do not match the linear");
  ConvCall call;
  call.input = input;
  call.rows = input.rows;
  call.output = output.buffer;
  call.residual = std::move(residual);
  call.scale = std::move(scale);
  call.gelu = gelu;
  addConv(graph, linear, call, what);
}

void addSnake(metal::CommandGraph &graph, const CodecSnake &snake, const CodecRows &input,
              const CodecRows &output) {
  requireSnake(snake, input.width);
  if (output.rows != input.rows || output.width != input.width)
    throw std::invalid_argument("codec snake output does not match its input");
  requireBytes(input.buffer, input.bytes(), "codec snake input");
  requireBytes(output.buffer, output.bytes(), "codec snake output");
  const OmniCodecElementwiseParams params{input.rows, input.width, 0, 0};
  graph.add("omni_codec_elementwise", {input.buffer, snake.alpha, snake.beta, output.buffer}, params,
            {ceilDiv(uint64_t{input.rows} * input.width, 256), 1, 1});
}

// The histories a stream keeps, in the order the decode uses them: (length,
// width) of every causal convolution with taps.
std::vector<std::pair<uint32_t, uint32_t>> historyShapes(const CodecDecoderWeights &w) {
  std::vector<std::pair<uint32_t, uint32_t>> shapes;
  auto conv = [&](const CodecConv &c, uint32_t width) {
    if (c.taps > 1) shapes.emplace_back((c.taps - 1) * c.dilation, width);
  };
  conv(w.preConv, w.preConv.inputChannels);
  for (const CodecConvNeXt &stage : w.upsample) {
    conv(stage.upsample, stage.upsample.inputChannels);
    if (stage.depthwiseTaps > 1) shapes.emplace_back(stage.depthwiseTaps - 1, stage.upsample.inputChannels);
  }
  conv(w.initialConv, w.initialConv.inputChannels);
  for (const CodecDecoderBlock &block : w.blocks) {
    conv(block.upsample, block.upsample.inputChannels);
    for (const CodecResidualUnit &unit : block.units) {
      conv(unit.conv1, unit.conv1.inputChannels);
      conv(unit.conv2, unit.conv2.inputChannels);
    }
  }
  conv(w.outputConv, w.outputConv.inputChannels);
  return shapes;
}

} // namespace

uint32_t CodecDims::samplesPerFrame() const noexcept {
  uint32_t samples = 1;
  for (uint32_t rate : upsampleRates) samples *= rate;
  for (uint32_t ratio : upsamplingRatios) samples *= ratio;
  return samples;
}

CodecConv makeCodecConv(metal::MetalBackend &backend, const std::vector<float> &weight, const std::vector<float> *bias,
                        uint32_t outputChannels, uint32_t inputChannels, uint32_t taps, uint32_t dilation) {
  requireCount(weight, uint64_t{outputChannels} * inputChannels * taps, "codec conv weight");
  std::vector<float> packed(weight.size());
  for (uint32_t co = 0; co < outputChannels; ++co)
    for (uint32_t ci = 0; ci < inputChannels; ++ci)
      for (uint32_t k = 0; k < taps; ++k)
        packed[(uint64_t{k} * inputChannels + ci) * outputChannels + co] =
            weight[(uint64_t{co} * inputChannels + ci) * taps + k];
  CodecConv conv;
  conv.weight = uploadFloats(backend, packed, "codec conv weight");
  if (outputChannels % 8 == 0) conv.weightHalf = uploadHalves(backend, packed, "codec conv weight fp16");
  if (bias) {
    requireCount(*bias, outputChannels, "codec conv bias");
    conv.bias = uploadFloats(backend, *bias, "codec conv bias");
  }
  conv.inputChannels = inputChannels;
  conv.outputChannels = outputChannels;
  conv.taps = taps;
  conv.dilation = dilation;
  requireConv(conv, "codec conv");
  return conv;
}

CodecConv makeCodecTransposedConv(metal::MetalBackend &backend, const std::vector<float> &weight,
                                  const std::vector<float> *bias, uint32_t inputChannels, uint32_t outputChannels,
                                  uint32_t stride, uint32_t kernel) {
  if (!stride || !kernel || kernel % stride)
    throw std::invalid_argument("a codec transposed convolution's kernel is a multiple of its stride");
  const uint32_t taps = kernel / stride;
  requireCount(weight, uint64_t{inputChannels} * outputChannels * kernel, "codec transposed conv weight");
  // out[s i + r][co] = sum_j x[i - j] W[ci][co][s j + r]; tap k reads row
  // i - (taps - 1) + k, so tap k holds j = taps - 1 - k.
  const uint32_t phases = stride * outputChannels;
  std::vector<float> packed(uint64_t{taps} * inputChannels * phases);
  for (uint32_t ci = 0; ci < inputChannels; ++ci)
    for (uint32_t co = 0; co < outputChannels; ++co)
      for (uint32_t kk = 0; kk < kernel; ++kk) {
        const uint32_t r = kk % stride, j = kk / stride, k = taps - 1 - j;
        packed[(uint64_t{k} * inputChannels + ci) * phases + r * outputChannels + co] =
            weight[(uint64_t{ci} * outputChannels + co) * kernel + kk];
      }
  CodecConv conv;
  conv.weight = uploadFloats(backend, packed, "codec transposed conv weight");
  if (phases % 8 == 0) conv.weightHalf = uploadHalves(backend, packed, "codec transposed conv weight fp16");
  if (bias) {
    requireCount(*bias, outputChannels, "codec transposed conv bias");
    std::vector<float> repeated(phases);
    for (uint32_t r = 0; r < stride; ++r) std::copy(bias->begin(), bias->end(), repeated.begin() + r * outputChannels);
    conv.bias = uploadFloats(backend, repeated, "codec transposed conv bias");
  }
  conv.inputChannels = inputChannels;
  conv.outputChannels = phases;
  conv.taps = taps;
  conv.dilation = 1;
  requireConv(conv, "codec transposed conv");
  return conv;
}

CodecConv makeCodecLinear(metal::MetalBackend &backend, const std::vector<float> &weight, const std::vector<float> *bias,
                          uint32_t outputChannels, uint32_t inputChannels) {
  return makeCodecConv(backend, weight, bias, outputChannels, inputChannels, 1, 1);
}

CodecSnake makeCodecSnake(metal::MetalBackend &backend, const std::vector<float> &alpha, const std::vector<float> &beta) {
  if (alpha.empty() || alpha.size() != beta.size()) throw std::invalid_argument("codec snake alpha and beta differ");
  std::vector<float> scaledAlpha(alpha.size()), inverseBeta(beta.size());
  for (size_t c = 0; c < alpha.size(); ++c) {
    scaledAlpha[c] = std::exp(alpha[c]);
    inverseBeta[c] = 1.0F / (std::exp(beta[c]) + 1e-9F);
  }
  CodecSnake snake;
  snake.alpha = uploadFloats(backend, scaledAlpha, "codec snake alpha");
  snake.beta = uploadFloats(backend, inverseBeta, "codec snake beta");
  snake.channels = static_cast<uint32_t>(alpha.size());
  return snake;
}

metal::MetalBuffer makeCodecVector(metal::MetalBackend &backend, const std::vector<float> &values) {
  return uploadFloats(backend, values, "codec vector");
}

// ----------------------------------------------------------------- decoder

namespace {

// The largest tensor of a decode of `frames` frames, in bytes: the slabs
// hold every intermediate of the convolutional path.
uint64_t largestTensorBytes(const CodecDims &dims, uint32_t frames) {
  uint64_t largest = uint64_t{frames} * std::max(dims.codebookDim, dims.latent);
  uint64_t rows = frames;
  for (uint32_t ratio : dims.upsamplingRatios) {
    rows *= ratio;
    largest = std::max(largest, rows * 4 * dims.latent);
  }
  uint32_t channels = dims.decoderDim;
  largest = std::max(largest, rows * channels);
  for (uint32_t rate : dims.upsampleRates) {
    rows *= rate;
    channels /= 2;
    largest = std::max(largest, rows * channels);
  }
  return largest * 4;
}

} // namespace

CodecStream::CodecStream(metal::MetalBackend &backend, const CodecDecoderWeights &weights, uint32_t maximumFrames)
    : maximumFrames_(maximumFrames) {
  const CodecDims &d = weights.dims;
  if (!maximumFrames) throw std::invalid_argument("a codec stream takes at least one frame per chunk");
  if (!d.window || d.window > OMNI_CODEC_MAXIMUM_WINDOW)
    throw std::invalid_argument("the codec decoder's attention window is 1 .. 256 keys");
  capacity_ = d.window - 1 + maximumFrames;
  if (weights.preConv.taps <= 1) throw std::invalid_argument("the codec pre-convolution keeps a history");
  frontHistories_ = 1;
  for (const CodecConvNeXt &stage : weights.upsample)
    frontHistories_ += (stage.upsample.taps > 1 ? 1u : 0u) + (stage.depthwiseTaps > 1 ? 1u : 0u);
  auto allocate = [&](uint64_t bytes, metal::BufferStorage storage, const char *label) {
    bytes_ += bytes;
    return backend.allocateBuffer(bytes, storage, label);
  };
  for (const auto &[length, width] : historyShapes(weights)) {
    CodecHistory history;
    history.length = length;
    history.width = width;
    for (metal::MetalBuffer &half : history.rows) {
      half = allocate(uint64_t{length} * width * 4, metal::BufferStorage::Shared, "codec stream history");
      std::memset(half.contents(), 0, uint64_t{length} * width * 4);
    }
    histories_.push_back(std::move(history));
  }
  const uint64_t ringBytes = uint64_t{capacity_} * d.heads * kHeadDim * 4;
  for (uint32_t l = 0; l < d.layers; ++l) {
    keys_.push_back(allocate(ringBytes, metal::BufferStorage::Private, "codec stream keys"));
    values_.push_back(allocate(ringBytes, metal::BufferStorage::Private, "codec stream values"));
  }
}

void CodecStream::reset() {
  for (CodecHistory &history : histories_)
    for (metal::MetalBuffer &half : history.rows) std::memset(half.contents(), 0, half.sizeBytes());
  position_ = 0;
  phase_ = 0;
}

// One decode's context: the graph, the stream (none for a whole sequence)
// and the histories it has handed out so far.
struct CodecDecoder::Encode final {
  metal::CommandGraph &graph;
  uint32_t frames = 0;
  CodecStream *stream = nullptr;
  size_t nextHistory = 0;
  uint32_t phase = 0;  // the history half this chunk reads
  size_t current = 0; // the slab of the rows between the front and the stack
  // The stream's history of the next causal convolution with taps.
  [[nodiscard]] HistoryUse history(const CodecConv &conv) {
    if (!stream || conv.taps <= 1) return {};
    if (nextHistory >= stream->histories_.size()) throw std::logic_error("codec stream has too few histories");
    return {&stream->histories_[nextHistory++], phase};
  }
  [[nodiscard]] uint32_t position() const noexcept { return stream ? stream->position_ : 0; }
};

uint64_t CodecDecoder::workspaceBytes(const CodecDims &dims, uint32_t maximumFrames) {
  const uint64_t attention = uint64_t{dims.heads} * kHeadDim;
  const uint64_t small =
      uint64_t{maximumFrames} * (2 * dims.hidden + 3 * attention + attention + 3 * dims.intermediate) * 4;
  const uint64_t rings = uint64_t{dims.layers} * 2 * maximumFrames * attention * 4;
  return 3 * largestTensorBytes(dims, maximumFrames) + small + rings;
}

CodecDecoder::CodecDecoder(metal::MetalBackend &backend, CodecDecoderWeights weights, uint32_t maximumFrames)
    : backend_(backend), weights_(std::move(weights)), maximumFrames_(maximumFrames) {
  const CodecDims &d = weights_.dims;
  if (!maximumFrames) throw std::invalid_argument("the codec decoder takes at least one frame");
  if (!d.window || d.window > OMNI_CODEC_MAXIMUM_WINDOW)
    throw std::invalid_argument("the codec decoder's attention window is 1 .. 256 keys");
  if (d.layers != weights_.layers.size() || d.upsamplingRatios.size() != weights_.upsample.size() ||
      d.upsampleRates.size() != weights_.blocks.size())
    throw std::invalid_argument("codec decoder weights do not match their dims");
  if (d.decoderDim % (1u << d.upsampleRates.size()))
    throw std::invalid_argument("the codec decoder dim does not halve through its blocks");
  slabBytes_ = largestTensorBytes(d, maximumFrames);
  for (size_t i = 0; i < slabs_.size(); ++i)
    slabs_[i] = backend_.allocateBuffer(slabBytes_, metal::BufferStorage::Private, "codec slab");
  const uint64_t attention = uint64_t{d.heads} * kHeadDim;
  auto scratch = [&](uint64_t width, const char *label) {
    return backend_.allocateBuffer(uint64_t{maximumFrames} * width * 4, metal::BufferStorage::Private, label);
  };
  residual_ = scratch(d.hidden, "codec residual");
  normed_ = scratch(d.hidden, "codec normed");
  qkv_ = scratch(3 * attention, "codec qkv");
  attention_ = scratch(attention, "codec attention");
  gateUp_ = scratch(2 * d.intermediate, "codec gate up");
  hidden_ = scratch(d.intermediate, "codec hidden");
  for (uint32_t l = 0; l < d.layers; ++l) {
    keys_.push_back(scratch(attention, "codec keys"));
    values_.push_back(scratch(attention, "codec values"));
  }
}

void CodecDecoder::addTransformer(Encode &e, const CodecRows &input, const CodecRows &output) {
  metal::CommandGraph &graph = e.graph;
  const CodecDims &d = weights_.dims;
  const uint32_t rows = input.rows;
  const uint32_t attention = d.heads * kHeadDim;
  const uint32_t capacity = e.stream ? e.stream->capacity_ : maximumFrames_;
  const uint32_t position = e.position();
  const CodecRows x{residual_, rows, d.hidden};
  const CodecRows n{normed_, rows, d.hidden};
  const CodecRows qkv{qkv_, rows, 3 * attention};
  const CodecRows att{attention_, rows, attention};
  const CodecRows gu{gateUp_, rows, 2 * d.intermediate};
  const CodecRows h{hidden_, rows, d.intermediate};
  // A chunk's rows join the window - 1 positions before it in the ring; a
  // whole sequence starts at position 0.
  if (capacity < (e.stream ? d.window - 1 + rows : rows))
    throw std::invalid_argument("codec key ring is shorter than the window");
  addLinear(graph, weights_.inputProj, input, x, {}, {}, false, "codec input projection");
  for (uint32_t l = 0; l < d.layers; ++l) {
    const CodecTransformerLayer &layer = weights_.layers[l];
    const metal::MetalBuffer &keys = e.stream ? e.stream->keys_[l] : keys_[l];
    const metal::MetalBuffer &values = e.stream ? e.stream->values_[l] : values_[l];
    requireBytes(keys, uint64_t{capacity} * attention * 4, "codec keys");
    requireBytes(values, uint64_t{capacity} * attention * 4, "codec values");
    Norms::addRms(graph, x.buffer, layer.inputNorm, n.buffer, rows, d.hidden, d.rmsEpsilon, true);
    addLinear(graph, layer.qkv, n, qkv, {}, {}, false, "codec qkv projection");
    const OmniCodecRopeParams rope{rows, d.heads, d.ropeTheta, position};
    graph.add("omni_codec_rope", {qkv.buffer}, rope, {ceilDiv(uint64_t{rows} * d.heads * kHeadDim, 256), 1, 1});
    const OmniCodecAppendParams append{rows, d.heads, position, capacity};
    graph.add("omni_codec_kv_append", {qkv.buffer, keys, values}, append,
              {ceilDiv(uint64_t{rows} * attention, 256), 1, 1});
    OmniCodecAttentionParams attentionParams{};
    attentionParams.rows = rows;
    attentionParams.heads = d.heads;
    attentionParams.window = d.window;
    attentionParams.scale = 1.0F / std::sqrt(float(kHeadDim));
    attentionParams.position = position;
    attentionParams.capacity = capacity;
    requireBytes(att.buffer, att.bytes(), "codec attention output");
    graph.add("omni_codec_attention", {qkv.buffer, keys, values, att.buffer}, attentionParams,
              {ceilDiv(uint64_t{rows} * d.heads, OMNI_CODEC_ATTENTION_GROUPS), 1, 1},
              {OMNI_CODEC_ATTENTION_GROUPS * 32, 1, 1});
    addLinear(graph, layer.o, att, x, x.buffer, layer.attentionScale, false, "codec attention projection");
    Norms::addRms(graph, x.buffer, layer.postNorm, n.buffer, rows, d.hidden, d.rmsEpsilon, true);
    addLinear(graph, layer.gateUp, n, gu, {}, {}, false, "codec gate up projection");
    requireBytes(h.buffer, h.bytes(), "codec hidden");
    const OmniCodecElementwiseParams gate{rows, d.intermediate, 1, 0};
    graph.add("omni_codec_elementwise", {gu.buffer, gu.buffer, gu.buffer, h.buffer}, gate,
              {ceilDiv(uint64_t{rows} * d.intermediate, 256), 1, 1});
    addLinear(graph, layer.down, h, x, x.buffer, layer.mlpScale, false, "codec down projection");
  }
  Norms::addRms(graph, x.buffer, weights_.finalNorm, n.buffer, rows, d.hidden, d.rmsEpsilon, true);
  addLinear(graph, weights_.outputProj, n, output, {}, {}, false, "codec output projection");
}

void CodecDecoder::addConvNeXt(Encode &e, const CodecConvNeXt &stage, const CodecRows &input,
                               const CodecRows &output, const CodecRows &scratchA, const CodecRows &scratchB) {
  metal::CommandGraph &graph = e.graph;
  const uint32_t channels = input.width;
  const uint32_t rows = input.rows * stage.ratio;
  // output = convT(input); then output += gamma * pw2(gelu(pw1(ln(dw(output))))).
  const CodecRows y{output.buffer, rows, channels};
  const CodecRows dw{scratchA.buffer, rows, channels};
  const CodecRows ln{scratchB.buffer, rows, channels};
  const CodecRows wide{scratchA.buffer, rows, 4 * channels};
  if (output.rows != rows || output.width != channels) throw std::invalid_argument("codec upsample output rows mismatch");
  addTransposedConv(graph, stage.upsample, stage.ratio, input, y, e.history(stage.upsample), "codec upsample");
  requireBytes(stage.depthwiseWeight, uint64_t{stage.depthwiseTaps} * channels * 4, "codec depthwise weight");
  requireBytes(stage.depthwiseBias, uint64_t{channels} * 4, "codec depthwise bias");
  requireBytes(dw.buffer, dw.bytes(), "codec depthwise output");
  HistoryUse history;
  if (e.stream && stage.depthwiseTaps > 1) {
    if (e.nextHistory >= e.stream->histories_.size()) throw std::logic_error("codec stream has too few histories");
    history = {&e.stream->histories_[e.nextHistory++], e.phase};
  }
  requireHistory(history, channels, stage.depthwiseTaps - 1, "codec depthwise");
  const OmniCodecDepthwiseParams depthwise{rows, channels, stage.depthwiseTaps, history.rows()};
  const metal::MetalBuffer historyRows = history.history ? history.history->rows[history.phase] : y.buffer;
  graph.add("omni_codec_depthwise", {y.buffer, stage.depthwiseWeight, stage.depthwiseBias, dw.buffer, historyRows},
            depthwise, {ceilDiv(uint64_t{rows} * channels, 256), 1, 1});
  addHistoryUpdate(graph, history, y);
  Norms::addLayer(graph, dw.buffer, stage.normWeight, stage.normBias, ln.buffer, rows, channels, 1e-6F, true);
  addLinear(graph, stage.pw1, ln, wide, {}, {}, true, "codec pointwise 1");
  addLinear(graph, stage.pw2, wide, y, y.buffer, stage.gamma, false, "codec pointwise 2");
}

void CodecDecoder::addBlock(Encode &e, const CodecDecoderBlock &block, const CodecRows &input,
                            const CodecRows &output, const CodecRows &scratch) {
  metal::CommandGraph &graph = e.graph;
  const uint32_t channels = input.width / 2;
  const uint32_t rows = input.rows * block.rate;
  // snake(input) in scratch, then y = convT; per unit a = snake(y) over the
  // dead input, b = conv1(a) in scratch, a = snake(b), y += conv2(a).
  const CodecRows activatedInput{scratch.buffer, input.rows, input.width};
  const CodecRows y{output.buffer, rows, channels};
  const CodecRows a{input.buffer, rows, channels};
  const CodecRows b{scratch.buffer, rows, channels};
  if (output.rows != rows || output.width != channels) throw std::invalid_argument("codec block output rows mismatch");
  addSnake(graph, block.snake, input, activatedInput);
  addTransposedConv(graph, block.upsample, block.rate, activatedInput, y, e.history(block.upsample),
                    "codec block upsample");
  for (const CodecResidualUnit &unit : block.units) {
    addSnake(graph, unit.act1, y, a);
    addCausalConv(graph, unit.conv1, a, b, {}, e.history(unit.conv1), "codec residual conv 1");
    addSnake(graph, unit.act2, b, a);
    addCausalConv(graph, unit.conv2, a, y, y.buffer, e.history(unit.conv2), "codec residual conv 2");
  }
}

void CodecDecoder::encodeFront(Encode &e, metal::MetalBuffer codes) {
  metal::CommandGraph &graph = e.graph;
  const CodecDims &d = weights_.dims;
  const uint32_t frames = e.frames;
  if (!frames || frames > maximumFrames_)
    throw std::invalid_argument("codec decode takes 1 .. " + std::to_string(maximumFrames_) + " frames");
  requireBytes(codes, uint64_t{frames} * d.quantizers * 4, "codec codes");
  const uint32_t half = d.codebookDim / 2;
  requireBytes(weights_.codebooks, uint64_t{d.quantizers} * d.codebookSize * half * 4, "codec codebook");

  auto slab = [&](size_t index, uint32_t rows, uint32_t width) {
    const CodecRows r{slabs_[index], rows, width};
    requireBytes(r.buffer, r.bytes(), "codec slab");
    return r;
  };
  // RVQ lookup and projection, pre-conv, transformer.
  const CodecRows looked = slab(0, frames, d.codebookDim);
  const OmniCodecLookupParams lookup{frames, d.quantizers, d.codebookSize, half};
  graph.add("omni_codec_lookup", {codes, weights_.codebooks, looked.buffer}, lookup,
            {ceilDiv(uint64_t{frames} * d.codebookDim, 256), 1, 1});
  const CodecRows quantized = slab(1, frames, d.codebookDim);
  addLinear(graph, weights_.rvqProjection, looked, quantized, {}, {}, false, "codec RVQ projection");
  const CodecRows pre = slab(2, frames, d.latent);
  addCausalConv(graph, weights_.preConv, quantized, pre, {}, e.history(weights_.preConv), "codec pre conv");
  CodecRows latent = slab(0, frames, d.latent);
  addTransformer(e, pre, latent);

  // ConvNeXt upsample stages: input slab i -> output slab (i + 1) % 3.
  size_t current = 0;
  for (const CodecConvNeXt &stage : weights_.upsample) {
    const size_t next = (current + 1) % 3, other = (current + 2) % 3;
    const CodecRows out = slab(next, latent.rows * stage.ratio, d.latent);
    addConvNeXt(e, stage, latent, out, slab(other, out.rows, 4 * d.latent), slab(current, out.rows, d.latent));
    latent = out;
    current = next;
  }
  e.current = current;
}

void CodecDecoder::encodeStack(Encode &e, metal::MetalBuffer output) {
  metal::CommandGraph &graph = e.graph;
  const CodecDims &d = weights_.dims;
  const uint64_t samples = uint64_t{e.frames} * d.samplesPerFrame();
  requireBytes(output, samples * 4, "codec output");
  auto slab = [&](size_t index, uint32_t rows, uint32_t width) {
    const CodecRows r{slabs_[index], rows, width};
    requireBytes(r.buffer, r.bytes(), "codec slab");
    return r;
  };
  const CodecRows latent = stackInput(e.frames);
  size_t current = e.current;

  // Decoder stack.
  size_t next = (current + 1) % 3;
  CodecRows x = slab(next, latent.rows, d.decoderDim);
  addCausalConv(graph, weights_.initialConv, latent, x, {}, e.history(weights_.initialConv), "codec initial conv");
  current = next;
  for (const CodecDecoderBlock &block : weights_.blocks) {
    next = (current + 1) % 3;
    const size_t other = (current + 2) % 3;
    const CodecRows out = slab(next, x.rows * block.rate, x.width / 2);
    addBlock(e, block, x, out, slab(other, out.rows, out.width));
    x = out;
    current = next;
  }
  const CodecRows activated = slab((current + 1) % 3, x.rows, x.width);
  addSnake(graph, weights_.outputSnake, x, activated);
  const CodecConv &outputConv = weights_.outputConv;
  if (outputConv.inputChannels != x.width || outputConv.outputChannels != 1)
    throw std::invalid_argument("codec output conv does not match the decoder's channels");
  requireBytes(outputConv.weight, uint64_t{outputConv.taps} * x.width * 4, "codec output weight");
  requireBytes(outputConv.bias, 4, "codec output bias");
  if (x.rows != samples) throw std::logic_error("codec decoder rows do not reach the samples per frame");
  const HistoryUse history = e.history(outputConv);
  requireHistory(history, x.width, outputConv.taps - 1, "codec output conv");
  const OmniCodecOutputParams out{x.rows, x.width, outputConv.taps, history.rows()};
  const metal::MetalBuffer historyRows = history.history ? history.history->rows[history.phase] : activated.buffer;
  graph.add("omni_codec_output", {activated.buffer, outputConv.weight, outputConv.bias, output, historyRows}, out,
            {ceilDiv(samples, 256), 1, 1});
  addHistoryUpdate(graph, history, activated);
  if (e.stream && e.nextHistory != e.stream->histories_.size())
    throw std::logic_error("codec stream has unused histories");
}

CodecRows CodecDecoder::stackInput(uint32_t frames) const {
  if (!frames || frames > maximumFrames_)
    throw std::invalid_argument("codec decode takes 1 .. " + std::to_string(maximumFrames_) + " frames");
  uint32_t rows = frames;
  for (uint32_t ratio : weights_.dims.upsamplingRatios) rows *= ratio;
  // The front's last ConvNeXt stage writes slab (stages % 3).
  return {slabs_[weights_.upsample.size() % 3], rows, weights_.dims.latent};
}

void CodecDecoder::encode(Encode &e, metal::MetalBuffer codes, metal::MetalBuffer output) {
  if (!e.frames || e.frames > maximumFrames_)
    throw std::invalid_argument("codec decode takes 1 .. " + std::to_string(maximumFrames_) + " frames");
  requireBytes(codes, uint64_t{e.frames} * weights_.dims.quantizers * 4, "codec codes");
  requireBytes(output, uint64_t{e.frames} * weights_.dims.samplesPerFrame() * 4, "codec output");
  encodeFront(e, std::move(codes));
  encodeStack(e, std::move(output));
  if (e.stream) {
    e.stream->position_ += e.frames;
    e.stream->phase_ ^= 1;
  }
}

void CodecDecoder::addDecode(metal::CommandGraph &graph, metal::MetalBuffer codes, uint32_t frames,
                             metal::MetalBuffer output) {
  Encode e{graph, frames, nullptr, 0, 0, 0};
  encode(e, std::move(codes), std::move(output));
}

void CodecDecoder::addDecodeStream(metal::CommandGraph &graph, CodecStream &stream, metal::MetalBuffer codes,
                                   uint32_t frames, metal::MetalBuffer output) {
  if (!frames || frames > stream.maximumFrames_)
    throw std::invalid_argument("codec stream chunks take 1 .. " + std::to_string(stream.maximumFrames_) + " frames");
  Encode e{graph, frames, &stream, 0, stream.phase_, 0};
  encode(e, std::move(codes), std::move(output));
}

void CodecDecoder::addFrontStream(metal::CommandGraph &graph, CodecStream &stream, metal::MetalBuffer codes,
                                  uint32_t frames) {
  if (!frames || frames > stream.maximumFrames_)
    throw std::invalid_argument("codec stream chunks take 1 .. " + std::to_string(stream.maximumFrames_) + " frames");
  Encode e{graph, frames, &stream, 0, stream.phase_, 0};
  encodeFront(e, std::move(codes));
  if (e.nextHistory != stream.frontHistories_) throw std::logic_error("codec stream front histories mismatch");
  stream.position_ += frames;
  stream.phase_ ^= 1;
}

void CodecDecoder::addStackStream(metal::CommandGraph &graph, CodecStream &stream, uint32_t frames, uint32_t phase,
                                  metal::MetalBuffer output) {
  if (!frames || frames > stream.maximumFrames_)
    throw std::invalid_argument("codec stream chunks take 1 .. " + std::to_string(stream.maximumFrames_) + " frames");
  Encode e{graph, frames, &stream, stream.frontHistories_, phase & 1u, weights_.upsample.size() % 3};
  encodeStack(e, std::move(output));
}

} // namespace splash::ops::omni
