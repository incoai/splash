// The audio tower's weights from an MLX checkpoint (ops/OmniAudioEncoder.hpp
// declares loadAudioEncoderWeights): bf16 tensors as stored, biases as fp32.

#include "ops/OmniAudioEncoder.hpp"

#include "model/SafetensorsCheckpoint.hpp"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops::omni {
namespace {

constexpr uint32_t kChannels = OMNI_AUDIO_CONV_CHANNELS;
constexpr uint32_t kPaddedChannels = OMNI_AUDIO_CONV_PADDED_CHANNELS;
constexpr uint32_t kTaps = OMNI_AUDIO_CONV_TAPS;
constexpr uint32_t kPaddedTaps = OMNI_AUDIO_CONV_PADDED_TAPS;
constexpr uint32_t kConvOutInput = OMNI_AUDIO_CONV_OUT_INPUT;

} // namespace

// ---------------------------------------------------------------- loading

namespace {

using splash::model::SafetensorsCheckpoint;
using splash::model::SourceTensor;

float bf16ToFloat(uint16_t bits) noexcept { return std::bit_cast<float>(uint32_t{bits} << 16); }

const SourceTensor &requireBf16(const SafetensorsCheckpoint &checkpoint, const std::string &name,
                                std::initializer_list<uint64_t> shape) {
  const SourceTensor &tensor = checkpoint.require(name);
  if (tensor.dtype != "BF16") throw std::runtime_error(name + ": expected a BF16 tensor, found " + tensor.dtype);
  if (!std::equal(tensor.shape.begin(), tensor.shape.end(), shape.begin(), shape.end()))
    throw std::runtime_error(name + ": unexpected shape");
  return tensor;
}

std::vector<uint16_t> readBf16(const SourceTensor &tensor) {
  std::vector<uint16_t> values(tensor.bytes / 2);
  tensor.read(0, {reinterpret_cast<uint8_t *>(values.data()), tensor.bytes});
  return values;
}

metal::MetalBuffer upload(metal::MetalBackend &backend, const void *data, uint64_t bytes, const std::string &label) {
  metal::MetalBuffer buffer = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), data, bytes);
  return buffer;
}

// A bf16 tensor as stored.
metal::MetalBuffer bf16Buffer(metal::MetalBackend &backend, const SafetensorsCheckpoint &checkpoint,
                              const std::string &name, std::initializer_list<uint64_t> shape) {
  const std::vector<uint16_t> values = readBf16(requireBf16(checkpoint, name, shape));
  return upload(backend, values.data(), values.size() * 2, name);
}

// A bf16 vector as fp32, zero-extended to `padded` values.
metal::MetalBuffer f32Buffer(metal::MetalBackend &backend, const SafetensorsCheckpoint &checkpoint,
                             const std::string &name, uint64_t count, uint64_t padded) {
  const std::vector<uint16_t> values = readBf16(requireBf16(checkpoint, name, {count}));
  std::vector<float> floats(padded, 0.0F);
  for (uint64_t i = 0; i < count; ++i) floats[i] = bf16ToFloat(values[i]);
  return upload(backend, floats.data(), floats.size() * 4, name);
}

AudioLayerNorm loadLayerNorm(metal::MetalBackend &backend, const SafetensorsCheckpoint &checkpoint,
                             const std::string &prefix, uint32_t width) {
  return {bf16Buffer(backend, checkpoint, prefix + ".weight", {width}),
          bf16Buffer(backend, checkpoint, prefix + ".bias", {width})};
}

DenseWeights loadLinear(metal::MetalBackend &backend, const SafetensorsCheckpoint &checkpoint,
                        const std::string &prefix, uint32_t outputSize, uint32_t inputSize, bool bias) {
  DenseWeights weights;
  weights.weight = bf16Buffer(backend, checkpoint, prefix + ".weight", {outputSize, inputSize});
  if (bias) weights.bias = f32Buffer(backend, checkpoint, prefix + ".bias", outputSize, outputSize);
  weights.outputSize = outputSize;
  weights.inputSize = inputSize;
  return weights;
}

// conv2d2 / conv2d3: [480][3][3][480] to the padded GEMM weight [512][4352].
DenseWeights loadConvGemm(metal::MetalBackend &backend, const SafetensorsCheckpoint &checkpoint,
                          const std::string &prefix) {
  const std::vector<uint16_t> source =
      readBf16(requireBf16(checkpoint, prefix + ".weight", {kChannels, 3, 3, kChannels}));
  std::vector<uint16_t> padded(size_t(kPaddedChannels) * kPaddedTaps, 0);
  for (uint32_t co = 0; co < kChannels; ++co)
    std::copy_n(source.begin() + size_t(co) * kTaps, kTaps, padded.begin() + size_t(co) * kPaddedTaps);
  DenseWeights weights;
  weights.weight = upload(backend, padded.data(), padded.size() * 2, prefix + ".weight");
  weights.bias = f32Buffer(backend, checkpoint, prefix + ".bias", kChannels, kPaddedChannels);
  weights.outputSize = kPaddedChannels;
  weights.inputSize = kPaddedTaps;
  return weights;
}

// q_proj, k_proj, v_proj stacked into one [3 hidden][hidden] projection.
DenseWeights loadQkv(metal::MetalBackend &backend, const SafetensorsCheckpoint &checkpoint,
                     const std::string &prefix, uint32_t hidden) {
  std::vector<uint16_t> weight;
  std::vector<float> bias;
  weight.reserve(size_t(3) * hidden * hidden);
  bias.reserve(size_t(3) * hidden);
  for (const char *name : {"q_proj", "k_proj", "v_proj"}) {
    const std::string projection = prefix + "." + name;
    const std::vector<uint16_t> w = readBf16(requireBf16(checkpoint, projection + ".weight", {hidden, hidden}));
    weight.insert(weight.end(), w.begin(), w.end());
    const std::vector<uint16_t> b = readBf16(requireBf16(checkpoint, projection + ".bias", {hidden}));
    for (uint16_t bits : b) bias.push_back(bf16ToFloat(bits));
  }
  DenseWeights weights;
  weights.weight = upload(backend, weight.data(), weight.size() * 2, prefix + ".qkv.weight");
  weights.bias = upload(backend, bias.data(), bias.size() * 4, prefix + ".qkv.bias");
  weights.outputSize = 3 * hidden;
  weights.inputSize = hidden;
  return weights;
}

} // namespace

AudioEncoderWeights loadAudioEncoderWeights(metal::MetalBackend &backend, const SafetensorsCheckpoint &checkpoint,
                                            const AudioEncoderDims &dims) {
  const std::string tower = "audio_tower.";
  const uint32_t hidden = dims.hidden;
  AudioEncoderWeights weights;
  weights.dims = dims;
  weights.conv1Weight = bf16Buffer(backend, checkpoint, tower + "conv2d1.weight", {kChannels, 3, 3, 1});
  weights.conv1Bias = f32Buffer(backend, checkpoint, tower + "conv2d1.bias", kChannels, kChannels);
  weights.conv2 = loadConvGemm(backend, checkpoint, tower + "conv2d2");
  weights.conv3 = loadConvGemm(backend, checkpoint, tower + "conv2d3");
  weights.convOut = loadLinear(backend, checkpoint, tower + "conv_out", hidden, kConvOutInput, false);
  for (uint32_t index = 0;; ++index) {
    const std::string prefix = tower + "layers." + std::to_string(index);
    if (!checkpoint.find(prefix + ".fc1.weight")) break;
    AudioEncoderLayerWeights layer;
    layer.attentionNorm = loadLayerNorm(backend, checkpoint, prefix + ".self_attn_layer_norm", hidden);
    layer.qkv = loadQkv(backend, checkpoint, prefix + ".self_attn", hidden);
    layer.out = loadLinear(backend, checkpoint, prefix + ".self_attn.out_proj", hidden, hidden, true);
    layer.ffnNorm = loadLayerNorm(backend, checkpoint, prefix + ".final_layer_norm", hidden);
    layer.fc1 = loadLinear(backend, checkpoint, prefix + ".fc1", dims.ffn, hidden, true);
    layer.fc2 = loadLinear(backend, checkpoint, prefix + ".fc2", hidden, dims.ffn, true);
    weights.layers.push_back(std::move(layer));
  }
  if (weights.layers.size() != dims.layers)
    throw std::runtime_error("the checkpoint has " + std::to_string(weights.layers.size()) + " audio_tower layers, expected " +
                             std::to_string(dims.layers));
  weights.postNorm = loadLayerNorm(backend, checkpoint, tower + "ln_post", hidden);
  weights.proj1 = loadLinear(backend, checkpoint, tower + "proj1", hidden, hidden, true);
  weights.proj2 = loadLinear(backend, checkpoint, tower + "proj2", dims.output, hidden, true);
  return weights;
}

} // namespace splash::ops::omni
