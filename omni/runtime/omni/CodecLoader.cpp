// The speech tokenizer decoder's weights from its safetensors file
// (ops/OmniCodec.hpp declares loadCodecDecoderWeights): every tensor
// converted to fp32 and repacked into the kernels' layouts.

#include "ops/OmniCodec.hpp"

#include "model/SafetensorsCheckpoint.hpp"
#include "ops/BufferExtent.hpp"
#include "ops/Omni.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops::omni {
namespace {

constexpr uint32_t kHeadDim = OMNI_CODEC_HEAD_DIM;

metal::MetalBuffer uploadFloats(metal::MetalBackend &backend, const std::vector<float> &values, const char *label) {
  if (values.empty()) throw std::invalid_argument(std::string(label) + ": an empty tensor");
  metal::MetalBuffer buffer = backend.allocateBuffer(values.size() * sizeof(float), metal::BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(float));
  return buffer;
}

} // namespace

// ------------------------------------------------------------------ loader

namespace {

float halfToFloat(uint16_t bits) {
  const uint32_t sign = (bits & 0x8000u) << 16;
  uint32_t exponent = (bits >> 10) & 0x1Fu;
  uint32_t mantissa = bits & 0x3FFu;
  uint32_t result;
  if (exponent == 0) {
    if (mantissa == 0) {
      result = sign;
    } else {
      exponent = 127 - 15 + 1;
      while (!(mantissa & 0x400u)) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x3FFu;
      result = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    result = sign | 0x7F800000u | (mantissa << 13);
  } else {
    result = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  return std::bit_cast<float>(result);
}

class DecoderReader final {
public:
  explicit DecoderReader(const std::filesystem::path &directory) : checkpoint_(directory) {}

  std::vector<float> floats(const std::string &name, std::vector<uint64_t> shape) {
    const model::SourceTensor &tensor = checkpoint_.require(name);
    if (tensor.shape != shape) {
      std::string got, want;
      for (uint64_t d : tensor.shape) got += std::to_string(d) + " ";
      for (uint64_t d : shape) want += std::to_string(d) + " ";
      throw std::runtime_error("speech tokenizer tensor " + name + " has shape [" + got + "], expected [" + want + "]");
    }
    uint64_t count = 1;
    for (uint64_t d : shape) count *= d;
    std::vector<float> values(count);
    if (tensor.dtype == "F32") {
      tensor.read(0, {reinterpret_cast<uint8_t *>(values.data()), count * 4});
    } else if (tensor.dtype == "BF16" || tensor.dtype == "F16") {
      std::vector<uint16_t> bits(count);
      tensor.read(0, {reinterpret_cast<uint8_t *>(bits.data()), count * 2});
      for (uint64_t i = 0; i < count; ++i)
        values[i] = tensor.dtype == "BF16" ? std::bit_cast<float>(uint32_t{bits[i]} << 16) : halfToFloat(bits[i]);
    } else {
      throw std::runtime_error("speech tokenizer tensor " + name + " has dtype " + tensor.dtype + ", expected F32");
    }
    return values;
  }

  CodecConv conv(metal::MetalBackend &backend, const std::string &prefix, uint32_t out, uint32_t in, uint32_t taps,
                 uint32_t dilation = 1) {
    const std::vector<float> weight = floats(prefix + ".weight", {out, in, taps});
    const std::vector<float> bias = floats(prefix + ".bias", {out});
    return makeCodecConv(backend, weight, &bias, out, in, taps, dilation);
  }
  CodecConv transposed(metal::MetalBackend &backend, const std::string &prefix, uint32_t in, uint32_t out,
                       uint32_t stride, uint32_t kernel) {
    const std::vector<float> weight = floats(prefix + ".weight", {in, out, kernel});
    const std::vector<float> bias = floats(prefix + ".bias", {out});
    return makeCodecTransposedConv(backend, weight, &bias, in, out, stride, kernel);
  }
  CodecConv linear(metal::MetalBackend &backend, const std::string &prefix, uint32_t out, uint32_t in, bool bias) {
    const std::vector<float> weight = floats(prefix + ".weight", {out, in});
    if (!bias) return makeCodecLinear(backend, weight, nullptr, out, in);
    const std::vector<float> b = floats(prefix + ".bias", {out});
    return makeCodecLinear(backend, weight, &b, out, in);
  }
  CodecSnake snake(metal::MetalBackend &backend, const std::string &prefix, uint32_t channels) {
    return makeCodecSnake(backend, floats(prefix + ".alpha", {channels}), floats(prefix + ".beta", {channels}));
  }
  metal::MetalBuffer vector(metal::MetalBackend &backend, const std::string &name, uint32_t count) {
    return uploadFloats(backend, floats(name, {count}), "codec vector");
  }

private:
  model::SafetensorsCheckpoint checkpoint_;
};

} // namespace

CodecDecoderWeights loadCodecDecoderWeights(metal::MetalBackend &backend,
                                            const std::filesystem::path &speechTokenizerSafetensors) {
  const std::filesystem::path directory = std::filesystem::is_directory(speechTokenizerSafetensors)
                                              ? speechTokenizerSafetensors
                                              : speechTokenizerSafetensors.parent_path();
  DecoderReader reader(directory);
  CodecDecoderWeights w;
  const CodecDims &d = w.dims;
  const uint32_t half = d.codebookDim / 2;

  // RVQ: codebook q = embedding_sum / max(cluster_usage, 1e-5); the first
  // quantizer's and the rest's output projections side by side.
  std::vector<float> codebooks(uint64_t{d.quantizers} * d.codebookSize * half);
  for (uint32_t q = 0; q < d.quantizers; ++q) {
    const std::string prefix = q == 0 ? "decoder.quantizer.rvq_first.vq.layers.0._codebook"
                                      : "decoder.quantizer.rvq_rest.vq.layers." + std::to_string(q - 1) + "._codebook";
    const std::vector<float> sum = reader.floats(prefix + ".embedding_sum", {d.codebookSize, half});
    const std::vector<float> usage = reader.floats(prefix + ".cluster_usage", {d.codebookSize});
    for (uint32_t e = 0; e < d.codebookSize; ++e) {
      const float divisor = std::max(usage[e], 1e-5F);
      for (uint32_t i = 0; i < half; ++i)
        codebooks[(uint64_t{q} * d.codebookSize + e) * half + i] = sum[uint64_t{e} * half + i] / divisor;
    }
  }
  w.codebooks = uploadFloats(backend, codebooks, "codec codebooks");
  {
    const std::vector<float> first =
        reader.floats("decoder.quantizer.rvq_first.output_proj.weight", {d.codebookDim, half, 1});
    const std::vector<float> rest =
        reader.floats("decoder.quantizer.rvq_rest.output_proj.weight", {d.codebookDim, half, 1});
    std::vector<float> joined(uint64_t{d.codebookDim} * d.codebookDim); // [out][in], in = first | rest
    for (uint32_t co = 0; co < d.codebookDim; ++co)
      for (uint32_t ci = 0; ci < half; ++ci) {
        joined[uint64_t{co} * d.codebookDim + ci] = first[uint64_t{co} * half + ci];
        joined[uint64_t{co} * d.codebookDim + half + ci] = rest[uint64_t{co} * half + ci];
      }
    w.rvqProjection = makeCodecLinear(backend, joined, nullptr, d.codebookDim, d.codebookDim);
  }

  w.preConv = reader.conv(backend, "decoder.pre_conv.conv", d.latent, d.codebookDim, 3);

  // Transformer.
  const uint32_t attention = d.heads * kHeadDim;
  w.inputProj = reader.linear(backend, "decoder.pre_transformer.input_proj", d.hidden, d.latent, true);
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string prefix = "decoder.pre_transformer.layers." + std::to_string(l) + ".";
    CodecTransformerLayer layer;
    layer.inputNorm = reader.vector(backend, prefix + "input_layernorm.weight", d.hidden);
    layer.postNorm = reader.vector(backend, prefix + "post_attention_layernorm.weight", d.hidden);
    layer.attentionScale = reader.vector(backend, prefix + "self_attn_layer_scale.scale", d.hidden);
    layer.mlpScale = reader.vector(backend, prefix + "mlp_layer_scale.scale", d.hidden);
    std::vector<float> qkv(uint64_t{3} * attention * d.hidden);
    const char *names[3] = {"q_proj", "k_proj", "v_proj"};
    for (uint32_t i = 0; i < 3; ++i) {
      const std::vector<float> part = reader.floats(prefix + "self_attn." + names[i] + ".weight", {attention, d.hidden});
      std::copy(part.begin(), part.end(), qkv.begin() + uint64_t{i} * attention * d.hidden);
    }
    layer.qkv = makeCodecLinear(backend, qkv, nullptr, 3 * attention, d.hidden);
    layer.o = reader.linear(backend, prefix + "self_attn.o_proj", d.hidden, attention, false);
    std::vector<float> gateUp(uint64_t{2} * d.intermediate * d.hidden);
    const std::vector<float> gate = reader.floats(prefix + "mlp.gate_proj.weight", {d.intermediate, d.hidden});
    const std::vector<float> up = reader.floats(prefix + "mlp.up_proj.weight", {d.intermediate, d.hidden});
    std::copy(gate.begin(), gate.end(), gateUp.begin());
    std::copy(up.begin(), up.end(), gateUp.begin() + gate.size());
    layer.gateUp = makeCodecLinear(backend, gateUp, nullptr, 2 * d.intermediate, d.hidden);
    layer.down = reader.linear(backend, prefix + "mlp.down_proj", d.hidden, d.intermediate, false);
    w.layers.push_back(std::move(layer));
  }
  w.finalNorm = reader.vector(backend, "decoder.pre_transformer.norm.weight", d.hidden);
  w.outputProj = reader.linear(backend, "decoder.pre_transformer.output_proj", d.latent, d.hidden, true);

  // Upsample stages.
  for (uint32_t s = 0; s < d.upsamplingRatios.size(); ++s) {
    const std::string prefix = "decoder.upsample." + std::to_string(s) + ".";
    CodecConvNeXt stage;
    stage.ratio = d.upsamplingRatios[s];
    stage.upsample = reader.transposed(backend, prefix + "0.conv", d.latent, d.latent, stage.ratio, stage.ratio);
    const std::vector<float> dw = reader.floats(prefix + "1.dwconv.conv.weight", {d.latent, 1, 7});
    std::vector<float> dwPacked(uint64_t{7} * d.latent);
    for (uint32_t c = 0; c < d.latent; ++c)
      for (uint32_t k = 0; k < 7; ++k) dwPacked[uint64_t{k} * d.latent + c] = dw[uint64_t{c} * 7 + k];
    stage.depthwiseWeight = uploadFloats(backend, dwPacked, "codec depthwise weight");
    stage.depthwiseBias = reader.vector(backend, prefix + "1.dwconv.conv.bias", d.latent);
    stage.depthwiseTaps = 7;
    stage.normWeight = reader.vector(backend, prefix + "1.norm.weight", d.latent);
    stage.normBias = reader.vector(backend, prefix + "1.norm.bias", d.latent);
    stage.pw1 = reader.linear(backend, prefix + "1.pwconv1", 4 * d.latent, d.latent, true);
    stage.pw2 = reader.linear(backend, prefix + "1.pwconv2", d.latent, 4 * d.latent, true);
    stage.gamma = reader.vector(backend, prefix + "1.gamma", d.latent);
    w.upsample.push_back(std::move(stage));
  }

  // Decoder stack.
  w.initialConv = reader.conv(backend, "decoder.decoder.0.conv", d.decoderDim, d.latent, 7);
  uint32_t channels = d.decoderDim;
  for (uint32_t b = 0; b < d.upsampleRates.size(); ++b) {
    const std::string prefix = "decoder.decoder." + std::to_string(b + 1) + ".block.";
    CodecDecoderBlock block;
    block.rate = d.upsampleRates[b];
    block.snake = reader.snake(backend, prefix + "0", channels);
    const uint32_t out = channels / 2;
    block.upsample = reader.transposed(backend, prefix + "1.conv", channels, out, block.rate, 2 * block.rate);
    const uint32_t dilations[3] = {1, 3, 9};
    for (uint32_t u = 0; u < 3; ++u) {
      const std::string unit = prefix + std::to_string(u + 2) + ".";
      block.units[u].act1 = reader.snake(backend, unit + "act1", out);
      block.units[u].conv1 = reader.conv(backend, unit + "conv1.conv", out, out, 7, dilations[u]);
      block.units[u].act2 = reader.snake(backend, unit + "act2", out);
      block.units[u].conv2 = reader.conv(backend, unit + "conv2.conv", out, out, 1);
    }
    w.blocks.push_back(std::move(block));
    channels = out;
  }
  const uint32_t last = 5 + static_cast<uint32_t>(d.upsampleRates.size()) - 4;
  w.outputSnake = reader.snake(backend, "decoder.decoder." + std::to_string(last), channels);
  {
    const std::string prefix = "decoder.decoder." + std::to_string(last + 1) + ".conv";
    const std::vector<float> weight = reader.floats(prefix + ".weight", {1, channels, 7});
    const std::vector<float> bias = reader.floats(prefix + ".bias", {1});
    std::vector<float> packed(uint64_t{7} * channels); // [taps][channels]
    for (uint32_t c = 0; c < channels; ++c)
      for (uint32_t k = 0; k < 7; ++k) packed[uint64_t{k} * channels + c] = weight[uint64_t{c} * 7 + k];
    w.outputConv.weight = uploadFloats(backend, packed, "codec output weight");
    w.outputConv.bias = uploadFloats(backend, bias, "codec output bias");
    w.outputConv.inputChannels = channels;
    w.outputConv.outputChannels = 1;
    w.outputConv.taps = 7;
  }
  return w;
}

} // namespace splash::ops::omni
