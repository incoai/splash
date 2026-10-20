#include "video/TextEncoder.hpp"

#include "metal/CommandGraph.hpp"
#include "metal/abi/QuantFormat.h"
#include "model/GgufFile.hpp"
#include "model/GgufImage.hpp"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightStore.hpp"
#include "ops/Embedding.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "video/TextEncoderABI.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::video {
namespace {

// Qwen3-VL-32B's language model.
constexpr uint32_t kSourceLayers = 64;
constexpr uint32_t kHidden = TextEncoder::kHiddenSize;
constexpr uint32_t kIntermediate = 25600;
constexpr uint32_t kVocabulary = 151936;
constexpr uint32_t kQueryHeads = 64;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kHead = SPLASH_TEXT_ENCODER_HEAD_DIMENSION;
constexpr uint32_t kQueryWidth = kQueryHeads * kHead;
constexpr uint32_t kKvWidth = kKvHeads * kHead;
constexpr uint32_t kQkvWidth = kQueryWidth + 2 * kKvWidth;
constexpr double kRotaryTheta = 5'000'000.0;
constexpr float kNormEpsilon = 1e-6F;
// MLX's affine quantization of the checkpoint.
constexpr uint32_t kBits = 4, kGroup = 64;
const std::string kPrefix = "language_model.model.";

double number(NSDictionary *object, NSString *key) {
  id value = object[key];
  if (![value isKindOfClass:[NSNumber class]])
    throw std::invalid_argument("text encoder config has no number " + std::string(key.UTF8String));
  return [value doubleValue];
}

// The checkpoint's config.json must describe Qwen3-VL-32B's language model,
// quantized as the images read it.
void requireConfiguration(const std::filesystem::path &path) {
  @autoreleasepool {
    NSData *data = [NSData dataWithContentsOfFile:@(path.c_str())];
    NSDictionary *config = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    NSDictionary *text = [config isKindOfClass:[NSDictionary class]] ? config[@"text_config"] : nil;
    NSDictionary *quantization = [config isKindOfClass:[NSDictionary class]] ? config[@"quantization"] : nil;
    if (![text isKindOfClass:[NSDictionary class]] || ![quantization isKindOfClass:[NSDictionary class]])
      throw std::invalid_argument("text encoder config " + path.string() +
                                  " holds no text_config and quantization");
    const std::pair<NSString *, double> text_fields[] = {
        {@"num_hidden_layers", kSourceLayers}, {@"hidden_size", kHidden},
        {@"intermediate_size", kIntermediate}, {@"vocab_size", kVocabulary},
        {@"num_attention_heads", kQueryHeads}, {@"num_key_value_heads", kKvHeads},
        {@"head_dim", kHead},                  {@"rope_theta", kRotaryTheta},
        {@"rms_norm_eps", 1e-6}};
    for (const auto &[key, value] : text_fields)
      if (number(text, key) != value)
        throw std::invalid_argument("text encoder config " + std::string(key.UTF8String) +
                                    " is not Qwen3-VL-32B's");
    if (number(quantization, @"bits") != kBits || number(quantization, @"group_size") != kGroup)
      throw std::invalid_argument("text encoder weights are not 4-bit in groups of 64");
  }
}

// An MLX affine module's codes, scales and biases as rows of its format.
model::gguf::TensorRows affineRows(const model::SafetensorsCheckpoint &checkpoint, const std::string &module,
                                   uint64_t rows, uint64_t columns) {
  const model::SourceTensor &codes = checkpoint.require(module + ".weight");
  const model::SourceTensor &scales = checkpoint.require(module + ".scales");
  const model::SourceTensor &biases = checkpoint.require(module + ".biases");
  const std::vector<uint64_t> groups{rows, columns / kGroup};
  if (codes.dtype != "U32" || codes.shape != std::vector<uint64_t>{rows, columns * kBits / 32} ||
      scales.dtype != "BF16" || scales.shape != groups || biases.dtype != "BF16" || biases.shape != groups)
    throw std::invalid_argument("text encoder tensor type or shape does not match: " + module);
  const QuantFormat &format = kQuantFormats[quant_affine_format_of(kBits, kGroup)];
  return {module,
          format.ggml_type,
          0,
          rows,
          model::ggufRowBytes(format, columns),
          {},
          codes.file,
          {&codes, &scales, &biases}};
}

void norm(model::gguf::ImageBuilder &image, const model::SafetensorsCheckpoint &checkpoint,
          const std::string &name, uint64_t elements) {
  const model::SourceTensor &weight = checkpoint.require(name + ".weight");
  if (weight.dtype != "BF16" || weight.shape != std::vector<uint64_t>{elements})
    throw std::invalid_argument("text encoder norm type or shape does not match: " + name);
  image.copy({name, model::ggml::kBF16, weight.offset, 1, weight.bytes, {}, weight.file});
}

void projection(model::gguf::ImageBuilder &image, const model::SafetensorsCheckpoint &checkpoint,
                const std::string &module, uint64_t rows, uint64_t columns) {
  const model::gguf::TensorRows source = affineRows(checkpoint, module, rows, columns);
  model::gguf::Repack repack = image.planes(gguf_format_of(source.type), rows, columns, module);
  repack.sources.push_back(source);
  image.repack(std::move(repack));
}

// A layer's image: its norms and projections in the order Layer reads them.
model::gguf::Image layerImage(const model::SafetensorsCheckpoint &checkpoint, uint32_t layer) {
  model::gguf::ImageBuilder image("layer-" + std::to_string(layer) + ".bin", layer, 0);
  const std::string prefix = kPrefix + "layers." + std::to_string(layer) + ".";
  const std::string attention = prefix + "self_attn.", mlp = prefix + "mlp.";
  norm(image, checkpoint, prefix + "input_layernorm", kHidden);
  projection(image, checkpoint, attention + "q_proj", kQueryWidth, kHidden);
  projection(image, checkpoint, attention + "k_proj", kKvWidth, kHidden);
  projection(image, checkpoint, attention + "v_proj", kKvWidth, kHidden);
  norm(image, checkpoint, attention + "q_norm", kHead);
  norm(image, checkpoint, attention + "k_norm", kHead);
  projection(image, checkpoint, attention + "o_proj", kHidden, kQueryWidth);
  norm(image, checkpoint, prefix + "post_attention_layernorm", kHidden);
  projection(image, checkpoint, mlp + "gate_proj", kIntermediate, kHidden);
  projection(image, checkpoint, mlp + "up_proj", kIntermediate, kHidden);
  projection(image, checkpoint, mlp + "down_proj", kHidden, kIntermediate);
  return image.finish();
}

// The token table's native rows, which the token gather reads.
model::gguf::Image embeddingImage(const model::SafetensorsCheckpoint &checkpoint) {
  model::gguf::ImageBuilder image("embedding.bin", kVocabulary, kHidden);
  const model::gguf::TensorRows rows = affineRows(checkpoint, kPrefix + "embed_tokens", kVocabulary, kHidden);
  image.descriptor(rows.type, kVocabulary, kHidden, {}, {kVocabulary * rows.rowBytes, 0, 0}, rows.name);
  image.copy(rows);
  return image.finish();
}

struct Layer {
  ops::NormWeights inputNorm;
  ops::Projection qkv;
  ops::NormWeights queryNorm, keyNorm;
  ops::Projection output;
  ops::NormWeights postNorm;
  ops::Projection gate, up, down;
};

Layer readLayer(model::WeightFile &file) {
  Layer layer;
  layer.inputNorm = model::readNorm(file, kHidden, false, "input-norm");
  ops::BlockWeights qkv;
  uint32_t column = 0;
  for (const char *part : {"q", "k", "v"}) {
    ops::QuantizedSegment segment = model::readQuantizedSegment(file, part);
    segment.columnOffset = column;
    column += segment.outputSize;
    qkv.segments.push_back(std::move(segment));
  }
  layer.qkv = ops::Projection(kQkvWidth, kHidden, std::move(qkv));
  layer.queryNorm = model::readNorm(file, kHead, false, "query-norm");
  layer.keyNorm = model::readNorm(file, kHead, false, "key-norm");
  layer.output = model::readBlockProjection(file, kHidden, kQueryWidth, "attention-output");
  layer.postNorm = model::readNorm(file, kHidden, false, "post-attention-norm");
  layer.gate = model::readBlockProjection(file, kIntermediate, kHidden, "gate");
  layer.up = model::readBlockProjection(file, kIntermediate, kHidden, "up");
  layer.down = model::readBlockProjection(file, kHidden, kIntermediate, "down");
  file.finish();
  return layer;
}

} // namespace

struct TextEncoder::Impl {
  Impl(metal::MetalBackend &backend, const std::filesystem::path &directory)
      : backend(backend), checkpoint(directory), linear(backend.capabilities()) {
    requireConfiguration(directory / "config.json");
    for (uint32_t layer = 0; layer < kLayers; ++layer) images.push_back(layerImage(checkpoint, layer));
    images.push_back(embeddingImage(checkpoint));
  }

  model::WeightFile load(model::WeightImages &memory, size_t index) {
    const model::gguf::Image &plan = images[index];
    return memory.load({"encoder/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
                        [this, index](std::span<uint8_t>, const metal::MetalBuffer &buffer) {
                          model::writeGgufImage(backend, buffer, images[index]);
                          checkpoint.checkUnchanged();
                        }});
  }

  metal::MetalBuffer allocate(uint64_t bytes, const char *label) {
    return bytes ? backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label) : metal::MetalBuffer{};
  }

  metal::MetalBackend &backend;
  model::SafetensorsCheckpoint checkpoint;
  ops::Linear linear;
  std::vector<model::gguf::Image> images; // layers, then the embedding
};

TextEncoder::TextEncoder(metal::MetalBackend &backend, const std::filesystem::path &directory)
    : impl_(std::make_unique<Impl>(backend, directory)) {}

TextEncoder::~TextEncoder() = default;

std::vector<float> TextEncoder::encode(std::span<const uint32_t> tokens) {
  const auto rows = static_cast<uint32_t>(tokens.size());
  if (!rows || rows > kMaximumTokens)
    throw std::invalid_argument("a text encoder prompt holds 1 to " + std::to_string(kMaximumTokens) + " tokens");
  if (std::any_of(tokens.begin(), tokens.end(), [](uint32_t token) { return token >= kVocabulary; }))
    throw std::invalid_argument("a prompt token is outside the text encoder's vocabulary");
  Impl &e = *impl_;
  model::WeightImages memory(e.backend, "text-encoder");
  std::vector<Layer> layers;
  layers.reserve(kLayers);
  for (uint32_t index = 0; index < kLayers; ++index) {
    model::WeightFile file = e.load(memory, index);
    layers.push_back(readLayer(file));
  }
  model::WeightFile tableFile = e.load(memory, kLayers);
  const ops::EmbeddingWeights table = model::readBlockEmbedding(tableFile, kVocabulary, kHidden, "embedding");
  tableFile.finish();

  // Every projection's input and output hold the rows its plan stores.
  const Layer &first = layers.front();
  uint32_t storage = rows;
  ops::LinearScratchSize scratchSize;
  for (const auto &[projection, epilogue] :
       {std::pair{&first.qkv, ops::LinearEpilogue::None}, {&first.output, ops::LinearEpilogue::Residual},
        {&first.gate, ops::LinearEpilogue::None}, {&first.up, ops::LinearEpilogue::UpWithGate},
        {&first.down, ops::LinearEpilogue::Residual}}) {
    storage = std::max(storage, e.linear.prefillPlan(*projection, rows, epilogue).storageRows());
    scratchSize.include(e.linear.prefillScratchSize(projection->shape()));
  }
  const auto rowsOf = [&](uint32_t width) { return uint64_t{storage} * width * sizeof(uint16_t); };
  const metal::MetalBuffer tokenIds = e.allocate(uint64_t{rows} * sizeof(uint32_t), "encoder-tokens");
  const metal::MetalBuffer hiddenA = e.allocate(rowsOf(kHidden), "encoder-hidden-a");
  const metal::MetalBuffer hiddenB = e.allocate(rowsOf(kHidden), "encoder-hidden-b");
  const metal::MetalBuffer qkv = e.allocate(rowsOf(kQkvWidth), "encoder-qkv");
  const metal::MetalBuffer queries = e.allocate(rowsOf(kQueryWidth), "encoder-queries");
  const metal::MetalBuffer keys = e.allocate(rowsOf(kKvWidth), "encoder-keys");
  const metal::MetalBuffer values = e.allocate(rowsOf(kKvWidth), "encoder-values");
  const metal::MetalBuffer attention = e.allocate(rowsOf(kQueryWidth), "encoder-attention");
  const metal::MetalBuffer attended = e.allocate(rowsOf(kHidden), "encoder-attended");
  const metal::MetalBuffer frequencies = e.allocate(kHead / 2 * sizeof(float), "encoder-inverse-frequencies");
  const ops::PrefillFfnBuffers ffn{
      e.allocate(rowsOf(kHidden), "encoder-normalized"),
      e.allocate(rowsOf(kIntermediate), "encoder-gate"),
      e.allocate(rowsOf(kIntermediate), "encoder-intermediate"),
      {e.allocate(scratchSize.input, "encoder-linear-input"), e.allocate(scratchSize.sums, "encoder-linear-sums"),
       e.allocate(scratchSize.partials, "encoder-linear-partials"),
       e.allocate(scratchSize.counters, "encoder-linear-counters")}};
  std::memcpy(tokenIds.contents(), tokens.data(), tokens.size_bytes());
  // inv_freq[i] = theta^(-2i / d), in fp32 as the reference computes it.
  auto *inverse = static_cast<float *>(frequencies.contents());
  for (uint32_t i = 0; i < kHead / 2; ++i)
    inverse[i] = 1.0F / std::pow(static_cast<float>(kRotaryTheta), float(2 * i) / float(kHead));

  const auto submit = [&](const metal::CommandGraph &graph) {
    static_cast<void>(e.backend.submitCommandAsync(graph.dispatches()).wait());
  };
  {
    metal::CommandGraph graph;
    ops::Embedding::add(graph, tokenIds, table, hiddenA, rows);
    submit(graph);
  }
  const TextEncoderHeadParams rotation{rows, kQueryHeads, kKvHeads, kNormEpsilon};
  const TextEncoderHeadParams scores{rows, kQueryHeads, kKvHeads, 1.0F / std::sqrt(float(kHead))};
  for (uint32_t index = 0; index < kLayers; ++index) {
    const Layer &layer = layers[index];
    const metal::MetalBuffer &input = index & 1 ? hiddenB : hiddenA;
    const metal::MetalBuffer &output = index & 1 ? hiddenA : hiddenB;
    metal::CommandGraph graph;
    ops::Normalization::addRms(graph, input, layer.inputNorm, ffn.normalized, kHidden, rows);
    e.linear.addPrefill(graph, ffn.normalized, layer.qkv, qkv, rows, ffn.scratch);
    graph.add("text_encoder_qk_norm_rope",
              {qkv, layer.queryNorm.buffer, layer.keyNorm.buffer, frequencies, queries, keys, values}, rotation,
              {rows, kQueryHeads + 2 * kKvHeads, 1}, {32, 1, 1});
    graph.add("text_encoder_attention", {queries, keys, values, attention}, scores, {rows, kQueryHeads, 1},
              {32, 1, 1});
    e.linear.addPrefillResidual(graph, attention, layer.output, input, attended, rows, ffn.scratch);
    ops::Normalization::addRms(graph, attended, layer.postNorm, ffn.normalized, kHidden, rows);
    e.linear.addPrefillSwiGlu(graph, {&layer.gate, &layer.up, &layer.down}, ffn, attended, output, rows);
    submit(graph);
  }
  const auto *result = static_cast<const uint16_t *>((kLayers & 1 ? hiddenB : hiddenA).contents());
  std::vector<float> hidden(size_t{rows} * kHidden);
  for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = std::bit_cast<float>(uint32_t{result[i]} << 16);
  return hidden;
}

} // namespace splash::video
