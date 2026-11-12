#include "video/model/TextEncoder.hpp"

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
#include "metal/abi/video/TextEncoderABI.h"
#include "metal/abi/video/VisionTowerABI.h"
#include "video/model/VisionTowerWeights.hpp"
#include "video/ops/VisionTower.hpp"
#include "AwakeClock.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::video::model {
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
// MLX's affine quantization of the checkpoint: b-bit codes in groups of g
// with a bf16 scale and bias each, per module (mlx-community's uniform 4-, 5-,
// 6- and 8-bit checkpoints, and mixed recipes whose modules differ).
bool affineFormat(uint32_t bits, uint32_t group) {
  return (bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8) &&
         (group == 32 || group == 64 || group == 128) && quant_affine_format_of(bits, group) < GGUF_FMT_COUNT;
}
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
    // The default width and group; a module the config names apart (MLX's
    // mixed recipes) may differ, and affineRows reads each module's from its
    // tensors' shapes.
    if (!affineFormat(static_cast<uint32_t>(number(quantization, @"bits")),
                      static_cast<uint32_t>(number(quantization, @"group_size"))))
      throw std::invalid_argument("text encoder quantization is not an MLX affine format the engine reads");
  }
}

// An MLX affine module's codes, scales and biases as rows of its format.
splash::model::gguf::TensorRows affineRows(const splash::model::SafetensorsCheckpoint &checkpoint,
                                           const std::string &module, uint64_t rows, uint64_t columns) {
  const splash::model::SourceTensor &codes = checkpoint.require(module + ".weight");
  const splash::model::SourceTensor &scales = checkpoint.require(module + ".scales");
  const splash::model::SourceTensor &biases = checkpoint.require(module + ".biases");
  // The module's width and group, from its tensors' shapes.
  const bool shaped = codes.shape.size() == 2 && scales.shape.size() == 2 && codes.shape[0] == rows &&
                      scales.shape[0] == rows && scales.shape[1] && columns % scales.shape[1] == 0 &&
                      (codes.shape[1] * 32) % columns == 0;
  const auto bits = shaped ? static_cast<uint32_t>(codes.shape[1] * 32 / columns) : 0U;
  const auto group = shaped ? static_cast<uint32_t>(columns / scales.shape[1]) : 0U;
  const std::vector<uint64_t> groups{rows, shaped ? columns / group : 0};
  if (!shaped || !affineFormat(bits, group) || codes.dtype != "U32" ||
      codes.shape != std::vector<uint64_t>{rows, columns * bits / 32} || scales.dtype != "BF16" ||
      scales.shape != groups || biases.dtype != "BF16" || biases.shape != groups)
    throw std::invalid_argument("text encoder tensor type or shape does not match: " + module);
  const QuantFormat &format = kQuantFormats[quant_affine_format_of(bits, group)];
  return {module,
          format.ggml_type,
          0,
          rows,
          splash::model::ggufRowBytes(format, columns),
          {},
          codes.file,
          {&codes, &scales, &biases}};
}

void norm(splash::model::gguf::ImageBuilder &image, const splash::model::SafetensorsCheckpoint &checkpoint,
          const std::string &name, uint64_t elements) {
  const splash::model::SourceTensor &weight = checkpoint.require(name + ".weight");
  if (weight.dtype != "BF16" || weight.shape != std::vector<uint64_t>{elements})
    throw std::invalid_argument("text encoder norm type or shape does not match: " + name);
  image.copy({name, splash::model::ggml::kBF16, weight.offset, 1, weight.bytes, {}, weight.file});
}

void projection(splash::model::gguf::ImageBuilder &image, const splash::model::SafetensorsCheckpoint &checkpoint,
                const std::string &module, uint64_t rows, uint64_t columns) {
  const splash::model::gguf::TensorRows source = affineRows(checkpoint, module, rows, columns);
  splash::model::gguf::Repack repack = image.planes(gguf_format_of(source.type), rows, columns, module);
  repack.sources.push_back(source);
  image.repack(std::move(repack));
}

// A layer's image: its norms and projections in the order Layer reads them.
splash::model::gguf::Image layerImage(const splash::model::SafetensorsCheckpoint &checkpoint, uint32_t layer) {
  splash::model::gguf::ImageBuilder image("layer-" + std::to_string(layer) + ".bin", layer, 0);
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
splash::model::gguf::Image embeddingImage(const splash::model::SafetensorsCheckpoint &checkpoint) {
  splash::model::gguf::ImageBuilder image("embedding.bin", kVocabulary, kHidden);
  const splash::model::gguf::TensorRows rows = affineRows(checkpoint, kPrefix + "embed_tokens", kVocabulary, kHidden);
  image.descriptor(rows.type, kVocabulary, kHidden, {}, {kVocabulary * rows.rowBytes, 0, 0}, rows.name);
  image.copy(rows);
  return image.finish();
}

struct Layer {
  splash::ops::NormWeights inputNorm;
  splash::ops::Projection qkv;
  splash::ops::NormWeights queryNorm, keyNorm;
  splash::ops::Projection output;
  splash::ops::NormWeights postNorm;
  splash::ops::Projection gate, up, down;
};

Layer readLayer(splash::model::WeightFile &file) {
  Layer layer;
  layer.inputNorm = splash::model::readNorm(file, kHidden, false, "input-norm");
  splash::ops::BlockWeights qkv;
  uint32_t column = 0;
  for (const char *part : {"q", "k", "v"}) {
    splash::ops::QuantizedSegment segment = splash::model::readQuantizedSegment(file, part);
    segment.columnOffset = column;
    column += segment.outputSize;
    qkv.segments.push_back(std::move(segment));
  }
  layer.qkv = splash::ops::Projection(kQkvWidth, kHidden, std::move(qkv));
  layer.queryNorm = splash::model::readNorm(file, kHead, false, "query-norm");
  layer.keyNorm = splash::model::readNorm(file, kHead, false, "key-norm");
  layer.output = splash::model::readBlockProjection(file, kHidden, kQueryWidth, "attention-output");
  layer.postNorm = splash::model::readNorm(file, kHidden, false, "post-attention-norm");
  layer.gate = splash::model::readBlockProjection(file, kIntermediate, kHidden, "gate");
  layer.up = splash::model::readBlockProjection(file, kIntermediate, kHidden, "up");
  layer.down = splash::model::readBlockProjection(file, kHidden, kIntermediate, "down");
  file.finish();
  return layer;
}

} // namespace

struct TextEncoder::Impl {
  Impl(metal::MetalBackend &backend, const std::filesystem::path &directory)
      : backend(backend), checkpoint(directory), linear(backend.capabilities(), splash::ops::Int8Activations::Off) {
    requireConfiguration(directory / "config.json");
    for (uint32_t layer = 0; layer < kLayers; ++layer) images.push_back(layerImage(checkpoint, layer));
    images.push_back(embeddingImage(checkpoint));
  }

  splash::model::WeightFile load(splash::model::WeightImages &memory, size_t index) {
    const splash::model::gguf::Image &plan = images[index];
    return memory.load({"encoder/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
                        [this, index](std::span<uint8_t>, const metal::MetalBuffer &buffer) {
                          splash::model::writeGgufImage(backend, buffer, images[index]);
                          checkpoint.checkUnchanged();
                        }});
  }

  metal::MetalBuffer allocate(uint64_t bytes, const char *label) {
    return bytes ? backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label) : metal::MetalBuffer{};
  }

  // The vision tower on every image of a presentation, streamed a stage at a time: each command holds one stage's
  // weights (the embedding, a block, with a deepstack merger after blocks 8 / 16 / 24, the merger) and runs it on
  // every image. Returns the images' merger and deepstack rows; their other state is released.
  std::vector<ops::VisionTowerImage> tower(std::span<const TextEncoder::Image> images,
                                           TextEncoder::VisionTiming &timing) {
    const auto start = splash::AwakeClock::now();
    const uint64_t before = backend.refreshMemoryStats().allocatedBytes;
    double loading = 0;
    const auto timed = [&loading](auto &&load) {
      const auto t0 = splash::AwakeClock::now();
      auto value = load();
      loading += splash::millisecondsSince(t0) / 1000.0;
      return value;
    };
    const VisionTowerWeights weights(backend, checkpoint);
    std::vector<ops::VisionGrid> grids;
    uint32_t patches = 0;
    for (const TextEncoder::Image &image : images) {
      grids.push_back(ops::VisionTower::grid(image.height, image.width));
      patches = std::max(patches, grids.back().patches());
    }
    std::vector<ops::VisionTowerImage> states;
    {
      const ops::VisionTower tower(backend, patches);
      for (size_t i = 0; i < images.size(); ++i)
        states.push_back(tower.image(grids[i], images[i].rgb, images[i].rgb2));
      const auto submit = [this](const metal::CommandGraph &graph) {
        static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      };
      {
        const ops::VisionTowerEmbedding embedding = timed([&] { return weights.embedding(); });
        metal::CommandGraph graph;
        for (const auto &state : states) tower.addEmbedding(graph, embedding, state);
        submit(graph);
      }
      for (uint32_t index = 0; index < ops::VisionTower::kDepth; ++index) {
        const ops::VisionTowerBlock block = timed([&] { return weights.block(index); });
        metal::CommandGraph graph;
        for (const auto &state : states) tower.addBlock(graph, block, state);
        const auto deep = std::find(ops::VisionTower::kDeepstack.begin(), ops::VisionTower::kDeepstack.end(), index);
        std::optional<ops::VisionTowerMerger> merger;
        if (deep != ops::VisionTower::kDeepstack.end()) {
          const auto output = static_cast<uint32_t>(1 + (deep - ops::VisionTower::kDeepstack.begin()));
          merger = timed([&] { return weights.merger(output); });
          for (const auto &state : states) tower.addMerger(graph, *merger, state, output);
        }
        submit(graph);
      }
      const ops::VisionTowerMerger merger = timed([&] { return weights.merger(0); });
      metal::CommandGraph graph;
      for (const auto &state : states) tower.addMerger(graph, merger, state, 0);
      submit(graph);
    }
    const metal::MetalMemoryStats after = backend.refreshMemoryStats();
    timing.peakBytes = after.peakAllocatedBytes > before ? after.peakAllocatedBytes - before : 0;
    for (auto &state : states)
      state.pixels = state.pixels2 = state.hidden = state.ropeCos = state.ropeSin = metal::MetalBuffer{};
    timing.loadSeconds = loading;
    timing.seconds = splash::millisecondsSince(start) / 1000.0;
    timing.images = static_cast<uint32_t>(images.size());
    timing.rows = 0;
    for (const auto &grid : grids) timing.rows += grid.merged();
    return states;
  }

  metal::MetalBackend &backend;
  splash::model::SafetensorsCheckpoint checkpoint;
  splash::ops::Linear linear;
  std::vector<splash::model::gguf::Image> images; // layers, then the embedding
  TextEncoder::VisionTiming vision;
};

TextEncoder::TextEncoder(metal::MetalBackend &backend, const std::filesystem::path &directory)
    : impl_(std::make_unique<Impl>(backend, directory)) {}

TextEncoder::~TextEncoder() = default;

const TextEncoder::VisionTiming &TextEncoder::lastVision() const { return impl_->vision; }

std::vector<TextEncoder::VisionBlock> TextEncoder::visionBlocks(std::span<const uint32_t> tokens) {
  std::vector<VisionBlock> blocks;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i] == kImagePad || tokens[i] == kVideoPad || tokens[i] == kVisionEnd)
      throw std::invalid_argument("a vision pad or vision end outside a vision block of the presentation");
    if (tokens[i] != kVisionStart) continue;
    const uint32_t pad = i + 1 < tokens.size() ? tokens[i + 1] : 0;
    size_t j = i + 1;
    if (pad == kImagePad || pad == kVideoPad)
      while (j < tokens.size() && tokens[j] == pad) ++j;
    if (j == i + 1 || j == tokens.size() || tokens[j] != kVisionEnd)
      throw std::invalid_argument("a presentation's vision block is not <|vision_start|>, <|image_pad|>... or "
                                  "<|video_pad|>..., <|vision_end|>");
    blocks.push_back({static_cast<uint32_t>(i + 1), static_cast<uint32_t>(j - i - 1), pad == kVideoPad});
    i = j;
  }
  return blocks;
}

std::vector<uint8_t> TextEncoder::tags(std::span<const uint32_t> tokens) {
  std::vector<uint8_t> tags(tokens.size(), uint8_t{1});
  for (const VisionBlock &block : visionBlocks(tokens))
    std::fill_n(tags.begin() + block.start - 1, block.pads + 2, uint8_t{0});
  return tags;
}

uint32_t TextEncoder::imageTokens(uint32_t width, uint32_t height) {
  return ops::VisionTower::grid(height, width).merged();
}

namespace {
// Python's round() of a non-negative double: half to even.
uint64_t roundHalfEven(double value) {
  const double floor = std::floor(value);
  const double fraction = value - floor;
  const auto whole = static_cast<uint64_t>(floor);
  return fraction > 0.5 || (fraction == 0.5 && whole % 2) ? whole + 1 : whole;
}
} // namespace

std::vector<TextEncoder::VideoBlockFrames> TextEncoder::videoBlockFrames(uint32_t frames, double fps,
                                                                         double sampleFps) {
  constexpr size_t kTemporalPatch = 2;
  if (!(fps > 0) || !(sampleFps > 0) || !(fps / sampleFps >= 1))
    throw std::invalid_argument("a reference video is sampled at a positive rate no higher than its own");
  const double stride = fps / sampleFps;
  std::vector<uint32_t> indices;
  for (double cursor = 0; roundHalfEven(cursor) < frames; cursor += stride) {
    const auto index = static_cast<uint32_t>(roundHalfEven(cursor));
    if (indices.empty() || index > indices.back()) indices.push_back(index);
  }
  if (indices.size() < kTemporalPatch)
    throw std::invalid_argument("a reference video of " + std::to_string(frames) +
                                " frames yields fewer than two samples for the vision tower");
  std::vector<VideoBlockFrames> blocks;
  for (size_t i = 0; i < indices.size(); i += kTemporalPatch) {
    const size_t j = std::min(i + kTemporalPatch - 1, indices.size() - 1);
    blocks.push_back({indices[i], indices[j], (double(i) / sampleFps + double(j) / sampleFps) / 2});
  }
  return blocks;
}

std::vector<uint32_t> TextEncoder::positions(std::span<const uint32_t> tokens, std::span<const Image> images) {
  const std::vector<VisionBlock> blocks = visionBlocks(tokens);
  if (blocks.size() != images.size())
    throw std::invalid_argument("the presentation has " + std::to_string(blocks.size()) + " vision blocks for " +
                                std::to_string(images.size()) + " images");
  std::vector<uint32_t> out(tokens.size() * 3);
  uint32_t current = 0;
  size_t block = 0;
  for (size_t i = 0; i < tokens.size();) {
    if (block < blocks.size() && i == blocks[block].start) {
      if (!blocks[block].video && images[block].rgb2)
        throw std::invalid_argument("vision block " + std::to_string(block) +
                                    " is an image block; its image has no second frame");
      const ops::VisionGrid grid = ops::VisionTower::grid(images[block].height, images[block].width);
      if (grid.merged() != blocks[block].pads)
        throw std::invalid_argument("vision block " + std::to_string(block) + " holds " +
                                    std::to_string(blocks[block].pads) + " pads; its image takes " +
                                    std::to_string(grid.merged()));
      const uint32_t rows = grid.height / 2, columns = grid.width / 2;
      for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < columns; ++c) {
          uint32_t *p = &out[(i + size_t{r} * columns + c) * 3];
          p[0] = current;
          p[1] = current + r;
          p[2] = current + c;
        }
      i += blocks[block].pads;
      current += std::max(rows, columns);
      ++block;
      continue;
    }
    out[i * 3] = out[i * 3 + 1] = out[i * 3 + 2] = current++;
    ++i;
  }
  return out;
}

std::vector<float> TextEncoder::encode(std::span<const uint32_t> tokens, std::span<const Image> images) {
  const auto rows = static_cast<uint32_t>(tokens.size());
  if (!rows || rows > kMaximumPresentationTokens)
    throw std::invalid_argument("a text encoder prompt holds 1 to " + std::to_string(kMaximumPresentationTokens) +
                                " tokens");
  if (std::any_of(tokens.begin(), tokens.end(), [](uint32_t token) { return token >= kVocabulary; }))
    throw std::invalid_argument("a prompt token is outside the text encoder's vocabulary");
  Impl &e = *impl_;
  e.vision = {};
  // A presentation with images: its M-RoPE positions (which check its blocks against the images) and the tower's
  // rows, before any text layer loads.
  const std::vector<VisionBlock> blocks = visionBlocks(tokens);
  const bool vision = !blocks.empty() || !images.empty();
  std::vector<uint32_t> rotary;
  std::vector<ops::VisionTowerImage> towerRows;
  if (vision) {
    rotary = positions(tokens, images);
    towerRows = e.tower(images, e.vision);
  }
  // The weights stream a layer at a time: each layer's image loads into
  // memory of its own just before its command and is released once the
  // command has run, so the encode holds one layer, the activations and,
  // until the token gather has run, the token table. The commands are the
  // resident encode's, in its order: only the loads move between them.
  const auto loadLayer = [&e](splash::model::WeightImages &memory, uint32_t index) {
    splash::model::WeightFile file = e.load(memory, index);
    return readLayer(file);
  };
  auto memory = std::make_unique<splash::model::WeightImages>(e.backend, "text-encoder");
  std::optional<Layer> layer = loadLayer(*memory, 0);

  // The linears run in passes of at most kMaximumTokens rows, one pass for a
  // prompt within it; the norms, rotation and attention run over every row.
  // Every projection's input and output hold the rows its plan stores past a
  // pass's start: layer 0's plans size them, and every later layer's must fit.
  const uint32_t pass = std::min(rows, kMaximumTokens);
  const uint32_t passes = (rows + pass - 1) / pass;
  const uint32_t lastStart = (passes - 1) * pass;
  const auto sizes = [&](const Layer &weights) {
    uint32_t storage = pass;
    splash::ops::LinearScratchSize scratchSize;
    for (const auto &[projection, epilogue] :
         {std::pair{&weights.qkv, splash::ops::LinearEpilogue::None},
          {&weights.output, splash::ops::LinearEpilogue::Residual},
          {&weights.gate, splash::ops::LinearEpilogue::None},
          {&weights.up, splash::ops::LinearEpilogue::UpWithGate},
          {&weights.down, splash::ops::LinearEpilogue::Residual}}) {
      for (const uint32_t count : {pass, rows - lastStart})
        storage = std::max(storage, e.linear.prefillPlan(*projection, count, epilogue).storageRows());
      scratchSize.include(e.linear.prefillScratchSize(projection->shape()));
    }
    return std::pair{storage, scratchSize};
  };
  const auto planned = sizes(*layer);
  const uint32_t passStorage = planned.first;
  const uint32_t storage = std::max(rows, lastStart + passStorage);
  const splash::ops::LinearScratchSize scratchSize = planned.second;
  const auto rowsOf = [&](uint32_t width) { return uint64_t{storage} * width * sizeof(uint16_t); };
  const auto passRowsOf = [&](uint32_t width) { return uint64_t{passStorage} * width * sizeof(uint16_t); };
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
  const splash::ops::PrefillFfnBuffers ffn{
      e.allocate(rowsOf(kHidden), "encoder-normalized"),
      e.allocate(passRowsOf(kIntermediate), "encoder-gate"),
      e.allocate(passRowsOf(kIntermediate), "encoder-intermediate"),
      {e.allocate(scratchSize.input, "encoder-linear-input"), e.allocate(scratchSize.sums, "encoder-linear-sums"),
       e.allocate(scratchSize.partials, "encoder-linear-partials"),
       e.allocate(scratchSize.counters, "encoder-linear-counters")}};
  // A pass's rows of a buffer of every row (the buffer itself for a one-pass prompt).
  const auto at = [&](const metal::MetalBuffer &buffer, uint32_t start, uint32_t width) {
    return passes == 1 ? buffer
                       : e.backend.view(buffer, uint64_t{start} * width * sizeof(uint16_t), passRowsOf(width));
  };
  const auto eachPass = [&](auto &&add) {
    for (uint32_t start = 0; start < rows; start += pass) add(start, std::min(pass, rows - start));
  };
  std::memcpy(tokenIds.contents(), tokens.data(), tokens.size_bytes());
  // inv_freq[i] = theta^(-2i / d), in fp32 as the reference computes it.
  auto *inverse = static_cast<float *>(frequencies.contents());
  for (uint32_t i = 0; i < kHead / 2; ++i)
    inverse[i] = 1.0F / std::pow(static_cast<float>(kRotaryTheta), float(2 * i) / float(kHead));

  // A presentation with images: its rows' (t, h, w) positions and each image's pad rows.
  metal::MetalBuffer rotaryPositions;
  std::vector<metal::MetalBuffer> padRows;
  if (vision) {
    rotaryPositions = e.allocate(uint64_t{rows} * 4 * sizeof(uint32_t), "encoder-mrope-positions");
    auto *p = static_cast<uint32_t *>(rotaryPositions.contents());
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t axis = 0; axis < 3; ++axis) p[r * 4 + axis] = rotary[size_t{r} * 3 + axis];
      p[r * 4 + 3] = 0;
    }
    for (const VisionBlock &block : blocks) {
      padRows.push_back(e.allocate(uint64_t{block.pads} * sizeof(uint32_t), "encoder-vision-rows"));
      auto *index = static_cast<uint32_t *>(padRows.back().contents());
      for (uint32_t i = 0; i < block.pads; ++i) index[i] = block.start + i;
    }
  }
  // mode 0: the merger's rows replace the pad rows' embeddings; 1: deepstack merger `output - 1`'s rows are added.
  const auto addVisionRows = [&](metal::CommandGraph &graph, const metal::MetalBuffer &hidden, uint32_t output) {
    for (size_t i = 0; i < blocks.size(); ++i)
      graph.add("text_encoder_vision_rows", {hidden, padRows[i], towerRows[i].outputs[output]},
                TextEncoderVisionRowsParams{blocks[i].pads, kHidden, output ? 1U : 0U}, {blocks[i].pads, 1, 1});
  };

  const auto submit = [&](const metal::CommandGraph &graph) {
    static_cast<void>(e.backend.submitCommandAsync(graph.dispatches()).wait());
  };
  {
    // The token table loads after layer 0 and is released once gathered.
    splash::model::WeightImages tableMemory(e.backend, "text-encoder");
    splash::model::WeightFile tableFile = e.load(tableMemory, kLayers);
    const splash::ops::EmbeddingWeights table =
        splash::model::readBlockEmbedding(tableFile, kVocabulary, kHidden, "embedding");
    tableFile.finish();
    metal::CommandGraph graph;
    splash::ops::Embedding::add(graph, tokenIds, table, hiddenA, rows);
    if (vision) addVisionRows(graph, hiddenA, 0);
    submit(graph);
    tableMemory.release();
  }
  const TextEncoderHeadParams rotation{rows, kQueryHeads, kKvHeads, kNormEpsilon};
  const TextEncoderHeadParams scores{rows, kQueryHeads, kKvHeads, 1.0F / std::sqrt(float(kHead))};
  for (uint32_t index = 0; index < kLayers; ++index) {
    if (index) {
      memory = std::make_unique<splash::model::WeightImages>(e.backend, "text-encoder");
      layer = loadLayer(*memory, index);
      const auto [layerStorage, layerScratch] = sizes(*layer);
      if (layerStorage > passStorage || layerScratch.input > scratchSize.input ||
          layerScratch.sums > scratchSize.sums || layerScratch.partials > scratchSize.partials ||
          layerScratch.counters > scratchSize.counters || layerScratch.rotated > scratchSize.rotated)
        throw std::invalid_argument("text encoder layer " + std::to_string(index) +
                                    " needs more rows or scratch than layer 0");
    }
    const Layer &weights = *layer;
    const metal::MetalBuffer &input = index & 1 ? hiddenB : hiddenA;
    const metal::MetalBuffer &output = index & 1 ? hiddenA : hiddenB;
    metal::CommandGraph graph;
    splash::ops::Normalization::addRms(graph, input, weights.inputNorm, ffn.normalized, kHidden, rows);
    eachPass([&](uint32_t start, uint32_t count) {
      e.linear.addPrefill(graph, at(ffn.normalized, start, kHidden), weights.qkv, at(qkv, start, kQkvWidth), count,
                          ffn.scratch);
    });
    if (vision)
      graph.add("text_encoder_qk_norm_mrope",
                {qkv, weights.queryNorm.buffer, weights.keyNorm.buffer, frequencies, queries, keys, values,
                 rotaryPositions},
                rotation, {rows, kQueryHeads + 2 * kKvHeads, 1}, {32, 1, 1});
    else
      graph.add("text_encoder_qk_norm_rope",
                {qkv, weights.queryNorm.buffer, weights.keyNorm.buffer, frequencies, queries, keys, values}, rotation,
                {rows, kQueryHeads + 2 * kKvHeads, 1}, {32, 1, 1});
    graph.add("text_encoder_attention", {queries, keys, values, attention}, scores, {rows, kQueryHeads, 1},
              {32, 1, 1});
    eachPass([&](uint32_t start, uint32_t count) {
      e.linear.addPrefillResidual(graph, at(attention, start, kQueryWidth), weights.output, at(input, start, kHidden),
                                  at(attended, start, kHidden), count, ffn.scratch);
    });
    splash::ops::Normalization::addRms(graph, attended, weights.postNorm, ffn.normalized, kHidden, rows);
    eachPass([&](uint32_t start, uint32_t count) {
      const splash::ops::PrefillFfnBuffers passFfn{at(ffn.normalized, start, kHidden), ffn.gateScratch,
                                                   ffn.intermediate, ffn.scratch};
      e.linear.addPrefillSwiGlu(graph, {&weights.gate, &weights.up, &weights.down}, passFfn,
                                at(attended, start, kHidden), at(output, start, kHidden), count);
    });
    // Qwen3-VL's deepstack: the visual rows gain deepstack merger i's rows after decoder layer i.
    if (vision && index < ops::VisionTower::kDeepstack.size()) addVisionRows(graph, output, 1 + index);
    submit(graph);
    layer.reset();
    memory->release();
  }
  const auto *result = static_cast<const uint16_t *>((kLayers & 1 ? hiddenB : hiddenA).contents());
  std::vector<float> hidden(size_t{rows} * kHidden);
  for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = std::bit_cast<float>(uint32_t{result[i]} << 16);
  return hidden;
}

} // namespace splash::video::model
