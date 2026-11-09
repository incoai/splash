#include "model/DraftCheckpoint.hpp"
#include "metal/abi/DraftAttention.h"
#include "model/DFlashDraft.hpp"
#include "model/GgufPreparation.hpp"

#include <initializer_list>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace splash::model {
namespace {

// Plans one image from the checkpoint's BF16 tensors, which must have the
// shapes the draft reads.
class Builder {
public:
  Builder(const SafetensorsCheckpoint &checkpoint, std::string name, uint32_t layer, uint32_t type)
      : checkpoint_(checkpoint), image_(std::move(name), layer, type) {}

  // A tensor copied as stored: a norm, a convolution's base kernel or a
  // selector codebook.
  void stored(const std::string &name, const std::vector<uint64_t> &shape) {
    const SourceTensor &tensor = require(name, shape);
    image_.copy({name, ggml::kBF16, tensor.offset, 1, tensor.bytes, {}, tensor.file});
  }

  void stacked(std::initializer_list<std::string> names, uint64_t rows, uint64_t columns) {
    const uint64_t bytes = rows * columns * 2;
    uint64_t destination = image_.section(names.size() * bytes);
    for (const auto &name : names) {
      const auto &tensor = require(name, {rows, columns});
      image_.copyAt(destination, {name, ggml::kBF16, tensor.offset, 1, tensor.bytes, {}, tensor.file}, gguf::Conversion::None);
      destination += bytes;
    }
  }

  // A projection of `columns` inputs: its parts' rows in order, af4g64 planes.
  void projection(std::initializer_list<std::pair<std::string, uint64_t>> parts, uint64_t columns) {
    const QuantFormat &format = kQuantFormats[GGUF_FMT_AF4G64];
    std::vector<gguf::TensorRows> sources;
    uint64_t rows = 0;
    for (const auto &[name, count] : parts) {
      const SourceTensor &tensor = require(name + ".weight", {count, columns});
      sources.push_back({name, format.ggml_type, 0, count, ggufRowBytes(format, columns), {}, tensor.file,
                         {.bfloat16 = &tensor}});
      rows += count;
    }
    gguf::Repack repack = image_.planes(GGUF_FMT_AF4G64, rows, columns, parts.begin()->first);
    repack.sources = std::move(sources);
    image_.repack(std::move(repack));
  }

  [[nodiscard]] gguf::Image finish() { return image_.finish(); }

private:
  const SourceTensor &require(const std::string &name, const std::vector<uint64_t> &shape) const {
    const SourceTensor &tensor = checkpoint_.require(name);
    if (tensor.dtype != "BF16" || tensor.shape != shape)
      throw WeightStoreError("source tensor type or shape does not match: " + name);
    return tensor;
  }

  const SafetensorsCheckpoint &checkpoint_;
  gguf::ImageBuilder image_;
};

gguf::Image layerImage(const SafetensorsCheckpoint &checkpoint, const DFlashDraftLayout &layout, uint32_t layer) {
  Builder b(checkpoint, "layer-" + std::to_string(layer) + ".bin", layer, 0);
  const std::string prefix = "layers." + std::to_string(layer) + ".";
  const std::string attention = prefix + "self_attn.";
  const uint64_t hidden = layout.hiddenSize, kv = uint64_t{layout.kvHeads} * layout.attentionHeadDimension;
  const auto convolution = [&](const std::string &name) {
    b.stored(name + ".base_kernel", {SPLASH_DRAFT_CONVOLUTION_STAGES, SPLASH_DRAFT_CONVOLUTION_TAPS, hidden});
    b.projection({{name + ".kernel_projection", layout.dynamicSize}}, hidden);
  };
  b.stored(prefix + "input_layernorm.weight", {hidden});
  if (!layout.isDSpine()) convolution(prefix + "attention_conv");
  b.projection({{attention + "q_proj", layout.attentionSize}, {attention + "k_proj", kv}, {attention + "v_proj", kv}},
               hidden);
  b.stored(attention + "q_norm.weight", {layout.attentionHeadDimension});
  b.stored(attention + "k_norm.weight", {layout.attentionHeadDimension});
  b.projection({{attention + "o_proj", hidden}}, layout.attentionSize);
  b.stored(prefix + "post_attention_layernorm.weight", {hidden});
  if (!layout.isDSpine()) convolution(prefix + "mlp_conv");
  b.projection({{prefix + "mlp.gate_proj", layout.intermediateSize}}, hidden);
  b.projection({{prefix + "mlp.up_proj", layout.intermediateSize}}, hidden);
  b.projection({{prefix + "mlp.down_proj", hidden}}, layout.intermediateSize);
  return b.finish();
}

gguf::Image modelImage(const SafetensorsCheckpoint &checkpoint, const DFlashDraftLayout &layout) {
  Builder b(checkpoint, "model.bin", layout.layers, 1);
  const std::string selector = "candidate_selector.";
  const uint64_t hidden = layout.hiddenSize;
  b.projection({{"fc", hidden}}, layout.targetHiddenSize());
  b.stored("hidden_norm.weight", {hidden});
  b.stored("norm.weight", {hidden});
  if (layout.isDSpine()) {
    const std::string bridge = "selected_prefix_bridge.";
    const uint64_t code = layout.bridgeCodeSize, rank = layout.bridgeRank;
    b.stored(bridge + "early_read_weights", {layout.layers - 1, code, hidden});
    b.stored(bridge + "read_weight", {code, hidden});
    b.stored(bridge + "feature_write_weights", {layout.layers, hidden, rank});
    b.stacked({bridge + "gate.weight", bridge + "predecessor_gate.weight", bridge + "prev.weight"}, rank, code);
    b.stored(bridge + "gate.bias", {rank});
    b.stored(bridge + "codebook", {layout.vocabularySize, code});
  } else {
    b.projection({{selector + "hidden_projection", layout.selectorRank}}, hidden);
    b.stored(selector + "predecessor_codebook", {layout.vocabularySize, layout.selectorRank});
    b.stored(selector + "successor_codebook", {layout.vocabularySize, layout.selectorRank});
  }
  if (checkpoint.find(kDraftMaskEmbedding)) b.stored(std::string(kDraftMaskEmbedding), {hidden});
  return b.finish();
}

} // namespace

std::vector<gguf::Image> planDraftImages(const SafetensorsCheckpoint &checkpoint, const DFlashDraftLayout &layout) {
  if (layout.isDSpine()) {
    for (const char *name : {"draft_block_causal_version", "last_write_refine_version",
                           "selected_prefix_bridge.independent_write_version", "selected_prefix_bridge.terminal_disabled_version"}) {
      const auto &tensor = checkpoint.require(name);
      int64_t version = 0;
      if (tensor.dtype != "I64" || !tensor.shape.empty() || tensor.bytes != sizeof(version))
        throw WeightStoreError(std::string("invalid DSpine version tensor: ") + name);
      tensor.read(0, {reinterpret_cast<uint8_t *>(&version), sizeof(version)});
      if (version != 1) throw WeightStoreError(std::string("unsupported DSpine version: ") + name);
    }
    const auto &terminal = checkpoint.require("selected_prefix_bridge.out.weight");
    if (terminal.dtype != "BF16" || terminal.shape != std::vector<uint64_t>{layout.bridgeCodeSize, layout.bridgeRank})
      throw WeightStoreError("invalid DSpine terminal projection");
    std::array<uint16_t, 4096> values{};
    for (uint64_t at = 0; at < terminal.bytes; at += sizeof(values)) {
      const auto bytes = static_cast<size_t>(std::min<uint64_t>(sizeof(values), terminal.bytes - at));
      terminal.read(at, {reinterpret_cast<uint8_t *>(values.data()), bytes});
      for (size_t i = 0; i < bytes / 2; ++i)
        if (values[i] & 0x7fffu) throw WeightStoreError("DSpine terminal projection is not disabled");
    }
  }
  std::vector<gguf::Image> images;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) images.push_back(layerImage(checkpoint, layout, layer));
  images.push_back(modelImage(checkpoint, layout));
  return images;
}

uint64_t draftImageBytes(const std::filesystem::path &directory, const DFlashDraftLayout &layout) {
  const SafetensorsCheckpoint checkpoint(directory);
  uint64_t bytes = 0;
  for (const gguf::Image &image : planDraftImages(checkpoint, layout)) bytes += image.bytes;
  return bytes;
}

DraftCheckpointLoader::DraftCheckpointLoader(metal::MetalBackend &backend, WeightImages &images,
                                             const std::filesystem::path &directory, const DFlashDraftLayout &layout)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(directory)) {
  planned_->images = planDraftImages(planned_->checkpoint, layout);
}

WeightFile DraftCheckpointLoader::open(size_t index) {
  return loadImage(backend_, images_, "draft/", {planned_, &planned_->images[index]},
                   [planned = planned_] { planned->checkpoint.checkUnchanged(); });
}

WeightFile DraftCheckpointLoader::layer(uint32_t index) {
  if (index >= planned_->images.size() - 1) throw WeightStoreError("draft layer is out of range");
  return open(index);
}

WeightFile DraftCheckpointLoader::model() { return open(planned_->images.size() - 1); }

bool DraftCheckpointLoader::hasMaskEmbedding() const noexcept {
  return planned_->checkpoint.find(kDraftMaskEmbedding) != nullptr;
}

} // namespace splash::model
