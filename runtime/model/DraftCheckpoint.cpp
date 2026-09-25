#include "model/DraftCheckpoint.hpp"
#include "model/AffinePlan.hpp"
#include "model/DFlashDraft.hpp"

#include <string>
#include <utility>
#include <vector>

namespace splash::model {
namespace {

using affine::copy;
using affine::Image;

// A projection's BF16 parts, stacked in row order, which the writer quantizes;
// they fill its rows.
void quantized(Image &image, std::initializer_list<std::pair<std::string, uint32_t>> parts, uint32_t rows,
               uint32_t columns) {
  validateQ4Layout(rows, columns);
  affine::Section section;
  section.kind = affine::SectionKind::Quantize;
  section.rows = rows;
  section.columns = columns;
  section.bytes = uint64_t(rows) * columns / 2 + uint64_t(rows) * columns / 16;
  uint32_t sourceRows = 0;
  for (const auto &[name, count] : parts) {
    affine::ProjectionPart part{count, {}};
    part.fields.push_back({name + ".weight", {"BF16"}, {count, columns}});
    section.parts.push_back(std::move(part));
    sourceRows += count;
  }
  if (sourceRows != rows) throw WeightStoreError("draft projection parts do not fill its rows");
  affine::append(image, std::move(section));
}

// The sections of each file in the order DFlashDraft.cpp reads them.
Image layerImage(const DFlashDraftLayout &layout, uint32_t layer) {
  Image result = affine::image("layer-" + std::to_string(layer) + ".bin", kDFlashLayerMagic, layer, 0);
  const std::string prefix = "layers." + std::to_string(layer) + ".";
  const std::string attention = prefix + "self_attn.";
  const uint32_t hidden = layout.hiddenSize;
  const uint32_t kv = layout.kvHeads * layout.attentionHeadDimension;
  const auto convolution = [&](const std::string &name) {
    copy(result, name + ".base_kernel", {2, 2, hidden});
    quantized(result, {{name + ".kernel_projection", layout.dynamicSize}}, layout.dynamicSize, hidden);
  };
  copy(result, prefix + "input_layernorm.weight", {hidden});
  convolution(prefix + "attention_conv");
  quantized(result, {{attention + "q_proj", layout.attentionSize}, {attention + "k_proj", kv}, {attention + "v_proj", kv}},
            layout.qkvSize, hidden);
  copy(result, attention + "q_norm.weight", {layout.attentionHeadDimension});
  copy(result, attention + "k_norm.weight", {layout.attentionHeadDimension});
  quantized(result, {{attention + "o_proj", hidden}}, hidden, layout.attentionSize);
  copy(result, prefix + "post_attention_layernorm.weight", {hidden});
  convolution(prefix + "mlp_conv");
  quantized(result, {{prefix + "mlp.gate_proj", layout.intermediateSize}}, layout.intermediateSize, hidden);
  quantized(result, {{prefix + "mlp.up_proj", layout.intermediateSize}}, layout.intermediateSize, hidden);
  quantized(result, {{prefix + "mlp.down_proj", hidden}}, hidden, layout.intermediateSize);
  return result;
}

Image modelImage(const DFlashDraftLayout &layout) {
  Image result = affine::image("model.bin", kDFlashLayerMagic, layout.layers, 1);
  const std::string selector = "candidate_selector.";
  quantized(result, {{"fc", layout.hiddenSize}}, layout.hiddenSize, layout.targetHiddenSize);
  copy(result, "hidden_norm.weight", {layout.hiddenSize});
  copy(result, "norm.weight", {layout.hiddenSize});
  quantized(result, {{selector + "hidden_projection", layout.selectorRank}}, layout.selectorRank, layout.hiddenSize);
  copy(result, selector + "predecessor_codebook", {layout.vocabularySize, layout.selectorRank});
  copy(result, selector + "successor_codebook", {layout.vocabularySize, layout.selectorRank});
  return result;
}

// Every file of a layout: the layers, then model.bin.
std::vector<Image> draftImages(const DFlashDraftLayout &layout) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) result.push_back(layerImage(layout, layer));
  result.push_back(modelImage(layout));
  return result;
}

} // namespace

struct DraftCheckpointLoader::Impl {
  metal::MetalBackend &backend;
  SafetensorsCheckpoint source;
  std::vector<Image> images; // layers, then model.bin
  std::vector<PreparedWeight> weights;
  PreparedFiles files;
  Impl(metal::MetalBackend &backend, const std::filesystem::path &directory, const DFlashDraftLayout &layout,
       PreparationCheck admitConversion)
      : backend(backend), source(directory, [&backend] { backend.checkOperation(); }),
        files([&backend] { backend.checkOperation(); }, std::move(admitConversion),
              [this] { source.checkUnchanged(); }) {
    images = draftImages(layout);
    for (Image &image : images) {
      backend.checkOperation();
      affine::bind(image, source);
      weights.push_back(affine::affineImageWeight(image, "draft", directory.string()));
    }
  }
  WeightFile open(size_t index) {
    const Image &image = images[index];
    return files.open(backend, weights[index], affine::affineImageWriter(image), image.magic, image.layer,
                      image.type);
  }
};
DraftCheckpointLoader::DraftCheckpointLoader(metal::MetalBackend &backend, const std::filesystem::path &directory,
                                             const DFlashDraftLayout &layout, PreparationCheck admitConversion)
    : impl_(std::make_unique<Impl>(backend, directory, layout, std::move(admitConversion))) {}
DraftCheckpointLoader::~DraftCheckpointLoader() = default;
std::span<const PreparedWeight> DraftCheckpointLoader::weights() const noexcept { return impl_->weights; }
void DraftCheckpointLoader::prepare() {
  for (size_t index = 0; index < impl_->images.size(); ++index)
    static_cast<void>(impl_->files.prepare(impl_->weights[index], affine::affineImageWriter(impl_->images[index])));
}
WeightFile DraftCheckpointLoader::layer(uint32_t index) {
  if (index >= impl_->images.size() - 1) throw WeightStoreError("draft layer is out of range");
  return impl_->open(index);
}
WeightFile DraftCheckpointLoader::model() { return impl_->open(impl_->images.size() - 1); }

uint64_t preparedDraftBytes(const DFlashDraftLayout &layout) {
  uint64_t bytes = 0;
  for (const Image &image : draftImages(layout)) bytes += image.bytes;
  return bytes;
}

} // namespace splash::model
