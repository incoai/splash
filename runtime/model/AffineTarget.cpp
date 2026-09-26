#include "model/AffineTarget.hpp"
#include "model/AffinePlan.hpp"
#include "model/Qwen3_8.hpp"
#include "model/Qwen3_6Moe.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/StateLayout.hpp"
#include "model/WeightLayout.hpp"

namespace splash::model {
namespace {

using affine::append;
using affine::copy;
using affine::image;
using affine::Image;
using affine::ProjectionPart;
using affine::Section;
using affine::SectionKind;

// Projection parts, stacked in row order, padded with zero rows to `rows`.
void projection(Image &image, std::initializer_list<std::pair<std::string, uint32_t>> parts,
                uint32_t rows, uint32_t columns, uint32_t bits = 4, uint32_t experts = 1) {
  validateQ4Layout(rows, columns);
  Section section;
  section.kind = SectionKind::Projection;
  section.rows = rows;
  section.columns = columns;
  section.bits = bits;
  section.experts = experts;
  section.bytes = checkedWeightMultiply(uint64_t(rows) * columns * bits / 8 +
                                        uint64_t(rows) * columns / 16, experts, "affine projection");
  uint64_t sourceRows = 0;
  for (const auto &[name, count] : parts) {
    image.quantized.emplace_back(name, bits);
    ProjectionPart part{count, {}};
    for (size_t field = 0; field < 3; ++field) { // weight, scales, biases
      std::vector<uint64_t> shape{count, field ? columns / kQ4GroupElements : columns * bits / 32};
      if (experts > 1) shape.insert(shape.begin(), experts);
      part.fields.push_back({name + (field == 0 ? ".weight" : field == 1 ? ".scales" : ".biases"),
                             {field ? "BF16" : "U32"}, std::move(shape)});
    }
    sourceRows += count;
    section.parts.push_back(std::move(part));
  }
  if (!experts || sourceRows > rows || rows - sourceRows >= kQ4StorageN)
    throw WeightStoreError("invalid affine projection padding");
  append(image, std::move(section));
}

template<class Layout>
void validateConfiguration(const SafetensorsCheckpoint &source, const Layout &layout) {
  const std::pair<const char *, double> fields[] = {
      {"num_hidden_layers", layout.layers}, {"hidden_size", layout.hiddenSize},
      {"vocab_size", layout.vocabularySize}, {"head_dim", layout.attentionHeadDimension},
      {"num_attention_heads", layout.attentionQueryHeads}, {"num_key_value_heads", layout.attentionKvHeads},
      {"linear_num_key_heads", layout.gdnKeyHeads}, {"linear_num_value_heads", layout.gdnValueHeads},
      {"linear_key_head_dim", layout.gdnHeadDimension}, {"linear_value_head_dim", layout.gdnHeadDimension},
      {"linear_conv_kernel_dim", kGdnConvolutionTaps}, {"full_attention_interval", layout.fullAttentionPeriod},
      {"rms_norm_eps", 1e-6}, {"attention_bias", 0}, {"attn_output_gate", 1},
      {"tie_word_embeddings", 0}, {"rope_parameters.rope_theta", layout.rotaryTheta},
      {"rope_parameters.partial_rotary_factor", double(layout.rotaryPairs * 2) / layout.attentionHeadDimension}};
  for (const auto &[key, value] : fields) source.requireConfigNumber(key, value);
  source.requireConfigString("hidden_act", "silu");
  source.requireConfigString("rope_parameters.rope_type", "default");
  source.requireLayerTypes(layout.layers, layout.fullAttentionPeriod);
  if constexpr (Layout::ffnKind == QwenFfnKind::SparseMoe) {
    source.requireConfigString("model_type", "qwen3_5_moe_text");
    source.requireConfigNumber("num_experts", layout.experts);
    source.requireConfigNumber("num_experts_per_tok", layout.expertsPerToken);
    source.requireConfigNumber("moe_intermediate_size", layout.expertIntermediateSize);
    source.requireConfigNumber("shared_expert_intermediate_size", layout.expertIntermediateSize);
  } else {
    source.requireConfigString("model_type", "qwen3_5_text");
    source.requireConfigNumber("intermediate_size", layout.intermediateSize);
  }
}

template<class Layout>
Image layerImage(const Layout &layout, uint32_t layer) {
  const bool full = layout.isFullAttentionLayer(layer);
  Image result = image("layer-" + std::to_string(layer) + ".bin", Layout::layerMagic, layer, full ? 1u : 0u);
  const std::string prefix = "language_model.model.layers." + std::to_string(layer) + ".";
  copy(result, prefix + "input_layernorm.weight", {layout.hiddenSize});
  if (full) {
    const std::string attention = prefix + "self_attn.";
    projection(result, {{attention + "q_proj", 2 * layout.attentionWidth},
                                {attention + "k_proj", layout.attentionKvHeads * layout.attentionHeadDimension},
                                {attention + "v_proj", layout.attentionKvHeads * layout.attentionHeadDimension}},
               layout.packedFullWidth, layout.hiddenSize);
    copy(result, attention + "q_norm.weight", {layout.attentionHeadDimension});
    copy(result, attention + "k_norm.weight", {layout.attentionHeadDimension});
    projection(result, {{attention + "o_proj", layout.hiddenSize}}, layout.hiddenSize, layout.attentionWidth);
  } else {
    const std::string gdn = prefix + "linear_attn.";
    projection(result, {{gdn + "in_proj_qkv", layout.convolutionDimension},
                                {gdn + "in_proj_z", layout.attentionWidth},
                                {gdn + "in_proj_b", layout.gdnValueHeads},
                                {gdn + "in_proj_a", layout.gdnValueHeads}},
               layout.packedGdnWidth, layout.hiddenSize);
    copy(result, gdn + "conv1d.weight", {layout.convolutionDimension, kGdnConvolutionTaps, 1});
    Section decay;
    decay.kind = SectionKind::Decay;
    decay.input = {gdn + "A_log", {"BF16", "F32"}, {layout.gdnValueHeads}};
    decay.bytes = uint64_t(layout.gdnValueHeads) * sizeof(float);
    append(result, std::move(decay));
    copy(result, gdn + "dt_bias", {layout.gdnValueHeads});
    copy(result, gdn + "norm.weight", {layout.gdnHeadDimension});
    projection(result, {{gdn + "out_proj", layout.hiddenSize}}, layout.hiddenSize, layout.attentionWidth);
  }
  copy(result, prefix + "post_attention_layernorm.weight", {layout.hiddenSize});
  const std::string mlp = prefix + "mlp.";
  const auto ffn = [&](const std::string &name, uint32_t intermediate, uint32_t experts = 1) {
    for (const std::string projectionName : {"gate_proj", "up_proj", "down_proj"}) {
      const bool down = projectionName == "down_proj";
      const uint32_t n = down ? layout.hiddenSize : intermediate;
      const uint32_t k = down ? intermediate : layout.hiddenSize;
      projection(result, {{name + projectionName, n}}, n, k, 4, experts);
    }
  };
  if constexpr (Layout::ffnKind == QwenFfnKind::SparseMoe) {
    // The router and the shared-expert scalar gate are 8-bit, their rows padded to
    // whole 256-row tiles as the reader expects.
    projection(result, {{mlp + "gate", layout.experts}}, layout.experts, layout.hiddenSize, 8);
    ffn(mlp + "switch_mlp.", layout.expertIntermediateSize, layout.experts);
    ffn(mlp + "shared_expert.", layout.expertIntermediateSize);
    projection(result, {{mlp + "shared_expert_gate", 1}}, kQ4StorageN, layout.hiddenSize, 8);
  } else {
    ffn(mlp, layout.intermediateSize);
  }
  return result;
}

template<class Layout>
Image headImage(const Layout &layout) {
  Image result = image("head.bin", Layout::headMagic, layout.layers, 2);
  copy(result, "language_model.model.norm.weight", {layout.hiddenSize});
  projection(result, {{"language_model.lm_head", layout.vocabularySize}}, layout.vocabularySize, layout.hiddenSize);
  return result;
}

// The token rows as stored: 4-bit codes, scales and biases.
template<class Layout>
Image embeddingImage(const Layout &layout) {
  Image result = image("embedding.bin", kEmbeddingMagic, layout.vocabularySize, layout.hiddenSize);
  const std::string prefix = "language_model.model.embed_tokens";
  result.quantized.emplace_back(prefix, 4);
  copy(result, prefix + ".weight", {layout.vocabularySize, layout.hiddenSize / 8}, "U32");
  copy(result, prefix + ".scales", {layout.vocabularySize, layout.hiddenSize / kQ4GroupElements});
  copy(result, prefix + ".biases", {layout.vocabularySize, layout.hiddenSize / kQ4GroupElements});
  return result;
}

// Every image of a layout: the layers, the head, the embedding.
template<class Layout>
std::vector<Image> images(const Layout &layout) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) result.push_back(layerImage(layout, layer));
  result.push_back(headImage(layout));
  result.push_back(embeddingImage(layout));
  return result;
}

template<class Layout>
uint64_t preparedBytes(const Layout &layout) {
  uint64_t bytes = 0;
  for (const Image &image : images(layout)) bytes += image.bytes;
  return bytes;
}

} // namespace

struct AffineTargetLoader::Impl {
  metal::MetalBackend &backend;
  SafetensorsCheckpoint source;
  std::vector<Image> images; // layers, head, embedding
  std::vector<PreparedWeight> weights;
  PreparedFiles files;
  template<class Layout>
  Impl(metal::MetalBackend &backend, const std::filesystem::path &directory, const Layout &layout,
       PreparationCheck admitConversion)
      : backend(backend), source(directory, [&backend] { backend.checkOperation(); }),
        files([&backend] { backend.checkOperation(); }, std::move(admitConversion),
              [this] { source.checkUnchanged(); }) {
    validateConfiguration(source, layout);
    images = model::images(layout);
    for (Image &image : images) {
      backend.checkOperation();
      affine::bind(image, source);
      weights.push_back(affine::affineImageWeight(image, "target", directory.string()));
    }
  }
  WeightFile open(size_t index) {
    const Image &image = images[index];
    return files.open(backend, weights[index], affine::affineImageWriter(image), image.magic, image.layer,
                      image.type);
  }
};
AffineTargetLoader::AffineTargetLoader(metal::MetalBackend &backend, const std::filesystem::path &directory,
                                       const Qwen3_8Layout &layout, PreparationCheck admitConversion)
    : impl_(std::make_unique<Impl>(backend, directory, layout, std::move(admitConversion))) {}
AffineTargetLoader::AffineTargetLoader(metal::MetalBackend &backend, const std::filesystem::path &directory,
                                       const Qwen3_6MoeLayout &layout, PreparationCheck admitConversion)
    : impl_(std::make_unique<Impl>(backend, directory, layout, std::move(admitConversion))) {}
AffineTargetLoader::~AffineTargetLoader() = default;
std::span<const PreparedWeight> AffineTargetLoader::weights() const noexcept { return impl_->weights; }
void AffineTargetLoader::prepare() {
  for (size_t index = 0; index < impl_->images.size(); ++index)
    static_cast<void>(impl_->files.prepare(impl_->weights[index], affine::affineImageWriter(impl_->images[index])));
}
WeightFile AffineTargetLoader::layer(uint32_t index) {
  if (index >= impl_->images.size() - 2) throw WeightStoreError("target layer is out of range");
  return impl_->open(index);
}
WeightFile AffineTargetLoader::head() { return impl_->open(impl_->images.size() - 2); }
WeightFile AffineTargetLoader::embedding() { return impl_->open(impl_->images.size() - 1); }

uint64_t preparedAffineBytes(const Qwen3_8Layout &layout) { return preparedBytes(layout); }
uint64_t preparedAffineBytes(const Qwen3_6MoeLayout &layout) { return preparedBytes(layout); }
Image affineLayerImage(const Qwen3_8Layout &layout, uint32_t layer) { return layerImage(layout, layer); }
Image affineLayerImage(const Qwen3_6MoeLayout &layout, uint32_t layer) { return layerImage(layout, layer); }

} // namespace splash::model
