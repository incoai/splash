#include "omni/OmniLoader.hpp"

#include "Checked.hpp"
#include "model/AffinePlan.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightStore.hpp"
#include "ops/Omni.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace splash::omni {
namespace {

using model::WeightFile;
using model::WeightStoreError;
using model::affine::Image;
using model::affine::Section;
using model::affine::SectionKind;

constexpr uint32_t kQkvWidth = OMNI_DECODER_QKV_WIDTH;
constexpr uint32_t kAttentionWidth = OMNI_DECODER_ATTENTION_WIDTH;
constexpr uint32_t kHeadDim = OMNI_DECODER_HEAD_DIM;

// One image's entries, in file order, planned into its sections and read
// back from its file in the same order.
struct Entry final {
  enum class Kind { Q4Projection, DenseMatrix, Norm, Bias, Q4Table, Bf16Table } kind;
  std::string label;
  std::vector<std::pair<std::string, uint32_t>> parts; // Q4Projection: part name and rows
  std::string name;                                    // the others
  uint32_t rows = 0, columns = 0;
};

class ImageSpec final {
public:
  ImageSpec(std::string name, std::string_view magic, uint32_t layer, uint32_t type, bool quantized)
      : image_(model::affine::image(std::move(name), magic, layer, type)), quantized_(quantized) {}

  // A projection of `rows` output rows, fused from `parts` in row order:
  // affine Q4 when the checkpoint is quantized (rows padded with zero rows to
  // whole 256-row tiles), bf16 otherwise.
  void projection(std::string label, std::vector<std::pair<std::string, uint32_t>> parts, uint32_t rows,
                  uint32_t columns) {
    if (quantized_) {
      q4(std::move(label), std::move(parts), rows, columns);
      return;
    }
    if (parts.size() != 1 || parts.front().second != rows)
      throw WeightStoreError("bf16 projections are not fused: " + label);
    dense(std::move(label), parts.front().first, rows, columns);
  }
  void dense(std::string label, const std::string &name, uint32_t rows, uint32_t columns) {
    model::affine::copy(image_, name + ".weight", {rows, columns});
    entries_.push_back({Entry::Kind::DenseMatrix, std::move(label), {}, name, rows, columns});
  }
  void table(std::string label, const std::string &name, uint32_t rows, uint32_t columns) {
    model::affine::copy(image_, name + ".weight", {rows, columns});
    entries_.push_back({Entry::Kind::Bf16Table, std::move(label), {}, name, rows, columns});
  }
  void q4Table(std::string label, const std::string &name, uint32_t rows, uint32_t columns) {
    image_.quantized.emplace_back(name, 4);
    model::affine::copy(image_, name + ".weight", {rows, columns / 8}, "U32");
    model::affine::copy(image_, name + ".scales", {rows, columns / model::kQ4GroupElements});
    model::affine::copy(image_, name + ".biases", {rows, columns / model::kQ4GroupElements});
    entries_.push_back({Entry::Kind::Q4Table, std::move(label), {}, name, rows, columns});
  }
  void norm(std::string label, const std::string &name, uint32_t width) {
    model::affine::copy(image_, name, {width});
    entries_.push_back({Entry::Kind::Norm, std::move(label), {}, name, width, 0});
  }
  void bias(std::string label, const std::string &name, uint32_t width) {
    model::affine::copy(image_, name, {width});
    entries_.push_back({Entry::Kind::Bias, std::move(label), {}, name, width, 0});
  }
  [[nodiscard]] Image &image() noexcept { return image_; }
  [[nodiscard]] const Image &image() const noexcept { return image_; }
  [[nodiscard]] const std::vector<Entry> &entries() const noexcept { return entries_; }

private:
  void q4(std::string label, std::vector<std::pair<std::string, uint32_t>> parts, uint32_t rows, uint32_t columns) {
    const uint32_t padded = (rows + model::kQ4StorageN - 1) / model::kQ4StorageN * model::kQ4StorageN;
    model::validateQ4Layout(padded, columns);
    Section section;
    section.kind = SectionKind::Projection;
    section.rows = padded;
    section.columns = columns;
    section.bits = 4;
    section.experts = 1;
    section.bytes = checkedMultiply<WeightStoreError>(uint64_t(padded) * columns * 4 / 8 + uint64_t(padded) * columns / 16, 1,
                                                      "affine projection");
    uint64_t sourceRows = 0;
    for (const auto &[name, count] : parts) {
      image_.quantized.emplace_back(name, 4);
      model::affine::ProjectionPart part{count, {}};
      part.fields.push_back({name + ".weight", {"U32"}, {count, columns * 4 / 32}});
      part.fields.push_back({name + ".scales", {"BF16"}, {count, columns / model::kQ4GroupElements}});
      part.fields.push_back({name + ".biases", {"BF16"}, {count, columns / model::kQ4GroupElements}});
      sourceRows += count;
      section.parts.push_back(std::move(part));
    }
    if (sourceRows != rows || padded - sourceRows >= model::kQ4StorageN) throw WeightStoreError("invalid affine projection padding: " + label);
    model::affine::append(image_, std::move(section));
    entries_.push_back({Entry::Kind::Q4Projection, std::move(label), std::move(parts), {}, padded, columns});
  }

  Image image_;
  std::vector<Entry> entries_;
  bool quantized_;
};

// Reads an image's entries in order.
class ImageReader final {
public:
  ImageReader(WeightFile &file, const ImageSpec &spec) : file_(&file), spec_(&spec) {}

  [[nodiscard]] Qwen3Projection projection(std::string_view label) {
    const Entry &entry = next(label);
    if (entry.kind == Entry::Kind::Q4Projection)
      return model::readAffineProjection(*file_, entry.rows, entry.columns, label);
    if (entry.kind != Entry::Kind::DenseMatrix) throw WeightStoreError("entry is not a projection: " + std::string(label));
    return ops::omni::DenseWeights{file_->section(uint64_t{entry.rows} * entry.columns * 2, label), {}, entry.rows,
                                   entry.columns};
  }
  [[nodiscard]] metal::MetalBuffer table(std::string_view label) {
    const Entry &entry = next(label);
    if (entry.kind != Entry::Kind::Bf16Table && entry.kind != Entry::Kind::DenseMatrix)
      throw WeightStoreError("entry is not a table: " + std::string(label));
    return file_->section(uint64_t{entry.rows} * entry.columns * 2, label);
  }
  [[nodiscard]] ops::EmbeddingWeights q4Table(std::string_view label) {
    const Entry &entry = next(label);
    if (entry.kind != Entry::Kind::Q4Table) throw WeightStoreError("entry is not a Q4 table: " + std::string(label));
    return model::readAffineEmbedding(*file_, entry.rows, entry.columns, label);
  }
  [[nodiscard]] ops::NormWeights norm(std::string_view label) {
    const Entry &entry = next(label);
    if (entry.kind != Entry::Kind::Norm) throw WeightStoreError("entry is not a norm: " + std::string(label));
    return model::readNorm(*file_, entry.rows, label);
  }
  // The bf16 bias as stored; the loader converts it to fp32 after the load.
  [[nodiscard]] metal::MetalBuffer bias(std::string_view label) {
    const Entry &entry = next(label);
    if (entry.kind != Entry::Kind::Bias) throw WeightStoreError("entry is not a bias: " + std::string(label));
    return file_->section(uint64_t{entry.rows} * 2, label);
  }
  void finish() {
    if (index_ != spec_->entries().size()) throw WeightStoreError("image entries were not all read: " + spec_->image().name);
    file_->finish();
  }

private:
  const Entry &next(std::string_view label) {
    if (index_ >= spec_->entries().size()) throw WeightStoreError("read past the image's entries: " + std::string(label));
    const Entry &entry = spec_->entries()[index_++];
    if (entry.label != label) throw WeightStoreError("image entry order: expected " + entry.label + ", read " + std::string(label));
    return entry;
  }
  WeightFile *file_;
  const ImageSpec *spec_;
  size_t index_ = 0;
};

// The images of one checkpoint directory, loaded one by one; biases are
// converted once every image is loaded.
class Loader final {
public:
  Loader(metal::MetalBackend &backend, model::WeightImages &images, const std::filesystem::path &directory,
         std::string component)
      : backend_(&backend), images_(&images), planned_(std::make_shared<model::affine::PlannedCheckpoint>(directory)),
        component_(std::move(component)) {}

  // Plans, loads and reads one image through `read`.
  template <class Read> void load(ImageSpec spec, Read read) {
    model::affine::bind(spec.image(), planned_->source);
    planned_->images.push_back(spec.image());
    const size_t index = planned_->images.size() - 1;
    WeightFile file = images_->load(model::affine::imagePlan(planned_, index, component_));
    ImageReader reader(file, spec);
    read(reader);
    reader.finish();
  }
  // An fp32 copy of a bf16 bias, converted on the GPU when finish() runs.
  [[nodiscard]] metal::MetalBuffer float32(metal::MetalBuffer bias, uint32_t width, std::string_view label) {
    metal::MetalBuffer converted = backend_->allocateBuffer(uint64_t{width} * 4, metal::BufferStorage::Shared, label);
    ops::omni::Norms::addConvert(conversions_, std::move(bias), converted, width, true);
    return converted;
  }
  // `names` bf16 tables of [rows][columns], read as stored one after another
  // into one shared buffer.
  [[nodiscard]] metal::MetalBuffer directTables(const std::vector<std::string> &names, uint32_t rows, uint32_t columns,
                                                std::string_view label) {
    const uint64_t tableBytes = uint64_t{rows} * columns * 2;
    metal::MetalBuffer buffer = backend_->allocateBuffer(tableBytes * names.size(), metal::BufferStorage::Shared, label);
    auto *destination = static_cast<uint8_t *>(buffer.contents());
    for (size_t i = 0; i < names.size(); ++i) {
      const model::SourceTensor &tensor = planned_->source.require(names[i]);
      if (tensor.dtype != "BF16" || tensor.shape != std::vector<uint64_t>{rows, columns} || tensor.bytes != tableBytes)
        throw WeightStoreError("table type or shape does not match: " + names[i]);
      tensor.read(0, {destination + i * tableBytes, tableBytes});
    }
    planned_->source.checkUnchanged();
    return buffer;
  }
  void finish() {
    if (conversions_.empty()) return;
    metal::CommandTicket ticket = backend_->submitCommandAsync(conversions_.command());
    (void)ticket.wait();
  }

private:
  metal::MetalBackend *backend_;
  model::WeightImages *images_;
  std::shared_ptr<model::affine::PlannedCheckpoint> planned_;
  std::string component_;
  metal::CommandGraph conversions_;
};

// The Qwen3 decoder layer `layer` under `prefix` (ending in a dot).
ImageSpec layerSpec(const std::string &component, std::string_view magic, const std::string &prefix, uint32_t layer,
                    const Qwen3Dims &dims, bool quantized) {
  ImageSpec spec("layer-" + std::to_string(layer) + ".bin", magic, layer, 0, quantized);
  const std::string p = prefix + "layers." + std::to_string(layer) + ".";
  spec.norm("input_norm", p + "input_layernorm.weight", dims.hidden);
  if (quantized) {
    spec.projection("qkv", {{p + "self_attn.q_proj", kAttentionWidth}, {p + "self_attn.k_proj", 8 * kHeadDim},
                            {p + "self_attn.v_proj", 8 * kHeadDim}}, kQkvWidth, dims.hidden);
  } else {
    spec.dense("q", p + "self_attn.q_proj", kAttentionWidth, dims.hidden);
    spec.dense("k", p + "self_attn.k_proj", 8 * kHeadDim, dims.hidden);
    spec.dense("v", p + "self_attn.v_proj", 8 * kHeadDim, dims.hidden);
  }
  spec.norm("q_norm", p + "self_attn.q_norm.weight", kHeadDim);
  spec.norm("k_norm", p + "self_attn.k_norm.weight", kHeadDim);
  spec.projection("o", {{p + "self_attn.o_proj", dims.hidden}}, dims.hidden, kAttentionWidth);
  spec.norm("post_norm", p + "post_attention_layernorm.weight", dims.hidden);
  spec.projection("gate", {{p + "mlp.gate_proj", dims.intermediate}}, dims.intermediate, dims.hidden);
  spec.projection("up", {{p + "mlp.up_proj", dims.intermediate}}, dims.intermediate, dims.hidden);
  spec.projection("down", {{p + "mlp.down_proj", dims.hidden}}, dims.hidden, dims.intermediate);
  (void)component;
  return spec;
}

// A bf16 checkpoint stores q, k and v apart; the kernels read one fused
// [4096][hidden] matrix, so the three are copied into one buffer.
Qwen3Projection fuseQkv(metal::MetalBackend &backend, const ops::omni::DenseWeights &q,
                        const ops::omni::DenseWeights &k, const ops::omni::DenseWeights &v, uint32_t hidden) {
  metal::MetalBuffer fused = backend.allocateBuffer(uint64_t{kQkvWidth} * hidden * 2, metal::BufferStorage::Shared, "qkv");
  auto *destination = static_cast<uint8_t *>(fused.contents());
  const uint64_t qBytes = uint64_t{kAttentionWidth} * hidden * 2, kvBytes = uint64_t{8 * kHeadDim} * hidden * 2;
  std::memcpy(destination, q.weight.contents(), qBytes);
  std::memcpy(destination + qBytes, k.weight.contents(), kvBytes);
  std::memcpy(destination + qBytes + kvBytes, v.weight.contents(), kvBytes);
  return ops::omni::DenseWeights{fused, {}, kQkvWidth, hidden};
}

Qwen3LayerWeights readLayer(metal::MetalBackend &backend, ImageReader &reader, const Qwen3Dims &dims, bool quantized) {
  Qwen3LayerWeights w;
  w.inputNorm = reader.norm("input_norm");
  if (quantized) {
    w.qkv = reader.projection("qkv");
  } else {
    const auto q = std::get<ops::omni::DenseWeights>(reader.projection("q"));
    const auto k = std::get<ops::omni::DenseWeights>(reader.projection("k"));
    const auto v = std::get<ops::omni::DenseWeights>(reader.projection("v"));
    w.qkv = fuseQkv(backend, q, k, v, dims.hidden);
  }
  w.queryNorm = reader.norm("q_norm").buffer;
  w.keyNorm = reader.norm("k_norm").buffer;
  w.output = reader.projection("o");
  w.postAttentionNorm = reader.norm("post_norm");
  w.gate = reader.projection("gate");
  w.up = reader.projection("up");
  w.down = reader.projection("down");
  return w;
}

// The bf16 projections of a decoder, its head included.
void addDense(std::vector<ops::omni::DenseWeights *> &dense, Qwen3Projection &projection) {
  if (auto *d = std::get_if<ops::omni::DenseWeights>(&projection)) dense.push_back(d);
}
void addDense(std::vector<ops::omni::DenseWeights *> &dense, Qwen3Weights &weights) {
  for (Qwen3LayerWeights &layer : weights.layers)
    for (Qwen3Projection *p : {&layer.qkv, &layer.output, &layer.gate, &layer.up, &layer.down}) addDense(dense, *p);
  if (weights.head) addDense(dense, *weights.head);
}

// Packs each weight over the load threads in ranges of 64 rows: the ranges'
// escape blocks counted first, so each range numbers its own.
uint64_t packDense(metal::MetalBackend &backend, const std::vector<ops::omni::DenseWeights *> &dense, uint64_t limit) {
  const char *setting = std::getenv("OMNI_PACKED_BF16");
  if (dense.empty() || (setting && std::string_view(setting) == "0")) return 0;
  // The planes without escape records, and an escape in every 256 blocks
  // (the checkpoints have one in about 300).
  uint64_t estimate = 0;
  for (const ops::omni::DenseWeights *d : dense)
    estimate += ops::omni::packedDenseBytes(d->outputSize, d->inputSize, 0) +
                uint64_t{d->outputSize} * d->inputSize / 64 / 256 * OMNI_PACKED_ESCAPE_BYTES;
  if (estimate > limit) return 0;
  uint64_t allocated = 0;
  for (ops::omni::DenseWeights *d : dense) {
    const uint32_t n = d->outputSize, k = d->inputSize, ranges = n / 64;
    const std::span<const uint16_t> weights(static_cast<const uint16_t *>(d->weight.contents()), uint64_t{n} * k);
    if (!weights.data()) throw std::logic_error("bf16 weights are not host visible");
    std::vector<uint32_t> escapes(ranges + 1, 0);
    model::parallelFor(ranges, [&](size_t r, unsigned) {
      escapes[r + 1] = ops::omni::packedEscapeBlocks(weights.subspan(r * 64 * k, uint64_t{64} * k), k);
    });
    for (uint32_t r = 0; r < ranges; ++r) escapes[r + 1] += escapes[r];
    const uint64_t bytes = ops::omni::packedDenseBytes(n, k, escapes[ranges]);
    metal::MetalBuffer packed = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, "packed bf16");
    const std::span<uint8_t> destination(static_cast<uint8_t *>(packed.contents()), bytes);
    model::parallelFor(ranges, [&](size_t r, unsigned) {
      ops::omni::packDenseRows(weights, n, k, static_cast<uint32_t>(r * 64), 64, escapes[r], destination);
    });
    d->packed = std::move(packed);
    allocated += bytes;
  }
  return allocated;
}

void loadDecoderLayers(Loader &loader, metal::MetalBackend &backend, Qwen3Weights &weights, const std::string &component,
                       std::string_view magic, const std::string &prefix, bool quantized) {
  const Qwen3Dims &dims = weights.dims;
  for (uint32_t layer = 0; layer < dims.layers; ++layer)
    loader.load(layerSpec(component, magic, prefix, layer, dims, quantized),
                [&](ImageReader &reader) { weights.layers.push_back(readLayer(backend, reader, dims, quantized)); });
}

} // namespace

TtsWeights loadTtsWeights(metal::MetalBackend &backend, model::WeightImages &images, const std::filesystem::path &root,
                          const TtsConfig &c, bool quantized) {
  TtsWeights w;
  w.talker.dims = c.talker;
  w.predictor.dims = c.predictor;
  w.textEmbedding = HostTable(root, "talker.model.text_embedding.weight", c.textVocabulary, c.textHidden);
  Loader loader(backend, images, root, "tts");
  loadDecoderLayers(loader, backend, w.talker, "tts", "OMTK0001", "talker.model.", quantized);
  {
    ImageSpec spec("talker-head.bin", "OMTH0001", c.talker.layers, 1, quantized);
    spec.norm("final_norm", "talker.model.norm.weight", c.talker.hidden);
    spec.projection("codec_head", {{"talker.codec_head", c.codecVocabulary}}, c.codecVocabulary, c.talker.hidden);
    spec.projection("fc1", {{"talker.text_projection.linear_fc1", c.textHidden}}, c.textHidden, c.textHidden);
    spec.bias("fc1_bias", "talker.text_projection.linear_fc1.bias", c.textHidden);
    spec.projection("fc2", {{"talker.text_projection.linear_fc2", c.talker.hidden}}, c.talker.hidden, c.textHidden);
    spec.bias("fc2_bias", "talker.text_projection.linear_fc2.bias", c.talker.hidden);
    loader.load(std::move(spec), [&](ImageReader &reader) {
      w.talker.finalNorm = reader.norm("final_norm");
      w.talker.head = reader.projection("codec_head");
      w.textFc1 = reader.projection("fc1");
      w.textFc1Bias = loader.float32(reader.bias("fc1_bias"), c.textHidden, "fc1 bias");
      w.textFc2 = reader.projection("fc2");
      w.textFc2Bias = loader.float32(reader.bias("fc2_bias"), c.talker.hidden, "fc2 bias");
    });
  }
  {
    ImageSpec spec("talker-embeddings.bin", "OMTE0001", c.talker.layers, 2, quantized);
    spec.table("codec_embedding", "talker.model.codec_embedding", c.codecVocabulary, c.talker.hidden);
    loader.load(std::move(spec), [&](ImageReader &reader) {
      w.codecEmbedding = reader.table("codec_embedding");
    });
  }
  loadDecoderLayers(loader, backend, w.predictor, "tts", "OMPR0001", "talker.code_predictor.model.", quantized);
  {
    ImageSpec spec("predictor-head.bin", "OMPH0001", c.predictor.layers, 1, quantized);
    spec.norm("final_norm", "talker.code_predictor.model.norm.weight", c.predictor.hidden);
    if (c.predictorProjection()) {
      spec.projection("small_to_mtp", {{"talker.code_predictor.small_to_mtp_projection", c.predictor.hidden}},
                      c.predictor.hidden, c.talker.hidden);
      spec.bias("small_to_mtp_bias", "talker.code_predictor.small_to_mtp_projection.bias", c.predictor.hidden);
    }
    for (uint32_t group = 0; group + 1 < c.codeGroups; ++group)
      spec.projection("lm_head_" + std::to_string(group),
                      {{"talker.code_predictor.lm_head." + std::to_string(group), c.predictorVocabulary}},
                      c.predictorVocabulary, c.predictor.hidden);
    loader.load(std::move(spec), [&](ImageReader &reader) {
      w.predictor.finalNorm = reader.norm("final_norm");
      if (c.predictorProjection()) {
        w.smallToMtp = reader.projection("small_to_mtp");
        w.smallToMtpBias = loader.float32(reader.bias("small_to_mtp_bias"), c.predictor.hidden, "small_to_mtp bias");
      }
      for (uint32_t group = 0; group + 1 < c.codeGroups; ++group)
        w.predictorHeads.push_back(reader.projection("lm_head_" + std::to_string(group)));
    });
  }
  // The 15 predictor tables, read as stored into one buffer the step kernel
  // indexes at a fixed stride (they stay allocated across idle releases).
  {
    std::vector<std::string> names;
    for (uint32_t group = 0; group + 1 < c.codeGroups; ++group)
      names.push_back("talker.code_predictor.model.codec_embedding." + std::to_string(group) + ".weight");
    w.predictorEmbeddings = loader.directTables(names, c.predictorVocabulary, c.talker.hidden, "predictor embeddings");
  }
  loader.finish();
  return w;
}

AsrTextWeights loadAsrTextWeights(metal::MetalBackend &backend, model::WeightImages &images,
                                  const std::filesystem::path &root, const AsrConfig &c, bool quantized) {
  AsrTextWeights w;
  w.decoder.dims = c.decoder;
  Loader loader(backend, images, root, "asr");
  loadDecoderLayers(loader, backend, w.decoder, "asr", "OMAS0001", "model.", quantized);
  {
    ImageSpec spec("text-head.bin", "OMAH0001", c.decoder.layers, 1, quantized);
    spec.norm("final_norm", "model.norm.weight", c.decoder.hidden);
    if (quantized) {
      // The tied embedding twice: its Q4 rows for the gather, and reordered
      // into storage tiles (padded to whole tiles) for the head.
      spec.q4Table("embedding", "model.embed_tokens", c.vocabulary, c.decoder.hidden);
      spec.projection("head", {{"model.embed_tokens", c.vocabulary}}, c.vocabulary, c.decoder.hidden);
    } else {
      spec.dense("embedding", "model.embed_tokens", c.vocabulary, c.decoder.hidden);
    }
    loader.load(std::move(spec), [&](ImageReader &reader) {
      w.decoder.finalNorm = reader.norm("final_norm");
      if (quantized) {
        w.embedding = reader.q4Table("embedding");
        w.decoder.head = reader.projection("head");
      } else {
        const Qwen3Projection table = reader.projection("embedding");
        w.embedding = std::get<ops::omni::DenseWeights>(table).weight;
        w.decoder.head = table;
      }
    });
  }
  loader.finish();
  return w;
}

DFlashDraftWeights loadDFlashDraftWeights(metal::MetalBackend &backend, model::WeightImages &images,
                                          const std::filesystem::path &root, const DFlashDraftConfig &c) {
  DFlashDraftWeights w;
  w.decoder.dims = c.decoder;
  const uint32_t hidden = c.decoder.hidden, features = hidden * static_cast<uint32_t>(c.targetLayers.size());
  Loader loader(backend, images, root, "asr-dflash");
  loadDecoderLayers(loader, backend, w.decoder, "asr-dflash", "OMDF0001", "", c.quantized);
  {
    ImageSpec spec("draft-context.bin", "OMDC0001", c.decoder.layers, 1, c.quantized);
    spec.norm("final_norm", "norm.weight", hidden);
    spec.norm("hidden_norm", "hidden_norm.weight", hidden);
    if (c.quantized) spec.projection("fc", {{"fc", hidden}}, hidden, features);
    else spec.dense("fc", "fc", hidden, features);
    if (c.draftVocabulary) spec.dense("head", "lm_head", c.draftVocabulary, hidden);
    loader.load(std::move(spec), [&](ImageReader &reader) {
      w.decoder.finalNorm = reader.norm("final_norm");
      w.hiddenNorm = reader.norm("hidden_norm");
      w.fc = reader.projection("fc");
      if (c.draftVocabulary) w.decoder.head = reader.projection("head");
    });
  }
  loader.finish();
  if (c.draftVocabulary) {
    // d2t: each draft token's offset to its target token (int64).
    const model::SafetensorsCheckpoint checkpoint(root);
    const model::SourceTensor &d2t = checkpoint.require("d2t");
    if (d2t.dtype != "I64" || d2t.shape != std::vector<uint64_t>{c.draftVocabulary})
      throw std::runtime_error("the DFlash draft's d2t is not int64 [draft_vocab_size]");
    std::vector<int64_t> offsets(c.draftVocabulary);
    d2t.read(0, {reinterpret_cast<uint8_t *>(offsets.data()), offsets.size() * 8});
    for (uint32_t i = 0; i < c.draftVocabulary; ++i) {
      const int64_t token = int64_t{i} + offsets[i];
      if (token < 0 || token > INT32_MAX) throw std::runtime_error("the DFlash draft's d2t maps outside the vocabulary");
      w.tokenMap.push_back(static_cast<uint32_t>(token));
    }
    w.decoder.dims.headRows = w.decoder.dims.vocabulary = c.draftVocabulary;
  }
  return w;
}

HostTable::HostTable(const std::filesystem::path &directory, std::string_view name, uint32_t rows, uint32_t width)
    : rows_(rows), width_(width) {
  const model::SafetensorsCheckpoint checkpoint(directory);
  const model::SourceTensor &tensor = checkpoint.require(name);
  if (tensor.dtype != "BF16" || tensor.shape != std::vector<uint64_t>{rows, width})
    throw std::runtime_error(std::string(name) + " is not bf16 [" + std::to_string(rows) + "][" + std::to_string(width) +
                             "]");
  const int descriptor = ::open(tensor.file->path().c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) throw std::runtime_error("cannot open " + tensor.file->path().string());
  struct stat status {};
  const bool stated = ::fstat(descriptor, &status) == 0;
  const auto bytes = static_cast<size_t>(status.st_size);
  void *address = stated && bytes ? ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, descriptor, 0) : MAP_FAILED;
  ::close(descriptor);
  if (address == MAP_FAILED) throw std::runtime_error("cannot map " + tensor.file->path().string());
  // A request reads scattered rows: read ahead none of their neighbours. The
  // table's pages are read into the file cache in the background, so a
  // first request finds its rows there (about 3 ms of page faults for a
  // hundred rows read from the disk); the system may evict them, and they
  // are no Metal memory.
  (void)::madvise(address, bytes, MADV_RANDOM);
  mapping_ = std::shared_ptr<const uint8_t>(static_cast<const uint8_t *>(address),
                                            [bytes](const uint8_t *mapped) { ::munmap(const_cast<uint8_t *>(mapped), bytes); });
  const uint64_t start = tensor.file->dataOffset() + tensor.offset;
  if (start > bytes || tensor.bytes > bytes - start || tensor.bytes != uint64_t{rows} * width * 2)
    throw std::runtime_error(std::string(name) + " lies outside its file");
  data_ = mapping_.get() + start;
  const auto page = static_cast<uint64_t>(::getpagesize());
  const uint64_t first = start / page * page;
  (void)::madvise(const_cast<uint8_t *>(mapping_.get()) + first, start + tensor.bytes - first, MADV_WILLNEED);
}

void HostTable::copyRows(std::span<const uint32_t> indices, void *destination) const {
  const size_t rowBytes = size_t{width_} * 2;
  auto *out = static_cast<uint8_t *>(destination);
  for (const uint32_t index : indices) {
    if (index >= rows_) throw std::invalid_argument("a table row is out of range");
    std::memcpy(out, data_ + size_t{index} * rowBytes, rowBytes);
    out += rowBytes;
  }
}

uint64_t packDenseWeights(metal::MetalBackend &backend, TtsWeights &weights, uint64_t limit) {
  std::vector<ops::omni::DenseWeights *> dense;
  addDense(dense, weights.talker);
  addDense(dense, weights.predictor);
  for (Qwen3Projection *p : {&weights.textFc1, &weights.textFc2, &weights.smallToMtp}) addDense(dense, *p);
  for (Qwen3Projection &head : weights.predictorHeads) addDense(dense, head);
  return packDense(backend, dense, limit);
}

uint64_t packDenseWeights(metal::MetalBackend &backend, AsrTextWeights &weights, uint64_t limit) {
  std::vector<ops::omni::DenseWeights *> dense;
  addDense(dense, weights.decoder);
  return packDense(backend, dense, limit);
}

} // namespace splash::omni
