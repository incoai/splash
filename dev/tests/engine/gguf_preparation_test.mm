// GGUF weight preparation on the GPU: every format's planes through the
// production executor against the CPU reference, the bytes the loader
// prepares from the small dense and MoE targets, their golden hashes and
// planned weights, warm loads, offsets past 4 GiB, the cache keys, and the
// target loader's reading of a prepared GGUF.
//   gguf-preparation METALLIB GOLDENS
// GOLDENS is dev/tests/fixtures/weight-goldens/goldens.json; its README says
// how to update it.
#include "GgufFixtures.hpp"
#include "model/GgufPreparation.hpp"
#include "model/GgufTarget.hpp"
#include "model/Qwen3_8.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/WeightLayout.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <fstream>

using namespace gguf_fixtures;
using namespace gguf_reference;

namespace {

namespace ops = splash::ops;
using splash::metal::MetalBackend;

// The source row image row n reads: order's grouped value heads read
// llama.cpp's tiled ones.
uint64_t sourceRow(uint64_t n, const model::gguf::RowOrder &order) {
  if (n < order.from) return n;
  const uint64_t head = (n - order.from) / order.headRows;
  const uint64_t source = (head % order.valueHeadsPerKey) * order.keyHeads + head / order.valueHeadsPerKey;
  return order.from + source * order.headRows + (n - order.from) % order.headRows;
}

// Rows of `bytes` (rowBytes each) in image order.
std::vector<uint8_t> orderedRows(const std::vector<uint8_t> &bytes, uint64_t rowBytes,
                                 const model::gguf::RowOrder &order) {
  std::vector<uint8_t> out(bytes.size());
  for (uint64_t n = 0; n < bytes.size() / rowBytes; ++n)
    std::memcpy(out.data() + n * rowBytes, bytes.data() + sourceRow(n, order) * rowBytes, rowBytes);
  return out;
}

// The bf16 upper halves of F32 values.
std::vector<uint8_t> bfloat16Halves(const std::vector<uint8_t> &floats) {
  std::vector<uint8_t> out;
  for (size_t i = 0; i < floats.size(); i += 4) out.insert(out.end(), floats.begin() + i + 2, floats.begin() + i + 4);
  return out;
}

std::vector<uint8_t> slice(const std::vector<uint8_t> &bytes, uint64_t offset, uint64_t size) {
  if (offset > bytes.size() || size > bytes.size() - offset) return {};
  return {bytes.begin() + offset, bytes.begin() + offset + size};
}

std::vector<uint8_t> cachedImage(const model::WeightFile &weights) {
  return splash::test::readFile(std::filesystem::path(std::getenv("SPLASH_WEIGHT_CACHE")) /
                                weights.record().contentIdentity / "weights");
}

// Every byte of each image the loader prepares: the layers', the head's and
// the embedding's.
std::vector<std::vector<uint8_t>> preparedImages(MetalBackend &backend, const std::filesystem::path &path,
                                                 const model::gguf::TargetGeometry &geometry) {
  model::GgufTargetLoader loader(backend, path, geometry);
  std::vector<std::vector<uint8_t>> images;
  for (uint32_t layer = 0; layer < geometry.layers; ++layer) images.push_back(cachedImage(loader.layer(layer)));
  images.push_back(cachedImage(loader.head()));
  images.push_back(cachedImage(loader.embedding()));
  return images;
}

std::vector<model::gguf::Image> planned(const std::filesystem::path &path,
                                        const model::gguf::TargetGeometry &geometry) {
  model::WeightSource source(path);
  const model::GgufFile gguf(source);
  return model::gguf::planImages(gguf, geometry);
}

// Every image the loader prepares from the GGUF at path against the goldens of
// `name`, and every golden of `name` prepared.
void checkGoldenImages(MetalBackend &backend, const std::filesystem::path &path,
                       const model::gguf::TargetGeometry &geometry, const std::string &name, const Goldens &hashes) {
  model::GgufTargetLoader loader(backend, path, geometry);
  uint64_t compared = 0;
  const auto compare = [&](const model::WeightFile &weights) {
    const auto &record = weights.record();
    const std::string image = name + "/" + record.relativePath;
    // The model's disk check budgets weights(): it must be the files the
    // loader writes, in order.
    check(compared < loader.weights().size() && loader.weights()[compared].key == record.contentIdentity &&
              loader.weights()[compared].bytes == record.declaredBytes &&
              loader.weights()[compared].component == record.relativePath,
          "GGUF loader plans the file it writes: " + image);
    checkGolden(hashes, image, cachedImage(weights), "golden prepared bytes: " + image);
    ++compared;
  };
  for (uint32_t layer = 0; layer < geometry.layers; ++layer) compare(loader.layer(layer));
  compare(loader.head());
  compare(loader.embedding());
  const auto goldens = std::count_if(hashes.begin(), hashes.end(), [&](const auto &golden) {
    return golden.first.starts_with(name + "/");
  });
  check(uint64_t(goldens) == compared, "every golden " + name + " image is prepared");
  check(compared == loader.weights().size(), "GGUF loader plans only the files it writes: " + name);
}

// The dense target: the Q8_0 alpha/beta tensor, the F32 norms, the
// convolution, decay and time bias in grouped head order, the bf16-exact
// rule, the golden images and the cache keys.
void checkDense(MetalBackend &backend, const std::filesystem::path &directory, const Goldens &hashes) {
  SmallTarget target = smallTarget(false);
  const model::gguf::TargetGeometry &g = target.geometry;
  const auto path = directory / "dense.gguf";
  writeGguf(path, target.tensors, g);
  const std::vector<model::gguf::Image> images = planned(path, g);
  const std::vector<std::vector<uint8_t>> prepared = preparedImages(backend, path, g);
  checkGoldenImages(backend, path, g, "dense", hashes);

  // Beta rows, alpha rows, then zero rows up to one tile, as one Q8_0 tensor.
  const auto *alphaBeta = repackOf(images[0], "blk.0.ssm_beta.weight");
  if (!alphaBeta) throw std::runtime_error("the plan has no alpha/beta tensor");
  const uint32_t stride = rowBytes(Q80, g.hiddenSize);
  std::vector<uint8_t> rows;
  for (const model::gguf::TensorRows &source : alphaBeta->sources) {
    const auto ordered = orderedRows(target.data(source.name), stride, source.order);
    rows.insert(rows.end(), ordered.begin(), ordered.end());
  }
  rows.resize(QUANT_TILE_ROWS * stride);
  const Packed expected = repack(Q80, rows, QUANT_TILE_ROWS, g.hiddenSize, nullptr);
  check(slice(prepared[0], alphaBeta->plane0, expected.w0.size()) == expected.w0 &&
            slice(prepared[0], alphaBeta->meta, expected.meta.size()) == expected.meta,
        "prepared alpha/beta tensor matches the CPU reference");

  for (size_t index = 0; index < images.size(); ++index)
    for (const model::gguf::Copy &copy : images[index].copies)
      if (isNorm(copy.source.name)) {
        const auto &values = target.data(copy.source.name);
        check(slice(prepared[index], copy.destination, values.size()) == values,
              "prepared norm is the GGUF's F32 values as stored: " + copy.source.name);
      }
  const auto prepares = [&](const char *name, bool bfloat16) {
    const model::gguf::Copy *copy = copyOf(images[0], name);
    if (!copy) return false;
    const auto &values = target.data(name);
    const auto ordered = orderedRows(values, values.size() / copy->source.rows, copy->source.order);
    const auto expected = bfloat16 ? bfloat16Halves(ordered) : ordered;
    return slice(prepared[0], copy->destination, expected.size()) == expected;
  };
  check(prepares("blk.0.ssm_conv1d.weight", true), "prepared convolution: exact bf16 in grouped head order");
  check(prepares("blk.0.ssm_a", false), "prepared decay: F32 in grouped head order");
  check(prepares("blk.0.ssm_dt.bias", true), "prepared time bias: exact bf16 in grouped head order");

  for (const std::string name : {"blk.0.ssm_conv1d.weight", "blk.0.ssm_dt.bias"}) {
    std::vector<Tensor> inexact = target.tensors;
    const float value = 1.0f + 0x1p-10f;
    std::memcpy(tensorNamed(inexact, name).data.data() + 12, &value, 4);
    writeGguf(path, inexact, g);
    std::string refused;
    try {
      static_cast<void>(preparedImages(backend, path, g));
    } catch (const model::GgufError &error) {
      refused = error.what();
    }
    check(refused.find(name) != std::string::npos && refused.find("bf16") != std::string::npos,
          "preparation refuses to round " + name + " to bf16");
  }

  // Keys follow the tensor data the images read: a metadata edit (a chat
  // template, so every tensor moves in the file) keeps every prepared image,
  // a changed tensor byte prepares its image again.
  bool allowPreparation = true;
  const auto keys = [&] {
    model::GgufTargetLoader loader(backend, path, g, [&] {
      if (!allowPreparation) throw std::runtime_error("conversion forbidden");
    });
    return std::array<std::string, 2>{loader.layer(0).record().contentIdentity, loader.head().record().contentIdentity};
  };
  writeGguf(path, target.tensors, g);
  const auto original = keys();
  std::vector<test_gguf::Key> edited = metadata(g);
  edited.push_back(test_gguf::stringKey("tokenizer.chat_template", std::string(100, 'x')));
  splash::test::writeFile(path, test_gguf::file(edited, target.tensors));
  allowPreparation = false;
  check(keys() == original, "a GGUF metadata edit keeps every prepared image");
  target.data("blk.0.ssm_out.weight")[100] ^= 1;
  writeGguf(path, target.tensors, g);
  allowPreparation = true;
  check(keys()[0] != original[0], "a changed tensor byte prepares its image again");
}

// The MoE layer: the F32 alpha/beta tensor, the golden images, warm loads
// and tensor offsets past 4 GiB.
void checkMoe(MetalBackend &backend, const std::filesystem::path &directory, const Goldens &hashes) {
  SmallTarget target = smallTarget(true);
  const model::gguf::TargetGeometry &g = target.geometry;
  const auto path = directory / "moe.gguf";
  writeGguf(path, target.tensors, g);
  checkGoldenImages(backend, path, g, "moe", hashes);
  const std::vector<model::gguf::Image> images = planned(path, g);
  const auto *beta = copyOf(images[0], "blk.0.ssm_beta.weight"), *alpha = copyOf(images[0], "blk.0.ssm_alpha.weight");
  if (!beta || !alpha) throw std::runtime_error("the plan has no F32 alpha/beta tensor");

  bool allowPreparation = true;
  const auto load = [&] {
    model::GgufTargetLoader loader(backend, path, g, [&] {
      if (!allowPreparation) throw std::runtime_error("conversion forbidden on warm load");
    });
    auto weights = loader.layer(0);
    const auto bytes = weights.section(weights.record().declaredBytes - model::kWeightFileAlignment);
    const auto *begin = static_cast<const uint8_t *>(bytes.contents());
    std::vector<uint8_t> image(model::kWeightFileAlignment, 0);
    image.insert(image.end(), begin, begin + bytes.sizeBytes());
    weights.finish();
    return image;
  };
  const auto image = load();
  const uint64_t rowBytes = uint64_t{g.hiddenSize} * sizeof(float);
  std::vector<uint8_t> gates = orderedRows(target.data(beta->source.name), rowBytes, beta->source.order);
  const auto alphaRows = orderedRows(target.data(alpha->source.name), rowBytes, alpha->source.order);
  gates.insert(gates.end(), alphaRows.begin(), alphaRows.end());
  check(slice(image, beta->destination, gates.size()) == gates,
        "prepared F32 alpha/beta tensor: beta then alpha rows in grouped order");
  allowPreparation = false;
  check(load() == image, "GGUF warm load does not require conversion headroom");

  // A layer may have tensors on opposite sides of the 4 GiB boundary. Keep
  // the file sparse and zero the old location so a truncated offset cannot
  // read the right data.
  allowPreparation = true;
  constexpr uint64_t kDisplacement = uint64_t{1} << 32;
  for (const std::string name : {"blk.0.ffn_down_exps.weight", "blk.0.ffn_gate_inp.weight"}) {
    writeGguf(path, target.tensors, g);
    model::WeightSource source(path);
    const model::GgufFile original(source);
    const uint64_t offset = source.dataOffset() + original.require(name).offset;
    std::vector<Tensor> displaced = target.tensors;
    Tensor &moved = tensorNamed(displaced, name);
    moved.data.assign(moved.data.size(), 0);
    moved.displacement = kDisplacement;
    const auto bytes = test_gguf::file(metadata(g), displaced);
    {
      std::ofstream stream(path, std::ios::binary | std::ios::trunc);
      stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
      stream.seekp(static_cast<std::streamoff>(offset + kDisplacement));
      stream.write(reinterpret_cast<const char *>(target.data(name).data()), target.data(name).size());
    }
    check(load() == image, "loader keeps weights exact beyond 4 GiB: " + name);
  }
}

// The MoE layer with BF16 alpha/beta, which preparation widens to the F32
// values they equal.
void checkWidenedAlphaBeta(MetalBackend &backend, const std::filesystem::path &directory) {
  SmallTarget target = smallTarget(true);
  const model::gguf::TargetGeometry &g = target.geometry;
  const auto path = directory / "moe-bf16.gguf";
  std::map<std::string, std::vector<uint8_t>> floats; // the F32 values of the BF16 tensors
  uint32_t seed = 960;
  for (const std::string name : {"blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight"}) {
    floats[name] = floatValues(uint64_t{g.gdnValueHeads} * g.hiddenSize, ++seed, true);
    Tensor &tensor = tensorNamed(target.tensors, name);
    tensor.type = model::ggml::kBF16;
    tensor.data = bfloat16Halves(floats[name]);
  }
  writeGguf(path, target.tensors, g);
  const std::vector<model::gguf::Image> images = planned(path, g);
  const auto *beta = copyOf(images[0], "blk.0.ssm_beta.weight"), *alpha = copyOf(images[0], "blk.0.ssm_alpha.weight");
  if (!beta || !alpha) throw std::runtime_error("the plan has no alpha/beta tensor");
  check(beta->float32 && alpha->float32 && !beta->bfloat16 && !alpha->bfloat16 &&
            alpha->destination == beta->destination + 2 * target.data(beta->source.name).size(),
        "planner widens BF16 alpha/beta into one F32 tensor");
  model::GgufTargetLoader loader(backend, path, g);
  const std::vector<uint8_t> image = cachedImage(loader.layer(0));
  const uint64_t rowBytes = uint64_t{g.hiddenSize} * sizeof(float);
  std::vector<uint8_t> gates = orderedRows(floats[beta->source.name], rowBytes, beta->source.order);
  const auto alphaRows = orderedRows(floats[alpha->source.name], rowBytes, alpha->source.order);
  gates.insert(gates.end(), alphaRows.begin(), alphaRows.end());
  check(slice(image, beta->destination, gates.size()) == gates,
        "prepared BF16 alpha/beta: their F32 values, beta then alpha rows in grouped order");
}

// The segments of a block projection of `n` x `k`, by output width.
bool blockProjection(const ops::Projection &p, uint32_t n, uint32_t k, std::vector<uint32_t> widths) {
  if (p.layout() != ops::WeightLayout::Block32 || p.outputSize != n || p.inputSize != k ||
      p.blocks().segments.size() != widths.size())
    return false;
  uint32_t offset = 0;
  for (size_t i = 0; i < widths.size(); ++i) {
    const ops::QuantizedSegment &s = p.blocks().segments[i];
    if (s.columnOffset != offset || s.outputSize != widths[i] || s.inputSize != k) return false;
    offset += widths[i];
  }
  return true;
}

// A qwen35 target read from a GGUF through the production loader: every
// projection a block projection of the layout's sizes, a fused one a segment
// per tensor in the layout's padded width, the norms F32 and the GDN output in
// the GGUF's tiled head order; the head and token table as the GGUF stores them.
void checkDenseTarget(MetalBackend &backend, const std::filesystem::path &directory) {
  using namespace model::ggml;
  model::Qwen3_8Layout layout;
  layout.layers = 4;
  layout.hiddenSize = 256;
  layout.vocabularySize = 256;
  layout.gdnKeyHeads = 1;
  layout.gdnValueHeads = 2;
  layout.gdnHeadDimension = 128;
  layout.convolutionDimension = 512; // q and k of one head, v of two
  layout.packedGdnWidth = 1280;      // qkv | z | alpha-beta and one padding tile
  layout.attentionWidth = 256;
  layout.attentionQueryHeads = 2;
  layout.attentionKvHeads = 2;
  layout.attentionHeadDimension = 128;
  layout.packedFullWidth = 1024;     // q and its gate | k | v
  layout.intermediateSize = 512;
  layout.hiddenCaptureLayers.fill(layout.layers - 1);
  const uint32_t hidden = layout.hiddenSize, valueRows = layout.gdnValueHeads * layout.gdnHeadDimension;
  const uint32_t kvRows = layout.attentionKvHeads * layout.attentionHeadDimension;
  const model::gguf::TargetGeometry geometry = model::ggufTargetGeometry(layout);
  const auto target = directory / "target";
  std::filesystem::create_directory(target);
  writeGguf(target / "target.gguf",
            targetTensors(geometry, {{"attn_q.weight", kQ8_0},
                                     {"attn_k.weight", kQ4_K},
                                     {"attn_v.weight", kQ6_K},
                                     {"attn_output.weight", kQ8_0},
                                     {"attn_qkv.weight", kQ8_0},
                                     {"attn_gate.weight", kQ4_K},
                                     {"ssm_beta.weight", kQ8_0},
                                     {"ssm_alpha.weight", kQ8_0},
                                     {"ssm_out.weight", kQ8_0},
                                     {"ffn_gate.weight", kQ4_K},
                                     {"ffn_up.weight", kQ4_K},
                                     {"ffn_down.weight", kQ6_K},
                                     {"output.weight", kQ6_K},
                                     {"token_embd.weight", kQ8_0}}),
            geometry);
  model::GgufTargetLoader files(backend, model::findTargetGguf(target), geometry);
  const model::Qwen3_8Weights weights = model::loadQwen3_8Weights(backend, layout, files);
  check(weights.layers.size() == layout.layers, "GGUF target: every layer");
  check(weights.finalNorm.float32, "GGUF target: F32 final norm");
  check(blockProjection(weights.logitsProjection, layout.vocabularySize, hidden, {layout.vocabularySize}) &&
            std::string_view(weights.logitsProjection.blocks().segments.front().name()) == "q6k" &&
            weights.logitsProjection.destination == ops::FloatOutput::Float32,
        "GGUF target: logits a Q6_K block projection of vocabulary x hidden into fp32");
  check(weights.tokenEmbedding.layout() == ops::WeightLayout::Block32 &&
            weights.tokenEmbedding.blocks().formatId == GGUF_FMT_Q80 &&
            weights.tokenEmbedding.outputSize == layout.vocabularySize && weights.tokenEmbedding.inputSize == hidden,
        "GGUF target: token table Q8_0 blocks of vocabulary x hidden");
  check(model::qwenTargetGeometry(weights).valid(), "GGUF target: a valid target geometry");
  for (uint32_t index = 0; index < weights.layers.size(); ++index) {
    const auto &layer = weights.layers[index];
    const std::string at = "GGUF target layer " + std::to_string(index) + ": ";
    check(layer.inputNorm.float32 && layer.postAttentionNorm.float32, at + "F32 input and post-attention norms");
    check(blockProjection(layer.gateProjection, layout.intermediateSize, hidden, {layout.intermediateSize}),
          at + "gate block projection");
    check(blockProjection(layer.upProjection, layout.intermediateSize, hidden, {layout.intermediateSize}),
          at + "up block projection");
    check(blockProjection(layer.downProjection, hidden, layout.intermediateSize, {hidden}),
          at + "down block projection");
    if (const auto *gdn = std::get_if<model::QwenGdnWeights>(&layer.mixer)) {
      check(blockProjection(gdn->inputProjection, layout.packedGdnWidth, hidden,
                            {layout.convolutionDimension, valueRows, QUANT_TILE_ROWS}),
            at + "GDN input segments qkv | z | alpha-beta tile");
      check(blockProjection(gdn->outputProjection, hidden, valueRows, {hidden}), at + "GDN output block projection");
      check(gdn->mixerNorm.float32, at + "F32 GDN norm");
      check(gdn->outputHeadOrder == ops::GdnHeadOrder::Tiled, at + "GDN output in the GGUF's tiled head order");
    } else {
      const auto &attention = std::get<model::QwenAttentionWeights>(layer.mixer);
      check(blockProjection(attention.inputProjection, layout.packedFullWidth, hidden,
                            {2 * layout.attentionWidth, kvRows, kvRows}),
            at + "attention input segments q and gate | k | v");
      check(blockProjection(attention.outputProjection, hidden, layout.attentionWidth, {hidden}),
            at + "attention output block projection");
      check(attention.queryNorm.float32 && attention.keyNorm.float32, at + "F32 query and key norms");
    }
  }
}

// The tensor data section's offset in the executor's source file: not zero,
// so reads must add it.
constexpr uint64_t kSourceOffset = 96;

// Row tiles are whole (the planner requires rows and K to be multiples of 256);
// the shapes cover several tiles, super-blocks and groups, and a permuted row range.
struct Shape {
  uint32_t rows, K;
  model::gguf::RowOrder order;
};

// The narrowest K whose tile of source and plane bytes exceeds the
// preparation staging, so preparation must split the rows by columns.
uint32_t widerThanStaging(Fmt f) {
  const QuantFormat &format = kQuantFormats[f];
  const model::GgufPlaneBytes planes = model::ggufPlaneBytes(format, 1, model::kGgufBlockColumns);
  const uint64_t perBlock = model::ggufRowBytes(format, model::kGgufBlockColumns) + planes.plane0 + planes.plane1 +
                            planes.meta;
  return uint32_t((model::kWeightPreparationStagingBytes / (QUANT_TILE_ROWS * perBlock) + 1) *
                  model::kGgufBlockColumns);
}

// One quantized tensor through the production executor, from a file whose
// data section starts at kSourceOffset, against the CPU reference's planes.
void checkRepack(MetalBackend &backend, const std::filesystem::path &directory, Fmt f, const Shape &shape,
                 uint32_t seed) {
  const QuantFormat &layout = kQuantFormats[f];
  const uint32_t rows = shape.rows, K = shape.K, stride = rowBytes(f, K);
  const std::vector<uint8_t> native = fixture(f, rows, K, seed);
  const Packed expected = repack(f, orderedRows(native, stride, shape.order), rows, K, nullptr);
  const uint64_t plane0 = model::kWeightFileAlignment, plane1 = model::alignWeightOffset(plane0 + expected.w0.size());
  const uint64_t meta = layout.plane1_bytes ? model::alignWeightOffset(plane1 + expected.w1.size()) : plane1;
  const uint64_t bytes = model::alignWeightOffset(meta + expected.meta.size()) + model::kWeightFileAlignment;
  const std::string what = std::string("prepared ") + fmtName(f) + " rows=" + std::to_string(rows) +
                           " K=" + std::to_string(K) + (shape.order.from == UINT64_MAX ? "" : " permuted");
  const auto inputPath = directory / "repack-source", outputPath = directory / "repack-output";
  int output = -1;
  try {
    std::vector<uint8_t> input(kSourceOffset, 0);
    input.insert(input.end(), native.begin(), native.end());
    splash::test::writeFile(inputPath, input);
    output = open(outputPath.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (output < 0 || ftruncate(output, bytes)) throw std::runtime_error("cannot create the repack output");
    model::gguf::Repack step;
    step.format = f;
    step.rows = rows;
    step.columns = K;
    step.plane0 = plane0;
    step.plane1 = layout.plane1_bytes ? plane1 : 0;
    step.meta = meta;
    step.sources = {{"fixture", layout.ggml_type, 0, rows, stride, shape.order}};
    model::gguf::Image plan;
    plan.bytes = bytes;
    plan.repacks.push_back(step);
    const uint64_t before = backend.memoryStats().allocatedBytes;
    model::WeightSource source(inputPath);
    source.setDataOffset(kSourceOffset);
    model::writeGgufImage(backend, source, output, plan, [] {});
    check(backend.memoryStats().allocatedBytes == before, "repack releases its staging buffers");
    check(backend.memoryStats().peakAllocatedBytes <= model::kWeightPreparationStagingBytes,
          "repack staging stays within the preparation staging bound");
    std::vector<uint8_t> actual(bytes), reference(bytes, 0);
    model::readWeightBytes(output, 0, actual);
    std::copy(expected.w0.begin(), expected.w0.end(), reference.begin() + plane0);
    if (layout.plane1_bytes) std::copy(expected.w1.begin(), expected.w1.end(), reference.begin() + plane1);
    std::copy(expected.meta.begin(), expected.meta.end(), reference.begin() + meta);
    check(actual == reference, what);
  } catch (const std::exception &error) {
    check(false, what + ": " + error.what());
  }
  if (output >= 0) close(output);
}

void checkExecutor(MetalBackend &backend, const std::filesystem::path &directory) {
  const Shape shapes[] = {{512, 1024, {}}, {768, 1280, {256, 16, 8, 4}}, {768, 8448, {128, 16, 8, 5}}};
  for (int s = 0; s < 3; ++s)
    for (int f = 0; f < FMT_COUNT; ++f) checkRepack(backend, directory, Fmt(f), shapes[s], 100 + 8 * s + f);
  // Several bounded row batches, and rows wider than one staging step.
  for (Fmt format : {Q3K, Q80}) {
    checkRepack(backend, directory, format, {8704, 2048, {}}, 741);
    checkRepack(backend, directory, format, {256, widerThanStaging(format), {}}, 742);
  }
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 3) {
      std::fprintf(stderr, "usage: gguf-preparation METALLIB GOLDENS\n");
      return 2;
    }
    // Every run must exercise conversion, including the >4 GiB source offsets.
    const splash::test::TemporaryDirectory directory("splash-gguf-preparation");
    setenv("SPLASH_WEIGHT_CACHE", (directory.path() / "cache").c_str(), 1);
    const Goldens hashes = goldens(argv[2], @"gguf_images");
    MetalBackend backend(argv[1]);
    guarded("preparation of the dense target", [&] { checkDense(backend, directory.path(), hashes); });
    guarded("preparation of the MoE target", [&] { checkMoe(backend, directory.path(), hashes); });
    guarded("preparation of BF16 alpha/beta", [&] { checkWidenedAlphaBeta(backend, directory.path()); });
    guarded("the target loader", [&] { checkDenseTarget(backend, directory.path()); });
    checkExecutor(backend, directory.path());
    std::printf("%s (%d failures)\n", failures ? "GGUF preparation tests FAILED" : "GGUF preparation tests passed",
                failures);
    return failures ? 1 : 0;
  }
}
