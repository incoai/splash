// GGUF weight preparation on the GPU: every format's planes through the
// production executor against the CPU reference, the images the loader
// writes from the small dense and MoE targets, their golden hashes, offsets
// past 4 GiB, a restore of released images, the target loader's reading of a
// GGUF, llama.cpp's NVFP4 tensors, and the quantized tensors of MLX, Model
// Optimizer and compressed-tensors read from a safetensors checkpoint.
//   gguf-preparation METALLIB GOLDENS MLX_FIXTURE
// GOLDENS is dev/tests/fixtures/weight-goldens/goldens.json; its README says
// how to update it. MLX_FIXTURE is
// dev/tests/fixtures/mlx-quantization/fixture.json.
#include "GgufFixtures.hpp"
#include "TestAdmission.hpp"
#include "TestCheckpoint.hpp"
#include "model/GgufPreparation.hpp"
#include "model/GgufTarget.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/Qwen3_8.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"

#include <array>
#include <cstring>
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

// Loads every image of the GGUF at path into images: the layers', the head's
// and the embedding's.
void loadImages(MetalBackend &backend, model::WeightImages &images, const std::filesystem::path &path,
                const model::QwenTargetDimensions &geometry) {
  model::GgufTargetLoader loader(backend, images, {path}, geometry);
  for (uint32_t layer = 0; layer < geometry.layers; ++layer) static_cast<void>(loader.layer(layer));
  static_cast<void>(loader.head());
  static_cast<void>(loader.embedding());
}

std::vector<std::vector<uint8_t>> bytesOf(const model::WeightImages &images) {
  std::vector<std::vector<uint8_t>> result;
  for (const auto &image : images.contents()) result.emplace_back(image.bytes.begin(), image.bytes.end());
  return result;
}

// Every byte of each image the loader writes.
std::vector<std::vector<uint8_t>> preparedImages(MetalBackend &backend, const std::filesystem::path &path,
                                                 const model::QwenTargetDimensions &geometry) {
  model::WeightImages images(backend, "fixture");
  loadImages(backend, images, path, geometry);
  return bytesOf(images);
}

std::vector<model::gguf::Image> planned(const std::filesystem::path &path,
                                        const model::QwenTargetDimensions &geometry) {
  model::WeightSource source(path);
  const model::GgufFile gguf(source);
  return model::gguf::planImages(gguf, geometry);
}

// Every image the loader writes from the GGUF at path against the goldens of
// `name`, and every golden of `name` written; then the same bytes again once
// the images are released and restored.
void checkGoldenImages(MetalBackend &backend, const std::filesystem::path &path,
                       const model::QwenTargetDimensions &geometry, const std::string &name, const Goldens &hashes) {
  model::WeightImages images(backend, "fixture");
  loadImages(backend, images, path, geometry);
  const auto loaded = bytesOf(images);
  for (const auto &image : images.contents())
    checkGolden(hashes, name + "/" + std::string(image.component), image.bytes,
                "golden prepared bytes: " + name + "/" + std::string(image.component));
  const auto goldens = std::count_if(hashes.begin(), hashes.end(), [&](const auto &golden) {
    return golden.first.starts_with(name + "/");
  });
  check(uint64_t(goldens) == loaded.size(), "every golden " + name + " image is prepared");
  images.release();
  while (!images.restore(splash::test::admitAll)) {
  }
  check(bytesOf(images) == loaded, "restored " + name + " images are the bytes loaded");
}

// The dense target: the F32 norms, the convolution, decay and time bias in
// grouped head order, the bf16-exact rule and the golden images.
void checkDense(MetalBackend &backend, const std::filesystem::path &directory, const Goldens &hashes) {
  SmallTarget target = smallTarget(false);
  const model::QwenTargetDimensions &g = target.geometry;
  const auto path = directory / "dense.gguf";
  writeGguf(path, target.tensors, g);
  const std::vector<model::gguf::Image> images = planned(path, g);
  const std::vector<std::vector<uint8_t>> prepared = preparedImages(backend, path, g);
  checkGoldenImages(backend, path, g, "dense", hashes);

  for (size_t index = 0; index < images.size(); ++index)
    for (const model::gguf::Copy &copy : images[index].copies)
      if (isNorm(copy.source.name)) {
        const auto &values = target.data(copy.source.name);
        check(slice(prepared[index], copy.destination, values.size()) == values,
              "prepared norm is the GGUF's F32 values as stored: " + copy.source.name);
      }
  const auto prepares = [&](const char *name, model::gguf::Conversion conversion) {
    const model::gguf::Copy *copy = copyOf(images[0], name);
    if (!copy || copy->conversion != conversion) return false;
    const auto &values = target.data(name);
    const auto ordered = orderedRows(values, values.size() / copy->source.rows, copy->source.order);
    const auto expected =
        conversion == model::gguf::Conversion::NarrowToBfloat16 ? bfloat16Halves(ordered) : ordered;
    return slice(prepared[0], copy->destination, expected.size()) == expected;
  };
  check(prepares("blk.0.ssm_conv1d.weight", model::gguf::Conversion::NarrowToBfloat16),
        "prepared convolution: exact bf16 in grouped head order");
  check(prepares("blk.0.ssm_a", model::gguf::Conversion::None), "prepared decay: F32 in grouped head order");
  check(prepares("blk.0.ssm_dt.bias", model::gguf::Conversion::NarrowToBfloat16),
        "prepared time bias: exact bf16 in grouped head order");

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
}

// alpha/beta of every GGUF tensor type of a format on either architecture
// (llama.cpp's NVFP4 as NVFP4's native rows): one tensor of the beta then the
// alpha rows in grouped head order, then zero rows through the tile, every
// plane equal to the CPU reference's. The expected row order is derived from
// the geometry, not from the plan.
void checkQuantizedAlphaBeta(MetalBackend &backend, const std::filesystem::path &directory) {
  for (bool moe : {false, true})
    for (uint32_t format = 0; format < GGUF_FMT_COUNT; ++format) {
      const std::optional<uint32_t> type = ggufType(format);
      if (!type) continue;
      SmallTarget target = smallTarget(moe);
      const model::QwenTargetDimensions &g = target.geometry;
      const Fmt f = Fmt(format);
      const uint32_t stride = rowBytes(f, g.hiddenSize);
      uint32_t seed = 1200 + 2 * format;
      std::vector<uint8_t> rows;
      for (const char *name : {"blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight"}) {
        Tensor &tensor = tensorNamed(target.tensors, name);
        tensor.type = *type;
        std::vector<uint8_t> native = fixture(f, g.gdnValueHeads, g.hiddenSize, ++seed);
        // llama.cpp's NVFP4 without a .scale tensor: g is 1.
        if (f == NVFP4) setTensorScales(native, f, g.hiddenSize, [](uint32_t) { return 1.0f; });
        tensor.data = f == NVFP4 ? blockNvfp4Rows(native) : native;
        // Grouped value head v of key head k is llama.cpp's tiled row v * keyHeads + k.
        for (uint32_t key = 0; key < g.gdnKeyHeads; ++key)
          for (uint32_t value = 0; value < g.gdnValueHeads / g.gdnKeyHeads; ++value) {
            const uint64_t source = (value * g.gdnKeyHeads + key) * uint64_t{stride};
            rows.insert(rows.end(), native.begin() + source, native.begin() + source + stride);
          }
      }
      rows.resize(QUANT_TILE_ROWS * stride);
      const Packed expected = repack(f, rows, QUANT_TILE_ROWS, g.hiddenSize, nullptr);
      const auto path =
          directory / (std::string(model::gguf::architecture(g.ffnKind)) + "-" + fmtName(format) + ".gguf");
      writeGguf(path, target.tensors, g);
      const auto images = planned(path, g);
      const auto prepared = preparedImages(backend, path, g);
      const auto *pair = repackOf(images[0], "blk.0.ssm_beta.weight");
      if (!pair) throw std::runtime_error("the plan has no quantized alpha/beta tensor");
      const auto matches = [&](uint64_t offset, const std::vector<uint8_t> &bytes) {
        return slice(prepared[0], offset, bytes.size()) == bytes;
      };
      check(pair->format == format && matches(pair->plane0, expected.w0) &&
                (!kQuantFormats[format].plane1_bytes || matches(pair->plane1, expected.w1)) &&
                matches(pair->meta, expected.meta),
            std::string("prepared ") + fmtName(format) + " alpha/beta and zero rows match the CPU reference (" +
                model::gguf::architecture(g.ffnKind) + ")");
    }
}

// The MoE layer: the F32 alpha/beta tensor, the golden images and tensor
// offsets past 4 GiB.
void checkMoe(MetalBackend &backend, const std::filesystem::path &directory, const Goldens &hashes) {
  SmallTarget target = smallTarget(true);
  const model::QwenTargetDimensions &g = target.geometry;
  const auto path = directory / "moe.gguf";
  writeGguf(path, target.tensors, g);
  checkGoldenImages(backend, path, g, "moe", hashes);
  const std::vector<model::gguf::Image> images = planned(path, g);
  const auto *beta = copyOf(images[0], "blk.0.ssm_beta.weight"), *alpha = copyOf(images[0], "blk.0.ssm_alpha.weight");
  if (!beta || !alpha) throw std::runtime_error("the plan has no F32 alpha/beta tensor");

  const auto load = [&] { return preparedImages(backend, path, g).front(); };
  const auto image = load();
  const uint64_t rowBytes = uint64_t{g.hiddenSize} * sizeof(float);
  std::vector<uint8_t> gates = orderedRows(target.data(beta->source.name), rowBytes, beta->source.order);
  const auto alphaRows = orderedRows(target.data(alpha->source.name), rowBytes, alpha->source.order);
  gates.insert(gates.end(), alphaRows.begin(), alphaRows.end());
  check(slice(image, beta->destination, gates.size()) == gates,
        "prepared F32 alpha/beta tensor: beta then alpha rows in grouped order");

  // A layer may have tensors on opposite sides of the 4 GiB boundary. Keep
  // the file sparse and zero the old location so a truncated offset cannot
  // read the right data.
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
  const model::QwenTargetDimensions &g = target.geometry;
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
  check(beta->conversion == model::gguf::Conversion::WidenToFloat32 &&
            alpha->conversion == model::gguf::Conversion::WidenToFloat32 &&
            alpha->destination == beta->destination + 2 * target.data(beta->source.name).size(),
        "planner widens BF16 alpha/beta into one F32 tensor");
  const std::vector<uint8_t> image = preparedImages(backend, path, g).front();
  const uint64_t rowBytes = uint64_t{g.hiddenSize} * sizeof(float);
  std::vector<uint8_t> gates = orderedRows(floats[beta->source.name], rowBytes, beta->source.order);
  const auto alphaRows = orderedRows(floats[alpha->source.name], rowBytes, alpha->source.order);
  gates.insert(gates.end(), alphaRows.begin(), alphaRows.end());
  check(slice(image, beta->destination, gates.size()) == gates,
        "prepared BF16 alpha/beta: their F32 values, beta then alpha rows in grouped order");
}

// The segments of a block projection of `n` x `k`, by output width.
bool blockProjection(const ops::Projection &p, uint32_t n, uint32_t k, std::vector<uint32_t> widths) {
  if (p.outputSize != n || p.inputSize != k || p.blocks().segments.size() != widths.size()) return false;
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
  const auto target = directory / "target";
  std::filesystem::create_directory(target);
  writeGguf(target / "target.gguf",
            targetTensors(layout, {{"attn_q.weight", kQ8_0},
                                   {"attn_k.weight", kQ4_K},
                                   {"attn_v.weight", kQ6_K},
                                   {"attn_output.weight", kQ8_0},
                                   {"attn_qkv.weight", kQ8_0},
                                   {"attn_gate.weight", kQ4_K},
                                   {"ssm_beta.weight", kIQ4_XS},
                                   {"ssm_alpha.weight", kIQ4_XS},
                                   {"ssm_out.weight", kQ8_0},
                                   {"ffn_gate.weight", kQ4_K},
                                   {"ffn_up.weight", kQ4_K},
                                   {"ffn_down.weight", kQ6_K},
                                   {"output.weight", kQ6_K},
                                   {"token_embd.weight", kIQ4_XS}}),
            layout);
  model::WeightImages images(backend, "fixture");
  model::GgufTargetLoader files(backend, images, model::findTargetGgufs(target), layout);
  const model::Qwen3_8Weights weights = model::loadQwen3_8Weights(backend, layout, files);
  check(weights.layers.size() == layout.layers, "GGUF target: every layer");
  check(weights.finalNorm.float32, "GGUF target: F32 final norm");
  check(blockProjection(weights.logitsProjection, layout.vocabularySize, hidden, {layout.vocabularySize}) &&
            std::string_view(weights.logitsProjection.blocks().segments.front().name()) == "q6k" &&
            weights.logitsProjection.destination == ops::FloatOutput::Float32,
        "GGUF target: logits a Q6_K block projection of vocabulary x hidden into fp32");
  check(weights.tokenEmbedding.blocks().formatId == GGUF_FMT_IQ4XS &&
            weights.tokenEmbedding.outputSize == layout.vocabularySize && weights.tokenEmbedding.inputSize == hidden,
        "GGUF target: token table IQ4_XS blocks of vocabulary x hidden");
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
                            {layout.convolutionDimension, valueRows, QUANT_TILE_ROWS}) &&
                gdn->inputProjection.blocks().segments[2].formatId == GGUF_FMT_IQ4XS,
            at + "GDN input segments qkv | z | IQ4_XS alpha-beta tile");
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

// The narrowest K whose tile of source rows exceeds the repack staging, so
// preparation must split the rows by columns.
uint32_t widerThanStaging(Fmt f) {
  const uint64_t perBlock = model::ggufRowBytes(kQuantFormats[f], model::kGgufBlockColumns);
  return uint32_t((model::kGgufRepackStagingBytes / (QUANT_TILE_ROWS * perBlock) + 1) * model::kGgufBlockColumns);
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
  const auto inputPath = directory / "repack-source";
  try {
    std::vector<uint8_t> input(kSourceOffset, 0);
    input.insert(input.end(), native.begin(), native.end());
    splash::test::writeFile(inputPath, input);
    model::WeightSource source(inputPath);
    source.setDataOffset(kSourceOffset);
    model::gguf::Repack step;
    step.format = f;
    step.rows = rows;
    step.columns = K;
    step.plane0 = plane0;
    step.plane1 = layout.plane1_bytes ? plane1 : 0;
    step.meta = meta;
    step.sources = {{"fixture", layout.ggml_type, 0, rows, stride, shape.order, &source}};
    model::gguf::Image plan;
    plan.bytes = bytes;
    plan.repacks.push_back(step);
    const auto image = backend.allocateBuffer(bytes, splash::metal::BufferStorage::Shared, "repack-output");
    // The writer writes every byte: none of these is left.
    std::memset(image.contents(), 0xFF, bytes);
    const auto before = backend.memoryStats();
    model::writeGgufImage(backend, image, plan);
    const auto after = backend.memoryStats();
    check(after.allocatedBytes == before.allocatedBytes, "repack releases its staging buffer");
    check(after.peakAllocatedBytes <=
              std::max(before.peakAllocatedBytes, before.allocatedBytes + model::kGgufRepackStagingBytes),
          "repack staging stays within the repack staging bound");
    const auto actual = model::contentsOf(image);
    std::vector<uint8_t> reference(bytes, 0);
    std::copy(expected.w0.begin(), expected.w0.end(), reference.begin() + plane0);
    if (layout.plane1_bytes) std::copy(expected.w1.begin(), expected.w1.end(), reference.begin() + plane1);
    std::copy(expected.meta.begin(), expected.meta.end(), reference.begin() + meta);
    check(std::equal(actual.begin(), actual.end(), reference.begin(), reference.end()), what);
  } catch (const std::exception &error) {
    check(false, what + ": " + error.what());
  }
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

// Each MLX fixture tensor through the production writer, read from a
// safetensors checkpoint as the planner binds it (model/SafetensorsImage.hpp):
// its planes in one 256-row tile, the native rows the token gather reads and,
// for an affine tensor, the F32 values the MoE router's dequantization writes,
// against the CPU reference's planes of the loader's native rows (mlxNative)
// and MLX's own values.
void checkMlxSources(MetalBackend &backend, const std::filesystem::path &directory, const char *fixture) {
  for (const MlxTensor &tensor : mlxFixture(fixture)) {
    const Fmt f = tensor.format();
    const QuantFormat &layout = kQuantFormats[f];
    const uint32_t rows = tensor.rows, K = tensor.columns, groups = K / tensor.group;
    const auto root = directory / (std::string("mlx-") + fmtName(f));
    std::vector<splash::test::SyntheticTensor> shard{
        {"m.weight", "U32", {rows, K * tensor.bits / 32}, tensor.weight},
        {"m.scales", tensor.affine() ? "BF16" : "U8", {rows, groups}, tensor.scales}};
    if (tensor.affine()) shard.push_back({"m.biases", "BF16", {rows, groups}, tensor.biases});
    splash::test::writeSyntheticShard(root / "model.safetensors", shard);
    const model::SafetensorsCheckpoint checkpoint(root);
    const model::SourceTensor &codes = checkpoint.require("m.weight");
    const model::gguf::TensorRows source{
        "m", layout.ggml_type, 0, rows, rowBytes(f, K), {}, codes.file,
        {&codes, &checkpoint.require("m.scales"), tensor.affine() ? &checkpoint.require("m.biases") : nullptr}};
    const std::vector<uint8_t> native = tensor.native();
    std::vector<uint8_t> tile = native;
    tile.resize(size_t(QUANT_TILE_ROWS) * rowBytes(f, K), 0);
    const Packed expected = repack(f, tile, QUANT_TILE_ROWS, K, nullptr);

    model::gguf::ImageBuilder builder("mlx.bin", 0, 0);
    model::gguf::Repack planes = builder.planes(f, QUANT_TILE_ROWS, K, "m");
    planes.sources.push_back(source);
    const model::gguf::Repack step = planes;
    builder.repack(std::move(planes));
    const uint64_t nativeAt = builder.section(native.size());
    builder.copyAt(nativeAt, source, model::gguf::Conversion::None);
    const uint64_t valuesAt = builder.section(tensor.values.size());
    builder.copyAt(valuesAt, source, model::gguf::Conversion::DequantizeToFloat32);
    const model::gguf::Image plan = builder.finish();
    const auto image = backend.allocateBuffer(plan.bytes, splash::metal::BufferStorage::Shared, "mlx-image");
    model::writeGgufImage(backend, image, plan);
    const auto actual = model::contentsOf(image);
    const auto holds = [&](uint64_t at, const std::vector<uint8_t> &bytes) {
      return at + bytes.size() <= actual.size() && std::equal(bytes.begin(), bytes.end(), actual.begin() + at);
    };
    const std::string name = std::string("MLX ") + (tensor.affine() ? "" : tensor.mode + " as ") + fmtName(f);
    check(holds(step.plane0, expected.w0) && (!layout.plane1_bytes || holds(step.plane1, expected.w1)) &&
              holds(step.meta, expected.meta),
          name + ": planes from its codes and scales");
    check(holds(nativeAt, native), name + ": the token gather's native rows");
    // By value: MLX writes an mxfp4 code of -0 as -0, the kernels as +0.
    std::vector<float> expectedValues(tensor.values.size() / 4), values(expectedValues.size());
    std::memcpy(expectedValues.data(), tensor.values.data(), tensor.values.size());
    std::memcpy(values.data(), actual.data() + valuesAt, tensor.values.size());
    check(values == expectedValues, name + ": dequantized to MLX's F32 values");
  }
}

// The tensors of a reference NVFP4 or FP8 tensor (makeNative) as Model
// Optimizer and compressed-tensors store them, read back into the reference's
// native rows, its planes and its F32 values. NVFP4: U8 codes [rows, K / 2]
// and F8_E4M3 scales [rows, K / 16] with the tensor scale g, Model
// Optimizer's F32 weight_scale_2 beside its .weight, compressed-tensors' F32
// weight_global_scale 1 / g beside its .weight_packed; FP8: F8_E4M3 values
// [rows, K] with Model Optimizer's F32 weight_scale g or compressed-tensors'
// BF16 weight_scale [rows, 1], each row's g. And a transformers RMSNorm
// weight w, read as the F32 1 + w.
void checkFloatSources(MetalBackend &backend, const std::filesystem::path &directory) {
  std::mt19937 rng(7);
  for (const bool compressed : {false, true})
    for (const Fmt f : {NVFP4, FP8}) {
      const QuantFormat &layout = kQuantFormats[f];
      const uint32_t rows = 96, K = 1280, blocks = K / layout.block_elements;
      // The rows padded with zero rows to whole tiles.
      const uint32_t tiled = (rows + QUANT_TILE_ROWS - 1) / QUANT_TILE_ROWS * QUANT_TILE_ROWS;
      std::vector<uint8_t> native = makeNative(f, rows, K, rng);
      // compressed-tensors' g in every block: the F32 reciprocal of a global
      // scale (NVFP4), or each row's bf16 scale (FP8).
      const float global = std::uniform_real_distribution<float>(1e4f, 1e5f)(rng);
      std::vector<uint8_t> globalScale(4), rowScales(2 * rows);
      std::memcpy(globalScale.data(), &global, 4);
      for (uint32_t r = 0; r < rows; ++r) {
        const uint16_t bits = f2bf(std::uniform_real_distribution<float>(0.0005f, 0.004f)(rng));
        std::memcpy(rowScales.data() + 2 * r, &bits, 2);
      }
      if (compressed)
        setTensorScales(native, f, K, [&](uint32_t r) {
          return f == NVFP4 ? 1.0f / global : bf2f(uint16_t(rowScales[2 * r] | rowScales[2 * r + 1] << 8));
        });
      std::vector<uint8_t> codes, scales, scale(4);
      for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t b = 0; b < blocks; ++b) {
          const uint8_t *blk = native.data() + (size_t(r) * blocks + b) * layout.block_bytes;
          if (f == NVFP4) {
            scales.insert(scales.end(), blk, blk + 16);
            codes.insert(codes.end(), blk + kNvfp4Codes, blk + kNvfp4Codes + 128);
          } else {
            codes.insert(codes.end(), blk + kFp8Values, blk + kFp8Values + 256);
          }
        }
      std::memcpy(scale.data(), native.data() + (f == NVFP4 ? kNvfp4Scale : 0), 4);
      const std::string convention = compressed ? "compressed-tensors" : "Model Optimizer";
      const auto root = directory / ((compressed ? "compressed-tensors-" : "modelopt-") + std::string(fmtName(f)));
      using splash::test::SyntheticTensor;
      const std::vector<SyntheticTensor> shard =
          f == NVFP4 && compressed ? std::vector<SyntheticTensor>{{"m.weight_packed", "U8", {rows, K / 2}, codes},
                                                                  {"m.weight_scale", "F8_E4M3", {rows, K / 16}, scales},
                                                                  {"m.weight_global_scale", "F32", {1}, globalScale}}
          : f == NVFP4 ? std::vector<SyntheticTensor>{{"m.weight", "U8", {rows, K / 2}, codes},
                                                      {"m.weight_scale", "F8_E4M3", {rows, K / 16}, scales},
                                                      {"m.weight_scale_2", "F32", {}, scale}}
          : compressed ? std::vector<SyntheticTensor>{{"m.weight", "F8_E4M3", {rows, K}, codes},
                                                      {"m.weight_scale", "BF16", {rows, 1}, rowScales}}
                       : std::vector<SyntheticTensor>{{"m.weight", "F8_E4M3", {rows, K}, codes},
                                                      {"m.weight_scale", "F32", {}, scale}};
      splash::test::writeSyntheticShard(root / "model.safetensors", shard);
      const model::SafetensorsCheckpoint checkpoint(root);
      const model::SourceTensor &weight = checkpoint.require(f == NVFP4 && compressed ? "m.weight_packed" : "m.weight");
      const model::SourceTensor &g = checkpoint.require(f == FP8      ? "m.weight_scale"
                                                        : compressed ? "m.weight_global_scale"
                                                                     : "m.weight_scale_2");
      model::gguf::TensorRows source{"m", layout.ggml_type, 0, rows, rowBytes(f, K), {}, weight.file,
                                     {&weight, f == NVFP4 ? &checkpoint.require("m.weight_scale") : nullptr}};
      // compressed-tensors' NVFP4 holds 1 / g, and its FP8 each row's g.
      source.scale = {g.file, g.offset, g.bytes, compressed && f == FP8 ? 1 : rows, g.dtype == "BF16",
                      compressed && f == NVFP4};
      std::vector<uint8_t> tile = native;
      tile.resize(size_t(tiled) * rowBytes(f, K), 0);
      std::vector<float> reference;
      const Packed expected = repack(f, tile, tiled, K, &reference);
      reference.resize(size_t(rows) * K);

      model::gguf::ImageBuilder builder("float.bin", 0, 0);
      model::gguf::Repack planes = builder.planes(f, tiled, K, "m");
      planes.sources.push_back(source);
      const model::gguf::Repack step = planes;
      builder.repack(std::move(planes));
      const uint64_t nativeAt = builder.section(native.size());
      builder.copyAt(nativeAt, source, model::gguf::Conversion::None);
      const uint64_t valuesAt = builder.section(reference.size() * 4);
      builder.copyAt(valuesAt, source, model::gguf::Conversion::DequantizeToFloat32);
      const model::gguf::Image plan = builder.finish();
      const auto image = backend.allocateBuffer(plan.bytes, splash::metal::BufferStorage::Shared, "float-image");
      model::writeGgufImage(backend, image, plan);
      const auto actual = model::contentsOf(image);
      const auto holds = [&](uint64_t at, const std::vector<uint8_t> &bytes) {
        return at + bytes.size() <= actual.size() && std::equal(bytes.begin(), bytes.end(), actual.begin() + at);
      };
      const std::string name = convention + " " + fmtName(f);
      check(holds(step.plane0, expected.w0) && holds(step.meta, expected.meta), name + ": planes from its tensors");
      check(holds(nativeAt, native), name + ": native rows of its scales, tensor scale and codes");
      std::vector<float> values(reference.size());
      std::memcpy(values.data(), actual.data() + valuesAt, values.size() * 4);
      check(values == reference, name + ": dequantized to its F32 values");
    }
  // RMSNorm weights a transformers checkpoint stores 1 below the norm's: the
  // F32 1 + w, exact for these bf16 values.
  std::vector<uint8_t> weights(2 * 512);
  for (size_t i = 0; i < weights.size(); i += 2) {
    const uint16_t bits = f2bf(std::uniform_real_distribution<float>(-0.5f, 0.5f)(rng));
    std::memcpy(weights.data() + i, &bits, 2);
  }
  const auto root = directory / "centered-norm";
  splash::test::writeSyntheticShard(root / "model.safetensors", {{"n.weight", "BF16", {512}, weights}});
  const model::SafetensorsCheckpoint checkpoint(root);
  const model::SourceTensor &norm = checkpoint.require("n.weight");
  model::gguf::ImageBuilder builder("norm.bin", 0, 0);
  builder.copy({"n.weight", model::ggml::kBF16, norm.offset, 1, norm.bytes, {}, norm.file},
               model::gguf::Conversion::CenteredNorm);
  const model::gguf::Image plan = builder.finish();
  const auto image = backend.allocateBuffer(plan.bytes, splash::metal::BufferStorage::Shared, "norm-image");
  model::writeGgufImage(backend, image, plan);
  const auto actual = model::contentsOf(image);
  std::vector<float> expected(512), got(512);
  for (size_t i = 0; i < expected.size(); ++i) {
    uint16_t bits;
    std::memcpy(&bits, weights.data() + 2 * i, 2);
    expected[i] = 1.0f + bf2f(bits);
  }
  std::memcpy(got.data(), actual.data() + plan.copies.front().destination, got.size() * 4);
  check(got == expected, "transformers RMSNorm weight read as the F32 1 + w");
}

// llama.cpp's NVFP4 (ggml type 40) prepares into NVFP4's native rows: a
// token table of block_nvfp4 whose scales take every byte reads as llama.cpp
// reads it (blockNvfp4Values), and projections made from reference native
// rows (blockNvfp4Rows) prepare into those rows' planes with their .scale's g:
// a dense FFN projection and the GDN's grouped qkv rows with one value, the
// MoE's gate experts with one per expert. A .scale of another count, or
// beside a weight of another type, is refused.
void checkGgufNvfp4(MetalBackend &backend, const std::filesystem::path &directory) {
  std::mt19937 rng(41);
  struct Converted final {
    std::string name;
    uint32_t rows, columns, scales; // scales: the .scale tensor's values
    model::gguf::RowOrder order{};
    std::vector<uint8_t> native{};
  };
  // The planner's refusal of these tensors.
  const auto refusal = [](const std::filesystem::path &path, const std::vector<Tensor> &tensors,
                          const model::QwenTargetDimensions &g) {
    writeGguf(path, tensors, g);
    try {
      static_cast<void>(planned(path, g));
    } catch (const model::GgufError &error) {
      return std::string(error.what());
    }
    return std::string();
  };
  for (const bool moe : {false, true}) {
    SmallTarget target = smallTarget(moe);
    const model::QwenTargetDimensions &g = target.geometry;
    const uint32_t valueRows = g.gdnValueHeads * g.gdnHeadDimension;
    std::vector<Converted> tensors =
        moe ? std::vector<Converted>{{"blk.0.ffn_gate_exps.weight", g.experts * g.expertIntermediateSize, g.hiddenSize,
                                      g.experts}}
            : std::vector<Converted>{{"blk.0.ffn_up.weight", g.intermediateSize, g.hiddenSize, 1},
                                     {"blk.0.attn_qkv.weight", g.convolutionDimension, g.hiddenSize, 1,
                                      {g.convolutionDimension - valueRows, g.gdnHeadDimension, g.gdnKeyHeads,
                                       g.gdnValueHeads / g.gdnKeyHeads}}};
    for (Converted &c : tensors) {
      c.native = makeNative(NVFP4, c.rows, c.columns, rng);
      std::vector<float> scales(c.scales);
      for (float &scale : scales) scale = std::uniform_real_distribution<float>(0.5f, 2.0f)(rng);
      setTensorScales(c.native, NVFP4, c.columns, [&](uint32_t r) { return scales[r / (c.rows / c.scales)]; });
      Tensor &tensor = tensorNamed(target.tensors, c.name);
      tensor.type = model::ggml::kNVFP4;
      tensor.data = blockNvfp4Rows(c.native);
      std::vector<uint8_t> values(scales.size() * 4);
      std::memcpy(values.data(), scales.data(), values.size());
      target.tensors.push_back({c.name.substr(0, c.name.size() - 7) + ".scale", {c.scales}, 0, values});
    }
    std::vector<uint8_t> table;
    if (!moe) {
      Tensor &tokens = tensorNamed(target.tensors, "token_embd.weight");
      tokens.type = model::ggml::kNVFP4;
      tokens.data = table = blockNvfp4Fixture(g.vocabularySize, g.hiddenSize, 43);
    }
    const auto path = directory / (std::string(moe ? "moe" : "dense") + "-nvfp4.gguf");
    writeGguf(path, target.tensors, g);
    const auto images = planned(path, g);
    const auto prepared = preparedImages(backend, path, g);
    for (const Converted &c : tensors) {
      const model::gguf::Repack *planes = repackOf(images[0], c.name);
      const Packed expected =
          repack(NVFP4, orderedRows(c.native, rowBytes(NVFP4, c.columns), c.order), c.rows, c.columns, nullptr);
      check(planes && planes->format == GGUF_FMT_NVFP4 &&
                slice(prepared[0], planes->plane0, expected.w0.size()) == expected.w0 &&
                slice(prepared[0], planes->meta, expected.meta.size()) == expected.meta,
            "llama.cpp NVFP4 " + c.name + " prepares into the planes of its NVFP4 rows, its .scale as g");
    }
    if (!moe) {
      const model::gguf::Copy *copy = copyOf(images.back(), "token_embd.weight");
      const uint64_t stride = rowBytes(NVFP4, g.hiddenSize);
      const std::vector<uint8_t> native =
          copy ? slice(prepared.back(), copy->destination, g.vocabularySize * stride) : std::vector<uint8_t>{};
      std::vector<float> values(size_t(g.vocabularySize) * g.hiddenSize);
      for (uint32_t r = 0; copy && r < g.vocabularySize; ++r)
        rowValues(NVFP4, native.data() + r * stride, g.hiddenSize, values.data() + size_t(r) * g.hiddenSize);
      check(copy && values == blockNvfp4Values(table),
            "a llama.cpp NVFP4 token table prepares into native rows of its values, each scale byte read as "
            "llama.cpp reads it");
    }
    // A .scale of another count than one, or one per expert of an experts tensor.
    const std::string scale = moe ? "blk.0.ffn_gate_exps.scale" : "blk.0.ffn_up.scale";
    std::vector<Tensor> twoValues = target.tensors;
    tensorNamed(twoValues, scale) = {scale, {2}, 0, std::vector<uint8_t>(8, 0)};
    check(refusal(path, twoValues, g).find(scale + (moe ? " must hold one F32 value per expert for "
                                                        : " must hold one F32 value for ")) != std::string::npos,
          "a .scale tensor of another count than its weight's is refused: " + scale);
    if (moe) continue;
    // A .scale beside a weight of another type, which llama.cpp would apply.
    std::vector<Tensor> beside = target.tensors;
    beside.push_back({"blk.0.ffn_gate.scale", {1}, 0, std::vector<uint8_t>{0x00, 0x00, 0x80, 0x3F}});
    check(refusal(path, beside, g).find("blk.0.ffn_gate.scale scales a ") != std::string::npos,
          "a .scale tensor beside a weight of another type than NVFP4 is refused");
  }
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 4) {
      std::fprintf(stderr, "usage: gguf-preparation METALLIB GOLDENS MLX_FIXTURE\n");
      return 2;
    }
    const splash::test::TemporaryDirectory directory("splash-gguf-preparation");
    const Goldens hashes = goldens(argv[2], @"gguf_images");
    MetalBackend backend(argv[1]);
    // First, so that the peak allocation it checks against is its own.
    checkExecutor(backend, directory.path());
    guarded("preparation of the dense target", [&] { checkDense(backend, directory.path(), hashes); });
    guarded("preparation of the MoE target", [&] { checkMoe(backend, directory.path(), hashes); });
    guarded("preparation of BF16 alpha/beta", [&] { checkWidenedAlphaBeta(backend, directory.path()); });
    guarded("preparation of quantized alpha/beta", [&] { checkQuantizedAlphaBeta(backend, directory.path()); });
    guarded("llama.cpp NVFP4", [&] { checkGgufNvfp4(backend, directory.path()); });
    guarded("the target loader", [&] { checkDenseTarget(backend, directory.path()); });
    guarded("MLX sources", [&] { checkMlxSources(backend, directory.path(), argv[3]); });
    guarded("NVFP4 and FP8 sources", [&] { checkFloatSources(backend, directory.path()); });
    std::printf("%s (%d failures)\n", failures ? "GGUF preparation tests FAILED" : "GGUF preparation tests passed",
                failures);
    return failures ? 1 : 0;
  }
}
