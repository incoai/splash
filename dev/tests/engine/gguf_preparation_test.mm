// GGUF weight preparation on the GPU: every format's planes through the
// production executor against the CPU reference, the images the loader
// writes from the small dense and MoE targets, their golden hashes, offsets
// past 4 GiB, a restore of released images, the writer's chunk order, a load
// that fails mid-image, the target loader's reading of a GGUF, llama.cpp's
// NVFP4 tensors, the quantized tensors of MLX, Model Optimizer and
// compressed-tensors read from a safetensors checkpoint, and Inkling-2-Mini's
// checkpoints in its release's and its MLX converters' names and layouts,
// whose images are its GGUF's.
//   gguf-preparation METALLIB GOLDENS MLX_FIXTURE
// GOLDENS is dev/tests/fixtures/weight-goldens/goldens.json; its README says
// how to update it. MLX_FIXTURE is
// dev/tests/fixtures/mlx-quantization/fixture.json.
#include "GgufFixtures.hpp"
#include "Inkling2Fixtures.hpp"
#include "TestAdmission.hpp"
#include "TestCheckpoint.hpp"
#include "model/GgufPreparation.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/GgufTarget.hpp"
#include "model/Inkling2Loop.hpp"
#include "model/ModelSizing.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/SafetensorsTarget.hpp"
#include "model/Qwen3_8.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <tuple>
#include <utility>

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
  const model::GgufTargetPlan plan{geometry.layers, geometry.leadingDenseLayers,
                                   [&](const model::GgufFile &file, model::gguf::PlanOptions options) {
                                     return model::gguf::planImages(file, architecture(geometry), geometry, options);
                                   }};
  model::GgufTargetLoader loader(backend, images, {path}, plan);
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
  return model::gguf::planImages(gguf, architecture(geometry), geometry);
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
  splash::test::restoreWeights(images);
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

// alpha/beta of every GGUF tensor type of a format but BF16 and F16 on either
// architecture (llama.cpp's NVFP4 as NVFP4's native rows): one tensor of the
// beta then the alpha rows in grouped head order, then zero rows through the
// tile, every plane equal to the CPU reference's. Beside an alpha of another type (F32 on
// the dense target, the next format's on the MoE one), the beta and alpha rows
// in grouped head order as one F32 tensor of GGML's values, bit for bit. The
// expected row order is derived from the geometry, not from the plan.
void checkQuantizedAlphaBeta(MetalBackend &backend, const std::filesystem::path &directory) {
  std::vector<uint32_t> formats;
  for (uint32_t format = 0; format < GGUF_FMT_COUNT; ++format)
    if (ggufType(format)) formats.push_back(format);
  for (bool moe : {false, true})
    for (size_t index = 0; index < formats.size(); ++index) {
      const uint32_t format = formats[index];
      SmallTarget target = smallTarget(moe);
      const model::QwenTargetDimensions &g = target.geometry;
      // Grouped value head v of key head k is llama.cpp's tiled row v * keyHeads + k.
      const auto grouped = [&](const std::vector<uint8_t> &tiled, uint64_t stride) {
        std::vector<uint8_t> rows;
        for (uint32_t key = 0; key < g.gdnKeyHeads; ++key)
          for (uint32_t value = 0; value < g.gdnValueHeads / g.gdnKeyHeads; ++value) {
            const uint64_t source = (value * g.gdnKeyHeads + key) * stride;
            rows.insert(rows.end(), tiled.begin() + source, tiled.begin() + source + stride);
          }
        return rows;
      };
      // A tensor of format f's native rows, and their F32 values.
      uint32_t seed = 1200 + 2 * format;
      const auto store = [&](const char *name, uint32_t f, std::vector<uint8_t> &values) {
        Tensor &tensor = tensorNamed(target.tensors, name);
        tensor.type = *ggufType(f);
        std::vector<uint8_t> native = fixture(Fmt(f), g.gdnValueHeads, g.hiddenSize, ++seed);
        // llama.cpp's NVFP4 without a .scale tensor: g is 1.
        if (f == NVFP4) setTensorScales(native, NVFP4, g.hiddenSize, [](uint32_t) { return 1.0f; });
        tensor.data = f == NVFP4 ? blockNvfp4Rows(native) : native;
        values.resize(uint64_t{g.gdnValueHeads} * g.hiddenSize * sizeof(float));
        for (uint32_t row = 0; row < g.gdnValueHeads; ++row)
          rowValues(Fmt(f), native.data() + row * uint64_t{rowBytes(Fmt(f), g.hiddenSize)}, g.hiddenSize,
                    reinterpret_cast<float *>(values.data()) + uint64_t{row} * g.hiddenSize);
        return native;
      };
      const Fmt f = Fmt(format);
      const uint32_t stride = rowBytes(f, g.hiddenSize);
      std::vector<uint8_t> rows, betaValues, alphaValues;
      for (const auto &[name, values] :
           {std::pair{"blk.0.ssm_beta.weight", &betaValues}, std::pair{"blk.0.ssm_alpha.weight", &alphaValues}}) {
        const std::vector<uint8_t> ordered = grouped(store(name, format, *values), stride);
        rows.insert(rows.end(), ordered.begin(), ordered.end());
      }
      const std::string arch = architecture(g);
      // A pair of one 16-bit float type is a float tensor (checkFloatAlphaBeta).
      if (!float16(f)) {
        rows.resize(QUANT_TILE_ROWS * stride);
        const Packed expected = repack(f, rows, QUANT_TILE_ROWS, g.hiddenSize, nullptr);
        const auto path = directory / (arch + "-" + fmtName(format) + ".gguf");
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
                  arch + ")");
      }

      // The same beta beside an alpha of another type.
      std::string partner = "F32";
      if (moe) {
        const uint32_t next = formats[(index + 1) % formats.size()];
        static_cast<void>(store("blk.0.ssm_alpha.weight", next, alphaValues));
        partner = fmtName(next);
      } else {
        Tensor &alpha = tensorNamed(target.tensors, "blk.0.ssm_alpha.weight");
        alpha.type = model::ggml::kF32;
        alpha.data = alphaValues = floatValues(uint64_t{g.gdnValueHeads} * g.hiddenSize, ++seed);
      }
      const uint64_t floatStride = uint64_t{g.hiddenSize} * sizeof(float);
      std::vector<uint8_t> values = grouped(betaValues, floatStride);
      const std::vector<uint8_t> alphaRows = grouped(alphaValues, floatStride);
      values.insert(values.end(), alphaRows.begin(), alphaRows.end());
      const auto mixedPath = directory / (arch + "-" + fmtName(format) + "-" + partner + ".gguf");
      writeGguf(mixedPath, target.tensors, g);
      const auto *beta = copyOf(planned(mixedPath, g)[0], "blk.0.ssm_beta.weight");
      check(beta && slice(preparedImages(backend, mixedPath, g)[0], beta->destination, values.size()) == values,
            std::string("prepared ") + fmtName(format) + " beta beside a " + partner +
                " alpha: GGML's F32 values of both, beta then alpha rows (" + arch + ")");
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

  const auto fusedPath = directory / "moe-fused.gguf";
  writeGguf(fusedPath, withFusedGateUp(target.tensors, 0, g.experts), g);
  check(preparedImages(backend, fusedPath, g) == preparedImages(backend, path, g),
        "llama.cpp's fused ffn_gate_up_exps prepares into the images of the gate and up tensors it holds");

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

// The MoE layer with its router, shared-expert scalar gate and norms in 16-bit
// floats, as some GGUFs store them, but the scalar gate in `gate`, BF16 or
// Q8_0: the BF16 router's rows as stored, which the float projection runs, and
// the F32 values the scalar gate (widened, or decoded on the GPU) and the F16
// and BF16 norms hold.
void checkSixteenBitFloats(MetalBackend &backend, const std::filesystem::path &directory, Fmt gate) {
  SmallTarget target = smallTarget(true);
  const model::QwenTargetDimensions &g = target.geometry;
  const auto path = directory / "moe-16-bit-floats.gguf";
  const struct Case {
    std::string name;
    Fmt format;
    uint32_t rows;
    bool asStored;
  } cases[] = {{"blk.0.ffn_gate_inp.weight", BF16, g.experts, true},
               {"blk.0.ffn_gate_inp_shexp.weight", gate, 1, false},
               {"blk.0.attn_norm.weight", F16, 1, false},
               {"blk.0.post_attention_norm.weight", BF16, 1, false}};
  std::map<std::string, std::vector<uint8_t>> expected;
  uint32_t seed = 980;
  for (const Case &c : cases) {
    Tensor &tensor = tensorNamed(target.tensors, c.name);
    tensor.type = *ggufType(c.format);
    tensor.data = fixture(c.format, c.rows, g.hiddenSize, ++seed);
    std::vector<uint8_t> &bytes = expected[c.name];
    if (c.asStored) {
      bytes = tensor.data;
      continue;
    }
    bytes.resize(uint64_t{c.rows} * g.hiddenSize * sizeof(float));
    for (uint32_t row = 0; row < c.rows; ++row)
      rowValues(c.format, tensor.data.data() + row * uint64_t{rowBytes(c.format, g.hiddenSize)}, g.hiddenSize,
                reinterpret_cast<float *>(bytes.data()) + uint64_t{row} * g.hiddenSize);
  }
  writeGguf(path, target.tensors, g);
  const std::vector<model::gguf::Image> images = planned(path, g);
  const std::vector<uint8_t> image = preparedImages(backend, path, g).front();
  for (const Case &c : cases) {
    const auto *copy = copyOf(images[0], c.name);
    using model::gguf::Conversion;
    const auto conversion = c.asStored        ? Conversion::None
                            : c.format == Q80 ? Conversion::DequantizeToFloat32
                                              : Conversion::WidenToFloat32;
    check(copy && copy->conversion == conversion && slice(image, copy->destination, expected[c.name].size()) ==
                                                         expected[c.name],
          std::string("prepared ") + fmtName(c.format) + " " + c.name +
              (c.asStored ? ": its rows as stored" : ": the F32 values it holds"));
  }
}

// The MoE layer with a router of a quantized type (Q8_0, Q4_K): its native
// rows as stored, which the float projection decodes, as an MLX checkpoint's
// quantized router loads.
void checkQuantizedRouter(MetalBackend &backend, const std::filesystem::path &directory) {
  for (const Fmt f : {Q80, Q4K}) {
    SmallTarget target = smallTarget(true);
    const model::QwenTargetDimensions &g = target.geometry;
    const auto path = directory / (std::string("moe-router-") + fmtName(f) + ".gguf");
    const std::string name = "blk.0.ffn_gate_inp.weight";
    Tensor &tensor = tensorNamed(target.tensors, name);
    tensor.type = *ggufType(f);
    tensor.data = fixture(f, g.experts, g.hiddenSize, 990 + f);
    writeGguf(path, target.tensors, g);
    const std::vector<model::gguf::Image> images = planned(path, g);
    const std::vector<uint8_t> image = preparedImages(backend, path, g).front();
    const auto *copy = copyOf(images[0], name);
    check(copy && copy->conversion == model::gguf::Conversion::None &&
              slice(image, copy->destination, tensor.data.size()) == tensor.data,
          std::string("prepared ") + fmtName(f) + " router: its native rows as stored");
  }
}

// MXFP8's block of 32 values by a search of every E8M0 exponent, the smallest
// first, for one at which each value is an E4M3 code's (the reference's
// e4m3); none when no exponent holds them all.
std::optional<std::array<uint8_t, 33>> searchedMxfp8Block(const float *values) {
  std::map<float, uint8_t> codes; // each positive finite E4M3 value's code, 0 among them
  for (uint32_t code = 0; code < 0x7F; ++code) codes.emplace(gguf_reference::e4m3(uint8_t(code)), uint8_t(code));
  for (int e = 0; e < 255; ++e) {
    std::array<uint8_t, 33> block{uint8_t(e)};
    bool held = true;
    for (uint32_t i = 0; held && i < 32; ++i) {
      const auto found = codes.find(std::fabs(float(std::ldexp(double(values[i]), 127 - e))));
      held = found != codes.end() && std::ldexp(double(found->first), e - 127) == std::fabs(double(values[i]));
      if (held) block[1 + i] = uint8_t(found->second | (std::signbit(values[i]) ? 0x80 : 0));
    }
    if (held) return block;
  }
  return std::nullopt;
}

// gguf::mxfp8Block against the search of every exponent: MXFP8 blocks of
// random codes at each E8M0 exponent whose values a float holds, half of
// them of small codes (whose largest the smallest exponents hold below
// E4M3's range); each also with one value given a fifth significant bit,
// which no exponent holds, and with one value 2^-10 of its exponent's scale;
// zeros of either sign, and a block with an infinity or a NaN.
void checkMxfp8Blocks() {
  std::mt19937 rng(1420);
  uint32_t blocks = 0, agreed = 0, exact = 0;
  const auto agree = [&](const std::array<float, 32> &values) {
    std::array<uint8_t, 33> block{};
    const bool held = model::gguf::mxfp8Block(values, block);
    const std::optional<std::array<uint8_t, 33>> searched = searchedMxfp8Block(values.data());
    ++blocks;
    exact += held;
    agreed += held == searched.has_value() && (!held || block == *searched);
  };
  for (int e = 0; e < 255; ++e)
    for (int trial = 0; trial < 4; ++trial) {
      std::array<float, 32> values;
      for (float &value : values) {
        uint8_t code = uint8_t(rng() % 0x7F | (rng() % 2 ? 0x80 : 0));
        if (trial % 2) code &= 0x8F;
        value = float(std::ldexp(double(gguf_reference::e4m3(code)), e - 127));
      }
      if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); })) continue;
      agree(values);
      std::array<float, 32> wider = values, smaller = values;
      const size_t at = rng() % 32;
      if (values[at] != 0.0F)
        wider[at] = float(values[at] + std::ldexp(std::copysign(1.0, values[at]), std::ilogb(values[at]) - 4));
      smaller[at] = float(std::ldexp(1.0, e - 127 - 10));
      agree(wider);
      agree(smaller);
    }
  std::array<float, 32> zeros{};
  for (size_t i = 0; i < zeros.size(); i += 2) zeros[i] = -0.0F;
  agree(zeros);
  for (const float special : {INFINITY, NAN}) {
    std::array<float, 32> values{};
    values[5] = special;
    agree(values);
  }
  check(agreed == blocks && exact > blocks / 3 && exact < blocks * 2 / 3,
        "MXFP8 blocks take the exponent and codes a search of every exponent finds, " + std::to_string(exact) +
            " of " + std::to_string(blocks) + " exact");
}

// A MoE layer's routed experts of BF16 values, 320 inputs wide but down's 256,
// as Unsloth's UD-Q8_K_XL widens a release's MXFP8 experts: of MXFP8's values
// (the reference's MXFP8 rows), each tensor MXFP8 planes of the blocks a
// search of every exponent finds; of other values, BF16 planes as stored where
// a tensor's rows are whole 256-value blocks (down) and the reference
// quantization's Q8_0 where they are not (gate and up). A plan without the
// loader's look at the values (exactlyMxfp8), as of a header alone, takes the
// latter for both.
void checkFloatExperts(MetalBackend &backend, const std::filesystem::path &directory) {
  SmallTarget target = smallTarget(true);
  model::QwenTargetDimensions &g = target.geometry;
  g.hiddenSize = 320;
  TensorTypes types;
  for (const char *name : {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "ffn_gate_shexp.weight",
                           "ffn_up_shexp.weight", "ffn_down_shexp.weight", "output.weight", "token_embd.weight"})
    types[name] = kQ8_0;
  for (const char *name : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"})
    types[name] = *ggufType(BF16);
  target.tensors = targetTensors(g, types);
  randomize(target.tensors, 1400);
  const uint32_t width = g.expertIntermediateSize;
  const struct Expert {
    std::string name;
    uint32_t rows, columns;
  } experts[] = {{"blk.0.ffn_gate_exps.weight", g.experts * width, g.hiddenSize},
                 {"blk.0.ffn_up_exps.weight", g.experts * width, g.hiddenSize},
                 {"blk.0.ffn_down_exps.weight", g.experts * g.hiddenSize, width}};
  for (const bool mxfp8 : {true, false}) {
    std::mt19937 rng(1410);
    std::map<std::string, Packed> expected;
    for (const Expert &e : experts) {
      Tensor &tensor = tensorNamed(target.tensors, e.name);
      std::vector<float> values(uint64_t{e.rows} * e.columns);
      if (mxfp8) {
        const std::vector<uint8_t> native = makeNative(MXFP8, e.rows, e.columns, rng);
        for (uint32_t row = 0; row < e.rows; ++row)
          rowValues(MXFP8, native.data() + uint64_t{row} * rowBytes(MXFP8, e.columns), e.columns,
                    values.data() + uint64_t{row} * e.columns);
        for (size_t i = 0; i < values.size(); ++i) {
          const uint16_t bits = f2bf(values[i]);
          std::memcpy(tensor.data.data() + 2 * i, &bits, 2);
          if (bf2f(bits) != values[i]) throw std::runtime_error("an MXFP8 value no BF16 holds");
        }
      } else {
        tensor.data = fixture(BF16, e.rows, e.columns, uint32_t(rng()));
        for (size_t i = 0; i < values.size(); ++i) {
          uint16_t bits;
          std::memcpy(&bits, tensor.data.data() + 2 * i, 2);
          values[i] = bf2f(bits);
        }
      }
      std::vector<uint8_t> rows;
      if (mxfp8) {
        for (size_t at = 0; at < values.size(); at += 32) {
          const auto block = searchedMxfp8Block(values.data() + at);
          if (!block) throw std::runtime_error("no exponent holds an MXFP8 block");
          rows.insert(rows.end(), block->begin(), block->end());
        }
      }
      const Fmt format = mxfp8 ? MXFP8 : e.columns % 256 ? Q80 : BF16;
      expected[e.name] = repack(format, mxfp8 ? rows : format == Q80 ? quantizeQ8_0(values) : tensor.data, e.rows,
                                e.columns, nullptr);
    }
    const auto path = directory / (std::string("moe-bf16-experts-") + (mxfp8 ? "mxfp8" : "other") + ".gguf");
    writeGguf(path, target.tensors, g);
    model::WeightSource source(path);
    const model::GgufFile gguf(source);
    const std::vector<model::gguf::Image> images =
        model::gguf::planImages(gguf, architecture(g), g, {.exactlyMxfp8 = model::exactlyMxfp8});
    const std::vector<model::gguf::Image> headerOnly = model::gguf::planImages(gguf, architecture(g), g);
    const std::vector<uint8_t> image = preparedImages(backend, path, g).front();
    for (const Expert &e : experts) {
      const model::gguf::Repack *planes = repackOf(images[0], e.name), *fallback = repackOf(headerOnly[0], e.name);
      const Fmt format = mxfp8 ? MXFP8 : e.columns % 256 ? Q80 : BF16;
      const Packed &want = expected[e.name];
      check(planes && planes->format == format && fallback && fallback->format == (e.columns % 256 ? Q80 : BF16) &&
                slice(image, planes->plane0, want.w0.size()) == want.w0 &&
                slice(image, planes->meta, want.meta.size()) == want.meta,
            std::string("BF16 experts of ") + (mxfp8 ? "MXFP8's" : "other") + " values prepare into " +
                fmtName(format) + " planes: " + e.name);
    }
  }
}

// The MoE layer with BF16 or F16 alpha/beta: a BF16 pair stays one BF16
// tensor of their rows as stored, which the float projection reads as it
// reads their F32 values; preparation decodes an F16 pair on the GPU into
// one F32 tensor of the values they equal. Both at the target's width and
// 320 values wide, whole Q8_0 blocks (its projections' format there) but
// not whole 256-value blocks, which rows as stored need not be.
void checkFloatAlphaBeta(MetalBackend &backend, const std::filesystem::path &directory) {
  for (const uint32_t hidden : {512u, 320u})
    for (const Fmt f : {BF16, F16}) {
      SmallTarget target = smallTarget(true);
      model::QwenTargetDimensions &g = target.geometry;
      if (hidden != g.hiddenSize) {
        g.hiddenSize = hidden;
        TensorTypes types;
        for (const char *name : {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "ffn_gate_exps.weight",
                                 "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight",
                                 "ffn_up_shexp.weight", "ffn_down_shexp.weight", "output.weight", "token_embd.weight"})
          types[name] = kQ8_0;
        target.tensors = targetTensors(g, types);
        randomize(target.tensors, 1300);
      }
      const std::string what = std::string(fmtName(f)) + " alpha/beta " + std::to_string(hidden) + " wide";
      const auto path = directory / ("moe-" + std::string(fmtName(f)) + "-" + std::to_string(hidden) + ".gguf");
      const bool stored = f == BF16;
      std::map<std::string, std::vector<uint8_t>> rows; // the BF16 rows as stored, or the F32 values of the F16 ones
      uint32_t seed = 960;
      for (const std::string name : {"blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight"}) {
        Tensor &tensor = tensorNamed(target.tensors, name);
        tensor.type = *ggufType(f);
        tensor.data = fixture(f, g.gdnValueHeads, g.hiddenSize, ++seed);
        std::vector<uint8_t> &values = rows[name];
        if (stored) {
          values = tensor.data;
          continue;
        }
        values.resize(uint64_t{g.gdnValueHeads} * g.hiddenSize * sizeof(float));
        for (uint32_t row = 0; row < g.gdnValueHeads; ++row)
          rowValues(f, tensor.data.data() + row * uint64_t{rowBytes(f, g.hiddenSize)}, g.hiddenSize,
                    reinterpret_cast<float *>(values.data()) + uint64_t{row} * g.hiddenSize);
      }
      writeGguf(path, target.tensors, g);
      const std::vector<model::gguf::Image> images = planned(path, g);
      const auto *beta = copyOf(images[0], "blk.0.ssm_beta.weight");
      const auto *alpha = copyOf(images[0], "blk.0.ssm_alpha.weight");
      if (!beta || !alpha) throw std::runtime_error("the plan has no alpha/beta tensor");
      const auto conversion = stored ? model::gguf::Conversion::None : model::gguf::Conversion::DequantizeToFloat32;
      check(beta->conversion == conversion && alpha->conversion == conversion &&
                alpha->destination == beta->destination + rows[beta->source.name].size(),
            "planner makes " + what + " one " + (stored ? "BF16" : "F32") + " tensor");
      const std::vector<uint8_t> image = preparedImages(backend, path, g).front();
      const uint64_t rowSize = uint64_t{g.hiddenSize} * (stored ? sizeof(uint16_t) : sizeof(float));
      std::vector<uint8_t> gates = orderedRows(rows[beta->source.name], rowSize, beta->source.order);
      const auto alphaRows = orderedRows(rows[alpha->source.name], rowSize, alpha->source.order);
      gates.insert(gates.end(), alphaRows.begin(), alphaRows.end());
      check(slice(image, beta->destination, gates.size()) == gates,
            "prepared " + what + ": " + (stored ? "their rows as stored" : "their F32 values") +
                ", beta then alpha rows in grouped order");
    }
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
// Its FFN gate is F16 and its up Q4_K, a gate/up pair the target geometry takes.
void checkDenseTarget(MetalBackend &backend, const std::filesystem::path &directory) {
  model::Qwen3_8Layout layout;
  layout.layers = 4;
  layout.hiddenSize = 256;
  layout.vocabularySize = 256;
  // Not the Qwen tokenizer's 248,320 tokens: its draft proposes from all of them.
  layout.draftVocabulary = {};
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
                                   {"ffn_gate.weight", model::ggml::kF16},
                                   {"ffn_up.weight", kQ4_K},
                                   {"ffn_down.weight", kQ6_K},
                                   {"output.weight", kQ6_K},
                                   {"token_embd.weight", kIQ4_XS}}),
            layout);
  model::WeightImages images(backend, "fixture");
  model::GgufTargetLoader files(backend, images, model::findTargetGgufs(target), model::ggufTargetPlan(layout));
  const model::Qwen3_8Weights weights = model::loadQwen3_8Weights(backend, layout, files);
  check(weights.layers.size() == layout.layers, "GGUF target: every layer");
  check(weights.finalNorm.float32, "GGUF target: F32 final norm");
  check(blockProjection(weights.logitsProjection, layout.vocabularySize, hidden, {layout.vocabularySize}) &&
            std::string_view(weights.logitsProjection.blocks().segments.front().name()) == "q6k" &&
            weights.logitsProjection.destination == ops::FloatOutput::Float32,
        "GGUF target: logits a Q6_K block projection of vocabulary x hidden into fp32");
  check(weights.tokenEmbedding.rows().formatId == GGUF_FMT_IQ4XS &&
            weights.tokenEmbedding.outputSize == layout.vocabularySize && weights.tokenEmbedding.inputSize == hidden,
        "GGUF target: token table IQ4_XS blocks of vocabulary x hidden");
  const std::array<uint32_t, 2> captured{1, layout.layers - 1};
  const model::QwenTargetGeometry geometry = model::qwenTargetGeometry(weights, captured);
  check(geometry.valid(), "GGUF target: a valid target geometry");
  {
    // Read over released images, each reserved, the target holds no memory
    // and has the geometry of its load, from which a start plans its arenas
    // before loading it (model::plannedTargetGeometry).
    const uint64_t before = backend.memoryStats().allocatedBytes;
    model::WeightImages reserved(backend, "fixture");
    reserved.release();
    model::GgufTargetLoader planned(backend, reserved, model::findTargetGgufs(target), model::ggufTargetPlan(layout));
    const model::QwenTargetGeometry read =
        model::qwenTargetGeometry(model::loadQwen3_8Weights(backend, layout, planned), captured);
    check(backend.memoryStats().allocatedBytes == before && read.prefillProjections == geometry.prefillProjections &&
              read.decodeProjections == geometry.decodeProjections &&
              read.gateUpProjections == geometry.gateUpProjections && read.moeRouterFormat == geometry.moeRouterFormat,
          "GGUF target: its reserved images held memory, or gave another geometry than its load");
  }
  for (uint32_t index = 0; index < weights.layers.size(); ++index) {
    const auto &layer = weights.layers[index];
    const std::string at = "GGUF target layer " + std::to_string(index) + ": ";
    check(layer.inputNorm.float32 && layer.postAttentionNorm.float32, at + "F32 input and post-attention norms");
    check(blockProjection(layer.gateProjection, layout.intermediateSize, hidden, {layout.intermediateSize}) &&
              layer.gateProjection.blocks().segments.front().formatId == GGUF_FMT_F16,
          at + "gate F16 block projection");
    check(blockProjection(layer.upProjection, layout.intermediateSize, hidden, {layout.intermediateSize}) &&
              layer.upProjection.blocks().segments.front().formatId == GGUF_FMT_Q4K,
          at + "up Q4_K block projection");
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

// Row tiles are whole (the planner requires whole plane tiles of rows);
// the shapes cover several tiles, super-blocks and groups, and a permuted row range.
// A tensor of `windows` column windows is [rows, windows * K], repacked as the
// windows' rows in turn (model::gguf::TensorRows::windows).
struct Shape {
  uint32_t rows, K;
  model::gguf::RowOrder order;
  uint32_t windows = 1;
};

// The narrowest K whose tile of source rows exceeds the repack staging, so
// preparation must split the rows by columns.
uint32_t widerThanStaging(Fmt f) {
  const uint64_t perBlock = model::ggufRowBytes(kQuantFormats[f], model::kGgufBlockColumns);
  return uint32_t((model::kGgufRepackStagingBytes / (QUANT_TILE_ROWS * perBlock) + 1) * model::kGgufBlockColumns);
}

// One quantized tensor through the production executor, from a file whose
// data section starts at kSourceOffset, against the CPU reference's planes of
// its rows in image order, window after window. With blockNvfp4 the file
// holds an NVFP4 tensor as llama.cpp's NVFP4 rows (blockNvfp4Rows), which the
// reader converts window by window (TensorRows::ggufNvfp4RowBytes).
void checkRepack(MetalBackend &backend, const std::filesystem::path &directory, Fmt f, const Shape &shape,
                 uint32_t seed, bool blockNvfp4 = false) {
  const QuantFormat &layout = kQuantFormats[f];
  const uint32_t windows = shape.windows, rows = shape.rows * windows, K = shape.K;
  const uint32_t stride = rowBytes(f, windows * K), windowBytes = rowBytes(f, K);
  std::vector<uint8_t> native = fixture(f, shape.rows, windows * K, seed);
  // llama.cpp's NVFP4 without a .scale tensor: g is 1.
  if (blockNvfp4) setTensorScales(native, f, windows * K, [](uint32_t) { return 1.0f; });
  const std::vector<uint8_t> ordered = orderedRows(native, stride, shape.order);
  std::vector<uint8_t> windowRows;
  for (uint32_t window = 0; window < windows; ++window)
    for (uint32_t row = 0; row < shape.rows; ++row) {
      const auto at = ordered.begin() + uint64_t{row} * stride + uint64_t{window} * windowBytes;
      windowRows.insert(windowRows.end(), at, at + windowBytes);
    }
  const Packed expected = repack(f, windowRows, rows, K, nullptr);
  const uint64_t plane0 = model::kWeightFileAlignment, plane1 = model::alignWeightOffset(plane0 + expected.w0.size());
  const uint64_t meta = layout.plane1_bytes ? model::alignWeightOffset(plane1 + expected.w1.size()) : plane1;
  const uint64_t bytes = model::alignWeightOffset(meta + expected.meta.size()) + model::kWeightFileAlignment;
  const std::string what = std::string("prepared ") + fmtName(f) + " rows=" + std::to_string(rows) +
                           " K=" + std::to_string(K) + (shape.order.from == UINT64_MAX ? "" : " permuted") +
                           (windows == 1 ? "" : " windows=" + std::to_string(windows)) +
                           (blockNvfp4 ? " from llama.cpp NVFP4" : "");
  const auto inputPath = directory / "repack-source";
  try {
    std::vector<uint8_t> input(kSourceOffset, 0);
    const std::vector<uint8_t> stored = blockNvfp4 ? blockNvfp4Rows(native, f) : native;
    input.insert(input.end(), stored.begin(), stored.end());
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
    for (uint32_t window = 0; window < windows; ++window) {
      model::gguf::TensorRows sources{"fixture", layout.ggml_type, 0, shape.rows, stride, shape.order, &source};
      sources.windows = windows;
      sources.window = window;
      if (blockNvfp4) sources.ggufNvfp4RowBytes = uint64_t{windows} * K / 64 * kBlockNvfp4Bytes;
      step.sources.push_back(sources);
    }
    model::gguf::Image plan;
    plan.bytes = bytes;
    plan.repacks.push_back(step);
    const auto image = backend.allocateBuffer(bytes, splash::metal::BufferStorage::Shared, "repack-output");
    // The writer writes every byte: none of these is left.
    std::memset(image.contents(), 0xFF, bytes);
    const auto before = backend.memoryStats();
    model::writeGgufImage(backend, image, plan);
    const auto after = backend.memoryStats();
    check(after.allocatedBytes == before.allocatedBytes, "repack releases its staging buffers");
    check(after.peakAllocatedBytes <=
              std::max(before.peakAllocatedBytes, before.allocatedBytes + 2 * model::kGgufRepackStagingBytes),
          "repack staging stays within two repack stagings");
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
  // Widths of 64-column spans not a whole superblock, in every format whose
  // width is a multiple of 64; an odd number of 64-row tiles by an odd number
  // of spans leaves half a threadgroup of (row, group) items in the last one.
  for (int f = 0; f < FMT_COUNT; ++f)
    if (model::ggufColumnUnit(kQuantFormats[f]) == 64)
      for (const Shape &shape : {Shape{192, 320, {}}, Shape{64, 64, {}}, Shape{192, 640, {}}})
        checkRepack(backend, directory, Fmt(f), shape, 300 + f + shape.K);
  // Several bounded row batches, and rows wider than one staging step.
  for (Fmt format : {Q3K, Q80}) {
    checkRepack(backend, directory, format, {8704, 2048, {}}, 741);
    checkRepack(backend, directory, format, {256, widerThanStaging(format), {}}, 742);
  }
  // Column windows, a shared expert's down projection split into its slices
  // (Flint's 2048 rows of two 512-input windows) in every format whose blocks
  // a window holds whole, permuted rows too, windows wider than one staging
  // step, and windows of llama.cpp's NVFP4 rows.
  for (int f = 0; f < FMT_COUNT; ++f)
    if (512 % model::ggufColumnUnit(kQuantFormats[f]) == 0) {
      checkRepack(backend, directory, Fmt(f), {2048, 512, {}, 2}, 500 + f);
      checkRepack(backend, directory, Fmt(f), {768, 512, {256, 16, 8, 4}, 3}, 600 + f);
    }
  checkRepack(backend, directory, Q80, {256, widerThanStaging(Q80), {}, 2}, 743);
  checkRepack(backend, directory, NVFP4, {2048, 512, {}, 2}, 744, true);
  checkRepack(backend, directory, NVFP4, {768, 512, {256, 16, 8, 4}, 3}, 745, true);
  // llama.cpp's NVFP4 rows that are not whole 256-element blocks (an expert down projection of 1152
  // inputs) as NVFP4's form of 64-element blocks.
  checkRepack(backend, directory, NVFP464, {256, 1152, {}}, 746, true);
}

// The writer's chunk order (model::pipelineChunks), with tickets that record
// what happens to them: chunk i is staged while chunk i - 1's command runs,
// and that command is waited for before chunk i's is submitted, one in
// flight. A stage that throws leaves the command in flight to its ticket's
// destruction (a CommandTicket's waits for the GPU) and submits nothing more.
void checkPipelineOrder() {
  std::vector<std::string> events;
  struct Ticket {
    std::vector<std::string> *events;
    size_t chunk;
    bool ended = false;
    Ticket(std::vector<std::string> *events, size_t chunk) : events(events), chunk(chunk) {}
    Ticket(Ticket &&other) noexcept : events(std::exchange(other.events, nullptr)), chunk(other.chunk),
                                      ended(other.ended) {}
    Ticket &operator=(Ticket &&) = delete;
    ~Ticket() {
      if (events && !ended) events->push_back("destroy " + std::to_string(chunk));
    }
    int wait() {
      ended = true;
      events->push_back("wait " + std::to_string(chunk));
      return 0;
    }
  };
  const auto run = [&](size_t count, size_t failing) {
    events.clear();
    try {
      model::pipelineChunks(
          count,
          [&](size_t chunk, unsigned buffer) {
            events.push_back("stage " + std::to_string(chunk) + " in " + std::to_string(buffer));
            if (chunk == failing) throw std::runtime_error("stage failed");
          },
          [&](size_t chunk, unsigned buffer) {
            events.push_back("submit " + std::to_string(chunk) + " from " + std::to_string(buffer));
            return Ticket(&events, chunk);
          });
    } catch (const std::runtime_error &) {
      events.push_back("failed");
    }
  };
  run(3, SIZE_MAX);
  check(events == std::vector<std::string>{"stage 0 in 0", "submit 0 from 0", "stage 1 in 1", "wait 0",
                                           "submit 1 from 1", "stage 2 in 0", "wait 1", "submit 2 from 0", "wait 2"},
        "the next chunk is staged while the last chunk's command runs, one command in flight");
  run(3, 2);
  check(events == std::vector<std::string>{"stage 0 in 0", "submit 0 from 0", "stage 1 in 1", "wait 0",
                                           "submit 1 from 1", "stage 2 in 0", "destroy 1", "failed"},
        "a stage that fails ends the command in flight by its ticket and submits no other");
  run(0, SIZE_MAX);
  check(events.empty(), "no chunk, no command");
}

// A source that ends inside an image's second chunk fails its load while the
// first chunk's command may still be in flight: the load fails naming the
// truncation, keeps no image, and gives back the image's memory and both
// stagings once that command ended.
void checkFailedLoad(MetalBackend &backend, const std::filesystem::path &directory) {
  const Fmt f = Q80;
  const uint32_t rows = 32768, K = 2048; // 71 MB of rows: three chunks of the 32 MiB staging
  const QuantFormat &layout = kQuantFormats[f];
  const uint64_t stride = rowBytes(f, K);
  const std::vector<uint8_t> native = fixture(f, rows, K, 901);
  const auto path = directory / "truncated-source";
  std::vector<uint8_t> input(kSourceOffset, 0);
  input.insert(input.end(), native.begin(), native.begin() + native.size() / 2);
  splash::test::writeFile(path, input);
  model::WeightSource source(path);
  source.setDataOffset(kSourceOffset);
  const auto planes = model::ggufPlaneBytes(layout, rows, K);
  auto plan = std::make_shared<model::gguf::Image>();
  plan->name = "truncated.bin";
  model::gguf::Repack step;
  step.format = f;
  step.rows = rows;
  step.columns = K;
  step.plane0 = model::kWeightFileAlignment;
  step.plane1 = 0;
  step.meta = model::alignWeightOffset(step.plane0 + planes.plane0);
  step.sources.push_back({"fixture", layout.ggml_type, 0, rows, stride, {}, &source});
  plan->repacks.push_back(step);
  plan->bytes = model::alignWeightOffset(step.meta + planes.meta) + model::kWeightFileAlignment;
  plan->residentBytes = plan->bytes;
  model::WeightImages images(backend, "fixture");
  const uint64_t before = backend.memoryStats().allocatedBytes;
  std::string error;
  try {
    static_cast<void>(model::loadImage(backend, images, "", plan, [] {}));
  } catch (const std::exception &failure) {
    error = failure.what();
  }
  check(error.find("truncated") != std::string::npos, "a source truncated inside the second chunk fails the load");
  check(images.contents().empty(), "a load that failed keeps no image");
  check(backend.memoryStats().allocatedBytes == before, "a load that failed gives back its image and stagings");
}

// Each MLX fixture tensor through the production writer, read from a
// safetensors checkpoint as the planner binds it (model/SafetensorsImage.hpp):
// its planes in whole plane tiles, the native rows the token gather and the
// float projection read and, for an affine tensor, the F32 values its
// dequantization writes (a shared-expert gate's, GDN alpha's or beta's),
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
    // The tensor as two column windows (a shared expert's down projection in
    // its slices): window 1's rows after window 0's, from whole rows read.
    if ((K / 2) % model::ggufColumnUnit(layout) == 0) {
      const uint64_t stride = rowBytes(f, K), half = rowBytes(f, K / 2);
      std::vector<uint8_t> windowRows;
      for (uint64_t window = 0; window < 2; ++window)
        for (uint64_t row = 0; row < rows; ++row) {
          const auto at = native.begin() + static_cast<std::ptrdiff_t>(row * stride + window * half);
          windowRows.insert(windowRows.end(), at, at + static_cast<std::ptrdiff_t>(half));
        }
      windowRows.resize(size_t(2 * QUANT_TILE_ROWS) * half, 0);
      const Packed windowed = repack(f, windowRows, 2 * QUANT_TILE_ROWS, K / 2, nullptr);
      model::gguf::ImageBuilder windowBuilder("mlx-windows.bin", 0, 0);
      model::gguf::Repack windowPlanes = windowBuilder.planes(f, 2 * QUANT_TILE_ROWS, K / 2, "m");
      for (uint32_t window = 0; window < 2; ++window) {
        model::gguf::TensorRows slice = source;
        slice.windows = 2;
        slice.window = window;
        windowPlanes.sources.push_back(slice);
      }
      const model::gguf::Repack windowStep = windowPlanes;
      windowBuilder.repack(std::move(windowPlanes));
      const model::gguf::Image windowPlan = windowBuilder.finish();
      const auto windowImage = backend.allocateBuffer(windowPlan.bytes, splash::metal::BufferStorage::Shared,
                                                      "mlx-windows-image");
      model::writeGgufImage(backend, windowImage, windowPlan);
      const auto written = model::contentsOf(windowImage);
      const auto windowHolds = [&](uint64_t at, const std::vector<uint8_t> &bytes) {
        return at + bytes.size() <= written.size() && std::equal(bytes.begin(), bytes.end(), written.begin() + at);
      };
      check(windowHolds(windowStep.plane0, windowed.w0) &&
                (!layout.plane1_bytes || windowHolds(windowStep.plane1, windowed.w1)) &&
                windowHolds(windowStep.meta, windowed.meta),
            name + ": planes of its two column windows");
    }
    // A streamed expert's load reads an affine, mxfp4 or mxfp8 tensor's planes
    // as the checkpoint stores them (storedPlanes), and builds an nvfp4 one's
    // native rows; a permuted tensor's rows are built too.
    model::gguf::Repack exact = step;
    exact.rows = rows;
    const auto stored = model::storedPlanes(exact);
    const auto fileHolds = [&](const model::PlaneRows &plane, const std::vector<uint8_t> &bytes) {
      if (!plane.file || plane.rowBytes * rows != bytes.size()) return false;
      std::vector<uint8_t> read(bytes.size());
      plane.file->readFile(plane.offset, read);
      return read == bytes;
    };
    if (f == gguf_reference::NVFP4)
      check(!stored, name + ": no planes read as stored");
    else
      check(stored && fileHolds((*stored)[0], tensor.weight) && fileHolds((*stored)[1], tensor.scales) &&
                (tensor.affine() ? fileHolds((*stored)[2], tensor.biases) : !(*stored)[2].rowBytes),
            name + ": its planes as the checkpoint stores them");
    exact.sources.front().order = {0, 1, 1, 1};
    check(!model::storedPlanes(exact), name + ": permuted rows have no stored planes");
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
// rows (blockNvfp4Rows) prepare into those rows' planes with their .scale's g
// (a dense FFN projection and the GDN's grouped qkv rows, one value each),
// the MoE's gate experts with a g of 1 and their .scale, one value per
// expert, in the layer's expert scales beside 1s for up and down. A .scale of
// another count, or beside a projection of another type, is refused.
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
      // A projection's rows hold its .scale as g; an experts tensor's hold 1 (its .scale is the layer's).
      setTensorScales(c.native, NVFP4, c.columns,
                      [&](uint32_t r) { return moe ? 1.0F : scales[r / (c.rows / c.scales)]; });
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
            "llama.cpp NVFP4 " + c.name + " prepares into the planes of its NVFP4 rows" +
                (moe ? ", g 1" : ", its .scale as g"));
    }
    if (moe) {
      // Gate's scales, then up's and down's 1s.
      const std::string scale = "blk.0.ffn_gate_exps.scale";
      const model::gguf::Copy *copy = copyOf(images[0], scale);
      std::vector<uint8_t> expected = target.data(scale);
      const std::vector<float> ones(2 * g.experts, 1.0F);
      expected.insert(expected.end(), reinterpret_cast<const uint8_t *>(ones.data()),
                      reinterpret_cast<const uint8_t *>(ones.data() + ones.size()));
      check(copy && slice(prepared[0], copy->destination, expected.size()) == expected,
            "llama.cpp NVFP4 experts' .scale prepares into the layer's expert scales, 1s for the others");
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
    // A .scale beside a projection of another type, which llama.cpp would apply.
    std::vector<Tensor> beside = target.tensors;
    beside.push_back({"blk.0.ffn_gate.scale", {1}, 0, std::vector<uint8_t>{0x00, 0x00, 0x80, 0x3F}});
    check(refusal(path, beside, g).find("ffn_gate.scale (unused) in 1 layer") != std::string::npos,
          "a .scale tensor beside a projection of another type than NVFP4 is refused");
  }
}

// Inkling-2-Mini's checkpoints (model/Inkling2LoopSafetensors.cpp) of one
// model, in its release's names and layouts and in its MLX converters'
// (oMLX's, mlx-vlm's and mlx-lm's), of seeded values in the engine test's
// layout (inkling2EngineLayout: its experts' down of 320 inputs takes NVFP4's
// 64-element form) with four query heads over one KV head: BF16 projections,
// head and token table, but the output projection, NVFP4 with a tensor scale
// as each converter stores one; BF16 norms, clips, convolutions, relative
// bias and router (the relative bias and a full layer's static taps stored
// transposed, the taps last first); F32 queries, router bias and message
// gates of BF16 values (mlx-vlm's down projections BF16, a call of both
// formats as mlx-community's conversions hold them); NVFP4 experts with each
// expert's amax over a denominator of 2688 (MLX's global_scale the amax,
// mlx-lm's weight_scale_2 the quotient), MXFP8 ones in the last block. Each
// checkpoint holds them as its converter does:
// its convolutions' shapes, oMLX's fused projections and its message gates
// packed and transposed, the experts' gate and up interleaved a row each
// (mlx-lm's apart), the release's message queries in two lists of every loop
// layer's. Every checkpoint's
// images are the release's bit for bit, and the release's are those of its
// GGUF as llama.cpp's converter writes it (Q8_0 projections and head of the
// reference quantization, NVFP4 output projections and experts and their
// .scale, BF16 sinks, tap projections, router and token table, F32 the rest),
// but the last block's, whose MXFP8 experts no GGUF type holds. The
// release's own gates, BF16 in every call, stay BF16, as its GGUF's do.
void checkInkling2Checkpoints(MetalBackend &backend, const std::filesystem::path &directory) {
  model::Inkling2LoopLayout g = inkling2EngineLayout();
  g.attentionWidth = 512;
  g.attentionQueryHeads = 4;
  g.attentionKvHeads = 1;
  g.packedFullWidth = g.queryProjectionWidth();
  g.relativeExtent = 32;
  const uint64_t hidden = g.hiddenSize, width = g.attentionWidth, kv = g.kvWidth(), rel = g.relativeWidth(),
                 heads = g.attentionQueryHeads, taps = g.convolutionTaps, tapRows = g.convolutionTapsWidth(),
                 experts = g.experts, expert = g.expertIntermediateSize, rank = g.loopGateRank,
                 vocabulary = g.vocabularySize, loops = g.loopEndLayer - g.loopFirstLayer;
  enum Naming { Release, OMlx, MlxVlm, MlxLm, kNamings };
  using Names = std::array<std::string, kNamings>;
  const Names models{"model.llm.", "model.", "model.language_model.", "language_model.model."};
  std::array<std::vector<splash::test::SyntheticTensor>, kNamings> shards;
  std::vector<Tensor> gguf;
  std::mt19937 rng(1500);
  const auto draw = [&](uint64_t count, float deviation) {
    std::normal_distribution<float> value(0.0f, deviation);
    std::vector<float> values(count);
    for (float &v : values) v = value(rng);
    return values;
  };
  // The F32 bytes of values; their BF16 bytes, to whose values they round.
  const auto floatBytes = [](const std::vector<float> &values) {
    std::vector<uint8_t> bytes(values.size() * 4);
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
  };
  const auto bfloatBytes = [](std::vector<float> &values) {
    std::vector<uint8_t> bytes(values.size() * 2);
    for (size_t i = 0; i < values.size(); ++i) {
      const uint16_t bits = f2bf(values[i]);
      values[i] = bf2f(bits);
      std::memcpy(bytes.data() + 2 * i, &bits, 2);
    }
    return bytes;
  };
  // The transpose of a [rows, columns] matrix, its rows last first with `reversed`.
  const auto transposed = [](const std::vector<float> &values, uint64_t rows, uint64_t columns, bool reversed) {
    std::vector<float> result(values.size());
    for (uint64_t r = 0; r < rows; ++r)
      for (uint64_t c = 0; c < columns; ++c)
        result[(reversed ? columns - 1 - c : c) * rows + r] = values[r * columns + c];
    return result;
  };
  const auto concat = [](std::initializer_list<const std::vector<uint8_t> *> parts) {
    std::vector<uint8_t> bytes;
    for (const std::vector<uint8_t> *part : parts) bytes.insert(bytes.end(), part->begin(), part->end());
    return bytes;
  };
  const auto prefixed = [](const Names &prefixes, const Names &names) {
    Names result;
    for (int naming = 0; naming < kNamings; ++naming) result[naming] = prefixes[naming] + names[naming];
    return result;
  };
  const auto put = [&](Naming naming, std::string name, std::string dtype, std::vector<uint64_t> shape,
                       std::vector<uint8_t> bytes) {
    shards[naming].push_back({std::move(name), std::move(dtype), std::move(shape), std::move(bytes)});
  };
  // A tensor every checkpoint holds by its own name, and the GGUF's F32 tensor of its values.
  const auto everywhere = [&](const Names &names, const std::string &dtype, const std::vector<uint64_t> &shape,
                              const std::vector<uint8_t> &bytes, const std::string &ggufName,
                              std::vector<uint64_t> dims, const std::vector<float> &values) {
    for (int naming = 0; naming < kNamings; ++naming) put(Naming(naming), names[naming], dtype, shape, bytes);
    gguf.push_back({ggufName, std::move(dims), model::ggml::kF32, floatBytes(values)});
  };
  // F32 values; BF16 values of the GGUF's [rows, columns], which the
  // checkpoints store `transpose`d (the transpose's rows last first with
  // `reversed`) in `shape`.
  const auto f32 = [&](const Names &names, const std::vector<uint64_t> &shape, const std::string &ggufName,
                       std::vector<uint64_t> dims, float deviation) {
    const std::vector<float> values =
        draw(std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<>()), deviation);
    everywhere(names, "F32", shape, floatBytes(values), ggufName, std::move(dims), values);
  };
  const auto bf16 = [&](const Names &names, const std::vector<uint64_t> &shape, const std::string &ggufName,
                        uint64_t rows, uint64_t columns, float deviation, bool transpose = false,
                        bool reversed = false) {
    std::vector<float> values = draw(rows * columns, deviation);
    static_cast<void>(bfloatBytes(values));
    std::vector<float> stored = transpose ? transposed(values, rows, columns, reversed) : values;
    everywhere(names, "BF16", shape, bfloatBytes(stored), ggufName,
               rows == 1 ? std::vector<uint64_t>{columns} : std::vector<uint64_t>{columns, rows}, values);
  };
  // A convolution's BF16 taps [channels, taps] of the GGUF's, each
  // checkpoint's in its converter's shape: the release's [channels, 1, taps],
  // oMLX's [channels, taps], MLX's [channels, taps, 1].
  const auto convolution = [&](const Names &names, uint64_t channels, const std::string &ggufName) {
    std::vector<float> values = draw(channels * taps, 0.25f);
    const std::vector<uint8_t> bytes = bfloatBytes(values);
    const std::array<std::vector<uint64_t>, kNamings> shapes{
        {{channels, 1, taps}, {channels, taps}, {channels, taps, 1}, {channels, taps, 1}}};
    for (int naming = 0; naming < kNamings; ++naming) put(Naming(naming), names[naming], "BF16", shapes[naming], bytes);
    gguf.push_back({ggufName, {taps, channels}, model::ggml::kF32, floatBytes(values)});
  };
  // A BF16 projection [rows, columns]: its values and bytes.
  const auto projection = [&](uint64_t rows, uint64_t columns) {
    std::vector<float> values = draw(rows * columns, 1.0f / std::sqrt(float(columns)));
    std::vector<uint8_t> bytes = bfloatBytes(values);
    return std::pair{std::move(values), std::move(bytes)};
  };
  // Rows select(r), r < count, of rows of rowBytes each.
  const auto pick = [](const std::vector<uint8_t> &bytes, uint64_t rowBytes, uint64_t count, auto select) {
    std::vector<uint8_t> rows;
    for (uint64_t r = 0; r < count; ++r) {
      const auto from = bytes.begin() + int64_t(select(r) * rowBytes);
      rows.insert(rows.end(), from, from + int64_t(rowBytes));
    }
    return rows;
  };
  // Each gate's BF16 bytes, by the release's name and the GGUF's.
  std::map<std::string, std::vector<uint8_t>> bfloatGates;
  // Each gate's down projection's BF16 bytes, by the GGUF's name.
  std::map<std::string, std::vector<uint8_t>> bfloatDowns;
  std::vector<float> crossQueries(2 * loops * hidden), crossKvQueries(2 * loops * hidden), messageQueries;
  for (uint32_t layer = 0; layer < g.weightLayers; ++layer) {
    const std::string l = std::to_string(layer), gp = "blk." + l + ".";
    const Names p = prefixed(models, {"layers." + l + ".", "layers." + l + ".", "layers." + l + ".", "layers." + l + "."});
    const bool full = !g.slides(layer);
    if (const uint32_t queries = g.mixerQueries(layer)) {
      Names names = prefixed(p, {"", "attn_input.query", "attention_input.query", "attention_input.query"});
      names[Release] = models[Release] + "attn_res.attn_queries." + std::to_string(layer - 1);
      f32(names, {queries, hidden}, gp + "attn_res_query.weight",
          queries == 1 ? std::vector<uint64_t>{hidden} : std::vector<uint64_t>{hidden, queries}, 0.5f);
    }
    if (!full)
      bf16(prefixed(p, {"attn_norm.weight", "attn_norm.weight", "input_layernorm.weight", "input_layernorm.weight"}),
           {hidden}, gp + "attn_norm.weight", 1, hidden, 1.0f);
    for (const std::string input : {"q", "k", "v"})
      if (full)
        bf16(prefixed(p, {input + "_input_norm.weight", input + "_input_norm.weight", input + "_input_layernorm.weight",
                          input + "_input_layernorm.weight"}),
             {hidden}, gp + "attn_" + input + "_inp_norm.weight", 1, hidden, 1.0f);
    // A projection of the release and mlx-vlm and mlx-lm (self_attn.<mlx>);
    // the GGUF's Q8_0 rows of the reference quantization, BF16 the sinks and
    // the taps' projections.
    const auto attention = [&](const char *release, const char *mlx, const std::string &ggufName, uint64_t rows,
                               uint64_t columns) {
      auto [values, bytes] = projection(rows, columns);
      put(Release, p[Release] + release + ".weight", "BF16", {rows, columns}, bytes);
      for (const Naming naming : {MlxVlm, MlxLm})
        put(naming, p[naming] + "self_attn." + mlx + ".weight", "BF16", {rows, columns}, bytes);
      if (ggufName == "attn_sink_proj" || ggufName.ends_with("_conv_proj"))
        gguf.push_back({gp + ggufName + ".weight", {columns, rows}, model::ggml::kBF16, bytes});
      else
        gguf.push_back({gp + ggufName + ".weight", {columns, rows}, kQ8_0, quantizeQ8_0(values)});
      return bytes;
    };
    const std::vector<uint8_t> q = attention("attn.wq_du", "q_proj", "attn_q", width, hidden),
                               gate = attention("attn.gate_attn", "gate_proj", "attn_gate", width, hidden),
                               relative = attention("attn.wr_du", "r_proj", "attn_rel", rel, hidden),
                               sinks = attention("attn.attn_sink_proj", "sink_proj", "attn_sink_proj", heads, hidden),
                               k = attention("attn.wk_dv", "k_proj", "attn_k", kv, hidden),
                               v = attention("attn.wv_dv", "v_proj", "attn_v", kv, hidden);
    // The output projection NVFP4 with a tensor scale g, as each converter
    // stores one: Model Optimizer's codes, scales and weight_scale_2 (g) by
    // the release's names, MLX's with its global_scale (g's amax, over the
    // 6 * 448 MLX's dequantization divides by) or mlx-lm's weight_scale_2;
    // the GGUF's llama.cpp NVFP4 and its .scale.
    {
      const std::vector<uint8_t> native = makeNative(NVFP4, uint32_t(hidden), uint32_t(width), rng);
      std::vector<uint8_t> codes, scales;
      for (uint64_t at = 0; at < native.size(); at += kQuantFormats[NVFP4].block_bytes) {
        scales.insert(scales.end(), native.begin() + int64_t(at), native.begin() + int64_t(at + kNvfp4Scale));
        codes.insert(codes.end(), native.begin() + int64_t(at + kNvfp4Codes),
                     native.begin() + int64_t(at + kQuantFormats[NVFP4].block_bytes));
      }
      const float amax = std::uniform_real_distribution<float>(0.5f, 8.0f)(rng), g = amax / 2688.0f;
      const std::string release = p[Release] + "attn.wo_ud";
      put(Release, release + ".weight", "U8", {hidden, width / 2}, codes);
      put(Release, release + ".weight_scale", "F8_E4M3", {hidden, width / 16}, scales);
      put(Release, release + ".weight_scale_2", "F32", {}, floatBytes({g}));
      for (const Naming naming : {OMlx, MlxVlm, MlxLm}) {
        const std::string module = p[naming] + (naming == OMlx ? "attn.o_proj" : "self_attn.o_proj");
        put(naming, module + ".weight", "U32", {hidden, width / 8}, codes);
        put(naming, module + ".scales", "U8", {hidden, width / 16}, scales);
        if (naming == MlxLm) put(naming, module + ".weight_scale_2", "F32", {}, floatBytes({g}));
        else put(naming, module + ".global_scale", "F32", {}, floatBytes({amax}));
      }
      gguf.push_back({gp + "attn_output.weight", {width, hidden}, model::ggml::kNVFP4, blockNvfp4Rows(native)});
      gguf.push_back({gp + "attn_output.scale", {1}, model::ggml::kF32, floatBytes({g})});
    }
    // oMLX's fused projections: a sliding layer's in_proj; a full layer's
    // q_proj, and its k_proj and v_proj with their taps' projections.
    if (!full) {
      put(OMlx, p[OMlx] + "attn.in_proj.weight", "BF16", {2 * width + 2 * kv + rel + heads, hidden},
          concat({&q, &k, &v, &relative, &gate, &sinks}));
    } else {
      const std::vector<uint8_t> kTaps = attention("attn.k_sconv.weight_proj", "k_sconv.weight_proj",
                                                   "attn_k_conv_proj", tapRows, hidden),
                                 vTaps = attention("attn.v_sconv.weight_proj", "v_sconv.weight_proj",
                                                   "attn_v_conv_proj", tapRows, hidden);
      put(OMlx, p[OMlx] + "attn.q_proj.weight", "BF16", {2 * width + rel + heads, hidden},
          concat({&q, &relative, &gate, &sinks}));
      put(OMlx, p[OMlx] + "attn.k_proj.weight", "BF16", {kv + tapRows, hidden}, concat({&k, &kTaps}));
      put(OMlx, p[OMlx] + "attn.v_proj.weight", "BF16", {kv + tapRows, hidden}, concat({&v, &vTaps}));
    }
    bf16(prefixed(p, {"attn.q_l2_clip.scale", "attn.q_scale", "self_attn.q_l2_clip.scale", "self_attn.q_scale"}),
         {heads, g.attentionHeadDimension}, gp + "attn_q_clip.weight", heads, g.attentionHeadDimension, 0.1f);
    bf16(prefixed(p, {"attn.rel_l2_clip.scale", "attn.r_scale", "self_attn.rel_l2_clip.scale", "self_attn.rel_scale"}),
         {heads, g.relativeRank}, gp + "attn_rel_clip.weight", heads, g.relativeRank, 0.1f);
    bf16(prefixed(p, {"attn.rel_logits_proj.proj", "attn.rel_proj", "self_attn.rel_logits_proj.proj",
                      "self_attn.rel_proj"}),
         {g.relativeRank, g.relativeExtent}, gp + "attn_rel_bias.weight", g.relativeExtent, g.relativeRank, 0.1f,
         true);
    for (const std::string input : {"k", "v"}) {
      const std::string sconv = "attn." + input, mlx = "self_attn." + input + "_sconv.";
      if (full)
        bf16(prefixed(p, {sconv + "_sconv.static_w", sconv + "_static", mlx + "static_weight", mlx + "static_weight"}),
             {taps, kv}, gp + "attn_" + input + "_conv.weight", kv, taps, 0.25f, true, true);
      else
        convolution(prefixed(p, {sconv + "_sconv.weight", sconv + "_conv", mlx + "conv1d.weight", mlx + "conv.weight"}),
                    kv, gp + "attn_" + input + "_conv.weight");
    }
    convolution(prefixed(p, {"attn_sconv.weight", "attn_conv", "post_attention_convolution.conv1d.weight",
                             "attn_sconv.conv.weight"}),
                hidden, gp + "attn_post_conv.weight");
    {
      Names names = prefixed(p, {"", "mlp_input.query", "mlp_input.query", "mlp_input.query"});
      names[Release] = models[Release] + "attn_res.mlp_queries." + l;
      f32(names, {hidden}, gp + "ffn_res_query.weight", {hidden}, 0.5f);
    }
    bf16(prefixed(p, {"mlp_norm.weight", "mlp_norm.weight", "post_attention_layernorm.weight",
                      "post_attention_layernorm.weight"}),
         {hidden}, gp + "ffn_norm.weight", 1, hidden, 1.0f);
    {
      // The router BF16, as the release stores it, and its rows as stored.
      std::vector<float> router = draw(experts * hidden, 0.01f);
      const std::vector<uint8_t> bytes = bfloatBytes(router);
      const Names names = prefixed(p, {"mlp.gate.weight", "mlp.router_weight", "mlp.gate.weight", "mlp.gate_weight"});
      for (int naming = 0; naming < kNamings; ++naming) put(Naming(naming), names[naming], "BF16", {experts, hidden}, bytes);
      gguf.push_back({gp + "ffn_gate_inp.weight", {hidden, experts}, model::ggml::kBF16, bytes});
    }
    f32(prefixed(p, {"mlp.gate.bias", "mlp.router_bias", "mlp.gate.e_score_correction_bias",
                     "mlp.e_score_correction_bias"}),
        {experts}, gp + "exp_probs_b.bias", {experts}, 1.0f);
    convolution(prefixed(p, {"mlp_sconv.weight", "mlp_conv", "post_mlp_convolution.conv1d.weight",
                             "mlp_sconv.conv.weight"}),
                hidden, gp + "ffn_post_conv.weight");

    // The routed experts: native rows of each expert's gate rows, then its up
    // rows (the GGUF's fused gate and up), and of its down rows; the
    // checkpoints' codes and scales of them (MLX's codes in U32 words).
    const bool mxfp8 = layer + g.blockResidualPeriod >= g.weightLayers;
    const uint64_t valuesPerByte = mxfp8 ? 1 : 2, group = mxfp8 ? 32 : 16;
    struct Experts final {
      Fmt format;
      std::vector<uint8_t> native, codes{}, scales{};
    };
    const auto expertRows = [&](uint64_t rows, uint64_t columns) {
      const Fmt format = mxfp8 ? MXFP8 : columns % 256 ? NVFP464 : NVFP4;
      const QuantFormat &layout = kQuantFormats[format];
      Experts result{format, makeNative(format, uint32_t(experts * rows), uint32_t(columns), rng)};
      // A block's NVFP4 scales (before its g) or MXFP8 exponent, then its codes.
      for (uint64_t at = 0; at < result.native.size(); at += layout.block_bytes) {
        const auto block = result.native.begin() + int64_t(at);
        result.scales.insert(result.scales.end(), block, block + (mxfp8 ? 1 : layout.meta_bytes - 4));
        result.codes.insert(result.codes.end(), block + layout.meta_bytes, block + layout.block_bytes);
      }
      return result;
    };
    const Experts gateUp = expertRows(2 * expert, hidden), down = expertRows(hidden, expert);
    const auto releaseExperts = [&](const std::string &module, const std::vector<uint8_t> &codes,
                                    const std::vector<uint8_t> &scales, uint64_t rows, uint64_t columns) {
      put(Release, module + "_weight", mxfp8 ? "F8_E4M3" : "U8", {experts, rows, columns / valuesPerByte}, codes);
      put(Release, module + "_weight_scale", mxfp8 ? "U8" : "F8_E4M3", {experts, rows, columns / group}, scales);
    };
    const auto mlxExperts = [&](Naming naming, const std::string &module, const std::vector<uint8_t> &codes,
                                const std::vector<uint8_t> &scales, uint64_t rows, uint64_t columns) {
      put(naming, module + ".weight", "U32", {experts, rows, columns / valuesPerByte / 4}, codes);
      put(naming, module + ".scales", "U8", {experts, rows, columns / group}, scales);
    };
    // The release's w13 interleaves gate and up a row each, as oMLX's w13
    // and mlx-vlm's gate_up_proj keep them; mlx-lm's holds them apart.
    const uint64_t gateUpCodes = gateUp.codes.size() / (experts * 2 * expert),
                   gateUpScales = gateUp.scales.size() / (experts * 2 * expert);
    const auto interleaved = [&](uint64_t r) {
      return r / (2 * expert) * 2 * expert + r % 2 * expert + r % (2 * expert) / 2;
    };
    const std::vector<uint8_t> w13 = pick(gateUp.codes, gateUpCodes, experts * 2 * expert, interleaved),
                               w13Scales = pick(gateUp.scales, gateUpScales, experts * 2 * expert, interleaved);
    releaseExperts(p[Release] + "mlp.experts.w13", w13, w13Scales, 2 * expert, hidden);
    releaseExperts(p[Release] + "mlp.experts.w2", down.codes, down.scales, hidden, expert);
    mlxExperts(OMlx, p[OMlx] + "mlp.experts.w13", w13, w13Scales, 2 * expert, hidden);
    mlxExperts(OMlx, p[OMlx] + "mlp.experts.w2", down.codes, down.scales, hidden, expert);
    mlxExperts(MlxVlm, p[MlxVlm] + "mlp.experts.gate_up_proj", w13, w13Scales, 2 * expert, hidden);
    mlxExperts(MlxVlm, p[MlxVlm] + "mlp.experts.down_proj", down.codes, down.scales, hidden, expert);
    const std::array<std::string, 2> halves{"mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj"};
    for (const uint64_t up : {0, 1}) {
      const auto half = [&](uint64_t r) { return r / expert * 2 * expert + up * expert + r % expert; };
      mlxExperts(MlxLm, p[MlxLm] + halves[up], pick(gateUp.codes, gateUpCodes, experts * expert, half),
                 pick(gateUp.scales, gateUpScales, experts * expert, half), expert, hidden);
    }
    mlxExperts(MlxLm, p[MlxLm] + "mlp.switch_mlp.down_proj", down.codes, down.scales, hidden, expert);
    if (mxfp8) {
      // Q8_0 in the GGUF, which has no MXFP8 type.
      gguf.push_back({gp + "ffn_gate_up_exps.weight", {hidden, 2 * expert, experts}, kQ8_0,
                      fixture(Q80, uint32_t(experts * 2 * expert), uint32_t(hidden), layer)});
      gguf.push_back({gp + "ffn_down_exps.weight", {expert, hidden, experts}, kQ8_0,
                      fixture(Q80, uint32_t(experts * hidden), uint32_t(expert), layer)});
    } else {
      // An NVFP4 expert's global scale: the release's amax over its
      // denominator, MLX's global_scale the amax, mlx-lm's weight_scale_2 and
      // the GGUF's .scale their quotient.
      std::uniform_real_distribution<float> amax(0.5f, 8.0f);
      std::vector<float> gateUpAmax(experts), downAmax(experts), gateUpScale(experts), downScale(experts);
      for (uint64_t e = 0; e < experts; ++e) {
        gateUpAmax[e] = amax(rng);
        downAmax[e] = amax(rng);
        gateUpScale[e] = gateUpAmax[e] / 2688.0f;
        downScale[e] = downAmax[e] / 2688.0f;
      }
      for (const auto &[module, values] : {std::pair{"mlp.experts.w13", &gateUpAmax}, {"mlp.experts.w2", &downAmax}}) {
        put(Release, p[Release] + module + "_weight_amax", "F32", {experts}, floatBytes(*values));
        put(Release, p[Release] + module + "_weight_outer_scale_denominator", "F32", {1}, floatBytes({2688.0f}));
      }
      put(OMlx, p[OMlx] + "mlp.experts.w13.global_scale", "F32", {experts}, floatBytes(gateUpAmax));
      put(OMlx, p[OMlx] + "mlp.experts.w2.global_scale", "F32", {experts}, floatBytes(downAmax));
      put(MlxVlm, p[MlxVlm] + "mlp.experts.gate_up_proj.global_scale", "F32", {experts}, floatBytes(gateUpAmax));
      put(MlxVlm, p[MlxVlm] + "mlp.experts.down_proj.global_scale", "F32", {experts}, floatBytes(downAmax));
      for (const std::string &half : halves)
        put(MlxLm, p[MlxLm] + half + ".weight_scale_2", "F32", {experts}, floatBytes(gateUpScale));
      put(MlxLm, p[MlxLm] + "mlp.switch_mlp.down_proj.weight_scale_2", "F32", {experts}, floatBytes(downScale));
      gguf.push_back({gp + "ffn_gate_up_exps.weight", {hidden, 2 * expert, experts}, model::ggml::kNVFP4,
                      blockNvfp4Rows(gateUp.native, gateUp.format)});
      gguf.push_back({gp + "ffn_gate_up_exps.scale", {experts}, model::ggml::kF32, floatBytes(gateUpScale)});
      gguf.push_back({gp + "ffn_down_exps.weight", {expert, hidden, experts}, model::ggml::kNVFP4,
                      blockNvfp4Rows(down.native, down.format)});
      gguf.push_back({gp + "ffn_down_exps.scale", {experts}, model::ggml::kF32, floatBytes(downScale)});
    }
    if (!g.schedule().isLoopLayer(layer)) continue;

    // A loop layer's message gates of its q, k, v and FFN inputs, oMLX's down
    // projections transposed in loop_down and its gains and forgets
    // transposed side by side in loop_gain_forget; and its message queries
    // (q, k, v, FFN), the release's rows of its two lists of every loop
    // layer's (q and FFN, k and v), mlx-vlm's and mlx-lm's of one.
    std::vector<float> loopDown, loopGainForget;
    for (const auto &[input, controller, messageGate] :
         {std::tuple{"q", "attn", "q"}, std::tuple{"k", "k", "k"}, std::tuple{"v", "v", "v"},
          std::tuple{"ffn", "mlp", "mlp"}}) {
      std::array<std::vector<float>, 3> projections;
      for (const auto &[index, name, mlx, rows, columns] :
           {std::tuple{0, "down", "down_proj", rank, 2 * hidden}, std::tuple{1, "forget", "forget_gate_proj", hidden, rank},
            std::tuple{2, "gain", "message_gate_proj", hidden, rank}}) {
        projections[index] = draw(rows * columns, 0.1f);
        const std::vector<uint8_t> bfloat = bfloatBytes(projections[index]), bytes = floatBytes(projections[index]);
        const std::string release = p[Release] + "controllers." + controller + "." + name + "_weight",
                          converted = gp + "loop_" + input + "_ctrl_" + name + ".weight";
        put(Release, release, "F32", {rows, columns}, bytes);
        for (const Naming naming : {MlxVlm, MlxLm}) {
          const bool narrow = naming == MlxVlm && index == 0;
          put(naming, p[naming] + messageGate + "_message_gate." + mlx + ".weight", narrow ? "BF16" : "F32",
              {rows, columns}, narrow ? bfloat : bytes);
        }
        gguf.push_back({converted, {columns, rows}, model::ggml::kF32, bytes});
        bfloatGates[release] = bfloatGates[converted] = bfloat;
        if (index == 0) bfloatDowns[converted] = bfloat;
      }
      const std::vector<float> downT = transposed(projections[0], rank, 2 * hidden, false),
                               forgetT = transposed(projections[1], hidden, rank, false),
                               gainT = transposed(projections[2], hidden, rank, false);
      loopDown.insert(loopDown.end(), downT.begin(), downT.end());
      for (uint64_t r = 0; r < rank; ++r) {
        loopGainForget.insert(loopGainForget.end(), gainT.begin() + int64_t(r * hidden),
                              gainT.begin() + int64_t((r + 1) * hidden));
        loopGainForget.insert(loopGainForget.end(), forgetT.begin() + int64_t(r * hidden),
                              forgetT.begin() + int64_t((r + 1) * hidden));
      }
    }
    put(OMlx, p[OMlx] + "loop_down", "F32", {4, 2, hidden, rank}, floatBytes(loopDown));
    put(OMlx, p[OMlx] + "loop_gain_forget", "F32", {4, rank, 2 * hidden}, floatBytes(loopGainForget));
    const std::vector<float> queries = draw(4 * hidden, 0.5f);
    const uint64_t index = layer - g.loopFirstLayer;
    std::copy_n(queries.begin(), hidden, crossQueries.begin() + int64_t(2 * index * hidden));
    std::copy_n(queries.begin() + int64_t(hidden), 2 * hidden, crossKvQueries.begin() + int64_t(2 * index * hidden));
    std::copy_n(queries.begin() + int64_t(3 * hidden), hidden, crossQueries.begin() + int64_t((2 * index + 1) * hidden));
    messageQueries.insert(messageQueries.end(), queries.begin(), queries.end());
    put(OMlx, p[OMlx] + "loop_message.query", "F32", {4, hidden}, floatBytes(queries));
    gguf.push_back({gp + "loop_msg_query.weight", {hidden, 4}, model::ggml::kF32, floatBytes(queries)});
  }
  put(Release, models[Release] + "middle_cycle_cross_pass_queries", "F32", {2 * loops, hidden}, floatBytes(crossQueries));
  put(Release, models[Release] + "middle_cycle_cross_pass_kv_queries", "F32", {2 * loops, hidden},
      floatBytes(crossKvQueries));
  for (const Naming naming : {MlxVlm, MlxLm})
    put(naming, models[naming] + "message_input.query", "F32", {loops, 4, hidden}, floatBytes(messageQueries));

  // The head's norms and query, the head and the token table.
  bf16(prefixed(models, {"embed_norm.weight", "embed_norm.weight", "embed_norm.weight", "embed_norm.weight"}), {hidden},
       "token_embd_norm.weight", 1, hidden, 1.0f);
  f32(prefixed(models, {"attn_res.output_query", "final_input.query", "final_input.query", "final_input.query"}),
      {hidden}, "output_res_query.weight", {hidden}, 0.5f);
  bf16(prefixed(models, {"norm.weight", "norm.weight", "norm.weight", "norm.weight"}), {hidden}, "output_norm.weight",
       1, hidden, 1.0f);
  const auto [head, headBytes] = projection(vocabulary, hidden);
  const auto [table, tableBytes] = projection(vocabulary, hidden);
  const Names headNames{models[Release] + "unembed.weight", "lm_head.weight", "lm_head.weight",
                        "language_model.lm_head.weight"};
  const Names tableNames = prefixed(models, {"embed.weight", "embed.weight", "embed_tokens.weight", "embed_tokens.weight"});
  for (int naming = 0; naming < kNamings; ++naming) {
    put(Naming(naming), headNames[naming], "BF16", {vocabulary, hidden}, headBytes);
    put(Naming(naming), tableNames[naming], "BF16", {vocabulary, hidden}, tableBytes);
  }
  gguf.push_back({"output.weight", {hidden, vocabulary}, kQ8_0, quantizeQ8_0(head)});
  gguf.push_back({"token_embd.weight", {hidden, vocabulary}, model::ggml::kBF16, tableBytes});

  // Each image of a plan, as written, by its digest.
  const auto written = [&](const std::vector<model::gguf::Image> &plans) {
    std::vector<std::string> digests;
    for (const model::gguf::Image &plan : plans) {
      const auto image = backend.allocateBuffer(plan.bytes, splash::metal::BufferStorage::Shared, "inkling2-image");
      model::writeGgufImage(backend, image, plan);
      digests.push_back(model::weightDigest(model::contentsOf(image)));
    }
    return digests;
  };
  std::vector<std::string> release, mlxVlm;
  std::vector<uint64_t> releaseBytes;
  const std::array<const char *, kNamings> namings{"its release's", "oMLX's", "mlx-vlm's", "mlx-lm's"};
  for (int naming = 0; naming < kNamings; ++naming) {
    const auto root = directory / ("inkling2-" + std::to_string(naming));
    splash::test::writeSyntheticShard(root / "target" / "model.safetensors", shards[naming]);
    const model::SafetensorsCheckpoint checkpoint(root / "target");
    const std::vector<model::gguf::Image> plans = model::safetensorsTargetPlan(g).images(checkpoint, {}).images;
    const std::vector<std::string> digests = written(plans);
    std::vector<uint64_t> imageBytes;
    for (const model::gguf::Image &plan : plans) imageBytes.push_back(plan.bytes);
    if (naming == Release) {
      release = digests;
      releaseBytes = imageBytes;
    }
    if (naming == MlxVlm) {
      // mlx-vlm keeps each gate's down projection BF16 beside F32 forget and
      // gain (ops::MessageGate takes a call's down format apart from its
      // forget and gain format), so its loop layers stay smaller than the
      // all-F32 release's while every other image matches bit for bit.
      mlxVlm = digests;
      bool matches = digests.size() == release.size();
      for (size_t image = 0; matches && image < digests.size(); ++image)
        matches = image < g.weightLayers && g.schedule().isLoopLayer(static_cast<uint32_t>(image))
                      ? digests[image] != release[image] && imageBytes[image] < releaseBytes[image]
                      : digests[image] == release[image];
      check(matches, "an Inkling-2-Mini checkpoint in mlx-vlm's names keeps BF16 down projections beside F32 forget "
                     "and gain");
    } else {
      check(digests.size() == g.weightLayers + 2 && digests == release,
            std::string("an Inkling-2-Mini checkpoint in ") + namings[naming] +
                " names and layouts writes the release's images");
    }
    // A model of it, with a draft, is sized as its plan loads it
    // (streamableWeights), as the server sizes the model it serves.
    if (naming != Release) continue;
    model::DFlashDraftLayout draft;
    draft.layers = 2;
    draft.hiddenSize = g.hiddenSize;
    draft.vocabularySize = g.vocabularySize;
    draft.dynamicSize = draft.qkvSize = 256;
    draft.attentionSize = draft.intermediateSize = draft.selectorRank = 256;
    draft.attentionHeadDimension = 64;
    draft.kvHeads = 2;
    draft.targetLayers = {1, 3};
    draft.targetLayerCount = 2;
    splash::test::writeSyntheticShard(root / "draft" / "model.safetensors", splash::test::draftTensors(draft));
    const model::ModelDescriptor descriptor =
        model::makeModelDescriptor("Inkling-2-Mini", g, draft, ops::visionLayout(g.visionTower, g.hiddenSize),
                                   model::TargetSource::Safetensors,
                                   model::VisionSource::None);
    uint64_t planned = model::draftImageBytes(root / "draft", draft);
    for (const model::gguf::Image &plan : plans) planned += model::gguf::loadedBytes(plan);
    check(model::streamableWeights(root, descriptor, {}).loadedBytes == planned,
          "an Inkling-2-Mini checkpoint is sized as its plan loads it");
  }
  const auto path = directory / "inkling2.gguf";
  splash::test::writeFile(path, test_gguf::file(inkling2Metadata(g), gguf));
  model::WeightSource source(path);
  const std::vector<std::string> converted = written(model::ggufTargetPlan(g).images(model::GgufFile(source), {}));
  bool same = converted.size() == release.size();
  for (size_t image = 0; same && image < converted.size(); ++image)
    same = converted[image] == release[image] ||
           (image < g.weightLayers && image + g.blockResidualPeriod >= g.weightLayers);
  check(same, "Inkling-2-Mini's release writes its GGUF's images, but its MXFP8 experts'");

  // The release's own format, BF16 gates in every call: kept as stored, so
  // its images are not the F32 release's but those of its GGUF of BF16 gates.
  std::vector<splash::test::SyntheticTensor> bfloatShard = shards[Release];
  std::vector<Tensor> bfloatGguf = gguf;
  for (splash::test::SyntheticTensor &tensor : bfloatShard)
    if (const auto found = bfloatGates.find(tensor.name); found != bfloatGates.end()) {
      tensor.dtype = "BF16";
      tensor.data = found->second;
    }
  for (Tensor &tensor : bfloatGguf)
    if (const auto found = bfloatGates.find(tensor.name); found != bfloatGates.end()) {
      tensor.type = model::ggml::kBF16;
      tensor.data = found->second;
    }
  splash::test::writeSyntheticShard(directory / "inkling2-bf16-gates" / "target" / "model.safetensors", bfloatShard);
  const model::SafetensorsCheckpoint bfloatCheckpoint(directory / "inkling2-bf16-gates" / "target");
  const std::vector<std::string> kept = written(model::safetensorsTargetPlan(g).images(bfloatCheckpoint, {}).images);
  const auto bfloatPath = directory / "inkling2-bf16-gates.gguf";
  splash::test::writeFile(bfloatPath, test_gguf::file(inkling2Metadata(g), bfloatGguf));
  model::WeightSource bfloatSource(bfloatPath);
  const std::vector<std::string> keptConverted =
      written(model::ggufTargetPlan(g).images(model::GgufFile(bfloatSource), {}));
  bool keeps = kept != release && keptConverted.size() == kept.size();
  for (size_t image = 0; keeps && image < kept.size(); ++image)
    keeps = keptConverted[image] == kept[image] ||
            (image < g.weightLayers && image + g.blockResidualPeriod >= g.weightLayers);
  check(keeps, "an Inkling-2-Mini checkpoint of BF16 gates keeps them, as its GGUF does");

  // A GGUF whose gates keep down in BF16 and forget/gain in F32 keeps each
  // group as stored, writing mlx-vlm's images (but its MXFP8 experts').
  std::vector<Tensor> mixedDownGguf = gguf;
  for (Tensor &tensor : mixedDownGguf)
    if (const auto found = bfloatDowns.find(tensor.name); found != bfloatDowns.end()) {
      tensor.type = model::ggml::kBF16;
      tensor.data = found->second;
    }
  const auto mixedDownPath = directory / "inkling2-bf16-down-gates.gguf";
  splash::test::writeFile(mixedDownPath, test_gguf::file(inkling2Metadata(g), mixedDownGguf));
  model::WeightSource mixedDownSource(mixedDownPath);
  const std::vector<std::string> mixedDown =
      written(model::ggufTargetPlan(g).images(model::GgufFile(mixedDownSource), {}));
  bool keepsMixed = mixedDown != converted && mixedDown.size() == mlxVlm.size();
  for (size_t image = 0; keepsMixed && image < mixedDown.size(); ++image)
    keepsMixed = mixedDown[image] == mlxVlm[image] ||
                 (image < g.weightLayers && image + g.blockResidualPeriod >= g.weightLayers);
  check(keepsMixed, "an Inkling-2-Mini GGUF of BF16 down beside F32 forget/gain keeps them, as mlx-vlm's does");

  // A GGUF whose calls mix BF16 and other formats within down (q BF16 beside
  // k/v F32) or within forget/gain (forget BF16 beside gain F32) widens each
  // mixed group's BF16 tensors to F32, writing the all-F32 GGUF's images.
  std::vector<Tensor> mixedGroupGguf = gguf;
  for (Tensor &tensor : mixedGroupGguf)
    if (tensor.name.ends_with("loop_q_ctrl_down.weight") || tensor.name.ends_with("loop_q_ctrl_forget.weight") ||
        tensor.name.ends_with("loop_ffn_ctrl_forget.weight")) {
      const auto found = bfloatGates.find(tensor.name);
      if (found != bfloatGates.end()) {
        tensor.type = model::ggml::kBF16;
        tensor.data = found->second;
      }
    }
  const auto mixedGroupPath = directory / "inkling2-mixed-group-gates.gguf";
  splash::test::writeFile(mixedGroupPath, test_gguf::file(inkling2Metadata(g), mixedGroupGguf));
  model::WeightSource mixedGroupSource(mixedGroupPath);
  check(written(model::ggufTargetPlan(g).images(model::GgufFile(mixedGroupSource), {})) == converted,
        "an Inkling-2-Mini GGUF mixing BF16 and F32 within down or forget/gain widens the mixed group to F32");
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
    checkPipelineOrder();
    checkMxfp8Blocks();
    // First of the GPU's, so that the peak allocation it checks against is its own.
    checkExecutor(backend, directory.path());
    guarded("a load that fails mid-image", [&] { checkFailedLoad(backend, directory.path()); });
    guarded("preparation of the dense target", [&] { checkDense(backend, directory.path(), hashes); });
    guarded("preparation of the MoE target", [&] { checkMoe(backend, directory.path(), hashes); });
    guarded("preparation of BF16 and F16 alpha/beta", [&] { checkFloatAlphaBeta(backend, directory.path()); });
    guarded("preparation of 16-bit float routers, gates and norms",
            [&] {
              for (const Fmt gate : {BF16, Q80}) checkSixteenBitFloats(backend, directory.path(), gate);
            });
    guarded("preparation of quantized routers", [&] { checkQuantizedRouter(backend, directory.path()); });
    guarded("preparation of BF16 experts", [&] { checkFloatExperts(backend, directory.path()); });
    guarded("preparation of quantized alpha/beta", [&] { checkQuantizedAlphaBeta(backend, directory.path()); });
    guarded("llama.cpp NVFP4", [&] { checkGgufNvfp4(backend, directory.path()); });
    guarded("the target loader", [&] { checkDenseTarget(backend, directory.path()); });
    guarded("MLX sources", [&] { checkMlxSources(backend, directory.path(), argv[3]); });
    guarded("NVFP4 and FP8 sources", [&] { checkFloatSources(backend, directory.path()); });
    guarded("Inkling-2-Mini's checkpoints", [&] { checkInkling2Checkpoints(backend, directory.path()); });
    std::printf("%s (%d failures)\n", failures ? "GGUF preparation tests FAILED" : "GGUF preparation tests passed",
                failures);
    return failures ? 1 : 0;
  }
}
