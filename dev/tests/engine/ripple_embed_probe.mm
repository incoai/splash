// The token table probe: the GPU's gathered rows of the real model's
// embedding against the CPU's dequantization of the GGUF's own bytes. A
// mismatch names the token-table path (the release's IQ4_XS table, which no
// earlier Ripple test exercised) as the corruption.
#include "TestChecks.hpp"
#include "engine/MemoryGovernor.hpp"
#include "model/ModelFactory.hpp"
#include "model/RuntimeArenas.hpp"
#include "ops/ExecutionPlans.hpp"
#include "model/Ripple.hpp"
#include "LinearNumerics.hpp"
#include "ops/Embedding.hpp"

#import <Foundation/Foundation.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::model::LoadedModel;
using splash::ops::Embedding;
using splash::test::require;

constexpr uint32_t kHidden = 5120;
constexpr float kEmbedScale = 71.5541763305664F;

// ------------------------------------------------------------- GGUF reading

float f16Bits(uint16_t bits) {
  const uint32_t sign = bits >> 15 & 1, exponent = bits >> 10 & 0x1F, fraction = bits & 0x3FF;
  if (!exponent) return std::ldexp(fraction, -24) * (sign ? -1.0F : 1.0F);
  if (exponent == 31) return sign ? -INFINITY : INFINITY;
  return std::ldexp((fraction | 0x400) * (sign ? -1.0F : 1.0F), static_cast<int>(exponent) - 25);
}

// The IQ4_NL grid the format's codes index.
constexpr float kIQ4NLValues[16] = {-127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
                                    1.0f,    13.0f,   25.0f,  38.0f,  53.0f,  69.0f,  89.0f,  120.0f};

// Dequantizes `count` values of a row stored in the GGUF at `base`: Q8_0
// blocks of a float16 scale and 32 int8, or IQ4_XS superblocks of 256.
std::vector<float> rowValues(const uint8_t *base, uint64_t count, uint32_t type) {
  std::vector<float> out(count);
  if (type == 8) { // Q8_0
    for (uint64_t block = 0; block * 32 < count; ++block) {
      const float scale = f16Bits(static_cast<uint16_t>(base[block * 34] | base[block * 34 + 1] << 8));
      for (uint32_t element = 0; element < 32 && block * 32 + element < count; ++element)
        out[block * 32 + element] = static_cast<int8_t>(base[block * 34 + 2 + element]) * scale;
    }
  } else if (type == 23) { // IQ4_XS: 136-byte blocks of 256, the engine's layout
    for (uint64_t block = 0; block * 256 < count; ++block) {
      const uint8_t *b = base + block * 136;
      const float d = f16Bits(static_cast<uint16_t>(b[0] | b[1] << 8));
      const uint16_t sh = static_cast<uint16_t>(b[2] | b[3] << 8);
      const uint8_t *qs = b + 8;
      for (uint32_t group = 0; group < 8; ++group) {
        const int ls = ((b[4 + group / 2] >> 4 * (group % 2)) & 0xf) | (((sh >> 2 * group) & 3) << 4);
        const float scale = d * (ls - 32);
        const uint8_t *q = qs + group * 16;
        for (uint32_t k = 0; k < 32; ++k) {
          const uint8_t code = k < 16 ? (q[k] & 15) : (q[k - 16] >> 4);
          const uint64_t index = block * 256 + uint64_t{group} * 32 + k;
          if (index < count) out[index] = scale * kIQ4NLValues[code];
        }
      }
    }
  } else throw std::runtime_error("unexpected GGUF type in the probe");
  return out;
}

class Gguf final {
public:
  explicit Gguf(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("the GGUF is missing");
    std::vector<char> header(16 << 20);
    in.read(header.data(), static_cast<std::streamsize>(header.size()));
    uint64_t cursor = 24;
    auto stringAt = [&](uint64_t at) {
      uint64_t length = 0;
      std::memcpy(&length, header.data() + at, 8);
      return std::string(header.data() + at + 8, length);
    };
    auto readScalar = [&](uint64_t &at, uint32_t bytes) {
      uint64_t value = 0;
      std::memcpy(&value, header.data() + at, bytes);
      at += bytes;
      return value;
    };
    uint64_t tensorCount = 0, metadataCount = 0;
    std::memcpy(&tensorCount, header.data() + 8, 8);
    std::memcpy(&metadataCount, header.data() + 16, 8);
    for (uint64_t index = 0; index < metadataCount; ++index) {
      const uint64_t keyLength = readScalar(cursor, 8);
      cursor += keyLength;
      const uint32_t kind = static_cast<uint32_t>(readScalar(cursor, 4));
      if (kind == 8) {
        const uint64_t length = readScalar(cursor, 8);
        cursor += length;
      } else if (kind == 9) {
        const uint32_t element = static_cast<uint32_t>(readScalar(cursor, 4));
        const uint64_t count = readScalar(cursor, 8);
        for (uint64_t i = 0; i < count; ++i) {
          if (element == 8) {
            const uint64_t length = readScalar(cursor, 8);
            cursor += length;
          } else if (element == 4 || element == 5 || element == 6) cursor += 4;
          else if (element == 7) cursor += 1;
          else if (element == 10 || element == 11 || element == 12) cursor += 8;
          else throw std::runtime_error("unexpected array element kind");
        }
      } else if (kind == 4 || kind == 5 || kind == 6) cursor += 4;
      else if (kind == 7) cursor += 1;
      else if (kind == 10 || kind == 11 || kind == 12) cursor += 8;
      else throw std::runtime_error("unexpected metadata kind");
    }
    for (uint64_t index = 0; index < tensorCount; ++index) {
      const uint64_t nameLength = readScalar(cursor, 8);
      Tensor tensor;
      tensor.name = std::string(header.data() + cursor, nameLength);
      cursor += nameLength;
      const uint32_t dimensions = static_cast<uint32_t>(readScalar(cursor, 4));
      for (uint32_t axis = 0; axis < dimensions; ++axis)
        tensor.shape.push_back(readScalar(cursor, 8));
      tensor.type = static_cast<uint32_t>(readScalar(cursor, 4));
      tensor.offset = readScalar(cursor, 8);
      tensors_.push_back(tensor);
    }
    const uint64_t aligned = cursor;
    dataStart_ = aligned + (aligned % 32 ? 32 - aligned % 32 : 0);
    file_ = std::ifstream(path, std::ios::binary);
    (void)stringAt; (void)readScalar;
  }

  std::vector<float> row(const std::string &name, uint64_t row, uint64_t columns) const {
    for (const auto &tensor : tensors_)
      if (tensor.name == name) {
        const uint64_t rowBytes = tensor.type == 8 ? columns / 32 * 34 : columns / 256 * 136;
        std::vector<char> bytes(rowBytes);
        file_.clear();
        file_.seekg(static_cast<std::streamoff>(dataStart_ + tensor.offset + row * rowBytes));
        file_.read(bytes.data(), static_cast<std::streamsize>(rowBytes));
        return rowValues(reinterpret_cast<const uint8_t *>(bytes.data()), columns, tensor.type);
      }
    throw std::runtime_error("no tensor " + name);
  }

  uint32_t type(const std::string &name) const {
    for (const auto &tensor : tensors_)
      if (tensor.name == name) return tensor.type;
    throw std::runtime_error("no tensor " + name);
  }

private:
  struct Tensor final {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t type = 0;
    uint64_t offset = 0;
  };
  std::vector<struct Tensor> tensors_;
  uint64_t dataStart_ = 0;
  mutable std::ifstream file_;
};

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3) throw std::invalid_argument("usage: ripple-embed-probe METALLIB MODEL_ROOT");
    MetalBackend backend(argv[1]);
    const std::filesystem::path root(argv[2]);

    splash::model::RippleLayout layout;
    layout.nativeContextTokens = 262144;
    layout.layers = 80;
    layout.hiddenSize = kHidden;
    layout.vocabularySize = 100352;
    layout.attentionWidth = 80 * 128;
    layout.attentionQueryHeads = 80;
    layout.attentionKvHeads = 8;
    layout.attentionHeadDimension = 128;
    layout.rotaryPairs = 64;
    layout.rotaryTheta = 10000.0F;
    layout.intermediateSize = 40960;
    layout.embeddingScale = kEmbedScale;
    layout.swigluClamp = 20.0F;
    layout.valueResidual = true;
    layout.postFfnNorm = true;
    layout.maskToken = 100278;
    layout.stopTokens = {100265, 100257};
    // The family's 4+1 pattern: four sliding layers, then a full one.
    uint64_t sliding[2] = {}, full[2] = {};
    for (uint32_t layer = 0; layer < 80; ++layer)
      (layer % 5 == 4 ? full : sliding)[layer / 64] |= 1ULL << (layer % 64);
    layout.slidingLayers = {sliding[0], sliding[1]};
    layout.unrotatedLayers = {full[0], full[1]};
    layout.fullAttentionPeriod = 1;
    layout.slidingWindow = 1024;
    layout.ffnKind = splash::model::QwenFfnKind::Dense;
    layout.packedFullWidth = 2 * (80 * 128) + 2 * (8 * 128);

    const std::vector<uint32_t> draftLayers{1, 20, 39, 58, 77};
    const splash::model::DFlashDraftLayout draft = splash::model::draftLayout(
        kHidden, 100352, 5, 40960, 10000.0F, draftLayers, SPLASH_DRAFT_VARIANTS[1]);
    const splash::model::ModelDescriptor descriptor = splash::model::makeModelDescriptor(
        "ripple-65b-real", layout, draft, {.outputHiddenSize = kHidden},
        splash::model::TargetSource::Gguf, splash::model::VisionSource::None);
    std::cerr << "probe: loading\n" << std::flush;
    const LoadedModel model = splash::model::loadModel(backend, root, descriptor);
    std::cerr << "probe: loaded\n" << std::flush;

    // The gather of two rows through the real target's own dispatch.
    std::cerr << "probe: plans\n" << std::flush;
    const splash::ops::ExecutionPlans operators(backend.capabilities());
    std::cerr << "probe: geometry\n" << std::flush;
    const splash::model::RuntimeGeometry geometry =
        splash::model::RuntimeGeometry::from(model, splash::kv::Format::BFloat16, splash::ops::GdnStateFormat::Float32);
    std::cerr << "probe: target\n" << std::flush;
    auto target = std::visit(
        [&](const auto &weights) {
          return makeQwenTarget(weights, geometry.target, backend, operators, geometry.prefillChunks);
        },
        model.target);
    std::cerr << "probe: target built\n" << std::flush;
    const std::vector<uint32_t> tokens{100257, 791};
    std::cerr << "probe: tokens\n" << std::flush;
    MetalBuffer tokenBuffer = backend.allocateBuffer(tokens.size() * 4, BufferStorage::Shared, "tokens");
    std::memcpy(tokenBuffer.contents(), tokens.data(), tokens.size() * 4);
    std::cerr << "probe: output\n" << std::flush;
    MetalBuffer gathered =
        backend.allocateBuffer(tokens.size() * kHidden * 2, BufferStorage::Shared, "gathered");
    std::cerr << "probe: graph\n" << std::flush;
    CommandGraph graph;
    target->addEmbedding(graph, tokenBuffer, gathered, static_cast<uint32_t>(tokens.size()));
    std::cerr << "probe: encoded\n" << std::flush;
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    std::cerr << "probe: ran\n" << std::flush;

    // The CPU's dequantization of the same rows from the GGUF.
    Gguf gguf(root / "target" / "ripple.gguf");
    std::cout << "token_embd type " << gguf.type("token_embd.weight") << "\n";
    const auto *values = static_cast<const uint16_t *>(gathered.contents());
    double worst = 0.0;
    for (size_t index = 0; index < tokens.size(); ++index) {
      const auto expected = gguf.row("token_embd.weight", tokens[index], kHidden);
      for (uint32_t column = 0; column < kHidden; ++column) {
        const float got = splash::test::bf16ToFloat(values[index * kHidden + column]);
        const double want = expected[column] * kEmbedScale;
        worst = std::max(worst, std::fabs(got - want));
        if (index == 0 && column < 6)
          std::cout << "  [" << column << "] got " << got << ", want " << want << '\n';
      }
    }
    std::cout << "ripple-embed-probe: worst |gpu - cpu| = " << worst << (worst < 1.0 ? " PASS\n" : " FAIL\n");
    return worst < 1.0 ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "ripple-embed-probe: FAIL: " << error.what() << '\n';
    return 1;
  }
}
