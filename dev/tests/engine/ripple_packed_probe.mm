// The packed-row probe: the real layer-0 mixer's packed projection runs on a
// ramp input on the GPU, and the output row is compared against the two
// candidate row orders the packed q could hold - the sources' concatenation
// (queries, then gates, then keys) and the per-head interleave the prepare
// kernel reads (each head's queries then its gate). Whichever matches names
// the truth; neither does means the projection itself is broken.
#include "LinearNumerics.hpp"
#include "TestChecks.hpp"
#include "model/ModelFactory.hpp"
#include "model/RuntimeArenas.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/PageStorage.hpp"

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
using splash::model::QwenTargetPrefillBuffers;
using splash::test::bf16ToFloat;
using splash::test::require;

constexpr uint32_t kMixerRows = 2 * (80 * 128) + 8 * 128; // the fused q+gate+k
constexpr uint32_t kHidden = 5120;
constexpr uint32_t kPacked = 22528;
constexpr float kEmbedScale = 71.5541763305664F;

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 3) throw std::invalid_argument("usage: ripple-packed-probe METALLIB MODEL_ROOT [--tiny]");
    MetalBackend backend(argv[1]);
    const std::filesystem::path root(argv[2]);

    const bool tiny = argc > 3 && std::string(argv[3]) == "--tiny";
    const uint32_t layers = tiny ? 2 : 80, hidden = tiny ? 256 : kHidden, vocab = tiny ? 256 : 100352,
                   intermediate = tiny ? 512 : 40960;
    splash::model::RippleLayout layout;
    layout.nativeContextTokens = tiny ? 4096u : 262144u;
    layout.layers = layers;
    layout.hiddenSize = hidden;
    layout.vocabularySize = vocab;
    layout.attentionWidth = 80 * 128;
    layout.attentionQueryHeads = 80;
    layout.attentionKvHeads = 8;
    layout.attentionHeadDimension = 128;
    layout.rotaryPairs = 64;
    layout.rotaryTheta = 10000.0F;
    layout.intermediateSize = intermediate;
    layout.embeddingScale = tiny ? 1.0F : kEmbedScale;
    layout.swigluClamp = tiny ? 6.0F : 20.0F;
    layout.valueResidual = true;
    layout.postFfnNorm = true;
    layout.maskToken = tiny ? 200u : 100278u;
    layout.stopTokens = tiny ? splash::ops::StopTokens{1, 0} : splash::ops::StopTokens{100265, 100257};
    uint64_t sliding[2] = {}, full[2] = {};
    for (uint32_t layer = 0; layer < layers; ++layer) {
      const bool isFull = tiny ? layer == 1 : layer % 5 == 4;
      (isFull ? full : sliding)[layer / 64] |= 1ULL << (layer % 64);
    }
    layout.slidingLayers = {sliding[0], sliding[1]};
    layout.unrotatedLayers = {full[0], full[1]};
    layout.fullAttentionPeriod = 1;
    layout.slidingWindow = 1024;
    layout.ffnKind = splash::model::QwenFfnKind::Dense;
    layout.packedFullWidth = 2 * (80 * 128) + 2 * (8 * 128);

    const std::vector<uint32_t> draftLayers = tiny ? std::vector<uint32_t>{0}
                                                   : std::vector<uint32_t>{1, 20, 39, 58, 77};
    const splash::model::DFlashDraftLayout draft =
        splash::model::draftLayout(hidden, vocab, tiny ? 1 : 5, intermediate, 10000.0F, draftLayers,
                                   tiny ? SPLASH_DRAFT_VARIANTS[0] : SPLASH_DRAFT_VARIANTS[1]);
    const splash::model::ModelDescriptor descriptor = splash::model::makeModelDescriptor(
        "ripple-65b-real", layout, draft, {.outputHiddenSize = hidden},
        splash::model::TargetSource::Gguf, splash::model::VisionSource::None);
    std::cerr << "probe: descriptor valid=" << descriptor.valid() << ", vocab " << descriptor.capabilities.vocabularySize
              << ", ctx " << descriptor.capabilities.maximumContextTokens << ", kv win "
              << descriptor.targetKvLayout.windowTokens << " layers " << descriptor.targetKvLayout.attentionLayers
              << " draft hidden " << descriptor.draft.hiddenSize << '\n';
    const LoadedModel model = splash::model::loadModel(backend, root, descriptor);

    const splash::ops::ExecutionPlans operators(backend.capabilities());
    const splash::model::RuntimeGeometry geometry =
        splash::model::RuntimeGeometry::from(model, splash::kv::Format::BFloat16,
                                             splash::ops::GdnStateFormat::Float32);
    auto target = std::visit(
        [&](const auto &weights) {
          return splash::model::makeQwenTarget(weights, geometry.target, backend, operators,
                                               geometry.prefillChunks);
        },
        model.target);
    (void)target;

    // The ramp input, one row, through the layer-0 mixer's packed projection.
    std::vector<float> ramp(kHidden);
    for (uint32_t column = 0; column < kHidden; ++column) ramp[column] = float(column % 17) - 8.0F;
    const splash::ops::PrefillAttentionPlan plan = operators.prefillAttention(
        4, 80, model.targetKvLayout(splash::kv::Format::BFloat16));
    MetalBuffer packed;
    splash::model::withPrefillArena(backend, model, operators, splash::kv::Format::BFloat16,
                            [&](const QwenTargetPrefillBuffers &buffers) {
      // Four rows of the ramp, as a prefill chunk's rows would be.
      MetalBuffer input = backend.allocateBuffer(uint64_t{8} * kHidden * 2, BufferStorage::Shared, "input");
      std::memset(input.contents(), 0, input.sizeBytes());
      auto *inBits = static_cast<uint16_t *>(input.contents());
      for (uint32_t row = 0; row < 4; ++row)
        for (uint32_t column = 0; column < kHidden; ++column)
          inBits[uint64_t{row} * kHidden + column] = splash::test::floatToBf16(ramp[column]);
      // The prefill linear's plan writes rows in tiles of eight.
    packed = backend.allocateBuffer(uint64_t{8} * kMixerRows * 2, BufferStorage::Shared, "packed");
      CommandGraph graph;
      const auto &ripple = std::get<splash::model::RippleWeights>(model.target);
      const auto &mixer = std::get<splash::model::QwenAttentionWeights>(ripple.layers[0].mixer);
      operators.linear().addPrefill(graph, input, mixer.inputProjection, packed, 4, buffers.linearScratch);
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    });

    // The packed row, as the GPU computed it.
    const auto *bits = static_cast<const uint16_t *>(packed.contents());
    std::vector<float> got(kPacked);
    for (uint32_t index = 0; index < kPacked; ++index) got[index] = bf16ToFloat(bits[index]);

    // The prepare kernel on the packed row: identity norms, cos=1/sin=0
    // tables, and the queries it writes, dumped. The slot h read tells which
    // GGUF head's rows the kernel takes for head h.
    {
      MetalBuffer queries = backend.allocateBuffer(uint64_t{8} * 80 * 128 * 2, BufferStorage::Shared, "queries");
      MetalBuffer keys = backend.allocateBuffer(uint64_t{8} * 8 * 128 * 2, BufferStorage::Shared, "keys");
      MetalBuffer values = backend.allocateBuffer(uint64_t{8} * 8 * 128 * 2, BufferStorage::Shared, "values");
      MetalBuffer queryNormW = backend.allocateBuffer(128 * 4, BufferStorage::Shared, "qn");
      MetalBuffer keyNormW = backend.allocateBuffer(128 * 4, BufferStorage::Shared, "kn");
      for (uint32_t i = 0; i < 128; ++i) {
        static_cast<float *>(queryNormW.contents())[i] = 1.0F;
        static_cast<float *>(keyNormW.contents())[i] = 1.0F;
      }
      MetalBuffer cosT = backend.allocateBuffer(uint64_t{8} * 64 * 4, BufferStorage::Shared, "cos");
      MetalBuffer sinT = backend.allocateBuffer(uint64_t{8} * 64 * 4, BufferStorage::Shared, "sin");
      for (uint32_t i = 0; i < 8 * 64; ++i) {
        static_cast<float *>(cosT.contents())[i] = 1.0F;
        static_cast<float *>(sinT.contents())[i] = 0.0F;
      }
      CommandGraph prepare;
      splash::ops::PagedAttention::addPrefillProjection(
          prepare, backend.view(packed, 0, uint64_t{layout.packedFullWidth} * 2),
          {queryNormW, true}, {keyNormW, true}, cosT, sinT, queries, keys, values, 1, 8, 80,
          model.targetKvLayout(splash::kv::Format::BFloat16), {});
      static_cast<void>(backend.submitCommandAsync(prepare.dispatches()).wait());
      const auto *qBits = static_cast<const uint16_t *>(queries.contents());
      // Slot h's 128 values against every GGUF head's queries (bf16-rounded
      // projections are already in the packed row; the norm is identity, so
      // the values are the packed ones reordered).
      std::vector<float> slot(128);
      std::string summary;
      for (uint32_t head = 0; head < 6; ++head) {
        const uint32_t kv = head / 10, local = head % 10;
        const uint64_t base = ((uint64_t{kv} * 8 + 0) * 10 + local) * 128;
        for (uint32_t d = 0; d < 128; ++d) slot[d] = bf16ToFloat(qBits[base + d]);
        double bestConcat = 1e30, bestInter = 1e30;
        uint32_t concatHead = 0, interHead = 0;
        // The kernel norms each head's 128 values by their rms; the
        // candidates are compared the same way.
        const auto normalized = [&](uint64_t at) {
          double sum = 0;
          std::array<float, 128> values{};
          for (uint32_t d = 0; d < 128; ++d) {
            values[d] = bf16ToFloat(bits[at + d]);
            sum += double(values[d]) * values[d];
          }
          const float inverse = 1.0F / std::sqrt(float(sum / 128) + 1e-6F);
          for (uint32_t d = 0; d < 128; ++d) values[d] *= inverse;
          return values;
        };
        for (uint32_t candidate = 0; candidate < 80; ++candidate) {
          const auto concatValues = normalized(uint64_t{candidate} * 128);
          const auto interValues =
              normalized(uint64_t{candidate % 10} * 256 + uint64_t{candidate / 10} * 128);
          double concatError = 0, interError = 0;
          for (uint32_t d = 0; d < 128; ++d) {
            concatError += std::fabs(slot[d] - concatValues[d]);
            interError += std::fabs(slot[d] - interValues[d]);
          }
          if (concatError < bestConcat) { bestConcat = concatError; concatHead = candidate; }
          if (interError < bestInter) { bestInter = interError; interHead = candidate; }
        }
        summary += " slot" + std::to_string(head) + "=q" + std::to_string(concatHead) + "/" +
                   std::to_string(interHead);
      }
      std::cout << "prepare slots (concat/interleave best):" << summary << '\n';
    }

    // The attention over the four prepared rows: unit norms, identity RoPE,
    // one page, and the outputs dumped. The reference computes each row's
    // attention from the packed queries and keys directly.
    {
      constexpr uint32_t kStride = 32;
      MetalBuffer queries = backend.allocateBuffer(uint64_t{80} * kStride * 128 * 2, BufferStorage::Shared, "q");
      MetalBuffer keys = backend.allocateBuffer(uint64_t{8} * kStride * 128 * 2, BufferStorage::Shared, "k");
      MetalBuffer values = backend.allocateBuffer(uint64_t{8} * kStride * 128 * 2, BufferStorage::Shared, "v");
      MetalBuffer queryNormW = backend.allocateBuffer(128 * 4, BufferStorage::Shared, "qn");
      MetalBuffer keyNormW = backend.allocateBuffer(128 * 4, BufferStorage::Shared, "qn2");
      for (uint32_t i = 0; i < 128; ++i) {
        static_cast<float *>(queryNormW.contents())[i] = 1.0F;
        static_cast<float *>(keyNormW.contents())[i] = 1.0F;
      }
      MetalBuffer cosT = backend.allocateBuffer(uint64_t{kStride} * 64 * 4, BufferStorage::Shared, "cos");
      MetalBuffer sinT = backend.allocateBuffer(uint64_t{kStride} * 64 * 4, BufferStorage::Shared, "sin");
      for (uint32_t i = 0; i < kStride * 64; ++i) {
        static_cast<float *>(cosT.contents())[i] = 1.0F;
        static_cast<float *>(sinT.contents())[i] = 0.0F;
      }
      MetalBuffer attention = backend.allocateBuffer(uint64_t{80} * kStride * 128 * 2, BufferStorage::Shared, "attn");
      MetalBuffer partials = backend.allocateBuffer(plan.workspace.partialsBytes, BufferStorage::Shared, "part");
      MetalBuffer statistics = backend.allocateBuffer(plan.workspace.statisticsBytes, BufferStorage::Shared, "stat");
      MetalBuffer table = backend.allocateBuffer(sizeof(SplashKvPage), BufferStorage::Shared, "pt");
      static_cast<SplashKvPage *>(table.contents())[0] = 0; // page 0
      CommandGraph attend;
      splash::ops::PagedAttention::addPrefillProjection(
          attend, backend.view(packed, 0, uint64_t{4} * layout.packedFullWidth * 2), {queryNormW, true},
          {keyNormW, true}, cosT, sinT, queries, keys, values, 4, kStride, 80,
          model.targetKvLayout(splash::kv::Format::BFloat16), {});
      splash::ops::PagedAttention::addPrefillStore(
          attend, {}, keys, values, table,
          splash::ops::PagedAttention::prefillParams(0, 4, kStride, 1),
          model.targetKvLayout(splash::kv::Format::BFloat16));
      splash::ops::PagedAttention::addPrefill(
          attend, {}, queries, attention, partials, statistics, table,
          splash::ops::PagedAttention::prefillParams(0, 4, kStride, 1), plan, {});
      static_cast<void>(backend.submitCommandAsync(attend.dispatches()).wait());
      const auto *aBits = static_cast<const uint16_t *>(attention.contents());
      // The output is [kv head][row][query head in group][dimension]; rows
      // that share a softmax state read the same at every slot.
      double same = 0, magnitude1 = 0, magnitude0 = 0;
      for (uint32_t kv = 0; kv < 8; ++kv)
        for (uint32_t local = 0; local < 10; ++local) {
          const uint64_t row0 = ((uint64_t{kv} * kStride + 0) * 10 + local) * 128;
          const uint64_t row1 = ((uint64_t{kv} * kStride + 1) * 10 + local) * 128;
          for (uint32_t d = 0; d < 128; ++d) {
            same += std::fabs(bf16ToFloat(aBits[row1 + d]) - bf16ToFloat(aBits[row0 + d]));
            magnitude1 += std::fabs(bf16ToFloat(aBits[row1 + d]));
            magnitude0 += std::fabs(bf16ToFloat(aBits[row0 + d]));
          }
        }
      std::cout << "attention: mean |row1 - row0| = " << same / (80.0 * 128) << ", mean |row0| = "
                << magnitude0 / (80.0 * 128) << ", mean |row1| = " << magnitude1 / (80.0 * 128) << '\n';
    }

    // Layer 0 alone through the real target: the packed rows, the staged
    // K/V and the attention output it left behind, all dumped.
    {
      MetalBuffer tokens = backend.allocateBuffer(4 * 4, BufferStorage::Shared, "tokens");
      static_cast<uint32_t *>(tokens.contents())[0] = 0;
      static_cast<uint32_t *>(tokens.contents())[1] = 5;
      static_cast<uint32_t *>(tokens.contents())[2] = 17;
      static_cast<uint32_t *>(tokens.contents())[3] = 200;
      splash::model::withPrefillArena(backend, model, operators, splash::kv::Format::BFloat16,
                                      [&](const splash::model::QwenTargetPrefillBuffers &buffers) {
        CommandGraph graph;
        std::vector<uint32_t> positions;
        for (uint32_t index = 0; index < 4; ++index) positions.insert(positions.end(), {index, index, index});
        const MetalBuffer positionBuffer = backend.allocateBuffer(positions.size() * 4, BufferStorage::Shared, "pos");
        std::memcpy(positionBuffer.contents(), positions.data(), positions.size() * 4);
        const MetalBuffer none = backend.allocateBuffer(4, BufferStorage::Shared, "unbound");
        splash::ops::RoPE::addTables(graph, positionBuffer, none, buffers.ropeInverseFrequencies, none,
                                     buffers.ropeCos, buffers.ropeSin, none, none,
                                     {.target_rows = 4, .draft_rows = 0, .target_scale = 1.0F}, 4, 64);
        target->addEmbedding(graph, tokens, buffers.hidden[0], 4);
        splash::model::QwenTargetPrefillSequence sequence;
        sequence.lane = 0;
        sequence.rowBegin = 0;
        sequence.rows = 4;
        sequence.tokens = std::span<const uint32_t>(tokens ? std::vector<uint32_t>{0, 5, 17, 200}
                                                           : std::vector<uint32_t>{});
        static_cast<void>(sequence.tokens.empty());
        sequence.request = splash::model::QwenTargetRequest{
            std::span<const uint32_t>(),
            [](uint64_t logical) { return std::array<uint32_t, 3>{uint32_t(logical), uint32_t(logical), uint32_t(logical)}; }};
        sequence.attentionStride = 32;
        sequence.chunk = splash::ops::PagedAttention::prefillParams(0, 4, 32, 1);
        MetalBuffer table = backend.allocateBuffer(sizeof(SplashKvPage), BufferStorage::Shared, "pt");
        static_cast<SplashKvPage *>(table.contents())[0] = pages.entries({0})[0];
        sequence.pageTable = table;
        splash::model::QwenTargetPrefillChunk chunk;
        chunk.buffers = buffers;
        chunk.sequences = {&sequence, 1};
        chunk.rows = 4;
        splash::model::SpilledKvReads spilled;
        target->addPrefill(graph, {&chunk, 1}, pages.storage(), spilled, {0, 1});
        static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
        const auto dump = [&](const char *name, const metal::MetalBuffer &buffer, uint64_t values) {
          const auto *bits = static_cast<const uint16_t *>(buffer.contents());
          std::vector<float> out(values);
          for (uint64_t index = 0; index < values; ++index) out[index] = splash::test::bf16ToFloat(bits[index]);
          std::ofstream file(std::string("/tmp/layer0-") + name + ".f32", std::ios::binary);
          file.write(reinterpret_cast<const char *>(out.data()), static_cast<std::streamsize>(values * 4));
        };
        dump("packed", buffers.fullPacked, uint64_t{8} * 22528);
        dump("keys", buffers.chunkKeys, uint64_t{8} * 8 * 128);
        dump("values", buffers.chunkValues, uint64_t{8} * 8 * 128);
        dump("attention", buffers.fullAttention, uint64_t{32} * 80 * 128);
      });
    }

    // The packed row, as floats, dumped for the caller's CPU comparison.
    {
      const auto *bits = static_cast<const uint16_t *>(packed.contents());
      std::vector<float> values(kMixerRows);
      for (uint32_t index = 0; index < kMixerRows; ++index)
        values[index] = splash::test::bf16ToFloat(bits[index]);
      std::ofstream out(root / "packed-row.f32", std::ios::binary);
      out.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(kMixerRows * 4));
    }
    std::cout << "ripple-packed-probe: PASS (packed row dumped)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ripple-packed-probe: FAIL: " << error.what() << '\n';
    return 1;
  }
}
