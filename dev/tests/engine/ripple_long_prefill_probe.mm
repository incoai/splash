// The long-prefill finiteness probe: the real Ripple target's prefill,
// chunked the way the runtime chunks a long conversation (2048-row chunks
// over one lane's growing context, each chunk its own command), each chunk's
// last final-hidden row checked for finiteness, and a failing chunk re-run
// with a layer-range prefix to name the layer whose output first goes
// non-finite. The draft's context path is not in the probe: what it chases
// is the target's own rows.
//
// usage: ripple-long-prefill-probe METALLIB MODEL_ROOT TOKENS_FILE
#include "TestChecks.hpp"
#include "LinearNumerics.hpp"
#include "ClaimScope.hpp"
#include "engine/MemoryGovernor.hpp"
#include "model/QwenState.hpp"
#include "model/RuntimeArenas.hpp"
#include "ops/RowCopy.hpp"
#include "ops/RoPE.hpp"

#import <Foundation/Foundation.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::model::LoadedModel;
using splash::model::QwenTarget;
using splash::model::QwenTargetPrefillBuffers;
using splash::model::QwenTargetPrefillChunk;
using splash::model::QwenTargetPrefillSequence;
using splash::model::QwenTargetRequest;
using splash::model::RuntimeGeometry;
using splash::model::SpilledKvReads;
using splash::ops::ExecutionPlans;
using splash::test::require;

using namespace splash;

constexpr uint32_t kChunkRows = SPLASH_PREFILL_TOKEN_BUDGET;

std::vector<uint32_t> readTokens(const char *path) {
  std::ifstream in(path);
  std::vector<uint32_t> tokens;
  uint32_t id = 0;
  while (in >> id) tokens.push_back(id);
  return tokens;
}

splash::metal::MetalBuffer upload(splash::metal::MetalBackend &backend,
                                  const std::vector<uint32_t> &values, const char *label) {
  splash::metal::MetalBuffer buffer =
      backend.allocateBuffer(values.size() * 4, splash::metal::BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), values.data(), values.size() * 4);
  return buffer;
}

bool finiteRow(const void *data, uint32_t hidden) {
  const auto *values = static_cast<const uint16_t *>(data);
  for (uint32_t i = 0; i < hidden; ++i) {
    const uint32_t bits = static_cast<uint32_t>(values[i]) << 16;
    if ((bits & 0x7fffffffu) > 0x7f800000u || bits == 0xff800000u) return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
    if (argc != 4) throw std::invalid_argument(
        "usage: ripple-long-prefill-probe METALLIB MODEL_ROOT TOKENS_FILE");
    splash::metal::MetalBackend backend(argv[1]);
    const std::filesystem::path root(argv[2]);
    const std::vector<uint32_t> tokens = readTokens(argv[3]);
    std::cout << "tokens: " << tokens.size() << '\n';

    const splash::model::ModelDescriptor descriptor = splash::model::inspectModelRoot(root);
    splash::test::require(descriptor.valid(), "the probe's model descriptor is not valid");
    const splash::model::LoadedModel model = splash::model::loadModel(backend, root, descriptor);
    const splash::ops::ExecutionPlans operators(backend.capabilities());

    const splash::model::RuntimeGeometry geometry = splash::model::RuntimeGeometry::from(
        model, splash::kv::Format::Int8, splash::ops::GdnStateFormat::Float32);
    auto target = std::visit(
        [&](const auto &weights) {
          return splash::model::makeQwenTarget(weights, geometry.target, backend, operators,
                                               geometry.prefillChunks);
        },
        model.target);

    engine::MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
                                    engine::queryHostMemory, 0);
    const splash::test::ClaimScope claim(governor, splash::engine::ClaimClass::Floor);
    const splash::kv::Layout kvLayout = model.targetKvLayout(splash::kv::Format::Int8);
    const uint32_t extent = kvLayout.minimumExtentPages();
    const uint32_t pages = (tokens.size() + splash::kv::kPageTokens - 1) /
                           splash::kv::kPageTokens / extent * extent + extent;
    splash::kv::PageStorage kv(backend, governor.allocationAdmission(), kvLayout,
                               pages / extent * extent, extent);
    for (uint32_t e = 0; e < pages / extent; ++e)
      splash::test::require(static_cast<bool>(kv.allocateExtent(e)), "the probe's extent did not allocate");
    splash::model::QwenStateStorage states(backend, governor.allocationAdmission(),
                                           model.stateLayout(splash::ops::GdnStateFormat::Float32), nullptr);

    const uint32_t hidden = geometry.target.hiddenSize;
    const uint32_t layers = geometry.target.layers;
    const uint32_t chunks = (tokens.size() + kChunkRows - 1) / kChunkRows;
    std::vector<uint32_t> pageIds(pages);
    for (uint32_t page = 0; page < pages; ++page) pageIds[page] = page;

    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
      const uint32_t position = chunk * kChunkRows;
      const uint32_t rows = std::min<uint32_t>(kChunkRows, tokens.size() - position);
      const std::vector<uint32_t> chunkTokens(tokens.begin() + position,
                                              tokens.begin() + position + rows);
      model::withPrefillArena(backend, model, operators, splash::kv::Format::Int8,
                              [&](const splash::model::QwenTargetPrefillBuffers &buffers) {
        const splash::metal::MetalBuffer tokenBuffer = upload(backend, chunkTokens, "prefill tokens");
        const splash::metal::MetalBuffer positionBuffer =
            upload(backend, [&] {
              std::vector<uint32_t> values;
              values.reserve(static_cast<uint64_t>(rows) * 3);
              for (uint32_t row = 0; row < rows; ++row)
                values.insert(values.end(), {position + row, position + row, position + row});
              return values;
            }(), "rope positions");
        const splash::metal::MetalBuffer none =
            backend.allocateBuffer(4, splash::metal::BufferStorage::Shared, "unbound");
        // The chunk's page table covers the whole history: the params' page
        // table entries count the pages from the sequence's first page.
        splash::metal::MetalBuffer table = backend.allocateBuffer(
            static_cast<uint64_t>(pageIds.size()) * sizeof(SplashKvPage),
            splash::metal::BufferStorage::Shared, "page table");
        kv.writeEntries(pageIds, 0, table);
        // The chunk's full-stack command, as the runtime encodes it: the
        // last row checked for finiteness. A failing chunk bisects the
        // layers, each a command of its own.
        splash::metal::CommandGraph graph;
        QwenTargetPrefillSequence sequence;
        sequence.lane = 0;
        sequence.rowBegin = 0;
        sequence.rows = rows;
        sequence.tokens = chunkTokens;
        sequence.request = QwenTargetRequest{
            chunkTokens,
            [](uint64_t logical) {
              return std::array<uint32_t, 3>{uint32_t(logical), uint32_t(logical), uint32_t(logical)};
            }};
        sequence.attentionStride =
          (rows + splash::kv::kPageTokens - 1) / splash::kv::kPageTokens * splash::kv::kPageTokens;
        sequence.queryOffset = position;
        sequence.kvOffset = position;
        sequence.chunk = ops::PagedAttention::prefillParams(
            position, rows, sequence.attentionStride,
            (position + rows + splash::kv::kPageTokens - 1) / splash::kv::kPageTokens);
        sequence.pageTable = table;
        QwenTargetPrefillChunk modelChunk;
        modelChunk.buffers = buffers;
        modelChunk.sequences = {&sequence, 1};
        modelChunk.rows = rows;

        ops::RoPE::addTables(graph, positionBuffer, none, buffers.ropeInverseFrequencies, none,
                             buffers.ropeCos, buffers.ropeSin, none, none,
                             {.target_rows = rows, .draft_rows = 0, .target_scale = 1.0F},
                             rows, geometry.target.rotaryPairs);
        target->addEmbedding(graph, tokenBuffer, buffers.hidden[0], rows);
        SpilledKvReads spilled;
        const std::vector<MetalBuffer> finalHidden =
            target->addPrefill(graph, {&modelChunk, 1}, kv, spilled, {0, layers});
        (void)finalHidden;
        splash::metal::MetalBuffer probeRow = backend.allocateBuffer(
            static_cast<uint64_t>(hidden) * 2, splash::metal::BufferStorage::Shared, "probe row");
        splash::ops::RowCopy::add(graph, buffers.hidden[layers & 1],
                                  {rows - 1, hidden, 0}, probeRow, {0, hidden, 0}, 1, hidden);
        static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
        if (finiteRow(probeRow.contents(), hidden)) {
          std::cout << "chunk " << chunk << " (rows " << position << ".." << position + rows
                    << "): finite through the whole stack" << std::endl;
          return;
        }
        std::cout << "chunk " << chunk << " (rows " << position << ".." << position + rows
                  << "): last row NOT FINITE" << std::endl;
        uint32_t lo = 1, hi = layers;
        while (lo < hi) {
          const uint32_t mid = (lo + hi) / 2;
          splash::metal::CommandGraph layerGraph;
          splash::ops::RoPE::addTables(layerGraph, positionBuffer, none, buffers.ropeInverseFrequencies, none,
                                       buffers.ropeCos, buffers.ropeSin, none, none,
                                       {.target_rows = rows, .draft_rows = 0, .target_scale = 1.0F},
                                       rows, geometry.target.rotaryPairs);
          target->addEmbedding(layerGraph, tokenBuffer, buffers.hidden[0], rows);
          SpilledKvReads layerSpilled;
          const std::vector<MetalBuffer> partialHidden =
              target->addPrefill(layerGraph, {&modelChunk, 1}, kv, layerSpilled, {0, mid});
          (void)partialHidden;
          splash::metal::MetalBuffer layerRow = backend.allocateBuffer(
              static_cast<uint64_t>(hidden) * 2, splash::metal::BufferStorage::Shared, "layer row");
          splash::ops::RowCopy::add(layerGraph, buffers.hidden[mid & 1],
                                    {rows - 1, hidden, 0}, layerRow, {0, hidden, 0}, 1, hidden);
          static_cast<void>(backend.submitCommandAsync(layerGraph.dispatches()).wait());
          const bool finite = finiteRow(layerRow.contents(), hidden);
          std::cout << "  layers [0, " << mid << "): last row "
                    << (finite ? "finite" : "NOT FINITE") << std::endl;
          if (finite) lo = mid + 1;
          else hi = mid;
        }
        std::cout << "chunk " << chunk << ": the non-finite enters at layer " << lo << std::endl;
        throw std::runtime_error("the probe reproduced the non-finite row");
      });
    }
    std::cout << "ripple-long-prefill-probe: every chunk's last row finite\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ripple-long-prefill-probe: FAIL: " << error.what() << '\n';
    return 1;
  }
  }
}
