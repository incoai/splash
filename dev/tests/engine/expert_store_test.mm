// Streamed experts (model::ExpertStore) at five MoE shapes in turn, a small
// one, the 35B's routing, Qwen3.8-Flash-Next's, Flint's and DeepSeek-V4.1's
// (whose decode steps predict the next block's routes with its router's
// selection bias): routed experts read from a file into slots of one block's
// experts and three more, which two blocks share, run decode steps and
// prefill chunks bitwise as the resident experts do, their slots holding the
// reference repack's planes, in as many rounds as the staging needs (a
// decode step's split around the passes of the experts that were resident),
// and read ahead the next round's experts or the next block's a long chunk
// routes or a decode step predicts; a decode step's experts outlive long
// chunks of other blocks; the union of a block's steps loads at once; blocks
// of other formats share the slots of the widest expert's size, through a
// release and a restore, and a store of one block is its floor chunk alone,
// refused when the largest buffer does not hold it; long chunks load in
// rounds through half the staging, also across a release while the next
// block's experts are read ahead; a read ahead that fails fails no load that
// does not route its expert, a read of a routed expert that fails fails the
// load, and a source written since it was opened fails the load's command,
// which the store reports; a command buffer that fails before a load's first
// round fails its command with its own error alone, the load's later rounds
// not run; memory one byte short of a block's slots is refused; the slots lie
// in a floor chunk and smaller elastic chunks, the last of fewer slots,
// bitwise on the staged and the register tile, through a release and a
// restore; the store gives its elastic chunks back one at a time down to its
// floor and takes them back, bitwise at every step, its slot tables and
// address tables never naming a chunk given back, also while a read-ahead
// still reads, through an idle release, a restore at the cap, refused and
// granted regrows, a restore refused past an elastic chunk and a release to
// the floor at once; blocks of three formats give back and take again the
// chunks of their shared slots, whose floor takes more than one chunk, each
// block's experts bitwise wherever the compactions moved its slots; a load
// whose host steps do not run after a shrink repacks nothing; and slots that
// would take more chunks than the expert kernels' address table holds are
// refused.
#include "MoeFixture.hpp"
#include "TestAdmission.hpp"
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ExpertStore.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/Model.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightSource.hpp"
#include "ops/MoE.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using splash::metal::AllocationFailure;
using splash::metal::AllocationResult;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBackendError;
using splash::metal::MetalBuffer;
using splash::model::ExpertStore;
using splash::model::ExpertStoreOptions;
using splash::model::ExpertStoreStats;
using splash::model::StreamedBlocks;
using splash::ops::BlockExpertProjection;
using splash::ops::BlockMoeWeights;
using splash::ops::MoE;
using splash::ops::MoeConfig;
using splash::ops::MoeExpertTile;
using splash::ops::MoeLoadPart;
using splash::ops::MoePhase;
using splash::ops::MoePlan;
using splash::ops::MoeShape;
using splash::ops::QuantizedSegment;
using splash::test::require;
using namespace gguf_reference;
using namespace moe_fixture;

// The shapes the store runs at: 16 experts, top 4, hidden 1024, 512 wide; the
// 35B's 256 experts, top 8, at hidden 256 and 512 wide; Qwen3.8-Flash-Next's
// 512 experts, top 10, hidden 256, 640 wide; Flint's 256 experts, top 6 by a
// sigmoid router, 512 wide, with a shared expert of two ungated slices, at
// hidden 256; and DeepSeek-V4.1's 384 experts, top 6 by a biased
// sqrt-softplus router (whose bias a decode step's prediction of the next
// block's routes reads), with an ungated shared expert, at hidden 256 and 512
// wide.
constexpr MoeShape kShapes[] = {{1024, 16, 4, 512},
                                {256, 256, 8, 512},
                                {256, 512, 10, 640},
                                {.hiddenSize = 256,
                                 .experts = 256,
                                 .expertsPerToken = 6,
                                 .expertIntermediateSize = 512,
                                 .router = splash::ops::MoeRouter::SigmoidNormalized,
                                 .sharedSlices = 2,
                                 .sharedGate = false},
                                {.hiddenSize = 256,
                                 .experts = 384,
                                 .expertsPerToken = 6,
                                 .expertIntermediateSize = 512,
                                 .router = splash::ops::MoeRouter::SqrtSoftplusNormalized,
                                 .routeScale = 1.5f,
                                 .swigluLimit = 10.0f,
                                 .sharedSlices = 1,
                                 .sharedGate = false}};

// Memory a store gets past the slots a test asks for, less than any slot
// takes: the store, which takes the most slots its memory holds, takes
// exactly those.
constexpr uint64_t kSpareBytes = 4096;

// A prefill chunk of `rows` rows on the staged 32-row tiles.
MoePlan prefillPlan(const MoeShape &shape, uint32_t rows) {
  return MoE::prefillPlan(shape, rows, MoeConfig{MoeExpertTile::M32});
}

// A model's routed experts' native rows written to `file`, gate, up and down
// back to back as a GGUF holds them, and the two blocks of them a store
// streams, two layers of a target.
struct ExpertFile {
  ExpertFile(const Model &m, std::filesystem::path file) : path(std::move(file)) {
    std::vector<uint8_t> bytes;
    std::array<uint64_t, 3> offsets{};
    for (size_t p = 0; p < 3; ++p) {
      offsets[p] = bytes.size();
      bytes.insert(bytes.end(), m.routed[p].native.begin(), m.routed[p].native.end());
    }
    splash::test::writeFile(path, bytes);
    source = std::make_shared<splash::model::WeightSource>(path);
    splash::model::gguf::StreamedExperts experts;
    for (size_t p = 0; p < 3; ++p) {
      const Tensor &t = m.routed[p];
      const Fmt f = m.formats[p];
      splash::model::gguf::Repack &repack = experts.projections[p];
      repack.format = f;
      repack.rows = t.rows;
      repack.columns = t.columns;
      repack.sources.push_back({"experts-" + std::to_string(p), kQuantFormats[f].ggml_type, offsets[p], t.rows,
                                rowBytes(f, t.columns), {}, source.get()});
    }
    blocks.blocks = {experts, experts};
    blocks.sources = source;
    blocks.checks = {[file = source.get()] { file->checkUnchanged(); }};
  }
  ExpertFile(const ExpertFile &) = delete;
  ExpertFile &operator=(const ExpertFile &) = delete;

  std::filesystem::path path;
  std::shared_ptr<splash::model::WeightSource> source;
  StreamedBlocks blocks;
};

// The bytes of a slot of the model's experts: every plane of an expert in
// turn, each from a multiple of 256 bytes, in whole kWeightFileAlignment
// units (model/ExpertStore.cpp).
uint64_t slotBytes(const Model &m) {
  uint64_t bytes = 0;
  for (size_t p = 0; p < 3; ++p) {
    const splash::model::GgufPlaneBytes planes = splash::model::ggufPlaneBytes(
        kQuantFormats[m.formats[p]], m.routed[p].rows / m.shape.experts, m.routed[p].columns);
    for (const uint64_t plane : {planes.plane0, planes.plane1, planes.meta}) bytes += (plane + 255) / 256 * 256;
  }
  return splash::model::alignWeightOffset(bytes);
}

// The bytes of `slots` experts of the model's formats in one chunk, slot
// after slot, which a store's chunks of those slots take as well.
uint64_t arenaBytes(const Model &m, uint64_t slots) { return slots * slotBytes(m); }

// The readers every store of the test reads on.
const std::shared_ptr<splash::model::SsdReaders> &readers() {
  static const auto shared = std::make_shared<splash::model::SsdReaders>(splash::model::kSsdReaders);
  return shared;
}

// The options of a store of `blocks` whose chunks hold `arena` bytes of slots
// (ExpertStoreOptions::memoryBytes: the slots, its staging and tables).
ExpertStoreOptions storeOptions(uint64_t arena, const StreamedBlocks &blocks, const MoeShape &shape) {
  return {.memoryBytes = arena + ExpertStore::fixedBytes(blocks, shape), .readers = readers()};
}

// The host steps of a load of `plan`'s rows, a round each: the staging
// places the experts of a block a decode cycle's rows may route, and at least
// two; a decode step's load that they hold stages them all in one round, any
// other load half of them a round. A decode step's round on staged tiles
// splits around its resident experts' passes into two host steps
// (ExpertStore::addSplitLoadStart). A long chunk's load whose next block
// streams (`nextStreams`) reads it ahead in one step more.
uint32_t loadSteps(const MoeShape &shape, const MoePlan &plan, bool nextStreams) {
  using splash::model::ExecutionLimits;
  const uint64_t places = std::max<uint64_t>(
      2, std::min<uint64_t>(shape.experts, uint64_t{ExecutionLimits::maximumBatchWidth} *
                                               ExecutionLimits::targetVerifyRows * shape.expertsPerToken));
  const uint64_t routable = std::min<uint64_t>(shape.experts, uint64_t{plan.rows()} * shape.expertsPerToken);
  const uint64_t perRound = plan.phase() == MoePhase::Decode && routable <= places ? places : places / 2;
  if (plan.phase() == MoePhase::Decode && routable <= places &&
      plan.configuration().ggufTile == splash::ops::MoeGgufTile::Staged)
    return 2;
  const bool longChunk = plan.phase() == MoePhase::Prefill &&
                         uint64_t{plan.rows()} * shape.expertsPerToken >= 3ULL * shape.experts;
  return static_cast<uint32_t>((routable + perRound - 1) / perRound) + (longChunk && nextStreams);
}

// A step's rows of input and residual and its output, fresh.
void freshRows(MetalBackend &backend, Buffers &b, const MoeShape &shape) {
  b.input = activations(uint64_t{kMaximumRows} * shape.hiddenSize);
  b.residual = activations(uint64_t{kMaximumRows} * shape.hiddenSize);
  b.moe.input = bfloatBuffer(backend, b.input, "moe-input");
  b.moe.residual = bfloatBuffer(backend, b.residual, "moe-residual");
  b.moe.output = zeros(backend, uint64_t{kMaximumRows} * shape.hiddenSize * 2, "moe-output");
}

// The output rows of one MoE dispatch of `weights` over the first plan.rows() input rows.
std::vector<uint16_t> runWeights(MetalBackend &backend, const BlockMoeWeights &weights, Buffers &b,
                                 const MoePlan &plan) {
  allocate(backend, b, plan);
  std::memset(b.moe.output.contents(), 0, b.moe.output.sizeBytes());
  CommandGraph graph;
  MoE::add(graph, b.moe, weights, plan);
  static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
  const auto *output = static_cast<const uint16_t *>(b.moe.output.contents());
  return {output, output + uint64_t{plan.rows()} * plan.shape().hiddenSize};
}

// The chunk of `chunks` that holds `slot`, and the slot's place in it: the
// floor chunk below floorSlots, an elastic chunk from it on
// (metal/abi/MoE.h).
std::pair<uint32_t, uint32_t> chunkOf(const splash::ops::ExpertChunks &chunks, uint32_t slot) {
  if (slot < chunks.floorSlots()) return {0, slot};
  const uint32_t elastic = slot - chunks.floorSlots();
  return {1 + elastic / chunks.chunkSlots(), elastic % chunks.chunkSlots()};
}

// Every expert the last dispatch routed holds its slot's planes, in the
// slot's chunk, as the reference repacked them: a store's slots
// chunks.slotBytes() apart in every plane's view.
bool slotsHoldTheirExperts(const Model &m, const BlockMoeWeights &streamed, const Buffers &b, const MoePlan &plan) {
  const auto *selected = static_cast<const uint32_t *>(b.moe.scratch.selectedExperts.contents());
  const auto *table = static_cast<const uint32_t *>(streamed.slotTable.contents());
  const std::vector<MetalBuffer> &chunks = *streamed.chunks.planes();
  const BlockExpertProjection *resident[] = {&m.weights.gate, &m.weights.up, &m.weights.down};
  for (uint64_t route = 0; route < uint64_t{plan.rows()} * m.shape.routesPerToken(); ++route) {
    const uint32_t expert = selected[route];
    if (expert >= m.shape.experts) continue;
    const auto [chunk, local] = chunkOf(streamed.chunks, table[expert]);
    for (size_t p = 0; p < 3; ++p) {
      const QuantizedSegment &experts = resident[p]->routed;
      const uint64_t rows = experts.outputSize / m.shape.experts;
      const splash::model::GgufPlaneBytes planes =
          splash::model::ggufPlaneBytes(kQuantFormats[experts.formatId], rows, experts.inputSize);
      const MetalBuffer *want[] = {&experts.plane0, &experts.plane1, &experts.meta};
      const uint64_t bytes[] = {planes.plane0, planes.plane1, planes.meta};
      for (size_t plane = 0; plane < 3; ++plane) {
        if (!bytes[plane]) continue;
        const MetalBuffer &region = chunks[uint64_t{chunk} * MOE_EXPERT_CHUNK_ADDRESSES + p * 3 + plane];
        const uint64_t stride = streamed.chunks.slotBytes() ? streamed.chunks.slotBytes() : bytes[plane];
        const auto *got = static_cast<const uint8_t *>(region.contents()) + local * stride;
        const auto *wanted = static_cast<const uint8_t *>(want[plane]->contents()) + expert * bytes[plane];
        if (std::memcmp(got, wanted, bytes[plane])) return false;
      }
    }
  }
  return true;
}

int streamedExperts(MetalBackend &backend, const MoeShape &shape, const std::filesystem::path &directory) {
  int failures = 0;
  uint64_t decodePrefetched = 0;
  for (const Fmt f : {IQ2S, IQ3S, IQ3XXS, Q4K, Q80, IQ4NL, MXFP4}) {
    const Model m = makeModel(backend, shape, f, true);
    const std::string formats = formatNames(m);
    ExpertFile file(m, directory / "experts.bin");
    const uint64_t slots = shape.experts + 3;
    ExpertStore store(backend, file.blocks, shape,
                      storeOptions(arenaBytes(m, slots) + kSpareBytes, file.blocks, shape));
    require(store.stats().slots == slots, formats + ": the store holds another number of slots");
    std::array<BlockMoeWeights, 2> streamed{m.weights, m.weights};
    for (uint32_t block = 0; block < 2; ++block) store.bind(block, streamed[block]);
    Buffers b;
    const int before = failures;
    // Decode steps of one to four lanes and prefill chunks of fresh rows, the
    // two blocks taking turns, so each block's load evicts the other's
    // experts; the long chunks' first blocks read the second's ahead. A
    // decode step's blocks read the same rows, so the first block (the first
    // layer) predicts the second's routes exactly (both have the same router)
    // and reads them ahead.
    const MoePlan plans[] = {MoE::decodePlan(shape, 1, MoeConfig{}), MoE::decodePlan(shape, 4, MoeConfig{}),
                             prefillPlan(shape, 27),                 prefillPlan(shape, kMaximumRows),
                             MoE::decodePlan(shape, 3, MoeConfig{}), prefillPlan(shape, 9),
                             MoE::decodePlan(shape, 2, MoeConfig{}), prefillPlan(shape, 64)};
    uint32_t index = 0;
    std::array<uint64_t, 2> rounds{};
    for (const MoePlan &plan : plans)
      for (uint32_t block = 0; block < 2; ++block, ++index) {
        rounds[static_cast<size_t>(plan.phase())] += loadSteps(shape, plan, block == 0);
        const bool prefill = plan.phase() == MoePhase::Prefill;
        if (block == 0 || prefill) freshRows(backend, b, shape);
        const std::string label = formats + (prefill ? " prefill rows=" : " decode rows=") +
                                  std::to_string(plan.rows()) + " block " + std::to_string(block);
        const std::vector<uint16_t> want = runWeights(backend, m.weights, b, plan);
        const std::vector<uint16_t> got = runWeights(backend, streamed[block], b, plan);
        if (got != want) {
          printf("  %s: streamed experts' output differs from the resident experts' FAIL\n", label.c_str());
          ++failures;
        } else if (!slotsHoldTheirExperts(m, streamed[block], b, plan)) {
          printf("  %s: a routed expert's slot differs from its reference planes FAIL\n", label.c_str());
          ++failures;
        }
      }
    const ExpertStoreStats stats = store.stats();
    uint64_t loads = 0, routed = 0, read = 0, readBytes = 0;
    for (const ExpertStoreStats::Use &use : stats.uses) {
      loads += use.loads;
      routed += use.routed;
      read += use.read;
      readBytes += use.readBytes;
    }
    const ExpertStoreStats::Use &prefill = stats.uses[0], &decode = stats.uses[1];
    decodePrefetched += decode.prefetched;
    if (loads != index || read <= stats.slots || stats.resident > stats.slots || routed < read || !prefill.loads ||
        !decode.loads || !prefill.prefetched || prefill.prefetched > prefill.read || decode.prefetched > decode.read ||
        prefill.readAhead < prefill.prefetched || decode.readAhead < decode.prefetched || !stats.failure.empty()) {
      printf("  %s: %llu loads routed %llu experts and read %llu (ahead: prefill %llu of %llu, decode %llu of %llu) "
             "into %llu slots%s%s FAIL\n",
             formats.c_str(), (unsigned long long)loads, (unsigned long long)routed, (unsigned long long)read,
             (unsigned long long)prefill.prefetched, (unsigned long long)prefill.readAhead,
             (unsigned long long)decode.prefetched, (unsigned long long)decode.readAhead,
             (unsigned long long)stats.slots, stats.failure.empty() ? "" : ", failed: ", stats.failure.c_str());
      ++failures;
    }
    // Every round of a load is one host step, whose round trip its command's
    // completion reported.
    for (size_t use = 0; use < 2; ++use)
      if (const ExpertStoreStats::Use &counted = stats.uses[use];
          counted.pickup.count != rounds[use] || counted.work.count != rounds[use] ||
          counted.resume.count != rounds[use]) {
        printf("  %s: %llu loads in %llu rounds reported %llu/%llu/%llu host-step round trips FAIL\n",
               formats.c_str(), (unsigned long long)counted.loads, (unsigned long long)rounds[use],
               (unsigned long long)counted.pickup.count, (unsigned long long)counted.work.count,
               (unsigned long long)counted.resume.count);
        ++failures;
      }
    printf("%-20s streamed decode B1-4, prefill 9/27/64/263 over two blocks: %llu of %llu routed experts read in "
           "%llu rounds (ahead: prefill %llu, decode %llu of %llu), %.1f MiB, bitwise the resident experts' %s\n",
           formats.c_str(), (unsigned long long)read, (unsigned long long)routed,
           (unsigned long long)(rounds[0] + rounds[1]), (unsigned long long)prefill.prefetched,
           (unsigned long long)decode.prefetched, (unsigned long long)decode.read, double(readBytes) / (1 << 20),
           failures > before ? "FAIL" : "ok");
  }
  // The first block's prediction is exact, and the second block's load
  // follows it by two commands: some of its misses were read ahead whole.
  if (!decodePrefetched) {
    printf("  streamed decode steps never copied a predicted expert read ahead FAIL\n");
    ++failures;
  }
  return failures;
}

// A long prefill chunk's loads are recycled before the experts a decode step
// routed: three blocks share the slots of one block and the decode step's
// experts and three slots, and after a decode step of the last block two
// long chunks of the others load every expert of each, yet the same decode
// step reads nothing again (one recency order would evict its experts for
// the second chunk).
int streamedChunkOrder(MetalBackend &backend, const MoeShape &shape, const std::filesystem::path &directory) {
  const Model m = makeModel(backend, shape, Q4K, true);
  ExpertFile file(m, directory / "experts.bin");
  StreamedBlocks blocks = file.blocks;
  blocks.blocks.push_back(blocks.blocks.front());
  Buffers b;
  const MoePlan decode = MoE::decodePlan(shape, 1, MoeConfig{});
  const MoePlan chunk = prefillPlan(shape, kMaximumRows);
  freshRows(backend, b, shape);
  const std::vector<float> decodeInput = b.input, decodeResidual = b.residual;
  // The resident experts' routing of the decode rows.
  static_cast<void>(runWeights(backend, m.weights, b, decode));
  const auto *selected = static_cast<const uint32_t *>(b.moe.scratch.selectedExperts.contents());
  std::set<uint32_t> routed;
  for (uint32_t route = 0; route < decode.rows() * shape.routesPerToken(); ++route)
    if (selected[route] < shape.experts) routed.insert(selected[route]);
  const uint64_t slots = shape.experts + routed.size() + 3;
  ExpertStore store(backend, blocks, shape, storeOptions(arenaBytes(m, slots) + kSpareBytes, blocks, shape));
  std::array<BlockMoeWeights, 3> streamed{m.weights, m.weights, m.weights};
  for (uint32_t block = 0; block < 3; ++block) store.bind(block, streamed[block]);
  int failures = 0;
  const auto run = [&](uint32_t block, const MoePlan &plan, const std::string &label) {
    const std::vector<uint16_t> want = runWeights(backend, m.weights, b, plan);
    if (runWeights(backend, streamed[block], b, plan) != want) {
      printf("  streamed chunk order, %s: output differs from the resident experts' FAIL\n", label.c_str());
      ++failures;
    }
  };
  const auto decodeAgain = [&](const std::string &label) {
    b.input = decodeInput;
    b.residual = decodeResidual;
    b.moe.input = bfloatBuffer(backend, b.input, "moe-input");
    b.moe.residual = bfloatBuffer(backend, b.residual, "moe-residual");
    run(2, decode, label);
  };
  decodeAgain("decode step");
  const uint64_t decodeRead = store.stats().uses[1].read;
  freshRows(backend, b, shape);
  run(0, chunk, "first chunk");
  freshRows(backend, b, shape);
  run(1, chunk, "second chunk");
  decodeAgain("decode step again");
  const ExpertStoreStats stats = store.stats();
  const uint64_t again = stats.uses[1].read - decodeRead;
  if (stats.slots != slots || decodeRead != routed.size() || again) {
    printf("  streamed chunk order: %llu slots, the decode step read %llu of %zu experts, again %llu FAIL\n",
           (unsigned long long)stats.slots, (unsigned long long)decodeRead, routed.size(), (unsigned long long)again);
    ++failures;
  }
  printf("streamed experts: a decode step's %zu experts outlive two long chunks of other blocks in %llu slots %s\n",
         routed.size(), (unsigned long long)slots, failures ? "FAIL" : "ok");
  return failures;
}

// Steps of one block that share a load (the chunks of a layer-major prefill
// group): each routes into its own scratch, one load of their rows' rounds
// stages the union of their experts, and each step's routed work then gives
// the resident experts' bits. The two blocks take turns, so each union
// evicts the other block's experts; each load counts once.
int streamedUnionLoads(MetalBackend &backend, const MoeShape &shape, const std::filesystem::path &directory) {
  int failures = 0;
  for (const Fmt f : {Q4K, IQ3XXS}) {
    const Model m = makeModel(backend, shape, f, true);
    ExpertFile file(m, directory / "experts.bin");
    const uint64_t slots = shape.experts + 3;
    ExpertStore store(backend, file.blocks, shape,
                      storeOptions(arenaBytes(m, slots) + kSpareBytes, file.blocks, shape));
    std::array<BlockMoeWeights, 2> streamed{m.weights, m.weights};
    for (uint32_t block = 0; block < 2; ++block) store.bind(block, streamed[block]);
    constexpr std::array<uint32_t, 3> kRows{kMaximumRows, 27, 9};
    std::array<Buffers, kRows.size()> steps;
    std::vector<MoePlan> plans;
    for (const uint32_t rows : kRows) plans.push_back(prefillPlan(shape, rows));
    const int before = failures;
    for (uint32_t turn = 0; turn < 4; ++turn) {
      const uint32_t block = turn % 2;
      std::vector<std::vector<uint16_t>> want;
      std::vector<MoeLoadPart> parts;
      for (size_t step = 0; step < steps.size(); ++step) {
        Buffers &b = steps[step];
        freshRows(backend, b, shape);
        want.push_back(runWeights(backend, m.weights, b, plans[step]));
        std::memset(b.moe.output.contents(), 0, b.moe.output.sizeBytes());
        parts.push_back({&b.moe, &plans[step]});
      }
      const ExpertStoreStats::Use used = store.stats().uses[0];
      CommandGraph graph;
      for (size_t step = 0; step < steps.size(); ++step)
        MoE::addRouting(graph, steps[step].moe, streamed[block], plans[step]);
      store.addLoad(graph, parts, block);
      for (size_t step = 0; step < steps.size(); ++step)
        MoE::addRouted(graph, steps[step].moe, streamed[block], plans[step]);
      static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
      for (size_t step = 0; step < steps.size(); ++step) {
        const auto *output = static_cast<const uint16_t *>(steps[step].moe.output.contents());
        if (!std::equal(want[step].begin(), want[step].end(), output) ||
            !slotsHoldTheirExperts(m, streamed[block], steps[step], plans[step])) {
          printf("  %s block %u step of %u rows: a union load's output or slots differ from the resident "
                 "experts' FAIL\n",
                 fmtName(f), block, kRows[step]);
          ++failures;
        }
      }
      const ExpertStoreStats::Use &use = store.stats().uses[0];
      const uint32_t rounds = loadSteps(shape, prefillPlan(shape, kRows[0] + kRows[1] + kRows[2]), block == 0);
      if (use.loads != used.loads + 1 || use.pickup.count != used.pickup.count + rounds) {
        printf("  %s: a union load counted %llu loads in %llu host steps, not %u FAIL\n", fmtName(f),
               (unsigned long long)(use.loads - used.loads),
               (unsigned long long)(use.pickup.count - used.pickup.count), rounds);
        ++failures;
      }
    }
    printf("%-20s streamed union loads of three steps (263/27/9 rows), two blocks in turn, one load each: "
           "bitwise the resident experts' %s\n",
           fmtName(f), failures > before ? "FAIL" : "ok");
  }
  return failures;
}

// Replaces a method of the class of `object`, every command buffer's, while
// it lives.
class MethodReplacement final {
public:
  MethodReplacement(id object, SEL selector, IMP replacement)
      : method_(class_getInstanceMethod(object_getClass(object), selector)) {
    require(method_ != nullptr, "replacement method is missing");
    original = method_setImplementation(method_, replacement);
  }
  ~MethodReplacement() { method_setImplementation(method_, original); }
  MethodReplacement(const MethodReplacement &) = delete;
  MethodReplacement &operator=(const MethodReplacement &) = delete;
  IMP original = nullptr;

private:
  Method method_ = nullptr;
};

// Once armed, the command buffer that raises a command's first host step's
// event (its value 1) never raises it and fails once it ends, as a buffer the
// GPU faults in, or whose wait Metal times out, does.
std::atomic<bool> faultFirstStep{false};
std::atomic<void *> faulted{nullptr};
IMP originalSignal = nullptr, originalStatus = nullptr, originalError = nullptr;

void signalUnlessFaulted(id command, SEL selector, id<MTLSharedEvent> event, uint64_t value) {
  if (value != 1 || !faultFirstStep.exchange(false)) {
    reinterpret_cast<void (*)(id, SEL, id<MTLSharedEvent>, uint64_t)>(originalSignal)(command, selector, event, value);
    return;
  }
  [command addCompletedHandler:^(id<MTLCommandBuffer> ended) {
    faulted.store((__bridge void *)ended);
  }];
}

MTLCommandBufferStatus faultedStatus(id command, SEL selector) {
  if ((__bridge void *)command == faulted.load()) return MTLCommandBufferStatusError;
  return reinterpret_cast<MTLCommandBufferStatus (*)(id, SEL)>(originalStatus)(command, selector);
}

NSError *faultedError(id command, SEL selector) {
  if ((__bridge void *)command == faulted.load())
    return [NSError errorWithDomain:MTLCommandBufferErrorDomain
                               code:MTLCommandBufferErrorTimeout
                           userInfo:@{NSLocalizedDescriptionKey : @"test timeout before a load's first round"}];
  return reinterpret_cast<NSError *(*)(id, SEL)>(originalError)(command, selector);
}

// A command buffer that fails before a load's first round (a GPU fault, or a
// wait Metal times out) leaves the round unrun: the load's later rounds do
// not run either, and the command fails with that buffer's error alone, not
// with a round of the load following another block's load. Block 0's long
// chunk loads, then block 1's, in rounds, with the buffer before its first
// round failing.
int streamedFaultBeforeRound(const std::string &metallibPath, const MoeShape &shape,
                             const std::filesystem::path &directory) {
  MetalBackend backend(metallibPath);
  const Model m = makeModel(backend, shape, Q4K, true);
  ExpertFile file(m, directory / "experts.bin");
  const uint64_t slots = 2 * shape.experts;
  ExpertStore store(backend, file.blocks, shape, storeOptions(arenaBytes(m, slots), file.blocks, shape));
  std::array<BlockMoeWeights, 2> streamed{m.weights, m.weights};
  for (uint32_t block = 0; block < 2; ++block) store.bind(block, streamed[block]);
  Buffers b;
  freshRows(backend, b, shape);
  const MoePlan chunk = prefillPlan(shape, kMaximumRows);
  static_cast<void>(runWeights(backend, streamed[0], b, chunk));
  std::string error;
  {
    id<MTLCommandBuffer> command = [[MTLCreateSystemDefaultDevice() newCommandQueue] commandBuffer];
    MethodReplacement signal(command, @selector(encodeSignalEvent:value:),
                             reinterpret_cast<IMP>(signalUnlessFaulted));
    originalSignal = signal.original;
    MethodReplacement status(command, @selector(status), reinterpret_cast<IMP>(faultedStatus));
    originalStatus = status.original;
    MethodReplacement failure(command, @selector(error), reinterpret_cast<IMP>(faultedError));
    originalError = failure.original;
    faultFirstStep = true;
    try {
      static_cast<void>(runWeights(backend, streamed[1], b, chunk));
    } catch (const MetalBackendError &caught) {
      error = caught.what();
    }
  }
  faultFirstStep = false;
  faulted.store(nullptr);
  const uint32_t rounds = loadSteps(shape, chunk, false);
  if (rounds < 2 || error.find("test timeout before a load's first round") == std::string::npos ||
      error.find("host step failed") != std::string::npos) {
    printf("  a buffer that failed before block 1's load of %u rounds failed its command with \"%s\" FAIL\n", rounds,
           error.c_str());
    return 1;
  }
  printf("streamed experts: a buffer that fails before a load's first round fails its command alone, the load's %u "
         "rounds not run ok\n",
         rounds);
  return 0;
}

// Blocks whose experts take other formats (a layer a dynamic quantization
// gave formats of its own) share the slots of the widest expert's size: the
// trunk's two blocks in Q4_K and the head's block in Q8_0 all stream through
// them, in the same chunks, each step bitwise the resident experts', the
// head's too, before and after a release and a restore, after which no
// expert is resident; each load runs its rounds. A store of the head's block
// alone holds every one of its experts, in its floor chunk alone: under a
// largest buffer of exactly its experts' bytes the store takes it, and under
// one a byte smaller refuses it; its steps give the resident experts' bits
// and load nothing (no host step, no read).
int streamedMixedLayouts(MetalBackend &backend, const MoeShape &shape, const std::filesystem::path &directory) {
  const Model trunk = makeModel(backend, shape, Q4K, true), head = makeModel(backend, shape, Q80, true);
  ExpertFile trunkFile(trunk, directory / "trunk.bin"), headFile(head, directory / "head.bin");
  StreamedBlocks blocks = trunkFile.blocks;
  blocks.blocks.push_back(headFile.blocks.blocks.front());
  blocks.checks.push_back(headFile.blocks.checks.front());
  const uint64_t slots = shape.experts + 3, width = std::max(slotBytes(trunk), slotBytes(head));
  ExpertStore store(backend, blocks, shape, storeOptions(slots * width + kSpareBytes, blocks, shape));
  require(store.stats().slots == slots && store.stats().resident == 0,
          "blocks of other formats hold another number of slots, or some expert before any load");
  std::array<BlockMoeWeights, 3> streamed{trunk.weights, trunk.weights, head.weights};
  for (uint32_t block = 0; block < 3; ++block) store.bind(block, streamed[block]);
  // Gate plane0 starts every slot: the head's views and the trunk's are of
  // the same chunks.
  const std::vector<MetalBuffer> &trunkViews = *streamed[0].chunks.planes(), &headViews = *streamed[2].chunks.planes();
  require(streamed[0].stream && streamed[1].stream && streamed[2].stream && headViews.size() == trunkViews.size() &&
              trunkViews.front().contents() == headViews.front().contents() &&
              streamed[2].chunks.slotBytes() == width && streamed[0].chunks.slotBytes() == width,
          "a block of another format streams apart from the others, or in slots of another size");
  const MoePlan chunk = prefillPlan(shape, kMaximumRows);
  const MoePlan decode = MoE::decodePlan(shape, 1, MoeConfig{});
  int failures = 0;
  const struct Step {
    uint32_t block;
    const MoePlan *plan;
    bool restores; // after a release and a restore of the store
  } steps[] = {{0, &chunk, false}, {2, &chunk, false}, {2, &decode, false},
               {1, &chunk, false}, {2, &chunk, true},  {0, &decode, false}};
  for (const Step &step : steps) {
    if (step.restores) {
      store.release();
      require(store.restore(splash::test::admitAll, splash::test::admitAll) && store.stats().resident == 0,
              "a restored store holds experts before any load");
    }
    const Model &m = step.block == 2 ? head : trunk;
    Buffers b;
    freshRows(backend, b, shape);
    const std::vector<uint16_t> want = runWeights(backend, m.weights, b, *step.plan);
    const ExpertStoreStats before = store.stats();
    const std::vector<uint16_t> got = runWeights(backend, streamed[step.block], b, *step.plan);
    const ExpertStoreStats after = store.stats();
    uint64_t hostSteps = 0;
    for (size_t use = 0; use < 2; ++use) hostSteps += after.uses[use].pickup.count - before.uses[use].pickup.count;
    if (got != want || !slotsHoldTheirExperts(m, streamed[step.block], b, *step.plan) ||
        hostSteps != loadSteps(shape, *step.plan, step.block < 2)) {
      printf("  block %u%s: %llu host steps, or its output or slots differ from the resident experts' FAIL\n",
             step.block, step.restores ? " after a restore" : "", (unsigned long long)hostSteps);
      ++failures;
    }
  }
  // A store of the head's block alone.
  StreamedBlocks lone = headFile.blocks;
  lone.blocks.resize(1);
  ExpertStoreOptions options = storeOptions(arenaBytes(head, shape.experts) + kSpareBytes, lone, shape);
  options.bufferBytes = arenaBytes(head, shape.experts) - 1;
  splash::test::rejects([&] { ExpertStore refused(backend, lone, shape, options); },
                        "more than the device's largest buffer",
                        "a store of one block larger than the largest buffer was taken");
  ++options.bufferBytes;
  ExpertStore alone(backend, lone, shape, options);
  BlockMoeWeights single = head.weights;
  alone.bind(0, single);
  require(alone.stats().slots == shape.experts && alone.stats().resident == shape.experts && !single.stream &&
              single.chunks.planes()->size() == MOE_EXPERT_CHUNK_ADDRESSES &&
              single.chunks.floorSlots() == shape.experts,
          "a store of one block is not its floor chunk alone of every one of its experts, or it streams");
  for (const MoePlan *plan : {&chunk, &decode}) {
    Buffers b;
    freshRows(backend, b, shape);
    const std::vector<uint16_t> want = runWeights(backend, head.weights, b, *plan);
    const ExpertStoreStats before = alone.stats();
    const bool same = runWeights(backend, single, b, *plan) == want && slotsHoldTheirExperts(head, single, b, *plan);
    const ExpertStoreStats after = alone.stats();
    bool loaded = false;
    for (size_t use = 0; use < 2; ++use)
      loaded = loaded || after.uses[use].read != before.uses[use].read ||
               after.uses[use].pickup.count != before.uses[use].pickup.count;
    if (!same || loaded) {
      printf("  a store of one block, %u rows: its output or slots differ from the resident experts', or it "
             "loaded FAIL\n",
             plan->rows());
      ++failures;
    }
  }
  printf("streamed experts of two formats in slots of the wider one's size, before and after a restore, and a "
         "store of one block in its floor chunk alone: bitwise the resident experts' %s\n",
         failures ? "FAIL" : "ok");
  return failures;
}

// Long prefill chunks load in rounds through half the staging each, the
// readers reading the next round's experts ahead into the other half while
// the GPU repacks, and after the last round the next block's: a store of
// two blocks' slots holds both blocks' experts, so a decode step of the first
// chunk's rows after the chunks reads nothing. A release while the
// read-ahead of the second
// block's experts runs ends it: the chunks after the restore read every
// expert they route again. Each step gives the resident experts' bits, and
// each load's host steps are its rounds.
int streamedRounds(MetalBackend &backend, const MoeShape &shape, const std::filesystem::path &directory) {
  const Model m = makeModel(backend, shape, Q4K, true);
  ExpertFile file(m, directory / "experts.bin");
  const uint64_t slots = 2 * shape.experts;
  ExpertStore store(backend, file.blocks, shape, storeOptions(arenaBytes(m, slots), file.blocks, shape));
  std::array<BlockMoeWeights, 2> streamed{m.weights, m.weights};
  for (uint32_t block = 0; block < 2; ++block) store.bind(block, streamed[block]);
  const MoePlan decode = MoE::decodePlan(shape, 1, MoeConfig{});
  const MoePlan chunk = prefillPlan(shape, kMaximumRows);
  int failures = 0;
  const struct Step {
    uint32_t block;
    bool prefill;
    bool releases; // before the step, while the last step's read-ahead runs
    bool reads;    // every expert it routes, else none
  } steps[] = {{0, true, false, true},
               {1, true, false, true},
               {0, false, false, false},
               {0, true, true, true},
               {1, true, true, true}};
  Buffers b;
  std::vector<float> firstInput, firstResidual;
  for (const Step &step : steps) {
    if (step.releases) {
      store.release();
      require(store.restore(splash::test::admitAll, splash::test::admitAll), "the store was not restored");
    }
    const MoePlan &plan = step.prefill ? chunk : decode;
    if (step.prefill) {
      freshRows(backend, b, shape);
      if (firstInput.empty()) {
        firstInput = b.input;
        firstResidual = b.residual;
      }
    } else {
      b.input = firstInput;
      b.residual = firstResidual;
      b.moe.input = bfloatBuffer(backend, b.input, "moe-input");
      b.moe.residual = bfloatBuffer(backend, b.residual, "moe-residual");
    }
    const std::vector<uint16_t> want = runWeights(backend, m.weights, b, plan);
    const ExpertStoreStats::Use used = store.stats().uses[step.prefill ? 0 : 1];
    const std::vector<uint16_t> got = runWeights(backend, streamed[step.block], b, plan);
    const ExpertStoreStats::Use &use = store.stats().uses[step.prefill ? 0 : 1];
    const uint64_t hostSteps = use.pickup.count - used.pickup.count, read = use.read - used.read,
                   routed = use.routed - used.routed;
    const uint32_t rounds = loadSteps(shape, plan, step.block == 0);
    if (got != want || !slotsHoldTheirExperts(m, streamed[step.block], b, plan) || hostSteps != rounds ||
        read != (step.reads ? routed : 0) || !store.stats().failure.empty()) {
      printf("  block %u %s%s: %llu host steps of %u rounds, %llu of %llu routed experts read, or its output or "
             "slots differ from the resident experts' FAIL\n",
             step.block, step.prefill ? "chunk" : "decode step", step.releases ? " after a release" : "",
             (unsigned long long)hostSteps, rounds, (unsigned long long)read, (unsigned long long)routed);
      ++failures;
    }
  }
  printf("streamed experts loading long chunks in %u rounds through a staging of %llu KiB, across a release: "
         "bitwise the resident experts' %s\n",
         loadSteps(shape, chunk, false), (unsigned long long)(store.stats().stagingBytes >> 10),
         failures ? "FAIL" : "ok");
  return failures;
}

// A read that fails twice fails a load only when the load routes its
// expert. Of two blocks, the second reads a file whose last bytes were cut,
// so its last expert's rows cannot be read (and those whose rows,
// widened to whole pages, end past the cut). A decode step of the first
// block predicts the second block's routes and reads them ahead, the last
// expert among them, which fails; a chunk of the second block that routes
// none of the experts cut runs as the resident experts do, and the store
// reports no failure; the store then gives a chunk above its floor back. A
// chunk that routes the last expert fails its command and frees the slots it
// claimed; the store, which holds its failure, then neither gives another
// chunk back nor would take one; the backend refuses every later command,
// and the store keeps its failure.
int streamedReadFailures(const std::string &metallibPath, const MoeShape &shape,
                         const std::filesystem::path &directory) {
  MetalBackend backend(metallibPath);
  const Model m = makeModel(backend, shape, Q4K, true);
  ExpertFile intact(m, directory / "intact.bin"), cut(m, directory / "cut.bin");
  StreamedBlocks blocks;
  blocks.blocks = {intact.blocks.blocks.front(), cut.blocks.blocks.front()};
  const uint64_t slots = 2 * shape.experts;
  ExpertStoreOptions options = storeOptions(arenaBytes(m, slots), blocks, shape);
  options.floorBytes = ExpertStore::fixedBytes(blocks, shape) + arenaBytes(m, shape.experts);
  options.elasticChunkBytes = arenaBytes(m, slots / 8);
  ExpertStore store(backend, blocks, shape, options);
  std::array<BlockMoeWeights, 2> streamed{m.weights, m.weights};
  for (uint32_t block = 0; block < 2; ++block) store.bind(block, streamed[block]);
  // The down rows are the file's last: cut half of the last expert's.
  const uint64_t fileBytes = std::filesystem::file_size(cut.path);
  const uint64_t downBytes = m.routed[2].native.size() / shape.experts;
  std::filesystem::resize_file(cut.path, fileBytes - downBytes / 2);
  const uint64_t downStart = fileBytes - m.routed[2].native.size();
  constexpr uint64_t kPage = 16 << 10;
  std::set<uint32_t> unreadable;
  for (uint32_t expert = 0; expert < shape.experts; ++expert)
    if ((downStart + (expert + 1) * downBytes + kPage - 1) / kPage * kPage > fileBytes - downBytes / 2)
      unreadable.insert(expert);
  // Fresh rows until the resident experts' routing of `plan`'s rows routes
  // the last expert or, with !last, none of the unreadable ones.
  const auto rowsRouting = [&](Buffers &b, const MoePlan &plan, bool last) {
    for (int attempt = 0; attempt < 1000; ++attempt) {
      freshRows(backend, b, shape);
      static_cast<void>(runWeights(backend, m.weights, b, plan));
      const auto *selected = static_cast<const uint32_t *>(b.moe.scratch.selectedExperts.contents());
      bool routesLast = false, routesUnreadable = false;
      for (uint64_t route = 0; route < uint64_t{plan.rows()} * shape.routesPerToken(); ++route) {
        routesLast |= selected[route] == shape.experts - 1;
        routesUnreadable |= unreadable.contains(selected[route]);
      }
      if (last ? routesLast : !routesUnreadable) return;
    }
    throw std::runtime_error("no rows route as the test needs");
  };
  int failures = 0;
  const MoePlan decode = MoE::decodePlan(shape, 1, MoeConfig{}), row = prefillPlan(shape, 1);
  Buffers b;
  // The first block's decode step twice: the second finds its experts
  // resident and reads the predicted ones ahead into all but its places.
  rowsRouting(b, decode, true);
  for (int pass = 0; pass < 2; ++pass) static_cast<void>(runWeights(backend, streamed[0], b, decode));
  rowsRouting(b, row, false);
  const std::vector<uint16_t> want = runWeights(backend, m.weights, b, row);
  if (runWeights(backend, streamed[1], b, row) != want || !store.stats().failure.empty()) {
    printf("  streamed read failures: a chunk that routes no expert cut differs or fails (%s) FAIL\n",
           store.stats().failure.c_str());
    ++failures;
  }
  if (!store.releaseAboveFloor() || !store.releaseAboveFloor()) {
    printf("  streamed read failures: the store above its floor gave no chunk back before the failed load FAIL\n");
    ++failures;
  }
  const uint64_t resident = store.stats().resident;
  rowsRouting(b, row, true);
  std::string error;
  try {
    static_cast<void>(runWeights(backend, streamed[1], b, row));
  } catch (const MetalBackendError &caught) {
    error = caught.what();
  }
  const ExpertStoreStats stats = store.stats();
  // The store refuses to shrink or grow while it holds a failure: it gives
  // no chunk back, though some lie above its floor, and would take none
  // back, though it gave some.
  if (store.releaseAboveFloor() || store.growthBytes() || store.stats().chunks != stats.chunks ||
      stats.chunks <= 1) {
    printf("  streamed read failures: the store gave a chunk back or would take one after a failed load FAIL\n");
    ++failures;
  }
  if (error.find("host step failed") == std::string::npos || stats.failure.find("truncated") == std::string::npos ||
      stats.resident != resident) {
    printf("  streamed read failures: a chunk that routes the expert cut did not fail its command (%s), the store "
           "reports \"%s\", or %llu experts are resident, not %llu FAIL\n",
           error.c_str(), stats.failure.c_str(), (unsigned long long)stats.resident, (unsigned long long)resident);
    ++failures;
  }
  error.clear();
  try {
    static_cast<void>(runWeights(backend, streamed[0], b, row));
  } catch (const MetalBackendError &caught) {
    error = caught.what();
  }
  if (error.empty() || store.stats().failure.find("truncated") == std::string::npos) {
    printf("  streamed read failures: a load after a failed one ran, or the store forgot its failure FAIL\n");
    ++failures;
  }
  printf("streamed experts: a failed read ahead of an expert no load routes fails nothing; a failed read of a routed "
         "expert fails its command, and the store keeps its failure %s\n",
         failures ? "FAIL" : "ok");
  return failures;
}

// A store refuses memory a byte short of a block's slots beside its staging
// and tables, and takes exactly that much.
int streamedRefusal(MetalBackend &backend, const MoeShape &shape, const std::filesystem::path &directory) {
  const Model m = makeModel(backend, shape, Q4K, true);
  ExpertFile file(m, directory / "experts.bin");
  ExpertStoreOptions options = storeOptions(arenaBytes(m, shape.experts), file.blocks, shape);
  --options.memoryBytes;
  std::string refusal;
  try {
    ExpertStore refused(backend, file.blocks, shape, options);
  } catch (const std::invalid_argument &error) {
    refusal = error.what();
  }
  ++options.memoryBytes;
  ExpertStore store(backend, file.blocks, shape, options);
  if (refusal.find("holds fewer slots") == std::string::npos || store.stats().slots != shape.experts) {
    printf("streamed experts: memory a byte short of a block's slots was not refused (%s), or exactly enough holds "
           "%llu slots FAIL\n",
           refusal.c_str(), (unsigned long long)store.stats().slots);
    return 1;
  }
  printf("streamed experts: memory a byte short of a block's slots is refused, exactly enough taken ok\n");
  return 0;
}

// A size keeps its slots in a floor chunk and elastic chunks of about
// ExpertStoreOptions::elasticChunkBytes: two blocks, in memory for both
// blocks' experts but three, with elastic chunks of a fourteenth of those
// slots' bytes, take them in a floor chunk of the floor share, more than a
// block's experts, and three elastic chunks of fewer slots than the floor,
// the last holding fewer, which together take the bytes of one chunk of them
// all and which the blocks share. Decode steps of one to four lanes and
// prefill chunks on the staged and the register tile, the blocks taking turns
// so that their slots recycle across the chunks, give the resident experts'
// bits, their slots holding their planes (gate and up in Q5_K, whose second
// plane every chunk holds a region of), before and after a release and a
// restore, the blocks' experts sitting in more than one chunk.
int streamedChunks(MetalBackend &backend, const MoeShape &shape, const std::filesystem::path &directory) {
  using splash::ops::MoeGgufTile;
  const Model m = makeModel(backend, shape, Q5K, true, MoeGgufTile::Register);
  ExpertFile file(m, directory / "experts.bin");
  const uint64_t slots = 2 * shape.experts - 3, arena = arenaBytes(m, slots);
  ExpertStoreOptions options = storeOptions(arena + kSpareBytes, file.blocks, shape);
  options.elasticChunkBytes = arenaBytes(m, slots / 14);
  ExpertStore store(backend, file.blocks, shape, options);
  std::array<BlockMoeWeights, 2> streamed{m.weights, m.weights};
  for (uint32_t block = 0; block < 2; ++block) store.bind(block, streamed[block]);
  const splash::ops::ExpertChunks &chunks = streamed[0].chunks;
  const uint64_t count = chunks.planes()->size() / MOE_EXPERT_CHUNK_ADDRESSES;
  const uint64_t floorSlots = chunks.floorSlots(), chunkSlots = chunks.chunkSlots();
  const uint64_t elastic = slots > floorSlots ? slots - floorSlots : 0;
  uint64_t chunkBytes = arenaBytes(m, std::min(floorSlots, slots));
  for (uint64_t first = floorSlots; first < slots; first += chunkSlots)
    chunkBytes += arenaBytes(m, std::min(chunkSlots, slots - first));
  int failures = 0;
  if (store.stats().slots != slots || count != 4 || floorSlots <= shape.experts || chunkSlots >= floorSlots ||
      !elastic || elastic % chunkSlots == 0 || count != 1 + (elastic + chunkSlots - 1) / chunkSlots ||
      streamed[1].chunks.planes() != chunks.planes() || chunkBytes != arena || store.stats().arenaBytes != arena) {
    printf("  chunks: %llu slots in a floor of %llu and %llu chunks in all of %llu slots, %llu bytes (%llu), not "
           "%llu slots in a floor of more than a block's experts and three smaller elastic chunks ending in a "
           "partial one, of %llu bytes FAIL\n",
           (unsigned long long)store.stats().slots, (unsigned long long)floorSlots, (unsigned long long)count,
           (unsigned long long)chunkSlots, (unsigned long long)store.stats().arenaBytes,
           (unsigned long long)chunkBytes, (unsigned long long)slots, (unsigned long long)arena);
    ++failures;
  }
  const MoeConfig registerTile{.expertTile = MoeExpertTile::M8, .ggufTile = MoeGgufTile::Register};
  std::vector<MoePlan> plans;
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    plans.push_back(MoE::decodePlan(shape, lanes, MoeConfig{}));
    plans.push_back(MoE::decodePlan(shape, lanes, registerTile));
  }
  for (const uint32_t rows : {9u, 64u, kMaximumRows}) {
    plans.push_back(prefillPlan(shape, rows));
    plans.push_back(MoE::prefillPlan(shape, rows, registerTile));
  }
  Buffers b;
  for (const bool restored : {false, true}) {
    if (restored) {
      store.release();
      require(store.restore(splash::test::admitAll, splash::test::admitAll), "the chunked store did not restore");
    }
    for (const MoePlan &plan : plans)
      for (uint32_t block = 0; block < 2; ++block) {
        freshRows(backend, b, shape);
        const std::vector<uint16_t> want = runWeights(backend, m.weights, b, plan);
        if (runWeights(backend, streamed[block], b, plan) != want ||
            !slotsHoldTheirExperts(m, streamed[block], b, plan)) {
          printf("  chunks%s, block %u, %u rows on the %s tile: output or slots differ from the resident experts' "
                 "FAIL\n",
                 restored ? " after a restore" : "", block, plan.rows(),
                 plan.configuration().ggufTile == MoeGgufTile::Register ? "register" : "staged");
          ++failures;
        }
      }
  }
  // The blocks' resident experts sit in more than one chunk.
  std::set<uint32_t> named;
  for (const BlockMoeWeights &weights : streamed)
    for (uint32_t expert = 0; expert < shape.experts; ++expert)
      named.insert(chunkOf(chunks, static_cast<const uint32_t *>(weights.slotTable.contents())[expert]).first);
  if (named.size() < 2) {
    printf("  chunks: the resident experts sit in one chunk FAIL\n");
    ++failures;
  }
  printf("streamed experts in a floor of %llu slots and %llu chunks in all of %llu, the last of fewer: bitwise the "
         "resident experts' on the staged and register tiles, through a restore %s\n",
         (unsigned long long)floorSlots, (unsigned long long)count, (unsigned long long)chunkSlots,
         failures ? "FAIL" : "ok");
  return failures;
}

// A size gives its elastic chunks back one at a time, the last live first, down
// to its floor, and takes them back: two blocks in memory for both blocks'
// experts but an eighth of a block's, with a floor of a block's experts
// (ExpertStoreOptions::floorBytes) and elastic chunks of an eighth of those
// slots' bytes. Each chunk given back returns its bytes; the experts it held
// that fit move below it, a shrink reads nothing, the store keeps as many
// experts as the live slots hold, no slot table names a slot past them, and
// both blocks' chunks name the live chunks alone, one list per count of them
// (the backend refuses a command whose address table reaches memory given
// back). Decode steps and prefill chunks after each step give the resident
// experts' bits, their slots holding their planes. A chunk also goes while a
// read-ahead of the next block's experts still reads into the staging, at once
// after the command whose last host step started it (a long chunk's read of the
// next block, a decode step's of the experts its prediction names), and the
// next block's load takes over what it read where the compaction left its
// slots. At the floor nothing more goes. An idle release after the shrinks
// releases the live chunks and counts none live, and a restore takes every
// chunk back; a refused regrow leaves its chunk released, granted ones add free
// slots up to the cap and read nothing; a restore whose optional admission
// refuses past an elastic chunk keeps the chunks before it, which regrows
// complete; and a release to the floor gives every chunk above it back at once.
int streamedElastic(MetalBackend &backend, const MoeShape &shape, const std::filesystem::path &directory) {
  const Model m = makeModel(backend, shape, Q5K, true);
  ExpertFile file(m, directory / "experts.bin");
  const uint64_t slots = 2 * shape.experts - shape.experts / 8, fixed = ExpertStore::fixedBytes(file.blocks, shape);
  ExpertStoreOptions options = storeOptions(arenaBytes(m, slots) + kSpareBytes, file.blocks, shape);
  options.floorBytes = fixed + arenaBytes(m, shape.experts);
  options.elasticChunkBytes = arenaBytes(m, slots / 8);
  ExpertStore store(backend, file.blocks, shape, options);
  std::array<BlockMoeWeights, 2> streamed{m.weights, m.weights};
  for (uint32_t block = 0; block < 2; ++block) store.bind(block, streamed[block]);
  const splash::ops::ExpertChunks &chunks = streamed[0].chunks;
  const uint32_t capChunks = chunks.chunks();
  const uint64_t floorSlots = chunks.floorSlots(), chunkSlots = chunks.chunkSlots();
  // The slots the first `live` chunks hold.
  const auto slotsOf = [&](uint32_t live) {
    return live ? std::min<uint64_t>(slots, floorSlots + uint64_t{live - 1} * chunkSlots) : 0;
  };
  int failures = 0;
  const auto fail = [&](const std::string &what) {
    printf("  elastic chunks %s FAIL\n", what.c_str());
    ++failures;
  };
  ExpertStoreStats stats = store.stats();
  if (capChunks < 3 || stats.capChunks != capChunks || stats.chunks != capChunks || stats.slots != slots ||
      floorSlots < shape.experts || stats.capBytes != fixed + arenaBytes(m, slots) ||
      stats.floorBytes != fixed + arenaBytes(m, floorSlots))
    fail("at the cap: " + std::to_string(stats.chunks) + " of " + std::to_string(capChunks) + " chunks of " +
         std::to_string(stats.slots) + " slots in a floor of " + std::to_string(floorSlots) + ", " +
         std::to_string(stats.capBytes) + " bytes at the cap and " + std::to_string(stats.floorBytes) +
         " at the floor, not at least three chunks of " + std::to_string(slots) +
         " slots in a floor of a block's experts");
  // Each count of live chunks has its one list, which both blocks name.
  std::vector<const std::vector<MetalBuffer> *> lists(capChunks + 1, nullptr);
  const auto requireLists = [&](uint32_t live, const std::string &when) {
    const std::vector<MetalBuffer> *list = chunks.planes().get();
    if (!lists[live]) lists[live] = list;
    bool distinct = true;
    for (uint32_t other = 1; other <= capChunks; ++other)
      distinct = distinct && (other == live || lists[other] != list);
    if (chunks.liveChunks() != live || list->size() != uint64_t{live} * MOE_EXPERT_CHUNK_ADDRESSES ||
        streamed[1].chunks.planes() != chunks.planes() || lists[live] != list || !distinct)
      fail(when + ": the blocks' chunks do not name the " + std::to_string(live) + " live chunks in their one list");
  };
  // No slot table names a slot past the live chunks'.
  const auto tablesBelow = [&](uint64_t live) {
    for (const BlockMoeWeights &weights : streamed) {
      const auto *table = static_cast<const uint32_t *>(weights.slotTable.contents());
      for (uint32_t expert = 0; expert < shape.experts; ++expert)
        if (table[expert] >= live) return false;
    }
    return true;
  };
  // Nothing was read since `before`.
  const auto readNothing = [&](const ExpertStoreStats &before) {
    const ExpertStoreStats now = store.stats();
    for (size_t use = 0; use < 2; ++use)
      if (now.uses[use].read != before.uses[use].read || now.uses[use].readBytes != before.uses[use].readBytes)
        return false;
    return true;
  };
  std::vector<MoePlan> plans;
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) plans.push_back(MoE::decodePlan(shape, lanes, MoeConfig{}));
  for (const uint32_t rows : {9u, 64u, kMaximumRows}) plans.push_back(prefillPlan(shape, rows));
  Buffers b;
  const auto run = [&](const std::string &when) {
    for (const MoePlan &plan : plans)
      for (uint32_t block = 0; block < 2; ++block) {
        freshRows(backend, b, shape);
        const std::vector<uint16_t> want = runWeights(backend, m.weights, b, plan);
        if (runWeights(backend, streamed[block], b, plan) != want ||
            !slotsHoldTheirExperts(m, streamed[block], b, plan))
          fail(when + ", block " + std::to_string(block) + ", " + std::to_string(plan.rows()) +
               " rows: output or slots differ from the resident experts'");
      }
    if (!tablesBelow(store.stats().slots)) fail(when + ": a load claimed a slot past the live chunks");
  };
  // A chunk goes at once after block 0's `plan` on fresh rows, whose
  // command ended with a read-ahead of block 1's experts started, and block
  // 1's same plan on the same rows takes over what it read (the plan's
  // use's prefetched); where `evicts`, the store is full, so that the
  // compaction evicts experts as well as moving them. Block 0's plan runs
  // twice, and its second load finds every expert it routes resident and
  // stages none: a decode step's read-ahead takes only the places its round
  // left free, and a first load of four lanes at 16 experts stages an expert
  // in every place. A long chunk's read-ahead takes the whole staging for
  // block 1's experts that are not resident: its rows route more than fifteen
  // sixteenths of a block's experts, which block 0 then holds and block 1
  // routes alike (the same rows and router), in fewer slots than twice those,
  // so block 1 lacks some it routes.
  const auto shrinkUnderReadAhead = [&](const MoePlan &plan, bool evicts, const std::string &when) {
    const size_t use = static_cast<size_t>(plan.phase());
    freshRows(backend, b, shape);
    const std::vector<uint16_t> want = runWeights(backend, m.weights, b, plan);
    for (uint32_t step = 1; step <= 2; ++step)
      if (runWeights(backend, streamed[0], b, plan) != want)
        fail(when + ", block 0's step " + std::to_string(step) + ": output differs");
    const ExpertStoreStats before = store.stats();
    if (!store.releaseAboveFloor()) fail(when + ": no chunk went");
    const ExpertStoreStats shrunk = store.stats();
    if (evicts && before.resident <= shrunk.slots)
      fail(when + ": the store held " + std::to_string(before.resident) + " experts, which " +
           std::to_string(shrunk.slots) + " slots hold without an eviction");
    if (runWeights(backend, streamed[1], b, plan) != want || !slotsHoldTheirExperts(m, streamed[1], b, plan))
      fail(when + ", block 1: output or slots differ from the resident experts'");
    const ExpertStoreStats after = store.stats();
    if (!tablesBelow(after.slots)) fail(when + ": a slot table names a slot past the live chunks");
    if (after.uses[use].prefetched <= before.uses[use].prefetched)
      fail(when + ": block 1's load took over nothing read ahead");
  };
  const MoePlan longChunk = prefillPlan(shape, kMaximumRows);
  requireLists(capChunks, "at the cap");
  run("at the cap");
  shrinkUnderReadAhead(longChunk, true, "a shrink at the cap under a long chunk's read-ahead");
  // Down to the floor, a chunk at a time.
  uint32_t live = capChunks - 1;
  requireLists(live, "after a shrink under a read-ahead");
  stats = store.stats();
  for (uint64_t bytes; (bytes = store.releaseAboveFloor()) != 0;) {
    const ExpertStoreStats before = stats;
    stats = store.stats();
    const std::string when = "after a shrink to " + std::to_string(--live);
    const uint64_t liveSlots = slotsOf(live);
    if (bytes != arenaBytes(m, slotsOf(live + 1) - liveSlots) || stats.chunks != live || stats.slots != liveSlots ||
        stats.arenaBytes != before.arenaBytes - bytes || stats.shrinks != before.shrinks + 1 ||
        stats.resident != std::min(before.resident, liveSlots))
      fail(when + ": gave back " + std::to_string(bytes) + " bytes and holds " + std::to_string(stats.resident) +
           " experts in " + std::to_string(stats.slots) + " slots of " + std::to_string(stats.chunks) +
           " chunks, not its last chunk's bytes and as many of its " + std::to_string(before.resident) +
           " experts as " + std::to_string(liveSlots) + " slots hold");
    if (!readNothing(before)) fail(when + ": the shrink read experts");
    if (!tablesBelow(liveSlots)) fail(when + ": a slot table names a slot past the live chunks");
    requireLists(live, when);
    run(when);
    stats = store.stats();
  }
  if (live != 1 || stats.slots != floorSlots || stats.arenaBytes + fixed != stats.floorBytes ||
      store.releaseAboveFloor() || store.growthBytes() != arenaBytes(m, slotsOf(2) - floorSlots))
    fail("at the floor: " + std::to_string(live) + " chunks of " + std::to_string(stats.slots) +
         " slots, or it gives more back or would take another chunk than the next");
  // An idle release after the shrinks, which counts nothing live, and a
  // restore at the cap.
  store.release();
  stats = store.stats();
  if (!store.released() || stats.chunks || stats.arenaBytes || stats.slots || stats.resident)
    fail("after a release: " + std::to_string(stats.chunks) + " chunks of " + std::to_string(stats.slots) +
         " slots counted live");
  require(store.restore(splash::test::admitAll, splash::test::admitAll), "the shrunk store did not restore");
  stats = store.stats();
  if (stats.chunks != capChunks || stats.slots != slots || stats.arenaBytes + fixed != stats.capBytes)
    fail("after a restore: " + std::to_string(stats.chunks) + " chunks of " + std::to_string(stats.slots) +
         " slots, not the cap's");
  requireLists(capChunks, "after a restore");
  // Block 1 holds no expert after the restore, so block 0's decode step
  // reads ahead the experts its prediction names; then, the store full, a
  // long chunk's.
  shrinkUnderReadAhead(MoE::decodePlan(shape, 4, MoeConfig{}), false,
                       "a shrink after a restore under a decode step's read-ahead");
  run("after a restore");
  shrinkUnderReadAhead(longChunk, true, "a shrink after a restore under a long chunk's read-ahead");
  // A refused regrow leaves the next chunk released, granted ones take them
  // back with their slots free.
  live = capChunks - 2;
  const auto refuse = [](uint64_t, const std::function<void()> &) -> AllocationResult {
    return AllocationFailure::EngineBudget;
  };
  stats = store.stats();
  if (store.growTowardCap(refuse).failure != AllocationFailure::EngineBudget || store.stats().chunks != live ||
      store.stats().regrows != stats.regrows)
    fail("a refused regrow did not report its refusal, or took its chunk");
  requireLists(live, "after a refused regrow");
  run("after a refused regrow");
  for (uint64_t bytes; live < capChunks && (bytes = store.growthBytes()) != 0;) {
    const ExpertStoreStats before = store.stats();
    const std::string when = "after a regrow to " + std::to_string(++live);
    if (!store.growTowardCap(splash::test::admitAll)) fail(when + ": refused under an admission of all");
    stats = store.stats();
    if (bytes != arenaBytes(m, slotsOf(live) - slotsOf(live - 1)) || stats.chunks != live ||
        stats.slots != slotsOf(live) || stats.arenaBytes != before.arenaBytes + bytes ||
        stats.regrows != before.regrows + 1 || stats.resident != before.resident || !readNothing(before))
      fail(when + ": took " + std::to_string(bytes) + " bytes into " + std::to_string(stats.chunks) +
           " chunks of " + std::to_string(stats.slots) + " slots, not the next chunk's free slots alone");
    requireLists(live, when);
    run(when);
  }
  stats = store.stats();
  if (live != capChunks || store.growthBytes() || !store.growTowardCap(splash::test::admitAll) ||
      store.stats().regrows != stats.regrows)
    fail("at the cap: " + std::to_string(live) + " chunks, or a regrow took one more");
  // A restore whose optional admission refuses the second elastic chunk
  // keeps the floor and the first, and regrows take the rest.
  store.release();
  uint32_t admissions = 0;
  const auto firstElastic = [&](uint64_t, const std::function<void()> &allocate) -> AllocationResult {
    if (admissions++ >= 1) return AllocationFailure::HostPressure;
    allocate();
    return {};
  };
  require(store.restore(splash::test::admitAll, firstElastic), "a restore refused past an elastic chunk did not end");
  stats = store.stats();
  if (admissions != 2 || stats.chunks != 2 || stats.slots != slotsOf(2))
    fail("after a restore refused past an elastic chunk: " + std::to_string(stats.chunks) + " chunks after " +
         std::to_string(admissions) + " optional admissions, not the floor and one elastic chunk after two");
  requireLists(2, "after a restore refused past an elastic chunk");
  run("after a restore refused past an elastic chunk");
  for (live = 2; live < capChunks && store.growthBytes(); ++live)
    if (!store.growTowardCap(splash::test::admitAll)) fail("a regrow after a partial restore was refused");
  if (live != capChunks || store.growthBytes() || store.stats().chunks != capChunks)
    fail("regrows after a partial restore took " + std::to_string(live) + " chunks, not " +
         std::to_string(capChunks));
  requireLists(capChunks, "after regrows from a partial restore");
  run("after regrows from a partial restore");
  // Critical pressure gives every chunk above the floor back at once.
  stats = store.stats();
  const uint64_t toFloor = store.releaseToFloor();
  const ExpertStoreStats atFloor = store.stats();
  if (toFloor != arenaBytes(m, slots) - arenaBytes(m, floorSlots) || atFloor.chunks != 1 ||
      atFloor.slots != floorSlots || atFloor.shrinks != stats.shrinks + capChunks - 1 ||
      atFloor.resident != std::min(stats.resident, floorSlots) || store.releaseToFloor() ||
      store.releaseAboveFloor() || !readNothing(stats))
    fail("a release to the floor gave back " + std::to_string(toFloor) + " bytes into " +
         std::to_string(atFloor.chunks) + " chunks of " + std::to_string(atFloor.slots) +
         " slots, not every chunk above the floor at once");
  requireLists(1, "after a release to the floor");
  run("after a release to the floor");
  printf("streamed experts in %u chunks, a floor of %llu slots and %llu-slot elastic chunks: given back down to "
         "the floor and taken back, through a release, a restore and refused regrows and restores, bitwise the "
         "resident experts' %s\n",
         capChunks, (unsigned long long)floorSlots, (unsigned long long)chunkSlots, failures ? "FAIL" : "ok");
  return failures;
}

// Blocks of three formats give back and take again the chunks of the slots
// they share, the widest expert's: two blocks of Q8_0, two of Q5_K and one of
// IQ2_S, in memory for every block's experts but three, under a floor of two
// blocks' experts and a quarter more and a largest buffer of a block's
// slots, which holds less than the floor, so that the floor lies in the
// floor chunk and the elastic chunks after it (ExpertChunkLayout::
// floorChunks, at least two). The store lays the slots out as
// streamedExpertChunkLayout does a slot of one plane of a slot's bytes. Each
// shrink gives back the last live chunk, its bytes, down to the floor chunks,
// past which no slot table names a slot, the floor's bytes those chunks'; each
// regrow takes the next chunk back; every block's views name the live chunks
// alone. After a release, a restore whose optional admission refuses the
// first elastic chunk brings the floor back in one admission, and regrows
// bring the cap back. Every block's decode steps and prefill chunks give the
// resident experts' bits at every step, their slots holding their planes
// wherever the compactions moved them.
int streamedElasticLayouts(MetalBackend &backend, const MoeShape &shape, const std::filesystem::path &directory) {
  using splash::model::ExpertChunkLayout;
  const Model big = makeModel(backend, shape, Q80, true), small = makeModel(backend, shape, Q5K, true),
              narrow = makeModel(backend, shape, IQ2S, true);
  ExpertFile bigFile(big, directory / "big.bin"), smallFile(small, directory / "small.bin"),
      narrowFile(narrow, directory / "narrow.bin");
  StreamedBlocks blocks = bigFile.blocks;
  blocks.blocks.insert(blocks.blocks.end(), smallFile.blocks.blocks.begin(), smallFile.blocks.blocks.end());
  blocks.blocks.push_back(narrowFile.blocks.blocks.front());
  blocks.checks.push_back(smallFile.blocks.checks.front());
  blocks.checks.push_back(narrowFile.blocks.checks.front());
  const Model *models[] = {&big, &big, &small, &small, &narrow};
  const uint64_t width = std::max({slotBytes(big), slotBytes(small), slotBytes(narrow)});
  const uint64_t cap = 5 * shape.experts - 3, floor = 2 * shape.experts + shape.experts / 4;
  const uint64_t fixed = ExpertStore::fixedBytes(blocks, shape);
  ExpertStoreOptions options = storeOptions(cap * width + kSpareBytes, blocks, shape);
  options.floorBytes = fixed + floor * width;
  options.bufferBytes = shape.experts * width;
  options.elasticChunkBytes = options.bufferBytes / 2;
  ExpertStore store(backend, blocks, shape, options);
  std::array<BlockMoeWeights, 5> streamed{big.weights, big.weights, small.weights, small.weights, narrow.weights};
  for (uint32_t block = 0; block < 5; ++block) store.bind(block, streamed[block]);
  int failures = 0;
  const auto fail = [&](const std::string &what) {
    printf("  elastic layouts %s FAIL\n", what.c_str());
    ++failures;
  };
  // The layout as the store lays it out.
  splash::model::ExpertSlotPlanes planes{};
  planes[0][0] = width;
  const ExpertChunkLayout layout =
      splash::model::streamedExpertChunkLayout(planes, static_cast<uint32_t>(cap), static_cast<uint32_t>(floor),
                                               shape.experts, options.bufferBytes, options.elasticChunkBytes);
  for (uint32_t block = 0; block < 5; ++block) {
    const splash::ops::ExpertChunks &chunks = streamed[block].chunks;
    if (streamed[block].slots != cap || chunks.chunks() != layout.chunks || chunks.floorSlots() != layout.floorSlots ||
        chunks.chunkSlots() != layout.chunkSlots || chunks.slotBytes() != width || layout.floorChunks < 2 ||
        layout.chunks <= layout.floorChunks)
      fail("block " + std::to_string(block) + ": " + std::to_string(chunks.chunks()) + " chunks of a floor chunk of " +
           std::to_string(chunks.floorSlots()) + " slots and elastic chunks of " +
           std::to_string(chunks.chunkSlots()) + ", not the layout's " + std::to_string(layout.chunks) +
           " chunks with its floor in " + std::to_string(layout.floorChunks) + ", at least two, and more above it");
  }
  // The blocks of a format share their views, those of other formats views
  // of the same chunks.
  if (streamed[0].chunks.planes() != streamed[1].chunks.planes() ||
      streamed[2].chunks.planes() != streamed[3].chunks.planes() ||
      streamed[0].chunks.planes() == streamed[2].chunks.planes() ||
      streamed[0].chunks.planes()->front().contents() != streamed[4].chunks.planes()->front().contents())
    fail("the blocks' views are not their format's, of the same chunks");
  if (failures) return failures;
  const auto liveOf = [&](uint32_t block) { return streamed[block].chunks.liveChunks(); };
  std::vector<MoePlan> plans{MoE::decodePlan(shape, 2, MoeConfig{}), prefillPlan(shape, 64),
                             prefillPlan(shape, kMaximumRows)};
  Buffers b;
  // Every block's plans, and its address and slot tables naming the live
  // chunks alone.
  const auto run = [&](const std::string &when) {
    for (const MoePlan &plan : plans)
      for (uint32_t block = 0; block < 5; ++block) {
        freshRows(backend, b, shape);
        const std::vector<uint16_t> want = runWeights(backend, models[block]->weights, b, plan);
        if (runWeights(backend, streamed[block], b, plan) != want ||
            !slotsHoldTheirExperts(*models[block], streamed[block], b, plan))
          fail(when + ", block " + std::to_string(block) + ", " + std::to_string(plan.rows()) +
               " rows: output or slots differ from the resident experts'");
      }
    for (uint32_t block = 0; block < 5; ++block) {
      const splash::ops::ExpertChunks &chunks = streamed[block].chunks;
      const uint64_t live = layout.first(chunks.liveChunks());
      const auto *table = static_cast<const uint32_t *>(streamed[block].slotTable.contents());
      for (uint32_t expert = 0; expert < shape.experts; ++expert)
        if (table[expert] >= live) {
          fail(when + ", block " + std::to_string(block) + ": its slot table names a slot past the live chunks");
          break;
        }
      if (chunks.liveChunks() != liveOf(0) ||
          chunks.planes()->size() != uint64_t{chunks.liveChunks()} * MOE_EXPERT_CHUNK_ADDRESSES)
        fail(when + ", block " + std::to_string(block) + ": its address table names other than the live chunks");
    }
  };
  run("at the cap");
  // Down to the floor, a chunk at a time.
  for (int step = 0;; ++step) {
    const uint32_t before = liveOf(0);
    const uint64_t bytes = store.releaseAboveFloor();
    const std::string when = "after shrink " + std::to_string(step);
    if (before == layout.floorChunks) {
      if (bytes) fail(when + ": the store at its floor gave a chunk back");
      break;
    }
    if (liveOf(0) != before - 1 || bytes != layout.held(before - 1) * width) {
      fail(when + ": gave back " + std::to_string(bytes) + " bytes, not its last live chunk's");
      if (!bytes) break;
    }
    run(when);
  }
  const ExpertStoreStats stats = store.stats();
  if (liveOf(0) != layout.floorChunks || layout.first(layout.floorChunks) < floor ||
      layout.first(layout.floorChunks - 1) >= floor ||
      stats.floorBytes != fixed + uint64_t{layout.first(layout.floorChunks)} * width || store.releaseAboveFloor() ||
      store.growthBytes() != layout.held(liveOf(0)) * width)
    fail("at the floor: " + std::to_string(liveOf(0)) + " chunks and " + std::to_string(stats.floorBytes) +
         " bytes, not the " + std::to_string(layout.floorChunks) + " chunks its floor of " + std::to_string(floor) +
         " slots takes, or it gives more back or would take another chunk than the next");
  // Back to the cap, a chunk at a time.
  while (store.growthBytes()) {
    const uint32_t before = liveOf(0);
    if (!store.growTowardCap(splash::test::admitAll) || liveOf(0) != before + 1) {
      fail("a regrow took another chunk than the next");
      break;
    }
  }
  run("after the regrows");
  if (liveOf(0) != layout.chunks) fail("after the regrows: the store is not at its cap");
  // A restore whose optional admission refuses the first elastic chunk.
  store.release();
  uint32_t needed = 0, optional = 0;
  const auto admit = [&](uint64_t, const std::function<void()> &allocate) -> AllocationResult {
    ++needed;
    allocate();
    return {};
  };
  const auto refuse = [&](uint64_t, const std::function<void()> &) -> AllocationResult {
    ++optional;
    return AllocationFailure::HostPressure;
  };
  require(store.restore(admit, refuse), "a restore refused at its first elastic chunk did not end");
  if (needed != 1 || optional != 1 || liveOf(0) != layout.floorChunks)
    fail("a restore refused at its first elastic chunk took " + std::to_string(needed) + " and " +
         std::to_string(optional) + " admissions, not one for the floor and one refused");
  run("after a restore at the floor");
  while (store.growthBytes())
    if (!store.growTowardCap(splash::test::admitAll)) {
      fail("a regrow after a restore at the floor was refused");
      break;
    }
  if (liveOf(0) != layout.chunks) fail("regrows after a restore at the floor did not bring the cap back");
  run("after regrows from the floor");
  printf("streamed experts of three formats in %u chunks of shared slots, their floor in %u: given back and taken "
         "back, through a restore at the floor, bitwise the resident experts' %s\n",
         layout.chunks, layout.floorChunks, failures ? "FAIL" : "ok");
  return failures;
}

// A load whose host steps do not run (one before them threw, as a streamed
// table's stage that fails does) still dispatches its repack, which then
// has no job: a shrink forgot the last load's, whose slots it gave back. Two
// blocks in memory for both blocks' experts but three, with a floor of a
// block's experts and elastic chunks of an eighth of the slots' bytes: block
// 1's long chunk loads into the floor chunk, then block 0's into the elastic
// chunks above it, its last round staging into the last; every chunk above
// the floor goes; then a command whose first host step throws before block
// 0's decode step's load fails with that step's error alone, and leaves the
// floor chunk's bytes and the slot tables as they were (run under
// MTL_SHADER_VALIDATION=1, a repack of the slots given back would also be an
// access out of bounds).
int streamedSkippedLoadAfterShrink(const std::string &metallibPath, const MoeShape &shape,
                                   const std::filesystem::path &directory) {
  MetalBackend backend(metallibPath);
  const Model m = makeModel(backend, shape, Q4K, true);
  ExpertFile file(m, directory / "experts.bin");
  const uint64_t slots = 2 * shape.experts - 3;
  ExpertStoreOptions options = storeOptions(arenaBytes(m, slots) + kSpareBytes, file.blocks, shape);
  options.floorBytes = ExpertStore::fixedBytes(file.blocks, shape) + arenaBytes(m, shape.experts);
  options.elasticChunkBytes = arenaBytes(m, slots / 8);
  ExpertStore store(backend, file.blocks, shape, options);
  std::array<BlockMoeWeights, 2> streamed{m.weights, m.weights};
  for (uint32_t block = 0; block < 2; ++block) store.bind(block, streamed[block]);
  Buffers b;
  freshRows(backend, b, shape);
  const MoePlan chunk = prefillPlan(shape, kMaximumRows), decode = MoE::decodePlan(shape, 1, MoeConfig{});
  static_cast<void>(runWeights(backend, streamed[1], b, chunk));
  static_cast<void>(runWeights(backend, streamed[0], b, chunk));
  if (!store.releaseToFloor() || streamed[0].chunks.liveChunks() != 1) {
    printf("  a skipped load after a shrink: the store gave nothing back FAIL\n");
    return 1;
  }
  // The floor chunk's regions and the slot tables, as they are.
  const auto contents = [](const MetalBuffer &buffer) {
    const auto *bytes = static_cast<const uint8_t *>(buffer.contents());
    return std::vector<uint8_t>(bytes, bytes + buffer.sizeBytes());
  };
  std::vector<std::vector<uint8_t>> regions;
  for (const MetalBuffer &region : *streamed[0].chunks.planes()) regions.push_back(contents(region));
  const std::vector<uint8_t> tables[] = {contents(streamed[0].slotTable), contents(streamed[1].slotTable)};
  allocate(backend, b, decode);
  CommandGraph graph;
  graph.host([] { throw std::runtime_error("test step before a load"); });
  MoE::add(graph, b.moe, streamed[0], decode);
  std::string error;
  try {
    static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
  } catch (const MetalBackendError &caught) {
    error = caught.what();
  }
  bool unchanged = contents(streamed[0].slotTable) == tables[0] && contents(streamed[1].slotTable) == tables[1];
  for (size_t region = 0; region < regions.size(); ++region)
    unchanged = unchanged && contents((*streamed[0].chunks.planes())[region]) == regions[region];
  if (error.find("test step before a load") == std::string::npos || !unchanged) {
    printf("  a load whose host steps did not run after a shrink failed its command with \"%s\", or a repack "
           "wrote the floor chunk or a slot table changed FAIL\n",
           error.c_str());
    return 1;
  }
  printf("streamed experts: a load whose host steps do not run after a shrink repacks nothing ok\n");
  return 0;
}

// Slots that would take more than MOE_EXPERT_MAX_CHUNKS chunks of the largest
// buffer, which the expert kernels' address table holds at most, are
// refused: two blocks in memory for a block's experts and three more, under a
// largest buffer of a sixtieth of their bytes.
int streamedChunkLimit(MetalBackend &backend, const MoeShape &shape, const std::filesystem::path &directory) {
  const Model m = makeModel(backend, shape, Q4K, true);
  ExpertFile file(m, directory / "experts.bin");
  const uint64_t slots = shape.experts + 3, arena = arenaBytes(m, slots);
  ExpertStoreOptions options = storeOptions(arena + kSpareBytes, file.blocks, shape);
  options.bufferBytes = arena / 60;
  const std::string expected = "takes more than " + std::to_string(MOE_EXPERT_MAX_CHUNKS);
  std::string refusal;
  try {
    ExpertStore refused(backend, file.blocks, shape, options);
  } catch (const std::invalid_argument &error) {
    refusal = error.what();
  }
  if (refusal.find(expected) == std::string::npos) {
    printf("streamed experts: a size of more than %u chunks of the largest buffer was not refused (%s) FAIL\n",
           MOE_EXPERT_MAX_CHUNKS, refusal.c_str());
    return 1;
  }
  printf("streamed experts: a size of more than %u chunks of the largest buffer is refused ok\n",
         MOE_EXPERT_MAX_CHUNKS);
  return 0;
}

// A source written since it was opened fails the load that reads it, which
// fails its command, and every later load; the store reports why.
int streamedSourceChanges(const std::string &metallibPath, const MoeShape &shape,
                          const std::filesystem::path &directory) {
  MetalBackend backend(metallibPath);
  const Model m = makeModel(backend, shape, Q4K, true);
  ExpertFile file(m, directory / "experts.bin");
  const uint64_t slots = shape.experts;
  ExpertStore store(backend, file.blocks, shape, storeOptions(arenaBytes(m, slots), file.blocks, shape));
  std::array<BlockMoeWeights, 2> blocks{m.weights, m.weights};
  for (uint32_t block = 0; block < 2; ++block) store.bind(block, blocks[block]);
  const BlockMoeWeights &streamed = blocks[0];
  Buffers b;
  freshRows(backend, b, shape);
  // A one-row chunk loads top-k of the block's experts, which leaves the long
  // chunk below experts to read; a decode step's rows could route them all.
  static_cast<void>(runWeights(backend, streamed, b, prefillPlan(shape, 1)));
  {
    std::ofstream append(file.path, std::ios::binary | std::ios::app);
    append.put(0);
  }
  std::string error;
  try {
    // The chunk's rows route experts that are not resident.
    static_cast<void>(runWeights(backend, streamed, b, prefillPlan(shape, kMaximumRows)));
  } catch (const MetalBackendError &caught) {
    error = caught.what();
  }
  const std::string failure = store.stats().failure;
  if (error.find("host step failed") == std::string::npos ||
      error.find("written while the model is loaded") == std::string::npos ||
      failure.find("written while the model is loaded") == std::string::npos) {
    printf("streamed experts: a load of a written source did not fail its command (%s), or the store reports "
           "\"%s\" FAIL\n",
           error.c_str(), failure.c_str());
    return 1;
  }
  printf("streamed experts: a load of a written source fails its command, and the store reports it ok\n");
  return 0;
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: expert_store_test <metallib>\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      const splash::test::TemporaryDirectory directory("expert-store");
      int failures = 0;
      for (const MoeShape &shape : kShapes) {
        printf("%u experts, top %u, hidden %u, %u wide:\n", shape.experts, shape.expertsPerToken, shape.hiddenSize,
               shape.expertIntermediateSize);
        failures += streamedExperts(backend, shape, directory.path()) +
                    streamedChunkOrder(backend, shape, directory.path()) +
                    streamedUnionLoads(backend, shape, directory.path()) +
                    streamedMixedLayouts(backend, shape, directory.path()) +
                    streamedRounds(backend, shape, directory.path()) +
                    streamedReadFailures(argv[1], shape, directory.path()) +
                    streamedFaultBeforeRound(argv[1], shape, directory.path()) +
                    streamedRefusal(backend, shape, directory.path()) +
                    streamedChunks(backend, shape, directory.path()) +
                    streamedElastic(backend, shape, directory.path()) +
                    streamedElasticLayouts(backend, shape, directory.path()) +
                    streamedSkippedLoadAfterShrink(argv[1], shape, directory.path()) +
                    streamedChunkLimit(backend, shape, directory.path()) +
                    streamedSourceChanges(argv[1], shape, directory.path());
      }
      if (failures) {
        std::cerr << "expert_store_test: " << failures << " failures\n";
        return 1;
      }
      std::cout << "expert_store_test: PASS\n";
    } catch (const std::exception &error) {
      std::cerr << "FAIL: " << error.what() << '\n';
      return 1;
    }
  }
  return 0;
}
