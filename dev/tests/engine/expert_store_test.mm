// Streamed experts (model::ExpertStore) at three MoE shapes in turn, a small
// one, the 35B's routing and Qwen3.8-Flash-Next's: routed experts read from a
// file into an arena of one block's experts and three more, which two blocks
// share, run decode steps and prefill chunks bitwise as the resident experts
// do, their slots holding the reference repack's planes, in as many rounds as
// the staging needs, and read ahead the next round's experts or the next
// block's a long chunk routes or a decode step predicts; a decode step's
// experts outlive long chunks of other blocks; the union of a block's steps
// loads at once; a size of one block holds every expert, through a release
// and a restore; long chunks load in rounds through half the staging, also
// across a release while the next block's experts are read ahead; a read
// ahead that fails fails no load that does not route its expert, a read of a
// routed expert that fails fails the load, and a source written since it was
// opened fails the load's command, which the store reports; memory one
// byte short of a block's slots is refused; and a size whose slots would take
// more than the largest buffer keeps its blocks in arenas that fit.
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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

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
// 35B's 256 experts, top 8, at hidden 256 and 512 wide; and
// Qwen3.8-Flash-Next's 512 experts, top 10, hidden 256, 640 wide.
constexpr MoeShape kShapes[] = {{1024, 16, 4, 512}, {256, 256, 8, 512}, {256, 512, 10, 640}};

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

// The arena bytes of `slots` experts of the model's formats, each plane in a
// region of its own (model/ExpertStore.cpp).
uint64_t arenaBytes(const Model &m, uint64_t slots) {
  uint64_t bytes = 0;
  for (size_t p = 0; p < 3; ++p) {
    const splash::model::GgufPlaneBytes planes = splash::model::ggufPlaneBytes(
        kQuantFormats[m.formats[p]], m.routed[p].rows / m.shape.experts, m.routed[p].columns);
    for (const uint64_t plane : {planes.plane0, planes.plane1, planes.meta})
      bytes += splash::model::alignWeightOffset(slots * plane);
  }
  return bytes;
}

// The readers every store of the test reads on.
const std::shared_ptr<splash::model::WeightReaders> &readers() {
  static const auto shared = std::make_shared<splash::model::WeightReaders>(splash::model::kWeightReaders);
  return shared;
}

// The options of a store of `blocks` whose arenas hold `arena` bytes of slots
// (ExpertStoreOptions::memoryBytes: the slots, its staging and tables).
ExpertStoreOptions storeOptions(uint64_t arena, const StreamedBlocks &blocks, const MoeShape &shape) {
  return {.memoryBytes = arena + ExpertStore::fixedBytes(blocks, shape), .readers = readers()};
}

// The host steps of a load of `plan`'s rows, a round each: the staging
// places the experts of a block a decode cycle's rows may route, and at least
// two; a decode step's load that they hold stages them all in one round, any
// other load half of them a round. A long chunk's load whose next block
// streams (`nextStreams`) reads it ahead in one step more.
uint32_t loadSteps(const MoeShape &shape, const MoePlan &plan, bool nextStreams) {
  using splash::model::ExecutionLimits;
  const uint64_t places = std::max<uint64_t>(
      2, std::min<uint64_t>(shape.experts, uint64_t{ExecutionLimits::maximumBatchWidth} *
                                               ExecutionLimits::targetVerifyRows * shape.expertsPerToken));
  const uint64_t routable = std::min<uint64_t>(shape.experts, uint64_t{plan.rows()} * shape.expertsPerToken);
  const uint64_t perRound = plan.phase() == MoePhase::Decode && routable <= places ? places : places / 2;
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

// Every expert the last dispatch routed holds its slot's planes as the
// reference repacked them.
bool slotsHoldTheirExperts(const Model &m, const BlockMoeWeights &streamed, const Buffers &b, const MoePlan &plan) {
  const auto *selected = static_cast<const uint32_t *>(b.moe.scratch.selectedExperts.contents());
  const auto *table = static_cast<const uint32_t *>(streamed.slotTable.contents());
  const BlockExpertProjection *projections[] = {&streamed.gate, &streamed.up, &streamed.down};
  const BlockExpertProjection *resident[] = {&m.weights.gate, &m.weights.up, &m.weights.down};
  for (uint64_t route = 0; route < uint64_t{plan.rows()} * m.shape.routesPerToken(); ++route) {
    const uint32_t expert = selected[route];
    if (expert >= m.shape.experts) continue;
    for (size_t p = 0; p < 3; ++p) {
      const QuantizedSegment &slots = projections[p]->routed, &experts = resident[p]->routed;
      const uint64_t rows = experts.outputSize / m.shape.experts;
      const splash::model::GgufPlaneBytes planes =
          splash::model::ggufPlaneBytes(kQuantFormats[experts.formatId], rows, experts.inputSize);
      const std::pair<const MetalBuffer *, const MetalBuffer *> pairs[] = {
          {&slots.plane0, &experts.plane0}, {&slots.plane1, &experts.plane1}, {&slots.meta, &experts.meta}};
      const uint64_t bytes[] = {planes.plane0, planes.plane1, planes.meta};
      for (size_t plane = 0; plane < 3; ++plane) {
        if (!bytes[plane]) continue;
        const auto *got = static_cast<const uint8_t *>(pairs[plane].first->contents()) + table[expert] * bytes[plane];
        const auto *want = static_cast<const uint8_t *>(pairs[plane].second->contents()) + expert * bytes[plane];
        if (std::memcmp(got, want, bytes[plane])) return false;
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
    require(store.stats().slots == slots, formats + ": the arena holds another number of slots");
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
// routed: three blocks share an arena of one block and the decode step's
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

// A size of one block in a format of its own (a layer a dynamic quantization
// gave other formats) holds every one of its experts: the store reads them
// when it is built and again when it is restored, and that block's steps
// load nothing (no host step, no read), while each load of the streamed size
// runs its rounds. Each step gives the resident experts' bits.
int streamedLoneBlockSize(MetalBackend &backend, const MoeShape &shape, const std::filesystem::path &directory) {
  const Model trunk = makeModel(backend, shape, Q4K, true), head = makeModel(backend, shape, Q80, true);
  ExpertFile trunkFile(trunk, directory / "trunk.bin"), headFile(head, directory / "head.bin");
  StreamedBlocks blocks = trunkFile.blocks;
  blocks.blocks.push_back(headFile.blocks.blocks.front());
  blocks.checks.push_back(headFile.blocks.checks.front());
  const uint64_t trunkSlots = shape.experts + 3;
  ExpertStore store(backend, blocks, shape,
                    storeOptions(arenaBytes(trunk, trunkSlots) + arenaBytes(head, shape.experts) + kSpareBytes,
                                 blocks, shape));
  require(store.stats().slots == trunkSlots + shape.experts && store.stats().resident == shape.experts,
          "a size of one block holds another number of slots, or not every expert of its block");
  std::array<BlockMoeWeights, 3> streamed{trunk.weights, trunk.weights, head.weights};
  for (uint32_t block = 0; block < 3; ++block) store.bind(block, streamed[block]);
  require(streamed[0].stream && streamed[1].stream && !streamed[2].stream,
          "a block of a size of one block streams, or a streamed block does not");
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
      require(store.restore(splash::test::admitAll) && store.stats().resident == shape.experts,
              "a restored store does not hold every expert of a size of one block");
    }
    const Model &m = step.block == 2 ? head : trunk;
    Buffers b;
    freshRows(backend, b, shape);
    const std::vector<uint16_t> want = runWeights(backend, m.weights, b, *step.plan);
    const ExpertStoreStats before = store.stats();
    const std::vector<uint16_t> got = runWeights(backend, streamed[step.block], b, *step.plan);
    const ExpertStoreStats after = store.stats();
    uint64_t hostSteps = 0, read = 0;
    for (size_t use = 0; use < 2; ++use) {
      hostSteps += after.uses[use].pickup.count - before.uses[use].pickup.count;
      read += after.uses[use].read - before.uses[use].read;
    }
    if (got != want || !slotsHoldTheirExperts(m, streamed[step.block], b, *step.plan) ||
        hostSteps != (step.block == 2 ? 0 : loadSteps(shape, *step.plan, step.block == 0)) ||
        (step.block == 2 && read)) {
      printf("  block %u%s: %llu host steps read %llu experts, or its output or slots differ from the resident "
             "experts' FAIL\n",
             step.block, step.restores ? " after a restore" : "", (unsigned long long)hostSteps,
             (unsigned long long)read);
      ++failures;
    }
  }
  printf("streamed experts beside a size of one block, which holds every expert and loads none, before and after "
         "a restore: bitwise the resident experts' %s\n",
         failures ? "FAIL" : "ok");
  return failures;
}

// Long prefill chunks load in rounds through half the staging each, the
// readers reading the next round's experts ahead into the other half while
// the GPU repacks, and after the last round the next block's: an arena of
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
      require(store.restore(splash::test::admitAll), "the store was not restored");
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
// reports no failure. A chunk that routes the last expert fails its command
// and frees the slots it claimed; the backend then refuses every later
// command, and the store keeps its failure.
int streamedReadFailures(const std::string &metallibPath, const MoeShape &shape,
                         const std::filesystem::path &directory) {
  MetalBackend backend(metallibPath);
  const Model m = makeModel(backend, shape, Q4K, true);
  ExpertFile intact(m, directory / "intact.bin"), cut(m, directory / "cut.bin");
  StreamedBlocks blocks;
  blocks.blocks = {intact.blocks.blocks.front(), cut.blocks.blocks.front()};
  const uint64_t slots = 2 * shape.experts;
  ExpertStore store(backend, blocks, shape, storeOptions(arenaBytes(m, slots), blocks, shape));
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
  const uint64_t resident = store.stats().resident;
  rowsRouting(b, row, true);
  std::string error;
  try {
    static_cast<void>(runWeights(backend, streamed[1], b, row));
  } catch (const MetalBackendError &caught) {
    error = caught.what();
  }
  const ExpertStoreStats stats = store.stats();
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

// A streamed size whose slots would take more than the largest buffer
// (ExpertStoreOptions::bufferBytes) keeps its blocks in arenas that each fit,
// each recycling its own slots: four blocks of one layout, in memory for two
// arenas of a block's experts and three more, under a largest buffer of one,
// take two arenas of two blocks. Each block's long chunk and decode step give
// the resident experts' bits.
int streamedSplitArenas(MetalBackend &backend, const MoeShape &shape, const std::filesystem::path &directory) {
  const Model m = makeModel(backend, shape, Q4K, true);
  ExpertFile file(m, directory / "experts.bin");
  StreamedBlocks blocks = file.blocks;
  blocks.blocks.insert(blocks.blocks.end(), file.blocks.blocks.begin(), file.blocks.blocks.end());
  const uint64_t arena = arenaBytes(m, shape.experts + 3);
  ExpertStoreOptions options = storeOptions(2 * arena + kSpareBytes, blocks, shape);
  options.bufferBytes = arena + kSpareBytes;
  ExpertStore store(backend, blocks, shape, options);
  std::array<BlockMoeWeights, 4> streamed{m.weights, m.weights, m.weights, m.weights};
  for (uint32_t block = 0; block < 4; ++block) store.bind(block, streamed[block]);
  const auto arenaOf = [&](uint32_t block) { return streamed[block].gate.routed.plane0.contents(); };
  int failures = 0;
  if (arenaOf(0) != arenaOf(1) || arenaOf(2) != arenaOf(3) || arenaOf(0) == arenaOf(2) ||
      store.stats().slots < 2 * shape.experts || store.stats().arenaBytes > 2 * (arena + kSpareBytes)) {
    printf("  split arenas: blocks are not in two arenas of two, or they hold %llu slots in %llu bytes FAIL\n",
           (unsigned long long)store.stats().slots, (unsigned long long)store.stats().arenaBytes);
    ++failures;
  }
  const MoePlan chunk = prefillPlan(shape, kMaximumRows), decode = MoE::decodePlan(shape, 2, MoeConfig{});
  Buffers b;
  for (const MoePlan *plan : {&chunk, &decode})
    for (uint32_t block = 0; block < 4; ++block) {
      freshRows(backend, b, shape);
      const std::vector<uint16_t> want = runWeights(backend, m.weights, b, *plan);
      if (runWeights(backend, streamed[block], b, *plan) != want ||
          !slotsHoldTheirExperts(m, streamed[block], b, *plan)) {
        printf("  split arenas, block %u %s: output or slots differ from the resident experts' FAIL\n", block,
               plan == &chunk ? "chunk" : "decode step");
        ++failures;
      }
    }
  printf("streamed experts in arenas of at most the largest buffer, two of two blocks: bitwise the resident "
         "experts' %s\n",
         failures ? "FAIL" : "ok");
  return failures;
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
                    streamedLoneBlockSize(backend, shape, directory.path()) +
                    streamedRounds(backend, shape, directory.path()) +
                    streamedReadFailures(argv[1], shape, directory.path()) +
                    streamedRefusal(backend, shape, directory.path()) +
                    streamedSplitArenas(backend, shape, directory.path()) +
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
