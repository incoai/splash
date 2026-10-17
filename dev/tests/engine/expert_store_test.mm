// Streamed experts (model::ExpertStore) at two MoE shapes in turn, a small
// one and Qwen3.8-Flash-Next's routing: routed experts read from a file into
// an arena of one block's experts and three more, which two blocks share, run
// decode steps and prefill chunks bitwise as the resident experts do, their
// slots holding the reference repack's planes, and read ahead the next
// block's experts a long chunk routes or a decode step predicts; a decode
// step's experts outlive long chunks of other blocks; the union of a block's
// steps loads in one host step; a size of one block holds every expert,
// through a release and a restore; long chunks stage in slots they borrow and
// give back; and a source written since it was opened fails the load's
// command, which the store reports.
#include "MoeFixture.hpp"
#include "TestAdmission.hpp"
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ExpertStore.hpp"
#include "model/GgufImageLayout.hpp"
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

// The shapes the store runs at: 16 experts, top 4, hidden 1024, 512 wide, and
// Qwen3.8-Flash-Next's 512 experts, top 10, hidden 256, 640 wide.
constexpr MoeShape kShapes[] = {{1024, 16, 4, 512}, {256, 512, 10, 640}};

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
    source = std::make_unique<splash::model::WeightSource>(path);
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
    blocks.checks = {[file = source.get()] { file->checkUnchanged(); }};
  }
  ExpertFile(const ExpertFile &) = delete;
  ExpertFile &operator=(const ExpertFile &) = delete;

  std::filesystem::path path;
  std::unique_ptr<splash::model::WeightSource> source;
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

// The memory of a store of `blocks` whose arenas hold `arena` bytes of
// slots (ExpertStoreOptions::memoryBytes): the slots, its staging and tables.
uint64_t storeBytes(uint64_t arena, const StreamedBlocks &blocks, const MoeShape &shape) {
  return arena + ExpertStore::fixedBytes(blocks, shape);
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
    const uint64_t borrowed = ExpertStore::borrowedSlots(file.blocks, shape), slots = shape.experts + 3 + borrowed;
    ExpertStore store(backend, file.blocks, shape,
                      {.memoryBytes = storeBytes(arenaBytes(m, slots) + kSpareBytes, file.blocks, shape)});
    require(store.stats().slots == slots && store.stats().borrowed == borrowed,
            formats + ": the arena holds another number of slots");
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
    for (const MoePlan &plan : plans)
      for (uint32_t block = 0; block < 2; ++block, ++index) {
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
    // Every load is one host step, whose round trip its command's completion
    // reported.
    for (const ExpertStoreStats::Use &use : stats.uses)
      if (use.pickup.count != use.loads || use.work.count != use.pickup.count || use.resume.count != use.pickup.count) {
        printf("  %s: %llu loads reported %llu/%llu/%llu host-step round trips FAIL\n", formats.c_str(),
               (unsigned long long)use.loads, (unsigned long long)use.pickup.count,
               (unsigned long long)use.work.count, (unsigned long long)use.resume.count);
        ++failures;
      }
    printf("%-20s streamed decode B1-4, prefill 9/27/64/263 over two blocks: %llu of %llu routed experts read "
           "(ahead: prefill %llu, decode %llu of %llu), %.1f MiB, bitwise the resident experts' %s\n",
           formats.c_str(), (unsigned long long)read, (unsigned long long)routed,
           (unsigned long long)prefill.prefetched, (unsigned long long)decode.prefetched,
           (unsigned long long)decode.read, double(readBytes) / (1 << 20), failures > before ? "FAIL" : "ok");
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
// experts and three slots (besides those prefill loads borrow), and after a
// decode step of the last block two long chunks of the others load every
// expert of each, yet the same decode step reads nothing again (one recency
// order would evict its experts for the second chunk).
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
  const uint64_t slots = shape.experts + routed.size() + 3 + ExpertStore::borrowedSlots(blocks, shape);
  ExpertStore store(backend, blocks, shape,
                    {.memoryBytes = storeBytes(arenaBytes(m, slots) + kSpareBytes, blocks, shape)});
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
// group): each routes into its own scratch, one load, one host step, stages
// the union of their experts, and each step's routed work then gives the
// resident experts' bits. The two blocks take turns, so each union evicts
// the other block's experts; each load counts once.
int streamedUnionLoads(MetalBackend &backend, const MoeShape &shape, const std::filesystem::path &directory) {
  int failures = 0;
  for (const Fmt f : {Q4K, IQ3XXS}) {
    const Model m = makeModel(backend, shape, f, true);
    ExpertFile file(m, directory / "experts.bin");
    const uint64_t slots = shape.experts + 3 + ExpertStore::borrowedSlots(file.blocks, shape);
    ExpertStore store(backend, file.blocks, shape,
                      {.memoryBytes = storeBytes(arenaBytes(m, slots) + kSpareBytes, file.blocks, shape)});
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
      if (use.loads != used.loads + 1 || use.pickup.count != used.pickup.count + 1) {
        printf("  %s: a union load counted %llu loads in %llu host steps FAIL\n", fmtName(f),
               (unsigned long long)(use.loads - used.loads),
               (unsigned long long)(use.pickup.count - used.pickup.count));
        ++failures;
      }
    }
    printf("%-20s streamed union loads of three steps (263/27/9 rows), two blocks in turn, one host step each: "
           "bitwise the resident experts' %s\n",
           fmtName(f), failures > before ? "FAIL" : "ok");
  }
  return failures;
}

// A size of one block in a format of its own (a layer a dynamic quantization
// gave other formats) holds every one of its experts: the store reads them
// when it is built and again when it is restored, and that block's steps
// load nothing (no host step, no read), while each load of the streamed size
// is one host step. Each step gives the resident experts' bits.
int streamedLoneBlockSize(MetalBackend &backend, const MoeShape &shape, const std::filesystem::path &directory) {
  const Model trunk = makeModel(backend, shape, Q4K, true), head = makeModel(backend, shape, Q80, true);
  ExpertFile trunkFile(trunk, directory / "trunk.bin"), headFile(head, directory / "head.bin");
  StreamedBlocks blocks = trunkFile.blocks;
  blocks.blocks.push_back(headFile.blocks.blocks.front());
  blocks.checks.push_back(headFile.blocks.checks.front());
  const uint64_t trunkSlots = shape.experts + 3 + ExpertStore::borrowedSlots(blocks, shape);
  ExpertStore store(
      backend, blocks, shape,
      {.memoryBytes = storeBytes(arenaBytes(trunk, trunkSlots) + arenaBytes(head, shape.experts) + kSpareBytes, blocks,
                                 shape)});
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
        hostSteps != (step.block == 2 ? 0 : 1) || (step.block == 2 && read)) {
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

// Prefill chunks whose rows may route more experts than the staging places
// stage in slots borrowed from the end of the streamed size's arena,
// whatever the formats, and the decode step after them gives them back:
// while they are borrowed no routed expert lives there, and that decode
// step's misses take them first. Without borrowed slots (the staging places
// a whole block) every load stages in the staging. Each load is one host
// step, and each step gives the resident experts' bits.
int streamedBorrowedStaging(MetalBackend &backend, const MoeShape &shape, const std::filesystem::path &directory) {
  const Model m = makeModel(backend, shape, Q4K, true);
  ExpertFile file(m, directory / "experts.bin");
  const uint64_t borrowed = ExpertStore::borrowedSlots(file.blocks, shape), slots = 2 * shape.experts + borrowed;
  ExpertStore store(backend, file.blocks, shape,
                    {.memoryBytes = storeBytes(arenaBytes(m, slots) + kSpareBytes, file.blocks, shape)});
  require(store.stats().slots == slots && store.stats().borrowed == borrowed,
          "the streamed size holds or lends another number of slots");
  const uint64_t lent = slots - borrowed;
  std::array<BlockMoeWeights, 2> streamed{m.weights, m.weights};
  for (uint32_t block = 0; block < 2; ++block) store.bind(block, streamed[block]);
  const MoePlan decode = MoE::decodePlan(shape, 1, MoeConfig{});
  const MoePlan chunk = prefillPlan(shape, kMaximumRows);
  int failures = 0;
  const struct Step {
    uint32_t block;
    bool prefill;
    bool givesBack; // the decode step after the chunks, whose misses take the borrowed slots first
  } steps[] = {{0, false, false}, {1, false, false}, {0, true, false}, {1, true, false},
               {0, false, true},  {1, false, false}, {0, true, false}};
  for (const Step &step : steps) {
    const MoePlan &plan = step.prefill ? chunk : decode;
    Buffers b;
    freshRows(backend, b, shape);
    const std::vector<uint16_t> want = runWeights(backend, m.weights, b, plan);
    const ExpertStoreStats::Use used = store.stats().uses[step.prefill ? 0 : 1];
    const std::vector<uint16_t> got = runWeights(backend, streamed[step.block], b, plan);
    const ExpertStoreStats::Use &use = store.stats().uses[step.prefill ? 0 : 1];
    const uint64_t hostSteps = use.pickup.count - used.pickup.count, read = use.read - used.read;
    // The highest slot of an expert the step routed.
    const auto *selected = static_cast<const uint32_t *>(b.moe.scratch.selectedExperts.contents());
    const auto *table = static_cast<const uint32_t *>(streamed[step.block].slotTable.contents());
    uint32_t highest = 0;
    for (uint64_t route = 0; route < uint64_t{plan.rows()} * shape.routesPerToken(); ++route)
      if (selected[route] < shape.experts) highest = std::max(highest, table[selected[route]]);
    const bool placed = !borrowed || (step.prefill ? highest < lent : !step.givesBack || !read || highest >= lent);
    if (got != want || !slotsHoldTheirExperts(m, streamed[step.block], b, plan) || hostSteps != 1 || !placed) {
      printf("  block %u %s: %llu host steps, %llu read, highest routed slot %u of %llu lent from %llu, or its "
             "output or slots differ from the resident experts' FAIL\n",
             step.block, step.prefill ? "chunk" : "decode step", (unsigned long long)hostSteps,
             (unsigned long long)read, highest, (unsigned long long)borrowed, (unsigned long long)lent);
      ++failures;
    }
  }
  printf("streamed experts staging long chunks in %llu borrowed slots of %llu, one host step a load: bitwise the "
         "resident experts' %s\n",
         (unsigned long long)borrowed, (unsigned long long)slots, failures ? "FAIL" : "ok");
  return failures;
}

// A source written since it was opened fails the load that reads it, which
// fails its command, and every later load; the store reports why.
int streamedSourceChanges(const std::string &metallibPath, const MoeShape &shape,
                          const std::filesystem::path &directory) {
  MetalBackend backend(metallibPath);
  const Model m = makeModel(backend, shape, Q4K, true);
  ExpertFile file(m, directory / "experts.bin");
  const uint64_t slots = shape.experts + ExpertStore::borrowedSlots(file.blocks, shape);
  ExpertStore store(backend, file.blocks, shape, {.memoryBytes = storeBytes(arenaBytes(m, slots), file.blocks, shape)});
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
                    streamedBorrowedStaging(backend, shape, directory.path()) +
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
