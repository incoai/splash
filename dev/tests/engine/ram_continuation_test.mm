// Synthetic-weight numerical unit test; never loads the production model.
#include "SyntheticModels.hpp"
#include "ClaimScope.hpp"
#include "TestChecks.hpp"
#include "TestKvTier.hpp"
#include "ops/KvSlotFormat.hpp"
#include "model/Runtime.hpp"
#include "model/RuntimeInstrumentation.hpp"
#include "model/QwenState.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <chrono>
#include <thread>
#include <iostream>
#include <functional>
#include <memory>
#include <set>
#include <vector>

using namespace splash;
using namespace splash::engine;
using splash::test::require;
using splash::test::rejects;

namespace {
std::vector<uint8_t> bytes(const metal::MetalBuffer &buffer) {
  if (!buffer.sizeBytes()) return {};
  const auto *begin = static_cast<const uint8_t *>(buffer.contents());
  require(begin, "continuation test buffer is not CPU-visible");
  return {begin, begin + buffer.sizeBytes()};
}
struct State final {
  model::QwenLaneMetadata metadata;
  std::vector<uint8_t> current, next, window, pending;
  std::vector<std::vector<uint8_t>> rings;
};
State state(const model::Runtime &runtime, const model::QwenStateStorage &storage, uint32_t lane) {
  State result{storage.metadata(lane), bytes(storage.current(lane).stateBase),
               bytes(storage.next(lane).stateBase), bytes(storage.window(lane)),
               model::RuntimeInstrumentation::pendingGdnRows(runtime, storage.metadata(lane).requestId), {}};
  for (const auto &ring : storage.draft(lane)) {
    result.rings.push_back(bytes(ring.keys));
    result.rings.push_back(bytes(ring.values));
  }
  return result;
}
void equal(const State &a, const State &b) {
  require(a.metadata.requestId == b.metadata.requestId &&
              a.metadata.activeParity == b.metadata.activeParity &&
              a.metadata.lengths == b.metadata.lengths &&
              a.metadata.pendingPath == b.metadata.pendingPath &&
              a.current == b.current && a.next == b.next && a.window == b.window &&
              a.pending == b.pending && a.rings == b.rings,
          "park/rebind changed recurrent state, tape, draft context or metadata");
}
ModelStepResult step(model::Runtime &runtime, uint64_t id, uint64_t position,
                     const std::vector<uint32_t> &pages, std::span<const uint32_t> prompt = {},
                     bool probePark = false, bool probeRestore = false) {
  const bool prefill = !prompt.empty();
  const uint32_t count = static_cast<uint32_t>(prompt.size());
  BatchPlan plan{.kind = prefill ? WorkKind::Prefill : WorkKind::Decode,
                 .items = {{id, count}}, .decodeStage = DecodeStage::Regular};
  ModelBatchItem item{id, position, count, pages};
  item.pageTableRevision = 1;
  item.inputTokens = prompt;
  if (!prefill)
    require(runtime.reserveSpilledReads(id, {.kind = WorkKind::Decode}).granted(),
            "continuation decode hot floor was refused");
  auto ticket = runtime.submit(plan, {&item, 1}, {});
  if (probePark)
    require(!runtime.parkDecodedRequest(202).granted(), "a live command permitted parking");
  if (probeRestore)
    require(!runtime.restoreParkedRequest(202).granted(), "a live command permitted restoration");
  const auto results = ticket->wait();
  require(results.size() == 1, "synthetic continuation lost its result");
  require(results.front().failure.empty(), "synthetic continuation command failed for " +
          std::to_string(id) + " at " + std::to_string(position) + ": " + results.front().failure);
  return results.front();
}
std::vector<uint32_t> singletonMasks(std::span<const uint32_t> tokens, uint32_t vocabulary) {
  const uint32_t stride = (vocabulary + 31) / 32;
  std::vector<uint32_t> words(tokens.size() * stride, 0);
  for (size_t row = 0; row < tokens.size(); ++row) {
    require(tokens[row] < vocabulary, "partial-acceptance mask token is outside vocabulary");
    words[row * stride + tokens[row] / 32] |= 1u << (tokens[row] % 32);
  }
  return words;
}
void initialMask(model::Runtime &runtime, uint64_t id, uint64_t position,
                 const std::vector<uint32_t> &pages, uint32_t vocabulary,
                 const std::vector<uint32_t> &given = {}, bool zeroChain = false) {
  const std::array<uint32_t, 1> token{0};
  std::vector<uint32_t> forced;
  if (zeroChain) for (uint32_t row = 0; row < model::ExecutionLimits::targetVerifyRows; ++row)
    forced.insert(forced.end(), {0, 0});
  require(!runtime.provideMask(id, given.empty() ? singletonMasks(token, vocabulary) : given, forced),
          "initial mask was refused");
  BatchPlan plan{.kind = WorkKind::Decode, .items = {{id, 0}},
                 .decodeStage = DecodeStage::ApplyInitialMask};
  ModelBatchItem item{id, position, 0, pages};
  item.pageTableRevision = 1;
  auto ticket = runtime.submit(plan, {&item, 1}, {});
  const auto results = ticket->wait();
  require(results.size() == 1 && results.front().failure.empty() && results.front().outputTokens.empty(),
          "initial mask selection failed or committed an output row");
}
ModelStepResult maskedStep(model::Runtime &runtime, metal::MetalBackend &backend,
                            uint64_t id, uint64_t position, const std::vector<uint32_t> &pages,
                            const std::function<std::vector<uint32_t>(const ModelMaskRequest &)> &maskFor) {
  require(runtime.reserveSpilledReads(id, {.kind = WorkKind::Decode}).granted(),
          "partial-acceptance hot floor was refused");
  BatchPlan plan{.kind = WorkKind::Decode, .items = {{id, 0}}, .decodeStage = DecodeStage::Regular};
  ModelBatchItem item{id, position, 0, pages};
  item.pageTableRevision = 1;
  auto ticket = runtime.submit(plan, {&item, 1}, {});
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
  const auto advance = [&] {
    require(std::chrono::steady_clock::now() < deadline, "partial-acceptance ticket timed out");
    ticket->advance();
    std::this_thread::yield();
  };
  std::vector<ModelMaskRequest> requests;
  while (requests.empty()) {
    advance();
    requests = ticket->takeMaskRequests();
    require(!ticket->ready(), "masked decode finished without its verify masks");
  }
  require(requests.size() == 1 && requests.front().requestId == id,
          "partial-acceptance simulation belongs to another request");
  const auto &asked = requests.front();
  // Drain target forward without supplying the mask. A host-only mask wait
  // must not be mistaken for a safe point merely because the GPU is idle.
  while (backend.commandInFlight()) advance();
  require(!runtime.parkDecodedRequest(id).granted(), "mask-waiting request was parked with an idle GPU");
  require(!runtime.provideMask(id, maskFor(asked), {}), "verify masks were refused");
  while (!ticket->ready()) advance();
  const auto results = ticket->wait();
  require(results.size() == 1 && results.front().failure.empty() &&
              results.front().acceptedDraftTokens <= results.front().draftedTokens,
          "masked continuation command failed");
  return results.front();
}
ModelStepResult partialStep(model::Runtime &runtime, metal::MetalBackend &backend,
                            uint64_t id, uint64_t position, const std::vector<uint32_t> &pages,
                            uint32_t vocabulary, uint32_t accepted) {
  const auto result = maskedStep(runtime, backend, id, position, pages, [&](const auto &asked) {
    const auto &simulation = asked.simulationTokens;
    require(asked.parents.size() == simulation.size(), "partial fixture has no tree parents");
    std::vector<uint32_t> depths(simulation.size(), 0);
    uint32_t deepest = 0;
    for (uint32_t row = 1; row < simulation.size(); ++row) {
      require(asked.parents[row] < row, "partial fixture parent is not earlier than its child");
      depths[row] = depths[asked.parents[row]] + 1;
      if (depths[row] >= depths[deepest]) deepest = row;
    }
    require(accepted > 0 && depths[deepest] >= accepted, "fixture tree depth " + std::to_string(depths[deepest]) + " is too shallow for acceptance " + std::to_string(accepted));
    std::vector<uint32_t> path;
    for (uint32_t row = deepest;; row = asked.parents[row]) {
      path.push_back(row);
      if (!row) break;
    }
    std::reverse(path.begin(), path.end());
    std::vector<uint32_t> selected(simulation.size() + 1, 0);
    selected[0] = simulation.front();
    for (uint32_t depth = 0; depth < accepted; ++depth)
      selected[path[depth] + 1] = simulation[path[depth + 1]];
    uint32_t mismatch = 0;
    while (mismatch < vocabulary && std::any_of(simulation.begin() + 1, simulation.end(),
        [&](const auto &token) { const auto row = &token - simulation.data();
          return asked.parents[row] == path[accepted] && token == mismatch; })) ++mismatch;
    require(mismatch < vocabulary, "partial fixture has every token as a child");
    selected[path[accepted] + 1] = mismatch;
    return singletonMasks(selected, vocabulary);
  });
  require(result.acceptedDraftTokens == accepted && result.outputTokens.size() == accepted + 1,
          "mask did not force the requested intermediate acceptance length");
  return result;
}

// Follow the real-model spill oracle's SlotFormat -> Index-page transition,
// but use its in-memory copy to inject read faults without touching any SSD.
void spilledContinuation(RuntimeResources &resources, model::Runtime &runtime) {
  auto context = resources.modelContext();
  auto &pool = resources.kvPool();
  auto &cache = *context.blockCache;
  const auto shape = *context.model.descriptor.capabilities.sparseKv;
  const uint32_t densePages = (shape.denseHistoryTokens + kv::kPageTokens - 1) / kv::kPageTokens;
  const uint32_t count = (densePages + 5) * kv::kPageTokens;
  const uint32_t pageCount = count / kv::kPageTokens + 4;
  const kv::SlotFormat format(context.kvPages.layout());
  require(format.version() == 1, "continuation fixture has no spillable slot format");
  std::vector<uint32_t> prompt(count);
  for (uint32_t i = 0; i < count; ++i) prompt[i] = 10 + i % 100;
  uint64_t block = 10'000;
  // A successful cold read, a failed cold read, then a fresh successful
  // request: failed commands are ended, as Engine requires, never retried
  // in-place with recurrent/draft scratch that the failed GPU command wrote.
  for (const bool fail : {false, true, false}) {
    const auto acquire = [&] {
      auto result = pool.acquirePages(pageCount);
      require(result.granted() && result.pages.size() == pageCount, "spill fixture pages refused");
      for (const auto page : result.pages) pool.clearPage(page);
      return result.pages;
    };
    auto resident = acquire(), spilled = acquire(), intruderPages = acquire();
    EngineRequest a, b;
    a.id = 1001; a.prompt = prompt; a.maxNewTokens = 4096;
    a.flags = RequestIgnoreEndOfSequence; a.sampling.temperature = 0.7F; a.sampling.seed = 47;
    b = a; b.id = 1002;
    runtime.beginColdRequest(a.modelView(), 0);
    runtime.beginColdRequest(b.modelView(), 1);
    for (uint32_t at = 0; at < count;) {
      const uint32_t rows = std::min(count - at, runtime.commandLimits().commandRows);
      const auto input = std::span(prompt).subspan(at, rows);
      static_cast<void>(step(runtime, a.id, at, resident, input));
      static_cast<void>(step(runtime, b.id, at, spilled, input));
      at += rows;
    }
    const auto initialA = step(runtime, a.id, count, resident);
    const auto initialB = step(runtime, b.id, count, spilled);
    require(!initialA.outputTokens.empty() && initialA.outputTokens == initialB.outputTokens &&
                initialA.outputTokensWithoutKv == initialB.outputTokensWithoutKv,
            "spill controls differed before moving pages");
    const uint64_t position = count + initialA.outputTokens.size() - initialA.outputTokensWithoutKv;
    auto chunk = std::make_shared<model::KvPageSources::Chunk>();
    require(pageCount <= model::KvPageSources::kChunkPages, "spill fixture needs another source chunk");
    std::vector<std::shared_ptr<test::TestKvCopy>> copies;
    // Keep dense history and a complete trailing window page resident.
    for (uint32_t page = densePages; page + 1 < count / kv::kPageTokens; ++page) {
      std::vector<std::byte> payload;
      for (const auto span : format.transferSpans(context.kvPages, spilled[page]))
        payload.insert(payload.end(), span.begin(), span.end());
      auto copy = std::make_shared<test::TestKvCopy>(std::move(payload));
      const std::array old{spilled[page]};
      const auto moved = pool.movePages(old, kv::PageForm::Index);
      require(moved.granted(), "continuation page could not move to Index form");
      for (uint32_t layer = 0; layer < context.kvPages.layout().attentionLayers; ++layer)
        for (auto tensor : context.kvPages.layerSpans(old[0], layer))
          std::ranges::fill(tensor, std::byte{0x7f});
      spilled[page] = moved.moves.follow(old[0]);
      chunk->pages[page] = {++block, copy};
      ++chunk->indexPages;
      copies.push_back(std::move(copy));
    }
    const auto sources = std::make_shared<const model::KvPageSources>(
        std::vector<std::shared_ptr<const model::KvPageSources::Chunk>>{chunk}, pageCount);
    require(!copies.empty(), "continuation fixture did not spill any history");
    require(runtime.reserveSpilledReads(b.id, {.kind = WorkKind::Decode}).granted(),
            "spilled continuation hot floor refused");
    require(cache.holdsLane(1), "spilled request did not hold its hot floor");
    const auto saved = state(runtime, context.stateStorage, 1);
    require(saved.metadata.pendingPath > 0, "spilled continuation retained no speculative GDN rows");
    require(runtime.parkDecodedRequest(b.id).granted(), "spilled continuation could not park");
    require(!cache.holdsLane(1), "parked request retained its hot floor");
    while (cache.releaseAboveFloor()) {}
    // Overwrite the old decoder lane with a different request.
    EngineRequest intruder = a; intruder.id = 1003; intruder.prompt = {11, 12, 13};
    runtime.beginColdRequest(intruder.modelView(), 1);
    static_cast<void>(step(runtime, intruder.id, 0, intruderPages, intruder.prompt));
    const auto restored = runtime.restoreParkedRequest(b.id);
    require(restored.granted() && *restored.lane == 2, "spilled continuation did not rebind");
    equal(saved, state(runtime, context.stateStorage, 2));
    for (auto &copy : copies) copy->failAll = fail;
    const auto expected = step(runtime, a.id, position, resident);
    const uint32_t vocabulary = context.model.descriptor.capabilities.vocabularySize;
    const auto expectedSpan = model::RuntimeInstrumentation::lastLogits(runtime, 0).first(vocabulary);
    const std::vector<float> expectedLogits(expectedSpan.begin(), expectedSpan.end());
    require(std::any_of(expectedLogits.begin(), expectedLogits.end(),
                        [&](float value) { return value != expectedLogits.front(); }),
            "spilled continuation has only tied target logits");
    require(runtime.reserveSpilledReads(b.id, {.kind = WorkKind::Decode}).granted() && cache.holdsLane(2),
            "restored request could not reacquire its hot floor");
    BatchPlan plan{.kind = WorkKind::Decode, .items = {{b.id, 0}}, .decodeStage = DecodeStage::Regular};
    ModelBatchItem item{b.id, position, 0, spilled};
    item.pageTableRevision = 2;
    item.pageSources = sources;
    auto ticket = runtime.submit(plan, {&item, 1}, {});
    const auto results = ticket->wait();
    require(results.size() == 1 && results[0].requestId == b.id, "spilled continuation lost its result");
    const auto &actual = results[0];
    uint64_t reads = 0;
    for (const auto &copy : copies) reads += copy->reads();
    require(reads > 0, "continuation never read its spilled copies");
    if (fail) {
      require(actual.failure.empty() && !actual.unreadableKvBlocks.empty() && actual.outputTokens.empty(),
              "failed spilled reads silently committed continuation tokens");
      for (const auto unreadable : actual.unreadableKvBlocks)
        require(std::any_of(chunk->pages.begin(), chunk->pages.end(),
                            [&](const auto &source) { return source.block == unreadable; }),
                "failed continuation reported another request's spilled block");
    } else {
      require(actual.failure.empty() && actual.unreadableKvBlocks.empty() &&
                  !actual.outputTokens.empty() && actual.outputTokens == expected.outputTokens &&
                  actual.draftedTokens == expected.draftedTokens &&
                  actual.acceptedDraftTokens == expected.acceptedDraftTokens &&
                  actual.outputTokensWithoutKv == expected.outputTokensWithoutKv,
              "spilled continuation differed from resident control");
      const auto actualLogits = model::RuntimeInstrumentation::lastLogits(runtime, 0).first(vocabulary);
      require(std::equal(expectedLogits.begin(), expectedLogits.end(), actualLogits.begin()),
              "spilled continuation logits differed from resident control");
    }
    runtime.end(a.id); runtime.end(b.id); runtime.end(intruder.id);
    for (auto page : resident) pool.releasePage(page, false);
    for (auto page : spilled) pool.releasePage(page, false);
    for (auto page : intruderPages) pool.releasePage(page, false);
    std::cout << "spilled_continuation=PASS fault=" << fail << " reads=" << reads << '\n';
  }
}

void run(const char *metallib) {
  const test::TemporaryDirectory root("splash-ram-continuation");
  for (const bool flashNext : {false, true}) {
    auto config = flashNext
        ? test::flashNextConfig(metallib, root.path() / "flash-next", gguf_fixtures::kPleTableRows, 2560, 8)
        : test::loadableConfig(metallib, root.path() / "dense", "ram-continuation-unit");
    config.aneFfn.enabled = false;
    config.verifyRows = model::VerifyRows::All;
    if (flashNext) {
      config.cacheDisk = {.bytes = 2 * kGiB};
      config.cacheRoot = root.path() / "unit-cache";
      config.diskVolume = test::reading(test::localVolume(500 * kGiB));
      config.kvSpillAlways = true;
    }
    for (const auto format : {ops::GdnStateFormat::Float32, ops::GdnStateFormat::BFloat16}) {
      config.stateFormat = format;
      auto resources = RuntimeResources::create(config, 4096);
      if (flashNext)
        require(resources->modelContext().blockCache != nullptr, "Flash-Next fixture has no hot KV reader");
      auto &governor = resources->memoryGovernor();
      const test::ClaimScope claim(governor, ClaimClass::Floor);
      const auto &budget = resources->memoryPlan().breakdown();
      std::unique_ptr<model::Runtime> runtime;
      require(static_cast<bool>(governor.allocationAdmission()(
          budget.sharedPrefillBytes + budget.sharedDecodeBytes,
          [&] { runtime = std::make_unique<model::Runtime>(resources->modelContext()); })),
          "continuation runtime arenas were refused");
      require(runtime->capturesInDecode() == !flashNext,
              "decode-capture capability did not distinguish auxiliary-state Flash-Next");
      auto &storage = resources->modelContext().stateStorage;
      auto &pool = resources->kvPool();
      const auto takePages = [&] {
        auto acquired = pool.acquirePages(16);
        require(acquired.granted() && acquired.pages.size() == 16, "continuation KV pages were refused");
        for (const uint32_t page : acquired.pages) pool.clearPage(page);
        return acquired.pages;
      };
      const auto baselinePages = takePages(), parkedPages = takePages(), intruderPages = takePages();
      std::vector<uint32_t> prompt(40);
      for (uint32_t i = 0; i < prompt.size(); ++i) prompt[i] = 10 + i;
      const auto input = [&](uint64_t id) {
        EngineRequest request;
        request.id = id;
        request.prompt = prompt;
        request.maxNewTokens = 4096;
        request.flags = RequestIgnoreEndOfSequence;
        request.sampling.temperature = 0.8F;
        request.sampling.presencePenalty = 0.2F;
        request.sampling.frequencyPenalty = 0.1F;
        request.sampling.seed = 47;
        return request;
      };
      auto baseline = input(101), candidate = input(202), intruder = input(303);
      require(runtime->begin(baseline.modelView()).lane == 0 &&
                  runtime->begin(candidate.modelView()).lane == 1,
              "continuation lanes did not begin independently");
      require(!runtime->parkDecodedRequest(202).granted(), "an unfinished prefill was parked");
      runtime->setDraftContextPlan(101, planDraftContext(0, prompt.size(), {}));
      runtime->setDraftContextPlan(202, planDraftContext(0, prompt.size(), {}));
      static_cast<void>(step(*runtime, 101, 0, baselinePages, prompt));
      static_cast<void>(step(*runtime, 202, 0, parkedPages, prompt));
      uint64_t position = prompt.size();
      std::vector<uint32_t> output;
      const auto compareStep = [&] {
        const auto a = step(*runtime, 101, position, baselinePages, {}, true);
        const auto b = step(*runtime, 202, position, parkedPages);
        require(!a.finished && !b.finished && a.outputTokens == b.outputTokens &&
                    a.draftedTokens == b.draftedTokens && a.acceptedDraftTokens == b.acceptedDraftTokens &&
                    a.outputTokensWithoutKv == b.outputTokensWithoutKv,
                "parked stream changed sampled tokens or speculative acceptance");
        require(!a.outputTokens.empty(), "synthetic decode made no progress");
        position += a.outputTokens.size() - a.outputTokensWithoutKv;
        output.insert(output.end(), a.outputTokens.begin(), a.outputTokens.end());
      };
      for (uint32_t cycle = 0; cycle < 3; ++cycle) compareStep();
      std::set<uint32_t> parities;
      for (uint32_t rotation = 0; rotation < 2; ++rotation) {
        const uint32_t oldLane = rotation == 0 ? 1 : 2;
        const auto saved = state(*runtime, storage, oldLane);
        parities.insert(saved.metadata.activeParity);
        require(saved.metadata.pendingPath && !saved.pending.empty(), "fixture has no pending recurrent rows");
        const auto anchor = model::RuntimeInstrumentation::pendingToken(*runtime, 202);
        const auto tables = model::RuntimeInstrumentation::laneTables(*runtime, 202);
        const std::vector<SplashKvPage> expectedPages(tables.pages.begin(),
                                                      tables.pages.begin() + parkedPages.size());
        const auto before = runtime->actualRuntimeMemory();
        require(runtime->parkDecodedRequest(202).granted(), "decoded request could not be parked");
        require(runtime->actualRuntimeMemory().stateActualAllocatedBytes > before.stateActualAllocatedBytes,
                "parked scratch was omitted from runtime memory accounting");
        rejects([&] { static_cast<void>(runtime->resume(candidate.modelView())); }, "not suspended",
                "ordinary replay resume accepted a RAM continuation");
        // A different stream overwrites the released lane's tape, penalties and tables.
        runtime->beginColdRequest(intruder.modelView(), oldLane);
        require(storage.metadata(oldLane).requestId == 303, "intruder did not reuse the parked lane");
        static_cast<void>(step(*runtime, 303, 0, intruderPages, prompt, false, true));
        static_cast<void>(step(*runtime, 303, prompt.size(), intruderPages));
        auto filler = input(404);
        require(runtime->begin(filler.modelView()).granted(), "filler could not occupy a lane");
        auto other = input(405);
        require(runtime->begin(other.modelView()).granted(), "second filler could not occupy a lane");
        const auto full = runtime->restoreParkedRequest(202);
        require(!full.granted() && full.failure == StateFailure::ConcurrencyLimit,
                "a full-lane restore destroyed the parked continuation");
        runtime->end(404);
        const auto restored = runtime->restoreParkedRequest(202);
        require(restored.granted() && *restored.lane != oldLane, "continuation did not rebind to a free lane");
        equal(saved, state(*runtime, storage, *restored.lane));
        require(model::RuntimeInstrumentation::pendingToken(*runtime, 202) == anchor &&
                    runtime->actualRuntimeMemory().stateActualAllocatedBytes >= before.stateActualAllocatedBytes,
                "restore changed the sampling anchor or undercounted held state");
        runtime->end(303);
        runtime->end(405);
        for (uint32_t cycle = 0; cycle < (rotation == 0 ? 1u : 2u); ++cycle) compareStep();
        const auto rebound = model::RuntimeInstrumentation::laneTables(*runtime, 202);
        require(std::memcmp(rebound.pages.data(), expectedPages.data(),
                            expectedPages.size() * sizeof(SplashKvPage)) == 0,
                "lane rebinding retained another request's KV table at the same revision");
      }
      require(parities.size() == 2, "continuation fixture did not exercise both state parities");
      require(std::set<uint32_t>(output.begin(), output.end()).size() > 1,
              "continuation token parity used a constant-output fixture");
      require(runtime->parkDecodedRequest(202).granted(), "cancellation fixture could not park");
      runtime->end(202);
      rejects([&] { static_cast<void>(runtime->restoreParkedRequest(202)); }, "unknown request",
              "cancelled parked request was resurrected");
      runtime->end(101);
      // A greedy zero-weight control accepts an entire draft block. It
      // exercises the largest pending-row copy as well as sampled single rows.
      for (const auto *pages : {&baselinePages, &parkedPages, &intruderPages})
        for (const uint32_t page : *pages) pool.clearPage(page);
      auto greedyBase = input(601), greedyParked = input(602);
      greedyBase.sampling = {};
      greedyParked.sampling = {};
      greedyBase.constraint = greedyParked.constraint = ConstraintMode::TokenMask;
      runtime->beginColdRequest(greedyBase.modelView(), 0);
      runtime->beginColdRequest(greedyParked.modelView(), 1);
      static_cast<void>(step(*runtime, 601, 0, baselinePages, prompt));
      static_cast<void>(step(*runtime, 602, 0, parkedPages, prompt));
      const uint32_t fullVocabulary = std::visit([](const auto &layout) { return layout.vocabularySize; }, config.model.target);
      initialMask(*runtime, 601, prompt.size(), baselinePages, fullVocabulary, {}, true);
      initialMask(*runtime, 602, prompt.size(), parkedPages, fullVocabulary, {}, true);
      const auto allZeroMasks = [&](const ModelMaskRequest &asked) {
        return singletonMasks(std::vector<uint32_t>(asked.simulationTokens.size() + 1, 0), fullVocabulary);
      };
      model::RuntimeInstrumentation::forceStepRows(*runtime, model::ExecutionLimits::targetVerifyRows);
      const auto fullBase = maskedStep(*runtime, resources->backend(), 601, prompt.size(), baselinePages, allZeroMasks);
      const auto fullParked = maskedStep(*runtime, resources->backend(), 602, prompt.size(), parkedPages, allZeroMasks);
      require(fullBase.acceptedDraftTokens > 0 &&
                  fullBase.acceptedDraftTokens == fullBase.draftedTokens &&
                  fullBase.outputTokens == fullParked.outputTokens,
              "greedy continuation fixture did not accept the full draft block");
      const auto fullState = state(*runtime, storage, 1);
      require(fullState.metadata.pendingPath > 1 && runtime->parkDecodedRequest(602).granted(),
              "full-acceptance state did not park multiple retained rows");
      runtime->beginColdRequest(intruder.modelView(), 1);
      static_cast<void>(step(*runtime, 303, 0, intruderPages, prompt));
      static_cast<void>(step(*runtime, 303, prompt.size(), intruderPages));
      const auto fullRestored = runtime->restoreParkedRequest(602);
      require(fullRestored.granted() && *fullRestored.lane == 2,
              "full-acceptance continuation did not rebind");
      equal(fullState, state(*runtime, storage, 2));
      const uint64_t fullPosition = prompt.size() + fullBase.outputTokens.size();
      const auto afterBase = maskedStep(*runtime, resources->backend(), 601, fullPosition, baselinePages, allZeroMasks);
      const auto afterParked = maskedStep(*runtime, resources->backend(), 602, fullPosition, parkedPages, allZeroMasks);
      model::RuntimeInstrumentation::forceStepRows(*runtime, std::nullopt);
      require(afterBase.outputTokens == afterParked.outputTokens &&
                  afterBase.acceptedDraftTokens == afterParked.acceptedDraftTokens,
              "full-acceptance continuation changed output after restoration");
      runtime->end(601);
      runtime->end(602);
      runtime->end(303);
      const uint32_t vocabulary = std::visit([](const auto &layout) { return layout.vocabularySize; },
                                               config.model.target);
      uint32_t partialCases = 0, sparsePartialPaths = 0;
      model::RuntimeInstrumentation::forceStepRows(*runtime, model::ExecutionLimits::targetVerifyRows);
      for (uint32_t accepted = 1; accepted < model::ExecutionLimits::draftProposalTokens; ++accepted) {
        for (const auto *pages : {&baselinePages, &parkedPages, &intruderPages})
          for (const uint32_t page : *pages) pool.clearPage(page);
        auto maskedBase = input(701), maskedParked = input(702);
        if (accepted > 1) for (uint32_t row = 0; row < prompt.size(); ++row)
          maskedBase.prompt[row] = maskedParked.prompt[row] = (row + 8) % 16;
        maskedBase.flags = maskedParked.flags = 0;
        maskedBase.sampling = maskedParked.sampling = {};
        maskedBase.constraint = maskedParked.constraint = ConstraintMode::TokenMask;
        runtime->beginColdRequest(maskedBase.modelView(), 0);
        runtime->beginColdRequest(maskedParked.modelView(), 1);
        require(step(*runtime, 701, 0, baselinePages, maskedBase.prompt).nextDecodeStage == DecodeStage::ApplyInitialMask &&
                    step(*runtime, 702, 0, parkedPages, maskedParked.prompt).nextDecodeStage == DecodeStage::ApplyInitialMask,
                "masked fixture skipped its initial selection stage");
        require(!runtime->parkDecodedRequest(702).granted(), "initial-mask waiter was parked");
        initialMask(*runtime, 701, prompt.size(), baselinePages, vocabulary);
        initialMask(*runtime, 702, prompt.size(), parkedPages, vocabulary);
        const auto a = partialStep(*runtime, resources->backend(), 701, prompt.size(), baselinePages,
                                   vocabulary, accepted);
        const auto b = partialStep(*runtime, resources->backend(), 702, prompt.size(), parkedPages,
                                   vocabulary, accepted);
        require(a.outputTokens == b.outputTokens, "partial-acceptance controls diverged before parking");
        const auto saved = state(*runtime, storage, 1);
        if (saved.metadata.pendingPath & (saved.metadata.pendingPath + 1)) ++sparsePartialPaths;
        require(std::popcount(saved.metadata.pendingPath) == int(accepted + 1) && runtime->parkDecodedRequest(702).granted(),
                "intermediate pending state was not parked exactly");
        runtime->beginColdRequest(intruder.modelView(), 1);
        static_cast<void>(step(*runtime, 303, 0, intruderPages, prompt));
        static_cast<void>(step(*runtime, 303, prompt.size(), intruderPages));
        const auto restored = runtime->restoreParkedRequest(702);
        require(restored.granted() && *restored.lane == 2, "masked continuation did not rebind");
        equal(saved, state(*runtime, storage, 2));
        const uint64_t next = prompt.size() + a.outputTokens.size();
        const auto nextMasks = [&](const ModelMaskRequest &asked) {
          std::vector<uint32_t> selected(asked.simulationTokens.size() + 1, 0);
          selected[0] = asked.simulationTokens.front();
          return singletonMasks(selected, vocabulary);
        };
        const auto afterA = maskedStep(*runtime, resources->backend(), 701, next, baselinePages, nextMasks);
        const auto afterB = maskedStep(*runtime, resources->backend(), 702, next, parkedPages, nextMasks);
        require(afterA.outputTokens == afterB.outputTokens &&
                    afterA.acceptedDraftTokens == afterB.acceptedDraftTokens,
                "partial-acceptance stream changed after parking");
        runtime->end(701);
        runtime->end(702);
        runtime->end(303);
        ++partialCases;
      }
      require(sparsePartialPaths > 0, "partial fixtures never retained a nonconsecutive tree path");
      model::RuntimeInstrumentation::forceStepRows(*runtime, std::nullopt);
      // This is a byte-ownership test, not a valid numerical continuation:
      // canary-filled pending rows must never be folded by another decode.
      auto canary = input(711);
      runtime->beginColdRequest(canary.modelView(), 0);
      static_cast<void>(step(*runtime, 711, 0, baselinePages, prompt));
      static_cast<void>(step(*runtime, 711, prompt.size(), baselinePages));
      constexpr uint32_t canaryPath = 1u | (1u << 3) | (1u << 15);
      storage.swapDecodeParity(0, canaryPath);
      storage.swapDecodeParity(0, canaryPath);
      model::RuntimeInstrumentation::fillPendingTape(*runtime, 711, 41);
      const auto canaryState = state(*runtime, storage, 0);
      require(runtime->parkDecodedRequest(711).granted(), "sparse canary tape could not park");
      runtime->beginColdRequest(intruder.modelView(), 0);
      static_cast<void>(step(*runtime, 303, 0, intruderPages, prompt));
      static_cast<void>(step(*runtime, 303, prompt.size(), intruderPages));
      require(storage.metadata(0).activeParity == canaryState.metadata.activeParity,
              "canary intruder did not overwrite the vacated parity slot");
      model::RuntimeInstrumentation::fillPendingTape(*runtime, 303, 79);
      const auto canaryRestored = runtime->restoreParkedRequest(711);
      require(canaryRestored.granted() && *canaryRestored.lane == 1, "sparse canary tape did not rebind");
      equal(canaryState, state(*runtime, storage, 1));
      runtime->end(711);
      runtime->end(303);
      uint32_t thinkingCases = 0;
      for (const uint32_t thinkingTokens : {0u, 1u, 5u, 12u}) {
        for (const auto *pages : {&baselinePages, &parkedPages, &intruderPages})
          for (const uint32_t page : *pages) pool.clearPage(page);
        auto thinkingBase = input(801), thinkingParked = input(802);
        thinkingBase.sampling = thinkingParked.sampling = {};
        thinkingBase.thinkingBudget = thinkingParked.thinkingBudget = ThinkingBudget{thinkingTokens, {2, 3}, 4};
        runtime->beginColdRequest(thinkingBase.modelView(), 0);
        runtime->beginColdRequest(thinkingParked.modelView(), 1);
        static_cast<void>(step(*runtime, 801, 0, baselinePages, prompt));
        static_cast<void>(step(*runtime, 802, 0, parkedPages, prompt));
        const auto saved = state(*runtime, storage, 1);
        require(saved.metadata.pendingPath == 0 && runtime->parkDecodedRequest(802).granted(),
                "post-prefill thinking state did not park without a pending tape");
        const auto anchor = model::RuntimeInstrumentation::pendingToken(*runtime, 802);
        runtime->beginColdRequest(intruder.modelView(), 1);
        static_cast<void>(step(*runtime, 303, 0, intruderPages, prompt));
        static_cast<void>(step(*runtime, 303, prompt.size(), intruderPages));
        const auto restored = runtime->restoreParkedRequest(802);
        require(restored.granted() && *restored.lane == 2 &&
                    model::RuntimeInstrumentation::pendingToken(*runtime, 802) == anchor,
                "thinking state lost its pending anchor when rebound");
        equal(saved, state(*runtime, storage, 2));
        uint64_t next = prompt.size();
        std::vector<uint32_t> tokens;
        for (uint32_t cycle = 0; cycle < 40 && tokens.size() < thinkingTokens + 6; ++cycle) {
          const auto a = step(*runtime, 801, next, baselinePages);
          const auto b = step(*runtime, 802, next, parkedPages);
          require(a.outputTokens == b.outputTokens && a.acceptedDraftTokens == b.acceptedDraftTokens &&
                      a.outputTokensWithoutKv == b.outputTokensWithoutKv,
                  "thinking budget or end sequence changed after parking");
          next += a.outputTokens.size() - a.outputTokensWithoutKv;
          tokens.insert(tokens.end(), a.outputTokens.begin(), a.outputTokens.end());
        }
        const std::array<uint32_t, 2> end{2, 3};
        require(tokens.size() >= thinkingTokens + 6 && tokens[thinkingTokens] == 2 &&
                    tokens[thinkingTokens + 1] == 3 &&
                    std::search(tokens.begin(), tokens.end(), end.begin(), end.end()) != tokens.end() &&
                    std::count(tokens.begin(), tokens.end(), 2) == 1 &&
                    std::count(tokens.begin(), tokens.end(), 3) == 1,
                "thinking fixture lost, duplicated or reordered its end sequence");
        runtime->end(801);
        runtime->end(802);
        runtime->end(303);
        ++thinkingCases;
      }
      uint32_t combinedCases = 0;
      for (const float temperature : {0.0F, 0.7F}) {
        for (const uint32_t thinkingTokens : {0u, 1u, 5u, 12u}) {
          for (const auto *pages : {&baselinePages, &parkedPages, &intruderPages})
            for (const uint32_t page : *pages) pool.clearPage(page);
          // Select exact grammar/end safe points independently of the
          // tree's ability to accept several character tokens in one cycle.
          model::RuntimeInstrumentation::forceStepRows(*runtime, 1);
          auto aRequest = input(901), bRequest = input(902);
          aRequest.flags = bRequest.flags = 0;
          aRequest.constraint = bRequest.constraint = ConstraintMode::TokenMask;
          aRequest.sampling = bRequest.sampling = {};
          aRequest.sampling.temperature = bRequest.sampling.temperature = temperature;
          aRequest.sampling.seed = bRequest.sampling.seed = 47;
          aRequest.thinkingBudget = bRequest.thinkingBudget = ThinkingBudget{thinkingTokens, {2, 3}, 4};
          const uint32_t character = thinkingTokens ? thinkingTokens - 1 : 0;
          const uint32_t endAt = character + 3;
          const auto maskAt = [&](uint64_t position) {
            if (position >= character && position < endAt) {
              const std::array<uint32_t, 1> token{static_cast<uint32_t>(5 + position - character)};
              return singletonMasks(token, vocabulary);
            }
            std::vector<uint32_t> words((vocabulary + 31) / 32, ~0u);
            // Admit the desired thinking-end token without forcing it: a
            // lost thinking continuation would choose an ordinary token.
            for (const uint32_t token : {2u, 3u, 4u, 5u, 6u, 7u})
              if (!((position == endAt && token == 2) || (position == endAt + 1 && token == 3)))
                words[token / 32] &= ~(1u << (token % 32));
            return words;
          };
          runtime->beginColdRequest(aRequest.modelView(), 0);
          runtime->beginColdRequest(bRequest.modelView(), 1);
          require(step(*runtime, 901, 0, baselinePages, prompt).nextDecodeStage == DecodeStage::ApplyInitialMask &&
                      step(*runtime, 902, 0, parkedPages, prompt).nextDecodeStage == DecodeStage::ApplyInitialMask,
                  "combined fixture skipped its initial masks");
          initialMask(*runtime, 901, prompt.size(), baselinePages, vocabulary, maskAt(0));
          initialMask(*runtime, 902, prompt.size(), parkedPages, vocabulary, maskAt(0));
          std::vector<uint32_t> generated;
          const auto compareMasked = [&] {
            const uint64_t position = generated.size();
            const auto masks = [&](const ModelMaskRequest &asked) {
              std::vector<uint32_t> depths(asked.simulationTokens.size(), 0);
              require(asked.parents.size() == depths.size(), "combined fixture has no tree parents");
              for (uint32_t row = 1; row < depths.size(); ++row) {
                require(asked.parents[row] < row, "combined fixture has a forward tree parent");
                depths[row] = depths[asked.parents[row]] + 1;
              }
              auto words = maskAt(position);
              for (const auto depth : depths) {
                const auto next = maskAt(position + depth + 1);
                words.insert(words.end(), next.begin(), next.end());
              }
              return words;
            };
            const auto a = maskedStep(*runtime, resources->backend(), 901, prompt.size() + position,
                                      baselinePages, masks);
            const auto b = maskedStep(*runtime, resources->backend(), 902, prompt.size() + position,
                                      parkedPages, masks);
            require(!a.finished && !b.finished && !a.outputTokens.empty() &&
                        a.outputTokens == b.outputTokens && a.acceptedDraftTokens == b.acceptedDraftTokens &&
                        a.outputTokensWithoutKv == b.outputTokensWithoutKv,
                    "combined grammar/thinking continuation changed its next output");
            generated.insert(generated.end(), a.outputTokens.begin(), a.outputTokens.end());
          };
          uint32_t cycles = 0;
          while (model::RuntimeInstrumentation::pendingToken(*runtime, 902) != 6 && cycles++ < 20)
            compareMasked();
          require(model::RuntimeInstrumentation::pendingToken(*runtime, 902) == 6 &&
                      generated.size() == character + 1,
                  "fixture did not pause inside a grammar-delayed thinking end");
          const auto saved = state(*runtime, storage, 1);
          require(runtime->parkDecodedRequest(902).granted(), "combined continuation could not park");
          runtime->beginColdRequest(intruder.modelView(), 1);
          static_cast<void>(step(*runtime, 303, 0, intruderPages, prompt));
          static_cast<void>(step(*runtime, 303, prompt.size(), intruderPages));
          const auto restored = runtime->restoreParkedRequest(902);
          require(restored.granted() && *restored.lane == 2, "combined continuation did not rebind");
          equal(saved, state(*runtime, storage, 2));
          while (model::RuntimeInstrumentation::pendingToken(*runtime, 902) != 2 && cycles++ < 30)
            compareMasked();
          require(model::RuntimeInstrumentation::pendingToken(*runtime, 902) == 2 && generated.size() == endAt,
                  "fixture did not reach the first selected thinking-end token");
          runtime->end(303);
          const auto ending = state(*runtime, storage, 2);
          require(runtime->parkDecodedRequest(902).granted(), "partly selected thinking end could not park");
          runtime->beginColdRequest(intruder.modelView(), 2);
          static_cast<void>(step(*runtime, 303, 0, intruderPages, prompt));
          static_cast<void>(step(*runtime, 303, prompt.size(), intruderPages));
          const auto endingRestored = runtime->restoreParkedRequest(902);
          require(endingRestored.granted() && *endingRestored.lane == 1,
                  "partly selected thinking end did not rebind");
          equal(ending, state(*runtime, storage, 1));
          model::RuntimeInstrumentation::forceStepRows(*runtime, std::nullopt);
          while (generated.size() < endAt + 5 && cycles++ < 40) compareMasked();
          const std::array<uint32_t, 3> characterTokens{5, 6, 7};
          require(generated.size() >= endAt + 5 &&
                      std::equal(characterTokens.begin(), characterTokens.end(), generated.begin() + character) &&
                      generated[endAt] == 2 && generated[endAt + 1] == 3 &&
                      std::count(generated.begin(), generated.end(), 2) == 1 &&
                      std::count(generated.begin(), generated.end(), 3) == 1,
                  "combined continuation ended thinking early, twice or out of order");
          runtime->end(901);
          runtime->end(902);
          runtime->end(303);
          ++combinedCases;
        }
      }
      auto terminal = input(505);
      terminal.maxNewTokens = 1;
      for (const uint32_t page : intruderPages) pool.clearPage(page);
      runtime->beginColdRequest(terminal.modelView(), 0);
      const auto ended = step(*runtime, 505, 0, intruderPages, prompt);
      require(ended.outputTokens.size() == 1 && !runtime->parkDecodedRequest(505).granted(),
              "an already emitted terminal anchor was parked for another decode");
      runtime->end(505);
      for (uint32_t lane = 0; lane < model::ExecutionLimits::maximumBatchWidth; ++lane)
        require(!storage.metadata(lane).assigned(), "continuation test retained an active lane");
      for (const auto *pages : {&baselinePages, &parkedPages, &intruderPages})
        for (const uint32_t page : *pages) pool.releasePage(page, false);
      std::cout << "ram_continuation=PASS family=" << (flashNext ? "flash-next" : "dense")
                << " state=" << (format == ops::GdnStateFormat::Float32 ? "fp32" : "bf16")
                << " rotations=2 sampled_tokens=" << output.size()
                << " full_draft_accepted=" << fullBase.acceptedDraftTokens
                << " partial_cases=" << partialCases
                << " sparse_partial_paths=" << sparsePartialPaths
                << " thinking_cases=" << thinkingCases
                << " combined_cases=" << combinedCases << '\n';
    }
  }
}
void runSpilled(const char *metallib) {
  const test::TemporaryDirectory root("splash-spilled-continuation");
  auto config = test::flashNextConfig(metallib, root.path() / "model",
                                     gguf_fixtures::kPleTableRows, 2560, 8, true);
  config.aneFfn.enabled = false;
  config.verifyRows = model::VerifyRows::All;
  config.cacheDisk = {.bytes = 2 * kGiB};
  config.cacheRoot = root.path() / "unit-cache";
  config.diskVolume = test::reading(test::localVolume(500 * kGiB));
  config.kvSpillAlways = true;
  for (const auto format : {ops::GdnStateFormat::Float32, ops::GdnStateFormat::BFloat16}) {
    config.stateFormat = format;
    auto resources = RuntimeResources::create(config, 4096);
    const test::ClaimScope claim(resources->memoryGovernor(), ClaimClass::Floor);
    const auto &budget = resources->memoryPlan().breakdown();
    std::unique_ptr<model::Runtime> runtime;
    require(static_cast<bool>(resources->memoryGovernor().allocationAdmission()(
        budget.sharedPrefillBytes + budget.sharedDecodeBytes,
        [&] { runtime = std::make_unique<model::Runtime>(resources->modelContext()); })),
        "spilled continuation arenas refused");
    spilledContinuation(*resources, *runtime);
    std::cout << "spilled_format=PASS state="
              << (format == ops::GdnStateFormat::Float32 ? "fp32" : "bf16") << '\n';
  }
}

} // namespace
int main(int argc, const char **argv) {
  if (argc != 2) return EXIT_FAILURE;
  @autoreleasepool {
    try {
      run(argv[1]);
      runSpilled(argv[1]);
      return EXIT_SUCCESS;
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return EXIT_FAILURE;
    }
  }
}
