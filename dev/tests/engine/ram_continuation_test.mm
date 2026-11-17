// Synthetic-weight numerical unit test; never loads the production model.
#include "SyntheticModels.hpp"
#include "ClaimScope.hpp"
#include "TestChecks.hpp"
#include "model/Runtime.hpp"
#include "model/RuntimeInstrumentation.hpp"
#include "model/QwenState.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <chrono>
#include <thread>
#include <iostream>
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
              a.metadata.pendingRows == b.metadata.pendingRows &&
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
  require(results.size() == 1 && results.front().failure.empty(), "synthetic continuation command failed");
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
                 const std::vector<uint32_t> &pages, uint32_t vocabulary) {
  const std::array<uint32_t, 1> token{0};
  require(!runtime.provideMask(id, singletonMasks(token, vocabulary)), "initial mask was refused");
  BatchPlan plan{.kind = WorkKind::Decode, .items = {{id, 0}},
                 .decodeStage = DecodeStage::ApplyInitialMask};
  ModelBatchItem item{id, position, 0, pages};
  item.pageTableRevision = 1;
  auto ticket = runtime.submit(plan, {&item, 1}, {});
  const auto results = ticket->wait();
  require(results.size() == 1 && results.front().failure.empty() && results.front().outputTokens.empty(),
          "initial mask selection failed or committed an output row");
}
ModelStepResult partialStep(model::Runtime &runtime, metal::MetalBackend &backend,
                            uint64_t id, uint64_t position, const std::vector<uint32_t> &pages,
                            uint32_t vocabulary, uint32_t accepted) {
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
  const auto &simulation = requests.front().simulationTokens;
  require(accepted > 0 && accepted + 1 < simulation.size(), "fixture is not partial acceptance");
  // Drain target forward without supplying the mask. A host-only mask wait
  // must not be mistaken for a safe point merely because the GPU is idle.
  while (backend.commandInFlight()) advance();
  require(!runtime.parkDecodedRequest(id).granted(), "mask-waiting request was parked with an idle GPU");
  std::vector<uint32_t> selected(simulation.size() + 1, 0);
  selected[0] = simulation.front();
  for (uint32_t row = 1; row <= accepted; ++row) selected[row] = simulation[row];
  selected[accepted + 1] = simulation[accepted + 1] == 0 ? 1 : 0;
  require(!runtime.provideMask(id, singletonMasks(selected, vocabulary)), "verify masks were refused");
  while (!ticket->ready()) advance();
  const auto results = ticket->wait();
  require(results.size() == 1 && results.front().failure.empty() &&
              results.front().acceptedDraftTokens == accepted &&
              results.front().outputTokens.size() == accepted + 1,
          "mask did not force the requested intermediate acceptance length");
  return results.front();
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
        require(saved.metadata.pendingRows && !saved.pending.empty(), "fixture has no pending recurrent rows");
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
      runtime->beginColdRequest(greedyBase.modelView(), 0);
      runtime->beginColdRequest(greedyParked.modelView(), 1);
      static_cast<void>(step(*runtime, 601, 0, baselinePages, prompt));
      static_cast<void>(step(*runtime, 602, 0, parkedPages, prompt));
      const auto fullBase = step(*runtime, 601, prompt.size(), baselinePages);
      const auto fullParked = step(*runtime, 602, prompt.size(), parkedPages);
      require(fullBase.acceptedDraftTokens > 0 &&
                  fullBase.acceptedDraftTokens == fullBase.draftedTokens &&
                  fullBase.outputTokens == fullParked.outputTokens,
              "greedy continuation fixture did not accept the full draft block");
      const auto fullState = state(*runtime, storage, 1);
      require(fullState.metadata.pendingRows > 1 && runtime->parkDecodedRequest(602).granted(),
              "full-acceptance state did not park multiple retained rows");
      runtime->beginColdRequest(intruder.modelView(), 1);
      static_cast<void>(step(*runtime, 303, 0, intruderPages, prompt));
      static_cast<void>(step(*runtime, 303, prompt.size(), intruderPages));
      const auto fullRestored = runtime->restoreParkedRequest(602);
      require(fullRestored.granted() && *fullRestored.lane == 2,
              "full-acceptance continuation did not rebind");
      equal(fullState, state(*runtime, storage, 2));
      const uint64_t fullPosition = prompt.size() + fullBase.outputTokens.size();
      const auto afterBase = step(*runtime, 601, fullPosition, baselinePages);
      const auto afterParked = step(*runtime, 602, fullPosition, parkedPages);
      require(afterBase.outputTokens == afterParked.outputTokens &&
                  afterBase.acceptedDraftTokens == afterParked.acceptedDraftTokens,
              "full-acceptance continuation changed output after restoration");
      runtime->end(601);
      runtime->end(602);
      runtime->end(303);
      const uint32_t vocabulary = std::visit([](const auto &layout) { return layout.vocabularySize; },
                                               config.model.target);
      uint32_t partialCases = 0;
      for (uint32_t accepted = 1; accepted < model::ExecutionLimits::draftProposalTokens; ++accepted) {
        for (const auto *pages : {&baselinePages, &parkedPages, &intruderPages})
          for (const uint32_t page : *pages) pool.clearPage(page);
        auto maskedBase = input(701), maskedParked = input(702);
        maskedBase.flags = maskedParked.flags = 0;
        maskedBase.sampling = maskedParked.sampling = {};
        maskedBase.constraint = maskedParked.constraint = ConstraintMode::TokenMask;
        runtime->beginColdRequest(maskedBase.modelView(), 0);
        runtime->beginColdRequest(maskedParked.modelView(), 1);
        require(step(*runtime, 701, 0, baselinePages, prompt).nextDecodeStage == DecodeStage::ApplyInitialMask &&
                    step(*runtime, 702, 0, parkedPages, prompt).nextDecodeStage == DecodeStage::ApplyInitialMask,
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
        require(saved.metadata.pendingRows == accepted + 1 && runtime->parkDecodedRequest(702).granted(),
                "intermediate pending state was not parked exactly");
        runtime->beginColdRequest(intruder.modelView(), 1);
        static_cast<void>(step(*runtime, 303, 0, intruderPages, prompt));
        static_cast<void>(step(*runtime, 303, prompt.size(), intruderPages));
        const auto restored = runtime->restoreParkedRequest(702);
        require(restored.granted() && *restored.lane == 2, "masked continuation did not rebind");
        equal(saved, state(*runtime, storage, 2));
        const uint64_t next = prompt.size() + a.outputTokens.size();
        const auto afterA = partialStep(*runtime, resources->backend(), 701, next, baselinePages,
                                        vocabulary, accepted);
        const auto afterB = partialStep(*runtime, resources->backend(), 702, next, parkedPages,
                                        vocabulary, accepted);
        require(afterA.outputTokens == afterB.outputTokens &&
                    afterA.acceptedDraftTokens == afterB.acceptedDraftTokens,
                "partial-acceptance stream changed after parking");
        runtime->end(701);
        runtime->end(702);
        runtime->end(303);
        ++partialCases;
      }
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
        require(saved.metadata.pendingRows == 0 && runtime->parkDecodedRequest(802).granted(),
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
        for (uint32_t cycle = 0; cycle < 3; ++cycle) {
          const auto a = step(*runtime, 801, next, baselinePages);
          const auto b = step(*runtime, 802, next, parkedPages);
          require(a.outputTokens == b.outputTokens && a.acceptedDraftTokens == b.acceptedDraftTokens &&
                      a.outputTokensWithoutKv == b.outputTokensWithoutKv,
                  "thinking budget or end sequence changed after parking");
          next += a.outputTokens.size() - a.outputTokensWithoutKv;
          tokens.insert(tokens.end(), a.outputTokens.begin(), a.outputTokens.end());
        }
        const std::array<uint32_t, 2> end{2, 3};
        require(std::search(tokens.begin(), tokens.end(), end.begin(), end.end()) != tokens.end() &&
                    std::count(tokens.begin(), tokens.end(), 2) == 1 &&
                    std::count(tokens.begin(), tokens.end(), 3) == 1,
                "thinking fixture lost, duplicated or reordered its end sequence");
        runtime->end(801);
        runtime->end(802);
        runtime->end(303);
        ++thinkingCases;
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
                << " thinking_cases=" << thinkingCases << '\n';
    }
  }
}
} // namespace
int main(int argc, const char **argv) {
  if (argc != 2) return EXIT_FAILURE;
  @autoreleasepool {
    try {
      run(argv[1]);
      return EXIT_SUCCESS;
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return EXIT_FAILURE;
    }
  }
}
