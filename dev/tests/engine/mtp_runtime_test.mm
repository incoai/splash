// Synthetic Flash-Next weights through Runtime, its Metal commands and state
// storage. No installed model or fake executor is used.
#include "GgufFormatReference.hpp"
#include "TestChecks.hpp"
#include "engine/Types.hpp"
#include "metal/BackendInstrumentation.hpp"
#include "model/Qwen4DraftVocabulary.hpp"
#include "model/QwenState.hpp"
#include "model/Runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <unordered_map>

using namespace splash;
using namespace splash::model;
using splash::test::require;

namespace {
constexpr uint32_t hidden = 256, vocabulary = 512;

std::vector<uint32_t> contextTokens() {
  std::vector<uint32_t> tokens;
  for (uint32_t token = 2; token < vocabulary && tokens.size() < 3; ++token)
    if (!inQwen4DraftVocabulary(token)) tokens.push_back(token);
  require(tokens.size() == 3, "fixture has three outside-vocabulary context tokens");
  return tokens;
}

struct Fixture {
  metal::MetalBackend backend;
  bool contextWeights;
  LoadedModel model;
  ops::ExecutionPlans operators;
  std::unique_ptr<kv::PageStorage> pages;
  std::unique_ptr<QwenStateStorage> states;
  std::unique_ptr<Runtime> runtime;

  metal::MetalBuffer bytes(const void *data, uint64_t size) {
    auto buffer = backend.allocateBuffer(size, metal::BufferStorage::Shared, "mtp-runtime-fixture");
    std::memcpy(buffer.contents(), data, size);
    return buffer;
  }
  template<class T> metal::MetalBuffer upload(const std::vector<T> &values) {
    return bytes(values.data(), values.size() * sizeof(T));
  }
  ops::NormWeights norm(uint32_t width) { return {upload(std::vector<float>(width, 1.0F)), true}; }
  ops::Projection floats(uint32_t n, uint32_t k) {
    return {n, k, {{ops::QuantizedSegment::floats(n, k, upload(std::vector<float>(uint64_t{n} * k)))}}};
  }
  ops::QuantizedSegment quantized(uint32_t n, uint32_t k, bool head = false) {
    // Q8_0's exact +/-1 rows, all other rows zero, repacked by the shared reference.
    std::vector<uint8_t> native(uint64_t{n} * k / 32 * 34);
    if (head) {
      std::vector<std::pair<uint32_t, int>> rows{{0, 1}, {1, -1}};
      if (contextWeights && n == vocabulary) {
        const auto tokens = contextTokens();
        rows.insert(rows.end(), {{tokens[0], 2}, {tokens[1], 3}, {tokens[2], -4}});
      }
      for (const auto &[row, code] : rows)
        for (uint32_t group = 0; group < k / 32; ++group) {
          uint8_t *block = native.data() + (uint64_t{row} * k / 32 + group) * 34;
          const uint16_t scale = gguf_reference::f2h(1.0F);
          std::memcpy(block, &scale, 2);
          std::memset(block + 2, static_cast<uint8_t>(code), 32);
        }
    }
    const auto planes = gguf_reference::repack(gguf_reference::Q80, native, n, k, nullptr);
    return ops::QuantizedSegment::planes(GGUF_FMT_Q80, n, k, upload(planes.w0), {}, upload(planes.meta));
  }
  Qwen4HyperWeights hyper(bool inject = true) {
    return {norm(4 * hidden), floats(256, 4 * hidden), floats(4 * hidden, 256),
            inject ? upload(std::vector<float>(4 * 4 * hidden)) : metal::MetalBuffer{}};
  }
  ops::BlockMoeWeights moe() {
    ops::BlockExpertProjection forward{quantized(16 * 256, hidden), quantized(256, hidden)};
    ops::BlockExpertProjection down{quantized(16 * hidden, 256), quantized(hidden, 256)};
    return {floats(16, hidden).blocks().segments.front(), floats(1, hidden).blocks().segments.front(),
            forward, forward, down};
  }
  explicit Fixture(const char *library, bool adaptive = true, bool withContext = false)
      : backend(library), contextWeights(withContext), operators(backend.capabilities()) {
    setenv("SPLASH_MTP_ADAPTIVE", adaptive ? "1" : "0", 1);
    setenv("SPLASH_MTP_DRAFTS", "7", 1);
    setenv("SPLASH_MTP_DEPTH", "cost", 1);
    Qwen4ExpWeights weights;
    auto &layout = weights.layout;
    layout.maximumContextTokens = 4096;
    layout.layers = 1;
    layout.hiddenSize = hidden;
    layout.vocabularySize = vocabulary;
    // Retain Flash-Next's compiled attention/GDN head geometry; shrink only
    // the hidden width, layer count and experts.
    layout.fullAttentionPeriod = 4;
    layout.experts = 16;
    layout.expertsPerToken = 4;
    layout.expertIntermediateSize = 256;
    layout.hyperRank = 256;
    layout.hiddenCaptureLayers = {0};
    layout.maskToken = 511;
    layout.stopTokens = {509, 510};
    // One zero PLE head keeps hashing and history commits on their real path.
    layout.pleNgram = 2;
    layout.pleHeadsPerNgram = 1;
    layout.pleHeadDimension = 256;
    layout.pleConvolutionTaps = 2;
    layout.pleLayer = 0;
    weights.pleHash = {{1, 1, 1, 1}, {0}, {1}, 1};
    layout.mtpLayers = 1;
    layout.mtpDraftTokens = 256;
    std::vector<uint16_t> embeddings(uint64_t{vocabulary} * hidden, 0x3f80);
    // The negative lane emits token 1 forever; the positive lane token 0.
    std::vector<uint32_t> negative{1, 11};
    if (contextWeights) negative.push_back(contextTokens()[2]);
    for (uint32_t token : negative)
      std::fill_n(embeddings.begin() + token * hidden, hidden, 0xbf80);
    weights.tokenEmbedding = {vocabulary, hidden, {upload(embeddings), GGUF_FMT_BF16}};
    weights.logitsProjection = {vocabulary, hidden, {{quantized(vocabulary, hidden, true)}}};
    weights.logitsProjection.destination = ops::FloatOutput::Float32;
    Qwen4ExpLayerWeights trunk;
    trunk.mixerHyper = hyper();
    trunk.ffnHyper = hyper();
    trunk.mixer = QwenGdnWeights{floats(layout.packedGdnWidth, hidden),
                               upload(std::vector<float>(4 * layout.convolutionDimension)),
                               upload(std::vector<float>(layout.gdnValueHeads)),
                               upload(std::vector<float>(layout.gdnValueHeads)),
                               norm(layout.gdnHeadDimension), floats(hidden, layout.attentionWidth)};
    trunk.ffn = moe();
    weights.pleTable = {1, 256, {upload(std::vector<uint16_t>(256)), GGUF_FMT_BF16}};
    trunk.ple = Qwen4PleWeights{floats(4 * hidden, 256), floats(hidden, 256),
                               norm(4 * hidden), norm(4 * hidden), norm(4 * hidden),
                               upload(std::vector<float>(2 * 4 * hidden))};
    weights.layers.push_back(trunk);
    weights.finalHyper = hyper(false);
    Qwen4MtpWeights mtp;
    mtp.block.mixerHyper = hyper();
    mtp.block.ffnHyper = hyper();
    mtp.block.mixer = QwenAttentionWeights{floats(layout.packedFullWidth, hidden), norm(256), norm(256),
                                           floats(hidden, layout.attentionWidth)};
    mtp.block.indexer = Qwen4IndexerWeights{floats(512, hidden), floats(128, hidden), norm(128), norm(128)};
    mtp.block.ffn = moe();
    mtp.embedHidden = floats(hidden, 2 * hidden); // all-zero MTP streams propose token 0
    if (contextWeights) {
      // Preserve the token embedding's sign; only copied context head rows
      // can propose the target's outside-vocabulary winner.
      std::vector<float> identity(uint64_t{hidden} * 2 * hidden);
      for (uint32_t i = 0; i < hidden; ++i) identity[i * 2 * hidden + i] = 1.0F;
      mtp.embedHidden = {hidden, 2 * hidden, {{ops::QuantizedSegment::floats(hidden, 2 * hidden, upload(identity))}}};
    }
    mtp.embedNorm = norm(hidden);
    mtp.hiddenNorm = norm(4 * hidden);
    mtp.head = hyper(false);
    std::vector<uint32_t> tokens;
    for (uint32_t token = 0; token < vocabulary && tokens.size() < 256; ++token)
      if (inQwen4DraftVocabulary(token)) tokens.push_back(token);
    require(tokens.size() == 256 && tokens[0] == 0 && tokens[1] == 1,
            "fixture draft head maps static vocabulary rows");
    mtp.draftHead = Qwen4DraftHead{{256, hidden, {{quantized(256, hidden, true)}}}, upload(tokens)};
    mtp.draftHead->logits.destination = ops::FloatOutput::Float32;
    weights.mtp = mtp;
    model.target = weights;
    model.draft.layout = nullDraftLayout(hidden, vocabulary, hidden);
    model.descriptor.target = layout;
    model.descriptor.draftModel = false;
    model.descriptor.visionSource = VisionSource::None;
    model.descriptor.targetKvLayout = layout.kvLayout();
    model.descriptor.stateLayout = {layout.gdnStateLayout(), model.draft.layout.stateLayout()};
    model.descriptor.capabilities.maximumContextTokens = layout.maximumContextTokens;
    const metal::AllocationAdmission admit = [](uint64_t, const std::function<void()> &allocate) {
      allocate(); return metal::AllocationResult{};
    };
    const auto kvLayout = model.targetKvLayout(kv::Format::Int8);
    const uint32_t extent = kvLayout.minimumExtentPages();
    const uint32_t pageCount = ((64 + extent - 1) / extent) * extent;
    pages = std::make_unique<kv::PageStorage>(backend, admit, kvLayout, pageCount, extent);
    for (uint32_t i = 0; i < pageCount / extent; ++i)
      require(bool(pages->allocateExtent(i)), "fixture extent allocation");
    states = std::make_unique<QwenStateStorage>(backend, admit, model.stateLayout(ops::GdnStateFormat::Float32), nullptr);
    RuntimeContext context{backend, model, *pages, *states, operators};
    runtime = std::make_unique<Runtime>(context);
  }
  engine::EngineRequest request(uint64_t id, std::vector<uint32_t> prompt, bool constrained = false) {
    engine::EngineRequest result;
    result.id = id; result.prompt = std::move(prompt); result.maxNewTokens = 2048;
    result.constraint = constrained ? ConstraintMode::TokenMask : ConstraintMode::None;
    return result;
  }
  ModelStepResult prefill(const engine::EngineRequest &request, uint32_t lane) {
    runtime->beginColdRequest(request.modelView(), lane);
    return replay(request, lane);
  }
  ModelStepResult replay(const engine::EngineRequest &request, uint32_t lane) {
    const std::array<uint32_t, 1> table{lane};
    const uint32_t count = static_cast<uint32_t>(request.prompt.size());
    BatchPlan plan{.kind = WorkKind::Prefill, .items = {{request.id, count}}};
    ModelBatchItem item{request.id, 0, count, table}; item.inputTokens = request.prompt;
    item.pageTableRevision = ++revisions[request.id];
    return runtime->submit(plan, std::span(&item, 1), {})->wait().front();
  }
  std::unordered_map<uint64_t, uint64_t> revisions;
  std::array<std::array<uint32_t, 16>, 4> tables{};
  std::unique_ptr<ModelBatchTicket> submit(std::span<const uint64_t> ids, std::span<const uint32_t> lanes,
                                          bool constrained = false) {
    BatchPlan plan{.kind = WorkKind::Decode, .constrained = constrained};
    std::vector<ModelBatchItem> items;
    for (uint32_t i = 0; i < ids.size(); ++i) {
      plan.items.push_back({ids[i], 0});
      // Disjoint page ranges, enough for the acceptance-training history.
      for (uint32_t page = 0; page < 16; ++page) tables[i][page] = lanes[i] + page * 4;
      ModelBatchItem item{ids[i], states->metadata(lanes[i]).lengths.targetTokens, 0, tables[i]};
      item.pageTableRevision = ++revisions[ids[i]];
      items.push_back(item);
    }
    return runtime->submit(plan, items, [] {});
  }
  ModelStepResult decode(uint64_t id, uint32_t lane) {
    return submit(std::span(&id, 1), std::span(&lane, 1))->wait().front();
  }
};

uint32_t copyDispatches(Fixture &f) {
  const auto profile = metal::BackendInstrumentation::takeDispatchProfile(f.backend);
  return std::count_if(profile.begin(), profile.end(), [](const auto &d) {
    return d.pipelineName == "qwen4_copy_head_rows";
  });
}

void thinkingCutoff(const char *library) {
  for (const float temperature : {0.0F, 0.7F}) {
    Fixture f(library, false);
    const std::array<uint64_t, 2> ids{41, 42};
    const std::array<uint32_t, 2> lanes{3, 1};
    auto cut = f.request(ids[0], {10});
    cut.sampling.temperature = temperature;
    cut.sampling.topK = 1;
    cut.thinkingBudget = ThinkingBudget{1, {3, 4, 5}, 6};
    f.prefill(cut, lanes[0]); // selection consumes the last thinking token
    f.prefill(f.request(ids[1], {10}), lanes[1]);
    const auto results = f.submit(ids, lanes)->wait();
    const std::vector<uint32_t> prefix{0, 3, 4, 5};
    require(results[0].failure.empty() && results[0].outputTokens.size() >= prefix.size() &&
                std::equal(prefix.begin(), prefix.end(), results[0].outputTokens.begin()),
            "Runtime row-zero cutoff replaces MTP proposals before verify");
    require(results[0].draftedTokens == 4 && results[0].acceptedDraftTokens == 4,
            "Runtime forced end proposals are excluded from draft counts");
    require(results[1].failure.empty() && results[1].draftedTokens == 7 &&
                results[1].acceptedDraftTokens == 7 &&
                std::ranges::all_of(results[1].outputTokens, [](uint32_t t) { return t == 0; }),
            "Runtime thinking cutoff leaves the other lane's proposals intact");
    f.runtime->end(ids[0]); f.runtime->end(ids[1]);
  }
  std::cout << "runtime_thinking_cutoff=PASS\n";
}

void contextLifetime(const char *library) {
  Fixture f(library, false, true);
  const auto tokens = contextTokens();
  metal::BackendInstrumentation::setDispatchProfiling(f.backend, true);
  auto old = f.request(7, {tokens[0], tokens[1]});
  f.prefill(old, 0);
  require(f.decode(7, 0).acceptedDraftTokens == 7, "Runtime drafts from copied outside-vocabulary rows");
  require(copyDispatches(f) == 1, "Runtime initially copies context rows");
  f.decode(7, 0);
  require(copyDispatches(f) == 0, "Runtime reuses synchronized context rows");
  // Engine cancellation drains its in-flight ticket, then calls Runtime::end.
  auto cancelled = f.submit(std::array<uint64_t, 1>{7}, std::array<uint32_t, 1>{0});
  static_cast<void>(cancelled->wait());
  require(copyDispatches(f) == 0, "Runtime cancellation drains synchronized work");
  f.runtime->end(7);
  auto fresh = f.request(7, {tokens[2]});
  f.prefill(fresh, 0);
  const auto freshDecoded = f.decode(7, 0);
  require(freshDecoded.acceptedDraftTokens == 7, "Runtime uses the new request context weights and token map");
  require(copyDispatches(f) == 1, "Runtime cancellation and same-ID shrink recopies all context rows");
  f.runtime->suspend(7);
  // Occupy the released lane so production resume must migrate to lane 1.
  auto occupant = f.request(8, {tokens[0]}); f.prefill(occupant, 0);
  auto replay = fresh;
  replay.prompt.insert(replay.prompt.end(), freshDecoded.outputTokens.begin(), freshDecoded.outputTokens.end());
  const auto resumed = f.runtime->resume(replay.modelView());
  require(resumed.granted() && *resumed.lane != 0, "Runtime resumes on a different state lane");
  f.runtime->setDraftContextPlan(7, planDraftContext(0, replay.prompt.size(), {}));
  f.replay(replay, *resumed.lane);
  require(f.decode(7, *resumed.lane).acceptedDraftTokens == 7, "Runtime migrated context proposals match the target");
  require(copyDispatches(f) == 1, "Runtime migrated context is copied onto its new lane");
  f.runtime->end(7); f.runtime->end(8);
  // Also migrate the same ID back onto its old lane: suspend's release hook
  // must invalidate that lane even when its next owner has the same ID.
  f.prefill(fresh, 0);
  require(f.decode(7, 0).acceptedDraftTokens == 7, "Runtime uses the new request context weights and token map");
  require(copyDispatches(f) == 1, "Runtime release invalidates the original state lane");
  f.runtime->end(7);
  std::cout << "runtime_context_cancel_migration=PASS\n";
}

void constrainedTransitions(const char *library) {
  Fixture f(library, false, true);
  const auto context = contextTokens();
  auto request = f.request(9, {context[0], context[1]}, true);
  auto initial = f.prefill(request, 3);
  require(initial.nextDecodeStage == DecodeStage::ApplyInitialMask, "Runtime constrained prefill waits for initial mask");
  std::vector<uint32_t> mask(vocabulary / 32, ~0u);
  require(!f.runtime->provideMask(9, mask), "Runtime accepts initial mask");
  BatchPlan initialPlan{.kind = WorkKind::Decode, .constrained = true, .items = {{9, 0}},
                        .decodeStage = DecodeStage::ApplyInitialMask};
  const std::array<uint32_t, 1> pages{3};
  ModelBatchItem item{9, request.prompt.size(), 0, pages}; item.pageTableRevision = ++f.revisions[9];
  auto selected = f.runtime->submit(initialPlan, std::span(&item, 1), {})->wait();
  require(selected.front().nextDecodeStage == DecodeStage::Regular, "Runtime initial selection enters regular decode");
  auto ticket = f.submit(std::array<uint64_t, 1>{9}, std::array<uint32_t, 1>{3}, true);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  std::vector<ModelMaskRequest> requests;
  while (requests.empty() && std::chrono::steady_clock::now() < deadline) {
    requests = ticket->takeMaskRequests(); std::this_thread::yield();
  }
  require(requests.size() == 1 && requests[0].simulationTokens.size() == 8 && ticket->ownsMaskWait(9),
          "Runtime draft transitions to target forward and exposes constraint simulation");
  require(!ticket->ready(), "Runtime ticket waits for verify masks before commit");
  std::vector<uint32_t> masks(9 * vocabulary / 32, ~0u);
  require(!f.runtime->provideMask(9, masks), "Runtime accepts verify masks");
  while (!ticket->ready() && std::chrono::steady_clock::now() < deadline) {
    require(ticket->takeMaskRequests().empty(), "Runtime mask request is emitted once"); std::this_thread::yield();
  }
  require(ticket->ready(), "Runtime constrained forward/mask/commit completes");
  const auto result = ticket->wait().front();
  require(!ticket->ownsMaskWait(9) && result.nextDecodeStage == DecodeStage::Regular,
          "Runtime constrained commit clears its mask ownership");
  require(result.failure.empty() && result.draftedTokens == 7 && result.acceptedDraftTokens == 7,
          "Runtime constrained commit reports real retained proposals");
  // Cancel at mask wait as Engine does, drain the real ticket, then reuse the ID/lane.
  ticket = f.submit(std::array<uint64_t, 1>{9}, std::array<uint32_t, 1>{3}, true);
  requests.clear();
  while (requests.empty() && std::chrono::steady_clock::now() < deadline) requests = ticket->takeMaskRequests();
  require(requests.size() == 1, "Runtime cancellation reaches mask wait");
  ticket->abandonMask(9);
  while (!ticket->ready() && std::chrono::steady_clock::now() < deadline) static_cast<void>(ticket->takeMaskRequests());
  require(ticket->ready(), "Runtime abandoned mask drains its commit");
  static_cast<void>(ticket->wait()); f.runtime->end(9);
  metal::BackendInstrumentation::setDispatchProfiling(f.backend, true);
  f.prefill(f.request(9, {context[2]}), 3);
  require(f.decode(9, 3).failure.empty() && copyDispatches(f) == 1,
          "Runtime smaller context recopied after real constrained-ticket cancellation");
  f.runtime->end(9);
  std::cout << "runtime_constrained_transitions=PASS\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: mtp-runtime METALLIB");
    thinkingCutoff(argv[1]); contextLifetime(argv[1]); constrainedTransitions(argv[1]);
  } catch (const std::exception &error) {
    std::cerr << "mtp-runtime: FAIL: " << error.what() << '\n'; return 1;
  }
  std::cout << "MTP Runtime tests passed\n";
}
