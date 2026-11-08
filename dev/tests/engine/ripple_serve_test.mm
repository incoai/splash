// The real model's first served request: Ripple-65B-Preview (the release's
// AD-Q4_K_M GGUF and DFlash2 draft) loaded through the engine's real
// machinery, a prompt prefilled, and tokens generated through the model
// runtime's decode cycles — the draft proposes, the target verifies. With
// SPLASH_SERVE_BENCH the harness benchmarks instead: prefill chunks of
// rising length and decode cycles, each timed alone, so the prefill's and
// the cycle's costs split from one another.
//
// usage: ripple-serve-test METALLIB MODEL_ROOT TOKENS_FILE NEW_TOKENS
#include "TestChecks.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Types.hpp"
#include "model/QwenState.hpp"
#include "model/Runtime.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

using namespace splash;
using splash::metal::MetalBackend;
using splash::model::LoadedModel;
using splash::model::Runtime;
using splash::BatchPlan;
using splash::ModelBatchItem;
using splash::ModelStepResult;
using splash::test::require;

namespace {

constexpr uint32_t kLayers = 80;
constexpr uint32_t kHidden = 5120;
constexpr uint32_t kVocab = 100352;
constexpr float kEmbedScale = 71.5541763305664F;
constexpr uint32_t kBos = 100257;

splash::model::RippleLayout servedLayout() {
  splash::model::RippleLayout layout;
  layout.nativeContextTokens = 262144;
  layout.layers = kLayers;
  layout.hiddenSize = kHidden;
  layout.vocabularySize = kVocab;
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
  layout.stopTokens = {100265, kBos};
  uint64_t sliding[2] = {}, full[2] = {};
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    (layer % 5 == 4 ? full : sliding)[layer / 64] |= 1ULL << (layer % 64);
  layout.slidingLayers = {sliding[0], sliding[1]};
  layout.unrotatedLayers = {full[0], full[1]};
  layout.fullAttentionPeriod = 1;
  layout.slidingWindow = 1024;
  layout.ffnKind = splash::model::QwenFfnKind::Dense;
  layout.packedFullWidth = 2 * (80 * 128) + 2 * (8 * 128);
  return layout;
}

void runBench(splash::metal::MetalBackend &backend, LoadedModel &model, const splash::ops::ExecutionPlans &operators,
              kv::PageStorage &kv, splash::model::QwenStateStorage &states,
              const std::vector<uint32_t> &tokens, const std::vector<uint32_t> &pageTable,
              uint32_t budget) {
  using splash::test::require;
  Runtime runtime(model::RuntimeContext{backend, model, kv, states, operators, nullptr, nullptr, nullptr,
                                        splash::model::VerifyRows::All});
  std::cout << "bench: weights "
            << (model.targetActualAllocatedBytes() + model.draft.actualAllocatedBytes) / 1e9 << " GB\n";
  // Prefill chunks of rising length, each its own request and lane, ended
  // after its one chunk.
  uint32_t lane = 1;
      for (const uint32_t count : {32u, 128u, 512u, 2048u}) {
    const uint64_t id = 100 + count;
    engine::EngineRequest warm;
    warm.id = id;
    warm.prompt.assign(count, kBos);
    warm.maxNewTokens = 1;
        runtime.beginColdRequest(warm.modelView(), 1 + (lane++ % 3));
    std::vector<uint32_t> warmTokens(count, kBos);
    const auto began = std::chrono::steady_clock::now();
    BatchPlan plan{.kind = WorkKind::Prefill, .items = {{id, count, 0}}, .decodeStage = DecodeStage::Regular};
    ModelBatchItem item{id, 0, count, pageTable};
    item.pageTableRevision = 1;
    item.inputTokens = warmTokens;
    const std::vector<ModelStepResult> result =
        runtime.prefill(plan, std::span<const ModelBatchItem>(&item, 1));
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
    require(result.size() == 1 && result[0].requestId == id, "the bench prefill failed");
    std::cout << "bench prefill " << count << " tokens: " << seconds << " s ("
              << count / seconds << " tok/s)\n";
    runtime.end(id);
  }
  // The decode run: its prompt prefilled, then per-cycle timings.
  const uint64_t id = 1;
  engine::EngineRequest run;
  run.id = id;
  run.prompt = tokens;
  run.maxNewTokens = budget;
  runtime.beginColdRequest(run.modelView(), 0);
  {
    BatchPlan plan{.kind = WorkKind::Prefill, .items = {{id, static_cast<uint32_t>(tokens.size()), 0}},
                   .decodeStage = DecodeStage::Regular};
    ModelBatchItem item{id, 0, static_cast<uint32_t>(tokens.size()), pageTable};
    item.pageTableRevision = 1;
    item.inputTokens = tokens;
    const std::vector<ModelStepResult> result =
        runtime.prefill(plan, std::span<const ModelBatchItem>(&item, 1));
    require(result.size() == 1 && !result[0].finished, "the bench run's prefill ended early");
  }
  uint32_t generated = 0, drafted = 0, accepted = 0, cycles = 0;
  std::vector<double> cycleSeconds;
  while (generated < budget) {
    BatchPlan plan{.kind = WorkKind::Decode, .items = {{id, 0, 0}}, .decodeStage = DecodeStage::Regular};
    ModelBatchItem item{id, static_cast<uint64_t>(tokens.size()) + generated, 0, pageTable};
    item.pageTableRevision = 2;
    const auto cycleBegan = std::chrono::steady_clock::now();
    const std::vector<ModelStepResult> result =
        runtime.decode(plan, std::span<const ModelBatchItem>(&item, 1));
    cycleSeconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - cycleBegan).count());
    require(result.size() == 1 && result[0].requestId == id && result[0].failure.empty(),
            "the bench decode failed: " + result[0].failure);
    generated += static_cast<uint32_t>(result[0].outputTokens.size());
    drafted += result[0].draftedTokens;
    accepted += result[0].acceptedDraftTokens;
    ++cycles;
    if (result[0].finished) break;
  }
  double total = 0, best = cycleSeconds.empty() ? 0 : cycleSeconds.front(), worst = 0;
  for (double second : cycleSeconds) {
    total += second;
    best = std::min(best, second);
    worst = std::max(worst, second);
  }
  std::cout << "bench decode " << cycles << " cycles, " << generated << " tokens in " << total
            << " s: mean cycle " << (cycles ? total / cycles : 0.0) << " s, best " << best << ", worst " << worst
            << ", " << (total > 0 ? generated / total : 0.0) << " tok/s, "
            << drafted << " drafted, " << accepted << " accepted ("
            << (drafted ? 100.0 * accepted / drafted : 0.0) << "%)\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 5) throw std::invalid_argument("usage: ripple-serve-test METALLIB MODEL_ROOT TOKENS_FILE NEW_TOKENS");
    MetalBackend backend(argv[1]);
    const std::filesystem::path root(argv[2]);
    std::vector<uint32_t> tokens{kBos};
    {
      std::ifstream in(argv[3]);
      uint32_t id = 0;
      while (in >> id) tokens.push_back(id);
    }
    const uint32_t budget = static_cast<uint32_t>(std::stoul(argv[4]));

    const splash::model::RippleLayout layout = servedLayout();
    const std::vector<uint32_t> draftLayers{1, 20, 39, 58, 77};
    const splash::model::DFlashDraftLayout draft = splash::model::draftLayout(
        kHidden, kVocab, 5, 40960, 10000.0F, draftLayers, SPLASH_DRAFT_VARIANTS[1]);
    const splash::model::ModelDescriptor descriptor = splash::model::makeModelDescriptor(
        "ripple-65b-real", layout, draft, {.outputHiddenSize = kHidden},
        splash::model::TargetSource::Gguf, splash::model::VisionSource::None);
    require(descriptor.valid(), "the served model's descriptor is not valid");

    const auto loadBegan = std::chrono::steady_clock::now();
    LoadedModel model = splash::model::loadModel(backend, root, descriptor);
    const double loadSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - loadBegan).count();
    std::cout << "loaded in " << loadSeconds << " s\n";

    const splash::ops::ExecutionPlans operators(backend.capabilities());
    engine::MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
                                    engine::queryHostMemory, 0);
    const kv::Layout kvLayout = model.targetKvLayout(kv::Format::BFloat16);
    const uint32_t extent = kvLayout.minimumExtentPages();
    // A page of 32 tokens; the request's pages the runtime's tables name.
    const uint32_t pages = 64;
    kv::PageStorage kv(backend, governor.allocationAdmission(), kvLayout, pages / extent * extent, extent);
    for (uint32_t e = 0; e < pages / extent; ++e)
      require(static_cast<bool>(kv.allocateExtent(e)), "the served run's extent did not allocate");
    splash::model::QwenStateStorage states(backend, governor.allocationAdmission(),
                                           model.stateLayout(splash::ops::GdnStateFormat::Float32), nullptr);

    const bool bench = getenv("SPLASH_SERVE_BENCH") != nullptr;
    const std::vector<uint32_t> pageTable(pages);
    if (bench) {
      runBench(backend, model, operators, kv, states, tokens, pageTable, budget);
      std::cout << "ripple-serve-test: PASS (bench)\n";
      return 0;
    }

    Runtime runtime(model::RuntimeContext{backend, model, kv, states, operators, nullptr, nullptr, nullptr,
                                          splash::model::VerifyRows::All});

    engine::EngineRequest request;
    request.id = 1;
    request.prompt = tokens;
    request.maxNewTokens = budget;
    // Sampling parameters from the environment, so a served prompt can
    // sample: SPLASH_SERVE_TEMPERATURE and SPLASH_SERVE_SEED.
    if (const char *temperature = getenv("SPLASH_SERVE_TEMPERATURE"))
      request.sampling.temperature = std::strtof(temperature, nullptr);
    if (const char *seed = getenv("SPLASH_SERVE_SEED"))
      request.sampling.seed = std::strtoull(seed, nullptr, 10);
    runtime.beginColdRequest(request.modelView(), 0);

    uint64_t revision = 0;
    const auto began = std::chrono::steady_clock::now();
    // The prompt's chunks: the runtime's prefill group bounds each.
    uint32_t position = 0;
    while (position < tokens.size()) {
      const uint32_t count = std::min<uint32_t>(tokens.size() - position, runtime.prefillGroupRows(false));
      BatchPlan plan{.kind = WorkKind::Prefill, .items = {{1, count, 0}}, .decodeStage = DecodeStage::Regular};
      ModelBatchItem item{1, position, count, pageTable};
      item.pageTableRevision = ++revision;
      item.inputTokens = std::span<const uint32_t>(tokens).subspan(position, count);
      const std::vector<ModelStepResult> result =
          runtime.prefill(plan, std::span<const ModelBatchItem>(&item, 1));
      require(result.size() == 1 && result[0].requestId == 1, "the served prefill's result is missing");
      if (result[0].finished) break;
      position += count;
    }
    const double prefillSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
    std::cout << "prefilled " << tokens.size() << " tokens in " << prefillSeconds << " s\n";

    // The decode cycles: the draft proposes, the target verifies, the result
    // reports what the request generated and what the draft's proposal did.
    uint32_t generated = 0, drafted = 0, accepted = 0;
    double decodeSeconds = 0.0;
    std::vector<uint32_t> output;
    while (generated < budget) {
      BatchPlan plan{.kind = WorkKind::Decode, .items = {{1, 0, 0}}, .decodeStage = DecodeStage::Regular};
      ModelBatchItem item{1, tokens.size() + generated, 0, pageTable};
      item.pageTableRevision = revision;
      const auto cycleBegan = std::chrono::steady_clock::now();
      const std::vector<ModelStepResult> result =
          runtime.decode(plan, std::span<const ModelBatchItem>(&item, 1));
      decodeSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - cycleBegan).count();
      require(result.size() == 1 && result[0].requestId == 1 && result[0].failure.empty(),
              "the served decode cycle failed: " + result[0].failure);
      output.insert(output.end(), result[0].outputTokens.begin(), result[0].outputTokens.end());
      generated += static_cast<uint32_t>(result[0].outputTokens.size());
      drafted += result[0].draftedTokens;
      accepted += result[0].acceptedDraftTokens;
      if (result[0].finished) break;
      if (!output.empty() && (output.back() == 100265 || output.back() == kBos)) break;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
    std::cout << "generated " << generated << " tokens in " << decodeSeconds << " s of decode ("
              << (decodeSeconds > 0 ? generated / decodeSeconds : 0.0) << " tok/s), "
              << drafted << " drafted, " << accepted << " accepted ("
              << (drafted ? 100.0 * accepted / drafted : 0.0) << "%)\n";
    std::ofstream ids(argv[3] + std::string(".serve"));
    for (const uint32_t token : tokens) ids << token << '\n';
    for (const uint32_t token : output) ids << token << '\n';
    std::cout << "ripple-serve-test: PASS (" << seconds << " s total)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ripple-serve-test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
