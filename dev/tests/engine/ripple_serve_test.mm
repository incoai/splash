// The real model's first served request: the loaded Ripple-65B runs through
// the model::Runtime the engine serves through - its prefill chunks, then its
// decode cycles, in which the DFlash2 draft proposes and the target verifies
// - and the request's tokens, timing and acceptance counts are the report.
//
// usage: ripple-serve-test METALLIB MODEL_ROOT TOKENS_FILE NEW_TOKENS
#include "TestChecks.hpp"
#include "engine/MemoryGovernor.hpp"
#include "model/ModelFactory.hpp"
#include "engine/Types.hpp"
#include "model/QwenState.hpp"
#include "model/Runtime.hpp"

using namespace splash;

#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using splash::metal::MetalBackend;
using splash::model::LoadedModel;
using splash::model::Runtime;
using splash::test::require;

constexpr uint32_t kLayers = 80;
constexpr uint32_t kHidden = 5120;
constexpr uint32_t kVocab = 100352;
constexpr float kEmbedScale = 71.5541763305664F;
constexpr uint32_t kBos = 100257;

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 5) throw std::invalid_argument("usage: ripple-serve-test METALLIB MODEL_ROOT TOKENS_FILE NEW_TOKENS");
    MetalBackend backend(argv[1]);
    const std::filesystem::path root(argv[2]);
    std::vector<uint32_t> tokens{kBos};
    uint64_t revision = 0;
    {
      std::ifstream in(argv[3]);
      uint32_t id = 0;
      while (in >> id) tokens.push_back(id);
    }
    const uint32_t budget = static_cast<uint32_t>(std::stoul(argv[4]));

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

        Runtime runtime(model::RuntimeContext{backend, model, kv, states, operators, nullptr, nullptr, nullptr,
                                                 splash::model::VerifyRows::All});

    engine::EngineRequest request;
    request.id = 1;
    request.prompt = tokens;
    request.maxNewTokens = budget;
    request.sampling = {};
        runtime.beginColdRequest(request.modelView(), 0);

    const std::vector<uint32_t> pageTable(pages);
    const auto began = std::chrono::steady_clock::now();
    // The prompt's chunks: the runtime's prefill group bounds each.
    uint32_t position = 0;
    while (position < tokens.size()) {
      const uint32_t count = std::min<uint32_t>(tokens.size() - position, runtime.prefillGroupRows(false));
      BatchPlan plan{.kind = WorkKind::Prefill,
                            .items = {{1, count, 0}},
                            .decodeStage = DecodeStage::Regular};
      ModelBatchItem item{1, position, count, pageTable};
      item.pageTableRevision = ++revision;
      item.pageTableFirstChanged = 0;
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
    while (!output.empty() ? output.back() != 100265 && output.back() != kBos && generated < budget
                           : generated < budget) {
      if (!output.empty() && (output.back() == 100265 || output.back() == kBos)) break;
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
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
    std::cout << "generated " << generated << " tokens in " << decodeSeconds << " s of decode ("
              << (decodeSeconds > 0 ? generated / decodeSeconds : 0.0) << " tok/s), "
              << drafted << " drafted, " << accepted << " accepted ("
              << (drafted ? 100.0 * accepted / drafted : 0.0) << "%)\n";
    std::ofstream ids(argv[3] + std::string(".serve"), std::ios::binary);
    for (const uint32_t token : tokens) ids << token << '\n';
    for (const uint32_t token : output) ids << token << '\n';
    std::cout << "ripple-serve-test: PASS (" << seconds << " s total)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ripple-serve-test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
