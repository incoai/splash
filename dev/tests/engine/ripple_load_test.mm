// A tiny Ripple model loaded through the full pipeline: the planner plans
// its GGUF's images, the preparation writes them, the loader reads them into
// the target's and the draft's weights, and the target constructs. The
// model is 2 layers of the family's 4+1 pattern at the family's attention
// geometry (80 query heads over 8 KV heads of 128) and hidden 256, its
// draft 1 layer at the shipped draft variant's geometry.
#include "TestChecks.hpp"
#include "model/ModelFactory.hpp"
#include "model/ModelDescriptor.hpp"

#import <Foundation/Foundation.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

using splash::model::loadModel;
using splash::model::RippleLayout;

// The tiny model's layout, whose attention geometry the Ripple variant's
// kernels serve and whose dimensions the fixture's GGUF states.
RippleLayout tinyLayout() {
  RippleLayout layout;
  layout.nativeContextTokens = 4096;
  layout.layers = 2;
  layout.hiddenSize = 256;
  layout.vocabularySize = 256;
  layout.packedFullWidth = 2 * 10240 + 1024 + 1024;
  layout.attentionWidth = 10240;
  layout.attentionQueryHeads = 80;
  layout.attentionKvHeads = 8;
  layout.attentionHeadDimension = 128;
  layout.rotaryPairs = 64;
  layout.rotaryTheta = 10000.0F;
  layout.slidingWindow = 33;
  layout.slidingLayers = {1, 0};   // layer 0 slides
  layout.unrotatedLayers = {2, 0}; // layer 1 is full (NoPE)
  layout.fullAttentionPeriod = 1;
  layout.ffnKind = splash::model::QwenFfnKind::Dense;
  layout.intermediateSize = 512;
  layout.embeddingScale = 16.0F;
  layout.swigluClamp = 6.0F;
  layout.valueResidual = true;
  layout.postFfnNorm = true;
  layout.maskToken = 200;
  layout.stopTokens = {1, 0};
  return layout;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: ripple-load-test METALLIB");
    splash::metal::MetalBackend backend(argv[1]);

    const splash::model::RippleLayout layout = tinyLayout();
    const std::vector<uint32_t> targetLayers{0};
    const splash::model::DFlashDraftLayout draft = splash::model::draftLayout(
        layout.hiddenSize, layout.vocabularySize, 1, 512, 10000.0F, targetLayers,
        SPLASH_DRAFT_VARIANTS[0]);
    const splash::model::ModelDescriptor descriptor = splash::model::makeModelDescriptor(
        "test/ripple-tiny", layout, draft,
        {.outputHiddenSize = layout.hiddenSize},
        splash::model::TargetSource::Gguf, splash::model::VisionSource::None);
    if (!descriptor.valid())
      throw std::runtime_error("the tiny model's descriptor is not valid");

    const std::filesystem::path root("dev/tests/fixtures/ripple-tiny");
    const splash::model::LoadedModel model = loadModel(backend, root, descriptor);
    if (model.draft.layout.window != 2048 || model.draft.layout.queryHeads != 32)
      throw std::runtime_error("the draft's layout is not the shipped variant's");
    const auto *ripple = std::get_if<splash::model::RippleLayout>(&model.descriptor.target);
    if (!ripple || model.descriptor.draft.hiddenSize != layout.hiddenSize)
      throw std::runtime_error("the loaded model's descriptor is not the family's");
    std::cout << "ripple-load-test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ripple-load-test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
