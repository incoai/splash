// Prepares a tiny dense or MoE affine checkpoint, or a DFlash2 draft
// checkpoint, twice (cold, then warm without conversion headroom) and prints
// each prepared image's SHA-256. Every image must also equal the
// independently serialized file in FIXTURE/expected.
//
//   affine-preparation METALLIB FIXTURE dense|moe|draft
#include "model/AffineTarget.hpp"
#include "model/DFlashDraft.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "model/QwenTargetLoader.hpp"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace splash;
namespace {

std::vector<uint8_t> fileBytes(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), {}};
}

// The dimensions both fixtures share: two layers (GDN, then full attention)
// of width 256, every draft capture reading the last.
template <class Layout> Layout tinyLayout() {
  Layout layout;
  layout.layers = 2;
  layout.hiddenSize = 256;
  layout.vocabularySize = 256;
  layout.packedGdnWidth = 1024;
  layout.packedFullWidth = 768;
  layout.convolutionDimension = 512;
  layout.gdnKeyHeads = 2;
  layout.gdnValueHeads = 4;
  layout.gdnHeadDimension = 64;
  layout.attentionWidth = 256;
  layout.attentionQueryHeads = 4;
  layout.attentionKvHeads = 2;
  layout.attentionHeadDimension = 64;
  layout.rotaryPairs = 8;
  layout.fullAttentionPeriod = 2;
  layout.hiddenCaptureLayers.fill(1);
  return layout;
}

// Prepares the fixture at root with Loader twice; open opens every file of
// the loader, in the order it plans them, through check.
template <class Loader, class Layout, class Open>
void prepare(metal::MetalBackend &backend, const std::filesystem::path &root, const Layout &layout, Open open) {
  const std::filesystem::path cache(std::getenv("SPLASH_WEIGHT_CACHE"));
  bool cold = true;
  const auto admitConversion = [&] { if (!cold) throw std::runtime_error("conversion forbidden on warm load"); };
  for (unsigned pass = 0; pass < 2; ++pass) {
    Loader loader(backend, root, layout, admitConversion);
    size_t opened = 0;
    const auto check = [&](model::WeightFile weights) {
      const auto &record = weights.record();
      // The model's disk check budgets weights(): it must be the files the
      // loader writes, in order.
      if (opened == loader.weights().size())
        throw std::runtime_error("the loader writes a file it did not plan: " + record.relativePath);
      const model::PreparedWeight &planned = loader.weights()[opened++];
      if (record.contentIdentity != planned.key || record.declaredBytes != planned.bytes ||
          record.relativePath != planned.component)
        throw std::runtime_error("the loader's planned weights differ from its " + record.relativePath);
      const auto prepared = fileBytes(cache / record.contentIdentity / "weights");
      if (prepared.size() != record.declaredBytes) throw std::runtime_error("wrong image size");
      if (prepared != fileBytes(root / "expected" / std::filesystem::path(record.relativePath).filename()))
        throw std::runtime_error("affine fixture differs: " + record.relativePath);
      static_cast<void>(weights.section(record.declaredBytes - model::kWeightFileAlignment, {}));
      weights.finish();
      if (pass == 0) std::cout << "prepared " << record.relativePath << ' ' << model::weightDigest(prepared) << '\n';
    };
    open(loader, check);
    if (opened != loader.weights().size()) throw std::runtime_error("the loader plans files it never writes");
    cold = false;
  }
}

// Every image of the target fixtures, in plan order.
void openTarget(model::AffineTargetLoader &loader, const auto &check) {
  check(loader.layer(0));
  check(loader.layer(1));
  check(loader.head());
  check(loader.embedding());
}

// The draft fixture: two layers of width 256, one KV head.
model::DFlashDraftLayout tinyDraftLayout() {
  model::DFlashDraftLayout layout;
  layout.layers = 2;
  layout.hiddenSize = 256;
  layout.vocabularySize = 256;
  layout.dynamicSize = 256;
  layout.qkvSize = 256;
  layout.attentionSize = 128;
  layout.intermediateSize = 256;
  layout.attentionHeadDimension = 64;
  layout.targetHiddenSize = 256;
  layout.kvHeads = 1;
  return layout;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      const std::string_view kind = argc == 4 ? argv[3] : "";
      if (kind != "dense" && kind != "moe" && kind != "draft")
        throw std::runtime_error("usage: affine-preparation METALLIB FIXTURE dense|moe|draft");
      const std::filesystem::path root(argv[2]);
      setenv("SPLASH_WEIGHT_CACHE", (root / "cache").c_str(), 1);
      metal::MetalBackend backend(argv[1]);
      if (kind == "draft") {
        const model::DFlashDraftLayout layout = tinyDraftLayout();
        prepare<model::DraftCheckpointLoader>(backend, root, layout, [](auto &loader, const auto &check) {
          check(loader.layer(0));
          check(loader.layer(1));
          check(loader.model());
        });
        // The draft reads the prepared files as it reads a package's.
        model::DraftCheckpointLoader files(backend, root, layout, {});
        const model::DFlashDraftWeights draft = model::loadDFlashDraftWeights(backend, std::ref(files), layout);
        const auto affine = [](const ops::Projection &p, uint32_t n, uint32_t k) {
          return p.layout() == ops::WeightLayout::Affine64 && p.outputSize == n && p.inputSize == k;
        };
        bool read = draft.layers.size() == layout.layers && draft.files.size() == layout.layers + 1 &&
                    affine(draft.contextProjection, layout.hiddenSize, layout.targetHiddenSize) &&
                    affine(draft.selectorProjection, layout.selectorRank, layout.hiddenSize);
        for (const auto &layer : draft.layers)
          read = read && affine(layer.attentionDynamic, layout.dynamicSize, layout.hiddenSize) &&
                 affine(layer.qkvProjection, layout.qkvSize, layout.hiddenSize) &&
                 affine(layer.outputProjection, layout.hiddenSize, layout.attentionSize) &&
                 affine(layer.downProjection, layout.hiddenSize, layout.intermediateSize);
        if (!read) throw std::runtime_error("the draft loader misread the prepared draft files");
        std::cout << "affine preparation: exact independent draft fixture, quantization edge cases, fused qkv, "
                     "warm admission, planned weights, draft read PASS\n";
        return 0;
      }
      if (kind == "moe") {
        auto layout = tinyLayout<model::Qwen3_6MoeLayout>();
        layout.experts = 256;
        layout.expertsPerToken = 8;
        layout.expertIntermediateSize = 256;
        prepare<model::AffineTargetLoader>(backend, root, layout, [](auto &loader, const auto &check) {
          openTarget(loader, check);
        });
        std::cout << "affine preparation: exact independent fixture, MoE experts, 8-bit router and shared-expert "
                     "gate, warm admission, planned weights PASS\n";
        return 0;
      }
      auto layout = tinyLayout<model::Qwen3_8Layout>();
      layout.intermediateSize = 512;
      prepare<model::AffineTargetLoader>(backend, root, layout, [](auto &loader, const auto &check) {
        openTarget(loader, check);
      });
      // The target loader reads the prepared files as affine Q4 projections of
      // the layout's sizes with bf16 norms, the head into fp32 logits.
      model::AffineTargetLoader files(backend, root, layout, {});
      const model::Qwen3_8Weights weights = model::loadQwen3_8Weights(backend, layout, files);
      const auto affine = [](const ops::Projection &p, uint32_t n, uint32_t k) {
        return p.layout() == ops::WeightLayout::Affine64 && p.outputSize == n && p.inputSize == k;
      };
      bool read = weights.layers.size() == layout.layers && !weights.finalNorm.float32 &&
                  affine(weights.logitsProjection, layout.vocabularySize, layout.hiddenSize) &&
                  weights.logitsProjection.destination == ops::FloatOutput::Float32 &&
                  weights.tokenEmbedding.layout() == ops::WeightLayout::Affine64;
      for (const auto &layer : weights.layers) {
        read = read && !layer.inputNorm.float32 && !layer.postAttentionNorm.float32 &&
               affine(layer.gateProjection, layout.intermediateSize, layout.hiddenSize) &&
               affine(layer.downProjection, layout.hiddenSize, layout.intermediateSize);
        if (const auto *gdn = std::get_if<model::QwenGdnWeights>(&layer.mixer))
          read = read && affine(gdn->inputProjection, layout.packedGdnWidth, layout.hiddenSize) &&
                 gdn->outputHeadOrder == ops::GdnHeadOrder::Grouped && !gdn->mixerNorm.float32;
        else
          read = read && affine(std::get<model::QwenAttentionWeights>(layer.mixer).inputProjection,
                                layout.packedFullWidth, layout.hiddenSize);
      }
      if (!read) throw std::runtime_error("the target loader misread the prepared affine files");
      // The dense layout is checked like the MoE one, before any file is
      // opened: its capture layers and its convolution width against its GDN
      // heads.
      const auto inconsistent = [&](const model::Qwen3_8Layout &broken) {
        try {
          static_cast<void>(model::loadQwen3_8Weights(
              backend, broken, model::PackedTargetFiles<model::Qwen3_8Layout>{backend, root, broken}));
        } catch (const model::WeightStoreError &error) {
          return std::string_view(error.what()) == "Qwen target layout is inconsistent";
        }
        return false;
      };
      auto capturePastLastLayer = layout;
      capturePastLastLayer.hiddenCaptureLayers.back() = layout.layers;
      auto convolutionMismatch = layout;
      convolutionMismatch.convolutionDimension += layout.gdnHeadDimension;
      if (!inconsistent(capturePastLastLayer) || !inconsistent(convolutionMismatch))
        throw std::runtime_error("the target loader accepted an inconsistent dense layout");
      std::cout << "affine preparation: exact independent fixture, padding, fused order, gate/up, warm admission, "
                   "planned weights, target read, layout checks PASS\n";
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
}
