// Loads a tiny dense or MoE affine checkpoint, or a DFlash2 draft
// checkpoint, and prints each image's SHA-256. Every image must equal the
// independently serialized file in FIXTURE/expected, and again once the
// images are released and restored.
//
//   affine-preparation METALLIB FIXTURE dense|moe|draft
#include "model/AffinePlan.hpp"
#include "model/AffineTarget.hpp"
#include "model/DFlashDraft.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/WeightImages.hpp"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
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

// Loads the fixture at root with Loader; open opens every file of the
// loader, in the order it plans them, through check.
template <class Loader, class Layout, class Open>
void prepare(metal::MetalBackend &backend, const std::filesystem::path &root, const Layout &layout, Open open) {
  model::WeightImages images(backend);
  Loader loader(images, root, layout);
  open(loader, [](model::WeightFile weights) {
    static_cast<void>(weights.section(weights.record().declaredBytes - model::kWeightFileAlignment, {}));
    weights.finish();
  });
  std::vector<std::vector<uint8_t>> loaded;
  for (const auto &image : images.contents()) {
    loaded.emplace_back(image.bytes.begin(), image.bytes.end());
    if (loaded.back() != fileBytes(root / "expected" / std::filesystem::path(image.component).filename()))
      throw std::runtime_error("affine fixture differs: " + std::string(image.component));
    std::cout << "prepared " << image.component << ' ' << model::weightDigest(image.bytes) << '\n';
  }
  images.release();
  while (!images.restore()) {
  }
  for (size_t index = 0; index < loaded.size(); ++index) {
    const auto restored = images.contents()[index].bytes;
    if (!std::equal(restored.begin(), restored.end(), loaded[index].begin(), loaded[index].end()))
      throw std::runtime_error("a restored affine image differs: " + std::string(images.contents()[index].component));
  }
}

// The writer of each image, over memory of 0xFF bytes, writes its expected
// file: every byte of it.
void writeEveryByte(const std::filesystem::path &root, std::vector<model::affine::Image> images) {
  auto planned = std::make_shared<model::affine::PlannedCheckpoint>(root);
  planned->images = std::move(images);
  for (auto &image : planned->images) model::affine::bind(image, planned->source);
  for (size_t index = 0; index < planned->images.size(); ++index) {
    const model::ImagePlan plan = model::affine::imagePlan(planned, index, "");
    std::vector<uint8_t> bytes(plan.bytes, 0xFF);
    plan.write(bytes, {});
    if (bytes != fileBytes(root / "expected" / planned->images[index].name))
      throw std::runtime_error("an affine image left bytes unwritten: " + planned->images[index].name);
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
      metal::MetalBackend backend(argv[1]);
      if (kind == "draft") {
        const model::DFlashDraftLayout layout = tinyDraftLayout();
        prepare<model::DraftCheckpointLoader>(backend, root, layout, [](auto &loader, const auto &check) {
          check(loader.layer(0));
          check(loader.layer(1));
          check(loader.model());
        });
        writeEveryByte(root, model::draftCheckpointImages(layout));
        // The draft reads the prepared files as it reads a package's.
        model::WeightImages images(backend);
        model::DraftCheckpointLoader files(images, root, layout);
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
                     "restore, draft read PASS\n";
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
        writeEveryByte(root, model::affineTargetImages(layout));
        std::cout << "affine preparation: exact independent fixture, MoE experts, 8-bit router and shared-expert "
                     "gate, restore PASS\n";
        return 0;
      }
      auto layout = tinyLayout<model::Qwen3_8Layout>();
      layout.intermediateSize = 512;
      prepare<model::AffineTargetLoader>(backend, root, layout, [](auto &loader, const auto &check) {
        openTarget(loader, check);
      });
      writeEveryByte(root, model::affineTargetImages(layout));
      // The target loader reads the prepared files as affine Q4 projections of
      // the layout's sizes with bf16 norms, the head into fp32 logits.
      model::WeightImages images(backend);
      model::AffineTargetLoader files(images, root, layout);
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
              backend, broken, model::PackedTargetFiles<model::Qwen3_8Layout>{images, root, broken}));
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
      std::cout << "affine preparation: exact independent fixture, padding, fused order, gate/up, restore, "
                   "target read, layout checks PASS\n";
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
}
