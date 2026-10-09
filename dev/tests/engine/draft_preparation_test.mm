// A tiny DFlash2 draft checkpoint written into its block images, every byte
// against independent oracles: each copied tensor as stored, and each
// projection's planes the CPU reference's of the af4g64 rows
// run_draft_preparation.py quantizes as MLX does (FIXTURE/expected). Then
// the writer over memory of 0xFF bytes writes every byte, each image matches
// its golden hash, again once the images are released and restored, and the
// draft loader reads the images.
//
//   draft-preparation METALLIB FIXTURE GOLDENS
// GOLDENS is dev/tests/fixtures/weight-goldens/goldens.json; its README says
// how to update it.
#include "GgufFixtures.hpp"
#include "TestAdmission.hpp"
#include "model/DFlashDraft.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightImages.hpp"

#include <cstring>
#include <vector>

using namespace gguf_fixtures;
using namespace gguf_reference;

namespace {

namespace ops = splash::ops;
using splash::metal::MetalBackend;

// The draft fixture: two layers of width 256, two KV heads.
model::DFlashDraftLayout tinyDraftLayout() {
  model::DFlashDraftLayout layout;
  layout.layers = 2;
  layout.hiddenSize = 256;
  layout.vocabularySize = 256;
  layout.dynamicSize = 256;
  layout.qkvSize = 512;
  layout.attentionSize = 256;
  layout.intermediateSize = 256;
  layout.attentionHeadDimension = 64;
  layout.rotaryTheta = 10'000'000.0F;
  layout.targetHiddenSize = 256;
  layout.selectorRank = 256;
  layout.kvHeads = 2;
  return layout;
}

std::vector<uint8_t> tensorBytes(const model::SourceTensor &tensor) {
  std::vector<uint8_t> bytes(tensor.bytes);
  tensor.read(0, bytes);
  return bytes;
}

std::vector<uint8_t> bytesOf(std::span<const uint8_t> bytes) { return {bytes.begin(), bytes.end()}; }

// Whether `image` holds `bytes` at `offset`.
bool holds(const std::vector<uint8_t> &image, uint64_t offset, const std::vector<uint8_t> &bytes) {
  return offset + bytes.size() <= image.size() && std::equal(bytes.begin(), bytes.end(), image.begin() + offset);
}

// Each copy of the planned image as stored in the checkpoint, and the planes
// of each projection the CPU reference's of its parts' expected rows.
void checkImage(const std::vector<uint8_t> &image, const model::gguf::Image &plan,
                const model::SafetensorsCheckpoint &checkpoint, const model::SafetensorsCheckpoint &expected) {
  for (const model::gguf::Copy &copy : plan.copies)
    check(holds(image, copy.destination, tensorBytes(checkpoint.require(copy.source.name))),
          plan.name + " " + copy.source.name + ": as stored");
  for (const model::gguf::Repack &step : plan.repacks) {
    std::vector<uint8_t> native;
    for (const model::gguf::TensorRows &source : step.sources) {
      const std::vector<uint8_t> rows = tensorBytes(expected.require(source.name));
      native.insert(native.end(), rows.begin(), rows.end());
    }
    const Fmt f = Fmt(step.format);
    native.resize(size_t(step.rows) * rowBytes(f, uint32_t(step.columns)), 0);
    const Packed planes = repack(f, native, uint32_t(step.rows), uint32_t(step.columns), nullptr);
    check(step.format == GGUF_FMT_AF4G64 && holds(image, step.plane0, planes.w0) && holds(image, step.meta, planes.meta),
          plan.name + " " + step.sources.front().name + ": af4g64 planes of MLX's quantization");
  }
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 4) {
      std::fprintf(stderr, "usage: draft-preparation METALLIB FIXTURE GOLDENS\n");
      return 2;
    }
    const std::filesystem::path root(argv[2]);
    const Goldens hashes = goldens(argv[3], @"draft_images");
    MetalBackend backend(argv[1]);
    const model::DFlashDraftLayout layout = tinyDraftLayout();
    guarded("draft preparation", [&] {
      model::WeightImages images(backend, "fixture");
      model::DraftCheckpointLoader loader(backend, images, root, layout);
      const model::DFlashDraftWeights draft = model::loadDFlashDraftWeights(backend, loader, layout);
      const model::SafetensorsCheckpoint checkpoint(root), expected(root / "expected");
      const std::vector<model::gguf::Image> plans = model::planDraftImages(checkpoint, layout);
      const auto contents = images.contents();
      check(contents.size() == plans.size(), "the draft loads every planned image");
      std::vector<std::vector<uint8_t>> loaded;
      for (size_t index = 0; index < plans.size(); ++index) {
        const auto &image = contents[index];
        loaded.push_back(bytesOf(image.bytes));
        checkImage(loaded.back(), plans[index], checkpoint, expected);
        const auto buffer = backend.allocateBuffer(plans[index].bytes, splash::metal::BufferStorage::Shared, "image");
        std::memset(buffer.contents(), 0xFF, plans[index].bytes);
        model::writeGgufImage(backend, buffer, plans[index]);
        check(bytesOf(model::contentsOf(buffer)) == loaded.back(), plans[index].name + ": every byte written");
        checkGolden(hashes, std::string(image.component), image.bytes, std::string(image.component) + ": golden");
      }
      images.release();
      while (!images.restore(splash::test::admitAll)) {
      }
      for (size_t index = 0; index < loaded.size(); ++index)
        check(bytesOf(images.contents()[index].bytes) == loaded[index],
              std::string(images.contents()[index].component) + ": restored");
      // The draft loader reads every projection as one af4g64 segment of the
      // layout's sizes.
      const auto projection = [](const ops::Projection &p, uint32_t n, uint32_t k) {
        return p.outputSize == n && p.inputSize == k && p.blocks().segments.size() == 1 &&
               p.blocks().segments.front().formatId == GGUF_FMT_AF4G64;
      };
      bool read = draft.layers.size() == layout.layers && draft.files.size() == layout.layers + 1 &&
                  projection(draft.contextProjection, layout.hiddenSize, layout.targetHiddenSize) &&
                  projection(draft.selectorProjection, layout.selectorRank, layout.hiddenSize);
      for (const auto &layer : draft.layers)
        read = read && projection(layer.attentionDynamic, layout.dynamicSize, layout.hiddenSize) &&
               projection(layer.qkvProjection, layout.qkvSize, layout.hiddenSize) &&
               projection(layer.outputProjection, layout.hiddenSize, layout.attentionSize) &&
               projection(layer.downProjection, layout.hiddenSize, layout.intermediateSize);
      check(read, "the draft loader reads the draft images");
    });
    std::printf("%s (%d failures)\n", failures ? "draft preparation tests FAILED" : "draft preparation tests passed",
                failures);
    return failures ? 1 : 0;
  }
}
