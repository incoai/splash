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
#include "TestCheckpoint.hpp"
#include "TestFiles.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "model/DFlashDraft.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightImages.hpp"
#include "ops/Embedding.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
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
  layout.targetLayerCount = 1;
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

// A checkpoint that ships its mask token's embedding beside its weights
// (mask_embedding.safetensors): model.bin holds it after the codebooks, the
// loader reads it as shipped, and each lane's mask rows of the draft input
// take it, the lane's anchor row as the table gave it. An embedding of
// another shape is refused.
void checkMaskEmbedding(MetalBackend &backend, const std::filesystem::path &root,
                        const model::DFlashDraftLayout &layout) {
  const splash::test::TemporaryDirectory directory("splash-draft-mask");
  for (const auto &entry : std::filesystem::directory_iterator(root))
    if (entry.path().extension() == ".safetensors")
      std::filesystem::copy_file(entry.path(), directory.path() / entry.path().filename());
  const uint64_t rowBytes = uint64_t{layout.hiddenSize} * 2;
  std::vector<uint8_t> mask(rowBytes);
  for (uint32_t column = 0; column < layout.hiddenSize; ++column) {
    const uint16_t bits = uint16_t(0x3400 + column);
    std::memcpy(mask.data() + uint64_t{column} * 2, &bits, 2);
  }
  const std::filesystem::path path = directory.path() / "mask_embedding.safetensors";
  splash::test::writeSyntheticShard(path, {{"mask_embedding", "BF16", {layout.hiddenSize}, mask}});
  model::WeightImages images(backend, "mask fixture");
  model::DraftCheckpointLoader loader(backend, images, directory.path(), layout);
  check(loader.hasMaskEmbedding(), "the loader finds the mask embedding the checkpoint ships");
  const model::DFlashDraftWeights draft = model::loadDFlashDraftWeights(backend, loader, layout);
  check(draft.maskEmbedding && bytesOf(model::contentsOf(draft.maskEmbedding)) == mask,
        "the draft reads the mask embedding as shipped");
  // Two lanes of poisoned draft input rows.
  const uint32_t lanes = 2, rows = lanes * SPLASH_TARGET_VERIFY_ROWS;
  const auto input =
      backend.allocateBuffer(rows * rowBytes, splash::metal::BufferStorage::Shared, "draft input rows");
  std::memset(input.contents(), 0xA5, rows * rowBytes);
  splash::metal::CommandGraph graph;
  ops::Embedding::addDraftMaskRows(graph, draft.maskEmbedding, input, layout.hiddenSize, lanes);
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  const std::vector<uint8_t> written = bytesOf(model::contentsOf(input));
  bool set = true;
  for (uint32_t row = 0; row < rows; ++row) {
    const auto begin = written.begin() + row * rowBytes;
    set = set && (row % SPLASH_TARGET_VERIFY_ROWS
                      ? std::equal(mask.begin(), mask.end(), begin)
                      : std::all_of(begin, begin + rowBytes, [](uint8_t byte) { return byte == 0xA5; }));
  }
  check(set, "each lane's mask rows take the mask embedding, its anchor row as it was");
  splash::test::writeSyntheticShard(path, {{"mask_embedding", "BF16", {layout.hiddenSize / 2}, {}}});
  bool refused = false;
  try {
    const model::SafetensorsCheckpoint checkpoint(directory.path());
    static_cast<void>(model::planDraftImages(checkpoint, layout));
  } catch (const std::exception &error) {
    refused = std::string_view(error.what()).find("mask_embedding") != std::string_view::npos;
  }
  check(refused, "a mask embedding of another shape is refused");
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
                  projection(draft.contextProjection, layout.hiddenSize, layout.targetHiddenSize()) &&
                  projection(draft.selectorProjection, layout.selectorRank, layout.hiddenSize);
      for (const auto &layer : draft.layers)
        read = read && projection(layer.attentionDynamic, layout.dynamicSize, layout.hiddenSize) &&
               projection(layer.qkvProjection, layout.qkvSize, layout.hiddenSize) &&
               projection(layer.outputProjection, layout.hiddenSize, layout.attentionSize) &&
               projection(layer.downProjection, layout.hiddenSize, layout.intermediateSize);
      check(read && !draft.maskEmbedding, "the draft loader reads the draft images");
    });
    guarded("draft mask embedding", [&] { checkMaskEmbedding(backend, root, layout); });
    std::printf("%s (%d failures)\n", failures ? "draft preparation tests FAILED" : "draft preparation tests passed",
                failures);
    return failures ? 1 : 0;
  }
}
