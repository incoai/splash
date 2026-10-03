// Prepares the tiny vision source run_vision_preparation.py writes (depth 2,
// width 8, 2x2 patches) and prints the prepared file and its cache key.
//
//   vision-preparation mlx|gguf DIRECTORY cold|warm [EXPECTED]
//   vision-preparation padding DIRECTORY
//
// warm requires a cache hit. EXPECTED is an independently serialized file the
// prepared bytes and the size estimate must equal. padding writes a padded
// section after an unpadded one in DIRECTORY and checks its padding is zero.

#include "TestFiles.hpp"
#include "model/VisionLoader.hpp"
#include "model/VisionPreparation.hpp"
#include "model/WeightLayout.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;

namespace {

// A 1x8 section, then a 2x3 section stored as 2x4, written from one BF16
// source: the second section's rows reuse the staging the first filled, and
// its padding column must still be zero.
void checkPadding(const std::filesystem::path &directory) {
  std::vector<uint16_t> values(14);
  for (size_t i = 0; i < values.size(); ++i) values[i] = uint16_t(0x3F80 + i); // 1 + i/128
  test::writeFile(directory / "source.bin",
                  {reinterpret_cast<const uint8_t *>(values.data()), values.size() * sizeof(uint16_t)});
  const model::WeightSource source(directory / "source.bin");
  const model::SourceTensor first{&source, "BF16", {1, 8}, 0, 16}, padded{&source, "BF16", {2, 3}, 16, 12};
  model::vision::Plan plan{.patchSize = 2, .bytes = 3 * model::kWeightFileAlignment};
  plan.sections.push_back({.rows = 1, .columns = 8, .storedRows = 1, .storedColumns = 8,
                           .offset = model::kWeightFileAlignment, .inputs = {{"first", first}}});
  plan.sections.push_back({.rows = 2, .columns = 3, .storedRows = 2, .storedColumns = 4,
                           .offset = 2 * model::kWeightFileAlignment, .inputs = {{"padded", padded}}});
  const auto path = directory / "padded.bin";
  const int file = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
  if (file < 0) throw std::runtime_error("cannot create " + path.string());
  const bool sized = ftruncate(file, off_t(plan.bytes)) == 0;
  if (sized) model::vision::writeVision(file, plan, [] {});
  close(file);
  if (!sized) throw std::runtime_error("cannot size " + path.string());
  const std::vector<uint8_t> bytes = test::readFile(path);
  std::vector<uint16_t> section(8);
  std::memcpy(section.data(), bytes.data() + plan.sections[1].offset, section.size() * sizeof(uint16_t));
  const std::vector<uint16_t> expected{values[8], values[9], values[10], 0, values[11], values[12], values[13], 0};
  if (section != expected) throw std::runtime_error("a padded section's padding columns are not zero");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 3 && std::string_view(argv[1]) == "padding") {
      checkPadding(argv[2]);
      std::cout << "padding columns zero\n";
      return 0;
    }
    if (argc < 4 || argc > 5)
      throw std::runtime_error("usage: vision-preparation mlx|gguf DIRECTORY "
                               "cold|warm [EXPECTED] | padding DIRECTORY");
    const std::string format = argv[1], mode = argv[3];
    if ((format != "mlx" && format != "gguf") ||
        (mode != "cold" && mode != "warm"))
      throw std::runtime_error("invalid vision-preparation arguments");
    const auto source = format == "mlx" ? model::VisionSource::Mlx
                                        : model::VisionSource::Gguf;
    ops::VisionLayout layout;
    layout.depth = 2;
    layout.hiddenSize = 8;
    layout.patchSize = 2;
    layout.patchDimension = 24;
    layout.intermediateSize = 10;
    layout.paddedIntermediateSize = 16;
    layout.mergedHiddenSize = 32;
    layout.outputHiddenSize = 8;
    layout.heads = 2;
    layout.headDimension = 4;
    layout.positionGridSide = 2;
    const auto forbidden = [] {
      throw std::runtime_error("unexpected warm conversion");
    };
    const model::VisionLoader preparation(argv[2], source, layout, {},
                                          mode == "warm" ? model::PreparationCheck(forbidden)
                                                         : model::PreparationCheck());
    const auto path = preparation.prepare();
    if (model::VisionLoader(argv[2], source, layout, {}, forbidden).prepare() !=
        path)
      throw std::runtime_error("warm cache miss");
    if (argc == 5) {
      const auto actual = test::readFile(path), expected = test::readFile(argv[4]);
      const auto differs = std::mismatch(actual.begin(), actual.end(),
                                         expected.begin(), expected.end());
      if (differs.first != actual.end() || differs.second != expected.end())
        throw std::runtime_error(
            "prepared bytes differ from the oracle from byte " +
            std::to_string(differs.first - actual.begin()) + " (" +
            std::to_string(actual.size()) + " prepared, " +
            std::to_string(expected.size()) + " expected)");
      if (model::preparedVisionBytes(layout) != expected.size())
        throw std::runtime_error("vision size estimate differs from the oracle");
    }
    std::cout << path.string() << '\n' << preparation.weight().key << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
