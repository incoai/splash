#import <Foundation/Foundation.h>

#include "model/AffineTarget.hpp"
#include "model/WeightImages.hpp"
#include "model/ModelDescriptor.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>

using namespace splash;

void compare(model::WeightFile file, const std::filesystem::path &target, uint64_t decayOffset = 0, uint32_t decayHeads = 0) {
  const auto &record = file.record();
  const auto path = target / std::filesystem::path(record.relativePath).filename();
  if (std::filesystem::file_size(path) != record.declaredBytes)
    throw std::runtime_error("image size differs: " + record.relativePath);
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open packed oracle");
  try {
    std::array<uint8_t, 16> header;
    model::readWeightBytes(fd, 0, header);
    if (memcmp(header.data(), record.magic.data(), 8) || memcmp(header.data() + 8, &record.layer, 4) ||
        memcmp(header.data() + 12, &record.type, 4)) throw std::runtime_error("image header differs");
    const auto data = file.section(record.declaredBytes - model::kWeightFileAlignment, {});
    const auto *loaded = static_cast<const uint8_t *>(data.contents());
    std::vector<uint8_t> bytes(1024 * 1024);
    uint32_t maximumDecayUlp = 0;
    for (uint64_t at = 0; at < data.sizeBytes(); at += bytes.size()) {
      auto part = std::span(bytes).first(std::min<uint64_t>(bytes.size(), data.sizeBytes() - at));
      model::readWeightBytes(fd, model::kWeightFileAlignment + at, part);
      // Released dense packages used MLX's float exp for the small GDN
      // decay vector. The source adapter uses double exp rounded to float.
      // Only that named section permits a two-ULP difference; every packed
      // code, scale, bias, other tensor and padding byte remains exact.
      const uint64_t absolute = model::kWeightFileAlignment + at;
      for (uint32_t head = 0; head < decayHeads; ++head) {
        const uint64_t offset = decayOffset + head * sizeof(float);
        if (offset < absolute || offset + 4 > absolute + part.size()) continue;
        const size_t local = offset - absolute;
        uint32_t actual, expected;
        memcpy(&actual, loaded + at + local, 4);
        memcpy(&expected, part.data() + local, 4);
        float a, b;
        memcpy(&a, &actual, 4); memcpy(&b, &expected, 4);
        const uint32_t ulp = actual > expected ? actual - expected : expected - actual;
        if (!std::isfinite(a) || !std::isfinite(b) || !(a < 0 && b < 0) || ulp > 2)
          throw std::runtime_error("GDN decay differs by more than two ULP");
        maximumDecayUlp = std::max(maximumDecayUlp, ulp);
        memcpy(part.data() + local, &actual, 4);
      }
      if (memcmp(loaded + at, part.data(), part.size())) {
        const auto different = std::mismatch(part.begin(), part.end(), loaded + at).first;
        throw std::runtime_error("image bytes differ: " + record.relativePath + " offset=" +
                                 std::to_string(model::kWeightFileAlignment + at + (different - part.begin())));
      }
    }
    file.finish();
    std::cout << record.relativePath << " bytes=" << record.declaredBytes << " packed_exact=true decay_max_ulp=" << maximumDecayUlp << std::endl;
  } catch (...) { close(fd); throw; }
  close(fd);
}

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 4 || argc > 5) throw std::runtime_error("usage: affine-source-oracle METALLIB SOURCE PACKED_PACKAGE [LAYER]");
      const auto started = std::chrono::steady_clock::now();
      metal::MetalBackend backend(argv[1]);
      const std::filesystem::path package(argv[3]);
      const auto descriptor = model::inspectModelPackage(package);
      std::visit([&](const auto &layout) {
        // Each image is written into memory of its own, so one is held at a
        // time.
        const auto check = [&](const auto &load, uint64_t decayOffset = 0, uint32_t decayHeads = 0) {
          model::WeightImages images(backend);
          model::AffineTargetLoader loader(images, argv[2], layout);
          compare(load(loader), package / "target", decayOffset, decayHeads);
        };
        const uint32_t begin = argc == 5 ? std::stoul(argv[4]) : 0;
        const uint32_t end = argc == 5 ? begin + 1 : layout.layers;
        const std::vector<model::affine::Image> planned = model::affineTargetImages(layout);
        for (uint32_t layer = begin; layer < end; ++layer) {
          const auto &sections = planned.at(layer).sections;
          const auto decay = std::ranges::find(sections, model::affine::SectionKind::Decay,
                                               &model::affine::Section::kind);
          const bool gdn = decay != sections.end();
          check([&](auto &loader) { return loader.layer(layer); }, gdn ? decay->offset : 0,
                gdn ? uint32_t(decay->bytes / sizeof(float)) : 0);
        }
        if (argc != 5) {
          check([](auto &loader) { return loader.head(); });
          check([](auto &loader) { return loader.embedding(); });
        }
      }, descriptor.target);
      std::cout << "affine source oracle PASS seconds="
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << '\n';
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
}
