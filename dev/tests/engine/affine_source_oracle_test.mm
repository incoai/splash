#import <Foundation/Foundation.h>

#include "model/AffineTarget.hpp"
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

void compare(model::WeightFile file, const std::filesystem::path &target, bool loadOnly = false, uint64_t decayOffset = 0, uint32_t decayHeads = 0) {
  const auto &record = file.record();
  const auto path = target / std::filesystem::path(record.relativePath).filename();
  if (std::filesystem::file_size(path) != record.declaredBytes)
    throw std::runtime_error("prepared size differs: " + record.relativePath);
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open packed oracle");
  try {
    std::array<uint8_t, 16> header;
    model::readWeightBytes(fd, 0, header);
    if (memcmp(header.data(), record.magic.data(), 8) || memcmp(header.data() + 8, &record.layer, 4) ||
        memcmp(header.data() + 12, &record.type, 4)) throw std::runtime_error("prepared header differs");
    const auto data = file.section(record.declaredBytes - model::kWeightFileAlignment);
    const auto *prepared = static_cast<const uint8_t *>(data.contents());
    std::vector<uint8_t> bytes(1024 * 1024);
    uint32_t maximumDecayUlp = 0;
    for (uint64_t at = 0; !loadOnly && at < data.sizeBytes(); at += bytes.size()) {
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
        memcpy(&actual, prepared + at + local, 4);
        memcpy(&expected, part.data() + local, 4);
        float a, b;
        memcpy(&a, &actual, 4); memcpy(&b, &expected, 4);
        const uint32_t ulp = actual > expected ? actual - expected : expected - actual;
        if (!std::isfinite(a) || !std::isfinite(b) || !(a < 0 && b < 0) || ulp > 2)
          throw std::runtime_error("GDN decay differs by more than two ULP");
        maximumDecayUlp = std::max(maximumDecayUlp, ulp);
        memcpy(part.data() + local, &actual, 4);
      }
      if (memcmp(prepared + at, part.data(), part.size())) {
        const auto different = std::mismatch(part.begin(), part.end(), prepared + at).first;
        throw std::runtime_error("prepared bytes differ: " + record.relativePath + " offset=" +
                                 std::to_string(model::kWeightFileAlignment + at + (different - part.begin())));
      }
    }
    file.finish();
    std::cout << record.relativePath << " bytes=" << record.declaredBytes << (loadOnly ? " opened=true" : " packed_exact=true") << " decay_max_ulp=" << maximumDecayUlp << std::endl;
  } catch (...) { close(fd); throw; }
  close(fd);
}

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 4 || argc > 5) throw std::runtime_error("usage: affine-source-oracle METALLIB SOURCE PACKED_PACKAGE [LAYER|--load-only]");
      const bool loadOnly = argc == 5 && std::string_view(argv[4]) == "--load-only";
      const auto started = std::chrono::steady_clock::now();
      metal::MetalBackend backend(argv[1]);
      const std::filesystem::path package(argv[3]);
      const auto descriptor = model::inspectModelPackage(package);
      std::visit([&](const auto &layout) {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(layout)>, model::Qwen4ExpLayout>) {
          throw std::runtime_error("Qwen3.8-Flash-Next has no affine source");
        } else {
        model::AffineTargetLoader loader(backend, argv[2], layout);
        const uint32_t begin = argc == 5 && !loadOnly ? std::stoul(argv[4]) : 0;
        const uint32_t end = argc == 5 && !loadOnly ? begin + 1 : layout.layers;
        for (uint32_t layer = begin; layer < end; ++layer) {
          const auto sections = model::affineLayerImage(layout, layer).sections;
          const auto decay = std::ranges::find(sections, model::affine::SectionKind::Decay,
                                               &model::affine::Section::kind);
          const bool gdn = decay != sections.end();
          compare(loader.layer(layer), package / "target", loadOnly, gdn ? decay->offset : 0,
                  gdn ? uint32_t(decay->bytes / sizeof(float)) : 0);
        }
        if (argc != 5 || loadOnly) {
          compare(loader.head(), package / "target", loadOnly);
          compare(loader.embedding(), package / "target", loadOnly);
        }
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
