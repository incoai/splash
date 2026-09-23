#pragma once

// Plans the in-memory MDGG0001 images of a Qwen3.8 target read straight from a
// llama.cpp GGUF: section offsets, the bytes the CPU fills (header, descriptors,
// norms, convolution, decay, time bias, alpha/beta) and the GPU repacks/copies
// that move quantized rows into 256-column tiles. Layout: 16-byte header
// (magic, layer, type), then 16 KiB-aligned sections; each quantized tensor is a
// 64-byte descriptor, then its plane0, optional plane1 and meta planes in the
// layout of its format (metal/abi/QuantFormat.h).

#include <cstdint>
#include <string>
#include <vector>

#include "metal/abi/Gguf.h"
#include "model/GgufFile.hpp"

namespace splash::model::gguf {

inline constexpr uint64_t kSectionAlignment = 16384;
inline constexpr char kImageMagic[9] = "MDGG0001";

struct TargetGeometry {
  uint32_t layers = 64;
  uint32_t hiddenSize = 5120;
  uint32_t vocabularySize = 248320;
  uint32_t intermediateSize = 17408;
  uint32_t gdnKeyHeads = 16;
  uint32_t gdnValueHeads = 48;
  uint32_t gdnHeadDimension = 128;
  uint32_t convolutionDimension = 10240;
  uint32_t attentionWidth = 6144;
  uint32_t attentionHeadDimension = 256;
  uint32_t fullAttentionPeriod = 4;
  [[nodiscard]] bool isFullAttentionLayer(uint32_t layer) const noexcept {
    return (layer + 1) % fullAttentionPeriod == 0;
  }
};

struct Fill {
  uint64_t offset = 0;
  std::vector<uint8_t> bytes;
};
struct Repack {
  GgufRepackParams params{}; // src_offset is relative to sourceOffset until the executor binds it
  uint64_t sourceOffset = 0; // absolute file offset of the tensor data
  uint64_t sourceBytes = 0;
};
struct Copy {
  GgufCopyParams params{};
  uint64_t sourceOffset = 0;
  uint64_t sourceBytes = 0;
};
struct Image {
  std::string name; // layer-N.bin, head.bin, embedding.bin
  uint32_t layer = 0;
  uint32_t type = 0;
  uint64_t bytes = 0;
  std::vector<Fill> fills;
  std::vector<Repack> repacks;
  std::vector<Copy> copies;
  uint64_t sourceBegin = ~uint64_t{0}; // covering range of GPU-read source bytes
  uint64_t sourceEnd = 0;
};

class ImagePlanner final {
public:
  // Validates architecture, geometry and every tensor's presence, shape and
  // type; throws GgufError listing all offending tensors.
  ImagePlanner(const GgufFile &file, TargetGeometry geometry);
  [[nodiscard]] Image layer(uint32_t index) const;
  [[nodiscard]] Image head() const;
  [[nodiscard]] Image embedding() const;
  [[nodiscard]] const TargetGeometry &geometry() const noexcept { return geometry_; }
  // Sum of all image bytes, for memory accounting before allocation.
  [[nodiscard]] uint64_t totalBytes() const;

private:
  const GgufFile &file_;
  TargetGeometry geometry_;
};

} // namespace splash::model::gguf
