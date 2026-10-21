#pragma once

// Converted TAEH3 decoder (tools/convert_preview.py from madebyollin/taehv taeh3.safetensors, MIT): fp16 conv
// weights and fp32 biases in one blob taeh3_dec_f16.bin + manifest.json (byte offsets, 256-byte aligned).
//   conv3x3 W[src][tap 9][Cout][Cin] (MemBlock conv.0: src 0 = x, src 1 = past), tgrow 1x1 W[Cout][Cin],
//   up (nearest 2x + 3x3) W[phase 4][tap 2x2][Cout][Cin] pre-combined.
#include "video/h3/engine/Encoder.hpp"

#include <array>
#include <memory>
#include <string>

namespace h3::model {

struct PreviewMemBlock {
  BufferView conv0, bias0, conv2, bias2, conv4, bias4;
};

class PreviewWeights {
public:
  static std::unique_ptr<PreviewWeights> load(metal::MetalBackend &backend, const std::string &dir);

  BufferView inputConv, inputBias;          // decoder.1: 24 (32) -> 256
  std::array<PreviewMemBlock, 9> memBlocks;        // decoder.3-5, 9-11, 15-17
  std::array<BufferView, 3> tgrow;          // decoder.7, 13, 19
  std::array<BufferView, 3> upsample;       // decoder.8, 14, 20
  BufferView outputConv, outputBias;        // decoder.22: 64 -> 12 (16)

private:
  metal::MetalBuffer blob_;
};

} // namespace h3::model
