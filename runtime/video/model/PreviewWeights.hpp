#pragma once

// Converted TAEH3 decoder (tools/convert_preview.py from madebyollin/taehv taeh3.safetensors, MIT): the decoder's
// weights (ops/PreviewDecoder.hpp) in one blob taeh3_dec_f16.bin + manifest.json (byte offsets, 256-byte aligned).
#include "video/ops/PreviewDecoder.hpp"

#include <string>

namespace splash::video::model {

[[nodiscard]] ops::PreviewDecoderWeights loadPreviewDecoder(metal::MetalBackend &backend, const std::string &dir);

} // namespace splash::video::model
