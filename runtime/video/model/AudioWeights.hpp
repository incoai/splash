#pragma once

// Converted audio VAE decoder (tools/convert_audio.py from FL2VA/audio_vae): the decoder's weights
// (ops/AudioDecoder.hpp) in one blob audio_dec_f16.bin + manifest.json (byte offsets, 256-byte aligned).
#include "video/ops/AudioDecoder.hpp"

#include <string>

namespace splash::video::model {

[[nodiscard]] ops::AudioDecoderWeights loadAudioDecoder(metal::MetalBackend &backend, const std::string &dir);

} // namespace splash::video::model
