#pragma once

// Audio VAE decoder converted at installation (from FL2VA/audio_vae): the decoder's weights
// (video/ops/AudioDecoder.hpp) in one blob audio_dec_f16.bin + manifest.json (byte offsets, 256-byte aligned).
#include "video/ops/AudioDecoder.hpp"
#include "video/ops/AudioEncoder.hpp"

#include <string>

namespace splash::video::model {

[[nodiscard]] ops::AudioDecoderWeights loadAudioDecoder(metal::MetalBackend &backend, const std::string &dir);

// The audio VAE encoder (converted at installation): audio_enc_f16.bin +
// manifest_enc.json in audioDir, next to the decoder's files.
[[nodiscard]] ops::AudioEncoderWeights loadAudioEncoder(metal::MetalBackend &backend, const std::string &audioDir);

} // namespace splash::video::model
