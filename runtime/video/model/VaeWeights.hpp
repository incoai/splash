#pragma once

// Converted video VAE (tools/convert_vae.py). Decoder: one flat blob vae_dec_<tag>.bin + manifest_<tag>.json with the
// byte offsets of the globals and of every layer tensor (256-byte aligned); each layer's byte range is loaded into its
// own Metal buffer; its int8 layers are of the format QUANT_VIDEO_W8 (metal/abi/QuantFormat.h), whose Hadamard seed
// the manifest names. Encoder (FL2VA keyframes): vae_enc_f16.bin + manifest_enc.json, one buffer.
#include "video/ops/VaeDecoder.hpp"
#include "video/ops/VaeEncoder.hpp"

#include <string>

namespace splash::video::model {

[[nodiscard]] ops::VaeDecoderWeights loadVaeDecoder(metal::MetalBackend &backend, const std::string &dir,
                                                    const std::string &tag);
[[nodiscard]] ops::VaeEncoderWeights loadVaeEncoder(metal::MetalBackend &backend, const std::string &dir);

} // namespace splash::video::model
