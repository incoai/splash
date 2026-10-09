#pragma once

// Plans the images of an MLX target: the sections a GGUF target's images hold
// (model/GgufImage.hpp), in their order, which writeGgufImage writes from the
// checkpoint's tensors and BlockTargetFormat reads with bf16 norms and the GDN
// value heads grouped, as MLX keeps them. Every quantized tensor keeps the
// format MLX infers from its tensors, affine (metal/abi/QuantFormat.h) or
// mxfp4 (GGUF_FMT_MXFP4): projections and experts as planes, the token table
// as native rows. Norms, the convolution and dt_bias are copied as stored,
// and the GDN decay is float(-exp(double(A_log))). The MoE router and the
// shared-expert gate, which the block kernels read in F32, become the F32
// values s * code + z of their MLX quantization (or their bf16 values), and so
// do GDN alpha and beta unless both are quantized in one format.

#include "model/GgufImage.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/SafetensorsCheckpoint.hpp"

#include <string>
#include <vector>

namespace splash::model::mlx {

// The layers' images, then the head's and the embedding's. The checkpoint
// outlives them: their rows read its tensors.
[[nodiscard]] std::vector<gguf::Image> planImages(const SafetensorsCheckpoint &checkpoint,
                                                  const QwenTargetDimensions &geometry);

// The modules the images read only quantized, in their order: the layers'
// projections, the experts' and the shared expert's included, the head and
// the token table. planImages refuses a checkpoint holding one unquantized,
// and the configuration check one its config.json states unquantized
// (model/ModelDescriptor.mm). The router, the shared-expert gate and GDN alpha
// and beta are not among them.
[[nodiscard]] std::vector<std::string> quantizedModules(const QwenTargetDimensions &geometry);

} // namespace splash::model::mlx
