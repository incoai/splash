#pragma once

// Plans the images of a safetensors target, MLX's, Model Optimizer's or
// compressed-tensors': the sections a GGUF target's images hold
// (model/GgufImage.hpp), in their order, which writeGgufImage writes from the
// checkpoint's tensors and BlockTargetFormat reads with the GDN value heads
// grouped, as all keep them. Every quantized tensor keeps the format its
// tensors hold: MLX affine (metal/abi/QuantFormat.h), mxfp4 or nvfp4 as MLX
// infers it, or NVFP4 or FP8 as the other two store them: projections and
// experts as planes, a quantized token table as native rows. An unquantized
// token table is copied as its bf16 rows. Ordinary MLX norms stay bf16;
// complete MLX-LM and transformers exports' zero-centered RMSNorms become
// F32 1 + w, and their gated norms are widened without an offset. Convolution
// and dt_bias are copied as stored; the GDN decay is float(-exp(double(A_log))).
// The MoE router and shared-expert gate become the F32 values of their
// quantization (or of their bf16 weights), as do GDN alpha and beta unless both
// are quantized in one format.

#include "model/GgufImage.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/SafetensorsCheckpoint.hpp"

#include <optional>
#include <string>
#include <vector>

namespace splash::model::safetensors {

// The names a checkpoint's language-model modules take: MLX's (mlx-lm,
// mlx-vlm: language_model.model.*, the routed experts stacked under
// mlp.switch_mlp), the complete MLX-LM export's same names with retained MTP
// offset norms, or transformers' (model.language_model.*, lm_head, each
// routed expert a module mlp.experts.<e>). The latter two keep zero-centered
// RMSNorm weights, but the GDN's gated norm's weight is already multiplicative.
enum class ModuleNames : uint8_t { Mlx, MlxFull, Transformers };

// Transformers names have a root lm_head; complete MLX-LM exports retain
// mtp.pre_fc_norm_embedding.weight and mtp.pre_fc_norm_hidden.weight.
[[nodiscard]] ModuleNames moduleNames(const SafetensorsCheckpoint &checkpoint);

// Zero-centered norms become F32 1 + w; the gated norm is widened without
// adding an offset. Ordinary MLX checkpoints keep their BF16 norms as stored.
[[nodiscard]] constexpr bool float32Norms(ModuleNames names) noexcept { return names != ModuleNames::Mlx; }

// The native rows of an MLX quantized module, inferred from its codes,
// scales and biases and validated against its dimensions; none when it has
// no scales. The checkpoint outlives the returned rows.
[[nodiscard]] std::optional<gguf::TensorRows>
mlxQuantizedRows(const SafetensorsCheckpoint &checkpoint, const std::string &module,
                 uint64_t rows, uint64_t columns, uint64_t experts = 1);

// The layers' images, then the head's and the embedding's. The checkpoint
// outlives them: their rows read its tensors.
[[nodiscard]] std::vector<gguf::Image> planImages(const SafetensorsCheckpoint &checkpoint,
                                                  const QwenTargetDimensions &geometry);

// The modules the images read only quantized, in their order and by `names`:
// the layers' projections, the routed experts' (as a configuration names
// them: MLX's stacked tensor per projection, once transformers' mlp.experts) and
// the shared expert's, and the head. planImages refuses a checkpoint holding
// one unquantized, and the configuration check one its config.json states
// unquantized (model/ModelDescriptor.mm). The router, the shared-expert gate,
// GDN alpha and beta and the token table are not among them.
[[nodiscard]] std::vector<std::string> quantizedModules(const QwenTargetDimensions &geometry, ModuleNames names);

} // namespace splash::model::safetensors
