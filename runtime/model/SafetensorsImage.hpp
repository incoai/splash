#pragma once

// Plans the images of a safetensors target, MLX's, Model Optimizer's or
// compressed-tensors': the sections a GGUF target's images hold
// (model/GgufImage.hpp), in their order, which writeGgufImage writes from the
// checkpoint's tensors and BlockTargetFormat reads with the GDN value heads
// grouped, as all keep them. Every quantized tensor keeps the format its
// tensors hold: MLX affine (metal/abi/QuantFormat.h), mxfp4 or nvfp4 as MLX
// infers it, or NVFP4 or FP8 as the other two store them: projections and
// experts as planes, a quantized token table as native rows. An unquantized
// token table is copied as its bf16 rows. Norms are bf16 as stored, or the
// F32 1 + w of a w stored 1 below (float32Norms); the convolution and
// dt_bias are copied as stored, and the GDN decay is
// float(-exp(double(A_log))). The MoE router keeps the format it is stored
// in, as native rows when quantized, which the float projection decodes; the
// shared-expert gate, which the MoE selection reads in F32, becomes the F32
// values of its quantization (or of its bf16 weights), and so do GDN alpha
// and beta unless both are quantized in one format.

#include "model/GgufImage.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/SafetensorsCheckpoint.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace splash::model::safetensors {

// The format of MLX quantization `mode` at `bits` in groups of `group`
// (metal/abi/QuantFormat.h): affine at the widths QuantFormat.h holds, mxfp4
// 4 bits in groups of 32 or nvfp4 4 bits in groups of 16; GGUF_FMT_COUNT for
// one the images do not load. Both the configuration check and the images
// read it, and their refusals name what it takes in these two phrases.
[[nodiscard]] uint32_t mlxFormat(std::string_view mode, uint32_t bits, uint32_t group) noexcept;
inline constexpr std::string_view kMlxAffineFormats = "affine 2, 3, 4, 5, 6 or 8 bits in groups of 32, 64 or 128";
inline constexpr std::string_view kMlxFloatFormats = "mxfp4 or nvfp4";

// The names a checkpoint's language-model modules take: mlx-vlm's
// (language_model.model.*, language_model.lm_head) or mlx-lm's (model.*,
// lm_head), the routed experts of both stacked under mlp.switch_mlp; or
// transformers' (as NVIDIA's Model Optimizer keeps them:
// model.language_model.*, lm_head, each routed expert a module
// mlp.experts.<e> of its own).
enum class ModuleNames : uint8_t { Mlx, MlxLm, Transformers };

// The names the checkpoint's tensors take, by where its token table is:
// language_model.model.embed_tokens (mlx-vlm's), model.language_model.
// (transformers') or model.embed_tokens (mlx-lm's).
[[nodiscard]] ModuleNames moduleNames(const SafetensorsCheckpoint &checkpoint);

// Whether the images hold the norms as F32 rather than bf16 as stored, by
// the names and the family whose GGUFs are of `architecture`: a checkpoint's whose RMSNorm weights are
// stored 1 below the weights the norm multiplies by as their F32 1 + w, but
// the GDN's gated norm's widened (transformers' checkpoints of the Qwen3.5
// hybrid), and Flint's widened, as its GGUF's are; an MLX hybrid's, which
// mlx-lm stores shifted to the multipliers, as stored.
[[nodiscard]] bool float32Norms(ModuleNames names, std::string_view architecture);

// The layers' images, then the head's and the embedding's, of a target of
// `geometry`, its norms as float32Norms holds them for `architecture` (its
// layout's ggufArchitecture), the sections those its GGUF walk writes
// (model/GgufImage.hpp), in their order. The checkpoint outlives them: their
// rows read its tensors.
[[nodiscard]] std::vector<gguf::Image> planImages(const SafetensorsCheckpoint &checkpoint,
                                                  const QwenTargetDimensions &geometry,
                                                  std::string_view architecture, gguf::PlanOptions options = {});

// The modules the images read only quantized, in their order and by `names`:
// the layers' projections, the routed experts' (as a configuration names
// them: MLX's stacked tensor per projection, once transformers' mlp.experts) and
// the shared expert's, and the head. planImages refuses a checkpoint holding
// one unquantized, and the configuration check one its config.json states
// unquantized (model/ModelDescriptor.mm). The router, the shared-expert gate,
// GDN alpha and beta, Flint's sinks and the token table are not among them.
[[nodiscard]] std::vector<std::string> quantizedModules(const QwenTargetDimensions &geometry, ModuleNames names);

} // namespace splash::model::safetensors
