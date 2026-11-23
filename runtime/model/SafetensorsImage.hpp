#pragma once

// Plans the images of a safetensors target, MLX's, Model Optimizer's or
// compressed-tensors': the sections a GGUF target's images hold
// (model/GgufImage.hpp), in their order, which writeGgufImage writes from the
// checkpoint's tensors and BlockTargetFormat reads with the GDN value heads
// grouped, as all keep them. Every quantized tensor keeps the format its
// tensors hold: MLX affine (metal/abi/QuantFormat.h), mxfp4, mxfp8 or nvfp4
// as MLX infers it, or NVFP4 or FP8 as the other two store them: projections
// and experts as planes, a quantized token table as native rows. A projection
// or expert of BF16 or F16 weights takes Q8_0's planes, quantized from its
// values as the reference GGUF quantization of Q8_0 rounds them; an unquantized
// token table is copied as its bf16 rows. Norms are bf16 as stored, or the
// F32 1 + w of a w stored 1 below, or widened (float32Norms); the convolution and
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
// 4 bits or mxfp8 8 bits in groups of 32, or nvfp4 4 bits in groups of 16;
// GGUF_FMT_COUNT for one the images do not load. Both the configuration
// check and the images read it, and their refusals name what it takes in
// these two phrases.
[[nodiscard]] uint32_t mlxFormat(std::string_view mode, uint32_t bits, uint32_t group) noexcept;
inline constexpr std::string_view kMlxAffineFormats = "affine 2, 3, 4, 5, 6 or 8 bits in groups of 32, 64 or 128";
inline constexpr std::string_view kMlxFloatFormats = "mxfp4, mxfp8 or nvfp4";

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

// The prefix of layer `layer`'s modules by `names` (language_model.model.layers.N.
// and the like), for what a family reads beside its images.
[[nodiscard]] std::string layerPrefix(ModuleNames names, uint32_t layer);

// How a checkpoint stores its RMSNorm weights: as the multipliers the norms
// multiply by, or 1 below them, as transformers does (Qwen3.5 and later). No
// metadata records it and no naming decides it: mlx-vlm keeps
// Qwen3.8-Flash-Next's 1 below but the Qwen3.5 hybrids' as the multipliers,
// and conversions of either kind exist under every naming.
enum class NormStorage : uint8_t { Multipliers, BelowMultipliers };

// How the checkpoint stores its norms, told by its attention q and k norms,
// whose mean over every full-attention layer is a family's own
// (attentionNormMean) stored as the multipliers, and that less 1 stored
// below: the nearer of the two is the storage, so a checkpoint may lie up to
// 0.5 from its family's mean and still be read right. Every checkpoint
// measured lies within 0.001 of one of the two (Qwen3.8-27B-4bit's 1.4909,
// nvidia's 0.4909, flint-mixed-3-4bit's 1.7071; Flash-Next's were read to
// three places). The families' means, measured on their checkpoints (48 GB
// and 64 GB M5 Pro, 11-17 to 11-19; the 122B's from the Hub's bytes of its
// 24 norm tensors):
// Qwen3.8-27B 1.491 (mlx-community/Qwen3.8-27B-4bit; nvidia/Qwen3.8-27B-NVFP4,
// transformers' names, 0.491 below); Qwen3.6-35B-A3B 1.490
// (mlx-community/Qwen3.6-35B-A3B-4bit); Qwen3.5-122B-A10B 1.414
// (ornith-ai/Ornith-1.5-122B-MLX-4bit, a fine-tune of it);
// Qwen3.8-Flash-Next 1.454 (mlx-community/Qwen3.8-Flash-Next-OptiQ-2bit;
// mlx-community/Qwen3.8-Flash-Next-4bit 0.454 below); Flint 1.707
// (mlx-community/flint-mixed-3-4bit). Reads two tensors a full-attention
// layer: a loader decides it once and plans with the answer.
[[nodiscard]] NormStorage normStorage(const SafetensorsCheckpoint &checkpoint, const QwenTargetDimensions &geometry);

// Whether the images hold the norms as F32 rather than bf16 as stored: those
// stored below their multipliers as the F32 1 + w, but the GDN's gated
// norm's widened; else, for a geometry whose kernels read F32 norms alone
// (float32NormsOnly), widened; any other's as stored.
[[nodiscard]] bool float32Norms(NormStorage norms, const QwenTargetDimensions &geometry);

// The layers' images, then the head's and the embedding's, of a target of
// `geometry` whose checkpoint stores its norms as `norms` says (normStorage),
// the norms as float32Norms holds them, the sections those its GGUF walk
// writes (model/GgufImage.hpp), in their order. The checkpoint outlives them:
// their rows read its tensors.
[[nodiscard]] std::vector<gguf::Image> planImages(const SafetensorsCheckpoint &checkpoint,
                                                  const QwenTargetDimensions &geometry, NormStorage norms,
                                                  gguf::PlanOptions options = {});

} // namespace splash::model::safetensors
