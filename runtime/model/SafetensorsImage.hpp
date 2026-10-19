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
// F32 1 + w of a stored w (NormEncoding); the convolution and dt_bias are
// copied as stored, and the GDN decay is float(-exp(double(A_log))). The MoE
// router and the shared-expert gate, which the block kernels read in F32,
// become the F32 values of their quantization (or of their bf16 weights), and
// so do GDN alpha and beta unless both are quantized in one format.

#include "model/GgufImage.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/SafetensorsCheckpoint.hpp"

#include <optional>
#include <string>
#include <vector>

namespace splash::model::safetensors {

// The names a checkpoint's language-model modules take: MLX's (mlx-lm,
// mlx-vlm: language_model.model.*, the routed experts stacked under
// mlp.switch_mlp), or transformers' (as NVIDIA's Model Optimizer keeps them:
// model.language_model.*, lm_head, each routed expert a module mlp.experts.<e>
// of its own).
enum class ModuleNames : uint8_t { Mlx, Transformers };

// The names the checkpoint's tensors take: transformers' when it holds
// lm_head.weight (or compressed-tensors' NVFP4 lm_head.weight_packed).
[[nodiscard]] ModuleNames moduleNames(const SafetensorsCheckpoint &checkpoint);

// How a checkpoint stores the weights of its RMSNorms, which multiply by 1 + w
// for a zero-centered w: as the 1 + w they multiply by, or as w. The GDN's
// gated norm is not zero-centered: both store the weight it multiplies by.
enum class NormEncoding : uint8_t { Multiplier, Centered };

// How the checkpoint stores its RMSNorm weights. Transformers names store w.
// MLX names store either, which every layer's input norm tells apart: its
// multipliers stay near 1, so at most a quarter of a stored 1 + w's values are
// below 1/2 and at least three quarters of a stored w's. An input norm between
// the two, or input norms that disagree, are refused, naming the tensors. The
// other norms' multipliers run well above 1, where a stored w can look like a
// 1 + w, so they follow the input norms.
[[nodiscard]] NormEncoding normEncoding(const SafetensorsCheckpoint &checkpoint,
                                        const QwenTargetDimensions &geometry);

// Whether the images hold the norms as F32: the 1 + w of a stored w and the
// GDN's gated norm widened, rather than a stored 1 + w and the gated norm as
// stored, in bf16.
[[nodiscard]] constexpr bool float32Norms(NormEncoding norms) noexcept { return norms == NormEncoding::Centered; }

// How MLX quantizes a module, as it infers it from the module's codes
// [experts, rows, columns * bits / 32] (U32) and scales [experts, rows, columns
// / group size]: bf16 scales beside bf16 biases (affine), uint8 E8M0 exponents
// (mxfp4, 4 bits in groups of 32) or uint8 E4M3 scales (nvfp4, 4 bits in
// groups of 16). format is GGUF_FMT_COUNT for any other bits, group size or
// scales.
struct MlxQuantization {
  uint32_t format;
  uint64_t bits, group;
  gguf::SafetensorsSource tensors; // its codes, scales and an affine module's biases
};

// The quantization of an MLX module of rows x columns weights (of each of
// `experts`); none when the checkpoint holds its weight unquantized, without
// scales. Codes, scales or biases of other types or shapes are refused. The
// checkpoint outlives the tensors.
[[nodiscard]] std::optional<MlxQuantization> mlxQuantization(const SafetensorsCheckpoint &checkpoint,
                                                             const std::string &module, uint64_t rows,
                                                             uint64_t columns, uint64_t experts = 1);

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
