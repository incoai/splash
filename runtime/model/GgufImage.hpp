#pragma once

// Plans the images (model/GgufImageLayout.hpp) of a Qwen3.8 (qwen35)
// or Qwen3.6 MoE (qwen35moe) target read straight from a llama.cpp GGUF, from
// its metadata alone: section offsets, the header and descriptor bytes, and
// the source rows each tensor section is written from
// (model/GgufPreparation.hpp). A 3-D expert tensor is one quantized tensor of
// experts * N rows.

#include <cstdint>
#include <string>
#include <vector>

#include "model/GgufFile.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/QwenHybridLayout.hpp"

namespace splash::model::gguf {

// The general.architecture of a GGUF of a target with this FFN.
[[nodiscard]] constexpr const char *architecture(QwenFfnKind ffn) noexcept {
  return ffn == QwenFfnKind::SparseMoe ? "qwen35moe" : "qwen35";
}

// The order of a tensor's rows in the image. Rows below `from` keep their
// order; from there on, blocks of headRows rows are value heads, which
// llama.cpp stores tiled (value head of its key head * keyHeads + key head)
// and splash groups by key head (key head * valueHeadsPerKey + value head).
struct RowOrder {
  uint64_t from = UINT64_MAX; // UINT64_MAX: rows as stored
  uint32_t headRows = 0;
  uint32_t keyHeads = 0;
  uint32_t valueHeadsPerKey = 0;
};

// An MLX quantized tensor's codes and scales, and an affine one's biases,
// which the writer interleaves into the native rows of its format
// (metal/abi/QuantFormat.h): per group an affine tensor's bf16 scale, its
// bias and its codes, or an mxfp4 tensor's block_mxfp4, its E8M0 scale and
// codes (GGUF_FMT_MXFP4). Or a BF16 weight, which the writer quantizes into
// native af4g64 rows as MLX's affine quantization rounds it (a DFlash2
// draft's projections).
struct MlxSource {
  const SourceTensor *codes = nullptr, *scales = nullptr, *biases = nullptr;
  const SourceTensor *bfloat16 = nullptr;
};

// Rows [0, rows) of one source tensor in image order, read from `file`: rows
// of rowBytes bytes at `offset` of its tensor data, or, for an MLX tensor
// (mlx.codes or mlx.bfloat16 set), its format's native rows.
struct TensorRows {
  std::string name;
  uint32_t type = 0;   // ggml type, or QUANT_AFFINE_TYPE
  uint64_t offset = 0; // in the file's tensor data
  uint64_t rows = 0;
  uint64_t rowBytes = 0;
  RowOrder order{};
  const WeightSource *file = nullptr;
  MlxSource mlx{};
};

// Header and descriptor bytes.
struct Fill {
  uint64_t offset = 0;
  std::vector<uint8_t> bytes;
};
// How a copy writes each value: as stored, narrowed from F32 to the bf16
// value it equals exactly (rows the kernels read as bf16), widened from BF16
// to the F32 value it equals (rows the kernels read as F32), as the F32 decay
// -exp(A_log) of an MLX GDN's A_log, BF16 or F32, which float(-exp(double))
// rounds once, or as the F32 values of an MLX quantized tensor's rows (affine
// s * code + z, or mxfp4), which the kernels read unquantized (its MoE router
// and shared-expert gate, GDN alpha and beta of two formats).
enum class Conversion : uint8_t { None, NarrowToBfloat16, WidenToFloat32, Decay, DequantizeToFloat32 };
// Rows written back to back, each value converted as `conversion` says.
struct Copy {
  uint64_t destination = 0;
  TensorRows source;
  Conversion conversion = Conversion::None;
};
// Quantized rows repacked into the planes of their format; the rows of the
// sources in order, then zero rows up to `rows`.
struct Repack {
  uint32_t format = 0; // GGUF_FMT_*
  uint64_t rows = 0;
  uint64_t columns = 0;
  uint64_t plane0 = 0, plane1 = 0, meta = 0; // image offsets; plane1 when the format has one
  std::vector<TensorRows> sources;
};
struct Image {
  std::string name; // layer-N.bin, head.bin, embedding.bin
  std::string magic; // kGgufImageMagic
  uint32_t layer = 0;
  uint32_t type = 0;
  uint64_t bytes = 0;
  std::vector<Fill> fills;
  std::vector<Copy> copies;
  std::vector<Repack> repacks;
};

// The destination bytes of `sourceBytes` bytes of rows copied with
// `conversion`: halved when narrowed, doubled when widened (and for the decay
// of BF16 values), four per element when decoded to F32.
[[nodiscard]] uint64_t convertedBytes(const TensorRows &source, Conversion conversion, uint64_t sourceBytes);

// Lays out one image: its header block, then 16 KiB-aligned sections, each a
// descriptor, copied rows or a quantized tensor's planes. The planners of a
// GGUF target (planImages) and of an MLX target (model/MlxImage.hpp) build
// their images with it.
class ImageBuilder {
public:
  ImageBuilder(std::string name, uint32_t layer, uint32_t type);

  // A section of `bytes` bytes after the last one, and its offset.
  [[nodiscard]] uint64_t section(uint64_t bytes);
  // Rows copied into a section of their own, converted as `conversion` says.
  void copy(TensorRows source, Conversion conversion = Conversion::None);
  // Rows copied to `destination`, inside a section already laid out.
  void copyAt(uint64_t destination, TensorRows source, Conversion conversion);
  // The descriptor of a tensor of `type`, quantized in `format` (a float or
  // native-rows tensor has neither per-group nor meta bytes).
  void descriptor(uint32_t type, uint64_t rows, uint64_t columns, const QuantFormat &format,
                  const GgufPlaneBytes &bytes, const std::string &name);
  // The descriptor and planes of a [rows, columns] tensor in format, whose
  // sources the caller adds before repack() takes it.
  [[nodiscard]] Repack planes(uint32_t format, uint64_t rows, uint64_t columns, const std::string &name);
  void repack(Repack repack);
  [[nodiscard]] Image finish();

private:
  Image image_;
  uint64_t cursor_ = 0;
};

// The target geometry a GGUF's metadata declares, its architecture included,
// with the rotary embedding and norms the kernels compute: the RoPE base and
// rotated dimensions, the RMS epsilon and no RoPE scaling. planImages checks
// the file's first, and model-check the installer's copy before any weight
// download. Throws GgufError naming every mismatch.
void requireMetadata(const GgufMetadata &metadata, const QwenTargetDimensions &geometry);

// The layers' images, then the head's and the embedding's. Checks the
// metadata (requireMetadata) and each tensor's shape; throws GgufError naming
// every missing tensor and every tensor of a type this build cannot load.
[[nodiscard]] std::vector<Image> planImages(const GgufFile &file, const QwenTargetDimensions &geometry);

} // namespace splash::model::gguf
