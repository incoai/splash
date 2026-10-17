#pragma once

// Plans the images (model/GgufImageLayout.hpp) of a Qwen3.8 (qwen35)
// or Qwen3.6 MoE (qwen35moe) target read straight from a llama.cpp GGUF, from
// its metadata alone: section offsets, the header and descriptor bytes, and
// the source rows each tensor section is written from
// (model/GgufPreparation.hpp). A 3-D expert tensor is one quantized tensor of
// experts * N rows.

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "model/GgufFile.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/QwenHybridLayout.hpp"

namespace splash::model::gguf {

// `rows` rows and the zero rows that pad them to whole plane tiles.
[[nodiscard]] constexpr uint64_t tiledRows(uint64_t rows) noexcept {
  return (rows + QUANT_TILE_ROWS - 1) / QUANT_TILE_ROWS * QUANT_TILE_ROWS;
}

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

// A quantized safetensors tensor's codes and scales, and an affine one's
// biases, which the writer interleaves into the native rows of its format
// (metal/abi/QuantFormat.h): per group an affine tensor's bf16 scale, its
// bias and its codes, or an mxfp4 tensor's block_mxfp4, its E8M0 scale and
// codes (GGUF_FMT_MXFP4); per 256 elements an nvfp4 tensor's E4M3 scales, its
// row's tensor scale (TensorScale) and codes, or an fp8 tensor's row's tensor
// scale and E4M3 values. Or a BF16 weight, which the writer quantizes into
// native af4g64 rows as MLX's affine quantization rounds it (a DFlash2
// draft's projections).
struct SafetensorsSource {
  const SourceTensor *codes = nullptr, *scales = nullptr, *biases = nullptr;
  const SourceTensor *bfloat16 = nullptr;
};
// Whether a quantized safetensors tensor may be in a format: MLX affine,
// mxfp4, nvfp4 or fp8.
[[nodiscard]] constexpr bool safetensorsFormat(uint32_t format) {
  return quant_loader_format(format) || format == GGUF_FMT_MXFP4;
}

// The tensor scale g of an nvfp4 or fp8 tensor's rows: the F32 or BF16 values
// at `offset` of file's tensor data (`bytes` long), one for every
// rowsPerValue rows, or their reciprocals. Model Optimizer's weight_scale_2 or
// per-tensor fp8 weight_scale holds the tensor's g, compressed-tensors'
// weight_global_scale its 1 / g and its per-channel fp8 weight_scale each
// row's g [rows, 1], and a llama.cpp NVFP4 projection's .scale its g or
// each expert's. Without one (rowsPerValue 0), as in MLX's nvfp4, g is 1.
struct TensorScale {
  const WeightSource *file = nullptr;
  uint64_t offset = 0, bytes = 0;
  uint64_t rowsPerValue = 0;
  bool bfloat16 = false, reciprocal = false;
};

// Rows [first, first + rows) of one source tensor in image order, read from
// `file`: rows of rowBytes bytes at `offset` of its tensor data, or its
// format's native rows, which the writer builds from a quantized safetensors
// tensor's tensors (safetensors.codes or safetensors.bfloat16 set) or from
// llama.cpp's NVFP4 rows (ggufNvfp4RowBytes set: rows of block_nvfp4 that long
// from `offset` on).
struct TensorRows {
  std::string name;
  uint32_t type = 0;   // ggml type, or a loader's (metal/abi/QuantFormat.h)
  uint64_t offset = 0; // in the file's tensor data
  uint64_t rows = 0;
  uint64_t rowBytes = 0;
  RowOrder order{};
  const WeightSource *file = nullptr;
  SafetensorsSource safetensors{};
  TensorScale scale{};
  uint64_t ggufNvfp4RowBytes = 0;
  uint64_t first = 0; // the source row of image row 0
};

// Header and descriptor bytes.
struct Fill {
  uint64_t offset = 0;
  std::vector<uint8_t> bytes;
};
// How a copy writes each value: as stored, narrowed from F32 to the bf16
// value it equals exactly (rows the kernels read as bf16), widened from BF16
// or F16 to the F32 value it equals (rows the kernels read as F32), as the F32
// decay -exp(A_log) of a safetensors GDN's A_log, BF16 or F32, which
// float(-exp(double)) rounds once, as the F32 values of a quantized tensor's
// native rows, which the GPU decodes as the format's gather does
// (kernels/common/gguf_embedding_formats.h) and the kernels read unquantized
// (a safetensors MoE router and shared-expert gate, GDN alpha and beta of two
// types or of F16, and a GGUF's Q8_0 shared-expert scalar gate), or as the F32
// 1 + w of a BF16 RMSNorm weight w, which transformers stores 1 below the
// weight the norm multiplies by (exact but where |w| < 2^-16 or |w| >= 2^24,
// which it may round once).
enum class Conversion : uint8_t { None, NarrowToBfloat16, WidenToFloat32, Decay, DequantizeToFloat32, CenteredNorm };
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
// A layer's routed experts when they stream from their source instead of
// loading with the layer (PlanOptions::streamExperts, model/ExpertStore.hpp):
// the repacks of gate, up and down, each of experts * N rows and of no image
// section, whose rows the store reads an expert at a time.
struct StreamedExperts {
  std::array<Repack, 3> projections; // gate, up, down
};
// Where a table's rows are in its GGUF (tableRows), for reading them from the
// file (model/SparseRowStore.hpp) instead of loading the table's image: the
// rows exactly as the image would hold them.
struct StreamedRows final {
  std::string tensor;
  uint32_t file = 0;   // the part of a split GGUF that holds it
  uint64_t offset = 0; // of row 0, from the start of that file
  uint64_t rows = 0;
  uint32_t rowBytes = 0;
  uint32_t format = 0; // its gguf_embedding_format (GGUF_FMT_*)
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
  // A MoE layer's routed experts, when they stream: then not in the image.
  std::optional<StreamedExperts> experts;
  // A table's rows, when they stream (PlanOptions::streamTable): then the
  // image does not load, and stands in the weights' records as when it does,
  // as the rows read are its rows.
  std::optional<StreamedRows> streamedRows;
};

// How a target's images are planned: with streamExperts, every MoE layer's
// routed experts stream (Image::experts) instead of loading with it; with
// streamTable, the rows of a table the family's GGUF holds after the
// embedding (Qwen3.8-Flash-Next's PLE n-gram table) stream
// (Image::streamedRows) instead of loading as its image.
struct PlanOptions {
  bool streamExperts = false;
  bool streamTable = false;
};

// The routed experts each layer's image streams (Image::experts), in layer
// order; throws unless every layer's do.
[[nodiscard]] std::vector<StreamedExperts> streamedExperts(std::span<const Image> layers);
// The bytes of the planes of the routed experts an image streams
// (Image::experts), 0 for none.
[[nodiscard]] uint64_t streamedExpertBytes(const Image &image);
// The bytes of the table rows an image streams (Image::streamedRows), 0 for
// none.
[[nodiscard]] uint64_t streamedTableBytes(const Image &image);
// The bytes an image takes once loaded: none for a table whose rows stream.
[[nodiscard]] uint64_t loadedBytes(const Image &image);

// The destination bytes of `sourceBytes` bytes of rows copied with
// `conversion`: halved when narrowed, doubled when widened (and for the decay
// of BF16 values and a centered norm), four per element when decoded to F32.
[[nodiscard]] uint64_t convertedBytes(const TensorRows &source, Conversion conversion, uint64_t sourceBytes);

// Lays out one image: its header block, then 16 KiB-aligned sections, each a
// descriptor, copied rows or a quantized tensor's planes. The planners of a
// GGUF target (planImages) and of a safetensors target
// (model/SafetensorsImage.hpp) build their images with it.
class ImageBuilder {
public:
  // With streamExperts, routed experts stream (expertPlanes).
  ImageBuilder(std::string name, uint32_t layer, uint32_t type, bool streamExperts = false);

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
  // A layer's routed experts' projection of experts * N rows, gate, up and
  // down in this order: as planes() and repack() lay out a tensor, or, when
  // the builder streams experts, a repack of no image section, which
  // expertRepack() keeps in Image::experts.
  [[nodiscard]] Repack expertPlanes(uint32_t format, uint64_t rows, uint64_t columns, const std::string &name);
  void expertRepack(Repack repack);
  [[nodiscard]] Image finish();

private:
  Image image_;
  uint64_t cursor_ = 0;
  bool streamExperts_ = false;
  // The routed experts' projections expertRepack() took.
  uint32_t expertProjections_ = 0;
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
[[nodiscard]] std::vector<Image> planImages(const GgufFile &file, const QwenTargetDimensions &geometry,
                                            PlanOptions options = {});

// Where the GGUF holds the rows of the tensor `name`, a table of `columns`
// values a row whose image copies its native rows as stored (a family's
// table), for reading them from the file. Throws GgufError for a tensor of a
// type its image does not copy as stored, of another width or of a rotated
// GGUF.
[[nodiscard]] StreamedRows tableRows(const GgufFile &file, const std::string &name, uint64_t columns);

} // namespace splash::model::gguf
