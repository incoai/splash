#pragma once

// Plans the prepared images (model/GgufImageLayout.hpp) of a Qwen3.8 (qwen35)
// or Qwen3.6 MoE (qwen35moe) target read straight from a llama.cpp GGUF, from
// its metadata alone: section offsets, the header and descriptor bytes, and
// the source rows each tensor section is written from
// (model/GgufPreparation.hpp). A 3-D expert tensor is one quantized tensor of
// experts * N rows.

#include <cstdint>
#include <string>
#include <vector>

#include "model/GgufFile.hpp"

namespace splash::model::gguf {

struct TargetGeometry {
  uint32_t layers = 0;
  uint32_t hiddenSize = 0;
  uint32_t vocabularySize = 0;
  uint32_t intermediateSize = 0; // dense FFN
  uint32_t gdnKeyHeads = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t gdnHeadDimension = 0;
  uint32_t convolutionDimension = 0;
  uint32_t attentionWidth = 0;
  uint32_t attentionKvHeads = 0;
  uint32_t attentionHeadDimension = 0;
  // The rotated dimension pairs of each attention head and their RoPE base.
  uint32_t rotaryPairs = 0;
  float rotaryTheta = 0.0F;
  uint32_t fullAttentionPeriod = 0;
  // A sparse MoE FFN (qwen35moe) when experts is set; the shared expert has
  // the routed experts' intermediate width.
  uint32_t experts = 0;
  uint32_t expertsPerToken = 0;
  uint32_t expertIntermediateSize = 0;
  // Qwen3.8-Flash-Next (qwen4exp) when hyperConnections is set: the streams,
  // the mixes' rank, the QSA indexer and the PLE layer (pleLayer = layers
  // for none) with its heads of pleHeadDimension values and convolution taps.
  uint32_t hyperConnections = 0;
  uint32_t hyperRank = 0;
  uint32_t indexerHeads = 0;
  uint32_t indexerHeadDimension = 0;
  uint32_t indexerTokens = 0;
  uint32_t indexerBlockTokens = 0;
  uint32_t pleLayer = 0;
  uint32_t pleNgram = 0;
  uint32_t pleHeadsPerNgram = 0;
  uint32_t pleHeadDimension = 0;
  uint32_t pleConvolutionTaps = 0;
  [[nodiscard]] bool qwen4() const noexcept { return hyperConnections != 0; }
  [[nodiscard]] bool isFullAttentionLayer(uint32_t layer) const noexcept {
    return (layer + 1) % fullAttentionPeriod == 0;
  }
  [[nodiscard]] bool sparseMoe() const noexcept { return experts != 0; }
  // The general.architecture of a GGUF of this target.
  [[nodiscard]] const char *architecture() const noexcept {
    return qwen4() ? "qwen4exp" : sparseMoe() ? "qwen35moe" : "qwen35";
  }
};

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

// Rows [0, rows) of one source tensor in image order.
struct TensorRows {
  std::string name;
  uint32_t type = 0;   // ggml type
  uint64_t offset = 0; // in the file's tensor data
  uint64_t rows = 0;
  uint64_t rowBytes = 0;
  RowOrder order{};
  uint32_t file = 0;   // the file of a split GGUF it is in
};

// Header and descriptor bytes.
struct Fill {
  uint64_t offset = 0;
  std::vector<uint8_t> bytes;
};
// Rows written back to back as stored or, for F32 rows the kernels read as
// bf16, as the bf16 values they equal exactly, or for BF16 rows the kernels
// read as F32, as the F32 values they equal.
struct Copy {
  uint64_t destination = 0;
  TensorRows source;
  bool bfloat16 = false;
  bool float32 = false;
  // Q8_0 rows as the F32 values they equal (d * q, exact in F32): small
  // tensors a float kernel reads, as the MTP head's injection weights.
  bool dequantizeQ80 = false;
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
  uint32_t layer = 0;
  uint32_t type = 0;
  uint64_t bytes = 0;
  std::vector<Fill> fills;
  std::vector<Copy> copies;
  std::vector<Repack> repacks;
};

// The layers' images, then the head's and the embedding's, and for qwen4exp
// with a PLE layer the n-gram table's (ple.bin). Checks the
// architecture, the geometry the metadata declares, its rotary embedding and
// norms included, and each tensor's shape; throws GgufError naming every
// missing tensor and every tensor of a type this build cannot load.
[[nodiscard]] std::vector<Image> planImages(const GgufFile &file, const TargetGeometry &geometry);

// A qwen4exp MTP head's image (mtp.bin): its one full-attention block
// (blk.<layers>, the block's tensors as a target layer's), then the
// nextn tensors: eh_proj, enorm, hnorm and the head's mix. The geometry is
// the target's; the MTP file has no PLE.
[[nodiscard]] Image planMtpImage(const GgufFile &file, const TargetGeometry &geometry);

// The hash of a qwen4exp target's PLE n-gram embedding (llama.cpp
// qwen4exp.cpp llm_graph_input_qwen4exp_ple): per n-gram position the
// multiplier, per head its table rows' offset and count, and the table's rows.
struct PleHash {
  std::vector<uint64_t> multipliers;
  std::vector<uint32_t> headOffsets;
  std::vector<uint32_t> headVocabularies;
  uint64_t tableRows = 0;
  uint32_t eosToken = 0;
};
[[nodiscard]] PleHash readPleHash(const GgufFile &file, const TargetGeometry &geometry);

} // namespace splash::model::gguf
