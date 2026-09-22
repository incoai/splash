#include "FlashAffine.hpp"

#include "metal/abi/FlashAffine.h"

#include <array>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::flash {
namespace {

struct AffineMathProfile final {
  bool dense;
  bool expert;
};

bool readProfileSwitch(const char *name) {
  const char *value = std::getenv(name);
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}

const AffineMathProfile &mathProfile() {
  static const AffineMathProfile profile{
      readProfileSwitch("SPLASH_FLASH_QMV_F32"),
      readProfileSwitch("SPLASH_FLASH_EXPERT_QMV")};
  return profile;
}

} // namespace

bool flashAffineFastEnabled() { return mathProfile().dense; }
bool flashAffineExpertEnabled() { return mathProfile().expert; }

const char *flashAffineSemantics() {
  const bool dense = flashAffineFastEnabled();
  const bool expert = flashAffineExpertEnabled();
  if (dense && expert) return kFlashAffineCombinedSemantics;
  if (expert) return kFlashAffineExpertSemantics;
  return dense ? kFlashAffineFastSemantics : kFlashAffineSemantics;
}

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kOutputsPerGroup = kThreads / 32;

uint64_t multiply(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::invalid_argument("Flash affine byte extent overflows");
  return a * b;
}

uint64_t plus(uint64_t a, uint64_t b) {
  if (a > std::numeric_limits<uint64_t>::max() - b)
    throw std::invalid_argument("Flash affine byte extent overflows");
  return a + b;
}

void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes,
                  const char *name) {
  if (!bytes || !buffer || buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash affine insufficient ") + name);
}

void requireTensor(const FlashTensor *tensor, FlashDType dtype, uint64_t bytes,
                   const char *name) {
  if (!tensor || tensor->dtype != dtype || tensor->logicalBytes < bytes)
    throw std::invalid_argument(std::string("Flash affine invalid ") + name);
  requireBytes(tensor->buffer, bytes, name);
}

bool validBits(uint32_t bits) {
  return bits == 4 || bits == 5 || bits == 6 || bits == 8;
}

uint64_t extent(uint32_t experts, uint32_t outputs, uint64_t expertStride,
                uint64_t rowStride, uint64_t rowBytes) {
  const uint64_t matrixBytes = plus(multiply(outputs - 1, rowStride), rowBytes);
  if (rowStride < rowBytes || (experts > 1 && expertStride < matrixBytes))
    throw std::invalid_argument("Flash affine overlapping source strides");
  return plus(multiply(experts - 1, expertStride), matrixBytes);
}

void requireProjection(const FlashAffineProjection &p) {
  if (!p.experts || !p.outputSize || !p.inputSize || !validBits(p.bits) ||
      (p.groupSize != 32 && p.groupSize != 64 && p.groupSize != 128) ||
      p.inputSize % p.groupSize || p.parameterRowStrideBytes % 2 ||
      p.parameterExpertStrideBytes % 2)
    throw std::invalid_argument("Flash affine invalid projection geometry");
  const uint64_t packedBytes = (multiply(p.inputSize, p.bits) + 7) / 8;
  const uint64_t parameterBytes = multiply(p.inputSize / p.groupSize, 2);
  requireTensor(p.weights, FlashDType::U32,
                extent(p.experts, p.outputSize, p.weightExpertStrideBytes,
                       p.weightRowStrideBytes, packedBytes),
                "packed weights");
  const uint64_t coefficientExtent =
      extent(p.experts, p.outputSize, p.parameterExpertStrideBytes,
             p.parameterRowStrideBytes, parameterBytes);
  requireTensor(p.scales, FlashDType::BF16, coefficientExtent, "scales");
  requireTensor(p.biases, FlashDType::BF16, coefficientExtent, "biases");
}

FlashAffineParams params(const FlashAffineProjection &p, uint32_t rows,
                         uint32_t selections, uint32_t flags) {
  return {rows, selections, p.inputSize, p.outputSize, p.experts, p.bits,
          p.groupSize, flags, p.weightRowStrideBytes,
          p.weightExpertStrideBytes, p.parameterRowStrideBytes,
          p.parameterExpertStrideBytes};
}

void requireBuffers(const FlashAffineProjection &p, metal::MetalBuffer input,
                    metal::MetalBuffer output,
                    metal::MetalBuffer diagnostics, uint32_t rows,
                    uint32_t selections, bool inputPerSelection) {
  requireProjection(p);
  if (!rows || !selections || selections > p.experts)
    throw std::invalid_argument("Flash affine invalid row/selection count");
  const uint64_t inputRows = multiply(rows, inputPerSelection ? selections : 1);
  requireBytes(input, multiply(multiply(inputRows, p.inputSize), 2), "input");
  requireBytes(output, multiply(multiply(multiply(rows, selections),
                                         p.outputSize), 2), "output");
  requireBytes(diagnostics, sizeof(uint32_t), "diagnostics");
  if (input.sameView(output))
    throw std::invalid_argument("Flash affine input/output must be distinct");
}

struct AffineRoute final {
  const char *pipeline = "flash_affine_project";
  uint32_t columnsPerSimd = 1;
  uint32_t simdGroups = 8;
};

bool measuredDenseGeometry(const FlashAffineProjection &p) {
  if (p.experts != 1) return false;
  // Real source-weight B1/B8 comparisons were BF16 exact and all improved.
  // Geometry is {N,K,bits,group}; unmatched formats keep the generic control.
  constexpr std::array<std::array<uint32_t, 4>, 14> geometries{{
      {10240, 2560, 6, 64}, {10240, 2560, 4, 64},
      {10240, 2560, 5, 64}, {6144, 2560, 6, 64},
      {6144, 2560, 5, 128}, {2560, 6144, 5, 128},
      {48, 2560, 6, 64}, {48, 2560, 5, 128},
      {640, 2560, 8, 128}, {2560, 640, 8, 128},
      {12288, 2560, 4, 64}, {512, 2560, 4, 64},
      {2560, 6144, 4, 64}, {640, 2560, 4, 64}}};
  for (const auto &g : geometries)
    if (p.outputSize == g[0] && p.inputSize == g[1] &&
        p.bits == g[2] && p.groupSize == g[3])
      return true;
  return false;
}

AffineRoute route(const FlashAffineProjection &p, uint32_t rows,
                  bool selectedExperts = false, uint32_t selections = 1) {
  const bool expertQMV = flashAffineExpertEnabled();
  // Alternate math is confined to real-source-qualified dense Q4/Q5/Q6
  // GDN/QSA geometries. HC, all Q8 and selected experts retain their controls.
  if (flashAffineFastEnabled() && !selectedExperts && measuredDenseGeometry(p) &&
      (p.bits == 4 || p.bits == 5 || p.bits == 6)) {
    if (p.bits == 4)
      return {"flash_affine_mlx_qmv_f32xsum_v1_q4_g64", 4, 2};
    if (p.bits == 5 && p.groupSize == 64)
      return {"flash_affine_mlx_qmv_f32xsum_v1_q5_g64", 4, 2};
    if (p.bits == 5 && p.groupSize == 128)
      return {"flash_affine_mlx_qmv_f32xsum_v1_q5_g128", 4, 2};
    if (p.bits == 6)
      return {"flash_affine_mlx_qmv_f32xsum_v1_q6_g64", 4, 2};
  }
  // Qualified with the original checkpoint's selected expert planes at B1,
  // B8 and prefill32. The specialization keeps each lane's K/reduction order.
  const bool expertShape = p.experts == 512 &&
      ((p.outputSize == 640 && p.inputSize == 2560) ||
       (p.outputSize == 2560 && p.inputSize == 640));
  if (expertShape && p.bits == 4 && p.groupSize == 64) {
    const void *address = p.weights->buffer.contents();
    const bool aligned = address && reinterpret_cast<uintptr_t>(address) % 4 == 0 &&
        p.weightRowStrideBytes % 4 == 0 && p.weightExpertStrideBytes % 4 == 0;
    // Source-weight B1/B4 micro comparisons qualify an alternate explicit-F32
    // coefficient policy with contiguous lane-K assignment. Wider verification
    // rows require paired generation qualification under the same producer tag.
    if (expertQMV && selectedExperts && selections == 10 && rows && rows <= 16 && aligned) {
      if (p.outputSize == 640)
        return {"flash_expert_qmv_contig_k16_sg4_c2", 2, 4};
      return {"flash_expert_qmv_contig_k8_sg2_c4", 4, 2};
    }
    return {aligned ? "flash_affine_q4_g64_u32_c1"
                    : "flash_affine_q4_g64_c1", 1};
  }
  // The measured vocabulary projection is single-row. Other Q8 projections
  // and row counts retain the generic control until separately qualified.
  if (rows == 1 && p.experts == 1 && p.bits == 8 && p.groupSize == 64 &&
      p.outputSize == 248320 && p.inputSize == 2560)
    return {"flash_affine_q8_g64_c2", 2};
  // Original Q4/Q5/Q6/Q8 HC planes passed exact source-weight comparisons and
  // balanced timing at B1/B8, with down additionally measured at prefill32.
  const bool hcShape = p.experts == 1 &&
      ((p.outputSize == 320 && p.inputSize == 10240) ||
       (p.outputSize == 10240 && p.inputSize == 320) ||
       (p.outputSize == 4 && p.inputSize == 10240));
  const bool sharedGate = p.experts == 1 && p.bits == 8 &&
      p.outputSize == 1 && p.inputSize == 2560;
  if (p.groupSize == 64 && (hcShape || sharedGate)) {
    if (p.bits == 4) return {"flash_affine_q4_g64_grouped_c1", 1};
    if (p.bits == 5) return {"flash_affine_q5_g64_c1", 1};
    if (p.bits == 6) return {"flash_affine_q6_g64_c1", 1};
    if (p.bits == 8) return {"flash_affine_q8_g64_c1", 1};
  }
  if (measuredDenseGeometry(p)) {
    if (p.bits == 4) return {"flash_affine_q4_g64_grouped_c1", 1};
    if (p.bits == 5 && p.groupSize == 64)
      return {"flash_affine_q5_g64_c1", 1};
    if (p.bits == 5 && p.groupSize == 128)
      return {"flash_affine_q5_g128_c1", 1};
    if (p.bits == 6) return {"flash_affine_q6_g64_c1", 1};
    if (p.bits == 8) return {"flash_affine_q8_g128_c1", 1};
  }
  return {};
}

} // namespace

void addAffine(metal::CommandGraph &graph, metal::MetalBuffer input,
               const FlashAffineProjection &p, metal::MetalBuffer output,
               metal::MetalBuffer diagnostics, uint32_t rows) {
  requireBuffers(p, input, output, diagnostics, rows, 1, false);
  if (p.experts != 1)
    throw std::invalid_argument("Flash affine dense projection has experts");
  const auto selected = route(p, rows);
  // The expert-ID binding is a valid unused dummy in this mode.
  graph.add(selected.pipeline, {input, p.weights->buffer, p.scales->buffer,
                                     p.biases->buffer, input, output,
                                     diagnostics},
            params(p, rows, 1, 0),
            {(p.outputSize - 1) / (selected.simdGroups * selected.columnsPerSimd) + 1,
             rows, 1},
            {selected.simdGroups * 32, 1, 1});
}

void addGatheredAffine(metal::CommandGraph &graph, metal::MetalBuffer input,
                       const FlashAffineProjection &p,
                       metal::MetalBuffer expertIDs, metal::MetalBuffer output,
                       metal::MetalBuffer diagnostics, uint32_t rows,
                       uint32_t selections, bool inputPerSelection) {
  requireBuffers(p, input, output, diagnostics, rows, selections,
                 inputPerSelection);
  requireBytes(expertIDs, multiply(multiply(rows, selections), sizeof(int64_t)),
               "expert IDs");
  for (const auto &source : {input, expertIDs, diagnostics, p.weights->buffer,
                            p.scales->buffer, p.biases->buffer})
    if (output.sameView(source))
      throw std::invalid_argument("Flash gathered affine output/source alias");
  for (const auto &source : {input, expertIDs, p.weights->buffer,
                            p.scales->buffer, p.biases->buffer})
    if (diagnostics.sameView(source))
      throw std::invalid_argument("Flash gathered affine status/source alias");
  const auto selected = route(p, rows, true, selections);
  graph.add(selected.pipeline, {input, p.weights->buffer, p.scales->buffer,
                                     p.biases->buffer, expertIDs, output,
                                     diagnostics},
            params(p, rows, selections, 1u | (inputPerSelection ? 2u : 0u)),
            {(p.outputSize - 1) / (selected.simdGroups * selected.columnsPerSimd) + 1,
             rows, selections},
            {selected.simdGroups * 32, 1, 1});
}

void addDenseBF16(metal::CommandGraph &graph, metal::MetalBuffer input,
                  const FlashTensor &weights, metal::MetalBuffer output,
                  metal::MetalBuffer diagnostics, uint32_t rows) {
  if (!rows || weights.shape.size() != 2 || !weights.shape[0] ||
      !weights.shape[1] || weights.shape[0] > UINT32_MAX ||
      weights.shape[1] > UINT32_MAX)
    throw std::invalid_argument("Flash dense invalid matrix geometry");
  const auto n = static_cast<uint32_t>(weights.shape[0]);
  const auto k = static_cast<uint32_t>(weights.shape[1]);
  const uint64_t rowBytes = multiply(k, 2);
  requireTensor(&weights, FlashDType::BF16, multiply(n, rowBytes),
                "dense weights");
  requireBytes(input, multiply(rows, rowBytes), "dense input");
  requireBytes(output, multiply(multiply(rows, n), 2), "dense output");
  requireBytes(diagnostics, sizeof(uint32_t), "diagnostics");
  if (input.sameView(output))
    throw std::invalid_argument("Flash dense input/output must be distinct");
  graph.add("flash_dense_bf16_project", {input, weights.buffer, output,
                                         diagnostics},
            FlashDenseParams{rows, k, n, 0, rowBytes},
            {(n - 1) / kOutputsPerGroup + 1, rows, 1}, {kThreads, 1, 1});
}

void addAffineEmbedding(metal::CommandGraph &graph,
                        const FlashAffineProjection &p,
                        metal::MetalBuffer tokenIDs, metal::MetalBuffer output,
                        metal::MetalBuffer diagnostics, uint32_t rows) {
  requireProjection(p);
  if (!rows || p.experts != 1)
    throw std::invalid_argument("Flash embedding invalid geometry");
  requireBytes(tokenIDs, multiply(rows, sizeof(int64_t)), "token IDs");
  requireBytes(output, multiply(multiply(rows, p.inputSize), 2),
               "embedding output");
  requireBytes(diagnostics, sizeof(uint32_t), "diagnostics");
  graph.add("flash_affine_embedding", {tokenIDs, p.weights->buffer,
                                       p.scales->buffer, p.biases->buffer,
                                       output, diagnostics},
            FlashEmbeddingParams{rows, p.inputSize, p.outputSize, p.bits,
                                  p.groupSize, 0, p.weightRowStrideBytes,
                                  p.parameterRowStrideBytes},
            {(p.inputSize - 1) / kThreads + 1, rows, 1}, {kThreads, 1, 1});
}

uint32_t unpackAffineCode(std::span<const std::byte> packedRow, uint32_t bits,
                          uint64_t channel) {
  if (!validBits(bits))
    throw std::invalid_argument("Flash affine unsupported packing bit width");
  const uint64_t bitOffset = multiply(channel, bits);
  const uint64_t byteOffset = bitOffset / 8;
  const auto shift = static_cast<uint32_t>(bitOffset % 8);
  if (byteOffset >= packedRow.size() ||
      (shift + bits > 8 && byteOffset + 1 >= packedRow.size()))
    throw std::out_of_range("Flash affine packing reference exceeds row");
  uint32_t word = std::to_integer<uint8_t>(packedRow[byteOffset]);
  if (shift + bits > 8)
    word |= uint32_t{std::to_integer<uint8_t>(packedRow[byteOffset + 1])} << 8;
  return (word >> shift) & ((1u << bits) - 1);
}

} // namespace splash::flash
