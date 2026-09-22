#include "FlashGDN.hpp"

#include "metal/abi/FlashGDN.h"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace splash::flash {
namespace {

uint64_t multiply(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::invalid_argument("Flash GDN byte extent overflows");
  return a * b;
}

void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes,
                  const char *name) {
  if (!buffer || buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash GDN insufficient ") + name);
}

void requireTensor(const FlashTensor *tensor, uint64_t bytes,
                   const char *name) {
  if (!tensor || tensor->dtype != FlashDType::BF16 ||
      tensor->logicalBytes < bytes)
    throw std::invalid_argument(std::string("Flash GDN invalid ") + name);
  requireBytes(tensor->buffer, bytes, name);
}

uint64_t stride(uint64_t requested, uint64_t tight, uint32_t alignment) {
  if (!requested)
    return tight;
  if (requested < tight || requested % alignment)
    throw std::invalid_argument("Flash GDN invalid state lane stride");
  return requested;
}

uint64_t stateExtent(uint32_t lanes, uint64_t laneStride, uint64_t tight) {
  const uint64_t offset = multiply(lanes - 1, laneStride);
  if (offset > std::numeric_limits<uint64_t>::max() - tight)
    throw std::invalid_argument("Flash GDN state extent overflows");
  return offset + tight;
}

void requireDistinct(const FlashGDNBuffers &b, const FlashGDNState &s) {
  // Each work buffer has an independent lifetime and format. The backend's
  // views must also be disjoint when allocating them within one base buffer.
  const std::array buffers{b.qkv, b.z, b.a, b.b, b.mixed, b.decay, b.beta,
                           b.recurrentRows, b.output, b.diagnostics,
                           s.convolution, s.recurrent};
  for (size_t i = 0; i < buffers.size(); ++i)
    for (size_t j = i + 1; j < buffers.size(); ++j)
      if (buffers[i].sameView(buffers[j]))
        throw std::invalid_argument("Flash GDN work/state buffers must differ");
}

} // namespace

void addGDN(metal::CommandGraph &graph, const FlashGDNWeights &w,
            const FlashGDNBuffers &b, const FlashGDNState &s, uint32_t rows,
            uint32_t lanes, float normEpsilon) {
  if (!rows || rows > kFlashGDNMaximumRows || !lanes || lanes > 32 ||
      !std::isfinite(normEpsilon) || normEpsilon <= 0.0f)
    throw std::invalid_argument("Flash GDN unsupported row/lane geometry");
  requireTensor(w.convolution, uint64_t{kFlashGDNConvolutionWidth} * 4 * 2,
                "convolution weights");
  if (w.convolution->shape !=
          std::vector<uint64_t>{kFlashGDNConvolutionWidth, 4, 1} &&
      w.convolution->shape !=
          std::vector<uint64_t>{kFlashGDNConvolutionWidth, 4})
    throw std::invalid_argument("Flash GDN convolution weight layout differs");
  for (const auto &[tensor, name] :
       std::array<std::pair<const FlashTensor *, const char *>, 2>{
           {{w.aLog, "A_log"}, {w.timeBias, "time bias"}}}) {
    requireTensor(tensor, uint64_t{kFlashGDNValueHeads} * 2, name);
    if (tensor->shape != std::vector<uint64_t>{kFlashGDNValueHeads})
      throw std::invalid_argument("Flash GDN gate weight layout differs");
  }
  requireTensor(w.norm, uint64_t{kFlashGDNHeadDimension} * 2, "direct norm");
  if (w.norm->shape != std::vector<uint64_t>{kFlashGDNHeadDimension})
    throw std::invalid_argument("Flash GDN norm weight layout differs");
  const uint64_t rowCount = multiply(rows, lanes);
  const uint64_t qkvBytes = multiply(multiply(rowCount,
                                  kFlashGDNConvolutionWidth), 2);
  const uint64_t valueBytes =
      multiply(multiply(rowCount, kFlashGDNOutputWidth), 2);
  const uint64_t gateBytes =
      multiply(multiply(rowCount, kFlashGDNValueHeads), 2);
  requireBytes(b.qkv, qkvBytes, "qkv input");
  requireBytes(b.mixed, qkvBytes, "mixed scratch");
  requireBytes(b.z, valueBytes, "z input");
  requireBytes(b.recurrentRows, valueBytes, "recurrent row scratch");
  requireBytes(b.output, valueBytes, "output");
  requireBytes(b.a, gateBytes, "a input");
  requireBytes(b.b, gateBytes, "b input");
  requireBytes(b.beta, gateBytes, "beta scratch");
  requireBytes(b.decay, multiply(gateBytes, 2), "decay scratch");
  requireBytes(b.diagnostics, sizeof(uint32_t), "diagnostics");
  const uint64_t convStride = stride(s.convolutionLaneStrideBytes,
                                    flashGDNConvolutionLaneBytes(), 2);
  const uint64_t recurrentStride = stride(s.recurrentLaneStrideBytes,
                                         flashGDNRecurrentLaneBytes(), 4);
  requireBytes(s.convolution,
               stateExtent(lanes, convStride, flashGDNConvolutionLaneBytes()),
               "convolution state");
  requireBytes(s.recurrent,
               stateExtent(lanes, recurrentStride,
                           flashGDNRecurrentLaneBytes()),
               "recurrent state");
  requireDistinct(b, s);

  const FlashGDNParams p{rows, lanes, kFlashGDNKeyHeads,
                        kFlashGDNValueHeads, kFlashGDNHeadDimension,
                        kFlashGDNHeadDimension, 4, normEpsilon, convStride,
                        recurrentStride};
  graph.add("flash_gdn_convolution", {b.qkv, w.convolution->buffer,
                                      s.convolution, b.mixed, b.diagnostics},
            p, {kFlashGDNConvolutionWidth / 256, rows, lanes}, {256, 1, 1});
  graph.add("flash_gdn_normalize_qk", {b.mixed, b.diagnostics}, p,
            {kFlashGDNKeyHeads, rows, lanes}, {32, 1, 1});
  graph.add("flash_gdn_gates", {b.a, b.b, w.aLog->buffer, w.timeBias->buffer,
                                b.decay, b.beta, b.diagnostics},
            p, {1, rows, lanes}, {64, 1, 1});
  // Four SIMD groups own four independent value dimensions. Each lane holds
  // four key dimensions and keeps its state in registers across all rows.
  graph.add("flash_gdn_recurrence", {b.mixed, b.decay, b.beta, s.recurrent,
                                     b.recurrentRows, b.diagnostics},
            p, {kFlashGDNValueHeads, kFlashGDNHeadDimension / 4, lanes},
            {32, 4, 1});
  graph.add("flash_gdn_output", {b.recurrentRows, b.z, w.norm->buffer,
                                 b.output, b.diagnostics},
            p, {kFlashGDNValueHeads, rows, lanes}, {32, 1, 1});
  // The input history is read by every convolution task before it is updated.
  graph.add("flash_gdn_convolution_carry", {b.qkv, s.convolution,
                                            b.diagnostics},
            p, {kFlashGDNConvolutionWidth / 256, lanes, 1}, {256, 1, 1});
}

} // namespace splash::flash
