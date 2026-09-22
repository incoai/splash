#include "FlashGDNFused.hpp"
#include "FlashMTPWindow.hpp"

#include "metal/abi/FlashGDN.h"
#include "metal/abi/FlashGDNFused.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace splash::flash {

void addGDNFused(metal::CommandGraph &graph, const FlashGDNWeights &w,
                 const FlashGDNBuffers &b, const FlashGDNState &s,
                 uint32_t rows, uint32_t lanes, FlashGDNFusion fusion,
                 float normEpsilon) {
  if (fusion != FlashGDNFusion::Prepare &&
      fusion != FlashGDNFusion::PersistentHead &&
      fusion != FlashGDNFusion::PersistentHead512 &&
      fusion != FlashGDNFusion::PersistentHead1024)
    throw std::invalid_argument("Flash GDN unsupported fusion route");
  // Reuse all qualified host layout/extent/alias checks. This temporary list
  // is never submitted and can be factored into a common validator after GPU
  // qualification; it deliberately does not alter the frozen normal route.
  metal::CommandGraph validated;
  addGDN(validated, w, b, s, rows, lanes, normEpsilon);
  const uint64_t convStride = s.convolutionLaneStrideBytes
                                  ? s.convolutionLaneStrideBytes
                                  : flashGDNConvolutionLaneBytes();
  const uint64_t recurrentStride = s.recurrentLaneStrideBytes
                                       ? s.recurrentLaneStrideBytes
                                       : flashGDNRecurrentLaneBytes();
  const FlashGDNParams p{rows, lanes, kFlashGDNKeyHeads,
                        kFlashGDNValueHeads, kFlashGDNHeadDimension,
                        kFlashGDNHeadDimension, 4, normEpsilon, convStride,
                        recurrentStride};
  if (fusion == FlashGDNFusion::Prepare) {
    graph.add("flash_gdn_fused_prepare", {b.qkv, b.a, b.b,
                                          w.convolution->buffer, w.aLog->buffer,
                                          w.timeBias->buffer, s.convolution,
                                          b.mixed, b.decay, b.beta,
                                          b.diagnostics},
              p, {kFlashGDNKeyHeads, rows, lanes}, {128, 1, 1});
    graph.add("flash_gdn_recurrence", {b.mixed, b.decay, b.beta, s.recurrent,
                                       b.recurrentRows, b.diagnostics},
              p, {kFlashGDNValueHeads, kFlashGDNHeadDimension / 4, lanes},
              {32, 4, 1});
    graph.add("flash_gdn_output", {b.recurrentRows, b.z, w.norm->buffer,
                                   b.output, b.diagnostics},
              p, {kFlashGDNValueHeads, rows, lanes}, {32, 1, 1});
  } else {
    const uint32_t threads = fusion == FlashGDNFusion::PersistentHead512 ? 512
                              : fusion == FlashGDNFusion::PersistentHead1024 ? 1024 : 256;
    const char *pipeline = threads == 512 ? "flash_gdn_fused_persistent_sg16"
                            : threads == 1024 ? "flash_gdn_fused_persistent_sg32"
                                             : "flash_gdn_fused_persistent";
    graph.add(pipeline, {b.qkv, b.z, b.a, b.b,
                                             w.convolution->buffer,
                                             w.aLog->buffer, w.timeBias->buffer,
                                             w.norm->buffer, s.convolution,
                                             s.recurrent, b.mixed, b.decay,
                                             b.beta, b.recurrentRows, b.output,
                                             b.diagnostics},
              p, {kFlashGDNValueHeads, lanes, 1}, {threads, 1, 1});
  }
  graph.add("flash_gdn_convolution_carry", {b.qkv, s.convolution,
                                            b.diagnostics},
            p, {kFlashGDNConvolutionWidth / 256, lanes, 1}, {256, 1, 1});
}

void addGDNFusedCaptured(metal::CommandGraph &graph,
                         const FlashGDNWeights &w, const FlashGDNBuffers &b,
                         const FlashGDNState &s, const FlashGDNCapture &capture,
                         uint32_t rows, uint32_t lanes, float normEpsilon) {
  if (!rows || rows > kFlashSingletonMaximumVerifyRows || !lanes || lanes > 32)
    throw std::invalid_argument("Flash GDN capture requires rows1..16");
  if (capture.threadsPerHead != 256 && capture.threadsPerHead != 512 &&
      capture.threadsPerHead != 1024)
    throw std::invalid_argument("Flash GDN unsupported capture thread count");
  metal::CommandGraph validated;
  addGDN(validated, w, b, s, rows, lanes, normEpsilon);
  const uint32_t captureRows = capture.capturedRows == 0xffffffffu
                                   ? rows : capture.capturedRows;
  if (captureRows > rows)
    throw std::invalid_argument("Flash GDN capture count exceeds consumed rows");
  const uint64_t tight = flashGDNRecurrentLaneBytes();
  const uint64_t rowStride = capture.rowStrideBytes ? capture.rowStrideBytes : tight;
  if (rowStride < tight || rowStride % 4 ||
      rowStride > std::numeric_limits<uint64_t>::max() / rows)
    throw std::invalid_argument("Flash GDN capture invalid row stride");
  const uint64_t laneStride = capture.laneStrideBytes
                                   ? capture.laneStrideBytes : captureRows * rowStride;
  if (laneStride < captureRows * rowStride || laneStride % 4 ||
      laneStride > (std::numeric_limits<uint64_t>::max() - captureRows * rowStride) /
                       lanes)
    throw std::invalid_argument("Flash GDN capture invalid lane stride");
  const uint64_t bytes = captureRows
      ? (lanes - 1) * laneStride + (captureRows - 1) * rowStride + tight : 0;
  if (captureRows &&
      (!capture.recurrentStates || capture.recurrentStates.sizeBytes() < bytes))
    throw std::invalid_argument("Flash GDN capture insufficient tape");
  for (const auto &buffer : std::array{b.qkv, b.z, b.a, b.b, b.mixed, b.decay,
                                       b.beta, b.recurrentRows, b.output,
                                       b.diagnostics, s.convolution, s.recurrent})
    if (captureRows && capture.recurrentStates.sameView(buffer))
      throw std::invalid_argument("Flash GDN capture tape must be separate");
  const FlashGDNParams p{rows, lanes, 16, 48, 128, 128, 4, normEpsilon,
                        s.convolutionLaneStrideBytes
                            ? s.convolutionLaneStrideBytes : flashGDNConvolutionLaneBytes(),
                        s.recurrentLaneStrideBytes
                            ? s.recurrentLaneStrideBytes : flashGDNRecurrentLaneBytes()};
  const char *pipeline = capture.threadsPerHead == 512
      ? "flash_gdn_fused_persistent_capture_sg16"
      : capture.threadsPerHead == 1024
          ? "flash_gdn_fused_persistent_capture_sg32"
          : "flash_gdn_fused_persistent_capture";
  graph.add(pipeline,
            {b.qkv, b.z, b.a, b.b, w.convolution->buffer, w.aLog->buffer,
             w.timeBias->buffer, w.norm->buffer, s.convolution, s.recurrent,
             b.mixed, b.decay, b.beta, b.recurrentRows, b.output, b.diagnostics,
             captureRows ? capture.recurrentStates : s.recurrent},
            FlashGDNCaptureParams{p, rowStride, laneStride, captureRows, 0},
            {48, lanes, 1}, {capture.threadsPerHead, 1, 1});
  graph.add("flash_gdn_convolution_carry", {b.qkv, s.convolution, b.diagnostics},
            p, {kFlashGDNConvolutionWidth / 256, lanes, 1}, {256, 1, 1});
}

} // namespace splash::flash
