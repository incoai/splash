#include "FlashGDNStaged.hpp"

#include "metal/abi/FlashGDN.h"

#include <stdexcept>

namespace splash::flash {

void addGDNStagedPrefill(metal::CommandGraph &graph,
                        const FlashGDNWeights &w, const FlashGDNBuffers &b,
                        const FlashGDNState &s, uint32_t rows, uint32_t lanes,
                        FlashGDNStageTile tile, float normEpsilon) {
  const char *pipeline = nullptr;
  uint32_t values = 0;
  switch (tile) {
  case FlashGDNStageTile::Values8Time16:
    pipeline = "flash_gdn_staged_v8_t16"; values = 8; break;
  case FlashGDNStageTile::Values8Time32:
    pipeline = "flash_gdn_staged_v8_t32"; values = 8; break;
  case FlashGDNStageTile::Values16Time16:
    pipeline = "flash_gdn_staged_v16_t16"; values = 16; break;
  case FlashGDNStageTile::Values16Time32:
    pipeline = "flash_gdn_staged_v16_t32"; values = 16; break;
  default:
    throw std::invalid_argument("Flash GDN unsupported staged tile");
  }
  metal::CommandGraph validated;
  addGDN(validated, w, b, s, rows, lanes, normEpsilon);
  const FlashGDNParams p{rows, lanes, 16, 48, 128, 128, 4, normEpsilon,
                        s.convolutionLaneStrideBytes
                            ? s.convolutionLaneStrideBytes : flashGDNConvolutionLaneBytes(),
                        s.recurrentLaneStrideBytes
                            ? s.recurrentLaneStrideBytes : flashGDNRecurrentLaneBytes()};
  graph.add("flash_gdn_fused_prepare", {b.qkv, b.a, b.b, w.convolution->buffer,
                                        w.aLog->buffer, w.timeBias->buffer,
                                        s.convolution, b.mixed, b.decay,
                                        b.beta, b.diagnostics},
            p, {16, rows, lanes}, {128, 1, 1});
  graph.add(pipeline, {b.mixed, b.decay, b.beta, s.recurrent,
                        b.recurrentRows, b.diagnostics},
            p, {48, 128 / values, lanes}, {values * 32, 1, 1});
  graph.add("flash_gdn_output", {b.recurrentRows, b.z, w.norm->buffer,
                                 b.output, b.diagnostics},
            p, {48, rows, lanes}, {32, 1, 1});
  graph.add("flash_gdn_convolution_carry", {b.qkv, s.convolution, b.diagnostics},
            p, {40, lanes, 1}, {256, 1, 1});
}

} // namespace splash::flash
