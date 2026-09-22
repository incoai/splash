#include "FlashGDNSeparate.hpp"

#include "metal/abi/FlashGDNSeparate.h"

#include <stdexcept>
#include <string>

namespace splash::flash {
namespace {

void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes,
                  const char *name) {
  if (!buffer || buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash GDN separate insufficient ") + name);
}

} // namespace

void addGDNFusedSeparateStates(
    metal::CommandGraph &graph, const FlashGDNWeights &w,
    const FlashGDNBuffers &b, const std::array<FlashGDNState, 4> &states,
    uint32_t lanes, float normEpsilon) {
  if (!lanes || lanes > 4)
    throw std::invalid_argument("Flash GDN separate requires1..4one-token lanes");
  // Preserve the frozen route's complete weight/state/alias validation. Its
  // temporary one-lane plans are never submitted; the actual route has only
  // two dispatches and directly binds all live request states.
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    metal::CommandGraph validated;
    addGDN(validated, w, b, states[lane], 1, 1, normEpsilon);
    for (uint32_t other = 0; other < lane; ++other)
      if (states[lane].convolution.sameView(states[other].convolution) ||
          states[lane].convolution.sameView(states[other].recurrent) ||
          states[lane].recurrent.sameView(states[other].convolution) ||
          states[lane].recurrent.sameView(states[other].recurrent))
        throw std::invalid_argument("Flash GDN separate request states must differ");
  }
  const uint64_t qkvBytes = uint64_t{lanes} * kFlashGDNConvolutionWidth * 2;
  const uint64_t valueBytes = uint64_t{lanes} * kFlashGDNOutputWidth * 2;
  const uint64_t gateBytes = uint64_t{lanes} * kFlashGDNValueHeads * 2;
  requireBytes(b.qkv, qkvBytes, "qkv plane");
  requireBytes(b.mixed, qkvBytes, "mixed plane");
  requireBytes(b.z, valueBytes, "z plane");
  requireBytes(b.recurrentRows, valueBytes, "recurrent rows plane");
  requireBytes(b.output, valueBytes, "output plane");
  requireBytes(b.a, gateBytes, "a plane");
  requireBytes(b.b, gateBytes, "b plane");
  requireBytes(b.beta, gateBytes, "beta plane");
  requireBytes(b.decay, gateBytes * 2, "decay plane");
  auto state = [&](uint32_t lane) -> const FlashGDNState & {
    return states[lane < lanes ? lane : 0];
  };
  const FlashGDNSeparateParams p{lanes, normEpsilon};
  graph.add("flash_gdn_fused_separate_sg16",
            {b.qkv, b.z, b.a, b.b, w.convolution->buffer, w.aLog->buffer,
             w.timeBias->buffer, w.norm->buffer,
             state(0).convolution, state(1).convolution,
             state(2).convolution, state(3).convolution,
             state(0).recurrent, state(1).recurrent,
             state(2).recurrent, state(3).recurrent,
             b.mixed, b.decay, b.beta, b.recurrentRows, b.output, b.diagnostics},
            p, {48, 1, lanes}, {512, 1, 1});
  graph.add("flash_gdn_fused_separate_carry",
            {b.qkv, state(0).convolution, state(1).convolution,
             state(2).convolution, state(3).convolution, b.diagnostics},
            p, {kFlashGDNConvolutionWidth / 256, 1, lanes}, {256, 1, 1});
}

} // namespace splash::flash
