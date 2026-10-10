#include "ops/GDN.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

static_assert(offsetof(GDNDecodeBatchParams, conv_layer_bytes) == 8);

enum class KernelLayout : uint8_t { Value48, Value32 };

[[nodiscard]] KernelLayout kernelShape(const GdnShape &shape) {
  if (!shape.valid())
    throw std::invalid_argument("invalid GDN shape");
  if (shape == GdnShape{16, 48, 128, 10240, 16640})
    return KernelLayout::Value48;
  if (shape == GdnShape{16, 32, 128, 8192, 12544})
    return KernelLayout::Value32;
  throw std::invalid_argument("unsupported compiled GDN shape");
}

[[nodiscard]] const char *kernelName(KernelLayout shape,
                                     const char *value48,
                                     const char *value32) noexcept {
  return shape == KernelLayout::Value48 ? value48 : value32;
}

uint64_t valueWidth(const GdnShape &shape) { return uint64_t{shape.valueHeads} * shape.headDimension; }
// One layer's carried convolution rows and fp32 recurrent state, a head
// dimension square per value head.
uint64_t carriedBytes(const GdnShape &shape) {
  return uint64_t{SPLASH_GDN_CONVOLUTION_TAPS - 1} * shape.convolutionDimension * 2;
}
uint64_t recurrentBytes(const GdnShape &shape) { return valueWidth(shape) * shape.headDimension * sizeof(float); }

// The bytes of `rows` rows of the packed projection the kernels read: each
// row's q, k and v convolution inputs, z, then beta and alpha per value head.
uint64_t packedBytes(const GdnShape &shape, uint64_t rows) {
  return rowBytes(rows, shape.packedWidth, shape.convolutionDimension + valueWidth(shape) + 2 * shape.valueHeads, 2);
}

// The convolution weights of every channel and a_scale and dt_bias of every
// value head.
void requireMixerWeights(const GdnShape &shape, const metal::MetalBuffer &convolution,
                         const metal::MetalBuffer &decay, const metal::MetalBuffer &timeBias) {
  requireBytes(convolution, uint64_t{shape.convolutionDimension} * SPLASH_GDN_CONVOLUTION_TAPS * 2,
               "GDN convolution weight");
  requireBytes(decay, uint64_t{shape.valueHeads} * sizeof(float), "GDN decay weight");
  requireBytes(timeBias, uint64_t{shape.valueHeads} * 2, "GDN time bias");
}

// The fp32 decay and bf16 beta gates of `rows` rows of value heads.
void requireGates(const GdnShape &shape, uint64_t rows, const metal::MetalBuffer &decay,
                  const metal::MetalBuffer &beta) {
  requireBytes(decay, rows * shape.valueHeads * sizeof(float), "GDN decay");
  requireBytes(beta, rows * shape.valueHeads * 2, "GDN beta");
}

// Each running lane's current and next state cells reach the end of layer
// `layer`'s state: its carried rows from layer x convolutionLayerBytes, its
// recurrent state from convolutionStateBytes + layer x recurrentLayerBytes.
void requireStates(const GdnShape &shape, GdnStateStrides strides, uint32_t layer, uint32_t lanes,
                   std::span<const metal::MetalBuffer> current, std::span<const metal::MetalBuffer> next) {
  const uint64_t bytes =
      std::max(uint64_t{layer} * strides.convolutionLayerBytes + carriedBytes(shape),
               strides.convolutionStateBytes + uint64_t{layer} * strides.recurrentLayerBytes + recurrentBytes(shape));
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    requireBytes(current[lane], bytes, "GDN current state");
    requireBytes(next[lane], bytes, "GDN next state");
  }
}

} // namespace

void GDN::addPrefill(metal::CommandGraph &graph, GdnPrefillBuffers buffers,
                     GdnShape shape, uint32_t tokens, GdnHeadOrder order) {
  if (!tokens)
    throw std::invalid_argument("invalid GDN prefill geometry");
  const KernelLayout kernel = kernelShape(shape);
  const std::string gate = normKernel(kernelName(kernel, "prefill_gdn_gate", "prefill_gdn_gate_vh32"),
                                      buffers.mixerNorm, shape.headDimension);
  const uint64_t keyRows = uint64_t{tokens} * shape.keyHeads * shape.headDimension * 2;
  const uint64_t valueRows = tokens * valueWidth(shape) * 2;
  requireBytes(buffers.packed, packedBytes(shape, tokens), "GDN packed");
  requireMixerWeights(shape, buffers.convolutionWeights, buffers.decayWeights, buffers.timeBias);
  requireBytes(buffers.convolutionIn, carriedBytes(shape), "GDN convolution state");
  requireBytes(buffers.convolutionOut, carriedBytes(shape), "GDN next convolution state");
  requireBytes(buffers.queries, keyRows, "GDN query");
  requireBytes(buffers.keys, keyRows, "GDN key");
  requireBytes(buffers.values, valueRows, "GDN value");
  requireGates(shape, tokens, buffers.decay, buffers.beta);
  requireBytes(buffers.recurrentIn, recurrentBytes(shape), "GDN recurrent state");
  requireBytes(buffers.recurrentOut, recurrentBytes(shape), "GDN next recurrent state");
  requireBytes(buffers.recurrentRows, valueRows, "GDN recurrent row");
  requireBytes(buffers.hidden, valueRows, "GDN hidden");
  const GDNPrefillParams params{tokens};
  graph.add(kernelName(kernel, "prefill_gdn_prepare",
                       "prefill_gdn_prepare_vh32"),
            {buffers.packed, buffers.convolutionWeights,
             buffers.convolutionIn, buffers.convolutionOut, buffers.queries,
             buffers.keys, buffers.values, buffers.decayWeights,
             buffers.timeBias, buffers.decay, buffers.beta},
            params, {uint64_t{tokens} * shape.keyHeads, 1, 1},
            {shape.headDimension, 1, 1});
  graph.add(kernelName(kernel, "prefill_gdn_scan",
                       "prefill_gdn_scan_vh32"),
            {buffers.queries, buffers.keys, buffers.values, buffers.decay,
             buffers.beta, buffers.recurrentIn, buffers.recurrentOut,
             buffers.recurrentRows},
            params,
            {uint64_t{shape.valueHeads} * shape.headDimension /
                 SPLASH_GDN_SCAN_STATE_ROWS,
             1, 1},
            {SPLASH_GDN_SCAN_THREADS, 1, 1});
  graph.add(gate,
            {buffers.recurrentRows, buffers.packed, buffers.mixerNorm.buffer,
             buffers.hidden},
            GDNGatePrefillParams{order == GdnHeadOrder::Tiled},
            {uint64_t{tokens} * shape.valueHeads, 1, 1}, {128, 1, 1});
}

PreparedInput GDN::addDecode(metal::CommandGraph &graph, GdnDecodeBuffers buffers,
                             GdnShape shape, uint32_t lanes, uint32_t layer,
                             GdnStateStrides state, GdnHeadOrder order, LinearInput input) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH || !state.valid() || buffers.tapeLanes.size() < lanes)
    throw std::invalid_argument("invalid GDN decode geometry");
  const KernelLayout kernel = kernelShape(shape);
  // Lane l's rows of the packed and hidden rows are rows [8 l, 8 l + 8).
  const uint64_t rows = uint64_t{lanes} * SPLASH_TARGET_VERIFY_ROWS;
  requireBytes(buffers.packed, packedBytes(shape, rows), "GDN packed");
  requireMixerWeights(shape, buffers.convolutionWeights, buffers.decayWeights, buffers.timeBias);
  requireBytes(buffers.hidden, rows * valueWidth(shape) * 2, "GDN hidden");
  GDNDecodeBatchParams params{order == GdnHeadOrder::Tiled,
                              layer,
                              state.convolutionLayerBytes,
                              state.recurrentLayerBytes,
                              state.convolutionStateBytes,
                              {},
                              {},
                              {}};
  // Each lane reads its pending rows' tape of this layer and writes its
  // step's; no step's tape may overlap a tape another lane, or the step's own
  // lane, reads or writes.
  const uint64_t tapeBytes = gdnTapeLayerBytes(shape);
  const auto overlaps = [&](uint64_t a, uint64_t b) { return a < b + tapeBytes && b < a + tapeBytes; };
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const GdnTapeLane &tape = buffers.tapeLanes[lane];
    params.pending_tape[lane] = tape.pendingSlot + uint64_t{layer} * tapeBytes;
    params.step_tape[lane] = tape.stepSlot + uint64_t{layer} * tapeBytes;
    params.pending_rows[lane] = tape.pendingRows;
    if (tape.pendingRows > SPLASH_TARGET_VERIFY_ROWS || params.pending_tape[lane] % 16 ||
        params.step_tape[lane] % 16)
      throw std::invalid_argument("invalid GDN decode tape");
    for (uint32_t other = 0; other <= lane; ++other)
      if (overlaps(params.step_tape[lane], params.pending_tape[other]) ||
          overlaps(params.step_tape[other], params.pending_tape[lane]) ||
          (other != lane && overlaps(params.step_tape[lane], params.step_tape[other])))
        throw std::invalid_argument("invalid GDN decode tape");
    requireBytes(buffers.tape, std::max(params.pending_tape[lane], params.step_tape[lane]) + tapeBytes, "GDN tape");
  }
  std::vector<metal::MetalBuffer> bindings{buffers.packed,
                                           buffers.convolutionWeights};
  const bool prepare = input != LinearInput::Plain;
  if (prepare)
    requireTableScratch(buffers.linearScratch, input, shape.valueHeads * shape.headDimension,
                        lanes * SPLASH_TARGET_VERIFY_ROWS);
  bindings.reserve(prepare ? 17 : 15);
  appendLaneBindings(bindings, buffers.currentStates, buffers.nextStates);
  requireStates(shape, state, layer, lanes, buffers.currentStates, buffers.nextStates);
  bindings.insert(bindings.end(),
                  {buffers.tape, buffers.decayWeights, buffers.timeBias, buffers.mixerNorm.buffer,
                   buffers.hidden});
  if (prepare)
    bindings.insert(bindings.end(), {buffers.linearScratch.input, buffers.linearScratch.sums});
  const std::string name = std::string("verify_gdn_fused") + tableSuffix(input) + kernelName(kernel, "", "_vh32");
  graph.add(normKernel(name, buffers.mixerNorm, shape.headDimension), std::move(bindings), params,
            {shape.valueHeads, lanes, 1});
  if (!prepare) return {};
  return {buffers.hidden, input};
}

uint64_t gdnTapeLayerBytes(const GdnShape &shape) noexcept {
  return gdn_tape_layer_bytes(shape.convolutionDimension,
                              (shape.keyHeads + shape.valueHeads) * shape.headDimension,
                              shape.valueHeads);
}

} // namespace splash::ops
