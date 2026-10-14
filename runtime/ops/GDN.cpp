#include "ops/GDN.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
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
// One layer's carried convolution rows and recurrent state, a head dimension
// square per value head.
uint64_t carriedBytes(const GdnShape &shape) {
  return uint64_t{SPLASH_GDN_CONVOLUTION_TAPS - 1} * shape.convolutionDimension * 2;
}
uint64_t recurrentBytes(const GdnShape &shape, GdnStateFormat format) {
  return valueWidth(shape) * shape.headDimension * elementBytes(format);
}

// A bf16 state's kernels are the fp32 state's with _bf16_state appended.
std::string stateKernel(std::string name, GdnStateFormat format) {
  if (format == GdnStateFormat::BFloat16)
    name += "_bf16_state";
  return name;
}

// The values of a row of the packed projection the kernels read: its q, k
// and v convolution inputs, z, then beta and alpha per value head.
uint64_t packedValues(const GdnShape &shape) {
  return shape.convolutionDimension + valueWidth(shape) + 2 * shape.valueHeads;
}
// The bytes of `rows` packed rows, to the last row's last value read.
uint64_t packedBytes(const GdnShape &shape, uint64_t rows) {
  return rowBytes(rows, shape.packedWidth, packedValues(shape), 2);
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
               strides.convolutionStateBytes + uint64_t{layer} * strides.recurrentLayerBytes +
                   recurrentBytes(shape, strides.format));
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    requireBytes(current[lane], bytes, "GDN current state");
    requireBytes(next[lane], bytes, "GDN next state");
  }
}

} // namespace

void GDN::addPrefill(metal::CommandGraph &graph, const metal::MetalBackend &backend,
                     GdnPrefillBuffers buffers, GdnShape shape, uint32_t tokens,
                     GdnStateFormat state, GdnHeadOrder order,
                     std::span<const GdnPrefillCapture> captures) {
  if (!tokens || !validFormat(state))
    throw std::invalid_argument("invalid GDN prefill geometry");
  const KernelLayout kernel = kernelShape(shape);
  const std::string gate = normKernel(kernelName(kernel, "prefill_gdn_gate", "prefill_gdn_gate_vh32"),
                                      buffers.mixerNorm, shape.headDimension);
  const uint64_t keyWidth = uint64_t{shape.keyHeads} * shape.headDimension;
  const uint64_t keyRows = tokens * keyWidth * 2;
  const uint64_t valueRows = tokens * valueWidth(shape) * 2;
  requireBytes(buffers.packed, packedBytes(shape, tokens), "GDN packed");
  requireMixerWeights(shape, buffers.convolutionWeights, buffers.decayWeights, buffers.timeBias);
  requireBytes(buffers.convolutionIn, carriedBytes(shape), "GDN convolution state");
  requireBytes(buffers.convolutionOut, carriedBytes(shape), "GDN next convolution state");
  requireBytes(buffers.queries, keyRows, "GDN query");
  requireBytes(buffers.keys, keyRows, "GDN key");
  requireBytes(buffers.values, valueRows, "GDN value");
  requireGates(shape, tokens, buffers.decay, buffers.beta);
  requireBytes(buffers.recurrentIn, recurrentBytes(shape, state), "GDN recurrent state");
  requireBytes(buffers.recurrentOut, recurrentBytes(shape, state), "GDN next recurrent state");
  requireBytes(buffers.recurrentRows, valueRows, "GDN recurrent row");
  requireBytes(buffers.hidden, valueRows, "GDN hidden");
  uint32_t previous = 0;
  for (const GdnPrefillCapture &capture : captures) {
    if (capture.tokens <= previous || capture.tokens >= tokens)
      throw std::invalid_argument("GDN prefill captures are not ascending inside its rows");
    previous = capture.tokens;
    requireBytes(capture.convolution, carriedBytes(shape), "GDN captured convolution state");
    requireBytes(capture.recurrent, recurrentBytes(shape, state), "GDN captured recurrent state");
  }
  const std::string prepare = kernelName(kernel, "prefill_gdn_prepare", "prefill_gdn_prepare_vh32");
  const std::string scan = stateKernel(kernelName(kernel, "prefill_gdn_scan", "prefill_gdn_scan_vh32"), state);
  metal::MetalBuffer convolution = buffers.convolutionIn;
  metal::MetalBuffer recurrent = buffers.recurrentIn;
  uint32_t begin = 0;
  for (size_t index = 0; index <= captures.size(); ++index) {
    const bool last = index == captures.size();
    const uint32_t end = last ? tokens : captures[index].tokens;
    const metal::MetalBuffer convolutionOut = last ? buffers.convolutionOut : captures[index].convolution;
    const metal::MetalBuffer recurrentOut = last ? buffers.recurrentOut : captures[index].recurrent;
    // The run's rows of a buffer whose rows are `stride` values of `bytes`
    // bytes apart, to the last row's `width` values the kernels read.
    const auto rows = [&](const metal::MetalBuffer &buffer, uint64_t stride, uint64_t width, uint64_t bytes) {
      return backend.view(buffer, begin * stride * bytes, rowBytes(end - begin, stride, width, bytes));
    };
    const metal::MetalBuffer queries = rows(buffers.queries, keyWidth, keyWidth, 2);
    const metal::MetalBuffer keys = rows(buffers.keys, keyWidth, keyWidth, 2);
    const metal::MetalBuffer values = rows(buffers.values, valueWidth(shape), valueWidth(shape), 2);
    const metal::MetalBuffer decay = rows(buffers.decay, shape.valueHeads, shape.valueHeads, sizeof(float));
    const metal::MetalBuffer beta = rows(buffers.beta, shape.valueHeads, shape.valueHeads, 2);
    const GDNPrefillParams params{end - begin};
    graph.add(prepare,
              {rows(buffers.packed, shape.packedWidth, packedValues(shape), 2), buffers.convolutionWeights,
               convolution, convolutionOut, queries, keys, values, buffers.decayWeights, buffers.timeBias, decay,
               beta},
              params, {uint64_t{end - begin} * shape.keyHeads, 1, 1}, {shape.headDimension, 1, 1});
    graph.add(scan,
              {queries, keys, values, decay, beta, recurrent, recurrentOut,
               rows(buffers.recurrentRows, valueWidth(shape), valueWidth(shape), 2)},
              params, {uint64_t{shape.valueHeads} * shape.headDimension / SPLASH_GDN_SCAN_STATE_ROWS, 1, 1},
              {SPLASH_GDN_SCAN_THREADS, 1, 1});
    convolution = convolutionOut;
    recurrent = recurrentOut;
    begin = end;
  }
  graph.add(gate, {buffers.recurrentRows, buffers.packed, buffers.mixerNorm.buffer, buffers.hidden},
            GDNGatePrefillParams{order == GdnHeadOrder::Tiled}, {uint64_t{tokens} * shape.valueHeads, 1, 1},
            {128, 1, 1});
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
  // Validate the complete layer extent before adding the slot offset. An
  // overflowing offset could otherwise wrap past both the overlap and
  // buffer-extent checks below.
  const uint64_t layerOffset = uint64_t{layer} * tapeBytes;
  const uint64_t lastSlot = std::numeric_limits<uint64_t>::max() - layerOffset - tapeBytes;
  const auto overlaps = [&](uint64_t a, uint64_t b) { return a < b + tapeBytes && b < a + tapeBytes; };
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const GdnTapeLane &tape = buffers.tapeLanes[lane];
    if (tape.pendingSlot > lastSlot || tape.stepSlot > lastSlot)
      throw std::invalid_argument("invalid GDN decode tape");
    params.pending_tape[lane] = tape.pendingSlot + layerOffset;
    params.step_tape[lane] = tape.stepSlot + layerOffset;
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
  graph.add(stateKernel(normKernel(name, buffers.mixerNorm, shape.headDimension), state.format),
            std::move(bindings), params, {shape.valueHeads, lanes, 1});
  if (!prepare) return {};
  return {buffers.hidden, input};
}

uint64_t gdnTapeLayerBytes(const GdnShape &shape) noexcept {
  return gdn_tape_layer_bytes(shape.convolutionDimension,
                              (shape.keyHeads + shape.valueHeads) * shape.headDimension,
                              shape.valueHeads);
}

} // namespace splash::ops
