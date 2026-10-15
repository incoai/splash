#include "ops/Int8Ffn.hpp"

#include "Checked.hpp"
#include "metal/abi/AneFfn.h"
#include "metal/abi/Int8Ffn.h"
#include "ops/AneFfn.hpp"
#include "ops/BufferExtent.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace splash::ops {

Int8Ffn::Planes::Planes(const Projection &projection) {
  const QuantizedSegment &segment = projection.blocks().segments.front();
  requireSegmentPlanes(segment, "int8 FFN projection");
  groups = projection.planeInputSize() / 32;
  format = segment.formatId;
  buffers = {segment.plane0, segment.plane1Slot(), segment.meta};
}

const char *Int8Ffn::unsupported(const DeviceCapabilities &device, std::span<const SwiGluProjections> layers) {
  if (gpuFamilyClass(device.appleGpuFamily) != GpuFamilyClass::Apple10)
    return "int8 FFN needs an M5 or later GPU, whose int8 matmul runs faster than bf16";
  return unsupported(layers);
}

const char *Int8Ffn::unsupported(std::span<const SwiGluProjections> layers) {
  if (layers.empty()) return "int8 FFN has no layers";
  const uint32_t hidden = layers.front().gate->inputSize, intermediate = layers.front().gate->outputSize;
  if (hidden % ANE_FFN_INPUT_BLOCK || hidden % INT8_FFN_COLUMNS)
    return "int8 FFN needs a hidden size of whole input rotation blocks";
  if (intermediate % ANE_FFN_INTERMEDIATE_BLOCK || intermediate % INT8_FFN_COLUMNS)
    return "int8 FFN needs an intermediate size of whole intermediate rotation blocks";
  for (const SwiGluProjections &layer : layers) {
    for (const Projection *projection : {layer.gate, layer.up, layer.down}) {
      const std::vector<QuantizedSegment> &segments = projection->blocks().segments;
      if (segments.size() != 1 || segments.front().isFloat() || projection->rotation)
        return "int8 FFN needs unrotated quantized projections";
    }
    if (layer.gate->outputSize != intermediate || layer.gate->inputSize != hidden ||
        layer.up->outputSize != intermediate || layer.up->inputSize != hidden || layer.down->outputSize != hidden ||
        layer.down->inputSize != intermediate)
      return "int8 FFN layers differ in shape";
  }
  return nullptr;
}

Int8Ffn::Shape Int8Ffn::shapeOf(std::span<const SwiGluProjections> layers) {
  if (layers.empty()) throw std::invalid_argument("int8 FFN has no layers");
  return {layers.front().gate->inputSize, layers.front().gate->outputSize, static_cast<uint32_t>(layers.size())};
}

template <class MakeBuffer>
Int8Ffn::Memory Int8Ffn::allocate(const Shape &shape, MakeBuffer &&buffer, const Memory *scratch) {
  const uint64_t weights = uint64_t{shape.intermediate} * shape.hidden;
  Memory memory;
  if (scratch) memory = *scratch;
  memory.signs = buffer(ANE_FFN_INTERMEDIATE_BLOCK * sizeof(float), "int8 ffn signs");
  memory.rowScales =
      buffer(uint64_t{shape.layers} * (2 * shape.intermediate + shape.hidden) * sizeof(float), "int8 ffn row scales");
  if (scratch) return memory;
  memory.gate = buffer(weights, "int8 ffn gate");
  memory.up = buffer(weights, "int8 ffn up");
  memory.down = buffer(weights, "int8 ffn down");
  memory.inputs = buffer(uint64_t{kMaximumRows} * shape.hidden, "int8 ffn inputs");
  memory.inputScale = buffer(uint64_t{kMaximumRows} * sizeof(float), "int8 ffn input scales");
  memory.intermediate = buffer(uint64_t{kMaximumRows} * shape.intermediate, "int8 ffn intermediate");
  memory.intermediateScale = buffer(uint64_t{kMaximumRows} * sizeof(float), "int8 ffn intermediate scales");
  return memory;
}

uint64_t Int8Ffn::plannedBytes(uint32_t hidden, uint32_t intermediate, uint32_t layers, bool shared) {
  uint64_t bytes = 0;
  const Memory scratch;
  static_cast<void>(allocate(
      {hidden, intermediate, layers},
      [&](uint64_t size, const char *) {
        bytes += alignUp(size);
        return metal::MetalBuffer{};
      },
      shared ? &scratch : nullptr));
  return bytes;
}
uint64_t Int8Ffn::plannedBytes(std::span<const SwiGluProjections> layers) {
  const Shape shape = shapeOf(layers);
  return plannedBytes(shape.hidden, shape.intermediate, shape.layers, false);
}

Int8Ffn::Int8Ffn(metal::MetalBackend &backend, std::span<const SwiGluProjections> layers, const Int8Ffn *scratch)
    : backend_(backend), shape_(shapeOf(layers)) {
  if (const char *reason = unsupported(layers)) throw std::invalid_argument(reason);
  if (scratch && (scratch->shape_.hidden != shape_.hidden || scratch->shape_.intermediate < shape_.intermediate))
    throw std::invalid_argument("int8 FFN shares the scratch of an FFN of another shape");
  memory_ = allocate(
      shape_,
      [&](uint64_t bytes, const char *label) {
        metal::MetalBuffer buffer = backend_.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
        allocatedBytes_ += buffer.allocatedBytes();
        return buffer;
      },
      scratch ? &scratch->memory_ : nullptr);
  const std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> signs = aneFfnRotationSigns();
  std::memcpy(memory_.signs.contents(), signs.data(), sizeof signs);
  for (const SwiGluProjections &layer : layers)
    layers_.push_back({Planes(*layer.gate), Planes(*layer.up), Planes(*layer.down)});

  // Each row's scale over its inputs, rotated as its rows are.
  metal::CommandGraph graph;
  for (uint32_t layer = 0; layer < layers_.size(); ++layer) {
    const auto add = [&](Matrix matrix, const char *kernel, uint32_t width, uint32_t rows) {
      const Planes &planes = layers_[layer][static_cast<size_t>(matrix)];
      graph.add(kernel, {planes.buffers[0], planes.buffers[1], planes.buffers[2], rowScales(layer, matrix), memory_.signs},
                Int8FfnRowScaleParams{planes.groups, width, planes.format}, {rows / INT8_FFN_SIMDGROUPS, 1, 1},
                {INT8_FFN_THREADS, 1, 1});
    };
    add(Matrix::Gate, "int8_ffn_row_scale_inputs", shape_.hidden, shape_.intermediate);
    add(Matrix::Up, "int8_ffn_row_scale_inputs", shape_.hidden, shape_.intermediate);
    add(Matrix::Down, "int8_ffn_row_scale_intermediate", shape_.intermediate, shape_.hidden);
  }
  static_cast<void>(backend_.submitCommandAsync(graph.dispatches()).wait());
}

metal::MetalBuffer Int8Ffn::rowScales(uint32_t layer, Matrix matrix) const {
  const uint64_t perLayer = 2 * uint64_t{shape_.intermediate} + shape_.hidden, part = static_cast<uint64_t>(matrix);
  return backend_.view(memory_.rowScales, (layer * perLayer + part * shape_.intermediate) * sizeof(float),
                       uint64_t{matrix == Matrix::Down ? shape_.hidden : shape_.intermediate} * sizeof(float));
}

namespace {

// The grid of a GEMM dispatch of `rows` rows over `outputs` weight rows with
// `conversions` conversion threadgroups beside its tiles (Int8FfnGemmParams).
struct GemmGrid final {
  uint32_t rowTiles, columnTiles, conversionColumns;
  GemmGrid(uint32_t rows, uint32_t outputs, uint32_t conversions)
      : rowTiles((rows + INT8_FFN_ROWS - 1) / INT8_FFN_ROWS), columnTiles(outputs / INT8_FFN_COLUMNS),
        conversionColumns((conversions + columnTiles - 1) / columnTiles) {}
  [[nodiscard]] metal::DispatchSize size() const { return {rowTiles + conversionColumns, columnTiles, 1}; }
};

} // namespace

void Int8Ffn::addDown(metal::CommandGraph &graph, uint32_t layer, uint32_t next, metal::MetalBuffer residual,
                      metal::MetalBuffer output, uint32_t rows) const {
  const bool converts = next < layers_.size();
  // Without a next layer, the conversion binds this layer's planes and reads none.
  const uint32_t converted = converts ? next : layer;
  const Planes &gate = layers_[converted][static_cast<size_t>(Matrix::Gate)],
               &up = layers_[converted][static_cast<size_t>(Matrix::Up)];
  const uint32_t conversions = converts ? 2 * shape_.intermediate / INT8_FFN_SIMDGROUPS : 0;
  const GemmGrid grid(rows, shape_.hidden, conversions);
  graph.add("int8_ffn_down",
            {memory_.intermediate, memory_.intermediateScale, memory_.down, rowScales(layer, Matrix::Down), residual,
             output, gate.buffers[0], gate.buffers[1], gate.buffers[2], up.buffers[0], up.buffers[1], up.buffers[2],
             rowScales(converted, Matrix::Gate), rowScales(converted, Matrix::Up), memory_.gate, memory_.up,
             memory_.signs},
            Int8FfnGemmParams{rows, shape_.intermediate, shape_.hidden, grid.rowTiles, grid.conversionColumns,
                              conversions, shape_.hidden, shape_.intermediate, {gate.groups, up.groups},
                              {gate.format, up.format}},
            grid.size(), {INT8_FFN_THREADS, 1, 1});
}

void Int8Ffn::add(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn,
                  metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows) {
  if (!takes(rows)) throw std::invalid_argument("int8 FFN does not take a chunk of these rows");
  if (layer >= layers_.size()) throw std::invalid_argument("int8 FFN has no layer " + std::to_string(layer));
  const uint32_t hidden = shape_.hidden, intermediate = shape_.intermediate;
  requireBytes(ffn.normalized, uint64_t{rows} * hidden * sizeof(uint16_t), "int8 FFN input");
  requireBytes(ffn.intermediate, uint64_t{rows} * intermediate * sizeof(uint16_t), "int8 FFN intermediate");
  requireBytes(residual, uint64_t{rows} * hidden * sizeof(uint16_t), "int8 FFN residual");
  requireBytes(output, uint64_t{rows} * hidden * sizeof(uint16_t), "int8 FFN output");
  // A command's first layer converts its own gate and up alone, as does a
  // layer added out of order; each down converts the next layer's.
  if (!layer || layer != nextLayer_) addDown(graph, layer, layer, residual, output, 0);
  graph.add("int8_ffn_quantize_inputs", {ffn.normalized, memory_.signs, memory_.inputs, memory_.inputScale},
            Int8FfnQuantizeParams{hidden}, {rows, 1, 1}, {INT8_FFN_QUANTIZE_THREADS, 1, 1});
  const uint32_t conversions = hidden / INT8_FFN_SIMDGROUPS;
  const GemmGrid grid(rows, intermediate, conversions);
  const Planes &down = layers_[layer][static_cast<size_t>(Matrix::Down)];
  graph.add("int8_ffn_gate_up",
            {memory_.inputs, memory_.inputScale, memory_.gate, memory_.up, rowScales(layer, Matrix::Gate),
             rowScales(layer, Matrix::Up), ffn.intermediate, down.buffers[0], down.buffers[1], down.buffers[2],
             rowScales(layer, Matrix::Down), memory_.down, memory_.signs},
            Int8FfnGemmParams{rows, hidden, intermediate, grid.rowTiles, grid.conversionColumns, conversions,
                              intermediate, hidden, {down.groups, 0}, {down.format, 0}},
            grid.size(), {INT8_FFN_THREADS, 1, 1});
  graph.add("int8_ffn_quantize_intermediate",
            {ffn.intermediate, memory_.signs, memory_.intermediate, memory_.intermediateScale},
            Int8FfnQuantizeParams{intermediate}, {rows, 1, 1}, {INT8_FFN_QUANTIZE_THREADS, 1, 1});
  addDown(graph, layer, layer + 1, residual, output, rows);
  nextLayer_ = layer + 1;
}

} // namespace splash::ops
