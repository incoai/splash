#include "ops/AneFfn.hpp"

#include "Checked.hpp"
#include "StderrLine.hpp"
#include "metal/abi/AneFfn.h"
#include "metal/abi/QuantFormat.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {
namespace {

using Element = ane::Surface::Element;

// The ANE program multiplies gate's and up's hidden inputs, and down's ANE
// inputs, in segments of kSegment channels, a matmul each: on an M5 Pro, two
// of 2560 run a 5120-channel FFN about 14% faster than one of 5120 (ane-ffn's
// FFN, 2048 rows). The weights of a segment are staged in whole blocks of
// either rotation, and its inputs packed in whole tiles.
constexpr uint32_t kSegment = 2560;
static_assert(kSegment % ANE_FFN_INPUT_BLOCK == 0 && kSegment % ANE_FFN_INTERMEDIATE_BLOCK == 0 &&
                  kSegment % ANE_FFN_TILE == 0,
              "segments hold whole rotation blocks and packing tiles");
// The hidden channels ane_ffn_rotate's simdgroups rotate a block each of.
constexpr uint32_t kRotateGroup = ANE_FFN_INPUT_BLOCK * (ANE_FFN_ROTATE_THREADS / 32);
// The channels a split moves in: whole blocks of the intermediate rotation
// for the ANE, whole 256-row tiles of the weight planes for the GPU.
constexpr uint32_t kChannelUnit = std::max(QUANT_TILE_ROWS, ANE_FFN_INTERMEDIATE_BLOCK);
static_assert(kChannelUnit % ANE_FFN_WEIGHT_ROWS == 0 && kSegment % ANE_FFN_WEIGHT_ROWS == 0,
              "the weight kernels' threadgroups take whole rows of every matrix the split stages");
// How long an ANE evaluation may run once the GPU has raised its `ready`
// before the split stops (ane::Handoff). An evaluation takes 10-60 ms. Metal
// fails a command buffer whose event wait stays unmet for 5 s ("Caused GPU
// Timeout Error", macOS 27.0 on an M5 Max), and once two have failed so it
// ignores the process's later commands on that queue; the GPU waits on an
// evaluation only after its own part of the layer, so the release reaches a
// wait at most 2 s old, 3 s short of that.
constexpr auto kHandoffBound = std::chrono::seconds(2);
// How long the service may take to compile and load the program: four times
// the cold compile of the largest the split compiles, calibration's at share
// 0.8 (27 s on an M5 Max, about 30 s on an M6), so that a busy Mac still
// finishes and only a service that stopped answering runs out of it.
constexpr auto kProgramLimit = std::chrono::seconds(120);
// How long the service may take to unload the program for the idle release
// (release()) and to load it again (restore()), while the engine's loop and
// then the request that ends the idle wait: the service unloads a program in
// 3-4 ms and loads one it has compiled in 50-150 ms on an M5 Max, so a busy
// Mac still finishes, and one that stopped answering stops the split instead
// of holding the engine longer.
constexpr auto kReloadLimit = std::chrono::seconds(10);

// The GPU's channels of `intermediate` with the ANE taking `aneUnits`.
uint32_t gpuChannels(uint32_t intermediate, uint32_t aneUnits) {
  if (!aneUnits || aneUnits >= intermediate / kChannelUnit)
    throw std::invalid_argument("ANE FFN split leaves the GPU or the ANE no channels");
  return intermediate - aneUnits * kChannelUnit;
}

std::vector<uint32_t> segments(uint32_t channels) {
  std::vector<uint32_t> result;
  for (uint32_t begin = 0; begin < channels; begin += kSegment) result.push_back(std::min(kSegment, channels - begin));
  return result;
}

} // namespace

const char *AneFfn::unsupported(std::span<const SwiGluProjections> layers) {
  if (layers.empty()) return "ANE FFN split has no layers";
  const uint32_t hidden = layers.front().gate->inputSize, intermediate = layers.front().gate->outputSize;
  if (hidden % kSegment) return "ANE FFN split needs a hidden size of whole 2560-channel segments";
  if (hidden % kRotateGroup || hidden / kRotateGroup > ANE_FFN_ROTATE_BLOCKS)
    return "ANE FFN split needs a hidden size ane_ffn_rotate takes";
  if (intermediate % kChannelUnit || intermediate < 2 * kChannelUnit)
    return "ANE FFN split needs two or more whole rotation blocks of intermediate channels";
  for (const SwiGluProjections &layer : layers) {
    for (const Projection *projection : {layer.gate, layer.up, layer.down})
      if (!projection->takesPlaneViews())
        return "ANE FFN split needs unrotated quantized projections";
    if (layer.gate->outputSize != intermediate || layer.gate->inputSize != hidden ||
        layer.up->outputSize != intermediate || layer.up->inputSize != hidden || layer.down->outputSize != hidden ||
        layer.down->inputSize != intermediate)
      return "ANE FFN split layers differ in shape";
  }
  return nullptr;
}

uint32_t AneFfn::units(std::span<const SwiGluProjections> layers) {
  if (const char *reason = unsupported(layers)) throw std::invalid_argument(reason);
  return layers.front().gate->outputSize / kChannelUnit;
}

namespace {

// The rotation a weight kernel applies: over gate's and up's hidden inputs,
// or over down's intermediate inputs.
enum class Rotation : uint8_t { Inputs, Intermediate };
uint32_t blockOf(Rotation rotation) {
  return rotation == Rotation::Inputs ? ANE_FFN_INPUT_BLOCK : ANE_FFN_INTERMEDIATE_BLOCK;
}
std::string kernel(const char *name, Rotation rotation) {
  return std::string(name) + (rotation == Rotation::Inputs ? "_inputs" : "_intermediate");
}

// The Hadamard signs D of the rotations R = D H / sqrt(n) of inputs, weights
// and the ANE's intermediate rows alike: a block of n values takes the first n.
std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> rotationSigns() {
  std::mt19937 generator(20260930);
  std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> signs{};
  for (float &sign : signs) sign = (generator() & 1) ? -1.0f : 1.0f;
  return signs;
}

// The program's constant: the rotation of each of the ANE's intermediate
// channels as the [channels, ANE_FFN_INTERMEDIATE_BLOCK, 1, 1] weight of a
// grouped 1x1 convolution.
std::vector<uint8_t> rotationBlob(uint32_t channels, const std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> &signs) {
  constexpr uint32_t block = ANE_FFN_INTERMEDIATE_BLOCK;
  std::vector<_Float16> values(uint64_t{channels} * block);
  const float norm = 1.0f / std::sqrt(float(block));
  for (uint32_t channel = 0; channel < channels; ++channel)
    for (uint32_t input = 0; input < block; ++input) {
      const uint32_t output = channel % block;
      const float hadamard = (std::popcount(output & input) & 1) ? -1.0f : 1.0f;
      values[uint64_t{channel} * block + input] = _Float16(signs[output] * hadamard * norm);
    }
  return ane::constantBlob(values);
}

// The name of the program's function of `rows` rows.
std::string functionName(uint32_t rows) { return "ffn" + std::to_string(rows); }

// The MIL dimensions of `rows` rows of `width` values.
std::string dimensions(uint64_t rows, uint64_t width) {
  return "[1, 1, " + std::to_string(rows) + ", " + std::to_string(width) + "]";
}
std::string tensor(const char *type, uint64_t rows, uint64_t width) {
  return std::string("tensor<") + type + ", " + dimensions(rows, width) + ">";
}
// An fp16 constant of `value`, written exactly: MIL rounds it to the nearest
// fp16.
std::string fp16(double value) {
  char text[32];
  std::snprintf(text, sizeof text, "fp16(%a)", value);
  return text;
}
// The least peak the ANE quantizes a token's rotated intermediate row
// against: a quieter row takes coarser codes, each within this over 254 of
// the token's input scale.
constexpr double kIntermediateFloor = 0x1p-9;
static_assert(ANE_FFN_INT8_PEAK / kIntermediateFloor <= 65504.0,
              "the int8 peak over the floor fits fp16");

// The rows of the program's functions, ascending.
std::vector<uint32_t> functionRows() {
  std::vector<uint32_t> rows;
  for (uint32_t row = AneFfn::kMinimumRows; row <= AneFfn::kMaximumRows; row += AneFfn::kProgramStep)
    rows.push_back(row);
  return rows;
}

// What AneFfn::verify fills the ANE's output with before a check: an fp16
// NaN, which the join flags where the ANE leaves it.
constexpr uint16_t kHalfNaN = 0x7e00;
// The most the split's output may differ from the GPU's alone, as the RMS of
// their difference relative to the RMS of what the ANE adds over the GPU's
// part (AneFfn::verify): int8 leaves 2.5-2.8% on Qwen3.8-27B's MLX and GGUF
// layers on an M5 Max, and over three times that leaves room for other
// models and devices, where a function that computes other values differs by
// about all of the ANE's part.
constexpr double kMaximumAneError = 0.10;

// The leading `count` values of the CPU-visible bf16 `buffer`.
std::vector<float> bf16Values(const metal::MetalBuffer &buffer, uint64_t count) {
  const auto *bits = static_cast<const uint16_t *>(buffer.contents());
  if (!bits || buffer.sizeBytes() < count * sizeof(uint16_t))
    throw std::invalid_argument("ANE FFN check reads rows the CPU cannot");
  std::vector<float> values(count);
  for (uint64_t i = 0; i < count; ++i) values[i] = std::bit_cast<float>(uint32_t{bits[i]} << 16);
  return values;
}

} // namespace

AneFfn::Planes::Planes(const Projection &projection) {
  const QuantizedSegment &segment = projection.blocks().segments.front();
  requireSegmentPlanes(segment, "ANE FFN projection");
  groups = projection.inputSize / 32;
  format = segment.formatId;
  buffers = {segment.plane0, segment.plane1Slot(), segment.meta};
}

AneFfn::Shape AneFfn::shapeOf(std::span<const SwiGluProjections> layers, uint32_t aneUnits) {
  if (const char *reason = unsupported(layers)) throw std::invalid_argument(reason);
  Shape shape;
  shape.hidden = layers.front().gate->inputSize;
  shape.intermediate = layers.front().gate->outputSize;
  shape.layers = static_cast<uint32_t>(layers.size());
  shape.gpu = gpuChannels(shape.intermediate, aneUnits);
  shape.ane = shape.intermediate - shape.gpu;
  shape.downSegments = segments(shape.ane);
  return shape;
}

template <class MakeBuffer, class MakeSurface>
AneFfn::Memory AneFfn::allocate(const Shape &shape, MakeBuffer &&buffer, MakeSurface &&surface) {
  const uint32_t inputSegments = shape.hidden / kSegment;
  Memory memory;
  memory.signs = buffer(ANE_FFN_INTERMEDIATE_BLOCK * sizeof(float), "ane ffn signs");
  memory.rowScales = buffer(uint64_t{shape.layers} * (2 * shape.ane + shape.hidden) * sizeof(_Float16),
                            "ane ffn row scales");
  memory.rotated = buffer(uint64_t{kMaximumRows} * shape.hidden * sizeof(_Float16), "ane ffn rotated input");
  memory.status = buffer(sizeof(uint32_t), "ane ffn status");
  for (uint32_t k = 0; k < inputSegments; ++k) memory.inputs.push_back(surface(kSegment, kMaximumRows, Element::Int8));
  memory.tokenScale = surface(1, kMaximumRows, Element::Float16);
  memory.partial = surface(shape.hidden + 1, kMaximumRows, Element::Float16);
  for (Weights &set : memory.sets) {
    for (uint32_t k = 0; k < inputSegments; ++k) {
      set.gate.push_back(surface(shape.ane, kSegment, Element::Int8));
      set.up.push_back(surface(shape.ane, kSegment, Element::Int8));
    }
    for (const uint32_t width : shape.downSegments) set.down.push_back(surface(shape.hidden, width, Element::Int8));
    set.gateScale = surface(shape.ane, 1, Element::Float16);
    set.upScale = surface(shape.ane, 1, Element::Float16);
    set.downScale = surface(shape.hidden, 1, Element::Float16);
  }
  return memory;
}

uint64_t AneFfn::plannedBytes(std::span<const SwiGluProjections> layers, uint32_t aneUnits) {
  uint64_t bytes = 0;
  static_cast<void>(allocate(
      shapeOf(layers, aneUnits),
      [&](uint64_t size, const char *) {
        bytes += alignUp(size);
        return metal::MetalBuffer{};
      },
      [&](uint32_t rows, uint32_t width, Element element) {
        bytes += ane::Surface::bytes(rows, width, element);
        return ane::Surface{};
      }));
  return bytes;
}

// x<k>: the rotated input rows of segment k, channel-major in int8, and tx
// their per-token scales; wg<k> and wu<k>: gate's and up's rotated int8 rows
// over segment k, and sg and su their per-row scales; wd<i>: down's rotated
// int8 rows over segment i of its ANE inputs, and sd their per-row scales.
std::vector<AneFfn::Input> AneFfn::programInputs(const Shape &shape, const Memory &memory) {
  const auto both = [&](auto member) {
    return std::array{&(memory.sets[0].*member), &(memory.sets[1].*member)};
  };
  std::vector<Input> inputs;
  for (uint32_t k = 0; k < memory.inputs.size(); ++k)
    inputs.push_back({"x" + std::to_string(k), {&memory.inputs[k], &memory.inputs[k]}, kSegment, 0});
  inputs.push_back({"tx", {&memory.tokenScale, &memory.tokenScale}, 1, 0});
  for (uint32_t k = 0; k < memory.inputs.size(); ++k) {
    inputs.push_back(
        {"wg" + std::to_string(k), {&memory.sets[0].gate[k], &memory.sets[1].gate[k]}, shape.ane, kSegment});
    inputs.push_back({"wu" + std::to_string(k), {&memory.sets[0].up[k], &memory.sets[1].up[k]}, shape.ane, kSegment});
  }
  inputs.push_back({"sg", both(&Weights::gateScale), shape.ane, 1});
  inputs.push_back({"su", both(&Weights::upScale), shape.ane, 1});
  for (uint32_t i = 0; i < shape.downSegments.size(); ++i)
    inputs.push_back({"wd" + std::to_string(i), {&memory.sets[0].down[i], &memory.sets[1].down[i]}, shape.hidden,
                      shape.downSegments[i]});
  inputs.push_back({"sd", both(&Weights::downScale), shape.hidden, 1});
  return inputs;
}

// The ANE's share of the FFN over `rows` rows. Every int8 value is
// dequantized by 1 / ANE_FFN_INT8_UNIT against fp16 overflow, which the
// scales carry back. The intermediate rows are rotated and quantized per
// token here. The output holds down's rows before their tokens' scales, and
// in a last row each token's intermediate scale, which the join multiplies
// with the input's in fp32 (ane_ffn_join): the output at full scale could
// overflow fp16. The function reads and writes the leading `rows` of each row
// of the chunk's surfaces.
std::string AneFfn::function(const Shape &shape, std::span<const Input> inputs, const ane::Surface &output,
                             uint32_t rows) {
  const uint32_t inputSegments = shape.hidden / kSegment, channels = shape.ane;
  const std::string unit = fp16(1.0 / ANE_FFN_INT8_UNIT);
  std::string parameters, body;
  const auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
  const auto f16 = [&](const std::string &name, uint64_t height, uint64_t width, const std::string &expression) {
    line(tensor("fp16", height, width) + " " + name + " = " + expression);
  };
  for (const Input &input : inputs) {
    const uint32_t width = input.width ? input.width : rows;
    parameters +=
        (parameters.empty() ? "" : ", ") + input.surfaces[0]->bufferType(input.rows, width) + " " + input.name;
    line(tensor(ane::Surface::milType(input.surfaces[0]->element), input.rows, width) + " " + input.name +
         "_t = tensor_buffer_to_tensor<ios17>(input = " + input.name + ")");
  }
  const auto matmul = [&](const std::string &name, const std::string &weights, const std::string &values,
                          uint64_t height, uint64_t width) {
    f16(weights + "_d", height, width, "dequantize(input = " + weights + "_t, scale = " + unit + ")");
    line(tensor("fp16", height, rows) + " " + name +
         " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + weights + "_d, y = " + values + ")");
  };
  const auto sum = [&](const std::string &prefix, size_t terms, uint64_t height) {
    std::string total = prefix + "0";
    for (size_t term = 1; term < terms; ++term) {
      const std::string next = prefix + "_sum" + std::to_string(term);
      f16(next, height, rows, "add(x = " + total + ", y = " + prefix + std::to_string(term) + ")");
      total = next;
    }
    return total;
  };

  for (uint32_t k = 0; k < inputSegments; ++k) {
    const std::string x = "x" + std::to_string(k);
    f16(x + "_d", kSegment, rows, "dequantize(input = " + x + "_t, scale = " + unit + ")");
  }
  for (const char *projection : {"g", "u"}) {
    const std::string p = projection;
    for (uint32_t k = 0; k < inputSegments; ++k)
      matmul(p + "m" + std::to_string(k), "w" + p + std::to_string(k), "x" + std::to_string(k) + "_d", channels,
             kSegment);
    f16(p + "s", channels, rows, "mul(x = " + sum(p + "m", inputSegments, channels) + ", y = s" + p + "_t)");
  }
  const std::string c = std::to_string(channels), r = std::to_string(rows),
                    block = std::to_string(ANE_FFN_INTERMEDIATE_BLOCK);
  // silu(g) = g/2 (1 + tanh(g/2)): through the ANE's sigmoid, the split of
  // layers whose gate pre-activations are about 0.1 differed from the GPU
  // alone by 6.5% RMS, against 1.6% at about 1 (the ane-ffn test's split
  // of its former affine Q4 layers); through its tanh both differ by 1.6%.
  f16("gt", channels, rows, "mul(x = gs, y = tx_t)");
  f16("gh", channels, rows, "mul(x = gt, y = " + fp16(0.5) + ")");
  f16("th", channels, rows, "tanh(x = gh)");
  f16("tp", channels, rows, "add(x = th, y = " + fp16(1.0) + ")");
  f16("silu", channels, rows, "mul(x = gh, y = tp)");
  f16("h", channels, rows, "mul(x = silu, y = us)");
  line("tensor<fp16, [1, " + c + ", 1, " + r + "]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, " + c + ", 1, " +
       r + "]))");
  line("tensor<fp16, [" + c + ", " + block +
       ", 1, 1]> rotation = const()[name = string(\"rotation\"), val = tensor<fp16, [" + c + ", " + block +
       ", 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(" +
       std::to_string(ane::kConstantOffset) + ")))]");
  line("tensor<fp16, [1, " + c + ", 1, " + r + "]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32(" +
       std::to_string(channels / ANE_FFN_INTERMEDIATE_BLOCK) +
       "), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, "
       "1]), weight = rotation, x = h4)");
  f16("hr", channels, rows, "reshape(x = hr4, shape = tensor<int32, [4]>(" + dimensions(channels, rows) + "))");
  f16("habs", channels, rows, "abs(x = hr)");
  f16("peak", 1, rows, "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))");
  f16("floor", 1, rows, "maximum(x = peak, y = " + fp16(kIntermediateFloor) + ")");
  f16("inverse", 1, rows, "real_div(x = " + fp16(ANE_FFN_INT8_PEAK) + ", y = floor)");
  f16("hs", channels, rows, "mul(x = hr, y = inverse)");
  line(tensor("int8", channels, rows) + " hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))");
  f16("hd", channels, rows, "dequantize(input = hq, scale = " + unit + ")");
  f16("hscale", 1, rows, "mul(x = floor, y = " + fp16(double(ANE_FFN_INT8_UNIT) / ANE_FFN_INT8_PEAK) + ")");
  uint32_t begin = 0;
  for (size_t i = 0; i < shape.downSegments.size(); ++i) {
    const uint32_t width = shape.downSegments[i];
    const std::string index = std::to_string(i), slice = "hd" + index;
    f16(slice, width, rows,
        "slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, " + std::to_string(begin) +
            ", 0]), size = tensor<int32, [4]>(" + dimensions(width, rows) + "))");
    matmul("dm" + index, "wd" + index, slice, shape.hidden, width);
    begin += width;
  }
  f16("ds", shape.hidden, rows, "mul(x = " + sum("dm", shape.downSegments.size(), shape.hidden) + ", y = sd_t)");
  f16("yt", shape.hidden + 1, rows, "concat(axis = int32(2), interleave = bool(false), values = (ds, hscale))");
  line(output.bufferType(shape.hidden + 1, rows) +
       " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
       "strides = tensor<int64, [4]>(" + output.strides(shape.hidden + 1) + "))");
  return "    func " + functionName(rows) + "<ios18>(" + parameters + ") {\n" + body + "    } -> (y);\n";
}

std::string AneFfn::program(const Shape &shape, std::span<const Input> inputs, const ane::Surface &output,
                            std::span<const uint32_t> rows) {
  std::string text = "program(1.3)\n{\n";
  for (const uint32_t count : rows) text += function(shape, inputs, output, count);
  return text + "}\n";
}

AneFfn::AneFfn(metal::MetalBackend &backend, std::span<const SwiGluProjections> layers, uint32_t aneUnits,
               std::function<bool()> interrupted)
    : AneFfn(backend, layers, aneUnits, std::move(interrupted), functionRows()) {}

AneFfn::AneFfn(metal::MetalBackend &backend, std::span<const SwiGluProjections> layers, uint32_t aneUnits,
               std::function<bool()> interrupted, std::span<const uint32_t> functionRows)
    : backend_(backend), linear_(backend.capabilities()), shape_(shapeOf(layers, aneUnits)),
      handoff_(backend, kHandoffBound) {
  // What Metal allocates, which the memory audit compares with the plan.
  memory_ = allocate(
      shape_,
      [&](uint64_t bytes, const char *label) {
        metal::MetalBuffer buffer = backend_.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
        allocatedBytes_ += buffer.allocatedBytes();
        return buffer;
      },
      [&](uint32_t rows, uint32_t width, Element element) {
        ane::Surface surface = ane::Surface::create(backend_, rows, width, element);
        allocatedBytes_ += surface.buffer.allocatedBytes();
        return surface;
      });
  const std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> signs = rotationSigns();
  std::memcpy(memory_.signs.contents(), signs.data(), sizeof signs);
  *static_cast<uint32_t *>(memory_.status.contents()) = 0;

  for (const SwiGluProjections &source : layers)
    layers_.push_back({{Planes(*source.gate), Planes(*source.up), Planes(*source.down)},
                       source.gate->leadingRows(backend_, shape_.gpu),
                       source.up->leadingRows(backend_, shape_.gpu),
                       source.down->leadingInputs(shape_.gpu)});

  const std::vector<Input> inputs = programInputs(shape_, memory_);
  program_ = std::make_unique<ane::Program>(program(shape_, inputs, memory_.partial, functionRows),
                                            rotationBlob(shape_.ane, signs),
                                            ane::Program::Limits{kProgramLimit, std::move(interrupted)});
  for (const uint32_t rows : functionRows) {
    Evaluation &evaluation = evaluations_.emplace_back();
    evaluation.rows = rows;
    const uint32_t procedure = program_->procedure(functionName(rows));
    for (uint32_t set = 0; set < 2; ++set) {
      std::vector<ane::Surface> surfaces;
      for (const std::string &name : program_->inputs(procedure)) {
        const auto input = std::ranges::find(inputs, name, &Input::name);
        if (input == inputs.end()) throw std::logic_error("ANE FFN program has an unknown input " + name);
        surfaces.push_back(*input->surfaces[set]);
      }
      evaluation.bindings.push_back(program_->bind(procedure, surfaces, memory_.partial));
    }
  }

  // Each row's shared int8 scale over the ANE's share of its inputs.
  metal::CommandGraph graph;
  for (uint32_t layer = 0; layer < layers_.size(); ++layer) {
    const auto add = [&](Matrix matrix, uint32_t row, uint32_t input, uint32_t width, uint32_t rows,
                         Rotation rotation) {
      const Planes &planes = layers_[layer].planes[static_cast<size_t>(matrix)];
      graph.add(kernel("ane_ffn_row_scale_gguf", rotation),
                {planes.buffers[0], planes.buffers[1], planes.buffers[2], rowScales(layer, matrix), memory_.signs},
                AneFfnWeightParams{planes.groups, row, input, width, 0, 0, planes.format},
                {rows / ANE_FFN_WEIGHT_ROWS, 1, 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
    };
    add(Matrix::Gate, shape_.gpu, 0, shape_.hidden, shape_.ane, Rotation::Inputs);
    add(Matrix::Up, shape_.gpu, 0, shape_.hidden, shape_.ane, Rotation::Inputs);
    add(Matrix::Down, 0, shape_.gpu, shape_.ane, shape_.hidden, Rotation::Intermediate);
  }
  static_cast<void>(backend_.submitCommandAsync(graph.dispatches()).wait());
}

double AneFfn::share() const noexcept { return static_cast<double>(shape_.ane) / shape_.intermediate; }

void AneFfn::setMinimumRows(uint32_t rows) {
  if (rows < kMinimumRows || rows > kMaximumRows)
    throw std::invalid_argument("ANE FFN split has no function of " + std::to_string(rows) + " rows");
  minimumRows_ = rows;
}

bool AneFfn::splits(uint32_t rows) const {
  return rows >= minimumRows_ && rows <= kMaximumRows && !handoff_.retired();
}

metal::MetalBuffer AneFfn::rowScales(uint32_t layer, Matrix matrix) const {
  const uint64_t perLayer = 2 * shape_.ane + shape_.hidden, part = static_cast<uint64_t>(matrix);
  return backend_.view(memory_.rowScales, (layer * perLayer + part * shape_.ane) * sizeof(_Float16),
                       uint64_t{matrix == Matrix::Down ? shape_.hidden : shape_.ane} * sizeof(_Float16));
}

// Layer `layer`'s int8 weights and row scales into staging set `set`.
void AneFfn::addWeights(metal::CommandGraph &graph, uint32_t layer, uint32_t set) const {
  const Layer &source = layers_.at(layer);
  const Weights &target = memory_.sets[set];
  const auto add = [&](Matrix matrix, const ane::Surface &output, const ane::Surface &scale, uint32_t row,
                       uint32_t input, uint32_t width, uint32_t rows, Rotation rotation) {
    const Planes &planes = source.planes[static_cast<size_t>(matrix)];
    const uint32_t scaleStride = scale.strideBytes / uint32_t{sizeof(_Float16)};
    requireBytes(output.buffer, rowBytes(rows, output.strideBytes, width, 1), "ANE FFN weight surface");
    requireBytes(scale.buffer, rowBytes(rows, scaleStride, 1, sizeof(_Float16)), "ANE FFN weight scale surface");
    graph.add(kernel("ane_ffn_weights_gguf", rotation),
              {planes.buffers[0], planes.buffers[1], planes.buffers[2], rowScales(layer, matrix), output.buffer,
               scale.buffer, memory_.signs},
              AneFfnWeightParams{planes.groups, row, input, width, output.strideBytes, scaleStride, planes.format},
              {rows / ANE_FFN_WEIGHT_ROWS, width / blockOf(rotation), 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
  };
  for (uint32_t k = 0; k < target.gate.size(); ++k) {
    add(Matrix::Gate, target.gate[k], target.gateScale, shape_.gpu, k * kSegment, kSegment, shape_.ane,
        Rotation::Inputs);
    add(Matrix::Up, target.up[k], target.upScale, shape_.gpu, k * kSegment, kSegment, shape_.ane, Rotation::Inputs);
  }
  uint32_t begin = shape_.gpu;
  for (size_t i = 0; i < shape_.downSegments.size(); ++i) {
    add(Matrix::Down, target.down[i], target.downScale, 0, begin, shape_.downSegments[i], shape_.hidden,
        Rotation::Intermediate);
    begin += shape_.downSegments[i];
  }
}

bool AneFfn::begin() {
  // A command committed but never finished failed on the GPU, and the
  // evaluations it queued with it.
  if (std::exchange(unfinished_, false)) {
    handoff_.cancel("a command it split did not finish");
    warnStopped();
  }
  jobs_.clear();
  nextLayer_ = 0;
  return !handoff_.retired();
}

void AneFfn::add(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn,
                 metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows) {
  encode(graph, layer, ffn, residual, output, rows, Parts::Both);
}

void AneFfn::encode(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn,
                    metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows, Parts parts) {
  if (rows < kMinimumRows || rows > kMaximumRows)
    throw std::invalid_argument("ANE FFN split does not take a chunk of these rows");
  if (layer != nextLayer_ || layer >= layers_.size())
    throw std::logic_error("ANE FFN split adds layer " + std::to_string(layer) + ", not its next layer " +
                           std::to_string(nextLayer_) + " (begin() starts each command at layer 0)");
  const uint64_t hiddenBytes = uint64_t{rows} * shape_.hidden * sizeof(uint16_t);
  requireBytes(ffn.normalized, hiddenBytes, "ANE FFN input");
  requireBytes(output, hiddenBytes, "ANE FFN output");
  const Layer &current = layers_[layer];
  const uint32_t set = layer & 1, tiles = (rows + ANE_FFN_TILE - 1) / ANE_FFN_TILE;
  const bool ane = parts == Parts::Both;
  const auto found =
      std::ranges::find_if(evaluations_, [&](const Evaluation &evaluation) { return evaluation.rows >= rows; });
  if (found == evaluations_.end()) throw std::invalid_argument("ANE FFN has no function of the chunk's rows");
  const auto index = static_cast<uint32_t>(found - evaluations_.begin());
  // Each command stages layer 0's weights, then each layer the next one's.
  if (!layer) addWeights(graph, 0, 0);
  graph.add("ane_ffn_rotate", {ffn.normalized, memory_.signs, memory_.rotated, memory_.tokenScale.buffer},
            AneFfnRotateParams{shape_.hidden}, {rows, 1, 1}, {ANE_FFN_ROTATE_THREADS, 1, 1});
  for (uint32_t k = 0; k < memory_.inputs.size(); ++k)
    graph.add("ane_ffn_pack", {memory_.rotated, memory_.inputs[k].buffer},
              AneFfnPackParams{shape_.hidden, k * kSegment, memory_.inputs[k].strideBytes},
              {tiles, kSegment / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  const uint64_t ready = ane ? handoff_.next() : 0;
  if (ane) graph.signal(handoff_.event(), ready);
  linear_.addPrefillSwiGlu(graph, {&current.gate, &current.up, &current.down}, ffn, residual, output, rows);
  if (layer + 1 < layers_.size()) addWeights(graph, layer + 1, set ^ 1);
  const uint64_t done = ane ? handoff_.next() : 0;
  if (ane) graph.wait(handoff_.event(), done);
  graph.add("ane_ffn_join", {output, memory_.partial.buffer, memory_.tokenScale.buffer, memory_.status},
            AneFfnJoinParams{shape_.hidden, memory_.partial.strideBytes / uint32_t{sizeof(_Float16)}, rows},
            {tiles, shape_.hidden / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  if (ane) jobs_.push_back({index, set, ready, done});
  ++nextLayer_;
}

metal::CommandTicket AneFfn::commit(const metal::CommandGraph &graph, metal::CommandCompletion completion) {
  const std::vector<Job> jobs = std::exchange(jobs_, {});
  for (const Job &job : jobs) queue(job);
  unfinished_ = !jobs.empty();
  committedEvaluations_ = static_cast<uint32_t>(jobs.size());
  committedRows_ = jobs.empty() ? 0 : evaluations_[jobs.front().evaluation].rows;
  const auto cancel = [&](const std::string &reason) {
    if (!std::exchange(unfinished_, false)) return;
    handoff_.cancel(reason);
    warnStopped();
  };
  try {
    return backend_.submitCommandAsync(graph.command(), std::move(completion));
  } catch (const std::exception &error) {
    cancel(std::string("its Metal command was not submitted: ") + error.what());
    throw;
  } catch (...) {
    cancel("its Metal command was not submitted");
    throw;
  }
}

bool AneFfn::finish() {
  const std::optional<AwakeClock::duration> ran = completed();
  if (!ran) {
    warnStopped();
    return false;
  }
  if (!committedEvaluations_) return true;
  const double milliseconds = std::chrono::duration<double, std::milli>(*ran).count();
  ++served_.commands;
  served_.evaluations += committedEvaluations_;
  served_.milliseconds += milliseconds;
  if (std::string losing = breaker_.add(committedRows_, committedEvaluations_, milliseconds); !losing.empty()) {
    handoff_.retire(std::move(losing));
    warnStopped();
    if (lost_) std::exchange(lost_, {})();
  }
  return true;
}

std::optional<AwakeClock::duration> AneFfn::completed() {
  unfinished_ = false;
  std::optional<AwakeClock::duration> ran = handoff_.finish();
  if (std::exchange(*static_cast<uint32_t *>(memory_.status.contents()), 0u) && ran) {
    handoff_.retire("the Neural Engine's output or its scales were not finite");
    ran.reset();
  }
  return ran;
}

bool AneFfn::release() {
  if (unfinished_ || !jobs_.empty())
    throw std::logic_error("ANE FFN split released while a command it encoded is unfinished");
  if (released_ || (handoff_.retired() && !handoff_.idle())) return false;
  try {
    program_->unload({kReloadLimit, {}});
  } catch (const std::exception &error) {
    handoff_.retire(std::string("its program did not unload: ") + error.what());
    warnStopped();
    return false;
  }
  released_ = true;
  logLine("Neural Engine FFN program unloaded while idle");
  return true;
}

void AneFfn::restore() noexcept {
  // A stopped split keeps its program unloaded.
  if (!released_ || handoff_.retired()) return;
  released_ = false;
  const auto start = AwakeClock::now();
  std::string failure = "an unknown exception";
  try {
    program_->load({kReloadLimit, {}});
    logLine("Neural Engine FFN program reloaded in ", std::fixed, std::setprecision(2),
            millisecondsSince(start) / 1000.0, " s");
    return;
  } catch (const std::exception &error) {
    failure = error.what();
  } catch (...) {
  }
  handoff_.retire("its program did not load again: " + failure);
  warnStopped();
}

void AneFfn::warnStopped() {
  if (std::exchange(warned_, true)) return;
  logWarning("Neural Engine FFN split stopped (", handoff_.reason(),
             "); the GPU runs the prefill FFN alone until the engine restarts.");
}

void AneFfn::queue(const Job &job) {
  const ane::Program::Binding &binding = evaluations_.at(job.evaluation).bindings.at(job.set);
  handoff_.queue(job.ready, job.done,
                 [&](const metal::SharedEvent &event, uint64_t wait, uint64_t signal, ane::Handoff::Report report) {
                   program_->enqueue(binding, event, wait, signal, std::move(report));
                 });
}

void AneFfn::fillNormalized(const metal::MetalBuffer &normalized) {
  auto *bits = static_cast<uint16_t *>(normalized.contents());
  if (!bits) throw std::invalid_argument("ANE FFN check writes rows the CPU cannot");
  std::mt19937 random(20261001);
  std::uniform_real_distribution<float> uniform(-2.0f, 2.0f);
  for (uint64_t i = 0; i < normalized.sizeBytes() / sizeof(uint16_t); ++i)
    bits[i] = static_cast<uint16_t>(std::bit_cast<uint32_t>(uniform(random)) >> 16);
}

double AneFfn::verify(std::span<const SwiGluProjections> layers, const PrefillFfnBuffers &ffn,
                      const std::array<metal::MetalBuffer, 2> &hidden) {
  if (layers.size() != layers_.size() || layers.front().gate->outputSize != shape_.intermediate)
    throw std::invalid_argument("ANE FFN split verified on other layers than its own");
  if (handoff_.retired()) throw std::runtime_error("ANE FFN split stopped (" + handoff_.reason() + ")");
  fillNormalized(ffn.normalized);
  // Layer l adds its FFN of the normalized rows to hidden[l & 1] into
  // hidden[(l & 1) ^ 1], from zeros, on the GPU alone, as the split's GPU part
  // alone over a partial output of zeros, or split.
  enum class Way : uint8_t { Gpu, GpuPart, Split };
  const auto forward = [&](Way way, uint32_t count, uint32_t rows) {
    std::memset(hidden[0].contents(), 0, hidden[0].sizeBytes());
    if (way != Way::Gpu) {
      std::fill_n(static_cast<uint16_t *>(memory_.partial.buffer.contents()),
                  memory_.partial.buffer.sizeBytes() / sizeof(uint16_t), way == Way::Split ? kHalfNaN : 0);
      static_cast<void>(begin());
    }
    metal::CommandGraph graph;
    for (uint32_t layer = 0; layer < count; ++layer) {
      const metal::MetalBuffer &residual = hidden[layer & 1], &output = hidden[(layer & 1) ^ 1];
      if (way == Way::Gpu)
        linear_.addPrefillSwiGlu(graph, layers[layer], ffn, residual, output, rows);
      else
        encode(graph, layer, ffn, residual, output, rows, way == Way::Split ? Parts::Both : Parts::Gpu);
    }
    if (way == Way::Split) {
      static_cast<void>(commit(graph, {}).wait());
      if (!completed())
        throw std::runtime_error("ANE FFN split of " + std::to_string(rows) + " rows failed (" + reason() + ")");
    } else {
      static_cast<void>(backend_.submitCommandAsync(graph.command()).wait());
    }
    return bf16Values(hidden[count & 1], uint64_t{rows} * shape_.hidden);
  };
  // What the ANE adds over the GPU's part, the GPU alone's output less the
  // GPU part's, against the split's difference from the GPU alone, over the
  // split's rows.
  const uint32_t count = std::min<uint32_t>(2, shape_.layers);
  const std::vector<float> gpu = forward(Way::Gpu, count, kMaximumRows),
                           part = forward(Way::GpuPart, count, kMaximumRows);
  double worst = 0.0;
  for (const Evaluation &evaluation : evaluations_) {
    const uint32_t rows = evaluation.rows;
    const std::vector<float> split = forward(Way::Split, count, rows);
    const auto fail = [&](const std::string &why) {
      handoff_.retire("its output of " + std::to_string(rows) + " rows " + why);
      throw std::runtime_error("ANE FFN split's output of " + std::to_string(rows) + " rows " + why);
    };
    double difference = 0.0, contribution = 0.0;
    for (size_t i = 0; i < split.size(); ++i) {
      if (!std::isfinite(split[i])) fail("is not finite");
      difference += (double(split[i]) - gpu[i]) * (double(split[i]) - gpu[i]);
      contribution += (double(gpu[i]) - part[i]) * (double(gpu[i]) - part[i]);
    }
    const double error = std::sqrt(difference / contribution);
    if (!(error <= kMaximumAneError))
      fail("differs from the GPU alone's by " + std::to_string(100.0 * error) + "% of the Neural Engine's part");
    worst = std::max(worst, error);
  }
  return worst;
}

} // namespace splash::ops
