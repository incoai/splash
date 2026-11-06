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
#include <functional>
#include <iomanip>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
// for the ANE, whole plane tiles of the weight planes (QUANT_TILE_ROWS) for
// the GPU.
constexpr uint32_t kChannelUnit = std::max(QUANT_TILE_ROWS, ANE_FFN_INTERMEDIATE_BLOCK);
static_assert(kChannelUnit % ANE_FFN_WEIGHT_ROWS == 0 && kSegment % ANE_FFN_WEIGHT_ROWS == 0,
              "the weight kernels' threadgroups take whole rows of every matrix the split stages");
// How long an ANE evaluation may run once the GPU has raised its `ready`
// before the split stops (ane::Handoff). An evaluation takes 10-60 ms. Metal
// fails a command buffer whose event wait is still unmet 5 s after its queue
// reached it ("Caused GPU Timeout Error", macOS 27.0 on an M5 Max), and once
// two have failed so it ignores the process's later commands on that queue.
// The queue reaches each wait once the one before it was met, so at the
// release a wait has waited through one layer's GPU part and at most the
// bound; a command's first wait also through the work from the command's
// submission, which image encodes no longer add (Runtime::prefillAsync).
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
// How many times a split stopped by an evaluation that ran late starts over
// once the Neural Engine has reported it: a stall a busy Mac recovers from
// costs one chunk's rerun on the GPU, and one that keeps coming back stops
// the split for good after the last.
constexpr uint32_t kHandoffRestarts = 3;

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

AneFfn::Planes::Planes(const Projection &projection)
    : Planes(projection.blocks().segments.front(), projection.inputSize) {}

AneFfn::Planes::Planes(const QuantizedSegment &segment, uint32_t inputs) {
  requireSegmentPlanes(segment, "ANE FFN projection");
  groups = inputs / 32;
  format = segment.formatId;
  first = segment.columnOffset;
  rows = segment.outputSize;
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
    : backend_(backend), linear_(backend.capabilities()), interrupted_(std::move(interrupted)),
      shape_(shapeOf(layers, aneUnits)), handoff_(backend, kHandoffBound) {
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
                                            ane::Program::Limits{kProgramLimit, interrupted_});
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

bool AneFfn::restartable() const { return handoff_.overran() && restarts_ < kHandoffRestarts; }

bool AneFfn::begin() {
  // A command committed but never finished failed on the GPU, and the
  // evaluations it queued with it.
  if (std::exchange(unfinished_, false)) {
    handoff_.cancel("a command it split did not finish");
    warnStopped();
  }
  jobs_.clear();
  nextLayer_ = 0;
  for (const std::unique_ptr<ProjectionSplit> &split : projections_)
    if (split) {
      split->next = 0;
      split->staged = ProjectionSplit::kNotStaged;
      split->deferredDone = 0;
    }
  if (handoff_.retired() && !released_ && restartable() && handoff_.restart(backend_)) {
    warned_ = false;
    logLine("Neural Engine split resumed after a late evaluation (", ++restarts_, " of ", kHandoffRestarts, ")");
  }
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
  if (ane) addNextProjectionWeights(graph);
  const uint64_t done = ane ? handoff_.next() : 0;
  if (ane) graph.wait(handoff_.event(), done);
  graph.add("ane_ffn_join", {output, memory_.partial.buffer, memory_.tokenScale.buffer, memory_.status},
            AneFfnJoinParams{shape_.hidden, memory_.partial.strideBytes / uint32_t{sizeof(_Float16)}, rows},
            {tiles, shape_.hidden / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  if (ane) jobs_.push_back({index, set, ready, done});
  encodingRows_ = evaluations_[index].rows;
  ++nextLayer_;
}

metal::CommandTicket AneFfn::commit(const metal::CommandGraph &graph, metal::CommandCompletion completion) {
  committed_ = std::exchange(jobs_, {});
  for (const Job &job : committed_) queue(job);
  unfinished_ = !committed_.empty();
  committedRows_ = encodingRows_;
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
  const std::optional<std::vector<AwakeClock::duration>> ran = completed();
  const std::vector<Job> jobs = std::exchange(committed_, {});
  if (!ran) {
    warnStopped();
    return false;
  }
  if (jobs.empty()) return true;
  if (ran->size() != jobs.size()) throw std::logic_error("ANE FFN split's handoff finished other jobs than it took");
  // The Neural Engine's milliseconds and evaluations of the command by owner:
  // a projection's deferred columns count in its layer's evaluation.
  std::array<double, kFfn + 1> milliseconds{};
  std::array<uint32_t, kFfn + 1> evaluations{};
  for (size_t index = 0; index < jobs.size(); ++index) {
    milliseconds[jobs[index].owner] += std::chrono::duration<double, std::milli>((*ran)[index]).count();
    if (!jobs[index].deferred) ++evaluations[jobs[index].owner];
  }
  ++served_.commands;
  served_.evaluations += jobs.size();
  for (const double owner : milliseconds) served_.milliseconds += owner;
  // Any breaker that trips forgets the calibration, so the next start
  // calibrates again.
  const auto lose = [&] {
    if (lost_) std::exchange(lost_, {})();
  };
  for (size_t owner = 0; owner < projections_.size(); ++owner) {
    ProjectionSplit *split = projections_[owner].get();
    if (!split || split->lost || !evaluations[owner]) continue;
    if (std::string losing = split->breaker.add(committedRows_, evaluations[owner], milliseconds[owner]);
        !losing.empty()) {
      split->lost = true;
      logWarning("Neural Engine split of the ", ane_ffn::projectionName(split->kind), " projections stopped (",
                 losing, "); the GPU runs them alone until the engine restarts.");
      lose();
    }
  }
  if (!evaluations[kFfn]) return true;
  if (std::string losing = breaker_.add(committedRows_, evaluations[kFfn], milliseconds[kFfn]); !losing.empty()) {
    handoff_.retire(std::move(losing));
    warnStopped();
    lose();
  }
  return true;
}

std::optional<std::vector<AwakeClock::duration>> AneFfn::completed() {
  unfinished_ = false;
  std::optional<std::vector<AwakeClock::duration>> ran = handoff_.finishEach();
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
    for (const std::unique_ptr<ProjectionSplit> &split : projections_)
      if (split) split->program->unload({kReloadLimit, {}});
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
  // A stopped split keeps its program unloaded, unless it may still restart.
  if (!released_ || (handoff_.retired() && !restartable())) return;
  released_ = false;
  const auto start = AwakeClock::now();
  std::string failure = "an unknown exception";
  try {
    program_->load({kReloadLimit, {}});
    for (const std::unique_ptr<ProjectionSplit> &split : projections_)
      if (split) split->program->load({kReloadLimit, {}});
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
             restartable() ? "); the GPU runs the prefill FFN alone until the Neural Engine reports the evaluation, "
                             "then the split resumes."
                           : "); the GPU runs the prefill FFN alone until the engine restarts.");
}

void AneFfn::queue(const Job &job) {
  const ProjectionSplit *split = job.owner == kFfn ? nullptr : projections_.at(job.owner).get();
  ane::Program &program = split ? *split->program : *program_;
  const std::vector<Evaluation> &evaluations = !split        ? evaluations_
                                               : job.deferred ? split->deferred.evaluations
                                                              : split->ahead.evaluations;
  const ane::Program::Binding &binding = evaluations.at(job.evaluation).bindings.at(job.set);
  handoff_.queue(job.ready, job.done,
                 [&](const metal::SharedEvent &event, uint64_t wait, uint64_t signal, ane::Handoff::Report report) {
                   program.enqueue(binding, event, wait, signal, std::move(report));
                 });
}

std::vector<uint32_t> AneFfn::functionRowsOf() const {
  std::vector<uint32_t> rows;
  for (const Evaluation &evaluation : evaluations_) rows.push_back(evaluation.rows);
  return rows;
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


// The mixers' projections. A layer joins the ANE's ahead columns before its
// mixer reads them and the deferred ones before its gate, so neither waits
// on more than its own projection: each kind balances the GPU and the ANE
// on its own (ane_ffn::ProjectionKind), over the FFN's handoff and staging.

namespace {

using ane_ffn::Columns;
using ane_ffn::ProjectionKind;

// The kinds that read a mixer's rows and add the residual in their join;
// the others read the hidden rows the FFN's input surfaces hold.
bool outputKind(ProjectionKind kind) {
  return kind == ProjectionKind::GdnOutput || kind == ProjectionKind::AttentionOutput;
}

// The output columns a projection's tensors hold; the ones past them are
// padding, which nothing reads.
uint32_t tensorColumns(const Projection &projection) {
  const QuantizedSegment &last = projection.blocks().segments.back();
  return last.columnOffset + last.outputSize;
}

// Whether the ANE can stage `projection`'s rows: each tensor it stacks a
// quantized one in planes, of whole plane tiles, which the GPU reads
// unrotated over all its inputs.
bool stagesTensors(const Projection &projection) {
  if (projection.rotation || projection.planeInputs()) return false;
  for (const QuantizedSegment &segment : projection.blocks().segments)
    if (segment.isFloat() || segment.outputSize % QUANT_TILE_ROWS) return false;
  return true;
}

// The first tensor of a stacked projection, whose leading rows the GPU's
// part of a split views.
Projection firstTensor(const Projection &projection) {
  if (projection.blocks().segments.size() == 1) return projection;
  QuantizedSegment first = projection.blocks().segments.front();
  const uint32_t rows = first.outputSize;
  return Projection(rows, projection.inputSize, BlockWeights{{std::move(first)}});
}

// Calls `use(planes, begin, length, offset)` for each piece of the columns
// [begin, begin + length) one tensor holds, `begin` the tensor's own row and
// `offset` the piece's from the range's start.
template <class Tensors, class Use>
void eachTensorPiece(const Tensors &tensors, uint32_t begin, uint32_t length, Use &&use) {
  for (const auto &tensor : tensors) {
    const uint32_t from = std::max(begin, tensor.first), to = std::min(begin + length, tensor.first + tensor.rows);
    if (from < to) use(tensor, from - tensor.first, to - from, from - begin);
  }
}

uint32_t countOf(const Columns &columns) {
  uint32_t count = 0;
  for (const auto &[begin, length] : columns) count += length;
  return count;
}

// Whether ane_proj_join maps `columns`: a first range, then one more or equal
// blocks at a fixed stride, each of whole join tiles and whole rows of the
// weight kernels' threadgroups, past the GPU's `gpu` columns and inside the
// projection's `width`.
bool joinable(const Columns &columns, uint32_t gpu, uint32_t width) {
  for (const auto &[begin, length] : columns)
    if (!length || length % ANE_FFN_TILE || length % ANE_FFN_WEIGHT_ROWS || begin < gpu || begin + length > width)
      return false;
  if (columns.size() <= 2) return true;
  const uint32_t length = columns[1].second, stride = columns[2].first - columns[1].first;
  for (size_t index = 1; index < columns.size(); ++index)
    if (columns[index].second != length || columns[index].first != columns[1].first + (index - 1) * stride)
      return false;
  return stride >= length;
}

} // namespace

const char *AneFfn::unsupportedProjections(const ane_ffn::ProjectionLayout &layout) {
  // Two layers stage both weight sets, which verifyProjections() checks.
  if (layout.layers.size() < 2) return "ANE projection split needs two or more layers";
  const Projection &front = *layout.layers.front();
  // Columns past the tensors' are padding, which neither part writes.
  const uint32_t width = tensorColumns(front), inputs = front.inputSize;
  for (const Projection *projection : layout.layers) {
    if (!stagesTensors(*projection)) return "ANE projection split needs unrotated quantized projections";
    if (projection->outputSize != front.outputSize || projection->inputSize != inputs ||
        projection->blocks().segments.size() != front.blocks().segments.size() || tensorColumns(*projection) != width)
      return "ANE projection split layers differ in shape";
  }
  if (inputs % kRotateGroup || inputs / kRotateGroup > ANE_FFN_ROTATE_BLOCKS)
    return "ANE projection split needs inputs ane_ffn_rotate takes";
  if (layout.units < 2 || !layout.unitColumns || layout.unitColumns % QUANT_TILE_ROWS ||
      layout.unitColumns % ANE_FFN_TILE || uint64_t{layout.units} * layout.unitColumns > width)
    return "ANE projection split needs two or more units of whole plane tiles";
  if (!layout.ahead) return "ANE projection split has no columns ahead of the mixer";
  if (uint64_t{layout.units - 1} * layout.unitColumns > front.blocks().segments.front().outputSize)
    return "ANE projection split needs the GPU's columns in the projection's first tensor";
  for (uint32_t aneUnits = 1; aneUnits < layout.units; ++aneUnits) {
    const uint32_t gpu = (layout.units - aneUnits) * layout.unitColumns;
    const Columns ahead = layout.ahead(aneUnits), deferred = layout.deferred ? layout.deferred(aneUnits) : Columns{};
    if (ahead.empty() || !joinable(ahead, gpu, width) || (!deferred.empty() && !joinable(deferred, gpu, width)))
      return "ANE projection split's columns are not ones its join maps";
    if (!deferred.empty() && outputKind(layout.kind)) return "an output projection defers no columns";
    // Every column once, the GPU's or the ANE's: a column of neither would
    // hold what the output held before.
    std::vector<bool> covered(width);
    std::fill_n(covered.begin(), gpu, true);
    for (const Columns *columns : {&ahead, &deferred})
      for (const auto &[begin, length] : *columns)
        for (uint32_t column = begin; column < begin + length; ++column) {
          if (covered[column]) return "ANE projection split's columns overlap";
          covered[column] = true;
        }
    if (!std::ranges::all_of(covered, std::identity{})) return "ANE projection split leaves columns to neither part";
  }
  return nullptr;
}

template <class MakeBuffer, class MakeSurface>
std::unique_ptr<AneFfn::ProjectionSplit> AneFfn::allocateProjection(const ane_ffn::ProjectionLayout &layout,
                                                                    uint32_t aneUnits, MakeBuffer &&buffer,
                                                                    MakeSurface &&surface) {
  if (const char *reason = unsupportedProjections(layout)) throw std::invalid_argument(reason);
  if (!aneUnits || aneUnits >= layout.units)
    throw std::invalid_argument("ANE projection split leaves the GPU or the ANE no units");
  auto split = std::make_unique<ProjectionSplit>();
  split->kind = layout.kind;
  split->units = layout.units;
  split->aneUnits = aneUnits;
  split->width = layout.layers.front()->outputSize;
  split->inputs = layout.layers.front()->inputSize;
  split->gpu = (layout.units - aneUnits) * layout.unitColumns;
  split->segments = segments(split->inputs);
  split->layers = layout.layers;
  const uint64_t layers = layout.layers.size();
  split->gpuPart = buffer(uint64_t{kMaximumRows} * split->gpu * sizeof(uint16_t), "ane projection gpu part");
  if (outputKind(layout.kind)) {
    split->rotated =
        buffer(uint64_t{kMaximumRows} * split->inputs * sizeof(_Float16), "ane projection rotated input");
    for (const uint32_t width : split->segments) split->x.push_back(surface(width, kMaximumRows, Element::Int8));
  }
  const auto columns = [&](ProjectionColumns &part, Columns given, const char *label) {
    part.columns = std::move(given);
    part.count = countOf(part.columns);
    if (!part.count) return;
    part.rowScales = buffer(layers * part.count * sizeof(_Float16), label);
    for (std::vector<ane::Surface> &set : part.weights)
      for (const uint32_t width : split->segments) set.push_back(surface(part.count, width, Element::Int8));
    for (ane::Surface &scale : part.scales) scale = surface(part.count, 1, Element::Float16);
    part.partial = surface(part.count, kMaximumRows, Element::Float16);
  };
  columns(split->ahead, layout.ahead(aneUnits), "ane projection row scales");
  columns(split->deferred, layout.deferred ? layout.deferred(aneUnits) : Columns{},
          "ane projection deferred row scales");
  return split;
}

uint64_t AneFfn::plannedProjectionBytes(const ane_ffn::ProjectionLayout &layout, uint32_t aneUnits) {
  uint64_t bytes = 0;
  static_cast<void>(allocateProjection(
      layout, aneUnits,
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

std::string AneFfn::projectionFunction(bool deferred, uint32_t rows) {
  return (deferred ? "deferred" : "ahead") + std::to_string(rows);
}

// x<k>: the chunk's rotated int8 inputs of segment k, the FFN's for an input
// projection; w<k>: the ANE's rotated int8 rows of the projection over
// segment k; s: their per-row scales. A function's output holds the ANE's
// columns before each token's input scale, which the join applies.
std::string AneFfn::projectionProgram(const ProjectionSplit &split, const std::vector<ane::Surface> &x,
                                      std::span<const uint32_t> rows) {
  const std::string unit = fp16(1.0 / ANE_FFN_INT8_UNIT);
  std::string text = "program(1.3)\n{\n";
  for (const bool deferred : {false, true}) {
    const ProjectionColumns &part = deferred ? split.deferred : split.ahead;
    if (!part.count) continue;
    for (const uint32_t count : rows) {
      std::string parameters, body, total;
      const auto line = [&](const std::string &statement) { body += "        " + statement + ";\n"; };
      for (uint32_t k = 0; k < split.segments.size(); ++k) {
        const std::string i = std::to_string(k);
        const uint32_t width = split.segments[k];
        parameters += (k ? ", " : "") + x[k].bufferType(width, count) + " x" + i + ", " +
                      part.weights[0][k].bufferType(part.count, width) + " w" + i;
        line(tensor("int8", width, count) + " x" + i + "_t = tensor_buffer_to_tensor<ios17>(input = x" + i + ")");
        line(tensor("int8", part.count, width) + " w" + i + "_t = tensor_buffer_to_tensor<ios17>(input = w" + i +
             ")");
        line(tensor("fp16", width, count) + " x" + i + "_d = dequantize(input = x" + i + "_t, scale = " + unit + ")");
        line(tensor("fp16", part.count, width) + " w" + i + "_d = dequantize(input = w" + i + "_t, scale = " + unit +
             ")");
        line(tensor("fp16", part.count, count) + " m" + i +
             " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = w" + i + "_d, y = x" + i + "_d)");
        if (k) {
          line(tensor("fp16", part.count, count) + " sum" + i + " = add(x = " + total + ", y = m" + i + ")");
          total = "sum" + i;
        } else {
          total = "m0";
        }
      }
      parameters += ", " + part.scales[0].bufferType(part.count, 1) + " s";
      line(tensor("fp16", part.count, 1) + " s_t = tensor_buffer_to_tensor<ios17>(input = s)");
      line(tensor("fp16", part.count, count) + " ys = mul(x = " + total + ", y = s_t)");
      line(part.partial.bufferType(part.count, count) +
           " y = tensor_to_tensor_buffer<ios17>(input = ys, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
           "strides = tensor<int64, [4]>(" + part.partial.strides(part.count) + "))");
      text += "    func " + projectionFunction(deferred, count) + "<ios18>(" + parameters + ") {\n" + body +
              "    } -> (y);\n";
    }
  }
  return text + "}\n";
}

void AneFfn::addProjections(const ane_ffn::ProjectionLayout &layout, uint32_t aneUnits) {
  addProjections(layout, aneUnits, functionRowsOf());
}

void AneFfn::addProjections(const ane_ffn::ProjectionLayout &layout, uint32_t aneUnits,
                            std::span<const uint32_t> functionRows) {
  std::unique_ptr<ProjectionSplit> &slot = projections_.at(static_cast<size_t>(layout.kind));
  const std::string name = ane_ffn::projectionName(layout.kind);
  if (slot) throw std::logic_error("ANE split already takes the " + name + " projections");
  if (!outputKind(layout.kind) && !layout.layers.empty() && layout.layers.front()->inputSize != shape_.hidden)
    throw std::invalid_argument("ANE split's input projections read other rows than its FFN's");
  uint64_t allocated = 0;
  std::unique_ptr<ProjectionSplit> split = allocateProjection(
      layout, aneUnits,
      [&](uint64_t bytes, const char *label) {
        metal::MetalBuffer buffer = backend_.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
        allocated += buffer.allocatedBytes();
        return buffer;
      },
      [&](uint32_t rows, uint32_t width, Element element) {
        ane::Surface surface = ane::Surface::create(backend_, rows, width, element);
        allocated += surface.buffer.allocatedBytes();
        return surface;
      });
  split->allocatedBytes = allocated;
  for (const Projection *source : layout.layers) {
    std::vector<Planes> &tensors = split->planes.emplace_back();
    for (const QuantizedSegment &segment : source->blocks().segments) tensors.emplace_back(segment, source->inputSize);
    split->gpuRows.push_back(firstTensor(*source).leadingRows(backend_, split->gpu));
  }
  const std::vector<ane::Surface> &x = inputsOf(*split);
  split->program = std::make_unique<ane::Program>(projectionProgram(*split, x, functionRows),
                                                  std::span<const uint8_t>{},
                                                  ane::Program::Limits{kProgramLimit, interrupted_});
  for (const bool deferred : {false, true}) {
    ProjectionColumns &part = deferred ? split->deferred : split->ahead;
    if (!part.count) continue;
    for (const uint32_t rows : functionRows) {
      Evaluation &evaluation = part.evaluations.emplace_back();
      evaluation.rows = rows;
      const uint32_t procedure = split->program->procedure(projectionFunction(deferred, rows));
      for (uint32_t set = 0; set < 2; ++set) {
        std::vector<ane::Surface> surfaces;
        for (const std::string &input : split->program->inputs(procedure)) {
          const uint32_t k = input.size() > 1 ? static_cast<uint32_t>(std::stoul(input.substr(1))) : 0;
          if (input[0] == 'x')
            surfaces.push_back(x.at(k));
          else if (input[0] == 'w')
            surfaces.push_back(part.weights[set].at(k));
          else if (input == "s")
            surfaces.push_back(part.scales[set]);
          else
            throw std::logic_error("ANE projection program has an unknown input " + input);
        }
        evaluation.bindings.push_back(split->program->bind(procedure, surfaces, part.partial));
      }
    }
  }

  // Each ANE row's int8 scale over its rotated inputs, once.
  metal::CommandGraph graph;
  for (uint32_t layer = 0; layer < split->planes.size(); ++layer) {
    for (const ProjectionColumns *part : {&split->ahead, &split->deferred}) {
      uint32_t offset = 0;
      for (const auto &[begin, length] : part->columns) {
        eachTensorPiece(split->planes[layer], begin, length,
                        [&](const Planes &planes, uint32_t row, uint32_t rows, uint32_t piece) {
                          graph.add(kernel("ane_ffn_row_scale_gguf", Rotation::Inputs),
                                    {planes.buffers[0], planes.buffers[1], planes.buffers[2],
                                     backend_.view(part->rowScales,
                                                   (uint64_t{layer} * part->count + offset + piece) * sizeof(_Float16),
                                                   uint64_t{rows} * sizeof(_Float16)),
                                     memory_.signs},
                                    AneFfnWeightParams{planes.groups, row, 0, split->inputs, 0, 0, planes.format},
                                    {rows / ANE_FFN_WEIGHT_ROWS, 1, 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
                        });
        offset += length;
      }
    }
  }
  static_cast<void>(backend_.submitCommandAsync(graph.dispatches()).wait());
  allocatedBytes_ += split->allocatedBytes;
  slot = std::move(split);
}

void AneFfn::removeProjections(ane_ffn::ProjectionKind kind) {
  std::unique_ptr<ProjectionSplit> &slot = projections_.at(static_cast<size_t>(kind));
  if (!slot) return;
  if (unfinished_ || !jobs_.empty() || !handoff_.idle())
    throw std::logic_error("ANE projection split removed while its evaluations may run");
  allocatedBytes_ -= slot->allocatedBytes;
  slot.reset();
}

bool AneFfn::projects(ane_ffn::ProjectionKind kind, uint32_t rows) const {
  const ProjectionSplit *split = projection(kind);
  return split && !split->lost && splits(rows);
}

double AneFfn::projectionShare(ane_ffn::ProjectionKind kind) const noexcept {
  const ProjectionSplit *split = projection(kind);
  return split ? static_cast<double>(split->aneUnits) / split->units : 0.0;
}

void AneFfn::setProjectionBreaker(ane_ffn::ProjectionKind kind, ane_ffn::Breaker breaker) {
  ProjectionSplit *split = projection(kind);
  if (!split)
    throw std::logic_error(std::string("ANE split judges ") + ane_ffn::projectionName(kind) +
                           " projections it does not take");
  split->breaker = std::move(breaker);
}

// Each range's rows follow the range before's in the surfaces.
void AneFfn::addProjectionWeights(const ProjectionSplit &split, metal::CommandGraph &graph, uint32_t layer,
                                  uint32_t set) const {
  const std::vector<Planes> &tensors = split.planes.at(layer);
  for (const ProjectionColumns *part : {&split.ahead, &split.deferred}) {
    if (!part->count) continue;
    const ane::Surface &scale = part->scales[set];
    const uint32_t scaleStride = scale.strideBytes / uint32_t{sizeof(_Float16)};
    uint32_t input = 0;
    for (uint32_t k = 0; k < split.segments.size(); ++k) {
      const ane::Surface &output = part->weights[set][k];
      const uint32_t width = split.segments[k];
      requireBytes(output.buffer, rowBytes(part->count, output.strideBytes, width, 1),
                   "ANE projection weight surface");
      uint32_t offset = 0;
      for (const auto &[begin, length] : part->columns) {
        eachTensorPiece(tensors, begin, length, [&](const Planes &planes, uint32_t row, uint32_t rows, uint32_t piece) {
          const uint32_t at = offset + piece;
          graph.add(kernel("ane_ffn_weights_gguf", Rotation::Inputs),
                    {planes.buffers[0], planes.buffers[1], planes.buffers[2],
                     backend_.view(part->rowScales, (uint64_t{layer} * part->count + at) * sizeof(_Float16),
                                   uint64_t{rows} * sizeof(_Float16)),
                     backend_.view(output.buffer, uint64_t{at} * output.strideBytes, uint64_t{rows} * output.strideBytes),
                     backend_.view(scale.buffer, uint64_t{at} * scale.strideBytes, uint64_t{rows} * scale.strideBytes),
                     memory_.signs},
                    AneFfnWeightParams{planes.groups, row, input, width, output.strideBytes, scaleStride, planes.format},
                    {rows / ANE_FFN_WEIGHT_ROWS, width / blockOf(Rotation::Inputs), 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
        });
        offset += length;
      }
      input += width;
    }
  }
}

// The set a kind's next layer reads was last read by the layer before its
// last, whose evaluations the last's joins followed.
void AneFfn::addNextProjectionWeights(metal::CommandGraph &graph) {
  for (const std::unique_ptr<ProjectionSplit> &split : projections_) {
    if (!split || split->lost || split->next >= split->planes.size() || split->staged == split->next) continue;
    addProjectionWeights(*split, graph, split->next, split->next & 1);
    split->staged = split->next;
  }
}

AneProjJoinParams AneFfn::joinParams(const ProjectionSplit &split, const ProjectionColumns &columns,
                                     uint32_t rows) const {
  const Columns &ranges = columns.columns;
  AneProjJoinParams params{split.width,
                           &columns == &split.ahead ? split.gpu : 0,
                           columns.partial.strideBytes / uint32_t{sizeof(_Float16)},
                           rows,
                           ranges.front().first,
                           ranges.front().second,
                           0,
                           0,
                           0,
                           0};
  if (ranges.size() >= 2) params.begin2 = ranges[1].first;
  if (ranges.size() > 2) {
    params.block_length = ranges[1].second;
    params.block_stride = ranges[2].first - ranges[1].first;
  }
  return params;
}

void AneFfn::addProjection(ane_ffn::ProjectionKind kind, metal::CommandGraph &graph, uint32_t layer,
                           metal::MetalBuffer input, metal::MetalBuffer output, uint32_t rows, LinearScratch scratch,
                           metal::MetalBuffer residual) {
  encodeProjection(kind, graph, layer, std::move(input), std::move(output), rows, scratch, std::move(residual),
                   Parts::Both);
}

void AneFfn::addDeferredJoin(ane_ffn::ProjectionKind kind, metal::CommandGraph &graph, metal::MetalBuffer output,
                             uint32_t rows) {
  encodeDeferredJoin(kind, graph, std::move(output), rows, Parts::Both);
}

// The GPU's leading columns and the ANE's ahead columns of the layer, joined;
// the ANE's deferred columns start as its ahead ones end, and the GPU waits
// for them in encodeDeferredJoin(), which must come before the GPU next
// raises the event: a higher value would meet that wait.
void AneFfn::encodeProjection(ane_ffn::ProjectionKind kind, metal::CommandGraph &graph, uint32_t layer,
                              metal::MetalBuffer input, metal::MetalBuffer output, uint32_t rows,
                              LinearScratch scratch, metal::MetalBuffer residual, Parts parts) {
  ProjectionSplit *split = projection(kind);
  const std::string name = ane_ffn::projectionName(kind);
  if (!split || split->lost) throw std::logic_error("ANE split adds " + name + " projections it does not take");
  if (rows < kMinimumRows || rows > kMaximumRows)
    throw std::invalid_argument("ANE projection split does not take a chunk of these rows");
  if (layer != split->next || layer >= split->planes.size())
    throw std::logic_error("ANE split adds " + name + " projection " + std::to_string(layer) + ", not its next " +
                           std::to_string(split->next) + " (begin() starts each command at 0)");
  if (outputKind(kind) != static_cast<bool>(residual))
    throw std::invalid_argument("an output projection split takes its residual, an input projection none");
  if (split->deferredDone)
    throw std::logic_error("ANE split adds a " + name +
                           " projection before joining the last one's deferred columns");
  const uint64_t outputBytes = uint64_t{rows} * split->width * sizeof(uint16_t);
  requireBytes(input, uint64_t{rows} * split->inputs * sizeof(uint16_t), "ANE projection input");
  requireBytes(output, outputBytes, "ANE projection output");
  if (residual) requireBytes(residual, outputBytes, "ANE projection residual");
  const auto found = std::ranges::find_if(split->ahead.evaluations,
                                          [&](const Evaluation &evaluation) { return evaluation.rows >= rows; });
  if (found == split->ahead.evaluations.end())
    throw std::invalid_argument("ANE projection split has no function of the chunk's rows");
  const auto evaluation = static_cast<uint32_t>(found - split->ahead.evaluations.begin());
  const uint32_t set = layer & 1, tiles = (rows + ANE_FFN_TILE - 1) / ANE_FFN_TILE;
  const bool ane = parts == Parts::Both;
  const auto owner = static_cast<uint8_t>(kind);
  // A command's first layer of the kind stages its own rows; each later one
  // finds them staged during the FFN before it (addNextProjectionWeights).
  if (split->staged != layer) {
    addProjectionWeights(*split, graph, layer, set);
    split->staged = layer;
  }
  const metal::MetalBuffer rotated = split->rotated ? split->rotated : memory_.rotated;
  const std::vector<ane::Surface> &x = inputsOf(*split);
  graph.add("ane_ffn_rotate", {input, memory_.signs, rotated, memory_.tokenScale.buffer},
            AneFfnRotateParams{split->inputs}, {rows, 1, 1}, {ANE_FFN_ROTATE_THREADS, 1, 1});
  uint32_t channel = 0;
  for (uint32_t k = 0; k < split->segments.size(); ++k) {
    graph.add("ane_ffn_pack", {rotated, x[k].buffer}, AneFfnPackParams{split->inputs, channel, x[k].strideBytes},
              {tiles, split->segments[k] / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
    channel += split->segments[k];
  }
  const uint64_t ready = ane ? handoff_.next() : 0;
  if (ane) graph.signal(handoff_.event(), ready);
  linear_.addPrefill(graph, input, split->gpuRows[layer], split->gpuPart, rows, scratch);
  const uint64_t done = ane ? handoff_.next() : 0;
  if (ane) graph.wait(handoff_.event(), done);
  const AneProjJoinParams params = joinParams(*split, split->ahead, rows);
  const uint64_t columnTiles = (split->gpu + split->ahead.count) / ANE_FFN_TILE;
  if (residual)
    graph.add("ane_proj_join_residual",
              {output, split->gpuPart, split->ahead.partial.buffer, memory_.tokenScale.buffer, memory_.status,
               residual},
              params, {tiles, columnTiles, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  else
    graph.add("ane_proj_join",
              {output, split->gpuPart, split->ahead.partial.buffer, memory_.tokenScale.buffer, memory_.status},
              params, {tiles, columnTiles, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  if (ane) {
    jobs_.push_back({evaluation, set, ready, done, owner});
    if (split->deferred.count) {
      split->deferredDone = handoff_.next();
      jobs_.push_back({evaluation, set, done, split->deferredDone, owner, true});
    }
  }
  encodingRows_ = split->ahead.evaluations[evaluation].rows;
  ++split->next;
}

void AneFfn::encodeDeferredJoin(ane_ffn::ProjectionKind kind, metal::CommandGraph &graph, metal::MetalBuffer output,
                                uint32_t rows, Parts parts) {
  ProjectionSplit *split = projection(kind);
  const std::string name = ane_ffn::projectionName(kind);
  if (!split || !split->deferred.count || !split->next)
    throw std::logic_error("ANE split joins deferred columns of no " + name + " projection");
  if (parts == Parts::Both && !split->deferredDone)
    throw std::logic_error("ANE split joins the deferred columns of a " + name + " projection twice");
  requireBytes(output, uint64_t{rows} * split->width * sizeof(uint16_t), "ANE projection output");
  if (parts == Parts::Both) graph.wait(handoff_.event(), std::exchange(split->deferredDone, 0));
  graph.add("ane_proj_join",
            {output, split->gpuPart, split->deferred.partial.buffer, memory_.tokenScale.buffer, memory_.status},
            joinParams(*split, split->deferred, rows),
            {(rows + ANE_FFN_TILE - 1) / ANE_FFN_TILE, split->deferred.count / ANE_FFN_TILE, 1},
            {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
}

// What the ANE adds is the GPU alone's output in the ANE's columns, which the
// split's difference from the GPU alone is held against, as verify() holds
// the FFN's, over each function's rows of the first layer and then of the
// second, which reads the other weight set.
double AneFfn::verifyProjections(ane_ffn::ProjectionKind kind, metal::MetalBuffer input, metal::MetalBuffer output,
                                 LinearScratch scratch, metal::MetalBuffer residual) {
  try {
    return checkProjections(kind, std::move(input), std::move(output), scratch, std::move(residual));
  } catch (...) {
    // The split goes on without the kind; one whose evaluations may still
    // run keeps it, never to split it.
    try {
      removeProjections(kind);
    } catch (const std::logic_error &) {
      if (ProjectionSplit *split = projection(kind)) split->lost = true;
    }
    throw;
  }
}

double AneFfn::checkProjections(ane_ffn::ProjectionKind kind, metal::MetalBuffer input, metal::MetalBuffer output,
                                LinearScratch scratch, metal::MetalBuffer residual) {
  ProjectionSplit *split = projection(kind);
  const std::string name = ane_ffn::projectionName(kind);
  if (!split) throw std::logic_error("ANE split verifies " + name + " projections it does not take");
  if (handoff_.retired()) throw std::runtime_error("ANE FFN split stopped (" + handoff_.reason() + ")");
  fillNormalized(input);
  if (residual) std::memset(residual.contents(), 0, residual.sizeBytes());
  // The ANE's columns; padding past the tensors' columns neither part writes.
  const uint32_t columns = tensorColumns(*split->layers.front());
  std::vector<bool> ane(split->width);
  for (const ProjectionColumns *part : {&split->ahead, &split->deferred})
    for (const auto &[begin, length] : part->columns) std::fill_n(ane.begin() + begin, length, true);
  // Layers 0 to `last` of the kind, of `rows` rows, the GPU alone's or split.
  const auto forward = [&](bool splitting, uint32_t last, uint32_t rows) {
    metal::CommandGraph graph;
    if (splitting) {
      for (const ProjectionColumns *part : {&split->ahead, &split->deferred})
        if (part->count)
          std::fill_n(static_cast<uint16_t *>(part->partial.buffer.contents()),
                      part->partial.buffer.sizeBytes() / sizeof(uint16_t), kHalfNaN);
      static_cast<void>(begin());
    }
    for (uint32_t layer = 0; layer <= last; ++layer) {
      if (splitting) {
        addProjection(kind, graph, layer, input, output, rows, scratch, residual);
        if (split->deferred.count) addDeferredJoin(kind, graph, output, rows);
      } else if (residual) {
        linear_.addPrefillResidual(graph, input, *split->layers[layer], residual, output, rows, scratch);
      } else {
        linear_.addPrefill(graph, input, *split->layers[layer], output, rows, scratch);
      }
    }
    if (splitting) {
      static_cast<void>(commit(graph, {}).wait());
      if (!completed())
        throw std::runtime_error("ANE split of the " + name + " projections of " + std::to_string(rows) +
                                 " rows failed (" + reason() + ")");
    } else {
      static_cast<void>(backend_.submitCommandAsync(graph.command()).wait());
    }
    return bf16Values(output, uint64_t{rows} * split->width);
  };
  double worst = 0.0;
  for (uint32_t last = 0; last < 2; ++last) {
    const std::vector<float> gpu = forward(false, last, kMaximumRows);
    for (const Evaluation &evaluation : split->ahead.evaluations) {
      const uint32_t rows = evaluation.rows;
      const std::vector<float> got = forward(true, last, rows);
      const auto fail = [&](const std::string &why) {
        throw std::runtime_error("ANE split's " + name + " projections of " + std::to_string(rows) + " rows " + why);
      };
      double difference = 0.0, contribution = 0.0;
      for (size_t i = 0; i < got.size(); ++i) {
        if (i % split->width >= columns) continue;
        if (!std::isfinite(got[i])) fail("are not finite");
        difference += (double(got[i]) - gpu[i]) * (double(got[i]) - gpu[i]);
        if (ane[i % split->width]) contribution += double(gpu[i]) * gpu[i];
      }
      const double error = std::sqrt(difference / contribution);
      if (!(error <= kMaximumAneError))
        fail("differ from the GPU alone's by " + std::to_string(100.0 * error) + "% of the Neural Engine's part");
      worst = std::max(worst, error);
    }
  }
  return worst;
}

} // namespace splash::ops
