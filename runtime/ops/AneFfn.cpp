#include "ops/AneFfn.hpp"

#include "Checked.hpp"
#include "StderrLine.hpp"
#include "Supervision.hpp"
#include "metal/abi/AneFfn.h"
#include "metal/abi/QuantFormat.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <bit>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iomanip>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace splash::ops {
namespace {

using Element = ane::Surface::Element;

// The ANE program multiplies gate's and up's hidden inputs, and down's ANE
// inputs, in segments of kSegment channels and a last of the rest, a matmul
// each: on an M5 Pro, two of 2560 run a 5120-channel FFN about 14% faster
// than one of 5120 (ane-ffn's FFN, 2048 rows). The weights of a segment are
// staged in whole blocks of either rotation, and its inputs packed in whole
// tiles.
constexpr uint32_t kSegment = 2560;
static_assert(kSegment % ANE_FFN_INPUT_BLOCK == 0 && kSegment % ANE_FFN_INTERMEDIATE_BLOCK == 0 &&
                  kSegment % ANE_FFN_TILE == 0,
              "segments hold whole rotation blocks and packing tiles");
// The hidden channels ane_ffn_rotate's simdgroups rotate a block each of.
constexpr uint32_t kRotateGroup = ANE_FFN_INPUT_BLOCK * (ANE_FFN_ROTATE_THREADS / 32);
static_assert(kRotateGroup % ANE_FFN_TILE == 0 && kRotateGroup % ANE_FFN_WEIGHT_ROWS == 0,
              "a hidden size the rotation takes packs its last segment in whole tiles and stages down's rows whole");
// The channels a split moves in: whole blocks of the intermediate rotation
// for the ANE, whole plane tiles of the weight planes (QUANT_TILE_ROWS) for
// the GPU.
constexpr uint32_t kChannelUnit = std::max(QUANT_TILE_ROWS, ANE_FFN_INTERMEDIATE_BLOCK);
static_assert(kChannelUnit % ANE_FFN_WEIGHT_ROWS == 0,
              "the weight kernels' threadgroups take whole rows of gate's and up's ANE channels");
// How long an ANE evaluation may run once the GPU has raised its `ready`
// before the split stops (ane::Handoff). An evaluation takes 10-60 ms. Metal
// fails a command buffer whose event wait is still unmet
// supervision::kGpuWaitLimit after its queue reached it ("Caused GPU Timeout
// Error"), and once two have failed so it ignores the process's later
// commands on that queue. The queue reaches each wait once the one before it
// was met, so at the release a wait has waited through one layer's GPU part
// and at most the bound; a command's first wait also through the work from
// the command's submission, which image encodes no longer add
// (Runtime::prefillAsync).
constexpr auto kHandoffBound = std::chrono::seconds(2);
static_assert(kHandoffBound < supervision::kGpuWaitLimit);
// How long the service may take to compile and load the program, far past
// what it takes: its one function of kEvaluationRows rows compiles cold in
// 1.4 s on an M5 Max with the ANE taking every channel of Qwen3.8-27B's FFN,
// and in about five times that on an M6.
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
  if (!aneUnits || aneUnits > intermediate / kChannelUnit)
    throw std::invalid_argument("ANE FFN split gives the ANE no channels, or more than there are");
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

std::string AneFfn::calibrationKey(std::span<const SwiGluProjections> layers, const Linear &linear) {
  if (const char *reason = unsupported(layers)) throw std::invalid_argument(reason);
  std::string key = "ane-ffn calibration";
  for (const SwiGluProjections &layer : layers)
    for (const Projection *projection : {layer.gate, layer.up, layer.down})
      key += " " + std::to_string(projection->outputSize) + "x" + std::to_string(projection->inputSize) + ":g" +
             std::to_string(projection->blocks().segments.front().formatId);
  return key + " int8-activations " + (linear.int8Activations() == Int8Activations::On ? "on" : "off");
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

// The rows of the chunks verify() checks: one piece, pieces whose last pads,
// and the most.
constexpr std::array<uint32_t, 3> verifiedRows() {
  return {AneFfn::kEvaluationRows, AneFfn::kMaximumRows - AneFfn::kEvaluationRows / 2, AneFfn::kMaximumRows};
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
  shape.inputSegments = segments(shape.hidden);
  shape.downSegments = segments(shape.ane);
  return shape;
}

template <class MakeBuffer, class MakeSurface>
AneFfn::Memory AneFfn::allocate(const Shape &shape, uint32_t slots, MakeBuffer &&buffer, MakeSurface &&surface) {
  if (!slots) throw std::invalid_argument("ANE FFN split needs a slot");
  Memory memory;
  memory.signs = buffer(ANE_FFN_INTERMEDIATE_BLOCK * sizeof(float), "ane ffn signs");
  memory.rowScales = buffer(uint64_t{shape.layers} * (2 * shape.ane + shape.hidden) * sizeof(_Float16),
                            "ane ffn row scales");
  memory.rotated = buffer(uint64_t{kMaximumRows} * shape.hidden * sizeof(_Float16), "ane ffn rotated input");
  memory.status = buffer(sizeof(uint32_t), "ane ffn status");
  for (uint32_t index = 0; index < slots; ++index) {
    Slot &slot = memory.slots.emplace_back();
    for (uint32_t count = 0; count < kPieces; ++count) {
      Piece &piece = slot.pieces.emplace_back();
      for (const uint32_t width : shape.inputSegments)
        piece.inputs.push_back(surface(width, kEvaluationRows, Element::Int8));
      piece.tokenScale = surface(1, kEvaluationRows, Element::Float16);
      piece.partial = surface(shape.hidden + 1, kEvaluationRows, Element::Float16);
    }
  }
  for (Weights &set : memory.sets) {
    for (const uint32_t width : shape.inputSegments) {
      set.gate.push_back(surface(shape.ane, width, Element::Int8));
      set.up.push_back(surface(shape.ane, width, Element::Int8));
    }
    for (const uint32_t width : shape.downSegments) set.down.push_back(surface(shape.hidden, width, Element::Int8));
    set.gateScale = surface(shape.ane, 1, Element::Float16);
    set.upScale = surface(shape.ane, 1, Element::Float16);
    set.downScale = surface(shape.hidden, 1, Element::Float16);
  }
  return memory;
}

uint64_t AneFfn::plannedBytes(std::span<const SwiGluProjections> layers, uint32_t aneUnits, uint32_t slots) {
  uint64_t bytes = 0;
  static_cast<void>(allocate(
      shapeOf(layers, aneUnits), slots,
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
std::vector<AneFfn::Input> AneFfn::programInputs(const Shape &shape, const Memory &memory, uint32_t slot,
                                                 uint32_t piece) {
  const auto both = [&](auto member) {
    return std::array{&(memory.sets[0].*member), &(memory.sets[1].*member)};
  };
  const Piece &own = memory.slots.at(slot).pieces.at(piece);
  std::vector<Input> inputs;
  for (uint32_t k = 0; k < own.inputs.size(); ++k)
    inputs.push_back({"x" + std::to_string(k), {&own.inputs[k], &own.inputs[k]}, shape.inputSegments[k], 0});
  inputs.push_back({"tx", {&own.tokenScale, &own.tokenScale}, 1, 0});
  for (uint32_t k = 0; k < own.inputs.size(); ++k) {
    const uint32_t width = shape.inputSegments[k];
    inputs.push_back({"wg" + std::to_string(k), {&memory.sets[0].gate[k], &memory.sets[1].gate[k]}, shape.ane, width});
    inputs.push_back({"wu" + std::to_string(k), {&memory.sets[0].up[k], &memory.sets[1].up[k]}, shape.ane, width});
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
  const auto segmentCount = static_cast<uint32_t>(shape.inputSegments.size());
  const uint32_t channels = shape.ane;
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

  for (uint32_t k = 0; k < segmentCount; ++k) {
    const std::string x = "x" + std::to_string(k);
    f16(x + "_d", shape.inputSegments[k], rows, "dequantize(input = " + x + "_t, scale = " + unit + ")");
  }
  for (const char *projection : {"g", "u"}) {
    const std::string p = projection;
    for (uint32_t k = 0; k < segmentCount; ++k)
      matmul(p + "m" + std::to_string(k), "w" + p + std::to_string(k), "x" + std::to_string(k) + "_d", channels,
             shape.inputSegments[k]);
    f16(p + "s", channels, rows, "mul(x = " + sum(p + "m", segmentCount, channels) + ", y = s" + p + "_t)");
  }
  const std::string c = std::to_string(channels), r = std::to_string(rows),
                    block = std::to_string(ANE_FFN_INTERMEDIATE_BLOCK);
  // silu(g) = g/2 (1 + tanh(g/2)): through the ANE's sigmoid, layers whose
  // gate pre-activations are about 0.1 differed from the GPU alone by 6.5%
  // RMS, through its tanh by 1.6%.
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

std::string AneFfn::program(const Shape &shape, std::span<const Input> inputs, const ane::Surface &output) {
  return "program(1.3)\n{\n" + function(shape, inputs, output, kEvaluationRows) + "}\n";
}

// The jobs of a committed command not yet queued, in the order the GPU
// raises their events, and the thread that queues each once fewer than
// kQueueDepth are outstanding: the service refuses its 128th outstanding
// request, and a pipelined command holds one job per layer and slot. The ANE
// runs its requests in the order they are queued; once the GPU waits for a
// job, it has raised the `ready` of every job before it, which then run and
// report, so the job it waits for is queued.
struct AneFfn::Feeder final {
  static constexpr uint32_t kQueueDepth = 64;
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<Job> jobs;
  uint32_t outstanding = 0;
  bool stopped = false;
  std::thread thread;
};

AneFfn::AneFfn(metal::MetalBackend &backend, const Linear &linear, std::span<const SwiGluProjections> layers,
               uint32_t aneUnits, std::function<bool()> interrupted, uint32_t slots)
    : backend_(backend), linear_(linear), shape_(shapeOf(layers, aneUnits)), nextLayer_(slots, 0),
      pending_(slots), feeder_(std::make_shared<Feeder>()) {
  if (!slots) throw std::invalid_argument("ANE FFN split needs a slot");
  for (uint32_t slot = 0; slot < slots; ++slot)
    handoffs_.push_back(std::make_unique<ane::Handoff>(backend, kHandoffBound));
  // What Metal allocates, which the memory audit compares with the plan.
  memory_ = allocate(
      shape_, slots,
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

  for (const SwiGluProjections &source : layers) {
    Layer &layer = layers_.emplace_back(
        Layer{{Planes(*source.gate), Planes(*source.up), Planes(*source.down)}, {}, {}, {}});
    if (shape_.gpu) {
      layer.gate.emplace(source.gate->rows(backend_, {{0, shape_.gpu}}));
      layer.up.emplace(source.up->rows(backend_, {{0, shape_.gpu}}));
      layer.down.emplace(source.down->leadingInputs(shape_.gpu));
    }
  }

  // Every piece's surfaces have the first's shapes and strides, which the
  // program declares.
  program_ = std::make_unique<ane::Program>(
      program(shape_, programInputs(shape_, memory_, 0, 0), memory_.slots.front().pieces.front().partial),
      rotationBlob(shape_.ane, signs), ane::Program::Limits{kProgramLimit, std::move(interrupted)});
  const uint32_t procedure = program_->procedure(functionName(kEvaluationRows));
  for (uint32_t set = 0; set < 2; ++set)
    for (uint32_t slot = 0; slot < slots; ++slot)
      for (uint32_t piece = 0; piece < kPieces; ++piece) {
        const std::vector<Input> inputs = programInputs(shape_, memory_, slot, piece);
        std::vector<ane::Surface> surfaces;
        for (const std::string &name : program_->inputs(procedure)) {
          const auto input = std::ranges::find(inputs, name, &Input::name);
          if (input == inputs.end()) throw std::logic_error("ANE FFN program has an unknown input " + name);
          surfaces.push_back(*input->surfaces[set]);
        }
        bindings_.push_back(program_->bind(procedure, surfaces, memory_.slots[slot].pieces[piece].partial));
      }
  feeder_->thread = std::thread([this, feeder = feeder_] {
    std::unique_lock lock(feeder->mutex);
    for (;;) {
      feeder->changed.wait(lock, [&] {
        return feeder->stopped || (!feeder->jobs.empty() && feeder->outstanding < Feeder::kQueueDepth);
      });
      if (feeder->stopped) return;
      const Job job = feeder->jobs.front();
      feeder->jobs.pop_front();
      ++feeder->outstanding;
      lock.unlock();
      queue(job, [feeder] {
        std::lock_guard reported(feeder->mutex);
        --feeder->outstanding;
        feeder->changed.notify_all();
      });
      lock.lock();
    }
  });

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

AneFfn::~AneFfn() {
  // The feeder queues nothing more; then each handoff, destroyed first of the
  // members, waits for its slot's evaluations to report.
  {
    std::lock_guard lock(feeder_->mutex);
    feeder_->stopped = true;
    feeder_->jobs.clear();
  }
  feeder_->changed.notify_all();
  if (feeder_->thread.joinable()) feeder_->thread.join();
}

double AneFfn::share() const noexcept { return static_cast<double>(shape_.ane) / shape_.intermediate; }

bool AneFfn::retired() const {
  return std::ranges::any_of(handoffs_, [](const auto &handoff) { return handoff->retired(); });
}

std::string AneFfn::reason() const {
  for (const auto &handoff : handoffs_)
    if (std::string why = handoff->reason(); !why.empty()) return why;
  return {};
}

void AneFfn::retireAll(const std::string &reason) {
  for (const auto &handoff : handoffs_) handoff->retire(reason);
}

void AneFfn::cancelAll(const std::string &reason) {
  {
    std::lock_guard lock(feeder_->mutex);
    feeder_->jobs.clear();
  }
  for (const auto &handoff : handoffs_) handoff->cancel(reason);
}

void AneFfn::setMinimumRows(uint32_t rows) {
  if (rows < kMinimumRows || rows > (slots() > 1 ? kPipelinedOnly : kMaximumRows))
    throw std::invalid_argument("ANE FFN split takes no chunk of " + std::to_string(rows) + " rows");
  minimumRows_ = rows;
}

bool AneFfn::splits(uint32_t rows, bool pipelined) const {
  return rows >= (pipelined ? kMinimumRows : minimumRows_) && rows <= kMaximumRows && !retired();
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
  uint32_t input = 0;
  for (size_t k = 0; k < shape_.inputSegments.size(); ++k) {
    const uint32_t width = shape_.inputSegments[k];
    add(Matrix::Gate, target.gate[k], target.gateScale, shape_.gpu, input, width, shape_.ane, Rotation::Inputs);
    add(Matrix::Up, target.up[k], target.upScale, shape_.gpu, input, width, shape_.ane, Rotation::Inputs);
    input += width;
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
    cancelAll("a command it split did not finish");
    warnStopped();
  }
  jobs_.clear();
  std::ranges::fill(nextLayer_, 0u);
  std::ranges::fill(pending_, Pending{});
  staged_.reset();
  return !retired();
}

void AneFfn::add(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn,
                 metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows) {
  encode(graph, layer, ffn, residual, output, rows, Parts::Both);
}

void AneFfn::encode(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn,
                    metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows, Parts parts) {
  startIn(graph, layer, 0, ffn, std::move(residual), std::move(output), rows, parts);
  stageNext(graph, layer);
  join(graph, 0);
}

void AneFfn::start(metal::CommandGraph &graph, uint32_t layer, uint32_t slot, const PrefillFfnBuffers &ffn,
                   metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows, const NormWeights *norm) {
  if (norm && !fusesNorm()) throw std::logic_error("ANE FFN split with a GPU part fuses no norm");
  startIn(graph, layer, slot, ffn, std::move(residual), std::move(output), rows, Parts::Both, norm);
}

void AneFfn::stage(metal::CommandGraph &graph, uint32_t layer) {
  if (staged_ && *staged_ >= layer) return;
  if (layer >= layers_.size()) throw std::logic_error("ANE FFN split stages a layer past its last");
  addWeights(graph, layer, layer & 1);
  staged_ = layer;
}

void AneFfn::stageNext(metal::CommandGraph &graph, uint32_t layer) {
  if (layer + 1 < layers_.size()) stage(graph, layer + 1);
}

void AneFfn::startIn(metal::CommandGraph &graph, uint32_t layer, uint32_t slot, const PrefillFfnBuffers &ffn,
                     metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows, Parts parts,
                     const NormWeights *norm) {
  if (rows < kMinimumRows || rows > kMaximumRows)
    throw std::invalid_argument("ANE FFN split does not take a chunk of these rows");
  if (slot >= slots()) throw std::logic_error("ANE FFN split has no slot " + std::to_string(slot));
  if (layer != nextLayer_[slot] || layer >= layers_.size())
    throw std::logic_error("ANE FFN split adds layer " + std::to_string(layer) + ", not its next layer " +
                           std::to_string(nextLayer_[slot]) + " (begin() starts each command at layer 0)");
  if (pending_[slot].active) throw std::logic_error("ANE FFN split starts a slot it has not joined");
  const uint64_t hiddenBytes = uint64_t{rows} * shape_.hidden * sizeof(uint16_t);
  requireBytes(norm ? residual : ffn.normalized, hiddenBytes, "ANE FFN input");
  requireBytes(output, hiddenBytes, "ANE FFN output");
  const Layer &current = layers_[layer];
  const uint32_t set = layer & 1, pieces = (rows + kEvaluationRows - 1) / kEvaluationRows;
  const bool ane = parts == Parts::Both;
  // The layer's weights, unless an earlier chunk or stageNext() staged them.
  stage(graph, layer);
  const Slot &own = memory_.slots[slot];
  const metal::MetalBuffer &input = norm ? residual : ffn.normalized;
  // Each piece's rows, rotated into its own rows of the scratch, packed into
  // its surfaces.
  for (uint32_t index = 0; index < pieces; ++index) {
    const Piece &piece = own.pieces[index];
    const uint32_t first = index * kEvaluationRows, count = std::min(kEvaluationRows, rows - first);
    const metal::MetalBuffer in = hiddenRows(input, first, count), rotated = hiddenRows(memory_.rotated, first, count);
    if (norm)
      graph.add(normKernel("ane_ffn_norm_rotate", *norm, shape_.hidden),
                {in, norm->buffer, memory_.signs, rotated, piece.tokenScale.buffer}, AneFfnRotateParams{shape_.hidden},
                {count, 1, 1}, {ANE_FFN_ROTATE_THREADS, 1, 1});
    else
      graph.add("ane_ffn_rotate", {in, memory_.signs, rotated, piece.tokenScale.buffer},
                AneFfnRotateParams{shape_.hidden}, {count, 1, 1}, {ANE_FFN_ROTATE_THREADS, 1, 1});
    uint32_t channel = 0;
    for (size_t k = 0; k < shape_.inputSegments.size(); ++k) {
      const uint32_t width = shape_.inputSegments[k];
      graph.add("ane_ffn_pack", {rotated, piece.inputs[k].buffer},
                AneFfnPackParams{shape_.hidden, channel, piece.inputs[k].strideBytes},
                {(count + ANE_FFN_TILE - 1) / ANE_FFN_TILE, width / ANE_FFN_TILE, 1},
                {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
      channel += width;
    }
  }
  ane::Handoff &handoff = *handoffs_[slot];
  const uint64_t ready = ane ? handoff.next() : 0;
  if (ane) graph.signal(handoff.event(), ready);
  // Without GPU channels the join adds the ANE's FFN to the residual rows
  // themselves (ane_ffn_join_residual).
  if (shape_.gpu)
    linear_.addSwiGlu(graph, {&*current.gate, &*current.up, &*current.down}, ffn, residual, output, rows);
  // The pieces run in turn, each once the one before has raised the event,
  // so that each job's time is its own (ane::Handoff::finish).
  uint64_t done = ready;
  for (uint32_t index = 0; ane && index < pieces; ++index) {
    const uint64_t wait = done;
    done = handoff.next();
    jobs_.push_back({set, slot, index, index ? 0 : pieces * kEvaluationRows, wait, done});
  }
  pending_[slot] = {true, ane, done, std::move(output), shape_.gpu ? metal::MetalBuffer{} : std::move(residual), rows};
  ++nextLayer_[slot];
}

metal::MetalBuffer AneFfn::hiddenRows(const metal::MetalBuffer &buffer, uint32_t first, uint32_t count) const {
  const uint64_t row = uint64_t{shape_.hidden} * sizeof(uint16_t);
  return backend_.view(buffer, first * row, count * row);
}

void AneFfn::join(metal::CommandGraph &graph, uint32_t slot) {
  if (slot >= slots() || !pending_[slot].active) throw std::logic_error("ANE FFN split joins a slot it did not start");
  const Pending pending = std::exchange(pending_[slot], {});
  if (pending.ane) graph.wait(handoffs_[slot]->event(), pending.done);
  const Slot &own = memory_.slots[slot];
  for (uint32_t index = 0; index * kEvaluationRows < pending.rows; ++index) {
    const Piece &piece = own.pieces[index];
    const uint32_t first = index * kEvaluationRows, count = std::min(kEvaluationRows, pending.rows - first);
    const AneFfnJoinParams params{shape_.hidden, piece.partial.strideBytes / uint32_t{sizeof(_Float16)}, count};
    const std::array grid{(count + ANE_FFN_TILE - 1) / ANE_FFN_TILE, shape_.hidden / ANE_FFN_TILE, 1u};
    const metal::MetalBuffer output = hiddenRows(pending.output, first, count);
    if (pending.residual)
      graph.add("ane_ffn_join_residual",
                {output, piece.partial.buffer, piece.tokenScale.buffer, memory_.status,
                 hiddenRows(pending.residual, first, count)},
                params, {grid[0], grid[1], grid[2]}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
    else
      graph.add("ane_ffn_join", {output, piece.partial.buffer, piece.tokenScale.buffer, memory_.status}, params,
                {grid[0], grid[1], grid[2]}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  }
}

metal::CommandTicket AneFfn::commit(const metal::CommandGraph &graph, metal::CommandCompletion completion) {
  if (std::ranges::any_of(pending_, &Pending::active))
    throw std::logic_error("ANE FFN split commits a command with a slot it has not joined");
  const std::vector<Job> jobs = std::exchange(jobs_, {});
  unfinished_ = !jobs.empty();
  committedEvaluations_ = 0;
  committedRows_ = 0;
  committedSlots_ = false;
  for (const Job &job : jobs) {
    committedEvaluations_ += job.rows != 0;
    committedRows_ = std::max(committedRows_, job.rows);
    committedSlots_ |= job.slot != 0;
  }
  if (!jobs.empty()) {
    {
      std::lock_guard lock(feeder_->mutex);
      feeder_->jobs.insert(feeder_->jobs.end(), jobs.begin(), jobs.end());
    }
    feeder_->changed.notify_all();
  }
  const auto cancel = [&](const std::string &reason) {
    if (!std::exchange(unfinished_, false)) return;
    cancelAll(reason);
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
  if (std::string losing = breaker_.add(committedRows_, committedEvaluations_, milliseconds, committedSlots_);
      !losing.empty()) {
    retireAll(losing);
    warnStopped();
    if (lost_) std::exchange(lost_, {})();
  }
  return true;
}

std::optional<AwakeClock::duration> AneFfn::completed() {
  unfinished_ = false;
  // The ANE runs one evaluation at a time: the slots' jobs queue behind each
  // other's, so its time is what their spans cover, once.
  std::optional<std::vector<ane::Handoff::Span>> spans = std::vector<ane::Handoff::Span>{};
  for (const auto &handoff : handoffs_) {
    std::optional<std::vector<ane::Handoff::Span>> slot = handoff->finishSpans();
    if (!slot) spans.reset();
    else if (spans) spans->insert(spans->end(), slot->begin(), slot->end());
  }
  std::optional<AwakeClock::duration> ran;
  if (spans) ran = ane::Handoff::covered(std::move(*spans));
  if (std::exchange(*static_cast<uint32_t *>(memory_.status.contents()), 0u) && ran) {
    retireAll("the Neural Engine's output or its scales were not finite");
    ran.reset();
  }
  if (!ran) retireAll(reason().empty() ? "an evaluation failed" : reason());
  return ran;
}

bool AneFfn::release() {
  if (unfinished_ || !jobs_.empty())
    throw std::logic_error("ANE FFN split released while a command it encoded is unfinished");
  const bool idle = std::ranges::all_of(handoffs_, [](const auto &handoff) { return handoff->idle(); });
  if (released_ || (retired() && !idle)) return false;
  try {
    program_->unload({kReloadLimit, {}});
  } catch (const std::exception &error) {
    retireAll(std::string("its program did not unload: ") + error.what());
    warnStopped();
    return false;
  }
  released_ = true;
  logLine("Neural Engine FFN program unloaded while idle");
  return true;
}

void AneFfn::restore() noexcept {
  // A stopped split keeps its program unloaded.
  if (!released_ || retired()) return;
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
  try {
    retireAll("its program did not load again: " + failure);
  } catch (...) {
  }
  warnStopped();
}

void AneFfn::warnStopped() {
  if (std::exchange(warned_, true)) return;
  logWarning("Neural Engine FFN split stopped (", reason(),
             "); the GPU runs the prefill FFN alone until the engine restarts.");
}

void AneFfn::queue(const Job &job, std::function<void()> reported) {
  const auto once = std::make_shared<std::atomic<bool>>(false);
  const auto report = [once, reported = std::move(reported)] {
    if (!once->exchange(true)) reported();
  };
  const ane::Program::Binding &binding = bindings_.at((job.set * slots() + job.slot) * kPieces + job.piece);
  bool enqueued = false;
  handoffs_.at(job.slot)->queue(
      job.ready, job.done,
      [&](const metal::SharedEvent &event, uint64_t wait, uint64_t signal, ane::Handoff::Report handoff) {
        program_->enqueue(binding, event, wait, signal, [handoff = std::move(handoff), report](bool success) {
          handoff(success);
          report();
        });
        enqueued = true;
      });
  if (!enqueued) report();
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
  if (retired()) throw std::runtime_error("ANE FFN split stopped (" + reason() + ")");
  fillNormalized(ffn.normalized);
  // Layer l adds its FFN of the normalized rows to hidden[l & 1] into
  // hidden[(l & 1) ^ 1], from zeros, on the GPU alone, as the split's GPU part
  // alone over a partial output of zeros, or split.
  enum class Way : uint8_t { Gpu, GpuPart, Split };
  const auto forward = [&](Way way, uint32_t count, uint32_t rows) {
    std::memset(hidden[0].contents(), 0, hidden[0].sizeBytes());
    if (way != Way::Gpu) {
      for (const Piece &piece : memory_.slots.front().pieces)
        std::fill_n(static_cast<uint16_t *>(piece.partial.buffer.contents()),
                    piece.partial.buffer.sizeBytes() / sizeof(uint16_t), way == Way::Split ? kHalfNaN : 0);
      static_cast<void>(begin());
    }
    metal::CommandGraph graph;
    for (uint32_t layer = 0; layer < count; ++layer) {
      const metal::MetalBuffer &residual = hidden[layer & 1], &output = hidden[(layer & 1) ^ 1];
      if (way == Way::Gpu)
        linear_.addSwiGlu(graph, layers[layer], ffn, residual, output, rows);
      else
        encode(graph, layer, ffn, residual, output, rows, way == Way::Split ? Parts::Both : Parts::Gpu);
    }
    if (way == Way::Split) {
      static_cast<void>(commit(graph, {}).wait());
      if (!completed())
        throw ane::Refused("ANE FFN split of " + std::to_string(rows) + " rows failed (" + reason() + ")");
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
  for (const uint32_t rows : verifiedRows()) {
    const std::vector<float> split = forward(Way::Split, count, rows);
    const auto fail = [&](const std::string &why) {
      retireAll("its output of " + std::to_string(rows) + " rows " + why);
      throw ane::Refused("ANE FFN split's output of " + std::to_string(rows) + " rows " + why);
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
