#include "ops/AneFfn.hpp"

#include "Checked.hpp"
#include "StderrLine.hpp"
#include "Supervision.hpp"
#include "metal/abi/AneFfn.h"
#include "metal/abi/Gguf.h"
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
// inputs, in segments of kSegment channels and a last of the rest, a matmul
// each: on an M5 Pro, two of 2560 run a 5120-channel FFN about 14% faster
// than one of 5120 (ane-ffn's FFN, 2048 rows). The weights of a segment are
// staged in whole blocks of either rotation, and its inputs packed in whole
// tiles.
constexpr uint32_t kSegment = 2560;
static_assert(kSegment % ANE_FFN_INPUT_BLOCK == 0 && kSegment % ANE_FFN_INTERMEDIATE_BLOCK == 0 &&
                  kSegment % ANE_FFN_TILE == 0,
              "segments hold whole rotation blocks and packing tiles");
// Where the split runs gate and up in blocks (gateUpInBlocks), their matmuls
// each compute at most kOutputBlock of the ANE's channels, and each block
// runs through its scales, SiLU and product before the next. On an M6, with
// Qwen3.8-27B's FFN shapes on synthetic weights, the split layer of 2048 rows
// took 7.6% less than with one matmul of all the ANE's channels a segment at
// 26 of 34 units (26.4 against 28.6 ms) and 2.5% less at 10. Down in blocks
// of 1024 hidden channels took no less; each matmul cut into blocks joined at
// once, 5% more.
constexpr uint32_t kOutputBlock = 1024;
// Whether gate and up run in blocks on the GPU of `appleGpuFamily`'s Neural
// Engine: Apple11's and later's. On an M6, Qwen3.8-27B UD-IQ3_XXS, the
// blocks took requests' first tokens 10-12% sooner from 1,024 rows and 2-4%
// below; on two M5 Pros they took the ANE alone 1-7% longer at every
// function of the program (512-1,024 rows), which moved one step's least rows
// from 535 to 648 on the 20-core and its 640-row prompts 10% past 1.7.0's;
// on an M4 (Apple9) they took 2.5% less, for 46% more compile on a Mac whose
// first start takes two minutes. Every family compiles 30-45% longer with
// them.
constexpr bool gateUpInBlocks(uint32_t appleGpuFamily) noexcept { return appleGpuFamily >= 11; }

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
// How long the service may take to compile and load the program: four times
// the longest cold compile measured, the M6's of the program of 13 functions
// at 26 units (52 s), so that a busy Mac still finishes and only a service
// that stopped answering runs out of it.
constexpr auto kProgramLimit = std::chrono::seconds(210);
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

// The rank of the layers' LoRA updates (ProjectionLora) where every gate, up
// and down has one of that rank on its one tensor, and 0 where none has one;
// none where only some have, or they differ.
std::optional<uint32_t> loraRank(std::span<const SwiGluProjections> layers) {
  uint32_t rank = 0;
  bool every = true;
  for (const SwiGluProjections &layer : layers)
    for (const Projection *projection : {layer.gate, layer.up, layer.down}) {
      const ProjectionLora &update = projection->lora;
      if (!update) {
        every = false;
        continue;
      }
      if (update.slots != 1 || update.segments.size() != 1 || !update.segments.front().b ||
          (rank && update.rank != rank))
        return std::nullopt;
      rank = update.rank;
    }
  if (rank && !every) return std::nullopt;
  return rank;
}

} // namespace

const char *AneFfn::unsupported(std::span<const SwiGluProjections> layers) {
  if (layers.empty()) return "ANE FFN split has no layers";
  if (!loraRank(layers))
    return "ANE FFN split takes LoRA updates of one rank on the gate, up and down of every layer, or on none";
  if (!fits(static_cast<uint32_t>(layers.size()), Steps::One))
    return "ANE FFN split has more layers than the Neural Engine holds evaluations of in flight";
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
  // The rank of the layers' LoRA updates: with them the program does more on
  // the Neural Engine and the GPU's part more on the GPU.
  if (const uint32_t rank = *loraRank(layers)) key += " lora " + std::to_string(rank);
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

// The name of the program's function of `rows` rows over `channels` of the
// ANE's.
std::string functionName(uint32_t channels, uint32_t rows) {
  return "ffn" + std::to_string(channels) + "_" + std::to_string(rows);
}

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

// A function every kProgramStep rows from the least rows to `most`.
std::vector<uint32_t> functionRows(uint32_t most) {
  std::vector<uint32_t> rows;
  for (uint32_t row = AneFfn::kMinimumRows; row <= most; row += AneFfn::kProgramStep) rows.push_back(row);
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
  shape.inputSegments = segments(shape.hidden);
  shape.downSegments = segments(shape.ane);
  shape.rank = *loraRank(layers);
  return shape;
}

uint64_t AneFfn::loraLeadingBytes(const Shape &shape) {
  return uint64_t{shape.layers} * shape.rank * shape.gpu * sizeof(uint16_t);
}
uint64_t AneFfn::loraCodeBytes(const Shape &shape) { return uint64_t{shape.layers} * shape.rank * shape.ane; }
uint64_t AneFfn::loraBytes(const Shape &shape) {
  return loraLeadingBytes(shape) + loraCodeBytes(shape) + uint64_t{shape.layers} * shape.rank * sizeof(uint16_t);
}

uint64_t AneFfn::rowScaleBytes(const Shape &shape) {
  return uint64_t{shape.layers} * (2 * shape.ane + shape.hidden) * sizeof(_Float16);
}

template <class MakeBuffer, class MakeSurface>
AneFfn::Memory AneFfn::allocate(const Shape &shape, MakeBuffer &&buffer, MakeSurface &&surface) {
  Memory memory;
  memory.signs = buffer(ANE_FFN_INTERMEDIATE_BLOCK * sizeof(float), "ane ffn signs");
  memory.rotated = buffer(uint64_t{kMaximumRows} * shape.hidden * sizeof(_Float16), "ane ffn rotated input");
  memory.status = buffer(sizeof(uint32_t), "ane ffn status");
  for (const uint32_t width : shape.inputSegments) memory.inputs.push_back(surface(width, kMaximumRows, Element::Int8));
  memory.tokenScale = surface(1, kMaximumRows, Element::Float16);
  // LoRA: down's rotated A as rows of down's weights after its own, before
  // the intermediate scale.
  memory.partial = surface(shape.hidden + shape.rank + 1, kMaximumRows, Element::Float16);
  if (shape.rank) {
    memory.loraDownGpu = buffer(uint64_t{kMaximumRows} * shape.rank * sizeof(float), "ane ffn lora down t");
    memory.loraT = buffer(lora::rowsBytes(kMaximumRows, 2 * shape.rank), "ane ffn lora gate/up t");
    memory.loraDown = buffer(uint64_t{kMaximumRows} * shape.rank * sizeof(uint16_t), "ane ffn lora down rows");
    memory.loraDelta = buffer(uint64_t{kMaximumRows} * shape.hidden * sizeof(uint16_t), "ane ffn lora delta");
    memory.loraGate = surface(shape.rank, kMaximumRows, Element::Float16);
    memory.loraUp = surface(shape.rank, kMaximumRows, Element::Float16);
  }
  for (Weights &set : memory.sets) {
    for (const uint32_t width : shape.inputSegments) {
      set.gate.push_back(surface(shape.ane, width, Element::Int8));
      set.up.push_back(surface(shape.ane, width, Element::Int8));
    }
    for (const uint32_t width : shape.downSegments)
      set.down.push_back(surface(shape.hidden + shape.rank, width, Element::Int8));
    set.gateScale = surface(shape.ane, 1, Element::Float16);
    set.upScale = surface(shape.ane, 1, Element::Float16);
    set.downScale = surface(shape.hidden + shape.rank, 1, Element::Float16);
    if (shape.rank) {
      set.loraGate = surface(shape.ane, shape.rank, Element::Float16);
      set.loraUp = surface(shape.ane, shape.rank, Element::Float16);
    }
  }
  return memory;
}

uint64_t AneFfn::plannedBytes(std::span<const SwiGluProjections> layers, Shares shares) {
  uint64_t bytes = 0;
  for (const uint32_t units : {shares.oneStep, shares.twoSteps})
    if (units) {
      const Shape shape = shapeOf(layers, units);
      bytes += alignUp(rowScaleBytes(shape));
      if (shape.rank) bytes += alignUp(loraBytes(shape));
    }
  static_cast<void>(allocate(
      shapeOf(layers, shares.most()),
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
    inputs.push_back({"x" + std::to_string(k), {&memory.inputs[k], &memory.inputs[k]}, shape.inputSegments[k], 0});
  inputs.push_back({"tx", {&memory.tokenScale, &memory.tokenScale}, 1, 0});
  for (uint32_t k = 0; k < memory.inputs.size(); ++k) {
    const uint32_t width = shape.inputSegments[k];
    inputs.push_back({"wg" + std::to_string(k), {&memory.sets[0].gate[k], &memory.sets[1].gate[k]}, shape.ane, width});
    inputs.push_back({"wu" + std::to_string(k), {&memory.sets[0].up[k], &memory.sets[1].up[k]}, shape.ane, width});
  }
  inputs.push_back({"sg", both(&Weights::gateScale), shape.ane, 1});
  inputs.push_back({"su", both(&Weights::upScale), shape.ane, 1});
  for (uint32_t i = 0; i < shape.downSegments.size(); ++i)
    inputs.push_back({"wd" + std::to_string(i), {&memory.sets[0].down[i], &memory.sets[1].down[i]},
                      shape.hidden + shape.rank, shape.downSegments[i]});
  inputs.push_back({"sd", both(&Weights::downScale), shape.hidden + shape.rank, 1});
  // LoRA: xlg and xlu, gate's and up's rows t over their tokens' scales; wlg
  // and wlu, their B's rows over the ANE's channels.
  if (shape.rank) {
    inputs.push_back({"xlg", {&memory.loraGate, &memory.loraGate}, shape.rank, 0});
    inputs.push_back({"xlu", {&memory.loraUp, &memory.loraUp}, shape.rank, 0});
    inputs.push_back({"wlg", both(&Weights::loraGate), shape.ane, shape.rank});
    inputs.push_back({"wlu", both(&Weights::loraUp), shape.ane, shape.rank});
  }
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
                             uint32_t rows, bool blocks) {
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
  // The `count` rows from `begin` of `source`, `width` values each.
  const auto slice = [&](const char *type, const std::string &name, const std::string &source, uint64_t begin,
                         uint64_t count, uint64_t width) {
    line(tensor(type, count, width) + " " + name + " = slice_by_size(x = " + source +
         ", begin = tensor<int32, [4]>([0, 0, " + std::to_string(begin) + ", 0]), size = tensor<int32, [4]>(" +
         dimensions(count, width) + "))");
  };
  // `name`: the `count` output channels from `begin` of a projection of
  // `height`, the sum over its input segments (`matmuls`) of each segment's
  // int8 weight rows times its values, times those rows' `scales`.
  struct Matmul final {
    std::string weights, values;
    uint32_t width;
  };
  const auto project = [&](const std::string &name, std::span<const Matmul> matmuls, const std::string &scales,
                           uint64_t begin, uint64_t count, uint64_t height) {
    const bool whole = count == height;
    std::string total;
    for (size_t term = 0; term < matmuls.size(); ++term) {
      const auto &[weights, values, width] = matmuls[term];
      const std::string product = name + "_m" + std::to_string(term);
      if (!whole) slice("int8", product + "_w", weights + "_t", begin, count, width);
      f16(product + "_wd", count, width,
          "dequantize(input = " + (whole ? weights + "_t" : product + "_w") + ", scale = " + unit + ")");
      f16(product, count, rows,
          "matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + product + "_wd, y = " + values +
              ")");
      if (term) f16(product + "_sum", count, rows, "add(x = " + total + ", y = " + product + ")");
      total = term ? product + "_sum" : product;
    }
    if (!whole) slice("fp16", name + "_scale", scales + "_t", begin, count, 1);
    f16(name, count, rows, "mul(x = " + total + ", y = " + (whole ? scales + "_t" : name + "_scale") + ")");
  };
  // LoRA: `name`, the `count` channels from `begin` of gate's or up's
  // (`p`) scaled sums `sums` with B t over the token's scale, as the sums are
  // before it; the sums themselves under that name without an update.
  const auto updated = [&](const std::string &name, const std::string &p, uint64_t begin, uint64_t count) {
    if (!shape.rank) return name;
    const bool whole = count == channels;
    if (!whole) slice("fp16", name + "_lb", "wl" + p + "_t", begin, count, shape.rank);
    f16(name + "_l", count, rows,
        "matmul(transpose_x = bool(false), transpose_y = bool(false), x = " +
            (whole ? "wl" + p + "_t" : name + "_lb") + ", y = xl" + p + "_t)");
    return name + "_w";
  };
  const auto update = [&](const std::string &name, uint64_t count) {
    if (shape.rank) f16(name, count, rows, "add(x = " + name + "_w, y = " + name + "_l)");
  };
  // The ANE's channels through gate and up, their scales, SiLU and product:
  // where the chip runs them in blocks (gateUpInBlocks), a block at a time;
  // else each projection whole, as 1.7.0's program does.
  if (blocks) {
    std::vector<Matmul> gate, up;
    for (uint32_t k = 0; k < segmentCount; ++k) {
      const std::string index = std::to_string(k), x = "x" + index;
      f16(x + "_d", shape.inputSegments[k], rows, "dequantize(input = " + x + "_t, scale = " + unit + ")");
      gate.push_back({"wg" + index, x + "_d", shape.inputSegments[k]});
      up.push_back({"wu" + index, x + "_d", shape.inputSegments[k]});
    }
    // Each block of the ANE's channels through gate and up, their scales, SiLU
    // and their product, before the next block's; one block is h itself.
    std::string intermediate;
    for (uint64_t begin = 0; begin < channels; begin += kOutputBlock) {
      const uint64_t count = std::min<uint64_t>(kOutputBlock, channels - begin);
      const std::string b = "b" + std::to_string(begin / kOutputBlock) + "_",
                        h = channels > kOutputBlock ? b + "h" : "h";
      for (const char *projection : {"g", "u"}) {
        const std::string p = projection, sums = updated(b + p + "s", p, begin, count);
        project(sums, p == "g" ? gate : up, "s" + p, begin, count, channels);
        update(b + p + "s", count);
      }
      // silu(g) = g/2 (1 + tanh(g/2)): through the ANE's sigmoid, layers whose
      // gate pre-activations are about 0.1 differed from the GPU alone by 6.5%
      // RMS, through its tanh by 1.6%.
      f16(b + "gt", count, rows, "mul(x = " + b + "gs, y = tx_t)");
      f16(b + "gh", count, rows, "mul(x = " + b + "gt, y = " + fp16(0.5) + ")");
      f16(b + "th", count, rows, "tanh(x = " + b + "gh)");
      f16(b + "tp", count, rows, "add(x = " + b + "th, y = " + fp16(1.0) + ")");
      f16(b + "silu", count, rows, "mul(x = " + b + "gh, y = " + b + "tp)");
      f16(h, count, rows, "mul(x = " + b + "silu, y = " + b + "us)");
      intermediate += (intermediate.empty() ? "" : ", ") + h;
    }
    if (channels > kOutputBlock)
      f16("h", channels, rows, "concat(axis = int32(2), interleave = bool(false), values = (" + intermediate + "))");
  } else {
    for (uint32_t k = 0; k < segmentCount; ++k) {
      const std::string x = "x" + std::to_string(k);
      f16(x + "_d", shape.inputSegments[k], rows, "dequantize(input = " + x + "_t, scale = " + unit + ")");
    }
    for (const char *projection : {"g", "u"}) {
      const std::string p = projection;
      for (uint32_t k = 0; k < segmentCount; ++k)
        matmul(p + "m" + std::to_string(k), "w" + p + std::to_string(k), "x" + std::to_string(k) + "_d", channels,
               shape.inputSegments[k]);
      const std::string sums = updated(p + "s", p, 0, channels);
      f16(sums, channels, rows, "mul(x = " + sum(p + "m", segmentCount, channels) + ", y = s" + p + "_t)");
      update(p + "s", channels);
    }
    f16("gt", channels, rows, "mul(x = gs, y = tx_t)");
    f16("gh", channels, rows, "mul(x = gt, y = " + fp16(0.5) + ")");
    f16("th", channels, rows, "tanh(x = gh)");
    f16("tp", channels, rows, "add(x = th, y = " + fp16(1.0) + ")");
    f16("silu", channels, rows, "mul(x = gh, y = tp)");
    f16("h", channels, rows, "mul(x = silu, y = us)");
  }
  const std::string c = std::to_string(channels), r = std::to_string(rows),
                    block = std::to_string(ANE_FFN_INTERMEDIATE_BLOCK);
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
  // Down's rows, then with LoRA updates its rotated A's, then the
  // intermediate scale.
  const uint32_t products = shape.hidden + shape.rank, outputs = products + 1;
  if (blocks) {
    std::vector<Matmul> down;
    uint32_t input = 0;
    for (size_t i = 0; i < shape.downSegments.size(); ++i) {
      const uint32_t width = shape.downSegments[i];
      const std::string index = std::to_string(i);
      slice("fp16", "hd" + index, "hd", input, width, rows);
      down.push_back({"wd" + index, "hd" + index, width});
      input += width;
    }
    project("ds", down, "sd", 0, products, products);
  } else {
    uint32_t begin = 0;
    for (size_t i = 0; i < shape.downSegments.size(); ++i) {
      const uint32_t width = shape.downSegments[i];
      const std::string index = std::to_string(i), part = "hd" + index;
      f16(part, width, rows,
          "slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, " + std::to_string(begin) +
              ", 0]), size = tensor<int32, [4]>(" + dimensions(width, rows) + "))");
      matmul("dm" + index, "wd" + index, part, products, width);
      begin += width;
    }
    f16("ds", products, rows, "mul(x = " + sum("dm", shape.downSegments.size(), products) + ", y = sd_t)");
  }
  f16("yt", outputs, rows, "concat(axis = int32(2), interleave = bool(false), values = (ds, hscale))");
  line(output.bufferType(outputs, rows) +
       " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
       "strides = tensor<int64, [4]>(" + output.strides(outputs) + "))");
  return "    func " + functionName(shape.ane, rows) + "<ios18>(" + parameters + ") {\n" + body + "    } -> (y);\n";
}

AneFfn::AneFfn(metal::MetalBackend &backend, const Linear &linear, std::span<const SwiGluProjections> layers,
               Shares shares, Rows rows, std::function<bool()> interrupted, std::optional<bool> blocks)
    : AneFfn(backend, linear, layers, shares, std::move(interrupted), programRows(shares), blocks) {
  setRows(rows);
}

std::array<std::vector<uint32_t>, 2> AneFfn::programRows(Shares shares) {
  if (!shares.twoSteps) return {shares.oneStep ? functionRows(kMaximumRows) : std::vector<uint32_t>{}, {}};
  return {shares.oneStep ? functionRows(kTwoStepRows) : std::vector<uint32_t>{}, functionRows(kMaximumRows / 2)};
}

void AneFfn::setRows(Rows rows) {
  // The function that holds `count` rows of a kind of command: none holds
  // fewer than the least function's rows.
  const auto holds = [&](Steps steps, uint32_t count) {
    const std::optional<Share> &share = shares_[static_cast<size_t>(steps)];
    return share && !share->evaluations.empty() && count >= kMinimumRows &&
           share->evaluations.front().rows - kProgramStep < count && count <= share->evaluations.back().rows;
  };
  const uint32_t lastOneStep = rows.twoSteps && rows.twoSteps <= kMaximumRows ? rows.twoSteps - 1 : kMaximumRows;
  if ((rows.oneStep && (rows.oneStep > lastOneStep || !holds(Steps::One, rows.oneStep) ||
                        !holds(Steps::One, lastOneStep))) ||
      (rows.twoSteps && (rows.twoSteps < 2 * kMinimumRows || rows.twoSteps > kMaximumRows ||
                         !holds(Steps::Two, rows.twoSteps / 2) || !holds(Steps::Two, kMaximumRows / 2))) ||
      (!rows.oneStep && !rows.twoSteps))
    throw std::invalid_argument("ANE FFN split has no function of its rows");
  rows_ = rows;
}

AneFfn::AneFfn(metal::MetalBackend &backend, const Linear &linear, std::span<const SwiGluProjections> layers,
               Shares shares, std::function<bool()> interrupted,
               const std::array<std::vector<uint32_t>, 2> &functionRows, std::optional<bool> blocks)
    : backend_(backend), linear_(linear), shape_(shapeOf(layers, shares.most())),
      rows_{shares.oneStep ? kMinimumRows : 0, shares.twoSteps ? 2 * kMinimumRows : 0},
      steps_(shares.oneStep ? Steps::One : Steps::Two), handoff_(backend, kHandoffBound) {
  if (shares.twoSteps && !fits(static_cast<uint32_t>(layers.size()), Steps::Two))
    throw std::invalid_argument("ANE FFN split of " + std::to_string(layers.size()) +
                                " layers queues more evaluations in two steps than the Neural Engine holds in flight");
  // What Metal allocates, which the memory audit compares with the plan.
  const auto buffer = [&](uint64_t bytes, const char *label) {
    metal::MetalBuffer made = backend_.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
    allocatedBytes_ += made.allocatedBytes();
    return made;
  };
  memory_ = allocate(shape_, buffer, [&](uint32_t rows, uint32_t width, Element element) {
    ane::Surface surface = ane::Surface::create(backend_, rows, width, element);
    allocatedBytes_ += surface.buffer.allocatedBytes();
    return surface;
  });
  const std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> signs = rotationSigns();
  std::memcpy(memory_.signs.contents(), signs.data(), sizeof signs);
  *static_cast<uint32_t *>(memory_.status.contents()) = 0;

  for (const SwiGluProjections &source : layers) {
    planes_.push_back({Planes(*source.gate), Planes(*source.up), Planes(*source.down)});
    if (shape_.rank)
      loraB_.push_back({source.gate->lora.segments.front().b, source.up->lora.segments.front().b,
                        source.down->lora.segments.front().b});
  }
  // Each share's GPU parts and program text, a function of each of its rows.
  std::string text = "program(1.3)\n{\n";
  std::array<std::vector<Input>, 2> inputs;
  for (const Steps steps : {Steps::One, Steps::Two}) {
    const auto kind = static_cast<size_t>(steps);
    const uint32_t units = steps == Steps::One ? shares.oneStep : shares.twoSteps;
    if (!units != functionRows[kind].empty())
      throw std::invalid_argument("ANE FFN split has a kind of command without its share or its functions");
    if (!units) continue;
    Share &share = shares_[kind].emplace();
    share.shape = shapeOf(layers, units);
    share.rowScales = buffer(rowScaleBytes(share.shape), "ane ffn row scales");
    if (shape_.rank) share.lora = buffer(loraBytes(share.shape), "ane ffn lora down A");
    for (const SwiGluProjections &source : layers)
      share.layers.push_back({source.gate->rows(backend_, {{0, share.shape.gpu}}),
                              source.up->rows(backend_, {{0, share.shape.gpu}}),
                              source.down->leadingInputs(share.shape.gpu)});
    inputs[kind] = programInputs(share.shape, memory_);
    // Shares of the same channels share the functions of the same rows.
    for (const uint32_t rows : functionRows[kind])
      if (text.find(" " + functionName(share.shape.ane, rows) + "<") == std::string::npos)
        text += function(share.shape, inputs[kind], memory_.partial, rows,
                         blocks.value_or(gateUpInBlocks(backend.capabilities().appleGpuFamily)));
  }
  program_ = std::make_unique<ane::Program>(text + "}\n", rotationBlob(shape_.ane, signs),
                                            ane::Program::Limits{kProgramLimit, std::move(interrupted)});
  for (size_t kind = 0; kind < shares_.size(); ++kind) {
    if (!shares_[kind]) continue;
    for (const uint32_t rows : functionRows[kind]) {
      Evaluation &evaluation = shares_[kind]->evaluations.emplace_back();
      evaluation.rows = rows;
      const uint32_t procedure = program_->procedure(functionName(shares_[kind]->shape.ane, rows));
      for (uint32_t set = 0; set < 2; ++set) {
        std::vector<ane::Surface> surfaces;
        for (const std::string &name : program_->inputs(procedure)) {
          const auto input = std::ranges::find(inputs[kind], name, &Input::name);
          if (input == inputs[kind].end()) throw std::logic_error("ANE FFN program has an unknown input " + name);
          surfaces.push_back(*input->surfaces[set]);
        }
        evaluation.bindings.push_back(program_->bind(procedure, surfaces, memory_.partial));
      }
    }
  }

  // Each share's rows' shared int8 scales over the ANE's share of their inputs.
  metal::CommandGraph graph;
  for (const std::optional<Share> &share : shares_)
    for (uint32_t layer = 0; share && layer < planes_.size(); ++layer) {
      const Shape &channels = share->shape;
      const auto add = [&](Matrix matrix, uint32_t row, uint32_t input, uint32_t width, uint32_t rows,
                           Rotation rotation) {
        const Planes &planes = planes_[layer][static_cast<size_t>(matrix)];
        graph.add(kernel("ane_ffn_row_scale_gguf", rotation),
                  {planes.buffers[0], planes.buffers[1], planes.buffers[2], rowScales(*share, layer, matrix),
                   memory_.signs},
                  AneFfnWeightParams{planes.groups, row, input, width, 0, 0, planes.format, rows},
                  {rows / ANE_FFN_WEIGHT_ROWS, 1, 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
      };
      add(Matrix::Gate, channels.gpu, 0, channels.hidden, channels.ane, Rotation::Inputs);
      add(Matrix::Up, channels.gpu, 0, channels.hidden, channels.ane, Rotation::Inputs);
      add(Matrix::Down, 0, channels.gpu, channels.ane, channels.hidden, Rotation::Intermediate);
    }
  static_cast<void>(backend_.submitCommandAsync(graph.dispatches()).wait());
  if (shape_.rank) prepareLora(layers);
}

// Down's A as each share takes it (metal/abi/AneFfn.h, Share::lora): over the
// GPU's channels as it is, and over the ANE's, in the surfaces' order
// (ane_ffn_descending), rotated as down's rows are
// (ane_ffn_weights_gguf_intermediate), each block of
// ANE_FFN_INTERMEDIATE_BLOCK inputs through H times the signs over
// sqrt(block), scaled by a power of two that puts the layer's largest value
// in [0.5, 1), so that every row's scale is a normal half, and coded against
// its row's peak as a weight row is. Each share's scales are over its own
// channels, so that it computes what a split of that share alone computes.
void AneFfn::prepareLora(std::span<const SwiGluProjections> layers) {
  const uint32_t rank = shape_.rank, block = ANE_FFN_INTERMEDIATE_BLOCK;
  const std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> signs = rotationSigns();
  const float norm = 1.0f / std::sqrt(float(block));
  const auto value = [](uint16_t bits) { return std::bit_cast<float>(uint32_t{bits} << 16); };
  const auto bf16 = [](float v) {
    const uint32_t bits = std::bit_cast<uint32_t>(v);
    return static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
  };
  const auto code = [](float v) { return static_cast<int8_t>(std::clamp(std::rint(v), -127.0f, 127.0f)); };
  // A layer's rotated rows over the ANE's most channels, of which each share
  // takes the leading ones.
  std::vector<float> values(uint64_t{shape_.ane} * rank);
  for (uint32_t layer = 0; layer < shape_.layers; ++layer) {
    const QuantizedSegment &downA = layers[layer].down->lora.a;
    if (downA.inputSize != shape_.intermediate || layers[layer].gate->lora.a.inputSize != shape_.hidden ||
        layers[layer].up->lora.a.inputSize != shape_.hidden)
      throw std::invalid_argument("ANE FFN split does not take these LoRA updates");
    const auto *a = static_cast<const uint16_t *>(downA.plane0.contents());
    if (!a) throw std::invalid_argument("ANE FFN split reads LoRA rows the CPU cannot");
    for (uint32_t j = 0; j < rank; ++j) {
      const uint16_t *row = a + uint64_t{j} * shape_.intermediate;
      float *out = values.data() + uint64_t{j} * shape_.ane;
      for (uint32_t origin = 0; origin < shape_.ane; origin += block) {
        // The surfaces' unit at `origin`: the channels below those before it.
        const uint16_t *unit = row + shape_.intermediate - origin - block;
        float *v = out + origin;
        for (uint32_t i = 0; i < block; ++i) v[i] = value(unit[i]);
        for (uint32_t span = 1; span < block; span <<= 1)
          for (uint32_t i = 0; i < block; i += 2 * span)
            for (uint32_t k = i; k < i + span; ++k) {
              const float x = v[k], y = v[k + span];
              v[k] = x + y;
              v[k + span] = x - y;
            }
        for (uint32_t i = 0; i < block; ++i) v[i] *= signs[i] * norm;
      }
    }
    for (std::optional<Share> &share : shares_) {
      if (!share) continue;
      const uint32_t gpu = share->shape.gpu, ane = share->shape.ane;
      const uint64_t leadingBytes = uint64_t{rank} * gpu * sizeof(uint16_t);
      auto *base = static_cast<uint8_t *>(share->lora.contents());
      auto *leading = reinterpret_cast<uint16_t *>(base) + uint64_t{layer} * rank * gpu;
      auto *codes = reinterpret_cast<int8_t *>(base + loraLeadingBytes(share->shape)) + uint64_t{layer} * rank * ane;
      auto *rowScaleBits =
          reinterpret_cast<uint16_t *>(base + loraLeadingBytes(share->shape) + loraCodeBytes(share->shape)) +
          uint64_t{layer} * rank;
      float peak = 0.0f;
      for (uint32_t j = 0; j < rank; ++j) {
        std::memcpy(leading + uint64_t{j} * gpu, a + uint64_t{j} * shape_.intermediate,
                    uint64_t{gpu} * sizeof(uint16_t));
        const float *row = values.data() + uint64_t{j} * shape_.ane;
        for (uint32_t i = 0; i < ane; ++i) peak = std::max(peak, std::fabs(row[i]));
      }
      const int exponent = peak > 0.0f ? -static_cast<int>(std::floor(std::log2(peak))) - 1 : 0;
      const float scale = std::ldexp(1.0f, exponent);
      for (uint32_t j = 0; j < rank; ++j) {
        const float *row = values.data() + uint64_t{j} * shape_.ane;
        float rowPeak = 0.0f;
        for (uint32_t i = 0; i < ane; ++i) rowPeak = std::max(rowPeak, std::fabs(row[i]) * scale);
        // As ane_ffn_row_scale_gguf's, then the bf16 the staging kernel reads.
        const uint16_t bits = bf16(std::max(rowPeak, ANE_FFN_PEAK_FLOOR) / ANE_FFN_INT8_PEAK * ANE_FFN_INT8_UNIT);
        rowScaleBits[j] = bits;
        const float inverse = ANE_FFN_INT8_UNIT / value(bits);
        for (uint32_t i = 0; i < ane; ++i) codes[uint64_t{j} * ane + i] = code(row[i] * scale * inverse);
      }
      share->loraDownScale.push_back(std::ldexp(1.0f, -exponent));
      share->loraLeadingRows.push_back(QuantizedSegment::rows(
          GGUF_FMT_BF16, rank, gpu, backend_.view(share->lora, layer * leadingBytes, leadingBytes)));
    }
  }
  // The inputs t hold zeros until a step writes them.
  for (const ane::Surface *input : {&memory_.loraGate, &memory_.loraUp})
    std::memset(input->buffer.contents(), 0, input->buffer.sizeBytes());
}

double AneFfn::share(Steps steps) const noexcept {
  const std::optional<Share> &share = shares_[static_cast<size_t>(steps)];
  return share ? static_cast<double>(share->shape.ane) / share->shape.intermediate : 0.0;
}

uint32_t AneFfn::minimumRows() const noexcept {
  return rows_.oneStep && (!rows_.twoSteps || rows_.oneStep < rows_.twoSteps) ? rows_.oneStep : rows_.twoSteps;
}

bool AneFfn::splits(uint32_t rows) const {
  return rows >= minimumRows() && rows <= kMaximumRows && !handoff_.retired() && !pause_.paused();
}

AneFfn::Steps AneFfn::steps(uint32_t rows) const noexcept {
  return rows_.twoSteps && rows >= rows_.twoSteps ? Steps::Two : Steps::One;
}

// Halves, the first the larger: a command of 2,048 rows runs two of 1,024.
uint32_t AneFfn::firstStepRows(uint32_t rows) const {
  return splits(rows) && steps(rows) == Steps::Two ? (rows + 1) / 2 : rows;
}

const AneFfn::Share &AneFfn::share() const {
  const std::optional<Share> &share = shares_[static_cast<size_t>(steps_)];
  if (!share) throw std::logic_error("ANE FFN split takes no command of this kind");
  return *share;
}

metal::MetalBuffer AneFfn::rowScales(const Share &share, uint32_t layer, Matrix matrix) const {
  const Shape &channels = share.shape;
  const uint64_t perLayer = 2 * channels.ane + channels.hidden, part = static_cast<uint64_t>(matrix);
  return backend_.view(share.rowScales, (layer * perLayer + part * channels.ane) * sizeof(_Float16),
                       uint64_t{matrix == Matrix::Down ? channels.hidden : channels.ane} * sizeof(_Float16));
}

// Layer `layer`'s int8 weights and row scales of the command's share into
// staging set `set`: the leading rows of gate's and up's surfaces and the
// leading columns of down's, each 512-channel unit of the ANE's in
// descending order (ane_ffn_descending).
void AneFfn::addWeights(metal::CommandGraph &graph, uint32_t layer, uint32_t set) const {
  const std::array<Planes, 3> &source = planes_.at(layer);
  const Share &current = share();
  const Shape &channels = current.shape;
  const Weights &target = memory_.sets[set];
  const auto add = [&](Matrix matrix, const ane::Surface &output, const ane::Surface &scale, uint32_t row,
                       uint32_t input, uint32_t width, uint32_t rows, Rotation rotation) {
    const Planes &planes = source[static_cast<size_t>(matrix)];
    const uint32_t scaleStride = scale.strideBytes / uint32_t{sizeof(_Float16)};
    requireBytes(output.buffer, rowBytes(rows, output.strideBytes, width, 1), "ANE FFN weight surface");
    requireBytes(scale.buffer, rowBytes(rows, scaleStride, 1, sizeof(_Float16)), "ANE FFN weight scale surface");
    graph.add(kernel("ane_ffn_weights_gguf", rotation),
              {planes.buffers[0], planes.buffers[1], planes.buffers[2], rowScales(current, layer, matrix),
               output.buffer, scale.buffer, memory_.signs},
              AneFfnWeightParams{planes.groups, row, input, width, output.strideBytes, scaleStride, planes.format,
                                 rows},
              {rows / ANE_FFN_WEIGHT_ROWS, width / blockOf(rotation), 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
  };
  uint32_t input = 0;
  for (size_t k = 0; k < channels.inputSegments.size(); ++k) {
    const uint32_t width = channels.inputSegments[k];
    add(Matrix::Gate, target.gate[k], target.gateScale, channels.gpu, input, width, channels.ane, Rotation::Inputs);
    add(Matrix::Up, target.up[k], target.upScale, channels.gpu, input, width, channels.ane, Rotation::Inputs);
    input += width;
  }
  // Segment i's columns are the channels below the segments before it.
  uint32_t end = channels.intermediate;
  for (size_t i = 0; i < channels.downSegments.size(); ++i) {
    const uint32_t width = channels.downSegments[i];
    add(Matrix::Down, target.down[i], target.downScale, 0, end - width, width, channels.hidden,
        Rotation::Intermediate);
    end -= width;
  }
  if (!channels.rank) return;
  // LoRA: gate's and up's rows of B over the share's channels in fp16, in the
  // surfaces' order, and down's codes of its rotated A as the rows after its
  // own, with their scales.
  const uint32_t rank = channels.rank;
  const uint64_t matrix = uint64_t{channels.ane} * rank, codes = loraLeadingBytes(channels),
                 scales = codes + loraCodeBytes(channels);
  // `count` bf16 rows of `width` values into the surface's rows from `row`,
  // in fp16 times `scale`; with a span, the rows of the ANE's channels of a
  // projection of that many (AneFfnLoraRowsParams).
  const auto values = [&](metal::MetalBuffer rows, const ane::Surface &output, uint32_t count, uint32_t width,
                          uint32_t row, float scale, uint32_t span) {
    requireBytes(rows, uint64_t{span ? span : count} * width * sizeof(uint16_t), "ANE FFN LoRA rows");
    requireBytes(output.buffer, rowBytes(row + count, output.strideBytes, uint64_t{width} * sizeof(_Float16), 1),
                 "ANE FFN LoRA surface");
    graph.add("ane_ffn_lora_rows", {std::move(rows), output.buffer},
              AneFfnLoraRowsParams{count, width, width, 0, output.strideBytes, row, 0, scale, span},
              {(width + 31) / 32, (count + 7) / 8, 1}, {32, 8, 1});
  };
  const std::array<metal::MetalBuffer, 3> &b = loraB_.at(layer);
  values(b[0], target.loraGate, channels.ane, rank, 0, 1.0f / ANE_FFN_LORA_UNIT, channels.intermediate);
  values(b[1], target.loraUp, channels.ane, rank, 0, 1.0f / ANE_FFN_LORA_UNIT, channels.intermediate);
  uint32_t offset = 0;
  for (size_t i = 0; i < channels.downSegments.size(); ++i) {
    const ane::Surface &output = target.down[i];
    const uint32_t width = channels.downSegments[i];
    requireBytes(output.buffer, rowBytes(channels.hidden + rank, output.strideBytes, width, 1),
                 "ANE FFN LoRA surface");
    graph.add("ane_ffn_lora_codes", {backend_.view(current.lora, codes + layer * matrix, matrix), output.buffer},
              AneFfnLoraRowsParams{rank, width, channels.ane, offset, output.strideBytes, channels.hidden, 0, 1.0f, 0},
              {(width + 31) / 32, (rank + 7) / 8, 1}, {32, 8, 1});
    offset += width;
  }
  values(backend_.view(current.lora, scales + uint64_t{layer} * rank * sizeof(uint16_t), rank * sizeof(uint16_t)),
         target.downScale, rank, 1, channels.hidden, 1.0f, 0);
}

bool AneFfn::begin(Steps steps) {
  steps_ = steps;
  // A command committed but never finished failed on the GPU, and the
  // evaluations it queued with it.
  if (std::exchange(unfinished_, false)) {
    handoff_.cancel("a command it split did not finish");
    warnStopped();
  }
  jobs_.clear();
  layersAdded_ = 0;
  pending_.reset();
  return !handoff_.retired();
}

void AneFfn::add(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn,
                 metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows) {
  encode(graph, layer, ffn, residual, output, rows, Parts::Both);
}

// A step's layer: its rows rotated and packed for the ANE, the event raised
// for its evaluation, the GPU's channels, and with the command's first step
// of the layer the next layer's int8 weights into the other set, which the
// layer before read: every step of that layer joined before this add. Its
// join waits for join().
void AneFfn::encode(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn,
                    metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows, Parts parts) {
  if (rows < kMinimumRows || rows > kMaximumRows)
    throw std::invalid_argument("ANE FFN split does not take a step of these rows");
  if (pending_) throw std::logic_error("ANE FFN split adds a step before it joined the one before");
  const bool next = layer == layersAdded_, again = layersAdded_ && layer + 1 == layersAdded_;
  if ((!next && !again) || layer >= planes_.size())
    throw std::logic_error("ANE FFN split adds layer " + std::to_string(layer) + " after " +
                           std::to_string(layersAdded_) + " layers (begin() starts each command at layer 0)");
  const uint64_t hiddenBytes = uint64_t{rows} * shape_.hidden * sizeof(uint16_t);
  requireBytes(ffn.normalized, hiddenBytes, "ANE FFN input");
  requireBytes(output, hiddenBytes, "ANE FFN output");
  const Share &current = share();
  const uint32_t set = layer & 1, tiles = (rows + ANE_FFN_TILE - 1) / ANE_FFN_TILE;
  const bool ane = parts == Parts::Both;
  const auto found =
      std::ranges::find_if(current.evaluations, [&](const Evaluation &evaluation) { return evaluation.rows >= rows; });
  if (ane && found == current.evaluations.end())
    throw std::invalid_argument("ANE FFN has no function of the step's rows");
  const auto index = static_cast<uint32_t>(found - current.evaluations.begin());
  // Each command stages layer 0's weights, then each layer the next one's.
  if (!layersAdded_) addWeights(graph, 0, 0);
  graph.add("ane_ffn_rotate", {ffn.normalized, memory_.signs, memory_.rotated, memory_.tokenScale.buffer},
            AneFfnRotateParams{shape_.hidden}, {rows, 1, 1}, {ANE_FFN_ROTATE_THREADS, 1, 1});
  uint32_t channel = 0;
  for (size_t k = 0; k < shape_.inputSegments.size(); ++k) {
    const uint32_t width = shape_.inputSegments[k];
    graph.add("ane_ffn_pack", {memory_.rotated, memory_.inputs[k].buffer},
              AneFfnPackParams{shape_.hidden, channel, memory_.inputs[k].strideBytes},
              {tiles, width / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
    channel += width;
  }
  const GpuPart &part = current.layers[layer];
  // LoRA: gate's and up's t = A x of the step, for the ANE over each token's
  // scale; the GPU's part computes its own beside its sums
  // (Linear::addSwiGlu).
  const uint32_t rank = shape_.rank;
  if (rank) {
    const LoraRows down =
        lora::addDown(graph, linear_, ffn.normalized, rows, &part.gate.lora, &part.up.lora, memory_.loraT, false);
    size_t update = 0;
    for (const ane::Surface *input : {&memory_.loraGate, &memory_.loraUp})
      graph.add("ane_ffn_lora_inputs", {down.t, memory_.tokenScale.buffer, input->buffer},
                AneFfnLoraInputParams{rows, rank, input->strideBytes / uint32_t{sizeof(_Float16)}, down.stride,
                                      down.offsets[update++], ANE_FFN_LORA_UNIT},
                {(uint64_t{rows} * rank + 255) / 256, 1, 1}, {256, 1, 1});
  }
  const uint64_t ready = ane ? handoff_.next() : 0;
  if (ane) graph.signal(handoff_.event(), ready);
  // The GPU's channels, gate's and up's with their updates; down's view of
  // them has none, and its t over them joins the ANE's in join().
  linear_.addSwiGlu(graph, {&part.gate, &part.up, &part.down}, ffn, residual, output, rows);
  if (rank)
    addGgufFloat(graph, ffn.intermediate, current.loraLeadingRows.at(layer), memory_.loraDownGpu, rows, rank, 0,
                 FloatOutput::Float32, lora::floatTile(linear_, rows, rank, current.shape.gpu));
  if (next && layer + 1 < planes_.size()) addWeights(graph, layer + 1, set ^ 1);
  const uint64_t done = ane ? handoff_.next() : 0;
  if (ane) jobs_.push_back({index, set, ready, done});
  pending_ = Join{std::move(output), rows, layer, done};
  layersAdded_ = layer + 1;
}

void AneFfn::join(metal::CommandGraph &graph) {
  const std::optional<Join> step = std::exchange(pending_, std::nullopt);
  if (!step) return;
  if (step->done) graph.wait(handoff_.event(), step->done);
  const uint32_t tiles = (step->rows + ANE_FFN_TILE - 1) / ANE_FFN_TILE, rank = shape_.rank, rows = step->rows,
                 partialStride = memory_.partial.strideBytes / uint32_t{sizeof(_Float16)};
  const AneFfnJoinParams params{shape_.hidden, partialStride, rows, shape_.hidden + rank};
  const metal::DispatchSize grid{tiles, shape_.hidden / ANE_FFN_TILE, 1}, threads{ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1};
  if (!rank) {
    graph.add("ane_ffn_join", {step->output, memory_.partial.buffer, memory_.tokenScale.buffer, memory_.status},
              params, grid, threads);
    return;
  }
  // LoRA: down's t of both parts, then B t beside the ANE's part of the sums.
  graph.add("ane_ffn_lora_down",
            {memory_.loraDownGpu, memory_.partial.buffer, memory_.tokenScale.buffer, memory_.loraDown,
             memory_.status},
            AneFfnLoraDownParams{shape_.hidden, partialStride, rows, rank, share().loraDownScale.at(step->layer)},
            {(uint64_t{rows} * rank + 255) / 256, 1, 1}, {256, 1, 1});
  addGgufFloat(graph, memory_.loraDown,
               QuantizedSegment::rows(GGUF_FMT_BF16, shape_.hidden, rank, loraB_.at(step->layer)[2]),
               memory_.loraDelta, rows, shape_.hidden, 0, FloatOutput::BFloat16,
               lora::floatTile(linear_, rows, shape_.hidden, rank));
  graph.add("ane_ffn_join_lora",
            {step->output, memory_.partial.buffer, memory_.tokenScale.buffer, memory_.status, memory_.loraDelta},
            params, grid, threads);
}

metal::CommandTicket AneFfn::commit(const metal::CommandGraph &graph, metal::CommandCompletion completion) {
  if (pending_) throw std::logic_error("ANE FFN split commits a step it did not join");
  if (jobs_.size() > kMostInFlight)
    throw std::logic_error("ANE FFN command queues " + std::to_string(jobs_.size()) +
                           " evaluations, more than the Neural Engine holds in flight");
  const std::vector<Job> jobs = std::exchange(jobs_, {});
  unfinished_ = !jobs.empty();
  committedLayers_ = jobs.empty() ? 0 : layersAdded_;
  committedEvaluations_ = static_cast<uint32_t>(jobs.size());
  committedRows_ = 0;
  for (const Job &job : jobs) committedRows_ += share().evaluations[job.evaluation].rows;
  if (committedLayers_) committedRows_ /= committedLayers_;
  const auto cancel = [&](const std::string &reason) {
    if (!std::exchange(unfinished_, false)) return;
    handoff_.cancel(reason);
    warnStopped();
  };
  metal::CommandTicket ticket;
  try {
    ticket = backend_.submitCommandAsync(graph.command(), std::move(completion));
  } catch (const std::exception &error) {
    cancel(std::string("its Metal command was not submitted: ") + error.what());
    throw;
  } catch (...) {
    cancel("its Metal command was not submitted");
    throw;
  }
  // Queued once the command is submitted: after an idle the first request
  // waits for the Neural Engine to power on and load the program again, which
  // the GPU's work up to its first wait then overlaps. An evaluation queued
  // after the GPU raised its `ready` starts at once, its bound from then.
  for (const Job &job : jobs) queue(job);
  return ticket;
}

bool AneFfn::finish(const metal::CommandTiming &timing) {
  const std::optional<ane::Handoff::Ran> ran = completed();
  if (!ran) {
    warnStopped();
    return false;
  }
  std::string losing;
  if (committedLayers_) {
    const double milliseconds = std::chrono::duration<double, std::milli>(ran->total).count(),
                 longest = std::chrono::duration<double>(ran->longest).count();
    if (const std::optional<ane_ffn::Late> late =
            ane_ffn::late(timing.longestBufferSeconds, timing.medianBufferSeconds, longest))
      logWarning("Neural Engine FFN split: a buffer of a command took ", std::fixed, std::setprecision(2),
                 timing.longestBufferSeconds, " s against a median of ", std::setprecision(1),
                 1000.0 * timing.medianBufferSeconds, " ms; the ",
                 *late == ane_ffn::Late::NeuralEngine ? "Neural Engine was late" : "GPU was late",
                 " (the longest evaluation took ", 1000.0 * longest, " ms from its notice to its report).");
    ++served_.commands;
    served_.evaluations += committedEvaluations_;
    served_.milliseconds += milliseconds;
    const bool pipelined = committedEvaluations_ > committedLayers_;
    losing = breaker_.judge(committedRows_, committedLayers_, pipelined, milliseconds);
    const std::optional<double> alone = breaker_.alone(committedRows_, committedLayers_, pipelined);
    if (losing.empty() && alone &&
        ane_ffn::shared(1000.0 * timing.gpuSeconds, 1000.0 * timing.encodeGpuSeconds, *alone))
      logWarning("Neural Engine FFN split: a command took ", std::fixed, std::setprecision(2),
                 timing.gpuSeconds - timing.encodeGpuSeconds,
                 " s on the GPU against ", *alone / 1000.0, " s on the GPU alone over its layers; the GPU was busy "
                 "with another process's work.");
  }
  const AwakeClock::time_point now = AwakeClock::now();
  const ane_ffn::Pause::Verdict verdict = pause_.finished(now, committedLayers_ != 0, !losing.empty());
  if (verdict == ane_ffn::Pause::Verdict::Paused)
    logWarning("Neural Engine FFN split paused (", losing, "); the GPU runs the prefill FFN alone for ",
               std::chrono::duration_cast<std::chrono::minutes>(pause_.until() - now).count(),
               " min, then the split tries again.");
  if (verdict == ane_ffn::Pause::Verdict::Stopped) {
    handoff_.retire(std::move(losing));
    warnStopped();
  }
  return true;
}

std::optional<ane::Handoff::Ran> AneFfn::completed() {
  unfinished_ = false;
  std::optional<ane::Handoff::Ran> ran = handoff_.finish();
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
  const ane::Program::Binding &binding = share().evaluations.at(job.evaluation).bindings.at(job.set);
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
  if (layers.size() != planes_.size() || layers.front().gate->outputSize != shape_.intermediate)
    throw std::invalid_argument("ANE FFN split verified on other layers than its own");
  if (handoff_.retired()) throw std::runtime_error("ANE FFN split stopped (" + handoff_.reason() + ")");
  fillNormalized(ffn.normalized);
  // Layer l adds its FFN of the normalized rows to hidden[l & 1] into
  // hidden[(l & 1) ^ 1], from zeros, on the GPU alone, as the split's GPU part
  // alone over a partial output of zeros, or split.
  enum class Way : uint8_t { Gpu, GpuPart, Split };
  const auto forward = [&](Way way, uint32_t count, uint32_t rows, Steps steps) {
    std::memset(hidden[0].contents(), 0, hidden[0].sizeBytes());
    if (way != Way::Gpu) {
      std::fill_n(static_cast<uint16_t *>(memory_.partial.buffer.contents()),
                  memory_.partial.buffer.sizeBytes() / sizeof(uint16_t), way == Way::Split ? kHalfNaN : 0);
      static_cast<void>(begin(steps));
    }
    metal::CommandGraph graph;
    for (uint32_t layer = 0; layer < count; ++layer) {
      const metal::MetalBuffer &residual = hidden[layer & 1], &output = hidden[(layer & 1) ^ 1];
      if (way == Way::Gpu)
        linear_.addSwiGlu(graph, layers[layer], ffn, residual, output, rows);
      else {
        encode(graph, layer, ffn, residual, output, rows, way == Way::Split ? Parts::Both : Parts::Gpu);
        join(graph);
      }
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
  // split's rows, of each share's every function.
  const uint32_t count = std::min<uint32_t>(2, shape_.layers);
  const std::vector<float> gpu = forward(Way::Gpu, count, kMaximumRows, Steps::One);
  double worst = 0.0;
  for (const Steps steps : {Steps::One, Steps::Two}) {
    if (!shares_[static_cast<size_t>(steps)]) continue;
    const std::vector<float> part = forward(Way::GpuPart, count, kMaximumRows, steps);
    for (const Evaluation &evaluation : shares_[static_cast<size_t>(steps)]->evaluations) {
      const uint32_t rows = evaluation.rows;
      const std::vector<float> split = forward(Way::Split, count, rows, steps);
      const auto fail = [&](const std::string &why) {
        handoff_.retire("its output of " + std::to_string(rows) + " rows " + why);
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
  }
  return worst;
}

} // namespace splash::ops
