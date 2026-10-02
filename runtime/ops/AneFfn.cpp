#include "ops/AneFfn.hpp"

#include "metal/abi/AneFfn.h"
#include "metal/abi/QuantFormat.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {
namespace {

using Element = ane::Surface::Element;

// The rotation blocks (kernels/prefill/ane_ffn.metal) of the inputs, which
// gate and up multiply, and of the ANE's intermediate rows, which down
// multiplies: a larger block spreads the intermediate rows' outliers further
// before their per-token scale. Then the channels of an ANE input segment of
// gate and up, and of a segment of down's inputs.
constexpr uint32_t kBlock = 128;
constexpr uint32_t kIntermediateBlock = 512;
constexpr uint32_t kSegment = 2560;
constexpr uint32_t kQuantGroup = 64;
constexpr auto kCompletionTimeout = std::chrono::seconds(10);
// The channels a split moves in: whole blocks of the intermediate rotation
// for the ANE, whole 256-row tiles of the Q4 planes for the GPU.
constexpr uint32_t kChannelUnit = std::max(256u, kIntermediateBlock);

uint32_t gpuChannels(uint32_t intermediate, double share) {
  if (!(share > 0.0 && share < 1.0)) throw std::invalid_argument("ANE FFN share must lie in (0, 1)");
  if (intermediate % kChannelUnit)
    throw std::invalid_argument("ANE FFN split needs whole rotation blocks of channels");
  const auto units = static_cast<uint32_t>(std::lround((1.0 - share) * intermediate / kChannelUnit));
  if (!units || units >= intermediate / kChannelUnit)
    throw std::invalid_argument("ANE FFN share leaves the GPU or the ANE no channels");
  return units * kChannelUnit;
}

std::vector<uint32_t> segments(uint32_t channels) {
  std::vector<uint32_t> result;
  for (uint32_t begin = 0; begin < channels; begin += kSegment) result.push_back(std::min(kSegment, channels - begin));
  return result;
}

uint64_t pages(uint64_t bytes) { return (bytes + 16383) / 16384 * 16384; }

// A projection the split takes: affine Q4, or one quantized GGUF image tensor.
bool splittable(const Projection &projection) {
  if (projection.layout() == WeightLayout::Affine64) return true;
  const std::vector<QuantizedSegment> &segments = projection.blocks().segments;
  return segments.size() == 1 && !segments.front().isFloat() && !projection.rotation;
}

// Why the split does not take `layers`, or null when it does.
const char *unsupported(std::span<const SwiGluProjections> layers) {
  if (layers.empty()) return "ANE FFN split has no layers";
  const uint32_t hidden = layers.front().gate->inputSize, intermediate = layers.front().gate->outputSize;
  if (hidden % kSegment || hidden % (kBlock * 8)) return "ANE FFN split needs a hidden size of 2560-channel segments";
  if (intermediate % kChannelUnit) return "ANE FFN split needs whole rotation blocks of channels";
  for (const SwiGluProjections &layer : layers) {
    for (const Projection *projection : {layer.gate, layer.up, layer.down})
      if (!splittable(*projection)) return "ANE FFN split needs affine Q4 projections or quantized GGUF tensors";
    if (layer.gate->outputSize != intermediate || layer.gate->inputSize != hidden ||
        layer.up->outputSize != intermediate || layer.up->inputSize != hidden || layer.down->outputSize != hidden ||
        layer.down->inputSize != intermediate)
      return "ANE FFN split layers differ in shape";
  }
  return nullptr;
}

// A weight plane of a projection: per 256-row tile, `units` units of its inputs
// for each row, of `bytes` bytes each.
struct Plane {
  metal::MetalBuffer buffer;
  uint64_t units, bytes;
};
// The planes of a projection: the affine weights, scales and biases (units of
// 64 inputs), or a GGUF image's plane0, plane1 and meta (groups of 32, meta
// units of meta_groups).
std::vector<Plane> planes(const Projection &projection) {
  if (projection.layout() == WeightLayout::Affine64) {
    const AffineWeights &weights = projection.affine();
    const uint64_t units = projection.inputSize / kQuantGroup;
    return {{weights.weights, units, 32}, {weights.scales, units, 2}, {weights.biases, units, 2}};
  }
  const QuantizedSegment &segment = projection.blocks().segments.front();
  const QuantFormat &format = segment.format();
  const uint64_t groups = projection.inputSize / 32;
  std::vector<Plane> result{{segment.plane0, groups, format.plane0_bytes}};
  if (format.plane1_bytes) result.push_back({segment.plane1, groups, format.plane1_bytes});
  result.push_back({segment.meta, groups / format.meta_groups, format.meta_bytes});
  return result;
}
// A projection of `outputs` x `inputs` in the layout and format of `like`,
// over `views` of its planes in planes() order.
Projection projection(const Projection &like, uint32_t outputs, uint32_t inputs, std::vector<metal::MetalBuffer> views) {
  if (like.layout() == WeightLayout::Affine64) return Projection(outputs, inputs, AffineWeights{views[0], views[1], views[2]});
  const QuantizedSegment &segment = like.blocks().segments.front();
  const bool second = segment.format().plane1_bytes != 0;
  return Projection(outputs, inputs,
                    BlockWeights{{QuantizedSegment::planes(segment.formatId, outputs, inputs, views[0],
                                                           second ? views[1] : metal::MetalBuffer{}, views.back())}});
}
// A projection's weight planes as ane_ffn_weights and ane_ffn_row_scale bind
// them, and their kernel variant: the affine Q4 planes (groups of 64 inputs),
// or a GGUF image tensor's (the _gguf kernels, groups of 32 in its format).
struct WeightSource {
  metal::MetalBuffer a, b, c;
  uint32_t groups = 0, format = 0;
  std::string kernel;
};
WeightSource weightSource(const Projection &projection) {
  if (projection.layout() == WeightLayout::Affine64) {
    const AffineWeights &weights = projection.affine();
    return {weights.weights, weights.scales, weights.biases, projection.inputSize / kQuantGroup, 0, ""};
  }
  const QuantizedSegment &segment = projection.blocks().segments.front();
  return {segment.plane0, segment.plane1Slot(), segment.meta, projection.inputSize / 32, segment.formatId, "_gguf"};
}
// The Hadamard signs D of the rotations R = D H / sqrt(n) of inputs, weights
// and the ANE's intermediate rows alike: a block of n values takes the first n.
std::array<float, kIntermediateBlock> rotationSigns() {
  std::mt19937 generator(20260930);
  std::array<float, kIntermediateBlock> signs{};
  for (float &sign : signs) sign = (generator() & 1) ? -1.0f : 1.0f;
  return signs;
}

// A Core ML weight blob holding one fp16 tensor: the rotation of each of the
// ANE's intermediate channels as the [channels, kIntermediateBlock, 1, 1]
// weight of a grouped 1x1 convolution. Its metadata record is at offset 64.
std::vector<uint8_t> rotationBlob(uint32_t channels, const std::array<float, kIntermediateBlock> &signs) {
  constexpr uint32_t block = kIntermediateBlock;
  const uint64_t count = uint64_t{channels} * block;
  std::vector<uint8_t> blob(128 + count * sizeof(_Float16));
  const auto put = [&](size_t offset, auto value) { std::memcpy(blob.data() + offset, &value, sizeof value); };
  put(0, uint32_t{1});          // blobs
  put(4, uint32_t{2});          // storage version
  put(64, uint32_t{0xdeadbeef}); // metadata sentinel
  put(68, uint32_t{1});         // fp16
  put(72, count * sizeof(_Float16));
  put(80, uint64_t{128});
  auto *values = reinterpret_cast<_Float16 *>(blob.data() + 128);
  const float norm = 1.0f / std::sqrt(float(block));
  for (uint32_t channel = 0; channel < channels; ++channel)
    for (uint32_t input = 0; input < block; ++input) {
      const uint32_t output = channel % block;
      const float hadamard = (std::popcount(output & input) & 1) ? -1.0f : 1.0f;
      values[uint64_t{channel} * block + input] = _Float16(signs[output] * hadamard * norm);
    }
  return blob;
}

std::string shape(uint64_t rows, uint64_t width) {
  return "[1, 1, " + std::to_string(rows) + ", " + std::to_string(width) + "]";
}
std::string tensor(const char *type, uint64_t rows, uint64_t width) {
  return std::string("tensor<") + type + ", " + shape(rows, width) + ">";
}
std::string buffer(const char *type, uint64_t rows, uint64_t width, uint64_t stride) {
  const std::string plane = std::to_string(rows * stride);
  return std::string("tensor_buffer<") + type + ", shape=" + shape(rows, width) + ", strides=[" + plane + ", " +
         plane + ", " + std::to_string(stride) + ", 1], interleave_factors=[1, 1, 1, 1]>";
}

// The ANE's share of the FFN over `rows` rows as MIL. Inputs: x<k>, the rotated input rows
// channel-major in int8 segments, tx their per-token scales; w<g|u><k>, the
// rotated int8 gate and up rows of segment k, s<g|u> their per-row scales;
// wd<i> down's rotated int8 inputs of segment i, sd its per-row scales. Every
// int8 value is dequantized by 2^-7 against fp16 overflow, which the scales
// carry back. The intermediate rows are rotated and quantized per token here.
std::string ffnProgram(uint32_t hidden, uint32_t channels, const std::vector<uint32_t> &down, uint64_t rows) {
  const uint32_t inputs = hidden / kSegment;
  std::string parameters, body;
  const auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
  const auto input = [&](const char *type, const std::string &name, uint64_t height, uint64_t width) {
    const uint64_t stride = (width * (type[0] == 'i' ? 1 : 2) + 63) / 64 * 64 / (type[0] == 'i' ? 1 : 2);
    parameters += (parameters.empty() ? "" : ", ") + buffer(type, height, width, stride) + " " + name;
    line(tensor(type, height, width) + " " + name + "_t = tensor_buffer_to_tensor<ios17>(input = " + name + ")");
  };
  const auto matmul = [&](const std::string &name, const std::string &weights, const std::string &values,
                          uint64_t height, uint64_t width) {
    line(tensor("fp16", height, width) + " " + weights + "_d = dequantize(input = " + weights +
         "_t, scale = fp16(0x1p-7))");
    line(tensor("fp16", height, rows) + " " + name +
         " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + weights + "_d, y = " + values + ")");
  };
  const auto sum = [&](const std::string &prefix, size_t terms, uint64_t height) {
    std::string total = prefix + "0";
    for (size_t term = 1; term < terms; ++term) {
      const std::string next = prefix + "_sum" + std::to_string(term);
      line(tensor("fp16", height, rows) + " " + next + " = add(x = " + total + ", y = " + prefix +
           std::to_string(term) + ")");
      total = next;
    }
    return total;
  };
  const auto f16 = [&](const std::string &name, uint64_t height, uint64_t width, const std::string &expression) {
    line(tensor("fp16", height, width) + " " + name + " = " + expression);
  };

  for (uint32_t k = 0; k < inputs; ++k) {
    const std::string x = "x" + std::to_string(k);
    input("int8", x, kSegment, rows);
    f16(x + "_d", kSegment, rows, "dequantize(input = " + x + "_t, scale = fp16(0x1p-7))");
  }
  input("fp16", "tx", 1, rows);
  for (const char *projection : {"g", "u"}) {
    const std::string p = projection;
    input("fp16", "s" + p, channels, 1);
    for (uint32_t k = 0; k < inputs; ++k) {
      const std::string w = "w" + p + std::to_string(k);
      input("int8", w, channels, kSegment);
      matmul(p + "m" + std::to_string(k), w, "x" + std::to_string(k) + "_d", channels, kSegment);
    }
    f16(p + "s", channels, rows, "mul(x = " + sum(p + "m", inputs, channels) + ", y = s" + p + "_t)");
  }
  const std::string c = std::to_string(channels), r = std::to_string(rows);
  f16("gt", channels, rows, "mul(x = gs, y = tx_t)");
  f16("sig", channels, rows, "sigmoid(x = gt)");
  f16("silu", channels, rows, "mul(x = gt, y = sig)");
  f16("h", channels, rows, "mul(x = silu, y = us)");
  line("tensor<fp16, [1, " + c + ", 1, " + r + "]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, " + c + ", 1, " +
       r + "]))");
  const std::string block = std::to_string(kIntermediateBlock);
  line("tensor<fp16, [" + c + ", " + block + ", 1, 1]> rotation = const()[name = string(\"rotation\"), val = tensor<fp16, [" +
       c + ", " + block + ", 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(64)))]");
  line("tensor<fp16, [1, " + c + ", 1, " + r +
       "]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32(" + std::to_string(channels / kIntermediateBlock) +
       "), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, "
       "1]), weight = rotation, x = h4)");
  f16("hr", channels, rows, "reshape(x = hr4, shape = tensor<int32, [4]>(" + shape(channels, rows) + "))");
  f16("habs", channels, rows, "abs(x = hr)");
  f16("peak", 1, rows, "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))");
  f16("floor", 1, rows, "maximum(x = peak, y = fp16(0x1p-12))");
  f16("inverse", 1, rows, "real_div(x = fp16(0x1.fcp+6), y = floor)");
  f16("hs", channels, rows, "mul(x = hr, y = inverse)");
  line(tensor("int8", channels, rows) + " hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))");
  f16("hd", channels, rows, "dequantize(input = hq, scale = fp16(0x1p-7))");
  f16("hscale", 1, rows, "mul(x = floor, y = fp16(0x1.0204081020408p+0))");
  uint32_t begin = 0;
  for (size_t i = 0; i < down.size(); ++i) {
    const std::string index = std::to_string(i), slice = "hd" + index;
    f16(slice, down[i], rows,
        "slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, " + std::to_string(begin) +
            ", 0]), size = tensor<int32, [4]>(" + shape(down[i], rows) + "))");
    input("int8", "wd" + index, hidden, down[i]);
    matmul("dm" + index, "wd" + index, slice, hidden, down[i]);
    begin += down[i];
  }
  input("fp16", "sd", hidden, 1);
  f16("ds", hidden, rows, "mul(x = " + sum("dm", down.size(), hidden) + ", y = sd_t)");
  f16("ys", 1, rows, "mul(x = hscale, y = tx_t)");
  f16("yt", hidden, rows, "mul(x = ds, y = ys)");
  const std::string plane = std::to_string(uint64_t{hidden} * rows);
  line(buffer("fp16", hidden, rows, rows) +
       " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
       "strides = tensor<int64, [4]>([" + plane + ", " + plane + ", " + r + ", 1]))");
  return "program(1.3)\n{\n    func main_ane<ios18>(" + parameters + ") {\n" + body + "    } -> (y);\n}\n";
}

} // namespace

uint64_t AneFfn::plannedBytes(std::span<const SwiGluProjections> layers, double share) {
  const uint32_t hidden = layers.front().gate->inputSize, intermediate = layers.front().gate->outputSize;
  const uint32_t gpu = gpuChannels(intermediate, share), ane = intermediate - gpu;
  uint64_t bytes = pages(kIntermediateBlock * sizeof(float)) + pages(uint64_t{layers.size()} * (2 * ane + hidden) * 2) +
                   pages(uint64_t{kRows} * hidden * 2);
  for (uint32_t rows : kProgramRows)
    bytes += (hidden / kSegment) * ane::Surface::bytes(kSegment, rows, Element::Int8) +
             ane::Surface::bytes(1, rows, Element::Float16) + ane::Surface::bytes(hidden, rows, Element::Float16);
  uint64_t set = 2 * ane::Surface::bytes(ane, 1, Element::Float16) + ane::Surface::bytes(hidden, 1, Element::Float16) +
                 2 * (hidden / kSegment) * ane::Surface::bytes(ane, kSegment, Element::Int8);
  for (uint32_t width : segments(ane)) set += ane::Surface::bytes(hidden, width, Element::Int8);
  return bytes + 2 * set;
}

AneFfn::AneFfn(metal::MetalBackend &backend, const Linear &linear, std::span<const SwiGluProjections> layers,
               double share)
    : AneFfn(backend, linear, layers, share, kProgramRows) {}

AneFfn::AneFfn(metal::MetalBackend &backend, const Linear &linear, std::span<const SwiGluProjections> layers,
               double share, std::span<const uint32_t> programRows)
    : backend_(backend), linear_(linear) {
  if (const char *reason = unsupported(layers)) throw std::invalid_argument(reason);
  hidden_ = layers.front().gate->inputSize;
  intermediate_ = layers.front().gate->outputSize;
  gpuChannels_ = gpuChannels(intermediate_, share);
  aneChannels_ = intermediate_ - gpuChannels_;
  downSegments_ = segments(aneChannels_);

  // The bytes the backend counts, which the memory audit compares with the
  // runtime's categories: a small buffer can take less than its pages.
  const auto allocate = [&](uint64_t bytes, const char *label) {
    metal::MetalBuffer buffer = backend_.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
    allocatedBytes_ += buffer.allocatedBytes();
    return buffer;
  };
  const auto surface = [&](uint32_t rows, uint32_t width, Element element) {
    ane::Surface result = ane::Surface::create(backend_, rows, width, element);
    allocatedBytes_ += result.buffer.allocatedBytes();
    return result;
  };

  const std::array<float, kIntermediateBlock> signs = rotationSigns();
  signs_ = allocate(sizeof signs, "ane ffn signs");
  std::memcpy(signs_.contents(), signs.data(), sizeof signs);
  rowScales_ = allocate(uint64_t{layers.size()} * (2 * aneChannels_ + hidden_) * 2, "ane ffn row scales");
  rotated_ = allocate(uint64_t{kRows} * hidden_ * 2, "ane ffn rotated input");

  // The GPU's share: gate and up rows lead each projection's 256-row tiles,
  // and down's leading inputs each tile's groups.
  const auto leadingRows = [&](const Projection &source) {
    std::vector<metal::MetalBuffer> views;
    for (const Plane &plane : planes(source))
      views.push_back(backend_.view(plane.buffer, 0, uint64_t{gpuChannels_} * plane.units * plane.bytes));
    return projection(source, gpuChannels_, hidden_, std::move(views));
  };
  const auto leadingInputs = [&](const Projection &source) {
    std::vector<metal::MetalBuffer> views;
    for (const Plane &plane : planes(source)) views.push_back(plane.buffer);
    Projection result = projection(source, source.outputSize, gpuChannels_, std::move(views));
    result.planeInputs = source.inputSize;
    return result;
  };
  for (const SwiGluProjections &source : layers)
    layers_.push_back({source, leadingRows(*source.gate), leadingRows(*source.up), leadingInputs(*source.down)});

  for (Weights &set : sets_) {
    for (uint32_t k = 0; k < hidden_ / kSegment; ++k) {
      set.gate.push_back(surface(aneChannels_, kSegment, Element::Int8));
      set.up.push_back(surface(aneChannels_, kSegment, Element::Int8));
    }
    for (uint32_t width : downSegments_) set.down.push_back(surface(hidden_, width, Element::Int8));
    set.gateScale = surface(aneChannels_, 1, Element::Float16);
    set.upScale = surface(aneChannels_, 1, Element::Float16);
    set.downScale = surface(hidden_, 1, Element::Float16);
  }

  const std::vector<uint8_t> blob = rotationBlob(aneChannels_, signs);
  for (uint32_t rows : programRows) {
    Evaluation &evaluation = evaluations_.emplace_back();
    evaluation.rows = rows;
    for (uint32_t k = 0; k < hidden_ / kSegment; ++k)
      evaluation.inputs.push_back(surface(kSegment, rows, Element::Int8));
    evaluation.tokenScale = surface(1, rows, Element::Float16);
    evaluation.partial = surface(hidden_, rows, Element::Float16);
    evaluation.program = std::make_unique<ane::Program>(ffnProgram(hidden_, aneChannels_, downSegments_, rows), blob);
    for (uint32_t index = 0; index < 2; ++index) {
      const Weights &set = sets_[index];
      std::vector<ane::Surface> &bindings = evaluation.bindings[index];
      for (const std::string &name : evaluation.program->inputs()) {
        const auto segment = [&](size_t prefix) { return std::stoul(name.substr(prefix)); };
        if (name == "tx") bindings.push_back(evaluation.tokenScale);
        else if (name == "sg") bindings.push_back(set.gateScale);
        else if (name == "su") bindings.push_back(set.upScale);
        else if (name == "sd") bindings.push_back(set.downScale);
        else if (name.starts_with("wg")) bindings.push_back(set.gate.at(segment(2)));
        else if (name.starts_with("wu")) bindings.push_back(set.up.at(segment(2)));
        else if (name.starts_with("wd")) bindings.push_back(set.down.at(segment(2)));
        else if (name.starts_with("x")) bindings.push_back(evaluation.inputs.at(segment(1)));
        else throw std::logic_error("unknown ANE FFN program input " + name);
      }
    }
  }
  event_ = backend_.newSharedEvent();

  // Each row's shared int8 scale over the ANE's share of its inputs.
  metal::CommandGraph graph;
  for (uint32_t layer = 0; layer < layers_.size(); ++layer) {
    const SwiGluProjections &source = layers_[layer].source;
    const auto add = [&](const Projection &projection, uint32_t part, uint32_t row, uint32_t input, uint32_t width,
                         uint32_t rows, uint32_t block) {
      const WeightSource weights = weightSource(projection);
      graph.add("ane_ffn_row_scale" + weights.kernel + "_" + std::to_string(block),
                {weights.a, weights.b, weights.c, rowScales(layer, part), signs_},
                AneFfnWeightParams{weights.groups, row, input, width, 0, 0, weights.format}, {rows / 8, 1, 1});
    };
    add(*source.gate, 0, gpuChannels_, 0, hidden_, aneChannels_, kBlock);
    add(*source.up, 1, gpuChannels_, 0, hidden_, aneChannels_, kBlock);
    add(*source.down, 2, 0, gpuChannels_, aneChannels_, hidden_, kIntermediateBlock);
  }
  static_cast<void>(backend_.submitCommand(graph.dispatches()));
}

AneFfn::~AneFfn() { static_cast<void>(wait(true)); }

metal::MetalBuffer AneFfn::rowScales(uint32_t layer, uint32_t part) const {
  const uint64_t offset = (uint64_t{layer} * (2 * aneChannels_ + hidden_) + uint64_t{part} * aneChannels_) * 2;
  return backend_.view(rowScales_, offset, uint64_t{part < 2 ? aneChannels_ : hidden_} * 2);
}

// Layer `layer`'s int8 weights and row scales into staging set `set`.
void AneFfn::addWeights(metal::CommandGraph &graph, uint32_t layer, uint32_t set) const {
  const SwiGluProjections &source = layers_.at(layer).source;
  const Weights &target = sets_[set];
  const auto add = [&](const Projection &projection, const metal::MetalBuffer &scales, const ane::Surface &output,
                       const ane::Surface &scale, uint32_t row, uint32_t input, uint32_t width, uint32_t rows,
                       uint32_t block) {
    const WeightSource weights = weightSource(projection);
    graph.add("ane_ffn_weights" + weights.kernel + "_" + std::to_string(block),
              {weights.a, weights.b, weights.c, scales, output.buffer, scale.buffer, signs_},
              AneFfnWeightParams{weights.groups, row, input, width, output.strideBytes, scale.strideBytes / 2,
                                 weights.format},
              {rows / 8, width / block, 1});
  };
  for (uint32_t k = 0; k < target.gate.size(); ++k) {
    add(*source.gate, rowScales(layer, 0), target.gate[k], target.gateScale, gpuChannels_, k * kSegment, kSegment,
        aneChannels_, kBlock);
    add(*source.up, rowScales(layer, 1), target.up[k], target.upScale, gpuChannels_, k * kSegment, kSegment,
        aneChannels_, kBlock);
  }
  uint32_t begin = gpuChannels_;
  for (size_t i = 0; i < downSegments_.size(); ++i) {
    add(*source.down, rowScales(layer, 2), target.down[i], target.downScale, 0, begin, downSegments_[i], hidden_,
        kIntermediateBlock);
    begin += downSegments_[i];
  }
}

void AneFfn::add(metal::CommandGraph &graph, uint32_t layer, metal::MetalBuffer normalized, metal::MetalBuffer sums,
                 metal::MetalBuffer gateScratch, metal::MetalBuffer intermediate, metal::MetalBuffer downSums,
                 metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows, LinearScratch scratch) {
  encode(graph, layer, normalized, sums, gateScratch, intermediate, downSums, residual, output, rows, scratch,
         Parts::Both);
}

void AneFfn::encode(metal::CommandGraph &graph, uint32_t layer, metal::MetalBuffer normalized,
                    metal::MetalBuffer sums, metal::MetalBuffer gateScratch, metal::MetalBuffer intermediate,
                    metal::MetalBuffer downSums, metal::MetalBuffer residual, metal::MetalBuffer output,
                    uint32_t rows, LinearScratch scratch, Parts parts) {
  if (!rows || rows > kRows) throw std::invalid_argument("ANE FFN chunk exceeds its rows");
  const Layer &current = layers_.at(layer);
  const uint32_t set = layer & 1, tiles = (rows + 31) / 32;
  const bool gpu = parts != Parts::Ane, ane = parts != Parts::Gpu;
  const auto found =
      std::ranges::find_if(evaluations_, [&](const Evaluation &evaluation) { return evaluation.rows >= rows; });
  if (found == evaluations_.end()) throw std::invalid_argument("ANE FFN has no program of the chunk's rows");
  const auto index = static_cast<uint32_t>(found - evaluations_.begin());
  const Evaluation &evaluation = *found;
  // Each command stages layer 0's weights, then each layer the next one's.
  if (gpu && !layer) addWeights(graph, 0, 0);
  uint64_t wait = 0;
  if (ane) {
    graph.add("ane_ffn_rotate", {normalized, signs_, rotated_, evaluation.tokenScale.buffer},
              AneFfnRotateParams{hidden_}, {rows, 1, 1}, {256, 1, 1});
    for (uint32_t k = 0; k < evaluation.inputs.size(); ++k)
      graph.add("ane_ffn_pack", {rotated_, evaluation.inputs[k].buffer},
                AneFfnPackParams{hidden_, k * kSegment, evaluation.inputs[k].strideBytes}, {tiles, kSegment / 32, 1},
                {32, 8, 1});
    wait = ++value_;
    graph.signal(event_, wait);
  }
  if (gpu) {
    linear_.addPrefill(graph, normalized, current.gate, gateScratch, sums, rows, scratch);
    linear_.addPrefillUpWithGate(graph, normalized, current.up, gateScratch, intermediate, sums, downSums, rows,
                                 scratch);
    linear_.addPrefillResidual(graph, intermediate, current.down, residual, output, downSums, rows, scratch);
    if (layer + 1 < layers_.size()) addWeights(graph, layer + 1, set ^ 1);
  }
  if (ane) {
    const uint64_t signal = ++value_;
    graph.wait(event_, signal);
    graph.add("ane_ffn_join", {output, evaluation.partial.buffer},
              AneFfnJoinParams{hidden_, evaluation.partial.strideBytes / 2, rows}, {tiles, hidden_ / 32, 1},
              {32, 8, 1});
    jobs_.push_back({index, set, wait, signal});
  }
}

void AneFfn::submit() {
  std::vector<Job> jobs = std::exchange(jobs_, {});
  for (const Job &job : jobs) {
    try {
      const Evaluation &evaluation = evaluations_[job.evaluation];
      evaluation.program->enqueue(evaluation.bindings[job.set], evaluation.partial, event_, job.wait, job.signal,
                                  [completions = completions_](bool success) {
                                    std::lock_guard lock(completions->mutex);
                                    ++completions->completed;
                                    completions->failed |= !success;
                                    completions->changed.notify_all();
                                  });
    } catch (...) {
      static_cast<void>(wait(true));
      throw;
    }
    queued_.push_back(job);
    ++queuedCount_;
  }
}

void AneFfn::finish() {
  if (!wait(false)) throw std::runtime_error("ANE FFN evaluations did not complete");
  std::lock_guard lock(completions_->mutex);
  if (std::exchange(completions_->failed, false)) throw std::runtime_error("ANE FFN evaluation failed");
}

bool AneFfn::wait(bool release) {
  std::unique_lock lock(completions_->mutex);
  const uint64_t first = queuedCount_ - queued_.size();
  for (size_t index = 0; index < queued_.size(); ++index) {
    const uint64_t sequence = first + index + 1;
    if (release) ane::Program::release(event_, queued_[index].wait);
    if (!completions_->changed.wait_for(lock, kCompletionTimeout,
                                        [&] { return completions_->completed >= sequence; }))
      return false;
  }
  queued_.clear();
  return true;
}

// Calibration. At share s, a layer of a full chunk takes
//   T(s) = max(G(s), A(s), uG G(s) + uA A(s)):
// G(s) is the GPU's part alone (its channels and the staging of the ANE's
// int8 weights) and A(s) the ANE's (the packing, the evaluation and the
// join), each close to linear in s. The third term is the memory bandwidth
// both parts share when they run together, uG and uA each one's fraction of
// it alone. Timing each part alone at two shares gives G and A, and both
// together at those shares gives uG and uA where the bandwidth binds there.
// On Qwen3.8-27B's FFN this predicts the fastest share measured on the M4
// (0.83, no contention), the M5 Pro (0.52) and the M6, where contention moves
// it from where G meets A (0.68) to 0.92. Near a flat optimum the split takes
// the least share within kTolerance of it, for the ANE's memory and error.
bool AneFfn::supports(std::span<const SwiGluProjections> layers) { return !unsupported(layers); }

AneFfn::Calibration AneFfn::calibrate(metal::MetalBackend &backend, const Linear &linear,
                                      std::span<const SwiGluProjections> layers, const ChunkBuffers &buffers) {
  constexpr std::array<double, 2> kShares{0.4, 0.8};
  // The split's gain below which the GPU runs the FFN alone, the excess over
  // max(G, A) that shows the bandwidth binding, and the slack of the optimum.
  constexpr double kMinimumGain = 0.05, kBindingMargin = 0.03, kTolerance = 0.01;
  // Per-layer milliseconds come from commands of one and of four layers,
  // which cancels each command's fixed costs, of layers spread over the
  // model, whose formats can differ with depth; a fifth layer gives the
  // fourth one's next weights to stage.
  constexpr uint32_t kLayers = 4;
  if (layers.size() <= kLayers || !supports(layers)) return {};
  std::vector<SwiGluProjections> sampled;
  for (uint32_t index = 0; index <= kLayers; ++index) sampled.push_back(layers[index * (layers.size() - 1) / kLayers]);

  const auto fill = [](const metal::MetalBuffer &buffer, uint16_t value) {
    if (auto *data = static_cast<uint16_t *>(buffer.contents())) std::fill_n(data, buffer.sizeBytes() / 2, value);
  };
  // Inputs of the magnitude of normalized rows; timing does not depend on
  // their values.
  if (auto *data = static_cast<uint16_t *>(buffers.normalized.contents())) {
    std::mt19937 random(20261001);
    std::uniform_real_distribution<float> uniform(-2.0f, 2.0f);
    for (uint64_t i = 0; i < buffers.normalized.sizeBytes() / 2; ++i)
      data[i] = static_cast<uint16_t>(std::bit_cast<uint32_t>(uniform(random)) >> 16);
  }
  fill(buffers.sums, 0);
  for (const metal::MetalBuffer &hidden : buffers.hidden) fill(hidden, 0);

  struct Point final {
    double share, gpu, ane, both;
  };
  std::vector<Point> points;
  for (const double share : kShares) {
    AneFfn split(backend, linear, sampled, share, std::array{kRows});
    const auto milliseconds = [&](Parts parts, uint32_t count) {
      double best = std::numeric_limits<double>::infinity();
      for (int run = 0; run < 2; ++run) {
        metal::CommandGraph graph;
        split.begin();
        for (uint32_t layer = 0; layer < count; ++layer)
          split.encode(graph, layer, buffers.normalized, buffers.sums, buffers.gateScratch, buffers.intermediate,
                       buffers.downSums, buffers.hidden[layer & 1], buffers.hidden[(layer & 1) ^ 1], kRows,
                       buffers.scratch, parts);
        const auto start = std::chrono::steady_clock::now();
        split.submit();
        static_cast<void>(backend.submitCommand(graph.dispatches()));
        split.finish();
        best = std::min(best,
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
      }
      return best;
    };
    const auto perLayer = [&](Parts parts) {
      return (milliseconds(parts, kLayers) - milliseconds(parts, 1)) / (kLayers - 1);
    };
    // The first command compiles and wires what the others run.
    static_cast<void>(milliseconds(Parts::Both, kLayers));
    points.push_back({share, perLayer(Parts::Gpu), perLayer(Parts::Ane), perLayer(Parts::Both)});
  }

  struct Line final {
    double at0, slope;
    [[nodiscard]] double operator()(double share) const noexcept { return at0 + slope * share; }
  };
  const Point &p = points[0], &q = points[1];
  const Line gpu{p.gpu - (q.gpu - p.gpu) / (q.share - p.share) * p.share, (q.gpu - p.gpu) / (q.share - p.share)};
  const Line ane{p.ane - (q.ane - p.ane) / (q.share - p.share) * p.share, (q.ane - p.ane) / (q.share - p.share)};
  const auto excess = [](const Point &x) { return x.both / std::max(x.gpu, x.ane) - 1.0; };
  double uG = 0.0, uA = 0.0;
  bool solved = false;
  if (excess(p) > kBindingMargin && excess(q) > kBindingMargin) {
    const double det = p.gpu * q.ane - p.ane * q.gpu;
    if (std::abs(det) > 1e-9) {
      uG = (p.both * q.ane - p.ane * q.both) / det;
      uA = (p.gpu * q.both - p.both * q.gpu) / det;
      solved = uG >= 0.0 && uG <= 1.05 && uA >= 0.0 && uA <= 1.05;
    }
  }
  if (!solved) {
    // Binding at one share: the GPU's part alone nearly saturates memory, so
    // that share's excess over it is the ANE's fraction.
    const Point &x = excess(p) >= excess(q) ? p : q;
    uG = excess(x) > kBindingMargin ? 1.0 : 0.0;
    uA = uG > 0.0 ? (x.both - x.gpu) / x.ane : 0.0;
  }
  uG = std::clamp(uG, 0.0, 1.0);
  uA = std::clamp(uA, 0.0, 1.0);

  const uint32_t intermediate = layers.front().gate->outputSize, units = intermediate / kChannelUnit;
  const auto predicted = [&](uint32_t gpuUnits) {
    const double share = 1.0 - static_cast<double>(gpuUnits) / units, g = gpu(share), a = ane(share);
    return std::max({g, a, uG * g + uA * a});
  };
  double best = std::numeric_limits<double>::infinity();
  for (uint32_t gpuUnits = 1; gpuUnits < units; ++gpuUnits) best = std::min(best, predicted(gpuUnits));
  if (best > (1.0 - kMinimumGain) * gpu.at0) return {0.0, gpu.at0, gpu.at0};
  // The most GPU channels within the tolerance.
  uint32_t gpuUnits = units - 1;
  while (predicted(gpuUnits) > (1.0 + kTolerance) * best) --gpuUnits;
  return {1.0 - static_cast<double>(gpuUnits) / units, gpu.at0, predicted(gpuUnits)};
}

} // namespace splash::ops
