#include "ops/AneFfn.hpp"

#include "metal/abi/AneFfn.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
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

// Whole blocks of the intermediate rotation for the ANE, whole 256-row tiles
// of the Q4 planes for the GPU.
uint32_t gpuChannels(uint32_t intermediate, double share) {
  constexpr uint32_t unit = std::max(256u, kIntermediateBlock);
  if (!(share > 0.0 && share < 1.0)) throw std::invalid_argument("ANE FFN share must lie in (0, 1)");
  if (intermediate % unit) throw std::invalid_argument("ANE FFN split needs whole rotation blocks of channels");
  const auto units = static_cast<uint32_t>(std::lround((1.0 - share) * intermediate / unit));
  if (!units || units >= intermediate / unit)
    throw std::invalid_argument("ANE FFN share leaves the GPU or the ANE no channels");
  return units * unit;
}

std::vector<uint32_t> segments(uint32_t channels) {
  std::vector<uint32_t> result;
  for (uint32_t begin = 0; begin < channels; begin += kSegment) result.push_back(std::min(kSegment, channels - begin));
  return result;
}

uint64_t pages(uint64_t bytes) { return (bytes + 16383) / 16384 * 16384; }

// The GPU share of down: its first `channels` inputs of each 256-row tile.
uint64_t downShareBytes(uint32_t hidden, uint32_t channels) {
  return pages(uint64_t{hidden} * channels / 2 + uint64_t{hidden} * (channels / kQuantGroup) * 4);
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

// The ANE's share of the FFN as MIL. Inputs: x<k>, the rotated input rows
// channel-major in int8 segments, tx their per-token scales; w<g|u><k>, the
// rotated int8 gate and up rows of segment k, s<g|u> their per-row scales;
// wd<i> down's rotated int8 inputs of segment i, sd its per-row scales. Every
// int8 value is dequantized by 2^-7 against fp16 overflow, which the scales
// carry back. The intermediate rows are rotated and quantized per token here.
std::string ffnProgram(uint32_t hidden, uint32_t channels, const std::vector<uint32_t> &down) {
  const uint64_t rows = AneFfn::kRows;
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

uint64_t AneFfn::plannedBytes(uint32_t layers, uint32_t hidden, uint32_t intermediate, double share) {
  const uint32_t gpu = gpuChannels(intermediate, share), ane = intermediate - gpu;
  uint64_t bytes = pages(kIntermediateBlock * sizeof(float)) + pages(uint64_t{layers} * (2 * ane + hidden) * 2) +
                   pages(uint64_t{kRows} * hidden * 2) + uint64_t{layers} * downShareBytes(hidden, gpu) +
                   (hidden / kSegment) * ane::Surface::bytes(kSegment, kRows, Element::Int8) +
                   ane::Surface::bytes(1, kRows, Element::Float16) +
                   ane::Surface::bytes(hidden, kRows, Element::Float16);
  uint64_t set = 2 * ane::Surface::bytes(ane, 1, Element::Float16) + ane::Surface::bytes(hidden, 1, Element::Float16) +
                 2 * (hidden / kSegment) * ane::Surface::bytes(ane, kSegment, Element::Int8);
  for (uint32_t width : segments(ane)) set += ane::Surface::bytes(hidden, width, Element::Int8);
  return bytes + 2 * set;
}

AneFfn::AneFfn(metal::MetalBackend &backend, const Linear &linear, std::span<const SwiGluProjections> layers,
               double share)
    : backend_(backend), linear_(linear) {
  if (layers.empty()) throw std::invalid_argument("ANE FFN split has no layers");
  hidden_ = layers.front().gate->inputSize;
  intermediate_ = layers.front().gate->outputSize;
  gpuChannels_ = gpuChannels(intermediate_, share);
  aneChannels_ = intermediate_ - gpuChannels_;
  downSegments_ = segments(aneChannels_);
  if (hidden_ % kSegment || hidden_ % (kBlock * 8))
    throw std::invalid_argument("ANE FFN split needs a hidden size of 2560-channel segments");
  for (const SwiGluProjections &layer : layers) {
    for (const Projection *projection : {layer.gate, layer.up, layer.down})
      if (projection->layout() != WeightLayout::Affine64)
        throw std::invalid_argument("ANE FFN split needs affine Q4 projections");
    if (layer.gate->outputSize != intermediate_ || layer.gate->inputSize != hidden_ ||
        layer.up->outputSize != intermediate_ || layer.up->inputSize != hidden_ ||
        layer.down->outputSize != hidden_ || layer.down->inputSize != intermediate_)
      throw std::invalid_argument("ANE FFN split layers differ in shape");
  }

  const auto allocate = [&](uint64_t bytes, const char *label) {
    allocatedBytes_ += pages(bytes);
    return backend_.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
  };
  const auto surface = [&](uint32_t rows, uint32_t width, Element element) {
    allocatedBytes_ += ane::Surface::bytes(rows, width, element);
    return ane::Surface::create(backend_, rows, width, element);
  };

  const std::array<float, kIntermediateBlock> signs = rotationSigns();
  signs_ = allocate(sizeof signs, "ane ffn signs");
  std::memcpy(signs_.contents(), signs.data(), sizeof signs);
  rowScales_ = allocate(uint64_t{layers.size()} * (2 * aneChannels_ + hidden_) * 2, "ane ffn row scales");
  rotated_ = allocate(uint64_t{kRows} * hidden_ * 2, "ane ffn rotated input");

  // The GPU's share: gate and up rows lead each projection's 256-row tiles;
  // down's leading inputs are repacked out of each of its tiles.
  const uint32_t groups = intermediate_ / kQuantGroup, shareGroups = gpuChannels_ / kQuantGroup;
  const uint64_t tileWeights = uint64_t{shareGroups} * 256 * 32, tileScales = uint64_t{shareGroups} * 256 * 2;
  for (const SwiGluProjections &source : layers) {
    const auto prefix = [&](const Projection &projection) {
      const AffineWeights &weights = projection.affine();
      const uint64_t scaleBytes = uint64_t{gpuChannels_} * (hidden_ / kQuantGroup) * 2;
      return Projection(gpuChannels_, hidden_,
                        AffineWeights{backend_.view(weights.weights, 0, uint64_t{gpuChannels_} * hidden_ / 2),
                                      backend_.view(weights.scales, 0, scaleBytes),
                                      backend_.view(weights.biases, 0, scaleBytes)});
    };
    const AffineWeights &down = source.down->affine();
    const auto *weights = static_cast<const uint8_t *>(down.weights.contents());
    const auto *scales = static_cast<const uint8_t *>(down.scales.contents());
    const auto *biases = static_cast<const uint8_t *>(down.biases.contents());
    if (!weights || !scales || !biases) throw std::invalid_argument("ANE FFN split needs CPU-visible weights");
    const uint32_t tiles = hidden_ / 256;
    const metal::MetalBuffer packed = allocate(downShareBytes(hidden_, gpuChannels_), "ane ffn gpu down");
    AffineWeights share{backend_.view(packed, 0, tiles * tileWeights),
                        backend_.view(packed, tiles * tileWeights, tiles * tileScales),
                        backend_.view(packed, tiles * (tileWeights + tileScales), tiles * tileScales)};
    for (uint32_t tile = 0; tile < tiles; ++tile) {
      std::memcpy(static_cast<uint8_t *>(share.weights.contents()) + tile * tileWeights,
                  weights + uint64_t{tile} * groups * 256 * 32, tileWeights);
      std::memcpy(static_cast<uint8_t *>(share.scales.contents()) + tile * tileScales,
                  scales + uint64_t{tile} * groups * 256 * 2, tileScales);
      std::memcpy(static_cast<uint8_t *>(share.biases.contents()) + tile * tileScales,
                  biases + uint64_t{tile} * groups * 256 * 2, tileScales);
    }
    layers_.push_back({source, prefix(*source.gate), prefix(*source.up),
                       Projection(hidden_, gpuChannels_, std::move(share))});
  }

  for (uint32_t k = 0; k < hidden_ / kSegment; ++k) inputs_.push_back(surface(kSegment, kRows, Element::Int8));
  tokenScale_ = surface(1, kRows, Element::Float16);
  partial_ = surface(hidden_, kRows, Element::Float16);
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
  program_ = std::make_unique<ane::Program>(ffnProgram(hidden_, aneChannels_, downSegments_), blob);
  for (uint32_t index = 0; index < 2; ++index) {
    const Weights &set = sets_[index];
    for (const std::string &name : program_->inputs()) {
      const auto segment = [&](size_t prefix) { return std::stoul(name.substr(prefix)); };
      if (name == "tx") bindings_[index].push_back(tokenScale_);
      else if (name == "sg") bindings_[index].push_back(set.gateScale);
      else if (name == "su") bindings_[index].push_back(set.upScale);
      else if (name == "sd") bindings_[index].push_back(set.downScale);
      else if (name.starts_with("wg")) bindings_[index].push_back(set.gate.at(segment(2)));
      else if (name.starts_with("wu")) bindings_[index].push_back(set.up.at(segment(2)));
      else if (name.starts_with("wd")) bindings_[index].push_back(set.down.at(segment(2)));
      else if (name.starts_with("x")) bindings_[index].push_back(inputs_.at(segment(1)));
      else throw std::logic_error("unknown ANE FFN program input " + name);
    }
  }
  event_ = backend_.newSharedEvent();

  // Each row's shared int8 scale over the ANE's share of its inputs.
  metal::CommandGraph graph;
  for (uint32_t layer = 0; layer < layers_.size(); ++layer) {
    const SwiGluProjections &source = layers_[layer].source;
    const auto add = [&](const Projection &projection, uint32_t part, AneFfnWeightParams params, uint32_t rows,
                         uint32_t block) {
      const AffineWeights &weights = projection.affine();
      graph.add("ane_ffn_row_scale_" + std::to_string(block),
                {weights.weights, weights.scales, weights.biases, rowScales(layer, part), signs_}, params,
                {rows / 8, 1, 1});
    };
    add(*source.gate, 0, {hidden_ / kQuantGroup, gpuChannels_, 0, hidden_, 0, 0}, aneChannels_, kBlock);
    add(*source.up, 1, {hidden_ / kQuantGroup, gpuChannels_, 0, hidden_, 0, 0}, aneChannels_, kBlock);
    add(*source.down, 2, {groups, 0, gpuChannels_, aneChannels_, 0, 0}, hidden_, kIntermediateBlock);
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
                       const ane::Surface &scale, AneFfnWeightParams params, uint32_t rows, uint32_t block) {
    const AffineWeights &weights = projection.affine();
    params.stride = output.strideBytes;
    params.scale_stride = scale.strideBytes / 2;
    graph.add("ane_ffn_weights_" + std::to_string(block),
              {weights.weights, weights.scales, weights.biases, scales, output.buffer, scale.buffer, signs_}, params,
              {rows / 8, params.width / block, 1});
  };
  for (uint32_t k = 0; k < target.gate.size(); ++k) {
    const AneFfnWeightParams params{hidden_ / kQuantGroup, gpuChannels_, k * kSegment, kSegment, 0, 0};
    add(*source.gate, rowScales(layer, 0), target.gate[k], target.gateScale, params, aneChannels_, kBlock);
    add(*source.up, rowScales(layer, 1), target.up[k], target.upScale, params, aneChannels_, kBlock);
  }
  uint32_t begin = gpuChannels_;
  for (size_t i = 0; i < downSegments_.size(); ++i) {
    add(*source.down, rowScales(layer, 2), target.down[i], target.downScale,
        {intermediate_ / kQuantGroup, 0, begin, downSegments_[i], 0, 0}, hidden_, kIntermediateBlock);
    begin += downSegments_[i];
  }
}

void AneFfn::add(metal::CommandGraph &graph, uint32_t layer, metal::MetalBuffer normalized, metal::MetalBuffer sums,
                 metal::MetalBuffer gateScratch, metal::MetalBuffer intermediate, metal::MetalBuffer downSums,
                 metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows, LinearScratch scratch) {
  if (!rows || rows > kRows) throw std::invalid_argument("ANE FFN chunk exceeds its rows");
  const Layer &current = layers_.at(layer);
  const uint32_t set = layer & 1, tiles = (rows + 31) / 32;
  // Each command stages layer 0's weights, then each layer the next one's.
  if (!layer) addWeights(graph, 0, 0);
  graph.add("ane_ffn_rotate", {normalized, signs_, rotated_, tokenScale_.buffer}, AneFfnRotateParams{hidden_},
            {rows, 1, 1}, {256, 1, 1});
  for (uint32_t k = 0; k < inputs_.size(); ++k)
    graph.add("ane_ffn_pack", {rotated_, inputs_[k].buffer},
              AneFfnPackParams{hidden_, k * kSegment, inputs_[k].strideBytes}, {tiles, kSegment / 32, 1}, {32, 8, 1});
  const uint64_t wait = ++value_;
  graph.signal(event_, wait);
  linear_.addPrefill(graph, normalized, current.gate, gateScratch, sums, rows, scratch);
  linear_.addPrefillUpWithGate(graph, normalized, current.up, gateScratch, intermediate, sums, downSums, rows,
                               scratch);
  linear_.addPrefillResidual(graph, intermediate, current.down, residual, output, downSums, rows, scratch);
  if (layer + 1 < layers_.size()) addWeights(graph, layer + 1, set ^ 1);
  const uint64_t signal = ++value_;
  graph.wait(event_, signal);
  graph.add("ane_ffn_join", {output, partial_.buffer}, AneFfnJoinParams{hidden_, partial_.strideBytes / 2, rows},
            {tiles, hidden_ / 32, 1}, {32, 8, 1});
  jobs_.push_back({set, wait, signal});
}

void AneFfn::submit() {
  std::vector<Job> jobs = std::exchange(jobs_, {});
  for (const Job &job : jobs) {
    try {
      program_->enqueue(bindings_[job.set], partial_, event_, job.wait, job.signal,
                        [evaluations = evaluations_](bool success) {
                          std::lock_guard lock(evaluations->mutex);
                          ++evaluations->completed;
                          evaluations->failed |= !success;
                          evaluations->changed.notify_all();
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
  std::lock_guard lock(evaluations_->mutex);
  if (std::exchange(evaluations_->failed, false)) throw std::runtime_error("ANE FFN evaluation failed");
}

bool AneFfn::wait(bool release) {
  std::unique_lock lock(evaluations_->mutex);
  const uint64_t first = queuedCount_ - queued_.size();
  for (size_t index = 0; index < queued_.size(); ++index) {
    const uint64_t sequence = first + index + 1;
    if (release) ane::Program::release(event_, queued_[index].wait);
    if (!evaluations_->changed.wait_for(lock, kCompletionTimeout,
                                        [&] { return evaluations_->completed >= sequence; }))
      return false;
  }
  queued_.clear();
  return true;
}

} // namespace splash::ops
