#include "ops/AneFfn.hpp"

#include "Checked.hpp"
#include "StderrLine.hpp"
#include "metal/abi/AneFfn.h"
#include "metal/abi/Linear.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {
namespace {

using Element = ane::Surface::Element;

constexpr uint32_t kQuantGroup = 64;
static_assert(AneFfn::kChannelUnit % SPLASH_AFFINE_TILE_ROWS == 0 &&
                  AneFfn::kChannelUnit % ANE_FFN_INTERMEDIATE_BLOCK == 0 &&
                  AneFfn::kChannelUnit % ANE_FFN_WEIGHT_ROWS == 0,
              "a channel unit is whole plane tiles, rotation blocks and weight-kernel threadgroups");
static_assert(AneFfn::kMaximumRows % ANE_FFN_TILE == 0, "the surfaces hold whole packing tiles");
// How long an ANE evaluation may run once the GPU has raised its `ready`
// before the split stops (ane::Handoff). An evaluation takes 0.3-5 ms here.
// Metal fails a command buffer whose event wait stays unmet for 5 s; the GPU
// waits on an evaluation only after its own part of the layer, so the
// release reaches a wait at most 2 s old.
constexpr auto kHandoffBound = std::chrono::seconds(2);
// How long the service may take to compile and load the program: it compiles
// the Omni program in a few seconds, and loads a compiled one in well under
// one, so only a service that stopped answering runs out of it.
constexpr auto kProgramLimit = std::chrono::seconds(120);
// The least peak the ANE quantizes a token's rotated intermediate row
// against: a quieter row takes coarser codes, each within this over 254 of
// the token's input scale.
constexpr double kIntermediateFloor = 0x1p-9;
static_assert(ANE_FFN_INT8_PEAK / kIntermediateFloor <= 65504.0, "the int8 peak over the floor fits fp16");
// What verify() fills the ANE's output with before a check: an fp16 NaN,
// which the join flags where the ANE leaves it.
constexpr uint16_t kHalfNaN = 0x7e00;
// The most the split's output may differ from the GPU's alone, as the RMS of
// their difference relative to the RMS of what the ANE adds over the GPU's
// part (AneFfn::verify): int8 leaves 2.5-2.8% on Splash's models and
// about 1% on the Omni decoders' layers; a function that computes other
// values differs by about all of the ANE's part.
constexpr double kMaximumAneError = 0.10;

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
std::array<float, ANE_FFN_SIGNS> rotationSigns() {
  std::mt19937 generator(20260930);
  std::array<float, ANE_FFN_SIGNS> signs{};
  for (float &sign : signs) sign = (generator() & 1) ? -1.0f : 1.0f;
  return signs;
}

// The program's constant: the rotation of each of the ANE's intermediate
// channels as the [channels, ANE_FFN_INTERMEDIATE_BLOCK, 1, 1] weight of a
// grouped 1x1 convolution.
std::vector<uint8_t> rotationBlob(uint32_t channels, const std::array<float, ANE_FFN_SIGNS> &signs) {
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

const char *AneFfn::unsupported(std::span<const SwiGluProjections> layers) {
  if (layers.empty()) return "ANE FFN split has no layers";
  for (const SwiGluProjections &layer : layers)
    if (!layer.gate || !layer.up || !layer.down) return "ANE FFN split needs every layer's gate, up and down";
  const uint32_t hidden = layers.front().gate->inputSize, intermediate = layers.front().gate->outputSize;
  if (hidden % ANE_FFN_INPUT_BLOCK || hidden % ANE_FFN_TILE)
    return "ANE FFN split needs a hidden size ane_ffn_rotate takes";
  if (intermediate % kChannelUnit || intermediate < 2 * kChannelUnit)
    return "ANE FFN split needs two or more whole units of intermediate channels";
  for (const SwiGluProjections &layer : layers) {
    for (const Projection *projection : {layer.gate, layer.up, layer.down})
      if (!projection->takesPlaneViews()) return "ANE FFN split needs affine Q4 projections of their own planes";
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

AneFfn::Shape AneFfn::shapeOf(std::span<const SwiGluProjections> layers, uint32_t aneUnits) {
  const uint32_t total = units(layers);
  if (!aneUnits || aneUnits >= total) throw std::invalid_argument("ANE FFN split leaves the GPU or the ANE no channels");
  Shape shape;
  shape.hidden = layers.front().gate->inputSize;
  shape.intermediate = layers.front().gate->outputSize;
  shape.layers = static_cast<uint32_t>(layers.size());
  shape.ane = aneUnits * kChannelUnit;
  shape.gpu = shape.intermediate - shape.ane;
  return shape;
}

metal::MetalBuffer AneFfn::allocate(uint64_t bytes, const char *label) {
  metal::MetalBuffer buffer = backend_.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
  allocatedBytes_ += buffer.allocatedBytes();
  return buffer;
}

ane::Surface AneFfn::surface(uint32_t rows, uint32_t width, Element element) {
  ane::Surface result = ane::Surface::create(backend_, rows, width, element);
  allocatedBytes_ += result.buffer.allocatedBytes();
  return result;
}

// x: the rotated input rows, channel-major in int8, and tx their per-token
// scales; wg and wu: gate's and up's rotated int8 rows of the ANE's channels,
// and sg and su their per-row scales; wd: down's rotated int8 rows over the
// ANE's channels, and sd their per-row scales. The weights bind each layer's
// own surfaces.
std::vector<AneFfn::Input> AneFfn::programInputs() const {
  const auto each = [&](ane::Surface Weights::*member) {
    std::vector<const ane::Surface *> surfaces;
    for (const Weights &layer : weights_) surfaces.push_back(&(layer.*member));
    return surfaces;
  };
  return {{"x", {&inputs_}, shape_.hidden, 0},
          {"tx", {&tokenScale_}, 1, 0},
          {"wg", each(&Weights::gate), shape_.ane, shape_.hidden},
          {"wu", each(&Weights::up), shape_.ane, shape_.hidden},
          {"sg", each(&Weights::gateScale), shape_.ane, 1},
          {"su", each(&Weights::upScale), shape_.ane, 1},
          {"wd", each(&Weights::down), shape_.hidden, shape_.ane},
          {"sd", each(&Weights::downScale), shape_.hidden, 1}};
}

// The ANE's share of the FFN over `rows` rows. Every int8 value is
// dequantized by 1 / ANE_FFN_INT8_UNIT against fp16 overflow, which the
// scales carry back. Gate's output takes the token's input scale before the
// SiLU; up's does not, so that the intermediate rows stay small, and the
// join multiplies it in. The intermediate rows are rotated and quantized per
// token here. The output holds down's rows before their tokens' scales, and
// in a last row each token's intermediate scale, which the join multiplies
// with the input's in fp32 (ane_ffn_join): the output at full scale could
// overflow fp16. The function reads and writes the leading `rows` of each
// row of the chunk's surfaces.
std::string AneFfn::function(std::span<const Input> inputs, uint32_t rows) const {
  const uint32_t channels = shape_.ane, hidden = shape_.hidden;
  const std::string unit = fp16(1.0 / ANE_FFN_INT8_UNIT);
  std::string parameters, body;
  const auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
  const auto f16 = [&](const std::string &name, uint64_t height, uint64_t width, const std::string &expression) {
    line(tensor("fp16", height, width) + " " + name + " = " + expression);
  };
  for (const Input &input : inputs) {
    const uint32_t width = input.width ? input.width : rows;
    const ane::Surface &surface = *input.surfaces.front();
    parameters += (parameters.empty() ? "" : ", ") + surface.bufferType(input.rows, width) + " " + input.name;
    line(tensor(ane::Surface::milType(surface.element), input.rows, width) + " " + input.name +
         "_t = tensor_buffer_to_tensor<ios17>(input = " + input.name + ")");
  }
  const auto dequantized = [&](const std::string &name, uint64_t height, uint64_t width) {
    f16(name + "_d", height, width, "dequantize(input = " + name + "_t, scale = " + unit + ")");
  };
  const auto matmul = [&](const std::string &name, const std::string &weights, const std::string &values,
                          uint64_t height) {
    line(tensor("fp16", height, rows) + " " + name +
         " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + weights + ", y = " + values + ")");
  };
  dequantized("x", hidden, rows);
  dequantized("wg", channels, hidden);
  dequantized("wu", channels, hidden);
  dequantized("wd", hidden, channels);
  matmul("gm", "wg_d", "x_d", channels);
  matmul("um", "wu_d", "x_d", channels);
  f16("gs", channels, rows, "mul(x = gm, y = sg_t)");
  f16("us", channels, rows, "mul(x = um, y = su_t)");
  // silu(g) = g/2 (1 + tanh(g/2)): through the ANE's sigmoid, Splash's
  // layers whose gate pre-activations are about 0.1 differed from the GPU
  // alone by 6.5% RMS, against 1.6% at about 1; through its tanh both
  // differ by 1.6%.
  f16("gt", channels, rows, "mul(x = gs, y = tx_t)");
  f16("gh", channels, rows, "mul(x = gt, y = " + fp16(0.5) + ")");
  f16("th", channels, rows, "tanh(x = gh)");
  f16("tp", channels, rows, "add(x = th, y = " + fp16(1.0) + ")");
  f16("silu", channels, rows, "mul(x = gh, y = tp)");
  f16("h", channels, rows, "mul(x = silu, y = us)");
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
  matmul("dm", "wd_d", "hd", hidden);
  f16("ds", hidden, rows, "mul(x = dm, y = sd_t)");
  f16("yt", hidden + 1, rows, "concat(axis = int32(2), interleave = bool(false), values = (ds, hscale))");
  line(partial_.bufferType(hidden + 1, rows) +
       " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
       "strides = tensor<int64, [4]>(" + partial_.strides(hidden + 1) + "))");
  return "    func " + functionName(rows) + "<ios18>(" + parameters + ") {\n" + body + "    } -> (y);\n";
}

std::string AneFfn::program(std::span<const Input> inputs, std::span<const uint32_t> rows) const {
  std::string text = "program(1.3)\n{\n";
  for (const uint32_t count : rows) text += function(inputs, count);
  return text + "}\n";
}

AneFfn::AneFfn(metal::MetalBackend &backend, std::span<const SwiGluProjections> layers, uint32_t aneUnits,
               std::span<const uint32_t> functionRows, std::function<bool()> interrupted)
    : backend_(backend), linear_(backend.capabilities()), shape_(shapeOf(layers, aneUnits)),
      handoff_(backend, kHandoffBound) {
  if (functionRows.empty() || !std::ranges::is_sorted(functionRows) ||
      std::ranges::adjacent_find(functionRows) != functionRows.end() || functionRows.front() < ANE_FFN_TILE ||
      functionRows.back() > kMaximumRows ||
      std::ranges::any_of(functionRows, [](uint32_t rows) { return rows % ANE_FFN_TILE != 0; }))
    throw std::invalid_argument("ANE FFN functions take ascending whole packing tiles of rows up to the budget");
  minimumRows_ = functionRows.front();

  signs_ = allocate(ANE_FFN_SIGNS * sizeof(float), "ane ffn signs");
  rotated_ = allocate(uint64_t{kMaximumRows} * shape_.hidden * sizeof(_Float16), "ane ffn rotated input");
  status_ = allocate(sizeof(uint32_t), "ane ffn status");
  inputs_ = surface(shape_.hidden, kMaximumRows, Element::Int8);
  tokenScale_ = surface(1, kMaximumRows, Element::Float16);
  partial_ = surface(shape_.hidden + 1, kMaximumRows, Element::Float16);
  for (uint32_t layer = 0; layer < shape_.layers; ++layer)
    weights_.push_back({surface(shape_.ane, shape_.hidden, Element::Int8),
                        surface(shape_.ane, shape_.hidden, Element::Int8),
                        surface(shape_.hidden, shape_.ane, Element::Int8), surface(shape_.ane, 1, Element::Float16),
                        surface(shape_.ane, 1, Element::Float16), surface(shape_.hidden, 1, Element::Float16)});
  const std::array<float, ANE_FFN_SIGNS> signs = rotationSigns();
  std::memcpy(signs_.contents(), signs.data(), sizeof signs);
  *static_cast<uint32_t *>(status_.contents()) = 0;

  for (const SwiGluProjections &source : layers)
    layers_.push_back({source.gate->leadingRows(backend_, shape_.gpu), source.up->leadingRows(backend_, shape_.gpu),
                       source.down->leadingInputs(shape_.gpu)});
  stageWeights(layers);

  const std::vector<Input> inputs = programInputs();
  program_ = std::make_unique<ane::Program>(program(inputs, functionRows), rotationBlob(shape_.ane, signs),
                                            ane::Program::Limits{kProgramLimit, std::move(interrupted)});
  for (const uint32_t rows : functionRows) {
    Evaluation &evaluation = evaluations_.emplace_back();
    evaluation.rows = rows;
    const uint32_t procedure = program_->procedure(functionName(rows));
    for (uint32_t layer = 0; layer < shape_.layers; ++layer) {
      std::vector<ane::Surface> surfaces;
      for (const std::string &name : program_->inputs(procedure)) {
        const auto input = std::ranges::find(inputs, name, &Input::name);
        if (input == inputs.end()) throw std::logic_error("ANE FFN program has an unknown input " + name);
        surfaces.push_back(*input->surfaces[input->surfaces.size() == 1 ? 0 : layer]);
      }
      evaluation.bindings.push_back(program_->bind(procedure, surfaces, partial_));
    }
  }
}

// Each row's shared int8 scale over the ANE's share of its inputs, then the
// rotated int8 rows, of every layer, in one command.
void AneFfn::stageWeights(std::span<const SwiGluProjections> layers) {
  const uint64_t perLayer = 2 * uint64_t{shape_.ane} + shape_.hidden;
  const metal::MetalBuffer rowScales =
      backend_.allocateBuffer(perLayer * shape_.layers * sizeof(_Float16), metal::BufferStorage::Shared,
                              "ane ffn row scales");
  metal::CommandGraph graph;
  for (uint32_t layer = 0; layer < shape_.layers; ++layer) {
    const Weights &target = weights_[layer];
    uint64_t offset = layer * perLayer;
    const auto add = [&](const Projection &projection, const ane::Surface &output, const ane::Surface &scale,
                         uint32_t row, uint32_t input, uint32_t width, uint32_t rows, Rotation rotation) {
      const AffineWeights &planes = projection.affine();
      const uint32_t groups = projection.inputSize / kQuantGroup;
      const uint32_t scaleStride = scale.strideBytes / uint32_t{sizeof(_Float16)};
      requireBytes(output.buffer, rowBytes(rows, output.strideBytes, width, 1), "ANE FFN weight surface");
      requireBytes(scale.buffer, rowBytes(rows, scaleStride, 1, sizeof(_Float16)), "ANE FFN weight scale surface");
      const metal::MetalBuffer scales = backend_.view(rowScales, offset * sizeof(_Float16), rows * sizeof(_Float16));
      offset += rows;
      const AneFfnWeightParams params{groups, row, input, width, output.strideBytes, scaleStride};
      graph.add(kernel("ane_ffn_row_scale", rotation), {planes.weights, planes.scales, planes.biases, scales, signs_},
                params, {rows / ANE_FFN_WEIGHT_ROWS, 1, 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
      graph.add(kernel("ane_ffn_weights", rotation),
                {planes.weights, planes.scales, planes.biases, scales, output.buffer, scale.buffer, signs_}, params,
                {rows / ANE_FFN_WEIGHT_ROWS, width / blockOf(rotation), 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
    };
    const SwiGluProjections &source = layers[layer];
    add(*source.gate, target.gate, target.gateScale, shape_.gpu, 0, shape_.hidden, shape_.ane, Rotation::Inputs);
    add(*source.up, target.up, target.upScale, shape_.gpu, 0, shape_.hidden, shape_.ane, Rotation::Inputs);
    add(*source.down, target.down, target.downScale, 0, shape_.gpu, shape_.ane, shape_.hidden,
        Rotation::Intermediate);
  }
  static_cast<void>(backend_.submitCommandAsync(graph.dispatches()).wait());
}

double AneFfn::share() const noexcept { return static_cast<double>(shape_.ane) / shape_.intermediate; }

void AneFfn::setMinimumRows(uint32_t rows) {
  if (!rows || rows > maximumRows())
    throw std::invalid_argument("ANE FFN split has no function of " + std::to_string(rows) + " rows");
  minimumRows_ = rows;
}

bool AneFfn::splits(uint32_t rows) const {
  return rows >= minimumRows_ && rows <= maximumRows() && !handoff_.retired();
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
  encode(graph, layer, ffn, std::move(residual), std::move(output), rows, Parts::Both);
}

void AneFfn::encode(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn,
                    metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows, Parts parts) {
  const auto found =
      std::ranges::find_if(evaluations_, [&](const Evaluation &evaluation) { return evaluation.rows >= rows; });
  if (!rows || found == evaluations_.end()) throw std::invalid_argument("ANE FFN has no function of the chunk's rows");
  if (layer != nextLayer_ || layer >= layers_.size())
    throw std::logic_error("ANE FFN split adds layer " + std::to_string(layer) + ", not its next layer " +
                           std::to_string(nextLayer_) + " (begin() starts each command at layer 0)");
  const uint64_t hiddenBytes = uint64_t{rows} * shape_.hidden * sizeof(uint16_t);
  requireBytes(ffn.normalized, hiddenBytes, "ANE FFN input");
  requireBytes(output, hiddenBytes, "ANE FFN output");
  const Layer &current = layers_[layer];
  const uint32_t tiles = (rows + ANE_FFN_TILE - 1) / ANE_FFN_TILE;
  const bool ane = parts == Parts::Both;
  graph.add("ane_ffn_rotate", {ffn.normalized, signs_, rotated_, tokenScale_.buffer}, AneFfnRotateParams{shape_.hidden},
            {rows, 1, 1}, {ANE_FFN_ROTATE_THREADS, 1, 1});
  graph.add("ane_ffn_pack", {rotated_, inputs_.buffer}, AneFfnPackParams{shape_.hidden, inputs_.strideBytes},
            {tiles, shape_.hidden / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  const uint64_t ready = ane ? handoff_.next() : 0;
  if (ane) graph.signal(handoff_.event(), ready);
  linear_.addPrefillSwiGlu(graph, {&current.gate, &current.up, &current.down}, ffn, residual, output, rows);
  const uint64_t done = ane ? handoff_.next() : 0;
  if (ane) graph.wait(handoff_.event(), done);
  graph.add("ane_ffn_join", {output, partial_.buffer, tokenScale_.buffer, status_},
            AneFfnJoinParams{shape_.hidden, partial_.strideBytes / uint32_t{sizeof(_Float16)}, rows},
            {tiles, shape_.hidden / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  if (ane) jobs_.push_back({static_cast<uint32_t>(found - evaluations_.begin()), layer, ready, done});
  ++nextLayer_;
}

metal::CommandTicket AneFfn::commit(const metal::CommandGraph &graph, metal::CommandCompletion completion) {
  const std::vector<Job> jobs = std::exchange(jobs_, {});
  for (const Job &job : jobs) {
    const ane::Program::Binding &binding = evaluations_.at(job.evaluation).bindings.at(job.layer);
    handoff_.queue(job.ready, job.done,
                   [&](const metal::SharedEvent &event, uint64_t wait, uint64_t signal, ane::Handoff::Report report) {
                     program_->enqueue(binding, event, wait, signal, std::move(report));
                   });
  }
  unfinished_ = !jobs.empty();
  committedEvaluations_ = static_cast<uint32_t>(jobs.size());
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
  if (committedEvaluations_) lastAneMilliseconds_ = std::chrono::duration<double, std::milli>(*ran).count();
  return true;
}

std::optional<AwakeClock::duration> AneFfn::completed() {
  unfinished_ = false;
  std::optional<AwakeClock::duration> ran = handoff_.finish();
  if (std::exchange(*static_cast<uint32_t *>(status_.contents()), 0u) && ran) {
    handoff_.retire("the Neural Engine's output or its scales were not finite");
    ran.reset();
  }
  return ran;
}

void AneFfn::warnStopped() {
  if (std::exchange(warned_, true)) return;
  logWarning("Neural Engine FFN split stopped (", handoff_.reason(),
             "); the GPU runs the prefill FFN alone until the engine restarts.");
}

double AneFfn::verify(std::span<const SwiGluProjections> layers, const PrefillFfnBuffers &ffn,
                      const std::array<metal::MetalBuffer, 2> &hidden) {
  if (layers.size() != layers_.size() || layers.front().gate->outputSize != shape_.intermediate)
    throw std::invalid_argument("ANE FFN split verified on other layers than its own");
  if (handoff_.retired()) throw std::runtime_error("ANE FFN split stopped (" + handoff_.reason() + ")");
  // The same values of a normalized row's magnitude each time.
  auto *bits = static_cast<uint16_t *>(ffn.normalized.contents());
  if (!bits) throw std::invalid_argument("ANE FFN check writes rows the CPU cannot");
  std::mt19937 random(20261001);
  std::uniform_real_distribution<float> uniform(-2.0f, 2.0f);
  for (uint64_t i = 0; i < uint64_t{kMaximumRows} * shape_.hidden; ++i)
    bits[i] = static_cast<uint16_t>(std::bit_cast<uint32_t>(uniform(random)) >> 16);
  // Layer l adds its FFN of the normalized rows to hidden[l & 1] into
  // hidden[(l & 1) ^ 1], from zeros, on the GPU alone, as the split's GPU part
  // alone over a partial output of zeros, or split. The affine Q4 kernels
  // read the rows' sums, which each command computes first.
  enum class Way : uint8_t { Gpu, GpuPart, Split };
  const uint32_t count = std::min<uint32_t>(2, shape_.layers);
  const auto forward = [&](Way way, uint32_t rows) {
    std::memset(hidden[0].contents(), 0, hidden[0].sizeBytes());
    if (way != Way::Gpu) {
      std::fill_n(static_cast<uint16_t *>(partial_.buffer.contents()), partial_.buffer.sizeBytes() / sizeof(uint16_t),
                  way == Way::Split ? kHalfNaN : 0);
      static_cast<void>(begin());
    }
    metal::CommandGraph graph;
    linear_.addPrefillSums(graph, ffn.normalized, ffn.sums, *layers.front().gate, rows);
    for (uint32_t layer = 0; layer < count; ++layer) {
      const metal::MetalBuffer &residual = hidden[layer & 1], &output = hidden[(layer & 1) ^ 1];
      if (way == Way::Gpu)
        linear_.addPrefillSwiGlu(graph, layers[layer], ffn, residual, output, rows);
      else
        encode(graph, layer, ffn, residual, output, rows, way == Way::Split ? Parts::Both : Parts::Gpu);
    }
    if (way == Way::Split) {
      static_cast<void>(commit(graph).wait());
      if (!completed())
        throw std::runtime_error("ANE FFN split of " + std::to_string(rows) + " rows failed (" + reason() + ")");
    } else {
      static_cast<void>(backend_.submitCommandAsync(graph.command()).wait());
      *static_cast<uint32_t *>(status_.contents()) = 0;
    }
    return bf16Values(hidden[count & 1], uint64_t{rows} * shape_.hidden);
  };
  // What the ANE adds over the GPU's part, the GPU alone's output less the
  // GPU part's, against the split's difference from the GPU alone, over the
  // split's rows.
  const uint32_t largest = evaluations_.back().rows;
  const std::vector<float> gpu = forward(Way::Gpu, largest), part = forward(Way::GpuPart, largest);
  double worst = 0.0;
  for (const Evaluation &evaluation : evaluations_) {
    const uint32_t rows = evaluation.rows;
    const std::vector<float> split = forward(Way::Split, rows);
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
