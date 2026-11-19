#include "ops/OmniCodecAne.hpp"

#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace splash::ops::omni {
namespace {

using Element = ane::Surface::Element;

uint64_t ceilDiv(uint64_t value, uint64_t divisor) noexcept { return (value + divisor - 1) / divisor; }

std::string fp16(double value) {
  char text[40];
  std::snprintf(text, sizeof text, "fp16(%a)", value);
  return text;
}

std::string dims(std::initializer_list<uint64_t> values) {
  std::string text = "[";
  for (const uint64_t value : values) text += (text.size() > 1 ? ", " : "") + std::to_string(value);
  return text + "]";
}

// [1, channels, 1, columns] fp16, the convolutions' layout.
std::string tensor(uint64_t channels, uint64_t columns) { return "tensor<fp16, " + dims({1, channels, 1, columns}) + ">"; }

const float *floats(const metal::MetalBuffer &buffer, uint64_t count, const char *what) {
  requireBytes(buffer, count * 4, what);
  const auto *values = static_cast<const float *>(buffer.contents());
  if (!values) throw std::invalid_argument(std::string(what) + " is not in shared memory");
  return values;
}

// A convolution's weight as MIL's conv reads it, [out][in][1][taps], from
// the GEMM's [taps][in][out] fp32 layout.
std::vector<_Float16> convWeight(const CodecConv &conv) {
  const uint32_t in = conv.inputChannels, out = conv.outputChannels, taps = conv.taps;
  const float *packed = floats(conv.weight, uint64_t{taps} * in * out, "codec conv weight");
  std::vector<_Float16> weight(uint64_t{out} * in * taps);
  for (uint32_t k = 0; k < taps; ++k)
    for (uint32_t ci = 0; ci < in; ++ci)
      for (uint32_t co = 0; co < out; ++co)
        weight[(uint64_t{co} * in + ci) * taps + k] = static_cast<_Float16>(packed[(uint64_t{k} * in + ci) * out + co]);
  return weight;
}

std::vector<_Float16> halves(const metal::MetalBuffer &buffer, uint64_t count, const char *what) {
  const float *values = floats(buffer, count, what);
  std::vector<_Float16> result(count);
  for (uint64_t i = 0; i < count; ++i) result[i] = static_cast<_Float16>(values[i]);
  return result;
}

} // namespace

// The stack's layers with their constants' offsets in the weight blob.
struct CodecAneStack::Layers final {
  struct Conv final {
    std::string name;
    uint64_t weight = 0, bias = 0;
    uint32_t in = 0, out = 0, taps = 1, dilation = 1;
    uint32_t history = 0; // its history's index, for one with taps
  };
  struct Snake final {
    std::string name;
    uint64_t alpha = 0, beta = 0;
    uint32_t channels = 0;
  };
  struct Unit final {
    Snake act1, act2;
    Conv conv1, conv2;
  };
  struct Block final {
    Snake snake;
    Conv upsample; // phase-stacked: out = rate * channels over 2 taps
    uint32_t rate = 1, channels = 0;
    std::vector<Unit> units;
  };
  ane::Constants constants;
  uint32_t latent = 0, rowsPerFrame = 1, samplesPerFrame = 0;
  Conv initial;
  std::vector<Block> blocks;
  Snake outputSnake;
  Conv output;
  // (channels, length) of each history, in the stream's stackHistories() order.
  std::vector<std::pair<uint32_t, uint32_t>> histories;

  // A convolution; one with taps keeps the next history, of its input rows.
  Conv conv(const CodecConv &source, const std::string &name) {
    if (!source.bias) throw std::invalid_argument("codec ANE stack: " + name + " has no bias");
    Conv c{name, constants.add(convWeight(source)), 0, source.inputChannels, source.outputChannels, source.taps,
           source.dilation, 0};
    c.bias = constants.add(halves(source.bias, source.outputChannels, "codec conv bias"));
    if (c.taps > 1) {
      c.history = static_cast<uint32_t>(histories.size());
      histories.emplace_back(c.in, (c.taps - 1) * c.dilation);
    }
    return c;
  }
  Snake snake(const CodecSnake &source, const std::string &name) {
    return {name, constants.add(halves(source.alpha, source.channels, "codec snake alpha")),
            constants.add(halves(source.beta, source.channels, "codec snake beta")), source.channels};
  }
};

struct CodecAneStack::Lane final {
  ane::Surface input, samples;
  std::array<std::vector<ane::Surface>, 2> histories;
  uint32_t phase = 0;
  // By function, then phase.
  std::vector<ane::Program::Binding> bindings;
};

CodecAneStack::CodecAneStack(metal::MetalBackend &backend, const CodecDecoderWeights &weights,
                             std::vector<uint32_t> functionFrames, uint32_t lanes, ane::Program::Limits limits)
    : backend_(backend), functionFrames_(std::move(functionFrames)), layers_(std::make_unique<Layers>()) {
  const CodecDims &d = weights.dims;
  if (functionFrames_.empty() || !std::ranges::is_sorted(functionFrames_) || !functionFrames_.front() ||
      std::ranges::adjacent_find(functionFrames_) != functionFrames_.end())
    throw std::invalid_argument("codec ANE stack functions take ascending frame counts");
  if (!lanes) throw std::invalid_argument("codec ANE stack takes at least one lane");
  if (weights.blocks.size() != d.upsampleRates.size()) throw std::invalid_argument("codec weights do not match their dims");

  // The layers and their constants.
  Layers &l = *layers_;
  l.latent = d.latent;
  for (const uint32_t ratio : d.upsamplingRatios) l.rowsPerFrame *= ratio;
  l.samplesPerFrame = d.samplesPerFrame();
  const auto requireHistory = [](const CodecConv &conv, const char *what) {
    if (conv.taps < 2) throw std::invalid_argument(std::string("codec ANE stack: ") + what + " is not causal");
  };
  requireHistory(weights.initialConv, "the initial convolution");
  l.initial = l.conv(weights.initialConv, "initial");
  for (size_t b = 0; b < weights.blocks.size(); ++b) {
    const CodecDecoderBlock &source = weights.blocks[b];
    const std::string name = "b" + std::to_string(b);
    Layers::Block block;
    block.rate = source.rate;
    block.snake = l.snake(source.snake, name + "s");
    if (source.upsample.taps != 2 || source.upsample.outputChannels % source.rate)
      throw std::invalid_argument("codec ANE stack: a block's transposed convolution is not two taps of phases");
    block.channels = source.upsample.outputChannels / source.rate;
    block.upsample = l.conv(source.upsample, name + "u");
    for (size_t u = 0; u < source.units.size(); ++u) {
      const CodecResidualUnit &unit = source.units[u];
      const std::string unitName = name + "r" + std::to_string(u);
      requireHistory(unit.conv1, "a residual unit's first convolution");
      if (unit.conv2.taps != 1) throw std::invalid_argument("codec ANE stack: a residual unit's second convolution has taps");
      Layers::Unit layer{l.snake(unit.act1, unitName + "a"), l.snake(unit.act2, unitName + "b"),
                         l.conv(unit.conv1, unitName + "c"), l.conv(unit.conv2, unitName + "d")};
      block.units.push_back(std::move(layer));
    }
    l.blocks.push_back(std::move(block));
  }
  l.outputSnake = l.snake(weights.outputSnake, "os");
  requireHistory(weights.outputConv, "the output convolution");
  l.output = l.conv(weights.outputConv, "out");

  // Surfaces of every lane: the input for the largest function, the samples
  // in the pieces of any function, and two halves of the histories.
  const uint32_t largest = functionFrames_.back();
  uint32_t pieces = 1, pieceSamples = 1;
  for (const uint32_t frames : functionFrames_) {
    const Plan p = plan(frames);
    pieces = std::max(pieces, p.pieces);
    pieceSamples = std::max(pieceSamples, p.pieceSamples);
  }
  for (uint32_t i = 0; i < lanes; ++i) {
    auto lane = std::make_unique<Lane>();
    const auto surface = [&](uint32_t rows, uint32_t width) {
      ane::Surface created = ane::Surface::create(backend_, rows, width, Element::Float16);
      bytes_ += created.buffer.sizeBytes();
      return created;
    };
    lane->input = surface(l.latent, largest * l.rowsPerFrame);
    lane->samples = surface(pieces, pieceSamples);
    for (auto &half : lane->histories)
      for (const auto &[channels, length] : l.histories) half.push_back(surface(channels, length));
    lanes_.push_back(std::move(lane));
  }
  for (uint32_t i = 0; i < lanes; ++i) reset(i);

  source_ = "program(1.3)\n{\n";
  for (const uint32_t frames : functionFrames_) source_ += function(frames);
  source_ += "}\n";
  program_ = std::make_unique<ane::Program>(source_, l.constants.blob(), std::move(limits));

  for (const auto &lane : lanes_)
    for (const uint32_t frames : functionFrames_) {
      const uint32_t procedure = program_->procedure("f" + std::to_string(frames));
      for (uint32_t phase = 0; phase < 2; ++phase) {
        const auto find = [](const std::vector<std::string> &names, const std::string &name) {
          const auto found = std::ranges::find(names, name);
          if (found == names.end()) throw std::runtime_error("codec ANE program lacks " + name);
          return static_cast<size_t>(found - names.begin());
        };
        const std::vector<std::string> &inputNames = program_->inputs(procedure);
        const std::vector<std::string> &outputNames = program_->outputs(procedure);
        std::vector<ane::Surface> inputs(inputNames.size()), outputs(outputNames.size());
        inputs.at(find(inputNames, "x")) = lane->input;
        outputs.at(find(outputNames, "y")) = lane->samples;
        for (size_t h = 0; h < l.histories.size(); ++h) {
          inputs.at(find(inputNames, "h" + std::to_string(h))) = lane->histories[phase][h];
          outputs.at(find(outputNames, "n" + std::to_string(h))) = lane->histories[phase ^ 1][h];
        }
        lane->bindings.push_back(program_->bind(procedure, inputs, outputs));
      }
    }
}

CodecAneStack::~CodecAneStack() = default;

// The function of `frames` frames: x [latent][frames * 4] and the histories
// h0.. in; the samples y and the new histories n0.. out. The blocks from the
// plan's split on run per piece of the split block's input columns, each
// piece's convolutions reading the histories the piece before it left, and
// the pieces' samples are the rows of y.
std::string CodecAneStack::function(uint32_t frames) const {
  const Layers &l = *layers_;
  const Lane &shape = *lanes_.front();
  const Plan plan = this->plan(frames);
  std::string parameters, body, results = "y";
  const auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
  const auto constant = [&](const std::string &type, const std::string &name, uint64_t offset) {
    line(ane::Constants::declaration(type, name, offset));
  };
  const auto declareConv = [&](const Layers::Conv &c) {
    constant("tensor<fp16, " + dims({c.out, c.in, 1, c.taps}) + ">", c.name + "_w", c.weight);
    constant("tensor<fp16, " + dims({c.out}) + ">", c.name + "_b", c.bias);
  };
  const auto declareSnake = [&](const Layers::Snake &s) {
    constant(tensor(s.channels, 1), s.name + "_a", s.alpha);
    constant(tensor(s.channels, 1), s.name + "_b", s.beta);
  };
  // x + beta * sin(alpha x)^2 over [channels][columns].
  const auto snake = [&](const Layers::Snake &s, const std::string &x, uint32_t columns, const std::string &out) {
    const std::string t = tensor(s.channels, columns);
    line(t + " " + out + "_m = mul(x = " + x + ", y = " + s.name + "_a)");
    line(t + " " + out + "_s = sin(x = " + out + "_m)");
    line(t + " " + out + "_q = mul(x = " + out + "_s, y = " + out + "_s)");
    line(t + " " + out + "_p = mul(x = " + out + "_q, y = " + s.name + "_b)");
    line(t + " " + out + " = add(x = " + x + ", y = " + out + "_p)");
  };
  // Input surface `name` of `channels` x `columns` as the tensor `name`_4.
  const auto input = [&](const std::string &name, const ane::Surface &surface, uint32_t channels, uint32_t columns) {
    parameters += (parameters.empty() ? "" : ", ") + surface.bufferType(channels, columns) + " " + name;
    line("tensor<fp16, " + dims({1, 1, channels, columns}) + "> " + name + "_t = tensor_buffer_to_tensor<ios17>(input = " +
         name + ")");
    line(tensor(channels, columns) + " " + name + "_4 = reshape(x = " + name + "_t, shape = tensor<int32, [4]>(" +
         dims({1, channels, 1, columns}) + "))");
  };
  const auto output = [&](const std::string &name, const std::string &value, const ane::Surface &surface,
                          uint32_t channels, uint32_t columns) {
    line("tensor<fp16, " + dims({1, 1, channels, columns}) + "> " + name + "_t = reshape(x = " + value +
         ", shape = tensor<int32, [4]>(" + dims({1, 1, channels, columns}) + "))");
    line(surface.bufferType(channels, columns) + " " + name + " = tensor_to_tensor_buffer<ios17>(input = " + name +
         "_t, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>(" +
         surface.strides(channels) + "))");
  };
  // The tensor holding each history now: the input's, then the last that a
  // piece left.
  std::vector<std::string> histories;
  for (size_t h = 0; h < l.histories.size(); ++h) histories.push_back("h" + std::to_string(h) + "_4");
  // A causal convolution of `x` ([c.in][columns]) after its history: the
  // output [c.out][columns] in `out`; the new history is the concatenation's
  // last rows.
  const auto causal = [&](const Layers::Conv &c, const std::string &x, uint32_t columns, const std::string &out) {
    const auto [channels, length] = l.histories.at(c.history);
    if (channels != c.in || length != (c.taps - 1) * c.dilation)
      throw std::logic_error("codec ANE stack history does not match its convolution");
    line(tensor(channels, length + columns) + " " + out + "_c = concat(axis = int32(3), interleave = bool(false), "
         "values = (" + histories[c.history] + ", " + x + "))");
    line(tensor(channels, length) + " " + out + "_n = slice_by_size(x = " + out + "_c, begin = tensor<int32, [4]>(" +
         dims({0, 0, 0, columns}) + "), size = tensor<int32, [4]>(" + dims({1, channels, 1, length}) + "))");
    histories[c.history] = out + "_n";
    line(tensor(c.out, columns) + " " + out + " = conv(bias = " + c.name + "_b, dilations = tensor<int32, [2]>(" +
         dims({1, c.dilation}) + "), groups = int32(1), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = "
         "string(\"valid\"), strides = tensor<int32, [2]>([1, 1]), weight = " + c.name + "_w, x = " + out + "_c)");
  };
  // Block b over `x` ([2 c][columns]): the output [c][columns * rate].
  const auto block = [&](size_t b, const std::string &x, uint32_t columns, const std::string &name) {
    const Layers::Block &block = l.blocks[b];
    const uint32_t s = block.rate, c = block.channels;
    snake(block.snake, x, columns, name + "_a");
    // The s phases of each input row ([s][c][T] channels), interleaved into
    // rows: phase r of row t is row t s + r.
    causal(block.upsample, name + "_a", columns, name + "_p");
    std::string phases;
    for (uint32_t r = 0; r < s; ++r) {
      const std::string phase = name + "_q" + std::to_string(r);
      line(tensor(c, columns) + " " + phase + " = slice_by_size(x = " + name + "_p, begin = tensor<int32, [4]>(" +
           dims({0, uint64_t{r} * c, 0, 0}) + "), size = tensor<int32, [4]>(" + dims({1, c, 1, columns}) + "))");
      phases += (r ? ", " : "") + phase;
    }
    columns *= s;
    std::string y = name + "_y";
    line(tensor(c, columns) + " " + y + " = concat(axis = int32(3), interleave = bool(true), values = (" + phases + "))");
    for (size_t u = 0; u < block.units.size(); ++u) {
      const Layers::Unit &unit = block.units[u];
      const std::string r = name + "r" + std::to_string(u);
      snake(unit.act1, y, columns, r + "_a");
      causal(unit.conv1, r + "_a", columns, r + "_c1");
      snake(unit.act2, r + "_c1", columns, r + "_b");
      line(tensor(c, columns) + " " + r + "_c2 = conv(bias = " + unit.conv2.name + "_b, dilations = tensor<int32, "
           "[2]>([1, 1]), groups = int32(1), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), "
           "strides = tensor<int32, [2]>([1, 1]), weight = " + unit.conv2.name + "_w, x = " + r + "_b)");
      line(tensor(c, columns) + " " + r + "_y = add(x = " + y + ", y = " + r + "_c2)");
      y = r + "_y";
    }
    return y;
  };

  uint32_t columns = frames * l.rowsPerFrame;
  input("x", shape.input, l.latent, columns);
  for (size_t h = 0; h < l.histories.size(); ++h)
    input("h" + std::to_string(h), shape.histories[0][h], l.histories[h].first, l.histories[h].second);
  declareConv(l.initial);
  for (const Layers::Block &b : l.blocks) {
    declareSnake(b.snake);
    declareConv(b.upsample);
    for (const Layers::Unit &unit : b.units) {
      declareSnake(unit.act1);
      declareConv(unit.conv1);
      declareSnake(unit.act2);
      declareConv(unit.conv2);
    }
  }
  declareSnake(l.outputSnake);
  declareConv(l.output);

  causal(l.initial, "x_4", columns, "i");
  std::string y = "i";
  for (size_t b = 0; b < plan.split; ++b) {
    y = block(b, y, columns, "b" + std::to_string(b));
    columns *= l.blocks[b].rate;
  }
  // Per piece: the remaining blocks, the output SnakeBeta and convolution.
  const uint32_t piece = columns / plan.pieces;
  std::string samples;
  for (uint32_t p = 0; p < plan.pieces; ++p) {
    const std::string name = "p" + std::to_string(p);
    uint32_t width = piece;
    std::string z = y;
    if (plan.pieces > 1) {
      z = name + "_x";
      line(tensor(l.blocks[plan.split].upsample.in, piece) + " " + z + " = slice_by_size(x = " + y +
           ", begin = tensor<int32, [4]>(" + dims({0, 0, 0, uint64_t{p} * piece}) + "), size = tensor<int32, [4]>(" +
           dims({1, l.blocks[plan.split].upsample.in, 1, piece}) + "))");
    }
    for (size_t b = plan.split; b < l.blocks.size(); ++b) {
      z = block(b, z, width, name + "b" + std::to_string(b));
      width *= l.blocks[b].rate;
    }
    snake(l.outputSnake, z, width, name + "o_a");
    causal(l.output, name + "o_a", width, name + "o_c");
    line(tensor(1, width) + " " + name + "o_y = clip(alpha = " + fp16(-1.0) + ", beta = " + fp16(1.0) + ", x = " + name +
         "o_c)");
    samples += (p ? ", " : "") + name + "o_y";
  }
  const uint32_t width = piece * plan.samplesPerColumn;
  if (plan.pieces == 1) {
    output("y", samples, shape.samples, 1, width);
  } else {
    line(tensor(plan.pieces, width) + " o_y = concat(axis = int32(1), interleave = bool(false), values = (" + samples +
         "))");
    output("y", "o_y", shape.samples, plan.pieces, width);
  }
  for (size_t h = 0; h < l.histories.size(); ++h) {
    const std::string n = "n" + std::to_string(h);
    output(n, histories[h], shape.histories[0][h], l.histories[h].first, l.histories[h].second);
    results += ", " + n;
  }
  return "    func f" + std::to_string(frames) + "<ios18>(" + parameters + ") {\n" + body + "    } -> (" + results +
         ");\n";
}

CodecAneStack::Plan CodecAneStack::plan(uint32_t frames) const {
  const Layers &l = *layers_;
  Plan plan;
  plan.split = static_cast<uint32_t>(l.blocks.size());
  uint32_t columns = frames * l.rowsPerFrame;
  for (size_t b = 0; b < l.blocks.size(); ++b) {
    if (uint64_t{columns} * l.blocks[b].rate > kPieceColumns) {
      plan.split = static_cast<uint32_t>(b);
      break;
    }
    columns *= l.blocks[b].rate;
  }
  plan.samplesPerColumn = 1;
  for (size_t b = plan.split; b < l.blocks.size(); ++b) plan.samplesPerColumn *= l.blocks[b].rate;
  // The fewest pieces of the split block's input columns, in equal parts,
  // whose samples fit.
  for (plan.pieces = 1; plan.pieces < columns; ++plan.pieces)
    if (columns % plan.pieces == 0 && uint64_t{columns / plan.pieces} * plan.samplesPerColumn <= kPieceColumns) break;
  if (columns % plan.pieces || uint64_t{columns / plan.pieces} * plan.samplesPerColumn > kPieceColumns)
    throw std::invalid_argument("codec ANE stack cannot split " + std::to_string(frames) + " frames into pieces");
  plan.pieceSamples = columns / plan.pieces * plan.samplesPerColumn;
  return plan;
}

uint32_t CodecAneStack::functionFor(uint32_t frames) const noexcept {
  for (const uint32_t function : functionFrames_)
    if (function >= frames) return function;
  return 0;
}

void CodecAneStack::reset(uint32_t lane) {
  Lane &state = *lanes_.at(lane);
  for (auto &half : state.histories)
    for (ane::Surface &surface : half) std::memset(surface.buffer.contents(), 0, surface.buffer.sizeBytes());
  state.phase = 0;
}

void CodecAneStack::addInput(metal::CommandGraph &graph, uint32_t lane, const CodecRows &rows, uint32_t function) {
  const Lane &state = *lanes_.at(lane);
  const Layers &l = *layers_;
  const uint32_t columns = function * l.rowsPerFrame;
  if (functionFor(function) != function)
    throw std::invalid_argument("codec ANE stack has no function of " + std::to_string(function) + " frames");
  if (rows.width != l.latent || !rows.rows || rows.rows > columns)
    throw std::invalid_argument("codec ANE stack input rows do not fit the function");
  requireBytes(rows.buffer, rows.bytes(), "codec ANE stack input rows");
  const OmniCodecSurfaceParams params{rows.rows, columns, l.latent, state.input.strideBytes / 2};
  graph.add("omni_codec_to_surface", {rows.buffer, state.input.buffer}, params,
            {ceilDiv(uint64_t{columns} * l.latent, 256), 1, 1});
}

void CodecAneStack::enqueue(uint32_t lane, uint32_t function, const metal::SharedEvent &event, uint64_t wait,
                            uint64_t signal, std::function<void(bool)> done) {
  Lane &state = *lanes_.at(lane);
  const auto found = std::ranges::find(functionFrames_, function);
  if (found == functionFrames_.end())
    throw std::invalid_argument("codec ANE stack has no function of " + std::to_string(function) + " frames");
  const size_t index = static_cast<size_t>(found - functionFrames_.begin()) * 2 + state.phase;
  program_->enqueue(state.bindings.at(index), event, wait, signal, std::move(done));
}

void CodecAneStack::advance(uint32_t lane) { lanes_.at(lane)->phase ^= 1; }

void CodecAneStack::samples(uint32_t lane, uint32_t function, std::span<float> out) const {
  const Lane &state = *lanes_.at(lane);
  const Plan plan = this->plan(function);
  if (out.size() > uint64_t{plan.pieces} * plan.pieceSamples)
    throw std::invalid_argument("codec ANE stack function holds fewer samples");
  const auto *base = static_cast<const uint8_t *>(state.samples.buffer.contents());
  for (size_t i = 0; i < out.size(); ++i) {
    const size_t piece = i / plan.pieceSamples, offset = i % plan.pieceSamples;
    out[i] = static_cast<float>(reinterpret_cast<const _Float16 *>(base + piece * state.samples.strideBytes)[offset]);
  }
}

void CodecAneStack::addHistoriesToStream(metal::CommandGraph &graph, uint32_t lane, const CodecStream &stream,
                                         uint32_t phase) const {
  const Lane &state = *lanes_.at(lane);
  const std::span<const CodecHistory> histories = stream.stackHistories();
  if (histories.size() != layers_->histories.size()) throw std::invalid_argument("codec stream histories do not match");
  for (size_t h = 0; h < histories.size(); ++h) {
    const CodecHistory &history = histories[h];
    const ane::Surface &surface = state.histories[state.phase][h];
    if (history.width != layers_->histories[h].first || history.length != layers_->histories[h].second)
      throw std::invalid_argument("codec stream history does not match the ANE stack's");
    const OmniCodecSurfaceParams params{history.length, history.length, history.width, surface.strideBytes / 2};
    graph.add("omni_codec_from_surface", {surface.buffer, history.rows[phase & 1u]}, params,
              {ceilDiv(uint64_t{history.length} * history.width, 256), 1, 1});
  }
}

void CodecAneStack::addInputToRows(metal::CommandGraph &graph, uint32_t lane, const CodecRows &rows) const {
  const Lane &state = *lanes_.at(lane);
  if (rows.width != layers_->latent || rows.rows > functionFrames_.back() * layers_->rowsPerFrame)
    throw std::invalid_argument("codec ANE stack input rows do not fit");
  requireBytes(rows.buffer, rows.bytes(), "codec ANE stack rows");
  const OmniCodecSurfaceParams params{rows.rows, rows.rows, rows.width, state.input.strideBytes / 2};
  graph.add("omni_codec_from_surface", {state.input.buffer, rows.buffer}, params,
            {ceilDiv(uint64_t{rows.rows} * rows.width, 256), 1, 1});
}

void CodecAneStack::addHistoriesFromStream(metal::CommandGraph &graph, uint32_t lane, const CodecStream &stream,
                                           uint32_t phase) const {
  const Lane &state = *lanes_.at(lane);
  const std::span<const CodecHistory> histories = stream.stackHistories();
  if (histories.size() != layers_->histories.size()) throw std::invalid_argument("codec stream histories do not match");
  for (size_t h = 0; h < histories.size(); ++h) {
    const CodecHistory &history = histories[h];
    const ane::Surface &surface = state.histories[state.phase][h];
    if (history.width != layers_->histories[h].first || history.length != layers_->histories[h].second)
      throw std::invalid_argument("codec stream history does not match the ANE stack's");
    const OmniCodecSurfaceParams params{history.length, history.length, history.width, surface.strideBytes / 2};
    graph.add("omni_codec_to_surface", {history.rows[phase & 1u], surface.buffer}, params,
              {ceilDiv(uint64_t{history.length} * history.width, 256), 1, 1});
  }
}

} // namespace splash::ops::omni
