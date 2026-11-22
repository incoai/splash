// The short convolution, block residual mix, message gate and logit finish
// kernels (kernels/shared/short_convolution.metal, block_residual_mix.metal,
// message_gate.metal, logit_finish.metal) through their operators, against
// fp64 CPU references:
// - the convolution, static and dynamic taps, each output, over a prompt
//   chunk's packed sequences from their states, the state after them, a chunk
//   split in two carrying the state (the same bits as one chunk), states
//   captured inside the sequences by the convolution's dispatches;
// - a verify block's tree, whose rows tap their paths, the tape, and the
//   commit of accepted paths of two lanes, rejected rows and all;
// - the mix of 1, 3 and 4 queries over 1 to 5 states (bf16 stack planes, bf16
//   or fp32 current rows), plain and through its norms;
// - the message gate through its norm;
// - the logit finish's softcap and padding rows.
#include "ChainRows.hpp"
#include "../../../runtime/metal/CommandGraph.hpp"
#include "../../../runtime/metal/MetalBackend.hpp"
#include "../../../runtime/ops/BlockResidualMix.hpp"
#include "../../../runtime/ops/LogitFinish.hpp"
#include "../../../runtime/ops/MessageGate.hpp"
#include "../../../runtime/ops/ShortConvolution.hpp"
#include "ops/LaneRows.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
namespace ops = splash::ops;

int failures = 0;
void check(bool ok, const std::string &what) {
  std::cout << what << (ok ? " ok" : " FAIL") << '\n';
  if (!ok) ++failures;
}

uint16_t bf16(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  bits += 0x7FFF + ((bits >> 16) & 1);
  return uint16_t(bits >> 16);
}
float f32(uint16_t v) {
  const uint32_t bits = uint32_t(v) << 16;
  float out;
  std::memcpy(&out, &bits, 4);
  return out;
}
double roundBf16(double v) { return f32(bf16(static_cast<float>(v))); }

template <class T> MetalBuffer upload(MetalBackend &backend, const std::vector<T> &values) {
  MetalBuffer buffer = backend.allocateBuffer(std::max<uint64_t>(values.size() * sizeof(T), 16), BufferStorage::Shared,
                                              "convolution-mix-test");
  std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(T));
  return buffer;
}
template <class T> std::vector<T> download(const MetalBuffer &buffer, size_t count) {
  std::vector<T> out(count);
  std::memcpy(out.data(), buffer.contents(), count * sizeof(T));
  return out;
}
MetalBuffer scratch(MetalBackend &backend, uint64_t bytes) {
  return backend.allocateBuffer(std::max<uint64_t>(bytes, 16), BufferStorage::Shared, "convolution-mix-test");
}
std::vector<float> randoms(std::mt19937 &rng, size_t n, float scale) {
  std::normal_distribution<float> d(0.0f, scale);
  std::vector<float> v(n);
  for (float &x : v) x = d(rng);
  return v;
}
// Values a bf16 holds exactly, as a projection's rows and a GGUF's F32
// copies of bf16 weights are.
std::vector<float> bf16Values(std::vector<float> v) {
  for (float &x : v) x = f32(bf16(x));
  return v;
}
std::vector<uint16_t> bf16s(const std::vector<float> &v) {
  std::vector<uint16_t> out(v.size());
  std::transform(v.begin(), v.end(), out.begin(), bf16);
  return out;
}
// Within a bf16 step of want plus a small absolute floor: an fp32 result
// rounded to bf16 against the fp64 one.
bool closeBf16(double got, double want, double floor = 1e-3) {
  return std::fabs(got - want) <= std::fabs(want) * (1.0 / 128) + floor;
}
bool closeF32(double got, double want) { return std::fabs(got - want) <= 1e-5 * (1 + std::fabs(want)); }

void run(MetalBackend &backend, CommandGraph &graph) {
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}

MetalBuffer laneRowsOf(MetalBackend &backend, std::span<const ops::LaneRows> lanes) {
  return upload(backend, std::vector<ops::LaneRows>(lanes.begin(), lanes.end()));
}

constexpr uint32_t kTaps = SHORT_CONVOLUTION_TAPS, kStateRows = SHORT_CONVOLUTION_STATE_ROWS;
// The dynamic taps' rows, a projection's output of whole 64-column tiles, of
// which 8 heads x 4 taps are used.
constexpr uint32_t kDynamicStride = 64;

// A convolution's data: its inputs (bf16 values) of `rows` packed rows, its
// static taps, its rows' dynamic taps (bf16 values), and each sequence's
// state before (bf16 values), in a state matrix of `stride` columns whose
// columns from `column` are the convolution's.
struct ConvCase {
  ops::ShortConvShape shape;
  ops::ShortConvState state;
  uint32_t rows = 0;
  std::vector<float> input, taps, dynamic, residual;
  std::vector<std::vector<float>> states;  // a sequence's [kStateRows][stride]
};

ConvCase convCase(std::mt19937 &rng, ops::ShortConvShape shape, ops::ShortConvState state, uint32_t rows,
                  uint32_t sequences) {
  ConvCase c;
  c.shape = shape;
  c.state = state;
  c.rows = rows;
  c.input = bf16Values(randoms(rng, size_t{rows} * shape.channels, 1.0f));
  c.taps = randoms(rng, size_t{shape.channels} * kTaps, 0.4f);
  c.dynamic = bf16Values(randoms(rng, size_t{rows} * kDynamicStride, 0.3f));
  c.residual = randoms(rng, size_t{rows} * shape.channels, 1.0f);
  for (uint32_t s = 0; s < sequences; ++s)
    c.states.push_back(bf16Values(randoms(rng, size_t{kStateRows} * state.stride, 1.0f)));
  return c;
}

// The fp64 convolution of row `row` (packed) at channel ch: its own input
// and `back(k)`, the input k rows back (k = 1..3), before bf16 rounding.
template <class Back>
double convolve(const ConvCase &c, uint32_t row, uint32_t ch, Back back) {
  double sum = 0;
  const uint32_t heads = c.shape.dynamicHeads;
  for (uint32_t k = 0; k < kTaps; ++k) {
    // The static taps in a conv1d weight's order (the last weighs the row
    // itself), the dynamic ones by lag.
    double w = c.taps[size_t{ch} * kTaps + (kTaps - 1 - k)];
    if (heads) w += c.dynamic[size_t{row} * kDynamicStride + ch / (c.shape.channels / heads) * kTaps + k];
    sum += w * back(k);
  }
  const double own = c.input[size_t{row} * c.shape.channels + ch];
  return (sum + own) / std::sqrt(double((heads ? 2 : 1) * kTaps + 1));
}

struct ConvRun {
  std::vector<double> outputs;                 // as the output's type holds them
  std::vector<std::vector<uint16_t>> states;   // each sequence's state matrix after
  std::vector<std::vector<uint16_t>> captured; // each capture's
};

// A capture's sequence and row.
struct CaptureAt {
  uint32_t sequence = 0, row = 0;
};

// The prompt convolution of `lanes` (packed sequences) as one command,
// capturing the states `captures` name.
ConvRun runPrompt(MetalBackend &backend, const ConvCase &c, std::span<const ops::LaneRows> lanes,
                  ops::ShortConvOutput output, std::span<const CaptureAt> captures = {}) {
  const uint32_t channels = c.shape.channels, stride = c.state.stride;
  std::vector<MetalBuffer> in, out;
  for (size_t s = 0; s < lanes.size(); ++s) {
    in.push_back(upload(backend, bf16s(c.states[s])));
    out.push_back(upload(backend, bf16s(c.states[s])));  // columns past the convolution's stay
  }
  // Sum writes fp32 rows of a bf16 residual; RoundedSum bf16 rows of an
  // fp32 one.
  const bool sum = output == ops::ShortConvOutput::Sum;
  const MetalBuffer result = scratch(backend, size_t{c.rows} * channels * (sum ? 4 : 2));
  const MetalBuffer residual = output == ops::ShortConvOutput::Plain ? MetalBuffer{}
                               : sum                                 ? upload(backend, bf16s(c.residual))
                                                                     : upload(backend, c.residual);
  ops::ShortConvBuffers buffers{upload(backend, bf16s(c.input)), upload(backend, c.taps),
                                c.shape.dynamic() ? upload(backend, bf16s(c.dynamic)) : MetalBuffer{},
                                kDynamicStride, residual, result};
  std::vector<ops::ShortConvCapture> states;
  for (const CaptureAt &capture : captures)
    states.push_back({capture.sequence, capture.row, scratch(backend, ops::ShortConvolution::stateBytes(stride))});
  CommandGraph graph;
  ops::ShortConvolution::addConvolution(graph, buffers, c.shape, output, in, out, c.state, laneRowsOf(backend, lanes),
                                        {static_cast<uint32_t>(lanes.size()), c.rows, splash::test::mostRows(lanes)},
                                        states);
  run(backend, graph);
  ConvRun r;
  if (sum) {
    for (float v : download<float>(result, size_t{c.rows} * channels)) r.outputs.push_back(v);
  } else {
    for (uint16_t v : download<uint16_t>(result, size_t{c.rows} * channels)) r.outputs.push_back(f32(v));
  }
  for (const MetalBuffer &s : out) r.states.push_back(download<uint16_t>(s, size_t{kStateRows} * stride));
  for (const ops::ShortConvCapture &s : states)
    r.captured.push_back(download<uint16_t>(s.state, size_t{kStateRows} * stride));
  return r;
}

// The chunk's convolution of each output against fp64 and each sequence's
// state after its rows, exactly.
void convPromptTest(MetalBackend &backend, ops::ShortConvShape shape) {
  std::mt19937 rng(7 + shape.dynamicHeads);
  const ops::ShortConvState state{256, shape.channels + 512};
  // Two sequences of 9 and 2 rows (shorter than the state) packed.
  const std::array<ops::LaneRows, 2> lanes{{{0, 9}, {9, 2}}};
  const ConvCase c = convCase(rng, shape, state, 11, 2);
  const std::string scope = std::string("short convolution (") + (shape.dynamic() ? "dynamic" : "static") + ")";
  const auto input = [&](uint32_t s, int t, uint32_t ch) -> double {
    return t >= 0 ? c.input[size_t{lanes[s].first + uint32_t(t)} * shape.channels + ch]
                  : c.states[s][size_t(int(kStateRows) + t) * state.stride + state.column + ch];
  };
  for (const ops::ShortConvOutput output :
       {ops::ShortConvOutput::Plain, ops::ShortConvOutput::Sum, ops::ShortConvOutput::RoundedSum}) {
    const ConvRun got = runPrompt(backend, c, lanes, output);
    bool ok = true;
    for (uint32_t s = 0; s < lanes.size(); ++s)
      for (uint32_t t = 0; t < lanes[s].rows; ++t)
        for (uint32_t ch = 0; ch < shape.channels; ++ch) {
          const uint32_t row = lanes[s].first + t;
          const double y = convolve(c, row, ch, [&](uint32_t k) { return input(s, int(t) - int(k), ch); });
          const double g = got.outputs[size_t{row} * shape.channels + ch];
          if (output == ops::ShortConvOutput::Plain) {
            ok &= closeBf16(g, y);
          } else {
            // bf16(y) on the residual (bf16 for Sum): one bf16 step of y,
            // then (RoundedSum) the sum's.
            const double r = c.residual[size_t{row} * shape.channels + ch];
            const double sum = (output == ops::ShortConvOutput::Sum ? roundBf16(r) : r) + y;
            ok &= std::fabs(g - sum) <= std::fabs(y) / 128 + std::fabs(sum) / 128 + 1e-3;
          }
        }
    const char *name = output == ops::ShortConvOutput::Plain ? "bf16 rows"
                       : output == ops::ShortConvOutput::Sum ? "fp32 of a bf16 residual plus the rows"
                                                             : "bf16 of an fp32 residual plus the rows";
    check(ok, scope + ": " + name + " of two packed sequences from their states");
    bool okState = true;
    for (uint32_t s = 0; s < lanes.size(); ++s)
      for (uint32_t j = 0; j < kStateRows; ++j)
        for (uint32_t col = 0; col < state.stride; ++col) {
          const bool own = col >= state.column && col < state.column + shape.channels;
          const double want = own ? input(s, int(lanes[s].rows) - int(kStateRows) + int(j), col - state.column)
                                  : c.states[s][size_t{j} * state.stride + col];
          okState &= f32(got.states[s][size_t{j} * state.stride + col]) == want;
        }
    check(okState, scope + ": each sequence's state after its rows, its columns alone");
  }
  // The rows split into chunks of 4 and 5 rows of one sequence, carrying the
  // state, give the bits of one chunk of its 9 rows.
  const std::array<ops::LaneRows, 1> whole{{{0, 9}}};
  ConvCase nine = c;
  nine.rows = 9;
  const ConvRun one = runPrompt(backend, nine, whole, ops::ShortConvOutput::Plain);
  ConvCase first = c, second = c;
  first.rows = 4;
  first.input.resize(size_t{4} * shape.channels);
  second.rows = 5;
  second.input.assign(c.input.begin() + size_t{4} * shape.channels, c.input.begin() + size_t{9} * shape.channels);
  second.dynamic.assign(c.dynamic.begin() + size_t{4} * kDynamicStride, c.dynamic.end());
  const std::array<ops::LaneRows, 1> four{{{0, 4}}}, five{{{0, 5}}};
  const ConvRun a = runPrompt(backend, first, four, ops::ShortConvOutput::Plain);
  second.states = {std::vector<float>(a.states[0].size())};
  for (size_t i = 0; i < a.states[0].size(); ++i) second.states[0][i] = f32(a.states[0][i]);
  const ConvRun b = runPrompt(backend, second, five, ops::ShortConvOutput::Plain);
  std::vector<double> split = a.outputs;
  split.insert(split.end(), b.outputs.begin(), b.outputs.end());
  check(split == one.outputs && b.states[0] == one.states[0],
        scope + ": a chunk split in two carrying the state gives one chunk's bits and state");
  // States captured inside the sequences by the convolution's dispatch: six
  // (more than one dispatch writes), in both sequences, one at a sequence's
  // last row and one before the state's rows.
  const std::array<CaptureAt, 6> captures{{{0, 1}, {0, 5}, {1, 1}, {0, 9}, {1, 2}, {0, 2}}};
  const ConvRun got = runPrompt(backend, c, lanes, ops::ShortConvOutput::Plain, captures);
  bool okCaptures = true;
  for (size_t q = 0; q < captures.size(); ++q)
    for (uint32_t j = 0; j < kStateRows; ++j)
      for (uint32_t ch = 0; ch < shape.channels; ++ch)
        okCaptures &= f32(got.captured[q][size_t{j} * state.stride + state.column + ch]) ==
                      input(captures[q].sequence, int(captures[q].row) - int(kStateRows) + int(j), ch);
  check(okCaptures, scope + ": six states captured inside two sequences by the convolution's dispatches");
}

// Two lanes' verify blocks: lane 0 a tree (a chain of four, a branch off row
// 1 and one off the anchor), lane 1 a chain of three. Each row taps its path
// (then its lane's state), its input goes to the tape, and the commit of an
// accepted path per lane writes each lane's state after that path from the
// tape, every column of the matrix (the convolution's and the others').
void convTreeTest(MetalBackend &backend, ops::ShortConvShape shape) {
  std::mt19937 rng(23 + shape.dynamicHeads);
  const std::array<uint32_t, 8> parents{0, 0, 1, 2, 3, 1, 5, 0};
  const std::array<ops::LaneRows, 2> lanes{{{0, 8}, {8, 3}}};
  const uint32_t rows = 11;
  const ops::ShortConvState state{128, shape.channels + 384};
  const ConvCase c = convCase(rng, shape, state, rows, 2);
  std::vector<ops::TreeRow> tree(rows);
  for (uint32_t r = 0; r < 8; ++r)
    tree[r] = r ? ops::TreeRow{tree[parents[r]].ancestors | 1u << r, static_cast<uint16_t>(parents[r]),
                               static_cast<uint16_t>(tree[parents[r]].depth + 1)}
                : splash::test::chainRow(0);
  for (uint32_t r = 0; r < 3; ++r) tree[8 + r] = splash::test::chainRow(r);
  const std::string scope = std::string("short convolution tree (") + (shape.dynamic() ? "dynamic" : "static") + ")";
  // Input `depth` of packed row r's path in lane l, or its state's before
  // the anchor.
  const auto input = [&](uint32_t l, uint32_t r, int depth, uint32_t col) -> double {
    if (depth < 0) return c.states[l][size_t(int(kStateRows) + depth) * state.stride + col];
    uint32_t row = r - lanes[l].first;
    while (tree[lanes[l].first + row].depth > uint32_t(depth)) row = tree[lanes[l].first + row].parent;
    return col >= state.column && col < state.column + shape.channels
               ? c.input[size_t{lanes[l].first + row} * shape.channels + col - state.column]
               : 0.0;
  };
  std::vector<MetalBuffer> in;
  for (uint32_t l = 0; l < 2; ++l) in.push_back(upload(backend, bf16s(c.states[l])));
  const MetalBuffer output = scratch(backend, size_t{rows} * shape.channels * 2),
                    tape = upload(backend, std::vector<uint16_t>(size_t{rows} * state.stride, 0)),
                    layout = laneRowsOf(backend, lanes);
  ops::ShortConvBuffers buffers{upload(backend, bf16s(c.input)), upload(backend, c.taps),
                                shape.dynamic() ? upload(backend, bf16s(c.dynamic)) : MetalBuffer{}, kDynamicStride,
                                {}, output};
  CommandGraph graph;
  ops::ShortConvolution::addConvolution(graph, buffers, shape, ops::ShortConvOutput::Plain, in, {}, state, layout,
                                        {2, rows, 8}, {}, upload(backend, tree), tape);
  run(backend, graph);
  const auto got = download<uint16_t>(output, size_t{rows} * shape.channels);
  const auto gotTape = download<uint16_t>(tape, size_t{rows} * state.stride);
  bool ok = true, okTape = true, differs = false;
  for (uint32_t l = 0; l < 2; ++l)
    for (uint32_t r = lanes[l].first; r < lanes[l].first + lanes[l].rows; ++r)
      for (uint32_t ch = 0; ch < shape.channels; ++ch) {
        const uint32_t col = state.column + ch;
        const double y =
            convolve(c, r, ch, [&](uint32_t k) { return input(l, r, int(tree[r].depth) - int(k), col); });
        const double packed = convolve(c, r, ch, [&](uint32_t k) {
          const int t = int(r - lanes[l].first) - int(k);
          return t >= 0 ? c.input[size_t(lanes[l].first + t) * shape.channels + ch]
                        : c.states[l][size_t(int(kStateRows) + t) * state.stride + col];
        });
        ok &= closeBf16(f32(got[size_t{r} * shape.channels + ch]), y);
        differs |= std::fabs(packed - y) > 1e-2;
        okTape &= f32(gotTape[size_t{r} * state.stride + col]) == c.input[size_t{r} * shape.channels + ch];
      }
  check(differs, scope + ": a row's packed predecessors are not its path's");
  check(ok, scope + ": each row taps its path, then its lane's state");
  check(okTape, scope + ": each row's input in the tape at the convolution's columns");
  // The tape's other columns as a step's other convolutions would write them.
  std::vector<float> fullTape(size_t{rows} * state.stride);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t col = 0; col < state.stride; ++col)
      fullTape[size_t{r} * state.stride + col] = col >= state.column && col < state.column + shape.channels
                                                     ? c.input[size_t{r} * shape.channels + col - state.column]
                                                     : f32(bf16(float(r * 0.25 + col * 1e-3)));
  const MetalBuffer fullTapeBuffer = upload(backend, bf16s(fullTape));
  // Lane 0 accepts the path to row 6 (rows 0, 1, 5, 6: 2, 3, 4 and 7
  // rejected) or to row 1; lane 1 its whole chain or its anchor alone.
  for (const auto &[leaf, kept] : {std::pair<uint32_t, uint32_t>{6, 3}, std::pair<uint32_t, uint32_t>{1, 1}}) {
    std::vector<MetalBuffer> out{scratch(backend, ops::ShortConvolution::stateBytes(state.stride)),
                                 scratch(backend, ops::ShortConvolution::stateBytes(state.stride))};
    CommandGraph commit;
    ops::ShortConvolution::addCommit(commit, fullTapeBuffer, in, out,
                                     upload(backend, std::vector<uint32_t>{tree[leaf].ancestors,
                                                                           splash::test::chainPath(kept)}),
                                     state.stride, layout, {2, rows, 8});
    run(backend, commit);
    bool okCommit = true;
    for (uint32_t l = 0; l < 2; ++l) {
      const auto gotState = download<uint16_t>(out[l], size_t{kStateRows} * state.stride);
      const uint32_t last = l == 0 ? leaf : lanes[1].first + kept - 1;
      for (uint32_t j = 0; j < kStateRows; ++j)
        for (uint32_t col = 0; col < state.stride; ++col) {
          const int depth = int(tree[last].depth) + 1 - int(kStateRows) + int(j);
          double want;
          if (depth < 0) {
            want = c.states[l][size_t(int(kStateRows) + depth) * state.stride + col];
          } else {
            uint32_t row = last;
            while (tree[row].depth > uint32_t(depth)) row = lanes[l].first + tree[row].parent;
            want = fullTape[size_t{row} * state.stride + col];
          }
          okCommit &= f32(gotState[size_t{j} * state.stride + col]) == want;
        }
    }
    check(okCommit, scope + ": the commit of the paths to row " + std::to_string(leaf) + " and " +
                        std::to_string(kept) + " chain rows, every column");
  }
}

// The fp64 mix of row r: each query's output, and through its norm.
// The stack's planes hold planeRows rows each, more than the mix's.
struct MixCase {
  uint32_t hidden = 0, stack = 0, queries = 0, rows = 0, planeRows = 0;
  std::vector<float> stackRows, current, queryValues;
  std::vector<std::vector<float>> norms;
};

std::vector<double> mixReference(const MixCase &m, uint32_t row) {
  const uint32_t states = m.stack + 1, h = m.hidden;
  std::vector<std::vector<double>> normalized(states, std::vector<double>(h));
  for (uint32_t s = 0; s < states; ++s) {
    const float *x = s < m.stack ? &m.stackRows[(size_t{s} * m.planeRows + row) * h] : &m.current[size_t{row} * h];
    double squares = 0;
    for (uint32_t d = 0; d < h; ++d) squares += double(x[d]) * x[d];
    const double inverse = 1 / std::sqrt(squares / h + 1e-6);
    for (uint32_t d = 0; d < h; ++d) normalized[s][d] = x[d] * inverse;
  }
  std::vector<double> out(size_t{m.queries} * h);
  for (uint32_t q = 0; q < m.queries; ++q) {
    std::vector<double> scores(states);
    for (uint32_t s = 0; s < states; ++s) {
      double dot = 0;
      for (uint32_t d = 0; d < h; ++d) dot += m.queryValues[size_t{q} * h + d] * normalized[s][d];
      scores[s] = dot / std::sqrt(double(h));
    }
    const double maximum = *std::max_element(scores.begin(), scores.end());
    double total = 0;
    for (double &score : scores) total += (score = std::exp(score - maximum));
    for (uint32_t d = 0; d < h; ++d)
      for (uint32_t s = 0; s < states; ++s) out[size_t{q} * h + d] += scores[s] / total * normalized[s][d];
  }
  return out;
}

// The mix over bf16 stack planes and bf16 current rows (a layer's input) or
// fp32 ones (the residual inside a layer).
void mixTest(MetalBackend &backend, uint32_t stack, uint32_t queries, bool currentFloat) {
  std::mt19937 rng(stack * 10 + queries + (currentFloat ? 100 : 0));
  MixCase m;
  m.hidden = 2304;
  m.stack = stack;
  m.queries = queries;
  m.rows = 5;
  m.planeRows = 8;
  // States of different scales, as a stack's blocks are, and queries large
  // enough that the softmax does not flatten.
  m.stackRows = randoms(rng, size_t{m.planeRows} * stack * m.hidden, 1.0f);
  for (size_t i = 0; i < m.stackRows.size(); ++i)
    m.stackRows[i] *= 1.0f + float(i / (size_t{m.planeRows} * m.hidden));
  m.stackRows = bf16Values(m.stackRows);
  m.current = randoms(rng, size_t{m.rows} * m.hidden, 3.0f);
  if (!currentFloat) m.current = bf16Values(m.current);
  m.queryValues = randoms(rng, size_t{queries} * m.hidden, 0.2f);
  const std::string scope = "block residual mix of " + std::to_string(queries) + " queries over " +
                            std::to_string(stack + 1) + " states, " + (currentFloat ? "fp32" : "bf16") +
                            " current rows";
  const ops::BlockResidualStates states{stack ? upload(backend, bf16s(m.stackRows)) : MetalBuffer{}, m.planeRows,
                                        currentFloat ? upload(backend, m.current) : upload(backend, bf16s(m.current)),
                                        currentFloat};
  const ops::BlockResidualShape shape{m.hidden, stack, queries};
  const MetalBuffer queryBuffer = upload(backend, m.queryValues),
                    output = scratch(backend, size_t{m.rows} * queries * m.hidden * 4);
  CommandGraph graph;
  ops::BlockResidualMix::addMix(graph, states, queryBuffer, output, shape, m.rows);
  run(backend, graph);
  const auto got = download<float>(output, size_t{m.rows} * queries * m.hidden);
  bool ok = true;
  for (uint32_t r = 0; r < m.rows; ++r) {
    const auto want = mixReference(m, r);
    for (size_t i = 0; i < want.size(); ++i) ok &= closeF32(got[size_t{r} * queries * m.hidden + i], want[i]);
  }
  check(ok, scope);
  if (queries > BLOCK_RESIDUAL_MIX_MAXIMUM_NORMS) return;
  std::vector<ops::NormWeights> norms;
  std::vector<MetalBuffer> outputs;
  for (uint32_t q = 0; q < queries; ++q) {
    m.norms.push_back(randoms(rng, m.hidden, 0.5f));
    norms.push_back({upload(backend, m.norms.back()), true});
    outputs.push_back(scratch(backend, size_t{m.rows} * m.hidden * 2));
  }
  CommandGraph normed;
  ops::BlockResidualMix::addMixNorm(normed, states, queryBuffer, norms, outputs, shape, m.rows);
  run(backend, normed);
  bool okNorm = true;
  for (uint32_t r = 0; r < m.rows; ++r) {
    const auto want = mixReference(m, r);
    for (uint32_t q = 0; q < queries; ++q) {
      const auto gotRows = download<uint16_t>(outputs[q], size_t{m.rows} * m.hidden);
      double squares = 0;
      for (uint32_t d = 0; d < m.hidden; ++d)
        squares += want[size_t{q} * m.hidden + d] * want[size_t{q} * m.hidden + d];
      const double inverse = 1 / std::sqrt(squares / m.hidden + 1e-6);
      for (uint32_t d = 0; d < m.hidden; ++d)
        okNorm &= closeBf16(f32(gotRows[size_t{r} * m.hidden + d]),
                            want[size_t{q} * m.hidden + d] * inverse * m.norms[q][d]);
    }
  }
  check(okNorm, scope + ", each query through its own norm");
}

// The gate of rows whose x is query 1 of a mix of three and whose message
// is message 2 of four, against fp64 with the source's bf16 roundings of the
// down projection's input and output.
void gateTest(MetalBackend &backend) {
  std::mt19937 rng(31);
  const uint32_t hidden = 2304, rank = 64, rows = 6;
  const auto x = randoms(rng, size_t{rows} * 3 * hidden, 1.5f), message = randoms(rng, size_t{rows} * 4 * hidden, 0.7f),
             down = bf16Values(randoms(rng, size_t{rank} * 2 * hidden, 0.02f)),
             forget = randoms(rng, size_t{hidden} * rank, 0.1f), gain = randoms(rng, size_t{hidden} * rank, 0.1f),
             norm = randoms(rng, hidden, 0.5f);
  const ops::MessageGateWeights weights{ops::QuantizedSegment::floats(rank, 2 * hidden, upload(backend, down)),
                                        ops::QuantizedSegment::floats(hidden, rank, upload(backend, forget)),
                                        ops::QuantizedSegment::floats(hidden, rank, upload(backend, gain))};
  const ops::MessageGateBuffers buffers{upload(backend, x),
                                        3 * hidden,
                                        hidden,
                                        upload(backend, message),
                                        4 * hidden,
                                        2 * hidden,
                                        scratch(backend, size_t{rows} * 2 * hidden * 2),
                                        scratch(backend, size_t{rows} * rank * 2),
                                        scratch(backend, size_t{rows} * hidden * 4),
                                        scratch(backend, size_t{rows} * hidden * 4),
                                        scratch(backend, size_t{rows} * hidden * 2)};
  CommandGraph graph;
  ops::MessageGate::add(graph, buffers, weights, {upload(backend, norm), true}, hidden, rows,
                        ops::FloatTile::Simdgroup);
  run(backend, graph);
  const auto got = download<uint16_t>(buffers.output, size_t{rows} * hidden);
  bool ok = true;
  for (uint32_t r = 0; r < rows; ++r) {
    const float *xr = &x[size_t{r} * 3 * hidden + hidden], *mr = &message[size_t{r} * 4 * hidden + 2 * hidden];
    double xs = 0, ms = 0;
    for (uint32_t d = 0; d < hidden; ++d) {
      xs += double(xr[d]) * xr[d];
      ms += double(mr[d]) * mr[d];
    }
    xs = 1 / std::sqrt(xs / hidden + 1e-6);
    ms = 1 / std::sqrt(ms / hidden + 1e-6);
    std::vector<double> concat(2 * hidden), g(rank);
    for (uint32_t d = 0; d < hidden; ++d) {
      concat[d] = roundBf16(xr[d] * xs);
      concat[hidden + d] = roundBf16(mr[d] * ms);
    }
    for (uint32_t k = 0; k < rank; ++k) {
      double dot = 0;
      for (uint32_t i = 0; i < 2 * hidden; ++i) dot += down[size_t{k} * 2 * hidden + i] * concat[i];
      g[k] = roundBf16(dot);
    }
    std::vector<double> v(hidden);
    double squares = 0;
    for (uint32_t d = 0; d < hidden; ++d) {
      double f = 0, e = 0;
      for (uint32_t k = 0; k < rank; ++k) {
        f += forget[size_t{d} * rank + k] * g[k];
        e += gain[size_t{d} * rank + k] * g[k];
      }
      v[d] = (1 - std::tanh(f)) * xr[d] + std::tanh(e) * mr[d];
      squares += v[d] * v[d];
    }
    const double inverse = 1 / std::sqrt(squares / hidden + 1e-6);
    for (uint32_t d = 0; d < hidden; ++d)
      ok &= closeBf16(f32(got[size_t{r} * hidden + d]), v[d] * inverse * norm[d], 2e-3);
  }
  check(ok, "message gate through its norm, x and the message at columns of wider rows");
}

// The finish of rows of logits: the softcap below `tokens`, -inf past it.
void logitFinishTest(MetalBackend &backend) {
  std::mt19937 rng(3);
  const uint32_t vocabulary = 1000, tokens = 937, rows = 3;
  const float softcap = 30.0f;
  const auto logits = randoms(rng, size_t{rows} * vocabulary, 40.0f);
  const MetalBuffer buffer = upload(backend, logits);
  CommandGraph graph;
  ops::LogitFinish::add(graph, buffer, vocabulary, rows, {softcap, tokens});
  run(backend, graph);
  const auto got = download<float>(buffer, size_t{rows} * vocabulary);
  bool ok = true;
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t t = 0; t < vocabulary; ++t) {
      const size_t i = size_t{r} * vocabulary + t;
      ok &= t < tokens ? closeF32(got[i], softcap * std::tanh(double(logits[i]) / softcap))
                       : got[i] == -std::numeric_limits<float>::infinity();
    }
  check(ok, "logit finish: 30 tanh(l / 30) for the tokens, -inf for the padding rows");
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    try {
      if (argc != 2) throw std::runtime_error("usage: convolution-mix-kernels METALLIB");
      MetalBackend backend(argv[1]);
      // A key or value projection's dynamic convolution (8 heads of 128) and
      // an output's static one.
      for (const ops::ShortConvShape shape : {ops::ShortConvShape{2304, 0}, ops::ShortConvShape{1024, 8}}) {
        convPromptTest(backend, shape);
        convTreeTest(backend, shape);
      }
      for (const bool currentFloat : {false, true}) {
        for (const uint32_t queries : {1u, 3u, 4u})
          for (uint32_t stack = 1; stack <= 4; ++stack) mixTest(backend, stack, queries, currentFloat);
        mixTest(backend, 0, 1, currentFloat);
      }
      gateTest(backend);
      logitFinishTest(backend);
    } catch (const std::exception &error) {
      std::cerr << "convolution and mix kernel test error: " << error.what() << '\n';
      return 1;
    }
  }
  if (failures) {
    std::cout << "convolution and mix kernel tests FAILED (" << failures << " failures)\n";
    return 1;
  }
  std::cout << "convolution and mix kernel tests PASS\n";
  return 0;
}
