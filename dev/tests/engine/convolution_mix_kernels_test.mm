// The short convolution, block residual mix, message gate and logit finish
// kernels (kernels/shared/short_convolution.metal, block_residual_mix.metal,
// block_residual_mix_rows.metal, message_gate.metal, logit_finish.metal)
// through their operators, against fp64 CPU references, in the
// production-and-test library, which holds the message gate's reference path
// (message_gate_test.metal):
// - the convolution, static and dynamic taps, each output, over a prompt
//   chunk's packed sequences from their states, the state after them, a chunk
//   split in two carrying the state (the same bits as one chunk), states
//   captured inside the sequences by the convolution's dispatches;
// - a verify block's tree, whose rows tap their paths, the tape, and the
//   commit of accepted paths of two lanes, rejected rows and all, and a
//   decode capture's fold of a path's first rows, bit for bit the commit of
//   a lane that retained exactly those;
// - a verify step's two convolutions in one dispatch
//   (ShortConvolution::addConvolutionPair), the keys and values of rows of
//   their own or of one projection's rows of both, bit for bit each one's own
//   dispatch (`bitwise`, uninstrumented), the tape too, and each of those
//   reading a view of the wider rows the bits of its own rows';
// - a verify step's convolution in its MoE combine's dispatch
//   (MoE::addCombineConvolution, kernels/shared/moe.metal), bit for bit as
//   moe_combine and then the convolution of its rows (`bitwise`,
//   uninstrumented), and against fp64;
// - the mix of 1, 3 and 4 queries over 1 to 5 states (bf16 stack planes, bf16
//   or fp32 current rows), plain and through its norms, of 5 rows (a
//   threadgroup a row and query) and of BLOCK_RESIDUAL_MIX_QUERY_ROWS + 3 (a
//   threadgroup a row); two mixes in one call, bit for bit each one's own;
// - the mix's outputs normalized for the message gates (BlockResidualMixInput::
//   normalized), against fp64;
// - the message gate through its norm, its fused kernels' (from its inputs
//   normalized as the mixes normalize them) and its reference path's, and the
//   paths against each other;
// - the logit finish's softcap and padding rows.
#include "ChainRows.hpp"
#include "message_gate_reference.h"
#include "../../../runtime/metal/CommandGraph.hpp"
#include "../../../runtime/metal/MetalBackend.hpp"
#include "../../../runtime/ops/BlockResidualMix.hpp"
#include "../../../runtime/ops/LogitFinish.hpp"
#include "../../../runtime/ops/MessageGate.hpp"
#include "../../../runtime/ops/MoE.hpp"
#include "../../../runtime/ops/Normalization.hpp"
#include "../../../runtime/ops/ShortConvolution.hpp"
#include "ops/LaneRows.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
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
  // A decode capture's fold of a lane's path (lane 0's to row 6, lane 1's
  // whole chain) at each of its rows: the commit's bits where the lane
  // retained exactly the path's first `taken` rows, its other lane's
  // layout beside it; past the path's rows the fold leaves its state as it
  // was.
  const std::array<uint32_t, 2> paths{tree[6].ancestors, splash::test::chainPath(3)};
  const uint64_t stateBytes = ops::ShortConvolution::stateBytes(state.stride);
  const std::vector<uint16_t> untouched(stateBytes / 2, 0x7FC1);
  bool okFold = true, okPast = true;
  for (uint32_t l = 0; l < 2; ++l)
    for (uint32_t taken = 1; taken <= uint32_t(std::popcount(paths[l])) + 1; ++taken) {
      // The path's first `taken` rows, the commit's of a lane that retained
      // them; the other lane retains its whole path.
      uint32_t first = 0;
      for (uint32_t rest = paths[l], row = 0; row < taken && rest; ++row, rest &= rest - 1)
        first |= 1u << std::countr_zero(rest);
      std::array<uint32_t, 2> retained = paths;
      retained[l] = first;
      std::vector<MetalBuffer> out{scratch(backend, stateBytes), scratch(backend, stateBytes)};
      const MetalBuffer fold = upload(backend, untouched);
      CommandGraph graph;
      ops::ShortConvolution::addCommit(graph, fullTapeBuffer, in, out,
                                       upload(backend, std::vector<uint32_t>(retained.begin(), retained.end())),
                                       state.stride, layout, {2, rows, 8});
      ops::ShortConvolution::addFold(graph, fullTapeBuffer, in[l], fold, upload(backend, std::vector<uint32_t>{paths[l]}),
                                     state.stride, laneRowsOf(backend, std::span(&lanes[l], 1)), rows, taken);
      run(backend, graph);
      const auto folded = download<uint16_t>(fold, stateBytes / 2);
      if (taken > uint32_t(std::popcount(paths[l])))
        okPast &= folded == untouched;
      else
        okFold &= folded == download<uint16_t>(out[l], stateBytes / 2);
    }
  check(okFold, scope + ": a fold of each path's first rows, the commit of a lane that retained exactly those");
  check(okPast, scope + ": a fold past its path's rows leaves its state as it was");
}

// A verify step's keys and values in one dispatch
// (ShortConvolution::addConvolutionPair) against each one's own dispatch over
// the tree of convTreeTest, of their own inputs (`oneProjection` false) or of
// one projection's rows of both, keys then values (a view of the wider rows
// each, ShortConvBuffers::inputStride and inputColumn), their states'
// columns side by side in one matrix: uninstrumented (`exact`) each output
// bit for bit, under shader validation, which compiles the kernels apart,
// within a bf16 step; the tape exactly, its other columns untouched. Each
// convolution alone of a view of the wider rows writes the bits of its own
// rows'.
void convPairTest(MetalBackend &backend, ops::ShortConvShape shape, bool oneProjection, bool exact) {
  std::mt19937 rng(31 + shape.dynamicHeads + (oneProjection ? 1 : 0));
  const std::array<uint32_t, 8> parents{0, 0, 1, 2, 3, 1, 5, 0};
  const std::array<ops::LaneRows, 2> lanes{{{0, 8}, {8, 3}}};
  const uint32_t rows = 11, channels = shape.channels, stride = 2 * channels + 384;
  const std::array<ops::ShortConvState, 2> states{ops::ShortConvState{128, stride},
                                                  ops::ShortConvState{128 + channels, stride}};
  const ConvCase keys = convCase(rng, shape, states[0], rows, 2), values = convCase(rng, shape, states[1], rows, 2);
  std::vector<ops::TreeRow> tree(rows);
  for (uint32_t r = 0; r < 8; ++r)
    tree[r] = r ? ops::TreeRow{tree[parents[r]].ancestors | 1u << r, static_cast<uint16_t>(parents[r]),
                               static_cast<uint16_t>(tree[parents[r]].depth + 1)}
                : splash::test::chainRow(0);
  for (uint32_t r = 0; r < 3; ++r) tree[8 + r] = splash::test::chainRow(r);
  const std::string scope = std::string("short convolution pair (") + (shape.dynamic() ? "dynamic" : "static") +
                            (oneProjection ? ", one projection's rows" : ", rows of their own") + ")";
  // The lanes' state matrices: the keys' states at their columns, the values'
  // at theirs.
  std::vector<MetalBuffer> in;
  for (uint32_t l = 0; l < 2; ++l) {
    std::vector<float> matrix = keys.states[l];
    for (uint32_t j = 0; j < kStateRows; ++j)
      for (uint32_t ch = 0; ch < channels; ++ch) {
        const size_t at = size_t{j} * stride + states[1].column + ch;
        matrix[at] = values.states[l][at];
      }
    in.push_back(upload(backend, bf16s(matrix)));
  }
  // One projection's rows of both, keys then values.
  std::vector<float> both;
  for (uint32_t r = 0; r < rows; ++r)
    for (const ConvCase *c : {&keys, &values})
      both.insert(both.end(), c->input.begin() + size_t{r} * channels, c->input.begin() + size_t{r + 1} * channels);
  const MetalBuffer layout = laneRowsOf(backend, lanes), treeRows = upload(backend, tree),
                    wide = upload(backend, bf16s(both));
  const auto buffersOf = [&](const ConvCase &c, uint32_t column) {
    ops::ShortConvBuffers buffers{upload(backend, bf16s(c.input)), upload(backend, c.taps),
                                  shape.dynamic() ? upload(backend, bf16s(c.dynamic)) : MetalBuffer{}, kDynamicStride,
                                  {}, scratch(backend, size_t{rows} * channels * 2)};
    if (oneProjection) buffers.input = wide, buffers.inputStride = 2 * channels, buffers.inputColumn = column;
    return buffers;
  };
  // Each alone, of rows of its own, then both in one dispatch.
  const std::array<ops::ShortConvBuffers, 2> pair{buffersOf(keys, 0), buffersOf(values, channels)};
  std::array<ops::ShortConvBuffers, 2> alone = pair;
  for (size_t i = 0; i < alone.size(); ++i) {
    alone[i].input = upload(backend, bf16s(i ? values.input : keys.input));
    alone[i].inputStride = alone[i].inputColumn = 0;
    alone[i].output = scratch(backend, size_t{rows} * channels * 2);
  }
  const MetalBuffer aloneTape = upload(backend, std::vector<uint16_t>(size_t{rows} * stride, 0)),
                    pairTape = upload(backend, std::vector<uint16_t>(size_t{rows} * stride, 0));
  CommandGraph graph;
  for (size_t i = 0; i < alone.size(); ++i)
    ops::ShortConvolution::addConvolution(graph, alone[i], shape, ops::ShortConvOutput::Plain, in, {}, states[i],
                                          layout, {2, rows, 8}, {}, treeRows, aloneTape);
  ops::ShortConvolution::addConvolutionPair(graph, pair, shape, states, in, layout, {2, rows, 8}, treeRows, pairTape);
  run(backend, graph);
  const size_t values16 = size_t{rows} * channels;
  bool same = true, close = true;
  for (size_t i = 0; i < pair.size(); ++i) {
    const auto want = download<uint16_t>(alone[i].output, values16), got = download<uint16_t>(pair[i].output, values16);
    same &= want == got;
    for (size_t v = 0; v < values16; ++v) close &= closeBf16(f32(got[v]), f32(want[v]), 0);
  }
  check(exact ? same : close, scope + (exact ? ": each output bit for bit its own dispatch's"
                                             : ": each output within a bf16 step of its own dispatch's"));
  check(download<uint16_t>(aloneTape, size_t{rows} * stride) == download<uint16_t>(pairTape, size_t{rows} * stride),
        scope + ": the tape its own dispatches', its other columns untouched");
  // A convolution of a view of the wider rows against one of its own rows.
  if (oneProjection) {
    const MetalBuffer viewTape = upload(backend, std::vector<uint16_t>(size_t{rows} * stride, 0));
    std::array<ops::ShortConvBuffers, 2> views = pair;
    CommandGraph viewGraph;
    for (size_t i = 0; i < views.size(); ++i) {
      views[i].output = scratch(backend, size_t{rows} * channels * 2);
      ops::ShortConvolution::addConvolution(viewGraph, views[i], shape, ops::ShortConvOutput::Plain, in, {}, states[i],
                                            layout, {2, rows, 8}, {}, treeRows, viewTape);
    }
    run(backend, viewGraph);
    bool viewSame = true;
    for (size_t i = 0; i < views.size(); ++i)
      viewSame &= download<uint16_t>(views[i].output, values16) == download<uint16_t>(alone[i].output, values16);
    check(viewSame && download<uint16_t>(viewTape, size_t{rows} * stride) ==
                          download<uint16_t>(aloneTape, size_t{rows} * stride),
          scope + ": a convolution of a view of the wider rows the bits of its own rows'");
  }
}

// Host side, nothing run: a pair's convolutions each write their state's
// columns of the tape and their output in one dispatch, so addConvolutionPair
// refuses two whose columns overlap, by a column or wholly, and two of one
// output; Inkling-2-Mini's keys and values, a convolution's channels apart
// (Inkling2LoopSchedule::convolutionColumn), are encoded either way round.
// And a combine's convolution (verifyParams), which convolves the rows the
// combine computes, refuses a view of wider input rows.
void convPairRefusals(MetalBackend &backend) {
  constexpr uint32_t rows = 3, channels = 1024, stride = 2 * channels;
  const ops::ShortConvShape shape{channels, 0};
  const std::array<ops::LaneRows, 1> lanes{{{0, rows}}};
  const MetalBuffer layout = laneRowsOf(backend, lanes), tree = scratch(backend, rows * sizeof(ops::TreeRow)),
                    tape = scratch(backend, uint64_t{rows} * stride * 2),
                    state = scratch(backend, ops::ShortConvolution::stateBytes(stride));
  const std::vector<MetalBuffer> in{state};
  const auto buffersOf = [&] {
    return ops::ShortConvBuffers{.input = scratch(backend, uint64_t{rows} * channels * 2),
                                 .taps = scratch(backend, uint64_t{channels} * kTaps * 4),
                                 .output = scratch(backend, uint64_t{rows} * channels * 2)};
  };
  const std::array<ops::ShortConvBuffers, 2> pair{buffersOf(), buffersOf()};
  std::array<ops::ShortConvBuffers, 2> oneOutput = pair;
  oneOutput[1].output = oneOutput[0].output;
  // Whether the pair encodes, at its convolutions' state columns `first` and `second`.
  const auto encodes = [&](const std::array<ops::ShortConvBuffers, 2> &buffers, uint32_t first, uint32_t second) {
    const std::array<ops::ShortConvState, 2> states{ops::ShortConvState{first, stride},
                                                    ops::ShortConvState{second, stride}};
    CommandGraph graph;
    try {
      ops::ShortConvolution::addConvolutionPair(graph, buffers, shape, states, in, layout, {1, rows, rows}, tree,
                                                tape);
    } catch (const std::invalid_argument &) {
      return false;
    }
    return graph.dispatches().size() == 1;
  };
  check(encodes(pair, 0, channels) && encodes(pair, channels, 0),
        "short convolution pair: state columns side by side encoded");
  check(!encodes(pair, 0, channels - 1) && !encodes(pair, channels - 1, 0) && !encodes(pair, channels, channels),
        "short convolution pair: overlapping state columns refused");
  check(!encodes(oneOutput, 0, channels), "short convolution pair: one output refused");
  // The combine's convolution: plain, then of a view.
  ops::ShortConvBuffers combined{.taps = pair[0].taps,
                                 .residual = scratch(backend, uint64_t{rows} * channels * 4),
                                 .output = pair[0].output};
  const auto verifies = [&](const ops::ShortConvBuffers &buffers) {
    try {
      static_cast<void>(ops::ShortConvolution::verifyParams(buffers, shape, in, {0, stride}, layout,
                                                            {1, rows, rows}, tree, tape));
    } catch (const std::invalid_argument &) {
      return false;
    }
    return true;
  };
  const bool plain = verifies(combined);
  combined.inputStride = stride;
  const bool strided = verifies(combined);
  combined.inputStride = 0;
  combined.inputColumn = channels;
  check(plain && !strided && !verifies(combined),
        "short convolution of a combine's rows: a view of wider input rows refused");
}

// A verify step's combine running its convolution
// (MoE::addCombineConvolution, moe_combine_convolution) against
// moe_combine into bf16 rows and then the RoundedSum convolution of them
// along the lanes' trees: two lanes, a tree of 8 rows (as convTreeTest's) and
// a chain of 3, beside a row of no lane (every route left out, as a verified
// grouping leaves them); a lane of a verify block's 32 rows, a tree; and a
// lane of one row. 8 routes a row, every seventh left out, on zero residual
// rows (Inkling-2-Mini's) or others. Uninstrumented (`exact`) the output and
// the tape bit for bit, under shader validation, which compiles the kernels
// apart, within a bf16 step; and each lane row's output and tape against the
// fp64 combine and its convolution along the row's path; a row of no lane
// written by neither.
void combineConvolutionTest(MetalBackend &backend, bool exact) {
  struct Case {
    std::string name;
    std::vector<ops::LaneRows> lanes;
    uint32_t rows;
    bool zeroResidual;
  };
  const std::vector<Case> cases{{"two lanes and a row of no lane", {{0, 8}, {8, 3}}, 12, true},
                                {"a lane of 32 rows", {{0, 32}}, 32, false},
                                {"a lane of one row", {{0, 1}}, 1, false}};
  constexpr uint32_t channels = 2304, routes = 8;
  constexpr uint16_t kSentinel = 0x7FC1;
  for (const Case &k : cases) {
    std::mt19937 rng(41 + k.rows);
    const uint32_t rows = k.rows, grouped = rows * routes, lanes = static_cast<uint32_t>(k.lanes.size());
    const ops::ShortConvShape shape{channels, 0};
    const ops::ShortConvState state{128, channels + 384};
    const ConvCase c = convCase(rng, shape, state, rows, lanes);
    // Lane 0 of the first case convTreeTest's tree, of the second a tree of
    // 32 rows, the others chains.
    std::vector<ops::TreeRow> tree(rows, splash::test::chainRow(0));
    for (uint32_t l = 0; l < lanes; ++l)
      for (uint32_t r = 0; r < k.lanes[l].rows; ++r) {
        uint32_t parent = r - 1;
        if (l == 0 && k.lanes[l].rows == 8) parent = std::array<uint32_t, 8>{0, 0, 1, 2, 3, 1, 5, 0}[r];
        if (l == 0 && k.lanes[l].rows == 32 && r > 1 && r % 4 == 0) parent = r - 2;
        const uint32_t first = k.lanes[l].first;
        tree[first + r] = r ? ops::TreeRow{tree[first + parent].ancestors | 1u << r, static_cast<uint16_t>(parent),
                                           static_cast<uint16_t>(tree[first + parent].depth + 1)}
                            : splash::test::chainRow(0);
      }
    std::vector<bool> inLane(rows, false);
    for (const ops::LaneRows &lane : k.lanes)
      for (uint32_t r = lane.first; r < lane.first + lane.rows; ++r) inLane[r] = true;
    // Each route's grouped row, a shuffle of them, ~0u where left out.
    std::vector<uint32_t> routeRows(grouped);
    std::iota(routeRows.begin(), routeRows.end(), 0u);
    std::shuffle(routeRows.begin(), routeRows.end(), rng);
    for (uint32_t i = 0; i < grouped; ++i)
      if (i % 7 == 3 || !inLane[i / routes]) routeRows[i] = ~0u;
    std::vector<float> weights(grouped);
    for (float &w : weights) w = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
    const std::vector<float> expert = bf16Values(randoms(rng, size_t{grouped} * channels, 1.0f));
    const std::vector<float> sumResidual =
        k.zeroResidual ? std::vector<float>(size_t{rows} * channels, 0.0f)
                       : bf16Values(randoms(rng, size_t{rows} * channels, 0.5f));
    std::vector<MetalBuffer> in;
    for (uint32_t l = 0; l < lanes; ++l) in.push_back(upload(backend, bf16s(c.states[l])));
    const MetalBuffer expertBuffer = upload(backend, bf16s(expert)), routeBuffer = upload(backend, routeRows),
                      weightBuffer = upload(backend, weights), sumBuffer = upload(backend, bf16s(sumResidual)),
                      taps = upload(backend, c.taps), residual = upload(backend, c.residual),
                      treeBuffer = upload(backend, tree), layout = laneRowsOf(backend, k.lanes);
    const MoeCombineParams params{rows, channels, routes};
    const ops::LayoutRows layoutRows{lanes, rows, splash::test::mostRows(k.lanes)};
    const auto sentinels = [&](size_t count) { return upload(backend, std::vector<uint16_t>(count, kSentinel)); };
    // moe_combine, then the convolution of its rows.
    const MetalBuffer combined = scratch(backend, size_t{rows} * channels * 2),
                      outputBefore = sentinels(size_t{rows} * channels),
                      tapeBefore = sentinels(size_t{rows} * state.stride);
    CommandGraph before;
    before.add("moe_combine", {expertBuffer, routeBuffer, weightBuffer, sumBuffer, combined}, params,
               {rows, channels / 256, 1});
    ops::ShortConvolution::addConvolution(before, {combined, taps, {}, 0, residual, outputBefore}, shape,
                                          ops::ShortConvOutput::RoundedSum, in, {}, state, layout, layoutRows, {},
                                          treeBuffer, tapeBefore);
    run(backend, before);
    // The combine running the convolution.
    const MetalBuffer output = sentinels(size_t{rows} * channels), tape = sentinels(size_t{rows} * state.stride);
    CommandGraph fused;
    ops::MoE::addCombineConvolution(
        fused, {expertBuffer, routeBuffer, weightBuffer, sumBuffer, params},
        {{{}, taps, {}, 0, residual, output}, shape, in, state, layout, layoutRows, treeBuffer, tape});
    run(backend, fused);
    const auto got = download<uint16_t>(output, size_t{rows} * channels),
               want = download<uint16_t>(outputBefore, size_t{rows} * channels),
               gotTape = download<uint16_t>(tape, size_t{rows} * state.stride),
               wantTape = download<uint16_t>(tapeBefore, size_t{rows} * state.stride);
    // The fp64 combine of row r at channel ch, rounded to the bf16 the
    // convolution reads.
    const auto sum = [&](uint32_t r, uint32_t ch) {
      double value = sumResidual[size_t{r} * channels + ch];
      for (uint32_t slot = 0; slot < routes; ++slot) {
        const uint32_t row = routeRows[r * routes + slot];
        if (row != ~0u) value += double(weights[r * routes + slot]) * expert[size_t{row} * channels + ch];
      }
      return roundBf16(value);
    };
    bool same = true, sameTape = true, ok = true, okTape = true, untouched = true;
    for (uint32_t l = 0; l < lanes; ++l)
      for (uint32_t r = k.lanes[l].first; r < k.lanes[l].first + k.lanes[l].rows; ++r)
        for (uint32_t ch = 0; ch < channels; ++ch) {
          const size_t i = size_t{r} * channels + ch, t = size_t{r} * state.stride + state.column + ch;
          same &= exact ? got[i] == want[i] : closeBf16(f32(got[i]), f32(want[i]));
          sameTape &= exact ? gotTape[t] == wantTape[t] : closeBf16(f32(gotTape[t]), f32(wantTape[t]));
          // Input `depth` along the row's path, or the lane's state's.
          const auto input = [&](int depth) -> double {
            if (depth < 0) return c.states[l][size_t(int(kStateRows) + depth) * state.stride + state.column + ch];
            uint32_t row = r;
            while (tree[row].depth > uint32_t(depth)) row = k.lanes[l].first + tree[row].parent;
            return sum(row, ch);
          };
          double weighted = 0;
          for (uint32_t tap = 0; tap < kTaps; ++tap)
            weighted += c.taps[size_t{ch} * kTaps + (kTaps - 1 - tap)] * input(int(tree[r].depth) - int(tap));
          const double y = (weighted + sum(r, ch)) / std::sqrt(double(kTaps + 1));
          // bf16(y) on the fp32 residual: one bf16 step of y, then the sum's
          // (as convPromptTest's RoundedSum).
          const double total = c.residual[i] + y;
          ok &= std::fabs(f32(got[i]) - total) <= std::fabs(y) / 128 + std::fabs(total) / 128 + 1e-3;
          okTape &= closeBf16(f32(gotTape[t]), sum(r, ch));
        }
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t col = 0; col < state.stride; ++col) {
        const bool convolved = inLane[r] && col >= state.column && col < state.column + channels;
        if (!convolved) untouched &= gotTape[size_t{r} * state.stride + col] == kSentinel;
        if (!inLane[r] && col < channels) untouched &= got[size_t{r} * channels + col] == kSentinel;
      }
    const std::string scope = "MoE combine convolution (" + k.name + ")";
    check(same, scope + (exact ? ": the output bit for bit as the combine's convolution's"
                               : ": the output within a bf16 step of the combine's convolution's"));
    check(sameTape, scope + (exact ? ": the tape bit for bit as the combine's convolution's"
                                   : ": the tape within a bf16 step of the combine's convolution's"));
    check(ok, scope + ": each row the fp64 combine's convolution along its path");
    check(okTape, scope + ": each row's combined input in the tape");
    check(untouched, scope + ": no row of no lane and no other column written");
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
// fp32 ones (the residual inside a layer), of `rows` rows: at most
// BLOCK_RESIDUAL_MIX_QUERY_ROWS, a threadgroup a row and query, or more, a
// threadgroup a row (the _rows kernels).
void mixTest(MetalBackend &backend, uint32_t stack, uint32_t queries, bool currentFloat, uint32_t rows) {
  std::mt19937 rng(stack * 10 + queries + (currentFloat ? 100 : 0) + rows * 1000);
  MixCase m;
  m.hidden = 2304;
  m.stack = stack;
  m.queries = queries;
  m.rows = rows;
  m.planeRows = rows + 3;
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
                            " current rows, " + std::to_string(rows) + " rows (a threadgroup a " +
                            (rows <= BLOCK_RESIDUAL_MIX_QUERY_ROWS ? "row and query)" : "row)");
  const ops::BlockResidualStates states{stack ? upload(backend, bf16s(m.stackRows)) : MetalBuffer{}, m.planeRows,
                                        currentFloat ? upload(backend, m.current) : upload(backend, bf16s(m.current)),
                                        currentFloat};
  const ops::BlockResidualShape shape{m.hidden, stack, queries};
  const MetalBuffer queryBuffer = upload(backend, m.queryValues),
                    output = scratch(backend, size_t{m.rows} * queries * m.hidden * 4);
  // Plain, and with each query's output normalized for the message gates
  // too, which leaves the outputs as they are.
  const MetalBuffer normalizedRows = scratch(backend, size_t{m.rows} * queries * m.hidden * 2),
                    outputBeside = scratch(backend, size_t{m.rows} * queries * m.hidden * 4);
  CommandGraph graph;
  const std::array<ops::BlockResidualMixInput, 2> calls{
      ops::BlockResidualMixInput{states, queryBuffer, shape, output, {}},
      ops::BlockResidualMixInput{states, queryBuffer, shape, outputBeside, normalizedRows}};
  ops::BlockResidualMix::addMix(graph, std::span(&calls[0], 1), m.rows);
  ops::BlockResidualMix::addMix(graph, std::span(&calls[1], 1), m.rows);
  run(backend, graph);
  const auto got = download<float>(output, size_t{m.rows} * queries * m.hidden);
  const auto gotNormalized = download<uint16_t>(normalizedRows, size_t{m.rows} * queries * m.hidden);
  bool ok = true, okNormalized = true;
  for (uint32_t r = 0; r < m.rows; ++r) {
    const auto want = mixReference(m, r);
    for (size_t i = 0; i < want.size(); ++i) ok &= closeF32(got[size_t{r} * queries * m.hidden + i], want[i]);
    // Each query's output RMS-normalized without weights, rounded to bf16.
    for (uint32_t q = 0; q < queries; ++q) {
      double squares = 0;
      for (uint32_t d = 0; d < m.hidden; ++d) squares += want[size_t{q} * m.hidden + d] * want[size_t{q} * m.hidden + d];
      const double inverse = 1 / std::sqrt(squares / m.hidden + 1e-6);
      for (uint32_t d = 0; d < m.hidden; ++d)
        okNormalized &= closeBf16(f32(gotNormalized[(size_t{r} * queries + q) * m.hidden + d]),
                                  want[size_t{q} * m.hidden + d] * inverse);
    }
  }
  check(ok, scope);
  check(okNormalized && download<float>(outputBeside, size_t{m.rows} * queries * m.hidden) == got,
        scope + ", each query's output normalized for the message gates beside it");
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

// The test producer's parameters (rms_norm_epilogue_test.metal).
struct RmsNormEpilogueTestParams {
  RmsNormEpilogueParams norm;
  uint32_t simdgroup;
};
static_assert(sizeof(RmsNormEpilogueTestParams) == 20, "the test's parameters are 20 bytes on both sides");

// A producer's RMS norm epilogue (kernels/common/rms_norm_epilogue.h) in its
// test producer (rms_norm_epilogue_test): tiles of 32 columns, each storing
// bf16(a + b) of every row and its sums of squares, the last writing the
// rows' weighted RMS norm, against the fp64 norm of the stored rows and
// against norm_rms's of them (Normalization::addRms), each within a bf16
// step; the counter back at zero, so a second dispatch on the same scratch
// writes the same bits. A decode step's rows (1, 2, 3, 16, 128), the 35B's,
// Inkling's and the 27B's widths (2048, 2304, 5120), F32 and bf16 weights,
// and a threadgroup's tiles or a simdgroup's (to 16 rows).
void normEpilogueTest(MetalBackend &backend) {
  for (const uint32_t width : {2048u, 2304u, 5120u})
    for (const uint32_t rows : {1u, 2u, 3u, 16u, 128u})
      for (const bool floatWeights : {true, false})
        for (const bool simdgroup : {false, true}) {
          if (simdgroup && rows > 16) continue;
          std::mt19937 rng(width + rows * 7 + (floatWeights ? 1 : 0) + (simdgroup ? 2 : 0));
          const uint32_t tiles = width / 32;
          const size_t values = size_t{rows} * width;
          const std::vector<float> a = bf16Values(randoms(rng, values, 1.0f)), b = bf16Values(randoms(rng, values, 0.5f));
          std::vector<float> weights = randoms(rng, width, 0.5f);
          for (float &w : weights) w += 1.0f;
          if (!floatWeights) weights = bf16Values(weights);
          const MetalBuffer weightBuffer = floatWeights ? upload(backend, weights) : upload(backend, bf16s(weights));
          const ops::RmsNormEpilogue epilogue{{weightBuffer, floatWeights},
                                              scratch(backend, values * 2),
                                              scratch(backend, ops::RmsNormEpilogue::partialsBytes(tiles, rows)),
                                              upload(backend, std::vector<uint32_t>{0})};
          const RmsNormEpilogueTestParams params{epilogue.params(rows, width, tiles), simdgroup ? 1u : 0u};
          const MetalBuffer aBuffer = upload(backend, bf16s(a)), bBuffer = upload(backend, bf16s(b)),
                            stored = scratch(backend, values * 2), reference = scratch(backend, values * 2);
          const auto bound = epilogue.bindings();
          const auto produce = [&](CommandGraph &graph) {
            graph.add("rms_norm_epilogue_test", {aBuffer, bBuffer, stored, bound[0], bound[1], bound[2], bound[3]},
                      params, {tiles, 1, 1}, {256, 1, 1});
          };
          CommandGraph first;
          produce(first);
          ops::Normalization::addRms(first, stored, epilogue.weights, reference, width, rows);
          run(backend, first);
          const auto got = download<uint16_t>(epilogue.output, values), kernel = download<uint16_t>(reference, values),
                     rowValues = download<uint16_t>(stored, values);
          const bool zero = download<uint32_t>(epilogue.counter, 1)[0] == 0;
          CommandGraph second;
          produce(second);
          run(backend, second);
          const bool again = download<uint16_t>(epilogue.output, values) == got &&
                             download<uint32_t>(epilogue.counter, 1)[0] == 0;
          bool ok = true, okKernel = true;
          for (uint32_t r = 0; r < rows; ++r) {
            double squares = 0;
            for (uint32_t c = 0; c < width; ++c) {
              const double x = f32(rowValues[size_t{r} * width + c]);
              squares += x * x;
            }
            const double inverse = 1 / std::sqrt(squares / width + 1e-6);
            for (uint32_t c = 0; c < width; ++c) {
              const size_t i = size_t{r} * width + c;
              ok &= closeBf16(f32(got[i]), f32(rowValues[i]) * inverse * weights[c]);
              okKernel &= closeBf16(f32(got[i]), f32(kernel[i]));
            }
          }
          const std::string scope = "RMS norm epilogue of " + std::to_string(rows) + " rows of " +
                                    std::to_string(width) + ", " + (floatWeights ? "F32" : "bf16") + " weights, " +
                                    (simdgroup ? "a simdgroup's tiles" : "a threadgroup's tiles");
          check(ok && okKernel, scope + ": the stored rows' norm within a bf16 step of fp64's and of norm_rms's");
          check(zero && again, scope + ": the counter back at zero, and a second dispatch the same bits");
        }
}

// The mix's scores and the convolution's sums against fp64, as 1.7.4 took
// them and as this tree does (mix_conv_precision_test.metal): a mix of one
// query over five states, random and at magnitudes 1e-3 to 1e3, its states
// of scales 1 to 30 as a stack's blocks are, its query of sd 0.2, each
// output value's error in its row's root mean square (a value near zero
// counts by its row's size), max and mean, and its weights' relative error;
// and a static convolution of 4 taps at the same magnitudes, Sum onto a bf16
// residual, its outputs' error in units of their bf16 step. Printed, the
// tree's against 1.7.4's; it fails where the tree's error passes 1.7.4's by
// more than a tenth.
// The precision kernels' parameters (mix_conv_precision_test.metal).
struct MixPrecisionParams {
  BlockResidualMixParams mix;
  uint32_t onePass;
};
struct ConvPrecisionParams {
  uint32_t channels, ordered;
};
void mixConvPrecisionTest(MetalBackend &backend) {
  constexpr uint32_t hidden = 2304, rows = 32, stack = 4, channels = 2304;
  std::printf("op precision against fp64: magnitude, 1.7.4's and this tree's max / mean error\n");
  bool noWorse = true;
  for (const double magnitude : {1e-3, 1e-1, 1.0, 1e1, 1e3}) {
    std::mt19937 rng(static_cast<uint32_t>(1000 + std::log10(magnitude) * 10));
    MixCase m;
    m.hidden = hidden;
    m.stack = stack;
    m.queries = 1;
    m.rows = rows;
    m.planeRows = rows;
    m.stackRows = randoms(rng, size_t{rows} * stack * hidden, 1.0f);
    for (size_t i = 0; i < m.stackRows.size(); ++i)
      m.stackRows[i] *= float(magnitude * std::pow(3.0, double(i / (size_t{rows} * hidden))));
    m.stackRows = bf16Values(m.stackRows);
    m.current = randoms(rng, size_t{rows} * hidden, float(30 * magnitude));
    m.queryValues = randoms(rng, hidden, 0.2f);
    const MetalBuffer stackBuffer = upload(backend, bf16s(m.stackRows)), current = upload(backend, m.current),
                      query = upload(backend, m.queryValues);
    const BlockResidualMixParams p{rows, hidden, stack, 1, rows * hidden,
                                   static_cast<float>(1.0 / std::sqrt(double(hidden)))};
    double maxError[2]{}, meanError[2]{}, maxWeight[2]{};
    for (const uint32_t onePass : {0u, 1u}) {
      const MetalBuffer weights = scratch(backend, size_t{rows} * BLOCK_RESIDUAL_MIX_MAXIMUM_STATES * 4),
                        output = scratch(backend, size_t{rows} * hidden * 4);
      CommandGraph graph;
      graph.add("mix_precision_test", {stackBuffer, current, query, weights, output},
                MixPrecisionParams{p, onePass}, {rows, 1, 1}, {BLOCK_RESIDUAL_MIX_THREADS, 1, 1});
      run(backend, graph);
      const auto got = download<float>(output, size_t{rows} * hidden);
      const auto gotWeights = download<float>(weights, size_t{rows} * BLOCK_RESIDUAL_MIX_MAXIMUM_STATES);
      for (uint32_t r = 0; r < rows; ++r) {
        const auto want = mixReference(m, r);
        double squares = 0;
        for (const double v : want) squares += v * v;
        const double rms = std::sqrt(squares / hidden);
        for (uint32_t d = 0; d < hidden; ++d) {
          const double error = std::fabs(got[size_t{r} * hidden + d] - want[d]) / std::max(std::fabs(want[d]), rms);
          maxError[onePass] = std::max(maxError[onePass], error);
          meanError[onePass] += error / (double(rows) * hidden);
        }
        // The fp64 weights of the row's states.
        const uint32_t states = stack + 1;
        std::vector<double> scores(states);
        for (uint32_t s = 0; s < states; ++s) {
          const float *x = s < stack ? &m.stackRows[(size_t{s} * rows + r) * hidden] : &m.current[size_t{r} * hidden];
          double xx = 0, qx = 0;
          for (uint32_t d = 0; d < hidden; ++d) {
            xx += double(x[d]) * x[d];
            qx += double(m.queryValues[d]) * x[d];
          }
          scores[s] = qx / std::sqrt(xx / hidden + 1e-6) / std::sqrt(double(hidden));
        }
        const double top = *std::max_element(scores.begin(), scores.end());
        double total = 0;
        for (double &v : scores) total += (v = std::exp(v - top));
        for (uint32_t s = 0; s < states; ++s)
          maxWeight[onePass] = std::max(maxWeight[onePass],
                                        std::fabs(gotWeights[size_t{r} * BLOCK_RESIDUAL_MIX_MAXIMUM_STATES + s] -
                                                  scores[s] / total) / (scores[s] / total));
      }
    }
    std::printf("  mix scores, magnitude %g: outputs %.3g / %.3g against %.3g / %.3g, weights %.3g against %.3g\n",
                magnitude, maxError[0], meanError[0], maxError[1], meanError[1], maxWeight[0], maxWeight[1]);
    noWorse &= maxError[1] <= 1.1 * maxError[0] + 1e-9 && meanError[1] <= 1.1 * meanError[0] + 1e-12;
    // The convolution of a chain of rows at the magnitude.
    const std::vector<float> input = bf16Values(randoms(rng, size_t{rows} * channels, float(magnitude)));
    const std::vector<float> taps = randoms(rng, size_t{channels} * kTaps, 0.4f);
    const std::vector<float> residual = bf16Values(randoms(rng, size_t{rows} * channels, float(magnitude)));
    const MetalBuffer inputBuffer = upload(backend, bf16s(input)), tapsBuffer = upload(backend, taps),
                      residualBuffer = upload(backend, bf16s(residual));
    double convMax[2]{}, convMean[2]{};
    for (const uint32_t ordered : {0u, 1u}) {
      const MetalBuffer output = scratch(backend, size_t{rows} * channels * 4);
      CommandGraph graph;
      graph.add("conv_precision_test", {inputBuffer, tapsBuffer, residualBuffer, output}, ConvPrecisionParams{channels, ordered},
                {rows, (channels + 255) / 256, 1}, {256, 1, 1});
      run(backend, graph);
      const auto got = download<float>(output, size_t{rows} * channels);
      for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t ch = 0; ch < channels; ++ch) {
          double sum = 0;
          for (uint32_t k = 0; k < kTaps; ++k)
            sum += double(taps[size_t{ch} * kTaps + (kTaps - 1 - k)]) *
                   (r >= k ? double(input[size_t{r - k} * channels + ch]) : 0.0);
          const double y = (sum + input[size_t{r} * channels + ch]) / std::sqrt(double(kTaps + 1));
          // In bf16 steps of y, the rounding the kernel takes.
          const double step = std::max(std::fabs(y), 1e-30) / 128;
          const double error = std::fabs((got[size_t{r} * channels + ch] - residual[size_t{r} * channels + ch]) - y) / step;
          convMax[ordered] = std::max(convMax[ordered], error);
          convMean[ordered] += error / (double(rows) * channels);
        }
    }
    std::printf("  convolution, magnitude %g: %.3g / %.3g bf16 steps against %.3g / %.3g\n", magnitude, convMax[0],
                convMean[0], convMax[1], convMean[1]);
    noWorse &= convMax[1] <= 1.1 * convMax[0] + 1e-9 && convMean[1] <= 1.1 * convMean[0] + 1e-12;
  }
  check(noWorse, "the mix's one-pass scores and the convolution's ordered sums no less accurate against fp64 than "
                 "1.7.4's, at magnitudes 1e-3 to 1e3");
}

// A decode step's FFN mix running its mixer's output convolution
// (BlockResidualStates::convolution, the _convolved kernels) against that
// convolution's own dispatch (Sum along the lanes' trees) and then the mix of
// its rows: Inkling-2-Mini's FFN mix (one query over three states) with the
// rows its gate reads normalized, or through its norm. Two lanes and a row of
// no lane (a tree of 8 rows, as convTreeTest's, and a chain of 3) on a
// residual; a lane of 16 rows at a block boundary (no residual); a lane of
// one row. Uninstrumented (`exact`) the current rows and every output
// bit for bit, under shader validation, which compiles the kernels apart,
// within a bf16 step; the tape, copies, bit for bit either way; in one
// dispatch; a row of no lane's current row and tape not written. And eight
// lanes of 9 rows, past BLOCK_RESIDUAL_MIX_QUERY_ROWS (the _rows kernels),
// which do not convolve: the two dispatches, and the mix asked to convolve
// them refused (ops::BlockResidualMix::convolves).
void convolvedMixTest(MetalBackend &backend, bool exact) {
  struct Case {
    std::string name;
    std::vector<ops::LaneRows> lanes;
    uint32_t rows;
    bool residual;
  };
  std::vector<Case> cases{{"two lanes and a row of no lane", {{0, 8}, {8, 3}}, 12, true},
                          {"a lane of 16 rows at a block boundary", {{0, 16}}, 16, false},
                          {"a lane of one row", {{0, 1}}, 1, true}};
  Case wide{"eight lanes of 9 rows", {}, 72, true};
  for (uint32_t l = 0; l < 8; ++l) wide.lanes.push_back({l * 9, 9});
  cases.push_back(wide);
  constexpr uint32_t hidden = 2304, stack = 2;
  constexpr uint16_t kSentinel = 0x7FC1;
  for (const Case &k : cases) {
    std::mt19937 rng(97 + k.rows);
    const uint32_t rows = k.rows, lanes = static_cast<uint32_t>(k.lanes.size());
    const ops::ShortConvShape shape{hidden, 0};
    const ops::ShortConvState state{128, hidden + 384};
    const ConvCase c = convCase(rng, shape, state, rows, lanes);
    std::vector<ops::TreeRow> tree(rows, splash::test::chainRow(0));
    std::vector<bool> inLane(rows, false);
    for (uint32_t l = 0; l < lanes; ++l)
      for (uint32_t r = 0; r < k.lanes[l].rows; ++r) {
        const uint32_t first = k.lanes[l].first;
        const uint32_t parent =
            l == 0 && k.lanes[l].rows == 8 ? std::array<uint32_t, 8>{0, 0, 1, 2, 3, 1, 5, 0}[r] : r - 1;
        tree[first + r] = r ? ops::TreeRow{tree[first + parent].ancestors | 1u << r, static_cast<uint16_t>(parent),
                                           static_cast<uint16_t>(tree[first + parent].depth + 1)}
                            : splash::test::chainRow(0);
        inLane[first + r] = true;
      }
    std::vector<MetalBuffer> in;
    for (uint32_t l = 0; l < lanes; ++l) in.push_back(upload(backend, bf16s(c.states[l])));
    const MetalBuffer input = upload(backend, bf16s(c.input)), taps = upload(backend, c.taps),
                      residual = k.residual ? upload(backend, bf16s(c.residual)) : MetalBuffer{},
                      treeBuffer = upload(backend, tree), layout = laneRowsOf(backend, k.lanes);
    const ops::LayoutRows layoutRows{lanes, rows, splash::test::mostRows(k.lanes)};
    const uint32_t planeRows = rows + 3;
    const MetalBuffer stackRows = upload(backend, bf16s(randoms(rng, size_t{planeRows} * stack * hidden, 1.0f))),
                      queries = upload(backend, randoms(rng, hidden, 0.2f));
    const ops::NormWeights norm{upload(backend, randoms(rng, hidden, 0.5f)), true};
    // The current rows before: what a row of no lane mixes.
    const std::vector<float> current = randoms(rng, size_t{rows} * hidden, 3.0f);
    const ops::BlockResidualShape mixShape{hidden, stack, 1};
    for (const bool normed : {false, true}) {
      struct Run {
        MetalBuffer current, tape, output, normalized;
        size_t dispatches = 0;
      };
      const auto runMix = [&](bool fused) {
        Run r{upload(backend, current), upload(backend, std::vector<uint16_t>(size_t{rows} * state.stride, kSentinel)),
              scratch(backend, size_t{rows} * hidden * 4),
              upload(backend, std::vector<uint16_t>(size_t{rows} * hidden, kSentinel))};
        const ops::ShortConvBuffers buffers{input, taps, {}, 0, residual, r.current};
        const ops::ShortConvVerify convolution{buffers, shape, in, state, layout, layoutRows, treeBuffer, r.tape};
        CommandGraph graph;
        if (!fused)
          ops::ShortConvolution::addConvolution(graph, buffers, shape, ops::ShortConvOutput::Sum, in, {}, state,
                                                layout, layoutRows, {}, treeBuffer, r.tape);
        const ops::BlockResidualStates states{stackRows, planeRows, r.current, true, fused ? &convolution : nullptr};
        if (normed) {
          ops::BlockResidualMix::addMixNorm(graph, states, queries, std::span(&norm, 1), std::span(&r.normalized, 1),
                                            mixShape, rows);
        } else {
          const ops::BlockResidualMixInput mix{states, queries, mixShape, r.output, r.normalized};
          ops::BlockResidualMix::addMix(graph, std::span(&mix, 1), rows);
        }
        r.dispatches = graph.dispatches().size();
        run(backend, graph);
        return r;
      };
      const std::string scope = std::string("block residual mix running its convolution (") + k.name +
                                (normed ? ", through its norm)" : ", its gate's rows normalized)");
      const Run before = runMix(false);
      if (!ops::BlockResidualMix::convolves(rows)) {
        // More rows than a threadgroup mixes a row and query: the convolution's dispatch, then the _rows mix, as a
        // prompt chunk runs them; a mix asked to convolve them is refused.
        bool refused = false;
        try {
          static_cast<void>(runMix(true));
        } catch (const std::invalid_argument &) {
          refused = true;
        }
        check(refused && before.dispatches == 2,
              scope + ": a call of more rows than a row and query's mix takes runs the convolution's dispatch and "
                      "the mix's, and a mix asked to convolve them is refused");
        continue;
      }
      const Run fused = runMix(true);
      const auto values = size_t{rows} * hidden, tapeValues = size_t{rows} * state.stride;
      const auto gotCurrent = download<float>(fused.current, values),
                 wantCurrent = download<float>(before.current, values);
      const auto gotTape = download<uint16_t>(fused.tape, tapeValues),
                 wantTape = download<uint16_t>(before.tape, tapeValues);
      const auto gotOutput = download<float>(fused.output, values), wantOutput = download<float>(before.output, values);
      const auto gotNormalized = download<uint16_t>(fused.normalized, values),
                 wantNormalized = download<uint16_t>(before.normalized, values);
      // The tape holds copies of the inputs, and its columns past the
      // convolution's their sentinels: bit for bit either way.
      bool same = gotTape == wantTape, untouched = true;
      for (size_t i = 0; i < values; ++i) {
        same &= gotCurrent[i] == wantCurrent[i] ||
                (!exact && (closeF32(gotCurrent[i], wantCurrent[i]) || closeBf16(gotCurrent[i], wantCurrent[i])));
        same &= gotNormalized[i] == wantNormalized[i] ||
                (!exact && closeBf16(f32(gotNormalized[i]), f32(wantNormalized[i])));
        if (!normed) same &= gotOutput[i] == wantOutput[i] || (!exact && closeBf16(gotOutput[i], wantOutput[i]));
      }
      for (uint32_t r = 0; r < rows; ++r) {
        if (inLane[r]) continue;
        for (uint32_t col = 0; col < state.stride; ++col) untouched &= gotTape[size_t{r} * state.stride + col] == kSentinel;
        for (uint32_t d = 0; d < hidden; ++d)
          untouched &= gotCurrent[size_t{r} * hidden + d] == current[size_t{r} * hidden + d];
      }
      check(same && fused.dispatches == 1 && before.dispatches == 2,
            scope + (exact ? ": the current rows, the tape and the outputs bit for bit as the convolution's and "
                             "the mix's dispatches', in one"
                           : ": the current rows, the tape and the outputs within a bf16 step of the convolution's "
                             "and the mix's dispatches', in one"));
      check(untouched, scope + ": no row of no lane convolved");
    }
  }
}

// Two mixes in one call (a gated application's messages, four queries over
// two states, and its mixer's inputs, one or three queries over three)
// against each one's own call, of 5 rows (a threadgroup a row and query) and
// of BLOCK_RESIDUAL_MIX_QUERY_ROWS + 3 (a threadgroup a row and mix): each
// output bit for bit, the one kernel's either way.
void mixPairTest(MetalBackend &backend) {
  for (const uint32_t queries : {1u, 3u})
    for (const uint32_t rows : {5u, BLOCK_RESIDUAL_MIX_QUERY_ROWS + 3}) {
      std::mt19937 rng(71 + queries + rows);
      constexpr uint32_t hidden = 2304;
      const std::array<ops::BlockResidualShape, 2> shapes{ops::BlockResidualShape{hidden, 1, 4},
                                                          ops::BlockResidualShape{hidden, 2, queries}};
      std::vector<ops::BlockResidualMixInput> mixes;
      for (const ops::BlockResidualShape &shape : shapes) {
        const uint32_t planeRows = rows + 3;
        mixes.push_back({{upload(backend, bf16s(bf16Values(randoms(rng, size_t{planeRows} * shape.stack * hidden, 1.0f)))),
                          planeRows, upload(backend, bf16s(bf16Values(randoms(rng, size_t{rows} * hidden, 3.0f))))},
                         upload(backend, randoms(rng, size_t{shape.queries} * hidden, 0.2f)), shape,
                         scratch(backend, size_t{rows} * shape.queries * hidden * 4), {}});
      }
      CommandGraph pair;
      ops::BlockResidualMix::addMix(pair, mixes, rows);
      check(pair.dispatches().size() == 1, "two block residual mixes in one call: one dispatch");
      run(backend, pair);
      std::vector<std::vector<float>> together;
      for (const ops::BlockResidualMixInput &mix : mixes)
        together.push_back(download<float>(mix.output, size_t{rows} * mix.shape.queries * hidden));
      bool same = true;
      for (size_t m = 0; m < mixes.size(); ++m) {
        CommandGraph alone;
        ops::BlockResidualMix::addMix(alone, std::span(&mixes[m], 1), rows);
        run(backend, alone);
        same &= download<float>(mixes[m].output, size_t{rows} * mixes[m].shape.queries * hidden) == together[m];
      }
      check(same, "two block residual mixes in one call, 4 and " + std::to_string(queries) + " queries of " +
                      std::to_string(rows) + " rows: each output bit for bit as its own call's");
    }
}

// A call's gates on the same rows (MessageGateInput): x rows of a mix of
// three queries and four messages a row, gate g's x query `column(g)` (a
// full layer's q, k and v each its own; a sliding layer's all query 0) and
// its message g, each of its own weights (of bf16 values, as the bf16
// checkpoint's are, stored as a GGUF keeps them: F32 or BF16 rows), norm and
// output; `hidden` values a row and a rank of `rank`, Inkling-2-Mini's by
// default.
struct GateCase {
  static constexpr uint32_t kHidden = 2304, kRank = 64;
  uint32_t rows = 0, gates = 0;
  bool sharedX = false;
  uint32_t hidden = kHidden, rank = kRank;
  std::vector<float> x, message;
  std::vector<std::vector<float>> down, forget, gain, norm;

  GateCase(std::mt19937 &rng, uint32_t rows_, uint32_t gates_, bool sharedX_, uint32_t hidden_ = kHidden,
           uint32_t rank_ = kRank)
      : rows(rows_), gates(gates_), sharedX(sharedX_), hidden(hidden_), rank(rank_) {
    x = randoms(rng, size_t{rows} * 3 * hidden, 1.5f);
    message = randoms(rng, size_t{rows} * 4 * hidden, 0.7f);
    for (uint32_t g = 0; g < gates; ++g) {
      down.push_back(bf16Values(randoms(rng, size_t{rank} * 2 * hidden, 0.02f)));
      forget.push_back(bf16Values(randoms(rng, size_t{hidden} * rank, 0.1f)));
      gain.push_back(bf16Values(randoms(rng, size_t{hidden} * rank, 0.1f)));
      norm.push_back(randoms(rng, hidden, 0.5f));
    }
  }
  [[nodiscard]] uint32_t column(uint32_t g) const { return sharedX ? 0 : g; }
  [[nodiscard]] ops::MessageGateWeights weights(MetalBackend &backend, uint32_t g, bool bfloat16) const {
    const auto rows = [&](const std::vector<float> &values, uint32_t outputs, uint32_t inputs) {
      return bfloat16 ? ops::QuantizedSegment::rows(GGUF_FMT_BF16, outputs, inputs, upload(backend, bf16s(values)))
                      : ops::QuantizedSegment::floats(outputs, inputs, upload(backend, values));
    };
    return {rows(down[g], rank, 2 * hidden), rows(forget[g], hidden, rank), rows(gain[g], hidden, rank)};
  }
  // Gate g's row r against fp64, with the source's bf16 roundings of the
  // down projection's input and output, before the norm's weights' rounding.
  [[nodiscard]] std::vector<double> reference(uint32_t g, uint32_t r) const {
    const float *xr = &x[(size_t{r} * 3 + column(g)) * hidden], *mr = &message[(size_t{r} * 4 + g) * hidden];
    double xs = 0, ms = 0;
    for (uint32_t d = 0; d < hidden; ++d) {
      xs += double(xr[d]) * xr[d];
      ms += double(mr[d]) * mr[d];
    }
    xs = 1 / std::sqrt(xs / hidden + 1e-6);
    ms = 1 / std::sqrt(ms / hidden + 1e-6);
    std::vector<double> concat(2 * hidden), low(rank);
    for (uint32_t d = 0; d < hidden; ++d) {
      concat[d] = roundBf16(xr[d] * xs);
      concat[hidden + d] = roundBf16(mr[d] * ms);
    }
    for (uint32_t k = 0; k < rank; ++k) {
      double dot = 0;
      for (uint32_t i = 0; i < 2 * hidden; ++i) dot += down[g][size_t{k} * 2 * hidden + i] * concat[i];
      low[k] = roundBf16(dot);
    }
    std::vector<double> v(hidden);
    double squares = 0;
    for (uint32_t d = 0; d < hidden; ++d) {
      double f = 0, e = 0;
      for (uint32_t k = 0; k < rank; ++k) {
        f += forget[g][size_t{d} * rank + k] * low[k];
        e += gain[g][size_t{d} * rank + k] * low[k];
      }
      v[d] = (1 - std::tanh(f)) * xr[d] + std::tanh(e) * mr[d];
      squares += v[d] * v[d];
    }
    const double inverse = 1 / std::sqrt(squares / hidden + 1e-6);
    for (uint32_t d = 0; d < hidden; ++d) v[d] *= inverse * norm[g][d];
    return v;
  }
};

// The paths a test call runs: the engine's fused kernels (ops::MessageGate),
// or the reference path (message_gate_reference.h), whose kernels the
// production-and-test library holds.
enum class GatePath : uint8_t { Fused, Reference };

// The reference path's gates of `rows` rows: a gate at a time, its rows' and
// messages' norms and concatenation (message_gate_test_prepare) into
// buffers.concat [rows][2 hidden] bf16, its down projection into buffers.low
// [rows][rank] bf16 and its forget and gain projections into buffers.forget
// and buffers.gain [rows][hidden] fp32 on the simdgroup float tile
// (ops::addGgufFloat), and its gated rows' norm (message_gate_test_norm) into
// its output.
void addReferenceGates(CommandGraph &graph, const ops::MessageGateBuffers &buffers,
                       std::span<const ops::MessageGateInput> gates, uint32_t hidden, uint32_t rows) {
  const uint32_t rank = gates.front().weights.down.outputSize;
  const ops::FloatTile tile = ops::FloatTile::Simdgroup;
  for (const ops::MessageGateInput &gate : gates) {
    const MessageGateReferenceParams p{rows, hidden, buffers.xStride, gate.xColumn, buffers.messageStride,
                                       gate.messageColumn, rank};
    graph.add("message_gate_test_prepare", {buffers.x, buffers.message, buffers.concat}, p, {rows, 1, 1},
              {MESSAGE_GATE_REFERENCE_THREADS, 1, 1});
    ops::addGgufFloat(graph, buffers.concat, gate.weights.down, buffers.low, rows, rank, 0,
                      ops::FloatOutput::BFloat16, tile);
    const std::array projections{ops::FloatProjection{&gate.weights.forget, buffers.forget, hidden, 0, tile},
                                 ops::FloatProjection{&gate.weights.gain, buffers.gain, hidden, 0, tile}};
    ops::addGgufFloat(graph, buffers.low, projections, rows, ops::FloatOutput::Float32);
    graph.add("message_gate_test_norm",
              {buffers.x, buffers.message, buffers.forget, buffers.gain, gate.norm.buffer, gate.output}, p,
              {rows, 1, 1}, {MESSAGE_GATE_REFERENCE_THREADS, 1, 1});
  }
}

// The fused gates' inputs normalized (MessageGateBuffers::normalizedX,
// normalizedMessage): `count` fp32 rows of `slots` slots of `hidden` values
// into bf16 rows of their layout, as the mixes writing them normalize them
// (message_gate_test_normalize).
void addNormalized(CommandGraph &graph, const MetalBuffer &rows, const MetalBuffer &normalized, uint32_t slots,
                   uint32_t hidden, uint32_t count) {
  const MessageGateReferenceParams p{count, hidden, slots * hidden, 0, 0, 0, 0};
  graph.add("message_gate_test_normalize", {rows, normalized}, p, {count, slots, 1},
            {MESSAGE_GATE_REFERENCE_THREADS, 1, 1});
}

// Runs rows [first, first + count) of the case's gates as one call on
// `path`, into `outputs` (a buffer a gate, [rows][hidden] bf16), over views
// of the shared rows, a call of the fused gates (MessageGate::readsNormalized)
// first normalizing them as the mixes writing them do; the counters stay
// zero.
void runGates(MetalBackend &backend, const GateCase &c, const MetalBuffer &x, const MetalBuffer &message,
              std::span<const ops::MessageGateWeights> weights, std::span<const MetalBuffer> norms,
              std::span<const MetalBuffer> outputs, const MetalBuffer &counters, uint32_t first, uint32_t count,
              GatePath path = GatePath::Fused) {
  const uint32_t hidden = c.hidden, rank = c.rank;
  std::vector<ops::MessageGateInput> gates;
  for (uint32_t g = 0; g < c.gates; ++g)
    gates.push_back({weights[g], {norms[g], true}, c.column(g) * hidden, g * hidden,
                     backend.view(outputs[g], uint64_t{first} * hidden * 2, uint64_t{count} * hidden * 2)});
  const bool normalized = path == GatePath::Fused && ops::MessageGate::readsNormalized(count);
  const ops::MessageGateBuffers buffers{
      backend.view(x, uint64_t{first} * 3 * hidden * 4, uint64_t{count} * 3 * hidden * 4),
      3 * hidden,
      backend.view(message, uint64_t{first} * 4 * hidden * 4, uint64_t{count} * 4 * hidden * 4),
      4 * hidden,
      normalized ? scratch(backend, size_t{count} * 3 * hidden * 2) : MetalBuffer{},
      normalized ? scratch(backend, size_t{count} * 4 * hidden * 2) : MetalBuffer{},
      scratch(backend, size_t{count} * 2 * hidden * 2),
      scratch(backend, size_t{c.gates} * count * rank * 2),
      scratch(backend, size_t{c.gates} * count * hidden * 4),
      scratch(backend, size_t{count} * hidden * 4),
      counters};
  CommandGraph graph;
  if (normalized) {
    addNormalized(graph, buffers.x, buffers.normalizedX, 3, hidden, count);
    addNormalized(graph, buffers.message, buffers.normalizedMessage, 4, hidden, count);
  }
  if (path == GatePath::Reference) addReferenceGates(graph, buffers, gates, hidden, count);
  else ops::MessageGate::add(graph, buffers, gates, hidden, count);
  run(backend, graph);
}

// The kernels a call of `rows` rows on `path` runs.
std::string gateKernels(uint32_t rows, GatePath path) {
  return path == GatePath::Reference              ? "float projections"
         : rows <= ops::MessageGate::kFusedRows ? "fused"
                                                : "tiled";
}

// Each gate of a call of one or three gates (a layer's FFN gate; its q, k and
// v gates of x queries of their own or of one) against fp64 with the
// source's bf16 roundings of the down projection's input and output: fused
// (a decode step's rows: one group of a power of two, or groups of 4 rows
// and a part of one), tiled (more rows,
// a prompt chunk's: whole tiles and a part of one) and through the reference
// path's float projections, of F32 weights and of BF16, the counters back at
// zero. The weights' values are the same in either format, and so is every
// product, so the two formats' outputs are the same bits. Inkling-2-Mini's
// hidden and rank, or others the tiled kernel takes: fewer 64-value steps,
// and a rank short of its 64 columns, whose simdgroups past it read no down
// weights and whose lanes past it no forget or gain weights.
void gateTest(MetalBackend &backend, uint32_t rows, uint32_t gates, bool sharedX,
              GatePath path = GatePath::Fused, uint32_t hidden = GateCase::kHidden,
              uint32_t rank = GateCase::kRank) {
  std::mt19937 rng(31 + gates + (sharedX ? 7 : 0));
  const GateCase c(rng, rows, gates, sharedX, hidden, rank);
  const MetalBuffer x = upload(backend, c.x), message = upload(backend, c.message);
  std::vector<MetalBuffer> norms;
  for (uint32_t g = 0; g < gates; ++g) norms.push_back(upload(backend, c.norm[g]));
  const MetalBuffer counters = scratch(backend, ops::MessageGate::kCountersBytes);
  std::memset(counters.contents(), 0, counters.sizeBytes());
  // Each format's outputs, gate after gate.
  std::vector<std::vector<uint16_t>> got[2];
  bool ok = true;
  for (const bool bfloat16 : {false, true}) {
    std::vector<ops::MessageGateWeights> weights;
    std::vector<MetalBuffer> outputs;
    for (uint32_t g = 0; g < gates; ++g) {
      weights.push_back(c.weights(backend, g, bfloat16));
      outputs.push_back(scratch(backend, size_t{rows} * hidden * 2));
    }
    runGates(backend, c, x, message, weights, norms, outputs, counters, 0, rows, path);
    for (uint32_t g = 0; g < gates; ++g) {
      got[bfloat16].push_back(download<uint16_t>(outputs[g], size_t{rows} * hidden));
      for (uint32_t r = 0; r < rows; ++r) {
        const auto want = c.reference(g, r);
        for (uint32_t d = 0; d < hidden; ++d)
          ok &= closeBf16(f32(got[bfloat16][g][size_t{r} * hidden + d]), want[d], 2e-3);
      }
    }
  }
  const auto zeros = download<uint32_t>(counters, ops::MessageGate::kCountersBytes / 4);
  const std::string kind = gates == 1 ? "" : sharedX ? " of one x" : " of three x";
  const std::string shape = hidden == GateCase::kHidden && rank == GateCase::kRank
                                ? ""
                                : ", hidden " + std::to_string(hidden) + " and rank " + std::to_string(rank);
  check(ok && got[0] == got[1] && std::all_of(zeros.begin(), zeros.end(), [](uint32_t value) { return value == 0; }),
        "message gates, " + std::to_string(gates) + " a call" + kind + ", of " + std::to_string(rows) + " rows (" +
            gateKernels(rows, path) + shape +
            ") through their norms, x and the messages at columns of wider rows, F32 and BF16 weights alike");
}

// A decode step's gates in groups of each power of two (MessageGate::groupRows):
// a row's outputs are the same bits whatever its call's rows (1, 2, 3, 6, 13,
// 16 and 17, a group and a part), as a lane's rows are whatever lanes run
// beside it. Uninstrumented alone (`exact`): shader validation compiles the
// kernels apart.
void gateGroupBitsTest(MetalBackend &backend, bool exact) {
  if (!exact) return;
  std::mt19937 rng(43);
  constexpr uint32_t most = 17, gates = 3;
  const GateCase c(rng, most, gates, false);
  const MetalBuffer x = upload(backend, c.x), message = upload(backend, c.message);
  std::vector<MetalBuffer> norms;
  for (uint32_t g = 0; g < gates; ++g) norms.push_back(upload(backend, c.norm[g]));
  const MetalBuffer counters = scratch(backend, ops::MessageGate::kCountersBytes);
  std::memset(counters.contents(), 0, counters.sizeBytes());
  std::vector<ops::MessageGateWeights> weights;
  for (uint32_t g = 0; g < gates; ++g) weights.push_back(c.weights(backend, g, true));
  // Each call's first row of each gate.
  std::vector<std::vector<uint16_t>> first;
  for (const uint32_t rows : {1u, 2u, 3u, 6u, 13u, 16u, 17u}) {
    std::vector<MetalBuffer> outputs;
    for (uint32_t g = 0; g < gates; ++g) outputs.push_back(scratch(backend, size_t{rows} * c.hidden * 2));
    runGates(backend, c, x, message, weights, norms, outputs, counters, 0, rows);
    std::vector<uint16_t> row;
    for (uint32_t g = 0; g < gates; ++g) {
      const auto values = download<uint16_t>(outputs[g], c.hidden);
      row.insert(row.end(), values.begin(), values.end());
    }
    first.push_back(std::move(row));
  }
  bool same = true;
  for (const auto &row : first) same &= row == first.front();
  check(same, "message gates of a decode step: a row's bits the same in a call of 1, 2, 3, 6, 13, 16 and 17 rows");
}

// The gates' three paths on the same rows: a call of three gates of more rows
// than MessageGate::kFusedRows through the reference path's float projections
// (a gate at a time, the engine's path until the fused kernels took every
// call), the same call on the tiled kernel (a prompt chunk's fused gates:
// whole tiles and a part of one) and calls of at most kFusedRows of those
// rows (a decode step's: the fused kernels), each over views of the same
// buffers, row by row, each pair. Rounding points are
// the same; what moves is the order of the sums: the rows' and messages'
// squares (the fused kernels add a row's by quads), the rank values',
// forget's, gain's and the norm's. Through them a normalized input's or a
// rank value's bf16 rounding moves by a step now and then, which moves its
// row's every value by a little of the row's size: a difference counts in
// bf16 steps of the larger of the two values and their row's root mean
// square (a value near zero moves by its row's steps, not its own). An M5 Max
// (2026-11-22) measured 2,200 of 1,382,400 values differing between the
// float projections and the fused kernels (a share of 1.6e-3), each by one
// such step at most; bounded at twice those, 2 steps and a share of 3.2e-3.
// An M6 (2026-11-23) measured the tiled kernel's rows against the float
// projections' at 2,815 values (2.0e-3) and against the fused kernels' at
// 1,688, each by one step at most. A gate that read another row, message,
// column, rank value or gate's weights moves values by their row's size,
// hundreds of steps (a planted bug that normalized a message by its row's
// scale moved 1,366,693 values by up to 499).
constexpr double kPathsSteps = 2, kPathsShare = 3.2e-3;
void gatePathsTest(MetalBackend &backend) {
  std::mt19937 rng(37);
  const uint32_t rows = ops::MessageGate::kFusedRows + 72, gates = 3, hidden = GateCase::kHidden;
  const GateCase c(rng, rows, gates, false);
  std::vector<ops::MessageGateWeights> weights;
  std::vector<MetalBuffer> norms, general, tiled, fused;
  for (uint32_t g = 0; g < gates; ++g) {
    weights.push_back(c.weights(backend, g, true));
    norms.push_back(upload(backend, c.norm[g]));
    general.push_back(scratch(backend, size_t{rows} * hidden * 2));
    tiled.push_back(scratch(backend, size_t{rows} * hidden * 2));
    fused.push_back(scratch(backend, size_t{rows} * hidden * 2));
  }
  const MetalBuffer x = upload(backend, c.x), message = upload(backend, c.message),
                    counters = scratch(backend, ops::MessageGate::kCountersBytes);
  std::memset(counters.contents(), 0, counters.sizeBytes());
  runGates(backend, c, x, message, weights, norms, general, counters, 0, rows,
           GatePath::Reference);
  runGates(backend, c, x, message, weights, norms, tiled, counters, 0, rows);
  for (uint32_t first = 0; first < rows; first += ops::MessageGate::kFusedRows)
    runGates(backend, c, x, message, weights, norms, fused, counters, first,
             std::min(ops::MessageGate::kFusedRows, rows - first));
  const auto compare = [&](std::span<const MetalBuffer> one, std::span<const MetalBuffer> other,
                           const std::string &names) {
    double steps = 0;
    size_t differ = 0, values = 0;
    for (uint32_t g = 0; g < gates; ++g) {
      const auto a = download<uint16_t>(one[g], size_t{rows} * hidden),
                 b = download<uint16_t>(other[g], size_t{rows} * hidden);
      values += a.size();
      for (uint32_t r = 0; r < rows; ++r) {
        double squares = 0;
        for (uint32_t d = 0; d < hidden; ++d) {
          const double value = f32(a[size_t{r} * hidden + d]);
          squares += value * value;
        }
        const double size = std::sqrt(squares / hidden);
        for (size_t i = size_t{r} * hidden; i < size_t{r + 1} * hidden; ++i) {
          if (a[i] == b[i]) continue;
          ++differ;
          const double larger = std::max({std::fabs(double(f32(a[i]))), std::fabs(double(f32(b[i]))), size});
          int exponent = 0;
          std::frexp(larger, &exponent);
          steps = std::max(steps, std::fabs(double(f32(a[i])) - f32(b[i])) / std::ldexp(1.0, exponent - 8));
        }
      }
    }
    const double share = double(differ) / values;
    check(steps <= kPathsSteps && share <= kPathsShare,
          "message gates, 3 a call, of " + std::to_string(rows) + " rows: " + names + ": " + std::to_string(differ) +
              " of " + std::to_string(values) + " values differ, at most " + std::to_string(steps) +
              " bf16 steps (bounds " + std::to_string(kPathsShare) + " of them, " + std::to_string(kPathsSteps) +
              " steps)");
  };
  compare(general, fused, "the float projections' rows and the fused kernels'");
  compare(general, tiled, "the float projections' rows and the tiled kernel's");
  compare(tiled, fused, "the tiled kernel's rows and the fused kernels'");
}

// MessageGate::add refuses, saying why, the calls the fused kernels do not
// take, which no Inkling-2-Mini checkpoint makes (its target checks a layer's
// gates at load): gates of F32 and BF16 rows in one call, a rank short of a
// run of 8, and a hidden short of a 64-value step.
void gateRefusalTest(MetalBackend &backend) {
  const auto refused = [&](uint32_t hidden, uint32_t rank, bool mixed, const std::string &why) {
    std::mt19937 rng(41);
    const GateCase c(rng, 13, 3, false, hidden, rank);
    const MetalBuffer x = upload(backend, c.x), message = upload(backend, c.message),
                      counters = scratch(backend, ops::MessageGate::kCountersBytes);
    std::vector<ops::MessageGateWeights> weights;
    std::vector<MetalBuffer> norms, outputs;
    for (uint32_t g = 0; g < c.gates; ++g) {
      weights.push_back(c.weights(backend, g, mixed && g > 0));
      norms.push_back(upload(backend, c.norm[g]));
      outputs.push_back(scratch(backend, size_t{c.rows} * hidden * 2));
    }
    try {
      runGates(backend, c, x, message, weights, norms, outputs, counters, 0, c.rows);
    } catch (const std::invalid_argument &error) {
      return std::string(error.what()).find(why) != std::string::npos;
    }
    return false;
  };
  check(refused(GateCase::kHidden, GateCase::kRank, true, "all of one format") &&
            refused(GateCase::kHidden, 4, false, "a rank of 8 to 64") &&
            refused(96, GateCase::kRank, false, "whole 64-value steps"),
        "message gates the fused kernels do not take refused, saying why: F32 and BF16 in one call, a rank of 4, a "
        "hidden of 96");
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
      const bool bitwise = argc == 3 && std::string(argv[2]) == "bitwise";
      if (argc != 2 && !bitwise) throw std::runtime_error("usage: convolution-mix-kernels METALLIB [bitwise]");
      MetalBackend backend(argv[1]);
      if (bitwise) {
        // Uninstrumented: the combine's convolution and the pair's alone, exact.
        if (std::getenv("MTL_SHADER_VALIDATION")) throw std::runtime_error("bitwise runs without shader validation");
        combineConvolutionTest(backend, true);
        for (const ops::ShortConvShape shape : {ops::ShortConvShape{1024, 0}, ops::ShortConvShape{1024, 8}})
          for (const bool oneProjection : {false, true}) convPairTest(backend, shape, oneProjection, true);
        std::cout << (failures ? "convolution and mix kernel bitwise tests FAILED\n"
                               : "convolution and mix kernel bitwise tests PASS\n");
        return failures ? 1 : 0;
      }
      // A key or value projection's dynamic convolution (8 heads of 128) and
      // an output's static one.
      for (const ops::ShortConvShape shape : {ops::ShortConvShape{2304, 0}, ops::ShortConvShape{1024, 8}}) {
        convPromptTest(backend, shape);
        convTreeTest(backend, shape);
      }
      combineConvolutionTest(backend, !std::getenv("MTL_SHADER_VALIDATION"));
      // A sliding layer's keys and values (static taps) and a full layer's
      // (dynamic).
      for (const ops::ShortConvShape shape : {ops::ShortConvShape{1024, 0}, ops::ShortConvShape{1024, 8}})
        for (const bool oneProjection : {false, true})
          convPairTest(backend, shape, oneProjection, !std::getenv("MTL_SHADER_VALIDATION"));
      convPairRefusals(backend);
      for (const bool currentFloat : {false, true}) {
        for (const uint32_t queries : {1u, 3u, 4u})
          for (uint32_t stack = 1; stack <= 4; ++stack)
            for (const uint32_t rows : {5u, BLOCK_RESIDUAL_MIX_QUERY_ROWS + 3})
              mixTest(backend, stack, queries, currentFloat, rows);
        for (const uint32_t rows : {5u, BLOCK_RESIDUAL_MIX_QUERY_ROWS + 3}) mixTest(backend, 0, 1, currentFloat, rows);
      }
      mixPairTest(backend);
      convolvedMixTest(backend, !std::getenv("MTL_SHADER_VALIDATION"));
      mixConvPrecisionTest(backend);
      normEpilogueTest(backend);
      // A decode step's groups of each power of two (1 and 2 rows, 3 in 4) and of 4 rows with a part of one
      // (6, 13, 16 and the most rows), and a prompt chunk's tiles.
      for (const uint32_t rows : {1u, 2u, 3u, 6u, 13u, 16u, ops::MessageGate::kFusedRows,
                                  ops::MessageGate::kFusedRows + 1}) {
        gateTest(backend, rows, 1, true);
        gateTest(backend, rows, 3, false);
      }
      for (const uint32_t rows : {13u, ops::MessageGate::kFusedRows + 1}) gateTest(backend, rows, 3, true);
      for (const uint32_t rows : {13u, ops::MessageGate::kFusedRows + 1})
        gateTest(backend, rows, 3, false, GatePath::Reference);
      // The tiled kernel at other shapes: three 64-value steps a half at its
      // full rank and at ranks short of its 64 columns, its last tile of 1
      // row and of 31; and Inkling-2-Mini's hidden at rank 24, of 129 rows (of
      // 159, one value in 1,099,008 lands 1.15 times the tolerance from fp64
      // on the float projections and this kernel alike, 2.3 bf16 steps of its
      // row: a rounding the fp64 reference takes the other way, as one value
      // of 200 rows at rank 64 does on either path).
      for (const uint32_t rank : {64u, 8u, 24u, 56u})
        for (const uint32_t rows : {ops::MessageGate::kFusedRows + 1, ops::MessageGate::kFusedRows + 31}) {
          gateTest(backend, rows, 1, true, GatePath::Fused, 192, rank);
          gateTest(backend, rows, 3, false, GatePath::Fused, 192, rank);
        }
      gateTest(backend, ops::MessageGate::kFusedRows + 1, 1, true, GatePath::Fused, GateCase::kHidden,
               24);
      gateTest(backend, ops::MessageGate::kFusedRows + 1, 3, false, GatePath::Fused,
               GateCase::kHidden, 24);
      gatePathsTest(backend);
      gateGroupBitsTest(backend, !std::getenv("MTL_SHADER_VALIDATION"));
      gateRefusalTest(backend);
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
