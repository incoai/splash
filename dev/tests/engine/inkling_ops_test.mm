// Inkling 2's operators (kernels/shared/short_convolution.metal,
// block_residual.metal and loop_control.metal) through ops::ShortConvolution,
// ops::BlockResidual and ops::LoopControl, against fp64 CPU references of the
// model's computation:
// - the short convolution, static and with a global layer's dynamic per-head
//   taps, at each input and state element type: its rows against the
//   reference over a history, a prompt in chunks against the prompt in one
//   (bit for bit), several sequences of one dispatch against each alone, the
//   history after a sequence's first rows (a state captured inside its
//   command) and its accumulation into a residual;
// - verify steps: a tree whose rows' past inputs follow their paths, the
//   accepted rows of a prompt's tail decoded as a tree, a chain or a row at a
//   time against the prompt prefilled whole (bit for bit), and the commit of
//   each lane's accepted path for every unit of several lanes in one
//   dispatch, a layer's convolutions side by side;
// - the bf16 state's difference from an fp32 state's (printed);
// - the depth attention over 2 to 6 states with 1, 3 and 4 queries, from
//   separate buffers and from a bank, one of its outputs over a state;
// - the loop control of 1 and 3 streams, F32 and BF16 weights, in place and
//   over the messages;
// - the RMS norms between the fp32 residual and its bf16 rows
//   (ops::Normalization::addRmsRows), with F32 and BF16 multipliers;
// - the refusal of buffers one element short and of shapes the kernels do
//   not hold.
// With `bench`, the GPU time of each operator's dispatches at a decode step's
// and a prefill chunk's rows instead.
#include "ChainRows.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/BlockResidual.h"
#include "metal/abi/LoopControl.h"
#include "metal/abi/ShortConvolution.h"
#include "ops/BlockResidual.hpp"
#include "ops/LaneRows.hpp"
#include "ops/LoopControl.hpp"
#include "ops/Normalization.hpp"
#include "ops/ShortConvolution.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
namespace ops = splash::ops;
using ops::FloatElement;

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
// A value as an element of `element` holds it.
float as(float v, FloatElement element) { return element == FloatElement::Float32 ? v : f32(bf16(v)); }
std::string name(FloatElement element) { return ops::elementToken(element); }

MetalBuffer scratch(MetalBackend &backend, uint64_t bytes) {
  MetalBuffer buffer = backend.allocateBuffer(std::max<uint64_t>(bytes, 16), BufferStorage::Shared, "inkling-ops-test");
  std::memset(buffer.contents(), 0, buffer.sizeBytes());
  return buffer;
}
template <class T> MetalBuffer upload(MetalBackend &backend, const std::vector<T> &values) {
  MetalBuffer buffer = scratch(backend, values.size() * sizeof(T));
  std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(T));
  return buffer;
}
// fp32 values as `element`s.
MetalBuffer uploadAs(MetalBackend &backend, const std::vector<float> &values, FloatElement element) {
  if (element == FloatElement::Float32) return upload(backend, values);
  std::vector<uint16_t> halves(values.size());
  std::transform(values.begin(), values.end(), halves.begin(), bf16);
  return upload(backend, halves);
}
std::vector<float> downloadAs(const MetalBuffer &buffer, size_t count, FloatElement element, uint64_t offset = 0) {
  std::vector<float> out(count);
  const auto *bytes = static_cast<const uint8_t *>(buffer.contents()) + offset;
  if (element == FloatElement::Float32) {
    std::memcpy(out.data(), bytes, count * sizeof(float));
  } else {
    std::vector<uint16_t> halves(count);
    std::memcpy(halves.data(), bytes, count * sizeof(uint16_t));
    std::transform(halves.begin(), halves.end(), out.begin(), f32);
  }
  return out;
}
std::vector<float> download(const MetalBuffer &buffer, size_t count) {
  return downloadAs(buffer, count, FloatElement::Float32);
}
std::vector<float> randoms(std::mt19937 &rng, size_t n, float scale) {
  std::normal_distribution<float> d(0.0f, scale);
  std::vector<float> v(n);
  for (float &x : v) x = d(rng);
  return v;
}
void run(MetalBackend &backend, CommandGraph &graph) {
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}
MetalBuffer layoutOf(MetalBackend &backend, std::span<const ops::LaneRows> lanes) {
  return upload(backend, std::vector<ops::LaneRows>(lanes.begin(), lanes.end()));
}
bool throws(const std::function<void()> &f) {
  try {
    f();
  } catch (const std::invalid_argument &) {
    return true;
  }
  return false;
}
bool sameBits(const std::vector<float> &a, const std::vector<float> &b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// ---------------------------------------------------------------------------
// Short convolution.

constexpr uint32_t kTaps = SHORT_CONVOLUTION_TAPS, kHistory = SHORT_CONVOLUTION_HISTORY;

// A convolution and the rows a test runs through it: each row's input and
// dynamic taps in one packed row, as a global layer's K side holds them
// (inputs at 0, the dynamic taps after them), `stride` values a row.
struct ConvCase {
  ops::ShortConvolutionShape shape;
  uint32_t stride = 0;
  [[nodiscard]] uint32_t dynamicWidth() const { return shape.heads() * kTaps; }
  [[nodiscard]] std::string label() const {
    return std::string(shape.dynamic() ? "dynamic" : "static") + " convolution of " + std::to_string(shape.channels) +
           " channels, " + name(shape.input) + " inputs, " + name(shape.state) + " state";
  }
};

// A sequence's packed rows (inputs and dynamic taps) and its history before
// them, both as their elements hold them.
struct ConvSequence {
  std::vector<float> rows;
  std::vector<float> history;
};

ConvSequence convSequence(std::mt19937 &rng, const ConvCase &c, uint32_t rows, bool zeroHistory) {
  ConvSequence s{randoms(rng, size_t{rows} * c.stride, 1.0f), randoms(rng, size_t{kHistory} * c.shape.channels, 1.0f)};
  // Dynamic taps of the size the projection gives them.
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t i = 0; i < c.dynamicWidth(); ++i) s.rows[size_t{r} * c.stride + c.shape.channels + i] *= 0.3f;
  for (float &v : s.rows) v = as(v, c.shape.input);
  for (float &v : s.history) v = zeroHistory ? 0.0f : as(v, c.shape.state);
  return s;
}

// The kernel's output of a row of channel `ch`, in its own fp32 operations,
// from its own input `x`, its past inputs (lag 1 to 3, as the history keeps
// them) and its dynamic taps (null for a static convolution); `magnitude`
// the sum of the terms' magnitudes, a bound on the rounding.
float convolutionOf(const ConvCase &c, const std::vector<float> &taps, uint32_t ch, float x, const float *past,
                    const float *dynamic, double &magnitude) {
  float sum = 0.0f;
  magnitude = std::fabs(x);
  for (uint32_t j = 0; j < kTaps; ++j) {
    const uint32_t lag = kTaps - 1 - j;
    const float weight = taps[size_t{ch} * kTaps + j] + (dynamic ? dynamic[lag] : 0.0f);
    const float value = lag ? past[lag - 1] : x;
    sum += weight * value;
    magnitude += std::fabs(double(weight) * value);
  }
  magnitude *= c.shape.scale();
  return (x + sum) * c.shape.scale();
}

// The reference output of a sequence's rows over its history, and its history
// after them.
struct ConvReference {
  std::vector<float> output;
  std::vector<double> magnitude;
  std::vector<float> history;
};
ConvReference convReference(const ConvCase &c, const std::vector<float> &taps, const ConvSequence &s, uint32_t rows) {
  const uint32_t C = c.shape.channels;
  ConvReference ref{std::vector<float>(size_t{rows} * C), std::vector<double>(size_t{rows} * C),
                    std::vector<float>(size_t{kHistory} * C)};
  const auto input = [&](int t, uint32_t ch) -> float {
    return t >= 0 ? as(s.rows[size_t(t) * c.stride + ch], c.shape.state)
                  : s.history[size_t(int(kHistory) + t) * C + ch];
  };
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t ch = 0; ch < C; ++ch) {
      const float past[kHistory] = {input(int(r) - 1, ch), input(int(r) - 2, ch), input(int(r) - 3, ch)};
      const float *dynamic =
          c.shape.dynamic() ? &s.rows[size_t{r} * c.stride + C + (ch / c.shape.headChannels) * kTaps] : nullptr;
      ref.output[size_t{r} * C + ch] = convolutionOf(c, taps, ch, s.rows[size_t{r} * c.stride + ch], past, dynamic,
                                                     ref.magnitude[size_t{r} * C + ch]);
    }
  for (uint32_t j = 0; j < kHistory; ++j)
    for (uint32_t ch = 0; ch < C; ++ch) ref.history[size_t{j} * C + ch] = input(int(rows) - int(kHistory) + int(j), ch);
  return ref;
}

bool closeTo(const std::vector<float> &got, const std::vector<float> &want, const std::vector<double> &magnitude) {
  for (size_t i = 0; i < want.size(); ++i)
    if (!(std::fabs(double(got[i]) - want[i]) <= 1e-6 * magnitude[i] + 1e-30)) return false;
  return true;
}

// One sequence's buffers on the GPU: its packed rows, histories before and
// after, and the output.
struct ConvBuffers {
  MetalBuffer rows, historyIn, historyOut, output;
};
ConvBuffers convBuffers(MetalBackend &backend, const ConvCase &c, const ConvSequence &s, uint32_t rows) {
  return {uploadAs(backend, s.rows, c.shape.input), uploadAs(backend, s.history, c.shape.state),
          scratch(backend, ops::shortConvolutionHistoryBytes(c.shape.state, c.shape.channels)),
          scratch(backend, uint64_t{rows} * c.shape.channels * sizeof(float))};
}
ops::ShortConvolutionBuffers convolutionBuffers(MetalBackend &backend, const ConvCase &c, const MetalBuffer &rows,
                                                const MetalBuffer &taps, const MetalBuffer &output,
                                                uint32_t first = 0) {
  const uint64_t element = ops::elementBytes(c.shape.input), rowBytes = uint64_t{c.stride} * element;
  const uint64_t base = first * rowBytes, length = rows.sizeBytes() - base;
  ops::ShortConvolutionBuffers b{
      {backend.view(rows, base, length), c.stride}, {}, taps, {output, c.shape.channels}, false};
  if (c.shape.dynamic())
    b.dynamic = {backend.view(rows, base + uint64_t{c.shape.channels} * element, length - c.shape.channels * element),
                 c.stride};
  return b;
}

// A sequence's rows in one dispatch against the reference, and the same rows
// in chunks, carrying the history from each to the next, bit for bit; the
// history after the first rows of a chunk (a capture) against the history
// after a sequence of those rows; the output accumulated into a residual.
void convolutionPrefillTest(MetalBackend &backend, const ConvCase &c, bool zeroHistory) {
  std::mt19937 rng(11 + c.shape.channels + (c.shape.dynamic() ? 1 : 0) + 2 * uint32_t(c.shape.input) +
                   4 * uint32_t(c.shape.state));
  const uint32_t rows = 37, C = c.shape.channels;
  const std::string scope = c.label() + (zeroHistory ? " from the sequence's start" : " over a history");
  const ConvSequence s = convSequence(rng, c, rows, zeroHistory);
  std::vector<float> taps = randoms(rng, size_t{C} * kTaps, 0.5f);
  // A few channels with the GGUF's largest taps (up to 76.5).
  for (uint32_t ch = 0; ch < C; ch += 97) taps[size_t{ch} * kTaps + ch % kTaps] = 70.0f;
  const ConvReference ref = convReference(c, taps, s, rows);
  const MetalBuffer tapsBuffer = upload(backend, taps);
  const ConvBuffers b = convBuffers(backend, c, s, rows);
  const std::array<ops::LaneRows, 1> one{{{0, rows}}};
  {
    CommandGraph graph;
    ops::ShortConvolution::addConvolution(graph, convolutionBuffers(backend, c, b.rows, tapsBuffer, b.output),
                                          {std::span(&b.historyIn, 1), std::span(&b.historyOut, 1), C}, c.shape,
                                          layoutOf(backend, one), {1, rows, rows});
    run(backend, graph);
  }
  const std::vector<float> whole = download(b.output, size_t{rows} * C);
  check(closeTo(whole, ref.output, ref.magnitude), scope + ": rows against the reference");
  check(sameBits(downloadAs(b.historyOut, size_t{kHistory} * C, c.shape.state), ref.history),
        scope + ": the history after the rows");

  // In chunks of 1, 2, 5, 13 and the rest, each from the history the one
  // before wrote: the same bits.
  {
    const std::vector<uint32_t> chunks{1, 2, 5, 13, rows - 21};
    MetalBuffer output = scratch(backend, uint64_t{rows} * C * sizeof(float));
    std::array<MetalBuffer, 2> histories{scratch(backend, b.historyOut.sizeBytes()),
                                         scratch(backend, b.historyOut.sizeBytes())};
    MetalBuffer history = b.historyIn;
    uint32_t first = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
      CommandGraph graph;
      const std::array<ops::LaneRows, 1> chunk{{{0, chunks[i]}}};
      ops::ShortConvolutionBuffers buffers = convolutionBuffers(backend, c, b.rows, tapsBuffer, output, first);
      buffers.output.buffer =
          backend.view(output, uint64_t{first} * C * sizeof(float), uint64_t{chunks[i]} * C * sizeof(float));
      ops::ShortConvolution::addConvolution(graph, buffers,
                                            {std::span(&history, 1), std::span(&histories[i % 2], 1), C}, c.shape,
                                            layoutOf(backend, chunk), {1, chunks[i], chunks[i]});
      run(backend, graph);
      history = histories[i % 2];
      first += chunks[i];
    }
    check(sameBits(download(output, size_t{rows} * C), whole), scope + ": a prompt in five chunks, bit for bit");
    check(sameBits(downloadAs(history, size_t{kHistory} * C, c.shape.state), ref.history),
          scope + ": the history after the last chunk");
  }

  // The history after a sequence's first rows, inside its command.
  bool okCapture = true;
  for (const uint32_t captured : {1u, 2u, 3u, 4u, 30u}) {
    MetalBuffer out = scratch(backend, b.historyOut.sizeBytes());
    CommandGraph graph;
    ops::ShortConvolution::addHistory(graph, {b.rows, c.stride}, b.historyIn, out, C, c.shape, captured);
    run(backend, graph);
    okCapture &=
        sameBits(downloadAs(out, size_t{kHistory} * C, c.shape.state), convReference(c, taps, s, captured).history);
  }
  check(okCapture, scope + ": the history after a sequence's first 1, 2, 3, 4 and 30 rows");

  // Added to a residual.
  {
    const std::vector<float> residual = randoms(rng, size_t{rows} * C, 2.0f);
    MetalBuffer output = upload(backend, residual);
    ops::ShortConvolutionBuffers buffers = convolutionBuffers(backend, c, b.rows, tapsBuffer, output);
    buffers.accumulate = true;
    CommandGraph graph;
    ops::ShortConvolution::addConvolution(graph, buffers, {std::span(&b.historyIn, 1), {}, C}, c.shape,
                                          layoutOf(backend, one), {1, rows, rows});
    run(backend, graph);
    const std::vector<float> got = download(output, size_t{rows} * C);
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) ok &= got[i] == residual[i] + whole[i];
    check(ok, scope + ": accumulated into a residual");
  }
}

// Three sequences of 9, 1 and 20 rows in one dispatch, each from its own
// history, against each alone, bit for bit, with their histories after.
void convolutionSequencesTest(MetalBackend &backend, const ConvCase &c) {
  std::mt19937 rng(23);
  const uint32_t C = c.shape.channels;
  const std::vector<uint32_t> counts{9, 1, 20};
  const uint32_t rows = 30;
  const std::vector<float> taps = randoms(rng, size_t{C} * kTaps, 0.5f);
  const MetalBuffer tapsBuffer = upload(backend, taps);
  // The sequences' rows packed in order, from their own sequences' rows.
  std::vector<ConvSequence> sequences;
  std::vector<float> packed;
  std::vector<MetalBuffer> in, out;
  std::vector<ops::LaneRows> layout;
  for (const uint32_t n : counts) {
    sequences.push_back(convSequence(rng, c, n, false));
    layout.push_back({static_cast<uint32_t>(packed.size() / c.stride), n});
    packed.insert(packed.end(), sequences.back().rows.begin(), sequences.back().rows.end());
    in.push_back(uploadAs(backend, sequences.back().history, c.shape.state));
    out.push_back(scratch(backend, ops::shortConvolutionHistoryBytes(c.shape.state, C)));
  }
  const MetalBuffer rowsBuffer = uploadAs(backend, packed, c.shape.input),
                    output = scratch(backend, uint64_t{rows} * C * sizeof(float));
  CommandGraph graph;
  ops::ShortConvolution::addConvolution(graph, convolutionBuffers(backend, c, rowsBuffer, tapsBuffer, output),
                                        {in, out, C}, c.shape, layoutOf(backend, layout), {3, rows, 20});
  run(backend, graph);
  const std::vector<float> got = download(output, size_t{rows} * C);
  bool ok = true, okHistory = true;
  for (size_t i = 0; i < counts.size(); ++i) {
    const ConvBuffers alone = convBuffers(backend, c, sequences[i], counts[i]);
    CommandGraph single;
    const std::array<ops::LaneRows, 1> one{{{0, counts[i]}}};
    ops::ShortConvolution::addConvolution(single, convolutionBuffers(backend, c, alone.rows, tapsBuffer, alone.output),
                                          {std::span(&alone.historyIn, 1), std::span(&alone.historyOut, 1), C}, c.shape,
                                          layoutOf(backend, one), {1, counts[i], counts[i]});
    run(backend, single);
    const std::vector<float> want = download(alone.output, size_t{counts[i]} * C);
    ok &= std::memcmp(got.data() + size_t{layout[i].first} * C, want.data(), want.size() * sizeof(float)) == 0;
    okHistory &= sameBits(downloadAs(out[i], size_t{kHistory} * C, c.shape.state),
                          downloadAs(alone.historyOut, size_t{kHistory} * C, c.shape.state));
  }
  check(ok, c.label() + ": three sequences in one dispatch, each as alone");
  check(okHistory, c.label() + ": three sequences' histories after their rows");
}

// A verify block as a tree: row r follows parents[r] (row 0 the anchor).
std::vector<ops::TreeRow> treeOf(std::span<const uint32_t> parents) {
  std::vector<ops::TreeRow> tree(parents.size());
  for (uint32_t r = 0; r < parents.size(); ++r)
    tree[r] = r ? ops::TreeRow{tree[parents[r]].ancestors | 1u << r, static_cast<uint16_t>(parents[r]),
                               static_cast<uint16_t>(tree[parents[r]].depth + 1)}
                : splash::test::chainRow(0);
  return tree;
}

// The prompt's last `tail` rows decoded after a prefill of the rest, against
// the prompt prefilled whole, bit for bit: the rows as a verify block, a
// tree whose path holds them in order among rows of other inputs, the
// commit of all of its path and of a part of it; and the rows one at a time
// as verify blocks of one row, each committed before the next.
void convolutionDecodeTest(MetalBackend &backend, const ConvCase &c) {
  std::mt19937 rng(31 + c.shape.channels);
  const uint32_t prompt = 23, tail = 5, head = prompt - tail, C = c.shape.channels;
  const std::string scope = c.label() + ", decode after a prefill";
  const ConvSequence s = convSequence(rng, c, prompt, false);
  const std::vector<float> taps = randoms(rng, size_t{C} * kTaps, 0.5f);
  const MetalBuffer tapsBuffer = upload(backend, taps);
  const ConvReference whole = convReference(c, taps, s, prompt);
  const uint64_t historyBytes = ops::shortConvolutionHistoryBytes(c.shape.state, C);

  // The prefill of the head.
  const ConvBuffers b = convBuffers(backend, c, s, prompt);
  MetalBuffer afterHead = scratch(backend, historyBytes);
  {
    const std::array<ops::LaneRows, 1> one{{{0, head}}};
    CommandGraph graph;
    ops::ShortConvolution::addConvolution(graph, convolutionBuffers(backend, c, b.rows, tapsBuffer, b.output),
                                          {std::span(&b.historyIn, 1), std::span(&afterHead, 1), C}, c.shape,
                                          layoutOf(backend, one), {1, head, head});
    run(backend, graph);
  }
  // The tail's rows at 0, 2, 3, 5 and 7 of a block of 9: rows 1 (a branch
  // off the anchor), 4 (off row 2), 6 (off row 3) and 8 (off row 7) take
  // other inputs.
  const std::vector<uint32_t> parents{0, 0, 0, 2, 2, 3, 3, 5, 7};
  const std::vector<uint32_t> path{0, 2, 3, 5, 7};
  const auto tree = treeOf(parents);
  const auto blockRows = static_cast<uint32_t>(parents.size());
  std::vector<float> block = randoms(rng, size_t{blockRows} * c.stride, 1.0f);
  for (float &v : block) v = as(v, c.shape.input);
  for (uint32_t i = 0; i < tail; ++i)
    std::copy_n(s.rows.begin() + size_t{head + i} * c.stride, c.stride, block.begin() + size_t{path[i]} * c.stride);
  const MetalBuffer blockBuffer = uploadAs(backend, block, c.shape.input),
                    blockOutput = scratch(backend, uint64_t{blockRows} * C * sizeof(float)),
                    tape = scratch(backend, uint64_t{blockRows} * C * ops::elementBytes(c.shape.state)),
                    treeBuffer = upload(backend, tree);
  const std::array<ops::LaneRows, 1> lane{{{0, blockRows}}};
  {
    CommandGraph graph;
    ops::ShortConvolution::addConvolution(graph, convolutionBuffers(backend, c, blockBuffer, tapsBuffer, blockOutput),
                                          {std::span(&afterHead, 1), {}, C}, c.shape, layoutOf(backend, lane),
                                          {1, blockRows, blockRows}, {treeBuffer, {tape, C}});
    run(backend, graph);
  }
  const std::vector<float> got = download(blockOutput, size_t{blockRows} * C);
  // The prompt prefilled whole.
  std::vector<float> full;
  {
    const ConvBuffers w = convBuffers(backend, c, s, prompt);
    const std::array<ops::LaneRows, 1> one{{{0, prompt}}};
    CommandGraph graph;
    ops::ShortConvolution::addConvolution(graph, convolutionBuffers(backend, c, w.rows, tapsBuffer, w.output),
                                          {std::span(&w.historyIn, 1), {}, C}, c.shape, layoutOf(backend, one),
                                          {1, prompt, prompt});
    run(backend, graph);
    full = download(w.output, size_t{prompt} * C);
  }
  check(closeTo(full, whole.output, whole.magnitude), scope + ": the prompt prefilled whole against the reference");
  bool okPath = true;
  for (uint32_t i = 0; i < tail; ++i)
    okPath &= std::memcmp(got.data() + size_t{path[i]} * C, full.data() + size_t{head + i} * C, C * sizeof(float)) == 0;
  check(okPath, scope + ": a tree's path rows against the prompt's, bit for bit");
  // Every row against the reference of its path's inputs.
  bool okTree = true, differs = false;
  const std::vector<float> history = downloadAs(afterHead, size_t{kHistory} * C, c.shape.state);
  for (uint32_t r = 0; r < blockRows; ++r)
    for (uint32_t ch = 0; ch < C; ++ch) {
      float past[kHistory];
      uint32_t row = r;
      for (uint32_t lag = 1; lag <= kHistory; ++lag) {
        const int depth = int(tree[r].depth) - int(lag);
        if (depth < 0) {
          past[lag - 1] = history[size_t(int(kHistory) + depth) * C + ch];
          continue;
        }
        row = tree[row].parent;
        past[lag - 1] = as(block[size_t{row} * c.stride + ch], c.shape.state);
      }
      const float *dynamic =
          c.shape.dynamic() ? &block[size_t{r} * c.stride + C + (ch / c.shape.headChannels) * kTaps] : nullptr;
      double magnitude;
      const float expected = convolutionOf(c, taps, ch, block[size_t{r} * c.stride + ch], past, dynamic, magnitude);
      okTree &= std::fabs(double(got[size_t{r} * C + ch]) - expected) <= 1e-6 * magnitude + 1e-30;
      // Row 3's packed predecessor (row 2) is its parent, row 5's (row 4) is not.
      if (r == 5) differs |= std::fabs(as(block[4 * size_t{c.stride} + ch], c.shape.state) - past[0]) > 1e-3f;
    }
  check(okTree && differs, scope + ": each row's past inputs follow its path");
  // The tape holds each row's input as a history keeps it.
  {
    const std::vector<float> taped = downloadAs(tape, size_t{blockRows} * C, c.shape.state);
    bool ok = true;
    for (uint32_t r = 0; r < blockRows; ++r)
      for (uint32_t ch = 0; ch < C; ++ch)
        ok &= taped[size_t{r} * C + ch] == as(block[size_t{r} * c.stride + ch], c.shape.state);
    check(ok, scope + ": the tape of the block's inputs");
  }
  // The commit of all of the path, and of its first 1 and 2 rows.
  const ops::ShortConvolutionUnits units{C, 1, C, 0, C, 0, c.shape.state};
  for (const uint32_t accepted : {5u, 1u, 2u}) {
    MetalBuffer committed = scratch(backend, historyBytes);
    const uint32_t mask = tree[path[accepted - 1]].ancestors;
    CommandGraph graph;
    ops::ShortConvolution::addCommit(graph, tape, std::span(&afterHead, 1), std::span(&committed, 1),
                                     upload(backend, std::vector<uint32_t>{mask}), units, layoutOf(backend, lane),
                                     {1, blockRows, blockRows});
    run(backend, graph);
    check(sameBits(downloadAs(committed, size_t{kHistory} * C, c.shape.state),
                   convReference(c, taps, s, head + accepted).history),
          scope + ": the commit of " + std::to_string(accepted) + " accepted rows, as their prefill's history");
  }

  // One row a step: a block of one row, committed.
  {
    MetalBuffer history = afterHead;
    bool ok = true;
    for (uint32_t i = 0; i < tail; ++i) {
      const std::vector<float> row(s.rows.begin() + size_t{head + i} * c.stride,
                                   s.rows.begin() + size_t{head + i + 1} * c.stride);
      const MetalBuffer rowBuffer = uploadAs(backend, row, c.shape.input),
                        output = scratch(backend, uint64_t{C} * sizeof(float)),
                        rowTape = scratch(backend, uint64_t{C} * ops::elementBytes(c.shape.state));
      MetalBuffer next = scratch(backend, historyBytes);
      const std::array<ops::LaneRows, 1> single{{{0, 1}}};
      const MetalBuffer singleLayout = layoutOf(backend, single);
      CommandGraph graph;
      ops::ShortConvolution::addConvolution(
          graph, convolutionBuffers(backend, c, rowBuffer, tapsBuffer, output), {std::span(&history, 1), {}, C},
          c.shape, singleLayout, {1, 1, 1},
          {upload(backend, std::vector<ops::TreeRow>{splash::test::chainRow(0)}), {rowTape, C}});
      ops::ShortConvolution::addCommit(graph, rowTape, std::span(&history, 1), std::span(&next, 1),
                                       upload(backend, std::vector<uint32_t>{1}), units, singleLayout, {1, 1, 1});
      run(backend, graph);
      const std::vector<float> out = download(output, C);
      ok &= std::memcmp(out.data(), full.data() + size_t{head + i} * C, C * sizeof(float)) == 0;
      history = next;
    }
    check(ok, scope + ": the tail decoded a row at a time, bit for bit");
    check(sameBits(downloadAs(history, size_t{kHistory} * C, c.shape.state), whole.history),
          scope + ": the history after the tail decoded a row at a time");
  }
}

// Eight lanes' verify blocks, one in each lane binding, through two layer
// applications of a K convolution (dynamic, 1024 channels) and an attention
// output convolution (2304 channels) side by side in each application's
// state and on the tape, as a target lays them out, and one commit of every
// application of every lane, against each lane's paths.
void convolutionUnitsTest(MetalBackend &backend, FloatElement state) {
  std::mt19937 rng(57);
  const std::array<ConvCase, 2> convs{ConvCase{{1024, 128, FloatElement::Float32, state}, 1088},
                                      ConvCase{{2304, 0, FloatElement::Float32, state}, 2304}};
  const uint32_t width = 1024 + 2304, applications = 2;
  const std::array<uint32_t, 2> offsets{0, 1024};
  // Lane 0: a chain of 4 accepting 3; lane 1: a tree of 6 accepting the
  // path to row 4 (rows 0, 2, 4); the others anchors alone, chains and
  // trees of up to 5 rows.
  const std::vector<std::vector<uint32_t>> parents{{0, 0, 1, 2}, {0, 0, 0, 1, 2, 2}, {0},          {0, 0, 1},
                                                   {0, 0, 0},    {0, 0, 1, 2, 3},    {0, 0},       {0, 0, 1, 1}};
  const std::vector<uint32_t> leaves{2, 4, 0, 2, 1, 4, 1, 3};
  constexpr uint32_t laneCount = 8, longest = 6;
  static_assert(laneCount == SPLASH_MAXIMUM_BATCH_WIDTH, "a lane in each binding");
  std::vector<ops::LaneRows> lanes;
  std::vector<ops::TreeRow> tree;
  for (const auto &p : parents) {
    lanes.push_back({static_cast<uint32_t>(tree.size()), static_cast<uint32_t>(p.size())});
    const auto t = treeOf(p);
    tree.insert(tree.end(), t.begin(), t.end());
  }
  const auto rows = static_cast<uint32_t>(tree.size());
  const MetalBuffer treeBuffer = upload(backend, tree), layout = layoutOf(backend, lanes);
  const uint64_t element = ops::elementBytes(state), unitBytes = uint64_t{kHistory} * width * element;
  const uint64_t tapeUnitBytes = uint64_t{rows} * width * element;
  // Each lane's state before and after, the step's tape.
  std::vector<std::vector<float>> before(laneCount);
  std::vector<MetalBuffer> in, out;
  for (uint32_t l = 0; l < laneCount; ++l) {
    before[l] = randoms(rng, size_t{applications} * kHistory * width, 1.0f);
    for (float &v : before[l]) v = as(v, state);
    in.push_back(uploadAs(backend, before[l], state));
    out.push_back(scratch(backend, applications * unitBytes));
  }
  const MetalBuffer tape = scratch(backend, applications * tapeUnitBytes);
  std::vector<std::vector<float>> inputs; // [application * 2 + conv]: rows x stride
  CommandGraph graph;
  for (uint32_t a = 0; a < applications; ++a)
    for (uint32_t k = 0; k < 2; ++k) {
      const ConvCase &c = convs[k];
      inputs.push_back(randoms(rng, size_t{rows} * c.stride, 1.0f));
      std::vector<float> &x = inputs.back();
      if (c.shape.dynamic())
        for (uint32_t r = 0; r < rows; ++r)
          for (uint32_t i = 0; i < c.dynamicWidth(); ++i) x[size_t{r} * c.stride + c.shape.channels + i] *= 0.3f;
      const MetalBuffer rowsBuffer = upload(backend, x);
      const MetalBuffer output = scratch(backend, uint64_t{rows} * c.shape.channels * sizeof(float));
      const uint64_t at = a * unitBytes + offsets[k] * element;
      std::vector<MetalBuffer> histories;
      for (const MetalBuffer &lane : in) histories.push_back(backend.view(lane, at, lane.sizeBytes() - at));
      const uint64_t tapeAt = a * tapeUnitBytes + offsets[k] * element;
      ops::ShortConvolution::addConvolution(
          graph,
          convolutionBuffers(backend, c, rowsBuffer, upload(backend, randoms(rng, c.shape.channels * kTaps, 0.5f)),
                             output),
          {histories, {}, width}, c.shape, layout, {laneCount, rows, longest},
          {treeBuffer, {backend.view(tape, tapeAt, tape.sizeBytes() - tapeAt), width}});
    }
  std::vector<uint32_t> paths;
  for (uint32_t l = 0; l < laneCount; ++l) paths.push_back(tree[lanes[l].first + leaves[l]].ancestors);
  ops::ShortConvolution::addCommit(graph, tape, in, out, upload(backend, paths),
                                   {width, applications, width, kHistory * width, width, rows * width, state}, layout,
                                   {laneCount, rows, longest});
  run(backend, graph);
  bool ok = true;
  for (uint32_t l = 0; l < laneCount; ++l) {
    const std::vector<float> got = downloadAs(out[l], size_t{applications} * kHistory * width, state);
    // The lane's path rows in order.
    std::vector<uint32_t> pathRows;
    for (uint32_t bits = paths[l]; bits; bits &= bits - 1) pathRows.push_back(lanes[l].first + std::countr_zero(bits));
    const int n = int(pathRows.size());
    for (uint32_t a = 0; a < applications; ++a)
      for (uint32_t k = 0; k < 2; ++k)
        for (uint32_t j = 0; j < kHistory; ++j)
          for (uint32_t ch = 0; ch < convs[k].shape.channels; ++ch) {
            const int source = n - int(kHistory) + int(j);
            const float want =
                source < 0
                    ? before[l][(size_t{a} * kHistory + uint32_t(int(kHistory) + source)) * width + offsets[k] + ch]
                    : as(inputs[a * 2 + k][size_t{pathRows[size_t(source)]} * convs[k].stride + ch], state);
            ok &= got[(size_t{a} * kHistory + j) * width + offsets[k] + ch] == want;
          }
  }
  check(ok, "eight lanes' commit of two applications of a K and an output convolution side by side, " + name(state) +
                " state");
}

// The output with a bf16 history against an fp32 history's, the inputs fp32,
// over the GGUF's range of taps (up to 76.5): printed, as a measure for the
// choice of the state's element type.
void convolutionStateTypeMeasure(MetalBackend &backend) {
  std::mt19937 rng(77);
  const uint32_t rows = 512, C = 2304;
  std::vector<float> taps = randoms(rng, size_t{C} * kTaps, 0.5f);
  for (uint32_t ch = 0; ch < C; ch += 97) taps[size_t{ch} * kTaps + ch % kTaps] = 70.0f;
  const MetalBuffer tapsBuffer = upload(backend, taps);
  std::array<std::vector<float>, 2> outputs;
  ConvSequence s;
  for (const FloatElement state : {FloatElement::Float32, FloatElement::BFloat16}) {
    const ConvCase c{{C, 0, FloatElement::Float32, state}, C};
    if (s.rows.empty()) s = convSequence(rng, c, rows, true);
    const ConvBuffers b = convBuffers(backend, c, s, rows);
    const std::array<ops::LaneRows, 1> one{{{0, rows}}};
    CommandGraph graph;
    ops::ShortConvolution::addConvolution(graph, convolutionBuffers(backend, c, b.rows, tapsBuffer, b.output),
                                          {std::span(&b.historyIn, 1), {}, C}, c.shape, layoutOf(backend, one),
                                          {1, rows, rows});
    run(backend, graph);
    outputs[state == FloatElement::BFloat16] = download(b.output, size_t{rows} * C);
  }
  double squares = 0, differences = 0, worst = 0;
  for (size_t i = 0; i < outputs[0].size(); ++i) {
    const double d = double(outputs[1][i]) - outputs[0][i];
    squares += double(outputs[0][i]) * outputs[0][i];
    differences += d * d;
    worst = std::max(worst, std::fabs(d));
  }
  const double rms = std::sqrt(squares / double(outputs[0].size()));
  std::cout << "bf16 state against fp32 state: rms(difference) / rms(output) " << std::sqrt(differences / squares)
            << ", max |difference| / rms(output) " << worst / rms << '\n';
  check(std::sqrt(differences / squares) < 2e-3, "bf16 state: within bf16 rounding of the fp32 state's output");
}

void convolutionRefusals(MetalBackend &backend) {
  const uint32_t rows = 4;
  const std::array<ops::LaneRows, 1> one{{{0, rows}}};
  const MetalBuffer layout = layoutOf(backend, one);
  // A static convolution of rows `stride` wide, each buffer at its extent
  // less `short` bytes.
  const auto convolve = [&](const ConvCase &c, uint64_t inputShort, uint64_t outputShort, uint64_t historyShort) {
    const uint32_t C = c.shape.channels;
    const uint64_t inputBytes = (uint64_t{rows - 1} * c.stride + C + c.dynamicWidth()) * 4;
    const MetalBuffer input = scratch(backend, inputBytes), taps = scratch(backend, C * kTaps * 4),
                      output = scratch(backend, uint64_t{rows} * C * 4),
                      history = scratch(backend, ops::shortConvolutionHistoryBytes(c.shape.state, C));
    const MetalBuffer h = backend.view(history, 0, history.sizeBytes() - historyShort);
    CommandGraph graph;
    ops::ShortConvolution::addConvolution(graph,
                                          convolutionBuffers(backend, c,
                                                             backend.view(input, 0, inputBytes - inputShort), taps,
                                                             backend.view(output, 0, output.sizeBytes() - outputShort)),
                                          {std::span(&h, 1), {}, C}, c.shape, layout, {1, rows, rows});
  };
  const ConvCase plain{{2304, 0, FloatElement::Float32, FloatElement::BFloat16}, 2304};
  const ConvCase dynamic{{1024, 128, FloatElement::Float32, FloatElement::BFloat16}, 1088};
  bool ok = !throws([&] { convolve(plain, 0, 0, 0); }) && !throws([&] { convolve(dynamic, 0, 0, 0); });
  ok &= throws([&] { convolve(plain, 4, 0, 0); });
  ok &= throws([&] { convolve(plain, 0, 4, 0); });
  ok &= throws([&] { convolve(plain, 0, 0, 2); });
  // The dynamic taps end the K side's last row.
  ok &= throws([&] { convolve(dynamic, 4, 0, 0); });
  ok &= throws([&] { convolve({{1024, 100, FloatElement::Float32, FloatElement::BFloat16}, 1088}, 0, 0, 0); });
  check(ok, "short convolution: buffers at their extent, and refused one element short or of uneven heads");
}

// ---------------------------------------------------------------------------
// Block residual (depth attention).

constexpr uint32_t kHidden = 2304;

// n states and m queries over `rows` rows, from separate buffers (one of a
// wider stride) or views of one bank, against the fp64 reference; with
// `overState`, the first output is the last state's rows.
void blockResidualTest(MetalBackend &backend, uint32_t n, uint32_t m, uint32_t rows, bool bank, bool overState) {
  std::mt19937 rng(100 * n + 10 * m + rows);
  const std::string scope = "depth attention of " + std::to_string(m) + " queries over " + std::to_string(n) +
                            " states, " + std::to_string(rows) + " rows" + (bank ? ", a bank" : "") +
                            (overState ? ", an output over a state" : "");
  // Row r of state k: values of its own scale, so the norms matter.
  std::vector<std::vector<float>> states(n);
  std::vector<uint32_t> strides(n);
  for (uint32_t k = 0; k < n; ++k) {
    strides[k] = bank ? n * kHidden : kHidden + (k == 1 ? 64 : 0);
    states[k] = randoms(rng, size_t{rows} * kHidden, 1.0f);
    for (uint32_t r = 0; r < rows; ++r) {
      const float scale = 0.05f + float((r * 7 + k * 13) % 17);
      for (uint32_t c = 0; c < kHidden; ++c) states[k][size_t{r} * kHidden + c] *= scale;
    }
  }
  const std::vector<float> queries = randoms(rng, size_t{m} * kHidden, 3.0f);
  std::vector<ops::StridedRows> views;
  std::vector<MetalBuffer> owned;
  if (bank) {
    std::vector<float> packed(size_t{rows} * n * kHidden);
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t k = 0; k < n; ++k)
        std::copy_n(states[k].begin() + size_t{r} * kHidden, kHidden, packed.begin() + (size_t{r} * n + k) * kHidden);
    const MetalBuffer buffer = upload(backend, packed);
    for (uint32_t k = 0; k < n; ++k) {
      const uint64_t at = uint64_t{k} * kHidden * sizeof(float);
      views.push_back({backend.view(buffer, at, buffer.sizeBytes() - at), strides[k]});
    }
  } else {
    for (uint32_t k = 0; k < n; ++k) {
      std::vector<float> strided(size_t{rows} * strides[k]);
      for (uint32_t r = 0; r < rows; ++r)
        std::copy_n(states[k].begin() + size_t{r} * kHidden, kHidden, strided.begin() + size_t{r} * strides[k]);
      views.push_back({upload(backend, strided), strides[k]});
    }
  }
  const uint32_t outputStride = overState ? strides[n - 1] : kHidden;
  std::vector<MetalBuffer> outputs;
  for (uint32_t j = 0; j < m; ++j)
    outputs.push_back(j == 0 && overState ? views[n - 1].buffer
                                          : scratch(backend, uint64_t{rows} * outputStride * sizeof(float)));
  CommandGraph graph;
  ops::BlockResidual::add(graph, views, upload(backend, queries), outputs, outputStride, kHidden, rows);
  run(backend, graph);
  bool ok = true;
  double worst = 0;
  for (uint32_t r = 0; r < rows; ++r) {
    std::vector<double> inverse(n);
    for (uint32_t k = 0; k < n; ++k) {
      double squares = 0;
      for (uint32_t c = 0; c < kHidden; ++c) squares += std::pow(double(states[k][size_t{r} * kHidden + c]), 2);
      inverse[k] = 1 / std::sqrt(squares / kHidden + 1e-6);
    }
    for (uint32_t j = 0; j < m; ++j) {
      std::vector<double> logits(n);
      for (uint32_t k = 0; k < n; ++k) {
        double dot = 0;
        for (uint32_t c = 0; c < kHidden; ++c)
          dot += double(queries[size_t{j} * kHidden + c]) * states[k][size_t{r} * kHidden + c] * inverse[k];
        logits[k] = dot / std::sqrt(double(kHidden));
      }
      const double most = *std::max_element(logits.begin(), logits.end());
      double total = 0;
      for (double &l : logits) total += (l = std::exp(l - most));
      const std::vector<float> got =
          downloadAs(outputs[j], kHidden, FloatElement::Float32, uint64_t{r} * outputStride * sizeof(float));
      for (uint32_t c = 0; c < kHidden; ++c) {
        double want = 0;
        for (uint32_t k = 0; k < n; ++k) want += logits[k] / total * inverse[k] * states[k][size_t{r} * kHidden + c];
        const double error = std::fabs(got[c] - want);
        worst = std::max(worst, error);
        ok &= error <= 2e-5 * (1 + std::fabs(want));
      }
    }
  }
  check(ok, scope + " (worst |error| " + std::to_string(worst) + ")");
}

void blockResidualRefusals(MetalBackend &backend) {
  const uint32_t rows = 3;
  const MetalBuffer state = scratch(backend, uint64_t{rows} * kHidden * 4), queries = scratch(backend, 4 * kHidden * 4),
                    output = scratch(backend, uint64_t{rows} * kHidden * 4);
  const auto mix = [&](std::vector<ops::StridedRows> states, std::vector<MetalBuffer> outputs) {
    CommandGraph graph;
    ops::BlockResidual::add(graph, states, queries, outputs, kHidden, kHidden, rows);
  };
  const ops::StridedRows s{state, kHidden};
  bool ok = !throws([&] { mix({s, s}, {output, output, output, output}); });
  ok &= throws([&] { mix(std::vector<ops::StridedRows>(7, s), {output}); });
  ok &= throws([&] { mix({s}, std::vector<MetalBuffer>(5, output)); });
  ok &= throws([&] { mix({s, {backend.view(state, 0, state.sizeBytes() - 4), kHidden}}, {output}); });
  ok &= throws([&] { mix({s}, {backend.view(output, 0, output.sizeBytes() - 4)}); });
  check(ok, "depth attention: buffers at their extent, and refused one element short or past its states and queries");
}

// ---------------------------------------------------------------------------
// Loop control.

constexpr uint32_t kRank = 64;

struct LoopStreamData {
  std::vector<float> down, forget, gain, input, message;
};

// Where a stream's gate writes its output: rows of its own, its input's
// (in place) or its message's (a target's streams that share one input,
// Inkling's sliding layers' Q, K and V).
enum class LoopOutput { Own, Input, Message };

// `streams` streams over `rows` rows with weights of `element`, each output
// where `output` says, against the fp64 reference.
void loopControlTest(MetalBackend &backend, uint32_t streams, uint32_t rows, FloatElement element,
                     LoopOutput output) {
  std::mt19937 rng(300 + streams * 10 + rows);
  const bool inPlace = output == LoopOutput::Input, overMessage = output == LoopOutput::Message;
  const std::string scope = "loop control of " + std::to_string(streams) + " streams, " + std::to_string(rows) +
                            " rows, " + name(element) + " weights" + (inPlace ? ", in place" : "") +
                            (overMessage ? ", over the messages" : "");
  const uint32_t stride = kHidden + 64;
  std::vector<LoopStreamData> data(streams);
  std::vector<ops::LoopControlWeights> weights(streams);
  std::vector<ops::LoopControlStream> bound(streams);
  std::vector<MetalBuffer> outputs;
  for (uint32_t s = 0; s < streams; ++s) {
    LoopStreamData &d = data[s];
    d.down = randoms(rng, size_t{kRank} * 2 * kHidden, 2.0f / std::sqrt(2.0f * kHidden));
    d.forget = randoms(rng, size_t{kHidden} * kRank, 1.5f / 8);
    d.gain = randoms(rng, size_t{kHidden} * kRank, 1.5f / 8);
    for (auto *w : {&d.down, &d.forget, &d.gain})
      for (float &v : *w) v = as(v, element);
    d.input = randoms(rng, size_t{rows} * stride, 1.0f);
    d.message = randoms(rng, size_t{rows} * stride, 1.0f);
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t c = 0; c < stride; ++c) {
        d.input[size_t{r} * stride + c] *= 0.1f + float((r * 5 + s) % 11);
        d.message[size_t{r} * stride + c] *= 0.2f + float((r * 3 + s * 7) % 13);
      }
    weights[s] = {uploadAs(backend, d.down, element), uploadAs(backend, d.forget, element),
                  uploadAs(backend, d.gain, element), element};
    const MetalBuffer input = upload(backend, d.input), message = upload(backend, d.message);
    outputs.push_back(inPlace ? input : overMessage ? message : scratch(backend, uint64_t{rows} * stride * 4));
    bound[s] = {&weights[s], input, message, outputs.back(), scratch(backend, ops::LoopControl::lowBytes(kRank, rows))};
  }
  CommandGraph graph;
  ops::LoopControl::add(graph, bound, {kHidden, kRank, stride, stride, stride}, rows);
  run(backend, graph);
  bool ok = true;
  double worst = 0;
  for (uint32_t s = 0; s < streams; ++s) {
    const LoopStreamData &d = data[s];
    const std::vector<float> got = download(outputs[s], size_t{rows} * stride);
    for (uint32_t r = 0; r < rows; ++r) {
      const float *u = &d.input[size_t{r} * stride], *m = &d.message[size_t{r} * stride];
      double su = 0, sm = 0;
      for (uint32_t c = 0; c < kHidden; ++c) {
        su += double(u[c]) * u[c];
        sm += double(m[c]) * m[c];
      }
      su = 1 / std::sqrt(su / kHidden + 1e-6);
      sm = 1 / std::sqrt(sm / kHidden + 1e-6);
      std::vector<double> low(kRank);
      for (uint32_t o = 0; o < kRank; ++o) {
        double sum = 0;
        for (uint32_t c = 0; c < kHidden; ++c)
          sum += double(d.down[size_t{o} * 2 * kHidden + c]) * u[c] * su +
                 double(d.down[size_t{o} * 2 * kHidden + kHidden + c]) * m[c] * sm;
        low[o] = sum;
      }
      for (uint32_t c = 0; c < kHidden; ++c) {
        double f = 0, g = 0;
        for (uint32_t o = 0; o < kRank; ++o) {
          f += d.forget[size_t{c} * kRank + o] * low[o];
          g += d.gain[size_t{c} * kRank + o] * low[o];
        }
        const double want = (1 - std::tanh(f)) * u[c] + std::tanh(g) * m[c];
        const double error = std::fabs(got[size_t{r} * stride + c] - want);
        worst = std::max(worst, error / (std::fabs(u[c]) + std::fabs(m[c]) + 1e-6));
        ok &= error <= 2e-5 * (std::fabs(u[c]) + std::fabs(m[c])) + 1e-6;
      }
      // The padding past the row is the input's (in place), the message's or untouched.
      for (uint32_t c = kHidden; c < stride; ++c)
        ok &= got[size_t{r} * stride + c] == (inPlace ? u[c] : overMessage ? m[c] : 0.0f);
    }
  }
  check(ok, scope + " (worst error / (|u| + |m|) " + std::to_string(worst) + ")");
}

void loopControlRefusals(MetalBackend &backend) {
  const uint32_t rows = 2;
  const ops::LoopControlWeights w{scratch(backend, uint64_t{kRank} * 2 * kHidden * 4),
                                  scratch(backend, uint64_t{kHidden} * kRank * 4),
                                  scratch(backend, uint64_t{kHidden} * kRank * 4), FloatElement::Float32};
  const ops::LoopControlWeights half{scratch(backend, uint64_t{kRank} * 2 * kHidden * 2),
                                     scratch(backend, uint64_t{kHidden} * kRank * 2),
                                     scratch(backend, uint64_t{kHidden} * kRank * 2), FloatElement::BFloat16};
  const MetalBuffer rowsBuffer = scratch(backend, uint64_t{rows} * kHidden * 4),
                    low = scratch(backend, ops::LoopControl::lowBytes(kRank, rows));
  const ops::LoopControlStream stream{&w, rowsBuffer, rowsBuffer, rowsBuffer, low};
  const auto gate = [&](std::vector<ops::LoopControlStream> streams, ops::LoopControlShape shape) {
    CommandGraph graph;
    ops::LoopControl::add(graph, streams, shape, rows);
  };
  const ops::LoopControlShape shape{kHidden, kRank, kHidden, kHidden, kHidden};
  bool ok = !throws([&] { gate({stream, stream, stream, stream}, shape); });
  ok &= throws([&] { gate(std::vector<ops::LoopControlStream>(5, stream), shape); });
  ok &= throws([&] { gate({stream, {&half, rowsBuffer, rowsBuffer, rowsBuffer, low}}, shape); });
  ok &= throws(
      [&] { gate({{&w, rowsBuffer, rowsBuffer, rowsBuffer, backend.view(low, 0, low.sizeBytes() - 4)}}, shape); });
  ok &= throws([&] {
    gate({{&w, backend.view(rowsBuffer, 0, rowsBuffer.sizeBytes() - 4), rowsBuffer, rowsBuffer, low}}, shape);
  });
  ok &= throws([&] { gate({stream}, {kHidden - 64 + 8, kRank, kHidden, kHidden, kHidden}); });
  ok &= throws([&] { gate({stream}, {kHidden, 32, kHidden, kHidden, kHidden}); });
  check(ok, "loop control: buffers at their extent, and refused one element short, of mixed weights or of shapes "
            "the kernels do not hold");
}

// The RMS norm of `rows` rows of `input` elements into `output` elements
// (one fp32, the other bf16) with F32 or BF16 multipliers, each row of its own
// scale, against the fp64 reference: an fp32 output within 2e-6 of it
// relatively, a bf16 one the reference's rounding or a neighbour of it; and
// refused between rows of one element type.
void normRowsTest(MetalBackend &backend, FloatElement input, FloatElement output, bool float32Weights,
                  uint32_t rows) {
  std::mt19937 rng(500 + rows + (input == FloatElement::Float32 ? 1 : 0) + (float32Weights ? 2 : 0));
  const std::string scope = std::string("RMS norm of ") + std::to_string(rows) + " rows, " + name(input) + " to " +
                            name(output) + ", " + (float32Weights ? "F32" : "BF16") + " multipliers";
  std::vector<float> x = randoms(rng, size_t{rows} * kHidden, 1.0f), gamma = randoms(rng, kHidden, 0.5f);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t c = 0; c < kHidden; ++c) x[size_t{r} * kHidden + c] *= 0.01f + float((r * 7) % 23);
  for (float &v : x) v = as(v, input);
  const FloatElement weightElement = float32Weights ? FloatElement::Float32 : FloatElement::BFloat16;
  for (float &v : gamma) v = as(v, weightElement);
  const ops::NormWeights weights{uploadAs(backend, gamma, weightElement), float32Weights};
  const MetalBuffer out = scratch(backend, uint64_t{rows} * kHidden * ops::elementBytes(output));
  CommandGraph graph;
  ops::Normalization::addRmsRows(graph, uploadAs(backend, x, input), input, weights, out, output, kHidden, rows);
  run(backend, graph);
  const std::vector<float> got = downloadAs(out, size_t{rows} * kHidden, output);
  bool ok = true;
  for (uint32_t r = 0; r < rows; ++r) {
    double squares = 0;
    for (uint32_t c = 0; c < kHidden; ++c) squares += std::pow(double(x[size_t{r} * kHidden + c]), 2);
    const double inverse = 1 / std::sqrt(squares / kHidden + 1e-6);
    for (uint32_t c = 0; c < kHidden; ++c) {
      const double want = x[size_t{r} * kHidden + c] * inverse * gamma[c];
      const float value = got[size_t{r} * kHidden + c];
      if (output == FloatElement::Float32) {
        ok &= std::fabs(value - want) <= 2e-6 * std::fabs(want) + 1e-7;
      } else {
        // Within one bf16 step of the reference.
        const double step = std::ldexp(1.0, std::ilogb(std::fabs(want) + 1e-30) - 7);
        ok &= std::fabs(value - want) <= step;
      }
    }
  }
  ok &= throws([&] {
    CommandGraph refused;
    ops::Normalization::addRmsRows(refused, out, output, weights, out, output, kHidden, rows);
  });
  check(ok, scope);
}

// ---------------------------------------------------------------------------
// Timing (`bench`): a prefill chunk's and a decode step's dispatches.

void bench(MetalBackend &backend) {
  const auto measure = [&](const std::string &what, const std::function<void(CommandGraph &)> &add, uint32_t repeat) {
    CommandGraph warm;
    add(warm);
    run(backend, warm);
    CommandGraph graph;
    for (uint32_t i = 0; i < repeat; ++i) add(graph);
    // The least of several runs: production work on the GPU beside the test
    // only adds to a run's time.
    double us = INFINITY;
    for (uint32_t run = 0; run < 7; ++run)
      us = std::min(us, backend.submitCommandAsync(graph.dispatches()).wait().gpuSeconds * 1e6 / repeat);
    std::cout << what << ": " << us << " us of GPU time a dispatch group\n";
  };
  std::mt19937 rng(5);
  for (const uint32_t rows : {1u, 16u, 4096u}) {
    const std::string at = " at " + std::to_string(rows) + " rows";
    // The K convolution (dynamic) of a global layer, bf16 state.
    const ConvCase k{{1024, 128, FloatElement::Float32, FloatElement::BFloat16}, 1088};
    const MetalBuffer input = upload(backend, randoms(rng, size_t{rows} * k.stride, 1.0f)),
                      taps = upload(backend, randoms(rng, 1024 * kTaps, 0.5f)),
                      output = scratch(backend, uint64_t{rows} * 2304 * 4),
                      history = scratch(backend, ops::shortConvolutionHistoryBytes(FloatElement::BFloat16, 2304)),
                      next = scratch(backend, ops::shortConvolutionHistoryBytes(FloatElement::BFloat16, 2304));
    const std::array<ops::LaneRows, 1> one{{{0, rows}}};
    const MetalBuffer layout = layoutOf(backend, one);
    measure(
        "K convolution" + at,
        [&](CommandGraph &g) {
          ops::ShortConvolution::addConvolution(g, convolutionBuffers(backend, k, input, taps, output),
                                                {std::span(&history, 1), std::span(&next, 1), 1024}, k.shape, layout,
                                                {1, rows, rows});
        },
        50);
    // Depth attention of 5 states, 1 query.
    std::vector<ops::StridedRows> states;
    for (uint32_t s = 0; s < 5; ++s)
      states.push_back({upload(backend, randoms(rng, size_t{rows} * kHidden, 1.0f)), kHidden});
    const MetalBuffer queries = upload(backend, randoms(rng, 4 * kHidden, 1.0f));
    const std::vector<MetalBuffer> outs{scratch(backend, uint64_t{rows} * kHidden * 4)};
    measure(
        "depth attention of 5 states" + at,
        [&](CommandGraph &g) { ops::BlockResidual::add(g, states, queries, outs, kHidden, kHidden, rows); }, 50);
    // Loop control of 3 streams, F32 weights.
    std::vector<ops::LoopControlWeights> w;
    std::vector<ops::LoopControlStream> streams;
    for (uint32_t s = 0; s < 3; ++s)
      w.push_back({upload(backend, randoms(rng, size_t{kRank} * 2 * kHidden, 0.03f)),
                   upload(backend, randoms(rng, size_t{kHidden} * kRank, 0.1f)),
                   upload(backend, randoms(rng, size_t{kHidden} * kRank, 0.1f)), FloatElement::Float32});
    for (uint32_t s = 0; s < 3; ++s)
      streams.push_back({&w[s], states[s].buffer, states[s + 1].buffer, outs[0],
                         scratch(backend, ops::LoopControl::lowBytes(kRank, rows))});
    measure(
        "loop control of 3 streams" + at,
        [&](CommandGraph &g) { ops::LoopControl::add(g, streams, {kHidden, kRank, kHidden, kHidden, kHidden}, rows); },
        20);
    measure(
        "loop control of 1 stream" + at,
        [&](CommandGraph &g) {
          ops::LoopControl::add(g, std::span(streams).first(1), {kHidden, kRank, kHidden, kHidden, kHidden}, rows);
        },
        20);
  }
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    try {
      const bool timing = argc == 3 && std::string(argv[2]) == "bench";
      if (argc != 2 && !timing) throw std::runtime_error("usage: inkling-ops METALLIB [bench]");
      MetalBackend backend(argv[1]);
      if (timing) {
        bench(backend);
        return 0;
      }
      const FloatElement F = FloatElement::Float32, B = FloatElement::BFloat16;
      // K and V (dynamic, 1024 channels in a K side of 1088), the attention
      // and MoE outputs' (static, 2304), at every pair of element types.
      for (const FloatElement input : {F, B})
        for (const FloatElement state : {F, B}) {
          convolutionPrefillTest(backend, {{1024, 128, input, state}, 1088}, false);
          convolutionPrefillTest(backend, {{2304, 0, input, state}, 2304}, false);
        }
      convolutionPrefillTest(backend, {{1024, 128, F, B}, 1088}, true);
      convolutionPrefillTest(backend, {{2304, 0, F, F}, 2304 + 64}, true);
      convolutionSequencesTest(backend, {{1024, 128, F, B}, 1088});
      convolutionSequencesTest(backend, {{2304, 0, B, B}, 2304});
      for (const FloatElement state : {F, B}) {
        convolutionDecodeTest(backend, {{1024, 128, F, state}, 1088});
        convolutionDecodeTest(backend, {{2304, 0, F, state}, 2304});
        convolutionUnitsTest(backend, state);
      }
      convolutionDecodeTest(backend, {{1024, 128, B, B}, 1088});
      convolutionStateTypeMeasure(backend);
      convolutionRefusals(backend);

      for (uint32_t n = 2; n <= BLOCK_RESIDUAL_MAXIMUM_STATES; ++n)
        for (const uint32_t m : {1u, 3u, 4u}) blockResidualTest(backend, n, m, n == 5 ? 67 : 3, n % 2 == 0, false);
      blockResidualTest(backend, 5, 1, 1, false, true);
      blockResidualTest(backend, 3, 3, 9, true, true);
      blockResidualRefusals(backend);

      for (const FloatElement element : {F, B}) {
        loopControlTest(backend, 1, 1, element, LoopOutput::Own);
        loopControlTest(backend, 3, 7, element, LoopOutput::Own);
        loopControlTest(backend, 3, 70, element, LoopOutput::Input);
        loopControlTest(backend, 3, 41, element, LoopOutput::Message);
      }
      loopControlTest(backend, 4, 33, F, LoopOutput::Own);
      loopControlRefusals(backend);

      for (const bool float32Weights : {true, false}) {
        normRowsTest(backend, F, B, float32Weights, 37);
        normRowsTest(backend, B, F, float32Weights, 5);
      }
    } catch (const std::exception &e) {
      std::cerr << "inkling-ops: " << e.what() << '\n';
      return 1;
    }
    std::cout << (failures ? "FAIL" : "PASS") << " (" << failures << " failures)\n";
    return failures ? 1 : 0;
  }
}
