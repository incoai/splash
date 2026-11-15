#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "model/Model.hpp"
#include "ops/Embedding.hpp"
#include "ops/RowCopy.hpp"
#include "ops/Sampling.hpp"
#include "ops/LaneRows.hpp"
#include "ops/StopTokens.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::Embedding;
using splash::ops::RowCopy;
using splash::ops::RowRegion;
using splash::ops::StopTokens;

constexpr uint32_t kRows = splash::model::ExecutionLimits::targetVerifyRows;
constexpr uint32_t kProposals =
    splash::model::ExecutionLimits::draftProposalTokens;
constexpr uint32_t kLanes =
    splash::model::ExecutionLimits::maximumBatchWidth;

MetalBuffer shared(MetalBackend &backend, uint64_t bytes, const char *label) {
  return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
}

template <class T> T *contents(const MetalBuffer &buffer) {
  return static_cast<T *>(buffer.contents());
}

using splash::test::rejects;
using splash::test::require;
using splash::test::requireExtent;

constexpr std::array<uint32_t, kLanes> kAllRows{kRows, kRows, kRows, kRows};
// The stop token runWidth places, in the last place the kernel compares.
constexpr uint32_t kStopToken = 248044;
constexpr StopTokens kFourStops{248046, 150001, 150003, kStopToken};

// Each lane's verify block of kRows rows as a chain, from row lane x kRows.
MetalBuffer chainRows(MetalBackend &backend, std::span<const splash::ops::LaneRows> blocks, uint32_t rows) {
  MetalBuffer buffer = shared(backend, uint64_t{rows} * sizeof(splash::ops::TreeRow), "accept-tree");
  auto *tree = contents<splash::ops::TreeRow>(buffer);
  for (const splash::ops::LaneRows &block : blocks)
    for (uint32_t row = 0; row < block.rows; ++row)
      tree[block.first + row] = splash::ops::chainRow(row);
  return buffer;
}

// Lane stopLane's target selects `stopToken` at row stopRow, also its draft
// below its accepted rows; acceptance cuts the lane's tokens after it when it
// is one of `stops`. Each lane's block is a chain: its retained tokens are
// its first rows' selections, and its retained rows their mask.
void runWidth(MetalBackend &backend, uint32_t width,
              const std::array<uint32_t, kLanes> &acceptedReference,
              const std::array<uint32_t, kLanes> &remainingReference,
              uint32_t stopLane = kLanes, uint32_t stopRow = 0,
              const std::array<uint32_t, kLanes> &verifyRowsReference = kAllRows,
              uint32_t stopToken = kStopToken, const StopTokens &stops = kFourStops) {
  require(width >= 1 && width <= kLanes, "invalid test width");
  std::array<splash::ops::LaneRows, kLanes> blocks{};
  for (uint32_t lane = 0; lane < kLanes; ++lane)
    blocks[lane] = {lane * kRows, kRows};
  MetalBuffer input = shared(backend, kLanes * kRows * sizeof(uint32_t), "accept-input");
  const MetalBuffer tree = chainRows(backend, blocks, kLanes * kRows);
  MetalBuffer output = shared(backend, kLanes * kRows * sizeof(uint32_t),
                              "accept-output");
  MetalBuffer retained =
      shared(backend, kLanes * sizeof(uint32_t), "accept-retained");
  MetalBuffer accepted =
      shared(backend, kLanes * sizeof(uint32_t), "accept-count");
  MetalBuffer retainedRows = shared(backend, kLanes * sizeof(uint32_t), "accept-retained-rows");
  MetalBuffer verifyRows =
      shared(backend, kLanes * sizeof(uint32_t), "accept-verify-rows");
  std::copy(verifyRowsReference.begin(), verifyRowsReference.end(),
            contents<uint32_t>(verifyRows));

  auto *inputTokens = contents<uint32_t>(input);
  auto *targetTokens = contents<uint32_t>(output);
  std::memset(retained.contents(), 0, retained.sizeBytes());
  std::memset(accepted.contents(), 0, accepted.sizeBytes());
  std::memset(retainedRows.contents(), 0, retainedRows.sizeBytes());
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    inputTokens[lane * kRows] = 500 + lane;
    for (uint32_t token = 0; token < kProposals; ++token) {
      const uint32_t proposal = 1000 + lane * 100 + token;
      inputTokens[lane * kRows + token + 1] = proposal;
      targetTokens[lane * kRows + token] =
          token < acceptedReference[lane] ? proposal : proposal + 50;
    }
    targetTokens[lane * kRows + kRows - 1] = 9000 + lane;
  }

  if (stopLane < width) {
    require(stopRow < kRows, "invalid stop row");
    targetTokens[stopLane * kRows + stopRow] = stopToken;
    if (stopRow < acceptedReference[stopLane])
      inputTokens[stopLane * kRows + stopRow + 1] = stopToken;
  }
  const std::vector<uint32_t> selected(targetTokens, targetTokens + kLanes * kRows);

  const splash::ops::Sampling sampling(248320);
  CommandGraph graph;
  sampling.addAcceptance(graph, {input, tree, output, retained, accepted, retainedRows, verifyRows},
                         std::span(blocks).first(width), std::span(remainingReference).first(width), stops);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());

  const auto *retainedCounts = contents<uint32_t>(retained);
  const auto *acceptedCounts = contents<uint32_t>(accepted);
  for (uint32_t lane = 0; lane < width; ++lane) {
    uint32_t expectedRetained =
        std::min({acceptedReference[lane] + 1, remainingReference[lane],
                  verifyRowsReference[lane]});
    if (lane == stopLane && stopRow < expectedRetained && stops.contains(stopToken))
      expectedRetained = stopRow + 1;
    require(acceptedCounts[lane] == acceptedReference[lane],
            "accepted proposal count mismatch");
    require(retainedCounts[lane] == expectedRetained,
            "retained token count mismatch");
    require(contents<uint32_t>(retainedRows)[lane] == (1u << expectedRetained) - 1,
            "a chain's retained rows are not its first ones");
    require(std::equal(selected.begin() + lane * kRows, selected.begin() + lane * kRows + expectedRetained,
                       targetTokens + lane * kRows),
            "a chain's retained tokens are not its first rows' selections");
  }
}

// The row copies of production, bitwise against a CPU copy: the capture of
// 13 target hidden rows (width 2048) from source row 5 into the third slot of
// four of the captured rows from row 2, the gather of the last 3 of 11
// prefill rows into the head's input, and an image's embedding rows 1-2
// (width 5120) over a chunk's placeholder rows from row 2. Every value
// outside the destination region keeps its poison, and regions past a row are
// refused. Each buffer holds its region at its extent, the end of its last
// row's values, and is refused one value short.
void testRowCopy(MetalBackend &backend) {
  constexpr uint16_t kPoison = 0xA5A5;
  const auto check = [&](uint32_t sourceRows, RowRegion from,
                         uint32_t destinationRows, RowRegion to,
                         uint32_t rows, uint32_t width, const char *what) {
    MetalBuffer source = shared(backend, uint64_t{sourceRows} * from.stride * 2, "row-copy-source");
    MetalBuffer destination =
        shared(backend, uint64_t{destinationRows} * to.stride * 2, "row-copy-destination");
    auto *input = contents<uint16_t>(source);
    for (uint64_t i = 0; i < source.sizeBytes() / 2; ++i)
      input[i] = static_cast<uint16_t>(i * 2654435761u >> 16);
    std::vector<uint16_t> expected(destination.sizeBytes() / 2, kPoison);
    std::copy(expected.begin(), expected.end(), contents<uint16_t>(destination));
    for (uint32_t row = 0; row < rows; ++row)
      for (uint32_t column = 0; column < width; ++column)
        expected[uint64_t{to.row + row} * to.stride + to.column + column] =
            input[uint64_t{from.row + row} * from.stride + from.column + column];
    CommandGraph graph;
    RowCopy::add(graph, source, from, destination, to, rows, width);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    require(std::equal(expected.begin(), expected.end(), contents<uint16_t>(destination)), what);
  };
  constexpr uint32_t kWidth = 2048;
  check(18, {5, kWidth, 0}, 16, {2, 4 * kWidth, 2 * kWidth}, 13, kWidth,
        "captured rows differ from a CPU copy");
  check(11, {8, kWidth, 0}, kRows, {0, kWidth, 0}, 3, kWidth,
        "gathered rows differ from a CPU copy");
  constexpr uint32_t kImageWidth = 5120;
  check(4, {1, kImageWidth, 0}, 5, {2, kImageWidth, 0}, 2, kImageWidth,
        "image rows differ from a CPU copy");

  MetalBuffer rows = shared(backend, uint64_t{4} * kWidth * 2, "row-copy-invalid");
  CommandGraph invalid;
  const auto copy = [&](RowRegion from, RowRegion to, uint32_t count, uint32_t width) {
    RowCopy::add(invalid, rows, from, rows, to, count, width);
  };
  rejects([&] { copy({0, kWidth, 0}, {0, kWidth, 0}, 0, kWidth); }, "invalid row copy",
          "a row copy of no rows was accepted");
  rejects([&] { copy({0, kWidth, 0}, {0, kWidth, 0}, 1, 0); }, "invalid row copy",
          "a row copy of no values was accepted");
  rejects([&] { copy({0, kWidth, 1}, {0, kWidth, 0}, 1, kWidth); }, "invalid row copy",
          "a row copy past the end of its source row was accepted");
  rejects([&] { copy({0, kWidth, 0}, {0, kWidth / 2, 0}, 1, kWidth); }, "invalid row copy",
          "a row copy past the end of its destination row was accepted");
  require(invalid.empty(), "an invalid row copy encoded a dispatch");
  // The capture's regions: source rows 5-17 of 2048 values, destination rows
  // 2-14 of 8192 from column 4096.
  constexpr RowRegion from{5, kWidth, 0}, to{2, 4 * kWidth, 2 * kWidth};
  constexpr uint32_t copied = 13;
  const MetalBuffer source = shared(backend, uint64_t{18} * kWidth * 2, "row-copy-source"),
                    destination = shared(backend, uint64_t{16} * 4 * kWidth * 2, "row-copy-destination");
  requireExtent(backend, source, uint64_t{17 * kWidth + kWidth} * 2, 2, "row copy source",
                [&](CommandGraph &graph, const MetalBuffer &view) {
                  RowCopy::add(graph, view, from, destination, to, copied, kWidth);
                });
  requireExtent(backend, destination, (uint64_t{14} * 4 * kWidth + 2 * kWidth + kWidth) * 2, 2,
                "row copy destination", [&](CommandGraph &graph, const MetalBuffer &view) {
                  RowCopy::add(graph, source, from, view, to, copied, kWidth);
                });
}

// Each buffer the verify input reaches, at its extent and one element short,
// for three lanes: each lane's anchor, row 0 of its eight draft input rows,
// its seven proposals and its eight verify input rows.
void verifyInputExtents(MetalBackend &backend) {
  constexpr uint32_t lanes = 3, vocabulary = 1003;
  const std::array<splash::ops::LaneRows, lanes> blocks{{{0, kRows}, {kRows, kRows}, {2 * kRows, kRows}}};
  const std::array buffers{shared(backend, lanes * kRows * 4, "draft-input"),
                           shared(backend, lanes * kProposals * 4, "proposed"),
                           shared(backend, lanes * kRows * 4, "verify-input")};
  for (const auto &[index, bytes, what] :
       {std::tuple{size_t{0}, uint64_t{(lanes - 1) * kRows + 1} * 4, "draft input token"},
        std::tuple{size_t{1}, uint64_t{lanes * kProposals} * 4, "proposed token"},
        std::tuple{size_t{2}, uint64_t{lanes * kRows} * 4, "verify input token"}})
    requireExtent(backend, buffers[index], bytes, 4, what, [&](CommandGraph &graph, const MetalBuffer &view) {
      auto changed = buffers;
      changed[index] = view;
      Embedding::addVerifyInput(graph, changed[0], changed[1], changed[2], vocabulary, blocks);
    });
}

// Lanes whose verify blocks hold different row counts, packed one after
// another (5, 8, 1 and 6 rows from rows 0, 5, 13 and 14), take their verify
// input and accept their drafts as each lane does alone in a block from row
// 0: the same input rows, retained and accepted counts. Lane l's target
// selects its first accepted[l] proposals and then other tokens; it retains
// at most remaining[l] tokens and its verify rows, and a block checks one
// proposal fewer than its rows.
void packedLanes(MetalBackend &backend) {
  // Every token below is inside the vocabulary, which clamps none.
  constexpr uint32_t vocabulary = 248320;
  using splash::ops::LaneRows;
  const std::array<LaneRows, kLanes> packed{{{0, 5}, {5, 8}, {13, 1}, {14, 6}}};
  const std::array<uint32_t, kLanes> accepted{3, 7, 0, 5}, remaining{5, 8, 1, 4}, verifyRows{5, 6, 1, 6};
  const splash::ops::Sampling sampling(vocabulary);
  // A run of `lanes` lanes' buffers: draft input, proposals, verify input,
  // target tokens, retained and accepted counts, verify rows and retained
  // rows.
  struct Run final {
    std::array<MetalBuffer, 8> buffers;
  };
  const auto makeRun = [&](uint32_t lanes, uint32_t rows) {
    Run run;
    for (const auto &[index, count] : {std::pair{0, lanes * kRows}, std::pair{1, lanes * kProposals},
                                       std::pair{2, rows}, std::pair{3, rows}, std::pair{4, lanes},
                                       std::pair{5, lanes}, std::pair{6, lanes}, std::pair{7, lanes}}) {
      run.buffers[index] = shared(backend, uint64_t{count} * sizeof(uint32_t), "packed-lanes");
      std::memset(run.buffers[index].contents(), 0, run.buffers[index].sizeBytes());
    }
    return run;
  };
  const auto encode = [&](const Run &run, std::span<const LaneRows> blocks, std::span<const uint32_t> limits) {
    const auto &b = run.buffers;
    const MetalBuffer tree = chainRows(backend, blocks, static_cast<uint32_t>(b[2].sizeBytes() / 4));
    CommandGraph graph;
    Embedding::addVerifyInput(graph, b[0], b[1], b[2], vocabulary, blocks);
    sampling.addAcceptance(graph, {b[2], tree, b[3], b[4], b[5], b[7], b[6]}, blocks, limits, kFourStops);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  };
  // Lane l's anchor, proposals and target tokens, written from `row` of
  // `run`'s rows and lane `at` of its lanes.
  const auto fill = [&](const Run &run, uint32_t lane, uint32_t at, uint32_t row) {
    contents<uint32_t>(run.buffers[0])[at * kRows] = 500 + lane;
    for (uint32_t position = 0; position < kProposals; ++position)
      contents<uint32_t>(run.buffers[1])[at * kProposals + position] = 1000 + lane * 100 + position;
    for (uint32_t index = 0; index < packed[lane].rows; ++index)
      contents<uint32_t>(run.buffers[3])[row + index] =
          index < accepted[lane] ? 1000 + lane * 100 + index : 9000 + lane * 100 + index;
    contents<uint32_t>(run.buffers[6])[at] = verifyRows[lane];
  };
  const Run all = makeRun(kLanes, 20);
  for (uint32_t lane = 0; lane < kLanes; ++lane) fill(all, lane, lane, packed[lane].first);
  encode(all, packed, remaining);
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    const LaneRows block = packed[lane];
    const Run alone = makeRun(1, block.rows);
    fill(alone, lane, 0, 0);
    const std::array<LaneRows, 1> own{{{0, block.rows}}};
    encode(alone, own, std::span(remaining).subspan(lane, 1));
    const std::string label = "packed lane " + std::to_string(lane);
    require(std::equal(contents<uint32_t>(all.buffers[2]) + block.first,
                       contents<uint32_t>(all.buffers[2]) + block.first + block.rows,
                       contents<uint32_t>(alone.buffers[2])),
            label + ": its verify input differs from alone");
    const uint32_t retained = contents<uint32_t>(all.buffers[4])[lane];
    const uint32_t counted = contents<uint32_t>(all.buffers[5])[lane];
    require(retained == contents<uint32_t>(alone.buffers[4])[0] && counted == contents<uint32_t>(alone.buffers[5])[0] &&
                contents<uint32_t>(all.buffers[7])[lane] == contents<uint32_t>(alone.buffers[7])[0],
            label + ": its acceptance differs from alone");
    require(counted == accepted[lane] &&
                retained == std::min({accepted[lane] + 1, remaining[lane], verifyRows[lane]}),
            label + ": accepted " + std::to_string(counted) + " and retained " + std::to_string(retained));
    // The anchor, then the proposals the block's rows after it hold.
    require(contents<uint32_t>(all.buffers[2])[block.first] == 500 + lane &&
                (block.rows < 2 || contents<uint32_t>(all.buffers[2])[block.first + block.rows - 1] ==
                                       1000 + lane * 100 + block.rows - 2),
            label + ": its verify input is not its anchor and proposals");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: dflash-batch-control METALLIB");
    MetalBackend backend(argv[1]);
    constexpr std::array<uint32_t, kLanes> lowerAccepted{0, 1, 2, 3};
    constexpr std::array<uint32_t, kLanes> upperAccepted{4, 5, 6, 7};
    constexpr std::array<uint32_t, kLanes> fullRemaining{8, 8, 8, 8};
    for (uint32_t width = 1; width <= kLanes; ++width) {
      runWidth(backend, width, lowerAccepted, fullRemaining);
      runWidth(backend, width, upperAccepted, fullRemaining);
    }

    // Output limits and stop tokens shorten the committed prefix without
    // changing the physical eight-row graph.  The accepted proposal count is
    // still seven in every lane; only retained rows move.
    constexpr std::array<uint32_t, kLanes> allAccepted{7, 7, 7, 7};
    constexpr std::array<uint32_t, kLanes> shortRemaining{1, 2, 3, 8};
    runWidth(backend, kLanes, allAccepted, shortRemaining, 3, 3);
    // SPLASH_NO_TOKEN, which pads a list of two stop tokens, is no stop token
    // where acceptance retains it: a row with no selectable logit selects it,
    // and a draft that proposed it there is accepted, so the lane keeps its
    // later tokens and the runtime fails it on the sentinel.
    runWidth(backend, kLanes, allAccepted, fullRemaining, 1, 2, kAllRows, SPLASH_NO_TOKEN,
             StopTokens{248044, 248046});
    // A lane retains at most the rows its verify verified, below its budget
    // or above it; acceptance still counts every proposal it checked.
    constexpr std::array<uint32_t, kLanes> fewRows{2, 3, 5, 1};
    runWidth(backend, kLanes, allAccepted, fullRemaining, kLanes, 0, fewRows);
    runWidth(backend, kLanes, lowerAccepted, shortRemaining, kLanes, 0, fewRows);
    testRowCopy(backend);
    verifyInputExtents(backend);
    packedLanes(backend);
    std::cout << "dflash_batch_control_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "dflash_batch_control_metal_test: FAIL: " << error.what()
              << '\n';
    return 1;
  }
}
