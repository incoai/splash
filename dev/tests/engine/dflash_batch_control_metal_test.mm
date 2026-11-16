#include "ChainRows.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "model/Model.hpp"
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
#include <random>
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
using splash::ops::RowCopy;
using splash::ops::RowRegion;
using splash::ops::StopTokens;

constexpr uint32_t kRows = splash::model::ExecutionLimits::draftQueryRows;
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

// A lane's block in runWidth: its anchor and its draft's proposals.
constexpr uint32_t kBlock = kProposals + 1;
constexpr std::array<uint32_t, kLanes> kAllRows{kBlock, kBlock, kBlock, kBlock};
// The stop token runWidth places, in the last place the kernel compares.
constexpr uint32_t kStopToken = 248044;
constexpr StopTokens kFourStops{248046, 150001, 150003, kStopToken};

// Each lane's rows of `blocks` as a chain.
MetalBuffer chainRows(MetalBackend &backend, std::span<const splash::ops::LaneRows> blocks, uint32_t rows) {
  MetalBuffer buffer = shared(backend, uint64_t{rows} * sizeof(splash::ops::TreeRow), "accept-tree");
  auto *tree = contents<splash::ops::TreeRow>(buffer);
  for (const splash::ops::LaneRows &block : blocks)
    for (uint32_t row = 0; row < block.rows; ++row)
      tree[block.first + row] = splash::test::chainRow(row);
  return buffer;
}

// `blocks` as a step's proposer writes them on the device (SplashLaneRows).
MetalBuffer laneRows(MetalBackend &backend, std::span<const splash::ops::LaneRows> blocks) {
  MetalBuffer buffer = shared(backend, blocks.size() * sizeof(splash::ops::LaneRows), "accept-lane-rows");
  std::ranges::copy(blocks, contents<splash::ops::LaneRows>(buffer));
  return buffer;
}

// Lane stopLane's target selects `stopToken` at row stopRow, also its draft
// below its accepted rows; acceptance cuts the lane's tokens after it when it
// is one of `stops`. Each lane's rows are a chain from row lane x kBlock, of
// rowsReference[lane] rows, the rows past them to the next lane's of no lane,
// which it walks: its retained tokens are its first rows' selections, its
// retained rows their mask, and its accepted count the proposals among them.
void runWidth(MetalBackend &backend, uint32_t width,
              const std::array<uint32_t, kLanes> &acceptedReference,
              const std::array<uint32_t, kLanes> &remainingReference,
              uint32_t stopLane = kLanes, uint32_t stopRow = 0,
              const std::array<uint32_t, kLanes> &rowsReference = kAllRows,
              uint32_t stopToken = kStopToken, const StopTokens &stops = kFourStops) {
  require(width >= 1 && width <= kLanes, "invalid test width");
  std::array<splash::ops::LaneRows, kLanes> blocks{};
  for (uint32_t lane = 0; lane < kLanes; ++lane)
    blocks[lane] = {lane * kBlock, rowsReference[lane]};
  MetalBuffer input = shared(backend, kLanes * kBlock * sizeof(uint32_t), "accept-input");
  const MetalBuffer tree = chainRows(backend, blocks, kLanes * kBlock);
  const MetalBuffer lanes = laneRows(backend, blocks);
  MetalBuffer output = shared(backend, kLanes * kBlock * sizeof(uint32_t),
                              "accept-output");
  MetalBuffer retained =
      shared(backend, kLanes * sizeof(uint32_t), "accept-retained");
  MetalBuffer accepted =
      shared(backend, kLanes * sizeof(uint32_t), "accept-count");
  MetalBuffer retainedRows = shared(backend, kLanes * sizeof(uint32_t), "accept-retained-rows");

  auto *inputTokens = contents<uint32_t>(input);
  auto *targetTokens = contents<uint32_t>(output);
  std::memset(retained.contents(), 0, retained.sizeBytes());
  std::memset(accepted.contents(), 0, accepted.sizeBytes());
  std::memset(retainedRows.contents(), 0, retainedRows.sizeBytes());
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    inputTokens[lane * kBlock] = 500 + lane;
    for (uint32_t token = 0; token < kProposals; ++token) {
      const uint32_t proposal = 1000 + lane * 100 + token;
      inputTokens[lane * kBlock + token + 1] = proposal;
      targetTokens[lane * kBlock + token] =
          token < acceptedReference[lane] ? proposal : proposal + 50;
    }
    targetTokens[lane * kBlock + kBlock - 1] = 9000 + lane;
  }

  if (stopLane < width) {
    require(stopRow < kBlock, "invalid stop row");
    targetTokens[stopLane * kBlock + stopRow] = stopToken;
    if (stopRow < acceptedReference[stopLane])
      inputTokens[stopLane * kBlock + stopRow + 1] = stopToken;
  }
  const std::vector<uint32_t> selected(targetTokens, targetTokens + kLanes * kBlock);

  const splash::ops::Sampling sampling(248320);
  CommandGraph graph;
  sampling.addAcceptance(graph, {input, tree, output, retained, accepted, retainedRows, lanes},
                         {width, width * kBlock, kBlock}, std::span(remainingReference).first(width), stops);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());

  const auto *retainedCounts = contents<uint32_t>(retained);
  const auto *acceptedCounts = contents<uint32_t>(accepted);
  for (uint32_t lane = 0; lane < width; ++lane) {
    const uint32_t walked = std::min(acceptedReference[lane], rowsReference[lane] - 1);
    uint32_t expectedRetained = std::min(walked + 1, remainingReference[lane]);
    if (lane == stopLane && stopRow < expectedRetained && stops.contains(stopToken))
      expectedRetained = stopRow + 1;
    // Where the cap or a stop token cuts the walk, every retained token is an
    // accepted proposal.
    require(acceptedCounts[lane] == std::min(walked, expectedRetained),
            "accepted proposal count mismatch");
    require(retainedCounts[lane] == expectedRetained,
            "retained token count mismatch");
    require(contents<uint32_t>(retainedRows)[lane] == (1u << expectedRetained) - 1,
            "a chain's retained rows are not its first ones");
    require(std::equal(selected.begin() + lane * kBlock, selected.begin() + lane * kBlock + expectedRetained,
                       targetTokens + lane * kBlock),
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

// Lanes of chains packed one after another as packedLanes takes them: each
// lane's rows, accepted proposals and remaining tokens, the step's rows (two
// of no lane past the lanes') and its lane cap.
struct PackedChains {
  std::array<splash::ops::LaneRows, kLanes> packed;
  std::array<uint32_t, kLanes> accepted, remaining;
  uint32_t rows, cap;
};

// Lanes of different row counts, packed one after another, and two rows of no
// lane after them, accept their chains as each lane does alone in a block
// from row 0: the same retained and accepted counts and retained rows. Lane
// l's target selects its first accepted[l] proposals and then other tokens;
// it walks its rows, which check one proposal fewer, and retains at most
// remaining[l] tokens.
void packedLanes(MetalBackend &backend, const PackedChains &chains) {
  using splash::ops::LaneRows;
  const std::array<LaneRows, kLanes> &packed = chains.packed;
  const std::array<uint32_t, kLanes> &accepted = chains.accepted, &remaining = chains.remaining;
  const splash::ops::Sampling sampling(248320);
  // A run of `lanes` lanes' buffers: verify input, target tokens, retained
  // and accepted counts, and retained rows.
  struct Run final {
    std::array<MetalBuffer, 5> buffers;
  };
  const auto makeRun = [&](uint32_t lanes, uint32_t rows) {
    Run run;
    for (const auto &[index, count] : {std::pair{0, rows}, std::pair{1, rows}, std::pair{2, lanes},
                                       std::pair{3, lanes}, std::pair{4, lanes}}) {
      run.buffers[index] = shared(backend, uint64_t{count} * sizeof(uint32_t), "packed-lanes");
      std::memset(run.buffers[index].contents(), 0, run.buffers[index].sizeBytes());
    }
    return run;
  };
  const auto encode = [&](const Run &run, std::span<const LaneRows> blocks, std::span<const uint32_t> limits) {
    const auto &b = run.buffers;
    const uint32_t rows = static_cast<uint32_t>(b[0].sizeBytes() / 4);
    const MetalBuffer tree = chainRows(backend, blocks, rows);
    const uint32_t lanes = static_cast<uint32_t>(blocks.size());
    CommandGraph graph;
    sampling.addAcceptance(graph, {b[0], tree, b[1], b[2], b[3], b[4], laneRows(backend, blocks)},
                           {lanes, rows, chains.cap}, limits, kFourStops);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  };
  // Lane l's anchor, proposals and target tokens, written from `row` of
  // `run`'s rows.
  const auto fill = [&](const Run &run, uint32_t lane, uint32_t row) {
    for (uint32_t index = 0; index < packed[lane].rows; ++index) {
      contents<uint32_t>(run.buffers[0])[row + index] = index ? 1000 + lane * 100 + index - 1 : 500 + lane;
      contents<uint32_t>(run.buffers[1])[row + index] =
          index < accepted[lane] ? 1000 + lane * 100 + index : 9000 + lane * 100 + index;
    }
  };
  const Run all = makeRun(kLanes, chains.rows);
  for (uint32_t lane = 0; lane < kLanes; ++lane) fill(all, lane, packed[lane].first);
  encode(all, packed, remaining);
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    const LaneRows block = packed[lane];
    const Run alone = makeRun(1, block.rows);
    fill(alone, lane, 0);
    const std::array<LaneRows, 1> own{{{0, block.rows}}};
    encode(alone, own, std::span(remaining).subspan(lane, 1));
    const std::string label = "packed lane " + std::to_string(lane) + " at a cap of " + std::to_string(chains.cap);
    const uint32_t retained = contents<uint32_t>(all.buffers[2])[lane];
    const uint32_t counted = contents<uint32_t>(all.buffers[3])[lane];
    require(retained == contents<uint32_t>(alone.buffers[2])[0] && counted == contents<uint32_t>(alone.buffers[3])[0] &&
                contents<uint32_t>(all.buffers[4])[lane] == contents<uint32_t>(alone.buffers[4])[0],
            label + ": its acceptance differs from alone");
    const uint32_t walked = std::min(accepted[lane], block.rows - 1);
    require(retained == std::min(walked + 1, remaining[lane]) && counted == std::min(walked, retained),
            label + ": accepted " + std::to_string(counted) + " and retained " + std::to_string(retained));
  }
}

// Acceptance down random trees against a reference walk. A lane's rows each
// follow a random earlier row and take tokens of a small alphabet (siblings
// apart), and each row's selection is one of its children's tokens most of the
// time, else another token or now and then a stop token; the lane holds a
// random number of its tree's rows, the rest of no lane, and retains at most
// a random number. The walk follows from the anchor the child among the
// lane's rows whose token is the row's selection; it retains at most the
// limit and none after a stop token,
// writes their selections from the lane's first row and their mask, its
// deepest retained row's ancestors, and counts the rows it reached past the
// anchor as accepted, those it retained where it cut the path. Trees of a
// draft's block of rows, and of a verify block's, whose walks reach rows
// past the eighth.
void treeAcceptance(MetalBackend &backend, uint32_t rows) {
  using splash::ops::LaneRows;
  using splash::ops::TreeRow;
  constexpr uint32_t kMostRows = SPLASH_TARGET_VERIFY_ROWS;
  require(rows <= kMostRows, "a tree of more rows than a verify block");
  std::mt19937 random(0x74726565);
  const auto below = [&](uint32_t bound) { return static_cast<uint32_t>(random() % bound); };
  // Siblings take distinct tokens of six, or in a tree of more rows of as
  // many as a row's most children, so that every row finds one.
  const uint32_t alphabet = std::max(6U, rows - 1);
  const splash::ops::Sampling sampling(248320);
  std::array<LaneRows, kLanes> blocks{};
  for (uint32_t lane = 0; lane < kLanes; ++lane)
    blocks[lane] = {lane * rows, rows};
  const MetalBuffer lanes = laneRows(backend, blocks);
  const auto buffer = [&](uint64_t bytes) {
    MetalBuffer result = shared(backend, bytes, "tree-acceptance");
    std::memset(result.contents(), 0, bytes);
    return result;
  };
  const MetalBuffer input = buffer(kLanes * rows * 4), tree = buffer(kLanes * rows * sizeof(TreeRow)),
                    output = buffer(kLanes * rows * 4), retained = buffer(kLanes * 4), accepted = buffer(kLanes * 4),
                    retainedRows = buffer(kLanes * 4);
  uint32_t deep = 0, pastEight = 0;
  for (uint32_t trial = 0; trial < 300; ++trial) {
    std::array<std::array<uint32_t, kMostRows>, kLanes> parents{}, tokens{}, selections{}, ancestors{};
    std::array<uint32_t, kLanes> limits{}, verified{};
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      tokens[lane][0] = 500 + lane;
      ancestors[lane][0] = 1;
      for (uint32_t row = 1; row < rows; ++row) {
        // Deep trees as often as wide ones.
        parents[lane][row] = trial % 2 ? row - 1 - below(std::min(row, 2U)) : below(row);
        uint32_t token = 0;
        for (bool taken = true; taken;) {
          token = 1000 + below(alphabet);
          taken = false;
          for (uint32_t sibling = 1; sibling < row; ++sibling)
            taken = taken || (parents[lane][sibling] == parents[lane][row] && tokens[lane][sibling] == token);
        }
        tokens[lane][row] = token;
        ancestors[lane][row] = ancestors[lane][parents[lane][row]] | (1U << row);
      }
      for (uint32_t row = 0; row < rows; ++row) {
        std::vector<uint32_t> children;
        for (uint32_t child = row + 1; child < rows; ++child)
          if (parents[lane][child] == row)
            children.push_back(child);
        const uint32_t draw = below(20);
        selections[lane][row] = draw == 0                       ? kStopToken
                                : draw < 15 && !children.empty() ? tokens[lane][children[below(children.size())]]
                                                                 : 2000 + below(1000);
      }
      verified[lane] = 1 + below(rows);
      limits[lane] = 1 + below(rows);
      for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t at = lane * rows + row;
        contents<uint32_t>(input)[at] = tokens[lane][row];
        contents<TreeRow>(tree)[at] = {ancestors[lane][row],
                                       static_cast<uint16_t>(row ? parents[lane][row] : SPLASH_TREE_ROOT),
                                       static_cast<uint16_t>(row ? contents<TreeRow>(tree)[lane * rows +
                                                                                           parents[lane][row]].depth +
                                                                       1
                                                                 : 0)};
        contents<uint32_t>(output)[at] = selections[lane][row];
      }
      contents<LaneRows>(lanes)[lane].rows = verified[lane];
    }
    CommandGraph graph;
    sampling.addAcceptance(graph, {input, tree, output, retained, accepted, retainedRows, lanes},
                           {kLanes, kLanes * rows, rows}, limits, kFourStops);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      std::vector<uint32_t> path{0};
      for (bool moved = true; moved;) {
        moved = false;
        for (uint32_t child = path.back() + 1; child < verified[lane] && !moved; ++child)
          if (parents[lane][child] == path.back() && tokens[lane][child] == selections[lane][path.back()]) {
            path.push_back(child);
            moved = true;
          }
      }
      uint32_t kept = std::min(static_cast<uint32_t>(path.size()), limits[lane]);
      for (uint32_t i = 0; i < kept; ++i)
        if (kFourStops.contains(selections[lane][path[i]])) {
          kept = i + 1;
          break;
        }
      deep += kept > 2;
      pastEight += path[kept - 1] >= 8;
      const std::string label = "tree of " + std::to_string(rows) + " rows, trial " + std::to_string(trial) +
                                " lane " + std::to_string(lane);
      require(contents<uint32_t>(accepted)[lane] == std::min<uint32_t>(path.size() - 1, kept),
              label + ": accepted count differs");
      require(contents<uint32_t>(retained)[lane] == kept, label + ": retained count differs");
      require(contents<uint32_t>(retainedRows)[lane] == ancestors[lane][path[kept - 1]],
              label + ": retained rows differ");
      for (uint32_t i = 0; i < kept; ++i)
        require(contents<uint32_t>(output)[lane * rows + i] == selections[lane][path[i]],
                label + ": a retained token is not its path row's selection");
    }
  }
  std::cout << "tree acceptance of " << rows << " rows: " << deep << " walks retained more than two rows, "
            << pastEight << " a row past the eighth\n";
  require(deep >= 100, "too few walks retained more than two rows");
  // 72 of the 1,200 walks of 16-row trees retain a row past the eighth.
  require(rows <= 8 || pastEight >= 50, "too few walks of a verify block's rows retained a row past the eighth");
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
    // changing the physical eight-row graph: a lane whose limit or stop token
    // cuts its seven accepted proposals counts those it retains.
    constexpr std::array<uint32_t, kLanes> allAccepted{7, 7, 7, 7};
    constexpr std::array<uint32_t, kLanes> shortRemaining{1, 2, 3, 8};
    runWidth(backend, kLanes, allAccepted, shortRemaining, 3, 3);
    // SPLASH_NO_TOKEN, which pads a list of two stop tokens, is no stop token
    // where acceptance retains it: a row with no selectable logit selects it,
    // and a draft that proposed it there is accepted, so the lane keeps its
    // later tokens and the runtime fails it on the sentinel.
    runWidth(backend, kLanes, allAccepted, fullRemaining, 1, 2, kAllRows, SPLASH_NO_TOKEN,
             StopTokens{248044, 248046});
    // A lane walks only its rows, where the rows past them to the next lane's
    // are of no lane, below its budget or above it.
    constexpr std::array<uint32_t, kLanes> fewRows{2, 3, 5, 1};
    runWidth(backend, kLanes, allAccepted, fullRemaining, kLanes, 0, fewRows);
    runWidth(backend, kLanes, lowerAccepted, shortRemaining, kLanes, 0, fewRows);
    testRowCopy(backend);
    // Lanes of 5, 6, 1 and 6 rows from rows 0, 5, 11 and 12 at the narrow
    // cap, and of 16, 5, 9 and 12 from rows 0, 16, 21 and 30 at a verify
    // block's.
    packedLanes(backend, {{{{0, 5}, {5, 6}, {11, 1}, {12, 6}}}, {3, 7, 0, 5}, {5, 8, 1, 4}, 20, kRows});
    packedLanes(backend, {{{{0, 16}, {16, 5}, {21, 9}, {30, 12}}},
                          {14, 3, 0, 9},
                          {16, 5, 9, 7},
                          44,
                          SPLASH_TARGET_VERIFY_ROWS});
    for (const uint32_t rows : {kRows, SPLASH_TARGET_VERIFY_ROWS})
      treeAcceptance(backend, rows);
    std::cout << "dflash_batch_control_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "dflash_batch_control_metal_test: FAIL: " << error.what()
              << '\n';
    return 1;
  }
}
