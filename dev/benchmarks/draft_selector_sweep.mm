// The draft selector's top-16 pass (draft_select_top16_sharded) alone and the
// whole selector (DraftSelector::add: the top-16 pass, the edges and the
// tree), timed as a decode cycle runs them, for each lane count and shard
// count, so that the shard law (DraftSelector::shards) can be fitted on every
// core count. Kernel timings: each lane's eight query rows of random logits
// over Inkling-2-Mini's head (200,058 tokens in rows of 200,064 columns),
// `repeats` selections a command, each over its own rows.
//
//   draft-selector-sweep METALLIB [--lanes 1,2,4,8] [--shards 2,4,8,16,32]
//                        [--tokens 200058] [--stride 200064] [--repeats 4]
//                        [--rounds 15]
//
// Prints for each lane count and shard count the median and the least fused
// GPU microseconds a selection over `rounds` interleaved rounds after a
// warm-up (timeInterleaved), of the top-16 pass and of the whole selector,
// and the shards the law takes on this GPU at each lane count.

#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "ops/DraftSelector.hpp"
#include "ops/KernelCalibration.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;

namespace {

constexpr uint32_t kRows = SPLASH_DRAFT_QUERY_ROWS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kCandidates = SPLASH_DRAFT_CANDIDATES;
constexpr uint32_t kRank = SPLASH_DRAFT_SELECTOR_RANK;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kBlock = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kChain = SPLASH_TREE_CHAIN_TOKENS;
constexpr uint32_t kAgreements = SPLASH_TREE_AGREEMENTS;
constexpr uint32_t kChildren = SPLASH_TREE_CHILDREN;

std::vector<uint32_t> parseList(const std::string &text) {
  std::vector<uint32_t> values;
  size_t start = 0;
  while (start <= text.size()) {
    const size_t end = text.find(',', start);
    const std::string part = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (part.empty()) throw std::invalid_argument("an empty list entry");
    values.push_back(static_cast<uint32_t>(std::stoul(part)));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return values;
}

uint32_t nextRandom(uint64_t &state) {
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<uint32_t>(state >> 33);
}

MetalBuffer zeroed(MetalBackend &backend, uint64_t bytes, const char *label) {
  MetalBuffer buffer = backend.allocateBuffer(std::max<uint64_t>(bytes, 4), BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, buffer.sizeBytes());
  return buffer;
}

// Logits of a peaked row: N(0, 2)-like noise with a few dozen tokens lifted,
// as a draft's rows are; bf16 values of 0.5 + k / 1024 for the codebooks and
// the selector hidden rows.
MetalBuffer logitRows(MetalBackend &backend, uint64_t rows, uint32_t stride, uint32_t tokens, uint64_t seed) {
  MetalBuffer buffer = backend.allocateBuffer(rows * stride * sizeof(float), BufferStorage::Shared, "draft logits");
  auto *values = static_cast<float *>(buffer.contents());
  uint64_t state = seed * 2654435761ULL + 1;
  for (uint64_t row = 0; row < rows; ++row) {
    float *r = values + row * stride;
    for (uint32_t column = 0; column < stride; ++column) {
      const float u = static_cast<float>(nextRandom(state) & 0xFFFF) / 65536.0F;
      r[column] = column < tokens ? 4.0F * (u - 0.5F) : INFINITY;
    }
    for (uint32_t lifted = 0; lifted < 48; ++lifted)
      r[nextRandom(state) % tokens] = 8.0F + static_cast<float>(nextRandom(state) % 1024) / 64.0F;
  }
  return buffer;
}

MetalBuffer bfloatRows(MetalBackend &backend, uint64_t count, uint64_t seed, const char *label) {
  MetalBuffer buffer = backend.allocateBuffer(count * sizeof(uint16_t), BufferStorage::Shared, label);
  auto *values = static_cast<uint16_t *>(buffer.contents());
  uint64_t state = seed * 2654435761ULL + 7;
  for (uint64_t index = 0; index < count; ++index) {
    const float value = (static_cast<float>(nextRandom(state) >> 26) / 64.0F - 0.5F) * 0.25F;
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    values[index] = static_cast<uint16_t>(bits >> 16);
  }
  return buffer;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

// Each graph's GPU microseconds a selection over `rounds` rounds, every
// round running each graph once in order, after a warm-up that runs them
// all for at least half a second of GPU time, so that the GPU's clocks have
// risen before the first measurement and drift reaches every graph alike.
std::vector<std::vector<double>> timeInterleaved(MetalBackend &backend, const std::vector<CommandGraph> &graphs,
                                                 uint32_t rounds, uint32_t repeats) {
  double warm = 0;
  for (uint32_t pass = 0; pass < 2 || warm < 0.5; ++pass)
    for (const CommandGraph &graph : graphs) warm += backend.submitCommandAsync(graph.dispatches()).wait().gpuSeconds;
  std::vector<std::vector<double>> microseconds(graphs.size());
  for (uint32_t round = 0; round < rounds; ++round)
    for (size_t index = 0; index < graphs.size(); ++index)
      microseconds[index].push_back(backend.submitCommandAsync(graphs[index].dispatches()).wait().gpuSeconds * 1e6 /
                                    repeats);
  return microseconds;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2)
      throw std::invalid_argument("usage: draft-selector-sweep METALLIB [--lanes L,...] [--shards S,...] "
                                  "[--tokens N] [--stride N] [--repeats N] [--rounds N]");
    std::vector<uint32_t> laneCounts{1, 2, 4, 8};
    std::vector<uint32_t> shardCounts;
    for (uint32_t shards = 2; shards <= SPLASH_DRAFT_SAMPLING_MOST_SHARDS; shards *= 2) shardCounts.push_back(shards);
    uint32_t tokens = 200058, stride = 200064, repeats = 4, rounds = 15;
    for (int index = 2; index < argc; index += 2) {
      const std::string option = argv[index];
      if (index + 1 >= argc) throw std::invalid_argument(option + " takes a value");
      const std::string value = argv[index + 1];
      if (option == "--lanes") laneCounts = parseList(value);
      else if (option == "--shards") shardCounts = parseList(value);
      else if (option == "--tokens") tokens = static_cast<uint32_t>(std::stoul(value));
      else if (option == "--stride") stride = static_cast<uint32_t>(std::stoul(value));
      else if (option == "--repeats") repeats = static_cast<uint32_t>(std::stoul(value));
      else if (option == "--rounds") rounds = static_cast<uint32_t>(std::stoul(value));
      else throw std::invalid_argument("unknown option " + option);
    }
    for (const uint32_t lanes : laneCounts)
      if (!lanes || lanes > kLanes) throw std::invalid_argument("--lanes takes 1 to a batch's lanes");
    if (!tokens || stride < tokens) throw std::invalid_argument("--stride holds at least --tokens columns");
    if (!repeats || !rounds) throw std::invalid_argument("--repeats and --rounds take at least 1");

    MetalBackend backend(argv[1]);
    const auto &device = backend.capabilities();
    const KernelCalibration calibration = KernelCalibration::of(device);
    const uint32_t cores = calibration.cores;
    std::cout << "device " << device.deviceName << ", " << cores << " GPU cores, " << tokens << " tokens in rows of "
              << stride << ", " << repeats << " selections a command, " << rounds << " rounds\n";

    const DraftSelector selector(tokens);
    const DraftCodebooks codebooks{bfloatRows(backend, uint64_t{tokens} * kRank, 1, "draft predecessor codebook"),
                                   bfloatRows(backend, uint64_t{tokens} * kRank, 2, "draft successor codebook")};
    for (const uint32_t lanes : laneCounts) {
      const DraftSelectorWorkspace workspace = DraftSelector::workspace(lanes * kPositions);
      const uint64_t treeRows = uint64_t{lanes} * kBlock;
      std::vector<DraftSelectorBuffers> selections;
      for (uint32_t repeat = 0; repeat < repeats; ++repeat)
        selections.push_back(
            {logitRows(backend, uint64_t{lanes} * kRows, stride, tokens, uint64_t{lanes} * 131 + repeat),
             zeroed(backend, workspace.partialIdsBytes, "draft selector partial id"),
             zeroed(backend, workspace.partialValuesBytes, "draft selector partial value"),
             zeroed(backend, workspace.candidatesBytes, "draft candidate"),
             zeroed(backend, workspace.unaryBytes, "draft candidate score"),
             bfloatRows(backend, uint64_t{lanes} * kRows * kRank, 3 + repeat, "draft selector hidden"),
             zeroed(backend, uint64_t{lanes} * kChain * sizeof(uint32_t), "draft tree chain token"),
             zeroed(backend, uint64_t{lanes} * kChain * kAgreements * sizeof(float), "draft tree chain rate"),
             zeroed(backend, treeRows * sizeof(uint32_t), "draft tree token"),
             zeroed(backend, treeRows * sizeof(TreeRow), "draft tree row"),
             zeroed(backend, treeRows * 2 * sizeof(float), "draft tree score"),
             zeroed(backend, uint64_t{lanes} * sizeof(uint32_t), "draft tree size"),
             zeroed(backend, treeRows * kChildren, "draft tree draw"),
             zeroed(backend, treeRows * kCandidates * sizeof(float), "draft tree next"),
             zeroed(backend, treeRows * sizeof(float), "draft tree cap"),
             {}, {}, {}, {}});
      std::vector<DraftTreeLane> trees;
      for (uint32_t lane = 0; lane < lanes; ++lane) trees.push_back({lane + 1, kBlock - 1, 0, kChildren});
      const uint32_t chosen = DraftSelector::shards(lanes, cores, calibration.draft.topThreadgroupsPerFourCores);
      // The top-16 pass alone, as DraftSelector::add encodes it, and the
      // whole selector, at each shard count.
      std::vector<CommandGraph> graphs;
      for (const uint32_t shards : shardCounts) {
        CommandGraph top, whole;
        SelectorBatchParams params{};
        params.lanes = lanes;
        params.vocabulary = tokens;
        params.columns = tokens;
        params.stride = stride;
        params.shards = shards;
        for (uint32_t lane = 0; lane < lanes; ++lane) params.anchor[lane] = trees[lane].anchor;
        for (const DraftSelectorBuffers &buffers : selections) {
          top.add("draft_select_top16_sharded", {buffers.logits, buffers.partialIds, buffers.partialValues}, params,
                  {uint64_t{lanes} * kPositions * shards, 1, 1});
          selector.add(whole, buffers, codebooks, {}, stride, trees, shards);
        }
        graphs.push_back(std::move(top));
        graphs.push_back(std::move(whole));
      }
      const std::vector<std::vector<double>> microseconds = timeInterleaved(backend, graphs, rounds, repeats);
      for (size_t index = 0; index < shardCounts.size(); ++index) {
        const std::vector<double> &top = microseconds[2 * index], &whole = microseconds[2 * index + 1];
        std::printf("lanes %u shards %2u: top-16 median %7.1f us, least %7.1f; selector median %7.1f us, least %7.1f%s\n",
                    lanes, shardCounts[index], median(top), *std::min_element(top.begin(), top.end()), median(whole),
                    *std::min_element(whole.begin(), whole.end()), shardCounts[index] == chosen ? "  (law)" : "");
      }
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft-selector-sweep: " << error.what() << '\n';
    return 1;
  }
}
