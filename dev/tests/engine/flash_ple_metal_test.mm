// Root-run qualification only. No GPU commands are submitted without an
// explicit metallib path. Synthetic shards are tiny; optional real-checkpoint
// qualification compares selected PLE rows without dequantizing the table.
#include "flash/FlashPLE.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace splash::flash;
using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
namespace fs = std::filesystem;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

uint16_t bits(float value) {
  uint32_t raw;
  std::memcpy(&raw, &value, 4);
  raw += 0x7fff + ((raw >> 16) & 1);
  return static_cast<uint16_t>(raw >> 16);
}

float number(uint16_t value) {
  const uint32_t raw = uint32_t(value) << 16;
  float result;
  std::memcpy(&result, &raw, 4);
  return result;
}

template <class T> std::vector<T> read(const fs::path &path) {
  const uint64_t size = fs::file_size(path);
  require(size && size % sizeof(T) == 0, "invalid fixture extent " + path.string());
  std::vector<T> result(size / sizeof(T));
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char *>(result.data()),
            static_cast<std::streamsize>(size));
  require(bool(file), "cannot read fixture " + path.string());
  return result;
}

template <class T>
MetalBuffer buffer(MetalBackend &backend, std::span<const T> values,
                    const char *label) {
  auto result = backend.allocateBuffer(values.size_bytes(), BufferStorage::Shared,
                                        label);
  std::memcpy(result.contents(), values.data(), values.size_bytes());
  return result;
}

template <class T>
MetalBuffer buffer(MetalBackend &backend, const std::vector<T> &values,
                    const char *label) {
  return buffer(backend, std::span<const T>(values), label);
}

MetalBuffer emptyBF16(MetalBackend &backend, uint64_t elements) {
  auto result = backend.allocateBuffer(elements * 2, BufferStorage::Shared,
                                        "PLE oracle output");
  std::fill_n(static_cast<uint16_t *>(result.contents()), elements,
               uint16_t{0x7fc1});
  return result;
}

template <class T>
FlashTensor tensor(MetalBackend &backend, const std::vector<T> &values,
                    FlashDType dtype, std::vector<uint64_t> shape) {
  return {buffer(backend, values, "PLE oracle constant"), dtype,
           std::move(shape), values.size() * sizeof(T)};
}

template <class T>
void exact(const MetalBuffer &actual, std::span<const T> expected,
            const char *label) {
  require(actual.sizeBytes() >= expected.size_bytes(), "comparison view too small");
  require(std::memcmp(actual.contents(), expected.data(), expected.size_bytes()) == 0,
           std::string(label) + " is not exact");
}

struct Comparison {
  uint64_t elements = 0;
  uint64_t exact = 0;
  double maxAbsolute = 0;
};

void compare(const MetalBuffer &actual, std::span<const uint16_t> expected,
              const char *label, Comparison &count, bool requireExact = false) {
  require(actual.sizeBytes() >= expected.size_bytes(), "comparison view too small");
  const auto *observed = static_cast<const uint16_t *>(actual.contents());
  for (size_t i = 0; i < expected.size(); ++i) {
    ++count.elements;
    if (observed[i] == expected[i]) {
      ++count.exact;
      continue;
    }
    const float got = number(observed[i]), wanted = number(expected[i]);
    const double delta = std::fabs(double(got) - wanted);
    count.maxAbsolute = std::max(count.maxAbsolute, delta);
    int exponent = 0;
    std::frexp(std::fabs(wanted), &exponent);
    const double ulp = std::max(std::ldexp(1.0, exponent - 8),
                                std::ldexp(1.0, -133));
    require(!requireExact && std::isfinite(got) && std::isfinite(wanted) &&
                delta <= 2 * ulp + 1e-5,
             std::string(label) + " mismatch at " + std::to_string(i) +
                 " got=" + std::to_string(got) +
                 " wanted=" + std::to_string(wanted));
  }
}

FlashPLEWeights syntheticWeights(MetalBackend &backend) {
  FlashPLEWeights w;
  const std::vector<int64_t> multipliers{INT64_MAX, -17, INT64_MIN + 3};
  std::vector<int64_t> sizes(16, 17), offsets(16);
  for (uint32_t head = 0; head < 16; ++head) offsets[head] = head * 17;
  w.multipliers = tensor(backend, multipliers, FlashDType::I64, {3});
  w.headVocabularySizes = tensor(backend, sizes, FlashDType::I64, {16});
  w.headOffsets = tensor(backend, offsets, FlashDType::I64, {16});
  w.sharedScale = tensor(backend, std::vector<uint16_t>{bits(0.00019931793212890625f)},
                         FlashDType::BF16, {1});
  for (uint32_t shard = 0; shard < 128; ++shard) {
    std::vector<uint32_t> packed(4 * 20, 0);
    std::vector<uint16_t> scales(4 * 5), biases(4 * 5);
    for (uint32_t row = 0; row < 4; ++row) {
      for (uint32_t channel = 0; channel < 160; ++channel) {
        const uint32_t code = (shard * 4 + row + channel * 7) % 16;
        packed[row * 20 + channel / 8] |= code << ((channel % 8) * 4);
      }
      for (uint32_t group = 0; group < 5; ++group) {
        scales[row * 5 + group] =
            bits((shard % 2 ? -1.0f : 1.0f) * float(group + 1) / 32);
        biases[row * 5 + group] = bits(float(int((shard + row) % 17) - 8) / 16);
      }
    }
    w.shards.push_back({buffer(backend, packed, "tiny PLE Q4 rows"),
                        buffer(backend, scales, "tiny PLE scales"),
                        buffer(backend, biases, "tiny PLE biases"), 4, 80, 10});
  }
  return w;
}

std::vector<uint16_t> gatherReference(const FlashPLEWeights &w,
                                      std::span<const int64_t> ids) {
  std::vector<uint16_t> output(ids.size() * 160);
  const float shared = number(*static_cast<const uint16_t *>(w.sharedScale.buffer.contents()));
  for (size_t i = 0; i < ids.size(); ++i) {
    require(ids[i] >= 0 && uint64_t(ids[i]) < w.tableRows(), "reference invalid ID");
    const auto &shard = w.shards[ids[i] / 4];
    const uint64_t row = ids[i] % 4;
    const auto *packed = static_cast<const uint8_t *>(shard.weights.contents()) + row * 80;
    const auto *scales = static_cast<const uint16_t *>(shard.scales.contents()) + row * 5;
    const auto *biases = static_cast<const uint16_t *>(shard.biases.contents()) + row * 5;
    for (uint32_t channel = 0; channel < 160; ++channel) {
      const uint32_t code = (packed[channel / 2] >> ((channel % 2) * 4)) & 15;
      const uint16_t rowValue = bits(float(code) * number(scales[channel / 32]) +
                                     number(biases[channel / 32]));
      output[i * 160 + channel] = bits(number(rowValue) * shared);
    }
  }
  return output;
}

void synthetic(MetalBackend &backend, Comparison &count) {
  auto w = syntheticWeights(backend);
  const std::array<int64_t, 3> multipliers{INT64_MAX, -17, INT64_MIN + 3};
  std::array<int64_t, 16> sizes{}, offsets{};
  sizes.fill(17);
  for (uint32_t head = 0; head < 16; ++head) offsets[head] = head * 17;
  for (const uint32_t rows : {1u, 2u, 3u, 9u, 19u, 224u}) {
    FlashPLEGeometry g{2, rows};
    std::vector<int64_t> tokens(uint64_t(g.lanes) * rows);
    for (size_t i = 0; i < tokens.size(); ++i)
      tokens[i] = i % 7 == 2 ? g.eosToken : int64_t((i * 37 + 2) % 248320);
    std::vector<int64_t> previous{3, 4, g.eosToken, g.eosToken};
    auto expectedHistory = previous;
    const auto expectedIDs = computePLENgramIDs(tokens, expectedHistory,
                                                 multipliers, sizes, offsets,
                                                 g, w.tableRows());
    auto tokenBuffer = buffer(backend, tokens, "PLE synthetic tokens");
    auto historyBuffer = buffer(backend, previous, "PLE synthetic history");
    auto ids = backend.allocateBuffer(expectedIDs.size() * 8);
    auto output = emptyBF16(backend, expectedIDs.size() * 160);
    auto diagnostics = buffer(backend, std::vector<uint32_t>{0}, "PLE status");
    CommandGraph graph;
    addPLENgramIDs(graph, w, tokenBuffer, historyBuffer, ids, diagnostics, g);
    addPLEGather(graph, w, ids, output, diagnostics, g);
    static_cast<void>(backend.submitCommand(graph.dispatches()));
    exact<int64_t>(ids, expectedIDs, "I64 overflow/EOS hash");
    exact<int64_t>(historyBuffer, expectedHistory, "token history");
    const auto expected = gatherReference(w, expectedIDs);
    compare(output, expected, "affine gather after hash", count, true);
    require(*static_cast<const uint32_t *>(diagnostics.contents()) == 0,
             "valid synthetic hash/gather set diagnostics");
    std::cout << "synthetic_hash_gather rows=" << rows << " lanes=2 exact\n";
  }

  // Every shard and each last/first row crosses an eight-shard batch boundary.
  FlashPLEGeometry g{1, 64};
  std::vector<int64_t> routed(64 * 16);
  for (size_t i = 0; i < routed.size(); ++i) routed[i] = i % 512;
  auto ids = buffer(backend, routed, "all PLE shard boundary IDs");
  auto output = emptyBF16(backend, routed.size() * 160);
  auto diagnostics = buffer(backend, std::vector<uint32_t>{0}, "PLE status");
  CommandGraph graph;
  addPLEGather(graph, w, ids, output, diagnostics, g);
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  compare(output, gatherReference(w, routed), "all128 sparse shard routes", count, true);
  require(*static_cast<const uint32_t *>(diagnostics.contents()) == 0,
           "all-shard gather set diagnostics");
  auto *deviceIDs = static_cast<int64_t *>(ids.contents());
  deviceIDs[0] = -1;
  deviceIDs[1] = 512;
  CommandGraph invalid;
  addPLEGather(invalid, w, ids, output, diagnostics, g);
  static_cast<void>(backend.submitCommand(invalid.dispatches()));
  require((*static_cast<const uint32_t *>(diagnostics.contents()) &
            kFlashPLEInvalidIndex) != 0, "bad gather IDs did not set diagnostics");
  require(std::isnan(number(static_cast<const uint16_t *>(output.contents())[0])) &&
              std::isnan(number(static_cast<const uint16_t *>(output.contents())[160])),
           "bad gather IDs did not produce NaN");
  // Bad device-written token IDs and malformed stored checkpoint arrays must
  // remain explicit terminal diagnostics, rather than indexing a source table.
  auto invalidTokens = buffer(backend, std::vector<int64_t>{-1, g.eosToken},
                               "invalid device-written PLE token");
  auto invalidHistory = buffer(backend,
                                std::vector<int64_t>{g.eosToken, g.eosToken},
                                "invalid-token PLE history");
  auto hashIDs = backend.allocateBuffer(2 * 16 * 8);
  *static_cast<uint32_t *>(diagnostics.contents()) = 0;
  CommandGraph badHash;
  addPLENgramIDs(badHash, w, invalidTokens, invalidHistory, hashIDs, diagnostics,
                  FlashPLEGeometry{1, 2});
  static_cast<void>(backend.submitCommand(badHash.dispatches()));
  require((*static_cast<const uint32_t *>(diagnostics.contents()) & 1u) != 0 &&
              static_cast<const int64_t *>(hashIDs.contents())[0] == -1,
           "invalid token hash was not diagnosed");
  auto malformed = w;
  malformed.headVocabularySizes = tensor(backend, std::vector<int64_t>(16, 0),
                                          FlashDType::I64, {16});
  *static_cast<int64_t *>(invalidTokens.contents()) = 7;
  *static_cast<uint32_t *>(diagnostics.contents()) = 0;
  CommandGraph badSizes;
  addPLENgramIDs(badSizes, malformed, invalidTokens, invalidHistory, hashIDs,
                  diagnostics, FlashPLEGeometry{1, 2});
  static_cast<void>(backend.submitCommand(badSizes.dispatches()));
  require((*static_cast<const uint32_t *>(diagnostics.contents()) & 2u) != 0,
           "malformed stored head sizes were not diagnosed");
  CommandGraph rejected;
  bool caught = false;
  try {
    addPLENgramIDs(rejected, w, invalidTokens, invalidHistory, hashIDs,
                    diagnostics, FlashPLEGeometry{1, 0});
  } catch (const std::invalid_argument &) { caught = true; }
  require(caught && rejected.empty(), "invalid hash graph geometry was accepted");
  std::cout << "all128_sparse_shards=exact invalid_ids=flagged\n";
}

FlashPLEGeometry fixtureGeometry(const fs::path &directory) {
  NSString *path = [NSString stringWithUTF8String:(directory / "manifest.json").c_str()];
  NSData *data = [NSData dataWithContentsOfFile:path];
  require(data != nil, "cannot read post manifest.json");
  NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
  require([manifest isKindOfClass:NSDictionary.class], "invalid post manifest.json");
  NSDictionary *json = manifest[@"post_geometry"];
  require([json isKindOfClass:NSDictionary.class], "missing post geometry");
  auto value = [&](NSString *key) {
    NSNumber *number = json[key];
    require([number isKindOfClass:NSNumber.class], "missing post geometry field");
    return number.unsignedIntValue;
  };
  return {value(@"lanes"), value(@"rows"), value(@"width"), value(@"streams")};
}

template <class T>
std::vector<T> chunk(const std::vector<T> &source, uint32_t lanes,
                      uint32_t sourceRows, uint32_t width,
                      uint32_t begin, uint32_t rows) {
  std::vector<T> result(uint64_t(lanes) * rows * width);
  for (uint32_t lane = 0; lane < lanes; ++lane)
    std::copy_n(source.data() + (uint64_t(lane) * sourceRows + begin) * width,
                  uint64_t(rows) * width,
                  result.data() + uint64_t(lane) * rows * width);
  return result;
}

void postFixture(MetalBackend &backend, const fs::path &directory,
                   Comparison &count) {
  const auto g = fixtureGeometry(directory);
  auto bf16 = [&](const char *name) { return read<uint16_t>(directory / name); };
  const auto hyper = bf16("post-hyper.u16"), key = bf16("post-key.u16");
  const auto values = bf16("post-value.u16");
  const auto initial = bf16("post-state_initial.u16");
  const auto mask = read<uint32_t>(directory / "post-mask.u32");
  FlashPLEWeights w;
  const uint64_t width = uint64_t(g.width) * g.streams;
  w.normKey = tensor(backend, bf16("post-norm_key.u16"), FlashDType::BF16, {width});
  w.normQuery = tensor(backend, bf16("post-norm_query.u16"), FlashDType::BF16, {width});
  w.normConvolution = tensor(backend, bf16("post-norm_conv.u16"), FlashDType::BF16, {width});
  w.convolution = tensor(backend, bf16("post-conv.u16"), FlashDType::BF16, {width, 4, 1});
  const auto expected = bf16("post-expected_output.u16");
  const auto expectedState = bf16("post-expected_state.u16");
  auto state = buffer(backend, initial, "PLE post initial state");
  auto output = emptyBF16(backend, uint64_t(g.lanes) * g.rows * width);
  auto hyperBuffer = buffer(backend, hyper, "PLE post hyper input");
  auto keyBuffer = buffer(backend, key, "PLE post key projection");
  auto valueBuffer = buffer(backend, values, "PLE post value projection");
  auto maskBuffer = buffer(backend, mask, "PLE post mask");
  auto diagnostics = buffer(backend, std::vector<uint32_t>{0}, "PLE status");
  const uint64_t scratchElements = uint64_t(g.lanes) * g.rows * width;
  FlashPLEPostScratch scratch{emptyBF16(backend, scratchElements),
                              emptyBF16(backend, scratchElements),
                              emptyBF16(backend, scratchElements),
                              emptyBF16(backend, scratchElements)};
  auto malformed = w;
  malformed.convolution.shape = {width, 3, 1};
  CommandGraph rejected;
  bool caught = false;
  try {
    addPLEPostProject(rejected, malformed, hyperBuffer, keyBuffer, valueBuffer,
                      scratch, state, output, diagnostics, g, maskBuffer);
  } catch (const std::invalid_argument &) { caught = true; }
  require(caught && rejected.empty(), "invalid PLE convolution partially built graph");
  CommandGraph graph;
  addPLEPostProject(graph, w, hyperBuffer, keyBuffer, valueBuffer, scratch,
                    state, output, diagnostics, g, maskBuffer);
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  if (const char *dumpRoot = std::getenv("SPLASH_PLE_ORACLE_DUMP_DIR")) {
    const fs::path dumpDirectory = fs::path(dumpRoot) / directory.filename();
    fs::create_directories(dumpDirectory);
    auto dump = [&](const char *name, const MetalBuffer &source, uint64_t bytes) {
      require(source.sizeBytes() >= bytes, "dump source extent mismatch");
      std::ofstream file(dumpDirectory / name, std::ios::binary);
      file.write(static_cast<const char *>(source.contents()),
                  static_cast<std::streamsize>(bytes));
      require(bool(file), "cannot write PLE diagnostic dump");
    };
    dump("native-normalized_keys.u16", scratch.normalizedKeys, scratchElements * 2);
    dump("native-normalized_queries.u16", scratch.normalizedQueries, scratchElements * 2);
    dump("native-gated.u16", scratch.gatedValues, scratchElements * 2);
    dump("native-normalized_conv.u16", scratch.normalizedConvolution, scratchElements * 2);
    dump("native-output.u16", output, scratchElements * 2);
    dump("native-state.u16", state, initial.size() * 2);
  }
  compare(output, expected, "independent Metal PLE post output", count);
  compare(state, expectedState, "independent Metal PLE post carry", count);
  compare(scratch.gatedValues, bf16("post-expected_gated.u16"), "PLE gated values", count);
  compare(scratch.normalizedConvolution,
            bf16("post-expected_normalized_conv.u16"), "PLE conv norm", count);
  require(*static_cast<const uint32_t *>(diagnostics.contents()) == 0,
           "valid post fixture set diagnostics");

  // The same work split across chunk boundaries must have exact native BF16
  // output and final state, independently of CPU reduction-tree tolerances.
  auto splitState = buffer(backend, initial, "PLE split initial state");
  std::vector<uint16_t> splitOutput(expected.size());
  uint32_t begin = 0;
  for (uint32_t countRows : {2u, 1u, 4u, g.rows}) {
    if (begin >= g.rows) break;
    countRows = std::min(countRows, g.rows - begin);
    auto partGeometry = g;
    partGeometry.rows = countRows;
    auto partHyper = buffer(backend, chunk(hyper, g.lanes, g.rows, width, begin,
                                           countRows), "PLE chunk hyper");
    auto partKeys = buffer(backend, chunk(key, g.lanes, g.rows, width, begin,
                                          countRows), "PLE chunk keys");
    auto partValues = buffer(backend, chunk(values, g.lanes, g.rows, g.width,
                                             begin, countRows), "PLE chunk values");
    auto partMask = buffer(backend, chunk(mask, g.lanes, g.rows, 1, begin,
                                          countRows), "PLE chunk mask");
    const uint64_t elements = uint64_t(g.lanes) * countRows * width;
    FlashPLEPostScratch partScratch{emptyBF16(backend, elements),
                                     emptyBF16(backend, elements),
                                     emptyBF16(backend, elements),
                                     emptyBF16(backend, elements)};
    auto partOutput = emptyBF16(backend, elements);
    CommandGraph part;
    addPLEPostProject(part, w, partHyper, partKeys, partValues, partScratch,
                      splitState, partOutput, diagnostics, partGeometry, partMask);
    static_cast<void>(backend.submitCommand(part.dispatches()));
    for (uint32_t lane = 0; lane < g.lanes; ++lane)
      std::memcpy(splitOutput.data() + (uint64_t(lane) * g.rows + begin) * width,
                    static_cast<const uint16_t *>(partOutput.contents()) +
                        uint64_t(lane) * countRows * width,
                    uint64_t(countRows) * width * 2);
    begin += countRows;
  }
  require(std::memcmp(output.contents(), splitOutput.data(), splitOutput.size() * 2) == 0,
           "chunking changed native PLE output");
  require(state.sizeBytes() == splitState.sizeBytes() &&
              std::memcmp(state.contents(), splitState.contents(),
                            initial.size() * 2) == 0,
           "chunking changed native PLE convolution state");
  require(*static_cast<const uint32_t *>(diagnostics.contents()) == 0,
           "split post fixture set diagnostics");
  auto injected = emptyBF16(backend, scratchElements);
  CommandGraph inject;
  addPLEInject(inject, hyperBuffer, output, injected, g);
  static_cast<void>(backend.submitCommand(inject.dispatches()));
  std::vector<uint16_t> injectReference(splitOutput.size());
  for (size_t i = 0; i < injectReference.size(); ++i)
    injectReference[i] = bits(number(hyper[i]) +
                               number(static_cast<const uint16_t *>(output.contents())[i]));
  exact<uint16_t>(injected, injectReference, "PLE injection BF16 boundary");
  // Speculative rollback retains an explicit prefix independently per lane,
  // and must use snapshots from before the complete verify command.
  std::vector<int64_t> beforeHistory(uint64_t(g.lanes) * 2);
  std::vector<int64_t> tokens(uint64_t(g.lanes) * g.rows);
  for (uint32_t lane = 0; lane < g.lanes; ++lane) {
    beforeHistory[lane * 2] = lane ? 4 : g.eosToken;
    beforeHistory[lane * 2 + 1] = lane ? 5 : g.eosToken;
    for (uint32_t row = 0; row < g.rows; ++row)
      tokens[uint64_t(lane) * g.rows + row] = row == 3 ? g.eosToken :
                                               row * 3 + lane * 17 + 9;
  }
  auto beforeHistoryBuffer = buffer(backend, beforeHistory, "PLE rollback history snapshot");
  auto beforeState = buffer(backend, initial, "PLE rollback convolution snapshot");
  auto rollbackTokens = buffer(backend, tokens, "PLE verify tokens");
  auto restoredHistory = backend.allocateBuffer(beforeHistory.size() * 8);
  auto restoredState = emptyBF16(backend, initial.size());
  const auto *nativeNorm = static_cast<const uint16_t *>(scratch.normalizedConvolution.contents());
  for (const uint32_t baseKept : {0u, 1u, 2u, 3u, 8u, 9u, 10u, g.rows}) {
    std::vector<uint32_t> kept(g.lanes);
    std::vector<int64_t> wantedHistory(beforeHistory.size());
    std::vector<uint16_t> wantedState(initial.size());
    for (uint32_t lane = 0; lane < g.lanes; ++lane) {
      kept[lane] = std::min(g.rows, baseKept + lane);
      for (uint32_t slot = 0; slot < 2; ++slot) {
        const uint64_t timeline = uint64_t(kept[lane]) + slot;
        wantedHistory[uint64_t(lane) * 2 + slot] = timeline < 2 ?
            beforeHistory[uint64_t(lane) * 2 + timeline] :
            tokens[uint64_t(lane) * g.rows + timeline - 2];
      }
      for (uint32_t row = 0; row < 9; ++row) {
        const uint64_t timeline = uint64_t(kept[lane]) + row;
        const uint16_t *source = timeline < 9 ?
            initial.data() + (uint64_t(lane) * 9 + timeline) * width :
            nativeNorm + (uint64_t(lane) * g.rows + timeline - 9) * width;
        std::copy_n(source, width,
                      wantedState.data() + (uint64_t(lane) * 9 + row) * width);
      }
    }
    auto counts = buffer(backend, kept, "per-lane retained prefix counts");
    CommandGraph restore;
    addPLERestorePrefix(restore, beforeHistoryBuffer, rollbackTokens,
                         beforeState, scratch.normalizedConvolution, counts,
                         restoredHistory, restoredState, diagnostics, g);
    static_cast<void>(backend.submitCommand(restore.dispatches()));
    exact<int64_t>(restoredHistory, wantedHistory, "per-lane restored PLE history");
    exact<uint16_t>(restoredState, wantedState, "per-lane restored PLE convolution");
    exact<int64_t>(beforeHistoryBuffer, beforeHistory, "retained history snapshot");
    exact<uint16_t>(beforeState, initial, "retained convolution snapshot");
    require(*static_cast<const uint32_t *>(diagnostics.contents()) == 0,
             "valid PLE prefix restoration set diagnostics");
  }
  auto badCounts = buffer(backend, std::vector<uint32_t>(g.lanes, g.rows + 1),
                           "invalid retained prefix counts");
  CommandGraph badRestore;
  addPLERestorePrefix(badRestore, beforeHistoryBuffer, rollbackTokens, beforeState,
                       scratch.normalizedConvolution, badCounts, restoredHistory,
                       restoredState, diagnostics, g);
  static_cast<void>(backend.submitCommand(badRestore.dispatches()));
  require((*static_cast<const uint32_t *>(diagnostics.contents()) & 2u) != 0,
           "invalid per-lane prefix count did not set diagnostics");
  std::cout << "post_fixture=" << directory.filename().string()
            << " rows=" << g.rows << " lanes=" << g.lanes << " H=" << g.width
            << " native_split_output_and_state=exact injection=exact"
            << " unequal_lane_rollback=exact\n";
}

void realGather(MetalBackend &backend, const fs::path &fixture,
                  const fs::path &model, Comparison &count) {
  auto allWeights = FlashWeights::load(backend, model);
  auto w = FlashPLEWeights::fromWeights(allWeights);
  const auto tokens = read<int64_t>(fixture / "hash-tokens.i64");
  const auto history = read<int64_t>(fixture / "hash-initial_history.i64");
  const auto expectedIDs = read<int64_t>(fixture / "hash-expected_ids.i64");
  const auto expectedHistory = read<int64_t>(fixture / "hash-expected_history.i64");
  const auto expectedOutput = read<uint16_t>(fixture / "hash-expected_gather.u16");
  const FlashPLEGeometry g{1, static_cast<uint32_t>(tokens.size())};
  auto tokenBuffer = buffer(backend, tokens, "real PLE oracle tokens");
  auto historyBuffer = buffer(backend, history, "real PLE oracle history");
  auto ids = backend.allocateBuffer(expectedIDs.size() * 8);
  auto output = emptyBF16(backend, expectedOutput.size());
  auto diagnostics = buffer(backend, std::vector<uint32_t>{0}, "PLE status");
  CommandGraph graph;
  addPLENgramIDs(graph, w, tokenBuffer, historyBuffer, ids, diagnostics, g);
  addPLEGather(graph, w, ids, output, diagnostics, g);
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  exact<int64_t>(ids, expectedIDs, "stored checkpoint PLE hash");
  exact<int64_t>(historyBuffer, expectedHistory, "stored checkpoint PLE history");
  compare(output, expectedOutput, "selected actual PLE row gather", count, true);
  require(*static_cast<const uint32_t *>(diagnostics.contents()) == 0,
           "real PLE hash/gather set diagnostics");
  std::cout << "actual_checkpoint_hash_and_selected_Q4G32_rows=exact\n";
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc >= 2 && argc <= 4,
               "usage: flash_ple_metal_test metallib [fixture-root [derived-model]]");
      MetalBackend backend(argv[1]);
      Comparison count;
      synthetic(backend, count);
      if (argc >= 3) {
        const fs::path root(argv[2]);
        if (fs::exists(root / "post-hyper.u16")) postFixture(backend, root, count);
        for (const auto &entry : fs::directory_iterator(root))
          if (entry.is_directory() &&
              fs::exists(entry.path() / "post-hyper.u16"))
            postFixture(backend, entry.path(), count);
        if (argc == 4) realGather(backend, root, argv[3], count);
      }
      std::cout << "PLE qualified elements=" << count.elements
                << " exact=" << count.exact
                << " max_abs=" << count.maxAbsolute << '\n';
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "Flash PLE oracle failed: " << error.what() << '\n';
      return 1;
    }
  }
}
