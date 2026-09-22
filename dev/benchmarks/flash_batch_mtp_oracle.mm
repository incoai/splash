#include "flash/FlashBatchMTPForward.hpp"
#include "flash/FlashDenseCache.hpp"
#include "engine/Json.hpp"
#import <Foundation/Foundation.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace splash;
using namespace splash::flash;
constexpr uint32_t kWidth = 10240, kVocabulary = 248320;
uint16_t bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}
float f32(uint16_t value) { return std::bit_cast<float>(uint32_t{value} << 16); }
bool environment(const char *name, bool fallback = false) {
  const char *source = std::getenv(name);
  if (!source) return fallback;
  if (std::string_view(source) == "1") return true;
  if (std::string_view(source) == "0") return false;
  throw std::invalid_argument(std::string(name) + " must be0 or1");
}
std::vector<uint16_t> copied(const metal::MetalBuffer &buffer, uint64_t count) {
  if (!buffer || !buffer.contents() || buffer.sizeBytes() < count * 2)
    throw std::runtime_error("batch head oracle received an invalid borrowed result");
  const auto *base = static_cast<const uint16_t *>(buffer.contents());
  return {base, base + count};
}
struct Comparison {
  uint64_t elements = 0, differences = 0, commands = 0, scenarios = 0;
  double maximumRelativeL2 = 0.0, sequentialGpu = 0.0, batchGpu = 0.0;
  void compare(std::span<const uint16_t> actual, std::span<const uint16_t> expected,
      bool exact, const char *kind) {
    if (actual.size() != expected.size()) throw std::runtime_error("oracle output size differs");
    double error = 0.0, norm = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
      ++elements;
      differences += actual[index] != expected[index];
      const double a = f32(actual[index]), e = f32(expected[index]);
      if (!std::isfinite(a) || !std::isfinite(e))
        throw std::runtime_error("oracle found nonfinite head output");
      error += (a - e) * (a - e); norm += e * e;
    }
    const double relative = std::sqrt(error / std::max(1e-30, norm));
    maximumRelativeL2 = std::max(maximumRelativeL2, relative);
    if ((exact && error != 0.0) || relative > 0.01)
      throw std::runtime_error(std::string(kind) + " comparison failed; relativeL2=" + std::to_string(relative));
  }
};
template <class Operation> void invalid(Operation operation) {
  bool failed = false;
  try { operation(); } catch (const std::invalid_argument &) { failed = true; }
  if (!failed) throw std::runtime_error("malformed batch head call was not rejected before GPU work");
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 4)
        throw std::invalid_argument("usage: flash-batch-mtp-oracle METALLIB PACKAGE REPORT_JSON");
      const bool exact = environment("FLASH_BATCH_MTP_ORACLE_EXACT", true);
      const bool cachedVocabulary = environment("FLASH_BATCH_MTP_ORACLE_CACHED_VOCAB");
      uint32_t maximumRowsPerLane = 4;
      uint32_t primeBlocks = 1;
      if (const char *limit = std::getenv("FLASH_BATCH_MTP_ORACLE_ROWS_PER_LANE")) {
        const std::string text(limit);
        size_t consumed = 0;
        const auto parsed = std::stoul(text, &consumed);
        if (consumed != text.size() || parsed < 4 || parsed > 128)
          throw std::invalid_argument("FLASH_BATCH_MTP_ORACLE_ROWS_PER_LANE must be4..128");
        maximumRowsPerLane = static_cast<uint32_t>(parsed);
      }
      if (const char *source = std::getenv("FLASH_BATCH_MTP_ORACLE_PRIME_BLOCKS")) {
        const std::string text(source);
        if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
          throw std::invalid_argument("FLASH_BATCH_MTP_ORACLE_PRIME_BLOCKS must be 1..16");
        size_t consumed = 0;
        const auto parsed = std::stoul(text, &consumed);
        if (consumed != text.size() || parsed < 1 || parsed > 16)
          throw std::invalid_argument("FLASH_BATCH_MTP_ORACLE_PRIME_BLOCKS must be 1..16");
        primeBlocks = static_cast<uint32_t>(parsed);
        if (maximumRowsPerLane <= 4 && primeBlocks != 1)
          throw std::invalid_argument("multiple prime blocks require rows per lane >4");
      }
      metal::MetalBackend backend(argv[1]);
      const auto weights = FlashWeights::load(backend, argv[2]);
      FlashMTPForward sequential(backend, weights, 4096, 128);
      const uint64_t sequentialPlanned = FlashMTPForward::workspacePlannedBytes(4096, 128) +
          (environment("SPLASH_FLASH_DENSE_CACHE")
              ? FlashMTPForward::denseCachePlannedBytes(weights) : 0);
      if (sequential.workspaceBytes() > sequentialPlanned)
        throw std::runtime_error("sequential head allocation exceeds admission estimate");
      std::unique_ptr<FlashDenseCache> vocabulary;
      if (cachedVocabulary) {
        const std::array<std::string, 1> prefix{"language_model.lm_head"};
        vocabulary = std::make_unique<FlashDenseCache>(backend, weights, prefix);
      }
      FlashBatchMTPForward batch(sequential, 4, maximumRowsPerLane,
          vocabulary ? &vocabulary->tensor("language_model.lm_head") : nullptr);
      if (std::string_view(batch.attentionRouteSemantics()) != sequential.attentionRouteSemantics())
        throw std::runtime_error("sequential and batch head attention policies differ");
      if (batch.workspaceBytes() > FlashBatchMTPForward::workspacePlannedBytes(
          4096, 4, maximumRowsPerLane, cachedVocabulary))
        throw std::runtime_error("batch head allocation exceeds admission estimate");
      std::array<FlashMTPState, 4> reference, actual;
      for (uint32_t lane = 0; lane < 4; ++lane) {
        reference[lane] = sequential.createState(); actual[lane] = sequential.createState();
      }
      const auto input = backend.allocateBuffer(uint64_t{4} * maximumRowsPerLane * kWidth * 2,
          metal::BufferStorage::Shared, "flash-batch-mtp-oracle-input");
      Comparison comparison;
      std::vector<uint16_t> latestBatchHidden;
      metal::MetalBuffer latestBorrowedHidden;
      const auto run = [&](std::span<const uint32_t> laneIDs,
          std::span<const uint32_t> counts, FlashMTPLogits mode, uint32_t salt,
          bool alias = false) {
        uint32_t total = 0;
        for (uint32_t count : counts) total += count;
        std::vector<FlashMTPState *> states;
        std::vector<uint32_t> tokens, offsets{0};
        auto *values = static_cast<uint16_t *>(input.contents());
        for (uint32_t lane = 0; lane < laneIDs.size(); ++lane) {
          const uint32_t id = laneIDs[lane];
          states.push_back(&actual[id]);
          if (reference[id].logicalLength() != actual[id].logicalLength())
            throw std::runtime_error("sequential/batch head logical offsets differ before call");
          for (uint32_t row = 0; row < counts[lane]; ++row) {
            const uint64_t position = reference[id].logicalLength() + row;
            tokens.push_back(static_cast<uint32_t>((id * 173 + position * 31 + salt * 17) % 240000));
            for (uint32_t column = 0; column < kWidth; ++column) {
              const float value = std::sin(float(column % 997) * 0.041f +
                  float(position) * 0.173f + float(id + salt) * 0.3f) *
                  (0.2f + float((column / 2560 + id) % 4) * 0.12f);
              values[uint64_t{offsets.back() + row} * kWidth + column] = bf16(value);
            }
          }
          offsets.push_back(offsets.back() + counts[lane]);
        }
        const auto featureInput = alias ? latestBorrowedHidden
            : backend.view(input, 0, uint64_t{total} * kWidth * 2);
        const auto featureBits = copied(featureInput, uint64_t{total} * kWidth);
        const auto result = batch.forward(states, featureInput, tokens, counts, mode);
        ++comparison.commands; ++comparison.scenarios;
        comparison.batchGpu += result.timing.gpuSeconds;
        if (result.lanes != laneIDs.size() || result.laneOffsets != offsets ||
            result.logicalLengths.size() != laneIDs.size() ||
            result.logitRows != (mode == FlashMTPLogits::None ? 0 :
                mode == FlashMTPLogits::Last ? laneIDs.size() : total))
          throw std::runtime_error("batch head result has wrong real lane/row metadata");
        latestBatchHidden = copied(result.hiddenBF16, uint64_t{total} * kWidth);
        latestBorrowedHidden = result.hiddenBF16;
        const auto actualLogits = mode == FlashMTPLogits::None ? std::vector<uint16_t>{}
            : copied(result.logitsBF16, uint64_t{result.logitRows} * kVocabulary);
        // Copy every feature before reusing input to feed the single-request
        // oracle. The batch result belongs to a different scratch arena.
        std::memcpy(input.contents(), featureBits.data(), featureBits.size() * 2);
        for (uint32_t lane = 0; lane < laneIDs.size(); ++lane) {
          const uint32_t id = laneIDs[lane], begin = offsets[lane], rows = counts[lane];
          const auto hidden = backend.view(input, uint64_t{begin} * kWidth * 2,
              uint64_t{rows} * kWidth * 2);
          const auto expected = sequential.forward(reference[id], hidden,
              std::span<const uint32_t>(tokens).subspan(begin, rows), mode);
          ++comparison.commands;
          comparison.sequentialGpu += expected.timing.gpuSeconds;
          const auto expectedHidden = copied(expected.hiddenBF16, uint64_t{rows} * kWidth);
          const std::string label = "scenario" + std::to_string(salt) + "-lane" +
              std::to_string(id) + "-count" + std::to_string(rows);
          comparison.compare(std::span<const uint16_t>(latestBatchHidden).subspan(
              uint64_t{begin} * kWidth, uint64_t{rows} * kWidth), expectedHidden,
              exact, (label + "-premixer").c_str());
          if (mode != FlashMTPLogits::None) {
            const uint32_t logitBegin = mode == FlashMTPLogits::Last ? lane : begin;
            const auto expectedLogits = copied(expected.logitsBF16, uint64_t{expected.logitRows} * kVocabulary);
            comparison.compare(std::span<const uint16_t>(actualLogits).subspan(
                uint64_t{logitBegin} * kVocabulary, expectedLogits.size()),
                expectedLogits, exact, (label + "-logits").c_str());
          }
          if (result.logicalLengths[lane] != reference[id].logicalLength() ||
              actual[id].logicalLength() != reference[id].logicalLength())
            throw std::runtime_error("batch head advanced wrong independent head offset");
        }
      };
      const std::array<uint32_t, 4> all{0, 1, 2, 3}, one{1, 1, 1, 1}, ragged{1, 2, 3, 4};
      run(all, ragged, FlashMTPLogits::None, 1);
      run(all, one, FlashMTPLogits::Last, 2);
      run(all, one, FlashMTPLogits::Last, 3, true); // Same-arena input alias.
      run(all, {std::array<uint32_t, 4>{4, 3, 2, 1}}, FlashMTPLogits::All, 4);
      for (uint32_t lane = 0; lane < 4; ++lane) {
        const auto keep = reference[lane].logicalLength() - (lane % 3 + 1);
        sequential.truncate(reference[lane], keep); sequential.truncate(actual[lane], keep);
      }
      run(all, {std::array<uint32_t, 4>{2, 4, 1, 3}}, FlashMTPLogits::Last, 5);
      run({std::array<uint32_t, 2>{3, 1}}, {std::array<uint32_t, 2>{4, 1}}, FlashMTPLogits::All, 6);
      run({std::array<uint32_t, 1>{2}}, {std::array<uint32_t, 1>{3}}, FlashMTPLogits::Last, 7);
      run({std::array<uint32_t, 3>{1, 3, 0}}, {std::array<uint32_t, 3>{1, 3, 2}}, FlashMTPLogits::None, 8);
      run(all, {std::array<uint32_t, 4>{4, 4, 4, 4}}, FlashMTPLogits::All, 9);
      if (maximumRowsPerLane > 4) {
        const std::array<uint32_t, 4> prime{
            maximumRowsPerLane, maximumRowsPerLane, maximumRowsPerLane, maximumRowsPerLane};
        for (uint32_t block = 0; block < primeBlocks; ++block)
          run(all, prime, FlashMTPLogits::None, 10 + block * 2);
        run(all, one, FlashMTPLogits::Last, 11 + (primeBlocks - 1) * 2);
        if (primeBlocks > 1) {
          // Exercise long-cache rollback and both measured decode windows,
          // with distinct real offsets maintained by each head state.
          for (uint32_t id = 0; id < 4; ++id) {
            const uint64_t keep = reference[id].logicalLength() - (id + 1);
            sequential.truncate(reference[id], keep);
            sequential.truncate(actual[id], keep);
          }
          run(all, ragged, FlashMTPLogits::All, 100);
          if (maximumRowsPerLane >= 8)
            run(all, {std::array<uint32_t, 4>{5, 6, 7, 8}}, FlashMTPLogits::Last, 101);
        }
      }
      const auto before = actual[0].logicalLength();
      std::array<FlashMTPState *, 2> duplicate{&actual[0], &actual[0]};
      invalid([&] { (void)batch.forward(duplicate, input, std::array<uint32_t, 2>{1, 2},
          std::array<uint32_t, 2>{1, 1}); });
      std::array<FlashMTPState *, 1> lane{&actual[0]};
      invalid([&] { (void)batch.forward(lane, input, std::array<uint32_t, 1>{1},
          std::array<uint32_t, 1>{0}); });
      invalid([&] { (void)batch.forward(lane, input, std::array<uint32_t, 1>{kVocabulary},
          std::array<uint32_t, 1>{1}); });
      invalid([&] { (void)batch.forward(lane, input, std::array<uint32_t, 5>{1, 2, 3, 4, 5},
          std::array<uint32_t, 1>{maximumRowsPerLane + 1}); });
      invalid([&] { (void)batch.forward(lane, input, std::span<const uint32_t>{},
          std::array<uint32_t, 1>{1}); });
      invalid([&] { (void)batch.forward(lane, backend.view(input, 0, kWidth * 2 - 2),
          std::array<uint32_t, 1>{1}, std::array<uint32_t, 1>{1}); });
      invalid([&] { (void)batch.forward(lane, input, std::array<uint32_t, 1>{1},
          std::array<uint32_t, 1>{1}, static_cast<FlashMTPLogits>(255)); });
      FlashMTPState unrelated;
      std::array<FlashMTPState *, 1> unowned{&unrelated};
      invalid([&] { (void)batch.forward(unowned, input, std::array<uint32_t, 1>{1},
          std::array<uint32_t, 1>{1}); });
      if (maximumRowsPerLane > 4) {
        std::array<FlashMTPState *, 4> allStates{&actual[0], &actual[1], &actual[2], &actual[3]};
        invalid([&] { (void)batch.forward(allStates, input,
            std::array<uint32_t, 20>{}, std::array<uint32_t, 4>{5, 5, 5, 5}, FlashMTPLogits::All); });
      }
      if (actual[0].logicalLength() != before || actual[0].poisoned())
        throw std::runtime_error("invalid host batch mutated/poisoned healthy state");
      std::ofstream report(argv[3]);
      report << "{\"pass\":true,\"gpu_work\":true,\"exact_required\":" << (exact ? "true" : "false")
             << ",\"semantics\":" << json::quote(kFlashBatchMTPSemantics)
             << ",\"attention_route\":" << json::quote(batch.attentionRouteSemantics())
             << ",\"sequential_attention_route\":" << json::quote(sequential.attentionRouteSemantics())
             << ",\"dense_cache\":" << (environment("SPLASH_FLASH_DENSE_CACHE") ? "true" : "false")
             << ",\"fused_hc\":" << (environment("SPLASH_FLASH_FUSE_HC") ? "true" : "false")
             << ",\"shared_cached_vocabulary\":" << (cachedVocabulary ? "true" : "false")
             << ",\"maximum_rows_per_lane\":" << maximumRowsPerLane
             << ",\"prime_blocks\":" << primeBlocks
             << ",\"sequential_workspace_bytes\":" << sequential.workspaceBytes()
             << ",\"sequential_planned_workspace_bytes\":" << sequentialPlanned
             << ",\"workspace_bytes\":" << batch.workspaceBytes()
             << ",\"planned_workspace_bytes\":" << FlashBatchMTPForward::workspacePlannedBytes(
                 4096, 4, maximumRowsPerLane, cachedVocabulary)
             << ",\"scenarios\":" << comparison.scenarios << ",\"commands\":" << comparison.commands
             << ",\"elements\":" << comparison.elements << ",\"differences\":" << comparison.differences
             << ",\"max_relative_l2\":" << comparison.maximumRelativeL2
             << ",\"sequential_gpu_seconds\":" << comparison.sequentialGpu
             << ",\"batch_gpu_seconds\":" << comparison.batchGpu << "}\n";
      if (!report) throw std::runtime_error("could not write batch head oracle report");
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "flash-batch-mtp-oracle: " << error.what() << '\n'; return 1;
    }
  }
}
