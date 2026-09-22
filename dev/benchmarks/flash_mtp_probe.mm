#include "flash/FlashForward.hpp"
#include "flash/FlashMTP.hpp"
#include "flash/FlashMTPWindow.hpp"
#include "engine/Json.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using namespace splash;
constexpr uint32_t kVocabulary = 248320;
constexpr uint32_t kHyper = 10240;
bool stop(uint32_t token) { return token == 248044 || token == 248046; }
uint32_t argmax(const metal::MetalBuffer &buffer, uint32_t row = 0) {
  if (!buffer || !buffer.contents() || buffer.sizeBytes() < uint64_t{row + 1} * kVocabulary * 2)
    throw std::logic_error("MTP probe logits have invalid extent/storage");
  const auto *values = static_cast<const uint16_t *>(buffer.contents()) + uint64_t{row} * kVocabulary;
  uint32_t best = 0;
  float maximum = -INFINITY;
  for (uint32_t token = 0; token < kVocabulary; ++token) {
    const float value = std::bit_cast<float>(uint32_t{values[token]} << 16);
    if (!std::isfinite(value)) throw std::runtime_error("MTP probe found nonfinite logits");
    if (value > maximum) { maximum = value; best = token; }
  }
  return best;
}
uint32_t environmentUInt(const char *name, uint32_t fallback, uint32_t minimum, uint32_t maximum) {
  const char *source = std::getenv(name);
  if (!source) return fallback;
  const std::string value(source);
  size_t consumed = 0;
  const auto parsed = std::stoul(value, &consumed);
  if (consumed != value.size() || parsed < minimum || parsed > maximum)
    throw std::invalid_argument(std::string(name) + " is outside its allowed range");
  return static_cast<uint32_t>(parsed);
}
struct Measurement {
  std::vector<uint32_t> output;
  uint64_t cycles = 0, accepted = 0, drafted = 0, committedHeadCalls = 0;
  std::array<uint64_t, flash::kFlashSingletonMaximumVerifyRows> acceptedPrefixes{};
  double prefillWall = 0, decodeWall = 0, prefillGpu = 0, decodeGpu = 0;
  double headGpu = 0, verifyGpu = 0, commitGpu = 0;
  std::string kernelRoutes;
  std::string headAttentionRoute;
};

Measurement runAR(metal::MetalBackend &backend, const flash::FlashWeights &weights,
    std::span<const uint32_t> prompt, uint32_t maximumTokens, uint32_t capacity) {
  Measurement result;
  flash::FlashForward forward(backend, weights, capacity, 128);
  result.kernelRoutes = forward.kernelRoutes();
  auto state = forward.createState();
  flash::FlashForwardResult current;
  const auto prefillBegan = Clock::now();
  for (size_t begin = 0; begin < prompt.size(); begin += 128) {
    current = forward.forward(state, prompt.subspan(begin, std::min<size_t>(128, prompt.size() - begin)));
    result.prefillGpu += current.timing.gpuSeconds;
  }
  result.prefillWall = std::chrono::duration<double>(Clock::now() - prefillBegan).count();
  const auto decodeBegan = Clock::now();
  for (uint32_t index = 0; index < maximumTokens; ++index) {
    const uint32_t token = argmax(current.logitsBF16);
    result.output.push_back(token);
    if (stop(token) || result.output.size() == maximumTokens) break;
    current = forward.forward(state, std::span<const uint32_t>(&token, 1));
    result.decodeGpu += current.timing.gpuSeconds;
    ++result.cycles;
  }
  result.decodeWall = std::chrono::duration<double>(Clock::now() - decodeBegan).count();
  return result;
}

Measurement runMTP(metal::MetalBackend &backend, const flash::FlashWeights &weights,
    std::span<const uint32_t> prompt, uint32_t maximumTokens, uint32_t capacity, uint32_t depth) {
  Measurement result;
  flash::FlashForward target(backend, weights, capacity, 128, depth + 1);
  result.kernelRoutes = target.kernelRoutes();
  flash::FlashMTPForward head(backend, weights, capacity, 128);
  result.headAttentionRoute = head.attentionRouteSemantics();
  const char *denseCacheFlag = std::getenv("SPLASH_FLASH_DENSE_CACHE");
  const uint64_t headPlanned = flash::FlashMTPForward::workspacePlannedBytes(capacity, 128) +
      (denseCacheFlag && std::string_view(denseCacheFlag) == "1"
          ? flash::FlashMTPForward::denseCachePlannedBytes(weights) : 0);
  if (head.workspaceBytes() > headPlanned)
    throw std::runtime_error("MTP probe head workspace exceeds admission estimate");
  auto targetState = target.createState();
  auto headState = head.createState();
  flash::FlashForwardResult current;
  const auto prefillBegan = Clock::now();
  for (size_t begin = 0; begin < prompt.size(); begin += 128) {
    const size_t rows = std::min<size_t>(128, prompt.size() - begin);
    current = target.forward(targetState, prompt.subspan(begin, rows), false, true);
    result.prefillGpu += current.timing.gpuSeconds;
    const size_t primeRows = std::min<size_t>(rows, prompt.size() - begin - 1);
    if (primeRows) {
      const auto hidden = backend.view(current.hiddenBF16, 0, primeRows * kHyper * 2);
      const auto folded = head.forward(headState, hidden, prompt.subspan(begin + 1, primeRows), flash::FlashMTPLogits::None);
      result.prefillGpu += folded.timing.gpuSeconds;
    }
  }
  if (headState.logicalLength() != prompt.size() - 1)
    throw std::logic_error("MTP prompt priming offset mismatch");
  result.prefillWall = std::chrono::duration<double>(Clock::now() - prefillBegan).count();
  uint32_t anchor = argmax(current.logitsBF16);
  result.output.push_back(anchor);
  if (stop(anchor) || result.output.size() == maximumTokens) return result;
  const uint32_t finalPromptRows = (prompt.size() - 1) % 128 + 1;
  auto previousHidden = backend.view(current.hiddenBF16,
      uint64_t{finalPromptRows - 1} * kHyper * 2, uint64_t{kHyper} * 2);
  std::vector<uint32_t> foldTokens{anchor};
  const auto decodeBegan = Clock::now();
  for (;;) {
    const uint32_t activeDepth = std::min<uint32_t>(depth,
        static_cast<uint32_t>(maximumTokens - result.output.size() - 1));
    flash::FlashMTPResult folded;
    // Keep committed head projections on the qualified raw-coefficient
    // decode path. A sixteen-row call would switch to BF16 cached matrices.
    for (uint32_t begin = 0; begin < foldTokens.size();) {
      const auto count = flash::flashMTPCommittedFoldChunkRows(
          static_cast<uint32_t>(foldTokens.size() - begin));
      if (!count) throw std::logic_error("MTP probe committed fold exceeds bounded chunks");
      const bool last = begin + *count == foldTokens.size();
      const auto hidden = backend.view(previousHidden, uint64_t{begin} * kHyper * 2,
          uint64_t{*count} * kHyper * 2);
      folded = head.forward(headState, hidden, std::span(foldTokens).subspan(begin, *count),
          last && activeDepth ? flash::FlashMTPLogits::Last : flash::FlashMTPLogits::None);
      result.headGpu += folded.timing.gpuSeconds;
      result.decodeGpu += folded.timing.gpuSeconds;
      ++result.committedHeadCalls;
      begin += *count;
    }
    const uint64_t foldedLength = headState.logicalLength();
    std::vector<uint32_t> inputs{anchor};
    auto headResult = folded;
    for (uint32_t draft = 0; draft < activeDepth; ++draft) {
      const uint32_t token = argmax(headResult.logitsBF16);
      inputs.push_back(token);
      if (stop(token)) break;
      if (draft + 1 < activeDepth) {
        const auto chainHidden = backend.view(headResult.hiddenBF16,
            uint64_t{headResult.hiddenRows - 1} * kHyper * 2, uint64_t{kHyper} * 2);
        headResult = head.forward(headState, chainHidden, std::span<const uint32_t>(&token, 1));
        result.headGpu += headResult.timing.gpuSeconds;
        result.decodeGpu += headResult.timing.gpuSeconds;
      }
    }
    const auto verified = target.verify(targetState, inputs);
    result.verifyGpu += verified.timing.gpuSeconds;
    result.decodeGpu += verified.timing.gpuSeconds;
    ++result.cycles;
    const uint32_t drafts = static_cast<uint32_t>(inputs.size() - 1);
    result.drafted += drafts;
    std::array<uint32_t, flash::kFlashSingletonMaximumVerifyRows> predictions{};
    for (uint32_t row = 0; row < verified.logitRows; ++row)
      predictions[row] = argmax(verified.logitsBF16, row);
    const std::array<uint32_t, 2> stops{248044, 248046};
    const auto accepted = flash::flashMTPAcceptGreedyPrefix(inputs,
        std::span(predictions).first(inputs.size()),
        static_cast<uint32_t>(maximumTokens - result.output.size()), stops);
    if (!accepted) throw std::logic_error("MTP probe acceptance has invalid geometry");
    // Stop/max budget can end inside a verified prefix; only materialize the
    // exact input prefix needed by its final emitted prediction.
    const uint32_t retained = accepted->retainedRows;
    const bool finished = accepted->finish != flash::FlashMTPPrefixFinish::None;
    result.output.insert(result.output.end(), accepted->output.begin(), accepted->output.begin() + retained);
    const auto committed = target.commitVerify(targetState, retained);
    result.commitGpu += committed.gpuSeconds;
    result.decodeGpu += committed.gpuSeconds;
    result.accepted += retained - 1;
    ++result.acceptedPrefixes[retained - 1];
    head.truncate(headState, foldedLength);
    std::cerr << "native_mtp cycle=" << result.cycles << " drafted=" << drafts
              << " accepted=" << retained - 1 << " outputs=" << result.output.size()
              << " verify_gpu_ms=" << verified.timing.gpuSeconds * 1000 << '\n';
    if (finished) break;
    // Fold the complete committed pair window in bounded head chunks next
    // cycle. Its last row drives draft1. Target hidden remains valid until the next
    // target.verify() call; the head has an independent scratch arena.
    previousHidden = backend.view(verified.hiddenBF16, 0, uint64_t{retained} * kHyper * 2);
    foldTokens.assign(result.output.end() - retained, result.output.end());
    anchor = result.output.back();
    if (targetState.logicalLength() + 1 != prompt.size() + result.output.size())
      throw std::logic_error("MTP target commit offset mismatch");
  }
  result.decodeWall = std::chrono::duration<double>(Clock::now() - decodeBegan).count();
  return result;
}
void writeMeasurement(std::ostream &out, const Measurement &result) {
  out << "{\"kernel_routes\":" << json::quote(result.kernelRoutes)
      << ",\"head_attention_route\":" << (result.headAttentionRoute.empty()
          ? "null" : json::quote(result.headAttentionRoute)) << ",\"output_tokens\":[";
  for (size_t i = 0; i < result.output.size(); ++i) { if (i) out << ','; out << result.output[i]; }
  out << "],\"cycles\":" << result.cycles << ",\"accepted\":" << result.accepted
      << ",\"drafted\":" << result.drafted << ",\"committed_head_calls\":" << result.committedHeadCalls
      << ",\"accepted_prefix_histogram\":[";
  for (size_t i = 0; i < result.acceptedPrefixes.size(); ++i) { if (i) out << ','; out << result.acceptedPrefixes[i]; }
  out << "],\"prefill_wall_seconds\":" << result.prefillWall
      << ",\"decode_wall_seconds\":" << result.decodeWall
      << ",\"prefill_gpu_seconds\":" << result.prefillGpu
      << ",\"decode_gpu_seconds\":" << result.decodeGpu
      << ",\"head_gpu_seconds\":" << result.headGpu
      << ",\"verify_gpu_seconds\":" << result.verifyGpu
      << ",\"commit_gpu_seconds\":" << result.commitGpu
      << ",\"decode_tokens_per_second\":"
      << (result.decodeWall > 0 ? double(result.output.size() - 1) / result.decodeWall : 0.0) << '}';
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 5)
        throw std::invalid_argument("usage: flash-mtp-probe METALLIB PACKAGE TOKENS_JSON REPORT_JSON");
      NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[3]]];
      if (!data) throw std::invalid_argument("could not read prompt tokens");
      id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:nullptr];
      if (![object isKindOfClass:[NSArray class]]) throw std::invalid_argument("prompt must be a JSON array");
      std::vector<uint32_t> prompt;
      for (id value in static_cast<NSArray *>(object)) {
        if (![value isKindOfClass:[NSNumber class]]) throw std::invalid_argument("prompt token must be numeric");
        NSNumber *token = value;
        if (token.longLongValue < 0 || token.unsignedLongLongValue >= kVocabulary)
          throw std::invalid_argument("prompt token out of vocabulary");
        prompt.push_back(static_cast<uint32_t>(token.unsignedLongLongValue));
      }
      if (prompt.empty()) throw std::invalid_argument("empty prompt");
      const uint32_t maximumTokens = environmentUInt("FLASH_MTP_PROBE_MAX_TOKENS", 64, 1, 512);
      const uint32_t depth = environmentUInt("FLASH_MTP_PROBE_DEPTH", 3, 0,
          flash::kFlashSingletonMaximumMTPDepth);
      const bool residencyEnabled = environmentUInt("FLASH_MTP_PROBE_RESIDENT", 1, 0, 1);
      const char *modeEnvironment = std::getenv("FLASH_MTP_PROBE_MODE");
      const std::string mode = modeEnvironment ? modeEnvironment : "paired";
      if (mode != "ar" && mode != "mtp" && mode != "paired") throw std::invalid_argument("probe mode must be ar, mtp or paired");
      if (prompt.size() + maximumTokens + 4 > 262144) throw std::invalid_argument("prompt/output exceeds native context");
      const uint32_t capacity = std::max<uint32_t>(4096, prompt.size() + maximumTokens + 4);
      metal::MetalBackend backend(argv[1]);
      const auto weights = flash::FlashWeights::load(backend, argv[2]);
      metal::ResidencyLease residency;
      if (residencyEnabled) residency = backend.requestWeightResidency(weights.immutableWeightBuffers(), "flash-mtp-probe-weights");
      Measurement ar, mtp;
      if (mode != "mtp") ar = runAR(backend, weights, prompt, maximumTokens, capacity);
      if (mode != "ar") mtp = runMTP(backend, weights, prompt, maximumTokens, capacity, depth);
      const bool paired = mode == "paired";
      const bool identical = !paired || ar.output == mtp.output;
      std::ofstream out(argv[4]);
      out << "{\"pass\":" << (identical ? "true" : "false") << ",\"mode\":" << json::quote(mode)
          << ",\"paired_tokens_identical\":" << (paired ? (identical ? "true" : "false") : "null")
          << ",\"depth\":" << depth << ",\"prompt_tokens\":" << prompt.size()
          << ",\"source_identity\":" << json::quote(weights.sourceIdentity())
          << ",\"forward_semantics\":" << json::quote(flash::kFlashForwardSemantics)
          << ",\"mtp_semantics\":" << json::quote(flash::kFlashMTPSemantics)
          << ",\"committed_head_fold_maximum_rows\":" << flash::kFlashMTPMaximumCommittedFoldRows
          << ",\"committed_head_fold_semantics\":\"ordered chunks at most8 preserve raw decode coefficient policy; priming remains unchanged\""
          << ",\"residency_requested\":" << (residencyEnabled ? "true" : "false") << ",\"ar\":";
      writeMeasurement(out, ar); out << ",\"mtp\":"; writeMeasurement(out, mtp); out << "}\n";
      if (!out) throw std::runtime_error("could not write MTP probe report");
      return identical ? 0 : 2;
    } catch (const std::exception &error) {
      std::cerr << "flash-mtp-probe: " << error.what() << '\n'; return 1;
    }
  }
}
