#include "flash/FlashForward.hpp"
#include "flash/FlashDenseCache.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"
#include "metal/ProfilingJson.hpp"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <vector>

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 5)
        throw std::invalid_argument("usage: flash-forward-oracle METALLIB PACKAGE TOKENS_JSON REPORT_JSON");
      NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[3]]];
      id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:nullptr];
      if (![object isKindOfClass:[NSArray class]])
        throw std::invalid_argument("tokens must be a JSON array");
      std::vector<uint32_t> tokens;
      for (NSNumber *word in static_cast<NSArray *>(object)) {
        if (![word isKindOfClass:[NSNumber class]] || word.longLongValue < 0 ||
            word.unsignedLongLongValue >= 248320)
          throw std::invalid_argument("invalid prompt token");
        tokens.push_back(static_cast<uint32_t>(word.unsignedLongLongValue));
      }
      if (tokens.empty()) throw std::invalid_argument("empty prompt");
      splash::metal::MetalBackend backend(argv[1]);
      const char *profile = std::getenv("FLASH_ORACLE_PROFILE");
      const std::string profileMode = profile ? profile : "off";
      if (profileMode == "stage")
        backend.setCommandDispatchProfiling(
            splash::metal::CommandDispatchProfilingMode::StagePerDispatch);
      else if (profileMode == "command")
        backend.setCommandDispatchProfiling(
            splash::metal::CommandDispatchProfilingMode::Command);
      else if (profileMode != "off")
        throw std::invalid_argument("FLASH_ORACLE_PROFILE must be off, command or stage");
      std::ofstream trace;
      if (profileMode != "off") {
        trace.open(std::string(argv[4]) + ".trace.jsonl");
        if (!trace) throw std::runtime_error("could not open profiling trace");
      }
      auto takeProfiles = [&](const char *phase, uint32_t rows) {
        for (const auto &entry : backend.takeCommandDispatchProfiles()) {
          trace << "{\"phase\":" << splash::json::quote(phase)
                << ",\"rows\":" << rows << ",\"command\":";
          splash::profiling::writeJson(trace, entry);
          trace << "}\n";
        }
      };
      uint32_t maximumTokens = 32;
      if (const char *limit = std::getenv("FLASH_ORACLE_MAX_TOKENS")) {
        const std::string value(limit);
        size_t consumed = 0;
        const unsigned long parsed = std::stoul(value, &consumed);
        if (consumed != value.size() || parsed < 1 || parsed > 256)
          throw std::invalid_argument("FLASH_ORACLE_MAX_TOKENS must be 1 through 256");
        maximumTokens = static_cast<uint32_t>(parsed);
      }
      uint32_t prefillRows = 128;
      if (const char *setting = std::getenv("FLASH_ORACLE_PREFILL_ROWS")) {
        const std::string value(setting);
        size_t consumed = 0;
        const unsigned long parsed = std::stoul(value, &consumed);
        constexpr uint32_t choices[]{32, 64, 128, 256, 512, 1024, 2048};
        if (consumed != value.size() ||
            std::find(std::begin(choices), std::end(choices), parsed) == std::end(choices))
          throw std::invalid_argument("FLASH_ORACLE_PREFILL_ROWS must be 32,64,128,256,512,1024 or 2048");
        prefillRows = static_cast<uint32_t>(parsed);
      }
      const auto weights = splash::flash::FlashWeights::load(backend, argv[2]);
      bool requestResidency = false;
      if (const char *resident = std::getenv("FLASH_ORACLE_RESIDENT")) {
        const std::string value(resident);
        if (value != "0" && value != "1")
          throw std::invalid_argument("FLASH_ORACLE_RESIDENT must be 0 or 1");
        requestResidency = value == "1";
      }
      splash::metal::ResidencyLease residency;
      if (requestResidency) {
        const auto buffers = weights.immutableWeightBuffers();
        residency = backend.requestWeightResidency(buffers, "flash-oracle-weights");
      }
      const uint32_t capacity = std::max<uint32_t>(4096, tokens.size() + maximumTokens);
      const uint64_t physical = NSProcessInfo.processInfo.physicalMemory;
      const uint64_t reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
      splash::engine::MemoryGovernor governor(backend, physical - reserve, reserve);
      uint64_t planned = splash::flash::FlashForward::workspacePlannedBytes(capacity, prefillRows) +
          splash::flash::FlashForward::requestStateBytes(capacity) +
          splash::flash::FlashForward::expertCachePlannedBytes(weights);
      planned += splash::flash::FlashForward::floatDenseCachePlannedBytes(weights);
      planned += splash::flash::FlashForward::int8HeadPlannedBytes(weights);
      const auto enabled = [](const char *name) {
        const char *value = std::getenv(name);
        return value && std::string(value) == "1";
      };
      if (enabled("SPLASH_FLASH_DENSE_CACHE"))
        planned += splash::flash::FlashDenseCache::plannedBytes(
            weights, splash::flash::FlashDenseCache::defaultPrefixes(weights, true));
      if (enabled("SPLASH_FLASH_QSA_F32")) planned += 16ULL << 20;
      if (enabled("SPLASH_FLASH_BLOCKED_MOE") && prefillRows >= 256)
        planned += uint64_t{prefillRows} * 10 * 11520 + (2ULL << 20);
      auto reservation = governor.tryReserve(planned);
      if (!reservation) throw std::runtime_error("governor denied oracle workspace before construction");
      splash::flash::FlashForward forward(backend, weights, capacity, prefillRows);
      takeProfiles("startup", 0);
      std::ofstream routes;
      if (const char *capture = std::getenv("SPLASH_FLASH_CAPTURE_EXPERT_IDS");
          capture && std::string(capture) == "1") {
        routes.open(std::string(argv[4]) + ".expert-ids.jsonl");
        if (!routes) throw std::runtime_error("could not open expert-route capture");
      }
      const auto takeRoutes = [&](const char *phase) {
        if (!routes.is_open()) return;
        uint32_t rows = 0, stride = 0;
        const auto buffer = forward.capturedExpertIDs(rows, stride);
        if (!buffer || !buffer.contents() || !rows || rows > stride)
          throw std::runtime_error("invalid expert-route capture extent");
        const auto *ids = static_cast<const int64_t *>(buffer.contents());
        routes << "{\"source_identity\":" << splash::json::quote(weights.sourceIdentity())
               << ",\"phase\":" << splash::json::quote(phase) << ",\"rows\":" << rows
               << ",\"layer_ids\":[";
        for (uint32_t layer = 0; layer < weights.descriptor().layers; ++layer) {
          if (layer) routes << ',';
          routes << '[';
          for (uint32_t route = 0; route < rows * 10; ++route) {
            const auto id = ids[uint64_t{layer} * stride * 10 + route];
            if (id < 0 || id >= 512) throw std::runtime_error("invalid captured expert ID");
            if (route) routes << ',';
            routes << id;
          }
          routes << ']';
        }
        routes << "]}\n";
        if (!routes) throw std::runtime_error("could not write expert-route capture");
      };
      auto state = forward.createState();
      if (forward.workspaceBytes() + splash::flash::FlashForward::requestStateBytes(capacity) > planned)
        throw std::runtime_error("oracle allocation exceeds reserved plan");
      reservation->commit();
      std::vector<uint32_t> output;
      double gpu = 0.0, wall = 0.0;
      std::vector<double> prefillGpu, decodeGpu;
      splash::flash::FlashForwardResult result;
      for (size_t begin = 0; begin < tokens.size(); begin += prefillRows) {
        result = forward.forward(state, std::span<const uint32_t>(tokens).subspan(
            begin, std::min<size_t>(prefillRows, tokens.size() - begin)));
        gpu += result.timing.gpuSeconds;
        wall += result.timing.wallSeconds;
        prefillGpu.push_back(result.timing.gpuSeconds);
        takeProfiles("prefill", std::min<size_t>(prefillRows, tokens.size() - begin));
        takeRoutes("prefill");
      }
      for (uint32_t index = 0; index < maximumTokens; ++index) {
        const auto *logits = static_cast<const uint16_t *>(result.logitsBF16.contents());
        uint32_t best = 0;
        float maximum = -INFINITY;
        for (uint32_t token = 0; token < 248320; ++token) {
          const float score = std::bit_cast<float>(uint32_t(logits[token]) << 16);
          if (!std::isfinite(score)) throw std::runtime_error("nonfinite logits");
          if (score > maximum) { maximum = score;best = token; }
        }
        output.push_back(best);
        if (best == 248044 || best == 248046 || index + 1 == maximumTokens) break;
        result = forward.forward(state, std::span<const uint32_t>(&best, 1));
        gpu += result.timing.gpuSeconds;
        wall += result.timing.wallSeconds;
        decodeGpu.push_back(result.timing.gpuSeconds);
        takeProfiles("decode", 1);
        takeRoutes("decode");
        std::cerr << "native_flash token=" << index + 1 << " gpu_ms="
                  << result.timing.gpuSeconds * 1000 << '\n';
      }
      std::ofstream out(argv[4]);
      out << "{\"pass\":true,\"source_identity\":"
          << splash::json::quote(weights.sourceIdentity())
          << ",\"manifest_fingerprint\":" << splash::json::quote(weights.manifestFingerprint())
          << ",\"forward_semantics\":" << splash::json::quote(splash::flash::kFlashForwardSemantics)
          << ",\"kernel_routes\":" << splash::json::quote(forward.kernelRoutes())
          << ",\"residency_requested\":" << (requestResidency ? "true" : "false")
          << ",\"resident_buffer_count\":" << residency.bufferCount()
          << ",\"resident_allocation_bytes\":" << residency.byteCount()
          << ",\"gpu_seconds\":" << gpu << ",\"wall_seconds\":" << wall
          << ",\"profile_mode\":" << splash::json::quote(profileMode)
          << ",\"maximum_prefill_rows\":" << prefillRows
          << ",\"prompt_tokens\":" << tokens.size() << ",\"output_tokens\":[";
      for (size_t i = 0; i < output.size(); ++i) { if(i)out << ',';out << output[i]; }
      out << "],\"prefill_gpu_seconds\":[";
      for (size_t i = 0; i < prefillGpu.size(); ++i) { if(i)out << ',';out << prefillGpu[i]; }
      out << "],\"decode_gpu_seconds\":[";
      for (size_t i = 0; i < decodeGpu.size(); ++i) { if(i)out << ',';out << decodeGpu[i]; }
      out << "]}\n";
      if (!out) throw std::runtime_error("could not write report");
      if (trace.is_open() && !trace) throw std::runtime_error("could not write profiling trace");
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "flash-forward-oracle: " << error.what() << '\n';return 1;
    }
  }
}
