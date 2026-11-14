#pragma once

// Development-tool input validation only. No Metal device, model or cache is opened.
#include "LoadOptions.hpp"
#include <fstream>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <CommonCrypto/CommonDigest.h>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <span>
#include <sstream>
#include <vector>

namespace splash::tools {
inline std::string profileJsonString(std::string_view value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << char(c);
    else if (c < 32) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << unsigned(c) << std::dec;
    else out << char(c);
  }
  out << '"';
  return out.str();
}
inline uint32_t profileCount(std::string_view text, uint32_t maximum, const char *what) {
  uint32_t value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || !value || value > maximum)
    throw std::invalid_argument(std::string(what) + " is out of range");
  return value;
}
inline std::string profileNumber(double value) {
  if (!std::isfinite(value) || value < 0) throw std::invalid_argument("invalid profile observation");
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}
inline void writePrivateProfileOutput(const std::string &path, const std::string &text) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (fd < 0) throw std::runtime_error("could not exclusively create profile output");
  size_t offset = 0;
  while (offset < text.size()) {
    const auto written = ::write(fd, text.data() + offset, text.size() - offset);
    if (written <= 0) { ::close(fd); throw std::runtime_error("could not write profile output"); }
    offset += static_cast<size_t>(written);
  }
  if (::close(fd)) throw std::runtime_error("could not close profile output");
}
struct DecodeProfileOptions {
  uint32_t promptTokens = 512, cycles = 4, maxContext = 262144;
  uint64_t maxMemory = 0;
  std::string activations = "auto", state = "fp32", kv = "int8", verify = "auto";
  std::string inputTokens, dispatchList, report, images;
  model::ModelLoadOptions load;
  bool configOnly = false, requireFlashNext = false, promptExplicit = false, contextExplicit = false, textOptions = false;
  std::vector<uint32_t> tokens;
  uint32_t generationReserve() const { return static_cast<uint32_t>(8ull * (9ull * cycles + 1)); }

  static DecodeProfileOptions parse(std::span<const std::string_view> args) {
    DecodeProfileOptions p;
    std::vector<std::string_view> seen;
    for (size_t i = 0; i < args.size();) {
      const auto option = args[i++];
      if (std::find(seen.begin(), seen.end(), option) != seen.end())
        throw std::invalid_argument("duplicate option " + std::string(option));
      seen.push_back(option);
      if (option == "--config-only") { p.configOnly = true; continue; }
      if (option == "--require-flash-next") { p.requireFlashNext = p.textOptions = true; continue; }
      if (i == args.size()) throw std::invalid_argument(std::string(option) + " requires a value");
      const auto value = args[i++];
      if (value.empty()) throw std::invalid_argument(std::string(option) + " requires a nonempty value");
      if (option == "--images") { if (value.empty()) throw std::invalid_argument("empty image grid"); p.images = value; continue; }
      if (option == "--cycles") { p.cycles = profileCount(value, 100000, "cycles"); continue; }
      p.textOptions = true;
      if (option == "--prompt-tokens") { p.promptTokens = profileCount(value, 1048576, "prompt tokens"); p.promptExplicit = true; }
      else if (option == "--max-context") { p.maxContext = profileCount(value, 1048576, "max context"); p.contextExplicit = true; }
      else if (option == "--max-memory") {
        p.maxMemory = parseBytes(value);
        if (!p.maxMemory) throw std::invalid_argument("--max-memory requires positive bytes or K/M/G");
      } else if (option == "--activations") {
        if (value != "auto" && value != "bf16") throw std::invalid_argument("--activations requires auto or bf16");
        p.activations = value;
      } else if (option == "--state-format") {
        if (value != "fp32" && value != "bf16") throw std::invalid_argument("--state-format requires fp32 or bf16");
        p.state = value;
      } else if (option == "--kv-format") {
        if (value != "int8" && value != "bf16") throw std::invalid_argument("--kv-format requires int8 or bf16");
        p.kv = value;
      } else if (option == "--verify-rows") {
        if (value != "auto" && value != "all") throw std::invalid_argument("--verify-rows requires auto or all");
        p.verify = value;
      } else if (option == "--input-tokens") p.inputTokens = value;
      else if (option == "--dispatch-list") p.dispatchList = value;
      else if (option == "--report") p.report = value;
      else if (!takeLoadOption(option, value, p.load)) throw std::invalid_argument("unknown option " + std::string(option));
    }
    if (!p.images.empty() && p.textOptions) throw std::invalid_argument("--images takes no text options");
    if (!p.inputTokens.empty()) {
      if (p.promptExplicit) throw std::invalid_argument("--input-tokens and --prompt-tokens are exclusive");
      if (!std::filesystem::is_regular_file(p.inputTokens) || std::filesystem::file_size(p.inputTokens) > uint64_t{p.maxContext} * 11)
        throw std::invalid_argument("token fixture must be a bounded regular file");
      std::ifstream input(p.inputTokens);
      if (!input) throw std::invalid_argument("cannot read token fixture");
      std::string word;
      while (input >> word) {
        if (word.size() > 10 || p.tokens.size() >= p.maxContext)
          throw std::invalid_argument("token fixture exceeds its bounds");
        uint32_t token = 0;
        const auto [end, error] = std::from_chars(word.data(), word.data() + word.size(), token);
        if (error != std::errc{} || end != word.data() + word.size()) throw std::invalid_argument("invalid token ID");
        p.tokens.push_back(token);
      }
      if (!input.eof() || p.tokens.empty()) throw std::invalid_argument("empty or unreadable token fixture");
      p.promptTokens = static_cast<uint32_t>(p.tokens.size());
    }
    // Warmup + ordinary and replayed B1/B2/B3/B4 cycles can advance a lane.
    const uint64_t reserve = 8ull * (9ull * p.cycles + 1);
    if (p.images.empty() && (p.promptTokens > p.maxContext || reserve > p.maxContext - p.promptTokens))
      throw std::invalid_argument("prompt and profiling cycles exceed max context");
    if (p.requireFlashNext && (p.activations != "bf16" || p.state != "bf16" || p.kv != "int8" ||
        p.maxMemory != (96ull << 30) || !p.contextExplicit || p.maxContext != 262144 || p.inputTokens.empty()))
      throw std::invalid_argument("matched Flash-Next requires BF16 activations/state, INT8 KV, 96G, 262144 context and a token fixture");
    if (!p.dispatchList.empty() && std::filesystem::exists(p.dispatchList)) throw std::invalid_argument("dispatch list must be a new file");
    if (!p.report.empty() && !p.dispatchList.empty() && std::filesystem::absolute(p.report).lexically_normal() == std::filesystem::absolute(p.dispatchList).lexically_normal())
      throw std::invalid_argument("report and dispatch list must differ");
    if (!p.report.empty() && std::filesystem::exists(p.report)) throw std::invalid_argument("report must be a new file");
    return p;
  }
  std::string tokenDigest() const {
    if (tokens.empty()) return "null";
    std::vector<unsigned char> bytes;
    bytes.reserve(tokens.size() * 4);
    for (uint32_t token : tokens) for (unsigned shift = 0; shift < 32; shift += 8) bytes.push_back(token >> shift);
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest;
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
    std::ostringstream out;
    for (auto byte : digest) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    return profileJsonString(out.str());
  }
  std::string json() const {
    std::ostringstream out;
    out << "{\"schema\":1,\"scope\":\"native_runtime_fixture\",\"input_kind\":"
        << profileJsonString(!images.empty() ? "synthetic_image_grid" : inputTokens.empty() ? "synthetic" : "private_token_fixture")
        << ",\"input_token_ids_sha256_le\":" << tokenDigest()
        << ",\"prompt_tokens\":" << promptTokens << ",\"cycles\":" << cycles
        << ",\"requested_activations\":" << profileJsonString(activations)
        << ",\"state_format\":" << profileJsonString(state) << ",\"kv_format\":" << profileJsonString(kv)
        << ",\"verify_rows\":" << profileJsonString(verify) << ",\"requested_max_context_tokens\":" << (contextExplicit ? std::to_string(maxContext) : "null")
        << ",\"fixture_context_bound_tokens\":" << maxContext
        << ",\"requested_memory_bytes\":" << (maxMemory ? std::to_string(maxMemory) : "null")
        << ",\"expert_memory_bytes\":" << load.expertMemoryBytes
        << ",\"ple_memory_bytes\":" << load.streamedTableBytes
        << ",\"generation_reserve_tokens\":" << generationReserve()
        << ",\"require_flash_next\":" << (requireFlashNext ? "true" : "false")
        << ",\"config_only\":" << (configOnly ? "true" : "false")
        << ",\"fusion_enabled\":false,\"http_ttft_s\":null,\"queue_wait_s\":null,\"ssd_cache_reuse\":null"
        << ",\"limitations\":[\"No HTTP, persistent KV cache, or production admission is exercised.\","
           "\"GPU replay changes command boundaries; attributed time is not serving latency.\","
           "\"Memory scope is resident-weight precheck and KV/state governor admission, not a full engine/host/cache budget.\"]}";
    return out.str();
  }
};
} // namespace splash::tools
