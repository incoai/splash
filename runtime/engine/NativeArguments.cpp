#include "engine/NativeArguments.hpp"

#include <charconv>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace splash::engine {
namespace {

template <typename T>
bool parsePositive(std::string_view value, T &result) {
  const char *end = value.data() + value.size();
  auto parsed = std::from_chars(value.data(), end, result);
  return parsed.ec == std::errc{} && parsed.ptr == end && result != 0;
}

uint64_t parseMaxMemory(std::string_view value) {
  if (value == "auto")
    return 0;
  uint64_t result = 0;
  if (!parsePositive(value, result))
    throw UsageError("MAX_MEMORY_BYTES must be auto or a positive integer");
  return result;
}

// Whether `value` is a context of 1 to the model's own tokens, which
// `result` then holds.
bool parseContext(std::string_view value,
                  const model::ModelCapabilities &capabilities, uint32_t &result) {
  return parsePositive(value, result) &&
         result <= capabilities.maximumContextTokens;
}

uint32_t parseMaxContext(std::string_view value,
                         const model::ModelCapabilities &capabilities) {
  if (value == "auto")
    return 0;
  uint32_t result = 0;
  if (!parseContext(value, capabilities, result)) {
    throw UsageError("MAX_CONTEXT must be auto or an integer in [1, " +
                     std::to_string(capabilities.maximumContextTokens) + "]");
  }
  return result;
}

uint32_t parseMaxImagePatches(std::string_view value) {
  uint32_t result = 0;
  if (!parsePositive(value, result) || result % 4 ||
      result > ops::kMaximumImagePatches)
    throw UsageError("--max-image-patches requires a positive multiple of 4 "
                     "up to " + std::to_string(ops::kMaximumImagePatches));
  return result;
}

bool parseFinite(std::string_view value, double &result) {
  const char *end = value.data() + value.size();
  auto parsed = std::from_chars(value.data(), end, result);
  return parsed.ec == std::errc{} && parsed.ptr == end && std::isfinite(result);
}

double parseIdleRelease(std::string_view value) {
  if (value == "off")
    return std::numeric_limits<double>::infinity();
  double result = 0.0;
  if (!parseFinite(value, result) || result <= 0.0)
    throw UsageError("--idle-release requires off or a positive number of seconds");
  return result;
}

double parseDecodeShare(std::string_view value) {
  double result = 0.0;
  if (!parseFinite(value, result) || result < 0.0)
    throw UsageError("--decode-share requires a nonnegative number");
  return result;
}

std::filesystem::path requireModelRoot(std::string_view argument) {
  std::error_code error;
  const std::filesystem::path root =
      std::filesystem::canonical(std::filesystem::path(argument), error);
  if (error || !std::filesystem::is_directory(root, error))
    throw UsageError("MODEL_DIRECTORY must name an existing directory");
  for (const char *role : {"target", "draft"}) {
    if (!std::filesystem::is_directory(root / role, error))
      throw UsageError(
          "MODEL_DIRECTORY must hold the model's target/ and draft/ directories");
  }
  return root;
}

} // namespace

NativeArguments parseNativeArguments(int argc, const char *const *argv) {
  if (argc < 5 || std::string_view(argv[1]) != "serve-native") {
    throw UsageError("expected the serve-native command");
  }
  NativeArguments result;
  // Read once the model's own context is known.
  std::optional<std::string_view> relaunchContext;
  int next = 5;
  if (next < argc && !std::string_view(argv[next]).starts_with("--")) {
    const std::string_view quota(argv[next++]);
    result.maxCacheDisk = {};
    if (quota != "0" && !parsePositive(quota, result.maxCacheDisk.bytes))
      throw UsageError("MAX_CACHE_DISK_BYTES must be a nonnegative integer");
  }
  // Options follow as --name value pairs; a missing value fails its check.
  for (; next < argc; next += 2) {
    const std::string_view option(argv[next]);
    const std::string_view value(next + 1 < argc ? argv[next + 1] : "");
    if (option == "--kv-format") {
      if (value != "int8" && value != "bf16")
        throw UsageError("--kv-format requires int8 or bf16");
      result.kvFormat = value == "int8" ? kv::Format::Int8 : kv::Format::BFloat16;
    } else if (option == "--state-format") {
      if (value != "fp32" && value != "bf16")
        throw UsageError("--state-format requires fp32 or bf16");
      result.stateFormat =
          value == "fp32" ? ops::GdnStateFormat::Float32 : ops::GdnStateFormat::BFloat16;
    } else if (option == "--decode-share") {
      result.decodeShare = parseDecodeShare(value);
    } else if (option == "--max-image-patches") {
      result.maxImagePatches = parseMaxImagePatches(value);
    } else if (option == "--persistent-cache") {
      if (value != "on" && value != "off")
        throw UsageError("--persistent-cache requires on or off");
      result.persistentCache = value == "on";
    } else if (option == "--max-cache-disk-writes") {
      if (value != "0" && !parsePositive(value, result.maxCacheDiskWritesBytes))
        throw UsageError("--max-cache-disk-writes requires a nonnegative integer byte count");
    } else if (option == "--cache-dir") {
      if (value.empty())
        throw UsageError("--cache-dir requires a directory");
      result.cacheRoot = std::filesystem::absolute(std::filesystem::path(value));
    } else if (option == "--idle-release") {
      result.idleReleaseSeconds = parseIdleRelease(value);
    } else if (option == "--idle-sleep") {
      if (value != "prevent" && value != "allow")
        throw UsageError("--idle-sleep requires prevent or allow");
      result.preventIdleSleep = value == "prevent";
    } else if (option == "--ple-memory") {
      if (!parsePositive(value, result.streamedTableBytes))
        throw UsageError("--ple-memory requires a positive byte count");
    } else if (option == "--expert-memory") {
      if (!parsePositive(value, result.expertMemoryBytes))
        throw UsageError("--expert-memory requires a positive byte count");
    } else if (option == "--ane") {
      if (value != "on" && value != "off")
        throw UsageError("--ane requires on or off");
      result.neuralEngine = value == "on";
    } else if (option == "--relaunch-context") {
      relaunchContext = value;
    } else {
      throw UsageError("unexpected argument " + std::string(option));
    }
  }
  if (result.persistentCache && !result.maxCacheDisk.automatic && !result.maxCacheDisk.bytes)
    throw UsageError("--persistent-cache requires an SSD cache, which MAX_CACHE_DISK_BYTES 0 turns off");
  if (result.maxCacheDiskWritesBytes && !result.maxCacheDisk.automatic && !result.maxCacheDisk.bytes)
    throw UsageError("--max-cache-disk-writes requires an SSD cache, which MAX_CACHE_DISK_BYTES 0 turns off");
  if (!result.cacheRoot.empty() && !result.persistentCache)
    throw UsageError("--cache-dir requires --persistent-cache on");
  result.modelRoot = requireModelRoot(argv[2]);
  result.model = model::inspectModelRoot(result.modelRoot);
  result.maxContext = parseMaxContext(argv[3], result.model.capabilities);
  if (relaunchContext) {
    if (result.maxContext)
      throw UsageError("--relaunch-context requires MAX_CONTEXT auto");
    if (!parseContext(*relaunchContext, result.model.capabilities, result.relaunchContext))
      throw UsageError("--relaunch-context requires an integer in [1, " +
                       std::to_string(result.model.capabilities.maximumContextTokens) + "]");
  }
  result.maxMemoryBytes = parseMaxMemory(argv[4]);
  return result;
}

} // namespace splash::engine
