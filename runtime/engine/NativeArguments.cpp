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
bool parseInteger(std::string_view value, T &result, bool allowZero = false) {
  const char *end = value.data() + value.size();
  auto parsed = std::from_chars(value.data(), end, result);
  return parsed.ec == std::errc{} && parsed.ptr == end && (allowZero || result != 0);
}

template <typename T>
T parseAutoPositive(std::string_view value, const char *error) {
  if (value == "auto") return 0;
  T result = 0;
  if (!parseInteger(value, result)) throw UsageError(error);
  return result;
}

bool parseChoice(std::string_view option, std::string_view value, std::string_view yes, std::string_view no) {
  if (value == yes) return true;
  if (value == no) return false;
  throw UsageError(std::string(option) + " requires " + std::string(yes) + " or " + std::string(no));
}

uint32_t parseMaxImagePatches(std::string_view value) {
  uint32_t result = 0;
  if (!parseInteger(value, result) || result % 4 ||
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

// BYTES|auto|off, as the server writes the memory of a store of streamed
// weights (--expert-memory, --ple-memory).
StreamedSetting parseStreamedSetting(std::string_view option, std::string_view value) {
  if (value == "auto")
    return {};
  if (value == "off")
    return {StreamedSetting::Mode::Off, 0};
  StreamedSetting result{StreamedSetting::Mode::Size, 0};
  if (!parseInteger(value, result.bytes))
    throw UsageError(std::string(option) + " requires a positive byte count, auto or off");
  return result;
}

std::filesystem::path requireModelRoot(std::string_view argument) {
  std::error_code error;
  const std::filesystem::path root =
      std::filesystem::canonical(std::filesystem::path(argument), error);
  if (error || !std::filesystem::is_directory(root, error))
    throw ModelRootError("MODEL_DIRECTORY must name an existing directory");
  for (const char *role : {"target", "draft"}) {
    if (!std::filesystem::is_directory(root / role, error))
      throw ModelRootError(
          "MODEL_DIRECTORY must hold the model's target/ and draft/ directories");
  }
  return root;
}

// The model `root` holds (model::inspectModelRoot), whose refusal is the
// root's.
model::ModelDescriptor inspectModel(const std::filesystem::path &root) {
  try {
    return model::inspectModelRoot(root);
  } catch (const std::exception &error) {
    throw ModelRootError(error.what());
  }
}

} // namespace

NativeArguments parseNativeArguments(int argc, const char *const *argv) {
  if (argc < 5)
    throw UsageError("serve-native requires MODEL_DIRECTORY, MAX_CONTEXT and MAX_MEMORY_BYTES");
  NativeArguments result;
  // Read once the model's own context is known.
  std::optional<std::string_view> relaunchContext;
  int next = 5;
  if (next < argc && !std::string_view(argv[next]).starts_with("--")) {
    const std::string_view quota(argv[next++]);
    result.maxCacheDisk = {};
    if (quota != "0" && !parseInteger(quota, result.maxCacheDisk.bytes))
      throw UsageError("MAX_CACHE_DISK_BYTES must be a nonnegative integer");
  }
  // Options follow as --name value pairs; a missing value fails its check.
  for (; next < argc; next += 2) {
    const std::string_view option(argv[next]);
    const std::string_view value(next + 1 < argc ? argv[next + 1] : "");
    if (option == "--kv-format") {
      result.kvFormat = parseChoice(option, value, "int8", "bf16") ? kv::Format::Int8 : kv::Format::BFloat16;
    } else if (option == "--state-format") {
      result.stateFormat = parseChoice(option, value, "fp32", "bf16") ? ops::GdnStateFormat::Float32 : ops::GdnStateFormat::BFloat16;
    } else if (option == "--cached-states") {
      result.exactCachedStates = !parseChoice(option, value, "int8", "exact");
    } else if (option == "--foreground-turn-tokens") {
      if (!parseInteger(value, result.foregroundTurnTokens, true) || result.foregroundTurnTokens > 4096)
        throw UsageError("--foreground-turn-tokens requires an integer in [0, 4096]");
    } else if (option == "--decode-share") {
      result.decodeShare = parseDecodeShare(value);
    } else if (option == "--max-image-patches") {
      result.maxImagePatches = parseMaxImagePatches(value);
    } else if (option == "--persistent-cache") {
      result.persistentCache = parseChoice(option, value, "on", "off");
    } else if (option == "--max-cache-disk-writes") {
      if (value != "0" && !parseInteger(value, result.maxCacheDiskWritesBytes))
        throw UsageError("--max-cache-disk-writes requires a nonnegative integer byte count");
    } else if (option == "--cache-dir") {
      if (value.empty())
        throw UsageError("--cache-dir requires a directory");
      result.cacheRoot = std::filesystem::absolute(std::filesystem::path(value));
    } else if (option == "--idle-release") {
      result.idleReleaseSeconds = parseIdleRelease(value);
    } else if (option == "--idle-sleep") {
      result.preventIdleSleep = parseChoice(option, value, "prevent", "allow");
    } else if (option == "--ple-memory") {
      result.pleMemory = parseStreamedSetting(option, value);
    } else if (option == "--expert-memory") {
      result.expertMemory = parseStreamedSetting(option, value);
    } else if (option == "--sparse-kv-memory") {
      uint64_t bytes = 0;
      if (value != "auto" && !parseInteger(value, bytes))
        throw UsageError("--sparse-kv-memory requires a positive byte count or auto");
      result.sparseKvMemory = bytes ? std::optional<uint64_t>(bytes) : std::nullopt;
    } else if (option == "--kv-spill") {
      if (value != "always")
        throw UsageError("--kv-spill requires always");
      result.kvSpillAlways = true;
    } else if (option == "--kv-window-trim") {
      result.kvWindowTrim = parseChoice(option, value, "on", "off");
    } else if (option == "--ane") {
      result.neuralEngine = parseChoice(option, value, "on", "off");
    } else if (option == "--activations") {
      result.int8Activations = parseChoice(option, value, "auto", "bf16") ? ops::Int8Activations::On : ops::Int8Activations::Off;
    } else if (option == "--verify-rows") {
      result.verifyRows = parseChoice(option, value, "auto", "all") ? model::VerifyRows::Auto : model::VerifyRows::All;
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
  model::ModelDescriptor inspected = inspectModel(result.modelRoot);
  result.maxContext = parseAutoPositive<uint32_t>(argv[3], "MAX_CONTEXT must be auto or a positive integer");
  try {
    result.model = model::servedFor(std::move(inspected), result.maxContext);
  } catch (const std::invalid_argument &error) {
    throw UsageError(std::string("MAX_CONTEXT: ") + error.what());
  }
  // A relaunch's context is one its first engine served: MAX_CONTEXT at
  // most, which a relaunch passes as that engine's, else one the model holds.
  if (relaunchContext) {
    const uint32_t most =
        result.maxContext ? result.maxContext : result.model.capabilities.maximumContextTokens;
    if ((!parseInteger(*relaunchContext, result.relaunchContext) || result.relaunchContext > most))
      throw UsageError("--relaunch-context requires an integer in [1, " + std::to_string(most) + "]");
  }
  result.maxMemoryBytes = parseAutoPositive<uint64_t>(argv[4], "MAX_MEMORY_BYTES must be auto or a positive integer");
  return result;
}

} // namespace splash::engine
