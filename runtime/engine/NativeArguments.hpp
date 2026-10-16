#pragma once

#include "engine/Engine.hpp"
#include "engine/RuntimeResources.hpp"
#include "model/ModelDescriptor.hpp"

#include <cstdint>
#include <filesystem>
#include <stdexcept>

namespace splash::engine {

// A command line the native process cannot run.
class UsageError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// What the serve-native command line asks for. server/server.py
// `_native_command` writes it, and dev/tests/engine/native_command_golden.txt
// pins the lines both sides agree on.
struct NativeArguments final {
  std::filesystem::path modelRoot;
  model::ModelDescriptor model;
  // Zero for auto.
  uint32_t maxContext = 0;
  // --relaunch-context TOKENS, with MAX_CONTEXT auto: the context the
  // server's first engine served, which the server passes to every relaunch
  // (RuntimeResourcesConfig::relaunchContext); zero for none.
  uint32_t relaunchContext = 0;
  uint64_t maxMemoryBytes = 0;
  // Automatic without MAX_CACHE_DISK_BYTES, as the server's default.
  CacheDiskQuota maxCacheDisk{.automatic = true};
  // --max-cache-disk-writes BYTES: the SSD cache's write budget, the most it
  // writes a day on average; zero for none.
  uint64_t maxCacheDiskWritesBytes = 0;
  // --persistent-cache on|off: whether the cache tier keeps its files for the
  // next process, instead of temporary ones.
  bool persistentCache = false;
  // --cache-dir: where persistent caches live; empty for the default,
  // ~/Library/Caches/Splash/prefix-cache.
  std::filesystem::path cacheRoot;
  kv::Format kvFormat = kv::Format::Int8;
  ops::GdnStateFormat stateFormat = ops::GdnStateFormat::Float32;
  double decodeShare = EngineConfig{}.decodeShare;
  uint32_t maxImagePatches = ops::kMaximumImagePatches;
  // --idle-release SECONDS|off, infinite for off.
  double idleReleaseSeconds = RuntimeResourcesConfig{}.idleReleaseSeconds;
  // --idle-sleep prevent|allow: whether the engine keeps the Mac from sleeping
  // automatically while it holds a request.
  bool preventIdleSleep = true;
  // --ane on|off: whether a dense target's prefill FFN may split with the
  // Neural Engine when that is faster (splash serve --disable-ane turns it
  // off).
  bool neuralEngine = true;
  // --ple-ssd-cache BYTES: stream a Qwen3.8-Flash-Next GGUF's PLE table from
  // its file through a host row cache of BYTES in all (at least
  // kMinimumPleCacheBytes) instead of loading it; 0, without the option,
  // loads it (RuntimeResourcesConfig::pleCacheBytes).
  uint64_t pleCacheBytes = 0;
};

inline constexpr uint64_t kMinimumPleCacheBytes = 64ULL << 20;

// Reads `serve-native MODEL_DIRECTORY MAX_CONTEXT|auto MAX_MEMORY_BYTES|auto
// [MAX_CACHE_DISK_BYTES] [--name value]...` and inspects the model the
// directory, its model root, holds. Throws UsageError for a command line it
// refuses.
[[nodiscard]] NativeArguments parseNativeArguments(int argc,
                                                   const char *const *argv);

} // namespace splash::engine
