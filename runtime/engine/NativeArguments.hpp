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
  uint64_t maxMemoryBytes = 0;
  uint64_t maxCacheDiskBytes = 0;
  // --cache-dir: where the cache tier keeps its files for the next process;
  // empty for temporary ones.
  std::filesystem::path persistentCacheRoot;
  kv::Format kvFormat = kv::Format::Int8;
  double decodeShare = EngineConfig{}.decodeShare;
  uint32_t maxImagePatches = ops::kMaximumImagePatches;
  // --idle-release SECONDS|off, infinite for off.
  double idleReleaseSeconds = RuntimeResourcesConfig{}.idleReleaseSeconds;
  // --idle-sleep prevent|allow: whether the engine keeps the Mac from sleeping
  // automatically while it holds a request.
  bool preventIdleSleep = true;
};

// Reads `serve-native MODEL_DIRECTORY MAX_CONTEXT|auto MAX_MEMORY_BYTES|auto
// [MAX_CACHE_DISK_BYTES] [--name value]...` and inspects the model package
// the directory holds. Throws UsageError for a command line it refuses.
[[nodiscard]] NativeArguments parseNativeArguments(int argc,
                                                   const char *const *argv);

} // namespace splash::engine
