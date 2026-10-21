#pragma once

#include "engine/Engine.hpp"
#include "engine/RuntimeResources.hpp"
#include "model/ModelDescriptor.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
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
  // The model the directory holds, served for MAX_CONTEXT (model::servedFor):
  // with YaRN for a context past its native window.
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
  // --ple-memory BYTES|auto|off and --expert-memory BYTES|auto|off: a
  // Qwen3.8-Flash-Next GGUF's PLE table and a MoE target's routed experts,
  // through a store of BYTES that streams the rest from the SSD, as the
  // memory planner decides (auto, without the option) or loaded whole (off)
  // (RuntimeResourcesConfig::pleMemory, expertMemory).
  StreamedSetting pleMemory;
  StreamedSetting expertMemory;
  // --sparse-kv-memory BYTES|auto: the target of a block-sparse target's hot
  // block cache; none, auto, its floor (RuntimeResourcesConfig::sparseKvMemory).
  std::optional<uint64_t> sparseKvMemory;
  // --kv-spill always: every request of a target that can spill keeps its
  // KV on the SSD cache, for validation; without it, the requests memory
  // cannot hold (EngineSpillConfig::always).
  bool kvSpillAlways = false;
  // --verify-rows auto|all: the rows of the draft's block a decode
  // cycle verifies (RuntimeResourcesConfig::verifyRows).
  model::VerifyRows verifyRows = model::VerifyRows::Auto;
};

// Reads `serve-native MODEL_DIRECTORY MAX_CONTEXT|auto MAX_MEMORY_BYTES|auto
// [MAX_CACHE_DISK_BYTES] [--name value]...` and inspects the model the
// directory, its model root, holds. Throws UsageError for a command line it
// refuses.
[[nodiscard]] NativeArguments parseNativeArguments(int argc,
                                                   const char *const *argv);

} // namespace splash::engine
