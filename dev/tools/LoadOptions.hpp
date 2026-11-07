#pragma once

// The options with which the native tools that load a model (the runtime
// oracle, weight-digests, decode-profile and teacher-logits) stream parts of
// its weights from their files, named as splash serve's, and what a load with
// them holds.

#include "model/ModelFactory.hpp"
#include "model/ModelSizing.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace splash::tools {

// A positive byte count with an optional K, M or G suffix, of 1024, 1024^2 or
// 1024^3; zero for anything else.
[[nodiscard]] inline uint64_t parseBytes(std::string_view value) {
  uint64_t unit = 1;
  if (!value.empty())
    if (const size_t power = std::string_view("KMG").find(value.back()); power != std::string_view::npos) {
      unit = uint64_t{1} << (10 * (power + 1));
      value.remove_suffix(1);
    }
  uint64_t count = 0;
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), count);
  if (error != std::errc{} || end != value.data() + value.size() || count > UINT64_MAX / unit)
    return 0;
  return count * unit;
}

// Takes `option` and its `value` into `options` when it is one of these, and
// returns false for any other option: --expert-memory SIZE streams a MoE
// target's routed experts through a store of SIZE bytes, and --ple-memory
// SIZE a Qwen3.8-Flash-Next target's PLE table, where its rows can stream,
// through a row cache of SIZE bytes (model::ModelLoadOptions); SIZE as
// parseBytes reads it, without the fraction (7.5G) and the spellings (7GB,
// 7g) splash serve also takes. Unlike splash serve, which loads a part whole
// where SIZE holds all of it, any SIZE streams. The expert store is SIZE
// from the load on, all of it its floor, as splash serve's is.
// Throws std::invalid_argument for a value that is not a size.
[[nodiscard]] inline bool takeLoadOption(std::string_view option, std::string_view value,
                                         model::ModelLoadOptions &options) {
  const bool experts = option == "--expert-memory";
  if (!experts && option != "--ple-memory") return false;
  const uint64_t bytes = parseBytes(value);
  if (!bytes) throw std::invalid_argument(std::string(option) + " takes a size such as " + (experts ? "16G" : "2G"));
  if (experts)
    options.expertMemoryBytes = options.expertFloorBytes = bytes;
  else
    options.streamedTableBytes = bytes;
  return true;
}

// What a load of the model in `root` with `options` holds, sized from its
// files before it loads (model::streamableWeights): the images that load,
// and the store of each part that streams, of the SIZE it was given, but no
// more than the part whole, which is all a store holds.
[[nodiscard]] inline uint64_t weightBytes(const std::filesystem::path &root, const model::ModelDescriptor &descriptor,
                                          const model::ModelLoadOptions &options) {
  const model::StreamableWeights streamable = model::streamableWeights(
      root, descriptor,
      {.streamExperts = options.expertMemoryBytes != 0, .streamTable = options.streamedTableBytes != 0});
  return streamable.loadedBytes + std::min(options.expertMemoryBytes, streamable.expertBytes) +
         std::min(options.streamedTableBytes, streamable.tableBytes);
}

} // namespace splash::tools
