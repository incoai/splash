#pragma once

// A control vector on a target's residual, from a GGUF as llama.cpp's
// --control-vector reads it: general.architecture "controlvector", and an F32
// tensor direction.L of the hidden size for each layer L it steers, which
// changes layer L's output (from 1: no file holds layer 0's). After each
// steered layer the residual takes scale times the direction (add, llama.cpp's
// own), or loses scale times its component along the direction (project).
// Qwen3.8-Flash-Next applies it to every hyper-connection stream after the
// layer's FFN (model/Qwen4ExpTarget.cpp, ops::HyperConnection::addSteer); the
// Qwen3.5 families to the bf16 residual itself
// (model/QwenTarget.cpp, ops::addResidualSteer). It is fixed for the process:
// every request runs on it.

#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {

enum class ControlVectorMode { Add, Project };

[[nodiscard]] std::string_view controlVectorModeName(ControlVectorMode mode) noexcept;

// What --control-vector and its options ask for; no file for none.
struct ControlVectorOptions final {
  std::filesystem::path path;
  float scale = 1.0F;
  // The layers it may steer, both included (--control-vector-layer-range):
  // every layer the file holds by default.
  uint32_t firstLayer = 0;
  uint32_t lastLayer = std::numeric_limits<uint32_t>::max();
  ControlVectorMode mode = ControlVectorMode::Add;

  [[nodiscard]] explicit operator bool() const noexcept { return !path.empty(); }
  bool operator==(const ControlVectorOptions &) const = default;
};

// A control vector as a target of `layers` layers of `hidden` values runs it:
// fp32 [layers][hidden] directions, a projection's of unit length and zero
// for a layer it leaves alone; the layers it steers, ascending: in the range,
// with a nonzero direction; and the SHA-256 of what it does (its mode, scale,
// layers and their directions), which names the caches it computed.
struct ControlVectorTable final {
  ControlVectorMode mode = ControlVectorMode::Add;
  float scale = 1.0F;
  uint32_t hidden = 0;
  std::vector<float> directions;
  std::vector<uint32_t> layers;
  std::string sha256;
};

// The table of the file `options` names for a target of `architecture`, the
// GGUF architecture a file's controlvector.model_hint must name when it names
// one. Throws for a file that is not a control vector, a tensor other than an
// F32 direction.L of `hidden` values for a layer in [1, layers), a value that
// is not finite, a range outside the target's layers, or one that steers no
// layer.
[[nodiscard]] ControlVectorTable readControlVector(const ControlVectorOptions &options,
                                                   std::string_view architecture, uint32_t layers,
                                                   uint32_t hidden);

// A loaded control vector: its table's directions in a Metal buffer.
struct ControlVector final {
  ControlVectorMode mode = ControlVectorMode::Add;
  float scale = 1.0F;
  metal::MetalBuffer directions;
  std::vector<uint32_t> layers;
  std::string sha256;

  [[nodiscard]] bool steers(uint32_t layer) const noexcept;
  // "project, layers 4..44 (41 steered), scale 1", as the engine logs it.
  [[nodiscard]] std::string describe() const;
};

[[nodiscard]] ControlVector loadControlVector(metal::MetalBackend &backend, const ControlVectorTable &table);

} // namespace splash::model
