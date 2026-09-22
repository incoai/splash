#pragma once

#include "FlashWeights.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::flash {

inline constexpr std::string_view kFlashOperandStoreSchema = "splash-local-affine-operands-v1";
inline constexpr uint64_t kFlashOperandStoreAlignment = 16384;
inline constexpr std::string_view kFlashOperandStoreMathVersion =
    "contract-off,f32-separated-mul-add,bf16-round-to-nearest-even,v1";

enum class FlashOperandFormat : uint8_t { BF16, F32 };

// Source geometry is part of the artifact contract, including otherwise unused
// expert strides. No stored matrix can be reused for a different raw view.
struct FlashOperandSpec final {
  std::string projection;
  FlashOperandFormat format = FlashOperandFormat::BF16;
  uint32_t experts = 1, outputSize = 0, inputSize = 0, bits = 0, groupSize = 0;
  uint64_t weightRowStrideBytes = 0, weightExpertStrideBytes = 0;
  uint64_t parameterRowStrideBytes = 0, parameterExpertStrideBytes = 0;
};

[[nodiscard]] FlashOperandSpec flashOperandSpec(std::string_view prefix,
    FlashOperandFormat format, const FlashAffineProjection &projection);

// Metadata inspection and source validation are CPU-only. Selected payloads
// are SHA256-verified before readonly zero-copy Shared Metal mapping. The same
// bytes replace an existing cache allocation; they are not charged twice.
class FlashOperandStore final {
public:
  [[nodiscard]] static FlashOperandStore load(const std::filesystem::path &directory,
      std::string_view sourceIdentity, std::string_view manifestFingerprint);
  [[nodiscard]] static std::unique_ptr<FlashOperandStore> fromEnvironment(const FlashWeights &weights);
  ~FlashOperandStore();
  FlashOperandStore(FlashOperandStore &&) noexcept;
  FlashOperandStore &operator=(FlashOperandStore &&) noexcept;
  FlashOperandStore(const FlashOperandStore &) = delete;
  FlashOperandStore &operator=(const FlashOperandStore &) = delete;

  void validateSource(const FlashWeights &weights) const;
  void verifyPayloads() const; // CPU-only; verifies every stored padded file.
  [[nodiscard]] bool contains(std::string_view prefix, FlashOperandFormat format) const noexcept;
  [[nodiscard]] FlashTensor mapTensor(metal::MetalBackend &backend, const FlashOperandSpec &expected) const;
  [[nodiscard]] const std::string &identitySha256() const;
  [[nodiscard]] uint64_t plannedBytes(std::span<const FlashOperandSpec> selection) const;
  [[nodiscard]] const std::vector<FlashOperandSpec> &specs() const;

private:
  FlashOperandStore();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Publishes to a fresh destination only, after payload/manifest fsync. Appends
// bounded writes from existing immutable Shared operands; no conversion occurs
// here. An abandoned or rejected writer removes only its own staging directory.
class FlashOperandStoreWriter final {
public:
  FlashOperandStoreWriter(const std::filesystem::path &destination,
      std::string_view sourceIdentity, std::string_view manifestFingerprint);
  ~FlashOperandStoreWriter();
  FlashOperandStoreWriter(const FlashOperandStoreWriter &) = delete;
  FlashOperandStoreWriter &operator=(const FlashOperandStoreWriter &) = delete;
  void append(const FlashOperandSpec &spec, std::span<const std::byte> operand);
  void append(const FlashOperandSpec &spec, const FlashTensor &operand);
  [[nodiscard]] std::string publish();
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::flash
