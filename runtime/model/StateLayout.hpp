#pragma once

#include <cstdint>

namespace splash::model {

// The GDN short convolution has four taps; the recurrent state keeps the
// three previous inputs.
inline constexpr uint32_t kGdnConvolutionTaps = 4;

// Physical state geometry is supplied by the paired target and draft models.
// The engine sees only opaque CompositeState handles and byte accounting.
struct GdnStateLayout final {
  static constexpr uint32_t alignmentBytes = 16 * 1024;
  static constexpr uint32_t bfloat16Bytes = 2;

  uint32_t layers = 0;
  uint32_t convolutionHistory = 0;
  uint32_t convolutionChannels = 0;
  uint32_t recurrentGroups = 0;
  uint32_t recurrentRows = 0;
  uint32_t recurrentColumns = 0;
  // State a family keeps beside its GDN layers' (Qwen3.8-Flash-Next's PLE
  // convolution history), after them in the cell.
  uint64_t auxiliaryBytes = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return layers && convolutionHistory && convolutionChannels &&
           recurrentGroups && recurrentRows && recurrentColumns;
  }
  [[nodiscard]] static constexpr uint64_t align(uint64_t bytes) noexcept {
    return (bytes + alignmentBytes - 1) & ~uint64_t(alignmentBytes - 1);
  }
  [[nodiscard]] constexpr uint64_t convolutionLayerBytes() const noexcept {
    return align(uint64_t{convolutionHistory} * convolutionChannels *
                 bfloat16Bytes);
  }
  [[nodiscard]] constexpr uint64_t convolutionBytes() const noexcept {
    return uint64_t{layers} * convolutionLayerBytes();
  }
  [[nodiscard]] constexpr uint64_t recurrentLayerBytes() const noexcept {
    return align(uint64_t{recurrentGroups} * recurrentRows *
                 recurrentColumns * sizeof(float));
  }
  [[nodiscard]] constexpr uint64_t recurrentBytes() const noexcept {
    return uint64_t{layers} * recurrentLayerBytes();
  }
  [[nodiscard]] constexpr uint64_t cellBytes() const noexcept {
    return convolutionBytes() + recurrentBytes() + align(auxiliaryBytes);
  }

  bool operator==(const GdnStateLayout &) const = default;
};

struct DraftStateLayout final {
  static constexpr uint32_t bfloat16Bytes = 2;

  uint32_t layers = 0;
  uint32_t kvHeads = 0;
  uint32_t tokens = 0;
  uint32_t headDimension = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return layers && kvHeads && tokens && headDimension;
  }
  [[nodiscard]] constexpr uint64_t tensorBytes() const noexcept {
    return uint64_t{kvHeads} * tokens * headDimension * bfloat16Bytes;
  }
  [[nodiscard]] constexpr uint64_t ringBytes() const noexcept {
    return uint64_t{layers} * 2 * tensorBytes();
  }

  bool operator==(const DraftStateLayout &) const = default;
};

struct CompositeStateLayout final {
  GdnStateLayout target;
  DraftStateLayout draft;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return target.valid() && draft.valid();
  }
  [[nodiscard]] constexpr uint64_t activeCellBytes() const noexcept {
    return 2 * target.cellBytes() + draft.ringBytes();
  }
  [[nodiscard]] constexpr uint64_t cachedBytes() const noexcept {
    return target.cellBytes() + draft.ringBytes();
  }

  bool operator==(const CompositeStateLayout &) const = default;
};

} // namespace splash::model
