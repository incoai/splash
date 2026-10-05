#pragma once

#include "Checked.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"

#include <cstdint>

namespace splash::model {

// The taps of the GDN short convolution the kernels compute; a state cell
// keeps the inputs of all taps but the current token's.
inline constexpr uint32_t kGdnConvolutionTaps = SPLASH_GDN_CONVOLUTION_TAPS;

// Physical state geometry is supplied by the paired target and draft models.
// The engine sees only opaque CompositeState handles and byte accounting.
struct GdnStateLayout final {
  static constexpr uint32_t bfloat16Bytes = 2;

  uint32_t layers = 0;
  uint32_t convolutionHistory = 0;
  uint32_t convolutionChannels = 0;
  uint32_t recurrentGroups = 0;
  uint32_t recurrentRows = 0;
  uint32_t recurrentColumns = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return layers && convolutionHistory && convolutionChannels &&
           recurrentGroups && recurrentRows && recurrentColumns;
  }
  [[nodiscard]] constexpr uint64_t convolutionLayerBytes() const noexcept {
    return alignUp(uint64_t{convolutionHistory} * convolutionChannels *
                   bfloat16Bytes);
  }
  [[nodiscard]] constexpr uint64_t convolutionBytes() const noexcept {
    return uint64_t{layers} * convolutionLayerBytes();
  }
  [[nodiscard]] constexpr uint64_t recurrentLayerBytes() const noexcept {
    return alignUp(uint64_t{recurrentGroups} * recurrentRows *
                   recurrentColumns * sizeof(float));
  }
  [[nodiscard]] constexpr uint64_t recurrentBytes() const noexcept {
    return uint64_t{layers} * recurrentLayerBytes();
  }
  [[nodiscard]] constexpr uint64_t cellBytes() const noexcept {
    return convolutionBytes() + recurrentBytes();
  }

  bool operator==(const GdnStateLayout &) const = default;
};

// One ring of SPLASH_DRAFT_SLIDING_WINDOW slots per KV head for the keys and
// one for the values of every draft layer.
struct DraftStateLayout final {
  static constexpr uint32_t bfloat16Bytes = 2;

  uint32_t layers = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return layers && kvHeads && headDimension;
  }
  [[nodiscard]] constexpr uint64_t tensorBytes() const noexcept {
    return uint64_t{kvHeads} * SPLASH_DRAFT_SLIDING_WINDOW * headDimension *
           bfloat16Bytes;
  }
  [[nodiscard]] constexpr uint64_t ringBytes() const noexcept {
    return uint64_t{layers} * 2 * tensorBytes();
  }

  bool operator==(const DraftStateLayout &) const = default;
};

struct CompositeStateLayout final {
  // The GDN cells a lane holds, with one draft ring.
  static constexpr uint32_t kLaneGdnCells = 2;

  GdnStateLayout target;
  DraftStateLayout draft;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return target.valid() && draft.valid();
  }
  [[nodiscard]] constexpr uint64_t laneBytes() const noexcept {
    return kLaneGdnCells * target.cellBytes() + draft.ringBytes();
  }
  [[nodiscard]] constexpr uint64_t cachedBytes() const noexcept {
    return target.cellBytes() + draft.ringBytes();
  }

  bool operator==(const CompositeStateLayout &) const = default;
};

} // namespace splash::model
