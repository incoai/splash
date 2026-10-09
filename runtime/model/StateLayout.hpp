#pragma once

#include "Checked.hpp"
#include "metal/abi/DraftAttention.h"
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
// one for the values of every draft layer, and the context window the rings
// are computed from: the draft's context row (contextWidth values) of each
// slot, in 4-bit codes (metal/abi/DraftAttention.h).
struct DraftStateLayout final {
  static constexpr uint32_t bfloat16Bytes = 2;

  uint32_t layers = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;
  uint32_t contextWidth = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return layers && kvHeads && headDimension && contextWidth &&
           contextWidth % SPLASH_DRAFT_CONTEXT_GROUP == 0;
  }
  [[nodiscard]] constexpr uint64_t tensorBytes() const noexcept {
    return uint64_t{kvHeads} * SPLASH_DRAFT_SLIDING_WINDOW * headDimension *
           bfloat16Bytes;
  }
  [[nodiscard]] constexpr uint64_t ringBytes() const noexcept {
    return uint64_t{layers} * 2 * tensorBytes();
  }
  [[nodiscard]] constexpr uint64_t windowBytes() const noexcept {
    return draft_context_window_bytes(contextWidth);
  }

  bool operator==(const DraftStateLayout &) const = default;
};

// A lane holds two GDN cells, its draft rings and their context window; a
// cached state one GDN cell and the window, from which a restore that needs
// the rings computes them again.
struct CompositeStateLayout final {
  static constexpr uint32_t kLaneGdnCells = 2;

  GdnStateLayout target;
  DraftStateLayout draft;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return target.valid() && draft.valid();
  }
  [[nodiscard]] constexpr uint64_t laneBytes() const noexcept {
    return kLaneGdnCells * target.cellBytes() + draft.ringBytes() +
           draft.windowBytes();
  }
  [[nodiscard]] constexpr uint64_t cachedBytes() const noexcept {
    return target.cellBytes() + draft.windowBytes();
  }

  bool operator==(const CompositeStateLayout &) const = default;
};

} // namespace splash::model
