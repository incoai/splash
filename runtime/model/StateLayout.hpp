#pragma once

#include "Checked.hpp"
#include "metal/abi/DraftAttention.h"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"
#include "ops/GDN.hpp"
#include "ops/StateCodec.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <array>
#include <cstdint>

namespace splash::model {

// The taps of the GDN short convolution the kernels compute; a state cell
// keeps the inputs of all taps but the current token's.
inline constexpr uint32_t kGdnConvolutionTaps = SPLASH_GDN_CONVOLUTION_TAPS;

// Physical state geometry is supplied by the paired target and draft models,
// and the running recurrent state's format by the runtime (--state-format).
// A cached state's cell holds the recurrent layers as int8 codes and bf16
// scales instead (ops::StateCodec, cachedCellBytes), or with
// --cached-states exact as the running state does. The engine sees only
// opaque CompositeState handles and byte accounting. A target without GDN
// layers keeps only its auxiliary state in a cell, if it has one: its cells
// may hold nothing.
struct GdnStateLayout final {
  static constexpr uint32_t bfloat16Bytes = 2;

  uint32_t layers = 0;
  uint32_t convolutionHistory = 0;
  uint32_t convolutionChannels = 0;
  uint32_t recurrentGroups = 0;
  uint32_t recurrentRows = 0;
  uint32_t recurrentColumns = 0;
  ops::GdnStateFormat format = ops::GdnStateFormat::Float32;
  // State a family keeps beside its GDN layers' (Qwen3.8-Flash-Next's PLE
  // convolution history), after them in the cell.
  uint64_t auxiliaryBytes = 0;
  // Whether a cached state holds the recurrent layers as codes; false keeps
  // them in `format` (--cached-states exact).
  bool cachedCodes = true;

  // Every GDN dimension is set with GDN layers and zero without.
  [[nodiscard]] constexpr bool valid() const noexcept {
    const std::array dimensions{convolutionHistory, convolutionChannels,
                                recurrentGroups, recurrentRows,
                                recurrentColumns};
    const auto set = [&](uint32_t dimension) { return !dimension == !layers; };
    return std::ranges::all_of(dimensions, set) && ops::validFormat(format) &&
           recurrentColumns % SPLASH_STATE_CODEC_BLOCK == 0;
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
                   recurrentColumns * ops::elementBytes(format));
  }
  [[nodiscard]] constexpr uint64_t recurrentBytes() const noexcept {
    return uint64_t{layers} * recurrentLayerBytes();
  }
  [[nodiscard]] constexpr uint64_t cellBytes() const noexcept {
    return convolutionBytes() + recurrentBytes() + alignUp(auxiliaryBytes);
  }
  // A cached state's recurrent layer: the codes and scales of every value
  // head's rows (ops::StateCodec), or a lane's layer.
  [[nodiscard]] constexpr uint64_t cachedRecurrentLayerBytes() const noexcept {
    return cachedCodes ? alignUp(state_codec_layer_bytes(recurrentGroups * recurrentRows, recurrentColumns))
                       : recurrentLayerBytes();
  }
  [[nodiscard]] constexpr uint64_t cachedRecurrentBytes() const noexcept {
    return uint64_t{layers} * cachedRecurrentLayerBytes();
  }
  // A cached state's cell: a lane's, but for its recurrent layers' codes.
  [[nodiscard]] constexpr uint64_t cachedCellBytes() const noexcept {
    return convolutionBytes() + cachedRecurrentBytes() + alignUp(auxiliaryBytes);
  }
  // The codec, or the copy, between a lane's recurrent layers and a cached
  // state's.
  [[nodiscard]] constexpr ops::StateCodecShape codecShape() const noexcept {
    return {layers,
            recurrentGroups * recurrentRows,
            recurrentColumns,
            recurrentLayerBytes(),
            cachedRecurrentLayerBytes(),
            format};
  }
  // A cell holds every layer's convolution state, then every layer's
  // recurrent state, then the auxiliary state.
  [[nodiscard]] constexpr ops::GdnStateStrides strides() const noexcept {
    return {convolutionLayerBytes(), recurrentLayerBytes(), convolutionBytes(),
            format};
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
  // The ring slots the draft's geometry keeps (the compiled draft variant's
  // window, the shipped drafts' unless another's config states one).
  uint32_t window = SPLASH_DRAFT_SLIDING_WINDOW;
  // The rows a draft cycle drafts and the target verifies (the compiled
  // draft variant's block; the ABI's 8 unless the variant states more).
  uint32_t rows = SPLASH_DRAFT_QUERY_ROWS;
  uint32_t slidingWindow = 0;
  kv::LayerMask fullAttentionLayers{};
  kv::LayerMask unrotatedLayers{};

  [[nodiscard]] constexpr uint32_t layerWindow(uint32_t layer) const noexcept {
    return slidingWindow && !fullAttentionLayers.test(layer) ? slidingWindow : window;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return layers && kvHeads && headDimension && contextWidth && window &&
           contextWidth % SPLASH_DRAFT_CONTEXT_GROUP == 0;
  }
  [[nodiscard]] constexpr uint64_t tensorBytes(uint32_t layer = 0) const noexcept {
    return uint64_t{kvHeads} * layerWindow(layer) * headDimension * bfloat16Bytes;
  }
  [[nodiscard]] constexpr uint64_t ringBytes() const noexcept {
    uint64_t bytes = 0;
    for (uint32_t layer = 0; layer < layers; ++layer) bytes += 2 * tensorBytes(layer);
    return bytes;
  }
  [[nodiscard]] constexpr uint64_t windowBytes() const noexcept {
    return draft_context_window_bytes(contextWidth, window);
  }

  bool operator==(const DraftStateLayout &) const = default;
};

// A lane holds two GDN cells, its draft rings and their context window; a
// cached state one cached cell (GdnStateLayout::cachedCellBytes) and the
// window, from which a restore that needs the rings computes them again. A
// cell of no bytes takes no memory (QwenGdnCell).
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
    return target.cachedCellBytes() + draft.windowBytes();
  }

  bool operator==(const CompositeStateLayout &) const = default;
};

} // namespace splash::model
