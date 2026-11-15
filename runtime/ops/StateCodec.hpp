#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/StateCodec.h"
#include "ops/GDN.hpp"

#include <cstdint>

namespace splash::ops {

// GDN recurrent layers as a running state holds them (GdnStateFormat) and as
// a cached state stores them: per layer, every value's int8 code, row by row,
// then each block's bf16 scale (metal/abi/StateCodec.h). A layer holds `rows`
// value rows (every value head's) of `columns` values (a head's key
// dimensions); consecutive layers are stateLayerBytes apart in the running
// state and codesLayerBytes apart in the cached one.
struct StateCodecShape final {
  uint32_t layers = 0;
  uint32_t rows = 0;
  uint32_t columns = 0;
  uint64_t stateLayerBytes = 0;
  uint64_t codesLayerBytes = 0;
  GdnStateFormat format{};

  [[nodiscard]] constexpr uint64_t stateBytes() const noexcept {
    return uint64_t{rows} * columns * elementBytes(format);
  }
  [[nodiscard]] constexpr uint64_t codesBytes() const noexcept { return state_codec_layer_bytes(rows, columns); }
  [[nodiscard]] constexpr bool valid() const noexcept {
    return layers && rows && columns && columns % SPLASH_STATE_CODEC_BLOCK == 0 && validFormat(format) &&
           stateLayerBytes >= stateBytes() && codesLayerBytes >= codesBytes();
  }
};

// Quantizes a running state's layers into a cached state's and dequantizes
// them back, one dispatch over every layer of `shape`; `state` and `codes`
// start at the first layer.
class StateCodec final {
public:
  static void addQuantize(metal::CommandGraph &graph, metal::MetalBuffer state, metal::MetalBuffer codes,
                          StateCodecShape shape);
  static void addDequantize(metal::CommandGraph &graph, metal::MetalBuffer codes, metal::MetalBuffer state,
                            StateCodecShape shape);
};

} // namespace splash::ops
