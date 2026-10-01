#pragma once

#include <cstdint>

namespace splash::ops {

// --ane-split: the prefill FFN on the GPU alone, split at a given share of
// its channels with the Neural Engine (ops/AneFfn.hpp), or split at the share
// measured fastest on the device.
struct AneSplit final {
  enum class Mode : uint8_t { Off, Fixed, Automatic };
  Mode mode = Mode::Off;
  // The Neural Engine's share of the channels, in (0, 1), for Fixed.
  double share = 0.0;
};

} // namespace splash::ops
