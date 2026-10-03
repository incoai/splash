#pragma once

#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::ops {

// Rows of `stride` bf16 values from row `row` of a buffer, of which a copy
// reads or writes the values from `column` on.
struct RowRegion final {
  uint32_t row = 0;
  uint32_t stride = 0;
  uint32_t column = 0;
};

// Copies rows of bf16 values between two row regions: a finished prompt's
// last prefill row into the head's input, and a capture layer's output into
// its slot of the draft's captured target hidden rows.
class RowCopy final {
public:
  // Copies `rows` rows of `width` values from `source` at `from` to
  // `destination` at `to`.
  static void add(metal::CommandGraph &graph, metal::MetalBuffer source,
                  RowRegion from, metal::MetalBuffer destination, RowRegion to,
                  uint32_t rows, uint32_t width);
};

} // namespace splash::ops
