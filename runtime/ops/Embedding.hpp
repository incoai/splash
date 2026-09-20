#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"

#include <cstdint>

namespace splash::ops {

class Embedding final {
public:
  static void add(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                  const Q4Projection &table, metal::MetalBuffer output,
                  uint32_t rows);
  static void add(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                  const Q8Projection &table, metal::MetalBuffer output,
                  uint32_t rows);
};

} // namespace splash::ops
