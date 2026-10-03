#include "ops/RowCopy.hpp"

#include "metal/abi/RowCopy.h"

#include <stdexcept>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kThreads = metal::CommandGraph::kDefaultThreads;

// Whether `buffer` holds `rows` rows of `width` values of `region`.
bool holds(const metal::MetalBuffer &buffer, RowRegion region, uint32_t rows,
           uint32_t width) {
  return region.column + uint64_t{width} <= region.stride &&
         buffer.sizeBytes() >=
             ((uint64_t{region.row} + rows - 1) * region.stride +
              region.column + width) * 2;
}

} // namespace

void RowCopy::add(metal::CommandGraph &graph, metal::MetalBuffer source,
                  RowRegion from, metal::MetalBuffer destination, RowRegion to,
                  uint32_t rows, uint32_t width) {
  if (!rows || !width || !holds(source, from, rows, width) ||
      !holds(destination, to, rows, width))
    throw std::invalid_argument("invalid row copy");
  const RowCopyParams params{rows,        width,     from.row,
                             from.stride, from.column, to.row,
                             to.stride,   to.column};
  // One thread per value.
  graph.add("copy_rows_bf16", {std::move(source), std::move(destination)},
            params, {(uint64_t{rows} * width + kThreads - 1) / kThreads, 1, 1},
            {kThreads, 1, 1});
}

} // namespace splash::ops
