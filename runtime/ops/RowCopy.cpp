#include "ops/RowCopy.hpp"

#include "metal/abi/RowCopy.h"
#include "ops/BufferExtent.hpp"

#include <stdexcept>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kThreads = metal::CommandGraph::kDefaultThreads;

// The bytes from a buffer's start to the end of `rows` rows of `width`
// values of `region`, which the copy reads or writes.
uint64_t regionBytes(RowRegion region, uint32_t rows, uint32_t width) {
  return rowBytes(uint64_t{region.row} + rows, region.stride, uint64_t{region.column} + width, 2);
}

} // namespace

void RowCopy::add(metal::CommandGraph &graph, metal::MetalBuffer source,
                  RowRegion from, metal::MetalBuffer destination, RowRegion to,
                  uint32_t rows, uint32_t width, RowLanes lanes) {
  if (!rows || !width || from.column + uint64_t{width} > from.stride ||
      to.column + uint64_t{width} > to.stride)
    throw std::invalid_argument("invalid row copy");
  requireBytes(source, regionBytes(from, rows, width), "row copy source");
  // A scattered destination's rows land past the flat span: the last group
  // sits (its groups - 1) lane strides out.
  uint64_t landedRows = rows;
  if (lanes.rows) {
    const uint64_t groups = (uint64_t{rows} + lanes.rows - 1) / lanes.rows;
    landedRows = (groups - 1) * lanes.stride + lanes.rows;
  }
  requireBytes(destination, rowBytes(uint64_t{to.row} + landedRows, to.stride,
                                     uint64_t{to.column} + width, 2),
               "row copy destination");
  const RowCopyParams params{rows,
                             width,
                             from.row,
                             from.stride,
                             from.column,
                             to.row,
                             to.stride,
                             to.column,
                             lanes.rows,
                             lanes.stride};
  // One thread per value.
  graph.add("copy_rows_bf16", {std::move(source), std::move(destination)},
            params, {(uint64_t{rows} * width + kThreads - 1) / kThreads, 1, 1},
            {kThreads, 1, 1});
}

} // namespace splash::ops
