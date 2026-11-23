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
uint64_t regionBytes(RowRegion region, uint32_t rows, uint32_t width, FloatElement element) {
  return rowBytes(uint64_t{region.row} + rows, region.stride, uint64_t{region.column} + width, elementBytes(element));
}

} // namespace

void RowCopy::add(metal::CommandGraph &graph, metal::MetalBuffer source,
                  RowRegion from, metal::MetalBuffer destination, RowRegion to,
                  uint32_t rows, uint32_t width, FloatElement sourceElement) {
  if (!rows || !width || from.column + uint64_t{width} > from.stride ||
      to.column + uint64_t{width} > to.stride)
    throw std::invalid_argument("invalid row copy");
  requireBytes(source, regionBytes(from, rows, width, sourceElement), "row copy source");
  requireBytes(destination, regionBytes(to, rows, width, FloatElement::BFloat16), "row copy destination");
  const RowCopyParams params{rows,        width,     from.row,
                             from.stride, from.column, to.row,
                             to.stride,   to.column};
  // One thread per value.
  graph.add(sourceElement == FloatElement::Float32 ? "copy_rows_f32_bf16" : "copy_rows_bf16",
            {std::move(source), std::move(destination)},
            params, {(uint64_t{rows} * width + kThreads - 1) / kThreads, 1, 1},
            {kThreads, 1, 1});
}

} // namespace splash::ops
