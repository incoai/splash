#include "model/GgufPreparation.hpp"

#include "metal/abi/GgufRepack.h"
#include "model/Bfloat16.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/WeightImages.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace splash::model {
namespace {

// A chunk's rows, columns and row bytes are within the staging bound, so they
// fit the repack kernel's 32-bit parameters; its planes' distances are checked
// where it is repacked.
static_assert(kGgufRepackStagingBytes <= std::numeric_limits<uint32_t>::max());

// The source row image row `row` of `rows` is read from.
uint64_t sourceRow(const gguf::TensorRows &rows, uint64_t row) {
  const gguf::RowOrder &order = rows.order;
  if (row < order.from) return row;
  if (!order.headRows || !order.keyHeads || !order.valueHeadsPerKey)
    throw GgufError("invalid weight row permutation");
  const uint64_t head = (row - order.from) / order.headRows;
  const uint64_t source = order.from +
      ((head % order.valueHeadsPerKey) * order.keyHeads + head / order.valueHeadsPerKey) * order.headRows +
      (row - order.from) % order.headRows;
  if (source >= rows.rows) throw GgufError("weight row permutation is out of bounds");
  return source;
}

void requireRange(uint64_t offset, uint64_t bytes, uint64_t available) {
  if (offset > available || bytes > available - offset)
    throw GgufError("prepared weight section is out of bounds");
}

// Image rows [first, first + count) of `rows`, bytes [column, column + span)
// of each, back to back. Consecutive source rows are read together.
void readRows(const WeightSource &source, const gguf::TensorRows &rows, uint64_t first,
              uint64_t count, uint64_t column, uint64_t span, uint8_t *to) {
  if (first > rows.rows || count > rows.rows - first || column > rows.rowBytes || span > rows.rowBytes - column)
    throw GgufError("prepared weight rows are out of bounds");
  for (uint64_t row = 0; row < count;) {
    const uint64_t start = sourceRow(rows, first + row);
    uint64_t run = 1;
    if (span == rows.rowBytes)
      while (row + run < count && sourceRow(rows, first + row + run) == start + run) ++run;
    source.readData(rows.offset + start * rows.rowBytes + column, {to + row * span, run * span});
    row += run;
  }
}

// Writes the bf16 values `count` F32 values equal: the kernels read these
// tensors as bf16, and loading never rounds a weight.
void narrowToBfloat16(const uint8_t *values, uint64_t count, uint8_t *to, const std::string &name) {
  for (uint64_t i = 0; i < count; ++i) {
    float value;
    std::memcpy(&value, values + 4 * i, 4);
    const std::optional<uint16_t> narrowed = exactBfloat16(value);
    if (!narrowed) throw GgufError(name + " is not bf16-exact; it needs an F32 path");
    std::memcpy(to + 2 * i, &*narrowed, 2);
  }
}

// Writes the F32 values `count` BF16 values equal.
void widenToFloat32(const uint8_t *values, uint64_t count, uint8_t *to) {
  for (uint64_t i = 0; i < count; ++i) {
    uint16_t value;
    std::memcpy(&value, values + 2 * i, 2);
    const uint32_t widened = uint32_t{value} << 16;
    std::memcpy(to + 4 * i, &widened, 4);
  }
}

// Destination bytes of source bytes: halved when narrowed, doubled when widened.
uint64_t copiedBytes(const gguf::Copy &copy, uint64_t sourceBytes) {
  switch (copy.conversion) {
  case gguf::Conversion::NarrowToBfloat16: return sourceBytes / 2;
  case gguf::Conversion::WidenToFloat32: return sourceBytes * 2;
  case gguf::Conversion::None: break;
  }
  return sourceBytes;
}

uint64_t copyBytes(const gguf::Copy &copy) {
  return copiedBytes(copy, copy.source.rows * copy.source.rowBytes);
}

// The tasks that write a copy into image, each of whole rows within
// kLoadStepBytes or, for a wider row, a piece of whole values of one row: a
// copy as stored reads in place, a converted one through its thread's
// staging.
void addCopyTasks(const WeightSource &source, uint8_t *image, const gguf::Copy &copy,
                  std::vector<std::function<void(std::vector<uint8_t> &)>> &tasks) {
  const gguf::TensorRows &rows = copy.source;
  const uint64_t span = std::min<uint64_t>(rows.rowBytes, kLoadStepBytes & ~uint64_t{3});
  const uint64_t batch = span == rows.rowBytes ? kLoadStepBytes / rows.rowBytes : 1;
  for (uint64_t first = 0; first < rows.rows; first += batch) {
    const uint64_t count = std::min(batch, rows.rows - first);
    for (uint64_t column = 0; column < rows.rowBytes; column += span) {
      const uint64_t width = std::min(span, rows.rowBytes - column);
      uint8_t *to = image + copy.destination + copiedBytes(copy, first * rows.rowBytes + column);
      tasks.push_back([&source, &copy, first, count, column, width, to](std::vector<uint8_t> &staging) {
        const gguf::TensorRows &rows = copy.source;
        if (copy.conversion == gguf::Conversion::None) {
          readRows(source, rows, first, count, column, width, to);
          return;
        }
        if (staging.size() < count * width) staging.resize(count * width);
        readRows(source, rows, first, count, column, width, staging.data());
        if (copy.conversion == gguf::Conversion::NarrowToBfloat16)
          narrowToBfloat16(staging.data(), count * width / 4, to, rows.name);
        else
          widenToFloat32(staging.data(), count * width / 2, to);
      });
    }
  }
}

// The plane0, plane1 and meta bytes of a [rows, columns] tensor in format.
std::array<uint64_t, 3> planeBytes(const QuantFormat &format, uint64_t rows, uint64_t columns) {
  const GgufPlaneBytes bytes = ggufPlaneBytes(format, rows, columns);
  return {bytes.plane0, bytes.plane1, bytes.meta};
}

// Where a repack's plane0, plane1 and meta start in the image.
std::array<uint64_t, 3> planeOffsets(const gguf::Repack &repack) {
  return {repack.plane0, repack.plane1, repack.meta};
}

// The rows and columns of one repack step and its staging: complete rows
// where they fit, so one read and one submission cover many plane tiles;
// very wide rows split within the same bound.
struct RepackChunk {
  uint64_t rows = 0, columns = 0, inputBytes = 0;
};

RepackChunk repackChunk(const gguf::Repack &repack) {
  const QuantFormat &format = kQuantFormats[repack.format];
  RepackChunk chunk;
  chunk.columns = std::min<uint64_t>(repack.columns,
      kGgufRepackStagingBytes / (QUANT_TILE_ROWS * ggufRowBytes(format, kGgufBlockColumns)) * kGgufBlockColumns);
  if (!chunk.columns) throw GgufError("weight row exceeds the repack staging");
  const uint64_t tileBytes = QUANT_TILE_ROWS * ggufRowBytes(format, chunk.columns);
  chunk.rows = chunk.columns == repack.columns
      ? std::min<uint64_t>(repack.rows, kGgufRepackStagingBytes / tileBytes * QUANT_TILE_ROWS)
      : QUANT_TILE_ROWS;
  chunk.inputBytes = chunk.rows * ggufRowBytes(format, chunk.columns);
  return chunk;
}

// A repack's plan within an image of imageBytes: its format, tile-aligned
// shape, sources of its row width and planes inside the image.
void requireRepack(const gguf::Repack &repack, uint64_t imageBytes) {
  if (repack.format >= GGUF_FMT_COUNT || !repack.rows || repack.rows % QUANT_TILE_ROWS || !repack.columns ||
      repack.columns % kGgufBlockColumns)
    throw GgufError("invalid prepared weight repack");
  const QuantFormat &format = kQuantFormats[repack.format];
  const uint64_t rowBytes = ggufRowBytes(format, repack.columns);
  uint64_t sourceRows = 0;
  for (const gguf::TensorRows &rows : repack.sources) {
    if (rows.rowBytes != rowBytes) throw GgufError("invalid prepared weight source size");
    sourceRows += rows.rows;
  }
  if (sourceRows > repack.rows) throw GgufError("invalid prepared weight source size");
  // A format without plane1 has an empty plane1 at offset 0.
  const auto offsets = planeOffsets(repack);
  const auto sizes = planeBytes(format, repack.rows, repack.columns);
  for (size_t plane = 0; plane < sizes.size(); ++plane) requireRange(offsets[plane], sizes[plane], imageBytes);
}

// Repacks a repack's rows chunk by chunk: threads read the rows into the
// input staging, and the GPU writes their planes in place into image.
void writeRepack(metal::MetalBackend &backend, const WeightSource &source, const metal::MetalBuffer &image,
                 const gguf::Repack &repack, const RepackChunk &chunk, const metal::MetalBuffer &input) {
  const QuantFormat &format = kQuantFormats[repack.format];
  const auto offsets = planeOffsets(repack);
  auto *host = static_cast<uint8_t *>(input.contents());
  for (uint64_t firstRow = 0; firstRow < repack.rows; firstRow += chunk.rows) {
    const uint64_t rows = std::min(chunk.rows, repack.rows - firstRow);
    for (uint64_t firstColumn = 0; firstColumn < repack.columns; firstColumn += chunk.columns) {
      const uint64_t columns = std::min(chunk.columns, repack.columns - firstColumn);
      const uint64_t chunkRowBytes = ggufRowBytes(format, columns);
      const uint64_t column = ggufRowBytes(format, firstColumn);
      // The sources' rows in image order, then zero rows, a task for each
      // run of rows within kLoadStepBytes.
      struct Read {
        const gguf::TensorRows *tensor;
        uint64_t first, count;
        uint8_t *to;
      };
      std::vector<Read> reads;
      const uint64_t batch = std::max<uint64_t>(1, kLoadStepBytes / chunkRowBytes);
      uint64_t start = 0;
      for (const gguf::TensorRows &tensor : repack.sources) {
        const uint64_t begin = std::max(firstRow, start), end = std::min(firstRow + rows, start + tensor.rows);
        for (uint64_t row = begin; row < end; row += batch)
          reads.push_back({&tensor, row - start, std::min(batch, end - row), host + (row - firstRow) * chunkRowBytes});
        start += tensor.rows;
      }
      parallelFor(reads.size(), [&](size_t index, unsigned) {
        const Read &read = reads[index];
        readRows(source, *read.tensor, read.first, read.count, column, chunkRowBytes, read.to);
      });
      if (start < firstRow + rows) {
        const uint64_t zero = std::max(start, firstRow);
        std::memset(host + (zero - firstRow) * chunkRowBytes, 0, (firstRow + rows - zero) * chunkRowBytes);
      }
      // A plane is [rows / QUANT_TILE_ROWS][units][QUANT_TILE_ROWS] tiles and
      // a chunk starts on a tile: after the planes of the rows above it and,
      // in its tile rows, of the columns before it. The kernel writes plane1
      // and meta at their distances from plane0, which the planner placed
      // after it; a format without plane1 writes none.
      const auto lengths = planeBytes(format, rows, columns);
      const auto above = planeBytes(format, firstRow, repack.columns);
      const auto before = planeBytes(format, QUANT_TILE_ROWS, firstColumn);
      std::array<uint64_t, 3> at;
      for (size_t plane = 0; plane < at.size(); ++plane) at[plane] = offsets[plane] + above[plane] + before[plane];
      const uint64_t extent = at[2] + lengths[2] - at[0];
      if ((lengths[1] && at[1] < at[0]) || at[2] < at[0] || extent > std::numeric_limits<uint32_t>::max())
        throw GgufError("a repack's planes do not follow its plane0 within 4 GiB");
      GgufRepackParams params{};
      params.rows = static_cast<uint32_t>(rows);
      params.input_size = static_cast<uint32_t>(columns);
      params.fmt = repack.format;
      params.src_row_bytes = static_cast<uint32_t>(chunkRowBytes);
      params.dst_plane1 = lengths[1] ? static_cast<uint32_t>(at[1] - at[0]) : 0;
      params.dst_meta = static_cast<uint32_t>(at[2] - at[0]);
      static_cast<void>(backend.submit({"gguf_repack", {{0, input}, {1, backend.view(image, at[0], extent)}},
                                        {{2, &params, sizeof(params)}}, {rows * (columns / 32) / 256, 1, 1},
                                        {256, 1, 1}}));
    }
  }
}

} // namespace

void writeGgufImage(metal::MetalBackend &backend, const WeightSource &source, const metal::MetalBuffer &image,
                    const gguf::Image &plan) {
  const std::span<uint8_t> destination = contentsOf(image);
  if (destination.size() != plan.bytes) throw GgufError("GGUF image size differs from its plan");
  std::vector<std::pair<uint64_t, uint64_t>> extents;
  for (const gguf::Fill &fill : plan.fills) extents.emplace_back(fill.offset, fill.bytes.size());
  for (const gguf::Copy &copy : plan.copies) {
    if (!copy.source.rows || !copy.source.rowBytes ||
        (copy.conversion != gguf::Conversion::None && copy.source.rowBytes % 4))
      throw GgufError("invalid prepared weight copy");
    extents.emplace_back(copy.destination, copyBytes(copy));
  }
  std::vector<RepackChunk> chunks;
  uint64_t staging = 0;
  for (const gguf::Repack &repack : plan.repacks) {
    requireRepack(repack, plan.bytes);
    const auto offsets = planeOffsets(repack);
    const auto sizes = planeBytes(kQuantFormats[repack.format], repack.rows, repack.columns);
    for (size_t plane = 0; plane < sizes.size(); ++plane) extents.emplace_back(offsets[plane], sizes[plane]);
    chunks.push_back(repackChunk(repack));
    staging = std::max(staging, chunks.back().inputBytes);
  }
  zeroUnwritten(destination, std::move(extents));
  for (const gguf::Fill &fill : plan.fills)
    std::memcpy(destination.data() + fill.offset, fill.bytes.data(), fill.bytes.size());
  // Copies keep their source precision and never pass through a
  // quantization operation.
  std::vector<std::function<void(std::vector<uint8_t> &)>> tasks;
  for (const gguf::Copy &copy : plan.copies) addCopyTasks(source, destination.data(), copy, tasks);
  std::vector<std::vector<uint8_t>> copyStaging(loadThreads());
  parallelFor(tasks.size(), [&](size_t index, unsigned thread) { tasks[index](copyStaging[thread]); });
  // One staging buffer serves every repack of the image.
  if (!plan.repacks.empty()) {
    const auto input = backend.allocateBuffer(staging, metal::BufferStorage::Shared, "load/rows");
    for (size_t i = 0; i < plan.repacks.size(); ++i)
      writeRepack(backend, source, image, plan.repacks[i], chunks[i], input);
  }
}

} // namespace splash::model
