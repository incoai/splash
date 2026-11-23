#include "model/GgufPreparation.hpp"

#include "metal/abi/Gguf.h"
#include "metal/abi/GgufRepack.h"
#include "model/Bfloat16.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/WeightImages.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <span>
#include <string>
#include <utility>
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
  if (row < order.from) return rows.first + row;
  if (!order.headRows || !order.keyHeads || !order.valueHeadsPerKey)
    throw GgufError("invalid weight row permutation");
  const uint64_t head = (row - order.from) / order.headRows;
  const uint64_t source = order.from +
      ((head % order.valueHeadsPerKey) * order.keyHeads + head / order.valueHeadsPerKey) * order.headRows +
      (row - order.from) % order.headRows;
  if (source >= rows.rows) throw GgufError("weight row permutation is out of bounds");
  return rows.first + source;
}

// Whether a source row of `rows` splits into its `windows` equal windows and
// it names one of them (gguf::TensorRows::windows), as imageRowBytes() needs.
bool validWindow(const gguf::TensorRows &rows) {
  return rows.windows && rows.window < rows.windows && rows.rowBytes % rows.windows == 0;
}

void requireRange(uint64_t offset, uint64_t bytes, uint64_t available) {
  if (offset > available || bytes > available - offset)
    throw GgufError("prepared weight section is out of bounds");
}

// How readRows reads a tensor's rows: as stored, or as its format's native
// rows, which it builds from a quantized safetensors tensor's tensors, from
// a GGUF's NVFP4 rows, or by quantizing a BF16 weight (into af4g64) or a
// GGUF tensor of BF16 or F16 values (into Q8_0, or exactly into MXFP8).
enum class RowReader : uint8_t { Stored, Safetensors, GgufNvfp4, QuantizedBfloat16, QuantizedGgufFloat };
RowReader rowReader(const gguf::TensorRows &rows) {
  if (rows.safetensors.codes) return RowReader::Safetensors;
  if (rows.ggufNvfp4RowBytes) return RowReader::GgufNvfp4;
  if (rows.safetensors.bfloat16) return RowReader::QuantizedBfloat16;
  if (rows.ggufFloatRowBytes) return RowReader::QuantizedGgufFloat;
  return RowReader::Stored;
}

// The format of the native rows a reader builds (rowReader).
const QuantFormat &nativeFormat(const gguf::TensorRows &rows) {
  const uint32_t format = gguf_format_of(rows.type);
  if (!gguf::safetensorsFormat(format)) throw GgufError("not a format the loader builds: " + rows.name);
  return kQuantFormats[format];
}

// The tensor scale g of source rows [start, start + count), each finite
// (gguf::TensorScale); 1 for rows without one.
std::vector<float> tensorScales(const gguf::TensorRows &rows, uint64_t start, uint64_t count) {
  const gguf::TensorScale &scale = rows.scale;
  if (!scale.rowsPerValue) return std::vector<float>(count, 1.0F);
  const uint64_t valueBytes = scale.bfloat16 ? 2 : 4, first = start / scale.rowsPerValue;
  std::vector<uint8_t> bytes(((start + count - 1) / scale.rowsPerValue - first + 1) * valueBytes);
  if (first * valueBytes > scale.bytes || bytes.size() > scale.bytes - first * valueBytes)
    throw GgufError("tensor scale of " + rows.name + " is out of bounds");
  scale.file->readData(scale.offset + first * valueBytes, bytes);
  std::vector<float> result(count);
  for (uint64_t row = 0; row < count; ++row) {
    uint32_t bits = 0;
    std::memcpy(&bits, bytes.data() + ((start + row) / scale.rowsPerValue - first) * valueBytes, valueBytes);
    float value = scale.bfloat16 ? widenBfloat16(uint16_t(bits)) : std::bit_cast<float>(bits);
    if (scale.reciprocal) value = 1.0F / value;
    if (!std::isfinite(value)) throw GgufError("non-finite tensor scale of " + rows.name);
    result[row] = value;
  }
  return result;
}

// The bytes of a native block of a quantized safetensors tensor's format
// from its codes tensor and from its scales tensor; an affine tensor's
// biases take as many as its scales.
struct StoredBlock final {
  uint64_t codes = 0, scales = 0;
};
StoredBlock storedBlock(uint32_t id, const QuantFormat &format) {
  const bool affine = quant_affine_format(id);
  return {format.block_bytes -
              (affine ? 4 : id == GGUF_FMT_NVFP4 ? QUANT_NVFP4_CODES : id == GGUF_FMT_FP8 ? QUANT_FP8_VALUES : 1),
          affine ? 2u : id == GGUF_FMT_NVFP4 ? QUANT_NVFP4_G : id == GGUF_FMT_FP8 ? 0u : 1u};
}

// Native bytes [column, column + span) of `count` source rows of a quantized
// safetensors tensor from source row `start` on, back to back, read from its
// tensors: per native block an affine tensor's scale, bias and codes; an
// mxfp4 tensor's block_mxfp4, its exponent and its codes with element j in
// the low nibble of byte j % 16 when j < 16, else in its high nibble (MLX
// packs element j at bits 4 j of the row); an nvfp4 tensor's 16 E4M3 scales,
// its row's tensor scale and its codes as stored; an fp8 tensor's row's tensor
// scale and its E4M3 values; an mxfp8 tensor's exponent and its E4M3 values
// (metal/abi/QuantFormat.h).
void readSafetensorsRows(const gguf::TensorRows &rows, uint64_t start, uint64_t count, uint64_t column, uint64_t span,
                 uint8_t *to) {
  const QuantFormat &format = nativeFormat(rows);
  const uint32_t id = gguf_format_of(rows.type);
  const bool affine = quant_affine_format(id);
  const uint64_t block = format.block_bytes, groups = rows.rowBytes / block;
  const auto [codes, scaleBytes] = storedBlock(id, format);
  if (column % block || span % block)
    throw GgufError("quantized safetensors rows are read by whole blocks: " + rows.name);
  // One read of each tensor per run of rows when they are read whole, else per row.
  const uint64_t perRead = span == rows.rowBytes ? count : 1, spanBlocks = span / block;
  std::vector<uint8_t> codeBytes(perRead * spanBlocks * codes), scales(perRead * spanBlocks * scaleBytes),
      biases(affine ? perRead * spanBlocks * 2 : 0);
  const std::vector<float> g = tensorScales(rows, start, count);
  for (uint64_t row = 0; row < count; row += perRead) {
    const uint64_t at = (start + row) * groups + column / block;
    rows.safetensors.codes->read(at * codes, codeBytes);
    if (scaleBytes) rows.safetensors.scales->read(at * scaleBytes, scales);
    if (affine) rows.safetensors.biases->read(at * 2, biases);
    for (uint64_t b = 0; b < perRead * spanBlocks; ++b) {
      uint8_t *out = to + row * span + b * block;
      const uint8_t *in = codeBytes.data() + b * codes;
      const float scale = g[row + b / spanBlocks];
      if (affine) {
        std::memcpy(out, scales.data() + 2 * b, 2);
        std::memcpy(out + 2, biases.data() + 2 * b, 2);
        std::memcpy(out + 4, in, codes);
      } else if (id == GGUF_FMT_NVFP4) {
        std::memcpy(out, scales.data() + QUANT_NVFP4_G * b, QUANT_NVFP4_G);
        std::memcpy(out + QUANT_NVFP4_G, &scale, 4);
        std::memcpy(out + QUANT_NVFP4_CODES, in, codes);
      } else if (id == GGUF_FMT_FP8) {
        std::memcpy(out, &scale, 4);
        std::memcpy(out + QUANT_FP8_VALUES, in, codes);
      } else if (id == GGUF_FMT_MXFP8) {
        out[0] = scales[b];
        std::memcpy(out + 1, in, codes);
      } else {
        out[0] = scales[b];
        for (uint64_t j = 0; j < 16; ++j)
          out[1 + j] = uint8_t(((in[j / 2] >> (4 * (j % 2))) & 15) | ((in[8 + j / 2] >> (4 * (j % 2))) & 15) << 4);
      }
    }
  }
}

// The BF16 nearest a finite value, ties to even.
uint16_t nearestBfloat16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

// A group of 64 BF16 weights as MLX's affine quantization rounds it to 4 bits
// (mlx.core.quantize, its Metal kernel), in float, into a native af4g64
// block: the range runs from the minimum to the maximum or 0, whichever is
// greater, the end of the range farther from zero is the bias, the scale is
// adjusted so that 0 falls on a code unless that code is 0, each code is
// round((w - bias) / scale), halves away from zero, clamped to [0, 15], and
// scale and bias are stored as BF16. The loops have no early exit, so they
// vectorize; the sign of a zero minimum or maximum, which the order of a
// vector reduction may change, does not reach any output.
void quantizeGroup(const uint8_t *weights, uint8_t *block) {
  std::array<float, 64> w;
  bool finite = true;
  for (size_t i = 0; i < w.size(); ++i) {
    uint16_t bits;
    std::memcpy(&bits, weights + 2 * i, 2);
    w[i] = widenBfloat16(bits);
    finite &= std::isfinite(w[i]);
  }
  if (!finite) throw GgufError("non-finite weight in a BF16 projection");
  float minimum = w[0], maximum = w[0];
  for (float value : w) {
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
  }
  maximum = std::max(0.0F, maximum);
  const bool minimumEdge = std::fabs(minimum) > std::fabs(maximum);
  float step = std::max((maximum - minimum) / 15.0F, 1e-7F);
  if (!minimumEdge) step = -step;
  const float edge = minimumEdge ? minimum : maximum;
  const float q0 = std::round(edge / step);
  float offset = 0.0F;
  if (q0 != 0.0F) {
    step = edge / q0;
    offset = edge;
  }
  std::array<uint8_t, 64> code;
  for (size_t i = 0; i < w.size(); ++i)
    code[i] = static_cast<uint8_t>(std::clamp(std::round((w[i] - offset) / step), 0.0F, 15.0F));
  const uint16_t scale = nearestBfloat16(step), bias = nearestBfloat16(offset);
  std::memcpy(block, &scale, 2);
  std::memcpy(block + 2, &bias, 2);
  for (size_t i = 0; i < w.size(); i += 2) block[4 + i / 2] = static_cast<uint8_t>(code[i] | code[i + 1] << 4);
}

// Native af4g64 bytes [column, column + span) of `count` rows of a BF16
// weight from source row `start` on, back to back, each group quantized as
// MLX's affine quantization rounds it. One read per run of rows when they are
// read whole, else per row.
void quantizeRows(const gguf::TensorRows &rows, uint64_t start, uint64_t count, uint64_t column, uint64_t span,
                  uint8_t *to) {
  const QuantFormat &format = kQuantFormats[GGUF_FMT_AF4G64];
  if (gguf_format_of(rows.type) != GGUF_FMT_AF4G64 || column % format.block_bytes || span % format.block_bytes)
    throw GgufError("a BF16 weight is quantized into whole af4g64 groups: " + rows.name);
  const uint64_t groupBytes = format.block_elements * 2, groups = rows.rowBytes / format.block_bytes;
  const uint64_t perRead = span == rows.rowBytes ? count : 1, spanGroups = span / format.block_bytes;
  std::vector<uint8_t> weights(perRead * spanGroups * groupBytes);
  for (uint64_t row = 0; row < count; row += perRead) {
    rows.safetensors.bfloat16->read(((start + row) * groups + column / format.block_bytes) * groupBytes, weights);
    for (uint64_t group = 0; group < perRead * spanGroups; ++group)
      quantizeGroup(weights.data() + group * groupBytes, to + row * span + group * format.block_bytes);
  }
}

// The values of a block_q8_0.
constexpr uint32_t kQ8_0BlockValues = kQuantFormats[GGUF_FMT_Q80].block_elements;

// The float the bits of a BF16 or, with float16, an F16 value hold.
float widenFloat16(uint16_t bits, bool float16) {
  return float16 ? static_cast<float>(std::bit_cast<__fp16>(bits)) : widenBfloat16(bits);
}

// A block's values as the reference quantize_row_q8_0_ref writes them into a
// block_q8_0, as GGUF's Q8_0 is made: d = max |x| / 127, stored as the
// nearest fp16, and each q = roundf(x * (1 / d)) of the unrounded d (0 when
// d is 0), halves away from zero. A value or a d fp16 does not hold is
// refused.
void quantizeQ8_0Block(const float *values, uint8_t *block, const std::string &name) {
  float maximum = 0.0F;
  bool finite = true;
  for (uint32_t i = 0; i < kQ8_0BlockValues; ++i) {
    finite &= std::isfinite(values[i]);
    maximum = std::max(maximum, std::fabs(values[i]));
  }
  const float d = maximum / 127, inverse = d != 0.0F ? 1.0F / d : 0.0F;
  const __fp16 scale = static_cast<__fp16>(d);
  if (!finite || !std::isfinite(static_cast<float>(scale)))
    throw GgufError(name + " holds a value Q8_0 cannot hold");
  std::memcpy(block, &scale, 2);
  for (uint32_t i = 0; i < kQ8_0BlockValues; ++i)
    block[2 + i] = static_cast<uint8_t>(static_cast<int8_t>(std::round(values[i] * inverse)));
}

// The MXFP8 native block of 32 values exactly (gguf::mxfp8Block), which the
// plan took them to be (gguf::PlanOptions::exactlyMxfp8); a value it is not
// is refused.
void quantizeMxfp8Block(const float *values, uint8_t *block, const std::string &name) {
  if (!gguf::mxfp8Block(std::span<const float, 32>(values, 32), std::span<uint8_t, 33>(block, 33)))
    throw GgufError(name + " holds a value MXFP8 does not hold exactly");
}

// Native Q8_0 or MXFP8 bytes (the format of rows.type) [column, column +
// span) of `count` rows of a GGUF tensor of BF16 or F16 values from source
// row `start` on, back to back, each 32 values built as quantizeQ8_0Block
// rounds them or as quantizeMxfp8Block holds them. One read per run of rows
// when they are read whole, else per row.
void quantizeFloat16Rows(const gguf::TensorRows &rows, uint64_t start, uint64_t count, uint64_t column,
                         uint64_t span, uint8_t *to) {
  const bool mxfp8 = gguf_format_of(rows.type) == GGUF_FMT_MXFP8;
  const QuantFormat &format = kQuantFormats[mxfp8 ? GGUF_FMT_MXFP8 : GGUF_FMT_Q80];
  if (column % format.block_bytes || span % format.block_bytes)
    throw GgufError(std::string(format.name) + " rows are built by whole blocks: " + rows.name);
  const bool float16 = rows.ggufFloatType == ggml::kF16;
  const uint64_t perRead = span == rows.rowBytes ? count : 1, blocks = span / format.block_bytes;
  std::vector<uint8_t> source(perRead * blocks * format.block_elements * 2);
  std::vector<float> values(source.size() / 2);
  for (uint64_t row = 0; row < count; row += perRead) {
    rows.file->readData(rows.offset + (start + row) * rows.ggufFloatRowBytes +
                            column / format.block_bytes * format.block_elements * 2,
                        source);
    for (size_t i = 0; i < values.size(); ++i) {
      uint16_t bits;
      std::memcpy(&bits, source.data() + 2 * i, 2);
      values[i] = widenFloat16(bits, float16);
    }
    for (uint64_t b = 0; b < perRead * blocks; ++b)
      (mxfp8 ? quantizeMxfp8Block : quantizeQ8_0Block)(values.data() + b * format.block_elements,
                                                        to + row * span + b * format.block_bytes, rows.name);
  }
}

// Native NVFP4 bytes [column, column + span) of `count` rows of llama.cpp's
// NVFP4 from source row `start` on, back to back: per 256 elements the 16
// UE4M3 scales of four block_nvfp4 as the E4M3 scales they are (llama.cpp
// reads 0x7F as 0 and ignores bit 7), the row's tensor scale, and the codes with
// element e in the low nibble of byte e / 2 when e is even, else in its high
// one (metal/abi/QuantFormat.h), where a block_nvfp4 holds element j of each
// 16 in the low nibble of byte j and element j + 8 in its high one. Per 64
// elements the same in NVFP4's form of 64-element blocks: its meta unit (the
// E4M3 scales, then g) leads each native block, its codes follow.
void readGgufNvfp4Rows(const gguf::TensorRows &rows, uint64_t start, uint64_t count, uint64_t column, uint64_t span,
                       uint8_t *to) {
  const QuantFormat &format = nativeFormat(rows);
  const uint64_t gAt = format.meta_bytes - sizeof(float), codesAt = format.meta_bytes;
  const GgmlTypeTraits &blocks = *ggmlTypeTraits(ggml::kNVFP4);
  const uint64_t perNative = format.block_elements / blocks.blockElements, nativeBlocks = span / format.block_bytes,
                 sourceSpan = nativeBlocks * perNative * blocks.blockBytes;
  // One read per run of rows when they are read whole (contiguous in the file), else per row.
  const uint64_t perRead = span == rows.rowBytes ? count : 1;
  std::vector<uint8_t> source(perRead * sourceSpan);
  const std::vector<float> g = tensorScales(rows, start, count);
  for (uint64_t row = 0; row < count; row += perRead) {
    rows.file->readData(rows.offset + (start + row) * rows.ggufNvfp4RowBytes +
                            column / format.block_bytes * perNative * blocks.blockBytes,
                        source);
    for (uint64_t b = 0; b < perRead * nativeBlocks; ++b) {
      uint8_t *out = to + row * span + b * format.block_bytes;
      for (uint64_t q = 0; q < perNative; ++q) {
        const uint8_t *in = source.data() + (b * perNative + q) * blocks.blockBytes;
        for (uint64_t s = 0; s < 4; ++s) {
          const uint8_t *codes = in + 4 + 8 * s;
          out[4 * q + s] = in[s] == 0x7F ? 0 : in[s] & 0x7F;
          // Elements 2 i and 2 i + 1: the low nibbles of bytes 2 i and 2 i + 1 for i < 4, else their high ones.
          uint8_t *pairs = out + codesAt + 32 * q + 8 * s;
          for (uint64_t i = 0; i < 4; ++i) {
            pairs[i] = uint8_t((codes[2 * i] & 15) | (codes[2 * i + 1] & 15) << 4);
            pairs[4 + i] = uint8_t(codes[2 * i] >> 4 | (codes[2 * i + 1] & 0xF0));
          }
        }
      }
      std::memcpy(out + gAt, &g[row + b / nativeBlocks], 4);
    }
  }
}

// Source rows [start, start + run) of `rows`, bytes [column, column + span)
// of each, back to back, by the tensor's reader.
void readSourceRows(const gguf::TensorRows &rows, uint64_t start, uint64_t run, uint64_t column, uint64_t span,
                    uint8_t *out) {
  switch (rowReader(rows)) {
  case RowReader::Stored: rows.file->readData(rows.offset + start * rows.rowBytes + column, {out, run * span}); break;
  case RowReader::Safetensors: readSafetensorsRows(rows, start, run, column, span, out); break;
  case RowReader::GgufNvfp4: readGgufNvfp4Rows(rows, start, run, column, span, out); break;
  case RowReader::QuantizedBfloat16: quantizeRows(rows, start, run, column, span, out); break;
  case RowReader::QuantizedGgufFloat: quantizeFloat16Rows(rows, start, run, column, span, out); break;
  }
}

// Image rows [first, first + count) of `rows`, bytes [column, column + span)
// of each, back to back, from their column window on. Consecutive source rows
// are read together where the bytes wanted are a whole window of each: a
// windowed tensor's as whole rows, whose window is copied out. Read a window
// at a time, Flint's shared expert down projection (47 layers of two windows
// of 2,048 rows) took 192,846 reads from its UD-Q4_K_XL GGUF and 580,819 from
// its MLX checkpoint (64 GB M5 Pro, 11-19). A column piece of a wide row is
// read row by row, as whole rows would multiply the bytes read.
void readRows(const gguf::TensorRows &rows, uint64_t first, uint64_t count, uint64_t column, uint64_t span,
              uint8_t *to) {
  if (!validWindow(rows) || first > rows.rows || count > rows.rows - first || column > rows.imageRowBytes() ||
      span > rows.imageRowBytes() - column)
    throw GgufError("prepared weight rows are out of bounds");
  const bool wholeWindows = span == rows.imageRowBytes();
  column += rows.window * rows.imageRowBytes();
  std::vector<uint8_t> wholeRows;
  for (uint64_t row = 0; row < count;) {
    const uint64_t start = sourceRow(rows, first + row);
    uint64_t run = 1;
    if (wholeWindows)
      while (row + run < count && sourceRow(rows, first + row + run) == start + run) ++run;
    uint8_t *out = to + row * span;
    if (wholeWindows && span != rows.rowBytes) {
      wholeRows.resize(run * rows.rowBytes);
      readSourceRows(rows, start, run, 0, rows.rowBytes, wholeRows.data());
      for (uint64_t r = 0; r < run; ++r) std::memcpy(out + r * span, wholeRows.data() + r * rows.rowBytes + column, span);
    } else {
      readSourceRows(rows, start, run, column, span, out);
    }
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

// Writes the F32 values `count` BF16 (or, with float16, F16) values equal.
void widenToFloat32(const uint8_t *values, uint64_t count, bool float16, uint8_t *to) {
  for (uint64_t i = 0; i < count; ++i) {
    uint16_t bits;
    std::memcpy(&bits, values + 2 * i, 2);
    const float widened = widenFloat16(bits, float16);
    std::memcpy(to + 4 * i, &widened, 4);
  }
}

// Writes the F32 1 + w of `count` BF16 RMSNorm weights w, which transformers
// stores 1 below the weights the norm multiplies by.
void centeredNorm(const uint8_t *values, uint64_t count, uint8_t *to) {
  for (uint64_t i = 0; i < count; ++i) {
    uint16_t weight;
    std::memcpy(&weight, values + 2 * i, 2);
    const float value = 1.0F + widenBfloat16(weight);
    std::memcpy(to + 4 * i, &value, 4);
  }
}

uint64_t copiedBytes(const gguf::Copy &copy, uint64_t sourceBytes) {
  return gguf::convertedBytes(copy.source, copy.conversion, sourceBytes);
}

uint64_t copyBytes(const gguf::Copy &copy) {
  return copiedBytes(copy, copy.source.rows * copy.source.rowBytes);
}

// The tasks that write a copy into image, each of whole rows within
// kLoadStepBytes or, for a wider row, a piece of whole values (whole native
// blocks of rows a reader builds) of one row: a copy as stored reads in
// place, a converted one through a staging vector of the task's own.
void addCopyTasks(uint8_t *image, const gguf::Copy &copy, std::vector<std::function<void()>> &tasks) {
  const gguf::TensorRows &rows = copy.source;
  const uint64_t unit = rowReader(rows) == RowReader::Stored ? 4 : nativeFormat(rows).block_bytes;
  const uint64_t span = std::min<uint64_t>(rows.rowBytes, kLoadStepBytes / unit * unit);
  const uint64_t batch = span == rows.rowBytes ? kLoadStepBytes / rows.rowBytes : 1;
  for (uint64_t first = 0; first < rows.rows; first += batch) {
    const uint64_t count = std::min(batch, rows.rows - first);
    for (uint64_t column = 0; column < rows.rowBytes; column += span) {
      const uint64_t width = std::min(span, rows.rowBytes - column);
      uint8_t *to = image + copy.destination + copiedBytes(copy, first * rows.rowBytes + column);
      tasks.push_back([&copy, first, count, column, width, to] {
        const gguf::TensorRows &rows = copy.source;
        if (copy.conversion == gguf::Conversion::None) {
          readRows(rows, first, count, column, width, to);
          return;
        }
        std::vector<uint8_t> staging(count * width);
        readRows(rows, first, count, column, width, staging.data());
        switch (copy.conversion) {
        case gguf::Conversion::NarrowToBfloat16:
          narrowToBfloat16(staging.data(), count * width / 4, to, rows.name);
          break;
        case gguf::Conversion::WidenToFloat32:
          widenToFloat32(staging.data(), count * width / 2, rows.type == ggml::kF16, to);
          break;
        case gguf::Conversion::CenteredNorm: centeredNorm(staging.data(), count * width / 2, to); break;
        case gguf::Conversion::Decay: {
          const bool bfloat16 = rows.type == ggml::kBF16;
          writeGdnDecay(staging.data(), count * width / (bfloat16 ? 2 : 4), bfloat16, to);
          break;
        }
        case gguf::Conversion::DequantizeToFloat32: // decoded on the GPU (addDecodedChunks)
        case gguf::Conversion::None: break;
        }
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
      repack.columns % ggufColumnUnit(kQuantFormats[repack.format]))
    throw GgufError("invalid prepared weight repack");
  const QuantFormat &format = kQuantFormats[repack.format];
  const uint64_t rowBytes = ggufRowBytes(format, repack.columns);
  uint64_t sourceRows = 0;
  for (const gguf::TensorRows &rows : repack.sources) {
    if (!validWindow(rows) || rows.imageRowBytes() != rowBytes)
      throw GgufError("invalid prepared weight source size");
    sourceRows += rows.rows;
  }
  if (sourceRows > repack.rows) throw GgufError("invalid prepared weight source size");
  // A format without plane1 has an empty plane1 at offset 0.
  const auto offsets = planeOffsets(repack);
  const auto sizes = planeBytes(format, repack.rows, repack.columns);
  for (size_t plane = 0; plane < sizes.size(); ++plane) requireRange(offsets[plane], sizes[plane], imageBytes);
}

// A chunk of an image's GPU writes: `stage` reads its rows into a staging
// buffer's host bytes on the load threads, and
// the dispatch of `pipeline` writes them from there into the image, the
// staging buffer bound at index `input` beside `buffers`.
struct GpuChunk {
  std::function<void(uint8_t *host)> stage;
  std::string pipeline;
  uint32_t input = 0;
  std::vector<metal::BufferBinding> buffers;
  uint32_t paramsIndex = 0;
  std::vector<uint8_t> params;
  uint64_t items = 0;
};

// The rows a load thread reads at once into a chunk's staging. A chunk is one
// tensor's rows, at most kGgufRepackStagingBytes, so this sets how many reads
// a chunk keeps in flight: 4 MiB reads left the 27B GGUF's small tensors at
// 2-3, and the layers read 4.6-5.0 GB/s against the SSD's 6.6-7. Cold loads
// by read size, two each (M5 Max 32-core, internal SSD; 27B MLX, then 27B
// UD-Q4_K_M GGUF): 4 MiB 4.49, 4.59 / 4.87, 4.69 s; 2 MiB 3.82, 3.86 / 4.25,
// 4.11; 1 MiB 3.57, 3.39 / 3.69, 3.83; 512 KiB 3.59, 3.63 / 3.85, 3.92;
// 256 KiB 4.08, 4.12 / 4.23, 4.24. Staging several chunks together in rounds
// of 64 or 128 MiB with 4 MiB reads gave 4.30, 4.19 and 4.50, 4.21 / 4.74,
// 4.48 and 4.52, 4.46 s, for four times the staging memory.
constexpr uint64_t kChunkReadBytes = 1 << 20;

template <class Params>
std::vector<uint8_t> paramBytes(const Params &params) {
  std::vector<uint8_t> bytes(sizeof(params));
  std::memcpy(bytes.data(), &params, sizeof(params));
  return bytes;
}

// A repack's chunks: threads read a chunk's rows into the staging, and the
// GPU writes their planes in place into image.
void addRepackChunks(metal::MetalBackend &backend, const metal::MetalBuffer &image, const gguf::Repack &repack,
                     const RepackChunk &chunk, std::vector<GpuChunk> &chunks) {
  const QuantFormat &format = kQuantFormats[repack.format];
  const auto offsets = planeOffsets(repack);
  for (uint64_t firstRow = 0; firstRow < repack.rows; firstRow += chunk.rows) {
    const uint64_t rows = std::min(chunk.rows, repack.rows - firstRow);
    for (uint64_t firstColumn = 0; firstColumn < repack.columns; firstColumn += chunk.columns) {
      const uint64_t columns = std::min(chunk.columns, repack.columns - firstColumn);
      const uint64_t chunkRowBytes = ggufRowBytes(format, columns);
      const uint64_t column = ggufRowBytes(format, firstColumn);
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
      // The sources' rows in image order, then zero rows, a task for each
      // run of rows within kChunkReadBytes.
      const auto stage = [&repack, firstRow, rows, column, chunkRowBytes](uint8_t *host) {
        struct Read {
          const gguf::TensorRows *tensor;
          uint64_t first, count;
          uint8_t *to;
        };
        std::vector<Read> reads;
        const uint64_t batch = std::max<uint64_t>(1, kChunkReadBytes / chunkRowBytes);
        uint64_t start = 0;
        for (const gguf::TensorRows &tensor : repack.sources) {
          const uint64_t begin = std::max(firstRow, start), end = std::min(firstRow + rows, start + tensor.rows);
          for (uint64_t row = begin; row < end; row += batch)
            reads.push_back(
                {&tensor, row - start, std::min(batch, end - row), host + (row - firstRow) * chunkRowBytes});
          start += tensor.rows;
        }
        parallelFor(reads.size(), [&](size_t index, unsigned) {
          const Read &read = reads[index];
          readRows(*read.tensor, read.first, read.count, column, chunkRowBytes, read.to);
        });
        if (start < firstRow + rows) {
          const uint64_t zero = std::max(start, firstRow);
          std::memset(host + (zero - firstRow) * chunkRowBytes, 0, (firstRow + rows - zero) * chunkRowBytes);
        }
      };
      // One thread per row and 32-column group, rounded up to whole
      // threadgroups: an odd number of 64-row tiles by an odd number of
      // 64-column spans leaves half a threadgroup, which the kernel bounds.
      chunks.push_back({stage, "gguf_repack", 0, {{1, backend.view(image, at[0], extent)}}, 2, paramBytes(params),
                        rows * (columns / 32)});
    }
  }
}

// The rows of a DequantizeToFloat32 copy that one run decodes: as many as the
// staging holds, whose elements the gather's 32-bit index counts (no format
// holds more than eight in a byte).
uint64_t decodedRun(const gguf::Copy &copy) {
  const gguf::TensorRows &rows = copy.source;
  const uint64_t run = std::min(rows.rows, kGgufRepackStagingBytes / rows.rowBytes);
  if (!run) throw GgufError("weight row exceeds the repack staging: " + rows.name);
  return run;
}

// A DequantizeToFloat32 copy's chunks: runs of its rows as stored staged,
// each gathered on the GPU (kernels/shared/embedding.metal,
// gguf_embed_f32_<format>) into the F32 values of its format's decode, the
// run's rows by their ids in `ids` (0, 1, ...).
void addDecodedChunks(metal::MetalBackend &backend, const metal::MetalBuffer &image, const gguf::Copy &copy,
                      const metal::MetalBuffer &ids, std::vector<GpuChunk> &chunks) {
  const gguf::TensorRows &rows = copy.source;
  const uint64_t columns = gguf::decodedColumns(rows), run = decodedRun(copy);
  const QuantFormat &format = kQuantFormats[gguf_format_of(rows.type)];
  for (uint64_t first = 0; first < rows.rows; first += run) {
    const uint64_t count = std::min(run, rows.rows - first);
    const GgufEmbedParams params{static_cast<uint32_t>(count), static_cast<uint32_t>(count),
                                 static_cast<uint32_t>(columns)};
    const metal::MetalBuffer output = backend.view(image, copy.destination + first * columns * sizeof(float),
                                                   count * columns * sizeof(float));
    // A task for each run of rows within kChunkReadBytes.
    const auto stage = [&rows, first, count](uint8_t *host) {
      const uint64_t batch = std::max<uint64_t>(1, kChunkReadBytes / rows.rowBytes);
      parallelFor((count + batch - 1) / batch, [&](size_t index, unsigned) {
        const uint64_t row = index * batch;
        readRows(rows, first + row, std::min(batch, count - row), 0, rows.rowBytes, host + row * rows.rowBytes);
      });
    };
    chunks.push_back({stage, std::string("gguf_embed_f32_") + format.name, 1, {{0, ids}, {2, output}}, 3,
                      paramBytes(params), count * columns});
  }
}

} // namespace

bool exactlyMxfp8(const WeightSource &file, const GgufTensor &tensor) {
  if (tensor.type != ggml::kBF16 && tensor.type != ggml::kF16)
    throw GgufError("not a 16-bit float tensor: " + tensor.name);
  // A start plans its target several times, its sizing's (model/ModelSizing.hpp) and its load's, each from a source
  // of its own on the same unchanged file (WeightSource::checkUnchanged). So the cache is the process's, a file known
  // by its identity whatever source holds it, and each tensor's values are read once, the first time a plan asks.
  static std::mutex mutex;
  static std::map<std::pair<std::array<int64_t, 5>, uint64_t>, bool> known;
  const std::pair key{file.identity(), tensor.offset};
  {
    const std::lock_guard lock(mutex);
    if (const auto found = known.find(key); found != known.end()) return found->second;
  }
  constexpr uint64_t kBlock = 32, kChunkValues = kChunkReadBytes / 2;
  const uint64_t values = tensor.elements();
  if (values % kBlock || tensor.bytes != values * 2) return false;
  const bool float16 = tensor.type == ggml::kF16;
  std::atomic<bool> exact = true;
  parallelFor((values + kChunkValues - 1) / kChunkValues, [&](size_t chunk, unsigned) {
    if (!exact.load(std::memory_order_relaxed)) return;
    const uint64_t first = chunk * kChunkValues, count = std::min(kChunkValues, values - first);
    std::vector<uint8_t> bytes(count * 2);
    file.readData(tensor.offset + first * 2, bytes);
    std::array<float, kBlock> block;
    std::array<uint8_t, kBlock + 1> encoded;
    for (uint64_t at = 0; at < count && exact.load(std::memory_order_relaxed); at += kBlock) {
      for (uint64_t i = 0; i < kBlock; ++i) {
        uint16_t bits;
        std::memcpy(&bits, bytes.data() + 2 * (at + i), 2);
        block[i] = widenFloat16(bits, float16);
      }
      if (!gguf::mxfp8Block(block, encoded)) exact.store(false, std::memory_order_relaxed);
    }
  });
  const std::lock_guard lock(mutex);
  known.emplace(key, exact.load());
  return exact;
}

void readRepackRows(const gguf::Repack &repack, uint64_t first, uint64_t count, uint8_t *to) {
  uint64_t sourceRows = 0;
  for (const gguf::TensorRows &tensor : repack.sources) sourceRows += tensor.rows;
  if (first > sourceRows || count > sourceRows - first) throw GgufError("repack rows are out of bounds");
  const uint64_t rowBytes = ggufRowBytes(kQuantFormats[repack.format], repack.columns);
  uint64_t start = 0;
  for (const gguf::TensorRows &tensor : repack.sources) {
    const uint64_t begin = std::max(first, start), end = std::min(first + count, start + tensor.rows);
    if (begin < end)
      readRows(tensor, begin - start, end - begin, 0, rowBytes, to + (begin - first) * rowBytes);
    start += tensor.rows;
  }
}

std::optional<FileRows> storedRepackRows(const gguf::Repack &repack, uint64_t first, uint64_t count) {
  const uint64_t rowBytes = ggufRowBytes(kQuantFormats[repack.format], repack.columns);
  uint64_t start = 0;
  for (const gguf::TensorRows &tensor : repack.sources) {
    if (first >= start && first - start < tensor.rows) {
      const uint64_t row = first - start;
      // A permutation (RowOrder) leaves single rows where they are.
      if (count > tensor.rows - row || rowReader(tensor) != RowReader::Stored || tensor.rowBytes != rowBytes ||
          !tensor.file || (count > 1 && tensor.order.from < row + count))
        return std::nullopt;
      return FileRows{tensor.file, tensor.file->dataOffset() + tensor.offset + sourceRow(tensor, row) * rowBytes};
    }
    start += tensor.rows;
  }
  throw GgufError("repack rows are out of bounds");
}

std::optional<std::array<PlaneRows, 3>> storedPlanes(const gguf::Repack &repack) {
  if (repack.sources.size() != 1) return std::nullopt;
  const gguf::TensorRows &rows = repack.sources.front();
  const uint32_t id = gguf_format_of(rows.type);
  if (rowReader(rows) != RowReader::Safetensors ||
      !(quant_affine_format(id) || id == GGUF_FMT_MXFP4 || id == GGUF_FMT_MXFP8) ||
      rows.order.from != UINT64_MAX ||
      rows.windows != 1 || rows.rows != repack.rows ||
      rows.rowBytes != ggufRowBytes(kQuantFormats[repack.format], repack.columns))
    return std::nullopt;
  const QuantFormat &format = nativeFormat(rows);
  const StoredBlock block = storedBlock(id, format);
  const uint64_t blocks = rows.rowBytes / format.block_bytes;
  // Plane rows of `bytes` a native block, row 0 the tensor's row rows.first.
  const auto plane = [&](const SourceTensor *tensor, uint64_t bytes) {
    if (!tensor || !bytes) return PlaneRows{};
    return PlaneRows{tensor->file, tensor->file->dataOffset() + tensor->offset + rows.first * blocks * bytes,
                     blocks * bytes};
  };
  const gguf::SafetensorsSource &source = rows.safetensors;
  return std::array{plane(source.codes, block.codes), plane(source.scales, block.scales),
                    plane(source.biases, quant_affine_format(id) ? block.scales : 0)};
}

void writeGgufImage(metal::MetalBackend &backend, const metal::MetalBuffer &image, const gguf::Image &plan) {
  const std::span<uint8_t> destination = contentsOf(image);
  if (destination.size() != plan.bytes) throw GgufError("GGUF image size differs from its plan");
  std::vector<std::pair<uint64_t, uint64_t>> extents;
  for (const gguf::Fill &fill : plan.fills) extents.emplace_back(fill.offset, fill.bytes.size());
  // Copies the CPU converts read whole 4-byte values; the GPU decodes the
  // others from their rows as stored.
  std::vector<const gguf::Copy *> decoded;
  uint64_t staging = 0, ids = 0;
  for (const gguf::Copy &copy : plan.copies) {
    const bool decodes = copy.conversion == gguf::Conversion::DequantizeToFloat32;
    if (!copy.source.rows || !copy.source.rowBytes || copy.source.windows != 1 ||
        (copy.conversion != gguf::Conversion::None && !decodes && copy.source.rowBytes % 4))
      throw GgufError("invalid prepared weight copy");
    extents.emplace_back(copy.destination, copyBytes(copy));
    if (!decodes) continue;
    decoded.push_back(&copy);
    staging = std::max(staging, decodedRun(copy) * copy.source.rowBytes);
    ids = std::max(ids, decodedRun(copy));
  }
  std::vector<RepackChunk> chunks;
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
  std::vector<std::function<void()>> tasks;
  for (const gguf::Copy &copy : plan.copies)
    if (copy.conversion != gguf::Conversion::DequantizeToFloat32) addCopyTasks(destination.data(), copy, tasks);
  parallelFor(tasks.size(), [&](size_t index, unsigned) { tasks[index](); });
  if (plan.repacks.empty() && decoded.empty()) return;
  metal::MetalBuffer rowIds;
  if (!decoded.empty()) {
    rowIds = backend.allocateBuffer(ids * sizeof(uint32_t), metal::BufferStorage::Shared, "load/row-ids");
    auto *id = static_cast<uint32_t *>(rowIds.contents());
    for (uint32_t row = 0; row < ids; ++row) id[row] = row;
  }
  std::vector<GpuChunk> gpuChunks;
  for (size_t i = 0; i < plan.repacks.size(); ++i)
    addRepackChunks(backend, image, plan.repacks[i], chunks[i], gpuChunks);
  for (const gguf::Copy *copy : decoded) addDecodedChunks(backend, image, *copy, rowIds, gpuChunks);
  // Two staging buffers serve every repack and decoded copy of the image,
  // one while a single chunk has nothing to overlap.
  std::array<metal::MetalBuffer, 2> inputs;
  for (size_t i = 0; i < std::min<size_t>(2, gpuChunks.size()); ++i)
    inputs[i] = backend.allocateBuffer(staging, metal::BufferStorage::Shared, "load/rows");
  pipelineChunks(
      gpuChunks.size(),
      [&](size_t index, unsigned buffer) {
        gpuChunks[index].stage(static_cast<uint8_t *>(inputs[buffer].contents()));
      },
      [&](size_t index, unsigned buffer) {
        const GpuChunk &chunk = gpuChunks[index];
        std::vector<metal::BufferBinding> buffers = chunk.buffers;
        buffers.push_back({chunk.input, inputs[buffer]});
        const metal::ComputeDispatch dispatch{chunk.pipeline,
                                              std::move(buffers),
                                              {{chunk.paramsIndex, chunk.params.data(), chunk.params.size()}},
                                              {(chunk.items + 255) / 256, 1, 1},
                                              {256, 1, 1}};
        return backend.submitCommandAsync(std::span<const metal::ComputeDispatch>(&dispatch, 1));
      });
}

WeightFile loadImage(metal::MetalBackend &backend, WeightImages &images, std::string_view prefix,
                     std::shared_ptr<const gguf::Image> plan, std::function<void()> checkUnchanged) {
  ImagePlan image{std::string(prefix) + plan->name, plan->magic, plan->layer, plan->type, plan->bytes,
                  [&backend, plan, checkUnchanged = std::move(checkUnchanged)](std::span<uint8_t>,
                                                                               const metal::MetalBuffer &buffer) {
                    writeGgufImage(backend, buffer, *plan);
                    checkUnchanged();
                  },
                  plan->residentBytes, std::shared_ptr<const std::vector<gguf::Fill>>(plan, &plan->fills)};
  return images.load(std::move(image));
}

WeightFileRecord imageRecord(const WeightImages &images, std::string_view prefix, const gguf::Image &plan) {
  return {std::string(prefix) + plan.name, plan.magic, plan.layer, plan.type, plan.residentBytes,
          images.contentIdentity()};
}

} // namespace splash::model
