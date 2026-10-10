#include "model/GgufPreparation.hpp"

#include "metal/abi/GgufRepack.h"
#include "metal/abi/QuantTables.h"
#include "model/Bfloat16.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/WeightImages.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
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

// How readRows reads a tensor's rows: as stored, or as its format's native
// rows, which it builds from a quantized safetensors tensor's tensors, from
// llama.cpp's NVFP4 rows, or by quantizing a BF16 weight.
enum class RowReader : uint8_t { Stored, Safetensors, GgufNvfp4, QuantizedBfloat16 };
RowReader rowReader(const gguf::TensorRows &rows) {
  if (rows.safetensors.codes) return RowReader::Safetensors;
  if (rows.ggufNvfp4RowBytes) return RowReader::GgufNvfp4;
  if (rows.safetensors.bfloat16) return RowReader::QuantizedBfloat16;
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

// Native bytes [column, column + span) of `count` source rows of a quantized
// safetensors tensor from source row `start` on, back to back, read from its
// tensors: per native block an affine tensor's scale, bias and codes; an
// mxfp4 tensor's block_mxfp4, its exponent and its codes with element j in
// the low nibble of byte j % 16 when j < 16, else in its high nibble (MLX
// packs element j at bits 4 j of the row); an nvfp4 tensor's 16 E4M3 scales,
// its row's tensor scale and its codes as stored; an fp8 tensor's row's tensor
// scale and its E4M3 values (metal/abi/QuantFormat.h).
void readSafetensorsRows(const gguf::TensorRows &rows, uint64_t start, uint64_t count, uint64_t column, uint64_t span,
                 uint8_t *to) {
  const QuantFormat &format = nativeFormat(rows);
  const uint32_t id = gguf_format_of(rows.type);
  const bool affine = quant_affine_format(id);
  // Bytes of a native block from the scales tensor (its biases take as many
  // as its scales) and from the codes tensor.
  const uint64_t block = format.block_bytes, groups = rows.rowBytes / block;
  const uint64_t scaleBytes = affine ? 2 : id == GGUF_FMT_NVFP4 ? QUANT_NVFP4_G : id == GGUF_FMT_FP8 ? 0 : 1;
  const uint64_t codes =
      block - (affine ? 4 : id == GGUF_FMT_NVFP4 ? QUANT_NVFP4_CODES : id == GGUF_FMT_FP8 ? QUANT_FP8_VALUES : 1);
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

// Native NVFP4 bytes [column, column + span) of `count` rows of llama.cpp's
// NVFP4 from source row `start` on, back to back: per 256 elements the 16
// UE4M3 scales of four block_nvfp4 as the E4M3 scales they are (llama.cpp
// reads 0x7F as 0 and ignores bit 7), the row's tensor scale, and the codes with
// element e in the low nibble of byte e / 2 when e is even, else in its high
// one (metal/abi/QuantFormat.h), where a block_nvfp4 holds element j of each
// 16 in the low nibble of byte j and element j + 8 in its high one.
void readGgufNvfp4Rows(const gguf::TensorRows &rows, uint64_t start, uint64_t count, uint64_t column, uint64_t span,
                       uint8_t *to) {
  const QuantFormat &format = kQuantFormats[GGUF_FMT_NVFP4];
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
          uint8_t *pairs = out + QUANT_NVFP4_CODES + 32 * q + 8 * s;
          for (uint64_t i = 0; i < 4; ++i) {
            pairs[i] = uint8_t((codes[2 * i] & 15) | (codes[2 * i + 1] & 15) << 4);
            pairs[4 + i] = uint8_t(codes[2 * i] >> 4 | (codes[2 * i + 1] & 0xF0));
          }
        }
      }
      std::memcpy(out + QUANT_NVFP4_G, &g[row + b / nativeBlocks], 4);
    }
  }
}

// Image rows [first, first + count) of `rows`, bytes [column, column + span)
// of each, back to back. Consecutive source rows are read together.
void readRows(const gguf::TensorRows &rows, uint64_t first, uint64_t count, uint64_t column, uint64_t span,
              uint8_t *to) {
  if (first > rows.rows || count > rows.rows - first || column > rows.rowBytes || span > rows.rowBytes - column)
    throw GgufError("prepared weight rows are out of bounds");
  for (uint64_t row = 0; row < count;) {
    const uint64_t start = sourceRow(rows, first + row);
    uint64_t run = 1;
    if (span == rows.rowBytes)
      while (row + run < count && sourceRow(rows, first + row + run) == start + run) ++run;
    uint8_t *out = to + row * span;
    switch (rowReader(rows)) {
    case RowReader::Stored: rows.file->readData(rows.offset + start * rows.rowBytes + column, {out, run * span}); break;
    case RowReader::Safetensors: readSafetensorsRows(rows, start, run, column, span, out); break;
    case RowReader::GgufNvfp4: readGgufNvfp4Rows(rows, start, run, column, span, out); break;
    case RowReader::QuantizedBfloat16: quantizeRows(rows, start, run, column, span, out); break;
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

// Writes the F32 values `count` BF16 values equal.
void widenToFloat32(const uint8_t *values, uint64_t count, uint8_t *to) {
  for (uint64_t i = 0; i < count; ++i) {
    uint16_t value;
    std::memcpy(&value, values + 2 * i, 2);
    const uint32_t widened = uint32_t{value} << 16;
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

// E4M3 byte b as the kernels read it (quant_e4m3_pair): 2^-8 times its value.
float e4m3Scaled(uint8_t b) {
  const int exponent = (b >> 3) & 15, mantissa = b & 7;
  const float magnitude = exponent ? std::ldexp(float(8 + mantissa), exponent - 18) : std::ldexp(float(mantissa), -17);
  return b & 0x80 ? -magnitude : magnitude;
}

// Writes the F32 values, as the kernels compute them, of `count` native
// blocks of format `id`: s * code + z (affine), kFP4Values[code] (twice the
// E2M1 value) times 2^(e - 128) (mxfp4) or times the 16-group's (128 g) *
// e4m3 / 2^8 (nvfp4), (256 g) times e4m3 / 2^8 (fp8), or d * q, exact in F32
// (GGUF Q8_0).
void dequantize(const uint8_t *blocks, uint64_t count, uint32_t id, uint8_t *to) {
  const QuantFormat &format = kQuantFormats[id];
  if (id == GGUF_FMT_Q80) {
    for (uint64_t block = 0; block < count; ++block) {
      const uint8_t *in = blocks + block * format.block_bytes;
      uint16_t d;
      std::memcpy(&d, in, 2);
      const float scale = static_cast<float>(std::bit_cast<_Float16>(d));
      for (uint32_t l = 0; l < format.block_elements; ++l) {
        const float value = scale * float(static_cast<int8_t>(in[2 + l]));
        std::memcpy(to + 4 * (block * format.block_elements + l), &value, 4);
      }
    }
    return;
  }
  if (id == GGUF_FMT_NVFP4 || id == GGUF_FMT_FP8) {
    for (uint64_t block = 0; block < count; ++block) {
      const uint8_t *in = blocks + block * format.block_bytes;
      float g;
      std::memcpy(&g, in + (id == GGUF_FMT_NVFP4 ? QUANT_NVFP4_G : 0), 4);
      for (uint32_t l = 0; l < format.block_elements; ++l) {
        const float value = id == GGUF_FMT_NVFP4
                                ? float(kFP4Values[(in[QUANT_NVFP4_CODES + l / 2] >> (4 * (l % 2))) & 15]) *
                                      (e4m3Scaled(in[l / 16]) * (128.0F * g))
                                : e4m3Scaled(in[QUANT_FP8_VALUES + l]) * (256.0F * g);
        std::memcpy(to + 4 * (block * format.block_elements + l), &value, 4);
      }
    }
    return;
  }
  if (id == GGUF_FMT_MXFP4) {
    for (uint64_t block = 0; block < count; ++block) {
      const uint8_t *in = blocks + block * format.block_bytes;
      const float scale = std::ldexp(1.0f, int{in[0]} - 128);
      for (uint32_t l = 0; l < format.block_elements; ++l) {
        const float value = float(kFP4Values[(in[1 + l % 16] >> (l < 16 ? 0 : 4)) & 15]) * scale;
        std::memcpy(to + 4 * (block * format.block_elements + l), &value, 4);
      }
    }
    return;
  }
  const uint32_t bits = quant_affine_bits(id), mask = (1u << bits) - 1;
  for (uint64_t block = 0; block < count; ++block) {
    const uint8_t *in = blocks + block * format.block_bytes;
    uint16_t s, z;
    std::memcpy(&s, in, 2);
    std::memcpy(&z, in + 2, 2);
    for (uint32_t l = 0; l < format.block_elements; ++l) {
      const uint32_t at = l * bits, shift = at % 8;
      uint32_t word = in[4 + at / 8];
      if (shift + bits > 8) word |= uint32_t{in[4 + at / 8 + 1]} << 8;
      const float value = std::fma(static_cast<float>((word >> shift) & mask), widenBfloat16(s), widenBfloat16(z));
      std::memcpy(to + 4 * (block * format.block_elements + l), &value, 4);
    }
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
// blocks of rows a reader builds, whole blocks of a GGUF tensor decoded to
// F32) of one row: a copy as stored reads in place, a converted one through
// its thread's staging.
void addCopyTasks(uint8_t *image, const gguf::Copy &copy,
                  std::vector<std::function<void(std::vector<uint8_t> &)>> &tasks) {
  const gguf::TensorRows &rows = copy.source;
  const uint64_t unit = rowReader(rows) != RowReader::Stored ? nativeFormat(rows).block_bytes
                        : copy.conversion == gguf::Conversion::DequantizeToFloat32
                            ? kQuantFormats[gguf_format_of(rows.type)].block_bytes
                            : 4;
  const uint64_t span = std::min<uint64_t>(rows.rowBytes, kLoadStepBytes / unit * unit);
  const uint64_t batch = span == rows.rowBytes ? kLoadStepBytes / rows.rowBytes : 1;
  for (uint64_t first = 0; first < rows.rows; first += batch) {
    const uint64_t count = std::min(batch, rows.rows - first);
    for (uint64_t column = 0; column < rows.rowBytes; column += span) {
      const uint64_t width = std::min(span, rows.rowBytes - column);
      uint8_t *to = image + copy.destination + copiedBytes(copy, first * rows.rowBytes + column);
      tasks.push_back([&copy, first, count, column, width, to](std::vector<uint8_t> &staging) {
        const gguf::TensorRows &rows = copy.source;
        if (copy.conversion == gguf::Conversion::None) {
          readRows(rows, first, count, column, width, to);
          return;
        }
        if (staging.size() < count * width) staging.resize(count * width);
        readRows(rows, first, count, column, width, staging.data());
        switch (copy.conversion) {
        case gguf::Conversion::NarrowToBfloat16:
          narrowToBfloat16(staging.data(), count * width / 4, to, rows.name);
          break;
        case gguf::Conversion::WidenToFloat32: widenToFloat32(staging.data(), count * width / 2, to); break;
        case gguf::Conversion::CenteredNorm: centeredNorm(staging.data(), count * width / 2, to); break;
        case gguf::Conversion::Decay: {
          const bool bfloat16 = rows.type == ggml::kBF16;
          writeGdnDecay(staging.data(), count * width / (bfloat16 ? 2 : 4), bfloat16, to);
          break;
        }
        case gguf::Conversion::DequantizeToFloat32:
          dequantize(staging.data(), count * width / kQuantFormats[gguf_format_of(rows.type)].block_bytes,
                     gguf_format_of(rows.type), to);
          break;
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
void writeRepack(metal::MetalBackend &backend, const metal::MetalBuffer &image, const gguf::Repack &repack,
                 const RepackChunk &chunk, const metal::MetalBuffer &input) {
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
        readRows(*read.tensor, read.first, read.count, column, chunkRowBytes, read.to);
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

void writeGgufImage(metal::MetalBackend &backend, const metal::MetalBuffer &image, const gguf::Image &plan) {
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
  for (const gguf::Copy &copy : plan.copies) addCopyTasks(destination.data(), copy, tasks);
  std::vector<std::vector<uint8_t>> copyStaging(loadThreads());
  parallelFor(tasks.size(), [&](size_t index, unsigned thread) { tasks[index](copyStaging[thread]); });
  // One staging buffer serves every repack of the image.
  if (!plan.repacks.empty()) {
    const auto input = backend.allocateBuffer(staging, metal::BufferStorage::Shared, "load/rows");
    for (size_t i = 0; i < plan.repacks.size(); ++i)
      writeRepack(backend, image, plan.repacks[i], chunks[i], input);
  }
}

} // namespace splash::model
