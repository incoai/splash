// Editing this file re-prepares every affine model. DFlash2 drafts are affine models too.
#include "WeightPreparationIdentity.hpp"
#include "model/AffinePreparation.hpp"
#include "model/WeightLayout.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace splash::model::affine {
namespace {

// Input and output staging: each half of the preparation bound.
constexpr uint64_t kChunkBytes = kWeightPreparationStagingBytes / 2;

// Bytes per row of a 64-column group: codes, scales, biases.
std::array<uint32_t, 3> groupBytes(const Section &section) { return {8 * section.bits, 2, 2}; }

// Bytes per row of a 64-column group of a BF16 weight.
constexpr uint32_t kBfloat16GroupBytes = kQ4GroupElements * kBFloat16Bytes;

// Groups of one field a 256-row tile step converts.
uint32_t chunkGroups(const Section &section, uint32_t unit) {
  return static_cast<uint32_t>(std::min<uint64_t>(section.columns / kQ4GroupElements, kChunkBytes / (kQ4StorageN * unit)));
}

// Gathers into input rows [firstRow, firstRow + 256) of groups [firstGroup,
// firstGroup + count) of one field of one expert, unit bytes per row of a
// group: each part's rows as stored, rows past the parts zero. A step of
// whole rows reads a part's rows at once, as they are contiguous in the
// source.
void gatherRows(const Section &section, uint32_t expert, size_t field, uint32_t unit, uint32_t firstRow,
                uint32_t firstGroup, uint32_t count, std::span<uint8_t> input) {
  const uint32_t groups = section.columns / kQ4GroupElements;
  const uint64_t rowBytes = uint64_t(count) * unit;
  const uint64_t sourceRowBytes = uint64_t(groups) * unit;
  uint32_t partStart = 0;
  for (const auto &part : section.parts) {
    const uint32_t begin = std::max(firstRow, partStart);
    const uint32_t end = std::min(firstRow + kQ4StorageN, partStart + part.rows);
    if (begin < end) {
      const uint64_t offset = (uint64_t(expert) * part.rows + begin - partStart) * sourceRowBytes +
                              uint64_t(firstGroup) * unit;
      uint8_t *to = input.data() + (begin - firstRow) * rowBytes;
      if (count == groups) {
        part.fields[field].tensor->read(offset, {to, (end - begin) * rowBytes});
      } else {
        for (uint32_t row = 0; row < end - begin; ++row)
          part.fields[field].tensor->read(offset + row * sourceRowBytes, {to + row * rowBytes, rowBytes});
      }
    }
    partStart += part.rows;
  }
  if (firstRow + kQ4StorageN > partStart) {
    const uint32_t gathered = std::max(firstRow, partStart) - firstRow;
    std::fill(input.begin() + gathered * rowBytes, input.begin() + kQ4StorageN * rowBytes, 0);
  }
}

// Transposes a gathered tile of 256 rows of count groups into count groups of
// 256 rows, unit bytes each.
void tile(std::span<const uint8_t> input, std::span<uint8_t> output, uint32_t count, uint32_t unit) {
  const uint64_t rowBytes = uint64_t(count) * unit;
  for (uint32_t group = 0; group < count; ++group)
    for (uint32_t row = 0; row < kQ4StorageN; ++row)
      std::memcpy(output.data() + (uint64_t(group) * kQ4StorageN + row) * unit,
                  input.data() + row * rowBytes + uint64_t(group) * unit, unit);
}

// Reorders each field of each expert into [rows / 256][groups][256] tiles of
// its group bytes; rows past the parts are zero.
void writeProjection(int destination, const Section &section, std::vector<uint8_t> &input,
                     std::vector<uint8_t> &output, const PreparationCheck &admit) {
  const uint32_t groups = section.columns / kQ4GroupElements;
  const auto unit = groupBytes(section);
  const uint64_t expertBytes = section.bytes / section.experts;
  for (uint32_t expert = 0; expert < section.experts; ++expert) {
    uint64_t fieldBase = section.offset + expert * expertBytes;
    for (size_t field = 0; field < unit.size(); ++field) {
      const uint32_t stepGroups = chunkGroups(section, unit[field]);
      for (uint32_t firstRow = 0; firstRow < section.rows; firstRow += kQ4StorageN) {
        for (uint32_t firstGroup = 0; firstGroup < groups; firstGroup += stepGroups) {
          admit();
          const uint32_t count = std::min(stepGroups, groups - firstGroup);
          gatherRows(section, expert, field, unit[field], firstRow, firstGroup, count, input);
          tile(input, output, count, unit[field]);
          const uint64_t offset = (uint64_t(firstRow / kQ4StorageN) * groups + firstGroup) * kQ4StorageN * unit[field];
          writeWeightBytes(destination, fieldBase + offset,
                           std::span(output).first(uint64_t(kQ4StorageN) * count * unit[field]));
        }
      }
      fieldBase += uint64_t(section.rows) * groups * unit[field];
    }
  }
}

float bfloat16Value(const uint8_t *bytes) {
  uint16_t bits;
  std::memcpy(&bits, bytes, sizeof bits);
  return std::bit_cast<float>(uint32_t(bits) << 16);
}

// The BF16 nearest a finite value, ties to even.
uint16_t nearestBfloat16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

// A group of 64 BF16 weights as MLX's affine quantization rounds it to 4 bits
// (mlx.core.quantize, its Metal kernel), in float: the range runs from the
// minimum to the maximum or 0, whichever is greater, the end of the range
// farther from zero is the bias, the scale is adjusted so that 0 falls on a
// code unless that code is 0, each code is round((w - bias) / scale), halves
// away from zero, clamped to [0, 15], and scale and bias are stored as BF16.
void quantizeGroup(const uint8_t *weights, uint8_t *codes, uint16_t &scale, uint16_t &bias) {
  std::array<float, kQ4GroupElements> w;
  for (size_t i = 0; i < w.size(); ++i) {
    w[i] = bfloat16Value(weights + i * kBFloat16Bytes);
    if (!std::isfinite(w[i])) throw std::runtime_error("non-finite weight in a BF16 projection");
  }
  const float minimum = *std::min_element(w.begin(), w.end());
  const float maximum = std::max(0.0F, *std::max_element(w.begin(), w.end()));
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
  const auto code = [&](float value) {
    return static_cast<uint8_t>(std::clamp(std::round((value - offset) / step), 0.0F, 15.0F));
  };
  for (size_t i = 0; i < w.size(); i += 2) codes[i / 2] = static_cast<uint8_t>(code(w[i]) | code(w[i + 1]) << 4);
  scale = nearestBfloat16(step);
  bias = nearestBfloat16(offset);
}

// Quantizes each step of 256 rows and count groups of a BF16 projection and
// writes its codes, scales and biases as writeProjection writes them: each
// field in [rows / 256][groups][256] tiles of its group bytes.
void writeQuantized(int destination, const Section &section, std::vector<uint8_t> &input,
                    std::vector<uint8_t> &output, const PreparationCheck &admit) {
  const uint32_t groups = section.columns / kQ4GroupElements;
  const auto unit = groupBytes(section);
  const uint32_t stepGroups = chunkGroups(section, kBfloat16GroupBytes);
  std::array<uint64_t, 3> fieldBase{section.offset};
  for (size_t field = 1; field < unit.size(); ++field)
    fieldBase[field] = fieldBase[field - 1] + uint64_t(section.rows) * groups * unit[field - 1];
  for (uint32_t firstRow = 0; firstRow < section.rows; firstRow += kQ4StorageN) {
    for (uint32_t firstGroup = 0; firstGroup < groups; firstGroup += stepGroups) {
      admit();
      const uint32_t count = std::min(stepGroups, groups - firstGroup);
      gatherRows(section, 0, 0, kBfloat16GroupBytes, firstRow, firstGroup, count, input);
      // The step's fields, one after another, each in [count][256] tile order.
      std::array<uint8_t *, 3> fields{output.data()};
      for (size_t field = 1; field < unit.size(); ++field)
        fields[field] = fields[field - 1] + uint64_t(count) * kQ4StorageN * unit[field - 1];
      for (uint32_t row = 0; row < kQ4StorageN; ++row) {
        for (uint32_t group = 0; group < count; ++group) {
          const uint64_t at = uint64_t(group) * kQ4StorageN + row;
          uint16_t scale, bias;
          quantizeGroup(input.data() + (uint64_t(row) * count + group) * kBfloat16GroupBytes,
                        fields[0] + at * unit[0], scale, bias);
          std::memcpy(fields[1] + at * unit[1], &scale, sizeof scale);
          std::memcpy(fields[2] + at * unit[2], &bias, sizeof bias);
        }
      }
      const uint64_t tile = (uint64_t(firstRow / kQ4StorageN) * groups + firstGroup) * kQ4StorageN;
      for (size_t field = 0; field < unit.size(); ++field)
        writeWeightBytes(destination, fieldBase[field] + tile * unit[field],
                         {fields[field], uint64_t(count) * kQ4StorageN * unit[field]});
    }
  }
}

// float(-exp(double(A_log))) of a BF16 or F32 vector. The A_log it reads and
// the decay it writes are staged together, within the staging bound of every
// conversion step.
void writeDecay(int destination, const Section &section) {
  const SourceTensor &tensor = *section.input.tensor;
  if (tensor.bytes + section.bytes > kWeightPreparationStagingBytes)
    throw std::runtime_error("decay tensor exceeds the preparation staging bound");
  std::vector<uint8_t> bytes(tensor.bytes);
  tensor.read(0, bytes);
  std::vector<float> values(section.bytes / sizeof(float));
  for (size_t i = 0; i < values.size(); ++i) {
    float logarithm;
    if (tensor.dtype == "BF16") logarithm = bfloat16Value(bytes.data() + i * kBFloat16Bytes);
    else std::memcpy(&logarithm, bytes.data() + i * 4, 4);
    values[i] = static_cast<float>(-std::exp(static_cast<double>(logarithm)));
    if (!std::isfinite(values[i])) throw std::runtime_error("non-finite GDN decay");
  }
  writeWeightBytes(destination, section.offset, {reinterpret_cast<const uint8_t *>(values.data()), section.bytes});
}

} // namespace

// The key: this code's identity, the plan and the bytes, dtype and shape of
// every tensor read. config.json is only validated against the layout, whose
// dimensions and quantization the plan records; nothing else in it changes
// these bytes.
PreparedWeight affineImageWeight(const Image &image, std::string_view directory, const std::string &source) {
  WeightIdentity identity("splash-affine-preparation-v2 " SPLASH_AFFINE_PREPARATION_ID);
  identity.record("image", image.magic, image.layer, image.type, image.bytes);
  for (const Section &section : image.sections) {
    identity.record("section", int(section.kind), section.offset, section.bytes, section.rows, section.columns,
                    section.experts, section.bits);
    if (section.parts.empty()) section.input.tensor->identify(identity);
    for (const ProjectionPart &part : section.parts) {
      identity.record("part", part.rows);
      for (const Input &field : part.fields) field.tensor->identify(identity);
    }
  }
  return identity.weight(image.bytes, std::string(directory) + "/" + image.name, source);
}

void writeAffineImage(int destination, const Image &image, const PreparationCheck &admit) {
  // One input and one output buffer serve every section.
  uint64_t inputBytes = 0, outputBytes = 0;
  for (const Section &section : image.sections) {
    switch (section.kind) {
    case SectionKind::Projection:
      for (uint32_t unit : groupBytes(section)) {
        const uint64_t bytes = uint64_t(kQ4StorageN) * chunkGroups(section, unit) * unit;
        inputBytes = std::max(inputBytes, bytes);
        outputBytes = std::max(outputBytes, bytes);
      }
      break;
    case SectionKind::Quantize: {
      const uint64_t rows = uint64_t(kQ4StorageN) * chunkGroups(section, kBfloat16GroupBytes);
      const auto unit = groupBytes(section);
      inputBytes = std::max(inputBytes, rows * kBfloat16GroupBytes);
      outputBytes = std::max(outputBytes, rows * (unit[0] + unit[1] + unit[2]));
      break;
    }
    case SectionKind::Copy:
      inputBytes = std::max(inputBytes, std::min(section.bytes, kChunkBytes));
      break;
    case SectionKind::Decay:
      break;
    }
  }
  std::vector<uint8_t> input(inputBytes), output(outputBytes);
  writeWeightBytes(destination, 0, weightFileHeader(image.magic, image.layer, image.type));
  for (const Section &section : image.sections) {
    admit();
    switch (section.kind) {
    case SectionKind::Projection:
      writeProjection(destination, section, input, output, admit);
      break;
    case SectionKind::Quantize:
      writeQuantized(destination, section, input, output, admit);
      break;
    case SectionKind::Decay:
      writeDecay(destination, section);
      break;
    case SectionKind::Copy:
      section.input.tensor->copy(destination, section.offset, input, admit);
      break;
    }
  }
}

} // namespace splash::model::affine
