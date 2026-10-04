#include "model/AffinePreparation.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>

namespace splash::model::affine {
namespace {

// Bytes per row of a 64-column group: codes, scales, biases.
std::array<uint32_t, 3> groupBytes(const Section &section) { return {8 * section.bits, 2, 2}; }

// Bytes per row of a 64-column group of a BF16 weight.
constexpr uint32_t kBfloat16GroupBytes = kQ4GroupElements * kBFloat16Bytes;

// Rows [firstRow, firstRow + rows) and groups [firstGroup, firstGroup +
// groups) of one field of experts [firstExpert, firstExpert + experts),
// which one task converts.
struct Step {
  uint32_t firstExpert = 0, experts = 1, field = 0;
  uint32_t firstRow = 0, rows = 0;
  uint32_t firstGroup = 0, groups = 0;
};

// The steps of one field, unit bytes per row of a group, each within
// kLoadStepBytes: whole experts at once, when their rows are those of a single
// part and fit, or per expert as many whole 256-row tiles as fit, or a tile
// in pieces of its groups.
std::vector<Step> steps(const Section &section, uint32_t field, uint32_t unit) {
  const uint32_t groups = section.columns / kQ4GroupElements;
  const uint64_t expertBytes = uint64_t(section.rows) * groups * unit;
  std::vector<Step> result;
  if (section.experts > 1 && section.parts.size() == 1 && section.parts.front().rows == section.rows &&
      expertBytes <= kLoadStepBytes) {
    const auto experts = static_cast<uint32_t>(kLoadStepBytes / expertBytes);
    for (uint32_t first = 0; first < section.experts; first += experts)
      result.push_back({first, std::min(experts, section.experts - first), field, 0, section.rows, 0, groups});
    return result;
  }
  const uint64_t tileBytes = uint64_t(kQ4StorageN) * groups * unit;
  const uint32_t rows = kQ4StorageN * static_cast<uint32_t>(std::max<uint64_t>(1, kLoadStepBytes / tileBytes));
  const uint32_t stepGroups = tileBytes <= kLoadStepBytes
      ? groups
      : static_cast<uint32_t>(std::max<uint64_t>(1, kLoadStepBytes / (uint64_t(kQ4StorageN) * unit)));
  for (uint32_t expert = 0; expert < section.experts; ++expert)
    for (uint32_t firstRow = 0; firstRow < section.rows; firstRow += rows)
      for (uint32_t firstGroup = 0; firstGroup < groups; firstGroup += stepGroups)
        result.push_back({expert, 1, field, firstRow, std::min(rows, section.rows - firstRow), firstGroup,
                          std::min(stepGroups, groups - firstGroup)});
  return result;
}

// Reads a step into input, its experts one after another, unit bytes per row
// of a group: each part's rows as stored, rows past the parts zero. Whole
// experts, and a step of whole rows of one, are read at once, as they are
// contiguous in the source.
void gatherRows(const Section &section, const Step &step, uint32_t unit, uint8_t *input) {
  const uint32_t groups = section.columns / kQ4GroupElements;
  const uint64_t rowBytes = uint64_t(step.groups) * unit;
  const uint64_t sourceRowBytes = uint64_t(groups) * unit;
  if (step.experts > 1) {
    const ProjectionPart &part = section.parts.front();
    part.fields[step.field].tensor->read(uint64_t(step.firstExpert) * part.rows * sourceRowBytes,
                                         {input, uint64_t(step.experts) * part.rows * sourceRowBytes});
    return;
  }
  uint32_t partStart = 0;
  for (const auto &part : section.parts) {
    const uint32_t begin = std::max(step.firstRow, partStart);
    const uint32_t end = std::min(step.firstRow + step.rows, partStart + part.rows);
    if (begin < end) {
      const SourceTensor &tensor = *part.fields[step.field].tensor;
      const uint64_t offset = (uint64_t(step.firstExpert) * part.rows + begin - partStart) * sourceRowBytes +
                              uint64_t(step.firstGroup) * unit;
      uint8_t *to = input + (begin - step.firstRow) * rowBytes;
      if (step.groups == groups) {
        tensor.read(offset, {to, (end - begin) * rowBytes});
      } else {
        for (uint32_t row = 0; row < end - begin; ++row)
          tensor.read(offset + row * sourceRowBytes, {to + row * rowBytes, rowBytes});
      }
    }
    partStart += part.rows;
  }
  if (step.firstRow + step.rows > partStart) {
    const uint32_t gathered = std::max(step.firstRow, partStart) - step.firstRow;
    std::memset(input + gathered * rowBytes, 0, (step.rows - gathered) * rowBytes);
  }
}

// Transposes 256 rows of `groups` groups of Unit bytes, rowBytes apart, into
// `groups` groups of 256 rows.
template <uint32_t Unit> void transposeTile(const uint8_t *rows, uint64_t rowBytes, uint32_t groups, uint8_t *to) {
  for (uint32_t group = 0; group < groups; ++group)
    for (uint32_t row = 0; row < kQ4StorageN; ++row)
      std::memcpy(to + (uint64_t(group) * kQ4StorageN + row) * Unit, rows + row * rowBytes + uint64_t(group) * Unit,
                  Unit);
}

// Reorders a step of a projection's field into the field's [rows / 256]
// [groups][256] tiles of unit bytes, the first expert's at `field` and each
// later one's a whole expert after it; rows past the parts are zero.
void writeProjectionStep(const Section &section, const Step &step, uint32_t unit, uint8_t *field,
                         std::vector<uint8_t> &input) {
  const uint32_t groups = section.columns / kQ4GroupElements;
  const uint64_t rowBytes = uint64_t(step.groups) * unit;
  const uint64_t expertRowBytes = uint64_t(step.rows) * rowBytes;
  if (input.size() < step.experts * expertRowBytes) input.resize(step.experts * expertRowBytes);
  gatherRows(section, step, unit, input.data());
  // Scales and biases, 4-bit and 8-bit codes.
  if (unit != 2 && unit != 32 && unit != 64) throw std::logic_error("unsupported affine group bytes");
  const auto transpose = unit == 2 ? transposeTile<2> : unit == 32 ? transposeTile<32> : transposeTile<64>;
  for (uint32_t expert = 0; expert < step.experts; ++expert) {
    uint8_t *expertField = field + uint64_t(expert) * (section.bytes / section.experts);
    for (uint32_t tile = 0; tile < step.rows / kQ4StorageN; ++tile)
      transpose(input.data() + expert * expertRowBytes + uint64_t(tile) * kQ4StorageN * rowBytes, rowBytes,
                step.groups,
                expertField +
                    (uint64_t(step.firstRow / kQ4StorageN + tile) * groups + step.firstGroup) * kQ4StorageN * unit);
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
// The loops have no early exit, so they vectorize; the sign of a zero
// minimum or maximum, which the order of a vector reduction may change, does
// not reach any output.
void quantizeGroup(const uint8_t *weights, uint8_t *codes, uint16_t &scale, uint16_t &bias) {
  std::array<float, kQ4GroupElements> w;
  bool finite = true;
  for (size_t i = 0; i < w.size(); ++i) {
    w[i] = bfloat16Value(weights + i * kBFloat16Bytes);
    finite &= std::isfinite(w[i]);
  }
  if (!finite) throw std::runtime_error("non-finite weight in a BF16 projection");
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
  std::array<uint8_t, kQ4GroupElements> code;
  for (size_t i = 0; i < w.size(); ++i)
    code[i] = static_cast<uint8_t>(std::clamp(std::round((w[i] - offset) / step), 0.0F, 15.0F));
  for (size_t i = 0; i < w.size(); i += 2) codes[i / 2] = static_cast<uint8_t>(code[i] | code[i + 1] << 4);
  scale = nearestBfloat16(step);
  bias = nearestBfloat16(offset);
}

// Quantizes a step of a BF16 projection into its codes, scales and biases at
// fields, each in [rows / 256][groups][256] tiles of its group bytes.
void writeQuantizedStep(const Section &section, const Step &step, const std::array<uint8_t *, 3> &fields,
                        std::vector<uint8_t> &input) {
  const uint32_t groups = section.columns / kQ4GroupElements;
  const auto unit = groupBytes(section);
  const uint64_t rowBytes = uint64_t(step.groups) * kBfloat16GroupBytes;
  if (input.size() < step.rows * rowBytes) input.resize(step.rows * rowBytes);
  gatherRows(section, step, kBfloat16GroupBytes, input.data());
  for (uint32_t row = 0; row < step.rows; ++row) {
    const uint32_t sourceRow = step.firstRow + row;
    for (uint32_t group = 0; group < step.groups; ++group) {
      const uint64_t at = (uint64_t(sourceRow / kQ4StorageN) * groups + step.firstGroup + group) * kQ4StorageN +
                          sourceRow % kQ4StorageN;
      uint16_t scale, bias;
      quantizeGroup(input.data() + row * rowBytes + uint64_t(group) * kBfloat16GroupBytes, fields[0] + at * unit[0],
                    scale, bias);
      std::memcpy(fields[1] + at * unit[1], &scale, sizeof scale);
      std::memcpy(fields[2] + at * unit[2], &bias, sizeof bias);
    }
  }
}

// float(-exp(double(A_log))) of a BF16 or F32 vector into destination.
void writeDecay(const Section &section, uint8_t *destination, std::vector<uint8_t> &input) {
  const SourceTensor &tensor = *section.input.tensor;
  if (input.size() < tensor.bytes) input.resize(tensor.bytes);
  tensor.read(0, {input.data(), tensor.bytes});
  for (uint64_t i = 0; i < section.bytes / sizeof(float); ++i) {
    float logarithm;
    if (tensor.dtype == "BF16") logarithm = bfloat16Value(input.data() + i * kBFloat16Bytes);
    else std::memcpy(&logarithm, input.data() + i * 4, 4);
    const auto value = static_cast<float>(-std::exp(static_cast<double>(logarithm)));
    if (!std::isfinite(value)) throw std::runtime_error("non-finite GDN decay");
    std::memcpy(destination + i * sizeof(float), &value, sizeof value);
  }
}

} // namespace

void writeAffineImage(std::span<uint8_t> destination, const Image &image) {
  if (destination.size() != image.bytes) throw std::invalid_argument("affine image destination size differs");
  const auto header = weightFileHeader(image.magic, image.layer, image.type);
  std::vector<std::pair<uint64_t, uint64_t>> extents{{0, header.size()}};
  for (const Section &section : image.sections) extents.emplace_back(section.offset, section.bytes);
  zeroUnwritten(destination, std::move(extents));
  std::memcpy(destination.data(), header.data(), header.size());
  // Every task writes bytes of its own and stages through its thread's buffer.
  std::vector<std::function<void(std::vector<uint8_t> &)>> tasks;
  for (const Section &section : image.sections) {
    uint8_t *const at = destination.data() + section.offset;
    const uint64_t fieldRows = uint64_t(section.rows) * (section.columns / kQ4GroupElements);
    switch (section.kind) {
    case SectionKind::Copy:
      for (uint64_t from = 0; from < section.bytes; from += kLoadStepBytes) {
        const uint64_t bytes = std::min(kLoadStepBytes, section.bytes - from);
        tasks.push_back([&section, at, from, bytes](std::vector<uint8_t> &) {
          section.input.tensor->read(from, {at + from, bytes});
        });
      }
      break;
    case SectionKind::Decay:
      tasks.push_back([&section, at](std::vector<uint8_t> &input) { writeDecay(section, at, input); });
      break;
    case SectionKind::Projection: {
      // Each expert's fields follow one another; field is expert 0's.
      const auto unit = groupBytes(section);
      uint8_t *field = at;
      for (uint32_t index = 0; index < unit.size(); ++index) {
        for (const Step &step : steps(section, index, unit[index]))
          tasks.push_back([&section, step, unit = unit[index],
                           field = field + uint64_t(step.firstExpert) * (section.bytes / section.experts)](
                              std::vector<uint8_t> &input) { writeProjectionStep(section, step, unit, field, input); });
        field += fieldRows * unit[index];
      }
      break;
    }
    case SectionKind::Quantize: {
      const auto unit = groupBytes(section);
      const std::array<uint8_t *, 3> fields{at, at + fieldRows * unit[0], at + fieldRows * (unit[0] + unit[1])};
      for (const Step &step : steps(section, 0, kBfloat16GroupBytes))
        tasks.push_back([&section, step, fields](std::vector<uint8_t> &input) {
          writeQuantizedStep(section, step, fields, input);
        });
      break;
    }
    }
  }
  std::vector<std::vector<uint8_t>> staging(loadThreads());
  parallelFor(tasks.size(), [&](size_t index, unsigned thread) { tasks[index](staging[thread]); });
}

} // namespace splash::model::affine
