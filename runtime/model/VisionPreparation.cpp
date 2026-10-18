#include "model/VisionPreparation.hpp"
#include "model/Bfloat16.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightStore.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <span>

namespace splash::model::vision {
namespace {

uint32_t elementBytes(const SourceTensor &tensor) { return tensor.dtype == "F32" ? 4 : 2; }

// Converts count values of tensor to BF16 bits. False when a value is not
// exactly a BF16.
bool convert(const SourceTensor &tensor, const uint8_t *source, uint16_t *destination, uint64_t count) {
  if (tensor.dtype == "BF16") {
    std::memcpy(destination, source, count * kBFloat16Bytes);
    return true;
  }
  for (uint64_t i = 0; i < count; ++i) {
    float value;
    if (tensor.dtype == "F32") {
      std::memcpy(&value, source + i * 4, 4);
    } else {
      _Float16 half;
      std::memcpy(&half, source + i * 2, 2);
      value = static_cast<float>(half);
    }
    const auto bits = exactBfloat16(value);
    if (!bits) return false;
    destination[i] = *bits;
  }
  return true;
}

uint64_t sourceRowBytes(const Input &in, uint32_t columns) {
  if (in.bits) return uint64_t(columns) * in.bits / 8 + uint64_t(columns / in.group) * 4;
  return uint64_t(columns) * elementBytes(in.tensor);
}

// Read each affine plane once for a batch, then decode directly into the
// image's strided BF16 rows. The thread reuses its one staging allocation.
void convertAffine(const Input &in, uint32_t first, uint32_t count, uint32_t columns,
                   uint16_t *destination, uint32_t stride, std::vector<uint8_t> &staging) {
  const uint64_t codeRowBytes = uint64_t(columns) * in.bits / 8;
  const uint32_t groups = columns / in.group, mask = (1u << in.bits) - 1;
  const uint64_t parameterRowBytes = uint64_t(groups) * sizeof(uint16_t);
  const uint64_t codeBytes = count * codeRowBytes, parameterBytes = count * parameterRowBytes;
  const uint64_t bytes = codeBytes + 2 * parameterBytes;
  if (staging.size() < bytes) staging.resize(bytes);
  in.tensor.read(first * codeRowBytes, std::span(staging).first(codeBytes));
  in.scales->read(first * parameterRowBytes, std::span(staging).subspan(codeBytes, parameterBytes));
  in.biases->read(first * parameterRowBytes, std::span(staging).subspan(codeBytes + parameterBytes, parameterBytes));
  for (uint32_t row = 0; row < count; ++row) {
    const uint8_t *codes = staging.data() + row * codeRowBytes;
    uint16_t *out = destination + uint64_t(row) * stride;
    uint32_t bit = 0;
    for (uint32_t group = 0; group < groups; ++group) {
      uint16_t scaleBits, biasBits;
      const uint64_t parameter = uint64_t(row) * parameterRowBytes + group * sizeof(uint16_t);
      std::memcpy(&scaleBits, staging.data() + codeBytes + parameter, sizeof(scaleBits));
      std::memcpy(&biasBits, staging.data() + codeBytes + parameterBytes + parameter, sizeof(biasBits));
      const float scale = widenBfloat16(scaleBits), bias = widenBfloat16(biasBits);
      for (uint32_t column = 0; column < in.group; ++column, bit += in.bits) {
        const uint32_t shift = bit % 8;
        uint32_t code = uint32_t(codes[bit / 8]) >> shift;
        if (shift + in.bits > 8) code |= uint32_t(codes[bit / 8 + 1]) << (8 - shift);
        out[group * in.group + column] = nearestBfloat16(scale * float(code & mask) + bias);
      }
    }
  }
}

// Source bytes and patch values of one task.
struct Staging {
  std::vector<uint8_t> source;
  std::vector<uint16_t> values;
};

// The tasks that write a section in batches of whole rows converted to BF16.
// The image's patch embedding orders a row [channel, frame, patch-row,
// patch-col]; MLX stores [frame, patch-row, patch-col, channel] and GGUF one
// [channel, patch-row, patch-col] tensor per frame. Padded columns are zero;
// padded rows are not written.
void addSectionTasks(uint8_t *image, const Section &s, uint32_t pixels,
                     std::vector<std::function<void(Staging &)>> &tasks) {
  const auto frames = static_cast<uint32_t>(s.inputs.size());
  const uint32_t columns = s.columns / frames;
  uint64_t stagedRowBytes = 0;
  for (const auto &in : s.inputs) stagedRowBytes = std::max(stagedRowBytes, sourceRowBytes(in, columns));
  if (s.patch) stagedRowBytes += uint64_t(s.columns) * kBFloat16Bytes;
  const auto batchRows = static_cast<uint32_t>(std::clamp<uint64_t>(kLoadStepBytes / stagedRowBytes, 1, s.rows));
  for (uint32_t row = 0; row < s.rows; row += batchRows) {
    const uint32_t count = std::min(batchRows, s.rows - row);
    auto *output = reinterpret_cast<uint16_t *>(image + s.offset) + uint64_t(row) * s.storedColumns;
    tasks.push_back([&s, pixels, frames, columns, row, count, output](Staging &staging) {
      // Converts count rows of one input to rows of stride BF16 values.
      const auto convertRows = [&](const Input &in, uint16_t *to, uint32_t stride) {
        if (in.bits) {
          convertAffine(in, row, count, columns, to, stride, staging.source);
          return;
        }
        const uint64_t bytes = uint64_t(columns) * elementBytes(in.tensor);
        if (staging.source.size() < count * bytes) staging.source.resize(count * bytes);
        in.tensor.read(row * bytes, std::span(staging.source).first(count * bytes));
        for (uint32_t r = 0; r < count; ++r)
          if (!convert(in.tensor, staging.source.data() + r * bytes, to + uint64_t(r) * stride, columns))
            throw WeightStoreError("vision tensor " + in.name + " in " + in.tensor.file->path().string() +
                                   " is not exactly representable in BF16");
      };
      if (!s.patch) {
        convertRows(s.inputs.front(), output, s.storedColumns);
      } else {
        if (staging.values.size() < uint64_t(count) * s.columns) staging.values.resize(uint64_t(count) * s.columns);
        uint16_t *values = staging.values.data();
        for (uint32_t frame = 0; frame < frames; ++frame)
          convertRows(s.inputs[frame], values + uint64_t(frame) * count * columns, columns);
        for (uint32_t r = 0; r < count; ++r)
          for (uint32_t c = 0; c < s.columns; ++c) {
            const uint32_t channel = c / (2 * pixels), frame = c / pixels % 2, pixel = c % pixels;
            output[uint64_t(r) * s.storedColumns + c] =
                frames == 1 ? values[uint64_t(r) * columns + (uint64_t(frame) * pixels + pixel) * 3 + channel]
                            : values[(uint64_t(frame) * count + r) * columns + uint64_t(channel) * pixels + pixel];
          }
      }
      for (uint32_t r = 0; r < count; ++r)
        std::fill_n(output + uint64_t(r) * s.storedColumns + s.columns, s.storedColumns - s.columns, uint16_t{0});
    });
  }
}

} // namespace

void writeVision(std::span<uint8_t> destination, const Plan &plan) {
  if (destination.size() != plan.bytes) throw WeightStoreError("vision image destination size differs");
  const auto header = weightFileHeader(kVisionMagic, plan.depth, 0);
  std::vector<std::pair<uint64_t, uint64_t>> extents{{0, header.size()}};
  for (const auto &s : plan.sections)
    extents.emplace_back(s.offset, uint64_t(s.rows) * s.storedColumns * kBFloat16Bytes);
  zeroUnwritten(destination, std::move(extents));
  std::memcpy(destination.data(), header.data(), header.size());
  std::vector<std::function<void(Staging &)>> tasks;
  for (const auto &s : plan.sections)
    addSectionTasks(destination.data(), s, plan.patchSize * plan.patchSize, tasks);
  std::vector<Staging> staging(loadThreads());
  parallelFor(tasks.size(), [&](size_t index, unsigned thread) { tasks[index](staging[thread]); });
}

} // namespace splash::model::vision
