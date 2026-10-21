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
  uint32_t widest = 0;
  for (const auto &in : s.inputs) widest = std::max(widest, elementBytes(in.tensor));
  const uint64_t stagedRowBytes = uint64_t(columns) * widest + (s.patch ? uint64_t(s.columns) * kBFloat16Bytes : 0);
  const auto batchRows = static_cast<uint32_t>(std::clamp<uint64_t>(kLoadStepBytes / stagedRowBytes, 1, s.rows));
  for (uint32_t row = 0; row < s.rows; row += batchRows) {
    const uint32_t count = std::min(batchRows, s.rows - row);
    auto *output = reinterpret_cast<uint16_t *>(image + s.offset) + uint64_t(row) * s.storedColumns;
    tasks.push_back([&s, pixels, frames, columns, row, count, output](Staging &staging) {
      // Converts count rows of one input to rows of stride BF16 values.
      const auto convertRows = [&](const Input &in, uint16_t *to, uint32_t stride) {
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
