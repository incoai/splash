#include "model/VisionPreparation.hpp"
#include "model/Bfloat16.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightStore.hpp"
#include "ops/Vision.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <span>

namespace splash::model::vision {
namespace {

// A Q8_0 block: an F16 scale and 32 int8 quants.
constexpr uint64_t kQ8Values = 32, kQ8BlockBytes = 34;

// The source bytes of `columns` values of tensor, a Q8_0 one's in whole
// blocks.
uint64_t sourceBytes(const SourceTensor &tensor, uint64_t columns) {
  if (tensor.dtype == "Q8_0") return columns / kQ8Values * kQ8BlockBytes;
  return columns * (tensor.dtype == "F32" ? 4 : 2);
}

// Converts count values of tensor to BF16 bits. False when a value is not
// exactly a BF16; a Q8_0 value is rounded to the nearest.
bool convert(const SourceTensor &tensor, const uint8_t *source, uint16_t *destination, uint64_t count) {
  if (tensor.dtype == "BF16") {
    std::memcpy(destination, source, count * kBFloat16Bytes);
    return true;
  }
  if (tensor.dtype == "Q8_0") {
    for (uint64_t block = 0; block < count / kQ8Values; ++block) {
      const uint8_t *quants = source + block * kQ8BlockBytes;
      _Float16 scale;
      std::memcpy(&scale, quants, 2);
      for (uint64_t i = 0; i < kQ8Values; ++i)
        destination[block * kQ8Values + i] =
            roundBfloat16(static_cast<float>(scale) * static_cast<float>(static_cast<int8_t>(quants[2 + i])));
    }
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
// patch-col], over the frames its width holds (Qwen's two, Inkling's one);
// MLX and transformers' linear patch embeddings store [frame, patch-row,
// patch-col, channel] and GGUF one [channel, patch-row, patch-col] tensor per
// frame. A stacked section's inputs each write their rows of it in turn.
// Padded columns are zero; padded rows are not written.
void addSectionTasks(uint8_t *image, const Section &s, uint32_t pixels,
                     std::vector<std::function<void(Staging &)>> &tasks) {
  if (s.inputs.empty()) return;
  const auto frames = static_cast<uint32_t>(s.stacked ? 1 : s.inputs.size());
  const auto parts = static_cast<uint32_t>(s.stacked ? s.inputs.size() : 1);
  const uint32_t columns = s.columns / frames, partRows = s.rows / parts;
  uint64_t widest = 0;
  for (const auto &in : s.inputs) widest = std::max(widest, sourceBytes(in.tensor, columns));
  const uint64_t stagedRowBytes = widest + (s.patch ? uint64_t(s.columns) * kBFloat16Bytes : 0);
  const auto batchRows = static_cast<uint32_t>(std::clamp<uint64_t>(kLoadStepBytes / stagedRowBytes, 1, partRows));
  for (uint32_t part = 0; part < parts; ++part) {
    for (uint32_t row = 0; row < partRows; row += batchRows) {
      const uint32_t count = std::min(batchRows, partRows - row);
      auto *output =
          reinterpret_cast<uint16_t *>(image + s.offset) + (uint64_t(part) * partRows + row) * s.storedColumns;
      tasks.push_back([&s, pixels, frames, columns, part, row, count, output](Staging &staging) {
        // Converts count rows of one input to rows of stride BF16 values.
        const auto convertRows = [&](const Input &in, uint16_t *to, uint32_t stride) {
          const uint64_t bytes = sourceBytes(in.tensor, columns);
          if (staging.source.size() < count * bytes) staging.source.resize(count * bytes);
          in.tensor.read(row * bytes, std::span(staging.source).first(count * bytes));
          for (uint32_t r = 0; r < count; ++r)
            if (!convert(in.tensor, staging.source.data() + r * bytes, to + uint64_t(r) * stride, columns))
              throw WeightStoreError("vision tensor " + in.name + " in " + in.tensor.file->path().string() +
                                     " is not exactly representable in BF16");
        };
        if (!s.patch) {
          convertRows(s.inputs[part], output, s.storedColumns);
        } else {
          if (staging.values.size() < uint64_t(count) * s.columns) staging.values.resize(uint64_t(count) * s.columns);
          uint16_t *values = staging.values.data();
          for (uint32_t frame = 0; frame < frames; ++frame)
            convertRows(s.inputs[frame], values + uint64_t(frame) * count * columns, columns);
          const uint32_t patchFrames = s.columns / (ops::kImageChannels * pixels);
          for (uint32_t r = 0; r < count; ++r)
            for (uint32_t c = 0; c < s.columns; ++c) {
              const uint32_t channel = c / (patchFrames * pixels), frame = c / pixels % patchFrames,
                             pixel = c % pixels;
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
}

} // namespace

void writeVision(std::span<uint8_t> destination, const Plan &plan) {
  if (destination.size() != plan.bytes) throw WeightStoreError("vision image destination size differs");
  const auto header = weightFileHeader(kVisionMagic, plan.depth, 0);
  std::vector<std::pair<uint64_t, uint64_t>> extents{{0, header.size()}};
  for (const auto &s : plan.sections)
    if (!s.inputs.empty()) extents.emplace_back(s.offset, uint64_t(s.rows) * s.storedColumns * kBFloat16Bytes);
  zeroUnwritten(destination, std::move(extents));
  std::memcpy(destination.data(), header.data(), header.size());
  std::vector<std::function<void(Staging &)>> tasks;
  for (const auto &s : plan.sections)
    addSectionTasks(destination.data(), s, plan.patchSize * plan.patchSize, tasks);
  std::vector<Staging> staging(loadThreads());
  parallelFor(tasks.size(), [&](size_t index, unsigned thread) { tasks[index](staging[thread]); });
}

} // namespace splash::model::vision
