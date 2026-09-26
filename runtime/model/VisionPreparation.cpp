// Editing this file re-prepares every vision model.
#include "model/VisionPreparation.hpp"
#include "WeightPreparationIdentity.hpp"
#include "model/Bfloat16.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightStore.hpp"

#include <algorithm>
#include <cstring>
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

// Source bytes, patch values and output rows of one batch, reused by every
// section of a file.
struct Staging {
  std::vector<uint8_t> source;
  std::vector<uint16_t> values, output;
};

// Writes a section in batches of whole rows converted to BF16. The packed
// patch embedding orders a row [channel, frame, patch-row, patch-col]; MLX
// stores [frame, patch-row, patch-col, channel] and GGUF one [channel,
// patch-row, patch-col] tensor per frame. Padded rows, columns and alignment
// stay zero: a prepared file starts zeroed.
void writeSection(int destination, const Section &s, uint32_t pixels, Staging &staging,
                  const PreparationCheck &admit) {
  const auto frames = static_cast<uint32_t>(s.inputs.size());
  const uint32_t columns = s.columns / frames;
  uint32_t widest = 0;
  for (const auto &in : s.inputs) widest = std::max(widest, elementBytes(in.tensor));
  const uint64_t storedRowBytes = uint64_t(s.storedColumns) * kBFloat16Bytes;
  const uint64_t rowBytes = uint64_t(columns) * widest + (s.patch ? uint64_t(s.columns) * kBFloat16Bytes : 0) +
                            storedRowBytes;
  const auto batchRows = static_cast<uint32_t>(
      std::clamp<uint64_t>(kWeightPreparationStagingBytes / rowBytes, 1, s.rows));
  auto &[source, values, output] = staging;
  source.resize(uint64_t(batchRows) * columns * widest);
  values.resize(s.patch ? uint64_t(batchRows) * s.columns : 0);
  output.resize(uint64_t(batchRows) * s.storedColumns);
  // Converts count rows of one input to rows of stride BF16 values.
  const auto convertRows = [&](const Input &in, uint32_t row, uint32_t count, uint16_t *to, uint32_t stride) {
    const uint64_t bytes = uint64_t(columns) * elementBytes(in.tensor);
    in.tensor.read(row * bytes, std::span(source).first(count * bytes));
    for (uint32_t r = 0; r < count; ++r)
      if (!convert(in.tensor, source.data() + r * bytes, to + uint64_t(r) * stride, columns))
        throw WeightStoreError("vision tensor " + in.name + " in " + in.tensor.file->path().string() +
                               " is not exactly representable in BF16");
  };
  for (uint32_t row = 0; row < s.rows; row += batchRows) {
    admit();
    const uint32_t count = std::min(batchRows, s.rows - row);
    if (!s.patch) {
      convertRows(s.inputs.front(), row, count, output.data(), s.storedColumns);
    } else {
      for (uint32_t frame = 0; frame < frames; ++frame)
        convertRows(s.inputs[frame], row, count, values.data() + uint64_t(frame) * count * columns, columns);
      for (uint32_t r = 0; r < count; ++r)
        for (uint32_t c = 0; c < s.columns; ++c) {
          const uint32_t channel = c / (2 * pixels), frame = c / pixels % 2, pixel = c % pixels;
          output[uint64_t(r) * s.storedColumns + c] =
              frames == 1 ? values[uint64_t(r) * columns + (uint64_t(frame) * pixels + pixel) * 3 + channel]
                          : values[(uint64_t(frame) * count + r) * columns + uint64_t(channel) * pixels + pixel];
        }
    }
    writeWeightBytes(destination, s.offset + row * storedRowBytes,
                     {reinterpret_cast<const uint8_t *>(output.data()), count * storedRowBytes});
  }
}

} // namespace

// The plan and the source tensors are all these bytes depend on; no
// configuration value enters them.
PreparedWeight visionWeight(const Plan &plan, const std::string &source) {
  WeightIdentity identity("splash-vision-preparation-v2 " SPLASH_VISION_PREPARATION_ID);
  identity.record("file", plan.depth, plan.patchSize, plan.bytes);
  for (const auto &s : plan.sections) {
    identity.record("section", s.offset, s.rows, s.columns, s.storedRows, s.storedColumns, s.patch);
    for (const auto &in : s.inputs) in.tensor.identify(identity);
  }
  return identity.weight(plan.bytes, "vision/model.bin", source);
}

void writeVision(int destination, const Plan &plan, const PreparationCheck &admit) {
  writeWeightBytes(destination, 0, weightFileHeader(kVisionMagic, plan.depth, 0));
  Staging staging;
  for (const auto &s : plan.sections) writeSection(destination, s, plan.patchSize * plan.patchSize, staging, admit);
}

} // namespace splash::model::vision
