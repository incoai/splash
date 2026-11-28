#include "model/VisionPreparation.hpp"
#include "model/Bfloat16.hpp"
#include "model/GgufFile.hpp"
#include "model/GgufImageLayout.hpp"
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

// The source bytes of `columns` values of a BF16, F16 or F32 tensor.
uint64_t sourceBytes(const SourceTensor &tensor, uint64_t columns) {
  return columns * (tensor.dtype == "F32" ? 4 : 2);
}

// Converts count values of a BF16, F16 or F32 tensor to BF16 bits. False when
// a value is not exactly a BF16.
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

// A Q8_0 matrix's planes (metal/abi/QuantFormat.h): its codes and scales.
GgufPlaneBytes q8Planes(const Section &s) { return ggufPlaneBytes(kQuantFormats[GGUF_FMT_Q80], s.rows, s.columns); }

// Source bytes and patch values of one task.
struct Staging {
  std::vector<uint8_t> source;
  std::vector<uint16_t> values;
};

// The tasks that write a Q8_0 matrix's planes from its rows' blocks, in
// batches of whole rows: block g of section row n is plane record
// quant_tile_index(n, g, G), its codes at their chunk-order slots
// (quant_slot) and its scale's two bytes in meta, as gguf_repack writes a
// Q8_0 tensor's.
void addQ8Tasks(uint8_t *image, const Section &s, std::vector<std::function<void(Staging &)>> &tasks) {
  const auto parts = static_cast<uint32_t>(s.inputs.size());
  const uint32_t partRows = s.rows / parts, groups = s.columns / kQ8Values;
  const uint64_t rowBytes = groups * kQ8BlockBytes;
  const auto [plane0, meta] = q8PlaneOffsets(s);
  const auto batchRows = static_cast<uint32_t>(std::clamp<uint64_t>(kLoadStepBytes / rowBytes, 1, partRows));
  for (uint32_t part = 0; part < parts; ++part)
    for (uint32_t row = 0; row < partRows; row += batchRows) {
      const uint32_t count = std::min(batchRows, partRows - row);
      tasks.push_back([&s, image, plane0, meta, part, partRows, row, count, groups, rowBytes](Staging &staging) {
        const Input &in = s.inputs[part];
        if (staging.source.size() < count * rowBytes) staging.source.resize(count * rowBytes);
        in.tensor.read(row * rowBytes, std::span(staging.source).first(count * rowBytes));
        for (uint32_t r = 0; r < count; ++r) {
          const uint32_t n = part * partRows + row + r;
          for (uint32_t g = 0; g < groups; ++g) {
            const uint8_t *block = staging.source.data() + r * rowBytes + g * kQ8BlockBytes;
            const uint64_t record = quant_tile_index(n, g, groups);
            uint8_t *codes = image + plane0 + record * kQ8Values;
            for (uint32_t e = 0; e < kQ8Values; ++e) codes[quant_slot(e)] = block[2 + e];
            std::memcpy(image + meta + record * 2, block, 2);
          }
        }
      });
    }
}

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

uint64_t sectionBytes(const Section &s) {
  if (!s.q8Planes) return uint64_t(s.storedRows) * s.storedColumns * kBFloat16Bytes;
  return q8PlaneOffsets(s).second + q8Planes(s).meta - s.offset;
}

std::pair<uint64_t, uint64_t> q8PlaneOffsets(const Section &s) {
  const uint64_t plane0 = alignWeightOffset(s.offset + GgufTensorDescriptor::kBytes);
  return {plane0, alignWeightOffset(plane0 + q8Planes(s).plane0)};
}

void placeSections(Plan &plan) {
  uint64_t at = kWeightFileAlignment;
  for (Section &s : plan.sections) {
    s.offset = at;
    at = alignWeightOffset(at + sectionBytes(s));
  }
  plan.bytes = at;
}

// Whether a projection's shape takes Q8_0 planes: unpadded columns of whole
// pairs of blocks (a 64-input span, ggufColumnUnit) and each of its `parts`
// stacked tensors whole 64-row plane tiles.
bool q8Shape(const Section &s, size_t parts) {
  return s.projection && parts && s.storedColumns == s.columns && s.rows / parts % QUANT_TILE_ROWS == 0 &&
         s.columns % ggufColumnUnit(kQuantFormats[GGUF_FMT_Q80]) == 0;
}

void placeQ8Matrices(Plan &plan) {
  for (Section &s : plan.sections) {
    const auto q8 = [](const Input &in) { return in.tensor.dtype == "Q8_0"; };
    s.q8Planes = q8Shape(s, s.inputs.size()) && std::all_of(s.inputs.begin(), s.inputs.end(), q8);
    if (s.q8Planes) continue;
    for (const Input &in : s.inputs)
      if (q8(in))
        throw WeightStoreError("vision tensor " + in.name + " in " + in.tensor.file->path().string() +
                               " is Q8_0; preparation reads Q8_0 only as a projection's planes: a matrix of whole "
                               "64-row tiles and 64-value spans, its q, k and v all Q8_0");
  }
  placeSections(plan);
}

std::string imageFormats(const Plan &plan) {
  std::string result;
  for (const Section &s : plan.sections)
    result += (s.gguf.empty() ? s.mlx : s.gguf) + (s.q8Planes ? " q80 " : " bf16 ") + std::to_string(s.rows) + "x" +
              std::to_string(s.columns) + "\n";
  return result;
}

void writeVision(std::span<uint8_t> destination, const Plan &plan) {
  if (destination.size() != plan.bytes) throw WeightStoreError("vision image destination size differs");
  const auto header = weightFileHeader(kVisionMagic, plan.depth, 0);
  std::vector<std::pair<uint64_t, uint64_t>> extents{{0, header.size()}};
  for (const auto &s : plan.sections) {
    if (s.inputs.empty()) continue;
    if (!s.q8Planes) {
      extents.emplace_back(s.offset, uint64_t(s.rows) * s.storedColumns * kBFloat16Bytes);
      continue;
    }
    const GgufPlaneBytes planes = q8Planes(s);
    const auto [plane0, meta] = q8PlaneOffsets(s);
    const GgufTensorDescriptor descriptor{ggml::kQ8_0, s.rows, s.columns, 32, 0, 2, 1, planes.plane0, 0, planes.meta};
    const auto bytes = descriptor.encode();
    std::memcpy(destination.data() + s.offset, bytes.data(), bytes.size());
    extents.insert(extents.end(), {{s.offset, bytes.size()}, {plane0, planes.plane0}, {meta, planes.meta}});
  }
  zeroUnwritten(destination, std::move(extents));
  std::memcpy(destination.data(), header.data(), header.size());
  std::vector<std::function<void(Staging &)>> tasks;
  for (const auto &s : plan.sections) {
    if (s.q8Planes) addQ8Tasks(destination.data(), s, tasks);
    else addSectionTasks(destination.data(), s, plan.patchSize * plan.patchSize, tasks);
  }
  std::vector<Staging> staging(loadThreads());
  parallelFor(tasks.size(), [&](size_t index, unsigned thread) { tasks[index](staging[thread]); });
}

} // namespace splash::model::vision
