#pragma once

#include "metal/MetalBackend.hpp"
#include "model/GgufImage.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightSource.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>

namespace splash::model {

// The Metal staging of the rows of one repack step, whatever the tensor,
// layer or expert count. An image's writer holds two (pipelineChunks).
inline constexpr uint64_t kGgufRepackStagingBytes = 32 << 20;

// Runs `count` chunks through two staging buffers: chunk i is staged into
// buffer i % 2 while the GPU command of chunk i - 1 runs, then that command
// is waited for and chunk i's submitted, so the reads and the GPU's writes
// overlap and one command is in flight at a time, as the backend requires.
// Buffer i % 2 is free when chunk i is staged: its last command, chunk
// i - 2's, ended before chunk i - 1's was submitted. stage(i, buffer)
// stages; submit(i, buffer) submits and returns the command's ticket, whose
// wait() ends it. A stage that throws leaves the command in flight to its
// ticket's destruction, which waits for it, so nothing writes the image
// after the writer failed. (Waiting for each command before staging the
// next kept the loading thread waiting on the GPU for 15% of a load; staged
// this way it waits 1-2%, and the 27B MLX's cold load takes 4.5 s instead
// of 5.2, the 27B and 35B UD-Q4_K_M GGUFs' 4.8 instead of 5.6 and 5.7
// instead of 6.7, M5 Max 32-core. The rows are still read at 4.1-4.4 GB/s
// against the SSD's 6.5-7, which this does not explain.)
template <class Stage, class Submit>
void pipelineChunks(size_t count, Stage &&stage, Submit &&submit) {
  std::optional<decltype(submit(size_t{0}, 0U))> pending;
  for (size_t chunk = 0; chunk < count; ++chunk) {
    const unsigned buffer = chunk % 2;
    stage(chunk, buffer);
    if (pending) static_cast<void>(pending->wait());
    pending.emplace(submit(chunk, buffer));
  }
  if (pending) static_cast<void>(pending->wait());
}

// Writes every byte of a planned image into its buffer: the header and
// descriptors, the copied rows and the planes the GPU repacks into it, each
// read from the source its rows name (a GGUF's, or an MLX checkpoint's
// tensors).
void writeGgufImage(metal::MetalBackend &backend, const metal::MetalBuffer &image, const gguf::Image &plan);

// Loads planned image `plan` into `images` as component `prefix` + its name:
// writeGgufImage writes it, then `checkUnchanged` finds the sources it was
// planned from unwritten since. The image holds both, to write it again on a
// restore. Its record declares the image's resident bytes
// (gguf::Image::residentBytes).
[[nodiscard]] WeightFile loadImage(metal::MetalBackend &backend, WeightImages &images, std::string_view prefix,
                                   std::shared_ptr<const gguf::Image> plan, std::function<void()> checkUnchanged);
// The record loadImage gives `plan`, for an image whose rows stream instead.
[[nodiscard]] WeightFileRecord imageRecord(const WeightImages &images, std::string_view prefix,
                                           const gguf::Image &plan);

// Rows [first, first + count) of a repack (its sources' rows in order), whole
// and back to back into `to`, count * ggufRowBytes(format, columns) bytes:
// the native rows the GPU repacks into those rows' planes. A streamed
// expert's rows are read with it (model/ExpertStore.hpp), from any thread.
void readRepackRows(const gguf::Repack &repack, uint64_t first, uint64_t count, uint8_t *to);

// Where rows [first, first + count) of a repack are in a file, when one of
// its sources stores them as they are and in order, so that they are read
// with the file's bytes alone: the file and the offset of the first row from
// its start. None when they are built (a quantized safetensors tensor's,
// NVFP4 rows, a quantized float tensor) or permuted, or span two sources.
struct FileRows final {
  const WeightSource *file = nullptr;
  uint64_t offset = 0;
};
[[nodiscard]] std::optional<FileRows> storedRepackRows(const gguf::Repack &repack, uint64_t first, uint64_t count);

// A plane of a repack's rows as its file stores them: where its row 0 is,
// from the file's start, and its bytes a row; none (rowBytes 0) for a
// format without that plane.
struct PlaneRows final {
  const WeightSource *file = nullptr;
  uint64_t offset = 0;
  uint64_t rowBytes = 0;
};
// The planes of a repack whose one source is an MLX affine, mxfp4 or mxfp8
// tensor of all its rows in order, from which the GPU builds the native
// blocks it repacks (moe_expert_load): an affine tensor's codes, scales and
// biases, an mxfp4 or mxfp8 tensor's codes and scales. None for any other
// repack, whose native rows readRepackRows builds.
[[nodiscard]] std::optional<std::array<PlaneRows, 3>> storedPlanes(const gguf::Repack &repack);

} // namespace splash::model
