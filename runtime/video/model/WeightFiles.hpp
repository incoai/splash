#pragma once

// The engine's weight files through Splash's readers: each converted directory of safetensors files (dit-<variant>/,
// fl2va/transformer/) as one checkpoint index (model/SafetensorsCheckpoint.hpp), the decoders' blobs as weight sources
// (model/WeightSource.hpp), read in slices on Splash's load threads (model/WeightImages.hpp parallelFor), and their
// JSON manifests as Splash reads JSON, with NSJSONSerialization.
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightSource.hpp"

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>

namespace splash::video::model {

using splash::model::SafetensorsCheckpoint;
using splash::model::SourceTensor;
using splash::model::WeightSource;

// The tensor `name`, which must hold `dtype` and, unless shape is empty, `shape`.
[[nodiscard]] const SourceTensor &tensor(const SafetensorsCheckpoint &checkpoint, std::string_view name,
                                         std::string_view dtype, const std::vector<int64_t> &shape = {});
// Every byte of the tensor, or bytes [offset, offset + bytes) of the file, into `destination`.
void read(const SourceTensor &tensor, void *destination);
void read(const WeightSource &source, uint64_t offset, uint64_t bytes, void *destination);

#ifdef __OBJC__
// The JSON object of a converter's manifest.
[[nodiscard]] NSDictionary *readManifest(const std::filesystem::path &path);
// A manifest's integer (an offset, a dimension), or nothing for any other value.
[[nodiscard]] std::optional<int64_t> integer(id value);
#endif

} // namespace splash::video::model
