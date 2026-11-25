#include "video/model/WeightFiles.hpp"

#include "model/WeightImages.hpp"
#include "model/WeightStore.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>

namespace splash::video::model {
namespace {

// Reads [0, bytes) into destination through readAt(offset, slice), a load step per slice.
template <class Read>
void readSliced(uint64_t bytes, void *destination, const Read &readAt) {
  auto *out = static_cast<uint8_t *>(destination);
  const uint64_t step = splash::model::kLoadStepBytes;
  splash::model::parallelFor((bytes + step - 1) / step, [&](size_t index, unsigned) {
    const uint64_t at = uint64_t(index) * step;
    readAt(at, std::span<uint8_t>(out + at, std::min(step, bytes - at)));
  });
}

std::string shapeText(const std::vector<uint64_t> &shape) {
  std::string text = "[";
  for (size_t i = 0; i < shape.size(); ++i) text += (i ? ", " : "") + std::to_string(shape[i]);
  return text + "]";
}

} // namespace

const SourceTensor &tensor(const SafetensorsCheckpoint &checkpoint, std::string_view name, std::string_view dtype,
                           const std::vector<int64_t> &shape) {
  const SourceTensor &found = checkpoint.require(name);
  const std::string where = found.file->path().string() + ": " + std::string(name);
  if (found.dtype != dtype)
    throw splash::model::WeightStoreError(where + " is " + found.dtype + ", expected " + std::string(dtype));
  if (!shape.empty() &&
      !std::ranges::equal(found.shape, shape, [](uint64_t a, int64_t b) { return b >= 0 && a == uint64_t(b); }))
    throw splash::model::WeightStoreError(where + " has the shape " + shapeText(found.shape));
  return found;
}

void read(const SourceTensor &tensor, void *destination) {
  readSliced(tensor.bytes, destination, [&](uint64_t at, std::span<uint8_t> slice) { tensor.read(at, slice); });
}

void read(const WeightSource &source, uint64_t offset, uint64_t bytes, void *destination) {
  readSliced(bytes, destination,
             [&](uint64_t at, std::span<uint8_t> slice) { source.readFile(offset + at, slice); });
}

NSDictionary *readManifest(const std::filesystem::path &path) {
  NSData *data = [NSData dataWithContentsOfFile:@(path.c_str())];
  if (!data) throw splash::model::WeightStoreError("cannot read " + path.string());
  NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
  if (![manifest isKindOfClass:[NSDictionary class]])
    throw splash::model::WeightStoreError(path.string() + " is not a JSON object");
  return manifest;
}

std::optional<int64_t> integer(id value) {
  if (![value isKindOfClass:[NSNumber class]] || CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID())
    return std::nullopt;
  const double real = [value doubleValue];
  if (!std::isfinite(real) || std::floor(real) != real || std::fabs(real) > 9007199254740991.0) return std::nullopt;
  return [value longLongValue];
}

} // namespace splash::video::model
