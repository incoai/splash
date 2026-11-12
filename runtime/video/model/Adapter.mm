#include "video/model/Adapter.hpp"

#include "metal/abi/video/Layout.h"

#include <cstdlib>
#include <stdexcept>

namespace splash::video::model {
namespace {

constexpr int64_t D = H3_DIT_HIDDEN, F = H3_DIT_FFN, C = H3_DIT_INNER, QKV = H3_DIT_QKV;
constexpr char kFormat[] = "h3-lora-v1";

struct Manifest {
  int32_t rank = 0;
  double scale = 1.0;
  std::string name;
};

Manifest readAdapterManifest(const std::filesystem::path &dir) {
  NSDictionary *m = readManifest(dir / "manifest.json");
  if (![m[@"format"] isEqual:[NSString stringWithUTF8String:kFormat]])
    throw std::invalid_argument((dir / "manifest.json").string() + ": not an adapter of format " + kFormat);
  Manifest out;
  const std::optional<int64_t> rank = integer(m[@"rank"]);
  NSNumber *scale = m[@"scale"];
  if (!rank || *rank <= 0 || *rank % 128 || ![scale isKindOfClass:[NSNumber class]])
    throw std::invalid_argument((dir / "manifest.json").string() + ": rank (a multiple of 128) and scale");
  if (*rank != 128) throw std::invalid_argument("the LoRA kernels take rank 128 adapters");
  out.rank = static_cast<int32_t>(*rank);
  out.scale = scale.doubleValue;
  NSString *name = m[@"name"];
  out.name = [name isKindOfClass:[NSString class]] ? name.UTF8String : dir.filename().string();
  return out;
}

} // namespace

std::vector<AdapterSpec> adapterSpecs(const std::filesystem::path &ditDir,
                                      const std::optional<std::vector<AdapterSpec>> &override) {
  if (override) return *override;
  @autoreleasepool {
    NSDictionary *m = readManifest(ditDir / "manifest.json");
    NSArray *list = m[@"adapters"];
    std::vector<AdapterSpec> out;
    if (!list) return out;
    if (![list isKindOfClass:[NSArray class]])
      throw std::invalid_argument((ditDir / "manifest.json").string() + ": adapters must be a list");
    for (id e in list) {
      // a converter's record of the adapters merged at conversion (VDN-H3: their names) is not a runtime adapter
      if (![e isKindOfClass:[NSDictionary class]]) continue;
      NSString *path = e[@"path"];
      NSNumber *scale = e[@"scale"];
      if (![path isKindOfClass:[NSString class]] || (scale && ![scale isKindOfClass:[NSNumber class]]))
        throw std::invalid_argument((ditDir / "manifest.json").string() + ": adapters are {path, scale}");
      std::filesystem::path p(path.UTF8String);
      out.push_back({p.is_absolute() ? p : ditDir / p, scale ? scale.doubleValue : 1.0});
    }
    return out;
  }
}

AdapterSpec parseAdapterSpec(const std::string &text) {
  const size_t colon = text.rfind(':');
  if (colon == std::string::npos || colon + 1 == text.size()) return {text, 1.0};
  char *end = nullptr;
  const double scale = std::strtod(text.c_str() + colon + 1, &end);
  if (end == text.c_str() + colon + 1 || *end) return {text, 1.0};  // a path with a colon
  return {text.substr(0, colon), scale};
}

std::unique_ptr<Adapter> Adapter::load(metal::MetalBackend &backend, const AdapterSpec &spec,
                                       const std::set<int> &qkfixBlocks) {
  auto out = std::unique_ptr<Adapter>(new Adapter());
  Manifest m;
  @autoreleasepool {
    m = readAdapterManifest(spec.path);
  }
  out->rank_ = m.rank;
  out->scale_ = static_cast<float>(m.scale * spec.scale);
  out->name_ = m.name;
  const SafetensorsCheckpoint checkpoint(spec.path);
  const int64_t r = m.rank;
  auto resident = [&](const std::string &name, const std::vector<int64_t> &shape) {
    const SourceTensor &source = tensor(checkpoint, name, "BF16", shape);
    metal::MetalBuffer buffer = allocate(backend, source.bytes, name);
    read(source, buffer.contents());
    out->residentBytes_ += buffer.sizeBytes();
    return buffer;
  };
  out->blocks_.resize(H3_DIT_BLOCKS);
  for (int i = 0; i < H3_DIT_BLOCKS; ++i) {
    AdapterBlock &b = out->blocks_[i];
    const std::string p = "blocks." + std::to_string(i) + ".lora.";
    b.qkv = {resident(p + "qkv.a", {3 * r, D}), resident(p + "qkv.b", {QKV, r}), 3, int32_t(QKV), int32_t(D)};
    b.out = {resident(p + "out.a", {r, C}), resident(p + "out.b", {D, r}), 1, int32_t(D), int32_t(C)};
    b.fc1 = {resident(p + "fc1.a", {r, D}), resident(p + "fc1.b", {2 * F, r}), 1, int32_t(2 * F), int32_t(D)};
    b.fc2 = {resident(p + "fc2.a", {r, F}), resident(p + "fc2.b", {D, r}), 1, int32_t(D), int32_t(F)};
    if (qkfixBlocks.count(i)) b.qkvPlain = resident(p + "qkv.a_plain", {3 * r, D});
  }
  return out;
}

std::vector<Adapter::RefinerBlock> Adapter::refiner(const AdapterSpec &spec, int blocks, float *scale) {
  Manifest m;
  @autoreleasepool {
    m = readAdapterManifest(spec.path);
  }
  *scale = static_cast<float>(m.scale * spec.scale);
  const SafetensorsCheckpoint checkpoint(spec.path);
  std::vector<RefinerBlock> out;
  if (!checkpoint.find("refiner.0.qkv.a")) return out;
  const int64_t r = m.rank;
  auto get = [&](const std::string &name, const std::vector<int64_t> &shape) {
    const SourceTensor &source = tensor(checkpoint, name, "BF16", shape);
    std::vector<uint16_t> v(source.bytes / 2);
    read(source, v.data());
    return v;
  };
  out.resize(blocks);
  for (int i = 0; i < blocks; ++i) {
    const std::string p = "refiner." + std::to_string(i) + ".";
    RefinerBlock &b = out[i];
    b.qkv = {get(p + "qkv.a", {3 * r, D}), get(p + "qkv.b", {QKV, 3 * r}), int32_t(3 * r)};
    b.out = {get(p + "out.a", {r, C}), get(p + "out.b", {D, r}), int32_t(r)};
    b.fc1 = {get(p + "fc1.a", {r, D}), get(p + "fc1.b", {2 * F, r}), int32_t(r)};
    b.fc2 = {get(p + "fc2.a", {r, F}), get(p + "fc2.b", {D, r}), int32_t(r)};
  }
  return out;
}

} // namespace splash::video::model
