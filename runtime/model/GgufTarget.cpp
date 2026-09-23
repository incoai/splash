#include "model/GgufTarget.hpp"

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <system_error>
#include <vector>

namespace splash::model {
namespace {

struct Mapping {
  void *address = nullptr;
  size_t bytes = 0;
  ~Mapping() {
    if (address) munmap(address, bytes);
  }
};

std::string describe(const char *what, int error) {
  return std::string(what) + ": " + std::system_category().message(error);
}

} // namespace

std::filesystem::path findTargetGguf(const std::filesystem::path &directory) {
  std::filesystem::path found;
  std::error_code error;
  for (const auto &entry : std::filesystem::directory_iterator(directory, error)) {
    if (entry.path().extension() != ".gguf") continue;
    if (!found.empty()) throw GgufError("target directory holds more than one GGUF: " + directory.string());
    found = entry.path();
  }
  if (error) throw GgufError("cannot list target directory: " + directory.string());
  if (found.empty()) throw GgufError("target directory holds no GGUF: " + directory.string());
  return found;
}

GgufTargetLoader::GgufTargetLoader(metal::MetalBackend &backend, std::filesystem::path path,
                                   gguf::TargetGeometry geometry)
    : backend_(&backend), file_(std::move(path)), planner_(file_, geometry) {
  descriptor_ = open(file_.path().c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor_ < 0) throw GgufError(describe("cannot open GGUF", errno) + ": " + file_.path().string());
}

GgufTargetLoader::~GgufTargetLoader() {
  if (descriptor_ >= 0) close(descriptor_);
}

WeightFile GgufTargetLoader::layer(uint32_t index) {
  const gguf::Image image = planner_.layer(index);
  return build(image, index, planner_.geometry().isFullAttentionLayer(index) ? 1u : 0u);
}

WeightFile GgufTargetLoader::head() {
  return build(planner_.head(), planner_.geometry().layers, 2u);
}

WeightFile GgufTargetLoader::embedding() {
  return build(planner_.embedding(), planner_.geometry().vocabularySize,
               planner_.geometry().hiddenSize);
}

metal::MetalBuffer GgufTargetLoader::mapTensor(uint64_t offset, uint64_t bytes) {
  const long pageSize = sysconf(_SC_PAGESIZE);
  if (pageSize <= 0) throw GgufError("unable to determine the page size");
  const uint64_t page = static_cast<uint64_t>(pageSize);
  const uint64_t base = offset / page * page;
  const uint64_t length = (offset - base + bytes + page - 1) / page * page;
  auto mapping = std::make_shared<Mapping>();
  mapping->address = mmap(nullptr, length, PROT_READ, MAP_SHARED, descriptor_,
                          static_cast<off_t>(base));
  if (mapping->address == MAP_FAILED) {
    mapping->address = nullptr;
    throw GgufError(describe("cannot map GGUF tensor", errno));
  }
  mapping->bytes = length;
  // Fault only this tensor's pages. GGUF offsets are 64-bit and tensors of
  // one layer need not be adjacent; neither gaps nor other layers need to
  // be resident for this upload. The shader reads a tensor-local view.
  volatile uint8_t sink = 0;
  for (uint64_t at = 0; at < length; at += page)
    sink ^= static_cast<const uint8_t *>(mapping->address)[at];
  const auto source = backend_->wrapSharedMemory(mapping->address, length, mapping, "gguf/source");
  return backend_->view(source, offset - base, bytes);
}

WeightFile GgufTargetLoader::build(const gguf::Image &image, uint32_t expectedLayer,
                                   uint32_t expectedType) {
  metal::MetalBuffer buffer = backend_->allocateBuffer(
      image.bytes, metal::BufferStorage::Shared, "target/" + image.name);
  auto *host = static_cast<uint8_t *>(buffer.contents());
  if (!host) throw GgufError("image buffer is not host visible: " + image.name);
  std::memset(host, 0, image.bytes);
  for (const gguf::Fill &fill : image.fills)
    std::memcpy(host + fill.offset, fill.bytes.data(), fill.bytes.size());

  if (!image.repacks.empty() || !image.copies.empty()) {
    std::vector<GgufRepackParams> repackParams;
    std::vector<GgufCopyParams> copyParams;
    repackParams.reserve(image.repacks.size());
    copyParams.reserve(image.copies.size());
    std::vector<metal::ComputeDispatch> dispatches;
    for (const gguf::Repack &repack : image.repacks) {
      GgufRepackParams params = repack.params;
      params.src_offset = 0;
      const auto source = mapTensor(repack.sourceOffset, repack.sourceBytes);
      repackParams.push_back(params);
      const uint64_t threads = uint64_t{params.rows} * (params.input_size / 32);
      dispatches.push_back({"gguf_repack", {{0, source}, {1, buffer}},
                            {{2, &repackParams.back(), sizeof(GgufRepackParams)}},
                            {(threads + 255) / 256, 1, 1}, {256, 1, 1}});
    }
    for (const gguf::Copy &copy : image.copies) {
      GgufCopyParams params = copy.params;
      params.src_offset = 0;
      const auto source = mapTensor(copy.sourceOffset, copy.sourceBytes);
      copyParams.push_back(params);
      dispatches.push_back({"gguf_copy", {{0, source}, {1, buffer}},
                            {{2, &copyParams.back(), sizeof(GgufCopyParams)}},
                            {((uint64_t{params.bytes} + 15) / 16 + 255) / 256, 1, 1}, {256, 1, 1}});
    }
    static_cast<void>(backend_->submitCommand(dispatches));
  }
  return WeightFile(*backend_, std::move(buffer), "target/" + image.name, kGgufImageMagic,
                    expectedLayer, expectedType);
}

} // namespace splash::model
