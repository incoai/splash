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

WeightFile GgufTargetLoader::build(const gguf::Image &image, uint32_t expectedLayer,
                                   uint32_t expectedType) {
  metal::MetalBuffer buffer = backend_->allocateBuffer(
      image.bytes, metal::BufferStorage::Shared, "target/" + image.name);
  auto *host = static_cast<uint8_t *>(buffer.contents());
  if (!host) throw GgufError("image buffer is not host visible: " + image.name);
  std::memset(host, 0, image.bytes);
  for (const gguf::Fill &fill : image.fills)
    std::memcpy(host + fill.offset, fill.bytes.data(), fill.bytes.size());

  if (image.sourceEnd > image.sourceBegin) {
    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0) throw GgufError("unable to determine the page size");
    const uint64_t page = static_cast<uint64_t>(pageSize);
    const uint64_t base = image.sourceBegin / page * page;
    const uint64_t length = (image.sourceEnd - base + page - 1) / page * page;
    auto mapping = std::make_shared<Mapping>();
    mapping->address = mmap(nullptr, length, PROT_READ, MAP_SHARED, descriptor_,
                            static_cast<off_t>(base));
    if (mapping->address == MAP_FAILED) {
      mapping->address = nullptr;
      throw GgufError(describe("cannot map GGUF tensors", errno));
    }
    mapping->bytes = length;
    // Fault the pages in on the CPU: sequential reads run at disk speed and
    // keep the GPU from stalling on page faults inside the command.
    volatile uint8_t sink = 0;
    for (uint64_t offset = 0; offset < length; offset += page)
      sink ^= static_cast<const uint8_t *>(mapping->address)[offset];
    metal::MetalBuffer source = backend_->wrapSharedMemory(
        mapping->address, length, mapping, "gguf/" + image.name);
    std::vector<KQRepackParams> repackParams;
    std::vector<KQCopyParams> copyParams;
    repackParams.reserve(image.repacks.size());
    copyParams.reserve(image.copies.size());
    std::vector<metal::ComputeDispatch> dispatches;
    for (const gguf::Repack &repack : image.repacks) {
      KQRepackParams params = repack.params;
      params.src_offset = static_cast<uint32_t>(repack.sourceOffset - base);
      repackParams.push_back(params);
      const uint64_t threads = uint64_t{params.rows} * (params.input_size / 32);
      dispatches.push_back({"kq_repack", {{0, source}, {1, buffer}},
                            {{2, &repackParams.back(), sizeof(KQRepackParams)}},
                            {(threads + 255) / 256, 1, 1}, {256, 1, 1}});
    }
    for (const gguf::Copy &copy : image.copies) {
      KQCopyParams params = copy.params;
      params.src_offset = static_cast<uint32_t>(copy.sourceOffset - base);
      copyParams.push_back(params);
      dispatches.push_back({"kq_copy", {{0, source}, {1, buffer}},
                            {{2, &copyParams.back(), sizeof(KQCopyParams)}},
                            {(uint64_t{params.bytes} / 16 + 255) / 256, 1, 1}, {256, 1, 1}});
    }
    static_cast<void>(backend_->submitCommand(dispatches));
  }
  return WeightFile(*backend_, std::move(buffer), "target/" + image.name, kKQuantMagic,
                    expectedLayer, expectedType);
}

} // namespace splash::model
