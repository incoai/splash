#include "WeightStore.hpp"

#include "Checked.hpp"
#include "metal/abi/Gguf.h"
#include "model/GgufFile.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/PreparedWeights.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <system_error>
#include <tuple>
#include <utility>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace splash::model {

static_assert(kWeightFileAlignment == kHostPageBytes, "weight files are mapped at host page boundaries");

namespace {

[[nodiscard]] uint64_t q4Elements(uint32_t outputSize, uint32_t inputSize) {
    if (!outputSize || !inputSize || inputSize % kQ4GroupElements) {
        throw WeightStoreError(
            "Q4 projection dimensions must be positive and input-aligned");
    }
    return checkedMultiply<WeightStoreError>(outputSize, inputSize, "Q4 element count");
}

} // namespace

uint64_t q4PackedBytes(uint32_t outputSize, uint32_t inputSize) {
    uint64_t elements = q4Elements(outputSize, inputSize);
    return checkedMultiply<WeightStoreError>(elements / 16, 9, "Q4 packed byte count");
}

void validateQ4Layout(uint32_t outputSize, uint32_t inputSize) {
    static_cast<void>(q4Elements(outputSize, inputSize));
    if (outputSize % kQ4StorageN) {
        throw WeightStoreError(
            "Q4 output dimension is incompatible with StorageN=256");
    }
}

namespace {

// The header weightFileHeader writes, which the first section follows.
constexpr uint64_t kHeaderBytes = std::tuple_size_v<decltype(weightFileHeader({}, 0, 0))>;

uint64_t alignPacked(uint64_t value) {
    static_cast<void>(
        checkedAdd<WeightStoreError>(value, kWeightFileAlignment - 1, "packed file alignment"));
    return alignWeightOffset(value);
}

std::string systemError(std::string_view operation,
                        const std::filesystem::path &path, int error) {
    return std::string(operation) + " " + path.string() + ": " +
        std::error_code(error, std::generic_category()).message();
}

static_assert(sizeof(size_t) == sizeof(uint64_t), "weight files are mapped whole");

class MappedRegion final {
public:
    // A prepared file is mapped only as the cache verified it.
    static std::shared_ptr<MappedRegion> openReadOnly(
        const std::filesystem::path &path, bool prepared) {
        int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            throw WeightStoreError(systemError("unable to open", path, errno));
        }

        struct stat status {};
        if (fstat(descriptor, &status) != 0) {
            int error = errno;
            close(descriptor);
            throw WeightStoreError(systemError("unable to stat", path, error));
        }
        if (!S_ISREG(status.st_mode) || status.st_size <= 0) {
            close(descriptor);
            throw WeightStoreError("packed file is not a non-empty regular file: " +
                                   path.string());
        }
        if (prepared) {
            try {
                requireVerifiedFile(descriptor, path);
            } catch (...) {
                close(descriptor);
                throw;
            }
        }
        uint64_t bytes = static_cast<uint64_t>(status.st_size);

        // Metal can materialize MAP_PRIVATE file mappings as anonymous dirty
        // pages on GPU use. Preserve file backing; pages held resident by Metal
        // are still wired and cannot be reclaimed until that residency ends.
        void *address = mmap(nullptr, static_cast<size_t>(bytes), PROT_READ,
                             MAP_SHARED, descriptor, 0);
        int mapError = errno;
        close(descriptor);
        if (address == MAP_FAILED) {
            throw WeightStoreError(
                systemError("unable to mmap", path, mapError));
        }
        return std::shared_ptr<MappedRegion>(
            new MappedRegion(address, bytes));
    }

    ~MappedRegion() {
        if (address_) {
            munmap(address_, static_cast<size_t>(bytes_));
        }
    }

    MappedRegion(const MappedRegion &) = delete;
    MappedRegion &operator=(const MappedRegion &) = delete;

    [[nodiscard]] void *address() const noexcept { return address_; }
    [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }

private:
    MappedRegion(void *address, uint64_t bytes)
        : address_(address), bytes_(bytes) {}

    void *address_ = nullptr;
    uint64_t bytes_ = 0;
};

} // namespace

struct WeightFile::Impl {
    metal::MetalBackend *backend = nullptr;
    std::shared_ptr<MappedRegion> mapping;
    metal::MetalBuffer base;
    WeightFileRecord record;
    uint64_t offset = kHeaderBytes;
};

namespace {
void checkWeightHeader(const uint8_t *header, uint64_t bytes, std::string_view expectedMagic,
                       uint32_t expectedLayer, uint32_t expectedType,
                       const std::string &what) {
    const auto expected = weightFileHeader(expectedMagic, expectedLayer, expectedType);
    if (bytes < expected.size() || bytes % kWeightFileAlignment) {
        throw WeightStoreError("packed file size is not 16 KiB-aligned: " + what);
    }
    if (std::memcmp(header, expected.data(), expected.size()) != 0) {
        throw WeightStoreError("packed file header mismatch: " + what);
    }
}
} // namespace

WeightFile::WeightFile(metal::MetalBackend &backend,
                       std::filesystem::path path,
                       std::string relativePath,
                       std::string_view expectedMagic,
                       uint32_t expectedLayer,
                       uint32_t expectedType, std::string contentIdentity)
    : impl_(std::make_unique<Impl>()) {
    impl_->backend = &backend;
    impl_->mapping = MappedRegion::openReadOnly(path, !contentIdentity.empty());
    checkWeightHeader(static_cast<const uint8_t *>(impl_->mapping->address()),
                      impl_->mapping->bytes(), expectedMagic, expectedLayer, expectedType,
                      path.string());
    impl_->record = {
        std::move(relativePath), std::string(expectedMagic), expectedLayer, expectedType,
        impl_->mapping->bytes(), std::move(contentIdentity),
    };
    impl_->base = backend.wrapSharedMemory(
        impl_->mapping->address(), impl_->mapping->bytes(), impl_->mapping,
        impl_->record.relativePath);
}

WeightFile::WeightFile(WeightFile &&) noexcept = default;
WeightFile &WeightFile::operator=(WeightFile &&) noexcept = default;

WeightFile::~WeightFile() = default;

metal::MetalBuffer WeightFile::section(uint64_t bytes,
                                       std::string_view label) {
    if (!bytes) throw WeightStoreError("packed section must not be empty");
    uint64_t start = alignPacked(impl_->offset);
    uint64_t end = checkedAdd<WeightStoreError>(start, bytes, "packed section end");
    if (start % kWeightFileAlignment || end > impl_->mapping->bytes()) {
        throw WeightStoreError(
            "packed file is truncated at section " + std::string(label));
    }
    impl_->offset = end;
    return impl_->backend->view(impl_->base, start, bytes);
}

std::vector<metal::MetalBuffer> WeightFile::split(std::initializer_list<uint64_t> parts,
                                                  std::string_view label) {
    uint64_t bytes = 0;
    for (uint64_t part : parts) bytes = checkedAdd<WeightStoreError>(bytes, part, "packed section size");
    const metal::MetalBuffer whole = section(bytes, label);
    std::vector<metal::MetalBuffer> views;
    uint64_t offset = 0;
    for (uint64_t part : parts) {
        views.push_back(impl_->backend->view(whole, offset, part));
        offset += part;
    }
    return views;
}

void WeightFile::finish() {
    uint64_t consumed = alignPacked(impl_->offset);
    if (consumed != impl_->mapping->bytes()) {
        throw WeightStoreError(
            "packed file has unconsumed or missing bytes: " +
            impl_->record.relativePath);
    }
}

const WeightFileRecord &WeightFile::record() const noexcept {
    return impl_->record;
}

ops::Projection readAffineProjection(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                     std::string_view label) {
    validateQ4Layout(outputSize, inputSize);
    const uint64_t elements = q4Elements(outputSize, inputSize);
    const std::vector<metal::MetalBuffer> planes =
        file.split({elements / 2, elements / 32, elements / 32}, label);
    return {outputSize, inputSize, ops::AffineWeights{planes[0], planes[1], planes[2]}};
}

ops::EmbeddingWeights readAffineEmbedding(WeightFile &file,
                                             uint32_t outputSize,
                                             uint32_t inputSize,
                                             std::string_view label) {
    const uint64_t elements = q4Elements(outputSize, inputSize);
    const std::string prefix(label);
    // Braced initializers read the sections in file order.
    return {outputSize, inputSize,
            ops::AffineWeights{
                file.section(elements / 2, prefix + "-weights"),
                file.section(elements / 32, prefix + "-scales"),
                file.section(elements / 32, prefix + "-biases"),
            }};
}

ops::NormWeights readNorm(WeightFile &file, uint32_t width, bool float32,
                          std::string_view label) {
    ops::NormWeights norm{{}, float32};
    norm.buffer = file.section(norm.bytes(width), label);
    return norm;
}

namespace {
GgufTensorDescriptor readGgufDescriptor(WeightFile &file, std::string_view label) {
    metal::MetalBuffer section = file.section(sizeof(GgufTensorDescriptor), std::string(label) + "-desc");
    const uint8_t *bytes = static_cast<const uint8_t *>(section.contents());
    if (!bytes) throw WeightStoreError("GGUF descriptor is not host visible");
    GgufTensorDescriptor d;
    std::memcpy(&d, bytes, sizeof d);
    // Float tensors are rows as stored; quantized ones fill whole tiles.
    if (!d.outputSize || !d.inputSize ||
        (d.type != ggml::kF32 && (d.outputSize % QUANT_TILE_ROWS || d.inputSize % kGgufBlockColumns)))
        throw WeightStoreError("GGUF tensor shape is not tile aligned: " + std::string(label));
    return d;
}
} // namespace

ops::QuantizedSegment readQuantizedSegment(WeightFile &file, std::string_view label) {
    const GgufTensorDescriptor d = readGgufDescriptor(file, label);
    if (d.type == ggml::kF32) {
        if (d.p0 || d.p1 || d.metaBytes || d.metaGroups || d.plane1Bytes || d.metaTotalBytes ||
            d.plane0Bytes != uint64_t{d.outputSize} * d.inputSize * sizeof(float))
            throw WeightStoreError("GGUF float section sizes are inconsistent: " + std::string(label));
        return ops::QuantizedSegment::floats(d.outputSize, d.inputSize,
                                             file.section(d.plane0Bytes, std::string(label) + "-floats"));
    }
    const uint32_t format = gguf_format_of(d.type);
    if (format == GGUF_FMT_COUNT)
        throw WeightStoreError("unsupported GGUF tensor type " + std::to_string(d.type));
    const QuantFormat &layout = kQuantFormats[format];
    const GgufPlaneBytes planes = ggufPlaneBytes(layout, d.outputSize, d.inputSize);
    if (d.p0 != layout.plane0_bytes || d.p1 != layout.plane1_bytes ||
        d.metaBytes != layout.meta_bytes || d.metaGroups != layout.meta_groups ||
        d.plane0Bytes != planes.plane0 || d.plane1Bytes != planes.plane1 || d.metaTotalBytes != planes.meta)
        throw WeightStoreError("GGUF section sizes are inconsistent: " + std::string(label));
    metal::MetalBuffer plane0 = file.section(d.plane0Bytes, std::string(label) + "-plane0");
    metal::MetalBuffer plane1 =
        d.plane1Bytes ? file.section(d.plane1Bytes, std::string(label) + "-plane1") : metal::MetalBuffer{};
    metal::MetalBuffer meta = file.section(d.metaTotalBytes, std::string(label) + "-meta");
    return ops::QuantizedSegment::planes(format, d.outputSize, d.inputSize, std::move(plane0),
                                         std::move(plane1), std::move(meta));
}

ops::Projection readBlockProjection(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                    std::string_view label) {
    ops::QuantizedSegment segment = readQuantizedSegment(file, label);
    if (segment.outputSize != outputSize || segment.inputSize != inputSize)
        throw WeightStoreError("GGUF tensor does not match the layout: " + std::string(label));
    return {outputSize, inputSize, ops::BlockWeights{{std::move(segment)}}};
}

ops::EmbeddingWeights readBlockEmbedding(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                         std::string_view label) {
    const GgufTensorDescriptor d = readGgufDescriptor(file, label);
    if (d.outputSize != outputSize || d.inputSize != inputSize)
        throw WeightStoreError("GGUF embedding does not match the layout: " + std::string(label));
    const uint32_t format = gguf_format_of(d.type);
    if (format == GGUF_FMT_COUNT ||
        d.plane0Bytes != d.outputSize * ggufRowBytes(kQuantFormats[format], d.inputSize))
        throw WeightStoreError("GGUF embedding rows are not native GGUF blocks: " + std::string(label));
    return {outputSize, inputSize,
            ops::NativeRows(file.section(d.plane0Bytes, std::string(label) + "-native"), format)};
}

ops::Q8Projection readAffineQ8Projection(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                         std::string_view label) {
    validateQ4Layout(outputSize, inputSize);
    const uint64_t elements = q4Elements(outputSize, inputSize);
    const std::vector<metal::MetalBuffer> planes = file.split({elements, elements / 32, elements / 32}, label);
    return {{planes[0], planes[1], planes[2]}, outputSize, inputSize};
}

ops::ExpertProjection
readAffineExpertProjection(WeightFile &file, uint32_t experts,
                           uint32_t outputSize, uint32_t inputSize,
                           std::string_view label) {
    if (!experts)
        throw WeightStoreError("expert projection requires experts");
    validateQ4Layout(outputSize, inputSize);
    const uint64_t stride = q4PackedBytes(outputSize, inputSize);
    return {
        file.section(checkedMultiply<WeightStoreError>(experts, stride,
                                                       "expert Q4 slab bytes"),
                     label),
        experts,
        outputSize,
        inputSize,
        stride,
    };
}

std::string weightManifestFingerprint(
    std::span<const WeightFileRecord> records) {
    std::vector<WeightFileRecord> sorted(records.begin(), records.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const WeightFileRecord &left,
                 const WeightFileRecord &right) {
                  return left.relativePath < right.relativePath;
              });
    std::ostringstream canonical;
    canonical << "splash-packed-manifest-v1\n";
    for (const WeightFileRecord &record : sorted) {
        canonical << record.relativePath << '\t' << record.declaredBytes
                  << '\t' << record.magic << '\t' << record.layer << '\t'
                  << record.type;
        if (!record.contentIdentity.empty()) canonical << '\t' << record.contentIdentity;
        canonical << '\n';
    }
    return weightDigest(canonical.str());
}

} // namespace splash::model
