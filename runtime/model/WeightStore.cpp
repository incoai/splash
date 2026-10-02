#include "WeightStore.hpp"

#include "metal/abi/Gguf.h"
#include "model/GgufImageLayout.hpp"
#include "model/PreparedWeights.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sstream>
#include <system_error>
#include <tuple>
#include <utility>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace splash::model {

uint64_t checkedWeightMultiply(uint64_t left, uint64_t right,
                         std::string_view description) {
    if (left && right > std::numeric_limits<uint64_t>::max() / left) {
        throw WeightStoreError(std::string(description) + " overflows");
    }
    return left * right;
}

namespace {

[[nodiscard]] uint64_t checkedWeightAdd(uint64_t left, uint64_t right,
                                        std::string_view description) {
    if (left > std::numeric_limits<uint64_t>::max() - right) {
        throw WeightStoreError(std::string(description) + " overflows");
    }
    return left + right;
}

[[nodiscard]] uint64_t q4Elements(uint32_t outputSize, uint32_t inputSize) {
    if (!outputSize || !inputSize || inputSize % kQ4GroupElements) {
        throw WeightStoreError(
            "Q4 projection dimensions must be positive and input-aligned");
    }
    return checkedWeightMultiply(outputSize, inputSize, "Q4 element count");
}

} // namespace

uint64_t q4PackedBytes(uint32_t outputSize, uint32_t inputSize) {
    uint64_t elements = q4Elements(outputSize, inputSize);
    return checkedWeightMultiply(elements / 16, 9, "Q4 packed byte count");
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
    static_cast<void>(checkedWeightAdd(value, kWeightFileAlignment - 1, "packed file alignment"));
    return alignWeightOffset(value);
}

std::string systemError(std::string_view operation,
                        const std::filesystem::path &path, int error) {
    return std::string(operation) + " " + path.string() + ": " +
        std::error_code(error, std::generic_category()).message();
}

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
        if (bytes > std::numeric_limits<size_t>::max()) {
            close(descriptor);
            throw WeightStoreError("packed file is too large to map: " +
                                   path.string());
        }

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
    uint64_t bytes = 0;
    WeightFileRecord record;
    uint64_t offset = kHeaderBytes;
    bool finished = false;
};

namespace {
void checkWeightHeader(const uint8_t *header, uint64_t bytes, std::string_view expectedMagic,
                       uint32_t expectedLayer, uint32_t expectedType,
                       const std::string &what) {
    if (expectedMagic.size() != 8) {
        throw WeightStoreError("packed file magic must contain eight bytes");
    }
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
    impl_->bytes = impl_->mapping->bytes();
    checkWeightHeader(static_cast<const uint8_t *>(impl_->mapping->address()),
                      impl_->bytes, expectedMagic, expectedLayer, expectedType,
                      path.string());
    impl_->record = {
        std::move(relativePath), std::string(expectedMagic), expectedLayer, expectedType,
        impl_->bytes, std::move(contentIdentity),
    };
    impl_->base = backend.wrapSharedMemory(
        impl_->mapping->address(), impl_->bytes, impl_->mapping,
        impl_->record.relativePath);
    // The weights outlive this loader: the base stays resident until its last
    // view is gone.
    backend.keepResident(impl_->base);
}

WeightFile::WeightFile(WeightFile &&) noexcept = default;
WeightFile &WeightFile::operator=(WeightFile &&) noexcept = default;

WeightFile::~WeightFile() = default;

metal::MetalBuffer WeightFile::section(uint64_t bytes,
                                       std::string_view label) {
    if (impl_->finished) {
        throw WeightStoreError("cannot add a section after packed file finish");
    }
    if (!bytes) throw WeightStoreError("packed section must not be empty");
    uint64_t start = alignPacked(impl_->offset);
    uint64_t end = checkedWeightAdd(start, bytes, "packed section end");
    if (start % kWeightFileAlignment || end > impl_->bytes) {
        throw WeightStoreError(
            "packed file is truncated at section " + std::string(label));
    }
    impl_->offset = end;
    return impl_->backend->view(impl_->base, start, bytes);
}

std::vector<metal::MetalBuffer> WeightFile::split(std::initializer_list<uint64_t> parts,
                                                  std::string_view label) {
    uint64_t bytes = 0;
    for (uint64_t part : parts) bytes = checkedWeightAdd(bytes, part, "packed section size");
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
    if (impl_->finished) return;
    uint64_t consumed = alignPacked(impl_->offset);
    if (consumed != impl_->bytes) {
        throw WeightStoreError(
            "packed file has unconsumed or missing bytes: " +
            impl_->record.relativePath);
    }
    impl_->finished = true;
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
// A table's native rows (`tiled` false) need not fill whole tiles: the
// gather reads any row of whole blocks.
GgufTensorDescriptor readGgufDescriptor(WeightFile &file, std::string_view label, bool tiled = true) {
    metal::MetalBuffer section = file.section(sizeof(GgufTensorDescriptor), std::string(label) + "-desc");
    const uint8_t *bytes = static_cast<const uint8_t *>(section.contents());
    if (!bytes) throw WeightStoreError("GGUF descriptor is not host visible");
    GgufTensorDescriptor d;
    std::memcpy(&d, bytes, sizeof d);
    // Float tensors are rows as stored; quantized ones fill whole tiles.
    if (!d.outputSize || !d.inputSize ||
        (tiled && d.type != GGUF_TYPE_F32 && (d.outputSize % QUANT_TILE_ROWS || d.inputSize % quant_column_unit(32))))
        throw WeightStoreError("GGUF tensor shape is not tile aligned: " + std::string(label));
    return d;
}
} // namespace

ops::QuantizedSegment readQuantizedSegment(WeightFile &file, std::string_view label) {
    const GgufTensorDescriptor d = readGgufDescriptor(file, label);
    if (d.type == GGUF_TYPE_F32) {
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
    const GgufTensorDescriptor d = readGgufDescriptor(file, label, false);
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
        file.section(checkedWeightMultiply(experts, stride,
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
    std::string value = canonical.str();
    if (value.size() > std::numeric_limits<CC_LONG>::max()) {
        throw WeightStoreError("manifest is too large to fingerprint");
    }
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    if (!CC_SHA256(value.data(), static_cast<CC_LONG>(value.size()),
                   digest.data())) {
        throw WeightStoreError("unable to calculate manifest SHA-256");
    }
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(digest.size() * 2);
    for (unsigned char byte : digest) {
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 0x0f]);
    }
    return result;
}

} // namespace splash::model
