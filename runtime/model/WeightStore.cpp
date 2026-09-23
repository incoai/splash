#include "WeightStore.hpp"

#include "metal/abi/Gguf.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sstream>
#include <system_error>
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

[[nodiscard]] uint64_t q8PackedBytes(uint32_t outputSize, uint32_t inputSize) {
    uint64_t elements = q4Elements(outputSize, inputSize);
    return checkedWeightAdd(
        elements,
        checkedWeightMultiply(elements / 32, 2, "Q8 parameter byte count"),
        "Q8 packed byte count");
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

uint64_t alignPacked(uint64_t value) {
    return checkedWeightAdd(value, kWeightFileAlignment - 1,
                            "packed file alignment") &
           ~(kWeightFileAlignment - 1);
}

uint32_t loadLittleEndian32(const uint8_t *bytes) {
    return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
        (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
}

std::string systemError(std::string_view operation,
                        const std::filesystem::path &path, int error) {
    return std::string(operation) + " " + path.string() + ": " +
        std::error_code(error, std::generic_category()).message();
}

class MappedRegion final {
public:
    static std::shared_ptr<MappedRegion> openReadOnly(
        const std::filesystem::path &path) {
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
        uint64_t bytes = static_cast<uint64_t>(status.st_size);
        if (bytes > std::numeric_limits<size_t>::max()) {
            close(descriptor);
            throw WeightStoreError("packed file is too large to map: " +
                                   path.string());
        }

        // Metal can materialize MAP_PRIVATE file mappings as anonymous dirty
        // pages on GPU use. Keep immutable weights file-backed and reclaimable.
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
    std::shared_ptr<MappedRegion> mapping; // null for in-memory images
    metal::MetalBuffer base;
    uint64_t bytes = 0;
    WeightFileRecord record;
    uint64_t offset = 16;
    bool finished = false;
};

namespace {
void checkWeightHeader(const uint8_t *header, uint64_t bytes, std::string_view expectedMagic,
                       uint32_t expectedLayer, uint32_t expectedType,
                       const std::string &what) {
    if (expectedMagic.size() != 8) {
        throw WeightStoreError("packed file magic must contain eight bytes");
    }
    if (bytes < 16 || bytes % kWeightFileAlignment) {
        throw WeightStoreError("packed file size is not 16 KiB-aligned: " + what);
    }
    uint32_t layer = loadLittleEndian32(header + 8);
    uint32_t type = loadLittleEndian32(header + 12);
    if (std::memcmp(header, expectedMagic.data(), 8) != 0 ||
        layer != expectedLayer || type != expectedType) {
        throw WeightStoreError("packed file header mismatch: " + what);
    }
}
} // namespace

WeightFile::WeightFile(metal::MetalBackend &backend,
                       std::filesystem::path path,
                       std::string relativePath,
                       std::string_view expectedMagic,
                       uint32_t expectedLayer,
                       uint32_t expectedType)
    : impl_(std::make_unique<Impl>()) {
    impl_->backend = &backend;
    impl_->mapping = MappedRegion::openReadOnly(path);
    impl_->bytes = impl_->mapping->bytes();
    checkWeightHeader(static_cast<const uint8_t *>(impl_->mapping->address()),
                      impl_->bytes, expectedMagic, expectedLayer, expectedType,
                      path.string());
    impl_->record = {
        std::move(relativePath), std::string(expectedMagic), expectedLayer, expectedType,
        impl_->bytes,
    };
    impl_->base = backend.wrapSharedMemory(
        impl_->mapping->address(), impl_->bytes, impl_->mapping,
        impl_->record.relativePath);
}

WeightFile::WeightFile(metal::MetalBackend &backend, metal::MetalBuffer image,
                       std::string relativePath, std::string_view expectedMagic,
                       uint32_t expectedLayer, uint32_t expectedType)
    : impl_(std::make_unique<Impl>()) {
    impl_->backend = &backend;
    impl_->bytes = image.sizeBytes();
    const auto *header = static_cast<const uint8_t *>(image.contents());
    if (!header) throw WeightStoreError("weight image is not host visible: " + relativePath);
    checkWeightHeader(header, impl_->bytes, expectedMagic, expectedLayer, expectedType,
                      relativePath);
    impl_->record = {
        std::move(relativePath), std::string(expectedMagic), expectedLayer, expectedType,
        impl_->bytes,
    };
    impl_->base = std::move(image);
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

ops::Q4Projection readQ4Projection(WeightFile &file,
                                   metal::MetalBackend &backend,
                                   uint32_t outputSize,
                                   uint32_t inputSize,
                                   std::string_view label) {
    validateQ4Layout(outputSize, inputSize);
    const uint64_t elements = q4Elements(outputSize, inputSize);
    const uint64_t weightBytes = elements / 2;
    const uint64_t parameterBytes = elements / 32;
    metal::MetalBuffer packed =
        file.section(q4PackedBytes(outputSize, inputSize), label);
    ops::Q4Projection result;
    result.weights = backend.view(packed, 0, weightBytes);
    result.scales = backend.view(packed, weightBytes, parameterBytes);
    result.biases =
        backend.view(packed, weightBytes + parameterBytes, parameterBytes);
    result.outputSize = outputSize;
    result.inputSize = inputSize;
    return result;
}

ops::Q4Projection readQ4ProjectionComponents(WeightFile &file,
                                             uint32_t outputSize,
                                             uint32_t inputSize,
                                             std::string_view label) {
    const uint64_t elements = q4Elements(outputSize, inputSize);
    const std::string prefix(label);
    ops::Q4Projection result;
    result.weights = file.section(elements / 2, prefix + "-weights");
    result.scales = file.section(elements / 32, prefix + "-scales");
    result.biases = file.section(elements / 32, prefix + "-biases");
    result.outputSize = outputSize;
    result.inputSize = inputSize;
    return result;
}

ops::NormWeights readNorm(WeightFile &file, uint32_t width, bool float32,
                          std::string_view label) {
    ops::NormWeights norm{{}, float32};
    norm.buffer = file.section(norm.bytes(width), label);
    return norm;
}

namespace {
struct GgufDescriptor {
    uint32_t type, outputSize, inputSize, p0, p1, metaBytes, metaGroups;
    uint64_t plane0Bytes, plane1Bytes, metaTotalBytes;
};
GgufDescriptor readGgufDescriptor(WeightFile &file, std::string_view label) {
    metal::MetalBuffer section = file.section(64, std::string(label) + "-desc");
    const uint8_t *bytes = static_cast<const uint8_t *>(section.contents());
    if (!bytes) throw WeightStoreError("GGUF descriptor is not host visible");
    GgufDescriptor d{};
    uint32_t words[7];
    std::memcpy(words, bytes, sizeof words);
    d.type = words[0]; d.outputSize = words[1]; d.inputSize = words[2]; d.p0 = words[3];
    d.p1 = words[4]; d.metaBytes = words[5]; d.metaGroups = words[6];
    std::memcpy(&d.plane0Bytes, bytes + 32, 8);
    std::memcpy(&d.plane1Bytes, bytes + 40, 8);
    std::memcpy(&d.metaTotalBytes, bytes + 48, 8);
    // Float tensors are rows as stored; quantized ones fill 256-column tiles.
    if (!d.outputSize || !d.inputSize ||
        (d.type != GGUF_TYPE_F32 && (d.outputSize % kQ4StorageN || d.inputSize % 256)))
        throw WeightStoreError("GGUF tensor shape is not tile aligned: " + std::string(label));
    return d;
}
} // namespace

ops::GgufSegment readGgufSegment(WeightFile &file, std::string_view label) {
    const GgufDescriptor d = readGgufDescriptor(file, label);
    if (d.type == GGUF_TYPE_F32) {
        if (d.p0 || d.p1 || d.metaBytes || d.metaGroups || d.plane1Bytes || d.metaTotalBytes ||
            d.plane0Bytes != uint64_t{d.outputSize} * d.inputSize * sizeof(float))
            throw WeightStoreError("GGUF float section sizes are inconsistent: " + std::string(label));
        ops::GgufSegment s;
        s.plane0 = file.section(d.plane0Bytes, std::string(label) + "-floats");
        s.type = d.type; s.outputSize = d.outputSize; s.inputSize = d.inputSize;
        s.formatId = GGUF_FMT_COUNT;
        s.format = "f32";
        return s;
    }
    const uint32_t format = gguf_format_of(d.type);
    if (format == GGUF_FMT_COUNT)
        throw WeightStoreError("unsupported GGUF tensor type " + std::to_string(d.type));
    const QuantFormat &layout = kQuantFormats[format];
    const uint64_t groups = uint64_t{d.inputSize} / 32;
    if (d.p0 != layout.plane0_bytes || d.p1 != layout.plane1_bytes ||
        d.metaBytes != layout.meta_bytes || d.metaGroups != layout.meta_groups ||
        d.plane0Bytes != uint64_t{d.outputSize} * groups * layout.plane0_bytes ||
        d.plane1Bytes != uint64_t{d.outputSize} * groups * layout.plane1_bytes ||
        d.metaTotalBytes != uint64_t{d.outputSize} * (groups / layout.meta_groups) * layout.meta_bytes)
        throw WeightStoreError("GGUF section sizes are inconsistent: " + std::string(label));
    ops::GgufSegment s;
    s.plane0 = file.section(d.plane0Bytes, std::string(label) + "-plane0");
    if (d.plane1Bytes) s.plane1 = file.section(d.plane1Bytes, std::string(label) + "-plane1");
    s.meta = file.section(d.metaTotalBytes, std::string(label) + "-meta");
    s.type = d.type; s.outputSize = d.outputSize; s.inputSize = d.inputSize;
    s.p0 = layout.plane0_bytes; s.p1 = layout.plane1_bytes;
    s.metaBytes = layout.meta_bytes; s.metaGroups = layout.meta_groups;
    s.formatId = format;
    s.format = layout.name;
    return s;
}

ops::Q4Projection readGgufProjection(WeightFile &file, std::string_view label) {
    ops::Q4Projection p;
    p.gguf.push_back(readGgufSegment(file, label));
    p.outputSize = p.gguf.front().outputSize;
    p.inputSize = p.gguf.front().inputSize;
    return p;
}

ops::Q4Projection readGgufEmbedding(WeightFile &file, std::string_view label) {
    const GgufDescriptor d = readGgufDescriptor(file, label);
    // Native rows, gathered by gguf_embed_<type>: block_q4_K, block_q6_K or block_q8_0.
    const uint32_t format = gguf_format_of(d.type);
    if ((format != GGUF_FMT_Q4K && format != GGUF_FMT_Q6K && format != GGUF_FMT_Q80) || d.inputSize % 256 ||
        d.plane0Bytes != uint64_t{d.outputSize} * (d.inputSize / kQuantFormats[format].block_elements) *
                             kQuantFormats[format].block_bytes)
        throw WeightStoreError("GGUF embedding must be native block_q4_K, block_q6_K or block_q8_0 rows");
    ops::Q4Projection p;
    ops::GgufSegment s;
    s.plane0 = file.section(d.plane0Bytes, std::string(label) + "-native");
    s.type = d.type; s.outputSize = d.outputSize; s.inputSize = d.inputSize;
    p.gguf.push_back(std::move(s));
    p.outputSize = d.outputSize;
    p.inputSize = d.inputSize;
    return p;
}

ops::Q8Projection readQ8Projection(WeightFile &file,
                                   metal::MetalBackend &backend,
                                   uint32_t outputSize,
                                   uint32_t inputSize,
                                   std::string_view label) {
    validateQ4Layout(outputSize, inputSize);
    const uint64_t elements = q4Elements(outputSize, inputSize);
    const uint64_t parameterBytes = elements / 32;
    metal::MetalBuffer packed =
        file.section(q8PackedBytes(outputSize, inputSize), label);
    return {
        backend.view(packed, 0, elements),
        backend.view(packed, elements, parameterBytes),
        backend.view(packed, elements + parameterBytes, parameterBytes),
        outputSize,
        inputSize,
    };
}

ops::ExpertQ4Projection
readExpertQ4Projection(WeightFile &file, uint32_t experts,
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
                  << record.type << '\n';
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
