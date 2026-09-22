#include "WeightStore.hpp"

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
    std::shared_ptr<MappedRegion> mapping;
    metal::MetalBuffer base;
    WeightFileRecord record;
    uint64_t offset = 16;
    bool finished = false;
};

WeightFile::WeightFile(metal::MetalBackend &backend,
                       std::filesystem::path path,
                       std::string relativePath,
                       std::string_view expectedMagic,
                       uint32_t expectedLayer,
                       uint32_t expectedType)
    : impl_(std::make_unique<Impl>()) {
    if (expectedMagic.size() != 8) {
        throw WeightStoreError("packed file magic must contain eight bytes");
    }
    impl_->backend = &backend;
    impl_->mapping = MappedRegion::openReadOnly(path);
    if (impl_->mapping->bytes() < 16 ||
        impl_->mapping->bytes() % kWeightFileAlignment) {
        throw WeightStoreError(
            "packed file size is not 16 KiB-aligned: " + path.string());
    }
    const auto *header = static_cast<const uint8_t *>(
        impl_->mapping->address());
    uint32_t layer = loadLittleEndian32(header + 8);
    uint32_t type = loadLittleEndian32(header + 12);
    if (std::memcmp(header, expectedMagic.data(), 8) != 0 ||
        layer != expectedLayer || type != expectedType) {
        throw WeightStoreError("packed file header mismatch: " + path.string());
    }

    impl_->record = {
        std::move(relativePath), std::string(expectedMagic), layer, type,
        impl_->mapping->bytes(),
    };
    impl_->base = backend.wrapSharedMemory(
        impl_->mapping->address(), impl_->mapping->bytes(), impl_->mapping,
        impl_->record.relativePath);
}

WeightFile::~WeightFile() = default;

metal::MetalBuffer WeightFile::section(uint64_t bytes,
                                       std::string_view label) {
    if (impl_->finished) {
        throw WeightStoreError("cannot add a section after packed file finish");
    }
    if (!bytes) throw WeightStoreError("packed section must not be empty");
    uint64_t start = alignPacked(impl_->offset);
    uint64_t end = checkedWeightAdd(start, bytes, "packed section end");
    if (start % kWeightFileAlignment || end > impl_->mapping->bytes()) {
        throw WeightStoreError(
            "packed file is truncated at section " + std::string(label));
    }
    impl_->offset = end;
    return impl_->backend->view(impl_->base, start, bytes);
}

void WeightFile::finish() {
    if (impl_->finished) return;
    uint64_t consumed = alignPacked(impl_->offset);
    if (consumed != impl_->mapping->bytes()) {
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

namespace {
struct KQuantDescriptor {
    uint32_t type, outputSize, inputSize, p0, p1, metaBytes, metaGroups, flags;
    uint64_t plane0Bytes, plane1Bytes, metaTotalBytes;
};
const char *kquantFormatName(uint32_t type) {
    switch (type) {
    case 12: return "q4k";
    case 23: return "iq4xs";
    case 20: return "iq4nl";
    case 13: return "q5k";
    case 14: return "q6k";
    case 11: return "q3k";
    case 8: return "q80";
    case 21: return "iq3s";
    default: throw WeightStoreError("unsupported GGUF tensor type " + std::to_string(type));
    }
}
KQuantDescriptor readKQuantDescriptor(WeightFile &file, std::string_view label) {
    metal::MetalBuffer section = file.section(64, std::string(label) + "-desc");
    const uint8_t *bytes = static_cast<const uint8_t *>(section.contents());
    if (!bytes) throw WeightStoreError("K-quant descriptor is not host visible");
    KQuantDescriptor d{};
    uint32_t words[8];
    std::memcpy(words, bytes, sizeof words);
    d.type = words[0]; d.outputSize = words[1]; d.inputSize = words[2]; d.p0 = words[3];
    d.p1 = words[4]; d.metaBytes = words[5]; d.metaGroups = words[6]; d.flags = words[7];
    std::memcpy(&d.plane0Bytes, bytes + 32, 8);
    std::memcpy(&d.plane1Bytes, bytes + 40, 8);
    std::memcpy(&d.metaTotalBytes, bytes + 48, 8);
    if (!d.outputSize || !d.inputSize || d.outputSize % kQ4StorageN || d.inputSize % 256)
        throw WeightStoreError("K-quant tensor shape is not tile aligned: " + std::string(label));
    return d;
}
} // namespace

ops::KQuantSegment readKQuantSegment(WeightFile &file, std::string_view label) {
    const KQuantDescriptor d = readKQuantDescriptor(file, label);
    const uint64_t groups = uint64_t{d.inputSize} / 32;
    if (d.plane0Bytes != uint64_t{d.outputSize} * groups * d.p0 ||
        d.plane1Bytes != uint64_t{d.outputSize} * groups * d.p1 ||
        d.metaTotalBytes != uint64_t{d.outputSize} * (groups / d.metaGroups) * d.metaBytes)
        throw WeightStoreError("K-quant section sizes are inconsistent: " + std::string(label));
    ops::KQuantSegment s;
    s.plane0 = file.section(d.plane0Bytes, std::string(label) + "-plane0");
    if (d.plane1Bytes) s.plane1 = file.section(d.plane1Bytes, std::string(label) + "-plane1");
    s.meta = file.section(d.metaTotalBytes, std::string(label) + "-meta");
    s.type = d.type; s.outputSize = d.outputSize; s.inputSize = d.inputSize;
    s.p0 = d.p0; s.p1 = d.p1; s.metaBytes = d.metaBytes; s.metaGroups = d.metaGroups;
    s.format = kquantFormatName(d.type);
    switch (d.type) {
    case 12: s.formatId = 0; break; case 23: s.formatId = 1; break; case 20: s.formatId = 2; break; case 13: s.formatId = 3; break;
    case 14: s.formatId = 4; break; case 11: s.formatId = 5; break; case 8: s.formatId = 6; break; default: s.formatId = 7; break;
    }
    return s;
}

ops::Q4Projection readKQuantProjection(WeightFile &file, std::string_view label) {
    ops::Q4Projection p;
    p.kq.push_back(readKQuantSegment(file, label));
    p.outputSize = p.kq.front().outputSize;
    p.inputSize = p.kq.front().inputSize;
    return p;
}

ops::Q4Projection readKQuantEmbedding(WeightFile &file, std::string_view label) {
    const KQuantDescriptor d = readKQuantDescriptor(file, label);
    // Native rows, gathered by kq_embed_<type>: block_q4_K (144 B / 256), block_q6_K
    // (210 B / 256) or block_q8_0 (34 B / 32).
    const char *format = nullptr;
    uint64_t rowBytes = 0;
    if (d.type == 12) { format = "q4k-native"; rowBytes = uint64_t{d.inputSize / 256} * 144; }
    else if (d.type == 14) { format = "q6k-native"; rowBytes = uint64_t{d.inputSize / 256} * 210; }
    else if (d.type == 8) { format = "q80-native"; rowBytes = uint64_t{d.inputSize / 32} * 34; }
    if (!format || d.inputSize % 256 || d.plane0Bytes != uint64_t{d.outputSize} * rowBytes)
        throw WeightStoreError("K-quant embedding must be native block_q4_K, block_q6_K or block_q8_0 rows");
    ops::Q4Projection p;
    ops::KQuantSegment s;
    s.plane0 = file.section(d.plane0Bytes, std::string(label) + "-native");
    s.type = d.type; s.outputSize = d.outputSize; s.inputSize = d.inputSize; s.format = format;
    p.kq.push_back(std::move(s));
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
