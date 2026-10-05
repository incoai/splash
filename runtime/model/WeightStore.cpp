#include "WeightStore.hpp"

#include "Checked.hpp"
#include "metal/abi/Gguf.h"
#include "model/GgufFile.hpp"
#include "model/GgufImageLayout.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <tuple>
#include <utility>

namespace splash::model {

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
        throw WeightStoreError("Q4 output dimension is not a whole number of " + std::to_string(kQ4StorageN) +
                               "-row storage tiles");
    }
}

namespace {

// The header weightFileHeader writes, which the first section follows.
constexpr uint64_t kHeaderBytes = std::tuple_size_v<decltype(weightFileHeader({}, 0, 0))>;

// Where the section after `offset` starts: the next section boundary.
uint64_t sectionStart(uint64_t offset) {
    static_cast<void>(
        checkedAdd<WeightStoreError>(offset, kWeightFileAlignment - 1, "weight image section offset"));
    return alignWeightOffset(offset);
}

} // namespace

struct WeightFile::Impl {
    metal::MetalBackend *backend = nullptr;
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
        throw WeightStoreError("weight image size is not 16 KiB-aligned: " + what);
    }
    if (std::memcmp(header, expected.data(), expected.size()) != 0) {
        throw WeightStoreError("weight image header mismatch: " + what);
    }
}
} // namespace

WeightFile::WeightFile(metal::MetalBackend &backend, metal::MetalBuffer image,
                       std::string relativePath, std::string_view expectedMagic,
                       uint32_t expectedLayer, uint32_t expectedType,
                       std::string contentIdentity)
    : impl_(std::make_unique<Impl>()) {
    const auto *header = static_cast<const uint8_t *>(image.contents());
    if (!header) throw WeightStoreError("weight image is not host visible: " + relativePath);
    checkWeightHeader(header, image.sizeBytes(), expectedMagic, expectedLayer, expectedType, relativePath);
    impl_->backend = &backend;
    impl_->record = {std::move(relativePath), std::string(expectedMagic), expectedLayer, expectedType,
                     image.sizeBytes(), std::move(contentIdentity)};
    impl_->base = std::move(image);
}

WeightFile::WeightFile(WeightFile &&) noexcept = default;
WeightFile &WeightFile::operator=(WeightFile &&) noexcept = default;

WeightFile::~WeightFile() = default;

metal::MetalBuffer WeightFile::section(uint64_t bytes,
                                       std::string_view label) {
    if (!bytes) throw WeightStoreError("weight image section must not be empty");
    uint64_t start = sectionStart(impl_->offset);
    uint64_t end = checkedAdd<WeightStoreError>(start, bytes, "weight image section end");
    if (start % kWeightFileAlignment || end > impl_->base.sizeBytes()) {
        throw WeightStoreError("weight image " + impl_->record.relativePath +
                               " is truncated at section " + std::string(label));
    }
    impl_->offset = end;
    return impl_->backend->view(impl_->base, start, bytes);
}

std::vector<metal::MetalBuffer> WeightFile::split(std::initializer_list<uint64_t> parts,
                                                  std::string_view label) {
    uint64_t bytes = 0;
    for (uint64_t part : parts) bytes = checkedAdd<WeightStoreError>(bytes, part, "weight image section size");
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
    uint64_t consumed = sectionStart(impl_->offset);
    if (consumed != impl_->base.sizeBytes()) {
        throw WeightStoreError(
            "weight image has unconsumed or missing bytes: " +
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
    metal::MetalBuffer section = file.section(GgufTensorDescriptor::kBytes, std::string(label) + "-desc");
    const uint8_t *bytes = static_cast<const uint8_t *>(section.contents());
    if (!bytes) throw WeightStoreError("GGUF descriptor is not host visible");
    const GgufTensorDescriptor d = GgufTensorDescriptor::decode(
        std::span<const uint8_t, GgufTensorDescriptor::kBytes>(bytes, GgufTensorDescriptor::kBytes));
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
                  << record.type << '\t' << record.contentIdentity << '\n';
    }
    return weightDigest(canonical.str());
}

std::string weightDigest(std::span<const uint8_t> bytes) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    // CommonCrypto takes 32-bit lengths.
    constexpr size_t kPieceBytes = size_t(1) << 30;
    for (size_t at = 0; at < bytes.size(); at += kPieceBytes)
        CC_SHA256_Update(&context, bytes.data() + at,
                         static_cast<CC_LONG>(std::min(kPieceBytes, bytes.size() - at)));
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    return digestHex(digest);
}

std::string weightDigest(std::string_view text) {
    return weightDigest({reinterpret_cast<const uint8_t *>(text.data()), text.size()});
}

} // namespace splash::model
