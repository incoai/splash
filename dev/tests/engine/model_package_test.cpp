#include "TestBuffers.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/ModelFactory.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"
#include "ops/Embedding.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

using splash::model::DFlashDraftLayout;
using splash::model::ModelDescriptor;
using splash::model::WeightFile;
using splash::model::WeightFileRecord;
using splash::model::WeightStoreError;
using splash::model::QwenAttentionWeights;
using splash::model::QwenGdnWeights;
using splash::model::Qwen3_8Layout;
using splash::model::Qwen3_8Weights;
using splash::ops::VisionLayout;
using splash::model::kWeightFileAlignment;
using splash::model::loadModelPackage;
using splash::model::makeModelDescriptor;
using splash::model::weightManifestFingerprint;
using splash::metal::BufferStorage;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::test::sharedBuffer;

constexpr std::string_view kDraftLayerMagic = "MDFD0004";
constexpr std::string_view kGgufImageMagic = "MDGG0001";
constexpr std::string_view kTargetEmbeddingMagic = "MDFE0001";
constexpr std::string_view kTargetHeadMagic = "MDFL0002";
constexpr std::string_view kTargetLayerMagic = "MDFL0006";
constexpr std::string_view kVisionMagic = "MDFV0001";

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

void testStartupCapabilities() {
    using splash::model::ExecutionLimits;
    require(ExecutionLimits::maximumBatchWidth == 4 &&
                ExecutionLimits::prefillTokenBudget == 2048 &&
                ExecutionLimits::draftQueryRows == 8 &&
                ExecutionLimits::draftProposalTokens == 7 &&
                ExecutionLimits::targetVerifyRows == 8 &&
                ExecutionLimits::draftContextTokens == 2048,
            "DFlash execution contract changed");
}

template <typename Function>
void requirePackedError(Function &&function, const std::string &message) {
    try {
        function();
    } catch (const WeightStoreError &) {
        return;
    }
    fail(message);
}

uint64_t alignPacked(uint64_t value) {
    return (value + kWeightFileAlignment - 1) &
        ~(kWeightFileAlignment - 1);
}

uint64_t checkedProduct(uint64_t left, uint64_t right) {
    if (left && right > std::numeric_limits<uint64_t>::max() / left) {
        fail("synthetic layout size overflow");
    }
    return left * right;
}

uint64_t q4Bytes(uint32_t outputSize, uint32_t inputSize) {
    return checkedProduct(outputSize, inputSize) * 9 / 16;
}

uint64_t declaredBytes(std::span<const WeightFileRecord> records) {
    uint64_t result = 0;
    for (const WeightFileRecord &record : records)
        result += record.declaredBytes;
    return result;
}

class TempDirectory final {
public:
    TempDirectory() {
        std::string pattern =
            (std::filesystem::temp_directory_path() /
             "splash-model-package.XXXXXX").string();
        char *created = mkdtemp(pattern.data());
        if (!created) fail("unable to create temporary directory");
        path_ = created;
    }

    ~TempDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path &path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void storeLittleEndian32(uint8_t *destination, uint32_t value) {
    destination[0] = static_cast<uint8_t>(value);
    destination[1] = static_cast<uint8_t>(value >> 8);
    destination[2] = static_cast<uint8_t>(value >> 16);
    destination[3] = static_cast<uint8_t>(value >> 24);
}

uint64_t writeWeightFile(const std::filesystem::path &path,
                         std::string_view magic, uint32_t layer,
                         uint32_t type,
                         std::span<const uint64_t> sections) {
    require(magic.size() == 8, "synthetic magic has the wrong size");
    std::filesystem::create_directories(path.parent_path());
    int descriptor = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC,
                          0600);
    if (descriptor < 0) fail("unable to create synthetic packed file");

    std::array<uint8_t, 16> header{};
    std::memcpy(header.data(), magic.data(), magic.size());
    storeLittleEndian32(header.data() + 8, layer);
    storeLittleEndian32(header.data() + 12, type);
    ssize_t written = pwrite(descriptor, header.data(), header.size(), 0);
    if (written != static_cast<ssize_t>(header.size())) {
        close(descriptor);
        fail("unable to write synthetic packed header");
    }

    uint64_t offset = header.size();
    for (uint64_t bytes : sections) {
        require(bytes > 0, "synthetic section is empty");
        offset = alignPacked(offset) + bytes;
    }
    uint64_t fileBytes = alignPacked(offset);
    if (fileBytes > static_cast<uint64_t>(
                        std::numeric_limits<off_t>::max()) ||
        ftruncate(descriptor, static_cast<off_t>(fileBytes)) != 0) {
        close(descriptor);
        fail("unable to size synthetic packed file");
    }
    close(descriptor);
    return fileBytes;
}

std::vector<uint64_t> targetLayerSections(
    const Qwen3_8Layout &layout, bool full) {
    constexpr uint64_t bf16 = 2;
    std::vector<uint64_t> result{
        uint64_t(layout.hiddenSize) * bf16,
        q4Bytes(full ? layout.packedFullWidth : layout.packedGdnWidth,
                layout.hiddenSize),
    };
    if (full) {
        result.insert(result.end(), {
            uint64_t(layout.attentionHeadDimension) * bf16,
            uint64_t(layout.attentionHeadDimension) * bf16,
            q4Bytes(layout.hiddenSize, layout.attentionWidth),
        });
    } else {
        result.insert(result.end(), {
            uint64_t(layout.convolutionDimension) * 4 * bf16,
            uint64_t(layout.gdnValueHeads) * 4,
            uint64_t(layout.gdnValueHeads) * bf16,
            uint64_t(layout.gdnHeadDimension) * bf16,
            q4Bytes(layout.hiddenSize, layout.attentionWidth),
        });
    }
    result.insert(result.end(), {
        uint64_t(layout.hiddenSize) * bf16,
        q4Bytes(layout.intermediateSize, layout.hiddenSize),
        q4Bytes(layout.intermediateSize, layout.hiddenSize),
        q4Bytes(layout.hiddenSize, layout.intermediateSize),
    });
    return result;
}

std::vector<uint64_t> draftLayerSections(const DFlashDraftLayout &layout) {
    constexpr uint64_t bf16 = 2;
    return {
        uint64_t(layout.hiddenSize) * bf16,
        uint64_t(4) * layout.hiddenSize * bf16,
        q4Bytes(layout.dynamicSize, layout.hiddenSize),
        q4Bytes(layout.qkvSize, layout.hiddenSize),
        uint64_t(layout.attentionHeadDimension) * bf16,
        uint64_t(layout.attentionHeadDimension) * bf16,
        q4Bytes(layout.hiddenSize, layout.attentionSize),
        uint64_t(layout.hiddenSize) * bf16,
        uint64_t(4) * layout.hiddenSize * bf16,
        q4Bytes(layout.dynamicSize, layout.hiddenSize),
        q4Bytes(layout.intermediateSize, layout.hiddenSize),
        q4Bytes(layout.intermediateSize, layout.hiddenSize),
        q4Bytes(layout.hiddenSize, layout.intermediateSize),
    };
}

std::vector<uint64_t> visionSections(const VisionLayout &layout) {
    constexpr uint64_t bf16 = 2;
    auto affine = [&](uint64_t outputSize, uint64_t inputSize,
                      std::vector<uint64_t> &sections) {
        sections.push_back(outputSize * inputSize * bf16);
        sections.push_back(outputSize * bf16);
    };
    auto norm = [&](std::vector<uint64_t> &sections) {
        sections.push_back(uint64_t(layout.hiddenSize) * bf16);
        sections.push_back(uint64_t(layout.hiddenSize) * bf16);
    };
    std::vector<uint64_t> result;
    affine(layout.hiddenSize, layout.patchDimension, result);
    result.push_back(uint64_t(layout.positionGridSide) *
                     layout.positionGridSide * layout.hiddenSize * bf16);
    for (uint32_t block = 0; block < layout.depth; ++block) {
        norm(result);
        affine(uint64_t(3) * layout.hiddenSize, layout.hiddenSize, result);
        affine(layout.hiddenSize, layout.hiddenSize, result);
        norm(result);
        affine(layout.paddedIntermediateSize, layout.hiddenSize, result);
        affine(layout.hiddenSize, layout.paddedIntermediateSize, result);
    }
    norm(result);
    affine(layout.mergedHiddenSize, layout.mergedHiddenSize, result);
    affine(layout.outputHiddenSize, layout.mergedHiddenSize, result);
    return result;
}

struct SyntheticAccounting {
    uint64_t targetBytes = 0;
    uint64_t draftBytes = 0;
    uint64_t visionBytes = 0;
};

SyntheticAccounting writeSyntheticPackage(
    const std::filesystem::path &root, const Qwen3_8Layout &target,
    const DFlashDraftLayout &draft, const VisionLayout &vision) {
    SyntheticAccounting result;
    for (uint32_t layer = 0; layer < target.layers; ++layer) {
        bool full = target.isFullAttentionLayer(layer);
        auto sections = targetLayerSections(target, full);
        result.targetBytes += writeWeightFile(
            root / "target" / ("layer-" + std::to_string(layer) + ".bin"),
            kTargetLayerMagic, layer, full ? 1U : 0U, sections);
    }
    std::array<uint64_t, 2> headSections{
        uint64_t(target.hiddenSize) * 2,
        q4Bytes(target.vocabularySize, target.hiddenSize),
    };
    result.targetBytes += writeWeightFile(
        root / "target/head.bin", kTargetHeadMagic, target.layers, 2,
        headSections);
    uint64_t embeddingElements =
        uint64_t(target.vocabularySize) * target.hiddenSize;
    std::array<uint64_t, 3> embeddingSections{
        embeddingElements / 2,
        embeddingElements / 32,
        embeddingElements / 32,
    };
    result.targetBytes += writeWeightFile(
        root / "target/embedding.bin", kTargetEmbeddingMagic,
        target.vocabularySize, target.hiddenSize, embeddingSections);

    for (uint32_t layer = 0; layer < draft.layers; ++layer) {
        auto sections = draftLayerSections(draft);
        result.draftBytes += writeWeightFile(
            root / "draft" / ("layer-" + std::to_string(layer) + ".bin"),
            kDraftLayerMagic, layer, 0, sections);
    }
    uint64_t codebookBytes =
        uint64_t(draft.vocabularySize) * draft.selectorRank * 2;
    std::array<uint64_t, 6> modelSections{
        q4Bytes(draft.hiddenSize, draft.targetHiddenSize),
        uint64_t(draft.hiddenSize) * 2,
        uint64_t(draft.hiddenSize) * 2,
        q4Bytes(draft.selectorRank, draft.hiddenSize),
        codebookBytes,
        codebookBytes,
    };
    result.draftBytes += writeWeightFile(
        root / "draft/model.bin", kDraftLayerMagic, draft.layers, 1,
        modelSections);
    auto sections = visionSections(vision);
    result.visionBytes += writeWeightFile(
        root / "vision/model.bin", kVisionMagic, vision.depth, 0, sections);
    return result;
}

// A packed file loaded into an image: its sections are aligned views the GPU
// reads and its memory is tracked; once released, a command that binds it
// fails until a restore reads the file again, which fails once the file was
// written.
void testWeightImages(MetalBackend &backend, const std::filesystem::path &root) {
    constexpr uint32_t elementCount = kWeightFileAlignment / sizeof(uint32_t);
    std::array<uint32_t, elementCount> expected{};
    for (uint32_t i = 0; i < elementCount; ++i) expected[i] = i * 17 + 3;
    std::array<uint64_t, 1> sections{sizeof(expected)};
    auto validPath = root / "valid.bin";
    uint64_t fileBytes = writeWeightFile(
        validPath, "TEST0001", 7, 9, sections);
    int descriptor = open(validPath.c_str(), O_WRONLY | O_CLOEXEC);
    require(descriptor >= 0 &&
                pwrite(descriptor, expected.data(), sizeof(expected),
                       kWeightFileAlignment) == static_cast<ssize_t>(sizeof(expected)),
            "unable to write the synthetic payload");
    close(descriptor);
    MetalBuffer output = backend.allocateBuffer(
        sizeof(expected), BufferStorage::Shared, "weight-readback");
    const uint64_t baseline = backend.memoryStats().allocatedBytes;
    MetalBuffer retained;
    const auto readBack = [&] {
        splash::metal::ComputeDispatch dispatch;
        dispatch.pipelineName = "test_copy_u32";
        dispatch.buffers = {{0, retained}, {1, output}};
        dispatch.bytes = {{2, &elementCount, sizeof(elementCount)}};
        dispatch.threadgroups = {(elementCount + 31) / 32, 1, 1};
        dispatch.threadsPerThreadgroup = {32, 1, 1};
        std::memset(output.contents(), 0, sizeof(expected));
        (void)backend.submit(dispatch);
        return std::memcmp(output.contents(), expected.data(), sizeof(expected)) == 0;
    };
    {
        splash::model::WeightImages images(backend);
        WeightFile file = images.load(
            splash::model::packedImage(validPath, "test/valid.bin", "TEST0001", 7, 9));
        retained = file.section(sizeof(expected), "payload");
        require(retained.contents() != nullptr &&
                    reinterpret_cast<uintptr_t>(retained.contents()) % kWeightFileAlignment == 0,
                "an image section is not a 16 KiB-aligned CPU-visible view");
        file.finish();
        require(backend.memoryStats().allocatedBytes >= baseline + fileBytes,
                "an image's memory was not tracked");
        require(readBack(), "GPU read of an image section was incorrect");
        images.release();
        require(!retained.contents() && backend.memoryStats().allocatedBytes == baseline,
                "released image memory remains");
        bool refused = false;
        try {
            (void)readBack();
        } catch (const splash::metal::MetalBackendError &error) {
            refused = std::string_view(error.what()).find("binds released memory") != std::string_view::npos;
        }
        require(refused, "a command bound released image memory");
        require(images.restore() && !images.released() &&
                    backend.memoryStats().allocatedBytes >= baseline + fileBytes && readBack(),
                "GPU read of a restored image section was incorrect");
        images.release();
        descriptor = open(validPath.c_str(), O_WRONLY | O_CLOEXEC);
        const uint32_t edit = 1;
        require(descriptor >= 0 &&
                    pwrite(descriptor, &edit, sizeof(edit), kWeightFileAlignment) == sizeof(edit),
                "unable to write the loaded file");
        close(descriptor);
        bool written = false;
        try {
            static_cast<void>(images.restore());
        } catch (const std::runtime_error &error) {
            written = std::string_view(error.what()).find("written while the model is loaded") !=
                      std::string_view::npos;
        }
        require(written, "a restore read a file written since it was opened");
    }
    retained = MetalBuffer{};
    require(backend.memoryStats().allocatedBytes == baseline,
            "a released image buffer remains in backend accounting");

    const auto load = [&](const std::filesystem::path &path, std::string_view magic, uint32_t layer,
                          uint32_t type) {
        splash::model::WeightImages images(backend);
        return images.load(splash::model::packedImage(path, "test/" + path.filename().string(), magic, layer, type));
    };
    auto headerPath = root / "header.bin";
    writeWeightFile(headerPath, "TEST0001", 7, 9, sections);
    requirePackedError([&] { (void)load(headerPath, "WRONG000", 7, 9); }, "wrong packed magic was accepted");
    requirePackedError([&] { (void)load(headerPath, "TEST0001", 8, 9); }, "wrong packed layer was accepted");
    requirePackedError([&] { (void)load(headerPath, "TEST0001", 7, 8); }, "wrong packed type was accepted");
    requirePackedError(
        [&] {
            WeightFile truncated = load(headerPath, "TEST0001", 7, 9);
            (void)truncated.section(fileBytes, {});
        },
        "truncated packed section was accepted");

    auto extraPath = root / "extra.bin";
    std::array<uint64_t, 2> extraSections{64, 64};
    writeWeightFile(extraPath, "TEST0001", 1, 2, extraSections);
    requirePackedError(
        [&] {
            WeightFile extra = load(extraPath, "TEST0001", 1, 2);
            (void)extra.section(64, {});
            extra.finish();
        },
        "unconsumed packed bytes were accepted");

    auto unalignedPath = root / "unaligned.bin";
    writeWeightFile(unalignedPath, "TEST0001", 1, 2, sections);
    require(truncate(unalignedPath.c_str(),
                     static_cast<off_t>(fileBytes - 1)) == 0,
            "unable to truncate synthetic file");
    requirePackedError([&] { (void)load(unalignedPath, "TEST0001", 1, 2); },
                       "unaligned packed file size was accepted");
}

// One tensor of a GGUF image: its 64-byte descriptor, then its sections.
std::filesystem::path writeGgufTensor(const std::filesystem::path &path, uint32_t type,
                                      uint32_t rows, uint32_t columns,
                                      std::array<uint32_t, 4> planeLayout,
                                      std::array<uint64_t, 3> planes) {
    std::vector<uint64_t> sections{64};
    for (uint64_t bytes : planes)
        if (bytes) sections.push_back(bytes);
    writeWeightFile(path, kGgufImageMagic, 0, 0, sections);
    std::array<uint8_t, 64> descriptor{};
    const std::array<uint32_t, 7> words{type, rows, columns, planeLayout[0], planeLayout[1],
                                        planeLayout[2], planeLayout[3]};
    std::memcpy(descriptor.data(), words.data(), sizeof(words));
    std::memcpy(descriptor.data() + 32, planes.data(), sizeof(planes));
    int file = open(path.c_str(), O_WRONLY | O_CLOEXEC);
    require(file >= 0 && pwrite(file, descriptor.data(), descriptor.size(),
                                kWeightFileAlignment) == static_cast<ssize_t>(descriptor.size()),
            "unable to write a synthetic GGUF descriptor");
    close(file);
    return path;
}

// GGUF image readers hold a tensor to the sizes the layout expects, and a
// token gather writes only into buffers holding its rows.
void testGgufImageLayout(MetalBackend &backend, const std::filesystem::path &root) {
    constexpr uint32_t rows = 256, columns = 256;
    // Q8_0 planes and native rows as the planner lays them out.
    const QuantFormat &q80 = kQuantFormats[GGUF_FMT_Q80];
    const splash::model::GgufPlaneBytes planes = splash::model::ggufPlaneBytes(q80, rows, columns);
    const auto projection = writeGgufTensor(root / "projection.bin", q80.ggml_type, rows, columns,
                                            {q80.plane0_bytes, q80.plane1_bytes, q80.meta_bytes, q80.meta_groups},
                                            {planes.plane0, planes.plane1, planes.meta});
    const auto embedding = writeGgufTensor(root / "embedding.bin", q80.ggml_type, rows, columns, {0, 0, 0, 0},
                                           {rows * splash::model::ggufRowBytes(q80, columns), 0, 0});
    splash::model::WeightImages images(backend);
    const auto loaded = [&](const std::filesystem::path &path) {
        return images.load(
            splash::model::packedImage(path, "test/" + path.filename().string(), kGgufImageMagic, 0, 0));
    };
    {
        // finish() proves the reader took exactly the descriptor, plane0 and
        // meta sections; the segment's format comes from the descriptor.
        WeightFile file = loaded(projection);
        const auto read = splash::model::readBlockProjection(file, rows, columns, "projection");
        file.finish();
        const auto &segment = read.blocks().segments.at(0);
        require(segment.formatId == GGUF_FMT_Q80 && !segment.plane1, "GGUF projection did not read a Q8_0 segment");
    }
    for (const auto [output, input] : {std::pair{2 * rows, columns}, std::pair{rows, 2 * columns}}) {
        requirePackedError(
            [&] {
                WeightFile file = loaded(projection);
                (void)splash::model::readBlockProjection(file, output, input, "projection");
            },
            "GGUF projection of other sizes than the layout's was accepted");
        requirePackedError(
            [&] {
                WeightFile file = loaded(embedding);
                (void)splash::model::readBlockEmbedding(file, output, input, "embedding");
            },
            "GGUF embedding of other sizes than the layout's was accepted");
    }
    WeightFile file = loaded(embedding);
    const auto table = splash::model::readBlockEmbedding(file, rows, columns, "embedding");
    file.finish();
    constexpr uint32_t gathered = 8;
    const MetalBuffer tokens = sharedBuffer(backend, gathered * sizeof(uint32_t));
    const MetalBuffer output =
        sharedBuffer(backend, uint64_t{gathered} * columns * splash::model::kBFloat16Bytes);
    splash::metal::CommandGraph graph;
    splash::ops::Embedding::add(graph, tokens, table, output, gathered);
    for (const auto &[tokenBytes, outputBytes] :
         {std::pair{tokens.sizeBytes() - sizeof(uint32_t), output.sizeBytes()},
          std::pair{tokens.sizeBytes(), output.sizeBytes() - splash::model::kBFloat16Bytes}}) {
        bool rejected = false;
        try {
            splash::ops::Embedding::add(graph, backend.view(tokens, 0, tokenBytes), table,
                                        backend.view(output, 0, outputBytes), gathered);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        require(rejected, "token gather past its buffers was accepted");
    }
}

void testSyntheticPackage(MetalBackend &backend,
                          const std::filesystem::path &root) {
    Qwen3_8Layout target;
    target.layers = 4;
    target.hiddenSize = 256;
    target.vocabularySize = 256;
    target.packedGdnWidth = 256;
    target.packedFullWidth = 256;
    target.convolutionDimension = 192;
    target.gdnKeyHeads = 1;
    target.gdnValueHeads = 1;
    target.gdnHeadDimension = 64;
    target.attentionWidth = 64;
    target.intermediateSize = 256;
    target.attentionQueryHeads = 1;
    target.attentionKvHeads = 1;
    target.attentionHeadDimension = 64;
    target.fullAttentionPeriod = 4;
    target.hiddenCaptureLayers.fill(target.layers - 1);

    DFlashDraftLayout draft;
    draft.layers = 2;
    draft.hiddenSize = 256;
    draft.vocabularySize = 256;
    draft.dynamicSize = 256;
    draft.qkvSize = 256;
    draft.attentionSize = 64;
    draft.intermediateSize = 256;
    draft.attentionHeadDimension = 64;
    draft.targetHiddenSize = target.capturedHiddenSize();
    draft.selectorRank = 256;

    VisionLayout vision;
    vision.depth = 2;
    vision.hiddenSize = 128;
    vision.patchDimension = 1536;
    vision.intermediateSize = 200;
    vision.paddedIntermediateSize = 256;
    vision.mergedHiddenSize = 512;
    vision.outputHiddenSize = 256;
    vision.heads = 2;
    vision.headDimension = 64;
    vision.positionGridSide = 4;

    SyntheticAccounting expected =
        writeSyntheticPackage(root, target, draft, vision);
    uint64_t baseline = backend.memoryStats().allocatedBytes;
    uint64_t actualTrackedBytes = 0;
    {
        ModelDescriptor descriptor =
            makeModelDescriptor("Qwen dense loader oracle", target, draft, vision);
        descriptor.sourceIdentity = "sources";
        auto package = loadModelPackage(backend, root, descriptor);
        const auto &loadedTarget = std::get<Qwen3_8Weights>(package.target);
        require(loadedTarget.layers.size() == target.layers,
                "target layer vector is incomplete");
        require(package.draft.layers.size() == draft.layers,
                "draft layer vector is incomplete");
        require(std::holds_alternative<QwenGdnWeights>(
                    loadedTarget.layers[0].mixer),
                "target GDN layer has the wrong typed layout");
        require(std::holds_alternative<QwenAttentionWeights>(
                    loadedTarget.layers[3].mixer),
                "target full-attention layer has the wrong typed layout");
        require(loadedTarget.files.size() == target.layers + 2,
                "target file records are incomplete");
        require(package.draft.files.size() == draft.layers + 1,
                "draft file records are incomplete");
        require(declaredBytes(loadedTarget.files) == expected.targetBytes,
                "target declared byte accounting is wrong");
        require(declaredBytes(package.draft.files) == expected.draftBytes,
                "draft declared byte accounting is wrong");
        require(package.vision.tensors.blocks.size() == vision.depth &&
                    package.vision.files.size() == 1 &&
                    declaredBytes(package.vision.files) == expected.visionBytes,
                "vision role records are incomplete");
        require(loadedTarget.actualAllocatedBytes +
                    package.draft.actualAllocatedBytes +
                    package.vision.actualAllocatedBytes ==
                    backend.memoryStats().allocatedBytes - baseline,
                "actual package allocation accounting is wrong");
        require(package.manifestFingerprintSha256.size() == 64,
                "manifest SHA-256 has the wrong length");

        std::vector<WeightFileRecord> records = loadedTarget.files;
        records.insert(records.end(), package.draft.files.begin(),
                       package.draft.files.end());
        records.insert(records.end(), package.vision.files.begin(),
                       package.vision.files.end());
        require(weightManifestFingerprint(records) ==
                    package.manifestFingerprintSha256,
                "combined manifest fingerprint is not reproducible");
        std::reverse(records.begin(), records.end());
        require(weightManifestFingerprint(records) ==
                    package.manifestFingerprintSha256,
                "manifest fingerprint depends on load order");
        records.front().declaredBytes += kWeightFileAlignment;
        require(weightManifestFingerprint(records) !=
                    package.manifestFingerprintSha256,
                "manifest fingerprint ignores declared file sizes");
        records.front().declaredBytes -= kWeightFileAlignment;
        require(std::all_of(records.begin(), records.end(),
                            [](const WeightFileRecord &record) {
                                return record.contentIdentity == "sources";
                            }),
                "an image does not record the sources it was written from");
        records.front().contentIdentity = "other sources";
        require(weightManifestFingerprint(records) !=
                    package.manifestFingerprintSha256,
                "manifest fingerprint ignores the sources' identity");

        require(package.draft.layers[0].attentionDynamic.outputSize ==
                        draft.dynamicSize &&
                    package.draft.layers[0].downProjection.outputSize ==
                        draft.hiddenSize,
                "draft projections lost their logical dimensions");
        actualTrackedBytes =
            backend.memoryStats().allocatedBytes - baseline;
        require(actualTrackedBytes >= expected.targetBytes +
                                          expected.draftBytes +
                                          expected.visionBytes,
                "backend actual allocation accounting is below logical bytes");

        // The weights go back to memory as they were loaded.
        std::vector<std::vector<uint8_t>> loaded;
        for (const auto &image : package.images->contents())
            loaded.emplace_back(image.bytes.begin(), image.bytes.end());
        package.images->release();
        require(backend.memoryStats().allocatedBytes - baseline ==
                    actualTrackedBytes - declaredBytes(loadedTarget.files) -
                        declaredBytes(package.draft.files) - declaredBytes(package.vision.files),
                "released weights remain in backend accounting");
        // They come back an image at a time, in load order.
        require(!package.images->restore() && !package.images->contents()[0].bytes.empty() &&
                    package.images->contents()[1].bytes.empty(),
                "a restore wrote back other than the next image");
        while (!package.images->restore()) {
        }
        const auto restored = package.images->contents();
        require(restored.size() == loaded.size() &&
                    std::equal(restored.begin(), restored.end(), loaded.begin(),
                               [](const auto &image, const std::vector<uint8_t> &bytes) {
                                   return std::equal(image.bytes.begin(), image.bytes.end(), bytes.begin(),
                                                     bytes.end());
                               }),
                "restored weights differ from the loaded ones");
    }
    require(backend.memoryStats().allocatedBytes == baseline,
            "model package allocations survived package destruction");

    std::cout << "synthetic declared_target=" << expected.targetBytes
              << " declared_draft=" << expected.draftBytes
              << " declared_vision=" << expected.visionBytes
              << " actual_tracked=" << actualTrackedBytes << '\n';
}

void validateRealPackage(MetalBackend &backend,
                         const std::filesystem::path &root) {
    uint64_t baseline = backend.memoryStats().allocatedBytes;
    uint64_t actualTrackedBytes = 0;
    uint64_t targetBytes = 0;
    uint64_t draftBytes = 0;
    uint64_t visionBytes = 0;
    std::string fingerprint;
    std::string name;
    {
        auto package = loadModelPackage(backend, root, splash::model::inspectModelPackage(root));
        targetBytes = declaredBytes(package.targetFiles());
        draftBytes = declaredBytes(package.draft.files);
        visionBytes = declaredBytes(package.vision.files);
        const uint32_t targetLayers = std::visit(
            [](const auto &weights) { return weights.layout.layers; },
            package.target);
        require(package.targetFiles().size() == targetLayers + 2,
                "real target file set is incomplete");
        require(package.draft.files.size() == package.draft.layout.layers + 1,
                "real draft file set is incomplete");
        require(package.vision.files.size() == 1,
                "real vision file set is incomplete");
        require(targetBytes && draftBytes && visionBytes,
                "real package has an empty role");
        actualTrackedBytes =
            backend.memoryStats().allocatedBytes - baseline;
        require(actualTrackedBytes >= targetBytes + draftBytes + visionBytes,
                "real package allocation accounting is below declared bytes");
        fingerprint = package.manifestFingerprintSha256;
        name = package.name();
    }
    require(backend.memoryStats().allocatedBytes == baseline,
            "real model images survived package destruction");
    std::cout << "real model=\"" << name << "\""
              << " declared_target=" << targetBytes
              << " declared_draft=" << draftBytes
              << " declared_vision=" << visionBytes
              << " actual_tracked=" << actualTrackedBytes
              << " manifest_sha256=" << fingerprint << '\n';
}

void testRealPackageMetadata(const std::filesystem::path &root) {
    std::ifstream input(root / "manifest.json");
    require(bool(input), "unable to read model manifest for format test");
    const std::string original{std::istreambuf_iterator<char>(input),
                               std::istreambuf_iterator<char>()};
    const std::string_view originalPrefix = "splash-packed-q4";
    const size_t offset = original.find(originalPrefix);
    require(offset != std::string::npos, "model manifest lacks a known format");

    TempDirectory temporary;
    std::filesystem::create_directory(temporary.path() / "tokenizer");
    std::filesystem::copy_file(root / "tokenizer/config.json",
                               temporary.path() / "tokenizer/config.json");
    const auto expected = splash::model::inspectModelPackage(root);
    for (std::string_view name : {std::string_view(expected.name),
                                  std::string_view("Community fine-tune")}) {
        for (std::string_view prefix : {"splash-packed-q4", "unknown-packed-q4"}) {
            std::string manifest = original;
            manifest.replace(offset, originalPrefix.size(), prefix);
            const std::string originalName = '"' + expected.name + '"';
            const size_t nameOffset = manifest.find(originalName);
            require(nameOffset != std::string::npos,
                    "model manifest lacks the expected display name");
            manifest.replace(nameOffset, originalName.size(),
                             '"' + std::string(name) + '"');
            {
                std::ofstream output(temporary.path() / "manifest.json");
                output << manifest;
                require(bool(output), "unable to write format test manifest");
            }
            try {
                const auto descriptor =
                    splash::model::inspectModelPackage(temporary.path());
                require(prefix != "unknown-packed-q4",
                        "unknown model format was accepted");
                require(descriptor.name == name &&
                            descriptor.target == expected.target &&
                            descriptor.draft == expected.draft &&
                            descriptor.valid(),
                        "model metadata changed the loaded layout or display name");
            } catch (const std::invalid_argument &error) {
                require(prefix == "unknown-packed-q4" &&
                            std::string_view(error.what()).find("weight format") !=
                                std::string_view::npos,
                        std::string("model metadata failed: ") + error.what());
            }
        }
    }
}

}  // namespace

int main(int argc, const char *argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: model_package_test <test.metallib> [models-root]\n";
        return 2;
    }
    try {
        testStartupCapabilities();
        MetalBackend backend(argv[1]);
        TempDirectory temporary;
        testWeightImages(backend, temporary.path());
        testGgufImageLayout(backend, temporary.path());
        testSyntheticPackage(backend, temporary.path() / "package");
        if (argc == 3) {
            testRealPackageMetadata(argv[2]);
            validateRealPackage(backend, argv[2]);
        }
        std::cout << "PASS ModelPackage\n";
    } catch (const std::exception &error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
