#include "TestAdmission.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "TestCheckpoint.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/ModelFactory.hpp"
#include "model/ModelSizing.hpp"
#include "model/SafetensorsImage.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightSource.hpp"
#include "ops/Embedding.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <span>
#include <string>
#include <system_error>
#include <tuple>
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
using splash::model::QwenAttentionWeights;
using splash::model::QwenGdnWeights;
using splash::model::Qwen3_6MoeLayout;
using splash::model::Qwen3_6MoeWeights;
using splash::model::Qwen3_8Layout;
using splash::model::Qwen3_8Weights;
using splash::model::TargetSource;
using splash::model::VisionSource;
using splash::ops::VisionLayout;
using splash::model::kWeightFileAlignment;
using splash::model::loadModel;
using splash::model::makeModelDescriptor;
using splash::model::streamableWeights;
using splash::model::weightManifestFingerprint;
using splash::metal::BufferStorage;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::test::SyntheticAccounting;
using splash::test::SyntheticTensor;
using splash::test::rejects;
using splash::test::sharedBuffer;
using splash::test::writeSyntheticModel;

constexpr std::string_view kGgufImageMagic = "MDGG0001";

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

// Writes a weight file of the given magic, layer and type whose sections have
// the given sizes, and returns its size.
uint64_t writeWeightFile(const std::filesystem::path &path, std::string_view magic, uint32_t layer,
                         uint32_t type, std::span<const uint64_t> sections) {
    require(magic.size() == 8, "synthetic magic has the wrong size");
    std::filesystem::create_directories(path.parent_path());
    const int descriptor = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    require(descriptor >= 0, "unable to create synthetic weight file");
    const std::array<uint8_t, 16> header = splash::model::weightFileHeader(magic, layer, type);
    uint64_t offset = header.size();
    for (uint64_t bytes : sections) {
        require(bytes > 0, "synthetic section is empty");
        offset = splash::model::alignWeightOffset(offset) + bytes;
    }
    const uint64_t fileBytes = splash::model::alignWeightOffset(offset);
    const bool written = pwrite(descriptor, header.data(), header.size(), 0) ==
                             static_cast<ssize_t>(header.size()) &&
                         fileBytes <= static_cast<uint64_t>(std::numeric_limits<off_t>::max()) &&
                         ftruncate(descriptor, static_cast<off_t>(fileBytes)) == 0;
    close(descriptor);
    require(written, "unable to write synthetic weight file");
    return fileBytes;
}

// The image of the weight file at path, read as it is.
splash::model::ImagePlan fileImage(const std::filesystem::path &path, std::string component,
                                   std::string_view magic, uint32_t layer, uint32_t type) {
    auto source = std::make_shared<splash::model::WeightSource>(path);
    return {std::move(component), std::string(magic), layer, type, source->bytes(),
            [source](std::span<uint8_t> destination, const MetalBuffer &) {
                source->readData(0, destination);
                source->checkUnchanged();
            }};
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
             "splash-model-loading.XXXXXX").string();
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

// A weight file loaded into an image: its sections are aligned views the GPU
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
        splash::model::WeightImages images(backend, "fixture");
        WeightFile file = images.load(
            fileImage(validPath, "test/valid.bin", "TEST0001", 7, 9));
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
        rejects([&] { (void)readBack(); }, "binds released memory", "a command bound released image memory");
        require(images.restore(splash::test::admitAll) && !images.released() &&
                    backend.memoryStats().allocatedBytes >= baseline + fileBytes && readBack(),
                "GPU read of a restored image section was incorrect");
        images.release();
        descriptor = open(validPath.c_str(), O_WRONLY | O_CLOEXEC);
        const uint32_t edit = 1;
        require(descriptor >= 0 &&
                    pwrite(descriptor, &edit, sizeof(edit), kWeightFileAlignment) == sizeof(edit),
                "unable to write the loaded file");
        close(descriptor);
        rejects([&] { static_cast<void>(images.restore(splash::test::admitAll)); }, "written while the model is loaded",
                "a restore read a file written since it was opened");
    }
    retained = MetalBuffer{};
    require(backend.memoryStats().allocatedBytes == baseline,
            "a released image buffer remains in backend accounting");

    const auto load = [&](const std::filesystem::path &path, std::string_view magic, uint32_t layer,
                          uint32_t type) {
        splash::model::WeightImages images(backend, "fixture");
        return images.load(fileImage(path, "test/" + path.filename().string(), magic, layer, type));
    };
    auto headerPath = root / "header.bin";
    writeWeightFile(headerPath, "TEST0001", 7, 9, sections);
    rejects([&] { (void)load(headerPath, "WRONG000", 7, 9); }, "weight image header mismatch",
            "wrong magic was accepted");
    rejects([&] { (void)load(headerPath, "TEST0001", 8, 9); }, "weight image header mismatch",
            "wrong layer was accepted");
    rejects([&] { (void)load(headerPath, "TEST0001", 7, 8); }, "weight image header mismatch",
            "wrong type was accepted");
    rejects(
        [&] {
            WeightFile truncated = load(headerPath, "TEST0001", 7, 9);
            (void)truncated.section(fileBytes, {});
        },
        "is truncated at section", "truncated section was accepted");

    auto extraPath = root / "extra.bin";
    std::array<uint64_t, 2> extraSections{64, 64};
    writeWeightFile(extraPath, "TEST0001", 1, 2, extraSections);
    rejects(
        [&] {
            WeightFile extra = load(extraPath, "TEST0001", 1, 2);
            (void)extra.section(64, {});
            extra.finish();
        },
        "weight image has unconsumed or missing bytes", "unconsumed bytes were accepted");

    auto unalignedPath = root / "unaligned.bin";
    writeWeightFile(unalignedPath, "TEST0001", 1, 2, sections);
    require(truncate(unalignedPath.c_str(),
                     static_cast<off_t>(fileBytes - 1)) == 0,
            "unable to truncate synthetic file");
    rejects([&] { (void)load(unalignedPath, "TEST0001", 1, 2); }, "weight image size is not 16 KiB-aligned",
            "unaligned file size was accepted");
}

// A restore admits each image's memory. An image admission refuses, and one
// whose writer is refused memory of its own, stays released and holds none,
// and the images before it stay; a release during a restore gives back the
// images it wrote back, and the next restore writes every image again as it
// was loaded.
void testRestoreAdmission(MetalBackend &backend, const std::filesystem::path &root) {
    const uint64_t baseline = backend.memoryStats().allocatedBytes;
    bool writerRefused = false;
    splash::model::WeightImages images(backend, "fixture");
    for (uint32_t index = 0; index < 2; ++index) {
        const std::array<uint64_t, 1> sections{kWeightFileAlignment};
        const auto path = root / ("restore-" + std::to_string(index) + ".bin");
        writeWeightFile(path, "TEST0001", index, 9, sections);
        const std::vector<uint8_t> payload(kWeightFileAlignment, static_cast<uint8_t>(index + 1));
        const int descriptor = open(path.c_str(), O_WRONLY | O_CLOEXEC);
        require(descriptor >= 0 &&
                    pwrite(descriptor, payload.data(), payload.size(), kWeightFileAlignment) ==
                        static_cast<ssize_t>(payload.size()),
                "unable to write a restore payload");
        close(descriptor);
        splash::model::ImagePlan image =
            fileImage(path, "test/" + path.filename().string(), "TEST0001", index, 9);
        image.write = [&writerRefused, write = std::move(image.write)](std::span<uint8_t> bytes,
                                                                       const MetalBuffer &buffer) {
            if (writerRefused)
                throw splash::metal::MetalAllocationError("writer refused memory");
            write(bytes, buffer);
        };
        WeightFile file = images.load(std::move(image));
        static_cast<void>(file.section(kWeightFileAlignment, "payload"));
        file.finish();
    }
    std::vector<std::vector<uint8_t>> loaded;
    for (const auto &image : images.contents())
        loaded.emplace_back(image.bytes.begin(), image.bytes.end());
    const auto holdsNone = [&] {
        return images.released() && images.contents()[0].bytes.empty() &&
               images.contents()[1].bytes.empty() && backend.memoryStats().allocatedBytes == baseline;
    };

    images.release();
    require(holdsNone(), "released images hold memory");
    require(!images.restore(splash::test::admitAll) && !images.contents()[0].bytes.empty() &&
                images.contents()[1].bytes.empty(),
            "a restore did not write back the first image alone");
    const uint64_t firstBack = backend.memoryStats().allocatedBytes;
    const auto keepsTheFirst = [&] {
        return images.released() && !images.contents()[0].bytes.empty() &&
               images.contents()[1].bytes.empty() && backend.memoryStats().allocatedBytes == firstBack;
    };
    const splash::metal::AllocationAdmission refuse = [](uint64_t, const std::function<void()> &) {
        return splash::metal::AllocationResult{splash::metal::AllocationFailure::HostPressure};
    };
    rejects([&] { static_cast<void>(images.restore(refuse)); }, "host memory reserve protected",
            "an image admission refused was restored");
    require(keepsTheFirst(), "an image admission refused holds memory, or the image before it went");
    writerRefused = true;
    rejects([&] { static_cast<void>(images.restore(splash::test::admitAll)); }, "writer refused memory",
            "an image whose writer was refused memory was restored");
    require(keepsTheFirst(), "an image whose writer was refused memory holds its own, or the image before it went");
    writerRefused = false;
    images.release();
    require(holdsNone(), "a release during a restore kept an image it wrote back");
    while (!images.restore(splash::test::admitAll)) {
    }
    for (size_t index = 0; index < loaded.size(); ++index) {
        const auto restored = images.contents()[index].bytes;
        require(std::equal(restored.begin(), restored.end(), loaded[index].begin(), loaded[index].end()),
                "a restore that started over wrote an image other than it was loaded");
    }
}

// One tensor of a GGUF image: its descriptor, then its sections.
std::filesystem::path writeGgufTensor(const std::filesystem::path &path,
                                      const splash::model::GgufTensorDescriptor &tensor) {
    std::vector<uint64_t> sections{splash::model::GgufTensorDescriptor::kBytes};
    for (uint64_t bytes : {tensor.plane0Bytes, tensor.plane1Bytes, tensor.metaTotalBytes})
        if (bytes) sections.push_back(bytes);
    writeWeightFile(path, kGgufImageMagic, 0, 0, sections);
    const auto descriptor = tensor.encode();
    int file = open(path.c_str(), O_WRONLY | O_CLOEXEC);
    require(file >= 0 && pwrite(file, descriptor.data(), descriptor.size(),
                                kWeightFileAlignment) == static_cast<ssize_t>(descriptor.size()),
            "unable to write a synthetic GGUF descriptor");
    close(file);
    return path;
}

// GGUF image readers hold a tensor to the sizes the layout expects, read F16
// and BF16 token rows as their formats' native rows, read a quantized
// tensor's native rows as a float segment only where one is read (the MoE
// router), never as a projection, and a token gather writes only into
// buffers holding its rows.
void testGgufImageLayout(MetalBackend &backend, const std::filesystem::path &root) {
    constexpr uint32_t rows = 256, columns = 256;
    // Q8_0 planes and native rows as the planner lays them out.
    const QuantFormat &q80 = kQuantFormats[GGUF_FMT_Q80];
    const splash::model::GgufPlaneBytes planes = splash::model::ggufPlaneBytes(q80, rows, columns);
    const auto projection = writeGgufTensor(root / "projection.bin",
                                            {.type = q80.ggml_type,
                                             .outputSize = rows,
                                             .inputSize = columns,
                                             .p0 = q80.plane0_bytes,
                                             .p1 = q80.plane1_bytes,
                                             .metaBytes = q80.meta_bytes,
                                             .metaGroups = q80.meta_groups,
                                             .plane0Bytes = planes.plane0,
                                             .plane1Bytes = planes.plane1,
                                             .metaTotalBytes = planes.meta});
    const auto embedding = writeGgufTensor(root / "embedding.bin",
                                           {.type = q80.ggml_type,
                                            .outputSize = rows,
                                            .inputSize = columns,
                                            .plane0Bytes = rows * splash::model::ggufRowBytes(q80, columns)});
    splash::model::WeightImages images(backend, "fixture");
    const auto loaded = [&](const std::filesystem::path &path) {
        return images.load(
            fileImage(path, "test/" + path.filename().string(), kGgufImageMagic, 0, 0));
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
        rejects(
            [&] {
                WeightFile file = loaded(projection);
                (void)splash::model::readBlockProjection(file, output, input, "projection");
            },
            "GGUF tensor does not match the layout",
            "GGUF projection of other sizes than the layout's was accepted");
        rejects(
            [&] {
                WeightFile file = loaded(embedding);
                (void)splash::model::readBlockEmbedding(file, output, input, "embedding");
            },
            "GGUF embedding does not match the layout",
            "GGUF embedding of other sizes than the layout's was accepted");
    }
    // Native Q8_0 rows read as the router's float segment, its rows as stored;
    // a projection refuses them, and the router refuses planes.
    {
        WeightFile file = loaded(embedding);
        const splash::ops::QuantizedSegment router = splash::model::readFloatSegment(file, "router");
        file.finish();
        require(router.isFloat() && router.quantizedRows() && router.formatId == GGUF_FMT_Q80 &&
                    router.outputSize == rows && router.inputSize == columns,
                "GGUF native rows did not read as a float segment");
    }
    rejects(
        [&] {
            WeightFile file = loaded(embedding);
            (void)splash::model::readBlockProjection(file, rows, columns, "projection");
        },
        "GGUF section sizes are inconsistent", "GGUF native rows were read as a projection");
    rejects(
        [&] {
            WeightFile file = loaded(projection);
            (void)splash::model::readFloatSegment(file, "router");
        },
        "GGUF rows are inconsistent", "GGUF planes were read as a float segment");
    // A GGUF's F16 and BF16 token rows read as the native rows of their formats.
    for (const auto &[type, format] : {std::pair{splash::model::ggml::kF16, GGUF_FMT_F16},
                                       std::pair{splash::model::ggml::kBF16, GGUF_FMT_BF16}}) {
        WeightFile file = loaded(writeGgufTensor(root / ("embedding-" + std::to_string(type) + ".bin"),
                                                 {.type = type,
                                                  .outputSize = rows,
                                                  .inputSize = columns,
                                                  .plane0Bytes = uint64_t{rows} * columns * 2}));
        const auto table = splash::model::readBlockEmbedding(file, rows, columns, "embedding");
        file.finish();
        require(table.blocks().formatId == format, "GGUF 16-bit embedding did not read as its format's rows");
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
    for (const auto &[tokenBytes, outputBytes, refusal] :
         {std::tuple{tokens.sizeBytes() - sizeof(uint32_t), output.sizeBytes(), "embedding token buffer holds"},
          std::tuple{tokens.sizeBytes(), output.sizeBytes() - splash::model::kBFloat16Bytes,
                     "embedding output buffer holds"}})
        rejects(
            [&] {
                splash::ops::Embedding::add(graph, backend.view(tokens, 0, tokenBytes), table,
                                            backend.view(output, 0, outputBytes), gathered);
            },
            refusal, "token gather past its buffers was accepted");
}

// The small layouts the synthetic models load: a target and its draft of
// every projection in whole plane tiles of a multiple of 256 inputs, as
// block images hold them, and a vision tower. A MoE target keeps its family's
// 256 experts, 8 per token.
template <class Layout = Qwen3_8Layout> Layout syntheticTarget() {
    Layout target;
    target.layers = 4;
    target.hiddenSize = 256;
    target.vocabularySize = 256;
    target.convolutionDimension = 512;
    target.gdnKeyHeads = 2;
    target.gdnValueHeads = 4;
    target.gdnHeadDimension = 64;
    target.attentionWidth = 256;
    target.packedGdnWidth = 1024;
    target.attentionQueryHeads = 4;
    target.attentionKvHeads = 4;
    target.attentionHeadDimension = 64;
    target.packedFullWidth = 1024;
    // The dense FFN, or each expert and the shared expert, 256 wide.
    if (target.ffnKind == splash::model::QwenFfnKind::SparseMoe)
        target.expertIntermediateSize = 256;
    else
        target.intermediateSize = 256;
    target.fullAttentionPeriod = 4;
    return target;
}

// The draft's hidden size and vocabulary are its target's.
template <class Layout> DFlashDraftLayout syntheticDraft(const Layout &target) {
    DFlashDraftLayout draft;
    draft.layers = 2;
    draft.hiddenSize = target.hiddenSize;
    draft.vocabularySize = target.vocabularySize;
    draft.dynamicSize = 256;
    draft.qkvSize = 512;
    draft.attentionSize = 256;
    draft.intermediateSize = 256;
    draft.attentionHeadDimension = 64;
    draft.rotaryTheta = 10'000'000.0F;
    draft.selectorRank = 256;
    draft.kvHeads = 2;
    // It reads a GDN layer and the attention layer.
    draft.targetLayers = {1, 3};
    draft.targetLayerCount = 2;
    return draft;
}

VisionLayout syntheticVision() {
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
    return vision;
}

// A synthetic installed model of small layouts loads every role, accounts
// each image's bytes and fingerprints what it loaded.
void testSyntheticModel(MetalBackend &backend,
                        const std::filesystem::path &root) {
    const Qwen3_8Layout target = syntheticTarget();
    const DFlashDraftLayout draft = syntheticDraft(target);
    const VisionLayout vision = syntheticVision();
    SyntheticAccounting expected =
        writeSyntheticModel(root, target, draft, vision);
    uint64_t baseline = backend.memoryStats().allocatedBytes;
    uint64_t actualTrackedBytes = 0;
    {
        ModelDescriptor descriptor =
            makeModelDescriptor("Qwen dense loader oracle", target, draft, vision,
                                TargetSource::Safetensors, VisionSource::Safetensors);
        descriptor.sourceIdentity = "sources";
        auto model = loadModel(backend, root, descriptor);
        const auto &loadedTarget = std::get<Qwen3_8Weights>(model.target);
        require(loadedTarget.layers.size() == target.layers,
                "target layer vector is incomplete");
        require(model.draft.layers.size() == draft.layers,
                "draft layer vector is incomplete");
        require(std::holds_alternative<QwenGdnWeights>(
                    loadedTarget.layers[0].mixer),
                "target GDN layer has the wrong typed layout");
        require(std::holds_alternative<QwenAttentionWeights>(
                    loadedTarget.layers[3].mixer),
                "target full-attention layer has the wrong typed layout");
        require(loadedTarget.files.size() == target.layers + 2,
                "target file records are incomplete");
        require(model.draft.files.size() == draft.layers + 1,
                "draft file records are incomplete");
        require(declaredBytes(loadedTarget.files) == expected.targetBytes,
                "target declared byte accounting is wrong");
        require(declaredBytes(model.draft.files) == expected.draftBytes,
                "draft declared byte accounting is wrong");
        require(model.vision.tensors.blocks.size() == vision.depth &&
                    model.vision.files.size() == 1 &&
                    declaredBytes(model.vision.files) == expected.visionBytes,
                "vision role records are incomplete");
        require(streamableWeights(root, descriptor, {}).loadedBytes ==
                    expected.targetBytes + expected.draftBytes +
                        expected.visionBytes,
                "the weight budget does not count every image");
        require(loadedTarget.actualAllocatedBytes +
                    model.draft.actualAllocatedBytes +
                    model.vision.actualAllocatedBytes ==
                    backend.memoryStats().allocatedBytes - baseline,
                "actual model allocation accounting is wrong");
        require(model.manifestFingerprintSha256.size() == 64,
                "manifest SHA-256 has the wrong length");

        std::vector<WeightFileRecord> records = loadedTarget.files;
        records.insert(records.end(), model.draft.files.begin(),
                       model.draft.files.end());
        records.insert(records.end(), model.vision.files.begin(),
                       model.vision.files.end());
        require(weightManifestFingerprint(records) ==
                    model.manifestFingerprintSha256,
                "combined manifest fingerprint is not reproducible");
        std::reverse(records.begin(), records.end());
        require(weightManifestFingerprint(records) ==
                    model.manifestFingerprintSha256,
                "manifest fingerprint depends on load order");
        records.front().declaredBytes += kWeightFileAlignment;
        require(weightManifestFingerprint(records) !=
                    model.manifestFingerprintSha256,
                "manifest fingerprint ignores declared file sizes");
        records.front().declaredBytes -= kWeightFileAlignment;
        require(std::all_of(records.begin(), records.end(),
                            [](const WeightFileRecord &record) {
                                return record.contentIdentity == "sources";
                            }),
                "an image does not record the sources it was written from");
        records.front().contentIdentity = "other sources";
        require(weightManifestFingerprint(records) !=
                    model.manifestFingerprintSha256,
                "manifest fingerprint ignores the sources' identity");

        require(model.draft.layers[0].attentionDynamic.outputSize ==
                        draft.dynamicSize &&
                    model.draft.layers[0].downProjection.outputSize ==
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
        for (const auto &image : model.images->contents())
            loaded.emplace_back(image.bytes.begin(), image.bytes.end());
        model.images->release();
        require(backend.memoryStats().allocatedBytes - baseline ==
                    actualTrackedBytes - declaredBytes(loadedTarget.files) -
                        declaredBytes(model.draft.files) - declaredBytes(model.vision.files),
                "released weights remain in backend accounting");
        // They come back an image at a time, in load order.
        require(!model.images->restore(splash::test::admitAll) && !model.images->contents()[0].bytes.empty() &&
                    model.images->contents()[1].bytes.empty(),
                "a restore wrote back other than the next image");
        while (!model.images->restore(splash::test::admitAll)) {
        }
        const auto restored = model.images->contents();
        require(restored.size() == loaded.size() &&
                    std::equal(restored.begin(), restored.end(), loaded.begin(),
                               [](const auto &image, const std::vector<uint8_t> &bytes) {
                                   return std::equal(image.bytes.begin(), image.bytes.end(), bytes.begin(),
                                                     bytes.end());
                               }),
                "restored weights differ from the loaded ones");
    }
    require(backend.memoryStats().allocatedBytes == baseline,
            "the model's allocations survived its destruction");

    std::cout << "synthetic declared_target=" << expected.targetBytes
              << " declared_draft=" << expected.draftBytes
              << " declared_vision=" << expected.visionBytes
              << " actual_tracked=" << actualTrackedBytes << '\n';
}

}  // namespace

// The format of each of a projection's segments, in order.
std::vector<uint32_t> segmentFormats(const splash::ops::Projection &projection) {
    std::vector<uint32_t> result;
    for (const auto &segment : projection.blocks().segments) result.push_back(segment.formatId);
    return result;
}

// Writes an MLX model of target under root, its target checkpoint holding
// tensors and its draft the synthetic one, and returns the descriptor that
// loads it as block images.
template <class Layout>
ModelDescriptor writeBlockModel(const std::filesystem::path &root, const Layout &target,
                                const std::vector<SyntheticTensor> &tensors) {
    const DFlashDraftLayout draft = syntheticDraft(target);
    splash::test::writeSyntheticShard(root / "target" / "model.safetensors", tensors);
    splash::test::writeSyntheticShard(root / "draft" / "model.safetensors", splash::test::draftTensors(draft));
    ModelDescriptor descriptor = makeModelDescriptor("Qwen block loader", target, draft, syntheticVision(),
                                                     TargetSource::Safetensors, VisionSource::None);
    descriptor.sourceIdentity = "sources";
    return descriptor;
}

// An MLX target loads as block images: each quantized module in the format of
// its tensors, GDN alpha and beta of two formats as F32, bf16 norms and
// grouped GDN value heads, and streamableWeights plans the bytes it loads. A
// raw checkpoint, whose convolution mlx-lm has not sanitized, is refused, and
// so is one whose head mlx-lm left unquantized; a bf16 token table loads as its
// bf16 rows.
void testSyntheticBlockModel(MetalBackend &backend, const std::filesystem::path &root) {
    const Qwen3_8Layout target = syntheticTarget();
    // Bits and group size by module, every MLX affine bit width and group size among them.
    const std::map<std::string_view, std::pair<uint32_t, uint32_t>> formats{
        {"in_proj_qkv", {3, 32}}, {"in_proj_z", {5, 128}}, {"in_proj_b", {4, 64}}, {"in_proj_a", {8, 64}},
        {"out_proj", {6, 64}},    {"q_proj", {2, 64}},     {"k_proj", {8, 32}},    {"v_proj", {4, 128}},
        {"o_proj", {5, 32}},      {"gate_proj", {4, 32}},  {"up_proj", {6, 128}},  {"down_proj", {3, 64}},
        {"lm_head", {8, 128}},    {"embed_tokens", {6, 32}}};
    const auto bitsAndGroup = [&](const std::string &module) {
        return formats.at(std::string_view(module).substr(module.rfind('.') + 1));
    };
    const auto format = [&](std::string_view name) {
        const auto [bits, group] = formats.at(name);
        return quant_affine_format_of(bits, group);
    };
    const ModelDescriptor descriptor =
        writeBlockModel(root, target, splash::test::mlxTargetTensors(target, bitsAndGroup));
    const uint64_t planned = splash::model::streamableWeights(root, descriptor, {}).loadedBytes;
    auto model = loadModel(backend, root, descriptor);
    const auto &weights = std::get<Qwen3_8Weights>(model.target);
    const auto &gdn = std::get<QwenGdnWeights>(weights.layers[0].mixer);
    require(segmentFormats(gdn.inputProjection) ==
                std::vector<uint32_t>{format("in_proj_qkv"), format("in_proj_z"),
                                      splash::ops::QuantizedSegment::kFloat32} &&
                segmentFormats(gdn.outputProjection) == std::vector<uint32_t>{format("out_proj")},
            "a block GDN did not keep its modules' formats and F32 alpha/beta");
    require(gdn.outputHeadOrder == splash::ops::GdnHeadOrder::Grouped && !weights.layers[0].inputNorm.float32,
            "an MLX block target did not keep its grouped value heads and bf16 norms");
    const auto &attention = std::get<QwenAttentionWeights>(weights.layers[3].mixer);
    require(segmentFormats(attention.inputProjection) ==
                    std::vector<uint32_t>{format("q_proj"), format("k_proj"), format("v_proj")} &&
                segmentFormats(attention.outputProjection) == std::vector<uint32_t>{format("o_proj")},
            "a block attention did not keep its modules' formats");
    require(segmentFormats(weights.layers[1].gateProjection) == std::vector<uint32_t>{format("gate_proj")} &&
                segmentFormats(weights.layers[1].upProjection) == std::vector<uint32_t>{format("up_proj")} &&
                segmentFormats(weights.layers[1].downProjection) == std::vector<uint32_t>{format("down_proj")} &&
                segmentFormats(weights.logitsProjection) == std::vector<uint32_t>{format("lm_head")} &&
                weights.tokenEmbedding.blocks().formatId == format("embed_tokens"),
            "a block FFN, head or token table did not keep its module's format");
    require(declaredBytes(weights.files) + declaredBytes(model.draft.files) == planned,
            "streamableWeights is not what an MLX block target loads");

    std::vector<splash::test::SyntheticTensor> raw = splash::test::mlxTargetTensors(target, bitsAndGroup);
    for (splash::test::SyntheticTensor &tensor : raw)
        if (tensor.name.ends_with("conv1d.weight")) tensor.shape = {tensor.shape[0], 1, tensor.shape[1]};
    splash::test::writeSyntheticShard(root / "target" / "model.safetensors", raw);
    rejects([&] { static_cast<void>(loadModel(backend, root, descriptor)); },
            "source tensor type or shape does not match: "
            "language_model.model.layers.0.linear_attn.conv1d.weight",
            "a raw checkpoint's convolution was accepted");

    // An unquantized module: its weight in bf16, without scales and biases.
    const auto unquantized = [&](const std::string &module) {
        std::vector<splash::test::SyntheticTensor> tensors;
        for (splash::test::SyntheticTensor &tensor : splash::test::mlxTargetTensors(target, bitsAndGroup))
            if (tensor.name == module + ".weight")
                tensors.push_back({tensor.name, "BF16", {target.vocabularySize, target.hiddenSize}});
            else if (tensor.name != module + ".scales" && tensor.name != module + ".biases")
                tensors.push_back(std::move(tensor));
        splash::test::writeSyntheticShard(root / "target" / "model.safetensors", tensors);
    };
    unquantized("language_model.lm_head");
    rejects([&] { static_cast<void>(loadModel(backend, root, descriptor)); }, "language_model.lm_head is unquantized",
            "an unquantized language_model.lm_head was accepted");
    // A module the checkpoint lacks altogether is missing, not unquantized.
    std::vector<splash::test::SyntheticTensor> lacking;
    for (splash::test::SyntheticTensor &tensor : splash::test::mlxTargetTensors(target, bitsAndGroup))
        if (!tensor.name.starts_with("language_model.lm_head."))
            lacking.push_back(std::move(tensor));
    splash::test::writeSyntheticShard(root / "target" / "model.safetensors", lacking);
    rejects([&] { static_cast<void>(loadModel(backend, root, descriptor)); },
            "missing source tensor: language_model.lm_head.weight",
            "a checkpoint without language_model.lm_head was accepted");
    unquantized("language_model.model.embed_tokens");
    require(std::get<Qwen3_8Weights>(loadModel(backend, root, descriptor).target).tokenEmbedding.blocks().formatId ==
                GGUF_FMT_BF16,
            "a bf16 token table did not load as its bf16 rows");
}

// The name of a convention of NVFP4 and FP8 checkpoints.
std::string checkpointName(splash::test::FloatCheckpoint checkpoint) {
    return checkpoint == splash::test::FloatCheckpoint::ModelOptimizer ? "Model Optimizer" : "compressed-tensors";
}

// A target of transformers names in NVFP4 and FP8, as Model Optimizer or
// compressed-tensors stores them, loads as block images: each module in its
// format with its convention's tensor scale, GDN alpha and beta of bf16 as
// F32, the RMSNorms as the F32 1 + w of the w stored but the GDN's gated norm
// as its w, the bf16 token table as its bf16 rows, and streamableWeights plans
// the bytes it loads.
void testSyntheticFloatModel(MetalBackend &backend, const std::filesystem::path &root,
                             splash::test::FloatCheckpoint checkpoint) {
    const Qwen3_8Layout target = syntheticTarget();
    const std::string name = checkpointName(checkpoint);
    // FP8 for the output projections and the head, as NVIDIA's and unsloth's
    // checkpoints keep some layers, NVFP4 for the others.
    const auto format = [](const std::string &module) -> uint32_t {
        return module.ends_with("o_proj") || module.ends_with("out_proj") || module == "lm_head" ? GGUF_FMT_FP8
                                                                                                 : GGUF_FMT_NVFP4;
    };
    const ModelDescriptor descriptor =
        writeBlockModel(root, target, splash::test::floatTargetTensors(target, format, checkpoint));
    const uint64_t planned = splash::model::streamableWeights(root, descriptor, {}).loadedBytes;
    auto model = loadModel(backend, root, descriptor);
    const auto &weights = std::get<Qwen3_8Weights>(model.target);
    const auto &gdn = std::get<QwenGdnWeights>(weights.layers[0].mixer);
    const auto &attention = std::get<QwenAttentionWeights>(weights.layers[3].mixer);
    const std::vector<uint32_t> nvfp4{GGUF_FMT_NVFP4}, fp8{GGUF_FMT_FP8};
    require(segmentFormats(gdn.inputProjection) ==
                    std::vector<uint32_t>{GGUF_FMT_NVFP4, GGUF_FMT_NVFP4, splash::ops::QuantizedSegment::kFloat32} &&
                segmentFormats(gdn.outputProjection) == fp8 &&
                segmentFormats(attention.inputProjection) ==
                    std::vector<uint32_t>{GGUF_FMT_NVFP4, GGUF_FMT_NVFP4, GGUF_FMT_NVFP4} &&
                segmentFormats(attention.outputProjection) == fp8 &&
                segmentFormats(weights.layers[1].gateProjection) == nvfp4 &&
                segmentFormats(weights.layers[1].upProjection) == nvfp4 &&
                segmentFormats(weights.layers[1].downProjection) == nvfp4 &&
                segmentFormats(weights.logitsProjection) == fp8 &&
                weights.tokenEmbedding.blocks().formatId == GGUF_FMT_BF16,
            "a " + name + " target did not keep its modules' formats and its bf16 token table");
    // Every stored norm weight is zero: 1 + w is 1, the gated norm's w 0.
    const auto first = [](const splash::ops::NormWeights &norm) {
        return norm.float32 ? static_cast<const float *>(norm.buffer.contents())[0] : -1.0f;
    };
    require(first(weights.layers[0].inputNorm) == 1.0f && first(weights.layers[0].postAttentionNorm) == 1.0f &&
                first(attention.queryNorm) == 1.0f && first(attention.keyNorm) == 1.0f &&
                first(weights.finalNorm) == 1.0f && first(gdn.mixerNorm) == 0.0f,
            "a " + name + " target's norms are not the F32 1 + w, its gated norm the F32 w");
    // The tensor scale g each convention stores: Model Optimizer's NVFP4
    // weight_scale_2 and FP8 weight_scale, g of the tensor; compressed-tensors'
    // weight_global_scale, 1 / g, and its bf16 FP8 weight_scale, each row's g.
    const splash::model::SafetensorsCheckpoint safetensors(root / "target");
    const std::vector<splash::model::gguf::Image> images = splash::model::safetensors::planImages(safetensors, target);
    const auto rowsOf = [&](const std::string &module) {
        for (const splash::model::gguf::Image &image : images)
            for (const splash::model::gguf::Repack &repack : image.repacks)
                for (const splash::model::gguf::TensorRows &rows : repack.sources)
                    if (rows.name == module) return rows;
        throw std::runtime_error("no planned rows of " + module);
    };
    const bool compressed = checkpoint == splash::test::FloatCheckpoint::CompressedTensors;
    const splash::model::gguf::TensorRows nvfp4Rows = rowsOf("model.language_model.layers.1.mlp.gate_proj"),
                                          fp8Rows = rowsOf("lm_head");
    const auto &g4 = nvfp4Rows.scale, &g8 = fp8Rows.scale;
    require(g4.rowsPerValue == nvfp4Rows.rows && !g4.bfloat16 && g4.reciprocal == compressed &&
                g8.rowsPerValue == (compressed ? 1 : fp8Rows.rows) && g8.bfloat16 == compressed && !g8.reciprocal,
            "a " + name + " target's tensor scales are not read as its convention stores them");
    require(declaredBytes(weights.files) + declaredBytes(model.draft.files) == planned,
            "streamableWeights is not what a " + name + " target loads");
}

// A MoE target in NVFP4 and FP8, as Model Optimizer or compressed-tensors
// stores it, loads as block images: each layer's routed experts, modules of
// their own, stacked in index order into the planes of their shared format,
// the shared expert in its own, the bf16 router as its BF16 rows and the
// shared-expert gate as F32, and streamableWeights plans the bytes it loads.
void testSyntheticFloatMoeModel(MetalBackend &backend, const std::filesystem::path &root,
                                splash::test::FloatCheckpoint checkpoint) {
    const auto target = syntheticTarget<Qwen3_6MoeLayout>();
    const std::string name = checkpointName(checkpoint);
    const auto format = [](const std::string &module) -> uint32_t {
        return module.find(".shared_expert.") != std::string::npos ? GGUF_FMT_FP8 : GGUF_FMT_NVFP4;
    };
    const ModelDescriptor descriptor =
        writeBlockModel(root, target, splash::test::floatTargetTensors(target, format, checkpoint));
    const uint64_t planned = splash::model::streamableWeights(root, descriptor, {}).loadedBytes;
    const auto model = loadModel(backend, root, descriptor);
    const auto &weights = std::get<Qwen3_6MoeWeights>(model.target);
    for (const auto &layer : weights.layers) {
        const splash::ops::BlockMoeWeights &ffn = layer.ffn;
        require(ffn.gate.routed.formatId == GGUF_FMT_NVFP4 && ffn.up.routed.formatId == GGUF_FMT_NVFP4 &&
                    ffn.down.routed.formatId == GGUF_FMT_NVFP4 && ffn.gate.shared.formatId == GGUF_FMT_FP8 &&
                    ffn.up.shared.formatId == GGUF_FMT_FP8 && ffn.down.shared.formatId == GGUF_FMT_FP8 &&
                    ffn.router.isFloat() && ffn.router.formatId == GGUF_FMT_BF16 &&
                    ffn.sharedScalarGate.formatId == splash::ops::QuantizedSegment::kFloat32,
                "a " + name + " MoE did not keep its experts' and its router's formats and an F32 shared-expert gate");
    }
    const splash::model::SafetensorsCheckpoint safetensors(root / "target");
    uint32_t stacks = 0;
    for (const splash::model::gguf::Image &image : splash::model::safetensors::planImages(safetensors, target))
        for (const splash::model::gguf::Repack &repack : image.repacks) {
            if (repack.sources.empty() || repack.sources.front().name.find(".experts.") == std::string::npos) continue;
            ++stacks;
            require(repack.sources.size() == target.experts, "a " + name + " layer did not stack all its experts");
            for (uint32_t expert = 0; expert < target.experts; ++expert)
                require(repack.sources[expert].name.find(".experts." + std::to_string(expert) + ".") !=
                            std::string::npos,
                        "a " + name + " layer's experts are not stacked in their index order");
        }
    require(stacks == 3 * target.layers, "a " + name + " layer's experts did not stack into three tensors");
    require(declaredBytes(weights.files) + declaredBytes(model.draft.files) == planned,
            "streamableWeights is not what a " + name + " MoE target loads");
}

// An MLX MoE target loads as block images: the experts in mixed formats, the
// routed down projection mxfp4, the 8-bit affine router as its native rows,
// which the float projection decodes, and the bf16 shared-expert gate as the
// F32 values the MoE selection reads; streamableWeights plans the bytes it
// loads.
void testSyntheticBlockMoeModel(MetalBackend &backend, const std::filesystem::path &root) {
    const auto target = syntheticTarget<Qwen3_6MoeLayout>();
    // Bits and group size of each FFN module (mxfp4's for the routed down
    // projection); the others 4-bit in groups of 64.
    const std::map<std::string_view, std::pair<uint32_t, uint32_t>> formats{
        {"gate", {8, 64}},
        {"switch_mlp.gate_proj", {5, 32}},
        {"switch_mlp.up_proj", {2, 128}},
        {"switch_mlp.down_proj", {4, 32}},
        {"shared_expert.gate_proj", {6, 64}},
        {"shared_expert.up_proj", {8, 32}},
        {"shared_expert.down_proj", {3, 128}},
        {"shared_expert_gate", {8, 64}}};
    const auto bitsAndGroup = [&](const std::string &module) -> std::pair<uint32_t, uint32_t> {
        const size_t ffn = module.find(".mlp.");
        if (ffn == std::string::npos) return {4, 64};
        return formats.at(std::string_view(module).substr(ffn + 5));
    };
    std::vector<SyntheticTensor> tensors = splash::test::mlxTargetTensors(target, bitsAndGroup);
    // The routed down projection's scales are mxfp4's E8M0 exponents, without
    // biases, and the shared-expert gate is a bf16 weight.
    std::erase_if(tensors, [](const SyntheticTensor &tensor) {
        return tensor.name.ends_with("switch_mlp.down_proj.biases") ||
               tensor.name.ends_with("shared_expert_gate.scales") ||
               tensor.name.ends_with("shared_expert_gate.biases");
    });
    for (SyntheticTensor &tensor : tensors) {
        if (tensor.name.ends_with("switch_mlp.down_proj.scales")) tensor.dtype = "U8";
        if (tensor.name.ends_with("shared_expert_gate.weight"))
            tensor = {tensor.name, "BF16", {1, target.hiddenSize}};
    }
    const ModelDescriptor descriptor = writeBlockModel(root, target, tensors);
    const uint64_t planned = splash::model::streamableWeights(root, descriptor, {}).loadedBytes;
    const auto model = loadModel(backend, root, descriptor);
    const auto &weights = std::get<Qwen3_6MoeWeights>(model.target);
    const auto format = [&](std::string_view name) {
        const auto [bits, group] = formats.at(name);
        return quant_affine_format_of(bits, group);
    };
    for (const auto &layer : weights.layers) {
        const splash::ops::BlockMoeWeights &ffn = layer.ffn;
        require(ffn.gate.routed.formatId == format("switch_mlp.gate_proj") &&
                    ffn.up.routed.formatId == format("switch_mlp.up_proj") &&
                    ffn.down.routed.formatId == GGUF_FMT_MXFP4 &&
                    ffn.gate.shared.formatId == format("shared_expert.gate_proj") &&
                    ffn.up.shared.formatId == format("shared_expert.up_proj") &&
                    ffn.down.shared.formatId == format("shared_expert.down_proj"),
                "a block MoE did not keep its experts' formats");
        require(ffn.router.quantizedRows() && ffn.router.formatId == format("gate") &&
                    ffn.router.outputSize == target.experts && ffn.router.inputSize == target.hiddenSize &&
                    ffn.router.plane0.sizeBytes() >= uint64_t{target.experts} *
                        splash::ops::QuantizedSegment::rowBytes(ffn.router.formatId, target.hiddenSize) &&
                    ffn.sharedScalarGate.formatId == splash::ops::QuantizedSegment::kFloat32 &&
                    ffn.sharedScalarGate.outputSize == 1,
                "a block MoE did not read its router as its native rows and its shared-expert gate as F32");
    }
    require(declaredBytes(weights.files) + declaredBytes(model.draft.files) == planned,
            "streamableWeights is not what an MLX block MoE target loads");
}

// Flash-Next's centered norms, split QSA projection, integer PLE hash and
// native table rows must be identical with the table resident or streamed.
void testSyntheticFlashNext(MetalBackend &backend, const std::filesystem::path &root, uint32_t headWidth) {
    auto target = syntheticTarget<splash::model::Qwen4ExpLayout>();
    target.experts = 16;
    target.expertsPerToken = 4;
    target.hyperRank = 256;
    target.indexerHeads = 2;
    target.indexerHeadDimension = 64;
    target.pleHeadsPerNgram = 4;
    target.pleHeadDimension = headWidth;
    const uint64_t groups = headWidth / 32, rowBytes = groups * 20, floatRowBytes = uint64_t{headWidth} * 2;
    auto tensors = splash::test::mlxTargetTensors(target, [](const std::string &) {
        return std::pair<uint32_t, uint32_t>{4, 32};
    });
    const std::string ple = "language_model.model.layers.1.ple.ple_embedding.";
    const auto integers = [&](const std::string &name, std::vector<uint64_t> values) {
        std::vector<uint8_t> data(values.size() * sizeof(uint64_t));
        std::memcpy(data.data(), values.data(), data.size());
        tensors.push_back({ple + name, "I64", {values.size()}, std::move(data)});
    };
    integers("layer_multipliers", {3, 5, 7});
    integers("ngram_heads_offsets", {0, 2, 4, 6, 8, 10, 12, 14});
    integers("ngram_heads_vocab_sizes", std::vector<uint64_t>(target.pleHeads(), 2));
    for (uint32_t shard = 0; shard < 2; ++shard) {
        const std::string module = ple + "ngram_embedding.shards." + std::to_string(shard);
        tensors.push_back({module + ".weight", "U32", {8, groups * 4}});
        tensors.push_back({module + ".scales", "BF16", {8, groups}});
        std::vector<uint16_t> values(8 * groups, uint16_t(shard ? 0x4000 : 0x3f80)); // biases 1 and 2
        std::vector<uint8_t> data(values.size() * 2);
        std::memcpy(data.data(), values.data(), data.size());
        tensors.push_back({module + ".biases", "BF16", {8, groups}, std::move(data)});
    }
    for (auto &tensor : tensors) {
        if (!tensor.name.ends_with("index_qk_proj.biases")) continue;
        std::vector<uint16_t> values(tensor.shape[0] * tensor.shape[1]);
        for (size_t row = 0; row < tensor.shape[0]; ++row)
            std::fill_n(values.begin() + row * tensor.shape[1], tensor.shape[1],
                          row < target.indexerHeads * target.indexerHeadDimension ? 0x3fc0 : 0x4020);
        tensor.data.resize(values.size() * 2);
        std::memcpy(tensor.data.data(), values.data(), tensor.data.size());
    }
    const auto descriptor = writeBlockModel(root, target, tensors);
    std::vector<uint8_t> residentRows;
    std::string fingerprint;
    {
        const auto planned = streamableWeights(root, descriptor, {});
        auto model = loadModel(backend, root, descriptor);
        const auto &weights = std::get<splash::model::Qwen4ExpWeights>(model.target);
        require(declaredBytes(weights.files) + declaredBytes(model.draft.files) == planned.loadedBytes,
                  "Flash-Next's resident safetensors size differs from its plan");
        require(weights.layers[0].mixerHyper.norm.float32 &&
                    static_cast<const float *>(weights.layers[0].mixerHyper.norm.buffer.contents())[0] == 1.0F &&
                    static_cast<const float *>(std::get<QwenGdnWeights>(weights.layers[0].mixer).mixerNorm.buffer.contents())[0] == 0.0F,
                  "Flash-Next's centered norms or uncentered gated norm were changed incorrectly");
        const auto &indexer = *weights.layers[3].indexer;
        require(static_cast<const float *>(indexer.query.blocks().segments[0].plane0.contents())[0] == 1.5F &&
                    static_cast<const float *>(indexer.key.blocks().segments[0].plane0.contents())[0] == 2.5F,
                  "Flash-Next's fused QSA projection was not split at its key rows");
        require(weights.pleHash.tableRows == 16 && weights.pleHash.multipliers[2] == 7 &&
                    weights.pleHash.headOffsets.back() == 14, "Flash-Next's PLE hash was changed");
        const auto &table = weights.pleTable.blocks();
        residentRows.resize(16 * rowBytes);
        std::memcpy(residentRows.data(), table.rows.contents(), residentRows.size());
        fingerprint = weights.manifestFingerprintSha256;
    }
    const auto planned = streamableWeights(root, descriptor, {.streamExperts = true, .streamTable = true});
    auto readers = std::make_shared<splash::model::SsdReaders>(2);
    splash::model::ModelLoadOptions options{.streamedTableBytes = planned.tableFloorBytes + 4096,
                                             .expertMemoryBytes = planned.expertFloorBytes, .readers = readers};
    auto model = loadModel(backend, root, descriptor, options);
    const auto &weights = std::get<splash::model::Qwen4ExpWeights>(model.target);
    require(weights.streamedTable && weights.experts && weights.manifestFingerprintSha256 == fingerprint,
              "Flash-Next's streaming changed the weight identity or failed to open its stores");
    std::vector<uint8_t> rows(3 * rowBytes);
    std::array<uint32_t, 3> slots{};
    const std::array<uint32_t, 3> ids{15, 0, 8};
    weights.streamedTable->stage(ids, rows, slots, splash::model::SparseRowUse::Decode);
    for (size_t index = 0; index < ids.size(); ++index)
        require(std::memcmp(rows.data() + slots[index] * rowBytes,
                              residentRows.data() + ids[index] * rowBytes, rowBytes) == 0,
                  "a streamed PLE row differs from the resident row across shards");

    // mlx-serve puts the concatenated codes/scales/biases in a safetensors
    // sidecar and has already folded the norm offsets into its BF16 gains.
    auto external = tensors;
    std::erase_if(external, [](const auto &tensor) { return tensor.name.find("ngram_embedding.shards.") != std::string::npos; });
    for (auto &tensor : external) {
        if (!tensor.name.ends_with("norm.weight") && tensor.name.find(".ple.norm_") == std::string::npos) continue;
        if (tensor.name.ends_with("linear_attn.norm.weight")) continue;
        uint64_t count = 1;
        for (const auto dimension : tensor.shape) count *= dimension;
        std::vector<uint16_t> gains(count, 0x3f80);
        tensor.data.resize(count * 2);
        std::memcpy(tensor.data.data(), gains.data(), tensor.data.size());
    }
    const auto sidecar = root / "sidecar";
    const auto sidecarDescriptor = writeBlockModel(sidecar, target, external);
    std::vector<uint16_t> biases(16 * groups, 0x3f80);
    std::fill(biases.begin() + 8 * groups, biases.end(), 0x4000);
    std::vector<uint8_t> biasBytes(biases.size() * 2);
    std::memcpy(biasBytes.data(), biases.data(), biasBytes.size());
    splash::test::writeSyntheticShard(sidecar / "target" / "ngram_table.bin",
        {{"weight", "U32", {16, groups * 4}}, {"scales", "BF16", {16, groups}},
          {"biases", "BF16", {16, groups}, biasBytes}}, "mlx-serve-ngram");
    auto externalModel = loadModel(backend, sidecar, sidecarDescriptor, options);
    const auto &externalWeights = std::get<splash::model::Qwen4ExpWeights>(externalModel.target);
    require(static_cast<const float *>(externalWeights.layers[0].mixerHyper.norm.buffer.contents())[0] == 1.0F,
              "an mlx-serve pack's already-folded norm was shifted a second time");
    externalWeights.streamedTable->stage(ids, rows, slots, splash::model::SparseRowUse::Decode);
    for (size_t index = 0; index < ids.size(); ++index)
        require(std::memcmp(rows.data() + slots[index] * rowBytes,
                              residentRows.data() + ids[index] * rowBytes, rowBytes) == 0,
                  "an external PLE row differs from its embedded row");
    {
        std::ofstream changed(sidecar / "target" / "ngram_table.bin", std::ios::binary | std::ios::app);
        changed.put(0);
    }
    rejects([&] { externalWeights.streamedTable->stage(ids, rows, slots, splash::model::SparseRowUse::Decode); },
              "was written", "a changed PLE sidecar was accepted");
    require(externalWeights.streamedTable->stats().poisoned, "a changed PLE sidecar did not poison its store");

    const auto bf16 = root / "bf16-sidecar";
    const auto bf16Descriptor = writeBlockModel(bf16, target, external);
    std::vector<uint16_t> bf16Values(16 * headWidth, 0x3f80);
    std::fill(bf16Values.begin() + 8 * headWidth, bf16Values.end(), 0x4000);
    std::vector<uint8_t> bf16Bytes(bf16Values.size() * 2);
    std::memcpy(bf16Bytes.data(), bf16Values.data(), bf16Bytes.size());
    splash::test::writeSyntheticShard(bf16 / "target" / "ngram_table.bin",
        {{"weight", "BF16", {16, headWidth}, bf16Bytes}}, "mlx-serve-ngram");
    const auto bf16Plan = streamableWeights(bf16, bf16Descriptor, {.streamTable = true});
    auto bf16Options = options;
    bf16Options.streamedTableBytes = bf16Plan.tableFloorBytes + 4096;
    auto bf16Model = loadModel(backend, bf16, bf16Descriptor, bf16Options);
    const auto &bf16Weights = std::get<splash::model::Qwen4ExpWeights>(bf16Model.target);
    require(bf16Weights.streamedTable->table().rowBytes == floatRowBytes && bf16Weights.pleFormat == GGUF_FMT_BF16,
              "an unquantized PLE sidecar did not preserve its BF16 rows");
    std::vector<uint8_t> bf16Rows(3 * floatRowBytes);
    bf16Weights.streamedTable->stage(ids, bf16Rows, slots, splash::model::SparseRowUse::Decode);
    for (size_t index = 0; index < ids.size(); ++index)
        require(std::memcmp(bf16Rows.data() + slots[index] * floatRowBytes,
                              bf16Bytes.data() + ids[index] * floatRowBytes, floatRowBytes) == 0,
                  "a BF16 PLE row was changed while streaming");
    auto bf16Resident = loadModel(backend, bf16, bf16Descriptor);
    const auto &residentTable = std::get<splash::model::Qwen4ExpWeights>(bf16Resident.target).pleTable;
    const auto gatherIds = sharedBuffer(backend, ids.size() * sizeof(uint32_t));
    const auto gathered = sharedBuffer(backend, bf16Rows.size());
    std::memcpy(gatherIds.contents(), ids.data(), ids.size() * sizeof(uint32_t));
    splash::metal::CommandGraph gather;
    splash::ops::Embedding::add(gather, gatherIds, residentTable, gathered, ids.size());
    static_cast<void>(backend.submitCommandAsync(gather.command()).wait());
    for (size_t index = 0; index < ids.size(); ++index)
        require(std::memcmp(static_cast<const uint8_t *>(gathered.contents()) + index * floatRowBytes,
                              bf16Bytes.data() + ids[index] * floatRowBytes, floatRowBytes) == 0,
                  "a short resident BF16 PLE row was changed by its Metal gather");

    auto invalid = tensors;
    std::erase_if(invalid, [&](const auto &tensor) { return tensor.name == ple + "ngram_heads_vocab_sizes"; });
    // Separate directory: changing a source while its model is live must not
    // invalidate the buffers above merely to exercise a parser refusal.
    const auto bad = root / "bad";
    const auto badDescriptor = writeBlockModel(bad, target, invalid);
    rejects([&] { static_cast<void>(loadModel(backend, bad, badDescriptor, options)); },
              "missing source tensor", "a PLE table without its hash metadata was accepted");
}

int main(int argc, const char *argv[]) {
    if (argc != 3) {
        std::cerr << "usage: model_loading_test <test.metallib> <splash.metallib>\n";
        return 2;
    }
    try {
        testStartupCapabilities();
        MetalBackend backend(argv[1]);
        TempDirectory temporary;
        testWeightImages(backend, temporary.path());
        testRestoreAdmission(backend, temporary.path());
        testGgufImageLayout(backend, temporary.path());
        // Block images are repacked by the production library's kernel.
        MetalBackend production(argv[2]);
        testSyntheticModel(production, temporary.path() / "model");
        testSyntheticBlockModel(production, temporary.path() / "block-model");
        testSyntheticBlockMoeModel(production, temporary.path() / "block-moe-model");
        for (const uint32_t width : {32u, 160u})
            testSyntheticFlashNext(production, temporary.path() / ("flash-next-model-" + std::to_string(width)), width);
        for (const auto checkpoint : {splash::test::FloatCheckpoint::ModelOptimizer,
                                      splash::test::FloatCheckpoint::CompressedTensors}) {
            const std::string at = checkpoint == splash::test::FloatCheckpoint::ModelOptimizer ? "model-optimizer"
                                                                                               : "compressed-tensors";
            testSyntheticFloatModel(production, temporary.path() / (at + "-model"), checkpoint);
            testSyntheticFloatMoeModel(production, temporary.path() / (at + "-moe-model"), checkpoint);
        }
        std::cout << "PASS model-loading\n";
    } catch (const std::exception &error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
