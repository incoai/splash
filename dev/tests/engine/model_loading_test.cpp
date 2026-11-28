#include "GgufFormatReference.hpp"
#include "TestAdmission.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "TestCheckpoint.hpp"
#include "model/Flint.hpp"
#include "model/Qwen4Exp.hpp"
#include "model/SparseRowStore.hpp"
#include "model/SsdReaders.hpp"
#include "model/GgufImage.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/GgufPreparation.hpp"
#include "model/ModelFactory.hpp"
#include "model/ModelSizing.hpp"
#include "model/SafetensorsImage.hpp"
#include "model/SafetensorsTarget.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightSource.hpp"
#include "ops/Embedding.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
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
#include <thread>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

using splash::model::DFlashDraftLayout;
using splash::model::FlintLayout;
using splash::model::FlintWeights;
using splash::model::Qwen4ExpLayout;
using splash::model::Qwen4ExpWeights;
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
    require(ExecutionLimits::maximumBatchWidth == 8 &&
                ExecutionLimits::prefillTokenBudget == 2048 &&
                ExecutionLimits::draftQueryRows == 8 &&
                ExecutionLimits::draftProposalTokens == 7 &&
                ExecutionLimits::targetVerifyRows == 16 && splash::ops::kNarrowLaneRows == 8 &&
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
        splash::test::restoreWeights(images);
        require(!images.released() && backend.memoryStats().allocatedBytes >= baseline + fileBytes && readBack(),
                "GPU read of a restored image section was incorrect");
        images.release();
        descriptor = open(validPath.c_str(), O_WRONLY | O_CLOEXEC);
        const uint32_t edit = 1;
        require(descriptor >= 0 &&
                    pwrite(descriptor, &edit, sizeof(edit), kWeightFileAlignment) == sizeof(edit),
                "unable to write the loaded file");
        close(descriptor);
        rejects([&] { splash::test::restoreWeights(images); }, "written while the model is loaded",
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

// Weights loaded while released are reserved: an image's buffer holds no
// memory and its sections are views of none, of their sizes, while its file
// reads the header and the descriptors its plan writes (ImagePlan::planned),
// so that what a reader parses of the weights costs no memory; an image whose
// plan holds none of them cannot be read so. The next restore writes the
// image, under the views read before, and no image loads while a restore is
// under way.
void testReservedImages(MetalBackend &backend, const std::filesystem::path &root) {
    constexpr uint32_t elementCount = kWeightFileAlignment / sizeof(uint32_t);
    std::array<uint32_t, elementCount> expected{};
    for (uint32_t i = 0; i < elementCount; ++i) expected[i] = i * 13 + 5;
    constexpr uint64_t kDescriptorBytes = splash::model::GgufTensorDescriptor::kBytes;
    const std::array<uint64_t, 2> sections{kDescriptorBytes, sizeof(expected)};
    const auto path = root / "reserved.bin";
    const uint64_t fileBytes = writeWeightFile(path, "TEST0001", 3, 4, sections);
    std::vector<uint8_t> described(kDescriptorBytes);
    for (size_t i = 0; i < described.size(); ++i) described[i] = static_cast<uint8_t>(i * 7 + 1);
    const int descriptor = open(path.c_str(), O_WRONLY | O_CLOEXEC);
    require(descriptor >= 0 &&
                pwrite(descriptor, described.data(), described.size(), kWeightFileAlignment) ==
                    static_cast<ssize_t>(described.size()) &&
                pwrite(descriptor, expected.data(), sizeof(expected), 2 * kWeightFileAlignment) ==
                    static_cast<ssize_t>(sizeof(expected)),
            "unable to write the reserved image's sections");
    close(descriptor);
    const std::array<uint8_t, 16> header = splash::model::weightFileHeader("TEST0001", 3, 4);
    const auto planned = std::make_shared<const std::vector<splash::model::gguf::Fill>>(
        std::vector<splash::model::gguf::Fill>{{0, {header.begin(), header.end()}},
                                               {kWeightFileAlignment, described}});
    const auto image = [&] {
        splash::model::ImagePlan plan = fileImage(path, "test/reserved.bin", "TEST0001", 3, 4);
        plan.planned = planned;
        return plan;
    };
    const uint64_t baseline = backend.memoryStats().allocatedBytes;
    splash::model::WeightImages images(backend, "fixture");
    images.release();
    WeightFile file = images.load(image());
    const std::span<const uint8_t> read = file.read(kDescriptorBytes, "descriptor");
    const MetalBuffer payload = file.section(sizeof(expected), "payload");
    file.finish();
    require(std::equal(read.begin(), read.end(), described.begin(), described.end()) && payload &&
                payload.sizeBytes() == sizeof(expected) && !payload.contents() &&
                backend.memoryStats().allocatedBytes == baseline && images.released(),
            "an image loaded while released was not reserved, or its file did not read its plan's bytes");
    rejects([&] { (void)images.load(fileImage(path, "test/unplanned.bin", "TEST0001", 3, 4)); },
            "weight image is not host visible", "an image without the bytes its readers parse was read without memory");
    require(!images.restore(splash::test::admitAll) && images.released(), "the restore did not start");
    rejects([&] { (void)images.load(image()); }, "while a restore is under way",
            "an image loaded while a restore was under way");
    splash::test::restoreWeights(images);
    require(!images.released() && backend.memoryStats().allocatedBytes >= baseline + fileBytes &&
                payload.contents() &&
                std::memcmp(payload.contents(), expected.data(), sizeof(expected)) == 0,
            "the restore did not write the reserved image under the views read before");
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
    splash::test::awaitWritten(images);
    rejects([&] { static_cast<void>(images.restore(refuse)); }, "host memory reserve protected",
            "an image admission refused was restored");
    require(keepsTheFirst(), "an image admission refused holds memory, or the image before it went");
    writerRefused = true;
    rejects([&] { splash::test::restoreWeights(images); }, "writer refused memory",
            "an image whose writer was refused memory was restored");
    require(keepsTheFirst(), "an image whose writer was refused memory holds its own, or the image before it went");
    writerRefused = false;
    images.release();
    require(holdsNone(), "a release during a restore kept an image it wrote back");
    splash::test::restoreWeights(images);
    for (size_t index = 0; index < loaded.size(); ++index) {
        const auto restored = images.contents()[index].bytes;
        require(std::equal(restored.begin(), restored.end(), loaded[index].begin(), loaded[index].end()),
                "a restore that started over wrote an image other than it was loaded");
    }
}

// A restore writes each image on a thread of its own, so that the engine's
// loop goes on answering while one is written: restore() returns while the
// writer runs, and until it ends the images stay released, writing() holds and
// no restore() may follow. The writer's thread calls the completion notifier
// once the write is over, so that the restore() it wakes the loop for goes
// on; a writer's failure comes from that restore(). A release during a write
// waits for it and gives its image back too.
void testRestoreWritesEachImageOnItsThread(MetalBackend &backend, const std::filesystem::path &root) {
    const uint64_t baseline = backend.memoryStats().allocatedBytes;
    const std::array<uint64_t, 1> sections{kWeightFileAlignment};
    const auto path = root / "threaded.bin";
    writeWeightFile(path, "TEST0001", 0, 9, sections);
    const std::vector<uint8_t> payload(kWeightFileAlignment, 0x5A);
    const int descriptor = open(path.c_str(), O_WRONLY | O_CLOEXEC);
    require(descriptor >= 0 &&
                pwrite(descriptor, payload.data(), payload.size(), kWeightFileAlignment) ==
                    static_cast<ssize_t>(payload.size()),
            "unable to write the threaded image's payload");
    close(descriptor);
    // While `hold` is set the writer waits for `go`, at most 10 s, and notes
    // whether restore() had returned by then; while `fail` is set it fails.
    std::atomic<bool> hold{false}, go{false}, returned{false}, returnedFirst{false}, fail{false};
    const auto await = [](const std::atomic<bool> &flag) {
        const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!flag && std::chrono::steady_clock::now() < limit)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return flag.load();
    };
    splash::model::WeightImages images(backend, "fixture");
    splash::model::ImagePlan image = fileImage(path, "test/threaded.bin", "TEST0001", 0, 9);
    image.write = [&, write = std::move(image.write)](std::span<uint8_t> bytes, const MetalBuffer &buffer) {
        if (hold) {
            static_cast<void>(await(go));
            returnedFirst = returned.load();
        }
        if (fail)
            throw std::runtime_error("the writer failed");
        write(bytes, buffer);
    };
    WeightFile file = images.load(std::move(image));
    static_cast<void>(file.section(kWeightFileAlignment, "payload"));
    file.finish();
    const std::vector<uint8_t> loaded(images.contents()[0].bytes.begin(), images.contents()[0].bytes.end());
    std::atomic<int> writtenCalls{0};
    std::atomic<bool> writtenAfterEnd{false}, writtenOnItsThread{false}, writtenSeen{false};
    const std::thread::id caller = std::this_thread::get_id();
    images.setCompletionNotifier([&] {
        writtenAfterEnd = !images.writing();
        writtenOnItsThread = std::this_thread::get_id() != caller;
        ++writtenCalls;
        writtenSeen = true;
    });

    images.release();
    hold = true;
    const bool restoredAtOnce = images.restore(splash::test::admitAll);
    returned = true;
    require(!restoredAtOnce && images.writing() && images.released(),
            "restore() waited for the image's write, or did not report it under way");
    rejects([&] { static_cast<void>(images.restore(splash::test::admitAll)); },
            "while an image is being written", "a restore went on while an image was written");
    go = true;
    require(await(writtenSeen) && returnedFirst && writtenCalls == 1 && writtenAfterEnd && writtenOnItsThread,
            "the write did not run after restore() returned, or did not call the notifier on its thread once over");
    require(images.restore(splash::test::admitAll) && !images.released() &&
                std::ranges::equal(images.contents()[0].bytes, loaded) && writtenCalls == 1,
            "the restore after the write did not end with the image as it was loaded");

    hold = false;
    fail = true;
    images.release();
    require(!images.restore(splash::test::admitAll), "a restore wrote its image before it returned");
    splash::test::awaitWritten(images);
    rejects([&] { static_cast<void>(images.restore(splash::test::admitAll)); }, "the writer failed",
            "a writer's failure did not come from the restore after it");
    fail = false;

    images.release();
    hold = true;
    go = false;
    require(!images.restore(splash::test::admitAll) && images.writing(), "the image was not being written");
    std::thread opener([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        go = true;
    });
    images.release();
    opener.join();
    require(!images.writing() && images.released() && images.contents()[0].bytes.empty() &&
                backend.memoryStats().allocatedBytes == baseline,
            "a release during a write kept the image being written");
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
// and BF16 token rows as stored, also of Flash-Next's 160 PLE values, which
// are no whole native block, read a quantized
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
    // A GGUF's F16 and BF16 token rows read as their formats' rows as stored.
    for (const auto &[type, format] : {std::pair{splash::model::ggml::kF16, GGUF_FMT_F16},
                                       std::pair{splash::model::ggml::kBF16, GGUF_FMT_BF16}})
        for (const uint32_t width : {columns, 160u}) {
            WeightFile file = loaded(writeGgufTensor(
                root / ("embedding-" + std::to_string(type) + "-" + std::to_string(width) + ".bin"),
                {.type = type, .outputSize = rows, .inputSize = width, .plane0Bytes = uint64_t{rows} * width * 2}));
            const auto table = splash::model::readBlockEmbedding(file, rows, width, "embedding");
            file.finish();
            require(table.rows().formatId == format,
                    "GGUF 16-bit embedding did not read as its format's rows at " + std::to_string(width) + " values");
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
    // Not the Qwen tokenizer's 248,320 tokens: its draft proposes from all of them.
    target.draftVocabulary = {};
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
        records.front().contentIdentity = "sources";
        // Every image, the target's, the draft's and the tower's, names the
        // formats its tensors load in, and another format is another manifest.
        require(std::all_of(records.begin(), records.end(),
                            [](const WeightFileRecord &record) { return !record.formats.empty(); }),
                "an image does not record the formats it loads in");
        records.front().formats += "x";
        require(weightManifestFingerprint(records) !=
                    model.manifestFingerprintSha256,
                "manifest fingerprint ignores the images' formats");

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
        require(!model.images->restore(splash::test::admitAll) &&
                    !model.images->contents()[0].bytes.empty() && model.images->contents()[1].bytes.empty(),
                "a restore wrote back other than the next image");
        splash::test::restoreWeights(*model.images);
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
// raw checkpoint, whose convolution mlx-lm has not sanitized, is refused; a
// head mlx-lm left unquantized, BF16 or F16, loads under --weights auto, the
// default, as the Q8_0 rows the reference GGUF quantization of Q8_0 makes of
// its values, and under original as its values as stored, in its own format;
// a bf16 token table as its bf16 rows.
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
                weights.tokenEmbedding.rows().formatId == format("embed_tokens"),
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

    // The same checkpoint with its attention q and k norms at `value`: nearer
    // the family's mean less 1 (Qwen3.8-27B's 1.491 less 1), its norms are
    // stored 1 below their multipliers and load as their F32 1 + w; nearer the
    // mean, even 0.3 from it, as the multipliers stored.
    const auto qkNorms = [&](uint16_t value) {
        std::vector<splash::test::SyntheticTensor> tensors = splash::test::mlxTargetTensors(target, bitsAndGroup);
        for (splash::test::SyntheticTensor &tensor : tensors)
            if (tensor.name.ends_with("q_norm.weight") || tensor.name.ends_with("k_norm.weight"))
                for (uint64_t at = 0; at < tensor.data.size(); at += 2)
                    tensor.data[at] = value & 0xFF, tensor.data[at + 1] = value >> 8;
        splash::test::writeSyntheticShard(root / "target" / "model.safetensors", tensors);
    };
    qkNorms(0x3F00); // 0.5
    require(std::get<Qwen3_8Weights>(loadModel(backend, root, descriptor).target).layers[0].inputNorm.float32,
            "an MLX block target whose norms are stored 1 below their multipliers did not load them as 1 + w");
    qkNorms(0x3F98); // 1.1875
    require(!std::get<Qwen3_8Weights>(loadModel(backend, root, descriptor).target).layers[0].inputNorm.float32,
            "an MLX block target whose q and k norms average 1.1875 did not load its norms as stored");

    // An unquantized module: its weight in `dtype` of the 16-bit values `bits`
    // (none for zeros), without scales and biases.
    const auto unquantized = [&](const std::string &module, const std::string &dtype = "BF16",
                                 const std::vector<uint16_t> &bits = {}) {
        std::vector<splash::test::SyntheticTensor> tensors;
        for (splash::test::SyntheticTensor &tensor : splash::test::mlxTargetTensors(target, bitsAndGroup))
            if (tensor.name == module + ".weight") {
                std::vector<uint8_t> data(bits.size() * 2);
                std::memcpy(data.data(), bits.data(), data.size());
                tensors.push_back({tensor.name, dtype, {target.vocabularySize, target.hiddenSize}, data});
            } else if (tensor.name != module + ".scales" && tensor.name != module + ".biases") {
                tensors.push_back(std::move(tensor));
            }
        splash::test::writeSyntheticShard(root / "target" / "model.safetensors", tensors);
    };
    // A head of random values, BF16 then F16, loads under auto as Q8_0, its planned rows those
    // gguf_reference::quantizeQ8_0 writes from the values, and under original in its own format, its planned
    // rows its values as stored.
    std::mt19937 rng(11);
    std::normal_distribution<float> weight(0.0f, 0.02f);
    using splash::model::gguf::Float16Weights;
    for (const bool float16 : {false, true}) {
        std::vector<uint16_t> bits(uint64_t{target.vocabularySize} * target.hiddenSize);
        std::vector<float> values(bits.size());
        for (size_t i = 0; i < bits.size(); ++i) {
            bits[i] = float16 ? gguf_reference::f2h(weight(rng)) : splash::test::floatToBf16(weight(rng));
            values[i] = float16 ? gguf_reference::h2f(bits[i]) : gguf_reference::bf2f(bits[i]);
        }
        unquantized("language_model.lm_head", float16 ? "F16" : "BF16", bits);
        for (const Float16Weights weights : {Float16Weights::Q8_0, Float16Weights::Published}) {
            const bool q8 = weights == Float16Weights::Q8_0;
            const std::string name = std::string("a ") + (float16 ? "F16" : "BF16") + " head" +
                                     (q8 ? " under q8" : " as published");
            const uint32_t format = q8 ? GGUF_FMT_Q80 : float16 ? GGUF_FMT_F16 : GGUF_FMT_BF16;
            require(segmentFormats(std::get<Qwen3_8Weights>(
                                       loadModel(backend, root, descriptor, {.weights = weights}).target)
                                       .logitsProjection) == std::vector<uint32_t>{format},
                    name + " did not load in " + kQuantFormats[format].name);
            const splash::model::SafetensorsCheckpoint checkpoint(root / "target");
            const std::vector<splash::model::gguf::Image> images = splash::model::safetensors::planImages(
                checkpoint, target, splash::model::safetensors::normStorage(checkpoint, target), {.weights = weights});
            const splash::model::gguf::Repack &head = images[target.layers].repacks.front();
            std::vector<uint8_t> rows(uint64_t{target.vocabularySize} *
                                      splash::model::ggufRowBytes(kQuantFormats[format], target.hiddenSize));
            splash::model::readRepackRows(head, 0, target.vocabularySize, rows.data());
            const std::vector<uint8_t> stored(reinterpret_cast<const uint8_t *>(bits.data()),
                                              reinterpret_cast<const uint8_t *>(bits.data() + bits.size()));
            require(head.format == format && rows == (q8 ? gguf_reference::quantizeQ8_0(values) : stored),
                    name + "'s planned rows are not " +
                        (q8 ? "the reference Q8_0 of its values" : "its values as stored"));
        }
    }
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
    require(std::get<Qwen3_8Weights>(loadModel(backend, root, descriptor).target).tokenEmbedding.rows().formatId ==
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
                weights.tokenEmbedding.rows().formatId == GGUF_FMT_BF16,
            "a " + name + " target did not keep its modules' formats and its bf16 token table");
    // Every stored norm weight is zero, but the q and k norms' 0.5: 1 + w is 1
    // (1.5), the gated norm's w 0.
    const auto first = [](const splash::ops::NormWeights &norm) {
        return norm.float32 ? static_cast<const float *>(norm.buffer.contents())[0] : -1.0f;
    };
    require(first(weights.layers[0].inputNorm) == 1.0f && first(weights.layers[0].postAttentionNorm) == 1.0f &&
                first(attention.queryNorm) == 1.5f && first(attention.keyNorm) == 1.5f &&
                first(weights.finalNorm) == 1.0f && first(gdn.mixerNorm) == 0.0f,
            "a " + name + " target's norms are not the F32 1 + w, its gated norm the F32 w");
    // The tensor scale g each convention stores: Model Optimizer's NVFP4
    // weight_scale_2 and FP8 weight_scale, g of the tensor; compressed-tensors'
    // weight_global_scale, 1 / g, and its bf16 FP8 weight_scale, each row's g.
    const splash::model::SafetensorsCheckpoint safetensors(root / "target");
    const std::vector<splash::model::gguf::Image> images = splash::model::safetensors::planImages(safetensors, target, splash::model::safetensors::normStorage(safetensors, target));
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
    // Under transformers' names too, the q and k norms tell the storage: at
    // 1.5, near Qwen3.8-27B's mean, the norms load as the multipliers stored.
    std::vector<splash::test::SyntheticTensor> multipliers = splash::test::floatTargetTensors(target, format, checkpoint);
    for (splash::test::SyntheticTensor &tensor : multipliers)
        if (tensor.name.ends_with("q_norm.weight") || tensor.name.ends_with("k_norm.weight")) {
            tensor.data.resize(tensor.shape.front() * 2);
            for (uint64_t at = 0; at < tensor.data.size(); at += 2) tensor.data[at] = 0xC0, tensor.data[at + 1] = 0x3F;
        }
    splash::test::writeSyntheticShard(root / "target" / "model.safetensors", multipliers);
    require(!std::get<Qwen3_8Weights>(loadModel(backend, root, descriptor).target).layers[0].inputNorm.float32,
            "a " + name + " target whose q and k norms are multipliers did not load its norms as stored");
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
    for (const splash::model::gguf::Image &image : splash::model::safetensors::planImages(safetensors, target, splash::model::safetensors::normStorage(safetensors, target)))
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
// routed gate projection nvfp4 with each expert's global_scale, which divided
// by 6 * 448 is its expert scale, the routed up projection BF16 values MXFP8
// holds exactly as MXFP8 (as a GGUF's 16-bit experts load), the routed down
// projection mxfp4, the 8-bit affine router as its native rows, which the
// float projection decodes, and the bf16 shared-expert gate as the F32 values
// the MoE selection reads; streamableWeights plans the bytes it loads.
void testSyntheticBlockMoeModel(MetalBackend &backend, const std::filesystem::path &root) {
    const auto target = syntheticTarget<Qwen3_6MoeLayout>();
    // Bits and group size of each FFN module (mxfp4's for the routed down
    // projection); the others 4-bit in groups of 64.
    const std::map<std::string_view, std::pair<uint32_t, uint32_t>> formats{
        {"gate", {8, 64}},
        {"switch_mlp.gate_proj", {4, 16}},
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
    // The routed gate projection's scales are nvfp4's E4M3 bytes (each below
    // 0x7F, no NaN) and the down projection's mxfp4's E8M0 exponents, both
    // without biases, the gate beside each expert's global_scale; the
    // shared-expert gate is a bf16 weight.
    std::erase_if(tensors, [](const SyntheticTensor &tensor) {
        return tensor.name.ends_with("switch_mlp.gate_proj.biases") ||
               tensor.name.ends_with("switch_mlp.down_proj.biases") ||
               tensor.name.ends_with("shared_expert_gate.scales") ||
               tensor.name.ends_with("shared_expert_gate.biases");
    });
    std::vector<float> globalScales(target.experts);
    for (uint32_t e = 0; e < target.experts; ++e) globalScales[e] = 0.5f + 0.25f * float(e);
    std::vector<uint8_t> globalBytes(globalScales.size() * 4);
    std::memcpy(globalBytes.data(), globalScales.data(), globalBytes.size());
    std::vector<SyntheticTensor> scales;
    for (SyntheticTensor &tensor : tensors) {
        if (tensor.name.ends_with("switch_mlp.gate_proj.scales")) {
            tensor.dtype = "U8";
            tensor.data.resize(tensor.data.size() / 2);
            for (uint8_t &byte : tensor.data) byte &= 0x3F;
            const std::string module = tensor.name.substr(0, tensor.name.size() - 7);
            scales.push_back({module + ".global_scale", "F32", {target.experts}, globalBytes});
        }
        if (tensor.name.ends_with("switch_mlp.down_proj.scales")) tensor.dtype = "U8";
        if (tensor.name.ends_with("shared_expert_gate.weight"))
            tensor = {tensor.name, "BF16", {1, target.hiddenSize}};
    }
    tensors.insert(tensors.end(), scales.begin(), scales.end());
    // The routed up projection a BF16 weight of k / 128 for k of -16..16, which
    // MXFP8 holds exactly: E4M3 values of at most three mantissa bits under one
    // power of two a block.
    std::erase_if(tensors, [](const SyntheticTensor &tensor) {
        return tensor.name.find("switch_mlp.up_proj.") != std::string::npos;
    });
    for (uint32_t layer = 0; layer < target.layers; ++layer) {
        const uint64_t values = uint64_t{target.experts} * target.expertIntermediateSize * target.hiddenSize;
        std::vector<uint8_t> bytes(values * 2);
        for (uint64_t i = 0; i < values; ++i) {
            const uint16_t bits = splash::test::floatToBf16(float(int64_t((i * 2654435761u + layer) % 33) - 16) / 128.0f);
            std::memcpy(bytes.data() + 2 * i, &bits, 2);
        }
        tensors.push_back({"language_model.model.layers." + std::to_string(layer) + ".mlp.switch_mlp.up_proj.weight",
                           "BF16", {target.experts, target.expertIntermediateSize, target.hiddenSize}, bytes});
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
        std::vector<float> expertScales(3 * target.experts);
        std::memcpy(expertScales.data(), ffn.expertScales.contents(), expertScales.size() * 4);
        bool scaled = true;
        for (uint32_t e = 0; e < target.experts; ++e)
            scaled &= expertScales[e] == globalScales[e] / (6.0f * 448.0f) &&
                      expertScales[target.experts + e] == 1.0f && expertScales[2 * target.experts + e] == 1.0f;
        require(scaled, "a block MoE's nvfp4 experts did not take their global scales as their expert scales");
        require(ffn.gate.routed.formatId == (target.hiddenSize % 256 ? GGUF_FMT_NVFP464 : GGUF_FMT_NVFP4) &&
                    ffn.up.routed.formatId == GGUF_FMT_MXFP8 &&
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

// An MLX Flint (mlx-community's, of mlx-lm's names) loads as the block images
// its GGUF's do: every layer's ungated attention with its sinks, the bf16
// values widened to the F32 the attention reads, its norms widened to the F32
// its GGUF's hold, for which alone its kernel shapes are compiled,
// the leading layers' dense FFN and the others' MoE of 3-bit experts, its
// bf16 router as stored and its shared expert two slices of the routed
// width, without a gate; its streamed experts are the MoE layers' alone, and
// streamableWeights plans the bytes it loads.
void testSyntheticFlintModel(MetalBackend &backend, const std::filesystem::path &root) {
    FlintLayout target;
    target.layers = 4;
    target.hiddenSize = 256;
    target.vocabularySize = 256;
    target.attentionWidth = 512; // four query heads of 128
    target.attentionQueryHeads = 4;
    target.attentionKvHeads = 1;
    target.attentionHeadDimension = 128;
    target.packedFullWidth = 512 + 2 * 128;
    target.slidingWindow = 33;
    target.slidingLayers = 0b0110;
    target.unrotatedLayers = 0b1001;
    target.firstDraftMaskToken = 240;
    target.lastDraftMaskToken = 255;
    target.maskToken = 240;
    target.stopTokens = {1, 2};
    target.intermediateSize = 256;
    target.experts = 8;
    target.expertsPerToken = 2;
    target.expertIntermediateSize = 256;
    target.leadingDenseLayers = 2;
    const auto bitsAndGroup = [](const std::string &module) {
        return module.find(".switch_mlp.") != std::string::npos ? std::pair{3u, 64u}
               : module.ends_with("embed_tokens")                 ? std::pair{8u, 64u}
                                                                  : std::pair{4u, 64u};
    };
    std::vector<SyntheticTensor> tensors = splash::test::mlxTargetTensors(target, bitsAndGroup);
    // Each query head's sink a bf16 value of its own: head h's h + 1.5.
    const auto sinkOf = [](uint32_t head) { return static_cast<float>(head) + 1.5F; };
    for (SyntheticTensor &tensor : tensors)
        if (tensor.name.ends_with("softmax_offset"))
            for (uint32_t head = 0; head < target.attentionQueryHeads; ++head) {
                const float value = sinkOf(head);
                uint32_t bits;
                std::memcpy(&bits, &value, sizeof bits);
                tensor.data.push_back(static_cast<uint8_t>(bits >> 16));
                tensor.data.push_back(static_cast<uint8_t>(bits >> 24));
            }
    const ModelDescriptor descriptor = writeBlockModel(root, target, tensors);
    const uint64_t planned = splash::model::streamableWeights(root, descriptor, {}).loadedBytes;
    const auto model = loadModel(backend, root, descriptor);
    const auto &weights = std::get<FlintWeights>(model.target);
    const uint32_t af4g64 = quant_affine_format_of(4, 64), af3g64 = quant_affine_format_of(3, 64);
    for (uint32_t index = 0; index < target.layers; ++index) {
        const auto &layer = weights.layers[index];
        const auto &attention = std::get<QwenAttentionWeights>(layer.mixer);
        const auto *sinks = static_cast<const float *>(attention.sinks.contents());
        bool widened = attention.sinks.sizeBytes() >= target.attentionQueryHeads * sizeof(float);
        for (uint32_t head = 0; widened && head < target.attentionQueryHeads; ++head)
            widened = sinks[head] == sinkOf(head);
        require(widened && segmentFormats(attention.inputProjection) ==
                               std::vector<uint32_t>{af4g64, af4g64, af4g64} &&
                    attention.inputProjection.outputSize == target.packedFullWidth && layer.inputNorm.float32 &&
                    layer.postAttentionNorm.float32 && attention.queryNorm.float32,
                "an MLX Flint layer did not read its ungated attention, its F32 sinks and its F32 norms");
        const splash::ops::BlockMoeWeights *moe = splash::model::qwenMoeBlock(layer);
        require((moe != nullptr) == (index >= target.leadingDenseLayers) &&
                    splash::model::denseFfn(layer).has_value() == (index < target.leadingDenseLayers),
                "an MLX Flint layer did not read its dense FFN or its MoE");
        if (!moe) continue;
        require(moe->gate.routed.formatId == af3g64 && moe->down.routed.formatId == af3g64 &&
                    moe->router.isFloat() && moe->router.formatId == GGUF_FMT_BF16 &&
                    moe->down.shared.formatId == af4g64,
                "an MLX Flint MoE did not keep its 3-bit experts, its bf16 router and its shared expert");
    }
    require(declaredBytes(weights.files) + declaredBytes(model.draft.files) == planned,
            "streamableWeights is not what an MLX Flint loads");
    const splash::model::SafetensorsCheckpoint safetensors(root / "target");
    const std::vector<splash::model::gguf::Image> images =
        splash::model::safetensors::planImages(safetensors, target, splash::model::safetensors::normStorage(safetensors, target));
    const auto repackOf = [&](uint32_t layer, const std::string &module) -> const splash::model::gguf::Repack * {
        for (const splash::model::gguf::Repack &repack : images[layer].repacks)
            if (!repack.sources.empty() && repack.sources.front().name == module) return &repack;
        return nullptr;
    };
    const auto *down = repackOf(2, "model.layers.2.mlp.shared_expert.down_proj");
    require(down && down->rows == 2 * target.hiddenSize && down->columns == target.expertIntermediateSize &&
                down->sources.size() == 2 && down->sources[0].windows == 2 && down->sources[1].window == 1,
            "an MLX Flint's shared expert down projection is not two slices of the routed width");
    splash::model::WeightImages streamedImages(backend, "sources");
    splash::model::SafetensorsTargetLoader streamed(backend, streamedImages, root / "target",
                                                     splash::model::safetensorsTargetPlan(target), {},
                                                     {.memoryBytes = 64ull << 20});
    require(streamed.streamedExperts().blocks.size() == target.layers - target.leadingDenseLayers,
            "an MLX Flint's streamed experts are not its MoE layers'");
    // Its q and k norms at 1.0625, nearer Flint's own mean less 1 (0.707)
    // than its mean (1.707): its norms are stored 1 below their multipliers,
    // and its zero input norms load as 1 + w. Read by the Qwen hybrids' means
    // (0.491 and 1.491), the same value is the multipliers: each family's
    // storage is told by its own mean.
    for (SyntheticTensor &tensor : tensors)
        if (tensor.name.ends_with("q_norm.weight") || tensor.name.ends_with("k_norm.weight"))
            for (uint64_t at = 0; at < tensor.data.size(); at += 2) tensor.data[at] = 0x88, tensor.data[at + 1] = 0x3F;
    splash::test::writeSyntheticShard(root / "target" / "model.safetensors", tensors);
    const auto below = loadModel(backend, root, descriptor);
    const splash::ops::NormWeights &inputNorm = std::get<FlintWeights>(below.target).layers[0].inputNorm;
    require(inputNorm.float32 && static_cast<const float *>(inputNorm.buffer.contents())[0] == 1.0f,
            "an MLX Flint whose q and k norms are stored 1 below its multipliers did not load its norms as 1 + w");
}

// A small Qwen3.8-Flash-Next of mlx-vlm's names and mlx-community's formats
// (af4g32, its router and shared-expert gate af8g64), its PLE table three
// shards of 64 rows, its hash the embedding's I64 buffers: two layers, a GDN
// layer that feeds the PLE and an attention layer with its QSA indexer.
Qwen4ExpLayout syntheticFlashNext() {
    Qwen4ExpLayout target;
    target.layers = 2;
    target.hiddenSize = 256;
    target.vocabularySize = 256;
    target.draftVocabulary = {};
    target.gdnKeyHeads = 2;
    target.gdnValueHeads = 2;
    target.gdnHeadDimension = 128;
    target.convolutionDimension = 768; // q and k of 2 heads, v of 2
    target.packedGdnWidth = 768 + 256 + 64; // qkv | z | beta, alpha, to whole 64-column tiles
    target.attentionWidth = 256; // two query heads of 128
    target.attentionQueryHeads = 2;
    target.attentionKvHeads = 1;
    target.attentionHeadDimension = 128;
    target.packedFullWidth = 2 * 256 + 2 * 128;
    target.fullAttentionPeriod = 2; // layer 1
    target.maskToken = 250;
    target.stopTokens = {1, 2};
    target.experts = 8;
    target.expertsPerToken = 2;
    target.expertIntermediateSize = 256;
    target.hyperConnections = 4;
    target.hyperRank = 64;
    target.indexerHeads = 2;
    target.indexerHeadDimension = 64;
    target.indexerTopBlocks = 2;
    target.indexerBlockTokens = 4;
    target.pleLayer = 0;
    target.pleNgram = 3;
    target.pleHeadsPerNgram = 2;
    target.pleHeadDimension = 64;
    target.pleConvolutionTaps = 4;
    return target;
}

constexpr uint32_t kPleShards = 3, kPleShardRows = 64;

// Its tensors: every hyper-connection and PLE norm a stored w of 0.5 (1.5
// multiplies), the hash's buffers and the table's shards of random codes,
// scales and biases; zeros elsewhere.
std::vector<SyntheticTensor> flashNextTensors(const Qwen4ExpLayout &g) {
    std::vector<SyntheticTensor> result;
    uint32_t seed = 4000;
    const auto random = [&](uint64_t bytes) {
        std::vector<uint8_t> data(bytes);
        for (uint8_t &byte : data) byte = static_cast<uint8_t>((seed = seed * 1664525u + 1013904223u) >> 24);
        return data;
    };
    const auto bfloat16 = [&](const std::string &name, std::vector<uint64_t> shape, std::vector<uint8_t> data = {}) {
        result.push_back({name, "BF16", std::move(shape), std::move(data)});
    };
    const auto half = [&](uint64_t elements) { // bf16 0.5 each
        std::vector<uint8_t> data;
        for (uint64_t i = 0; i < elements; ++i) data.insert(data.end(), {0x00, 0x3F});
        return data;
    };
    const auto quantized = [&](const std::string &module, uint64_t rows, uint64_t columns, uint64_t experts = 1,
                               bool random32 = false) {
        const bool wide = module.ends_with(".gate") || module.ends_with("shared_expert_gate");
        const uint32_t bits = wide ? 8 : 4, group = wide ? 64 : 32;
        const auto shape = [&](uint64_t last) {
            return experts > 1 ? std::vector<uint64_t>{experts, rows, last} : std::vector<uint64_t>{rows, last};
        };
        const uint64_t codes = rows * columns * bits / 32, groups = rows * columns / group;
        result.push_back({module + ".weight", "U32", shape(columns * bits / 32), random32 ? random(4 * codes) : std::vector<uint8_t>{}});
        bfloat16(module + ".scales", shape(columns / group), random32 ? random(2 * groups) : std::vector<uint8_t>{});
        bfloat16(module + ".biases", shape(columns / group), random32 ? random(2 * groups) : std::vector<uint8_t>{});
    };
    const auto int64s = [&](const std::string &name, std::vector<int64_t> values) {
        std::vector<uint8_t> data(values.size() * sizeof(int64_t));
        std::memcpy(data.data(), values.data(), data.size());
        result.push_back({name, "I64", {values.size()}, std::move(data)});
    };
    const uint64_t hidden = g.hiddenSize, streams = g.streamWidth();
    const auto hyper = [&](const std::string &stem, bool inject) {
        bfloat16(stem + "hc_norm.weight", {streams}, half(streams));
        quantized(stem + "input_mix_weight_down", g.hyperRank, streams);
        quantized(stem + "input_mix_weight_up", streams, g.hyperRank);
        if (inject) quantized(stem + "block_inject_weight", g.hyperConnections, streams);
    };
    for (uint32_t layer = 0; layer < g.layers; ++layer) {
        const std::string p = "language_model.model.layers." + std::to_string(layer) + ".";
        hyper(p + "attn_hyper_connection.", true);
        if (g.isFullAttentionLayer(layer)) {
            const std::string a = p + "self_attn.";
            quantized(a + "q_proj", 2ull * g.attentionWidth, hidden);
            quantized(a + "k_proj", 128, hidden);
            quantized(a + "v_proj", 128, hidden);
            // 0.5, nearer Flash-Next's 1.454 less 1 (the storage's) than 1.454.
            bfloat16(a + "q_norm.weight", {g.attentionHeadDimension}, half(g.attentionHeadDimension));
            bfloat16(a + "k_norm.weight", {g.attentionHeadDimension}, half(g.attentionHeadDimension));
            quantized(a + "o_proj", hidden, g.attentionWidth);
            quantized(a + "indexer.index_qk_proj", uint64_t{g.indexerHeads + 1} * g.indexerHeadDimension, hidden, 1,
                      true);
            bfloat16(a + "indexer.q_layernorm.weight", {g.indexerHeadDimension});
            bfloat16(a + "indexer.k_layernorm.weight", {g.indexerHeadDimension});
        } else {
            const std::string l = p + "linear_attn.";
            quantized(l + "in_proj_qkv", g.convolutionDimension, hidden);
            quantized(l + "in_proj_z", g.attentionWidth, hidden);
            quantized(l + "in_proj_b", g.gdnValueHeads, hidden);
            quantized(l + "in_proj_a", g.gdnValueHeads, hidden);
            bfloat16(l + "conv1d.weight", {g.convolutionDimension, splash::model::kGdnConvolutionTaps, 1});
            bfloat16(l + "A_log", {g.gdnValueHeads});
            bfloat16(l + "dt_bias", {g.gdnValueHeads});
            bfloat16(l + "norm.weight", {g.gdnHeadDimension});
            quantized(l + "out_proj", hidden, g.attentionWidth);
        }
        hyper(p + "mlp_hyper_connection.", true);
        const std::string m = p + "mlp.";
        quantized(m + "gate", g.experts, hidden);
        quantized(m + "switch_mlp.gate_proj", g.expertIntermediateSize, hidden, g.experts);
        quantized(m + "switch_mlp.up_proj", g.expertIntermediateSize, hidden, g.experts);
        quantized(m + "switch_mlp.down_proj", hidden, g.expertIntermediateSize, g.experts);
        quantized(m + "shared_expert.gate_proj", g.expertIntermediateSize, hidden);
        quantized(m + "shared_expert.up_proj", g.expertIntermediateSize, hidden);
        quantized(m + "shared_expert.down_proj", hidden, g.expertIntermediateSize);
        quantized(m + "shared_expert_gate", 1, hidden);
        if (layer != g.pleLayer) continue;
        const std::string e = p + "ple.";
        quantized(e + "key_proj", streams, g.pleWidth());
        quantized(e + "value_proj", hidden, g.pleWidth());
        for (const char *norm : {"norm_key", "norm_query", "norm_conv"})
            bfloat16(e + norm + ".weight", {streams}, half(streams));
        bfloat16(e + "conv1d.weight", {streams, g.pleConvolutionTaps, 1});
        int64s(e + "ple_embedding.layer_multipliers", {1000003, 999983, 0});
        int64s(e + "ple_embedding.ngram_heads_offsets", {0, 47, 94, 141});
        int64s(e + "ple_embedding.ngram_heads_vocab_sizes", {47, 47, 47, 47});
        for (uint32_t shard = 0; shard < kPleShards; ++shard)
            quantized(e + "ple_embedding.ngram_embedding.shards." + std::to_string(shard), kPleShardRows,
                      g.pleHeadDimension, 1, true);
    }
    hyper("language_model.model.hyper_connection_mixer.", false);
    quantized("language_model.lm_head", g.vocabularySize, hidden);
    quantized("language_model.model.embed_tokens", g.vocabularySize, hidden);
    return result;
}

// The af4g32 native row of row `row` of a synthetic module of `columns`
// inputs: per group of 32 its scale, its bias and its 16 bytes of codes
// (metal/abi/QuantFormat.h), from the module's tensors as MLX stores them.
std::vector<uint8_t> af4g32Row(const std::vector<SyntheticTensor> &tensors, const std::string &module, uint64_t row,
                               uint64_t columns) {
    const auto data = [&](const std::string &name) -> const std::vector<uint8_t> & {
        return std::find_if(tensors.begin(), tensors.end(), [&](const SyntheticTensor &t) { return t.name == name; })
            ->data;
    };
    const auto &codes = data(module + ".weight"), &scales = data(module + ".scales"),
               &biases = data(module + ".biases");
    const uint64_t groups = columns / 32;
    std::vector<uint8_t> result;
    for (uint64_t group = 0; group < groups; ++group) {
        const uint64_t at = (row * groups + group) * 2, code = (row * groups + group) * 16;
        result.insert(result.end(), scales.begin() + at, scales.begin() + at + 2);
        result.insert(result.end(), biases.begin() + at, biases.begin() + at + 2);
        result.insert(result.end(), codes.begin() + code, codes.begin() + code + 16);
    }
    return result;
}

// An MLX Qwen3.8-Flash-Next loads as the block images its GGUF's do: its
// hyper-connection and PLE norms, which mlx-vlm keeps 1 below the multiplier,
// as the F32 1 + w, its indexer's queries and key the rows of index_qk_proj
// as float segments, its PLE hash from the embedding's buffers and its table
// from its shards, loaded or streamed: a streamed row is its shard's codes,
// scales and biases as the af4g32 native row of each group's scale, bias and
// codes.
void testSyntheticFlashNext(MetalBackend &backend, const std::filesystem::path &root) {
    const Qwen4ExpLayout target = syntheticFlashNext();
    const std::vector<SyntheticTensor> tensors = flashNextTensors(target);
    splash::test::writeSyntheticShard(root / "target" / "model.safetensors", tensors);
    splash::model::WeightImages images(backend, "sources");
    splash::model::SafetensorsTargetLoader loaded(backend, images, root / "target",
                                                   splash::model::safetensorsTargetPlan(target));
    const Qwen4ExpWeights weights =
        splash::model::loadQwen4ExpWeights(backend, target, splash::model::QwenTargetFiles{std::ref(loaded)});
    const auto first = [](const splash::ops::NormWeights &norm) {
        return norm.float32 ? static_cast<const float *>(norm.buffer.contents())[0] : -1.0f;
    };
    const auto &layer0 = weights.layers[0], &layer1 = weights.layers[1];
    require(first(layer0.mixerHyper.norm) == 1.5F && first(layer1.ffnHyper.norm) == 1.5F &&
                first(weights.finalHyper.norm) == 1.5F && layer0.ple && first(layer0.ple->keyNorm) == 1.5F &&
                first(layer0.ple->convolutionNorm) == 1.5F,
            "an MLX Flash-Next's hyper-connection and PLE norms are not the F32 1 + w");
    require(layer1.indexer && layer1.indexer->query.outputSize == target.indexerHeads * target.indexerHeadDimension &&
                layer1.indexer->key.outputSize == target.indexerHeadDimension &&
                layer1.indexer->query.blocks().segments.size() == 1 &&
                layer1.indexer->query.blocks().segments.front().quantizedRows() &&
                layer1.indexer->key.blocks().segments.front().quantizedRows() && first(layer1.indexer->queryNorm) == 1.0F,
            "an MLX Flash-Next's indexer is not index_qk_proj's rows as float segments");
    // The queries are its rows from 0 on, the key its rows from the queries' end on.
    const std::string qk = "language_model.model.layers.1.self_attn.indexer.index_qk_proj";
    for (const auto &[projection, row] : {std::pair{&layer1.indexer->query, uint64_t{0}},
                                          std::pair{&layer1.indexer->key, uint64_t{target.indexerHeads} *
                                                                              target.indexerHeadDimension}}) {
        const std::vector<uint8_t> expected = af4g32Row(tensors, qk, row, target.hiddenSize);
        const auto *rows = static_cast<const uint8_t *>(projection->blocks().segments.front().plane0.contents());
        require(std::equal(expected.begin(), expected.end(), rows),
                "an MLX Flash-Next's indexer row " + std::to_string(row) + " of index_qk_proj is not its first");
    }
    require(weights.pleHash.multipliers[0] == 1000003 && weights.pleHash.multipliers[1] == 999983 &&
                weights.pleHash.headOffsets == std::vector<uint32_t>{0, 47, 94, 141} &&
                weights.pleHash.headVocabularies == std::vector<uint32_t>(4, 47) &&
                weights.pleHash.tableRows == kPleShards * kPleShardRows && !weights.streamedTable &&
                weights.pleTable.rows().formatId == quant_affine_format_of(4, 32),
            "an MLX Flash-Next's PLE hash or table is not its checkpoint's");

    // Streamed, rows across the shards' boundaries.
    splash::model::WeightImages streamedImages(backend, "sources");
    splash::model::SafetensorsTargetLoader streaming(
        backend, streamedImages, root / "target", splash::model::safetensorsTargetPlan(target),
        {.cacheBytes = 1 << 20, .readers = std::make_shared<splash::model::SsdReaders>(splash::model::kSsdReaders)});
    const Qwen4ExpWeights streamed =
        splash::model::loadQwen4ExpWeights(backend, target, splash::model::QwenTargetFiles{std::ref(streaming)});
    require(streamed.streamedTable && streamed.pleFormat == quant_affine_format_of(4, 32),
            "an MLX Flash-Next's table did not stream");
    const uint32_t rowBytes = streamed.streamedTable->table().rowBytes;
    const std::vector<uint32_t> ids{0, 63, 64, 191, 130};
    std::vector<uint8_t> rows(ids.size() * rowBytes);
    std::vector<uint32_t> slots(ids.size());
    streamed.streamedTable->stage(ids, rows, slots, splash::model::SparseRowUse::Decode);
    for (size_t index = 0; index < ids.size(); ++index) {
        const uint32_t shard = ids[index] / kPleShardRows, row = ids[index] % kPleShardRows;
        const std::vector<uint8_t> expected = af4g32Row(
            tensors, "language_model.model.layers.0.ple.ple_embedding.ngram_embedding.shards." + std::to_string(shard),
            row, target.pleHeadDimension);
        require(expected.size() == rowBytes &&
                    std::equal(expected.begin(), expected.end(), rows.begin() + uint64_t{slots[index]} * rowBytes),
                "an MLX Flash-Next's streamed PLE row " + std::to_string(ids[index]) +
                    " is not its shard's group scales, biases and codes");
    }
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
        testReservedImages(backend, temporary.path());
        testRestoreAdmission(backend, temporary.path());
        testRestoreWritesEachImageOnItsThread(backend, temporary.path());
        testGgufImageLayout(backend, temporary.path());
        // Block images are repacked by the production library's kernel.
        MetalBackend production(argv[2]);
        testSyntheticModel(production, temporary.path() / "model");
        testSyntheticBlockModel(production, temporary.path() / "block-model");
        testSyntheticBlockMoeModel(production, temporary.path() / "block-moe-model");
        testSyntheticFlintModel(production, temporary.path() / "flint-model");
        testSyntheticFlashNext(production, temporary.path() / "flash-next");
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
