#include "TestBuffers.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/ModelFactory.hpp"
#include "model/PreparedFiles.hpp"
#include "model/WeightLayout.hpp"
#include "ops/Embedding.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
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
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

using splash::model::DFlashDraftLayout;
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

bool addressIsMapped(void *address) {
    long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0) fail("unable to determine page size");
    char state = 0;
    errno = 0;
    return mincore(address, static_cast<size_t>(pageSize), &state) == 0;
}

void requireCleanFileMapping(void *pointer) {
    mach_vm_address_t address = reinterpret_cast<mach_vm_address_t>(pointer);
    mach_vm_size_t size = 0;
    natural_t depth = 0;
    vm_region_submap_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;
    require(mach_vm_region_recurse(
                mach_task_self(), &address, &size, &depth,
                reinterpret_cast<vm_region_recurse_info_t>(&info), &count) ==
                KERN_SUCCESS && address <= reinterpret_cast<uintptr_t>(pointer) &&
                reinterpret_cast<uintptr_t>(pointer) - address < size,
            "unable to inspect weight mapping");
    require(info.protection == VM_PROT_READ,
            "weight mapping is not read-only");
    require(info.external_pager && info.shadow_depth == 0 &&
                info.pages_dirtied == 0 && info.pages_swapped_out == 0,
            "GPU read turned file-backed weights into private dirty pages");
}

void testWeightFileValidationAndLifetime(MetalBackend &backend,
                                         const std::filesystem::path &root) {
    constexpr uint32_t elementCount = kWeightFileAlignment / sizeof(uint32_t);
    std::array<uint32_t, elementCount> expected{};
    for (uint32_t i = 0; i < elementCount; ++i) expected[i] = i * 17 + 3;
    std::array<uint64_t, 1> sections{sizeof(expected)};
    auto validPath = root / "valid.bin";
    uint64_t fileBytes = writeWeightFile(
        validPath, "TEST0001", 7, 9, sections);
    int descriptor = open(validPath.c_str(), O_WRONLY | O_CLOEXEC);
    require(descriptor >= 0, "unable to open synthetic payload");
    require(pwrite(descriptor, expected.data(), sizeof(expected),
                   kWeightFileAlignment) == static_cast<ssize_t>(sizeof(expected)) &&
                fsync(descriptor) == 0,
            "unable to persist synthetic payload before mapping");
    close(descriptor);
    uint64_t baseline = backend.memoryStats().allocatedBytes;
    MetalBuffer retained;
    void *mappedAddress = nullptr;
    {
        WeightFile file(
            backend, validPath, "test/valid.bin", "TEST0001", 7, 9);
        retained = file.section(sizeof(expected), "payload");
        mappedAddress = retained.contents();
        require(mappedAddress != nullptr, "mapped section is not CPU-visible");
        require(reinterpret_cast<uintptr_t>(mappedAddress) %
                    kWeightFileAlignment == 0,
                "mapped section start is not 16 KiB-aligned");
        file.finish();
        require(backend.memoryStats().allocatedBytes >= baseline + fileBytes,
                "zero-copy base allocation was not tracked");
    }
    require(addressIsMapped(mappedAddress),
            "mapping disappeared while a Metal view remained alive");
    {
        MetalBuffer output = backend.allocateBuffer(
            sizeof(expected), BufferStorage::Shared, "weight-readback");
        splash::metal::ComputeDispatch dispatch;
        dispatch.pipelineName = "test_copy_u32";
        dispatch.buffers = {{0, retained}, {1, output}};
        dispatch.bytes = {{2, &elementCount, sizeof(elementCount)}};
        dispatch.threadgroups = {(elementCount + 31) / 32, 1, 1};
        dispatch.threadsPerThreadgroup = {32, 1, 1};
        (void)backend.submit(dispatch);
        require(std::memcmp(output.contents(), expected.data(), sizeof(expected)) == 0,
                "GPU read of retained mapped weights was incorrect");
        requireCleanFileMapping(mappedAddress);
    }
    retained = MetalBuffer{};
    require(backend.memoryStats().allocatedBytes == baseline,
            "released mapped buffer remains in backend accounting");

    long pageSize = sysconf(_SC_PAGESIZE);
    require(pageSize > 0, "unable to determine page size");
    uint64_t ownerBytes = alignPacked(static_cast<uint64_t>(pageSize));
    void *ownerAddress = mmap(
        nullptr, static_cast<size_t>(ownerBytes), PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANON, -1, 0);
    require(ownerAddress != MAP_FAILED,
            "unable to create shared-memory lifetime test mapping");
    bool ownerReleased = false;
    std::shared_ptr<void> owner(
        ownerAddress, [&](void *address) {
            ownerReleased =
                munmap(address, static_cast<size_t>(ownerBytes)) == 0;
        });
    MetalBuffer base = backend.wrapSharedMemory(
        ownerAddress, ownerBytes, owner, "lifetime-owner-test");
    MetalBuffer ownerView = backend.view(base, 0, 64);
    owner.reset();
    base = MetalBuffer{};
    require(!ownerReleased,
            "shared-memory owner was released while a view remained alive");
    ownerView = MetalBuffer{};
    require(ownerReleased,
            "shared-memory owner was not released with its final view");

    requirePackedError(
        [&] {
            WeightFile wrong(
                backend, validPath, "test/valid.bin", "WRONG000", 7, 9);
        },
        "wrong packed magic was accepted");
    requirePackedError(
        [&] {
            WeightFile wrong(
                backend, validPath, "test/valid.bin", "TEST0001", 8, 9);
        },
        "wrong packed layer was accepted");
    requirePackedError(
        [&] {
            WeightFile wrong(
                backend, validPath, "test/valid.bin", "TEST0001", 7, 8);
        },
        "wrong packed type was accepted");
    requirePackedError(
        [&] {
            WeightFile truncated(
                backend, validPath, "test/valid.bin", "TEST0001", 7, 9);
            (void)truncated.section(fileBytes, {});
        },
        "truncated packed section was accepted");

    auto extraPath = root / "extra.bin";
    std::array<uint64_t, 2> extraSections{64, 64};
    writeWeightFile(extraPath, "TEST0001", 1, 2, extraSections);
    requirePackedError(
        [&] {
            WeightFile extra(
                backend, extraPath, "test/extra.bin", "TEST0001", 1, 2);
            (void)extra.section(64, {});
            extra.finish();
        },
        "unconsumed packed bytes were accepted");

    auto unalignedPath = root / "unaligned.bin";
    writeWeightFile(unalignedPath, "TEST0001", 1, 2, sections);
    require(truncate(unalignedPath.c_str(),
                     static_cast<off_t>(fileBytes - 1)) == 0,
            "unable to truncate synthetic file");
    requirePackedError(
        [&] {
            WeightFile unaligned(
                backend, unalignedPath, "test/unaligned.bin", "TEST0001",
                1, 2);
        },
        "unaligned packed file size was accepted");
}

// A prepared file is mapped only as the cache verified it: a file put in its
// place after prepare returned it is refused, whatever its bytes.
void testPreparedFileMapsOnlyAsVerified(MetalBackend &backend, const std::filesystem::path &root) {
    const std::filesystem::path cache = root / "verified-cache";
    std::filesystem::create_directories(cache);
    setenv("SPLASH_WEIGHT_CACHE", cache.c_str(), 1);
    const auto header = splash::model::weightFileHeader("TEST0001", 3, 4);
    const splash::model::PreparedWeight weight{std::string(64, 'c'), 2 * kWeightFileAlignment, "test/prepared.bin",
                                               std::string(64, 'd'), "/test"};
    const splash::model::WeightWriter write = [&](int destination, const splash::model::PreparationCheck &) {
        splash::model::writeWeightBytes(destination, 0, header);
    };
    const splash::model::PreparedFiles files({}, {}, {});
    static_cast<void>(files.open(backend, weight, write, "TEST0001", 3, 4));
    const std::filesystem::path path = files.prepare(weight, write);
    std::filesystem::rename(path, root / "replaced.bin");
    const std::array<uint64_t, 1> sections{64};
    writeWeightFile(path, "TEST0001", 3, 4, sections);
    bool refused = false;
    try {
        WeightFile replaced(backend, path, weight.component, "TEST0001", 3, 4, weight.key);
    } catch (const std::exception &error) {
        refused = std::string_view(error.what()).find("changed after verification") != std::string_view::npos;
    }
    require(refused, "a file replacing a verified prepared file was mapped");
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
    const auto mapped = [&](const std::filesystem::path &path) {
        return WeightFile(backend, path, "test/" + path.filename().string(), kGgufImageMagic, 0, 0);
    };
    {
        // finish() proves the reader took exactly the descriptor, plane0 and
        // meta sections; the segment's format comes from the descriptor.
        WeightFile file = mapped(projection);
        const auto read = splash::model::readBlockProjection(file, rows, columns, "projection");
        file.finish();
        const auto &segment = read.blocks().segments.at(0);
        require(segment.formatId == GGUF_FMT_Q80 && !segment.plane1, "GGUF projection did not read a Q8_0 segment");
    }
    for (const auto [output, input] : {std::pair{2 * rows, columns}, std::pair{rows, 2 * columns}}) {
        requirePackedError(
            [&] {
                WeightFile file = mapped(projection);
                (void)splash::model::readBlockProjection(file, output, input, "projection");
            },
            "GGUF projection of other sizes than the layout's was accepted");
        requirePackedError(
            [&] {
                WeightFile file = mapped(embedding);
                (void)splash::model::readBlockEmbedding(file, output, input, "embedding");
            },
            "GGUF embedding of other sizes than the layout's was accepted");
    }
    WeightFile file = mapped(embedding);
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

// An unquantized MLX vision tower of layout, every value zero: the
// vision_tower.* tensors the vision loader binds.
void writeMlxVisionTower(const std::filesystem::path &directory, const VisionLayout &layout) {
    std::vector<std::pair<std::string, std::vector<uint64_t>>> tensors;
    const auto affine = [&](const std::string &name, uint64_t rows, uint64_t columns) {
        tensors.push_back({name + ".weight", {rows, columns}});
        tensors.push_back({name + ".bias", {rows}});
    };
    const auto norm = [&](const std::string &name) {
        tensors.push_back({name + ".weight", {layout.hiddenSize}});
        tensors.push_back({name + ".bias", {layout.hiddenSize}});
    };
    tensors.push_back({"patch_embed.proj.weight", {layout.hiddenSize, 2, layout.patchSize, layout.patchSize, 3}});
    tensors.push_back({"patch_embed.proj.bias", {layout.hiddenSize}});
    tensors.push_back({"pos_embed.weight",
                       {uint64_t{layout.positionGridSide} * layout.positionGridSide, layout.hiddenSize}});
    for (uint32_t block = 0; block < layout.depth; ++block) {
        const std::string prefix = "blocks." + std::to_string(block) + ".";
        norm(prefix + "norm1");
        affine(prefix + "attn.qkv", 3 * layout.hiddenSize, layout.hiddenSize);
        affine(prefix + "attn.proj", layout.hiddenSize, layout.hiddenSize);
        norm(prefix + "norm2");
        affine(prefix + "mlp.linear_fc1", layout.intermediateSize, layout.hiddenSize);
        affine(prefix + "mlp.linear_fc2", layout.hiddenSize, layout.intermediateSize);
    }
    norm("merger.norm");
    affine("merger.linear_fc1", layout.mergedHiddenSize, layout.mergedHiddenSize);
    affine("merger.linear_fc2", layout.outputHiddenSize, layout.mergedHiddenSize);
    std::string header;
    uint64_t dataBytes = 0;
    for (const auto &[name, shape] : tensors) {
        uint64_t bytes = 2;
        std::string dimensions;
        for (uint64_t dimension : shape) {
            bytes *= dimension;
            dimensions += (dimensions.empty() ? "" : ",") + std::to_string(dimension);
        }
        header += (header.empty() ? "{" : ",") + std::string("\"vision_tower.") + name +
                  "\":{\"dtype\":\"BF16\",\"shape\":[" + dimensions + "],\"data_offsets\":[" +
                  std::to_string(dataBytes) + "," + std::to_string(dataBytes + bytes) + "]}";
        dataBytes += bytes;
    }
    header += "}";
    std::ofstream(directory / "config.json") << "{}";
    std::ofstream file(directory / "model.safetensors", std::ios::binary);
    const uint64_t headerBytes = header.size();
    file.write(reinterpret_cast<const char *>(&headerBytes), sizeof headerBytes);
    file << header << std::string(dataBytes, '\0');
}

// Every prepared file of a model is budgeted before the first is written: a
// prepared vision tower beside a packed target that the disk cannot hold
// fails the load before conversion is admitted and writes nothing.
void testModelDiskCheck(MetalBackend &backend, const std::filesystem::path &root, const Qwen3_8Layout &target,
                        const DFlashDraftLayout &draft, VisionLayout vision) {
    const std::filesystem::path cache = root / "cache";
    std::filesystem::create_directories(cache);
    setenv("SPLASH_WEIGHT_CACHE", cache.c_str(), 1);
    writeMlxVisionTower(root / "vision", vision);
    // The padding sizes the prepared file alone; the source keeps its shapes.
    // Grow it until the prepared tower exceeds this volume's free space.
    const uint64_t available = std::filesystem::space(cache).available;
    vision.paddedIntermediateSize = 1u << 20;
    while (splash::model::preparedVisionBytes(vision) <= available && vision.paddedIntermediateSize < (1u << 31))
        vision.paddedIntermediateSize *= 2;
    const uint64_t bytes = splash::model::preparedVisionBytes(vision);
    require(bytes > available, "the oversized vision tower fits on this volume");
    auto descriptor = makeModelDescriptor("Qwen dense disk check", target, draft, vision);
    descriptor.visionSource = splash::model::VisionSource::Mlx;
    bool admitted = false;
    std::string error;
    try {
        static_cast<void>(loadModelPackage(backend, root, descriptor, [&] { admitted = true; }));
    } catch (const std::exception &failure) {
        error = failure.what();
    }
    require(error.starts_with("not enough disk space to prepare weights: need " + std::to_string(bytes) + " bytes"),
            "the model disk check did not budget the vision tower: " + error);
    require(!admitted, "conversion was admitted before the model disk check");
    const auto planned = splash::model::planVisionLoader(backend, root, descriptor, {});
    require(!std::filesystem::exists(cache / planned->weight().key), "the model disk check wrote the vision tower");
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
        auto package = loadModelPackage(
            backend, root,
            makeModelDescriptor("Qwen dense loader oracle", target, draft,
                                vision), {});
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
        records.front().contentIdentity = std::string(64, 'a');
        const auto preparedIdentity = weightManifestFingerprint(records);
        require(preparedIdentity != package.manifestFingerprintSha256,
                "prepared content was omitted from runtime cache identity");
        records.front().contentIdentity = std::string(64, 'b');
        require(weightManifestFingerprint(records) != preparedIdentity,
                "same-shape different weights share a runtime cache identity");
        records.front().contentIdentity.clear();
        records.front().declaredBytes += kWeightFileAlignment;
        require(weightManifestFingerprint(records) !=
                    package.manifestFingerprintSha256,
                "manifest fingerprint ignores declared file sizes");

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
    }
    require(backend.memoryStats().allocatedBytes == baseline,
            "model package allocations survived package destruction");

    testModelDiskCheck(backend, root, target, draft, vision);

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
        auto package = loadModelPackage(backend, root, splash::model::inspectModelPackage(root), {});
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
            "real model mappings survived package destruction");
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
        testWeightFileValidationAndLifetime(backend, temporary.path());
        testPreparedFileMapsOnlyAsVerified(backend, temporary.path());
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
