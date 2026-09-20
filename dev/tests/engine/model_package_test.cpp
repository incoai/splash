#include "model/ModelFactory.hpp"

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
using splash::model::QwenAttentionQ8Weights;
using splash::model::QwenGdnQ8Weights;
using splash::model::Qwen3_8Q8Layout;
using splash::model::Qwen3_8Q8Weights;
using splash::ops::VisionLayout;
using splash::model::kWeightFileAlignment;
using splash::model::loadModelPackage;
using splash::model::makeModelDescriptor;
using splash::model::weightManifestFingerprint;
using splash::metal::BufferStorage;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

constexpr std::string_view kDraftLayerMagic = "MDFD0004";
constexpr std::string_view kTargetEmbeddingMagic = "MDFE0001";
constexpr std::string_view kTargetHeadMagic = "MDFL0002";
constexpr std::string_view kTargetLayerMagic = "MDFL0006";
constexpr std::string_view kTargetQ8LayerMagic = "MDFL0008";
constexpr std::string_view kTargetQ8HeadMagic = "MDFL0008";
constexpr std::string_view kTargetQ8EmbeddingMagic = "MDFE0008";
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

uint64_t q8Bytes(uint32_t outputSize, uint32_t inputSize) {
    return checkedProduct(outputSize, inputSize) * 17 / 16;
}

std::vector<uint64_t> targetLayerQ8Sections(
    const Qwen3_8Q8Layout &layout, bool full) {
    constexpr uint64_t bf16 = 2;
    std::vector<uint64_t> result{
        uint64_t(layout.hiddenSize) * bf16,
        q8Bytes(full ? layout.packedFullWidth : layout.packedGdnWidth,
                layout.hiddenSize),
    };
    if (full) {
        result.insert(result.end(), {
            uint64_t(layout.attentionHeadDimension) * bf16,
            uint64_t(layout.attentionHeadDimension) * bf16,
            q8Bytes(layout.hiddenSize, layout.attentionWidth),
        });
    } else {
        result.insert(result.end(), {
            uint64_t(layout.convolutionDimension) * 4 * bf16,
            uint64_t(layout.gdnValueHeads) * 4,
            uint64_t(layout.gdnValueHeads) * bf16,
            uint64_t(layout.gdnHeadDimension) * bf16,
            q8Bytes(layout.hiddenSize, layout.attentionWidth),
        });
    }
    result.insert(result.end(), {
        uint64_t(layout.hiddenSize) * bf16,
        q8Bytes(layout.intermediateSize, layout.hiddenSize),
        q8Bytes(layout.intermediateSize, layout.hiddenSize),
        q8Bytes(layout.hiddenSize, layout.intermediateSize),
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

SyntheticAccounting writeSyntheticQ8Package(
    const std::filesystem::path &root, const Qwen3_8Q8Layout &target,
    const DFlashDraftLayout &draft, const VisionLayout &vision) {
    SyntheticAccounting result;
    for (uint32_t layer = 0; layer < target.layers; ++layer) {
        bool full = target.isFullAttentionLayer(layer);
        auto sections = targetLayerQ8Sections(target, full);
        result.targetBytes += writeWeightFile(
            root / "target" / ("layer-" + std::to_string(layer) + ".bin"),
            kTargetQ8LayerMagic, layer, full ? 1U : 0U, sections);
    }
    std::array<uint64_t, 2> headSections{
        uint64_t(target.hiddenSize) * 2,
        q8Bytes(target.vocabularySize, target.hiddenSize),
    };
    result.targetBytes += writeWeightFile(
        root / "target/head.bin", kTargetQ8HeadMagic, target.layers, 2,
        headSections);
    uint64_t embeddingElements =
        uint64_t(target.vocabularySize) * target.hiddenSize;
    std::array<uint64_t, 3> embeddingSections{
        embeddingElements,
        embeddingElements / 32,
        embeddingElements / 32,
    };
    result.targetBytes += writeWeightFile(
        root / "target/embedding.bin", kTargetQ8EmbeddingMagic,
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
            (void)truncated.section(fileBytes);
        },
        "truncated packed section was accepted");

    auto extraPath = root / "extra.bin";
    std::array<uint64_t, 2> extraSections{64, 64};
    writeWeightFile(extraPath, "TEST0001", 1, 2, extraSections);
    requirePackedError(
        [&] {
            WeightFile extra(
                backend, extraPath, "test/extra.bin", "TEST0001", 1, 2);
            (void)extra.section(64);
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

void testSyntheticPackage(MetalBackend &backend,
                          const std::filesystem::path &root) {
    Qwen3_8Layout target;
    target.layers = 4;
    target.hiddenSize = 256;
    target.vocabularySize = 256;
    target.packedGdnWidth = 256;
    target.packedFullWidth = 256;
    target.convolutionDimension = 256;
    target.gdnKeyHeads = 2;
    target.gdnValueHeads = 4;
    target.gdnHeadDimension = 64;
    target.attentionWidth = 64;
    target.intermediateSize = 256;
    target.attentionQueryHeads = 1;
    target.attentionKvHeads = 1;
    target.attentionHeadDimension = 64;
    target.fullAttentionPeriod = 4;

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
                                vision));
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

    std::cout << "synthetic declared_target=" << expected.targetBytes
              << " declared_draft=" << expected.draftBytes
              << " declared_vision=" << expected.visionBytes
              << " actual_tracked=" << actualTrackedBytes << '\n';
}

void testSyntheticQ8Package(MetalBackend &backend,
                            const std::filesystem::path &root) {
    Qwen3_8Q8Layout target;
    target.layers = 4;
    target.hiddenSize = 256;
    target.vocabularySize = 256;
    target.packedGdnWidth = 256;
    target.packedFullWidth = 256;
    target.convolutionDimension = 256;
    target.gdnKeyHeads = 2;
    target.gdnValueHeads = 4;
    target.gdnHeadDimension = 64;
    target.attentionWidth = 64;
    target.intermediateSize = 256;
    target.attentionQueryHeads = 1;
    target.attentionKvHeads = 1;
    target.attentionHeadDimension = 64;
    target.fullAttentionPeriod = 4;

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
        writeSyntheticQ8Package(root, target, draft, vision);
    uint64_t baseline = backend.memoryStats().allocatedBytes;
    uint64_t actualTrackedBytes = 0;
    {
        auto package = loadModelPackage(
            backend, root,
            makeModelDescriptor("Qwen dense Q8 loader oracle", target, draft,
                                vision));
        const auto &loadedTarget = std::get<Qwen3_8Q8Weights>(package.target);
        require(loadedTarget.layers.size() == target.layers,
                "target Q8 layer vector is incomplete");
        require(package.draft.layers.size() == draft.layers,
                "draft layer vector is incomplete");
        require(std::holds_alternative<QwenGdnQ8Weights>(
                    loadedTarget.layers[0].mixer),
                "target GDN Q8 layer has the wrong typed layout");
        require(std::holds_alternative<QwenAttentionQ8Weights>(
                    loadedTarget.layers[3].mixer),
                "target full-attention Q8 layer has the wrong typed layout");
        require(loadedTarget.files.size() == target.layers + 2,
                "target Q8 file records are incomplete");
        require(package.draft.files.size() == draft.layers + 1,
                "draft file records are incomplete");
        require(declaredBytes(loadedTarget.files) == expected.targetBytes,
                "target Q8 declared byte accounting is wrong");
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
        actualTrackedBytes =
            backend.memoryStats().allocatedBytes - baseline;
        require(actualTrackedBytes >= expected.targetBytes +
                                          expected.draftBytes +
                                          expected.visionBytes,
                "backend actual allocation accounting is below logical bytes");
    }
    require(backend.memoryStats().allocatedBytes == baseline,
            "model package allocations survived package destruction");

    std::cout << "synthetic Q8 declared_target=" << expected.targetBytes
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
        auto package = loadModelPackage(backend, root);
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
    std::string_view originalPrefix = "splash-packed-q4";
    size_t offset = original.find(originalPrefix);
    if (offset == std::string::npos) {
        originalPrefix = "splash-packed-q8";
        offset = original.find(originalPrefix);
    }
    require(offset != std::string::npos, "model manifest lacks a known format");

    TempDirectory temporary;
    std::filesystem::create_directory(temporary.path() / "tokenizer");
    std::filesystem::copy_file(root / "tokenizer/config.json",
                               temporary.path() / "tokenizer/config.json");
    const auto expected = splash::model::inspectModelPackage(root);
    for (std::string_view name : {std::string_view(expected.name),
                                  std::string_view("Community fine-tune")}) {
        for (std::string_view prefix : {originalPrefix, std::string_view("unknown-packed-q4")}) {
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
        testSyntheticPackage(backend, temporary.path() / "package");
        testSyntheticQ8Package(backend, temporary.path() / "package_q8");
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
