#include "ops/PageStorage.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::kv {

static_assert(kv::kSparseMappingAlignmentBytes ==
                  metal::MetalBackend::kPlacementSparsePageBytes,
              "KV sparse mapping alignment must equal the Metal placement page");
namespace {

uint64_t checkedMultiply(uint64_t left, uint64_t right) {
    if (left && right > std::numeric_limits<uint64_t>::max() / left) {
        throw std::overflow_error("KV page storage size overflow");
    }
    return left * right;
}

}  // namespace

PageStorage::PageStorage(metal::MetalBackend &backend,
    metal::AllocationAdmission admitAllocation, Layout layout,
    uint32_t pageCount)
    : backend_(backend), admitAllocation_(std::move(admitAllocation)),
      layout_(layout), pageCount_(pageCount), layers_(layout.attentionLayers) {
    if (!admitAllocation_) {
        throw std::invalid_argument(
            "KV page storage requires physical allocation admission");
    }
    if (!layout_.valid()) {
        throw std::invalid_argument("KV page storage layout is invalid");
    }
    if (!pageCount || pageCount % sparseMappingBatchPages()) {
        throw std::invalid_argument(
            "KV page pool is not sparse-mapping aligned");
    }
    const StorageByteCounts bytes = layout_.storageByteCounts(pageCount);
    for (uint32_t layerIndex = 0; layerIndex < layers_.size();
         ++layerIndex) {
        std::string prefix = "kv-layer-" +
                             std::to_string(layerIndex) + "-";
        LayerStorage &storage = layers_[layerIndex];
        storage.format = layout_.format;
        storage.keyData = backend_.allocatePlacementSparseBuffer(
            checkedMultiply(pageCount, layout_.dataBytesPerLayerPage()),
            kSparseMappingAlignmentBytes, prefix + "keys");
        if (layout_.scaleBytesPerLayerPage()) {
            storage.keyScales = backend_.allocatePlacementSparseBuffer(
                checkedMultiply(pageCount, layout_.scaleBytesPerLayerPage()),
                kSparseMappingAlignmentBytes, prefix + "key-scales");
        }
        storage.valueData = backend_.allocatePlacementSparseBuffer(
            checkedMultiply(pageCount, layout_.dataBytesPerLayerPage()),
            kSparseMappingAlignmentBytes, prefix + "values");
        if (layout_.scaleBytesPerLayerPage()) {
            storage.valueScales = backend_.allocatePlacementSparseBuffer(
                checkedMultiply(pageCount, layout_.scaleBytesPerLayerPage()),
                kSparseMappingAlignmentBytes, prefix + "value-scales");
        }
    }
    if (declaredBytes() != bytes.total) {
        throw std::logic_error("KV storage accounting mismatch");
    }

    for (uint32_t first = 0; first < pageCount_;
         first += backingExtentPages()) {
        uint32_t count = std::min(backingExtentPages(), pageCount_ - first);
        if (count % sparseMappingBatchPages()) {
            throw std::logic_error("KV backing extent is not tile aligned");
        }
        extents_.push_back(Extent{first, count, std::nullopt});
    }
    // One small runway makes startup warmup and the first requests allocation
    // free. Every later extent remains virtual until real tokens need it.
    if (auto result = ensureResident(0); !result) {
        throw metal::MetalAllocationError(
            std::string("unable to allocate initial KV backing extent: ") +
                metal::allocationFailureName(result.failure), result.failure);
    }
}

PageStorage::~PageStorage() {
    // Pace GPU-written extent releases to avoid long stalls in the OS kernel.
    for (Extent &extent : extents_) {
        if (!backend_.healthy()) break;
        if (!extent.heap) continue;
        try {
            // unmapSparse waits for the previous extent's unmap itself.
            auto mappings = mappingsFor(extent);
            backend_.unmapSparse(mappings, std::move(*extent.heap));
        } catch (...) {
            // Shutdown cannot recover or safely report an exception. The
            // Metal resources are still released in deterministic C++ order.
        }
        extent.heap.reset();
    }
    if (!backend_.healthy()) return;
    try {
        backend_.drainSparseUnmaps();
    } catch (...) {
    }
}

uint64_t PageStorage::declaredBytes() const noexcept {
    return uint64_t(pageCount_) * layout_.bytesPerModelPage();
}

uint64_t PageStorage::actualAllocatedBytes() const noexcept {
    return residentBackingBytes_;
}

uint32_t PageStorage::residentPages() const noexcept {
    return residentPages_;
}

size_t PageStorage::extentIndex(uint32_t page) const {
    if (page >= pageCount_) throw std::out_of_range("invalid KV page id");
    return page / backingExtentPages();
}

uint32_t PageStorage::extentFirstPage(uint32_t page) const {
    return extents_.at(extentIndex(page)).firstPage;
}

uint32_t PageStorage::extentPageCount(uint32_t page) const {
    return extents_.at(extentIndex(page)).pageCount;
}

bool PageStorage::isResident(uint32_t page) const {
    return extents_.at(extentIndex(page)).heap.has_value();
}

std::vector<metal::SparseMapping> PageStorage::mappingsFor(
    const Extent &extent) const {
    std::vector<metal::SparseMapping> mappings;
    mappings.reserve(uint64_t{layout_.attentionLayers} * 4);
    uint64_t heapOffset = 0;
    auto append = [&](const metal::MetalBuffer &buffer,
                      uint64_t bytesPerPage) {
        if (!bytesPerPage) return;
        uint64_t bufferOffset = checkedMultiply(
            extent.firstPage, bytesPerPage);
        uint64_t size = checkedMultiply(extent.pageCount, bytesPerPage);
        mappings.push_back({buffer, bufferOffset, size, heapOffset});
        heapOffset += size;
    };
    for (const LayerStorage &storage : layers_) {
        append(storage.keyData, layout_.dataBytesPerLayerPage());
        append(storage.keyScales, layout_.scaleBytesPerLayerPage());
        append(storage.valueData, layout_.dataBytesPerLayerPage());
        append(storage.valueScales, layout_.scaleBytesPerLayerPage());
    }
    uint64_t expected = checkedMultiply(
        extent.pageCount, layout_.bytesPerModelPage());
    if (heapOffset != expected) {
        throw std::logic_error("KV sparse mapping geometry mismatch");
    }
    return mappings;
}

metal::AllocationResult PageStorage::ensureResident(uint32_t page) {
    Extent &extent = extents_.at(extentIndex(page));
    if (extent.heap) return true;
    uint64_t bytes = checkedMultiply(extent.pageCount,
                                     layout_.bytesPerModelPage());
    try {
        return admitAllocation_(bytes, [&] {
            std::optional<metal::SparseHeap> heap;
            heap.emplace(backend_.allocatePlacementHeap(
                bytes, kSparseMappingAlignmentBytes,
                "kv-extent-" + std::to_string(extent.firstPage)));
            if (heap->sizeBytes() != bytes) {
                throw std::logic_error(
                    "placement heap size differs from admitted KV extent bytes");
            }
            auto mappings = mappingsFor(extent);
            backend_.mapSparse(*heap, mappings);
            residentBackingBytes_ += heap->sizeBytes();
            residentPages_ += extent.pageCount;
            extent.heap = std::move(heap);
        });
    } catch (const metal::MetalAllocationError &error) {
        return error.failure();
    }
}

bool PageStorage::releaseBackingForPage(uint32_t page) {
    Extent &extent = extents_.at(extentIndex(page));
    if (!extent.heap) return false;
    const uint64_t heapBytes = extent.heap->sizeBytes();
    if (residentBackingBytes_ < heapBytes || residentPages_ < extent.pageCount) {
        throw std::logic_error("KV resident accounting underflowed");
    }
    auto mappings = mappingsFor(extent);
    // The backend owns the heap from here until the unmap has completed.
    // A validation failure throws before the move, leaving the extent intact.
    backend_.unmapSparse(mappings, std::move(*extent.heap));
    residentBackingBytes_ -= heapBytes;
    residentPages_ -= extent.pageCount;
    extent.heap.reset();
    return true;
}

bool PageStorage::releaseReady() const noexcept {
    return !backend_.sparseUnmapPending();
}

void PageStorage::awaitRelease() {
    backend_.drainSparseUnmaps();
}

const LayerStorage &PageStorage::layer(uint32_t index) const {
    if (index >= layers_.size()) {
        throw std::out_of_range("invalid attention layer index");
    }
    return layers_[index];
}

}  // namespace splash::kv
