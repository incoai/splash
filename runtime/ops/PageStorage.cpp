#include "ops/PageStorage.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::kv {

PageStorage::PageStorage(metal::MetalBackend &backend,
    metal::AllocationAdmission admitAllocation, Layout layout,
    uint32_t pageCount, uint32_t extentPages)
    : backend_(backend), admitAllocation_(std::move(admitAllocation)),
      layout_(layout), pageCount_(pageCount), extentPages_(extentPages) {
    if (!admitAllocation_) {
        throw std::invalid_argument(
            "KV page storage requires allocation admission");
    }
    if (!layout_.valid()) {
        throw std::invalid_argument("KV page storage layout is invalid");
    }
    // Page entries carry the index in an extent in their low bits, and
    // SplashKvLayer places a layer's region with a 32-bit offset, so an
    // extent stays below 4 GiB.
    if (!extentPages_ || extentPages_ % layout_.extentAlignmentPages() ||
        extentPages_ > SPLASH_KV_PAGE_INDEX_MASK ||
        extentBytes() > std::numeric_limits<uint32_t>::max() ||
        extentBytes() > backend_.capabilities().maxBufferLengthBytes) {
        throw std::invalid_argument("KV extent geometry is invalid");
    }
    if (!pageCount_ || pageCount_ % extentPages_) {
        throw std::invalid_argument(
            "KV page pool is not a whole number of extents");
    }
    // splash_kv_offset places an extent's regions one after another; a
    // layer's is the one Layout::region orders it at.
    const auto data = static_cast<uint32_t>(layout_.dataBytesPerLayerPage());
    const auto scale = static_cast<uint32_t>(layout_.scaleBytesPerLayerPage());
    regionLayers_.resize(layout_.attentionLayers);
    for (uint32_t layer = 0; layer < layout_.attentionLayers; ++layer) {
        layers_.push_back(splash_kv_layer(extentPages_, data, scale, layout_.region(layer)));
        regionLayers_[layout_.region(layer)] = layer;
    }
    extents_.resize(pageCount_ / extentPages_);
}

size_t PageStorage::extentIndex(uint32_t page) const {
    if (page >= pageCount_) throw std::out_of_range("invalid KV page id");
    return page / extentPages_;
}

const PageStorage::Extent &PageStorage::allocatedExtent(uint32_t page) const {
    const Extent &extent = extents_[extentIndex(page)];
    if (!extent.address) {
        throw std::logic_error("KV page " + std::to_string(page) +
                               " is in an extent that is not allocated");
    }
    return extent;
}

uint64_t PageStorage::actualAllocatedBytes() const noexcept {
    uint64_t bytes = 0;
    for (const Extent &extent : extents_)
        bytes += extent.buffer.sizeBytes();
    return bytes;
}

bool PageStorage::isAllocated(uint32_t page) const {
    return static_cast<bool>(extents_[extentIndex(page)].buffer);
}

metal::AllocationResult PageStorage::allocateExtent(uint32_t extent, PageForm form) {
    Extent &record = extents_.at(extent);
    if (record.buffer) {
        throw std::logic_error(
            "KV extent " + std::to_string(extent) + " is already allocated");
    }
    if (form == PageForm::Index && !layout_.spillable()) {
        throw std::invalid_argument(
            "KV pages of a layout that cannot spill are all Full");
    }
    if (form == PageForm::Slim && !layout_.slimmable()) {
        throw std::invalid_argument(
            "KV pages of a layout without a window keep every layer");
    }
    const uint64_t bytes = extentBytes(form);
    const std::string label = form == PageForm::Full   ? "kv-extent-"
                              : form == PageForm::Slim ? "kv-slim-extent-"
                                                       : "kv-index-extent-";
    return admitAllocation_(bytes, [&] {
        metal::MetalBuffer allocated = backend_.allocateBuffer(
            bytes, metal::BufferStorage::Shared,
            label + std::to_string(extent * extentPages_), metal::BufferReach::Addressed);
        if (allocated.gpuAddress() & SPLASH_KV_PAGE_INDEX_MASK) {
            throw std::logic_error(
                "KV extent address leaves no room for the page index");
        }
        const uint64_t address = allocated.gpuAddress();
        record = {std::move(allocated), address, form};
    });
}

void PageStorage::releaseExtent(uint32_t extent) {
    Extent &record = extents_.at(extent);
    if (!record.buffer) {
        throw std::logic_error(
            "KV extent " + std::to_string(extent) + " is not allocated");
    }
    if (backend_.commandInFlight()) {
        throw std::logic_error(
            "cannot release a KV extent while a command is in flight");
    }
    record = {};
}

void PageStorage::copyPages(std::span<const PageCopy> copies) {
    if (backend_.commandInFlight()) {
        throw std::logic_error(
            "cannot copy KV pages while a command is in flight");
    }
    for (const PageCopy &copy : copies) {
        if (!isAllocated(copy.from) || !isAllocated(copy.to)) {
            throw std::logic_error(
                "cannot copy a KV page of an extent that is not allocated");
        }
    }
    for (const PageCopy &copy : copies) {
        const uint32_t regions = std::min(layout_.regions(form(copy.from)),
                                          layout_.regions(form(copy.to)));
        const auto source = spans(copy.from, regions);
        const auto destination = spans(copy.to, regions);
        for (size_t tensor = 0; tensor < source.size(); ++tensor) {
            std::memcpy(destination[tensor].data(), source[tensor].data(),
                        source[tensor].size());
        }
    }
}

void PageStorage::clearPages(std::span<const uint32_t> pages) {
    for (const uint32_t page : pages) {
        if (!isAllocated(page)) {
            throw std::logic_error(
                "cannot clear a KV page of an extent that is not allocated");
        }
    }
    for (const uint32_t page : pages) {
        for (const std::span<std::byte> tensor : spans(page))
            std::memset(tensor.data(), 0, tensor.size());
    }
}

uint32_t PageStorage::indexOffset(uint32_t layer) const {
    if (!layout_.indexDimension || layer >= layout_.attentionLayers)
        throw std::out_of_range("KV layer has no index region");
    return static_cast<uint32_t>(uint64_t{extentPages_} * layer * layout_.indexBytesPerLayerPage());
}

PageForm PageStorage::form(uint32_t page) const { return allocatedExtent(page).form; }

template <class Entry>
void PageStorage::writeTable(std::span<const uint32_t> pages, uint32_t first,
                             const metal::MetalBuffer &table, Entry entry) const {
    auto *entries = static_cast<SplashKvPage *>(table.contents());
    if (!entries || table.sizeBytes() / sizeof(SplashKvPage) < pages.size()) {
        throw std::logic_error(
            "KV page table is not CPU-visible or too small for its entries");
    }
    if (first > pages.size()) {
        throw std::invalid_argument(
            "KV page table entries start past the end of its pages");
    }
    for (size_t index = first; index < pages.size(); ++index) {
        const uint32_t page = pages[index];
        entries[index] = entry(allocatedExtent(page), page % extentPages_);
    }
}

void PageStorage::writeEntries(std::span<const uint32_t> pages,
                               uint32_t first,
                               const metal::MetalBuffer &table) const {
    writeTable(pages, first, table, [](const Extent &extent, uint32_t index) {
        return extent.form == PageForm::Index ? SPLASH_KV_PAGE_ABSENT
                                              : splash_kv_page_entry(extent.address, index);
    });
}

void PageStorage::writeIndexEntries(std::span<const uint32_t> pages,
                                    uint32_t first,
                                    const metal::MetalBuffer &table) const {
    if (!layout_.indexDimension)
        throw std::logic_error("KV pages of a layout without an index have no index entries");
    writeTable(pages, first, table, [this](const Extent &extent, uint32_t index) {
        return splash_kv_page_entry(extent.address + indexPlaneOffset(extent.form), index);
    });
}

std::vector<std::span<std::byte>> PageStorage::spans(uint32_t page) const {
    return spans(page, layout_.regions(form(page)));
}

std::vector<std::span<std::byte>> PageStorage::spans(uint32_t page, uint32_t regions) const {
    std::vector<std::span<std::byte>> result;
    result.reserve(size_t{regions} * (layout_.scaleBytesPerLayerPage() ? 4 : 2) +
                   (layout_.indexDimension ? layout_.attentionLayers : 0));
    for (uint32_t region = 0; region < regions; ++region) {
        for (const std::span<std::byte> tensor : layerSpans(page, regionLayers_[region])) {
            if (!tensor.empty()) result.push_back(tensor);
        }
    }
    const std::vector<std::span<std::byte>> index = indexSpans(page);
    result.insert(result.end(), index.begin(), index.end());
    return result;
}

std::array<std::span<std::byte>, 4> PageStorage::layerSpans(uint32_t page, uint32_t layer) const {
    const Extent &extent = allocatedExtent(page);
    if (extent.form == PageForm::Index)
        throw std::logic_error("KV page " + std::to_string(page) + " holds only its index");
    if (layer >= layout_.attentionLayers)
        throw std::out_of_range("KV layer is outside the layout");
    if (layout_.region(layer) >= layout_.regions(extent.form))
        throw std::logic_error("KV page " + std::to_string(page) + " holds no windowed layer's K/V");
    auto *contents = static_cast<std::byte *>(extent.buffer.contents());
    const auto data = static_cast<uint32_t>(layout_.dataBytesPerLayerPage());
    const auto scale = static_cast<uint32_t>(layout_.scaleBytesPerLayerPage());
    const uint32_t index = page % extentPages_;
    std::array<std::span<std::byte>, 4> result;
    for (uint32_t tensor = SPLASH_KV_KEYS; tensor <= SPLASH_KV_VALUE_SCALES; ++tensor) {
        result[tensor] = {
            contents + splash_kv_offset(extentPages_, data, scale, layout_.region(layer), tensor, index),
            splash_kv_page_bytes(data, scale, tensor)};
    }
    return result;
}

std::vector<std::span<std::byte>> PageStorage::indexSpans(uint32_t page) const {
    const Extent &extent = allocatedExtent(page);
    std::vector<std::span<std::byte>> result;
    if (!layout_.indexDimension) return result;
    std::byte *plane = static_cast<std::byte *>(extent.buffer.contents()) + indexPlaneOffset(extent.form);
    const uint64_t bytes = layout_.indexBytesPerLayerPage();
    const uint32_t index = page % extentPages_;
    result.reserve(layout_.attentionLayers);
    for (uint32_t layer = 0; layer < layout_.attentionLayers; ++layer)
        result.emplace_back(plane + indexOffset(layer) + index * bytes, bytes);
    return result;
}

}  // namespace splash::kv
