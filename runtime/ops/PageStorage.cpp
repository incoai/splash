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
    const auto data = static_cast<uint32_t>(layout_.dataBytesPerLayerPage());
    const auto scale = static_cast<uint32_t>(layout_.scaleBytesPerLayerPage());
    for (uint32_t layer = 0; layer < layout_.attentionLayers; ++layer)
        layers_.push_back(splash_kv_layer(extentPages_, data, scale, layer));
    extents_.resize(pageCount_ / extentPages_);
    extentAddresses_.resize(extents_.size());
}

size_t PageStorage::extentIndex(uint32_t page) const {
    if (page >= pageCount_) throw std::out_of_range("invalid KV page id");
    return page / extentPages_;
}

uint64_t PageStorage::actualAllocatedBytes() const noexcept {
    const auto allocated = std::count_if(
        extents_.begin(), extents_.end(),
        [](const metal::MetalBuffer &extent) { return static_cast<bool>(extent); });
    return static_cast<uint64_t>(allocated) * extentBytes();
}

bool PageStorage::isAllocated(uint32_t page) const {
    return static_cast<bool>(extents_[extentIndex(page)]);
}

metal::AllocationResult PageStorage::allocateExtent(uint32_t extent) {
    metal::MetalBuffer &buffer = extents_.at(extent);
    if (buffer) {
        throw std::logic_error(
            "KV extent " + std::to_string(extent) + " is already allocated");
    }
    const uint64_t bytes = extentBytes();
    return admitAllocation_(bytes, [&] {
        metal::MetalBuffer allocated = backend_.allocateBuffer(
            bytes, metal::BufferStorage::Shared,
            "kv-extent-" + std::to_string(extent * extentPages_));
        if (allocated.gpuAddress() & SPLASH_KV_PAGE_INDEX_MASK) {
            throw std::logic_error(
                "KV extent address leaves no room for the page index");
        }
        extentAddresses_[extent] = allocated.gpuAddress();
        buffer = std::move(allocated);
    });
}

void PageStorage::releaseExtent(uint32_t extent) {
    metal::MetalBuffer &buffer = extents_.at(extent);
    if (!buffer) {
        throw std::logic_error(
            "KV extent " + std::to_string(extent) + " is not allocated");
    }
    if (backend_.commandInFlight()) {
        throw std::logic_error(
            "cannot release a KV extent while a command is in flight");
    }
    buffer = {};
    extentAddresses_[extent] = 0;
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
        const auto source = spans(copy.from);
        const auto destination = spans(copy.to);
        for (size_t tensor = 0; tensor < source.size(); ++tensor) {
            std::memcpy(destination[tensor].data(), source[tensor].data(),
                        source[tensor].size());
        }
    }
}

PageStorage::IndexPlacement PageStorage::indexPlacement(uint32_t layer) const {
    if (!layout_.indexDimension || layer >= layout_.attentionLayers)
        throw std::out_of_range("KV layer has no QSA index region");
    // After every layer's K/V region: per layer the keys of every page,
    // then their pooled keys.
    const uint64_t first = uint64_t{extentPages_} * layout_.attentionLayers * layout_.bytesPerLayerPage();
    const uint64_t keys = first + uint64_t{extentPages_} * layer * layout_.indexBytesPerLayerPage();
    const uint64_t pooled = keys + uint64_t{extentPages_} * layout_.indexKeyBytesPerLayerPage();
    return {static_cast<uint32_t>(keys), static_cast<uint32_t>(pooled)};
}

SplashKvPage PageStorage::entry(uint32_t page) const {
    const size_t extent = extentIndex(page);
    const uint64_t address = extentAddresses_[extent];
    if (!address) {
        throw std::logic_error("KV page " + std::to_string(page) +
                               " is in an extent that is not allocated");
    }
    return splash_kv_page_entry(
        address, page - static_cast<uint32_t>(extent) * extentPages_);
}

void PageStorage::writeEntries(std::span<const uint32_t> pages,
                               uint32_t first,
                               const metal::MetalBuffer &table) const {
    auto *entries = static_cast<SplashKvPage *>(table.contents());
    if (!entries || table.sizeBytes() / sizeof(SplashKvPage) < pages.size()) {
        throw std::logic_error(
            "KV page table is not CPU-visible or too small for its entries");
    }
    if (first > pages.size()) {
        throw std::invalid_argument(
            "KV page table entries start past the end of its pages");
    }
    for (size_t index = first; index < pages.size(); ++index)
        entries[index] = entry(pages[index]);
}

std::vector<std::span<std::byte>> PageStorage::spans(uint32_t page) const {
    auto *extent =
        static_cast<std::byte *>(extents_[extentIndex(page)].contents());
    if (!extent) {
        throw std::logic_error("KV page " + std::to_string(page) +
                               " is in an extent that is not allocated");
    }
    const auto data = static_cast<uint32_t>(layout_.dataBytesPerLayerPage());
    const auto scale = static_cast<uint32_t>(layout_.scaleBytesPerLayerPage());
    const uint32_t index = page % extentPages_;
    std::vector<std::span<std::byte>> result;
    result.reserve(size_t{layout_.attentionLayers} * ((scale ? 4 : 2) + (layout_.indexDimension ? 2 : 0)));
    for (uint32_t layer = 0; layer < layout_.attentionLayers; ++layer) {
        for (uint32_t tensor = SPLASH_KV_KEYS; tensor <= SPLASH_KV_VALUE_SCALES;
             ++tensor) {
            if (const uint32_t bytes = splash_kv_page_bytes(data, scale, tensor)) {
                result.emplace_back(extent + splash_kv_offset(extentPages_, data,
                                                              scale, layer,
                                                              tensor, index),
                                    bytes);
            }
        }
    }
    if (layout_.indexDimension) {
        const uint64_t keys = layout_.indexKeyBytesPerLayerPage();
        const uint64_t pooled = layout_.indexPooledBytesPerLayerPage();
        for (uint32_t layer = 0; layer < layout_.attentionLayers; ++layer) {
            const IndexPlacement placement = indexPlacement(layer);
            result.emplace_back(extent + placement.keysOffset + index * keys, keys);
            result.emplace_back(extent + placement.pooledOffset + index * pooled, pooled);
        }
    }
    return result;
}

}  // namespace splash::kv
