#include "TestBuffers.hpp"
#include "ops/PageStorage.hpp"
#include "engine/MemoryGovernor.hpp"
#include "tests/engine/TestChecks.hpp"
#include "tests/engine/TestPageEntries.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::engine;
using splash::test::entryOf;

namespace {

using splash::test::require;

template <typename Exception, typename Function>
void requireThrows(Function &&function, const char *message) {
    try {
        function();
    } catch (const Exception &) {
        return;
    }
    throw std::runtime_error(message);
}

constexpr kv::Layout kvLayout{16, 4, 256};
constexpr kv::Layout compactLayout{10, 2, 256};
constexpr kv::Layout bf16Layout{16, 4, 256, kv::Format::BFloat16};
constexpr kv::Layout compactBf16Layout{10, 2, 256, kv::Format::BFloat16};

// The host reaches a page through spans of its tensors, layer by layer as
// keys, key scales, values and value scales, BF16 without scales. Kernels
// find a page's keys at the page's slab in the layer's region, so that is
// where each layer's spans start, and the spans of an extent's pages cover
// the extent once, without gaps or overlaps.
void requireSpansTileExtent(const kv::PageStorage &storage, uint32_t firstPage) {
    const kv::Layout layout = storage.layout();
    const bool scaled = layout.format == kv::Format::Int8;
    const uint32_t tensors = scaled ? 4 : 2;
    std::byte *const extent = storage.spans(firstPage).front().data();
    std::vector<std::span<std::byte>> all;
    for (uint32_t index = 0; index < storage.extentPages(); ++index) {
        const auto spans = storage.spans(firstPage + index);
        require(spans.size() == layout.attentionLayers * tensors,
                "a page does not have one span per tensor of every layer");
        for (uint32_t span = 0; span < spans.size(); ++span) {
            const uint64_t bytes = scaled && span % 2 ? layout.scaleBytesPerLayerPage()
                                                      : layout.dataBytesPerLayerPage();
            require(spans[span].size() == bytes, "a page's span holds another tensor's bytes");
        }
        for (uint32_t layer = 0; layer < layout.attentionLayers; ++layer) {
            require(spans[layer * tensors].data() ==
                        extent + storage.layers()[layer].offset +
                            uint64_t{index} * layout.dataBytesPerLayerPage(),
                    "a layer's spans do not start at the page's keys in its region");
        }
        all.insert(all.end(), spans.begin(), spans.end());
    }
    std::sort(all.begin(), all.end(),
              [](auto left, auto right) { return left.data() < right.data(); });
    std::byte *next = extent;
    for (const auto span : all) {
        require(span.data() == next, "the spans of an extent's pages leave a gap or overlap");
        next += span.size();
    }
    require(next == extent + storage.extentBytes(), "the spans of an extent's pages do not cover it");
}

// Writing a table from an index on leaves the entries before it as they
// are, and refuses an index past the pages.
void entriesFromFirst(const kv::PageStorage &storage, const metal::MetalBuffer &table) {
    constexpr std::array<uint32_t, 3> pages{5, 200, 255};
    auto *entries = static_cast<SplashKvPage *>(table.contents());
    storage.writeEntries(pages, 0, table);
    const std::array<SplashKvPage, 3> written{entries[0], entries[1], entries[2]};
    constexpr SplashKvPage kSentinel = ~SplashKvPage{0};
    std::fill_n(entries, 3, kSentinel);
    storage.writeEntries(pages, 1, table);
    require(entries[0] == kSentinel && entries[1] == written[1] && entries[2] == written[2],
            "a table written from an index did not keep the entries before it");
    storage.writeEntries(pages, 3, table);
    require(entries[0] == kSentinel, "a table written past its last page changed");
    requireThrows<std::invalid_argument>(
        [&] { storage.writeEntries(std::array<uint32_t, 1>{5}, 2, table); },
        "a table was written from an index past its pages");
}

// An extent released and allocated again gets entries of its new buffer: a
// kernel writing through the table reaches the memory the host reads.
void entriesFollowAReallocatedExtent(metal::MetalBackend &backend, kv::PageStorage &storage,
                                     const metal::MetalBuffer &table) {
    storage.releaseExtent(1);
    require(static_cast<bool>(storage.allocateExtent(1)),
            "a released extent could not be allocated again");
    storage.writeEntries(std::array<uint32_t, 1>{128}, 0, table);
    const uint32_t words = 1024, seed = 0x5eed;
    (void)backend.submit({"addressed_write_u32", {{0, table}},
                          {{1, &words, sizeof(words)}, {2, &seed, sizeof(seed)}},
                          {words / 256, 1, 1}, {256, 1, 1}});
    const auto *written =
        reinterpret_cast<const uint32_t *>(storage.spans(128).front().data());
    for (uint32_t word = 0; word < words; ++word) {
        if (written[word] != (seed ^ word))
            throw std::runtime_error("a kernel did not reach a reallocated extent through its entry");
    }
}

void run(const std::string &metallib) {
    metal::MetalBackend backend(metallib);
    std::optional<uint64_t> elasticHostAvailable = 2ULL * 1024 * 1024 * 1024;
    MemoryGovernor hostGated(
        backend, backend.capabilities().recommendedMaxWorkingSetBytes,
        128ULL * 1024 * 1024,
        [&elasticHostAvailable] { return elasticHostAvailable; }, 0);
    kv::PageStorage hostGatedStorage(
        backend, hostGated.allocationAdmission(), kvLayout, 256, 128);
    require(hostGatedStorage.allocateExtent(0) &&
                hostGatedStorage.actualAllocatedBytes() ==
                    uint64_t{128} * hostGatedStorage.bytesPerPage(),
            "elastic Q8 storage did not allocate its first extent");
    elasticHostAvailable = 128ULL * 1024 * 1024;
    require(!hostGatedStorage.allocateExtent(1) &&
                hostGatedStorage.actualAllocatedBytes() ==
                    uint64_t{128} * hostGatedStorage.bytesPerPage(),
            "host pressure did not reject the next KV extent transactionally");
    elasticHostAvailable = 4ULL * 1024 * 1024 * 1024;
    require(hostGatedStorage.allocateExtent(1) &&
                hostGatedStorage.actualAllocatedBytes() ==
                    uint64_t{256} * hostGatedStorage.bytesPerPage(),
            "KV growth did not recover after host memory became available");

    MemoryGovernor governor(
        backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, queryHostAvailableMemory, 0);
    requireThrows<std::invalid_argument>(
        [&] { kv::PageStorage(backend, governor.allocationAdmission(), kvLayout, 192, 128); },
        "a pool of a part of an extent was accepted");
    requireThrows<std::invalid_argument>(
        [&] { kv::PageStorage(backend, governor.allocationAdmission(), kvLayout, 256, 64); },
        "an extent of a part of an alignment unit was accepted");

    metal::MetalBuffer table = test::sharedBuffer(backend, 4 * sizeof(SplashKvPage));
    const metal::MetalBuffer probe = test::sharedBuffer(backend, sizeof(SplashKvPage));
    metal::MetalBuffer word = test::sharedBuffer(backend, sizeof(uint32_t));
    const uint64_t before = backend.memoryStats().allocatedBytes;
    kv::PageStorage storage(backend, governor.allocationAdmission(), kvLayout, 384, 128);
    const uint64_t extentBytes = 128 * kvLayout.bytesPerModelPage();
    require(uint64_t{storage.pageCount()} * storage.bytesPerPage() == 3 * extentBytes && storage.extentBytes() == extentBytes &&
                storage.actualAllocatedBytes() == 0 &&
                backend.memoryStats().allocatedBytes == before,
            "KV page storage allocated an extent when it was built");
    require(storage.allocateExtent(0) && storage.actualAllocatedBytes() == extentBytes &&
                backend.memoryStats().allocatedBytes == before + extentBytes,
            "an extent was not allocated at exactly its size");
    require(storage.isAllocated(127) && !storage.isAllocated(128),
            "the first Q8 extent was not the one allocated");
    requireThrows<std::logic_error>([&] { (void)storage.allocateExtent(0); },
                                    "an allocated extent was allocated again");
    const SplashKvLayer layer = storage.layers()[15];
    require(storage.layers().size() == kvLayout.attentionLayers && layer.extent_pages == 128 &&
                layer.offset == 15 * 128 * kvLayout.bytesPerLayerPage(),
            "a layer's region does not follow the layers before it");

    const SplashKvPage runwayPage = entryOf(storage, 5, probe);
    require(runwayPage && (runwayPage & SPLASH_KV_PAGE_INDEX_MASK) == 5,
            "a page entry does not carry the page's index in its extent");
    requireThrows<std::logic_error>([&] { (void)storage.spans(200); },
                                    "a page of an unallocated extent received host memory");
    requireSpansTileExtent(storage, 0);
    requireThrows<std::logic_error>(
        [&] { storage.writeEntries(std::array<uint32_t, 1>{200}, 0, table); },
        "a table was written with a page of an unallocated extent");
    requireThrows<std::logic_error>(
        [&] { storage.writeEntries(std::array<uint32_t, 5>{0, 1, 2, 3, 4}, 0, table); },
        "a table too small for its entries was written");

    require(storage.allocateExtent(1) && storage.actualAllocatedBytes() == 2 * extentBytes &&
                backend.memoryStats().allocatedBytes == before + 2 * extentBytes,
            "growth did not add exactly one extent");
    storage.writeEntries(std::array<uint32_t, 4>{200, 5, 255, 128}, 0, table);
    const auto *entries = static_cast<const SplashKvPage *>(table.contents());
    // Pages 200, 255 and 128 share the second extent, at indices 72, 127, 0.
    require(entries[0] == entryOf(storage, 200, probe) &&
                (entries[0] & SPLASH_KV_PAGE_INDEX_MASK) == 72 &&
                entries[1] == runwayPage && entries[2] == entries[0] + 55 &&
                entries[3] == entries[0] - 72 && (runwayPage & ~uint64_t{SPLASH_KV_PAGE_INDEX_MASK}) !=
                                                     (entries[3] & ~uint64_t{SPLASH_KV_PAGE_INDEX_MASK}),
            "a page table does not hold the entries of its pages");

    // A command reaches extents through its tables without retaining them:
    // none is released while one is in flight.
    {
        const metal::ComputeDispatch kick{"residency_kick", {{0, word}}, {}, {1, 1, 1}, {1, 1, 1}};
        auto ticket = backend.submitAsync(kick);
        requireThrows<std::logic_error>(
            [&] { storage.releaseExtent(1); },
            "an extent was released while a command was in flight");
        require(storage.isAllocated(200) && entryOf(storage, 200, probe) == entries[0],
                "a refused release changed the extent");
        (void)ticket.wait();
    }
    storage.releaseExtent(1);
    require(!storage.isAllocated(200) && storage.actualAllocatedBytes() == extentBytes &&
                backend.memoryStats().allocatedBytes == before + extentBytes,
            "a released extent did not return its memory at once");
    requireThrows<std::logic_error>([&] { storage.releaseExtent(1); },
                                    "an unallocated extent was released again");
    require(storage.allocateExtent(1) &&
                (entryOf(storage, 255, probe) & SPLASH_KV_PAGE_INDEX_MASK) == 127,
            "a released extent could not be allocated again");
    entriesFromFirst(storage, table);
    entriesFollowAReallocatedExtent(backend, storage, table);

    // The pool moves a page by copying it: every tensor of every layer goes
    // to the other page, in another extent too, and the pages beside both
    // stay as they were. A command may still write the source, so nothing is
    // copied while one is in flight, nor when a page has no memory.
    const auto fill = [&](uint32_t page, uint8_t first) {
        uint8_t value = first;
        for (const auto span : storage.spans(page))
            std::memset(span.data(), value++, span.size());
    };
    const auto holds = [&](uint32_t page, uint8_t first) {
        uint8_t value = first;
        for (const auto span : storage.spans(page)) {
            const auto expected = static_cast<std::byte>(value++);
            if (!std::all_of(span.begin(), span.end(),
                             [&](std::byte byte) { return byte == expected; }))
                return false;
        }
        return true;
    };
    fill(5, 10);
    fill(6, 60);
    fill(200, 110);
    fill(201, 160);
    fill(255, 210);
    {
        const metal::ComputeDispatch kick{"residency_kick", {{0, word}}, {}, {1, 1, 1}, {1, 1, 1}};
        auto ticket = backend.submitAsync(kick);
        requireThrows<std::logic_error>(
            [&] { storage.copyPages(std::array<kv::PageCopy, 1>{{{5, 200}}}); },
            "a page was copied while a command was in flight");
        (void)ticket.wait();
    }
    requireThrows<std::logic_error>(
        [&] { storage.copyPages(std::array<kv::PageCopy, 2>{{{5, 200}, {6, 300}}}); },
        "a page was copied to an extent that is not allocated");
    require(holds(200, 110), "a refused copy changed a page");
    storage.copyPages(std::array<kv::PageCopy, 2>{{{5, 200}, {6, 255}}});
    require(holds(200, 10) && holds(255, 60) && holds(5, 10) && holds(6, 60) && holds(201, 160),
            "a copied page does not hold its source's tensors, or its neighbour changed");

    kv::PageStorage compactStorage(
        backend, governor.allocationAdmission(), compactLayout, 1024, 512);
    require(compactStorage.allocateExtent(0) && compactStorage.layers()[9].extent_pages == 512 &&
                compactStorage.layers()[9].offset ==
                    9 * 512 * compactLayout.bytesPerLayerPage() &&
                compactStorage.actualAllocatedBytes() ==
                    512 * compactLayout.bytesPerModelPage(),
            "model-provided compact Q8 geometry was not honored");

    for (const auto layout : {bf16Layout, compactBf16Layout}) {
        const uint32_t extent = layout.minimumExtentPages();
        kv::PageStorage bf16(backend, governor.allocationAdmission(), layout,
                             2 * extent, extent);
        const SplashKvLayer bf16Layer = bf16.layers()[layout.attentionLayers - 1];
        require(bf16Layer.offset == (layout.attentionLayers - 1) * extent * 2 *
                                               layout.dataBytesPerLayerPage(),
                "BF16 regions hold quantization scales or misplace a layer");
        require(bf16.allocateExtent(0) && !bf16.isAllocated(extent) &&
                    bf16.actualAllocatedBytes() == extent * layout.bytesPerModelPage(),
                "BF16 allocated more than its first admitted extent");
        requireSpansTileExtent(bf16, 0);
        require(bf16.allocateExtent(1) &&
                    bf16.actualAllocatedBytes() == uint64_t{2 * extent} * bf16.bytesPerPage(),
                "BF16 growth did not account for both extents");
        bf16.releaseExtent(1);
        require(!bf16.isAllocated(extent) && bf16.allocateExtent(1),
                "BF16 extent could not be allocated again after release");
    }
    std::cout << "KV page storage tests passed\n";
}

}  // namespace

int main(int argc, const char **argv) {
    if (argc != 2) {
        std::cerr << "usage: q8_page_storage_test METALLIB\n";
        return EXIT_FAILURE;
    }
    try {
        run(argv[1]);
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "q8 page storage test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
