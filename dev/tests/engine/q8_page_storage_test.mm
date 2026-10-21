#include "TestBuffers.hpp"
#include "ops/PageStorage.hpp"
#include "engine/MemoryGovernor.hpp"
#include "tests/engine/HostKvExtents.hpp"
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

using splash::test::rejects;
using splash::test::require;

constexpr kv::Layout kvLayout{16, 4, 256};
constexpr kv::Layout compactLayout{10, 2, 256};
constexpr kv::Layout bf16Layout{16, 4, 256, kv::Format::BFloat16};
constexpr kv::Layout compactBf16Layout{10, 2, 256, kv::Format::BFloat16};
// Qwen3.8-Flash-Next's: token-major values and a QSA index of 128-wide pooled
// keys.
constexpr kv::Layout indexedLayout{12, 2, 256, kv::Format::Int8, 128, true};
constexpr kv::Layout indexedBf16Layout{12, 2, 256, kv::Format::BFloat16, 128, true};

// The host reaches a Full page through spans of its tensors, layer by layer
// as keys, key scales, values and value scales, BF16 without scales, then
// each layer's QSA pooled keys, and an Index page through its pooled keys
// alone. Kernels find a page's keys at the page's slab in the layer's region
// and its pooled keys at its slab of the layer's index region in the
// extent's index plane, after every layer's region where there are K/V, so
// that is where those spans start, and the spans of an extent's pages cover
// the extent once, without gaps or overlaps.
void requireSpansTileExtent(const kv::PageStorage &storage, uint32_t firstPage) {
    const kv::Layout layout = storage.layout();
    const kv::PageForm form = storage.form(firstPage);
    const bool scaled = layout.format == kv::Format::Int8;
    const uint32_t tensors = scaled ? 4 : 2;
    const uint32_t kvSpans = form == kv::PageForm::Full ? layout.attentionLayers * tensors : 0;
    std::byte *const extent = storage.spans(firstPage).front().data();
    std::byte *const plane = extent + (kvSpans ? uint64_t{storage.extentPages()} * layout.kvBytesPerModelPage() : 0);
    std::vector<std::span<std::byte>> all;
    for (uint32_t index = 0; index < storage.extentPages(); ++index) {
        const auto spans = storage.spans(firstPage + index);
        require(spans.size() == kvSpans + (layout.indexDimension ? layout.attentionLayers : 0),
                "a page does not have one span per tensor of every layer");
        for (uint32_t span = 0; span < spans.size(); ++span) {
            const uint64_t bytes = span >= kvSpans                ? layout.indexBytesPerLayerPage()
                                   : scaled && span % 2 ? layout.scaleBytesPerLayerPage()
                                                        : layout.dataBytesPerLayerPage();
            require(spans[span].size() == bytes, "a page's span holds another tensor's bytes");
        }
        for (uint32_t layer = 0; layer < layout.attentionLayers; ++layer) {
            if (kvSpans)
                require(spans[layer * tensors].data() ==
                            extent + storage.layers()[layer].offset +
                                uint64_t{index} * layout.dataBytesPerLayerPage(),
                        "a layer's spans do not start at the page's keys in its region");
            if (layout.indexDimension)
                require(spans[kvSpans + layer].data() == plane + storage.indexOffset(layer) +
                                                             uint64_t{index} * layout.indexBytesPerLayerPage(),
                        "a layer's index span is not the page's pooled keys in its index region");
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
    require(next == extent + storage.extentBytes(form), "the spans of an extent's pages do not cover it");
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
    rejects([&] { storage.writeEntries(std::array<uint32_t, 1>{5}, 2, table); },
            "KV page table entries start past the end of its pages",
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

void requireSameSpans(std::span<const std::span<std::byte>> left, std::span<const std::span<std::byte>> right,
                      const char *message) {
    require(std::ranges::equal(left, right,
                               [](auto a, auto b) { return a.data() == b.data() && a.size() == b.size(); }),
            message);
}

// A layout that can spill also has extents of Index pages, which hold only
// the index plane. A page table reaches no K/V through an Index page
// (SPLASH_KV_PAGE_ABSENT); an index table reaches the index plane of a page
// of either form, as a kernel writing through it shows, and pages of the
// two forms copy their index to each other. HostKvExtents places both forms
// where the storage does.
void indexPagesHoldOnlyTheirIndex(metal::MetalBackend &backend, MemoryGovernor &governor) {
    for (const auto layout : {indexedLayout, indexedBf16Layout}) {
        const uint32_t extent = layout.minimumExtentPages();
        kv::PageStorage storage(backend, governor.allocationAdmission(), layout, 3 * extent, extent);
        require(storage.allocateExtent(0) && storage.allocateExtent(1, kv::PageForm::Index) &&
                    storage.extentBytes(kv::PageForm::Index) == uint64_t{extent} * 12 * 2048 &&
                    storage.actualAllocatedBytes() ==
                        storage.extentBytes() + storage.extentBytes(kv::PageForm::Index) &&
                    storage.form(extent - 1) == kv::PageForm::Full && storage.form(extent) == kv::PageForm::Index,
                "an extent of Index pages was not allocated at the bytes of their index");
        requireSpansTileExtent(storage, extent);
        rejects([&] { (void)storage.layerSpans(extent + 1, 0); },
                "KV page " + std::to_string(extent + 1) + " holds only its index",
                "an Index page had K/V spans");

        metal::MetalBuffer table = test::sharedBuffer(backend, 2 * sizeof(SplashKvPage));
        auto *entries = static_cast<SplashKvPage *>(table.contents());
        const std::array<uint32_t, 2> firsts{0, extent};
        storage.writeEntries(firsts, 0, table);
        const SplashKvPage kvEntry = entries[0];
        require(kvEntry && entries[1] == SPLASH_KV_PAGE_ABSENT, "an Index page's K/V entry is not absent");
        storage.writeIndexEntries(firsts, 0, table);
        require(entries[0] == kvEntry + uint64_t{extent} * layout.kvBytesPerModelPage() && entries[1] &&
                    !(entries[1] & SPLASH_KV_PAGE_INDEX_MASK),
                "an index entry does not reach its extent's index plane");
        const test::HostKvExtents host(
            layout, extent,
            {{storage.spans(0).front().data(), kvEntry, kv::PageForm::Full},
             {storage.indexSpans(extent).front().data(), entries[1], kv::PageForm::Index}});
        const uint32_t words = 512, seed = 0x1de;
        (void)backend.submit({"addressed_write_u32", {{0, table}},
                              {{1, &words, sizeof(words)}, {2, &seed, sizeof(seed)}},
                              {words / 256, 2, 1}, {256, 1, 1}});
        for (uint32_t row = 0; row < firsts.size(); ++row) {
            const auto *written = reinterpret_cast<const uint32_t *>(storage.indexSpans(firsts[row]).front().data());
            for (uint32_t word = 0; word < words; ++word)
                require(written[word] == (seed ^ (row * 131071u + word)),
                        "a kernel did not reach a page's pooled keys through its index entry");
        }

        // The index goes between the forms both ways; a Full page keeps its K/V.
        const auto fill = [&](std::span<const std::span<std::byte>> spans, std::byte value) {
            for (const auto span : spans) std::fill(span.begin(), span.end(), value);
        };
        const auto holds = [&](std::span<const std::span<std::byte>> spans, std::byte value) {
            return std::ranges::all_of(spans, [&](auto span) {
                return std::ranges::all_of(span, [&](std::byte byte) { return byte == value; });
            });
        };
        fill(storage.spans(2), std::byte{0x11});
        fill(storage.spans(3), std::byte{0x33});
        fill(storage.spans(extent + 3), std::byte{0x44});
        storage.copyPages(std::array{kv::PageCopy{2, extent + 2}});
        storage.copyPages(std::array{kv::PageCopy{extent + 2, 3}});
        require(holds(storage.indexSpans(extent + 2), std::byte{0x11}) &&
                    holds(storage.indexSpans(3), std::byte{0x11}) &&
                    holds(storage.layerSpans(3, layout.attentionLayers - 1), std::byte{0x33}),
                "a page's index did not move between the forms, or a Full page lost its K/V");
        storage.clearPages(std::array<uint32_t, 1>{extent + 2});
        require(holds(storage.spans(extent + 2), std::byte{0}) && holds(storage.spans(extent + 3), std::byte{0x44}),
                "a cleared Index page kept its index, or its neighbour changed");

        // HostKvExtents over the same memory: its spans and entries are the storage's.
        const auto tables = test::sharedBuffer(backend, 4 * sizeof(SplashKvPage));
        std::array<SplashKvPage, 4> hostEntries{};
        const std::array<uint32_t, 4> pages{1, extent - 1, extent + 1, 2 * extent - 1};
        for (const uint32_t page : pages) {
            requireSameSpans(host.indexSpans(page), storage.indexSpans(page),
                             "HostKvExtents misplaces a page's pooled keys");
            if (storage.form(page) == kv::PageForm::Full)
                requireSameSpans(host.layerSpans(page, 1), storage.layerSpans(page, 1),
                                 "HostKvExtents misplaces a page's K/V");
        }
        const auto *written = static_cast<const SplashKvPage *>(tables.contents());
        host.writeTable(pages, hostEntries.data());
        storage.writeEntries(pages, 0, tables);
        require(std::ranges::equal(hostEntries, std::span(written, pages.size())),
                "HostKvExtents writes other page table entries");
        host.writeIndexTable(pages, hostEntries.data());
        storage.writeIndexEntries(pages, 0, tables);
        require(std::ranges::equal(hostEntries, std::span(written, pages.size())) &&
                    host.indexOffset(11) == storage.indexOffset(11),
                "HostKvExtents writes other index table entries");

        storage.releaseExtent(1);
        require(storage.actualAllocatedBytes() == storage.extentBytes() && storage.allocateExtent(1),
                "a released Index extent did not return its memory");
    }
    kv::PageStorage dense(backend, governor.allocationAdmission(), kvLayout, 256, 128);
    rejects([&] { (void)dense.allocateExtent(0, kv::PageForm::Index); },
            "KV pages of a layout that cannot spill are all Full",
            "a layout that cannot spill allocated Index pages");
    require(static_cast<bool>(dense.allocateExtent(0)), "a dense extent was not allocated");
    const metal::MetalBuffer table = test::sharedBuffer(backend, sizeof(SplashKvPage));
    rejects([&] { dense.writeIndexEntries(std::array<uint32_t, 1>{0}, 0, table); },
            "KV pages of a layout without an index have no index entries",
            "a layout without an index wrote index entries");
}

void run(const std::string &metallib) {
    metal::MetalBackend backend(metallib);
    std::optional<uint64_t> elasticHostAvailable = 2ULL * 1024 * 1024 * 1024;
    MemoryGovernor hostGated(
        backend, backend.capabilities().recommendedMaxWorkingSetBytes,
        128ULL * 1024 * 1024, 128ULL * 1024 * 1024, [&elasticHostAvailable] {
          return std::optional<HostMemory>(HostMemory{*elasticHostAvailable, *elasticHostAvailable});
        }, 0);
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
        backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1, queryHostMemory, 0);
    rejects([&] { kv::PageStorage(backend, governor.allocationAdmission(), kvLayout, 192, 128); },
            "KV page pool is not a whole number of extents",
            "a pool of a part of an extent was accepted");
    rejects([&] { kv::PageStorage(backend, governor.allocationAdmission(), kvLayout, 256, 64); },
            "KV extent geometry is invalid",
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
    rejects([&] { (void)storage.allocateExtent(0); }, "KV extent 0 is already allocated",
            "an allocated extent was allocated again");
    const SplashKvLayer layer = storage.layers()[15];
    require(storage.layers().size() == kvLayout.attentionLayers && layer.extent_pages == 128 &&
                layer.offset == 15 * 128 * kvLayout.bytesPerLayerPage(),
            "a layer's region does not follow the layers before it");

    const SplashKvPage runwayPage = entryOf(storage, 5, probe);
    require(runwayPage && (runwayPage & SPLASH_KV_PAGE_INDEX_MASK) == 5,
            "a page entry does not carry the page's index in its extent");
    rejects([&] { (void)storage.spans(200); }, "KV page 200 is in an extent that is not allocated",
            "a page of an unallocated extent received host memory");
    requireSpansTileExtent(storage, 0);
    rejects([&] { storage.writeEntries(std::array<uint32_t, 1>{200}, 0, table); },
            "KV page 200 is in an extent that is not allocated",
            "a table was written with a page of an unallocated extent");
    rejects([&] { storage.writeEntries(std::array<uint32_t, 5>{0, 1, 2, 3, 4}, 0, table); },
            "KV page table is not CPU-visible or too small for its entries",
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
        auto ticket = backend.submitCommandAsync({&kick, 1});
        rejects([&] { storage.releaseExtent(1); },
                "cannot release a KV extent while a command is in flight",
                "an extent was released while a command was in flight");
        require(storage.isAllocated(200) && entryOf(storage, 200, probe) == entries[0],
                "a refused release changed the extent");
        (void)ticket.wait();
    }
    storage.releaseExtent(1);
    require(!storage.isAllocated(200) && storage.actualAllocatedBytes() == extentBytes &&
                backend.memoryStats().allocatedBytes == before + extentBytes,
            "a released extent did not return its memory at once");
    rejects([&] { storage.releaseExtent(1); }, "KV extent 1 is not allocated",
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
        auto ticket = backend.submitCommandAsync({&kick, 1});
        rejects([&] { storage.copyPages(std::array<kv::PageCopy, 1>{{{5, 200}}}); },
                "cannot copy KV pages while a command is in flight",
                "a page was copied while a command was in flight");
        (void)ticket.wait();
    }
    rejects([&] { storage.copyPages(std::array<kv::PageCopy, 2>{{{5, 200}, {6, 300}}}); },
            "cannot copy a KV page of an extent that is not allocated",
            "a page was copied to an extent that is not allocated");
    require(holds(200, 110), "a refused copy changed a page");
    storage.copyPages(std::array<kv::PageCopy, 2>{{{5, 200}, {6, 255}}});
    require(holds(200, 10) && holds(255, 60) && holds(5, 10) && holds(6, 60) && holds(201, 160),
            "a copied page does not hold its source's tensors, or its neighbour changed");

    // Clearing zeroes every tensor of every layer of a page and leaves its
    // neighbours; a page without memory clears nothing.
    const auto zeroed = [&](uint32_t page) {
        for (const auto span : storage.spans(page))
            if (!std::all_of(span.begin(), span.end(),
                             [](std::byte byte) { return byte == std::byte{0}; }))
                return false;
        return true;
    };
    rejects([&] { storage.clearPages(std::array<uint32_t, 2>{200, 300}); },
            "cannot clear a KV page of an extent that is not allocated",
            "a page of an extent that is not allocated was cleared");
    require(holds(200, 10), "a refused clear changed a page");
    storage.clearPages(std::array<uint32_t, 1>{200});
    require(zeroed(200) && holds(201, 160) && holds(255, 60),
            "a cleared page kept content, or its neighbour changed");

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
        // A BF16 page clears its keys and values, and leaves its neighbour.
        for (const uint32_t page : {0u, 1u})
            for (const auto span : bf16.spans(page))
                std::fill(span.begin(), span.end(), std::byte{0x5a});
        bf16.clearPages(std::array<uint32_t, 1>{0});
        const auto all = [&](uint32_t page, std::byte value) {
            for (const auto span : bf16.spans(page))
                if (!std::all_of(span.begin(), span.end(),
                                 [&](std::byte byte) { return byte == value; }))
                    return false;
            return true;
        };
        require(all(0, std::byte{0}) && all(1, std::byte{0x5a}),
                "a cleared BF16 page kept content, or its neighbour changed");
        require(bf16.allocateExtent(1) &&
                    bf16.actualAllocatedBytes() == uint64_t{2 * extent} * bf16.bytesPerPage(),
                "BF16 growth did not account for both extents");
        bf16.releaseExtent(1);
        require(!bf16.isAllocated(extent) && bf16.allocateExtent(1),
                "BF16 extent could not be allocated again after release");
    }
    // A QSA index: 2 KiB of pooled keys per layer page in the index plane
    // after every layer's K/V, which a copy moves and a clear zeroes with
    // the rest of the page.
    for (const auto layout : {indexedLayout, indexedBf16Layout}) {
        const uint32_t extent = layout.minimumExtentPages();
        require(layout.indexBytesPerLayerPage() == 2048 &&
                    layout.bytesPerModelPage() == 12 * (layout.bytesPerLayerPage() + 2048) &&
                    extent % layout.extentAlignmentPages() == 0,
                "a QSA index's page bytes or alignment changed");
        kv::PageStorage indexed(backend, governor.allocationAdmission(), layout, 2 * extent, extent);
        require(indexed.allocateExtent(0) && indexed.allocateExtent(1) && indexed.indexOffset(0) == 0 &&
                    indexed.indexOffset(11) == uint64_t{11} * extent * 2048 &&
                    uint64_t{extent} * layout.kvBytesPerModelPage() % kv::kExtentRegionAlignmentBytes == 0 &&
                    indexed.indexOffset(1) % kv::kExtentRegionAlignmentBytes == 0,
                "a QSA index region is misplaced");
        rejects([&] { (void)indexed.indexOffset(12); }, "KV layer has no index region",
                "a layer past the layout had an index region");
        requireSpansTileExtent(indexed, 0);
        const auto fill = [&](uint32_t page, std::byte value) {
            for (const auto span : indexed.spans(page)) std::fill(span.begin(), span.end(), value);
        };
        const auto holdsAll = [&](uint32_t page, std::byte value) {
            return std::ranges::all_of(indexed.spans(page), [&](auto span) {
                return std::ranges::all_of(span, [&](std::byte byte) { return byte == value; });
            });
        };
        fill(3, std::byte{0x3c});
        fill(extent + 1, std::byte{0});
        indexed.copyPages(std::array{kv::PageCopy{3, extent + 1}});
        require(holdsAll(extent + 1, std::byte{0x3c}), "a copied page left its pooled keys behind");
        indexed.clearPages(std::array<uint32_t, 1>{3});
        require(holdsAll(3, std::byte{0}), "a cleared page kept its pooled keys");
    }
    rejects([&] { (void)kv::PageStorage(backend, governor.allocationAdmission(), kvLayout, 256, 128).indexOffset(0); },
            "KV layer has no index region", "a layout without an index had an index region");
    indexPagesHoldOnlyTheirIndex(backend, governor);
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
