// How a KV page sits in its tier slot (kv::SlotFormat): a layout that cannot
// spill keeps its spans' order; a spillable one stores its index, then one
// record per layer and block at the pitch rule's pitch, so that each record is
// one aligned read of the fewest units. A write's spans and a whole restore
// move a page's bytes exactly, the zeros between staying out of it; a record,
// a layer's records and the index land where the kernels address those
// tokens (abi/KvExtent.h). Built under ASan too.

#include "HostKvPages.hpp"
#include "TestChecks.hpp"
#include "metal/abi/KvExtent.h"
#include "ops/KvSlotFormat.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;
using splash::kv::SlotFormat;
using splash::kv::SlotRange;
using splash::test::rejects;
using splash::test::require;

namespace {

constexpr uint64_t kUnit = SlotFormat::kReadAlignmentBytes;
constexpr uint32_t kBlocks = SlotFormat::kBlocksPerPage;
constexpr uint32_t kBlockTokens = SPLASH_ATTENTION_BLOCK_TOKENS;

// Qwen3.8-Flash-Next's, int8 and BF16: 12 layers of 2 heads, token-major
// values and a 128-wide index.
constexpr kv::Layout kFlashNext{12, 2, 256, kv::Format::Int8, 128, true};
constexpr kv::Layout kFlashNextBf16{12, 2, 256, kv::Format::BFloat16, 128, true};

// The bytes a write of the page stores, in slot order.
std::vector<std::byte> gather(const SlotFormat &format, const test::HostKvPages &pages, uint32_t page) {
  std::vector<std::byte> payload;
  for (const std::span<const std::byte> span : format.transferSpans(pages, page))
    payload.insert(payload.end(), span.begin(), span.end());
  return payload;
}

// Record (layer, block) as the kernels address its tokens: per tensor, each
// head's tokens of the block, keys and token-major values by
// splash_kv_key_element, scales by splash_kv_scale_element.
std::vector<std::byte> expectedRecord(const kv::Layout &layout, const test::HostKvPages &pages, uint32_t page,
                                      uint32_t layer, uint32_t block) {
  const SlotFormat::LayerSpans tensors = pages.layerSpans(page, layer);
  const uint64_t element = layout.format == kv::Format::Int8 ? 1 : 2;
  std::vector<std::byte> record;
  for (uint32_t tensor = SPLASH_KV_KEYS; tensor <= SPLASH_KV_VALUE_SCALES; ++tensor) {
    const bool scales = tensor % 2;
    if (scales && layout.format != kv::Format::Int8) continue;
    for (uint32_t head = 0; head < layout.kvHeads; ++head) {
      const uint32_t token = block * kBlockTokens;
      const uint64_t first = scales ? splash_kv_scale_element(head, token) * sizeof(float)
                                    : splash_kv_key_element(head, token, 0) * element;
      const uint64_t bytes = kBlockTokens * (scales ? sizeof(float) : SPLASH_KV_HEAD_DIMENSION * element);
      const auto run = tensors[tensor].subspan(first, bytes);
      record.insert(record.end(), run.begin(), run.end());
    }
  }
  return record;
}

[[nodiscard]] bool zero(std::span<const std::byte> bytes) {
  return std::ranges::all_of(bytes, [](std::byte value) { return value == std::byte{0}; });
}

// Layouts across heads, formats, index widths and layer counts, the padded
// ones included: one head leaves room after each record, an index of three
// layers room before the first.
const std::vector<kv::Layout> &spillableLayouts() {
  static const std::vector<kv::Layout> layouts{
      kFlashNext,
      kFlashNextBf16,
      {12, 4, 256, kv::Format::Int8, 128, true},
      {12, 4, 256, kv::Format::BFloat16, 128, true},
      {12, 2, 256, kv::Format::Int8, 64, true},
      {2, 2, 256, kv::Format::Int8, 64, true},
      {3, 2, 256, kv::Format::Int8, 128, true},
      {5, 1, 256, kv::Format::Int8, 128, true},
      {4, 3, 256, kv::Format::Int8, 64, true},
  };
  return layouts;
}

// Flash-Next's numbers: 4,160 B int8 records at their own size as pitch,
// each one aligned 8 KiB read, after a 24 KiB index; 8 KiB BF16 records.
void testFlashNextSlots() {
  const SlotFormat format(kFlashNext);
  require(format.version() == 1 && format.recordBytes() == 4160 && format.recordPitch() == 4160 &&
              format.indexRange() == SlotRange{0, 24576} && format.payloadBytes() == 423936,
          "Flash-Next int8 slots are not 4,160 B records after a 24 KiB index");
  require(format.recordRange(0, 0) == SlotRange{24576, 4160} &&
              format.recordRange(1, 3) == SlotRange{24576 + 11 * 4160, 4160} &&
              format.layerRange(5) == SlotRange{24576 + 5 * 33280, 33280},
          "Flash-Next int8 records or layer runs are not where the rule puts them");
  const std::vector<uint64_t> segments = format.segments();
  require(segments.size() == 97 && segments.front() == 24576 &&
              std::ranges::all_of(segments | std::views::drop(1), [](uint64_t bytes) { return bytes == 4160; }),
          "Flash-Next int8 slots are not the index and 96 records as segments");
  for (uint32_t layer = 0; layer < kFlashNext.attentionLayers; ++layer)
    for (uint32_t block = 0; block < kBlocks; ++block) {
      const SlotRange record = format.recordRange(layer, block);
      require(record.offset / kUnit + 1 == (record.offset + record.bytes - 1) / kUnit,
              "a Flash-Next int8 record is not one aligned 8 KiB read");
    }
  const SlotFormat bf16(kFlashNextBf16);
  require(bf16.recordBytes() == 8192 && bf16.recordPitch() == 8192 && bf16.payloadBytes() == 811008 &&
              bf16.recordRange(11, 7) == SlotRange{24576 + 95 * 8192, 8192},
          "Flash-Next BF16 slots are not 8 KiB records after a 24 KiB index");
}

// The pitch rule: records follow the index from its aligned end, each within
// the fewest aligned units it can, at the smallest pitch alignUp(record, 2^j)
// that does; the segments cover the payload.
void testPitchRule() {
  for (const kv::Layout &layout : spillableLayouts()) {
    const SlotFormat format(layout);
    const std::string name = std::to_string(layout.attentionLayers) + " layers of " + std::to_string(layout.kvHeads) +
                             " heads, " + std::string(kv::formatName(layout.format)) + ", index " +
                             std::to_string(layout.indexDimension);
    const uint64_t record = format.recordBytes(), pitch = format.recordPitch();
    const uint64_t base = format.recordRange(0, 0).offset, units = (record + kUnit - 1) / kUnit;
    require(record == layout.bytesPerLayerPage() / kBlocks && pitch >= record && pitch - record < kUnit &&
                base % kUnit == 0 && base >= layout.indexBytesPerModelPage() &&
                base - layout.indexBytesPerModelPage() < kUnit,
            name + ": records are not a block of a layer each, from the index's aligned end");
    const auto fits = [&](uint64_t candidate) {
      for (uint32_t index = 0; index < layout.attentionLayers * kBlocks; ++index)
        if ((base + index * candidate) % kUnit + record > units * kUnit) return false;
      return true;
    };
    require(fits(pitch), name + ": a record spans more aligned units than it must");
    for (uint64_t alignment = 1; alignment <= kUnit; alignment *= 2) {
      const uint64_t candidate = (record + alignment - 1) / alignment * alignment;
      require(candidate >= pitch || !fits(candidate), name + ": a smaller pitch keeps every record within its units");
    }
    uint64_t total = 0;
    for (const uint64_t bytes : format.segments()) total += bytes;
    const SlotRange last = format.recordRange(layout.attentionLayers - 1, kBlocks - 1);
    require(total == format.payloadBytes() && last.offset + last.bytes == total &&
                format.segments().front() == base,
            name + ": the segments are not the index region and each record, covering the payload");
  }
  const SlotFormat padded(kv::Layout{5, 1, 256, kv::Format::Int8, 128, true});
  const SlotFormat offset(kv::Layout{3, 2, 256, kv::Format::Int8, 128, true});
  require(padded.recordBytes() == 2080 && padded.recordPitch() == 4096 && offset.recordRange(0, 0).offset == 8192,
          "a record of one head does not take a unit of its own, or three layers' index no aligned end");
}

// A layout without block records, an index or token-major values, keeps its
// spans' order as one segment, a windowed layout's region by region
// (kv::Layout::region), has no records, and a whole restore puts the payload
// back in that order.
void testUnspillableLayouts() {
  for (const kv::Layout &layout : {kv::Layout{16, 4, 256}, kv::Layout{10, 2, 256, kv::Format::BFloat16},
                                   kv::Layout{12, 2, 256, kv::Format::Int8, 128, false},
                                   kv::Layout{12, 2, 256, kv::Format::Int8, 0, true},
                                   kv::Layout{6, 4, 128, kv::Format::Int8, 0, false, 512, 0b011010}}) {
    const SlotFormat format(layout);
    require(format.version() == 0 && format.payloadBytes() == layout.bytesPerModelPage() &&
                format.segments() == std::vector<uint64_t>{layout.bytesPerModelPage()},
            "a layout that cannot spill is not one segment of its page");
    test::HostKvPages pages(layout, 2);
    pages.fill(0, 7);
    const std::vector<std::span<const std::byte>> spans = format.transferSpans(pages, 0);
    const std::vector<std::span<std::byte>> expected = pages.spans(0);
    require(std::ranges::equal(spans, expected,
                               [](auto left, auto right) {
                                 return left.data() == right.data() && left.size() == right.size();
                               }),
            "a layout that cannot spill does not write its spans in order");
    format.scatterPage(gather(format, pages, 0), pages, 1);
    require(std::ranges::equal(pages.page(0), pages.page(1)), "a whole restore did not put the spans back in order");
    rejects([&] { static_cast<void>(format.indexRange()); }, "has no block records",
            "a layout that cannot spill has an index range");
    rejects([&] { static_cast<void>(format.recordRange(0, 0)); }, "has no block records",
            "a layout that cannot spill has records");
    rejects([&] { format.scatterRecord({}, pages.layerSpans(0, 0), 0); }, "has no block records",
            "a layout that cannot spill scattered a record");
  }
  rejects([] { static_cast<void>(SlotFormat(kv::Layout{})); }, "layout is invalid",
          "an invalid layout has a slot format");
}

// A Slim page of a windowed layout, which holds the regions of the layers
// no window bounds (kv::Layout::regions), stores those spans in order: the
// first bytes of a Full page's payload, which a Slim page restores into as
// a Full page restores, leaving the rest of its memory alone. A spillable
// layout's records hold Full pages alone.
void testSlimPayloads() {
  for (const kv::Layout &layout : {kv::Layout{6, 4, 128, kv::Format::Int8, 0, false, 512, 0b011010},
                                   kv::Layout{6, 4, 128, kv::Format::BFloat16, 0, false, 512, 0b111000}}) {
    const SlotFormat format(layout);
    const uint64_t slim = layout.bytesPerModelPage(kv::PageForm::Slim);
    require(layout.slimmable() && format.payloadBytes(kv::PageForm::Slim) == slim &&
                slim == 3 * layout.bytesPerLayerPage() && format.payloadBytes() == layout.bytesPerModelPage(),
            "a Slim page's payload is not the regions of its layers no window bounds");
    test::HostKvPages pages(layout, 2);
    pages.fill(0, 11);
    const std::vector<std::byte> full = gather(format, pages, 0);
    pages.setForm(0, kv::PageForm::Slim);
    const std::vector<std::byte> payload = gather(format, pages, 0);
    require(payload.size() == slim && std::ranges::equal(payload, std::span(full).first(slim)),
            "a Slim page's payload is not the first bytes of its Full payload");
    pages.setForm(1, kv::PageForm::Slim);
    format.scatterPage(payload, pages, 1);
    require(std::ranges::equal(pages.page(1).first(slim), pages.page(0).first(slim)) &&
                zero(pages.page(1).subspan(slim)),
            "a Slim restore did not put its regions back, or wrote past them");
    rejects([&] { format.scatterPage(full, pages, 1); }, "another size", "a Full payload was restored as Slim");
  }
  const SlotFormat records(kFlashNext);
  rejects([&] { static_cast<void>(records.payloadBytes(kv::PageForm::Slim)); }, "holds a Full page",
          "a slot of block records held a Slim page");
}

// A write stores the page's bytes as the format says, zeros in between: its
// index, then each record as the kernels address its tokens. A whole restore
// puts every byte back; a record, a layer's records and the index land in a
// page of another storage of the layout where they belong, touching nothing
// else.
void testTransfersAndScatters() {
  for (const kv::Layout &layout : spillableLayouts()) {
    const SlotFormat format(layout);
    test::HostKvPages pages(layout, 3);
    pages.fill(0, layout.attentionLayers * 31 + layout.kvHeads);
    const std::vector<std::byte> payload = gather(format, pages, 0);
    require(payload.size() == format.payloadBytes(), "a write's spans do not hold the payload");
    const SlotRange index = format.indexRange();
    const auto indexSpans = pages.indexSpans(0);
    std::vector<std::byte> expectedIndex;
    for (const auto span : indexSpans) expectedIndex.insert(expectedIndex.end(), span.begin(), span.end());
    require(std::ranges::equal(std::span(payload).subspan(index.offset, index.bytes), expectedIndex) &&
                zero(std::span(payload).subspan(index.bytes, format.recordRange(0, 0).offset - index.bytes)),
            "a write does not store the index layer by layer, zeros after it");
    uint64_t pageBytes = index.bytes;
    for (uint32_t layer = 0; layer < layout.attentionLayers; ++layer) {
      for (uint32_t block = 0; block < kBlocks; ++block) {
        const SlotRange record = format.recordRange(layer, block);
        require(std::ranges::equal(std::span(payload).subspan(record.offset, record.bytes),
                                   expectedRecord(layout, pages, 0, layer, block)),
                "a record does not hold its block's tokens of every tensor and head");
        const uint64_t end = record.offset + record.bytes;
        const uint64_t next = std::min<uint64_t>(record.offset + format.recordPitch(), payload.size());
        require(zero(std::span(payload).subspan(end, next - end)), "a write stores more than zeros between records");
        pageBytes += record.bytes;
      }
    }
    require(pageBytes == layout.bytesPerModelPage(), "a slot holds other than every byte of its page once");

    format.scatterPage(payload, pages, 1);
    require(std::ranges::equal(pages.page(0), pages.page(1)), "a whole restore did not put every byte back");

    // One record at a time into a page of storage without an index, as the
    // hot block cache's: each lands at its place and nowhere else.
    kv::Layout cacheLayout = layout;
    cacheLayout.indexDimension = 0;
    test::HostKvPages cache(cacheLayout, 1);
    for (uint32_t block = 0; block < kBlocks; ++block) {
      const SlotRange record = format.recordRange(2 % layout.attentionLayers, block);
      format.scatterRecord(std::span(payload).subspan(record.offset, record.bytes), cache.layerSpans(0, 1), block);
      require(std::ranges::equal(expectedRecord(cacheLayout, cache, 0, 1, block),
                                 expectedRecord(layout, pages, 0, 2 % layout.attentionLayers, block)),
              "a scattered record did not land at its place");
    }
    const SlotFormat::LayerSpans copied = cache.layerSpans(0, 1);
    const SlotFormat::LayerSpans source = pages.layerSpans(0, 2 % layout.attentionLayers);
    for (uint32_t tensor = 0; tensor < 4; ++tensor)
      require(std::ranges::equal(copied[tensor], source[tensor]), "a layer's scattered records differ from the layer");
    require(zero(cache.layerSpans(0, 0)[SPLASH_KV_KEYS]), "a scattered record touched another layer");

    // A layer's run into a one-layer staging page, and the index alone.
    kv::Layout stagingLayout = cacheLayout;
    stagingLayout.attentionLayers = 1;
    test::HostKvPages staging(stagingLayout, 1);
    const SlotRange run = format.layerRange(layout.attentionLayers - 1);
    format.scatterLayer(std::span(payload).subspan(run.offset, run.bytes), staging.layerSpans(0, 0));
    const SlotFormat::LayerSpans last = pages.layerSpans(0, layout.attentionLayers - 1);
    const SlotFormat::LayerSpans staged = staging.layerSpans(0, 0);
    for (uint32_t tensor = 0; tensor < 4; ++tensor)
      require(std::ranges::equal(staged[tensor], last[tensor]), "a layer's run did not land as the layer");
    format.scatterIndex(std::span(payload).subspan(index.offset, index.bytes), pages.indexSpans(2));
    const auto restoredIndex = pages.indexSpans(2);
    for (uint32_t layer = 0; layer < layout.attentionLayers; ++layer)
      require(std::ranges::equal(restoredIndex[layer], indexSpans[layer]), "an index restore did not land as the index");
    require(zero(pages.layerSpans(2, 0)[SPLASH_KV_KEYS]), "an index restore touched the page's K/V");
  }
}

// Reads of other sizes, places past a page and pages of another layout are
// refused before anything is written.
void testRejections() {
  const SlotFormat format(kFlashNext);
  test::HostKvPages pages(kFlashNext, 1), other(kv::Layout{12, 4, 256, kv::Format::Int8, 128, true}, 1);
  const std::vector<std::byte> record(format.recordBytes());
  rejects([&] { static_cast<void>(format.recordRange(12, 0)); }, "outside the page",
          "a record past the layers was placed");
  rejects([&] { static_cast<void>(format.recordRange(0, kBlocks)); }, "outside the page",
          "a record past the blocks was placed");
  rejects([&] { format.scatterRecord(record, pages.layerSpans(0, 0), kBlocks); }, "outside the page",
          "a record was scattered past the page's places");
  rejects([&] { format.scatterRecord(std::span(record).first(100), pages.layerSpans(0, 0), 0); }, "another size",
          "a short record was scattered");
  rejects([&] { format.scatterRecord(record, other.layerSpans(0, 0), 0); }, "another size",
          "a record was scattered into another layout's layer");
  rejects([&] { format.scatterLayer(record, pages.layerSpans(0, 0)); }, "another size",
          "one record was scattered as a layer's");
  rejects([&] { format.scatterIndex(record, pages.indexSpans(0)); }, "another size",
          "a short index was scattered");
  rejects([&] { format.scatterPage(record, pages, 0); }, "another size", "a short payload was restored");
  rejects([&] { static_cast<void>(format.transferSpans(other, 0)); }, "do not hold its layout",
          "another layout's page was written");
  require(zero(pages.page(0)), "a refused scatter wrote into the page");
}

} // namespace

int main() {
  try {
    testFlashNextSlots();
    testPitchRule();
    testUnspillableLayouts();
    testSlimPayloads();
    testTransfersAndScatters();
    testRejections();
  } catch (const std::exception &error) {
    std::cerr << "kv_slot_format_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "kv_slot_format_test: PASS\n";
  return 0;
}
