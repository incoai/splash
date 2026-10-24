#pragma once

#include "Checked.hpp"
#include "ops/PagedKv.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace splash::kv {

// Bytes of a page's tier slot (model::SlotFile): an offset from the slot's
// start and a length.
struct SlotRange final {
  uint64_t offset = 0;
  uint64_t bytes = 0;

  bool operator==(const SlotRange &) const = default;
};

// How a page sits in its tier slot, and the only definition of it.
//
// Version 0, for a layout that cannot spill (Layout::spillable): the page's
// spans in PageStorage::spans order, one segment. A Slim page's are the
// spans of the regions it holds and its index: without an index, the
// first bytes of a Full page's payload.
//
// Version 1, for a spillable layout: the page's index, layer by layer
// (indexBytesPerLayerPage each), then, from base = the index rounded up to
// kReadAlignmentBytes, one record per layer and block of
// SPLASH_ATTENTION_BLOCK_TOKENS tokens, record (layer, block) at base +
// (8 layer + block) pitch. A record holds the block's tokens of each tensor
// slab of the layer, tensor-major and head by head: keys, key scales, values,
// value scales (BF16 has no scales), so a slot is a permutation of the page's
// bytes, with zeros where the pitch or the base leaves room. The pitch is the
// smallest alignUp(recordBytes, 2^j), j <= 12, at which every record lies
// within the fewest aligned units it can, so a block is one aligned read
// (int8 Flash-Next: 4,160 B records, one 8 KiB read each). Its segments,
// which a persistent file checksums, are the index region up to base, then
// each record with the room after it.
//
// The page functions reach a page through `pages`: PageStorage, or any
// storage that reaches its pages as PageStorage does (spans, layerSpans,
// indexSpans).
class SlotFormat final {
public:
  // The alignment of a read beside the tier's worker (model::SlotFile::readAt).
  static constexpr uint64_t kReadAlignmentBytes = kSlotReadAlignmentBytes;
  // A version-1 page's blocks per layer: a location's places.
  static constexpr uint32_t kBlocksPerPage = 1u << SPLASH_ATTENTION_LOCATION_BLOCK_BITS;
  // A page's tensors of one layer, by SPLASH_KV_KEYS to
  // SPLASH_KV_VALUE_SCALES, empty scales for BF16 (PageStorage::layerSpans).
  using LayerSpans = std::array<std::span<std::byte>, 4>;

  // The format of the layout's pages: version 1 when it is spillable.
  explicit SlotFormat(const Layout &layout);

  [[nodiscard]] uint32_t version() const noexcept { return recordBytes_ ? 1 : 0; }
  [[nodiscard]] const Layout &layout() const noexcept { return layout_; }
  // The bytes a slot stores of a page of the form, a Full one's by default
  // (model::SlotFile::slotBytesFor rounds them up), which every slot has
  // room for. std::logic_error for a form version 1 does not store whole: a
  // spillable layout's slots hold Full pages, and an Index page's restore
  // reads indexRange().
  [[nodiscard]] uint64_t payloadBytes(PageForm form = PageForm::Full) const;
  // The byte lengths of the payload's consecutive segments from its start
  // (model::SlotFile::Options::segments): one for version 0.
  [[nodiscard]] std::vector<uint64_t> segments() const;

  // Version 1 only; std::logic_error for version 0, std::out_of_range for a
  // layer or block past the page. The page's index; one record and its
  // pitch; one layer's records, which a prefill's history reads together.
  [[nodiscard]] SlotRange indexRange() const;
  [[nodiscard]] uint64_t recordBytes() const;
  [[nodiscard]] uint64_t recordPitch() const;
  [[nodiscard]] SlotRange recordRange(uint32_t layer, uint32_t block) const;
  [[nodiscard]] SlotRange layerRange(uint32_t layer) const;

  // The page's bytes in slot order, with the zeros between, which a write
  // gathers (model::SlotFile::write): its spans for version 0, its index and
  // record pieces for version 1.
  template <class Pages>
  [[nodiscard]] std::vector<std::span<const std::byte>> transferSpans(const Pages &pages, uint32_t page) const;
  // What a read of the whole payload holds, into the page: a whole restore,
  // of what the page's form holds. std::invalid_argument for another
  // payload size.
  template <class Pages>
  void scatterPage(std::span<const std::byte> payload, const Pages &pages, uint32_t page) const;
  // Version 1 only, and std::invalid_argument for bytes or spans of other
  // sizes. What indexRange() read, into a page's index
  // (PageStorage::indexSpans).
  void scatterIndex(std::span<const std::byte> index, std::span<const std::span<std::byte>> indexSpans) const;
  // What recordRange(layer, block) read, into block `place` of a page's
  // layer (PageStorage::layerSpans of any storage with the layout's heads,
  // dimension and format: the pool, the hot block cache, staging).
  void scatterRecord(std::span<const std::byte> record, const LayerSpans &layer, uint32_t place) const;
  // What layerRange(layer) read, into a page's layer as scatterRecord does
  // each of its blocks at its own place.
  void scatterLayer(std::span<const std::byte> records, const LayerSpans &layer) const;

private:
  // Version 1: the index, then every record, the zeros between.
  [[nodiscard]] std::vector<std::span<const std::byte>> recordOrder(std::span<const std::span<std::byte>> index,
                                                                   std::span<const LayerSpans> layers) const;
  // `payload` into `spans` in order; std::invalid_argument unless they hold
  // it exactly.
  static void scatterInOrder(std::span<const std::byte> payload, std::span<const std::span<std::byte>> spans);
  // std::logic_error for version 0.
  void requireRecords() const;

  Layout layout_;
  // Version 1: a record, its pitch and where the first one starts; all zero
  // for version 0.
  uint64_t recordBytes_ = 0;
  uint64_t recordPitch_ = 0;
  uint64_t recordsOffset_ = 0;
};

template <class Pages>
std::vector<std::span<const std::byte>> SlotFormat::transferSpans(const Pages &pages, uint32_t page) const {
  if (!version()) {
    const std::vector<std::span<std::byte>> spans = pages.spans(page);
    return {spans.begin(), spans.end()};
  }
  std::vector<LayerSpans> layers(layout_.attentionLayers);
  for (uint32_t layer = 0; layer < layout_.attentionLayers; ++layer) layers[layer] = pages.layerSpans(page, layer);
  return recordOrder(pages.indexSpans(page), layers);
}

template <class Pages>
void SlotFormat::scatterPage(std::span<const std::byte> payload, const Pages &pages, uint32_t page) const {
  if (payload.size() != payloadBytes(pages.form(page)))
    throw std::invalid_argument("a KV slot's payload has another size");
  if (!version()) {
    scatterInOrder(payload, pages.spans(page));
    return;
  }
  const SlotRange index = indexRange();
  scatterIndex(payload.subspan(index.offset, index.bytes), pages.indexSpans(page));
  for (uint32_t layer = 0; layer < layout_.attentionLayers; ++layer) {
    const SlotRange records = layerRange(layer);
    scatterLayer(payload.subspan(records.offset, records.bytes), pages.layerSpans(page, layer));
  }
}

} // namespace splash::kv
