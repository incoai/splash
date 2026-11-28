#pragma once

#include "metal/abi/KvExtent.h"
#include "ops/KvSlotFormat.hpp"
#include "ops/PagedKv.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace splash::test {

// Pages in host memory, reached as kv::PageStorage reaches its pages: each
// slot's keys, key scales, values and value scales (kv::Layout::pageSlots),
// then each layer's index; a layer in its slot (kv::Layout::slot) on the page
// of its block that holds it, and a block's spans in its layers' regions'
// order (blockSpans). Every page is Full until a test gives it another form:
// the index form, which a tier reads its index alone into
// (engine::BasicKvPageTier).
class HostKvPages final {
public:
  HostKvPages(kv::Layout layout, uint32_t pages)
      : layout_(layout), bytes_(pages * layout.bytesPerModelPage(), std::byte{0}),
        forms_(pages, kv::PageForm::Full) {}

  [[nodiscard]] kv::Layout layout() const noexcept { return layout_; }
  [[nodiscard]] kv::PageForm form(uint32_t page) const { return forms_.at(page); }
  void setForm(uint32_t page, kv::PageForm form) { forms_.at(page) = form; }

  [[nodiscard]] kv::SlotFormat::LayerSpans layerSpans(uint32_t page, uint32_t layer) const {
    return slotSpans(page, layout_.slot(layer));
  }
  [[nodiscard]] kv::SlotFormat::LayerSpans slotSpans(uint32_t page, uint32_t slot) const {
    std::byte *at = bytes_.data() + page * layout_.bytesPerModelPage() + slot * layout_.bytesPerLayerPage();
    kv::SlotFormat::LayerSpans result;
    for (uint32_t tensor = SPLASH_KV_KEYS; tensor <= SPLASH_KV_VALUE_SCALES; ++tensor) {
      const uint64_t bytes = tensor % 2 ? layout_.scaleBytesPerLayerPage() : layout_.dataBytesPerLayerPage();
      result[tensor] = {at, bytes};
      at += bytes;
    }
    return result;
  }
  [[nodiscard]] std::vector<std::span<std::byte>> indexSpans(uint32_t page) const {
    std::byte *at =
        bytes_.data() + page * layout_.bytesPerModelPage() + uint64_t{layout_.pageSlots()} * layout_.bytesPerLayerPage();
    std::vector<std::span<std::byte>> result;
    for (uint32_t layer = 0; layer < layout_.attentionLayers && layout_.indexDimension; ++layer)
      result.emplace_back(at + layer * layout_.indexBytesPerLayerPage(), layout_.indexBytesPerLayerPage());
    return result;
  }
  [[nodiscard]] std::vector<std::span<std::byte>> spans(uint32_t page) const {
    std::vector<std::span<std::byte>> result;
    for (uint32_t slot = 0; slot < layout_.regions(form(page)); ++slot)
      for (const std::span<std::byte> tensor : slotSpans(page, slot))
        if (!tensor.empty()) result.push_back(tensor);
    const std::vector<std::span<std::byte>> index = indexSpans(page);
    result.insert(result.end(), index.begin(), index.end());
    return result;
  }
  [[nodiscard]] std::vector<std::span<std::byte>> blockSpans(std::span<const uint32_t> pages) const {
    std::vector<uint32_t> regionLayers(layout_.attentionLayers);
    for (uint32_t layer = 0; layer < layout_.attentionLayers; ++layer)
      regionLayers[layout_.region(layer)] = layer;
    std::vector<std::span<std::byte>> result;
    // A block without its window unit is its base page alone.
    for (const uint32_t layer : regionLayers)
      if (layout_.blockPage(layer) < pages.size())
        for (const std::span<std::byte> tensor : layerSpans(pages[layout_.blockPage(layer)], layer))
          if (!tensor.empty()) result.push_back(tensor);
    const std::vector<std::span<std::byte>> index = indexSpans(pages.front());
    result.insert(result.end(), index.begin(), index.end());
    return result;
  }
  [[nodiscard]] std::span<const std::byte> page(uint32_t page) const {
    return std::span(bytes_).subspan(page * layout_.bytesPerModelPage(), layout_.bytesPerModelPage());
  }
  // Every byte of the page a value of its own.
  void fill(uint32_t page, uint32_t seed) {
    uint32_t state = seed;
    for (std::byte &value : std::span(bytes_).subspan(page * layout_.bytesPerModelPage(), layout_.bytesPerModelPage())) {
      state = state * 1664525u + 1013904223u;
      value = static_cast<std::byte>(state >> 24);
    }
  }

private:
  kv::Layout layout_;
  mutable std::vector<std::byte> bytes_;
  std::vector<kv::PageForm> forms_;
};

} // namespace splash::test
