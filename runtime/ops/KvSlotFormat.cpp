#include "ops/KvSlotFormat.hpp"

#include <cstring>
#include <utility>

namespace splash::kv {

namespace {

constexpr uint64_t alignUp(uint64_t bytes, uint64_t alignment) noexcept {
  return (bytes + alignment - 1) / alignment * alignment;
}

// What a write stores where a slot holds no page bytes: less than a read
// unit, the most a pitch or the base leaves.
constexpr std::array<std::byte, SlotFormat::kReadAlignmentBytes> kZeros{};

// The pitch rule (SlotFormat): the smallest alignUp(record, 2^j) at which
// each of `records` records from `base`, itself aligned, lies within the
// fewest aligned units it can. A whole number of units always does.
uint64_t pitchFor(uint64_t base, uint64_t record, uint32_t records) noexcept {
  constexpr uint64_t unit = SlotFormat::kReadAlignmentBytes;
  const uint64_t units = alignUp(record, unit);
  for (uint64_t alignment = 1; alignment < unit; alignment *= 2) {
    const uint64_t pitch = alignUp(record, alignment);
    bool fits = true;
    for (uint32_t index = 0; index < records && fits; ++index)
      fits = (base + index * pitch) % unit + record <= units;
    if (fits) return pitch;
  }
  return units;
}

// Whether `layer` holds the layout's tensors of one page.
bool holdsLayer(const Layout &layout, const SlotFormat::LayerSpans &layer) noexcept {
  for (uint32_t tensor = SPLASH_KV_KEYS; tensor <= SPLASH_KV_VALUE_SCALES; ++tensor)
    if (layer[tensor].size() != (tensor % 2 ? layout.scaleBytesPerLayerPage() : layout.dataBytesPerLayerPage()))
      return false;
  return true;
}

// Visits block `place`'s pieces of a page's layer in record order. Each
// tensor's slab holds every head's tokens of the page in turn ([head][token]
// [...], values token-major as a spillable layout's are), so a head's block
// is one run of 1/(heads x blocks) of the slab.
template <class Visit>
void visitRecord(const Layout &layout, const SlotFormat::LayerSpans &layer, uint32_t place, Visit visit) {
  for (const std::span<std::byte> slab : layer) {
    if (slab.empty()) continue;
    const uint64_t run = slab.size() / (uint64_t{layout.kvHeads} * SlotFormat::kBlocksPerPage);
    for (uint32_t head = 0; head < layout.kvHeads; ++head)
      visit(slab.subspan((uint64_t{head} * SlotFormat::kBlocksPerPage + place) * run, run));
  }
}

} // namespace

SlotFormat::SlotFormat(const Layout &layout) : layout_(layout) {
  if (!layout_.valid()) throw std::invalid_argument("KV slot format layout is invalid");
  if (!layout_.spillable()) return;
  recordBytes_ = layout_.bytesPerLayerPage() / kBlocksPerPage;
  recordsOffset_ = alignUp(layout_.indexBytesPerModelPage(), kReadAlignmentBytes);
  recordPitch_ = pitchFor(recordsOffset_, recordBytes_, layout_.attentionLayers * kBlocksPerPage);
}

uint64_t SlotFormat::payloadBytes(PageForm form) const {
  if (!version()) return layout_.bytesPerModelPage(form);
  if (form != PageForm::Full) throw std::logic_error("a KV slot of block records holds a Full page");
  return recordsOffset_ + (uint64_t{layout_.attentionLayers} * kBlocksPerPage - 1) * recordPitch_ + recordBytes_;
}

std::vector<uint64_t> SlotFormat::segments() const {
  if (!version()) return {payloadBytes()};
  std::vector<uint64_t> result(uint64_t{layout_.attentionLayers} * kBlocksPerPage + 1, recordPitch_);
  result.front() = recordsOffset_;
  result.back() = recordBytes_;
  return result;
}

void SlotFormat::requireRecords() const {
  if (!version()) throw std::logic_error("a KV layout that cannot spill has no block records");
}

SlotRange SlotFormat::indexRange() const {
  requireRecords();
  return {0, layout_.indexBytesPerModelPage()};
}

uint64_t SlotFormat::recordBytes() const {
  requireRecords();
  return recordBytes_;
}

uint64_t SlotFormat::recordPitch() const {
  requireRecords();
  return recordPitch_;
}

SlotRange SlotFormat::recordRange(uint32_t layer, uint32_t block) const {
  requireRecords();
  if (layer >= layout_.attentionLayers || block >= kBlocksPerPage)
    throw std::out_of_range("KV block record is outside the page");
  return {recordsOffset_ + (uint64_t{layer} * kBlocksPerPage + block) * recordPitch_, recordBytes_};
}

SlotRange SlotFormat::layerRange(uint32_t layer) const {
  const SlotRange first = recordRange(layer, 0);
  return {first.offset, (kBlocksPerPage - 1) * recordPitch_ + recordBytes_};
}

std::vector<std::span<const std::byte>> SlotFormat::recordOrder(std::span<const std::span<std::byte>> index,
                                                                std::span<const LayerSpans> layers) const {
  if (index.size() != layout_.attentionLayers || layers.size() != layout_.attentionLayers)
    throw std::invalid_argument("a KV page's spans do not hold its layout");
  for (uint32_t layer = 0; layer < layout_.attentionLayers; ++layer)
    if (index[layer].size() != layout_.indexBytesPerLayerPage() || !holdsLayer(layout_, layers[layer]))
      throw std::invalid_argument("a KV page's spans do not hold its layout");
  std::vector<std::span<const std::byte>> result;
  result.reserve(index.size() + 1 + layers.size() * kBlocksPerPage * (4 * layout_.kvHeads + 1));
  result.assign(index.begin(), index.end());
  const auto pad = [&](uint64_t bytes) {
    if (bytes) result.emplace_back(kZeros.data(), bytes);
  };
  pad(recordsOffset_ - layout_.indexBytesPerModelPage());
  bool first = true;
  for (const LayerSpans &layer : layers) {
    for (uint32_t place = 0; place < kBlocksPerPage; ++place) {
      if (!std::exchange(first, false)) pad(recordPitch_ - recordBytes_);
      visitRecord(layout_, layer, place, [&](std::span<std::byte> piece) { result.emplace_back(piece); });
    }
  }
  return result;
}

void SlotFormat::scatterInOrder(std::span<const std::byte> payload, std::span<const std::span<std::byte>> spans) {
  uint64_t total = 0;
  for (const std::span<std::byte> span : spans) total += span.size();
  if (total != payload.size()) throw std::invalid_argument("a KV page's spans do not hold its slot's payload");
  for (const std::span<std::byte> span : spans) {
    std::memcpy(span.data(), payload.data(), span.size());
    payload = payload.subspan(span.size());
  }
}

void SlotFormat::scatterIndex(std::span<const std::byte> index,
                              std::span<const std::span<std::byte>> indexSpans) const {
  if (index.size() != indexRange().bytes || indexSpans.size() != layout_.attentionLayers)
    throw std::invalid_argument("a KV page's index read has another size than its index");
  scatterInOrder(index, indexSpans);
}

void SlotFormat::scatterRecord(std::span<const std::byte> record, const LayerSpans &layer, uint32_t place) const {
  requireRecords();
  if (place >= kBlocksPerPage) throw std::out_of_range("KV block record is outside the page");
  if (record.size() != recordBytes_ || !holdsLayer(layout_, layer))
    throw std::invalid_argument("a KV block record or its page's layer has another size");
  visitRecord(layout_, layer, place, [&](std::span<std::byte> piece) {
    std::memcpy(piece.data(), record.data(), piece.size());
    record = record.subspan(piece.size());
  });
}

void SlotFormat::scatterLayer(std::span<const std::byte> records, const LayerSpans &layer) const {
  requireRecords();
  if (records.size() != (kBlocksPerPage - 1) * recordPitch_ + recordBytes_)
    throw std::invalid_argument("a KV layer's records have another size");
  for (uint32_t place = 0; place < kBlocksPerPage; ++place)
    scatterRecord(records.subspan(place * recordPitch_, recordBytes_), layer, place);
}

} // namespace splash::kv
