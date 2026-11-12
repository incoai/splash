#include "video/model/WeightStream.hpp"

#include "AwakeClock.hpp"
#include "metal/abi/video/Layout.h"
#include "model/WeightImages.hpp"
#include "video/metal/DeviceProfile.hpp"

#include <algorithm>
#include <span>
#include <stdexcept>
#include <string>

namespace splash::video::model {
namespace {

using splash::AwakeClock;
using splash::millisecondsSince;

constexpr uint64_t kQkFixBytes = uint64_t(H3_DIT_QKV) * H3_DIT_HIDDEN * 2;

} // namespace

WeightStream::WeightStream(metal::MetalBackend &backend, const std::vector<StreamedTensor> &tensors,
                           const std::set<int> &qkfixBlocks)
    : readers_(splash::model::kSsdReaders) {
  // A slot tensor holds the largest of its block's streamed tensors (FastH3's qkv + gates are larger than VDN-H3's,
  // and it has no to_out_linear).
  uint64_t slotBytes[kTensors] = {};
  for (const StreamedTensor &t : tensors)
    if (t.slotTensor >= 0 && t.slotTensor < kTensors)
      slotBytes[t.slotTensor] = std::max(slotBytes[t.slotTensor], t.slotOffset + t.tensor->bytes);
  pieces_.resize(H3_DIT_BLOCKS);
  qkfixPieces_.resize(H3_DIT_BLOCKS);
  for (const StreamedTensor &t : tensors) {
    const uint64_t end = t.slotOffset + t.tensor->bytes;
    if (t.block < 0 || t.block >= H3_DIT_BLOCKS || t.slotTensor >= kTensors ||
        (t.slotTensor < 0 && end != kQkFixBytes))
      throw std::runtime_error("streamed tensor " + t.name + " of block " + std::to_string(t.block) +
                               " does not fit its stream slot");
    auto &pieces = t.slotTensor >= 0 ? pieces_[t.block] : qkfixPieces_[t.block];
    for (uint64_t at = 0; at < t.tensor->bytes; at += splash::model::kLoadStepBytes)
      pieces.push_back({t.tensor, at, std::min(splash::model::kLoadStepBytes, t.tensor->bytes - at), t.slotTensor,
                        t.slotOffset + at});
  }
  qkfixRank_.assign(H3_DIT_BLOCKS, -1);
  for (int b : qkfixBlocks) {
    if (qkfixPieces_[b].empty())
      throw std::runtime_error("QKFIX block " + std::to_string(b) + " has no streamed weights");
    qkfixRank_[b] = static_cast<int>(qkfixOrder_.size());
    qkfixOrder_.push_back(b);
  }
  slots_.resize(device::kStreamSlots);
  for (Slot &slot : slots_)
    for (int t = 0; t < kTensors; ++t) slot.tensor[t] = allocate(backend, slotBytes[t], "stream slot");
  if (!qkfixOrder_.empty()) {
    qkfixSlots_.resize(device::kStreamQkFixSlots);
    for (Slot &slot : qkfixSlots_) slot.tensor[0] = allocate(backend, kQkFixBytes, "stream QKFIX slot");
  }
  for (size_t i = 0; i < slots_.size(); ++i) fill(slots_[i], static_cast<int64_t>(i), false);
  for (size_t i = 0; i < qkfixSlots_.size(); ++i) fill(qkfixSlots_[i], static_cast<int64_t>(i), true);
}

WeightStream::~WeightStream() {
  for (std::vector<Slot> *ring : {&slots_, &qkfixSlots_})
    for (Slot &slot : *ring)
      if (slot.read) readers_.stop(*slot.read);
}

// Starts reading position n of a ring's sequence into its slot: block n % 50, or the QKFIX block of rank n % (QKFIX
// blocks per forward).
void WeightStream::fill(Slot &slot, int64_t n, bool qkfix) {
  const int block = qkfix ? qkfixOrder_[static_cast<size_t>(n % static_cast<int64_t>(qkfixOrder_.size()))]
                          : static_cast<int>(n % H3_DIT_BLOCKS);
  const std::vector<Piece> &pieces = (qkfix ? qkfixPieces_ : pieces_)[block];
  slot.fill = n;
  slot.read = readers_.start(
      pieces.size(),
      [&pieces, &slot](size_t i) {
        const Piece &p = pieces[i];
        auto *base = static_cast<uint8_t *>(slot.tensor[std::max(p.slotTensor, 0)].contents());
        p.tensor->read(p.at, std::span<uint8_t>(base + p.offset, p.bytes));
      },
      readers_.readers());
  for (const Piece &p : pieces) readBytes_ += p.bytes;
}

WeightStream::Slot &WeightStream::next(Slot **qkfix) {
  const int64_t n = encoderSequence_++;
  const int rank = qkfixRank_[static_cast<size_t>(n % H3_DIT_BLOCKS)];
  *qkfix = nullptr;
  if (rank >= 0) {
    const int64_t q = (n / H3_DIT_BLOCKS) * static_cast<int64_t>(qkfixOrder_.size()) + rank;
    *qkfix = &qkfixSlots_[static_cast<size_t>(q % static_cast<int64_t>(qkfixSlots_.size()))];
  }
  return slots_[static_cast<size_t>(n % static_cast<int64_t>(slots_.size()))];
}

void WeightStream::wait(Slot &slot) {
  readers_.finish(*slot.read);
  if (std::exception_ptr failure = slot.read->failure()) std::rethrow_exception(failure);
}

void WeightStream::waitRead(Slot &slot, Slot *qkfix) {
  const auto t0 = AwakeClock::now();
  wait(slot);
  if (qkfix) wait(*qkfix);
  const double ms = millisecondsSince(t0);
  waitMs_ += ms;
  if (ms > 1.0) ++waits_;
}

void WeightStream::release(Slot &slot, Slot *qkfix) {
  fill(slot, slot.fill + static_cast<int64_t>(slots_.size()), false);
  if (qkfix) fill(*qkfix, qkfix->fill + static_cast<int64_t>(qkfixSlots_.size()), true);
}

void WeightStream::resetStats() {
  readBytes_ = 0;
  waitMs_ = 0;
  waits_ = 0;
}

StreamStats WeightStream::stats() const { return {waitMs_, waits_, readBytes_ / 1e9}; }

size_t WeightStream::slotBytes() const {
  size_t total = 0;
  for (const Slot &slot : slots_)
    for (int t = 0; t < kTensors; ++t) total += slot.tensor[t].sizeBytes();
  for (const Slot &slot : qkfixSlots_) total += slot.tensor[0].sizeBytes();
  return total;
}

} // namespace splash::video::model
