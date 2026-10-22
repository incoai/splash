#include "video/h3/engine/WeightStream.hpp"

#include "AwakeClock.hpp"
#include "video/h3/engine/DeviceProfile.hpp"
#include "metal/abi/video/Layout.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace h3 {
namespace {

using splash::AwakeClock;
using splash::millisecondsSince;

// A slot starts free. libdispatch traps (SIGTRAP, "semaphore object deallocated while in use") when a semaphore is
// released with a value below the one it was created with, which a stream torn down mid-ring (h3_dit_release after a
// forward) always is when created with 1; created with 0 and signalled once, any value >= 0 is valid at release.
dispatch_semaphore_t freeSemaphore() {
  dispatch_semaphore_t s = dispatch_semaphore_create(0);
  dispatch_semaphore_signal(s);
  return s;
}

constexpr uint64_t kQkFixBytes = uint64_t(H3_DIT_QKV) * H3_DIT_HIDDEN * 2;

} // namespace

WeightStream::WeightStream(metal::MetalBackend &backend, const std::vector<model::StreamedTensor> &tensors, int slots) {
  if (slots < 1) throw std::invalid_argument("stream slots must be >= 1 (got " + std::to_string(slots) + ")");
  constexpr uint64_t D = H3_DIT_HIDDEN, F = H3_DIT_FFN, C = H3_DIT_INNER;
  constexpr uint64_t slotBytes[kTensors] = {(H3_DIT_QKV + H3_DIT_GATES) * D, D * C, 2 * F * D, D * F, D * C};
  blockTensors_.resize(H3_DIT_BLOCKS);
  for (const model::StreamedTensor &t : tensors) {
    const uint64_t end = t.slotOffset + t.tensor->bytes;
    if (t.block < 0 || t.block >= H3_DIT_BLOCKS ||
        (t.slotTensor >= 0 ? end > slotBytes[t.slotTensor] : end != kQkFixBytes))
      throw std::runtime_error("streamed tensor " + t.name + " of block " + std::to_string(t.block) +
                               " does not fit its stream slot");
    blockTensors_[t.block].push_back(t);
  }
  slots_.resize(slots);
  for (Slot &slot : slots_) {
    for (int t = 0; t < kTensors; ++t) slot.tensor[t] = allocate(backend, slotBytes[t], "stream slot");
    slot.ready = dispatch_semaphore_create(0);
    slot.free = freeSemaphore();
  }
}

WeightStream::~WeightStream() {
  if (thread_.joinable()) {
    stop_ = true;
    for (Slot &slot : slots_) dispatch_semaphore_signal(slot.free);
    for (Slot &slot : qkfixSlots_) dispatch_semaphore_signal(slot.free);
    thread_.join();
  }
}

void WeightStream::start(metal::MetalBackend &backend, int blocks, const std::set<int> &qkfixBlocks) {
  if (blocks < 1 || blocks > H3_DIT_BLOCKS)
    throw std::invalid_argument("stream: blocks must be 1.." + std::to_string(H3_DIT_BLOCKS));
  blocksPerForward_ = blocks;
  qkfixRank_.assign(blocks, -1);
  qkfixPerForward_ = 0;
  for (int b : qkfixBlocks) {
    if (std::ranges::none_of(blockTensors_[b], [](const model::StreamedTensor &t) { return t.slotTensor < 0; }))
      throw std::runtime_error("QKFIX block " + std::to_string(b) + " has no streamed weights");
    qkfixRank_[b] = qkfixPerForward_++;
  }
  if (qkfixPerForward_ > 0) {
    qkfixSlots_.resize(device::kStreamQkFixSlots);
    for (Slot &slot : qkfixSlots_) {
      slot.tensor[0] = allocate(backend, kQkFixBytes, "stream QKFIX slot");
      slot.ready = dispatch_semaphore_create(0);
      slot.free = freeSemaphore();
    }
  }
  thread_ = std::thread([this] { reader(); });
}

bool WeightStream::waitFree(dispatch_semaphore_t semaphore) {
  while (dispatch_semaphore_wait(semaphore, dispatch_time(DISPATCH_TIME_NOW, 200 * NSEC_PER_MSEC)) != 0)
    if (stop_) return false;
  return !stop_;
}

WeightStream::Slot *WeightStream::qkfixSlotOf(int64_t n) {
  const int rank = qkfixRank_[static_cast<size_t>(n % blocksPerForward_)];
  if (rank < 0) return nullptr;
  const int64_t q = (n / blocksPerForward_) * qkfixPerForward_ + rank;
  return &qkfixSlots_[static_cast<size_t>(q % static_cast<int64_t>(qkfixSlots_.size()))];
}

void WeightStream::reader() {
  for (int64_t n = 0; !stop_; ++n) {
    const int block = static_cast<int>(n % blocksPerForward_);
    Slot &slot = slots_[static_cast<size_t>(n % static_cast<int64_t>(slots_.size()))];
    if (!waitFree(slot.free)) break;
    Slot *qkfix = qkfixSlotOf(n);
    if (qkfix && !waitFree(qkfix->free)) break;
    const auto t0 = AwakeClock::now();
    int64_t bytes = 0;
    try {
      for (const model::StreamedTensor &t : blockTensors_[block]) {
        if (t.slotTensor < 0 && !qkfix) continue;
        void *slotTensor = t.slotTensor < 0 ? qkfix->tensor[0].contents() : slot.tensor[t.slotTensor].contents();
        model::read(*t.tensor, static_cast<uint8_t *>(slotTensor) + t.slotOffset);
        bytes += static_cast<int64_t>(t.tensor->bytes);
      }
    } catch (...) {
      failure_ = std::current_exception();
      failed_ = true;
      stop_ = true;
    }
    const auto us = static_cast<int64_t>(millisecondsSince(t0) * 1e3);
    readBytes_ += bytes;
    readUs_ += us;
    blocksRead_ += 1;
    if (us > readMaxUs_) readMaxUs_ = us;
    dispatch_semaphore_signal(slot.ready);
  }
}

WeightStream::Slot &WeightStream::next(Slot **qkfix) {
  const int64_t n = encoderSequence_++;
  *qkfix = qkfixSlotOf(n);
  return slots_[static_cast<size_t>(n % static_cast<int64_t>(slots_.size()))];
}

void WeightStream::waitRead(Slot &slot) {
  const auto t0 = AwakeClock::now();
  dispatch_semaphore_wait(slot.ready, DISPATCH_TIME_FOREVER);
  if (failed_) std::rethrow_exception(failure_);
  const double ms = millisecondsSince(t0);
  waitMs_ += ms;
  waitMaxMs_ = std::max(waitMaxMs_, ms);
  if (ms > 1.0) ++waits_;
}

void WeightStream::release(Slot &slot, Slot *qkfix) {
  dispatch_semaphore_signal(slot.free);
  if (qkfix) dispatch_semaphore_signal(qkfix->free);
}

void WeightStream::resetStats() {
  readBytes_ = 0;
  readUs_ = 0;
  readMaxUs_ = 0;
  blocksRead_ = 0;
  waitMs_ = waitMaxMs_ = 0;
  waits_ = 0;
}

StreamStats WeightStream::stats() const {
  StreamStats s;
  s.waitMs = waitMs_;
  s.waitMaxMs = waitMaxMs_;
  s.waits = waits_;
  s.readGb = readBytes_ / 1e9;
  s.readS = readUs_ / 1e6;
  s.readMaxMs = readMaxUs_ / 1e3;
  s.blocksRead = static_cast<int32_t>(blocksRead_.load());
  return s;
}

size_t WeightStream::slotBytes() const {
  size_t total = 0;
  for (const Slot &slot : slots_)
    for (int t = 0; t < kTensors; ++t) total += slot.tensor[t].sizeBytes();
  for (const Slot &slot : qkfixSlots_) total += slot.tensor[0].sizeBytes();
  return total;
}

} // namespace h3
