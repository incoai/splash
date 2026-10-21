#pragma once

#include "engine/KvTier.hpp"
#include "model/Model.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <vector>

namespace splash::test {

// A tier copy in memory (model::KvCopy), read on any thread as
// model::SlotFile::readAt reads a slot: the 4 KiB-aligned range around the
// bytes, cut at the copy's end, lands at the start of `buffer`, which must
// hold it, and the bytes are returned within it. A read returns nothing
// while failAll is set, the hook for a copy that cannot be read, and for the
// next failNext reads, the hook for a failure a retry may or may not meet.
// Set the bytes before reads start.
class TestKvCopy : public model::KvCopy {
public:
  static constexpr uint64_t kReadAlignment = 4096;

  TestKvCopy() : id_(++ids()) {}
  explicit TestKvCopy(std::vector<std::byte> content) : TestKvCopy() { bytes = std::move(content); }

  uint64_t id() const noexcept override { return id_; }
  std::optional<std::span<const std::byte>> readAt(uint64_t offset, uint64_t count,
                                                   std::span<std::byte> buffer) const noexcept override {
    reads_.fetch_add(1, std::memory_order_relaxed);
    if (failReads()) return std::nullopt;
    const uint64_t first = offset / kReadAlignment * kReadAlignment;
    const uint64_t end = std::min<uint64_t>((offset + count + kReadAlignment - 1) / kReadAlignment * kReadAlignment,
                                            bytes.size());
    if (offset + count > bytes.size() || end - first > buffer.size()) return std::nullopt;
    std::memcpy(buffer.data(), bytes.data() + first, end - first);
    return std::span<const std::byte>(buffer.data() + (offset - first), count);
  }
  [[nodiscard]] uint64_t reads() const noexcept { return reads_.load(std::memory_order_relaxed); }

  std::vector<std::byte> bytes;
  std::atomic<bool> failAll{false};
  mutable std::atomic<uint32_t> failNext{0};

private:
  static std::atomic<uint64_t> &ids() noexcept {
    static std::atomic<uint64_t> next{0};
    return next;
  }
  [[nodiscard]] bool failReads() const noexcept {
    if (failAll.load(std::memory_order_relaxed)) return true;
    uint32_t left = failNext.load(std::memory_order_relaxed);
    while (left && !failNext.compare_exchange_weak(left, left - 1, std::memory_order_relaxed)) {
    }
    return left != 0;
  }

  uint64_t id_;
  mutable std::atomic<uint64_t> reads_{0};
};

// A KV disk tier without a disk: slots count against a quota, a bounded
// number of transfers is in flight, writes of each class within their own
// bound, and transfers finish when the test says so, a restore only once
// submitted (submitRestores), as the tier's readers take it then. With a
// budget, a slot is reserved from it as its draw says
// (model::DiskBudget::reserve) instead of against the quota. Each slot is
// its own copy (TestKvCopy, copyOf), which reads of spilled KV hold as they
// hold a real slot: a copy outlives the cache's reference only as long as a
// reader keeps it, and its quota with it.
class TestKvTier final : public engine::KvTier {
public:
  struct Transfer final {
    bool ready = false;
    bool success = true;
    bool finished = false;
    bool submitted = true;
    // A write's class; none for a restore.
    std::optional<engine::WriteClass> write;
  };

  TestKvTier() = default;
  explicit TestKvTier(bool persistent) : persistentTier(persistent) {}

  uint64_t slotBytes() const noexcept override { return 100; }
  bool writable() const noexcept override { return writableFile; }
  bool persistent() const noexcept override { return persistentTier; }
  void label(const std::shared_ptr<engine::KvDiskSlot> &slot, std::vector<std::byte> label) override {
    static_cast<Slot &>(*slot).label = std::move(label);
  }
  bool canDemote(engine::WriteClass write) const noexcept override {
    const auto kind = static_cast<size_t>(write);
    return writableFile && inFlight() < transferLimit && writesInFlight_[kind] < writeLimits[kind];
  }
  bool canRestore() const noexcept override { return inFlight() < transferLimit; }
  std::shared_ptr<engine::KvDiskSlot> acquireSlot(model::DiskBudget::Draw draw) override {
    draws.push_back(draw);
    if (budget ? !budget->reserve(slotBytes(), draw) : slots >= capacity) return {};
    return std::make_shared<Slot>(*this);
  }
  std::unique_ptr<engine::KvTransfer>
  demote(uint32_t page, std::shared_ptr<engine::KvDiskSlot>, engine::WriteClass write,
         std::function<void()>) override {
    if (!canDemote(write)) return {};
    ++demotions;
    writes.push_back({page, write});
    return start(write);
  }
  std::unique_ptr<engine::KvTransfer>
  restore(std::shared_ptr<engine::KvDiskSlot>, uint32_t page, std::function<void()>) override {
    ++restoreCalls;
    if (!canRestore()) return {};
    ++restores;
    restoredPages.push_back(page);
    std::unique_ptr<engine::KvTransfer> transfer = start(std::nullopt);
    transfers.back()->submitted = false;
    return transfer;
  }
  void submitRestores() override {
    for (const auto &transfer : transfers)
      transfer->submitted = true;
  }
  std::shared_ptr<const model::KvCopy> copy(const std::shared_ptr<engine::KvDiskSlot> &slot) const override {
    return std::static_pointer_cast<const Slot>(slot);
  }
  void poll() override {}

  // A slot's copy, to give it bytes or make its reads fail.
  [[nodiscard]] static TestKvCopy &copyOf(const std::shared_ptr<engine::KvDiskSlot> &slot) {
    return static_cast<Slot &>(*slot);
  }

  // Finishes every transfer still in flight, of the restores those submitted.
  void complete(bool success = true) {
    for (auto &transfer : transfers) {
      if (transfer->finished || transfer->ready || !transfer->submitted) continue;
      transfer->ready = true;
      transfer->success = success;
    }
  }
  // Fails every write of the class still in flight.
  void fail(engine::WriteClass write) {
    for (auto &transfer : transfers) {
      if (transfer->finished || transfer->ready || transfer->write != write) continue;
      transfer->ready = true;
      transfer->success = false;
    }
  }
  [[nodiscard]] uint32_t inFlight() const noexcept { return inFlight_; }
  // The labels of the slots a persistent tier holds, as the next process
  // finds them, and a slot that process takes back.
  [[nodiscard]] std::vector<std::vector<std::byte>> labels() const {
    std::vector<std::vector<std::byte>> result;
    for (const Slot *slot : live_)
      if (!slot->label.empty()) result.push_back(slot->label);
    return result;
  }
  [[nodiscard]] std::shared_ptr<engine::KvDiskSlot> adopt() { return std::make_shared<Slot>(*this); }

  uint32_t slots = 0;
  uint32_t capacity = 4;
  // Reserves the slots instead of capacity, as SlotFile does.
  std::shared_ptr<model::DiskBudget> budget;
  uint32_t transferLimit = 2;
  // Writes in flight of each class at most (engine::WriteClass): by
  // default a production tier's shares (KvPageTier), 16 persistent copies
  // and 32 pages written through.
  std::array<uint32_t, 3> writeLimits{~0u, 16, 32};
  uint32_t demotions = 0;
  // Every write started, its page and class, and every draw a slot was
  // asked for.
  struct Write final {
    uint32_t page = 0;
    engine::WriteClass write = engine::WriteClass::Reclaim;
  };
  std::vector<Write> writes;
  std::vector<model::DiskBudget::Draw> draws;
  // Restores started, every restore() call, and the pages of those started.
  uint32_t restores = 0;
  uint32_t restoreCalls = 0;
  std::vector<uint32_t> restoredPages;
  bool writableFile = true;
  bool persistentTier = false;
  std::vector<std::shared_ptr<Transfer>> transfers;

private:
  struct Slot final : engine::KvDiskSlot, TestKvCopy {
    explicit Slot(TestKvTier &owner) : tier(owner), budget(owner.budget) {
      ++tier.slots;
      tier.live_.insert(this);
    }
    ~Slot() override {
      --tier.slots;
      tier.live_.erase(this);
      if (budget) budget->release(tier.slotBytes());
    }
    TestKvTier &tier;
    std::shared_ptr<model::DiskBudget> budget;
    std::vector<std::byte> label;
  };
  class Ticket final : public engine::KvTransfer {
  public:
    Ticket(TestKvTier &owner, std::shared_ptr<Transfer> transfer,
           std::optional<engine::WriteClass> write)
        : tier_(owner), transfer_(std::move(transfer)), write_(write) {}
    bool ready() const noexcept override { return transfer_->ready; }
    bool finish() override {
      if (!transfer_->finished) {
        transfer_->finished = true;
        --tier_.inFlight_;
        if (write_) --tier_.writesInFlight_[static_cast<size_t>(*write_)];
      }
      return transfer_->success;
    }

  private:
    TestKvTier &tier_;
    std::shared_ptr<Transfer> transfer_;
    std::optional<engine::WriteClass> write_;
  };

  // A write of its class, or a restore.
  std::unique_ptr<engine::KvTransfer> start(std::optional<engine::WriteClass> write) {
    auto transfer = std::make_shared<Transfer>();
    transfer->write = write;
    transfers.push_back(transfer);
    ++inFlight_;
    if (write) ++writesInFlight_[static_cast<size_t>(*write)];
    return std::make_unique<Ticket>(*this, std::move(transfer), write);
  }

  uint32_t inFlight_ = 0;
  std::array<uint32_t, 3> writesInFlight_{};
  std::set<const Slot *> live_;
};

// A request's page sources (model::KvPageSources), page by page.
inline std::shared_ptr<const model::KvPageSources> sourcesOf(const std::vector<model::KvPageSource> &pages) {
  std::vector<std::shared_ptr<const model::KvPageSources::Chunk>> chunks;
  for (size_t first = 0; first < pages.size(); first += model::KvPageSources::kChunkPages) {
    auto chunk = std::make_shared<model::KvPageSources::Chunk>();
    for (size_t page = first; page < std::min(pages.size(), first + model::KvPageSources::kChunkPages); ++page) {
      chunk->pages[page - first] = pages[page];
      chunk->indexPages += pages[page].indexed();
    }
    chunks.push_back(std::move(chunk));
  }
  return std::make_shared<const model::KvPageSources>(std::move(chunks), static_cast<uint32_t>(pages.size()));
}

} // namespace splash::test
