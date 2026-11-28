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
#include <stdexcept>
#include <vector>

namespace splash::test {

// A tier copy in memory (model::KvCopy), read on any thread as SlotFile::readAt reads a slot;
// failAll and failNext make reads fail. Set the bytes before reads start.
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

// A KV disk tier without a disk: copies count against a quota of copies (or, in bytes, a DiskBudget:
// 100 for a block's base part, standing in for what SlotFile counts of a part, which kv-page-tier-host
// holds a real tier to, and windowBytes for its window part where the tier has one), transfers finish
// when the test says so, and each copy is its own TestKvCopy that readers may outlive. Each part has its
// bytes, presence and label as the real tier's do. Tests on it check the cache's count of its slots, not
// the slot file's.
class TestKvTier final : public engine::KvTier {
public:
  struct Transfer final {
    bool ready = false;
    bool success = true;
    bool finished = false;
    bool submitted = true;
    // A write's class; none for a restore.
    std::optional<engine::WriteClass> write;
    // When it lands, on a tier with a rate.
    std::optional<double> dueMilliseconds;
  };

  TestKvTier() = default;
  explicit TestKvTier(bool persistent, uint64_t window = 0) : persistentTier(persistent), windowBytes(window) {}

  kv::Components components() const noexcept override {
    return kv::componentBit(kv::Component::Base) | (windowBytes ? kv::componentBit(kv::Component::Window) : 0);
  }
  uint64_t slotBytes(kv::Components parts) const override {
    return (parts & kv::componentBit(kv::Component::Base) ? 100 : 0) +
           (parts & kv::componentBit(kv::Component::Window) ? windowBytes : 0);
  }
  // A whole copy's.
  uint64_t slotBytes() const { return slotBytes(components()); }
  bool writable() const noexcept override { return writableFile; }
  bool persistent() const noexcept override { return persistentTier; }
  void label(const std::shared_ptr<engine::KvDiskSlot> &slot, kv::Components parts,
             std::vector<std::byte> label) override {
    for (const kv::Component part : kv::kAllComponents)
      if (parts & kv::componentBit(part)) static_cast<Slot &>(*slot).labels[static_cast<size_t>(part)] = label;
  }
  bool canDemote(engine::WriteClass write) const noexcept override {
    const auto kind = static_cast<size_t>(write);
    return writableFile && inFlight() < transferLimit && writesInFlight_[kind] < writeLimits[kind];
  }
  bool canRestore() const noexcept override { return inFlight() < transferLimit; }
  std::shared_ptr<engine::KvDiskSlot> acquireSlot(kv::Components parts, model::DiskBudget::Draw draw,
                                                  std::shared_ptr<engine::KvDiskSlot> into) override {
    draws.push_back(draw);
    if (into ? into->parts() & parts : slots >= capacity && !budget) return {};
    if (budget && !budget->reserve(slotBytes(parts), draw)) return {};
    std::shared_ptr<Slot> copy = into ? std::static_pointer_cast<Slot>(into) : std::make_shared<Slot>(*this);
    for (const kv::Component part : kv::kAllComponents)
      if (parts & kv::componentBit(part)) copy->hold(part, slotBytes(kv::componentBit(part)));
    return copy;
  }
  std::unique_ptr<engine::KvTransfer>
  demote(std::span<const uint32_t> pages, const std::shared_ptr<engine::KvDiskSlot> &copy, kv::Components parts,
         engine::WriteClass write, std::function<void()>) override {
    if (!canDemote(write)) return {};
    requireParts(*copy, parts, engine::KvDiskSlot::Presence::Writing);
    ++demotions;
    writes.push_back({pages.front(), write, parts});
    blocks.emplace_back(pages.begin(), pages.end());
    return start(write, std::static_pointer_cast<Slot>(copy), parts);
  }
  std::unique_ptr<engine::KvTransfer>
  restore(const std::shared_ptr<engine::KvDiskSlot> &copy, kv::Components parts, std::span<const uint32_t> pages,
          std::function<void()>) override {
    ++restoreCalls;
    if (!canRestore()) return {};
    requireParts(*copy, parts, engine::KvDiskSlot::Presence::Landed);
    ++restores;
    restoredPages.push_back(pages.front());
    restoredParts.push_back(parts);
    blocks.emplace_back(pages.begin(), pages.end());
    std::unique_ptr<engine::KvTransfer> transfer = start(std::nullopt);
    transfers.back()->submitted = false;
    return transfer;
  }
  void giveUp(const std::shared_ptr<engine::KvDiskSlot> &copy, kv::Components parts) override {
    if (!parts || parts & ~copy->parts()) throw std::logic_error("a test tier's copy gave up a part it lacks");
    for (const kv::Component part : kv::kAllComponents)
      if (parts & kv::componentBit(part)) static_cast<Slot &>(*copy).release(part);
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
    for (auto &transfer : open()) {
      if (transfer->finished || transfer->ready || !transfer->submitted) continue;
      transfer->ready = true;
      transfer->success = success;
    }
  }
  // With a rate, a transfer lands a slot's bytes at bytesPerMillisecond
  // after it starts, by the test's clock: the next of those to land, and
  // those due by `nowMilliseconds` landed (a restore once submitted).
  [[nodiscard]] std::optional<double> nextLanding() const {
    std::optional<double> next;
    for (const auto &transfer : open())
      if (!transfer->finished && !transfer->ready && transfer->submitted && transfer->dueMilliseconds)
        next = std::min(next.value_or(*transfer->dueMilliseconds), *transfer->dueMilliseconds);
    return next;
  }
  void land(double nowMilliseconds) {
    for (auto &transfer : open())
      if (!transfer->finished && !transfer->ready && transfer->submitted && transfer->dueMilliseconds &&
          *transfer->dueMilliseconds <= nowMilliseconds)
        transfer->ready = true;
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
  // finds them, and a slot that process takes back, within the tier's
  // budget if it has one.
  [[nodiscard]] std::vector<std::vector<std::byte>> labels(kv::Component part = kv::Component::Base) const {
    std::vector<std::vector<std::byte>> result;
    for (const Slot *slot : live_)
      if (!slot->labels[static_cast<size_t>(part)].empty()) result.push_back(slot->labels[static_cast<size_t>(part)]);
    return result;
  }
  // The labels a crash now leaves: those of the parts that landed, as a slot
  // file writes a part's record once the part has landed.
  [[nodiscard]] std::vector<std::vector<std::byte>> landedLabels(kv::Component part = kv::Component::Base) const {
    std::vector<std::vector<std::byte>> result;
    for (const Slot *slot : live_)
      if (!slot->labels[static_cast<size_t>(part)].empty() &&
          slot->presence(part) == engine::KvDiskSlot::Presence::Landed)
        result.push_back(slot->labels[static_cast<size_t>(part)]);
    return result;
  }
  [[nodiscard]] std::shared_ptr<engine::KvDiskSlot> adopt(kv::Component part = kv::Component::Base,
                                                          std::shared_ptr<engine::KvDiskSlot> into = {}) {
    const uint64_t bytes = slotBytes(kv::componentBit(part));
    if ((into && into->presence(part) != engine::KvDiskSlot::Presence::None) || (budget && !budget->reserve(bytes)))
      throw std::logic_error("a test tier takes back only what its budget holds, a part once");
    std::shared_ptr<Slot> copy = into ? std::static_pointer_cast<Slot>(into) : std::make_shared<Slot>(*this);
    copy->hold(part, bytes);
    copy->presences[static_cast<size_t>(part)] = engine::KvDiskSlot::Presence::Landed;
    return copy;
  }

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
    kv::Components parts = 0;
  };
  std::vector<Write> writes;
  // Every transfer's block's pages, base page first, in the order asked.
  std::vector<std::vector<uint32_t>> blocks;
  std::vector<model::DiskBudget::Draw> draws;
  // Restores started, every restore() call, and the pages of those started.
  uint32_t restores = 0;
  uint32_t restoreCalls = 0;
  std::vector<uint32_t> restoredPages;
  std::vector<kv::Components> restoredParts;
  bool writableFile = true;
  bool persistentTier = false;
  // A window part's bytes; none for a tier of one component.
  uint64_t windowBytes = 0;
  std::vector<std::shared_ptr<Transfer>> transfers;
  std::optional<double> bytesPerMillisecond;
  std::function<double()> clock;

private:
  struct Slot final : engine::KvDiskSlot, TestKvCopy {
    explicit Slot(TestKvTier &owner) : tier(owner), budget(owner.budget) {
      ++tier.slots;
      tier.live_.insert(this);
    }
    ~Slot() override {
      --tier.slots;
      tier.live_.erase(this);
      if (budget) budget->release(bytes());
    }
    uint64_t bytes() const noexcept override { return taken[0] + taken[1]; }
    uint64_t bytes(kv::Component part) const noexcept override { return taken[static_cast<size_t>(part)]; }
    Presence presence(kv::Component part) const noexcept override { return presences[static_cast<size_t>(part)]; }
    void hold(kv::Component part, uint64_t bytes) {
      taken[static_cast<size_t>(part)] = bytes;
      presences[static_cast<size_t>(part)] = Presence::Writing;
    }
    void release(kv::Component part) {
      if (budget) budget->release(taken[static_cast<size_t>(part)]);
      taken[static_cast<size_t>(part)] = 0;
      presences[static_cast<size_t>(part)] = Presence::None;
      labels[static_cast<size_t>(part)].clear();
    }
    TestKvTier &tier;
    std::shared_ptr<model::DiskBudget> budget;
    std::array<std::vector<std::byte>, kv::kComponents> labels;
    std::array<uint64_t, kv::kComponents> taken{};
    std::array<Presence, kv::kComponents> presences{};
  };
  static void requireParts(const engine::KvDiskSlot &copy, kv::Components parts, engine::KvDiskSlot::Presence presence) {
    for (const kv::Component part : kv::kAllComponents)
      if (parts & kv::componentBit(part) && copy.presence(part) != presence)
        throw std::logic_error("a test tier's copy has not the parts a call needs");
  }
  class Ticket final : public engine::KvTransfer {
  public:
    Ticket(TestKvTier &owner, std::shared_ptr<Transfer> transfer, std::optional<engine::WriteClass> write,
           std::shared_ptr<Slot> copy, kv::Components parts)
        : tier_(owner), transfer_(std::move(transfer)), write_(write), copy_(std::move(copy)), parts_(parts) {}
    bool ready() const noexcept override { return transfer_->ready; }
    bool finish() override {
      if (!transfer_->finished) {
        transfer_->finished = true;
        --tier_.inFlight_;
        if (write_) --tier_.writesInFlight_[static_cast<size_t>(*write_)];
        // A written part lands with its write.
        for (const kv::Component part : kv::kAllComponents)
          if (copy_ && transfer_->success && parts_ & kv::componentBit(part) && copy_->bytes(part))
            copy_->presences[static_cast<size_t>(part)] = Presence::Landed;
      }
      return transfer_->success;
    }

  private:
    using Presence = engine::KvDiskSlot::Presence;
    TestKvTier &tier_;
    std::shared_ptr<Transfer> transfer_;
    std::optional<engine::WriteClass> write_;
    std::shared_ptr<Slot> copy_;
    kv::Components parts_;
  };

  // A write of its class, of the copy's parts, or a restore.
  std::unique_ptr<engine::KvTransfer> start(std::optional<engine::WriteClass> write,
                                            std::shared_ptr<Slot> copy = {}, kv::Components parts = 0) {
    auto transfer = std::make_shared<Transfer>();
    transfer->write = write;
    if (bytesPerMillisecond)
      transfer->dueMilliseconds = clock() + double(slotBytes()) / *bytesPerMillisecond;
    transfers.push_back(transfer);
    ++inFlight_;
    if (write) ++writesInFlight_[static_cast<size_t>(*write)];
    return std::make_unique<Ticket>(*this, std::move(transfer), write, std::move(copy), parts);
  }

  // The transfers from the first one not finished on: every one before it
  // is, so a long run's scans skip them.
  [[nodiscard]] std::span<const std::shared_ptr<Transfer>> open() const {
    if (open_ > transfers.size())
      open_ = 0;
    while (open_ < transfers.size() && transfers[open_]->finished)
      ++open_;
    return std::span(transfers).subspan(open_);
  }

  uint32_t inFlight_ = 0;
  std::array<uint32_t, 3> writesInFlight_{};
  std::set<const Slot *> live_;
  mutable size_t open_ = 0;
};

// A fixture's tier: its quota of slots, its transfers in flight, and with a
// rate, a transfer lands bytes / rate after it starts. A persistent tier
// copies points ahead whatever memory holds.
struct TierOptions final {
  uint32_t capacity = 4, transferLimit = 2;
  std::optional<double> bytesPerMillisecond;
  bool persistent = false;
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
