#pragma once

#include "model/Model.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace splash::test {

// A KV disk tier without a disk: slots count against a quota, every
// transfer holds one staging slot until it finishes, and transfers finish
// when the test says so.
class TestKvTier final : public model::KvTier {
public:
  struct Transfer final {
    bool demotion = false;
    uint32_t page = 0;
    bool ready = false;
    bool success = true;
    bool finished = false;
  };

  uint64_t slotBytes() const noexcept override { return 100; }
  uint64_t capacityBytes() const noexcept override { return uint64_t{capacity} * 100; }
  uint64_t usedBytes() const noexcept override { return uint64_t{slots} * 100; }
  bool writable() const noexcept override { return writableFile; }
  std::shared_ptr<model::KvDiskSlot> acquireSlot() override {
    if (slots >= capacity) return {};
    return std::make_shared<Slot>(*this);
  }
  std::unique_ptr<model::KvTransfer>
  demote(uint32_t page, std::shared_ptr<model::KvDiskSlot>, std::function<void()>) override {
    if (!writableFile || staging >= stagingSlots) return {};
    ++demotions;
    return start(true, page);
  }
  std::unique_ptr<model::KvTransfer>
  restore(std::shared_ptr<model::KvDiskSlot>, uint32_t page, std::function<void()>) override {
    if (staging >= stagingSlots) return {};
    ++restores;
    return start(false, page);
  }
  bool copiesQueued() const noexcept override { return queued; }
  void poll() override {}

  // Finishes every transfer still in flight.
  void complete(bool success = true) {
    for (auto &transfer : transfers) {
      if (transfer->finished || transfer->ready) continue;
      transfer->ready = true;
      transfer->success = success;
    }
    queued = false;
  }
  [[nodiscard]] uint32_t inFlight() const noexcept {
    uint32_t count = 0;
    for (const auto &transfer : transfers) count += !transfer->finished;
    return count;
  }

  uint32_t slots = 0;
  uint32_t capacity = 4;
  uint32_t staging = 0;
  uint32_t stagingSlots = 2;
  uint32_t demotions = 0;
  uint32_t restores = 0;
  bool writableFile = true;
  bool queued = false;
  std::vector<std::shared_ptr<Transfer>> transfers;

private:
  struct Slot final : model::KvDiskSlot {
    explicit Slot(TestKvTier &owner) : tier(owner) { ++tier.slots; }
    ~Slot() override { --tier.slots; }
    TestKvTier &tier;
  };
  class Ticket final : public model::KvTransfer {
  public:
    Ticket(TestKvTier &owner, std::shared_ptr<Transfer> transfer)
        : tier_(owner), transfer_(std::move(transfer)) {}
    bool ready() const noexcept override { return transfer_->ready; }
    bool finish() override {
      if (!transfer_->finished) {
        transfer_->finished = true;
        --tier_.staging;
      }
      return transfer_->success;
    }

  private:
    TestKvTier &tier_;
    std::shared_ptr<Transfer> transfer_;
  };

  std::unique_ptr<model::KvTransfer> start(bool demotion, uint32_t page) {
    auto transfer = std::make_shared<Transfer>();
    transfer->demotion = demotion;
    transfer->page = page;
    transfers.push_back(transfer);
    ++staging;
    queued = true;
    return std::make_unique<Ticket>(*this, std::move(transfer));
  }
};

} // namespace splash::test
