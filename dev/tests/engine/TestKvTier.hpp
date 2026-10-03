#pragma once

#include "engine/KvTier.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace splash::test {

// A KV disk tier without a disk: slots count against a quota, a bounded
// number of transfers is in flight, and transfers finish when the test says
// so.
class TestKvTier final : public engine::KvTier {
public:
  struct Transfer final {
    bool ready = false;
    bool success = true;
    bool finished = false;
  };

  uint64_t slotBytes() const noexcept override { return 100; }
  bool writable() const noexcept override { return writableFile; }
  bool canDemote() const noexcept override { return writableFile && inFlight() < transferLimit; }
  bool canRestore() const noexcept override { return inFlight() < transferLimit; }
  std::shared_ptr<engine::KvDiskSlot> acquireSlot() override {
    if (slots >= capacity) return {};
    return std::make_shared<Slot>(*this);
  }
  std::unique_ptr<engine::KvTransfer>
  demote(uint32_t, std::shared_ptr<engine::KvDiskSlot>, std::function<void()>) override {
    if (!canDemote()) return {};
    ++demotions;
    return start();
  }
  std::unique_ptr<engine::KvTransfer>
  restore(std::shared_ptr<engine::KvDiskSlot>, uint32_t page, std::function<void()>) override {
    ++restoreCalls;
    if (!canRestore()) return {};
    ++restores;
    restoredPages.push_back(page);
    return start();
  }
  void poll() override {}

  // Finishes every transfer still in flight.
  void complete(bool success = true) {
    for (auto &transfer : transfers) {
      if (transfer->finished || transfer->ready) continue;
      transfer->ready = true;
      transfer->success = success;
    }
  }
  [[nodiscard]] uint32_t inFlight() const noexcept { return inFlight_; }

  uint32_t slots = 0;
  uint32_t capacity = 4;
  uint32_t transferLimit = 2;
  uint32_t demotions = 0;
  // Restores started, every restore() call, and the pages of those started.
  uint32_t restores = 0;
  uint32_t restoreCalls = 0;
  std::vector<uint32_t> restoredPages;
  bool writableFile = true;
  std::vector<std::shared_ptr<Transfer>> transfers;

private:
  struct Slot final : engine::KvDiskSlot {
    explicit Slot(TestKvTier &owner) : tier(owner) { ++tier.slots; }
    ~Slot() override { --tier.slots; }
    TestKvTier &tier;
  };
  class Ticket final : public engine::KvTransfer {
  public:
    Ticket(TestKvTier &owner, std::shared_ptr<Transfer> transfer)
        : tier_(owner), transfer_(std::move(transfer)) {}
    bool ready() const noexcept override { return transfer_->ready; }
    bool finish() override {
      if (!transfer_->finished) {
        transfer_->finished = true;
        --tier_.inFlight_;
      }
      return transfer_->success;
    }

  private:
    TestKvTier &tier_;
    std::shared_ptr<Transfer> transfer_;
  };

  std::unique_ptr<engine::KvTransfer> start() {
    auto transfer = std::make_shared<Transfer>();
    transfers.push_back(transfer);
    ++inFlight_;
    return std::make_unique<Ticket>(*this, std::move(transfer));
  }

  uint32_t inFlight_ = 0;
};

} // namespace splash::test
