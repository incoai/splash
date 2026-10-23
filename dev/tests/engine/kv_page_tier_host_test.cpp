// The KV tier over pages in host memory (engine::BasicKvPageTier), for the
// sanitizers: pages leave for their slots on the file's worker and come back
// on the SSD readers, index restores in a batch of their own beside the
// whole ones, while the engine thread polls; a restore of a slot never
// written fails alone; and a tier goes away with its transfers in flight,
// and right after its last batch retired, whose end ran on a reader, with
// nothing it started running past it. kv_page_tier_test runs the tier over
// the pool's extents beside the GPU. Built under ASan and TSan too.
#include "HostKvPages.hpp"
#include "TestChecks.hpp"
#include "engine/KvPageTier.hpp"
#include "model/SlotFile.hpp"
#include "model/SsdReaders.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace splash;
using engine::KvTransfer;
using engine::WriteClass;
using model::DiskBudget;
using model::SlotFile;
using splash::test::require;

namespace {

using Tier = engine::BasicKvPageTier<test::HostKvPages>;

constexpr auto cached = DiskBudget::Draw::Cache;

// A slot file of `slots` slots of the format's payload, checksummed by its
// segments.
std::shared_ptr<SlotFile> slotFile(const kv::SlotFormat &format, uint32_t slots,
                                   std::shared_ptr<DiskBudget> &budget) {
  const uint64_t bytes = SlotFile::slotBytesFor(format.payloadBytes());
  budget = std::make_shared<DiskBudget>(slots * bytes);
  return std::make_shared<SlotFile>(bytes, budget, SlotFile::Options{.segments = format.segments()});
}

// Drives the tier the way the engine does until every transfer is ready:
// restores asked for go out together, and polls retire what the worker and
// the readers did.
void runUntilReady(Tier &tier, std::span<const std::unique_ptr<KvTransfer>> transfers) {
  tier.submitRestores();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  for (;;) {
    tier.poll();
    bool ready = true;
    for (const std::unique_ptr<KvTransfer> &transfer : transfers)
      ready = ready && transfer->ready();
    if (ready)
      return;
    if (std::chrono::steady_clock::now() > deadline)
      throw std::runtime_error("KV transfers did not complete");
    std::this_thread::yield();
  }
}

std::vector<std::byte> indexOf(const test::HostKvPages &pages, uint32_t page) {
  std::vector<std::byte> bytes;
  for (const std::span<std::byte> span : pages.indexSpans(page))
    bytes.insert(bytes.end(), span.begin(), span.end());
  return bytes;
}

// Pages leave for their slots and come back byte for byte, all at once: a
// whole restore brings a page back, of a windowed layout every third one
// Slim, which its slot holds and a restore into a Slim page brings back,
// and, of a spillable layout, an index restore every other page's index
// alone. Each kind's batch wakes the engine once. A restore of a slot never
// written fails, the others beside it land.
void roundTrips(const std::shared_ptr<model::SsdReaders> &readers, kv::Layout layout) {
  constexpr uint32_t kPages = 48;
  test::HostKvPages pages(layout, 2 * kPages + 1);
  const kv::SlotFormat format(layout);
  std::shared_ptr<DiskBudget> budget;
  Tier tier(pages, format, slotFile(format, kPages + 1, budget), readers);
  std::vector<std::shared_ptr<engine::KvDiskSlot>> slots;
  std::vector<std::unique_ptr<KvTransfer>> writes;
  const auto slim = [&](uint32_t page) { return layout.slimmable() && page % 3 == 0; };
  for (uint32_t page = 0; page < kPages; ++page) {
    pages.fill(page, page + 1);
    if (slim(page)) {
      pages.setForm(page, kv::PageForm::Slim);
      pages.setForm(kPages + page, kv::PageForm::Slim);
    }
    slots.push_back(tier.acquireSlot(cached));
    writes.push_back(tier.demote(page, slots.back(), WriteClass::Reclaim, {}));
    require(writes.back() != nullptr, "a write within the tier's bound was refused");
  }
  runUntilReady(tier, writes);
  for (const std::unique_ptr<KvTransfer> &write : writes)
    require(write->finish(), "a write failed");

  std::atomic<uint32_t> wakes{0};
  std::vector<std::unique_ptr<KvTransfer>> restores;
  for (uint32_t page = 0; page < kPages; ++page) {
    if (layout.spillable() && page % 2)
      pages.setForm(kPages + page, kv::PageForm::Index);
    restores.push_back(tier.restore(slots[page], kPages + page, [&wakes] { wakes.fetch_add(1); }));
  }
  restores.push_back(tier.restore(tier.acquireSlot(cached), 2 * kPages, {}));
  runUntilReady(tier, restores);
  for (uint32_t page = 0; page < kPages; ++page) {
    const uint32_t into = kPages + page;
    const bool index = pages.form(into) == kv::PageForm::Index;
    const uint64_t bytes = layout.bytesPerModelPage(pages.form(into));
    require(restores[page]->finish() &&
                (index ? indexOf(pages, into) == indexOf(pages, page)
                       : std::ranges::equal(pages.page(into).first(bytes), pages.page(page).first(bytes))),
            "a page did not come back from its slot byte for byte");
  }
  require(!restores.back()->finish(), "a slot never written was restored");
  require(wakes.load() == (layout.spillable() ? 2u : 1u), "a batch of restores did not wake the engine once");
  std::cout << "round trips " << kv::formatName(layout.format) << " slot format v" << format.version() << '\n';
}

// A tier goes away with writes and restores in flight: the worker and the
// readers are done with its pages first. And a tier that goes right after
// its last restore retired leaves no batch end running on a reader, so the
// restore's completion never runs past it.
void teardown(const std::shared_ptr<model::SsdReaders> &readers, kv::Layout layout) {
  test::HostKvPages pages(layout, 16);
  const kv::SlotFormat format(layout);
  std::shared_ptr<DiskBudget> budget;
  const std::shared_ptr<SlotFile> file = slotFile(format, 16, budget);
  {
    Tier tier(pages, format, file, readers, 16);
    std::vector<std::unique_ptr<KvTransfer>> transfers;
    std::vector<std::shared_ptr<engine::KvDiskSlot>> slots;
    for (uint32_t page = 0; page < 8; ++page) {
      slots.push_back(tier.acquireSlot(cached));
      transfers.push_back(tier.demote(page, slots.back(), WriteClass::Reclaim, {}));
      require(transfers.back() != nullptr, "a write within the tier's bound was refused");
    }
    for (uint32_t page = 8; page < 16; ++page)
      transfers.push_back(tier.restore(slots[page - 8], page, {}));
    tier.submitRestores();
  }
  require(budget->usedBytes() == 0, "the tier left its pages with disk IO still running");
  for (int round = 0; round < 64; ++round) {
    auto wakes = std::make_unique<std::atomic<uint32_t>>(0);
    {
      Tier tier(pages, format, file, readers);
      const std::shared_ptr<engine::KvDiskSlot> slot = tier.acquireSlot(cached);
      std::vector<std::unique_ptr<KvTransfer>> transfers;
      transfers.push_back(tier.demote(0, slot, WriteClass::Reclaim, {}));
      runUntilReady(tier, transfers);
      // The completion takes a while, as a wake that waits for a lock does.
      transfers.push_back(tier.restore(slot, 1, [counter = wakes.get()] {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        counter->fetch_add(1);
      }));
      runUntilReady(tier, transfers);
    }
    require(wakes->load() == 1, "a batch's end ran past the tier that started it");
  }
  std::cout << "teardown " << kv::formatName(layout.format) << '\n';
}

} // namespace

int main() {
  try {
    const auto readers = std::make_shared<model::SsdReaders>(model::kSsdReaders);
    for (const kv::Format format : {kv::Format::Int8, kv::Format::BFloat16}) {
      // Qwen3.6-35B's layout, which keeps its spans' order, a windowed one
      // with Slim pages, and Qwen3.8-Flash-Next's, whose slots hold block
      // records.
      for (const kv::Layout layout : {kv::Layout{10, 2, 256, format},
                                      kv::Layout{6, 2, 256, format, 0, false, 512, 0b010110},
                                      kv::Layout{12, 2, 256, format, 128, true}}) {
        roundTrips(readers, layout);
        teardown(readers, layout);
      }
    }
  } catch (const std::exception &error) {
    std::cerr << "kv page tier host tests failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "kv page tier host tests passed\n";
  return 0;
}
