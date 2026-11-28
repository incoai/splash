// The KV tier over host-memory pages (engine::BasicKvPageTier) for the sanitizers: round trips,
// failed restores, teardown with transfers in flight, and a windowed block's copy as two parts in two
// files, each with its own slot, quota, write budget, presence and record. kv_page_tier_test covers the
// GPU side.
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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

using namespace splash;
using engine::KvTransfer;
using engine::WriteClass;
using model::DiskBudget;
using model::SlotFile;
using splash::test::require;

namespace {

using Tier = engine::BasicKvPageTier<test::HostKvPages>;

constexpr auto cached = DiskBudget::Draw::Cache;

// A slot file for each of the format's components, of `slots` slots each of their part's payload,
// checksummed by its segments, sharing a budget of `slots` whole copies.
std::vector<std::shared_ptr<SlotFile>> slotFiles(const kv::SlotFormat &format, uint32_t slots,
                                                 std::shared_ptr<DiskBudget> &budget) {
  uint64_t copyBytes = 0;
  for (const kv::Component part : kv::kAllComponents)
    if (format.layout().components() & kv::componentBit(part))
      copyBytes += SlotFile::slotBytesFor(format.payloadBytes(part));
  budget = std::make_shared<DiskBudget>(slots * copyBytes);
  std::vector<std::shared_ptr<SlotFile>> files;
  for (const kv::Component part : kv::kAllComponents)
    if (format.layout().components() & kv::componentBit(part))
      files.push_back(std::make_shared<SlotFile>(SlotFile::slotBytesFor(format.payloadBytes(part)), budget,
                                                 SlotFile::Options{.segments = format.segments(part)}));
  return files;
}

// Drives the tier as the engine does until every transfer is ready.
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

// Block b's pages, base page first: from b x kv::Layout::blockPages on.
std::vector<uint32_t> blockOf(const kv::Layout &layout, uint32_t block) {
  std::vector<uint32_t> pages(layout.blockPages());
  std::iota(pages.begin(), pages.end(), block * layout.blockPages());
  return pages;
}

// What block b holds of its K/V and index, in its slot's order.
std::vector<std::byte> contentOf(const test::HostKvPages &pages, uint32_t block) {
  std::vector<std::byte> bytes;
  for (const std::span<std::byte> span : pages.blockSpans(blockOf(pages.layout(), block)))
    bytes.insert(bytes.end(), span.begin(), span.end());
  return bytes;
}

void fillBlock(test::HostKvPages &pages, uint32_t block, uint32_t seed) {
  for (const uint32_t page : blockOf(pages.layout(), block))
    pages.fill(page, seed + 7919 * page);
}

std::vector<std::byte> indexOf(const test::HostKvPages &pages, uint32_t page) {
  std::vector<std::byte> bytes;
  for (const std::span<std::byte> span : pages.indexSpans(page))
    bytes.insert(bytes.end(), span.begin(), span.end());
  return bytes;
}

// Blocks round-trip byte for byte, a windowed layout's from all their pages (whole and index restores);
// a restore of an unwritten slot fails alone.
void roundTrips(const std::shared_ptr<model::SsdReaders> &readers, kv::Layout layout) {
  constexpr uint32_t kPages = 48;
  test::HostKvPages pages(layout, (2 * kPages + 1) * layout.blockPages());
  const kv::SlotFormat format(layout);
  std::shared_ptr<DiskBudget> budget;
  Tier tier(pages, format, slotFiles(format, kPages + 1, budget), readers);
  const kv::Components all = layout.components();
  std::vector<std::shared_ptr<engine::KvDiskSlot>> slots;
  std::vector<std::unique_ptr<KvTransfer>> writes;
  for (uint32_t page = 0; page < kPages; ++page) {
    fillBlock(pages, page, page + 1);
    slots.push_back(tier.acquireSlot(all, cached, nullptr));
    writes.push_back(tier.demote(blockOf(layout, page), slots.back(), all, WriteClass::Reclaim, {}));
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
    restores.push_back(
        tier.restore(slots[page], all, blockOf(layout, kPages + page), [&wakes] { wakes.fetch_add(1); }));
  }
  test::rejects([&] { static_cast<void>(tier.restore(tier.acquireSlot(all, cached, nullptr), all, blockOf(layout, 2 * kPages), {})); },
                "not as the tier's call needs", "a copy never written was restored");
  runUntilReady(tier, restores);
  for (uint32_t page = 0; page < kPages; ++page) {
    const uint32_t into = kPages + page;
    const bool index = pages.form(into) == kv::PageForm::Index;
    require(restores[page]->finish() &&
                (index ? indexOf(pages, into) == indexOf(pages, page) : contentOf(pages, into) == contentOf(pages, page)),
            "a block did not come back from its slot byte for byte");
  }
  require(wakes.load() == (layout.spillable() ? 2u : 1u), "a batch of restores did not wake the engine once");
  std::cout << "round trips " << kv::formatName(layout.format) << " slot format v" << format.version() << '\n';
}

// The next process restores a block from the copy an earlier process wrote, each part from its own
// file's record, which carries the label the copy's write gave every part. An adopted slot checks the
// whole segment a read touches, so the restore's buffer holds the slot. A part whose record is lost, as a
// crash between the two parts' records leaves it, leaves the other part's record alone in its file.
void adoptedCopiesRestore(const std::shared_ptr<model::SsdReaders> &readers, kv::Layout layout) {
  std::string name = (std::filesystem::temp_directory_path() / "splash-kv-tier-XXXXXX").string();
  require(::mkdtemp(name.data()) != nullptr, "temporary directory could not be made");
  const std::filesystem::path directory = name;
  const kv::SlotFormat format(layout);
  const kv::Components all = layout.components();
  const auto records = [&](size_t part) { return directory / ("records." + std::to_string(part)); };
  const auto open = [&] {
    const auto budget = std::make_shared<DiskBudget>(uint64_t{1} << 30);
    std::vector<std::shared_ptr<SlotFile>> files;
    for (const kv::Component part : kv::kAllComponents) {
      if (!(all & kv::componentBit(part)))
        continue;
      const size_t index = files.size();
      files.push_back(std::make_shared<SlotFile>(
          SlotFile::slotBytesFor(format.payloadBytes(part)), budget,
          SlotFile::Persistence{directory / ("slots." + std::to_string(index)), records(index), {std::byte{1}}, 8},
          SlotFile::Options{.segments = format.segments(part)}));
    }
    return files;
  };
  test::HostKvPages pages(layout, 2 * layout.blockPages());
  fillBlock(pages, 0, 7);
  {
    const std::vector<std::shared_ptr<SlotFile>> files = open();
    for (const auto &file : files)
      file->finishAdoption();
    Tier tier(pages, format, files, readers);
    const std::shared_ptr<engine::KvDiskSlot> slot = tier.acquireSlot(all, cached, nullptr);
    std::vector<std::unique_ptr<KvTransfer>> write;
    write.push_back(tier.demote(blockOf(layout, 0), slot, all, WriteClass::Reclaim, {}));
    tier.label(slot, all, std::vector<std::byte>(8, std::byte{2}));
    runUntilReady(tier, write);
    require(write.back()->finish() && slot->landed(all), "the block's copy was not written");
    for (const auto &file : files) {
      require(file->synchronize({})->wait(), "a part's file was not synchronized");
      file->seal();
    }
  }
  {
    const std::vector<std::shared_ptr<SlotFile>> files = open();
    Tier tier(pages, format, files, readers);
    std::shared_ptr<engine::KvDiskSlot> slot;
    for (size_t index = 0; index < files.size(); ++index) {
      require(files[index]->records().size() == 1 &&
                  files[index]->records()[0].label == std::vector<std::byte>(8, std::byte{2}),
              "a part's record did not come back with the copy's label");
      slot = tier.adopt(files[index]->records()[0], kv::kAllComponents[index], slot);
    }
    for (const auto &file : files)
      file->finishAdoption();
    require(slot->landed(all), "a copy taken back has a part that did not land");
    std::vector<std::unique_ptr<KvTransfer>> restore;
    restore.push_back(tier.restore(slot, all, blockOf(layout, 1), {}));
    runUntilReady(tier, restore);
    require(restore.back()->finish() && contentOf(pages, 1) == contentOf(pages, 0),
            "the next process did not restore a block from its copy");
    for (const auto &file : files)
      file->seal();
  }
  if (all != kv::componentBit(kv::Component::Base)) {
    std::filesystem::remove(records(1));
    const std::vector<std::shared_ptr<SlotFile>> files = open();
    require(files[0]->records().size() == 1 && files[1]->records().empty(),
            "a part's record depends on the other part's file");
  }
  std::filesystem::remove_all(directory);
  std::cout << "adopted copies restore " << kv::formatName(layout.format) << '\n';
}

// A windowed block's copy is two parts, each a slot of its own file holding its component's layers and
// no padding (5 windowed layers on 3 window pages of 2 slots: one unused), with its own bytes of the
// quota and of the write budget and its own presence: the parts are taken all or none, a write lands
// both, a restore reads the parts it names, and a part given up returns its bytes.
void windowedCopiesAreTwoParts(const std::shared_ptr<model::SsdReaders> &readers, kv::Format kvFormat) {
  const kv::Layout layout{7, 2, 256, kvFormat, 0, false, 512, 0b1110110};
  const kv::SlotFormat format(layout);
  constexpr auto base = kv::componentBit(kv::Component::Base), window = kv::componentBit(kv::Component::Window);
  require(layout.pageSlots() == 2 && layout.windowUnitPages() == 3 && layout.components() == (base | window) &&
              format.payloadBytes(kv::Component::Base) == 2 * layout.bytesPerLayerPage() &&
              format.payloadBytes(kv::Component::Window) == 5 * layout.bytesPerLayerPage(),
          "a windowed block's parts do not hold their layers alone");
  double now = 0;
  const uint64_t baseBytes = SlotFile::slotBytesFor(format.payloadBytes(kv::Component::Base));
  const uint64_t windowBytes = SlotFile::slotBytesFor(format.payloadBytes(kv::Component::Window));
  const auto budget = std::make_shared<DiskBudget>(2 * (baseBytes + windowBytes), [&] { return now; });
  budget->setWriteBudget(uint64_t{1} << 40);
  const std::vector<std::shared_ptr<SlotFile>> files{std::make_shared<SlotFile>(baseBytes, budget),
                                                     std::make_shared<SlotFile>(windowBytes, budget)};
  test::HostKvPages pages(layout, 3 * layout.blockPages());
  fillBlock(pages, 0, 11);
  Tier tier(pages, format, files, readers);
  require(tier.slotBytes(base) == files[0]->quotaBytesFor(format.payloadBytes(kv::Component::Base)) &&
              tier.slotBytes(window) == files[1]->quotaBytesFor(format.payloadBytes(kv::Component::Window)) &&
              tier.slotBytes(base | window) == tier.slotBytes(base) + tier.slotBytes(window),
          "a copy's quota is not its parts'");
  const std::shared_ptr<engine::KvDiskSlot> copy = tier.acquireSlot(base | window, cached, nullptr);
  require(copy && copy->bytes(kv::Component::Base) == tier.slotBytes(base) &&
              copy->bytes(kv::Component::Window) == tier.slotBytes(window) &&
              budget->usedBytes() == copy->bytes() && copy->bytes() == tier.slotBytes(base | window) &&
              copy->presence(kv::Component::Window) == engine::KvDiskSlot::Presence::Writing,
          "a copy's parts did not each take their own quota");
  const int64_t before = budget->writeBudgetBytes();
  std::vector<std::unique_ptr<KvTransfer>> write;
  write.push_back(tier.demote(blockOf(layout, 0), copy, base | window, WriteClass::Reclaim, {}));
  require(before - budget->writeBudgetBytes() == static_cast<int64_t>(copy->bytes()),
          "a copy's write did not charge each part's bytes to the write budget");
  runUntilReady(tier, write);
  require(write.back()->finish() && copy->landed(base | window), "a copy's write did not land both parts");

  // A budget that holds one more base part but no window part takes neither.
  const std::shared_ptr<engine::KvDiskSlot> second = tier.acquireSlot(base | window, cached, nullptr);
  budget->setLimitBytes(budget->usedBytes() + tier.slotBytes(base));
  require(second && !tier.acquireSlot(base | window, cached, nullptr) &&
              budget->usedBytes() == copy->bytes() + second->bytes(),
          "a copy took a part without the other");

  // A restore reads the parts it names: the base alone leaves the window unit as it was.
  std::vector<std::unique_ptr<KvTransfer>> restore;
  restore.push_back(tier.restore(copy, base, blockOf(layout, 1), {}));
  runUntilReady(tier, restore);
  const std::vector<uint32_t> source = blockOf(layout, 0), into = blockOf(layout, 1);
  require(restore.back()->finish() && std::ranges::equal(pages.page(into[0]), pages.page(source[0])) &&
              std::ranges::all_of(pages.page(into[1]), [](std::byte value) { return value == std::byte{0}; }),
          "a base restore did not read the base part alone");
  restore.push_back(tier.restore(copy, base | window, blockOf(layout, 2), {}));
  runUntilReady(tier, restore);
  require(restore.back()->finish() && contentOf(pages, 2) == contentOf(pages, 0),
          "a restore of both parts did not bring the block back");

  // A part given up returns its bytes, and the copy keeps the other.
  tier.giveUp(copy, window);
  require(copy->parts() == base && copy->landed(base) &&
              budget->usedBytes() == tier.slotBytes(base) + second->bytes(),
          "a window part given up did not return exactly its bytes");
  test::rejects([&] { tier.giveUp(copy, window); }, "lacks or writes", "a part was given up twice");
  std::cout << "windowed copies are two parts " << kv::formatName(kvFormat) << '\n';
}

// A part that fails fails its copy, whatever the other part did. The window file meets a file-size limit
// that the base file's write stays within: the write fails although the base part landed, both parts' bytes
// stay counted while the copy is held and return with it, and both were charged to the write budget. The
// window file then takes no writes, so the tier takes none though the base file still does. A restore whose
// window read fails, from a window slot changed on the drive after its record, fails though its base read
// landed.
void aPartThatFailsFailsItsCopy(const std::shared_ptr<model::SsdReaders> &readers, kv::Format kvFormat) {
  const kv::Layout layout{7, 2, 256, kvFormat, 0, false, 512, 0b1110110};
  const kv::SlotFormat format(layout);
  constexpr auto all = kv::componentBit(kv::Component::Base) | kv::componentBit(kv::Component::Window);
  const uint64_t baseBytes = SlotFile::slotBytesFor(format.payloadBytes(kv::Component::Base));
  const uint64_t windowBytes = SlotFile::slotBytesFor(format.payloadBytes(kv::Component::Window));
  test::HostKvPages pages(layout, 3 * layout.blockPages());
  fillBlock(pages, 0, 5);
  {
    double now = 0;
    const auto budget = std::make_shared<DiskBudget>(4 * (baseBytes + windowBytes), [&] { return now; });
    budget->setWriteBudget(uint64_t{1} << 40);
    const std::vector<std::shared_ptr<SlotFile>> files{std::make_shared<SlotFile>(baseBytes, budget),
                                                       std::make_shared<SlotFile>(windowBytes, budget)};
    Tier tier(pages, format, files, readers);
    std::vector<std::unique_ptr<KvTransfer>> writes;
    const std::shared_ptr<engine::KvDiskSlot> first = tier.acquireSlot(all, cached, nullptr);
    writes.push_back(tier.demote(blockOf(layout, 0), first, all, WriteClass::Reclaim, {}));
    runUntilReady(tier, writes);
    require(writes.back()->finish(), "the first copy was not written");
    // Each file is one slot long: the next base part ends at two base slots, the window part past them.
    rlimit original{};
    require(::getrlimit(RLIMIT_FSIZE, &original) == 0, "file limit unavailable");
    rlimit limited = original;
    limited.rlim_cur = 2 * baseBytes;
    std::shared_ptr<engine::KvDiskSlot> copy = tier.acquireSlot(all, cached, nullptr);
    const int64_t before = budget->writeBudgetBytes();
    require(::setrlimit(RLIMIT_FSIZE, &limited) == 0, "file limit could not be set");
    writes.push_back(tier.demote(blockOf(layout, 0), copy, all, WriteClass::Reclaim, {}));
    runUntilReady(tier, writes);
    require(::setrlimit(RLIMIT_FSIZE, &original) == 0, "file limit restore failed");
    require(!writes.back()->finish() && copy->landed(kv::componentBit(kv::Component::Base)) &&
                copy->presence(kv::Component::Window) == engine::KvDiskSlot::Presence::Writing &&
                before - budget->writeBudgetBytes() == static_cast<int64_t>(copy->bytes()) &&
                budget->usedBytes() == first->bytes() + copy->bytes(),
            "a write whose window part failed did not fail, or did not count both parts");
    copy.reset();
    require(budget->usedBytes() == first->bytes(), "a failed copy's parts did not return their bytes");
    require(files[0]->writable() && !files[1]->writable() && !tier.writable() &&
                !tier.canDemote(WriteClass::Reclaim) &&
                !tier.demote(blockOf(layout, 0), tier.acquireSlot(all, cached, nullptr), all, WriteClass::Reclaim, {}),
            "a tier whose window file stopped taking writes took one");
  }
  std::string name = (std::filesystem::temp_directory_path() / "splash-kv-tier-XXXXXX").string();
  require(::mkdtemp(name.data()) != nullptr, "temporary directory could not be made");
  const std::filesystem::path directory = name;
  const auto open = [&] {
    const auto budget = std::make_shared<DiskBudget>(uint64_t{1} << 30);
    std::vector<std::shared_ptr<SlotFile>> files;
    for (const kv::Component part : kv::kAllComponents) {
      const std::string index = std::to_string(static_cast<int>(part));
      files.push_back(std::make_shared<SlotFile>(
          SlotFile::slotBytesFor(format.payloadBytes(part)), budget,
          SlotFile::Persistence{directory / ("slots." + index), directory / ("records." + index), {std::byte{1}}, 8},
          SlotFile::Options{.segments = format.segments(part)}));
    }
    return files;
  };
  {
    const std::vector<std::shared_ptr<SlotFile>> files = open();
    for (const auto &file : files)
      file->finishAdoption();
    Tier tier(pages, format, files, readers);
    const std::shared_ptr<engine::KvDiskSlot> copy = tier.acquireSlot(all, cached, nullptr);
    std::vector<std::unique_ptr<KvTransfer>> write;
    write.push_back(tier.demote(blockOf(layout, 0), copy, all, WriteClass::Reclaim, {}));
    tier.label(copy, all, std::vector<std::byte>(8, std::byte{3}));
    runUntilReady(tier, write);
    require(write.back()->finish(), "the copy was not written");
    for (const auto &file : files) {
      require(file->synchronize({})->wait(), "a part's file was not synchronized");
      file->seal();
    }
  }
  {
    std::fstream window(directory / "slots.1", std::ios::in | std::ios::out | std::ios::binary);
    window.seekp(100);
    window.put(static_cast<char>(0x5a));
  }
  {
    const std::vector<std::shared_ptr<SlotFile>> files = open();
    Tier tier(pages, format, files, readers);
    std::shared_ptr<engine::KvDiskSlot> copy;
    for (size_t index = 0; index < files.size(); ++index)
      copy = tier.adopt(files[index]->records().at(0), kv::kAllComponents[index], copy);
    for (const auto &file : files)
      file->finishAdoption();
    std::vector<std::unique_ptr<KvTransfer>> restore;
    restore.push_back(tier.restore(copy, kv::componentBit(kv::Component::Base), blockOf(layout, 1), {}));
    restore.push_back(tier.restore(copy, all, blockOf(layout, 2), {}));
    runUntilReady(tier, restore);
    require(restore[0]->finish() && !restore[1]->finish(),
            "a restore whose window read failed did not fail alone, or the base part did not restore");
  }
  std::filesystem::remove_all(directory);
  std::cout << "a part that fails fails its copy " << kv::formatName(kvFormat) << '\n';
}

// A tier goes away with transfers in flight, and right after its last restore, without anything
// running past it.
void teardown(const std::shared_ptr<model::SsdReaders> &readers, kv::Layout layout) {
  test::HostKvPages pages(layout, 16 * layout.blockPages());
  const kv::SlotFormat format(layout);
  std::shared_ptr<DiskBudget> budget;
  const std::vector<std::shared_ptr<SlotFile>> files = slotFiles(format, 16, budget);
  const kv::Components all = layout.components();
  {
    Tier tier(pages, format, files, readers, 16);
    std::vector<std::unique_ptr<KvTransfer>> transfers;
    std::vector<std::shared_ptr<engine::KvDiskSlot>> slots;
    for (uint32_t page = 0; page < 8; ++page) {
      slots.push_back(tier.acquireSlot(all, cached, nullptr));
      transfers.push_back(tier.demote(blockOf(layout, page), slots.back(), all, WriteClass::Reclaim, {}));
      require(transfers.back() != nullptr, "a write within the tier's bound was refused");
      // The first four land before the tier goes, and are being restored when it does.
      if (page == 3)
        runUntilReady(tier, transfers);
    }
    for (uint32_t page = 8; page < 12; ++page)
      transfers.push_back(tier.restore(slots[page - 8], all, blockOf(layout, page), {}));
    tier.submitRestores();
  }
  require(budget->usedBytes() == 0, "the tier left its pages with disk IO still running");
  for (int round = 0; round < 64; ++round) {
    auto wakes = std::make_unique<std::atomic<uint32_t>>(0);
    {
      Tier tier(pages, format, files, readers);
      const std::shared_ptr<engine::KvDiskSlot> slot = tier.acquireSlot(all, cached, nullptr);
      std::vector<std::unique_ptr<KvTransfer>> transfers;
      transfers.push_back(tier.demote(blockOf(layout, 0), slot, all, WriteClass::Reclaim, {}));
      runUntilReady(tier, transfers);
      // The completion takes a while, as a wake that waits for a lock does.
      transfers.push_back(tier.restore(slot, all, blockOf(layout, 1), [counter = wakes.get()] {
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
      // Qwen3.6-35B's layout, which keeps its spans' order, a windowed one,
      // and Qwen3.8-Flash-Next's, whose slots hold block records.
      for (const kv::Layout layout : {kv::Layout{10, 2, 256, format},
                                      kv::Layout{6, 2, 256, format, 0, false, 512, 0b010110},
                                      kv::Layout{12, 2, 256, format, 128, true, 0, 0, kv::Reads::SelectedBlocks}}) {
        roundTrips(readers, layout);
        teardown(readers, layout);
        if (!layout.spillable())
          adoptedCopiesRestore(readers, layout);
      }
      windowedCopiesAreTwoParts(readers, format);
      aPartThatFailsFailsItsCopy(readers, format);
    }
  } catch (const std::exception &error) {
    std::cerr << "kv page tier host tests failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "kv page tier host tests passed\n";
  return 0;
}
