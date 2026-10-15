// The disk tier's limit on its volume (engine::DiskSpace): the rule, how it
// behaves as space comes and goes, the automatic tier's start, a measure of
// real files, and the other caches in a persistent cache's root.

#include "Checked.hpp"
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "engine/CacheDirectory.hpp"
#include "engine/DiskSpace.hpp"
#include "model/SlotFile.hpp"

#include <sys/time.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::engine;
using splash::test::require;

namespace {

constexpr uint64_t kMiB = uint64_t{1} << 20;
constexpr uint64_t kGiB = uint64_t{1} << 30;
constexpr uint64_t kQuota = 32 * kGiB;

DiskVolume localVolume(uint64_t availableBytes, uint64_t sizeBytes = 1000 * kGiB) {
  return {"/Volumes/Test", availableBytes, sizeBytes, true};
}

// The floor is 10 GiB, or 5% of a volume smaller than 200 GiB; the limit is
// a quarter of what the volume has beyond it, the tier's own blocks counted,
// at most the quota and never below zero.
void testTheRule() {
  require(diskFloorBytes(1000 * kGiB) == 10 * kGiB && diskFloorBytes(100 * kGiB) == 5 * kGiB &&
              diskFloorBytes(12 * kGiB) == 12 * kGiB / 20,
          "the floor is not 10 GiB or 5% of a small volume");
  const uint64_t floor = 10 * kGiB;
  require(diskLimitBytes(kQuota, 50 * kGiB, 0, floor) == 10 * kGiB &&
              diskLimitBytes(kQuota, 42 * kGiB, 8 * kGiB, floor) == 10 * kGiB,
          "the tier's own blocks moved its limit");
  require(diskLimitBytes(kQuota, 500 * kGiB, 0, floor) == kQuota &&
              diskLimitBytes(5 * kGiB, 500 * kGiB, 0, floor) == 5 * kGiB,
          "the limit passed the quota");
  require(!diskLimitBytes(kQuota, 9 * kGiB, 0, floor) &&
              diskLimitBytes(kQuota, 9 * kGiB, 3 * kGiB, floor) == kGiB / 2,
          "the limit below the floor is not what the tier holds beyond it");
}

// Tiers sharing a volume, each measuring in turn and filling or giving up
// to its limit, settle at (available - floor) / (tiers + 3) each, and the
// free space they leave stays above the floor.
void testTiersOnOneVolumeConverge() {
  const uint64_t floor = 10 * kGiB, before = 100 * kGiB;
  for (uint32_t tiers = 1; tiers <= 4; ++tiers) {
    std::vector<uint64_t> held(tiers, 0);
    uint64_t available = before;
    for (uint32_t round = 0; round < 200; ++round) {
      for (uint64_t &tier : held) {
        const uint64_t limit = diskLimitBytes(kQuota, available, tier, floor);
        available = available + tier - limit;
        tier = limit;
        require(available > floor, "tiers on one volume took it below the floor");
      }
    }
    const uint64_t settled = (before - floor) / (tiers + 3);
    for (const uint64_t tier : held)
      require(std::max(tier, settled) - std::min(tier, settled) < kMiB,
              "tiers on one volume did not settle at their share");
  }
}

// Blocks a snapshot keeps when the tier frees them return nothing to the
// volume. Each measure then lowers the limit by a quarter of what the last
// one freed: the tier gives up at most a third more than it first freed,
// and keeps the rest.
void testSnapshotCostsAtMostAThird() {
  const uint64_t floor = 10 * kGiB;
  uint64_t available = 50 * kGiB, held = 0;
  held = diskLimitBytes(kQuota, available, held, floor);
  available -= held;
  // Something else takes 20 GiB.
  available -= 20 * kGiB;
  const uint64_t start = held;
  uint64_t firstFreed = 0;
  for (uint32_t measure = 0; measure < 64; ++measure) {
    const uint64_t limit = diskLimitBytes(kQuota, available, held, floor);
    if (limit >= held)
      break;
    if (!firstFreed)
      firstFreed = held - limit;
    held = limit;
  }
  require(firstFreed && held && start - held <= firstFreed + firstFreed / 3 + 1,
          "a snapshot that keeps freed blocks drained the tier");
}

// The automatic tier turns on only on a local volume whose limit holds the
// minimum, the tier's own blocks counted.
void testAutomaticStart() {
  const uint64_t minimum = 2 * 153 * kMiB + kGiB;
  require(automaticDiskRefusal(std::nullopt, 0, minimum)->find("cannot be measured") !=
              std::string::npos,
          "an unmeasured volume did not refuse the automatic tier");
  require(automaticDiskRefusal(DiskVolume{"/Volumes/Share", 500 * kGiB, 1000 * kGiB, false}, 0,
                               minimum)
                  ->find("/Volumes/Share is not a local volume") != std::string::npos,
          "a network volume did not refuse the automatic tier");
  const std::optional<std::string> refusal =
      automaticDiskRefusal(localVolume(14 * kGiB), 0, minimum);
  require(refusal && refusal->find("14.0 GiB available on /Volumes/Test is too little beyond "
                                 "the 10.0 GiB it keeps free") != std::string::npos,
          "a volume with too little beyond its floor did not refuse the automatic tier");
  require(!automaticDiskRefusal(localVolume(18 * kGiB), 0, minimum) &&
              !automaticDiskRefusal(localVolume(12 * kGiB), 8 * kGiB, minimum),
          "a volume with room, or the room the tier's own files hold, refused the tier");
}

// A measure of a real file sets the limit by the rule from what it read and
// leaves the write budget alone: there is none unless the user sets one, so
// nothing is held back. A write that found the volume full while it shows
// none lowers the limit by the rule. One that found it full while it shows
// room may have met a volume that filled and freed again, and changes
// nothing; a second met a limit the measure cannot see: the tier keeps
// within what it holds then, replacing within that, until another such write
// lowers it again; its files go on taking writes. Off a local volume a
// second write that found it full does the same.
void testMeasureOfRealFiles() {
  auto budget = std::make_shared<model::DiskBudget>(kQuota);
  auto file = std::make_shared<model::SlotFile>(kHostPageBytes, budget);
  DiskSpace space(budget, {file});
  const std::optional<DiskVolume> &volume = space.volume();
  require(volume && volume->local && space.followed() && volume->sizeBytes &&
              budget->volumeAvailableBytes() == volume->availableBytes,
          "the temporary directory's volume was not measured");
  require(budget->limitBytes() ==
              diskLimitBytes(kQuota, volume->availableBytes,
                             space.heldBytes(), diskFloorBytes(volume->sizeBytes)),
          "the limit is not the rule's of what the measure read");
  require(budget->admits(false) && !budget->writeBudgetBytes() && !budget->admitsAgainAt(),
          "a measure set a write budget nobody asked for");

  budget->reportFull();
  space.measure(localVolume(kGiB));
  require(file->writable() && !budget->writesStopped() && !budget->limitBytes() &&
              !budget->fullReported(),
          "a full volume stopped the files or kept the limit");
  space.measure(localVolume(100 * kGiB));
  std::vector<std::shared_ptr<model::SlotFile::Slot>> slots{file->acquire(), file->acquire()};
  const uint64_t ruled = budget->limitBytes();
  budget->reportFull();
  space.measure(localVolume(100 * kGiB));
  require(ruled > 2 * kHostPageBytes && budget->limitBytes() == ruled,
          "one write that found the volume full while it shows room capped the tier");
  budget->reportFull();
  space.measure(localVolume(100 * kGiB));
  require(file->writable() && !budget->writesStopped() &&
              budget->limitBytes() == 2 * kHostPageBytes && !file->acquire(),
          "a second write that found the volume full while it shows room did not keep the tier "
          "within what it held");
  space.measure(localVolume(500 * kGiB));
  slots.pop_back();
  slots.push_back(file->acquire());
  require(budget->limitBytes() == 2 * kHostPageBytes && slots.back() && !file->acquire(),
          "the tier did not replace within what it held, or grew past it");
  slots.pop_back();
  budget->reportFull();
  space.measure(localVolume(500 * kGiB));
  require(budget->limitBytes() == kHostPageBytes,
          "another write that met the hidden limit did not lower it to what the tier held");

  // Off a local volume the quota is the limit, and a second write that found
  // the volume full keeps the tier within what it holds.
  auto other = std::make_shared<model::DiskBudget>(5 * kGiB);
  auto shared = std::make_shared<model::SlotFile>(kHostPageBytes, other);
  DiskSpace network(other, {shared});
  const DiskVolume share{"/Volumes/Share", kGiB, 1000 * kGiB, false};
  network.measure(share);
  require(!network.followed() && other->limitBytes() == 5 * kGiB,
          "a network volume's free space set the limit");
  const std::shared_ptr<model::SlotFile::Slot> held = shared->acquire();
  other->reportFull();
  network.measure(share);
  require(other->limitBytes() == 5 * kGiB,
          "one write that found a network volume full capped the tier");
  other->reportFull();
  network.measure(share);
  require(shared->writable() && other->limitBytes() == kHostPageBytes,
          "a full network volume did not keep the tier within what it held");
}

// Two other caches in a fresh root, a MiB of slots each, the first older.
std::vector<CacheDirectory::Idle> otherCaches(const std::filesystem::path &root) {
  long when = 1000000000;
  for (const char digit : {'a', 'b'}) {
    const std::string name(CacheDirectory::kNamespaceDigits, digit);
    auto directory = CacheDirectory::open(root, name, std::chrono::milliseconds(0));
    std::ofstream(directory->kvSlots()) << std::string(kMiB, 'x');
    const timeval used[2] = {{when, 0}, {when, 0}};
    for (const char *file : {"lock", "kv.slots"})
      require(::utimes((directory->path() / file).c_str(), used) == 0,
              "a cache file could not be aged");
    when += 1000;
  }
  std::vector<CacheDirectory::Idle> others = CacheDirectory::idle(root);
  require(others.size() == 2 &&
              others[0].path == root / std::string(CacheDirectory::kNamespaceDigits, 'a') &&
              others[0].bytes >= kMiB &&
              others[1].bytes >= kMiB,
          "the other caches were not found oldest first");
  return others;
}

// One limit bounds a persistent cache's root, and the tier may use all of
// it: the other caches nobody holds count among what the tier holds, and
// while the root holds more than its limit they go, the oldest first,
// before a copy of the tier's own. A shrinking volume makes them go.
void testOtherCachesGoFirst() {
  const test::TemporaryDirectory root("splash-disk-space-root");
  const std::vector<CacheDirectory::Idle> others = otherCaches(root.path());
  auto budget = std::make_shared<model::DiskBudget>(kQuota);
  auto file = std::make_shared<model::SlotFile>(kHostPageBytes, budget);
  DiskSpace space(budget, {file}, others);
  require(space.othersBytes() == others[0].bytes + others[1].bytes,
          "the other caches were not counted");
  // A volume that leaves the root a limit of what the others hold less half
  // a MiB: the oldest goes, the newer one stays.
  const uint64_t floor = diskFloorBytes(100 * kGiB);
  const uint64_t rootLimit = space.othersBytes() - kMiB / 2;
  space.measure(
      DiskVolume{"/Volumes/Test", floor + 4 * rootLimit - space.heldBytes(), 100 * kGiB, true});
  require(!std::filesystem::exists(others[0].path / "kv.slots") &&
              std::filesystem::exists(others[1].path / "kv.slots") &&
              space.othersBytes() == others[1].bytes &&
              budget->limitBytes() == rootLimit,
          "the oldest other cache did not go first, or the tier did not get the root's limit");
}

// The tier growing within the quota makes them go as well, on a volume
// whose free space the tier follows or on one it does not.
void testOtherCachesGoAsTheTierGrows() {
  for (const bool local : {true, false}) {
    const test::TemporaryDirectory root("splash-disk-space-root");
    const std::vector<CacheDirectory::Idle> others = otherCaches(root.path());
    const uint64_t quota = others[0].bytes + others[1].bytes + kMiB + kMiB / 2;
    auto budget = std::make_shared<model::DiskBudget>(quota);
    auto file = std::make_shared<model::SlotFile>(kMiB, budget);
    DiskSpace space(budget, {file}, others);
    const DiskVolume volume{"/Volumes/Test", 500 * kGiB, 1000 * kGiB, local};
    space.measure(volume);
    require(budget->limitBytes() == quota && space.othersBytes() == quota - kMiB - kMiB / 2,
            "the tier did not get the root's whole quota");
    const std::vector<std::byte> payload(kMiB, std::byte{1});
    std::vector<std::shared_ptr<model::SlotFile::Slot>> slots;
    const auto grow = [&] {
      slots.push_back(file->acquire());
      require(slots.back() && file->write(slots.back(), {payload}, {})->wait(),
              "the tier could not write within its limit");
      space.measure(volume);
    };
    grow();
    require(std::filesystem::exists(others[0].path / "kv.slots") &&
                space.othersBytes() == others[0].bytes + others[1].bytes,
            "another cache went while the root held less than its limit");
    grow();
    require(!std::filesystem::exists(others[0].path / "kv.slots") &&
                std::filesystem::exists(others[1].path / "kv.slots") &&
                space.othersBytes() == others[1].bytes && budget->limitBytes() == quota,
            local ? "the tier growing kept the oldest other cache"
                  : "the tier growing on a volume it does not follow kept the oldest other cache");
  }
}

} // namespace

int main() {
  try {
    testTheRule();
    testTiersOnOneVolumeConverge();
    testSnapshotCostsAtMostAThird();
    testAutomaticStart();
    testMeasureOfRealFiles();
    testOtherCachesGoFirst();
    testOtherCachesGoAsTheTierGrows();
    std::cout << "Disk space tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
