// The disk tier's limits on its volume (engine::DiskSpace): the rules, how
// they behave as space comes and goes, the automatic tier's start, a measure
// of real files, and the other caches in a persistent cache's root.

#include "Checked.hpp"
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "engine/CacheDirectory.hpp"
#include "engine/DiskSpace.hpp"
#include "model/SlotFile.hpp"

#include <sys/time.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
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
// DiskSpace's quota: --max-cache-disk auto's, or a size.
constexpr bool kAutomatic = true, kSize = false;

DiskVolume localVolume(uint64_t availableBytes, uint64_t sizeBytes = 1000 * kGiB) {
  return {"/Volumes/Test", availableBytes, sizeBytes, true};
}

// A tier's reader of its volume that reads `volume` at every measure, so the
// test Mac's own free space never sets a limit.
std::function<std::optional<DiskVolume>()> reads(const DiskVolume &volume) {
  return [volume] { return std::optional(volume); };
}

// A volume the test changes between a tier's measures: its reader reads what
// the test last set.
class TestVolume final {
public:
  explicit TestVolume(std::optional<DiskVolume> volume)
      : volume_(std::make_shared<std::optional<DiskVolume>>(std::move(volume))) {}
  [[nodiscard]] std::function<std::optional<DiskVolume>()> reader() const {
    return [volume = volume_] { return *volume; };
  }
  void set(std::optional<DiskVolume> volume) const { *volume_ = std::move(volume); }

private:
  std::shared_ptr<std::optional<DiskVolume>> volume_;
};

// The floor is 10 GiB, or 5% of a volume under 200 GiB; the automatic limit is a quarter of the
// room beyond it, capped by the quota.
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

// Automatic tiers sharing a volume settle at (available - floor) / (tiers + 3) each, above the
// floor.
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

// Blocks a snapshot keeps cost an automatic tier at most a third more than it first freed.
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

// Active KV may take all the room beyond the floor, up to the quota, while cached copies keep a
// quarter of it.
void testActiveKvTakesTheRoom() {
  const uint64_t floor = 10 * kGiB;
  require(diskActiveLimitBytes(kQuota, 30 * kGiB, 0, floor) == 20 * kGiB &&
              diskActiveLimitBytes(kQuota, 22 * kGiB, 8 * kGiB, floor) == 20 * kGiB &&
              diskActiveLimitBytes(kQuota, 500 * kGiB, 0, floor) == kQuota &&
              !diskActiveLimitBytes(kQuota, 9 * kGiB, 0, floor),
          "the active limit is not the room beyond the floor, at most the quota");
  require(volumeActiveLimitBytes(kQuota, localVolume(16 * kGiB, 460 * kGiB), 0) == 6 * kGiB &&
              volumeActiveLimitBytes(kQuota, DiskVolume{"/Volumes/Share", kGiB, 1000 * kGiB, false}, 0) == kQuota &&
              volumeActiveLimitBytes(kQuota, std::nullopt, 0) == kQuota,
          "a volume's active limit is not its room, or the quota off a volume the tier follows");
  auto budget = std::make_shared<model::DiskBudget>(kQuota);
  auto file = std::make_shared<model::SlotFile>(kHostPageBytes, budget);
  const TestVolume volume(localVolume(500 * kGiB));
  DiskSpace space(budget, kAutomatic, {file}, volume.reader());
  for (const auto [available, active, cached] :
       {std::array{16 * kGiB, 6 * kGiB, 3 * kGiB / 2}, std::array{24 * kGiB, 14 * kGiB, 7 * kGiB / 2}}) {
    volume.set(localVolume(available, 460 * kGiB));
    space.measure();
    require(space.heldBytes() == 0 && budget->activeLimitBytes() == active && budget->limitBytes() == cached,
            "a Mac mini's free space did not give active KV its room and cached copies a quarter of it");
  }
  volume.set(DiskVolume{"/Volumes/Share", kGiB, 1000 * kGiB, false});
  space.measure();
  require(budget->activeLimitBytes() == kQuota && budget->limitBytes() == kQuota,
          "a volume the tier does not follow did not leave both limits at the quota");
  const std::shared_ptr<model::SlotFile::Slot> held = file->acquire();
  volume.set(localVolume(100 * kGiB));
  for (int write = 0; write < 2; ++write) {
    budget->reportFull();
    space.measure();
  }
  require(budget->activeLimitBytes() == kHostPageBytes && budget->limitBytes() == kHostPageBytes,
          "a hidden limit did not cap active KV at what the tier holds");
}

// A --max-cache-disk size is held as given while the volume has room beyond the floor, else that
// room; auto holds a quarter, at most 32 GiB.
void testSizeIsHeldAsGiven() {
  const uint64_t size = 500 * kGiB, floor = 10 * kGiB;
  auto automaticBudget = std::make_shared<model::DiskBudget>(kAutomaticCacheDiskBytes);
  auto sizeBudget = std::make_shared<model::DiskBudget>(size);
  const TestVolume volume(localVolume(500 * kGiB));
  DiskSpace automatic(automaticBudget, kAutomatic,
                      {std::make_shared<model::SlotFile>(kHostPageBytes, automaticBudget)}, volume.reader());
  DiskSpace sized(sizeBudget, kSize, {std::make_shared<model::SlotFile>(kHostPageBytes, sizeBudget)},
                  volume.reader());
  for (const auto [available, automaticLimit, sizeLimit] :
       {std::array{floor + 7308 * kGiB / 10, kAutomaticCacheDiskBytes, size},
        std::array{floor + 72 * kGiB, 18 * kGiB, 72 * kGiB},
        std::array{9 * kGiB, uint64_t{0}, uint64_t{0}}}) {
    volume.set(localVolume(available, 2000 * kGiB));
    automatic.measure();
    sized.measure();
    require(automatic.heldBytes() == 0 && automaticBudget->limitBytes() == automaticLimit,
            "auto's limit is not a quarter of its volume's room, at most 32 GiB");
    require(sized.heldBytes() == 0 && sizeBudget->limitBytes() == sizeLimit &&
                sizeBudget->activeLimitBytes() == sizeLimit,
            "a size is not held as given while its volume has room for it beyond the floor, or else "
            "that room");
  }
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

// Measuring real files on the temporary directory's volume sets the limit by the rule; a full-volume
// write lowers it only when the volume shows no room.
void testMeasureOfRealFiles() {
  auto budget = std::make_shared<model::DiskBudget>(kQuota);
  auto file = std::make_shared<model::SlotFile>(kHostPageBytes, budget);
  const TestVolume measured(volumeOf(std::filesystem::temp_directory_path()));
  DiskSpace space(budget, kAutomatic, {file}, measured.reader());
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
  measured.set(localVolume(kGiB));
  space.measure();
  require(file->writable() && !budget->writesStopped() && !budget->limitBytes() &&
              !budget->fullReported(),
          "a full volume stopped the files or kept the limit");
  measured.set(localVolume(100 * kGiB));
  space.measure();
  std::vector<std::shared_ptr<model::SlotFile::Slot>> slots{file->acquire(), file->acquire()};
  const uint64_t ruled = budget->limitBytes();
  budget->reportFull();
  space.measure();
  require(ruled > 2 * kHostPageBytes && budget->limitBytes() == ruled,
          "one write that found the volume full while it shows room capped the tier");
  budget->reportFull();
  space.measure();
  require(file->writable() && !budget->writesStopped() &&
              budget->limitBytes() == 2 * kHostPageBytes && !file->acquire(),
          "a second write that found the volume full while it shows room did not keep the tier "
          "within what it held");
  measured.set(localVolume(500 * kGiB));
  space.measure();
  slots.pop_back();
  slots.push_back(file->acquire());
  require(budget->limitBytes() == 2 * kHostPageBytes && slots.back() && !file->acquire(),
          "the tier did not replace within what it held, or grew past it");
  slots.pop_back();
  budget->reportFull();
  space.measure();
  require(budget->limitBytes() == kHostPageBytes,
          "another write that met the hidden limit did not lower it to what the tier held");

  // Off a local volume the quota is the limit, and a second write that found
  // the volume full keeps the tier within what it holds.
  auto other = std::make_shared<model::DiskBudget>(5 * kGiB);
  auto shared = std::make_shared<model::SlotFile>(kHostPageBytes, other);
  const DiskVolume share{"/Volumes/Share", kGiB, 1000 * kGiB, false};
  DiskSpace network(other, kSize, {shared}, reads(share));
  require(!network.followed() && other->limitBytes() == 5 * kGiB,
          "a network volume's free space set the limit");
  const std::shared_ptr<model::SlotFile::Slot> held = shared->acquire();
  other->reportFull();
  network.measure();
  require(other->limitBytes() == 5 * kGiB,
          "one write that found a network volume full capped the tier");
  other->reportFull();
  network.measure();
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

// Other caches nobody holds in a persistent root count against the limit and go first, oldest
// first.
void testOtherCachesGoFirst() {
  const test::TemporaryDirectory root("splash-disk-space-root");
  const std::vector<CacheDirectory::Idle> others = otherCaches(root.path());
  auto budget = std::make_shared<model::DiskBudget>(kQuota);
  auto file = std::make_shared<model::SlotFile>(kHostPageBytes, budget);
  const TestVolume volume(localVolume(500 * kGiB));
  DiskSpace space(budget, kAutomatic, {file}, volume.reader(), others);
  require(space.othersBytes() == others[0].bytes + others[1].bytes,
          "the other caches were not counted");
  // A volume that leaves the root a limit of what the others hold less half
  // a MiB: the oldest goes, the newer one stays.
  const uint64_t floor = diskFloorBytes(100 * kGiB);
  const uint64_t rootLimit = space.othersBytes() - kMiB / 2;
  volume.set(DiskVolume{"/Volumes/Test", floor + 4 * rootLimit - space.heldBytes(), 100 * kGiB, true});
  space.measure();
  require(!std::filesystem::exists(others[0].path / "kv.slots") &&
              std::filesystem::exists(others[1].path / "kv.slots") &&
              space.othersBytes() == others[1].bytes &&
              budget->limitBytes() == rootLimit,
          "the oldest other cache did not go first, or the tier did not get the root's limit");
}

// Another cache removed by someone else stops counting at the next measure.
void testRemovedCachesStopCounting() {
  const test::TemporaryDirectory root("splash-disk-space-root");
  const std::vector<CacheDirectory::Idle> others = otherCaches(root.path());
  auto budget = std::make_shared<model::DiskBudget>(kQuota);
  auto file = std::make_shared<model::SlotFile>(kHostPageBytes, budget);
  const DiskVolume volume{"/Volumes/Test", 500 * kGiB, 1000 * kGiB, true};
  DiskSpace space(budget, kSize, {file}, reads(volume), others);
  std::filesystem::remove_all(others[1].path);
  space.measure();
  require(space.othersBytes() == others[0].bytes && space.heldBytes() == others[0].bytes,
          "a removed cache still counted among what the root holds");
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
    const DiskVolume volume{"/Volumes/Test", 500 * kGiB, 1000 * kGiB, local};
    DiskSpace space(budget, kSize, {file}, reads(volume), others);
    require(budget->limitBytes() == quota && space.othersBytes() == quota - kMiB - kMiB / 2,
            "the tier did not get the root's whole quota");
    const std::vector<std::byte> payload(kMiB, std::byte{1});
    std::vector<std::shared_ptr<model::SlotFile::Slot>> slots;
    const auto grow = [&] {
      slots.push_back(file->acquire());
      require(slots.back() && file->write(slots.back(), {payload}, {})->wait(),
              "the tier could not write within its limit");
      space.measure();
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
    testActiveKvTakesTheRoom();
    testSizeIsHeldAsGiven();
    testTiersOnOneVolumeConverge();
    testSnapshotCostsAtMostAThird();
    testAutomaticStart();
    testMeasureOfRealFiles();
    testOtherCachesGoFirst();
    testRemovedCachesStopCounting();
    testOtherCachesGoAsTheTierGrows();
    std::cout << "Disk space tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
