#pragma once

#include "engine/CacheDirectory.hpp"
#include "model/SlotFile.hpp"

#include <sys/mount.h>

#include <cstdint>
#include <deque>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace splash::engine {

// The volume a disk tier's files live on, as fstatfs says.
struct DiskVolume final {
  // Where it is mounted.
  std::string mountPoint;
  // What can be written now (f_bavail x f_bsize). Space macOS would first
  // have to purge, which the Finder counts as available, is not.
  uint64_t availableBytes = 0;
  uint64_t sizeBytes = 0;
  // A volume of this Mac's own (MNT_LOCAL), whose free space the tier
  // follows.
  bool local = false;

  [[nodiscard]] static DiskVolume of(const struct statfs &status);
};

// The volume the directory at `path`, or the nearest one above it that
// exists, lives on; empty when it cannot be measured.
[[nodiscard]] std::optional<DiskVolume> volumeOf(const std::filesystem::path &path);

// The most --max-cache-disk auto lets the tier hold.
inline constexpr uint64_t kAutomaticCacheDiskBytes = uint64_t{32} << 30;

// What a disk tier leaves free on its volume: 10 GiB, or 5% of a volume
// smaller than 200 GiB.
[[nodiscard]] uint64_t diskFloorBytes(uint64_t volumeBytes) noexcept;

// What a volume could give a disk tier beyond the floor, counting the blocks
// the tier's own files hold (`heldBytes`) as its:
//   max(0, available + held - floor).
[[nodiscard]] uint64_t diskRoomBytes(uint64_t availableBytes, uint64_t heldBytes,
                                     uint64_t floorBytes) noexcept;

// The most a disk tier may hold on a local volume: a quarter of its room
// there (diskRoomBytes), and at most the quota:
//   min(quota, max(0, available + held - floor) / 4).
// The tier's own writes and punches move bytes between available and held,
// so they leave the limit where it is; space anything else takes or frees
// moves it by a quarter. Several tiers on one volume each settle at
// (available - floor) / (tiers + 3) of what was available before them, so
// free space stays above the floor, and blocks a snapshot keeps when the tier
// frees them cost it at most a third more than it freed.
[[nodiscard]] uint64_t diskLimitBytes(uint64_t quotaBytes, uint64_t availableBytes,
                                      uint64_t heldBytes, uint64_t floorBytes) noexcept;

// Why the automatic quota leaves the tier off on this volume, where the
// tier's files would already hold `heldBytes`: it cannot be measured, it is
// not local, or its limit (diskLimitBytes) would not hold `minimumBytes`.
// Empty when the tier turns on.
[[nodiscard]] std::optional<std::string>
automaticDiskRefusal(const std::optional<DiskVolume> &volume, uint64_t heldBytes,
                     uint64_t minimumBytes);

// "12.3 GiB".
[[nodiscard]] std::string gibibytes(uint64_t bytes);

// Keeps a disk tier within what its volume leaves it. Each measure, at every
// control pass and when a write found the volume full, reads the volume
// through the first file's descriptor, which names the right volume however
// the file was made, adds the blocks the tier's files hold, and sets the
// budget's limit (diskLimitBytes); off a local volume the quota is the
// limit. In a persistent cache's root one limit bounds every namespace
// together, and the tier may use all of it: the others nobody holds count
// among what the tier holds, and while the root holds more than its limit,
// as the tier grows or the volume shrinks, they go, oldest first, before
// the tier gives up a copy of its own. A write that found the volume full
// while the volume shows room above the floor, or on a volume whose free
// space the tier does not follow, met a limit the measure cannot see, such as
// a volume quota or a disk image whose host is full, or a volume that filled
// and freed again before the measure. A second such write says it was a
// limit: the tier then keeps within what it holds, its live slots, for the
// rest of the process. It goes on serving and replacing its copies within
// that, and a later such write lowers it again.
class DiskSpace final {
public:
  // The files live on one volume; `others` are the root's namespaces nobody
  // holds (CacheDirectory::idle), oldest first, none for temporary files.
  // Measures the volume.
  DiskSpace(std::shared_ptr<model::DiskBudget> budget,
            std::vector<std::shared_ptr<model::SlotFile>> files,
            std::vector<CacheDirectory::Idle> others = {});
  DiskSpace(const DiskSpace &) = delete;
  DiskSpace &operator=(const DiskSpace &) = delete;

  // Reads the volume through the first file and sets the limit from it.
  // Engine thread.
  void measure();
  // The same with a reading of the volume the caller made.
  void measure(std::optional<DiskVolume> volume);
  // As of the last measure: the volume, empty when it could not be measured;
  // what the tier's files and the other namespaces held then; and what the
  // other namespaces held.
  [[nodiscard]] const std::optional<DiskVolume> &volume() const noexcept { return volume_; }
  [[nodiscard]] uint64_t heldBytes() const noexcept { return heldBytes_; }
  [[nodiscard]] uint64_t othersBytes() const noexcept { return othersBytes_; }
  // The volume's free space sets the limit: it is measured and local.
  [[nodiscard]] bool followed() const noexcept { return volume_ && volume_->local; }

private:
  std::shared_ptr<model::DiskBudget> budget_;
  std::vector<std::shared_ptr<model::SlotFile>> files_;
  std::deque<CacheDirectory::Idle> others_;
  uint64_t othersBytes_ = 0;
  std::optional<DiskVolume> volume_;
  uint64_t heldBytes_ = 0;
  // A write found the volume full while the measure showed room.
  bool fullWithRoom_ = false;
  // What the tier held when a write met a limit the measure cannot see;
  // none until one did.
  static constexpr uint64_t kNoCap = std::numeric_limits<uint64_t>::max();
  uint64_t heldCapBytes_ = kNoCap;
};

} // namespace splash::engine
