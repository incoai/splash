#include "engine/DiskSpace.hpp"

#include "StderrLine.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace splash::engine {
namespace {

constexpr uint64_t kGiB = uint64_t{1} << 30;
constexpr uint64_t kFloorBytes = 10 * kGiB;

} // namespace

DiskVolume DiskVolume::of(const struct statfs &status) {
  const uint64_t block = status.f_bsize;
  return {status.f_mntonname, uint64_t{status.f_bavail} * block,
          uint64_t{status.f_blocks} * block, (status.f_flags & MNT_LOCAL) != 0};
}

std::optional<DiskVolume> volumeOf(const std::filesystem::path &path) {
  std::filesystem::path directory = path;
  for (;;) {
    const int descriptor = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (descriptor >= 0) {
      struct statfs status {};
      const bool measured = ::fstatfs(descriptor, &status) == 0;
      ::close(descriptor);
      return measured ? std::optional(DiskVolume::of(status)) : std::nullopt;
    }
    if (!directory.has_relative_path())
      return std::nullopt;
    directory = directory.parent_path();
  }
}

uint64_t diskFloorBytes(uint64_t volumeBytes) noexcept {
  return std::min(kFloorBytes, volumeBytes / 20);
}

uint64_t diskRoomBytes(uint64_t availableBytes, uint64_t heldBytes, uint64_t floorBytes) noexcept {
  const uint64_t room = availableBytes + heldBytes;
  return room > floorBytes ? room - floorBytes : 0;
}

uint64_t diskLimitBytes(uint64_t quotaBytes, uint64_t availableBytes, uint64_t heldBytes,
                        uint64_t floorBytes) noexcept {
  return std::min(quotaBytes, diskRoomBytes(availableBytes, heldBytes, floorBytes) / 4);
}

uint64_t diskActiveLimitBytes(uint64_t quotaBytes, uint64_t availableBytes, uint64_t heldBytes,
                              uint64_t floorBytes) noexcept {
  return std::min(quotaBytes, diskRoomBytes(availableBytes, heldBytes, floorBytes));
}

uint64_t volumeActiveLimitBytes(uint64_t quotaBytes, const std::optional<DiskVolume> &volume,
                                uint64_t heldBytes) noexcept {
  if (!volume || !volume->local)
    return quotaBytes;
  return diskActiveLimitBytes(quotaBytes, volume->availableBytes, heldBytes, diskFloorBytes(volume->sizeBytes));
}

std::optional<std::string> automaticDiskRefusal(const std::optional<DiskVolume> &volume,
                                                uint64_t heldBytes, uint64_t minimumBytes) {
  if (!volume)
    return "its volume cannot be measured";
  if (!volume->local)
    return volume->mountPoint + " is not a local volume";
  const uint64_t floor = diskFloorBytes(volume->sizeBytes);
  if (diskLimitBytes(kAutomaticCacheDiskBytes, volume->availableBytes, heldBytes, floor) <
      minimumBytes)
    return gibibytes(volume->availableBytes) + " available on " + volume->mountPoint +
           " is too little beyond the " + gibibytes(floor) + " it keeps free";
  return std::nullopt;
}

std::string gibibytes(uint64_t bytes) {
  char text[32];
  std::snprintf(text, sizeof(text), "%.1f GiB", static_cast<double>(bytes) / kGiB);
  return text;
}

DiskSpace::DiskSpace(std::shared_ptr<model::DiskBudget> budget, bool automatic,
                     std::vector<std::shared_ptr<model::SlotFile>> files,
                     std::vector<CacheDirectory::Idle> others)
    : budget_(std::move(budget)), automatic_(automatic), files_(std::move(files)),
      others_(std::make_move_iterator(others.begin()), std::make_move_iterator(others.end())) {
  if (!budget_ || files_.empty())
    throw std::invalid_argument("a disk tier's space needs its budget and a file");
  for (const CacheDirectory::Idle &cache : others_)
    othersBytes_ += cache.bytes;
  measure();
}

void DiskSpace::measure() {
  const std::optional<struct statfs> status = files_.front()->volume();
  measure(status ? std::optional(DiskVolume::of(*status)) : std::nullopt);
}

void DiskSpace::measure(std::optional<DiskVolume> volume) {
  const bool full = budget_->takeFullReport();
  volume_ = std::move(volume);
  uint64_t own = 0;
  for (const std::shared_ptr<model::SlotFile> &file : files_)
    own += file->allocatedBytes();
  heldBytes_ = own + othersBytes_;
  if (volume_)
    budget_->setVolumeAvailableBytes(volume_->availableBytes);
  const uint64_t floor = volume_ ? diskFloorBytes(volume_->sizeBytes) : 0;
  // A write found the volume full while it shows room above the floor, or on
  // a volume the tier does not follow. The first such write may have met a
  // volume that filled and freed again before this measure; from the second
  // on, each met a limit the measure cannot see.
  if (full && (!followed() || volume_->availableBytes > floor) &&
      std::exchange(fullWithRoom_, true) && budget_->usedBytes() < heldCapBytes_) {
    const bool first = heldCapBytes_ == kNoCap;
    heldCapBytes_ = budget_->usedBytes();
    const std::string seen =
        !volume_          ? "its volume full, which cannot be measured"
        : !volume_->local ? volume_->mountPoint + " full, a volume whose free space it does not follow"
                          : volume_->mountPoint + " full while it shows " +
                                gibibytes(volume_->availableBytes) + " available";
    if (first)
      logLine("SSD cache: a second write found ", seen, "; the cache keeps within the ",
              gibibytes(heldCapBytes_), " it holds while this server runs.");
  }
  const uint64_t quota = budget_->capacityBytes();
  const uint64_t activeLimit = volumeActiveLimitBytes(quota, volume_, heldBytes_);
  // The automatic quota's copies keep a quarter of the room; a size's may
  // take all of it, as active KV may.
  const uint64_t rootLimit = automatic_ && followed()
                                 ? diskLimitBytes(quota, volume_->availableBytes, heldBytes_, floor)
                                 : activeLimit;
  // The root holds more than its limit: another model's cache nobody holds
  // goes, the oldest first, before a copy of this one's. One someone took
  // meanwhile is that process's to count.
  while (!others_.empty() && own + othersBytes_ > rootLimit) {
    const CacheDirectory::Idle oldest = std::move(others_.front());
    others_.pop_front();
    othersBytes_ -= oldest.bytes;
    if (CacheDirectory::remove(oldest))
      logLine("Persistent cache: removed ", oldest.path.string(), " (", gibibytes(oldest.bytes),
              "), the oldest cache nobody holds, to keep the cache root within ",
              gibibytes(rootLimit), ".");
  }
  // The tier may use the whole root: the others go as it grows, before
  // active KV takes the room beyond the root's limit.
  budget_->setLimitBytes(std::min(rootLimit, heldCapBytes_));
  budget_->setActiveLimitBytes(std::min(activeLimit, heldCapBytes_));
}

} // namespace splash::engine
