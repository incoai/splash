#include "model/KvCopyRead.hpp"

#include <new>
#include <utility>

namespace splash::model {
namespace {

constexpr uint64_t kAlignment = kv::SlotFormat::kReadAlignmentBytes;

constexpr uint64_t alignUp(uint64_t value) noexcept { return (value + kAlignment - 1) / kAlignment * kAlignment; }

} // namespace

KvReadBuffer::~KvReadBuffer() {
  if (bytes_)
    ::operator delete(bytes_, std::align_val_t{kAlignment});
}

std::span<std::byte> KvReadBuffer::hold(uint64_t bytes) {
  if (size_ < bytes) {
    if (bytes_)
      ::operator delete(std::exchange(bytes_, nullptr), std::align_val_t{kAlignment});
    size_ = 0;
    bytes_ = static_cast<std::byte *>(::operator new(bytes, std::align_val_t{kAlignment}));
    size_ = bytes;
  }
  return {bytes_, size_};
}

std::optional<std::span<const std::byte>> readKvCopy(const KvCopy &copy, kv::SlotRange range,
                                                     KvReadBuffer &buffer) {
  // The aligned range around the bytes, and a unit more: an adopted copy
  // reads the whole segments a range touches (SlotFile::readAt), and a range
  // of segments (the index, a block's record, a layer's, a whole page's)
  // leaves at most the room after a record, less than a unit, outside
  // (kv::SlotFormat).
  return copy.readAt(range.offset, range.bytes,
                     buffer.hold(alignUp(range.offset % kAlignment + range.bytes) + kAlignment));
}

std::optional<std::span<const std::byte>> readKvCopy(const KvCopy &copy, kv::SlotRange range) {
  thread_local KvReadBuffer buffer;
  return readKvCopy(copy, range, buffer);
}

} // namespace splash::model
