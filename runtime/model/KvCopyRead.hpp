#pragma once

// Reads of a tier copy's bytes (model::KvCopy::readAt) on the calling
// thread, as the KV tier's restores and the reads of spilled K/V make them on
// the SSD readers: into memory aligned for uncached IO. A task whose read
// failed throws, and the readers try it once more (SsdReaders).

#include "model/Model.hpp"
#include "ops/KvSlotFormat.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace splash::model {

// Memory reads of tier copies land in, aligned for uncached IO and grown to
// the largest read made into it.
class KvReadBuffer final {
public:
  KvReadBuffer() = default;
  ~KvReadBuffer();
  KvReadBuffer(const KvReadBuffer &) = delete;
  KvReadBuffer &operator=(const KvReadBuffer &) = delete;

  // At least `bytes` of the buffer; what it held before is gone.
  [[nodiscard]] std::span<std::byte> hold(uint64_t bytes);

private:
  std::byte *bytes_ = nullptr;
  uint64_t size_ = 0;
};

// The bytes of `range` of the copy, read into the buffer and valid until its
// next read, or nothing when the read failed.
[[nodiscard]] std::optional<std::span<const std::byte>> readKvCopy(const KvCopy &copy, kv::SlotRange range,
                                                                   KvReadBuffer &buffer);
// The same into the calling thread's own buffer, which its next read
// reuses: for reads no larger than a layer's records, which the readers make
// by the thousand. A read of a whole page brings its own buffer, so that no
// reader keeps a slot's worth of memory.
[[nodiscard]] std::optional<std::span<const std::byte>> readKvCopy(const KvCopy &copy, kv::SlotRange range);

} // namespace splash::model
