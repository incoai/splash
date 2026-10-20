#include "video/h3/engine/WeightStream.hpp"

#include "video/h3/engine/DeviceProfile.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>

namespace h3 {
namespace {

// A slot starts free. libdispatch traps (SIGTRAP, "semaphore object deallocated while in use") when a semaphore is
// released with a value below the one it was created with, which a stream torn down mid-ring (h3_dit_release after a
// forward) always is when created with 1; created with 0 and signalled once, any value >= 0 is valid at release.
dispatch_semaphore_t freeSemaphore() {
  dispatch_semaphore_t s = dispatch_semaphore_create(0);
  dispatch_semaphore_signal(s);
  return s;
}

} // namespace


Status WeightStream::open(metal::MetalBackend &backend, const std::string &image, int slots,
                          std::unique_ptr<WeightStream> &out) {
  if (slots < 1) return error(Code::Argument, "stream slots must be >= 1 (got %d)", slots);
  auto stream = std::unique_ptr<WeightStream>(new WeightStream());
  stream->path_ = image;
  stream->fd_ = ::open(image.c_str(), O_RDONLY);
  if (stream->fd_ < 0) return error(Code::Io, "cannot open stream image %s", image.c_str());
  if (device::kStreamNoCache) fcntl(stream->fd_, F_NOCACHE, 1);
  Header &h = stream->header_;
  H3_TRY(model::readFully(stream->fd_, 0, sizeof h, &h, image));
  if (std::memcmp(h.magic, "H3WSTRM1", 8) != 0 || h.version != 1 || h.tensors != kTensors)
    return error(Code::Io, "%s: not an H3 weight-stream image", image.c_str());
  stream->qkfixEntries_.resize(h.qkfixBlocks);
  if (h.qkfixBlocks)
    H3_TRY(model::readFully(stream->fd_, sizeof h, h.qkfixBlocks * sizeof(QkFixEntry), stream->qkfixEntries_.data(),
                            image));
  stream->slots_.resize(slots);
  for (Slot &slot : stream->slots_) {
    for (int t = 0; t < kTensors; ++t) H3_TRY(backend.allocate(h.tensorBytes[t], slot.tensor[t], "stream slot"));
    slot.ready = dispatch_semaphore_create(0);
    slot.free = freeSemaphore();
  }
  stream->io_ = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
  out = std::move(stream);
  return {};
}

WeightStream::~WeightStream() {
  if (started_) {
    stop_ = true;
    for (Slot &slot : slots_) dispatch_semaphore_signal(slot.free);
    for (Slot &slot : qkfixSlots_) dispatch_semaphore_signal(slot.free);
    thread_.join();
  }
  if (fd_ >= 0) ::close(fd_);
}

uint64_t WeightStream::tensorOffset(int block, Tensor t) const {
  return header_.blocksOffset + static_cast<uint64_t>(block) * header_.blockBytes + header_.tensorOffset[t];
}

Status WeightStream::qkfixOffset(int block, uint64_t &offset) const {
  for (const QkFixEntry &entry : qkfixEntries_)
    if (entry.block == block) {
      offset = entry.offset;
      return {};
    }
  return error(Code::Io, "QKFIX block %d not in the stream image %s", block, path_.c_str());
}

Status WeightStream::verify(const model::SafetensorsFile &file, uint64_t fileOffset, uint64_t bytes,
                            uint64_t imageOffset, const char *what, int block) const {
  const size_t n = std::min<uint64_t>(device::kStreamVerifyBytes, bytes);
  std::vector<uint8_t> a(n), b(n);
  for (uint64_t at : {uint64_t(0), bytes - n}) {
    H3_TRY(file.readRange(fileOffset + at, n, a.data()));
    H3_TRY(model::readFully(fd_, imageOffset + at, n, b.data(), path_));
    if (std::memcmp(a.data(), b.data(), n) != 0)
      return error(Code::Io, "stream image %s differs from the weights: block %d %s (rebuild it with "
                             "tools/convert_dit.py stream)", path_.c_str(), block, what);
  }
  return {};
}

Status WeightStream::start(metal::MetalBackend &backend, int blocks, const std::set<int> &qkfixBlocks) {
  if (header_.blocks < static_cast<uint32_t>(blocks))
    return error(Code::Io, "stream image %s has %u blocks, the model %d", path_.c_str(), header_.blocks, blocks);
  blocksPerForward_ = blocks;
  qkfixRank_.assign(blocks, -1);
  qkfixPerForward_ = 0;
  for (int b : qkfixBlocks) {
    uint64_t offset = 0;
    H3_TRY(qkfixOffset(b, offset));
    qkfixRank_[b] = qkfixPerForward_++;
  }
  if (qkfixPerForward_ > 0) {
    qkfixSlots_.resize(device::kStreamQkFixSlots);
    for (Slot &slot : qkfixSlots_) {
      H3_TRY(backend.allocate(size_t(H3_DIT_QKV) * H3_DIT_HIDDEN * 2, slot.tensor[0], "stream QKFIX slot"));
      slot.ready = dispatch_semaphore_create(0);
      slot.free = freeSemaphore();
    }
  }
  started_ = true;
  thread_ = std::thread([this] { reader(); });
  return {};
}

// Parallel chunked pread of [offset, offset + bytes) into dst.
bool WeightStream::readInto(void *dst, uint64_t offset, size_t bytes) {
  const size_t chunk = device::kStreamChunkBytes, chunks = (bytes + chunk - 1) / chunk;
  const int threads = device::kStreamReadThreads, fd = fd_;
  std::atomic<int> failed{0};
  std::atomic<int> *pFailed = &failed;
  char *d = static_cast<char *>(dst);
  dispatch_apply(static_cast<size_t>(threads), io_, ^(size_t t) {
    for (size_t c = t; c < chunks; c += threads) {
      const size_t o = c * chunk, m = std::min(chunk, bytes - o);
      size_t got = 0;
      while (got < m) {
        const ssize_t r = pread(fd, d + o + got, m - got, static_cast<off_t>(offset + o + got));
        if (r <= 0) {
          *pFailed = 1;
          return;
        }
        got += static_cast<size_t>(r);
      }
    }
  });
  return failed == 0;
}

bool WeightStream::waitFree(dispatch_semaphore_t semaphore) {
  while (dispatch_semaphore_wait(semaphore, dispatch_time(DISPATCH_TIME_NOW, 200 * NSEC_PER_MSEC)) != 0)
    if (stop_) return false;
  return !stop_;
}

WeightStream::Slot *WeightStream::qkfixSlotOf(int64_t n) {
  const int rank = qkfixRank_[static_cast<size_t>(n % blocksPerForward_)];
  if (rank < 0) return nullptr;
  const int64_t q = (n / blocksPerForward_) * qkfixPerForward_ + rank;
  return &qkfixSlots_[static_cast<size_t>(q % static_cast<int64_t>(qkfixSlots_.size()))];
}

void WeightStream::reader() {
  for (int64_t n = 0; !stop_; ++n) {
    const int block = static_cast<int>(n % blocksPerForward_);
    Slot &slot = slots_[static_cast<size_t>(n % static_cast<int64_t>(slots_.size()))];
    if (!waitFree(slot.free)) break;
    Slot *qkfix = qkfixSlotOf(n);
    if (qkfix && !waitFree(qkfix->free)) break;
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = true;
    int64_t bytes = 0;
    for (int t = 0; t < kTensors && ok; ++t) {
      ok = readInto(slot.tensor[t].contents, tensorOffset(block, static_cast<Tensor>(t)), header_.tensorBytes[t]);
      bytes += static_cast<int64_t>(header_.tensorBytes[t]);
    }
    if (ok && qkfix) {
      uint64_t offset = 0;
      ok = qkfixOffset(block, offset).ok() && readInto(qkfix->tensor[0].contents, offset, qkfix->tensor[0].length);
      bytes += static_cast<int64_t>(qkfix->tensor[0].length);
    }
    if (!ok) {
      fprintf(stderr, "h3: stream image read error, block %d\n", block);
      stop_ = true;
    }
    const int64_t us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    readBytes_ += bytes;
    readUs_ += us;
    blocksRead_ += 1;
    if (us > readMaxUs_) readMaxUs_ = us;
    dispatch_semaphore_signal(slot.ready);
  }
}

WeightStream::Slot &WeightStream::acquire(Slot **qkfix) {
  const int64_t n = encoderSequence_++;
  Slot &slot = slots_[static_cast<size_t>(n % static_cast<int64_t>(slots_.size()))];
  const auto t0 = std::chrono::steady_clock::now();
  dispatch_semaphore_wait(slot.ready, DISPATCH_TIME_FOREVER);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  waitMs_ += ms;
  waitMaxMs_ = std::max(waitMaxMs_, ms);
  if (ms > 1.0) ++waits_;
  *qkfix = qkfixSlotOf(n);
  return slot;
}

void WeightStream::resetStats() {
  readBytes_ = 0;
  readUs_ = 0;
  readMaxUs_ = 0;
  blocksRead_ = 0;
  waitMs_ = waitMaxMs_ = 0;
  waits_ = 0;
}

StreamStats WeightStream::stats() const {
  StreamStats s;
  s.waitMs = waitMs_;
  s.waitMaxMs = waitMaxMs_;
  s.waits = waits_;
  s.readGb = readBytes_ / 1e9;
  s.readS = readUs_ / 1e6;
  s.readMaxMs = readMaxUs_ / 1e3;
  s.blocksRead = static_cast<int32_t>(blocksRead_.load());
  return s;
}

size_t WeightStream::slotBytes() const {
  size_t total = 0;
  for (const Slot &slot : slots_)
    for (int t = 0; t < kTensors; ++t) total += slot.tensor[t].length;
  for (const Slot &slot : qkfixSlots_) total += slot.tensor[0].length;
  return total;
}

} // namespace h3
