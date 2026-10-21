#pragma once

// SSD weight streaming: the five int8 matrices of a block (qkv + gates, out, fc1, fc2, to_out_linear; 425 MB) are not
// resident. They live in one contiguous per-block image (tools/convert_dit.py stream) and a reader thread reads them
// into a ring of K block-sized slots ahead of the GPU, in an endless block sequence n = 0, 1, 2, ... (block n % 50 into
// slot n % K; forwards run back to back, so the next forward's first blocks are read while the current one finishes).
// QKFIX blocks' fp16 q / k / v weights (231 MB each) use a second ring.
// Per slot: `ready` (reader -> encoder, after the read) and `free` (encoder -> reader, once the command of the block
// that used the slot has completed).
//
// Image (little endian): magic "H3WSTRM1", u32 version 1, u32 blocks, u64 block bytes, u32 tensors (5), u32 QKFIX
// blocks, u64 blocks offset, u64 QKFIX offset, u64 QKFIX stride, u64 tensor offsets[5], u64 tensor bytes[5], then
// QKFIX entries (i32 block, i32 0, u64 offset); tensors 16 KiB aligned.
#include <dispatch/dispatch.h>

#include "video/h3/engine/Encoder.hpp"
#include "video/h3/model/WeightFiles.hpp"

#include <atomic>
#include <cstdint>
#include <exception>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace h3 {

struct StreamStats {
  double waitMs = 0, waitMaxMs = 0;  // encoder waits for a slot to be read
  int32_t waits = 0;                 // waits > 1 ms
  double readGb = 0, readS = 0, readMaxMs = 0;
  int32_t blocksRead = 0;
};

class WeightStream {
public:
  enum Tensor { Qkvg = 0, Out, Fc1, Fc2, OutLinear, kTensors };

  struct Slot {
    metal::MetalBuffer tensor[kTensors];
    dispatch_semaphore_t ready = nullptr, free = nullptr;
  };

  WeightStream(metal::MetalBackend &backend, const std::string &image, int slots);
  ~WeightStream();
  WeightStream(const WeightStream &) = delete;
  WeightStream &operator=(const WeightStream &) = delete;

  // Image offsets of a block's tensor and of a QKFIX block's fp16 weights.
  [[nodiscard]] uint64_t tensorOffset(int block, Tensor t) const;
  [[nodiscard]] uint64_t tensorBytes(Tensor t) const { return header_.tensorBytes[t]; }
  [[nodiscard]] int blocks() const { return static_cast<int>(header_.blocks); }
  [[nodiscard]] uint64_t qkfixOffset(int block) const;
  // First / last kStreamVerifyBytes of the tensor vs the same bytes at `imageOffset`.
  void verify(const model::SourceTensor &tensor, uint64_t imageOffset, const char *what, int block) const;

  // Starts the reader for `blocks` per forward and the QKFIX blocks among them.
  void start(metal::MetalBackend &backend, int blocks, const std::set<int> &qkfixBlocks);
  // Encoder: the slots of the next block in the sequence (qkfix null for other blocks), which its command may bind
  // once waitRead returned; release hands them back once the command completed. waitRead throws the reader's failure.
  Slot &next(Slot **qkfix);
  void waitRead(Slot &slot);
  static void release(Slot &slot, Slot *qkfix);
  void resetStats();
  [[nodiscard]] StreamStats stats() const;
  [[nodiscard]] int slots() const { return static_cast<int>(slots_.size()); }
  [[nodiscard]] size_t slotBytes() const;

private:
  struct Header {
    char magic[8];
    uint32_t version, blocks;
    uint64_t blockBytes;
    uint32_t tensors, qkfixBlocks;
    uint64_t blocksOffset, qkfixOffset, qkfixStride;
    uint64_t tensorOffset[kTensors], tensorBytes[kTensors];
  };
  struct QkFixEntry {
    int32_t block, pad;
    uint64_t offset;
  };

  bool waitFree(dispatch_semaphore_t semaphore);
  Slot *qkfixSlotOf(int64_t n);
  void reader();

  model::WeightSource image_;
  Header header_{};
  std::vector<QkFixEntry> qkfixEntries_;
  std::vector<Slot> slots_, qkfixSlots_;
  std::vector<int> qkfixRank_;  // block -> rank among the streamed QKFIX blocks, -1 = none
  int blocksPerForward_ = 0, qkfixPerForward_ = 0;
  std::thread thread_;
  std::atomic<bool> stop_{false};
  std::exception_ptr failure_;      // the reader's, set before failed_
  std::atomic<bool> failed_{false};
  int64_t encoderSequence_ = 0;
  std::atomic<int64_t> readBytes_{0}, readUs_{0}, readMaxUs_{0}, blocksRead_{0};
  double waitMs_ = 0, waitMaxMs_ = 0;
  int32_t waits_ = 0;
};

} // namespace h3
