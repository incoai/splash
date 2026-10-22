#pragma once

// SSD weight streaming: the five int8 matrices of a block (qkv + gates, out, fc1, fc2, to_out_linear; 425 MB) are not
// resident. A reader thread reads them from the block's own file (dit-<variant>/vdn-w8-blockNN.safetensors, opened
// without the buffer cache like every weight source) into a ring of K block-sized slots ahead of the GPU, in an
// endless block sequence n = 0, 1, 2, ... (block n % 50 into slot n % K; forwards run back to back, so the next
// forward's first blocks are read while the current one finishes). QKFIX blocks' fp16 q / k / v weights (231 MB each,
// qkfix-blockNN.safetensors) use a second ring.
// Per slot: `ready` (reader -> encoder, after the read) and `free` (encoder -> reader, once the command of the block
// that used the slot has completed).
#include <dispatch/dispatch.h>

#include "video/h3/engine/Encoder.hpp"
#include "video/h3/model/DitWeights.hpp"

#include <atomic>
#include <cstdint>
#include <exception>
#include <set>
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

  // The tensors DitWeights left on disk (DitWeights::streamed), which must stay open while the stream runs.
  WeightStream(metal::MetalBackend &backend, const std::vector<model::StreamedTensor> &tensors, int slots);
  ~WeightStream();
  WeightStream(const WeightStream &) = delete;
  WeightStream &operator=(const WeightStream &) = delete;

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
  bool waitFree(dispatch_semaphore_t semaphore);
  Slot *qkfixSlotOf(int64_t n);
  void reader();

  std::vector<std::vector<model::StreamedTensor>> blockTensors_;  // block -> its streamed tensors
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
