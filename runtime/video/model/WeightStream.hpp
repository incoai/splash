#pragma once

// SSD weight streaming: the five int8 matrices of a block (qkv + gates, out, fc1, fc2, to_out_linear; 425 MB) are not
// resident. Splash's SSD readers (model/SsdReaders.hpp) read them from the block's own file
// (dit-<variant>/vdn-w8-blockNN.safetensors, opened without the buffer cache like every weight source) into a ring of
// block-sized slots ahead of the GPU, in an endless block sequence n = 0, 1, 2, ... (block n % 50 into slot n % K;
// forwards run back to back, so the next forward's first blocks are read while the current one finishes). QKFIX
// blocks' fp16 q / k / v weights (231 MB each, qkfix-blockNN.safetensors) use a second ring.
// A slot's fill is a background batch of the readers, started once the command of the block that used the slot before
// has completed (release); the encoder finishes it before it submits the block (waitRead).
// Splash's model::ExpertStore streams routed experts into slots it picks by use; the DiT reads every block in order,
// so the ring keeps its own slots.
#include "model/SsdReaders.hpp"
#include "video/metal/Encoder.hpp"
#include "video/model/DitWeights.hpp"

#include <cstdint>
#include <memory>
#include <set>
#include <vector>

namespace splash::video::model {

struct StreamStats {
  double waitMs = 0;   // encoder waits for a slot to be read
  int32_t waits = 0;   // waits > 1 ms
  double readGb = 0;
};

class WeightStream {
public:
  enum Tensor { Qkvg = 0, Out, Fc1, Fc2, OutLinear, kTensors };

  struct Slot {
    metal::MetalBuffer tensor[kTensors];
    int64_t fill = 0;  // the position in its ring's sequence it holds
    std::shared_ptr<splash::model::SsdReaders::Batch> read;
  };

  // The tensors DitWeights left on disk (DitWeights::streamed), which must stay open while the stream runs; starts
  // reading the first blocks of forwards of all 50 blocks, the QKFIX blocks among them.
  WeightStream(metal::MetalBackend &backend, const std::vector<StreamedTensor> &tensors,
               const std::set<int> &qkfixBlocks);
  ~WeightStream();
  WeightStream(const WeightStream &) = delete;
  WeightStream &operator=(const WeightStream &) = delete;

  // Encoder: the slots of the next block in the sequence (qkfix null for other blocks), which its command may bind
  // once waitRead returned; release hands them back once the command completed. waitRead throws a read's failure.
  Slot &next(Slot **qkfix);
  void waitRead(Slot &slot, Slot *qkfix);
  void release(Slot &slot, Slot *qkfix);
  void resetStats();
  [[nodiscard]] StreamStats stats() const;
  [[nodiscard]] int slots() const { return static_cast<int>(slots_.size()); }
  [[nodiscard]] size_t slotBytes() const;

private:
  // A piece of a fill: bytes [at, at + bytes) of a tensor, to `offset` of its slot tensor (-1: the QKFIX slot's).
  struct Piece {
    const splash::model::SourceTensor *tensor;
    uint64_t at, bytes;
    int slotTensor;
    uint64_t offset;
  };

  void fill(Slot &slot, int64_t n, bool qkfix);
  void wait(Slot &slot);

  std::vector<std::vector<Piece>> pieces_, qkfixPieces_;  // block -> its pieces
  std::vector<int> qkfixRank_, qkfixOrder_;  // block -> rank among the QKFIX blocks (-1 none), rank -> block
  std::vector<Slot> slots_, qkfixSlots_;
  splash::model::SsdReaders readers_;
  int64_t encoderSequence_ = 0;
  uint64_t readBytes_ = 0;
  double waitMs_ = 0;
  int32_t waits_ = 0;
};

} // namespace splash::video::model
