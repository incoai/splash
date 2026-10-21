#pragma once

// Key-range table of the VDN-H3 window softmax (OpenVDN softmax_attention: keep (q, k) unless both are video and the
// key frame is outside window_bounds(F, radius, chunk)[query frame]; anchor frames 0 and F - 1 are keys of every
// video query and their queries are dense). Every row outside the video (text, audio) is global.
//
// int32 little endian. Header record (32 ints): magic 'VDKT', version 1, S, nqb, chunk, radius, anchors code 3 (both),
// text, cond 0, audio, F, T, pad 0, video start, video end. Then one record of 32 ints per 16-query block:
//   [0] union ranges  [1] mixed  [2] split row (rows < split use segment 0)  [3] [4] ranges of segments 0 / 1
//   [8..13] union ranges (k0, k1) x 3   [14..19] segment-0 ranges x 3   [20..25] segment-1 ranges x 3
// Ranges are half-open packed-row intervals, sorted and merged; non-mixed blocks repeat the union in both segments.
// Identical to dev/tools/keytable.py emit (checked by dev/tests/table_check.py against OpenVDN's mask).
//
// Work items for the window kernel: 64-row query groups whose 16-row blocks all see >= kDenseKeyMin keys are split
// into kKeyChunk-key items writing partials (merged by dit_window_combine); window items first.
#include "video/h3/engine/Encoder.hpp"
#include "metal/abi/video/AttentionABI.h"
#include "video/h3/model/Geometry.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace h3 {

class WindowTable {
public:
  // Throws std::invalid_argument for a layout the kernels do not cover.
  WindowTable(const model::SequenceLayout &layout, int32_t chunk, int32_t radius);

  [[nodiscard]] const std::vector<int32_t> &words() const { return words_; }
  [[nodiscard]] int32_t blocks() const { return nqb_; }
  [[nodiscard]] double density() const { return density_; }  // allowed (q, k) pairs / S^2
  [[nodiscard]] int32_t chunk() const { return chunk_; }
  [[nodiscard]] int32_t radius() const { return radius_; }
  // Linear-branch window bounds of the inner frames, rebased to inner indices (OpenVDN window_bounds, inclusive).
  [[nodiscard]] std::vector<std::array<int32_t, 2>> branchBounds() const;

  // Work items and the GPU copies of table, items and chunked groups (qg, slot0, chunks, 0).
  void upload(metal::MetalBackend &backend);
  [[nodiscard]] const std::vector<WindowItem> &items() const { return items_; }
  [[nodiscard]] int32_t partialSlots() const { return slots_; }
  [[nodiscard]] int32_t chunkedGroups() const { return static_cast<int32_t>(groups_.size()); }
  [[nodiscard]] const metal::MetalBuffer &tableBuffer() const { return table_; }
  [[nodiscard]] const metal::MetalBuffer &itemBuffer() const { return itemBuffer_; }
  [[nodiscard]] const metal::MetalBuffer &groupBuffer() const { return groupBuffer_; }

private:
  void buildItems();

  std::vector<int32_t> words_;
  int32_t rows_ = 0, nqb_ = 0, chunk_ = 0, radius_ = 0, frames_ = 0;
  double density_ = 0;
  std::vector<WindowItem> items_;
  std::vector<std::array<int32_t, 4>> groups_;
  int32_t slots_ = 0;
  metal::MetalBuffer table_, itemBuffer_, groupBuffer_;
};

} // namespace h3
