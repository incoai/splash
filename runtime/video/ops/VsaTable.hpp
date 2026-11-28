#pragma once

// Tile map of FastH3's Video Sparse Attention for one request (FastVideo's VSA-H3, metal/abi/video/VsaABI.h). The
// packed sequence [text L | conditioning C | audio conditioning Ca | audio Na | video F x G_H x G_W] is cut into
// tiles: the target video's latent grid into 4 x 4 x 4 cubes (t, h, w; smaller at the grid edges of any axis: 768p
// 24 x 42 and 480p 15 x 26 / 15 x 27 grids end in partial cubes), each dense prefix segment (SequenceLayout::
// prefixSegments: text, the keyframes' or each reference's rows, audio) into 64-row chunks of its own (FastVideo
// _h3_segment_tile_geometry of the segments (L, C, Na, video)).
// Inside the blocks the rows are stored in tile order: the video tiles in (t, h, w) tile order, the rows of a tile in
// (t, h, w) order, then the prefix tiles in packed order. Every per-row operation is order independent; the embedding
// and the final layer run in packed order around a gather (order) and its inverse. Video queries keep every prefix
// tile and their top k = ceil((1 - sparsity) V) video tiles (FastVideo compute_topk, the prefix keys "exempt");
// prefix queries, keyframe rows among them, attend to every row.
#include "video/metal/Encoder.hpp"
#include "video/ops/Geometry.hpp"

#include <cstdint>
#include <vector>

namespace splash::video::ops {

struct VsaSpec {
  int32_t tileT = 4, tileH = 4, tileW = 4;  // cube of latent tokens (FastH3: 64-token tiles)
  double sparsity = 0.0;                     // fraction of the video tiles each video query drops
  // Dense attention (the manifest's vsa.attention "dense": a DiT without the compression gate, sparsity 0, such as
  // pika-6's base): every query attends to every row; no gate rows, coarse stage or selection run, and the fine
  // stage's queries are the consecutive 64-row chunks of the tile order (denseTileBuffer), all full but the last.
  bool dense = false;
};

class VsaTable {
public:
  // Throws std::invalid_argument for a spec or layout the kernels do not cover.
  VsaTable(const SequenceLayout &layout, const VsaSpec &spec);

  [[nodiscard]] int32_t tiles() const { return static_cast<int32_t>(tiles_.size() / 2); }
  [[nodiscard]] int32_t paddedTiles() const { return (tiles() + 31) / 32 * 32; }  // coarse-stage matrices
  [[nodiscard]] int32_t videoTiles() const { return videoTiles_; }
  [[nodiscard]] int32_t videoRows() const { return videoRows_; }
  [[nodiscard]] int32_t rows() const { return static_cast<int32_t>(order_.size()); }
  [[nodiscard]] int32_t topk() const { return topk_; }
  [[nodiscard]] const VsaSpec &spec() const { return spec_; }
  // fraction of (query, key) pairs the fine stage computes
  [[nodiscard]] double density() const { return density_; }
  [[nodiscard]] const std::vector<int32_t> &order() const { return order_; }      // tile-order row -> packed row
  [[nodiscard]] const std::vector<int32_t> &inverse() const { return inverse_; }  // packed row -> tile-order row
  [[nodiscard]] const std::vector<int32_t> &tileWords() const { return tiles_; }  // [T][2] start, rows
  [[nodiscard]] const std::vector<int32_t> &rowTile() const { return rowTile_; }  // [rows] tile of a tile-order row
  [[nodiscard]] const std::vector<int32_t> &items() const { return items_; }      // attention items: prefix tiles first

  // The GPU copies of order, inverse, tiles, row tiles and items.
  void upload(metal::MetalBackend &backend);
  [[nodiscard]] const metal::MetalBuffer &orderBuffer() const { return orderBuffer_; }
  [[nodiscard]] const metal::MetalBuffer &inverseBuffer() const { return inverseBuffer_; }
  [[nodiscard]] const metal::MetalBuffer &tileBuffer() const { return tileBuffer_; }
  [[nodiscard]] const metal::MetalBuffer &rowTileBuffer() const { return rowTileBuffer_; }
  [[nodiscard]] const metal::MetalBuffer &itemBuffer() const { return itemBuffer_; }
  // Dense attention: the query chunks [64 i, 64 i + 64) of the tile order (int2 start, rows) and their items.
  [[nodiscard]] const metal::MetalBuffer &denseTileBuffer() const { return denseTileBuffer_; }
  [[nodiscard]] const metal::MetalBuffer &denseItemBuffer() const { return denseItemBuffer_; }
  [[nodiscard]] int32_t denseItems() const { return static_cast<int32_t>(denseItems_.size()); }

  // Tile order of a packed-order per-row array (tags, RoPE rows of `width` elements).
  template <class T>
  void permute(const T *packed, T *tileOrder, size_t width) const {
    for (size_t i = 0; i < order_.size(); ++i)
      for (size_t c = 0; c < width; ++c) tileOrder[i * width + c] = packed[size_t(order_[i]) * width + c];
  }

private:
  VsaSpec spec_;
  int32_t videoTiles_ = 0, videoRows_ = 0, topk_ = 0;
  double density_ = 0;
  std::vector<int32_t> order_, inverse_, tiles_, rowTile_, items_, denseTiles_, denseItems_;
  metal::MetalBuffer orderBuffer_, inverseBuffer_, tileBuffer_, rowTileBuffer_, itemBuffer_, denseTileBuffer_,
      denseItemBuffer_;
};

} // namespace splash::video::ops
