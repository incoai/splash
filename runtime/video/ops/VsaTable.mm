#include "video/ops/VsaTable.hpp"

#include "metal/abi/video/VsaABI.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>
#include <utility>

namespace splash::video::ops {

VsaTable::VsaTable(const SequenceLayout &lay, const VsaSpec &spec) : spec_(spec) {
  if (spec.tileT * spec.tileH * spec.tileW != H3_VSA_TILE)
    throw std::invalid_argument(
        std::format("VSA: tile {}x{}x{}, the kernels take 64-token tiles", spec.tileT, spec.tileH, spec.tileW));
  if (!(spec.sparsity >= 0.0 && spec.sparsity < 1.0)) throw std::invalid_argument("VSA: sparsity must be in [0, 1)");
  if (spec.dense && spec.sparsity != 0.0) throw std::invalid_argument("VSA: dense attention at sparsity 0 only");
  const int32_t F = lay.frames, GH = lay.gridH, GW = lay.gridW, M = lay.rows();
  const int32_t v0 = lay.videoStart();
  order_.reserve(M);
  auto tile = [&](int32_t start, int32_t n) {
    tiles_.push_back(start);
    tiles_.push_back(n);
  };
  // video cubes in (t, h, w) tile order, rows inside a cube in (t, h, w) order (FastVideo tile_partition_indices)
  for (int32_t f0 = 0; f0 < F; f0 += spec.tileT)
    for (int32_t y0 = 0; y0 < GH; y0 += spec.tileH)
      for (int32_t x0 = 0; x0 < GW; x0 += spec.tileW) {
        const int32_t start = static_cast<int32_t>(order_.size());
        for (int32_t f = f0; f < std::min(F, f0 + spec.tileT); ++f)
          for (int32_t y = y0; y < std::min(GH, y0 + spec.tileH); ++y)
            for (int32_t x = x0; x < std::min(GW, x0 + spec.tileW); ++x)
              order_.push_back(v0 + (f * GH + y) * GW + x);
        tile(start, static_cast<int32_t>(order_.size()) - start);
      }
  videoTiles_ = static_cast<int32_t>(tiles_.size() / 2);
  videoRows_ = static_cast<int32_t>(order_.size());
  // dense prefix: each segment (text, keyframe conditioning, audio) in 64-row chunks of its own
  for (auto [s0, n] : lay.prefixSegments())
    for (int32_t r = 0; r < n; r += H3_VSA_TILE) {
      const int32_t start = static_cast<int32_t>(order_.size());
      for (int32_t i = r; i < std::min(n, r + H3_VSA_TILE); ++i) order_.push_back(s0 + i);
      tile(start, static_cast<int32_t>(order_.size()) - start);
    }
  if (static_cast<int32_t>(order_.size()) != M) throw std::invalid_argument("VSA: the tile map misses rows");
  const int32_t T = tiles();
  if (T > H3_VSA_MAX_TILES) throw std::invalid_argument(std::format("VSA: {} tiles, at most {}", T, H3_VSA_MAX_TILES));
  inverse_.assign(M, -1);
  for (int32_t i = 0; i < M; ++i) inverse_[order_[i]] = i;
  rowTile_.assign(M, 0);
  for (int32_t j = 0; j < T; ++j)
    for (int32_t r = 0; r < tiles_[2 * j + 1]; ++r) rowTile_[tiles_[2 * j] + r] = j;
  // FastVideo compute_topk: max(1, min(ceil((1 - sparsity) * V), V)), in double as Python
  const int32_t V = videoTiles_;
  topk_ = std::max(1, std::min(static_cast<int32_t>(std::ceil((1.0 - spec.sparsity) * V)), V));
  // items: prefix tiles (dense, longest) first, then the video tiles
  for (int32_t j = V; j < T; ++j) items_.push_back(j);
  for (int32_t j = 0; j < V; ++j) items_.push_back(j);
  // density: prefix rows x M + video rows x (prefix rows + kept video rows on average)
  const double prefix = M - videoRows_, keptVideo = double(videoRows_) * topk_ / V;
  density_ = (prefix * M + double(videoRows_) * (prefix + keptVideo)) / (double(M) * M);
  if (spec.dense) {
    for (int32_t r = 0; r < M; r += H3_VSA_TILE) {
      denseItems_.push_back(static_cast<int32_t>(denseTiles_.size() / 2));
      denseTiles_.push_back(r);
      denseTiles_.push_back(std::min(H3_VSA_TILE, M - r));
    }
    density_ = 1.0;
  }
}

void VsaTable::upload(metal::MetalBackend &backend) {
  orderBuffer_ = video::upload(backend, order_.data(), order_.size() * 4, "vsa order");
  inverseBuffer_ = video::upload(backend, inverse_.data(), inverse_.size() * 4, "vsa inverse order");
  tileBuffer_ = video::upload(backend, tiles_.data(), tiles_.size() * 4, "vsa tiles");
  rowTileBuffer_ = video::upload(backend, rowTile_.data(), rowTile_.size() * 4, "vsa row tiles");
  itemBuffer_ = video::upload(backend, items_.data(), items_.size() * 4, "vsa items");
  if (spec_.dense) {
    denseTileBuffer_ = video::upload(backend, denseTiles_.data(), denseTiles_.size() * 4, "dense attention chunks");
    denseItemBuffer_ = video::upload(backend, denseItems_.data(), denseItems_.size() * 4, "dense attention items");
  }
}

} // namespace splash::video::ops
