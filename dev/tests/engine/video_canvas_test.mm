// MiniMax-H3's canvas tiers, on the CPU (video/ops/Geometry.hpp): the canvases a tier admits and those refused, and,
// for every named canvas and clip length, what the request derives from the token grid: the geometry, VDN-H3's
// key-range table, FastH3's VSA tile map (partial cubes on every axis, row for row against FastVideo's geometry), the
// video VAE's decode plan, the TAEH3 preview's frames and the DiT's activation plan. The named canvases' counts are
// written out (768p 1344 x 768: 1,008 rows per latent frame; 480p 832 x 480: 390), the rest follow from the grid.
#include "FastVideoTiles.hpp"
#include "TestChecks.hpp"
#include "metal/abi/video/Layout.h"
#include "metal/abi/video/VsaABI.h"
#include "video/model/MemoryPlan.hpp"
#include "video/ops/Geometry.hpp"
#include "video/ops/VsaTable.hpp"
#include "video/ops/WindowTable.hpp"
#include "video/stages/DenoisingStage.hpp"

#include <array>
#include <format>
#include <iostream>
#include <string>
#include <vector>

namespace {

using splash::test::require;
using splash::video::test::fastVideoTiles;
using namespace splash::video;

struct Canvas {
  int32_t width, height, tier, gridW, gridH, vaeTilesY, vaeTilesX;
};
// The named canvases: 768p 16:9 (today's) and 9:16; 480p 832 x 480 (FastVideo's), 864 x 480 (mlx-serve's), 480 x 832.
constexpr Canvas kCanvases[] = {
    {1344, 768, 768, 42, 24, 4, 7}, {768, 1344, 768, 24, 42, 7, 4}, {832, 480, 480, 26, 15, 3, 4},
    {864, 480, 480, 27, 15, 3, 5},  {480, 832, 480, 15, 26, 4, 3},
};
struct Clip {
  int32_t seconds, frames, latentFrames, audioRows;
};
// 3 s (the golden clips'), then the product's 5, 10 and 15 s.
constexpr Clip kClips[] = {{3, 73, 22, 244}, {5, 124, 37, 414}, {10, 243, 72, 810}, {15, 362, 107, 1206}};
constexpr int32_t kTextRows = 32;

// FastH3's tiles of a canvas and clip, written out: video tiles, partial ones, top k at sparsity 0.8 and 0.9.
struct TileCounts {
  int32_t width, height, frames, videoTiles, partial, topk8, topk9;
};
constexpr TileCounts kTileCounts[] = {
    {1344, 768, 73, 396, 96, 80, 40},   {1344, 768, 362, 1782, 222, 357, 179}, {768, 1344, 73, 396, 96, 80, 40},
    {832, 480, 73, 168, 78, 34, 17},    {832, 480, 124, 280, 118, 56, 28},     {832, 480, 243, 504, 180, 101, 51},
    {864, 480, 124, 280, 118, 56, 28},  {864, 480, 243, 504, 180, 101, 51},    {480, 832, 124, 280, 118, 56, 28},
};

int32_t ceilDiv(int32_t a, int32_t b) { return (a + b - 1) / b; }

ops::SequenceLayout layoutOf(const ops::VideoGeometry &g, int32_t keyframes, bool vsa) {
  ops::SequenceLayout lay;
  lay.text = kTextRows;
  lay.cond = keyframes * g.tokensPerFrame;
  lay.audio = g.audioRows;
  lay.frames = g.latentFrames;
  lay.tokensPerFrame = g.tokensPerFrame;
  lay.gridH = g.gridH;
  lay.gridW = g.gridW;
  lay.tailPad = vsa ? H3_VSA_TILE : 0;
  return lay;
}

void checkRefusals() {
  for (auto [w, h, tier] : std::vector<std::array<int32_t, 3>>{{1344, 768, 768}, {768, 1344, 768},  {768, 768, 768},
                                                                {1536, 768, 768}, {832, 480, 480},   {864, 480, 480},
                                                                {480, 832, 480},  {640, 480, 480},   {960, 480, 480},
                                                                {1280, 720, 0},   {1920, 1080, 0},   {1024, 576, 0},
                                                                {1600, 768, 0},   {992, 480, 0},     {832, 464, 0},
                                                                {848, 480, 0},    {0, 0, 0},         {-768, 1344, 0}}) {
    require(ops::canvasTier(w, h) == tier, std::format("canvas {} x {}: tier {}, not {}", w, h,
                                                       ops::canvasTier(w, h), tier));
    require(ops::canvasRefusal(w, h).empty() == (tier != 0), std::format("canvas {} x {}: refusal", w, h));
  }
  // 241 frames is not 17 n + 5: the 10 s clip is 243
  splash::test::rejects([] { (void)ops::VideoGeometry::of(241, 1344, 768); }, "17 n + 5", "241 frames were taken");
}

void checkVsa(const ops::SequenceLayout &lay, const Canvas &c, int32_t frames, int32_t &videoTiles, int32_t &partial,
              int32_t &topk8, int32_t &topk9) {
  const ops::VsaTable t8(lay, {4, 4, 4, 0.8}), t9(lay, {4, 4, 4, 0.9});
  const std::vector<std::vector<int32_t>> want = fastVideoTiles(lay);
  const int32_t prefix = static_cast<int32_t>(want.size()) - t8.videoTiles();
  require(t8.videoTiles() == ceilDiv(lay.frames, 4) * ceilDiv(c.gridH, 4) * ceilDiv(c.gridW, 4) &&
              t8.tiles() == static_cast<int32_t>(want.size()) && t8.tiles() <= H3_VSA_MAX_TILES &&
              t8.rows() == lay.rows() && t8.videoRows() == lay.frames * lay.tokensPerFrame,
          std::format("{} x {} {} frames: VSA tile counts", c.width, c.height, frames));
  for (int32_t j = 0; j < t8.tiles(); ++j) {
    const std::vector<int32_t> &rows = want[size_t(j < t8.videoTiles() ? prefix + j : j - t8.videoTiles())];
    const int32_t start = t8.tileWords()[2 * j], count = t8.tileWords()[2 * j + 1];
    require(count == static_cast<int32_t>(rows.size()), "a tile's row count differs from FastVideo's");
    for (int32_t r = 0; r < count; ++r)
      require(t8.order()[size_t(start + r)] == rows[size_t(r)] && t8.rowTile()[size_t(start + r)] == j,
              "a tile's rows differ from FastVideo's");
  }
  partial = 0;
  for (int32_t j = 0; j < t8.videoTiles(); ++j) partial += t8.tileWords()[2 * j + 1] < H3_VSA_TILE;
  // partial cubes: every tile on a short temporal, row or column edge
  const int32_t fullT = lay.frames / 4, fullH = c.gridH / 4, fullW = c.gridW / 4;
  require(partial == t8.videoTiles() - fullT * fullH * fullW, "partial tiles are not the grid's edge cubes");
  videoTiles = t8.videoTiles();
  topk8 = t8.topk();
  topk9 = t9.topk();
}

void checkCanvas(const Canvas &c) {
  require(ops::canvasTier(c.width, c.height) == c.tier, "a named canvas is outside its tier");
  for (const Clip &clip : kClips) {
    const ops::VideoGeometry g = ops::VideoGeometry::of(clip.frames, c.width, c.height);
    const std::string name = std::format("{} x {} {} s", c.width, c.height, clip.seconds);
    require(g.gridW == c.gridW && g.gridH == c.gridH && g.tokensPerFrame == c.gridW * c.gridH &&
                g.latentWidth == 2 * c.gridW && g.latentHeight == 2 * c.gridH && g.latentFrames == clip.latentFrames &&
                g.audioRows == clip.audioRows && g.videoRows() == clip.latentFrames * c.gridW * c.gridH,
            name + ": geometry");
    // VDN-H3: the key-range table of text-to-video and of one and two keyframes (c5r1 is temporal: whole frames)
    for (int32_t keyframes = 0; keyframes <= 2; ++keyframes) {
      const ops::SequenceLayout lay = layoutOf(g, keyframes, false);
      const ops::WindowTable table(lay, kWindowChunk, kWindowRadius);
      require(table.blocks() == ceilDiv(lay.rows(), H3_TABLE_ROWS) && table.density() > 0 && table.density() <= 1 &&
                  table.branchBounds().size() == size_t(lay.innerFrames()),
              name + ": key-range table");
    }
    const ops::SequenceLayout vdn = layoutOf(g, 0, false), fast = layoutOf(g, 0, true);
    int32_t videoTiles = 0, partial = 0, topk8 = 0, topk9 = 0;
    checkVsa(fast, c, clip.frames, videoTiles, partial, topk8, topk9);
    checkVsa(layoutOf(g, 2, true), c, clip.frames, videoTiles, partial, topk8, topk9);
    for (const TileCounts &t : kTileCounts)
      if (t.width == c.width && t.height == c.height && t.frames == clip.frames)
        require(videoTiles == t.videoTiles && partial == t.partial && topk8 == t.topk8 && topk9 == t.topk9,
                std::format("{}: {} video tiles ({} partial), top k {} / {}", name, videoTiles, partial, topk8, topk9));
    // the video VAE's 256 px tiles (latent-aligned overlaps, planVaeDecode throws otherwise) and the clip's frames
    const ops::VaePlan vae = ops::planVaeDecode(g);
    require(static_cast<int32_t>(vae.ys.size()) == c.vaeTilesY && static_cast<int32_t>(vae.xs.size()) == c.vaeTilesX &&
                vae.ys.back() + 256 == c.height && vae.xs.back() + 256 == c.width &&
                c.vaeTilesY <= H3_VAE_MAX_TILES && c.vaeTilesX <= H3_VAE_MAX_TILES &&
                vae.outputFrames() == clip.frames,
            name + ": video VAE plan");
    // activations: VDN-H3's (window softmax + linear branch) and FastH3's (VSA)
    const ops::WindowTable table(vdn, kWindowChunk, kWindowRadius);
    const ops::VsaTable vsa(fast, {4, 4, 4, 0.8});
    const double vdnGb = model::MemoryPlan::bytesFor(vdn, table.partialSlots()) / 1e9;
    const double fastGb = model::MemoryPlan::bytesFor(fast, 0, &vsa) / 1e9;
    std::cout << std::format("  {:<16} grid {:>2} x {:<2} = {:>4} rows/frame  F {:>3}  DiT rows {:>6}  VSA {:>4} tiles "
                             "({:>3} partial) top k {:>3} / {:<3}  VAE {} x {}  preview {:>3} frames  "
                             "activations VDN {:5.2f} GB, FastH3 {:5.2f} GB\n",
                             name, c.gridW, c.gridH, g.tokensPerFrame, g.latentFrames, vdn.rows(), videoTiles,
                             partial, topk8, topk9, c.vaeTilesY, c.vaeTilesX, ops::previewFrames(g.latentFrames),
                             vdnGb, fastGb);
  }
}

} // namespace

int main() {
  try {
    checkRefusals();
    for (const Canvas &c : kCanvases) checkCanvas(c);
    // a 9:16 canvas is its 16:9 twin's rows, transposed
    const ops::VideoGeometry wide = ops::VideoGeometry::of(124, 1344, 768);
    const ops::VideoGeometry tall = ops::VideoGeometry::of(124, 768, 1344);
    require(wide.videoRows() == tall.videoRows() &&
                model::MemoryPlan::bytesFor(layoutOf(wide, 0, false), 0) ==
                    model::MemoryPlan::bytesFor(layoutOf(tall, 0, false), 0),
            "9:16 and 16:9 plans differ");
    std::cout << "video_canvas_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "video_canvas_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
