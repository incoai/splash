// FastH3's request tables, on the CPU: the VSA tile map (video/ops/VsaTable.hpp) of a 3 s clip, text-to-video and
// with one and two keyframes, against FastVideo's VSA-H3 geometry (FastVideoTiles.hpp: _h3_segment_tile_geometry of the
// segments (L, C, Na, video): each dense segment in 64-row chunks, the video in 4 x 4 x 4 cubes of rows in (t, h, w)
// order) and a reference implementation's 3 s tile counts; and the trained-rung schedules (model::Prep::scheduleRungs)
// against the adaLN timesteps a reference conversion of the students records (manifest.json of dit-fasth3-8 and
// dit-fasth3-4); and the tile map of a Ref2VA sequence (video/ops/RowLayout.hpp: a Ref2VA example's references, a
// video and two images, at 832 x 480) in the engine's row order.
#include "FastVideoTiles.hpp"
#include "TestChecks.hpp"
#include "video/model/FastH3.hpp"
#include "video/model/Prep.hpp"
#include "video/ops/RowLayout.hpp"
#include "video/ops/VsaTable.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

using splash::test::require;
using splash::video::test::fastVideoTiles;
using namespace splash::video;

void checkTiles(int32_t keyframes, double sparsity, int32_t topk, int32_t tiles) {
  const ops::VideoGeometry g = ops::VideoGeometry::of(73, 1344, 768);
  ops::SequenceLayout lay;
  lay.text = 32;
  lay.cond = keyframes * g.tokensPerFrame;
  lay.audio = g.audioRows;
  lay.frames = g.latentFrames;
  lay.tokensPerFrame = g.tokensPerFrame;
  lay.gridH = g.gridH;
  lay.gridW = g.gridW;
  lay.tailPad = 64;
  const ops::VsaTable t(lay, {4, 4, 4, sparsity});
  const std::vector<std::vector<int32_t>> want = fastVideoTiles(lay);
  const int32_t prefix = static_cast<int32_t>(want.size()) - t.videoTiles();
  require(t.tiles() == tiles && static_cast<int32_t>(want.size()) == tiles && t.videoTiles() == 396 &&
              t.topk() == topk && t.rows() == lay.rows(),
          "tile counts differ from the reference's and FastVideo's");
  // the engine's storage order: the video tiles first, then the prefix tiles, each tile's rows in FastVideo's order
  for (int32_t j = 0; j < t.tiles(); ++j) {
    const std::vector<int32_t> &rows = want[size_t(j < t.videoTiles() ? prefix + j : j - t.videoTiles())];
    const int32_t start = t.tileWords()[2 * j], count = t.tileWords()[2 * j + 1];
    require(count == static_cast<int32_t>(rows.size()), "a tile's row count differs from FastVideo's");
    for (int32_t r = 0; r < count; ++r) {
      require(t.order()[size_t(start + r)] == rows[size_t(r)] && t.inverse()[size_t(rows[size_t(r)])] == start + r &&
                  t.rowTile()[size_t(start + r)] == j,
              "a tile's rows differ from FastVideo's");
    }
  }
  require(t.videoRows() == lay.frames * lay.tokensPerFrame && t.items().size() == size_t(t.tiles()) &&
              t.items().front() == t.videoTiles() && t.items()[size_t(prefix)] == 0,
          "the attention items are not the prefix tiles, then the video tiles");
  if (keyframes == 0 && sparsity == 0.8)
    require(std::fabs(t.density() - 0.22151857331271615) < 1e-12, "density differs from the reference's");
}

// Ref2VA: text with vision runs, a reference video with its soundtrack, two images, the target. The engine order puts
// the conditions after the text; the VSA cubes cover the target video alone, every prefix tile lies inside one
// segment, and FL2VA's row classes give the default segments.
void checkReferenceTiles(double sparsity, bool dense) {
  const ops::VideoGeometry g = ops::VideoGeometry::of(124, 832, 480);
  std::vector<uint8_t> tags(6393, 1);
  for (int32_t i = 17; i < 409; ++i) tags[size_t(i)] = 0;
  for (int32_t i = 2405; i < 4231; ++i) tags[size_t(i)] = 0;
  using Kind = ops::ReferenceShape::Kind;
  const std::vector<ops::ReferenceShape> refs = {
      {Kind::Video, 37, 30, 52, 207, 0}, {Kind::Image, 1, 64, 114, 0, 0}, {Kind::Image, 1, 64, 114, 0, 0}};
  const ops::RowLayout rows = ops::RowLayout::build(g, 6393, tags.data(), refs);
  const ops::SequenceLayout lay = rows.sequence(g, 64);
  require(lay.rows() == 39729 && lay.cond == 18078 && lay.condAudio == 414 && lay.audio == 414 &&
              lay.videoStart() == 39729 - 14430 && lay.adaModalities() == 5 && lay.finalSlots() == 5,
          "the Ref2VA sequence's counts differ");
  std::vector<int32_t> seen(size_t(lay.rows()), 0);
  for (int32_t r : rows.engineOrder) ++seen[size_t(r)];
  require(std::all_of(seen.begin(), seen.end(), [](int32_t n) { return n == 1; }), "the engine order is no permutation");
  std::vector<uint8_t> slots(size_t(lay.rows()));
  rows.toEngineOrder(rows.slots.data(), slots.data(), 1);
  for (int32_t r = 0; r < lay.rows(); ++r) {
    const int32_t want = r < lay.text ? -1 : r < lay.audioStart() ? 3 : r < lay.audioStart() + lay.condAudio ? 4
                                                                    : r < lay.videoStart() ? 2 : 0;
    require(want < 0 ? slots[size_t(r)] <= 1 : slots[size_t(r)] == want, "an engine-order row has the wrong slot");
  }
  const std::vector<std::pair<int32_t, int32_t>> segs = lay.prefixSegments();
  require(segs.size() == 5 && segs[0] == std::pair<int32_t, int32_t>{0, 6393} &&
              segs[1] == std::pair<int32_t, int32_t>{6393, 14430} && segs[2] == std::pair<int32_t, int32_t>{20823, 3648} &&
              segs[3] == std::pair<int32_t, int32_t>{24471, 414} && segs[4] == std::pair<int32_t, int32_t>{24885, 414},
          "the Ref2VA prefix segments differ");
  ops::VsaSpec spec{4, 4, 4, sparsity};
  spec.dense = dense;
  const ops::VsaTable t(lay, spec);
  require(t.rows() == lay.rows() && t.videoRows() == 14430, "the tile map's rows differ");
  for (int32_t j = 0; j < t.tiles(); ++j) {
    const int32_t start = t.tileWords()[2 * j], count = t.tileWords()[2 * j + 1];
    const int32_t first = t.order()[size_t(start)], lastRow = t.order()[size_t(start + count - 1)];
    if (j < t.videoTiles()) {
      require(first >= lay.videoStart() && lastRow >= lay.videoStart(), "a video tile leaves the target video");
      continue;
    }
    const auto in = [&](int32_t r) {
      for (size_t k = 0; k < segs.size(); ++k)
        if (r >= segs[k].first && r < segs[k].first + segs[k].second) return int32_t(k);
      return int32_t(-1);
    };
    require(count <= 64 && in(first) >= 0 && in(first) == in(lastRow) && lastRow - first == count - 1,
            "a prefix tile crosses its segment");
  }
  // FL2VA's row classes: the default segments
  const ops::RowLayout kf = ops::RowLayout::build(
      g, 32, nullptr, {{Kind::Keyframe, 1, g.latentHeight, g.latentWidth, 0, 0}, {Kind::Keyframe, 1, g.latentHeight, g.latentWidth, 0, 1}});
  ops::SequenceLayout def = kf.sequence(g, 64), given = def;
  def.segments.clear();
  const ops::VsaTable a(def, {4, 4, 4, sparsity}), b(given, {4, 4, 4, sparsity});
  require(a.order() == b.order() && a.tileWords() == b.tileWords(), "FL2VA's row-class segments change the tile map");
}

void checkSchedule(model::FastH3Contract c, const std::vector<float> &want) {
  model::Prepared p, k;
  model::Prep::scheduleRungs(c, false, p);
  require(p.steps == static_cast<int32_t>(c.rungs.size()) && p.table == want, "the rung schedule's timesteps differ");
  for (size_t s = 0; s < c.rungs.size(); ++s)
    require(p.table[size_t(p.tv[s])] == p.vT[s] && p.table[size_t(p.ta[s])] == p.aT[s] && p.vSigmas[s] > 0.0f,
            "a step's timestep index is wrong");
  require(p.vSigmas.back() == 0.0f && p.aSigmas.back() == 0.0f && p.cT.empty(), "the schedule ends at sigma 0");
  // keyframe rows: every step at 0.999, one more timestep, the text schedule's indices shifted by none
  model::Prep::scheduleRungs(c, true, k);
  std::vector<float> keyframed = want;
  keyframed.push_back(0.999f);
  require(k.table == keyframed && k.tv == p.tv && k.ta == p.ta, "the keyframe schedule's timesteps differ");
  for (size_t s = 0; s < c.rungs.size(); ++s)
    require(k.cT[s] == 0.999f && k.tc[s] == static_cast<int64_t>(want.size()), "a keyframe step's timestep is wrong");
}

} // namespace

int main() {
  try {
    checkTiles(0, 0.8, 80, 401);
    checkTiles(0, 0.9, 40, 401);
    checkTiles(1, 0.8, 80, 401 + 16);  // 1008 keyframe rows: 15 chunks of 64 and one of 48
    checkTiles(2, 0.9, 40, 401 + 32);
    checkReferenceTiles(0.9, false);
    checkReferenceTiles(0.0, true);
    model::FastH3Contract eight{{999, 874, 749, 624, 500, 375, 250, 125}, 10.0, 3.0, {4, 4, 4, 0.8}, {}};
    checkSchedule(eight, {0.00010013580322265625f, 0.00033360719680786133f, 0.014211654663085938f,
                          0.032424747943878174f, 0.04585158824920654f, 0.05683189630508423f, 0.09090906381607056f,
                          0.10048043727874756f, 0.1428571343421936f, 0.16725987195968628f, 0.23076921701431274f, 0.25f,
                          0.3571428656578064f, 0.4117646813392639f, 0.5f, 0.699999988079071f});
    model::FastH3Contract four{{999, 749, 500, 250}, 12.0, 3.0, {4, 4, 4, 0.9}, {}};
    checkSchedule(four, {8.344650268554688e-05f, 0.00033360719680786133f, 0.02716749906539917f, 0.07692307233810425f,
                         0.10048043727874756f, 0.19999998807907104f, 0.25f, 0.5f});
    std::cout << "video_vsa_table_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "video_vsa_table_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
