#include "video/ops/RowLayout.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace splash::video::ops {
namespace {

constexpr int32_t kAudioChannels = 2;
constexpr double kFrameRescale = 5.0 / 3.0;          // ROPE_FRAME_RESCALE
constexpr int kFramesPerLatent[5] = {1, 4, 4, 4, 4};  // ROPE_FRAMES_PER_LATENT, cyclic

// numpy linspace(start, stop, num, endpoint=False): arange(num) * ((stop - start) / num) + start
std::vector<double> linspaceNoEnd(double start, double stop, int num) {
  const double step = (stop - start) / num;
  std::vector<double> y(num);
  for (int i = 0; i < num; ++i) {
    double v = static_cast<double>(i) * step;
    y[i] = v + start;
  }
  return y;
}

// packing._spatial_position_grid: the aspect-normalized grid of one latent axis, x 32
std::vector<double> spatialGrid(int dim, int patch, double sqrtArea) {
  const double ratio = dim / sqrtArea;
  const double left = (1.0 - ratio) / 2.0;
  std::vector<double> g = linspaceNoEnd(left, left + ratio, dim / patch);
  for (double &v : g) v = v * 32;
  return g;
}

// numpy add.reduce of float64 (pairwise summation, 8 accumulators, blocks of 128) from the identity 0
double pairwiseSum(const double *a, int64_t n) {
  if (n < 8) {
    double r = 0.0;
    for (int64_t i = 0; i < n; ++i) r += a[i];
    return r;
  }
  if (n <= 128) {
    double r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    int64_t i = 8;
    for (; i < n - (n % 8); i += 8)
      for (int j = 0; j < 8; ++j) r[j] += a[i + j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  int64_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwiseSum(a, n2) + pairwiseSum(a + n2, n - n2);
}

// The rows of a latent grid [frames][h / 2][w / 2] and its (h, w) grids.
struct Grid {
  int32_t frames = 0, gh = 0, gw = 0;
  std::vector<double> hg, wg;
  [[nodiscard]] int32_t rows() const { return frames * gh * gw; }
};

Grid gridOf(int32_t frames, int32_t lh, int32_t lw, const char *what) {
  if (frames < 1 || lh < 2 || lw < 2 || lh % 2 || lw % 2)
    throw std::invalid_argument(std::string("row layout: ") + what + " latents " + std::to_string(frames) + " x " +
                                std::to_string(lh) + " x " + std::to_string(lw) + " (frames >= 1, even sides)");
  const double sqrtArea = std::sqrt(static_cast<double>(lh * lw));
  return {frames, lh / 2, lw / 2, spatialGrid(lh, 2, sqrtArea), spatialGrid(lw, 2, sqrtArea)};
}

// _temporal_position_grid: origin + concatenate([0], cumsum(spans[:-1])), spans = 5/3 x (1, 4, 4, 4, 4)[i % 5]
std::vector<double> temporalGrid(int32_t frames, double origin) {
  std::vector<double> t(frames);
  double acc = 0.0;
  for (int32_t f = 0; f < frames; ++f) {
    if (f > 0) acc = acc + kFrameRescale * kFramesPerLatent[(f - 1) % 5];
    t[f] = origin + acc;
  }
  return t;
}

class Builder {
public:
  explicit Builder(RowLayout &l) : l_(l) {}

  RowSegment &add(RowClass c, int32_t rows, int32_t ref) {
    RowSegment s{c, cursor_, rows, ref};
    cursor_ += rows;
    const RowClassInfo &info = rowClass(c);
    l_.pos.resize(size_t(cursor_) * 3, 0.0);
    l_.tags.resize(cursor_, info.tag);
    l_.slots.resize(cursor_, info.slot);
    l_.segments.push_back(s);
    return l_.segments.back();
  }
  // video-stream rows: time t[f] per latent frame, the grid's (h, w)
  void video(RowClass c, int32_t ref, const Grid &g, const std::vector<double> &t) {
    RowSegment &s = add(c, g.rows(), ref);
    s.frames = g.frames;
    s.latentHeight = 2 * g.gh;
    s.latentWidth = 2 * g.gw;
    for (int32_t f = 0; f < g.frames; ++f)
      for (int32_t y = 0; y < g.gh; ++y)
        for (int32_t x = 0; x < g.gw; ++x) {
          double *r = &l_.pos[size_t(s.start + (f * g.gh + y) * g.gw + x) * 3];
          r[0] = t[f];
          r[1] = g.hg[y];
          r[2] = g.wg[x];
        }
  }
  // audio-stream rows, channel-major: t = clock + k, w = the first / last value of a width grid
  void audio(RowClass c, int32_t ref, int32_t latents, double clock, const std::vector<double> &wg) {
    const RowSegment &s = add(c, kAudioChannels * latents, ref);
    for (int32_t ch = 0; ch < kAudioChannels; ++ch)
      for (int32_t k = 0; k < latents; ++k) {
        double *r = &l_.pos[size_t(s.start + ch * latents + k) * 3];
        r[0] = clock + static_cast<double>(k);
        r[2] = ch == 0 ? wg.front() : wg.back();
      }
  }

private:
  RowLayout &l_;
  int32_t cursor_ = 0;
};

} // namespace

RowLayout RowLayout::build(const VideoGeometry &target, int32_t textRows, const uint8_t *textTags,
                           const std::vector<ReferenceShape> &references) {
  if (textRows < 1) throw std::invalid_argument("row layout: no text rows");
  RowLayout l;
  Builder b(l);
  // text: runs of text and vision rows, at (i, 0, 0)
  for (int32_t i = 0; i < textRows;) {
    const uint8_t tag = textTags ? textTags[i] : uint8_t(1);
    if (tag > 1)
      throw std::invalid_argument("row layout: text row " + std::to_string(i) + " has tag " + std::to_string(tag) +
                                  " (1 text, 0 vision)");
    int32_t j = i + 1;
    while (j < textRows && (textTags ? textTags[j] : uint8_t(1)) == tag) ++j;
    b.add(tag ? RowClass::Text : RowClass::Vision, j - i, -1);
    i = j;
  }
  for (int32_t i = 0; i < textRows; ++i) l.pos[size_t(i) * 3] = static_cast<double>(i);
  l.textRows = textRows;

  const Grid tg = gridOf(target.latentFrames, target.latentHeight, target.latentWidth, "target");
  double clock = static_cast<double>(textRows);
  for (size_t k = 0; k < references.size(); ++k) {
    const ReferenceShape &r = references[k];
    const int32_t ref = static_cast<int32_t>(k);
    switch (r.kind) {
    case ReferenceShape::Kind::Image: {
      const Grid g = gridOf(1, r.latentHeight, r.latentWidth, "reference image");
      b.video(RowClass::RefImage, ref, g, {clock});
      clock += 1.0;
      break;
    }
    case ReferenceShape::Kind::Audio: {
      if (r.audioLatents < 1) throw std::invalid_argument("row layout: reference audio without latents");
      b.audio(RowClass::RefAudio, ref, r.audioLatents, clock, tg.wg);
      clock += static_cast<double>(r.audioLatents);
      break;
    }
    case ReferenceShape::Kind::Video: {
      if (r.audioLatents < 0) throw std::invalid_argument("row layout: negative soundtrack latents");
      const Grid g = gridOf(r.frames, r.latentHeight, r.latentWidth, "reference video");
      if (r.audioLatents) b.audio(RowClass::RefSoundtrack, ref, r.audioLatents, clock, g.wg);
      b.video(RowClass::RefVideo, ref, g, temporalGrid(g.frames, clock));
      double span = 0.0;  // the pipeline's sequential Python sum
      for (int32_t f = 0; f < g.frames; ++f) span += kFrameRescale * kFramesPerLatent[f % 5];
      clock += std::max(static_cast<double>(r.audioLatents), span);
      break;
    }
    case ReferenceShape::Kind::Keyframe: {
      // the first latent frame's time, or the last one's: clock + _temporal_position_span(F) - 5/3, the span summed
      // pairwise by numpy; the clock does not advance
      double t = clock;
      if (r.anchor == 1) {
        std::vector<double> spans(tg.frames, kFrameRescale);
        for (int32_t f = 0; f < tg.frames; ++f) spans[f] = spans[f] * kFramesPerLatent[f % 5];
        const double span = 0.0 + pairwiseSum(spans.data(), tg.frames);
        t = clock + span;
        t = t - kFrameRescale;
      } else if (r.anchor != 0) {
        throw std::invalid_argument("row layout: keyframe anchor " + std::to_string(r.anchor) + " (0 first, 1 last)");
      }
      Grid g = tg;
      g.frames = 1;
      b.video(RowClass::Keyframe, ref, g, {t});
      break;
    }
    }
  }
  l.targetClock = clock;
  b.audio(RowClass::TargetAudio, -1, target.audioRows / kAudioChannels, clock, tg.wg);
  b.video(RowClass::TargetVideo, -1, tg, temporalGrid(tg.frames, clock));

  // per-stream row indices, conditions first (packed order), then the target's
  for (const RowSegment &s : l.segments) {
    const RowClassInfo &info = rowClass(s.cls);
    std::vector<int64_t> &index = info.stream == RowStream::Video   ? l.videoIndices
                                  : info.stream == RowStream::Audio ? l.audioIndices
                                                                    : l.textIndices;
    for (int32_t i = s.start; i < s.end(); ++i) index.push_back(i);
    const bool target = info.update == RowUpdate::Noised;
    if (info.stream == RowStream::Video) (target ? l.targetVideoRows : l.condVideoRows) += s.rows;
    if (info.stream == RowStream::Audio) (target ? l.targetAudioRows : l.condAudioRows) += s.rows;
  }
  // engine order: text | video conditions | audio conditions | target audio | target video
  auto stage = [](const RowClassInfo &c) {
    if (c.stream == RowStream::Text) return 0;
    if (c.update != RowUpdate::Noised) return c.stream == RowStream::Video ? 1 : 2;
    return c.stream == RowStream::Audio ? 3 : 4;
  };
  RowClass last = RowClass::TargetVideo;
  int lastStage = -1;
  for (int k = 0; k < 5; ++k)
    for (const RowSegment &s : l.segments) {
      const RowClassInfo &info = rowClass(s.cls);
      if (stage(info) != k) continue;
      const int32_t e = static_cast<int32_t>(l.engineOrder.size());
      for (int32_t i = s.start; i < s.end(); ++i) l.engineOrder.push_back(i);
      if (k == 4) continue;  // the target video is the VSA region
      const bool joins = !l.engineSegments.empty() && lastStage == k && (k == 0 || s.cls == last);
      if (joins) l.engineSegments.back().second += s.rows;
      else l.engineSegments.push_back({e, s.rows});
      last = s.cls;
      lastStage = k;
    }
  return l;
}

SequenceLayout RowLayout::sequence(const VideoGeometry &target, int32_t tailPad) const {
  SequenceLayout s;
  s.text = textRows;
  s.cond = condVideoRows;
  s.condAudio = condAudioRows;
  s.audio = targetAudioRows;
  s.frames = target.latentFrames;
  s.tokensPerFrame = target.tokensPerFrame;
  s.gridH = target.gridH;
  s.gridW = target.gridW;
  s.tailPad = tailPad;
  s.segments = engineSegments;
  if (s.rows() != rows() || targetVideoRows != target.videoRows())
    throw std::logic_error("row layout: the sequence does not cover the rows");
  return s;
}

void RowLayout::timesteps(float videoT, float audioT, std::vector<float> &table, std::vector<int64_t> &index) const {
  auto timeOf = [&](RowTime t) {
    switch (t) {
    case RowTime::Video: return videoT;
    case RowTime::Condition: return std::max(videoT, kConditionTime);
    case RowTime::Clean: return 1.0f;
    case RowTime::Audio: return audioT;
    }
    return videoT;
  };
  table.clear();
  for (const RowSegment &s : segments) table.push_back(timeOf(rowClass(s.cls).time));
  std::sort(table.begin(), table.end());
  table.erase(std::unique(table.begin(), table.end()), table.end());
  index.assign(rows(), 0);
  for (const RowSegment &s : segments) {
    const float t = timeOf(rowClass(s.cls).time);
    const int64_t k = std::lower_bound(table.begin(), table.end(), t) - table.begin();
    std::fill(index.begin() + s.start, index.begin() + s.end(), k);
  }
}

namespace {
template <class Keep>
std::vector<RowRange> merged(const std::vector<RowSegment> &segments, Keep keep) {
  std::vector<RowRange> out;
  for (const RowSegment &s : segments) {
    if (!keep(rowClass(s.cls))) continue;
    if (!out.empty() && out.back().end() == s.start)
      out.back().rows += s.rows;
    else
      out.push_back({s.start, s.rows});
  }
  return out;
}
} // namespace

std::vector<RowRange> RowLayout::ranges(RowAttention role) const {
  return merged(segments, [&](const RowClassInfo &c) { return c.attention == role; });
}

std::vector<RowRange> RowLayout::fixedRanges() const {
  return merged(segments, [](const RowClassInfo &c) { return c.update != RowUpdate::Noised; });
}

} // namespace splash::video::ops
