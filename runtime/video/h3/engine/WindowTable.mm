#include "video/h3/engine/WindowTable.hpp"

#include "video/h3/engine/DeviceProfile.hpp"
#include "metal/abi/video/Layout.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace h3 {
namespace {

using Range = std::pair<int32_t, int32_t>;  // [k0, k1)
using Ranges = std::vector<Range>;

constexpr int32_t kAnchorsBoth = 3;

Ranges merge(Ranges ranges) {
  ranges.erase(std::remove_if(ranges.begin(), ranges.end(), [](const Range &r) { return r.second <= r.first; }),
               ranges.end());
  std::sort(ranges.begin(), ranges.end());
  Ranges out;
  for (const Range &r : ranges) {
    if (!out.empty() && r.first <= out.back().second) out.back().second = std::max(out.back().second, r.second);
    else out.push_back(r);
  }
  return out;
}

struct Window {
  const model::SequenceLayout &lay;
  int32_t chunk, radius;

  [[nodiscard]] int32_t videoEnd() const { return lay.rows(); }
  [[nodiscard]] int32_t frameStart(int32_t f) const { return lay.videoStart() + f * lay.tokensPerFrame; }
  // OpenVDN window_bounds (unclamped, inclusive)
  [[nodiscard]] std::pair<int32_t, int32_t> bounds(int32_t t) const {
    if (chunk <= 0) return {t - radius, t + radius};
    return {(t / chunk - radius) * chunk, (t / chunk + radius + 1) * chunk - 1};
  }
  // key-set class of query row r: -1 dense (text, audio, anchor frames), else its video frame
  [[nodiscard]] int32_t rowClass(int32_t r) const {
    if (r < lay.videoStart() || r >= videoEnd()) return -1;
    const int32_t f = (r - lay.videoStart()) / lay.tokensPerFrame;
    if (f == 0 || f == lay.frames - 1) return -1;
    return f;
  }
  [[nodiscard]] Ranges rangesFor(int32_t cls) const {
    const int32_t S = lay.rows();
    if (cls == -1) return {{0, S}};
    auto [lo, hi] = bounds(cls);
    lo = std::max(lo, 0);
    hi = std::min(hi, lay.frames - 1);
    return merge({{0, lay.videoStart()},
                  {videoEnd(), S},
                  {frameStart(lo), frameStart(hi + 1)},
                  {frameStart(0), frameStart(1)},
                  {frameStart(lay.frames - 1), frameStart(lay.frames)}});
  }
};

} // namespace

WindowTable::WindowTable(const model::SequenceLayout &lay, int32_t chunk, int32_t radius) {
  if (lay.frames < 3) throw std::invalid_argument("window attention needs >= 3 latent frames");
  if (radius < 0) throw std::invalid_argument("window radius must be >= 0");
  const Window window{lay, chunk, radius};
  const int32_t S = lay.rows(), nqb = (S + H3_TABLE_ROWS - 1) / H3_TABLE_ROWS;
  WindowTable &t = *this;
  t.rows_ = S;
  t.nqb_ = nqb;
  t.chunk_ = chunk;
  t.radius_ = radius;
  t.frames_ = lay.frames;
  t.words_.assign(static_cast<size_t>(nqb + 1) * H3_TABLE_RECORD, 0);
  const int32_t header[15] = {H3_TABLE_MAGIC, 1,         S,         nqb,        chunk,
                              radius,         kAnchorsBoth, lay.text, lay.cond,   lay.audio,
                              lay.frames,     lay.tokensPerFrame,     0, lay.videoStart(), window.videoEnd()};
  std::copy(header, header + 15, t.words_.begin());
  std::map<int32_t, Ranges> cache;
  auto rangesOf = [&](int32_t cls) -> const Ranges & {
    auto it = cache.find(cls);
    if (it == cache.end()) it = cache.emplace(cls, window.rangesFor(cls)).first;
    return it->second;
  };
  for (int32_t qb = 0; qb < nqb; ++qb) {
    const int32_t r0 = qb * H3_TABLE_ROWS, r1 = std::min((qb + 1) * H3_TABLE_ROWS, S);
    struct Segment {
      int32_t begin, end, cls;
    };
    std::vector<Segment> segs;
    for (int32_t r = r0; r < r1;) {
      const int32_t cls = window.rowClass(r);
      int32_t next;  // rows where the class can change: video start, video end, frame starts
      if (r < lay.videoStart()) next = lay.videoStart();
      else if (r >= window.videoEnd()) next = S;
      else next = lay.videoStart() + ((r - lay.videoStart()) / lay.tokensPerFrame + 1) * lay.tokensPerFrame;
      const int32_t e = std::min(next, r1);
      if (!segs.empty() && segs.back().cls == cls) segs.back().end = e;
      else segs.push_back({r, e, cls});
      r = e;
    }
    Ranges all;
    for (const Segment &s : segs) all.insert(all.end(), rangesOf(s.cls).begin(), rangesOf(s.cls).end());
    const Ranges u = merge(all);
    bool mixed = false;
    for (const Segment &s : segs) mixed |= rangesOf(s.cls) != rangesOf(segs[0].cls);
    if (u.size() > 3 || segs.size() > 2)
      throw std::invalid_argument("key-range table block " + std::to_string(qb) + ": " + std::to_string(u.size()) +
                                  " union ranges, " + std::to_string(segs.size()) +
                                  " segments (the kernels take 3 / 2)");
    int32_t *rec = &t.words_[static_cast<size_t>(qb + 1) * H3_TABLE_RECORD];
    rec[0] = static_cast<int32_t>(u.size());
    rec[1] = mixed ? 1 : 0;
    for (size_t j = 0; j < u.size(); ++j) {
      rec[8 + 2 * j] = u[j].first;
      rec[9 + 2 * j] = u[j].second;
    }
    if (mixed) {
      rec[2] = segs[1].begin;
      for (size_t s = 0; s < segs.size(); ++s) {
        const Ranges &sr = rangesOf(segs[s].cls);
        if (sr.size() > 3)
          throw std::invalid_argument("key-range table block " + std::to_string(qb) + ": a segment of " +
                                      std::to_string(sr.size()) + " ranges");
        rec[3 + s] = static_cast<int32_t>(sr.size());
        for (size_t j = 0; j < sr.size(); ++j) {
          rec[14 + 6 * s + 2 * j] = sr[j].first;
          rec[15 + 6 * s + 2 * j] = sr[j].second;
        }
      }
    } else {
      rec[2] = (qb + 1) * H3_TABLE_ROWS;
      rec[3] = rec[4] = static_cast<int32_t>(u.size());
      for (size_t j = 0; j < u.size(); ++j) {
        rec[14 + 2 * j] = rec[20 + 2 * j] = u[j].first;
        rec[15 + 2 * j] = rec[21 + 2 * j] = u[j].second;
      }
    }
  }
  // density: allowed (q, k) pairs over S^2
  double allowed = 0;
  for (int32_t b = 0; b < nqb; ++b) {
    const int32_t *r = &t.words_[static_cast<size_t>(1 + b) * H3_TABLE_RECORD];
    const int32_t q0 = b * H3_TABLE_ROWS, q1 = std::min(S, q0 + H3_TABLE_ROWS);
    const int32_t split = std::max(q0, std::min(q1, r[2]));
    for (int s = 0; s < 2; ++s) {
      const int32_t rows = s == 0 ? split - q0 : q1 - split;
      for (int j = 0; j < r[3 + s]; ++j) allowed += double(rows) * (r[15 + 6 * s + 2 * j] - r[14 + 6 * s + 2 * j]);
    }
  }
  t.density_ = allowed / (double(S) * S);
  t.buildItems();
}

std::vector<std::array<int32_t, 2>> WindowTable::branchBounds() const {
  std::vector<std::array<int32_t, 2>> b;
  for (int32_t t = 1; t < frames_ - 1; ++t) {
    int32_t lo, hi;
    if (chunk_ <= 0) {
      lo = t - radius_;
      hi = t + radius_;
    } else {
      lo = (t / chunk_ - radius_) * chunk_;
      hi = (t / chunk_ + radius_ + 1) * chunk_ - 1;
    }
    b.push_back({lo - 1, hi - 1});
  }
  return b;
}

void WindowTable::buildItems() {
  items_.clear();
  groups_.clear();
  slots_ = 0;
  const int32_t kc = device::kKeyChunk, S = rows_;
  const int32_t groups = (S + H3_ATTENTION_GROUP - 1) / H3_ATTENTION_GROUP;
  std::vector<WindowItem> chunked, window;
  for (int32_t g = 0; g < groups; ++g) {
    int64_t keys = int64_t(1) << 40;
    for (int32_t b = g * 4; b < std::min(nqb_, g * 4 + 4); ++b) {
      const int32_t *r = &words_[static_cast<size_t>(1 + b) * H3_TABLE_RECORD];
      int64_t k = 0;
      for (int j = 0; j < r[0]; ++j) k += r[9 + 2 * j] - r[8 + 2 * j];
      keys = std::min(keys, k);
    }
    if (keys >= device::kDenseKeyMin) {
      const int32_t chunks = (S + kc - 1) / kc;
      groups_.push_back({g, slots_, chunks, 0});
      for (int32_t c = 0; c < chunks; ++c) chunked.push_back({g, c * kc, std::min(S, (c + 1) * kc), slots_ + c});
      slots_ += chunks;
    } else {
      window.push_back({g, 0, S, -1});
    }
  }
  // dense (chunked) items after the window items
  static_assert(device::kDenseItemsLast);
  items_ = window;
  items_.insert(items_.end(), chunked.begin(), chunked.end());
}

void WindowTable::upload(metal::MetalBackend &backend) {
  table_ = h3::upload(backend, words_.data(), words_.size() * 4, "key-range table");
  itemBuffer_ = h3::upload(backend, items_.data(), items_.size() * sizeof(WindowItem), "window items");
  groupBuffer_ = h3::upload(backend, groups_.data(), groups_.size() * 16, "window groups");
}

} // namespace h3
