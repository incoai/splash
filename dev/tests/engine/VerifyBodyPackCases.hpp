#pragma once
#include "VerifyPackTestFixtures.hpp"
#include "ops/VerifyBodyPack.hpp"
struct Segment final {
  uint32_t offset, N, kind;
  Weight quant;
  MetalBuffer values;
  std::vector<float> decoded;
  uint64_t immutable;
};
struct Fixture final {
  uint32_t N = 0, K;
  std::vector<Segment> segments;
  ops::Projection projection;
  Fixture(MetalBackend &b, uint32_t k, const std::vector<uint32_t> &ns, uint32_t tail, std::mt19937 &rng,
          const std::vector<uint32_t> &bank, uint32_t padding = 0)
      : K(k) {
    std::vector<ops::QuantizedSegment> blocks;
    for (uint32_t n : ns) {
      auto q = weight(b, n, K, rng, bank);
      auto v = ops::QuantizedSegment::planes(GGUF_FMT_AF4G64, n, K, q.codes, {}, q.metadata);
      v.columnOffset = N;
      blocks.push_back(v);
      segments.push_back({N, n, 0, std::move(q), {}, {}, 0});
      N += n;
    }
    if (tail) {
      const uint32_t n = 96;
      std::uniform_real_distribution<float> d(-.125f, .125f);
      std::vector<float> v(uint64_t{n} * K);
      for (auto &x : v)
        x = tail == 2 ? bf16ToFloat(floatToBf16(d(rng))) : d(rng);
      auto buf =
          b.allocateBuffer(v.size() * (tail == 2 ? 2 : 4), metal::BufferStorage::Shared, "body float weight");
      if (tail == 2) {
        auto dst = static_cast<uint16_t *>(buf.contents());
        for (size_t i = 0; i < v.size(); ++i)
          dst[i] = floatToBf16(v[i]);
      } else
        std::memcpy(buf.contents(), v.data(), v.size() * 4);
      auto s =
          ops::QuantizedSegment::rows(tail == 2 ? GGUF_FMT_BF16 : ops::QuantizedSegment::kFloat32, n, K, buf);
      s.columnOffset = N;
      blocks.push_back(s);
      segments.push_back({N, n, tail, {}, buf, std::move(v), hash(buf)});
      N += n;
    }
    N += padding;
    projection = ops::Projection(N, K, ops::BlockWeights{blocks});
  }
  uint32_t columns() const { return segments.back().offset + segments.back().N; }
  bool immutable() const {
    for (const auto &s : segments) {
      if (s.kind ? hash(s.values) != s.immutable
                 : (hash(s.quant.codes) != s.quant.hc || hash(s.quant.metadata) != s.quant.hm))
        return false;
    }
    return true;
  }
  std::vector<float> row(uint32_t col) const {
    for (const auto &s : segments)
      if (col >= s.offset && col < s.offset + s.N) {
        uint32_t j = col - s.offset;
        std::vector<float> v(K);
        if (s.kind) {
          std::copy_n(s.decoded.begin() + uint64_t{j} * K, K, v.begin());
        } else {
          rowValues(Fmt(GGUF_FMT_AF4G64),
                    s.quant.native.data() + uint64_t{j} * rowBytes(Fmt(GGUF_FMT_AF4G64), K), K, v.data());
          for (auto &x : v)
            x = h2f(f2h(x));
        }
        return v;
      }
    throw std::runtime_error("column coverage");
  }
};
struct Run final {
  const Fixture &f;
  uint32_t lanes, S, walk;
  bool residual, candidate;
  std::array<uint32_t, 8> rows;
  std::vector<uint16_t> x, aux;
  Guarded input, addend, counts, pi, pr, po, output, partials, counters;
  Run(MetalBackend &b, const Fixture &fixture, uint32_t L, const std::array<uint32_t, 8> &r,
      const std::vector<uint16_t> &xb, const std::vector<uint16_t> &ab, uint32_t splits, uint32_t spread,
      bool ep, bool compact, uint32_t poison)
      : f(fixture), lanes(L), S(splits), walk(spread), residual(ep), candidate(compact), rows(r), x(xb),
        aux(ab), input(b, uint64_t{64} * f.K * 2, 0), addend(b, ep ? uint64_t{64} * f.N * 2 : 4, 0),
        counts(b, 32, 0), pi(b, uint64_t{32} * f.K * 2, 0xffffffff),
        pr(b, uint64_t{32} * f.N * 2, 0xffffffff), po(b, uint64_t{32} * f.N * 2, 0xffffffff),
        output(b, uint64_t{64} * f.N * 2, 0xffffffff), partials(b, uint64_t{S} * 64 * f.N * 4, poison),
        counters(b, uint64_t{(f.N + 63) / 64} * 4, 0) {
    std::memcpy(input.view.contents(), x.data(), x.size() * 2);
    std::memcpy(addend.view.contents(), aux.data(), aux.size() * 2);
    std::memcpy(counts.view.contents(), rows.data(), 32);
  }
  void add(CommandGraph &g, MetalBackend &b) const {
    ops::Linear linear(b.capabilities());
    auto plan = ops::Linear::plan({{f.N, f.K},
                                   lanes * 8,
                                   ops::LinearPhase::Decode,
                                   residual ? ops::LinearEpilogue::Residual : ops::LinearEpilogue::None},
                                  {ops::LinearTile::GgufStaged, S, bool(walk), false, false},
                                  ops::FloatOutput::BFloat16);
    ops::LinearScratch scratch{.partials = partials.view, .counters = counters.view};
    if (candidate) {
      check(ops::addVerifyBodyPack(linear, g, f.projection, plan, input.view,
                                   residual ? addend.view : MetalBuffer{}, output.view, scratch, counts.view,
                                   {pi.view, pr.view, po.view}, lanes),
            "actualbodyhelperinactive");
    } else
      linear.add(g,
                 {.input = input.view,
                  .output = output.view,
                  .residual = residual ? addend.view : MetalBuffer{},
                  .scratch = scratch},
                 f.projection, plan);
  }
  std::vector<uint16_t> selected() const {
    std::vector<uint16_t> v;
    auto p = static_cast<const uint16_t *>(output.view.contents());
    for (uint32_t l = 0; l < lanes; ++l)
      for (uint32_t r = 0; r < rows[l]; ++r) {
        auto a = p + (uint64_t{l} * 8 + r) * f.N;
        v.insert(v.end(), a, a + f.columns());
      }
    return v;
  }
  bool guard(uint32_t poison) const {
    for (const Guarded *p : {&input, &addend, &counts, &pi, &pr, &po, &output, &partials, &counters})
      if (!p->intact())
        return false;
    if (std::memcmp(input.view.contents(), x.data(), x.size() * 2) ||
        std::memcmp(addend.view.contents(), aux.data(), aux.size() * 2) ||
        std::memcmp(counts.view.contents(), rows.data(), 32))
      return false;
    auto c = static_cast<const uint32_t *>(counters.view.contents());
    if (!std::all_of(c, c + counters.bytes / 4, [](uint32_t v) { return v == 0; }))
      return false;
    auto v = selected();
    if (!std::all_of(v.begin(), v.end(), [](uint16_t a) { return std::isfinite(bf16ToFloat(a)); }))
      return false;
    auto pp = static_cast<const uint32_t *>(partials.view.contents());
    if (S == 1 && !std::all_of(pp, pp + partials.bytes / 4, [&](uint32_t v) { return v == poison; }))
      return false;
    auto untouched = [&](const Guarded &v) {
      auto p = static_cast<const uint32_t *>(v.view.contents());
      return std::all_of(p, p + v.bytes / 4, [](uint32_t x) { return x == 0xffffffff; });
    };
    if (!candidate && (!untouched(pi) || !untouched(pr) || !untouched(po)))
      return false;
    if (candidate && !residual && !untouched(pr))
      return false;
    for (uint32_t q = 0; q < 32; ++q)
      for (uint32_t n = f.columns(); n < f.N; ++n) {
        auto i = uint64_t{q} * f.N + n;
        if (static_cast<const uint16_t *>(po.view.contents())[i] != 0xffff ||
            static_cast<const uint16_t *>(pr.view.contents())[i] != 0xffff)
          return false;
      }
    if (S > 1) {
      for (uint32_t split = 0; split < S; ++split)
        for (uint32_t q = 0; q < (candidate ? 32u : 64u); ++q)
          for (uint32_t n = 0; n < f.N; ++n) {
            bool quantized = false;
            for (const auto &segment : f.segments)
              if (!segment.kind && n >= segment.offset && n < segment.offset + segment.N)
                quantized = true;
            if (!quantized && pp[(uint64_t{split} * (candidate ? 32 : 64) + q) * f.N + n] != poison)
              return false;
          }
      if (candidate && !std::all_of(pp + uint64_t{S} * 32 * f.N, pp + partials.bytes / 4,
                                    [&](uint32_t x) { return x == poison; }))
        return false;
    }
    auto pad = static_cast<const uint16_t *>(output.view.contents());
    for (uint32_t q = 0; q < 64; ++q)
      for (uint32_t n = f.columns(); n < f.N; ++n)
        if (pad[uint64_t{q} * f.N + n] != 0xffff)
          return false;
    if (candidate) {
      auto p = static_cast<const uint16_t *>(output.view.contents()),
           a = static_cast<const uint16_t *>(pi.view.contents()),
           d = static_cast<const uint16_t *>(pr.view.contents());
      for (uint32_t q = 0; q < 32; ++q) {
        uint32_t l = q / 4, r = q % 4;
        bool active = l < lanes && r < rows[l];
        for (uint32_t k = 0; k < f.K; ++k)
          if (a[uint64_t{q} * f.K + k] != (active ? x[(uint64_t{l} * 8 + r) * f.K + k] : 0))
            return false;
        if (residual)
          for (uint32_t n = 0; n < f.columns(); ++n)
            if (d[uint64_t{q} * f.N + n] != (active ? aux[(uint64_t{l} * 8 + r) * f.N + n] : 0))
              return false;
      }
      for (uint32_t q = 0; q < 64; ++q)
        if (q / 8 >= lanes || q % 8 >= rows[q / 8])
          for (uint32_t n = 0; n < f.columns(); ++n)
            if (p[uint64_t{q} * f.N + n] != 0)
              return false;
    }
    return true;
  }
  std::vector<uint32_t> selectedParts() const {
    std::vector<uint32_t> v;
    if (S == 1)
      return v;
    auto p = static_cast<const uint32_t *>(partials.view.contents());
    for (const auto &s : f.segments)
      if (!s.kind)
        for (uint32_t split = 0; split < S; ++split)
          for (uint32_t l = 0; l < lanes; ++l)
            for (uint32_t r = 0; r < rows[l]; ++r) {
              auto a = p + (uint64_t{split} * (candidate ? 32 : 64) + l * (candidate ? 4 : 8) + r) * f.N +
                       s.offset;
              v.insert(v.end(), a, a + s.N);
            }
    return v;
  }
  bool partialsExact(const Run &base) const {
    if (S == 1)
      return true;
    auto a = static_cast<const uint32_t *>(partials.view.contents()),
         z = static_cast<const uint32_t *>(base.partials.view.contents());
    for (const auto &s : f.segments)
      if (!s.kind)
        for (uint32_t split = 0; split < S; ++split)
          for (uint32_t l = 0; l < lanes; ++l)
            for (uint32_t r = 0; r < rows[l]; ++r)
              for (uint32_t n = s.offset; n < s.offset + s.N; ++n) {
                auto i = (uint64_t{split} * 32 + l * 4 + r) * f.N + n,
                     j = (uint64_t{split} * 64 + l * 8 + r) * f.N + n;
                if (a[i] != z[j] || !std::isfinite(std::bit_cast<float>(a[i])))
                  return false;
              }
    return true;
  }
  bool bounds() const {
    auto out = static_cast<const uint16_t *>(output.view.contents());
    std::vector<uint32_t> cols{0, 31, f.columns() / 2, f.columns() - 1};
    for (const auto &s : f.segments) {
      cols.push_back(s.offset);
      cols.push_back(s.offset + s.N - 1);
    }
    for (uint32_t n : cols) {
      auto v = f.row(n);
      for (uint32_t l = 0; l < lanes; ++l)
        for (uint32_t r : {0u, rows[l] - 1}) {
          uint32_t old = l * 8 + r;
          double sum = 0, mag = 0;
          for (uint32_t k = 0; k < f.K; ++k) {
            double z = double(bf16ToFloat(x[uint64_t{old} * f.K + k])) * v[k];
            sum += z;
            mag += std::abs(z);
          }
          const double u = std::ldexp(1., -24), ops = f.K + 2 * (f.K / 64) + 16,
                       gamma = ops * u / (1 - ops * u), e = gamma * mag + 1e-5;
          double lo = sum - e, hi = sum + e;
          auto at = uint64_t{old} * f.N + n;
          if (residual) {
            double q = bf16ToFloat(aux[at]);
            lo = float(float(lo) + q);
            hi = float(float(hi) + q);
          }
          double got = bf16ToFloat(out[at]);
          if (!std::isfinite(got) || got < bf16ToFloat(floatToBf16(float(lo))) ||
              got > bf16ToFloat(floatToBf16(float(hi))))
            return false;
        }
    }
    return true;
  }
};
