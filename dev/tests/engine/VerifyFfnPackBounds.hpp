#pragma once
#include "VerifyFfnPackChain.hpp"
namespace ffn_bounds {
struct DotHalf {
  double value = 0, magnitude = 0;
};
static std::vector<float> halfRow(const Weight &w, uint32_t n) {
  std::vector<float> v(w.K);
  rowValues(Fmt(GGUF_FMT_AF4G64), w.native.data() + uint64_t{n} * rowBytes(Fmt(GGUF_FMT_AF4G64), w.K), w.K,
            v.data());
  for (auto &f : v)
    f = h2f(f2h(f));
  return v;
}
static DotHalf dotHalf(const uint16_t *x, const std::vector<float> &w) {
  DotHalf d;
  for (uint32_t k = 0; k < w.size(); ++k) {
    double v = double(bf16ToFloat(x[k])) * w[k];
    d.value += v;
    d.magnitude += std::abs(v);
  }
  return d;
}
static double error(const DotHalf &d, uint32_t K) {
  const double u = std::ldexp(1., -24), ops = K + 2 * (K / 64) + 4;
  return ops * u / (1 - ops * u) * d.magnitude + 1e-5;
}
static double bf(double v) { return bf16ToFloat(floatToBf16(float(v))); }
static bool inside(uint16_t v, double lo, double hi) {
  auto x = bf16ToFloat(v);
  return std::isfinite(x) && x >= bf(lo) && x <= bf(hi);
}
// Same fast::exp2/silu error formula and silent-gate case as the native projection harness.
static std::pair<double, double> gateInterval(double lo, double hi, float gate) {
  const double t = std::abs(double(gate)) * 1.4426950408889634,
               s = double(gate) / (1 + std::exp(-double(gate))),
               e = (2 * (3 + std::floor(2 * t)) + 2 * std::log(2.) * t + 1 + 2 * 2.5) * std::ldexp(1., -24) *
                   std::abs(s);
  const bool silent = -double(gate) * 1.4426950408889634 >= 126 || std::abs(s) + e < std::ldexp(1., -126);
  double mn = INFINITY, mx = -INFINITY;
  for (double u : {bf(lo), bf(hi)})
    for (double g : {s - e, s + e, silent ? 0. : s}) {
      mn = std::min(mn, u * g);
      mx = std::max(mx, u * g);
    }
  if (std::min(std::abs(mn), std::abs(mx)) < std::ldexp(1., -126)) {
    mn = std::min(mn, 0.);
    mx = std::max(mx, 0.);
  }
  return {mn, mx};
}
struct SampleWeights {
  std::array<uint32_t, 4> columns;
  std::array<std::vector<float>, 4> g, u, d;
};
static SampleWeights prepare(const FfnWeights &w) {
  SampleWeights s{{0, 31, w.up.N / 2, w.up.N - 1}, {}, {}, {}};
  for (uint32_t i = 0; i < 4; ++i) {
    s.g[i] = halfRow(w.gate, s.columns[i]);
    s.u[i] = halfRow(w.up, s.columns[i]);
    s.d[i] = halfRow(w.down, std::array<uint32_t, 4>{0, 31, w.down.N / 2, w.down.N - 1}[i]);
  }
  return s;
}
static bool validate(const ChainBuffers &a, const ChainBuffers &base, const SampleWeights &s,
                     bool candidate) {
  const auto ag = static_cast<const uint16_t *>(base.gate.view.contents()),
             bm = static_cast<const uint16_t *>(base.mid.view.contents()),
             ao = static_cast<const uint16_t *>(a.out.view.contents());
  const bool cp = candidate && a.compact();
  const auto am = static_cast<const uint16_t *>((cp ? a.packedMid : a.mid).view.contents());
  const auto *c = static_cast<const VerifyPackControl *>(a.control.view.contents());
  for (uint32_t l = 0; l < a.lanes; ++l) {
    for (uint32_t rr : {0u, a.rows[l] - 1}) {
      uint32_t r = l * 8 + rr, pr = cp ? c->inverse[r] : r;
      for (uint32_t j = 0; j < 4; ++j) {
        uint32_t n = s.columns[j];
        auto dg = dotHalf(a.x.data() + uint64_t{r} * a.H, s.g[j]),
             du = dotHalf(a.x.data() + uint64_t{r} * a.H, s.u[j]);
        double eg = error(dg, a.H), eu = error(du, a.H);
        if (!inside(ag[uint64_t{r} * a.I + n], dg.value - eg, dg.value + eg))
          return false;
        auto [lo, hi] = gateInterval(du.value - eu, du.value + eu, bf16ToFloat(ag[uint64_t{r} * a.I + n]));
        if (!inside(am[uint64_t{pr} * a.I + n], lo, hi))
          return false;
        const uint32_t hn = std::array<uint32_t, 4>{0, 31, a.H / 2, a.H - 1}[j];
        auto dd = dotHalf(bm + uint64_t{r} * a.I, s.d[j]);
        double ed = error(dd, a.I), res = bf16ToFloat(a.residual[uint64_t{r} * a.H + hn]);
        if (!inside(ao[uint64_t{r} * a.H + hn], float(float(dd.value - ed) + res),
                    float(float(dd.value + ed) + res)))
          return false;
      }
    }
  }
  return true;
}
} // namespace ffn_bounds
